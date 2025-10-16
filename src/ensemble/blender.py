"""
Non-negative weight blender for OOF streams optimizing the competition composite.

Functions
- optimize_weights_per_month(oof_dict, y, ref_dates, reg=1e-4) -> dict
    For each month, finds non-negative weights summing to 1 maximizing ing_hubs_datathon_metric
    using a simple coordinate/random search on the simplex.
    Saves weights_per_month.json and weights_global.json under outputs/reports.

- derive_global_weights(month_weights, model_names, mode='median') -> np.ndarray
    Aggregates per-month weights (median or mean) and renormalizes to simplex.

- blend_scores(weights, pred_dict) -> np.ndarray
    Applies weights to prediction streams to produce a blended vector.

Notes
- This focuses on ranking-based optimization, not regression; NNLS is not directly suitable
  for our composite target, so we use a robust, small, constrained random/coordinate ascent.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    # Use competition metric if available in workspace
    from src.models.modeling_pipeline import ing_hubs_datathon_metric  # type: ignore
except Exception:  # pragma: no cover
    from sklearn.metrics import roc_auc_score

    def ing_hubs_datathon_metric(y_true, y_score):  # minimal fallback
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        auc = roc_auc_score(y_true, y_score)
        k = max(1, int(round(0.10 * len(y_true))))
        idx = np.argsort(-y_score)[:k]
        rec = float(y_true[idx].sum()) / max(1, int(y_true.sum())) if int(y_true.sum()) > 0 else 0.0
        lift = (float(y_true[idx].sum()) / k) / max(1e-12, (float(y_true.sum()) / len(y_true)))
        comp = (auc - 0.5) * 2.0 + 0.5 * rec + 0.5 * (lift / 2.0)
        return float(comp), {"auc": float(auc), "recall@10": float(rec), "lift@10": float(lift)}


def _to_months(ref_dates: Iterable) -> np.ndarray:
    return pd.to_datetime(pd.Series(ref_dates)).dt.to_period("M").values


def _normalize_simplex(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    if s <= 0.0:
        w[:] = 1.0 / len(w)
    else:
        w = w / s
    return w


def _score_combo(y: np.ndarray, mats: np.ndarray, w: np.ndarray) -> float:
    # mats: shape (n_models, n_samples)
    p = np.dot(w, mats)
    s, _ = ing_hubs_datathon_metric(y, p)
    return float(s)


def _search_weights(
    y: np.ndarray,
    mats: np.ndarray,
    iters: int = 400,
    seed: int = 42,
    start: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = mats.shape[0]
    if start is None:
        w = np.ones(m, dtype=float) / m
    else:
        w = _normalize_simplex(np.asarray(start, dtype=float))

    best_w = w.copy()
    best_s = _score_combo(y, mats, best_w)

    # Coordinate/Dirichlet hybrids
    grid = np.linspace(0.0, 1.0, 11)
    for t in range(iters):
        # 1) Coordinate line search toward basis vectors
        k = int(rng.integers(0, m))
        for alpha in grid:
            cand = (1.0 - alpha) * best_w
            cand[k] += alpha
            cand = _normalize_simplex(cand)
            s = _score_combo(y, mats, cand)
            if s > best_s:
                best_s = s
                best_w = cand

        # 2) Soft random exploration via Dirichlet around current best
        alpha_vec = 1.0 + 50.0 * best_w  # peaked around best
        cand = rng.dirichlet(alpha_vec)
        s = _score_combo(y, mats, cand)
        if s > best_s:
            best_s = s
            best_w = cand

    return best_w


def optimize_weights_per_month(
    oof_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    ref_dates: pd.Series,
    reg: float = 1e-4,
    max_iters: int = 400,
    seed: int = 42,
) -> Dict:
    """
    Optimize non-negative, sum-to-1 weights per validation month to maximize composite.
    Saves outputs/reports/weights_per_month.json and weights_global.json.
    Returns a dict with details and prints a compact table.
    """
    # Validate inputs
    names = sorted(list(oof_dict.keys()))
    if not names:
        raise ValueError("oof_dict is empty")
    n = None
    for k in names:
        arr = np.asarray(oof_dict[k], dtype=float)
        n = len(arr) if n is None else n
        if len(arr) != n:
            raise ValueError("All OOF arrays must have the same length")

    y = np.asarray(y, dtype=int)
    if len(y) != n:
        raise ValueError("y length must match OOF length")

    months = _to_months(ref_dates)
    uniq_months = list(sorted(pd.unique(months)))

    # Materialize matrices per month
    results = {}
    table_rows: List[str] = []

    # Header
    hdr = ["month", "composite"] + names
    table_rows.append(" | ".join([f"{h:>10}" for h in hdr]))

    # Optimization per month
    for vm in uniq_months:
        mask = (months == vm)
        if not np.any(mask):
            continue
        mats = np.vstack([np.asarray(oof_dict[k], dtype=float)[mask] for k in names])  # (m, n_vm)
        y_m = y[mask]
        w_best = _search_weights(y_m, mats, iters=max_iters, seed=seed)
        # Small L2 regularization pull toward uniform (post-hoc blend)
        if reg > 0:
            w_best = _normalize_simplex((1 - reg) * w_best + reg * (np.ones_like(w_best) / len(w_best)))
        s = _score_combo(y_m, mats, w_best)

        # Record
        weights_dict = {names[i]: float(w_best[i]) for i in range(len(names))}
        results[str(vm)] = {
            "composite": float(s),
            "weights": weights_dict,
        }
        row = [str(vm), f"{s:.6f}"] + [f"{weights_dict[nm]:.3f}" for nm in names]
        table_rows.append(" | ".join([f"{c:>10}" for c in row]))

    # Derive global weights
    w_global = derive_global_weights(results, names, mode="median")

    # Save JSON artifacts
    out_dir = os.path.join("outputs", "reports")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "weights_per_month.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    wg = {names[i]: float(w_global[i]) for i in range(len(names))}
    with open(os.path.join(out_dir, "weights_global.json"), "w", encoding="utf-8") as f:
        json.dump({"weights": wg}, f, indent=2)

    # Print compact table
    print("\nBlender - month-wise weights and composites:")
    for r in table_rows:
        print(r)

    print("\nAggregated blend weights (sum=1):")
    for nm in names:
        print(f"  {nm}: {wg[nm]:.3f}")

    return {
        "model_names": names,
        "weights_per_month": results,
        "weights_global": w_global,
        "table": "\n".join(table_rows),
    }


def derive_global_weights(month_weights: Dict[str, Dict], model_names: List[str], mode: str = "median") -> np.ndarray:
    """
    Aggregate per-month weights across months.
    mode: 'median' or 'mean'
    """
    mats = []
    for _m, d in month_weights.items():
        w = [float(d["weights"].get(nm, 0.0)) for nm in model_names]
        mats.append(w)
    if not mats:
        return np.ones(len(model_names), dtype=float) / max(1, len(model_names))
    W = np.asarray(mats, dtype=float)
    if mode == "mean":
        w = np.nanmean(W, axis=0)
    else:
        w = np.nanmedian(W, axis=0)
    return _normalize_simplex(w)


def blend_scores(weights: np.ndarray, pred_dict: Dict[str, np.ndarray]) -> np.ndarray:
    names = sorted(pred_dict.keys())
    mats = np.vstack([np.asarray(pred_dict[k], dtype=float) for k in names])  # (m, n)
    w = np.asarray(weights, dtype=float)
    if len(w) != len(names):
        raise ValueError("weights length must match number of prediction streams")
    return np.dot(w, mats)
