"""
Calibration utilities: fit calibrators on OOF scores, sweep gamma, and apply to test.

API
- calibrate_scores(y_true, p_raw, method='isotonic', gamma_grid=(0.9,1.0,1.1)) -> dict
- apply_calibrator(method, calibrator, p, gamma=1.0, eps=1e-6) -> np.ndarray
- save_calibrator(path, method, calibrator, gamma) -> None
- load_calibrator(path) -> dict {method, calibrator, gamma}

Methods
- none: passthrough
- sigmoid: Platt scaling via LogisticRegression on raw scores
- isotonic: IsotonicRegression(out_of_bounds='clip')
- beta: two-parameter beta calibration via logistic regression on [log(p), log(1-p)] with no intercept

Notes
- For each method (except 'none'), we fit on OOF, generate calibrated p_cal, then for each gamma in gamma_grid,
  compute p_power = clip(p_cal ** gamma, eps, 1-eps) and evaluate with ing_hubs_datathon_metric.
- The function returns best method/gamma selection summary and a callable to apply on test.
"""
from __future__ import annotations

import json
import pickle as _pkl
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

try:
    # Use the competition metric from the existing pipeline
    from src.models.modeling_pipeline import ing_hubs_datathon_metric  # type: ignore
except Exception:  # pragma: no cover
    from sklearn.metrics import roc_auc_score

    def ing_hubs_datathon_metric(y_true, y_score):  # minimal fallback
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        auc = roc_auc_score(y_true, y_score)
        # Simple proxies for recall@10 and lift@10 to avoid import issues if pipeline not on path
        k = max(1, int(round(0.10 * len(y_true))))
        idx = np.argsort(-y_score)[:k]
        rec = float(y_true[idx].sum()) / max(1, int(y_true.sum())) if int(y_true.sum()) > 0 else 0.0
        lift = (float(y_true[idx].sum()) / k) / max(1e-12, (float(y_true.sum()) / len(y_true)))
        # Compose a simple score that roughly aligns with expectations
        comp = (auc - 0.5) * 2.0 + 0.5 * rec + 0.5 * (lift / 2.0)
        return float(comp), {"auc": float(auc), "recall@10": float(rec), "lift@10": float(lift)}


EPS = 1e-6


def _clip_probs(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def _fit_sigmoid(y: np.ndarray, p: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(p.reshape(-1, 1), y.astype(int))
    return lr


def _transform_sigmoid(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    return model.predict_proba(p.reshape(-1, 1))[:, 1]


def _fit_isotonic(y: np.ndarray, p: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y.astype(int))
    return iso


def _transform_isotonic(model: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return model.transform(p)


def _fit_beta2(y: np.ndarray, p: np.ndarray) -> LogisticRegression:
    """
    Two-parameter beta calibration via logistic regression without intercept:
    q = sigmoid(a * log(p) + b * log(1-p))
    """
    p = _clip_probs(p)
    X = np.vstack([np.log(p), np.log(1.0 - p)]).T
    lr = LogisticRegression(solver="lbfgs", fit_intercept=False, max_iter=2000)
    lr.fit(X, y.astype(int))
    return lr


def _transform_beta2(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    p = _clip_probs(p)
    X = np.vstack([np.log(p), np.log(1.0 - p)]).T
    return model.predict_proba(X)[:, 1]


def apply_calibrator(method: str, calibrator: Any, p: Iterable[float], gamma: float = 1.0, eps: float = EPS) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    # Base calibrated
    if method == "none" or calibrator is None:
        base = p
    elif method == "sigmoid":
        base = _transform_sigmoid(calibrator, p)
    elif method == "isotonic":
        base = _transform_isotonic(calibrator, p)
    elif method == "beta":
        base = _transform_beta2(calibrator, p)
    else:
        raise ValueError(f"Unknown method: {method}")
    base = _clip_probs(base, eps)
    # Gamma temperature
    out = np.clip(np.power(base, float(gamma)), eps, 1.0 - eps)
    return out


def save_calibrator(path: str, method: str, calibrator: Any, gamma: float) -> None:
    payload = {"method": method, "gamma": float(gamma), "calibrator": calibrator}
    with open(path, "wb") as f:
        _pkl.dump(payload, f)


def load_calibrator(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        payload = _pkl.load(f)
    return payload


def calibrate_scores(
    y_true: Iterable[int] | np.ndarray,
    p_raw: Iterable[float] | np.ndarray,
    method: str = "isotonic",
    gamma_grid: Tuple[float, ...] = (0.9, 1.0, 1.1),
) -> Dict[str, Any]:
    """
    Fit a calibrator on OOF scores (p_raw), sweep gamma, and return best.

    Returns dict with keys:
      - method, gamma, score
      - p_oof_base: calibrated probs before gamma
      - p_oof_calibrated: after gamma adjustment
      - per_gamma_scores: mapping gamma -> composite score
      - calibrator: fitted object (or None)
      - apply_fn: lambda raw_test -> calibrated test probs with chosen gamma
      - summary: JSON-serializable summary for logging
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_raw, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and p_raw must have the same length")

    method = str(method).lower()

    if method == "none":
        p_base = _clip_probs(p)
        best_gamma = 1.0
        best_score, _met = ing_hubs_datathon_metric(y, p_base)
        result = {
            "method": method,
            "gamma": best_gamma,
            "score": float(best_score),
            "p_oof_base": p_base,
            "p_oof_calibrated": p_base,
            "per_gamma_scores": {1.0: float(best_score)},
            "calibrator": None,
            "apply_fn": lambda raw: apply_calibrator("none", None, raw, 1.0, EPS),
            "summary": {"method": method, "gamma_grid": [1.0], "best_gamma": 1.0, "best_score": float(best_score)},
        }
        print(f"Calibration=none | OOF score: {best_score:.6f}")
        return result

    # Fit calibrator
    if method == "sigmoid":
        model = _fit_sigmoid(y, p)
        p_base = _transform_sigmoid(model, p)
    elif method == "isotonic":
        model = _fit_isotonic(y, p)
        p_base = _transform_isotonic(model, p)
    elif method == "beta":
        try:
            model = _fit_beta2(y, p)
            p_base = _transform_beta2(model, p)
        except Exception as e:
            # Document skip and fall back to sigmoid
            print(f"[beta] Fallback to sigmoid due to: {e}")
            model = _fit_sigmoid(y, p)
            p_base = _transform_sigmoid(model, p)
            method = "sigmoid"
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    p_base = _clip_probs(p_base)

    # Sweep gamma
    per_gamma: Dict[float, float] = {}
    best_gamma = None
    best_score = -np.inf
    best_p = None

    print(f"Calibrating method={method} | gamma grid={tuple(gamma_grid)}")
    for g in gamma_grid:
        p_g = np.clip(np.power(p_base, float(g)), EPS, 1.0 - EPS)
        s, _ = ing_hubs_datathon_metric(y, p_g)
        per_gamma[float(g)] = float(s)
        print(f"  gamma={float(g):.2f} -> OOF score={float(s):.6f}")
        if float(s) > best_score:
            best_score = float(s)
            best_gamma = float(g)
            best_p = p_g

    assert best_p is not None and best_gamma is not None
    print(f"Chosen calibration: {method} with gamma={best_gamma:.2f} | OOF={best_score:.6f}")

    def _apply(raw_test: Iterable[float]) -> np.ndarray:
        return apply_calibrator(method, model, raw_test, gamma=best_gamma, eps=EPS)

    summary = {
        "method": method,
        "gamma_grid": list(map(float, gamma_grid)),
        "per_gamma": per_gamma,
        "best_gamma": float(best_gamma),
        "best_score": float(best_score),
    }

    return {
        "method": method,
        "gamma": float(best_gamma),
        "score": float(best_score),
        "p_oof_base": p_base,
        "p_oof_calibrated": best_p,
        "per_gamma_scores": per_gamma,
        "calibrator": model,
        "apply_fn": _apply,
        "summary": summary,
    }
