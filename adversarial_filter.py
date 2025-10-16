"""
Adversarial feature drift utility.

Functions:
- compute_domain_importance(X_train, X_test, n_splits=5, random_state=42) -> dict
- drop_top_drift_features(X_train, X_test, importances_df, k) -> (Xtr_red, Xte_red, dropped)
- run_fast_adversarial_filter(X_train, X_test, k) -> dict

CLI:
  python adversarial_filter.py --k 50 --pickles-dir .

Reads X_train.pkl / X_test.pkl and prints Domain AUCs before/after and top dropped features.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
except Exception as e:  # pragma: no cover
    raise RuntimeError("LightGBM is required. Please install with 'pip install lightgbm'.") from e


SEED = 42


def _assert_aligned_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train and X_test must be pandas DataFrames.")
    if not X_train.columns.equals(X_test.columns):
        diff1 = list(X_train.columns.difference(X_test.columns))
        diff2 = list(X_test.columns.difference(X_train.columns))
        msg = []
        if diff1:
            msg.append(f"in_train_not_in_test={len(diff1)} e.g. {diff1[:5]}")
        if diff2:
            msg.append(f"in_test_not_in_train={len(diff2)} e.g. {diff2[:5]}")
        raise ValueError("Column mismatch between X_train and X_test: " + " | ".join(msg))


def _ensure_numeric_clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df_numeric = df.copy()
    # Coerce all columns to numeric; log non-numeric conversions
    non_numeric: List[str] = []
    for c in df_numeric.columns:
        if not pd.api.types.is_numeric_dtype(df_numeric[c]):
            non_numeric.append(c)
            df_numeric[c] = pd.to_numeric(df_numeric[c], errors="coerce")
    if non_numeric:
        print(f"[{name}] coerced non-numeric -> numeric columns: {len(non_numeric)} (e.g., {non_numeric[:5]})")

    # Replace inf with NaN then fill
    arr = df_numeric.to_numpy()
    n_inf = int(np.isinf(arr).sum())
    if n_inf:
        print(f"[{name}] found {n_inf} inf/-inf values -> set to NaN")
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

    n_nan = int(df_numeric.isna().sum().sum())
    if n_nan:
        print(f"[{name}] found {n_nan} NaNs -> fill with 0")
        df_numeric = df_numeric.fillna(0)

    # Final guard: ensure numeric dtype
    for c in df_numeric.columns:
        if not pd.api.types.is_numeric_dtype(df_numeric[c]):
            raise TypeError(f"[{name}] column {c} is not numeric after coercion.")
    return df_numeric


def compute_domain_importance(
    X_train: pd.DataFrame, X_test: pd.DataFrame, n_splits: int = 5, random_state: int = SEED
) -> Dict[str, object]:
    """Train LightGBM to distinguish train vs test and compute feature drift importances.

    Returns a dict with:
      - domain_auc_mean: float
      - importances: pd.DataFrame with columns [feature, gain_mean, gain_std]
    """
    _assert_aligned_columns(X_train, X_test)
    X_train = _ensure_numeric_clean(X_train, "X_train")
    X_test = _ensure_numeric_clean(X_test, "X_test")

    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)])

    # Shuffle upfront for stability
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(X))
    X = X.iloc[idx].reset_index(drop=True)
    y = y[idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs: List[float] = []

    cols = list(X.columns)
    imp_mat: List[np.ndarray] = []

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": random_state,
        "feature_pre_filter": False,
        "num_leaves": 64,
        "max_depth": -1,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "num_threads": max(1, (os.cpu_count() or 8) // 2),
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], free_raw_data=True)
        dvalid = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], free_raw_data=True)

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Predict and AUC
        p = booster.predict(X.iloc[va_idx], num_iteration=booster.best_iteration)
        auc = roc_auc_score(y[va_idx], p)
        aucs.append(float(auc))

        # Importances aligned to all cols
        fold_names = booster.feature_name()
        gains = booster.feature_importance(importance_type="gain")
        s = pd.Series(gains, index=fold_names, dtype=float)
        s = s.reindex(cols, fill_value=0.0)
        imp_mat.append(s.to_numpy(copy=False))

    if imp_mat:
        imp_arr = np.vstack(imp_mat)
        gain_mean = imp_arr.mean(axis=0)
        gain_std = imp_arr.std(axis=0)
    else:  # pragma: no cover - edge case
        gain_mean = np.zeros(len(cols), dtype=float)
        gain_std = np.zeros(len(cols), dtype=float)

    importances = pd.DataFrame(
        {
            "feature": cols,
            "gain_mean": np.maximum(gain_mean, 0.0),
            "gain_std": np.maximum(gain_std, 0.0),
        }
    ).sort_values("gain_mean", ascending=False, ignore_index=True)

    out: Dict[str, object] = {
        "domain_auc_mean": float(np.mean(aucs) if len(aucs) else float("nan")),
        "importances": importances,
    }
    return out


def drop_top_drift_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, importances_df: pd.DataFrame, k: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if k <= 0:
        return X_train, X_test, []
    if not {"feature", "gain_mean"}.issubset(importances_df.columns):
        raise ValueError("importances_df must have columns: ['feature', 'gain_mean']")

    topk = importances_df.sort_values("gain_mean", ascending=False, ignore_index=True).head(k)
    drop_cols = topk["feature"].tolist()

    # Verify columns exist; ignore missing with warning
    missing = [c for c in drop_cols if c not in X_train.columns]
    if missing:
        print(f"[warn] {len(missing)} of top-k features not in X frames (ignored): {missing[:5]}")
    drop_cols = [c for c in drop_cols if c in X_train.columns]

    Xtr_red = X_train.drop(columns=drop_cols, errors="ignore")
    Xte_red = X_test.drop(columns=drop_cols, errors="ignore")
    return Xtr_red, Xte_red, drop_cols


def run_fast_adversarial_filter(X_train: pd.DataFrame, X_test: pd.DataFrame, k: int) -> Dict[str, object]:
    print(f"[info] Shapes before: train={X_train.shape}, test={X_test.shape}")
    res_before = compute_domain_importance(X_train, X_test, n_splits=5, random_state=SEED)
    auc_before = float(res_before["domain_auc_mean"])
    print(f"[info] Domain AUC (before): {auc_before:.6f}")

    Xtr_red, Xte_red, dropped = drop_top_drift_features(X_train, X_test, res_before["importances"], k=k)
    print(f"[info] Dropped {len(dropped)} features (top-{k}). First 20: {dropped[:20]}")

    res_after = compute_domain_importance(Xtr_red, Xte_red, n_splits=5, random_state=SEED)
    auc_after = float(res_after["domain_auc_mean"])
    print(f"[info] Shapes after: train={Xtr_red.shape}, test={Xte_red.shape}")
    print(f"[info] Domain AUC (after): {auc_after:.6f}")

    return {
        "domain_auc_before": auc_before,
        "domain_auc_after": auc_after,
        "dropped_features": dropped,
        "importances": res_before["importances"],
    }


def _load_pickles(pdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xtr_path = pdir / "X_train.pkl"
    xte_path = pdir / "X_test.pkl"
    if not xtr_path.exists() or not xte_path.exists():
        raise FileNotFoundError(f"Pickles not found under {pdir}. Expect X_train.pkl and X_test.pkl")
    X_train = pd.read_pickle(xtr_path)
    X_test = pd.read_pickle(xte_path)
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    return X_train, X_test


def _main_cli() -> int:
    ap = argparse.ArgumentParser(description="Fast adversarial drift filter")
    ap.add_argument("--k", type=int, default=50, help="Top-k drift features to drop")
    ap.add_argument(
        "--pickles-dir",
        type=str,
        default=".",
        help="Directory containing X_train.pkl and X_test.pkl",
    )
    args = ap.parse_args()

    pdir = Path(args.pickles_dir)
    X_train, X_test = _load_pickles(pdir)
    res = run_fast_adversarial_filter(X_train, X_test, k=args.k)

    # Print a compact summary
    print("\n=== Adversarial Filter Summary ===")
    print(f"Domain AUC before: {res['domain_auc_before']:.6f}")
    print(f"Domain AUC after:  {res['domain_auc_after']:.6f}")
    drops = res["dropped_features"]
    print(f"Dropped features ({len(drops)}): {drops[:20]}{' ...' if len(drops) > 20 else ''}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main_cli())
