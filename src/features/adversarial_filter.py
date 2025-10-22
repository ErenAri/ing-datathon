import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# LightGBM imports
try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("LightGBM is required for adversarial filtering") from e

# Reuse month-wise folds from modeling pipeline if available
try:
    from src.models.modeling_pipeline import month_folds
except Exception:
    month_folds = None


def _load_pickle_df(path: str) -> pd.DataFrame:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    # Some pickles might store numpy arrays; enforce DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, np.ndarray):
        return pd.DataFrame(obj)
    else:
        # Attempt to coerce
        return pd.DataFrame(obj)


def _compute_domain_auc(X_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """Train a fast domain classifier (train vs test) and return ROC-AUC on a holdout split.
    This is a quick single split (80/20) sanity check used for before/after reporting.
    """
    n = len(y_tr)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    cut = int(0.8 * n)
    tr_idx, va_idx = idx[:cut], idx[cut:]

    dtrain = lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx])
    dval = lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx], reference=dtrain)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 50,
        'verbose': -1,
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dval],
        valid_names=['valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    p = model.predict(X_tr[va_idx], num_iteration=model.best_iteration)
    y_true = np.asarray(y_tr[va_idx], dtype=int)
    y_score = np.asarray(p, dtype=float)
    return float(roc_auc_score(y_true, y_score))


def _time_stratified_cv_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    ref_dates: Optional[pd.Series],
    last_n_months: int = 6,
    gap_months: int = 1,
    n_boost: int = 800,
) -> Tuple[pd.Series, List[lgb.Booster]]:
    """Train a domain classifier with month-based folds and return average gain importances.
    Returns (gain_importance_series, models).
    """
    if ref_dates is not None:
        # Use StratifiedGroupKFold with month groups for train rows and pseudo-groups for test rows.
        from sklearn.model_selection import StratifiedGroupKFold
        n_splits = 5
        ref = pd.to_datetime(pd.Series(ref_dates)).reset_index(drop=True)
        # Domain mask
        m_train = (y == 1)
        m_test = ~m_train
        # Month groups for train
        months = ref.dt.to_period('M').astype(str)
        groups = np.empty(len(y), dtype=object)
        groups[m_train] = months[m_train]
        # Pseudo-groups for test to distribute across folds
        rng = np.random.default_rng(42)
        test_groups = rng.integers(0, n_splits, size=m_test.sum())
        groups[m_test] = [f"TEST-{g}" for g in test_groups]

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(sgkf.split(X, y, groups))

        def _fold_iter_sgkf():
            for i, (tr, va) in enumerate(splits):
                yield tr, va, f"SGKF-{i+1}"
    else:
        # Fallback: standard StratifiedKFold on y (domain labels) without time
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X, y))
        def _fold_iter_kf():
            for i, (tr, va) in enumerate(folds):
                yield tr, va, f"KF-{i+1}"

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 50,
        'verbose': -1,
    }

    gain_sum = np.zeros(X.shape[1], dtype=float)
    models: List[lgb.Booster] = []

    if ref_dates is not None:
        iter_fn = _fold_iter_sgkf
    else:
        iter_fn = _fold_iter_kf

    for fold, (tr_idx, va_idx, label) in enumerate(iter_fn()):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        dval = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_boost,
            valid_sets=[dval],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        models.append(model)
        # Accumulate gain importance
        imp = model.feature_importance(importance_type='gain')
        gain_sum += imp
        print(f"Fold {fold+1} [{label}] AUC={model.best_score.get('valid',{}).get('auc', np.nan):.4f}")

    gain_mean = gain_sum / max(1, len(models))
    imp_series = pd.Series(gain_mean, index=X.columns).sort_values(ascending=False)
    return imp_series, models


def _compute_shap_importance(models: List[lgb.Booster], X: pd.DataFrame, top_m: int = 2000) -> pd.Series:
    """Approximate SHAP importance by mean |shap| across a sample of rows and models."""
    try:
        import importlib
        shap = importlib.import_module('shap')
    except Exception:
        print("shap not installed; skipping SHAP importance and using gain only.")
        return pd.Series(np.zeros(X.shape[1]), index=X.columns)

    m = min(len(X), top_m)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(X), size=m, replace=False))
    Xs = X.iloc[idx]

    shap_sum = np.zeros(X.shape[1], dtype=float)
    for model in models:
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        shap_sum += np.mean(np.abs(sv), axis=0)
    shap_mean = shap_sum / max(1, len(models))
    return pd.Series(shap_mean, index=X.columns).sort_values(ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Adversarial filter to drop drift-prone features.")
    parser.add_argument("--drop-top-k", type=int, default=30, help="Number of highest drift features to drop")
    parser.add_argument("--clip-weights", nargs=2, type=float, default=[0.9, 1.1], metavar=("LOW","HIGH"), help="Clip domain weights to [low, high]")
    parser.add_argument("--save-suffix", type=str, default="advdrop", help="Suffix for saved files (e.g., advdrop30)")
    parser.add_argument("--use-shap", action="store_true", help="Combine SHAP with gain for ranking")
    args = parser.parse_args()

    k = int(args.drop_top_k)
    w_lo, w_hi = float(args.clip_weights[0]), float(args.clip_weights[1])
    suffix = str(args.save_suffix)

    # Load matrices (co-located in project root per save_training_data.py)
    if not Path('X_train.pkl').exists() or not Path('X_test.pkl').exists():
        raise FileNotFoundError("X_train.pkl / X_test.pkl not found. Run src/utils/save_training_data.py first.")

    X_train = _load_pickle_df('X_train.pkl')
    X_test = _load_pickle_df('X_test.pkl')

    # Align columns just in case
    common_cols = [c for c in X_train.columns if c in set(X_test.columns)]
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    # Domain labels: 1 for train, 0 for test
    y_dom = np.concatenate([np.ones(len(X_train), dtype=int), np.zeros(len(X_test), dtype=int)])
    X_dom = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Load ref_dates if available to enable month-based folds on train portion only
    ref_dates = None
    if Path('ref_dates.pkl').exists():
        with open('ref_dates.pkl', 'rb') as f:
            ref_dates = pickle.load(f)
        # Repeat a placeholder period for test rows to keep indexing consistent
        if isinstance(ref_dates, (pd.Series, pd.Index)):
            rd_train = pd.Series(pd.to_datetime(ref_dates), index=np.arange(len(ref_dates)))
        else:
            rd_train = pd.Series(pd.to_datetime(ref_dates))
        pad = pd.Series([rd_train.iloc[-1]] * len(X_test))
        ref_dates_all = pd.concat([rd_train, pad], axis=0, ignore_index=True)
    else:
        ref_dates_all = None

    # Report domain AUC before filtering (quick holdout)
    auc_before = _compute_domain_auc(X_dom.values.astype(float), y_dom)
    print(f"Domain ROC-AUC BEFORE filtering: {auc_before:.4f}")

    # Train CV domain classifier to rank features
    imp_gain, models = _time_stratified_cv_feature_importance(
        X_dom, y_dom, ref_dates=ref_dates_all if month_folds else None
    )

    if args.use_shap:
        imp_shap = _compute_shap_importance(models, X_dom)
        # Normalize and combine (mean of ranks)
        rank_gain = imp_gain.rank(ascending=False, method='average')
        rank_shap = imp_shap.rank(ascending=False, method='average')
        rank_comb = (rank_gain + rank_shap) / 2.0
        drift_order = rank_comb.sort_values().index.tolist()
    else:
        drift_order = imp_gain.index.tolist()

    drop_cols = drift_order[:k]
    print(f"Dropping top-{k} drift features:")
    for c in drop_cols[:min(k, 30)]:
        print(f"  - {c}")

    X_train_f = X_train.drop(columns=drop_cols, errors='ignore')
    X_test_f = X_test.drop(columns=drop_cols, errors='ignore')

    # Compute simple domain weights (train~test): inverse of domain prob, clipped
    # Fit a fresh small model on filtered features
    y_dom_train = np.concatenate([np.ones(len(X_train_f), dtype=int), np.zeros(len(X_test_f), dtype=int)])
    X_dom_f = pd.concat([X_train_f, X_test_f], axis=0, ignore_index=True)
    dtrain = lgb.Dataset(X_dom_f.values, label=y_dom_train)
    model_w = lgb.train(
        {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_data_in_leaf': 50,
            'verbose': -1,
        },
        dtrain,
        num_boost_round=400,
    )
    p_dom = np.asarray(model_w.predict(X_dom_f.values), dtype=float)
    # Higher probability -> more train-like; weight test-likeness (1-p)
    w = 1.0 - p_dom[: len(X_train_f)]
    w = np.clip(w, w_lo, w_hi)

    # Report domain AUC after filtering (quick holdout on filtered)
    auc_after = _compute_domain_auc(X_dom_f.values.astype(float), y_dom_train)
    print(f"Domain ROC-AUC AFTER filtering:  {auc_after:.4f}")

    # Save outputs
    out_dir = Path('data/processed')
    out_dir.mkdir(parents=True, exist_ok=True)

    suf = suffix
    with open(out_dir / f'X_train_{suf}.pkl', 'wb') as f:
        pickle.dump(X_train_f, f)
    with open(out_dir / f'X_test_{suf}.pkl', 'wb') as f:
        pickle.dump(X_test_f, f)
    with open(out_dir / f'kept_features_{suf}.txt', 'w', encoding='utf-8') as f:
        for c in X_train_f.columns:
            f.write(f"{c}\n")
    with open(out_dir / f'dropped_features_{suf}.txt', 'w', encoding='utf-8') as f:
        for c in drop_cols:
            f.write(f"{c}\n")
    with open(out_dir / f'domain_weights_{suf}.npy', 'wb') as f:
        np.save(f, w.astype(np.float32))

    print("\nSaved:")
    print(f"  - {out_dir / f'X_train_{suf}.pkl'}")
    print(f"  - {out_dir / f'X_test_{suf}.pkl'}")
    print(f"  - {out_dir / f'kept_features_{suf}.txt'}")
    print(f"  - {out_dir / f'dropped_features_{suf}.txt'}")
    print(f"  - {out_dir / f'domain_weights_{suf}.npy'} (clipped to [{w_lo}, {w_hi}])")


if __name__ == "__main__":
    main()
