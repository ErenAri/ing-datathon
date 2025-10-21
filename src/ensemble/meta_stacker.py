import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from src.models.modeling_pipeline import month_folds, ing_hubs_datathon_metric, oof_composite_monthwise

STABLE_FEATURES = [
    'avg_active_products',          # assumed alias for active_product_category_nbr_mean_12m or similar
    'tenure',
    'cc_amt_personal_z',
    'age',
    'lifecycle_days_since_first_txn'
]

BUNDLE_PATH = os.path.join('outputs', 'predictions', 'predictions_bundle.pkl')
SUBMISSIONS_DIR = os.path.join('data', 'submissions')

@dataclass
class MetaStackerConfig:
    use_ridge: bool = True
    enable_lightgbm: bool = True
    ridge_alpha_grid: Tuple[float, ...] = (0.1, 1.0, 5.0, 10.0)
    logistic_C_grid: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0)
    lgb_param_grid: Tuple[Dict[str, Any], ...] = (
        {'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'subsample': 0.9, 'reg_lambda': 0.0},
        {'num_leaves': 48, 'learning_rate': 0.04, 'feature_fraction': 0.7, 'subsample': 0.85, 'reg_lambda': 0.5},
    )
    lgb_n_estimators: int = 800
    last_n_months: int = 6
    gap_months: int = 1
    random_state: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_bundle(path: str) -> Dict[str, np.ndarray]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_meta_features(bundle: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    X_train_df = None
    X_test_df = None

    # Mandatory model OOF predictions
    required_oof = []
    name_map = {}
    for k in ['oof_lgb', 'oof_xgb', 'oof_cat', 'oof_two_stage_B']:
        if k in bundle:
            required_oof.append(bundle[k])
            name_map[k] = k.replace('oof_', '')
    if not required_oof:
        raise ValueError("No OOF predictions found in bundle.")

    X_meta = pd.DataFrame({name_map[k]: bundle[k] for k in name_map})

    # Attempt to append stable raw features from cached training/test matrices
    added_cols: List[str] = []
    try:
        with open('X_train.pkl', 'rb') as f:
            X_train_df = pickle.load(f)
        with open('X_test.pkl', 'rb') as f:
            X_test_df = pickle.load(f)
        if len(X_train_df) >= len(X_meta):
            for cand in STABLE_FEATURES:
                if cand in X_train_df.columns and cand in X_test_df.columns:
                    X_meta[cand] = np.asarray(X_train_df[cand])[: len(X_meta)]
                    added_cols.append(cand)
        else:
            print("[meta_stacker] Warning: X_train.pkl shorter than meta matrix; skipping raw feature augmentation.")
    except Exception:
        pass

    if added_cols:
        print(f"[meta_stacker] Appended stable features: {', '.join(added_cols)}")

    y = pd.Series(bundle['y_train'], name='y')
    ref_dates = pd.Series(bundle['ref_dates'], name='ref_date')

    # Test predictions for each base model
    test_meta = {}
    for k in ['test_lgb', 'test_xgb', 'test_cat', 'test_two_stage_B']:
        if k in bundle:
            test_meta[k.replace('test_', '')] = bundle[k]
    X_test_meta = pd.DataFrame(test_meta)

    if added_cols and X_test_df is not None:
        for cand in added_cols:
            if cand in X_test_df.columns:
                X_test_meta[cand] = np.asarray(X_test_df[cand])[: len(X_test_meta)]

    return X_meta, y, ref_dates, X_test_meta


def time_cv_train(X: pd.DataFrame, y: pd.Series, ref_dates: pd.Series, cfg: MetaStackerConfig):
    oof_log = np.zeros(len(X))
    oof_ridge = np.zeros(len(X)) if cfg.use_ridge else None
    oof_lgb = np.zeros(len(X)) if cfg.enable_lightgbm else None
    lgb_params_used: List[Dict[str, Any]] = []

    folds = list(month_folds(ref_dates, last_n=cfg.last_n_months, gap=cfg.gap_months))
    if not folds:
        raise ValueError("No time folds produced; check ref_dates and parameters.")

    # Standardize features once (fit only on train split inside loop to avoid leakage)
    for fold_id, (tr_idx, va_idx, mlabel) in enumerate(folds, 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Logistic hyperparam search
        best_log_score = -np.inf
        best_log_model = None
        for C in cfg.logistic_C_grid:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=200, random_state=cfg.random_state))
            ])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict_proba(X_va)[:, 1]
            # Flip-guard per fold
            try:
                auc = roc_auc_score(y_va, preds)
                if auc < 0.5:
                    preds = 1 - preds
            except Exception:
                pass
            score, _ = ing_hubs_datathon_metric(y_va, preds)
            if score > best_log_score:
                best_log_score = score
                best_log_model = (pipe, preds)
        if best_log_model is None:
            best_log_model = (pipe, np.zeros(len(va_idx), dtype=float))
        oof_log[va_idx] = best_log_model[1]
        print(f"Fold {fold_id} Logistic best composite={best_log_score:.6f} month={mlabel}")

        if cfg.use_ridge:
            best_ridge_score = -np.inf
            best_ridge_preds = None
            for alpha in cfg.ridge_alpha_grid:
                pipe_r = Pipeline([
                    ('scaler', StandardScaler()),
                    ('reg', Ridge(alpha=alpha))
                ])
                pipe_r.fit(X_tr, y_tr)
                raw = pipe_r.predict(X_va)
                preds = _sigmoid(np.asarray(raw, dtype=float))
                try:
                    auc = roc_auc_score(y_va, preds)
                    if auc < 0.5:
                        preds = 1 - preds
                except Exception:
                    pass
                score, _ = ing_hubs_datathon_metric(y_va, preds)
                if score > best_ridge_score:
                    best_ridge_score = score
                    best_ridge_preds = preds
            if oof_ridge is None:
                oof_ridge = np.zeros(len(X), dtype=float)
            if best_ridge_preds is None:
                best_ridge_preds = np.zeros(len(va_idx), dtype=float)
            oof_ridge[va_idx] = best_ridge_preds
            print(f"Fold {fold_id} Ridge best composite={best_ridge_score:.6f} month={mlabel}")

        if cfg.enable_lightgbm:
            best_lgb_score = -np.inf
            best_lgb_preds = None
            best_params = None
            for params in cfg.lgb_param_grid:
                booster = lgb.LGBMClassifier(
                    objective='binary',
                    n_estimators=cfg.lgb_n_estimators,
                    random_state=cfg.random_state,
                    **params,
                )
                booster.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )
                best_iter = booster.best_iteration_ if booster.best_iteration_ is not None else booster.n_estimators_
                preds = np.asarray(booster.predict_proba(X_va, num_iteration=best_iter))[:, 1]
                try:
                    auc = roc_auc_score(y_va, preds)
                    if auc < 0.5:
                        preds = 1 - preds
                except Exception:
                    pass
                score, _ = ing_hubs_datathon_metric(y_va, preds)
                if score > best_lgb_score:
                    best_lgb_score = score
                    best_lgb_preds = preds
                    best_params = booster.get_params()
            if oof_lgb is None:
                oof_lgb = np.zeros(len(X), dtype=float)
            if best_lgb_preds is None:
                best_lgb_preds = np.zeros(len(va_idx), dtype=float)
            oof_lgb[va_idx] = best_lgb_preds
            if best_params is not None:
                lgb_params_used.append(best_params)
            print(f"Fold {fold_id} LightGBM best composite={best_lgb_score:.6f} month={mlabel}")

    # Select model type by global OOF composite
    comp_log = oof_composite_monthwise(y, oof_log, ref_dates=ref_dates, last_n_months=cfg.last_n_months)
    print(f"Logistic global OOF composite={comp_log:.6f}")
    if cfg.use_ridge:
        comp_ridge = oof_composite_monthwise(y, oof_ridge, ref_dates=ref_dates, last_n_months=cfg.last_n_months)
        print(f"Ridge(global sigmoid) global OOF composite={comp_ridge:.6f}")
    else:
        comp_ridge = -np.inf

    if cfg.enable_lightgbm and oof_lgb is not None:
        comp_lgb = oof_composite_monthwise(y, oof_lgb, ref_dates=ref_dates, last_n_months=cfg.last_n_months)
        print(f"LightGBM global OOF composite={comp_lgb:.6f}")
    else:
        comp_lgb = -np.inf

    scores_map = {
        'logistic': comp_log,
        'ridge': comp_ridge,
        'lgb': comp_lgb,
    }
    chosen_label = max(scores_map, key=lambda k: scores_map[k])
    if chosen_label == 'logistic':
        chosen_oof = oof_log
    elif chosen_label == 'ridge':
        chosen_oof = oof_ridge
    else:
        chosen_oof = oof_lgb
    print(f"Chosen meta model: {chosen_label} (OOF composite={scores_map[chosen_label]:.6f})")

    return {
        'oof_log': oof_log,
        'oof_ridge': oof_ridge,
        'oof_lgb': oof_lgb,
        'lgb_params': lgb_params_used,
        'chosen_oof': chosen_oof,
        'chosen_model': chosen_label,
        'comp_log': comp_log,
        'comp_ridge': comp_ridge,
        'comp_lgb': comp_lgb,
    }


def fit_full_and_predict(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, ref_dates: pd.Series, cfg: MetaStackerConfig, chosen_model: str) -> np.ndarray:
    if chosen_model == 'logistic':
        # Pick best C by refitting using global composite on CV via internal search again
        best_C = None
        best_score = -np.inf
        for C in cfg.logistic_C_grid:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=300, random_state=cfg.random_state))
            ])
            pipe.fit(X, y)
            preds = pipe.predict_proba(X)[:, 1]
            score = oof_composite_monthwise(y, preds, ref_dates=ref_dates, last_n_months=cfg.last_n_months)
            if score > best_score:
                best_score = score
                best_C = C
                best_pipe = pipe
        print(f"Refit logistic with C={best_C} (full-data composite={best_score:.6f})")
        return best_pipe.predict_proba(X_test)[:, 1]
    elif chosen_model == 'lgb':
        best_score = -np.inf
        best_booster = None
        best_iter = None
        last_booster = None
        for params in cfg.lgb_param_grid:
            booster = lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                n_estimators=cfg.lgb_n_estimators,
                random_state=cfg.random_state,
                **params,
            )
            booster.fit(X, y)
            preds = np.asarray(booster.predict_proba(X))[:, 1]
            score = oof_composite_monthwise(y, preds, ref_dates=ref_dates, last_n_months=cfg.last_n_months)
            if score > best_score:
                best_score = score
                best_booster = booster
                best_iter = booster.n_estimators_
            last_booster = booster
        if best_booster is None:
            best_booster = last_booster
            best_iter = last_booster.n_estimators_ if last_booster is not None else cfg.lgb_n_estimators
        if best_iter is None:
            best_iter = cfg.lgb_n_estimators
        print(f"Refit LightGBM (full-data composite={best_score:.6f})")
        if best_booster is None:
            raise RuntimeError("LightGBM booster not available after refit.")
        return np.asarray(best_booster.predict_proba(X_test, num_iteration=best_iter))[:, 1]
    else:
        best_alpha = None
        best_score = -np.inf
        for alpha in cfg.ridge_alpha_grid:
            pipe_r = Pipeline([
                ('scaler', StandardScaler()),
                ('reg', Ridge(alpha=alpha))
            ])
            pipe_r.fit(X, y)
            raw = pipe_r.predict(X)
            preds = _sigmoid(np.asarray(raw, dtype=float))
            score = oof_composite_monthwise(y, preds, ref_dates=ref_dates, last_n_months=cfg.last_n_months)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_pipe = pipe_r
        print(f"Refit ridge with alpha={best_alpha} (full-data composite={best_score:.6f})")
    raw_test = best_pipe.predict(X_test)
    return _sigmoid(np.asarray(raw_test, dtype=float))


def main():
    if not os.path.exists(BUNDLE_PATH):
        raise SystemExit(f"Bundle not found at {BUNDLE_PATH}. Run main pipeline first.")

    bundle = load_bundle(BUNDLE_PATH)
    print(f"Loaded bundle keys: {sorted(bundle.keys())}")

    X_meta, y, ref_dates, X_test_meta = build_meta_features(bundle)
    print(f"Meta feature matrix shape: {X_meta.shape}; test meta shape: {X_test_meta.shape}")

    cfg = MetaStackerConfig()
    results = time_cv_train(X_meta, y, ref_dates, cfg)

    # Full-data refit & predict
    test_pred = fit_full_and_predict(X_meta, y, X_test_meta, ref_dates, cfg, results['chosen_model'])

    # Save submission
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    cust_count = len(X_test_meta)
    # We need customer IDs: attempt to reconstruct order from calibrated submission file
    try:
        base_sub = pd.read_csv(os.path.join(SUBMISSIONS_DIR, 'submission.csv'))
        cust_ids = base_sub['cust_id'].values
    except Exception:
        cust_ids = np.arange(1, cust_count + 1)

    sub_meta = pd.DataFrame({'cust_id': cust_ids, 'churn': test_pred})
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_meta.csv')
    sub_meta.to_csv(out_path, index=False)
    print(f"Saved meta submission: {out_path} shape={sub_meta.shape}")

    comp_meta = oof_composite_monthwise(y, results['chosen_oof'], ref_dates=ref_dates, last_n_months=cfg.last_n_months)
    print(f"Global OOF composite (chosen meta model) = {comp_meta:.6f}")

if __name__ == '__main__':
    main()
