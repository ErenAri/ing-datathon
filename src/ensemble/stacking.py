import argparse
import os
import glob
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.isotonic import IsotonicRegression

from src.models.modeling_pipeline import oof_composite_monthwise, ing_hubs_datathon_metric


def _read_parquet_oof_test(oof_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = glob.glob(os.path.join(oof_dir, "*.parquet"))
    cols_oof = {}
    cols_test = {}
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            # heuristic: if it has 'oof' and 'test' columns
            name = Path(fp).stem
            if 'oof' in df.columns:
                cols_oof[name] = df['oof'].astype(float).values
            if 'test' in df.columns:
                cols_test[name] = df['test'].astype(float).values
        except Exception:
            continue
    if not cols_oof:
        raise FileNotFoundError("No usable parquet OOF files found under outputs/oof")
    oof_df = pd.DataFrame(cols_oof)
    test_df = pd.DataFrame(cols_test) if cols_test else None
    return oof_df, test_df


def _read_bundle(bundle_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    with open(bundle_path, 'rb') as f:
        b = pickle.load(f)
    base_oof = {}
    base_test = {}
    for k in list(b.keys()):
        if k.startswith('oof_'):
            name = k.replace('oof_', '')
            base_oof[name] = np.asarray(b[k], dtype=float)
            t_key = 'test_' + name
            if t_key in b:
                base_test[name] = np.asarray(b[t_key], dtype=float)
    if not base_oof:
        raise RuntimeError("No base OOF predictions found in bundle")
    oof_df = pd.DataFrame(base_oof)
    test_df = pd.DataFrame(base_test) if base_test else None
    y = np.asarray(b.get('y_train'))
    ref_dates = np.asarray(b.get('ref_dates'))
    return oof_df, test_df, y, ref_dates


def _resolve_csv(filename: str) -> str:
    for p in [f"data/raw/{filename}", f"data/{filename}", filename]:
        if Path(p).exists():
            return p
    raise FileNotFoundError(filename)


def _get_test_ids() -> pd.Series:
    # Prefer reference_data_test.csv if present, else sample_submission.csv
    try:
        rdt = pd.read_csv(_resolve_csv('reference_data_test.csv'))
        return rdt['cust_id']
    except Exception:
        sub = pd.read_csv(_resolve_csv('sample_submission.csv'))
        return sub['cust_id']


def _pick_stable_features(X_tr: pd.DataFrame, X_te: pd.DataFrame, top_n: int = 5, allowlist: List[str] = None) -> List[str]:
    if allowlist:
        feats = [c for c in allowlist if c in X_tr.columns and c in X_te.columns]
        return feats[:top_n]
    # score stability by mean/std drift
    common = [c for c in X_tr.columns if c in X_te.columns]
    mu_tr = X_tr[common].mean(numeric_only=True)
    mu_te = X_te[common].mean(numeric_only=True)
    sd_tr = X_tr[common].std(numeric_only=True).replace(0, 1.0)
    sd_te = X_te[common].std(numeric_only=True).replace(0, 1.0)
    # lower is better
    score = (mu_tr - mu_te).abs() / (sd_tr + sd_te)
    score = score.replace([np.inf, -np.inf], np.nan).fillna(1e6)
    top = score.sort_values().index.tolist()
    return top[:top_n]


def _segment_isotonic(oof: np.ndarray, y: np.ndarray, test_raw: np.ndarray, tenure: np.ndarray | None = None):
    EPS = 1e-6
    if tenure is None:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof, y)
        return np.clip(iso.transform(oof), EPS, 1 - EPS), np.clip(iso.transform(test_raw), EPS, 1 - EPS)
    # segment by tenure quantiles
    try:
        edges = np.unique(np.quantile(tenure, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        if len(edges) < 3:
            raise ValueError
        oof_cal = np.zeros_like(oof)
        test_cal = np.zeros_like(test_raw)
        used = 0
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            m_o = (tenure >= lo) & (tenure <= hi)
            if m_o.sum() < 2000:
                continue
            iso_b = IsotonicRegression(out_of_bounds='clip')
            iso_b.fit(oof[m_o], y[m_o])
            oof_cal[m_o] = np.clip(iso_b.transform(oof[m_o]), EPS, 1 - EPS)
            # apply to test segment using same bounds
            # if test tenure not available, will be handled by caller
            used += 1
        if used == 0:
            raise ValueError
        # Fallback any zeros to global
        mask_zero = (oof_cal == 0)
        if mask_zero.any():
            iso_g = IsotonicRegression(out_of_bounds='clip')
            iso_g.fit(oof, y)
            oof_cal[mask_zero] = np.clip(iso_g.transform(oof[mask_zero]), EPS, 1 - EPS)
            test_cal = np.clip(iso_g.transform(test_raw), EPS, 1 - EPS)
        return oof_cal, test_cal
    except Exception:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof, y)
        return np.clip(iso.transform(oof), EPS, 1 - EPS), np.clip(iso.transform(test_raw), EPS, 1 - EPS)


def _beta_calibration(oof: np.ndarray, y: np.ndarray, test_raw: np.ndarray):
    EPS = 1e-6
    # logistic regression on log-odds features
    def make_X(p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.vstack([np.log(p), np.log(1 - p)]).T
    X_tr = make_X(oof)
    X_te = make_X(test_raw)
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(X_tr, y)
    return lr.predict_proba(X_tr)[:, 1], lr.predict_proba(X_te)[:, 1]


def main():
    parser = argparse.ArgumentParser(description="Stacking meta-learner (logistic regression) with calibration.")
    parser.add_argument("--meta-features", type=str, default="", help="Comma-separated feature names to include as meta features")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength for LogisticRegression")
    args = parser.parse_args()

    # Load base OOF/test predictions
    try:
        oof_df, test_df = _read_parquet_oof_test(os.path.join('outputs', 'oof'))
        # y/ref_dates required from bundle or separate files
        with open('y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open('ref_dates.pkl', 'rb') as f:
            ref_dates = np.asarray(pickle.load(f)).astype(str)
    except Exception:
        bundle_path = os.path.join('outputs', 'predictions', 'predictions_bundle.pkl')
        oof_df, test_df, y_train, ref_dates = _read_bundle(bundle_path)

    # Load feature matrices to source meta-features and tenure for calibration
    X_train = X_test = None
    tenure_tr = tenure_te = None
    meta_list = [s for s in args.meta_features.split(',') if s.strip()] if args.meta_features else []
    try:
        with open('X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open('X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        if 'tenure' in X_train.columns:
            tenure_tr = np.asarray(X_train['tenure'])
        if 'tenure' in X_test.columns:
            tenure_te = np.asarray(X_test['tenure'])
        if not meta_list:
            meta_list = _pick_stable_features(X_train, X_test, top_n=5)
    except Exception:
        # no feature matrices available; proceed with base predictions only
        X_train = X_test = None
        tenure_tr = tenure_te = None
        if not meta_list:
            meta_list = []

    # Build meta matrices
    meta_cols = list(oof_df.columns)
    M_tr = oof_df.copy()
    M_te = test_df.copy() if test_df is not None else None
    # Append requested meta features
    if meta_list and X_train is not None and X_test is not None:
        for c in meta_list:
            if c in X_train.columns and c in X_test.columns:
                M_tr[c] = X_train[c].astype(float).values
                if M_te is not None:
                    M_te[c] = X_test[c].astype(float).values
            else:
                print(f"Warning: meta feature '{c}' not found; skipping")

    # Ensure test matrix has exactly the same columns/order as train
    if M_te is not None:
        for c in M_tr.columns:
            if c not in M_te.columns:
                M_te[c] = 0.0
        # drop any extra columns on test
        extra = [c for c in M_te.columns if c not in M_tr.columns]
        if extra:
            M_te = M_te.drop(columns=extra)
        M_te = M_te[M_tr.columns]

    # Time-based folds using month_folds logic from modeling_pipeline (via ref_dates)
    from src.models.modeling_pipeline import month_folds
    folds = list(month_folds(pd.Series(pd.to_datetime(ref_dates)), last_n=6, gap=1))

    oof_meta = np.zeros(len(M_tr), dtype=float)
    test_meta_stack = []

    for fold, (tr_idx, va_idx, ml) in enumerate(folds, 1):
        X_tr_f = M_tr.iloc[tr_idx].values
        y_tr_f = np.asarray(y_train)[tr_idx]
        X_va_f = M_tr.iloc[va_idx].values
        # standardize based on train fold
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_f)
        X_va_s = scaler.transform(X_va_f)
        # model
        lr = LogisticRegression(C=float(args.C), solver='lbfgs', max_iter=2000)
        lr.fit(X_tr_s, y_tr_f)
        oof_meta[va_idx] = lr.predict_proba(X_va_s)[:, 1]
        if M_te is not None:
            X_te_s = scaler.transform(M_te.values)
            test_meta_stack.append(lr.predict_proba(X_te_s)[:, 1])
        print(f"Fold {fold} [{ml}] done")

    # Aggregate test preds
    if test_meta_stack:
        test_meta_raw = np.mean(np.vstack(test_meta_stack), axis=0)
    else:
        # If no test preds available (no test_df), cannot proceed
        raise RuntimeError("No base test predictions available for stacking")

    # Report OOF composite
    score_meta = oof_composite_monthwise(np.asarray(y_train), oof_meta, ref_dates=pd.Series(ref_dates), last_n_months=6)
    print(f"OOF Composite (meta): {score_meta:.6f}")

    # Calibration (same approach as main: isotonic vs beta, gamma sweep)
    iso_oof, iso_test = _segment_isotonic(oof_meta, np.asarray(y_train), test_meta_raw, tenure=tenure_tr)
    beta_oof, beta_test = _beta_calibration(oof_meta, np.asarray(y_train), test_meta_raw)
    # choose better by composite
    c_iso = oof_composite_monthwise(np.asarray(y_train), iso_oof, ref_dates=pd.Series(ref_dates), last_n_months=6)
    c_beta = oof_composite_monthwise(np.asarray(y_train), beta_oof, ref_dates=pd.Series(ref_dates), last_n_months=6)
    if c_iso >= c_beta:
        chosen_oof, chosen_test, chosen_name = iso_oof, iso_test, 'isotonic'
    else:
        chosen_oof, chosen_test, chosen_name = beta_oof, beta_test, 'beta'
    print(f"Chosen calibration: {chosen_name}")

    gammas = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
    best_gamma = 1.0
    best_c = -1
    EPS = 1e-6
    for g in gammas:
        p_adj = np.clip(np.power(chosen_oof, g), EPS, 1.0 - EPS)
        c = oof_composite_monthwise(np.asarray(y_train), p_adj, ref_dates=pd.Series(ref_dates), last_n_months=6)
        if c > best_c:
            best_c = c
            best_gamma = g
    print(f"Selected temperature gamma={best_gamma:.3f} (OOF composite={best_c:.6f})")
    calibrated_test = np.clip(np.power(chosen_test, best_gamma), EPS, 1.0 - EPS)

    # Tail correction similar to main
    if calibrated_test.min() > 0.10:
        for g2 in [1.1, 1.2, 1.3]:
            _oof_try = np.clip(np.power(chosen_oof, g2), EPS, 1.0 - EPS)
            _score_try = oof_composite_monthwise(np.asarray(y_train), _oof_try, ref_dates=pd.Series(ref_dates), last_n_months=6)
            if _score_try >= (best_c - 0.002):
                calibrated_test = np.clip(np.power(chosen_test, g2), EPS, 1.0 - EPS)
                print(f"Tail correction applied with gamma={g2}")
                break

    # Save OOF/test and submission
    os.makedirs(os.path.join('outputs', 'stacking'), exist_ok=True)
    np.save(os.path.join('outputs', 'stacking', 'oof_meta.npy'), oof_meta.astype(np.float32))
    np.save(os.path.join('outputs', 'stacking', 'test_meta_raw.npy'), test_meta_raw.astype(np.float32))
    np.save(os.path.join('outputs', 'stacking', 'test_meta_calibrated.npy'), calibrated_test.astype(np.float32))

    # Create submission
    cust_ids = _get_test_ids()
    submission = pd.DataFrame({'cust_id': cust_ids, 'churn': calibrated_test})
    os.makedirs(os.path.join('data', 'submissions'), exist_ok=True)
    sub_path = os.path.join('data', 'submissions', 'submission_stacking.csv')
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved -> {sub_path}")


if __name__ == "__main__":
    main()
"""
Stacking Ensemble for ING Datathon
==================================

This module implements meta-learning (stacking) on top of base model predictions.
Stacking trains a meta-model on the out-of-fold predictions of base models to learn
the optimal way to combine them.

Two approaches are supported:
1. Simple Stacking: Meta-model trained only on base model predictions
2. Feature Stacking: Meta-model trained on predictions + original features

Author: ING Datathon Team
Date: 2025-10-11
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_competition_metric(y_true, y_pred):
    """
    Calculate the ING Datathon competition metric.

    Metric: 40% Gini + 30% Recall@10% + 30% Lift@10%

    Args:
        y_true: True labels
        y_pred: Predicted probabilities

    Returns:
        float: Competition metric score
    """
    # Gini coefficient (2*AUC - 1)
    auc = roc_auc_score(y_true, y_pred)
    gini = 2 * auc - 1

    # Recall@10%: What % of churners are in top 10% by score
    n_top = int(len(y_pred) * 0.1)
    top_indices = np.argsort(y_pred)[-n_top:]
    recall_at_10 = y_true.iloc[top_indices].sum() / y_true.sum()

    # Lift@10%: (Recall@10%) / 0.10
    lift_at_10 = recall_at_10 / 0.1

    # Competition metric
    score = 0.4 * gini + 0.3 * recall_at_10 + 0.3 * lift_at_10

    return score


class StackingEnsemble:
    """
    Stacking Ensemble Meta-Learner

    This class implements stacking by training a meta-model on the predictions
    of base models. It supports both simple stacking (predictions only) and
    feature stacking (predictions + original features).

    The meta-model can be either LogisticRegression (simple, fast, less overfitting)
    or LightGBM (more powerful, potentially better performance).

    Attributes:
        n_folds (int): Number of CV folds for meta-model training
        random_state (int): Random seed for reproducibility
        meta_model_type (str): Type of meta-model ('logistic' or 'lgb')
        include_features (bool): Whether to include original features in meta-features
        meta_models (list): List of trained meta-models (one per fold)
        meta_feature_names (list): Names of meta-features
    """

    def __init__(self,
                 n_folds: int = 5,
                 random_state: int = 42,
                 meta_model_type: str = 'logistic',
                 include_features: bool = False):
        """
        Initialize StackingEnsemble.

        Args:
            n_folds: Number of CV folds for meta-model training
            random_state: Random seed for reproducibility
            meta_model_type: Type of meta-model ('logistic' or 'lgb')
            include_features: Whether to include original features in stacking
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_model_type = meta_model_type
        self.include_features = include_features
        self.meta_models = []
        self.meta_feature_names = []

        print(f"\n{'='*80}")
        print(f"STACKING ENSEMBLE INITIALIZATION")
        print(f"{'='*80}")
        print(f"Meta-model type: {meta_model_type}")
        print(f"Include original features: {include_features}")
        print(f"Number of folds: {n_folds}")
        print(f"{'='*80}\n")

    def create_meta_features(self,
                            oof_predictions_dict: Dict[str, np.ndarray],
                            X_train: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create meta-features from base model OOF predictions.

        Args:
            oof_predictions_dict: Dictionary mapping model names to OOF predictions
                                 e.g., {'lgb': oof_lgb, 'xgb': oof_xgb, ...}
            X_train: Original training features (optional, used if include_features=True)

        Returns:
            DataFrame of meta-features
        """
        print(f"\n{'='*80}")
        print(f"CREATING META-FEATURES")
        print(f"{'='*80}")

        # Start with base model predictions
        meta_features = pd.DataFrame()

        for model_name, predictions in oof_predictions_dict.items():
            col_name = f'oof_{model_name}'
            meta_features[col_name] = predictions
            print(f"Added OOF predictions from {model_name}")

        # Store meta-feature names (just predictions for now)
        self.meta_feature_names = meta_features.columns.tolist()

        # Optionally add original features
        if self.include_features and X_train is not None:
            print(f"\nAdding {len(X_train.columns)} original features to meta-features...")

            # Reset index to ensure alignment
            X_train_reset = X_train.reset_index(drop=True)
            meta_features.reset_index(drop=True, inplace=True)

            # Add original features
            for col in X_train.columns:
                meta_features[f'feat_{col}'] = X_train_reset[col].values

            print(f"Total meta-features: {len(meta_features.columns)}")
            print(f"  - Base model predictions: {len(oof_predictions_dict)}")
            print(f"  - Original features: {len(X_train.columns)}")
        else:
            print(f"\nTotal meta-features: {len(meta_features.columns)} (predictions only)")

        print(f"{'='*80}\n")

        return meta_features

    def train_meta_model(self,
                        meta_features: pd.DataFrame,
                        y_train: pd.Series) -> Tuple[list, np.ndarray, float]:
        """
        Train meta-model using cross-validation.

        Args:
            meta_features: Meta-features (base model predictions + optional original features)
            y_train: Training labels

        Returns:
            Tuple of (trained_models, oof_meta_predictions, meta_cv_score)
        """
        print(f"\n{'='*80}")
        print(f"TRAINING META-MODEL ({self.meta_model_type.upper()})")
        print(f"{'='*80}")
        print(f"Meta-features shape: {meta_features.shape}")
        print(f"Training samples: {len(y_train)}")
        print(f"Positive class rate: {y_train.mean():.4f}")
        print(f"{'='*80}\n")

        # Initialize
        self.meta_models = []
        oof_meta_predictions = np.zeros(len(y_train))

        # Cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(meta_features, y_train), 1):
            print(f"Fold {fold}/{self.n_folds}")
            print(f"-" * 40)

            # Split data
            X_train_fold = meta_features.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = meta_features.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]

            # Train meta-model
            if self.meta_model_type == 'logistic':
                # Logistic Regression meta-model
                meta_model = LogisticRegression(
                    C=1.0,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                meta_model.fit(X_train_fold, y_train_fold)

                # Predictions
                val_pred = meta_model.predict_proba(X_val_fold)[:, 1]

            elif self.meta_model_type == 'lgb':
                # LightGBM meta-model
                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 15,  # Small tree to avoid overfitting
                    'max_depth': 3,    # Shallow trees
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 50,  # High to avoid overfitting
                    'reg_alpha': 0.5,
                    'reg_lambda': 0.5,
                    'verbose': -1,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }

                meta_model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=200,  # Few rounds to avoid overfitting
                    valid_sets=[val_data],
                    valid_names=['valid'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20),
                        lgb.log_evaluation(period=0)  # Suppress output
                    ]
                )

                # Predictions
                val_pred = meta_model.predict(X_val_fold, num_iteration=meta_model.best_iteration)

            else:
                raise ValueError(f"Unknown meta_model_type: {self.meta_model_type}")

            # Store predictions
            oof_meta_predictions[val_idx] = val_pred

            # Calculate fold score
            fold_score = calculate_competition_metric(y_val_fold, val_pred)
            fold_scores.append(fold_score)

            # Store model
            self.meta_models.append(meta_model)

            print(f"  Fold {fold} Score: {fold_score:.6f}")
            print()

        # Overall meta-model score
        meta_cv_score = calculate_competition_metric(y_train, oof_meta_predictions)

        print(f"{'='*80}")
        print(f"META-MODEL TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Average Fold Score: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
        print(f"Overall CV Score:   {meta_cv_score:.6f}")
        print(f"{'='*80}\n")

        return self.meta_models, oof_meta_predictions, meta_cv_score

    def predict(self,
               test_predictions_dict: Dict[str, np.ndarray],
               X_test: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate final test predictions using trained meta-models.

        Args:
            test_predictions_dict: Dictionary mapping model names to test predictions
                                  e.g., {'lgb': test_lgb, 'xgb': test_xgb, ...}
            X_test: Original test features (optional, used if include_features=True)

        Returns:
            Final test predictions (averaged across meta-model folds)
        """
        print(f"\n{'='*80}")
        print(f"GENERATING STACKING PREDICTIONS")
        print(f"{'='*80}")

        # Create meta-features for test set
        meta_features_test = pd.DataFrame()

        for model_name, predictions in test_predictions_dict.items():
            col_name = f'oof_{model_name}'
            meta_features_test[col_name] = predictions

        # Add original features if needed
        if self.include_features and X_test is not None:
            X_test_reset = X_test.reset_index(drop=True)
            meta_features_test.reset_index(drop=True, inplace=True)

            for col in X_test.columns:
                meta_features_test[f'feat_{col}'] = X_test_reset[col].values

        print(f"Test meta-features shape: {meta_features_test.shape}")
        print(f"Number of meta-models: {len(self.meta_models)}")

        # Generate predictions from each meta-model fold
        test_predictions = []

        for fold, meta_model in enumerate(self.meta_models, 1):
            if self.meta_model_type == 'logistic':
                fold_pred = meta_model.predict_proba(meta_features_test)[:, 1]
            elif self.meta_model_type == 'lgb':
                fold_pred = meta_model.predict(meta_features_test, num_iteration=meta_model.best_iteration)

            test_predictions.append(fold_pred)

        # Average predictions across folds
        final_predictions = np.mean(test_predictions, axis=0)

        print(f"Final predictions generated (averaged across {len(self.meta_models)} folds)")
        print(f"Prediction range: [{final_predictions.min():.6f}, {final_predictions.max():.6f}]")
        print(f"Mean prediction: {final_predictions.mean():.6f}")
        print(f"{'='*80}\n")

        return final_predictions

    def train_and_predict(self,
                         base_models_oof: Dict[str, np.ndarray],
                         base_models_test: Dict[str, np.ndarray],
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_test: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Full stacking workflow: train meta-model and generate predictions.

        This is the main method that orchestrates the entire stacking process:
        1. Create meta-features from base model OOF predictions
        2. Train meta-model with cross-validation
        3. Generate final test predictions

        Args:
            base_models_oof: Dictionary of OOF predictions from base models
                            e.g., {'lgb': oof_lgb, 'xgb': oof_xgb, 'cat': oof_cat, ...}
            base_models_test: Dictionary of test predictions from base models
                             e.g., {'lgb': test_lgb, 'xgb': test_xgb, 'cat': test_cat, ...}
            X_train: Original training features
            y_train: Training labels
            X_test: Original test features (optional, only needed if include_features=True)

        Returns:
            Tuple of (oof_meta_predictions, test_meta_predictions, meta_cv_score)
        """
        print(f"\n{'#'*80}")
        print(f"# STACKING ENSEMBLE - FULL WORKFLOW")
        print(f"{'#'*80}")
        print(f"Base models: {list(base_models_oof.keys())}")
        print(f"Meta-model: {self.meta_model_type}")
        print(f"Include features: {self.include_features}")
        print(f"{'#'*80}\n")

        # Step 1: Create meta-features
        if self.include_features:
            meta_features = self.create_meta_features(base_models_oof, X_train)
        else:
            meta_features = self.create_meta_features(base_models_oof)

        # Step 2: Train meta-model
        meta_models, oof_meta_predictions, meta_cv_score = self.train_meta_model(
            meta_features, y_train
        )

        # Step 3: Generate test predictions
        if self.include_features:
            test_meta_predictions = self.predict(base_models_test, X_test)
        else:
            test_meta_predictions = self.predict(base_models_test)

        print(f"\n{'#'*80}")
        print(f"# STACKING ENSEMBLE - COMPLETE")
        print(f"{'#'*80}")
        print(f"Meta CV Score: {meta_cv_score:.6f}")
        print(f"OOF predictions shape: {oof_meta_predictions.shape}")
        print(f"Test predictions shape: {test_meta_predictions.shape}")
        print(f"{'#'*80}\n")

        return oof_meta_predictions, test_meta_predictions, meta_cv_score


def compare_stacking_approaches(base_models_oof: Dict[str, np.ndarray],
                               base_models_test: Dict[str, np.ndarray],
                               X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_test: Optional[pd.DataFrame] = None,
                               n_folds: int = 5,
                               random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Compare different stacking approaches and return all results.

    This utility function trains multiple stacking configurations:
    1. Simple stacking with Logistic Regression
    2. Simple stacking with LightGBM
    3. Feature stacking with Logistic Regression (if X_test provided)
    4. Feature stacking with LightGBM (if X_test provided)

    Args:
        base_models_oof: Dictionary of OOF predictions from base models
        base_models_test: Dictionary of test predictions from base models
        X_train: Original training features
        y_train: Training labels
        X_test: Original test features (optional)
        n_folds: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary mapping approach name to (oof_pred, test_pred, cv_score)
    """
    results = {}

    print(f"\n{'='*80}")
    print(f"COMPARING STACKING APPROACHES")
    print(f"{'='*80}\n")

    # 1. Simple stacking with Logistic Regression
    print(f"\n>>> Approach 1: Simple Stacking + Logistic Regression\n")
    stacker1 = StackingEnsemble(
        n_folds=n_folds,
        random_state=random_state,
        meta_model_type='logistic',
        include_features=False
    )
    oof1, test1, score1 = stacker1.train_and_predict(
        base_models_oof, base_models_test, X_train, y_train
    )
    results['simple_logistic'] = (oof1, test1, score1)

    # 2. Simple stacking with LightGBM
    print(f"\n>>> Approach 2: Simple Stacking + LightGBM\n")
    stacker2 = StackingEnsemble(
        n_folds=n_folds,
        random_state=random_state,
        meta_model_type='lgb',
        include_features=False
    )
    oof2, test2, score2 = stacker2.train_and_predict(
        base_models_oof, base_models_test, X_train, y_train
    )
    results['simple_lgb'] = (oof2, test2, score2)

    # 3. Feature stacking with Logistic Regression (if X_test provided)
    if X_test is not None:
        print(f"\n>>> Approach 3: Feature Stacking + Logistic Regression\n")
        stacker3 = StackingEnsemble(
            n_folds=n_folds,
            random_state=random_state,
            meta_model_type='logistic',
            include_features=True
        )
        oof3, test3, score3 = stacker3.train_and_predict(
            base_models_oof, base_models_test, X_train, y_train, X_test
        )
        results['feature_logistic'] = (oof3, test3, score3)

        # 4. Feature stacking with LightGBM
        print(f"\n>>> Approach 4: Feature Stacking + LightGBM\n")
        stacker4 = StackingEnsemble(
            n_folds=n_folds,
            random_state=random_state,
            meta_model_type='lgb',
            include_features=True
        )
        oof4, test4, score4 = stacker4.train_and_predict(
            base_models_oof, base_models_test, X_train, y_train, X_test
        )
        results['feature_lgb'] = (oof4, test4, score4)

    # Summary
    print(f"\n{'='*80}")
    print(f"STACKING COMPARISON SUMMARY")
    print(f"{'='*80}")
    for approach_name, (_, _, score) in results.items():
        print(f"{approach_name:25s}: {score:.6f}")
    print(f"{'='*80}\n")

    return results


# Example usage
if __name__ == "__main__":
    print("""
    ================================================================================
    STACKING ENSEMBLE - EXAMPLE USAGE
    ================================================================================

    This module provides stacking/meta-learning functionality for ensemble models.

    Basic Usage:
    -----------

    # After training base models (LightGBM, XGBoost, CatBoost, Two-Stage)
    # and obtaining OOF predictions and test predictions:

    from stacking import StackingEnsemble

    # Prepare dictionaries of predictions
    base_models_oof = {
        'lgb': oof_lgb,
        'xgb': oof_xgb,
        'cat': oof_cat,
        'two_stage': oof_two_stage
    }

    base_models_test = {
        'lgb': test_lgb,
        'xgb': test_xgb,
        'cat': test_cat,
        'two_stage': test_two_stage
    }

    # Simple Stacking (Logistic Regression on predictions only)
    stacker = StackingEnsemble(
        n_folds=5,
        random_state=42,
        meta_model_type='logistic',
        include_features=False
    )

    oof_stacked, test_stacked, cv_score = stacker.train_and_predict(
        base_models_oof,
        base_models_test,
        X_train,
        y_train
    )

    # Feature Stacking (LightGBM on predictions + original features)
    stacker = StackingEnsemble(
        n_folds=5,
        random_state=42,
        meta_model_type='lgb',
        include_features=True
    )

    oof_stacked, test_stacked, cv_score = stacker.train_and_predict(
        base_models_oof,
        base_models_test,
        X_train,
        y_train,
        X_test
    )

    # Compare all stacking approaches
    from stacking import compare_stacking_approaches

    results = compare_stacking_approaches(
        base_models_oof,
        base_models_test,
        X_train,
        y_train,
        X_test
    )

    ================================================================================
    """)
