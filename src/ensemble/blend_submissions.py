"""
Submission Blending for ING Datathon
====================================

This module provides advanced submission blending functionality to combine
multiple model predictions using various weighting strategies.

Blending Strategies:
1. Equal weights
2. Rank-based weights (based on CV scores)
3. Power mean ensembles (arithmetic, geometric, harmonic)
4. Median ensemble
5. Grid search over weight combinations
6. Hill climbing optimization

The blender can evaluate blends using OOF predictions to estimate CV scores
before generating final test submissions.

Author: ING Datathon Team
Date: 2025-10-12
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from itertools import product
import json
from scipy.optimize import minimize
from scipy.stats import gmean, hmean
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import pickle
from sklearn.linear_model import ElasticNet

try:
    from src.models.modeling_pipeline import ing_hubs_datathon_metric, oof_composite_monthwise  # type: ignore
except Exception:
    ing_hubs_datathon_metric = None
    oof_composite_monthwise = None


def calculate_competition_metric(
    y_true,
    y_pred,
    ref_dates: Optional[np.ndarray | pd.Series] = None,
    last_n_months: int = 6,
) -> float:
    """
    Calculate the ING Datathon composite score, preferring the official helper when available.
    """
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(y_pred, dtype=float)

    if oof_composite_monthwise is not None and ref_dates is not None:
        ref_ser = pd.to_datetime(pd.Series(ref_dates))
        return float(oof_composite_monthwise(y_arr, p_arr, ref_dates=ref_ser, last_n_months=last_n_months))

    if ing_hubs_datathon_metric is not None:
        score, _ = ing_hubs_datathon_metric(y_arr, p_arr)
        return float(score)

    # Fallback: replicate composite using simple components
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_arr, p_arr)
    gini = 2 * auc - 1
    k = max(1, int(len(p_arr) * 0.10))
    order = np.argsort(-p_arr)[:k]
    positives = y_arr.sum() if y_arr.sum() > 0 else 1.0
    recall_at_10 = y_arr[order].sum() / positives
    lift_at_10 = recall_at_10 / 0.10
    return float(0.4 * gini + 0.3 * recall_at_10 + 0.3 * lift_at_10)


class SubmissionBlender:
    """
    Advanced Submission Blending System

    This class implements multiple blending strategies to combine predictions
    from different models. It can evaluate blends using OOF predictions before
    generating final test submissions.

    Attributes:
        submission_dir (Path): Directory containing submission CSV files
        oof_dir (Path): Directory containing OOF prediction files (optional)
        y_train (np.ndarray): Training labels for OOF evaluation (optional)
        submissions (Dict): Loaded submission DataFrames
        oof_predictions (Dict): Loaded OOF predictions
        blend_results (List): List of blend results with scores and weights
    """

    def __init__(self,
                 submission_dir: str = 'submissions',
                 oof_dir: Optional[str] = None,
                 y_train: Optional[np.ndarray] = None):
        """
        Initialize SubmissionBlender.

        Args:
            submission_dir: Directory containing submission CSV files
            oof_dir: Directory containing OOF prediction files (optional)
            y_train: Training labels for evaluating OOF predictions (optional)
        """
        self.submission_dir = Path(submission_dir)
        self.oof_dir = Path(oof_dir) if oof_dir else None
        self.y_train = y_train
        self.submissions = {}
        self.oof_predictions = {}
        self.blend_results = []
        self.cv_scores = {}
        self.test_predictions = {}
        self.ref_dates = None

        print(f"\n{'='*80}")
        print(f"SUBMISSION BLENDER INITIALIZATION")
        print(f"{'='*80}")
        print(f"Submission directory: {self.submission_dir}")
        print(f"OOF directory: {self.oof_dir}")
        print(f"Training labels available: {y_train is not None}")
        print(f"{'='*80}\n")

    def load_submissions(self, pattern: str = 'submission_*.csv') -> None:
        """
        Load all submission CSV files from the submission directory.

        Args:
            pattern: Glob pattern to match submission files
        """
        print(f"\n{'='*80}")
        print(f"LOADING SUBMISSION FILES")
        print(f"{'='*80}")

        if not self.submission_dir.exists():
            raise FileNotFoundError(f"Submission directory not found: {self.submission_dir}")

        submission_files = sorted(self.submission_dir.glob(pattern))

        if not submission_files:
            raise FileNotFoundError(f"No submission files found matching pattern: {pattern}")

        for file_path in submission_files:
            # Extract model name from filename (e.g., submission_lgb.csv -> lgb)
            model_name = file_path.stem.replace('submission_', '')

            # Load submission
            df = pd.read_csv(file_path)

            # Validate format (should have cust_id and target columns)
            if 'cust_id' not in df.columns or 'target' not in df.columns:
                print(f"  ⚠ Skipping {file_path.name}: Missing required columns")
                continue

            self.submissions[model_name] = df
            print(f"  ✓ Loaded {model_name}: {len(df)} predictions")

        print(f"\nTotal submissions loaded: {len(self.submissions)}")
        print(f"{'='*80}\n")

    def load_oof_predictions(self, pattern: str = 'oof_*.npy') -> None:
        """
        Load OOF predictions for evaluating blends before submission.

        Args:
            pattern: Glob pattern to match OOF prediction files
        """
        if self.oof_dir is None:
            print("No OOF directory specified. Skipping OOF loading.")
            return

        print(f"\n{'='*80}")
        print(f"LOADING OOF PREDICTIONS")
        print(f"{'='*80}")

        if not self.oof_dir.exists():
            print(f"OOF directory not found: {self.oof_dir}")
            return

        oof_files = sorted(self.oof_dir.glob(pattern))

        # Attempt to load reference dates if not already set
        if self.ref_dates is None:
            for cand in ['ref_dates.pkl', os.path.join('data', 'processed', 'ref_dates.pkl')]:
                if os.path.exists(cand):
                    try:
                        with open(cand, 'rb') as f:
                            ref_obj = pickle.load(f)
                        self.ref_dates = np.asarray(ref_obj)
                        break
                    except Exception:
                        continue

        if not oof_files:
            print(f"No OOF files found matching pattern: {pattern}")
            return

        for file_path in oof_files:
            # Extract model name from filename (e.g., oof_lgb.npy -> lgb)
            model_name = file_path.stem.replace('oof_', '')

            # Load OOF predictions
            oof_pred = np.load(file_path)

            self.oof_predictions[model_name] = oof_pred
            print(f"  ✓ Loaded {model_name}: {len(oof_pred)} OOF predictions")

        print(f"\nTotal OOF predictions loaded: {len(self.oof_predictions)}")

        # Calculate CV scores if y_train is available
        if self.y_train is not None and len(self.oof_predictions) > 0:
            print(f"\nCalculating individual model CV scores:")
            print(f"-" * 40)
            for model_name, oof_pred in self.oof_predictions.items():
                cv_score = calculate_competition_metric(self.y_train, oof_pred, ref_dates=self.ref_dates)
                self.cv_scores[model_name] = cv_score
                print(f"  {model_name:20s}: {cv_score:.6f}")

        print(f"{'='*80}\n")

    def load_predictions_bundle(self, bundle_path: Optional[str] = None) -> None:
        """Load OOF/test predictions and labels from predictions_bundle.pkl.

        Args:
            bundle_path: Path to predictions bundle. Defaults to outputs/predictions/predictions_bundle.pkl
        """
        if bundle_path is None:
            bundle_path = os.path.join('outputs', 'predictions', 'predictions_bundle.pkl')

        if not os.path.exists(bundle_path):
            print(f"Bundle not found at {bundle_path}. Skipping bundle load.")
            return

        try:
            with open(bundle_path, 'rb') as f:
                bundle = pickle.load(f)

            key_map = {
                'oof_lgb': 'lgb',
                'oof_xgb': 'xgb',
                'oof_cat': 'cat',
                'oof_two_stage_B': 'two_stage_B',
                'oof_two_stage_A': 'two_stage_A',
                'test_lgb': 'lgb',
                'test_xgb': 'xgb',
                'test_cat': 'cat',
                'test_two_stage_B': 'two_stage_B',
                'test_two_stage_A': 'two_stage_A',
            }

            if 'y_train' in bundle:
                self.y_train = np.asarray(bundle['y_train']).astype(float)
            if 'ref_dates' in bundle:
                self.ref_dates = np.asarray(bundle['ref_dates'])

            for k, v in bundle.items():
                if k.startswith('oof_') and k in key_map:
                    self.oof_predictions[key_map[k]] = np.asarray(v).astype(float)
                if k.startswith('test_') and k in key_map:
                    self.test_predictions[key_map[k]] = np.asarray(v).astype(float)

            if self.y_train is not None and len(self.oof_predictions) > 0:
                print("\nCalculating individual model CV scores from bundle:")
                print("-" * 40)
                for model_name, oof_pred in self.oof_predictions.items():
                    cv_score = calculate_competition_metric(self.y_train, oof_pred, ref_dates=self.ref_dates)
                    self.cv_scores[model_name] = cv_score
                    print(f"  {model_name:20s}: {cv_score:.6f}")

            print("\n✓ Loaded predictions bundle and populated OOF/test maps.")
        except Exception as e:
            print(f"Failed to load bundle: {e}")

    # ---------------- Segment-wise NNLS with L2 ----------------
    def _ensure_segment_columns(self, X: pd.DataFrame, segments: List[str]) -> pd.DataFrame:
        X = X.copy()
        if 'tenure_bin' in segments and 'tenure_bin' not in X.columns and 'tenure' in X.columns:
            try:
                X['tenure_bin'] = pd.qcut(X['tenure'].fillna(X['tenure'].median()), q=4, labels=False, duplicates='drop')
            except Exception:
                X['tenure_bin'] = pd.cut(X['tenure'], bins=[-np.inf, 6, 12, 24, np.inf], labels=False)
        if 'avg_active_products_bin' in segments and 'avg_active_products_bin' not in X.columns:
            base_col = None
            for cand in ['avg_active_products', 'active_product_category_nbr_mean_12m']:
                if cand in X.columns:
                    base_col = cand
                    break
            if base_col is not None:
                try:
                    X['avg_active_products_bin'] = pd.qcut(X[base_col].fillna(X[base_col].median()), q=4, labels=False, duplicates='drop')
                except Exception:
                    X['avg_active_products_bin'] = pd.cut(X[base_col], bins=4, labels=False)
        return X

    def _build_segment_keys(self, X: pd.DataFrame, segments: List[str]) -> np.ndarray:
        seg_df = X[segments].copy()
        for c in segments:
            if seg_df[c].dtype.name == 'category':
                seg_df[c] = seg_df[c].cat.codes
        keys = list(map(tuple, seg_df.fillna(-1).astype(int).values))
        return np.array(keys, dtype=object)

    def _nnls_l2(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l2: float,
        model_names: List[str],
        ref_dates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        en = ElasticNet(alpha=l2, l1_ratio=0.0, fit_intercept=False, positive=True, max_iter=10000)
        en.fit(X, y)
        w = np.maximum(en.coef_.astype(float), 0.0)
        s = w.sum()
        if s <= 0:
            w = np.ones(X.shape[1], dtype=float) / float(X.shape[1])
        else:
            w = w / s
        pretty = ', '.join([f"{m}:{wt:.3f}" for m, wt in zip(model_names, w)])
        if ref_dates is not None:
            try:
                score = calculate_competition_metric(y, X @ w, ref_dates=ref_dates)
                print(f"    Weights -> {pretty} | composite={score:.6f}")
            except Exception:
                print(f"    Weights -> {pretty}")
        else:
            print(f"    Weights -> {pretty}")
        return w

    def optimize_segmentwise(self,
                              segments: List[str],
                              l2: float = 1e-3,
                              bundle_path: Optional[str] = None,
                              x_train_path: Optional[str] = None,
                              submission_ref_path: Optional[str] = None) -> Dict:
        print(f"\n{'='*80}")
        print(f"SEGMENT-WISE OPTIMIZATION (segments={segments}, l2={l2})")
        print(f"{'='*80}")
        # Load bundle
        self.load_predictions_bundle(bundle_path)
        if self.y_train is None or len(self.oof_predictions) == 0 or len(self.test_predictions) == 0:
            raise RuntimeError("OOF/test predictions and y_train are required. Ensure predictions_bundle.pkl exists.")
        model_names = sorted(list(set(self.oof_predictions.keys()) & set(self.test_predictions.keys())))
        if len(model_names) == 0:
            raise RuntimeError("No common models found between OOF and test predictions.")
        print(f"Models used: {model_names}")
        oof_matrix = np.column_stack([self.oof_predictions[m] for m in model_names])
        test_matrix = np.column_stack([self.test_predictions[m] for m in model_names])
        y = np.asarray(self.y_train, dtype=float)
        ref_dates_arr = np.asarray(self.ref_dates) if self.ref_dates is not None else None
        n = min(len(y), oof_matrix.shape[0])
        if n != len(y) or n != oof_matrix.shape[0]:
            print(f"⚠ Length mismatch: y={len(y)}, oof={oof_matrix.shape[0]} -> truncating to {n}")
            y = y[:n]
            oof_matrix = oof_matrix[:n]
            if ref_dates_arr is not None:
                ref_dates_arr = ref_dates_arr[:n]
                self.ref_dates = ref_dates_arr
        # Load X_train for segments
        if x_train_path is None:
            x_train_path = 'X_train.pkl'
        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"X_train pickle not found at {x_train_path}. Run save_training_data first.")
        with open(x_train_path, 'rb') as f:
            X_train_df = pickle.load(f)
        if len(X_train_df) != len(y):
            print(f"⚠ X_train rows ({len(X_train_df)}) != y_train ({len(y)}). Attempting to align by truncation.")
            m = min(len(X_train_df), len(y))
            X_train_df = X_train_df.iloc[:m].reset_index(drop=True)
            y = y[:m]
            oof_matrix = oof_matrix[:m]
            if ref_dates_arr is not None:
                ref_dates_arr = ref_dates_arr[:m]
                self.ref_dates = ref_dates_arr
        # Ensure segments
        X_train_df = self._ensure_segment_columns(X_train_df, segments)
        for s in segments:
            if s not in X_train_df.columns:
                raise KeyError(f"Segment column '{s}' not found or derivable in X_train.")
        # Build segments via groupby for robust indexing
        seg_df = X_train_df[segments].copy()
        # Ensure integer categories
        for c in segments:
            if seg_df[c].dtype.name == 'category':
                seg_df[c] = seg_df[c].cat.codes
            seg_df[c] = seg_df[c].astype(int)
        groups = seg_df.groupby(segments).groups  # dict: key -> Index of rows
        print(f"Found {len(groups)} unique segments")
        per_segment_weights = {}
        per_segment_counts = {}
        for key, idxs in groups.items():
            idx_list = list(idxs)
            n_seg = len(idx_list)
            if n_seg < max(200, oof_matrix.shape[1] * 50):
                continue
            print(f"  Segment {key} -> {n_seg} rows")
            X_seg = oof_matrix[idx_list]
            y_seg = y[idx_list]
            ref_seg = self.ref_dates[idx_list] if isinstance(self.ref_dates, (np.ndarray, list, pd.Series)) else None
            w_seg = self._nnls_l2(X_seg, y_seg, l2=l2, model_names=model_names, ref_dates=ref_seg)
            per_segment_weights[str(key)] = w_seg.tolist()
            per_segment_counts[str(key)] = n_seg
        print("\nFitting global weights as fallback...")
        w_global = self._nnls_l2(
            oof_matrix,
            y,
            l2=l2,
            model_names=model_names,
            ref_dates=self.ref_dates if isinstance(self.ref_dates, (np.ndarray, list, pd.Series)) else None,
        )
        if len(per_segment_weights) > 0:
            total = sum(per_segment_counts.values())
            w_sum = np.zeros_like(w_global)
            for k, w_list in per_segment_weights.items():
                w = np.asarray(w_list, dtype=float)
                w_sum += (per_segment_counts[k] / total) * w
            w_agg = w_sum
            s = w_agg.sum()
            if s > 0:
                w_agg = w_agg / s
        else:
            w_agg = w_global.copy()
        print("\nAggregated global weights:")
        print("    " + ', '.join([f"{m}:{wt:.3f}" for m, wt in zip(model_names, w_agg)]))
        # Temperature scaling search
        def logits(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        def inv_logits(z):
            return 1 / (1 + np.exp(-z))
        oof_blend = oof_matrix @ w_agg
        gammas = np.round(np.linspace(0.7, 1.3, 13), 3)
        best_gamma = 1.0
        best_score = -1e9
        for g in gammas:
            z = logits(oof_blend) * g
            p = inv_logits(z)
            sc = calculate_competition_metric(y, p, ref_dates=ref_dates_arr)
            if sc > best_score:
                best_score = sc
                best_gamma = float(g)
        print(f"\nChosen temperature gamma={best_gamma:.3f} (OOF composite={best_score:.6f})")
        test_blend_raw = test_matrix @ w_agg
        test_z = logits(test_blend_raw) * best_gamma
        test_blend = inv_logits(test_z)
        # Save submission
        if submission_ref_path is None:
            submission_ref_path = os.path.join('data', 'submissions', 'submission.csv')
        if not os.path.exists(submission_ref_path):
            raise FileNotFoundError(f"Reference submission not found at {submission_ref_path}")
        sub_df = pd.read_csv(submission_ref_path)
        target_col = 'target' if 'target' in sub_df.columns else ('churn' if 'churn' in sub_df.columns else None)
        if target_col is None:
            raise KeyError("Submission reference must have a 'target' or 'churn' column")
        sub_df[target_col] = test_blend
        out_dir = os.path.join('data', 'submissions')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'submission_segmentblend.csv')
        sub_df.to_csv(out_path, index=False)
        print(f"Submission saved -> {out_path}")
        try:
            with open(os.path.join(out_dir, 'last_update.txt'), 'w', encoding='utf-8') as _u:
                _u.write(f"updated_at={pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | file=submission_segmentblend.csv | rows={len(sub_df)}\n")
        except Exception:
            pass
        rep_dir = os.path.join('outputs', 'reports')
        os.makedirs(rep_dir, exist_ok=True)
        with open(os.path.join(rep_dir, 'segment_blend_weights.json'), 'w') as f:
            json.dump({m: float(w) for m, w in zip(model_names, w_agg)}, f, indent=2)
        with open(os.path.join(rep_dir, 'segment_blend_per_segment.json'), 'w') as f:
            json.dump({
                'segments': segments,
                'per_segment_weights': per_segment_weights,
                'per_segment_counts': per_segment_counts,
                'models': model_names,
                'l2': l2,
                'gamma': best_gamma
            }, f, indent=2)
        return {
            'models': model_names,
            'weights_global': w_agg.tolist(),
            'weights_per_segment': per_segment_weights,
            'counts_per_segment': per_segment_counts,
            'gamma': best_gamma,
            'oof_score': best_score,
            'submission_path': out_path
        }

    def create_equal_weight_blend(self) -> Tuple[pd.DataFrame, Dict[str, float], Optional[float]]:
        """
        Create blend with equal weights for all models.

        Returns:
            Tuple of (blended_submission, weights_dict, cv_score)
        """
        n_models = len(self.submissions)
        weights = {name: 1.0 / n_models for name in self.submissions.keys()}

        return self._create_weighted_blend(weights)

    def create_rank_based_blend(self, power: float = 1.0) -> Tuple[pd.DataFrame, Dict[str, float], Optional[float]]:
        """
        Create blend with weights based on model CV scores (rank-based).

        Models with higher CV scores get higher weights.

        Args:
            power: Exponent for rank weights (higher = more weight on top models)

        Returns:
            Tuple of (blended_submission, weights_dict, cv_score)
        """
        if not self.cv_scores:
            print("⚠ CV scores not available. Using equal weights instead.")
            return self.create_equal_weight_blend()

        # Sort models by CV score (descending)
        sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1], reverse=True)

        # Assign rank-based weights
        ranks = np.arange(1, len(sorted_models) + 1)
        rank_weights = (len(sorted_models) + 1 - ranks) ** power

        # Normalize weights
        rank_weights = rank_weights / rank_weights.sum()

        # Create weights dictionary
        weights = {model_name: weight for (model_name, _), weight in zip(sorted_models, rank_weights)}

        return self._create_weighted_blend(weights)

    def create_power_mean_blend(self, power: float = 1.0) -> Tuple[pd.DataFrame, Dict[str, float], Optional[float]]:
        """
        Create blend using power mean ensemble.

        Power values:
        - power = 1.0: Arithmetic mean (standard average)
        - power = 0.0: Geometric mean
        - power = -1.0: Harmonic mean
        - power > 1.0: Emphasizes larger values
        - power < 1.0: Emphasizes smaller values

        Args:
            power: Power for the mean (1.0=arithmetic, 0.0=geometric, -1.0=harmonic)

        Returns:
            Tuple of (blended_submission, weights_dict, cv_score)
        """
        print(f"\nCreating power mean blend (power={power})...")

        # Get reference submission for cust_id
        reference = list(self.submissions.values())[0].copy()

        # Stack all predictions
        predictions = np.column_stack([
            df.sort_values('cust_id')['target'].values
            for df in self.submissions.values()
        ])

        # Apply power mean
        if power == 0.0:
            # Geometric mean
            blended_pred = gmean(predictions, axis=1)
        elif power == -1.0:
            # Harmonic mean
            blended_pred = hmean(predictions, axis=1)
        else:
            # Power mean
            if power == 1.0:
                blended_pred = np.mean(predictions, axis=1)
            else:
                blended_pred = np.mean(predictions ** power, axis=1) ** (1.0 / power)

        # Create submission
        blended_submission = reference.copy()
        blended_submission['target'] = blended_pred

        # Calculate CV score if possible
        cv_score = None
        if self.oof_predictions and self.y_train is not None:
            oof_predictions = np.column_stack([
                self.oof_predictions[name] for name in self.submissions.keys()
                if name in self.oof_predictions
            ])

            if power == 0.0:
                blended_oof = gmean(oof_predictions, axis=1)
            elif power == -1.0:
                blended_oof = hmean(oof_predictions, axis=1)
            else:
                if power == 1.0:
                    blended_oof = np.mean(oof_predictions, axis=1)
                else:
                    blended_oof = np.mean(oof_predictions ** power, axis=1) ** (1.0 / power)

            cv_score = calculate_competition_metric(self.y_train, blended_oof, ref_dates=self.ref_dates)

        # For power mean, weights are not constant per model
        weights = {'power_mean': power}

        return blended_submission, weights, cv_score

    def create_median_blend(self) -> Tuple[pd.DataFrame, Dict[str, float], Optional[float]]:
        """
        Create blend using median of all predictions.

        Median is more robust to outliers than mean.

        Returns:
            Tuple of (blended_submission, weights_dict, cv_score)
        """
        print(f"\nCreating median blend...")

        # Get reference submission for cust_id
        reference = list(self.submissions.values())[0].copy()

        # Stack all predictions
        predictions = np.column_stack([
            df.sort_values('cust_id')['target'].values
            for df in self.submissions.values()
        ])

        # Calculate median
        blended_pred = np.median(predictions, axis=1)

        # Create submission
        blended_submission = reference.copy()
        blended_submission['target'] = blended_pred

        # Calculate CV score if possible
        cv_score = None
        if self.oof_predictions and self.y_train is not None:
            oof_predictions = np.column_stack([
                self.oof_predictions[name] for name in self.submissions.keys()
                if name in self.oof_predictions
            ])
            blended_oof = np.median(oof_predictions, axis=1)
            cv_score = calculate_competition_metric(self.y_train, blended_oof, ref_dates=self.ref_dates)
        # For typing consistency, return equal numeric weights summary
        weights = {name: 1.0 / len(self.submissions) for name in self.submissions.keys()}

        return blended_submission, weights, cv_score

    def _create_weighted_blend(self, weights: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, float], Optional[float]]:
        """
        Create weighted blend given a weights dictionary.

        Args:
            weights: Dictionary mapping model names to weights (should sum to 1.0)

        Returns:
            Tuple of (blended_submission, weights_dict, cv_score)
        """
        # Get reference submission for cust_id
        reference = list(self.submissions.values())[0].copy()

        # Initialize blended predictions
        blended_pred = np.zeros(len(reference))

        # Weighted sum
        for model_name, weight in weights.items():
            if model_name in self.submissions:
                sub = self.submissions[model_name].sort_values('cust_id')
                blended_pred += weight * sub['target'].values

        # Create submission
        blended_submission = reference.copy()
        blended_submission['target'] = blended_pred

        # Calculate CV score if OOF predictions available
        cv_score = None
        if self.oof_predictions and self.y_train is not None:
            blended_oof = np.zeros(len(self.y_train))
            for model_name, weight in weights.items():
                if model_name in self.oof_predictions:
                    blended_oof += weight * self.oof_predictions[model_name]

            cv_score = calculate_competition_metric(self.y_train, blended_oof)

        return blended_submission, weights, cv_score

    def grid_search_weights(self,
                           n_points: int = 5,
                           top_n: int = 10) -> List[Dict]:
        """
        Grid search over weight combinations.

        This generates weight combinations on a grid and evaluates each one.

        Args:
            n_points: Number of grid points per dimension (e.g., 5 = [0.0, 0.25, 0.5, 0.75, 1.0])
            top_n: Number of top blends to return

        Returns:
            List of top blend results sorted by CV score
        """
        print(f"\n{'='*80}")
        print(f"GRID SEARCH OVER WEIGHT COMBINATIONS")
        print(f"{'='*80}")
        print(f"Number of models: {len(self.submissions)}")
        print(f"Grid points per dimension: {n_points}")

        model_names = list(self.submissions.keys())
        n_models = len(model_names)

        # Generate grid points (excluding 0 for simplicity, but including small values)
        grid_points = np.linspace(0.1, 1.0, n_points)

        # Generate all combinations
        all_combinations = list(product(grid_points, repeat=n_models))

        # Filter to combinations that sum to approximately 1.0 (within tolerance)
        valid_combinations = [
            combo for combo in all_combinations
            if abs(sum(combo) - 1.0) < 0.15  # Tolerance
        ]

        # Normalize to sum exactly to 1.0
        valid_combinations = [
            tuple(w / sum(combo) for w in combo)
            for combo in valid_combinations
        ]

        print(f"Total valid weight combinations: {len(valid_combinations)}")
        print(f"Evaluating blends...")
        print(f"{'='*80}\n")

        results = []

        for i, combo in enumerate(valid_combinations, 1):
            # Create weights dictionary
            weights = {name: weight for name, weight in zip(model_names, combo)}

            # Create blend
            blended_sub, weights_dict, cv_score = self._create_weighted_blend(weights)

            # Store result
            result = {
                'blend_id': f'grid_{i:04d}',
                'weights': weights_dict,
                'cv_score': cv_score,
                'submission': blended_sub
            }
            results.append(result)

            # Progress
            if i % 50 == 0 or i == len(valid_combinations):
                if cv_score is not None:
                    print(f"  Evaluated {i}/{len(valid_combinations)} blends (latest CV: {cv_score:.6f})")
                else:
                    print(f"  Evaluated {i}/{len(valid_combinations)} blends")

        # Sort by CV score (descending)
        if results[0]['cv_score'] is not None:
            results.sort(key=lambda x: x['cv_score'], reverse=True)

        print(f"\n{'='*80}")
        print(f"GRID SEARCH COMPLETE")
        print(f"{'='*80}\n")

        return results[:top_n]

    def hill_climbing_weights(self,
                             n_iterations: int = 100,
                             n_starts: int = 5,
                             top_n: int = 10) -> List[Dict]:
        """
        Hill climbing optimization to find best weight combinations.

        Uses scipy.optimize.minimize with multiple random starts.

        Args:
            n_iterations: Maximum iterations per optimization run
            n_starts: Number of random starting points
            top_n: Number of top blends to return

        Returns:
            List of top blend results sorted by CV score
        """
        if not self.oof_predictions or self.y_train is None:
            print("⚠ OOF predictions not available. Cannot run hill climbing.")
            return []

        print(f"\n{'='*80}")
        print(f"HILL CLIMBING OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Number of models: {len(self.submissions)}")
        print(f"Number of random starts: {n_starts}")
        print(f"Max iterations per start: {n_iterations}")
        print(f"{'='*80}\n")

        model_names = list(self.submissions.keys())
        n_models = len(model_names)

        # Prepare OOF predictions matrix
        oof_matrix = np.column_stack([
            self.oof_predictions[name] for name in model_names
            if name in self.oof_predictions
        ])

        def objective(weights):
            """Objective to minimize (negative CV score)"""
            # Normalize weights to sum to 1
            weights = np.abs(weights)
            weights = weights / weights.sum()

            # Calculate blended OOF
            blended_oof = oof_matrix @ weights

            # Calculate CV score (negate for minimization)
            score = calculate_competition_metric(self.y_train, blended_oof, ref_dates=self.ref_dates)
            return -score

        results = []

        for start_idx in range(n_starts):
            print(f"Random start {start_idx + 1}/{n_starts}")
            print(f"-" * 40)

            # Random initial weights
            if start_idx == 0:
                # First start: equal weights
                initial_weights = np.ones(n_models) / n_models
            elif start_idx == 1 and self.cv_scores:
                # Second start: rank-based weights
                sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1], reverse=True)
                initial_weights = np.array([
                    self.cv_scores.get(name, 0.5) for name in model_names
                ])
                initial_weights = initial_weights / initial_weights.sum()
            else:
                # Random weights
                initial_weights = np.random.dirichlet(np.ones(n_models))

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * n_models,
                options={'maxiter': n_iterations}
            )

            # Normalize final weights
            final_weights = np.abs(result.x)
            final_weights = final_weights / final_weights.sum()

            # Calculate final score
            final_score = -result.fun

            # Create weights dictionary
            weights_dict = {name: weight for name, weight in zip(model_names, final_weights)}

            # Create submission
            blended_sub, _, cv_score = self._create_weighted_blend(weights_dict)

            # Store result
            results.append({
                'blend_id': f'hillclimb_{start_idx + 1:02d}',
                'weights': weights_dict,
                'cv_score': cv_score,
                'submission': blended_sub
            })

            print(f"  Final CV score: {final_score:.6f}")
            print(f"  Weights: {', '.join([f'{w:.3f}' for w in final_weights])}")
            print()

        # Sort by CV score (descending)
        results.sort(key=lambda x: x['cv_score'], reverse=True)

        print(f"{'='*80}")
        print(f"HILL CLIMBING COMPLETE")
        print(f"{'='*80}\n")

        return results[:top_n]

    def generate_all_blends(self,
                           grid_n_points: int = 4,
                           hillclimb_n_starts: int = 5,
                           top_n: int = 10) -> List[Dict]:
        """
        Generate blends using all strategies and return top N.

        This is the main method that runs all blending strategies:
        1. Equal weights
        2. Rank-based weights (if CV scores available)
        3. Arithmetic mean
        4. Geometric mean
        5. Harmonic mean
        6. Median
        7. Grid search
        8. Hill climbing (if OOF available)

        Args:
            grid_n_points: Number of grid points for grid search
            hillclimb_n_starts: Number of random starts for hill climbing
            top_n: Number of top blends to save

        Returns:
            List of top blend results sorted by CV score
        """
        print(f"\n{'#'*80}")
        print(f"# GENERATING ALL BLENDS")
        print(f"{'#'*80}\n")

        all_results = []

        # 1. Equal weights
        print(f"\n>>> Strategy 1: Equal Weights\n")
        blend_sub, weights, cv_score = self.create_equal_weight_blend()
        all_results.append({
            'blend_id': 'equal_weights',
            'weights': weights,
            'cv_score': cv_score,
            'submission': blend_sub
        })
        if cv_score:
            print(f"  CV Score: {cv_score:.6f}")

        # 2. Rank-based weights
        if self.cv_scores:
            print(f"\n>>> Strategy 2: Rank-Based Weights\n")
            blend_sub, weights, cv_score = self.create_rank_based_blend(power=1.0)
            all_results.append({
                'blend_id': 'rank_based',
                'weights': weights,
                'cv_score': cv_score,
                'submission': blend_sub
            })
            if cv_score:
                print(f"  CV Score: {cv_score:.6f}")

            # 2b. Rank-based with power=2
            print(f"\n>>> Strategy 2b: Rank-Based Weights (power=2)\n")
            blend_sub, weights, cv_score = self.create_rank_based_blend(power=2.0)
            all_results.append({
                'blend_id': 'rank_based_power2',
                'weights': weights,
                'cv_score': cv_score,
                'submission': blend_sub
            })
            if cv_score:
                print(f"  CV Score: {cv_score:.6f}")

        # 3. Arithmetic mean
        print(f"\n>>> Strategy 3: Arithmetic Mean\n")
        blend_sub, weights, cv_score = self.create_power_mean_blend(power=1.0)
        all_results.append({
            'blend_id': 'arithmetic_mean',
            'weights': weights,
            'cv_score': cv_score,
            'submission': blend_sub
        })
        if cv_score:
            print(f"  CV Score: {cv_score:.6f}")

        # 4. Geometric mean
        print(f"\n>>> Strategy 4: Geometric Mean\n")
        blend_sub, weights, cv_score = self.create_power_mean_blend(power=0.0)
        all_results.append({
            'blend_id': 'geometric_mean',
            'weights': weights,
            'cv_score': cv_score,
            'submission': blend_sub
        })
        if cv_score:
            print(f"  CV Score: {cv_score:.6f}")

        # 5. Harmonic mean
        print(f"\n>>> Strategy 5: Harmonic Mean\n")
        blend_sub, weights, cv_score = self.create_power_mean_blend(power=-1.0)
        all_results.append({
            'blend_id': 'harmonic_mean',
            'weights': weights,
            'cv_score': cv_score,
            'submission': blend_sub
        })
        if cv_score:
            print(f"  CV Score: {cv_score:.6f}")

        # 6. Median
        print(f"\n>>> Strategy 6: Median\n")
        blend_sub, weights, cv_score = self.create_median_blend()
        all_results.append({
            'blend_id': 'median',
            'weights': weights,
            'cv_score': cv_score,
            'submission': blend_sub
        })
        if cv_score:
            print(f"  CV Score: {cv_score:.6f}")

        # 7. Grid search
        print(f"\n>>> Strategy 7: Grid Search\n")
        grid_results = self.grid_search_weights(n_points=grid_n_points, top_n=top_n)
        all_results.extend(grid_results)

        # 8. Hill climbing
        if self.oof_predictions and self.y_train is not None:
            print(f"\n>>> Strategy 8: Hill Climbing\n")
            hillclimb_results = self.hill_climbing_weights(
                n_iterations=100,
                n_starts=hillclimb_n_starts,
                top_n=top_n
            )
            all_results.extend(hillclimb_results)

        # Sort all results by CV score
        if all_results[0]['cv_score'] is not None:
            all_results.sort(key=lambda x: x['cv_score'] if x['cv_score'] is not None else -999, reverse=True)

        # Store results
        self.blend_results = all_results

        print(f"\n{'#'*80}")
        print(f"# ALL BLENDS GENERATED")
        print(f"{'#'*80}")
        print(f"Total blends created: {len(all_results)}")
        print(f"{'#'*80}\n")

        return all_results[:top_n]

    def save_top_blends(self,
                       top_n: int = 10,
                       output_dir: str = 'blended_submissions') -> None:
        """
        Save top N blends as CSV files.

        Args:
            top_n: Number of top blends to save
            output_dir: Directory to save blended submissions
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n{'='*80}")
        print(f"SAVING TOP {top_n} BLENDS")
        print(f"{'='*80}")

        if not self.blend_results:
            print("No blend results available. Run generate_all_blends() first.")
            return

        # Get top N results
        top_results = self.blend_results[:top_n]

        # Save each blend
        summary_data = []

        for rank, result in enumerate(top_results, 1):
            blend_id = result['blend_id']
            weights = result['weights']
            cv_score = result['cv_score']
            submission = result['submission']

            # Create filename
            if cv_score is not None:
                filename = f"blend_{rank:02d}_{blend_id}_cv{cv_score:.6f}.csv"
            else:
                filename = f"blend_{rank:02d}_{blend_id}.csv"

            # Save submission
            filepath = output_path / filename
            submission.to_csv(filepath, index=False)

            # Add to summary
            summary_data.append({
                'rank': rank,
                'blend_id': blend_id,
                'cv_score': cv_score if cv_score is not None else 'N/A',
                'filename': filename,
                'weights': json.dumps(weights, indent=2)
            })

            print(f"  ✓ Rank {rank}: {filename}")

        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_path / 'blend_summary.csv'
        summary_df.to_csv(summary_path, index=False)

        # Save detailed summary with weights
        detailed_summary_path = output_path / 'blend_summary_detailed.txt'
        with open(detailed_summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BLEND SUMMARY - TOP {} SUBMISSIONS\n".format(top_n))
            f.write("="*80 + "\n\n")

            for data in summary_data:
                f.write(f"Rank {data['rank']}: {data['blend_id']}\n")
                f.write(f"  CV Score: {data['cv_score']}\n")
                f.write(f"  Filename: {data['filename']}\n")
                f.write(f"  Weights:\n")

                weights = json.loads(data['weights'])
                for model_name, weight in weights.items():
                    if isinstance(weight, (int, float)):
                        f.write(f"    {model_name:20s}: {weight:.6f}\n")
                    else:
                        f.write(f"    {model_name:20s}: {weight}\n")
                f.write("\n")

        print(f"\n  ✓ Summary saved to: {summary_path}")
        print(f"  ✓ Detailed summary saved to: {detailed_summary_path}")
        print(f"{'='*80}\n")


def main():
    """CLI entry point for blending utilities, including segment-wise optimization."""
    parser = argparse.ArgumentParser(description="Submission blending toolkit")
    parser.add_argument('--segments', type=str, default=None,
                        help="Comma-separated segment columns (e.g., tenure_bin,avg_active_products_bin)")
    parser.add_argument('--l2', type=float, default=1e-3, help="L2 regularization for NNLS (ElasticNet)")
    parser.add_argument('--bundle', type=str, default=None, help="Path to predictions_bundle.pkl")
    parser.add_argument('--xtrain', type=str, default=None, help="Path to X_train.pkl for segment bins")
    parser.add_argument('--submission-ref', type=str, default=None, help="Path to reference submission CSV for cust_id order")
    parser.add_argument('--demo', action='store_true', help="Print example usage and exit")

    args = parser.parse_args()

    if args.demo or args.segments is None:
        print(
            """
            ==============================================================================
            SUBMISSION BLENDER - EXAMPLE USAGE
            ==============================================================================

            Examples:

            # Segment-wise optimization with default bundle and X_train.pkl
            python -m src.ensemble.blend_submissions --segments tenure_bin,avg_active_products_bin --l2 1e-3

            # Classic usage (programmatic):
            from src.ensemble.blend_submissions import SubmissionBlender
            blender = SubmissionBlender(submission_dir='data/submissions', oof_dir='outputs/oof')
            blender.load_submissions(pattern='submission_*.csv')
            blender.load_oof_predictions(pattern='oof_*.npy')
            top_blends = blender.generate_all_blends()
            blender.save_top_blends(top_n=10, output_dir='outputs/blended_submissions')
            ==============================================================================
            """
        )
        if args.segments is None:
            return

    segments = [s.strip() for s in args.segments.split(',') if s.strip()]
    blender = SubmissionBlender(submission_dir=os.path.join('data', 'submissions'))
    result = blender.optimize_segmentwise(
        segments=segments,
        l2=args.l2,
        bundle_path=args.bundle,
        x_train_path=args.xtrain,
        submission_ref_path=args.submission_ref
    )
    print("\nCompleted segment-wise blend.")
    print(json.dumps({k: v for k, v in result.items() if k not in ['weights_per_segment']}, indent=2))


if __name__ == "__main__":
    main()
