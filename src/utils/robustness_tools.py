"""
Robustness Tools for Kaggle Competitions
Prevent public/private leaderboard gaps by ensuring model stability
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
from typing import List, Dict, Tuple, Optional
import warnings


def robust_time_split_cv(ref_dates: pd.Series, n_splits: int = 5,
                         test_size_months: int = 3, gap_months: int = 1) -> List[Dict]:
    """
    Create rolling time-based splits that simulate future prediction.

    This is CRITICAL for preventing public/private LB gaps because it ensures
    you validate on truly future data with a realistic time gap.

    Parameters
    ----------
    ref_dates : pd.Series
        Series of reference dates for each sample
    n_splits : int
        Number of train/test splits to create
    test_size_months : int
        Number of months to use for each test fold
    gap_months : int
        Number of months gap between train and test (prevents leakage)

    Returns
    -------
    List[Dict]
        List of dicts with keys: 'train_idx', 'test_idx', 'train_months', 'test_months'

    Example
    -------
    With n_splits=3, test_size=3, gap=1:
    - Split 1: Train on M1-M6  | Gap M7     | Test M8-M10
    - Split 2: Train on M1-M9  | Gap M10    | Test M11-M13
    - Split 3: Train on M1-M12 | Gap M13    | Test M14-M16
    """
    ref_ts = pd.to_datetime(ref_dates)
    months = pd.Series(ref_ts).dt.to_period('M')
    unique_months = sorted(months.unique())

    # Need at least test_size + gap + min_train months
    min_required = test_size_months + gap_months + 6
    if len(unique_months) < min_required:
        warnings.warn(f"Only {len(unique_months)} months available, need {min_required}. "
                     f"Reducing requirements.")
        test_size_months = max(1, len(unique_months) // 3)
        gap_months = max(0, len(unique_months) // 6)

    splits = []

    # Calculate step size between folds
    available_test_months = len(unique_months) - gap_months - 6
    step = max(1, available_test_months // (n_splits + 1))

    for i in range(n_splits):
        # Calculate the test period end
        test_end_idx = len(unique_months) - i * step
        test_start_idx = test_end_idx - test_size_months

        if test_start_idx < gap_months + 6:
            break  # Not enough data for this fold

        # Gap period
        gap_end_idx = test_start_idx
        gap_start_idx = gap_end_idx - gap_months

        # Training period (everything before gap)
        train_end_idx = gap_start_idx

        if train_end_idx < 6:  # Need minimum 6 months training
            continue

        train_months = unique_months[:train_end_idx]
        test_months = unique_months[test_start_idx:test_end_idx]

        train_mask = months.isin(train_months).values
        test_mask = months.isin(test_months).values

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_months': [str(m) for m in train_months[-3:]],  # Last 3 for display
                'test_months': [str(m) for m in test_months],
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            })

    # Reverse so most recent is last (matches typical CV iteration order)
    splits = list(reversed(splits))

    return splits


def check_model_stability(predictions_list: List[np.ndarray],
                           y_true: np.ndarray,
                           model_name: str = "Model") -> Dict:
    """
    Check if predictions from different folds/seeds agree (low variance = more stable).

    This is KEY for detecting overfitting. If your models don't agree on predictions,
    they're each overfitting to their training data in different ways.

    Parameters
    ----------
    predictions_list : List[np.ndarray]
        List of prediction arrays from different models/folds (same samples)
    y_true : np.ndarray
        True labels
    model_name : str
        Name for reporting

    Returns
    -------
    Dict
        Dictionary with stability metrics and is_stable flag
    """
    predictions = np.array(predictions_list)

    # 1. Prediction variance (lower is better)
    pred_std = predictions.std(axis=0).mean()

    # 2. Rank correlation between folds
    ranks = np.array([stats.rankdata(p) for p in predictions])
    rank_corrs = []
    for i in range(len(ranks)):
        for j in range(i+1, len(ranks)):
            corr, _ = stats.spearmanr(ranks[i], ranks[j])
            rank_corrs.append(corr)
    mean_rank_corr = np.mean(rank_corrs) if rank_corrs else 0.0

    # 3. Top-10% agreement
    k = int(0.1 * predictions.shape[1])
    top_k_sets = [set(np.argsort(-p)[:k]) for p in predictions]
    agreements = []
    for i in range(len(top_k_sets)):
        for j in range(i+1, len(top_k_sets)):
            overlap = len(top_k_sets[i] & top_k_sets[j])
            agreements.append(overlap / k)
    mean_agreement = np.mean(agreements) if agreements else 0.0

    # 4. AUC variance
    aucs = [roc_auc_score(y_true, p) for p in predictions]
    auc_std = np.std(aucs)

    # Determine if stable
    is_stable = (pred_std < 0.1 and mean_rank_corr > 0.85 and mean_agreement > 0.7)

    print(f"\n{'='*60}")
    print(f"{model_name.upper()} STABILITY REPORT")
    print(f"{'='*60}")
    print(f"Prediction Std Dev (avg):    {pred_std:.6f}  {'✓ STABLE' if pred_std < 0.1 else '⚠ UNSTABLE'}")
    print(f"Rank Correlation (folds):    {mean_rank_corr:.4f}  {'✓ STABLE' if mean_rank_corr > 0.85 else '⚠ UNSTABLE'}")
    print(f"Top-10% Agreement:           {mean_agreement:.4f}  {'✓ STABLE' if mean_agreement > 0.7 else '⚠ UNSTABLE'}")
    print(f"AUC Std Dev:                 {auc_std:.6f}")
    print(f"\nOverall: {'✓ MODEL IS STABLE' if is_stable else '⚠ MODEL IS UNSTABLE'}")
    print(f"{'='*60}\n")

    return {
        'pred_std': pred_std,
        'rank_corr': mean_rank_corr,
        'top10_agreement': mean_agreement,
        'auc_std': auc_std,
        'is_stable': is_stable
    }


def stable_feature_selection(X_train: pd.DataFrame, y_train: pd.Series,
                              ref_dates: pd.Series, n_iterations: int = 10,
                              stability_threshold: float = 0.7,
                              importance_threshold: float = 0.001) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features that are consistently important across different time periods.

    This PREVENTS overfitting by removing features that are only important
    due to noise or specific to your validation period.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    ref_dates : pd.Series
        Reference dates for time-based splits
    n_iterations : int
        Number of different time splits to test on
    stability_threshold : float
        Minimum fraction of folds where feature must be important (0-1)
    importance_threshold : float
        Minimum importance value to be considered "important"

    Returns
    -------
    Tuple[List[str], pd.DataFrame]
        (stable_feature_names, importance_statistics_df)
    """
    import lightgbm as lgb
    from collections import defaultdict

    print(f"\n{'='*60}")
    print(f"STABLE FEATURE SELECTION")
    print(f"{'='*60}")
    print(f"Testing feature stability across {n_iterations} time-based folds...")

    # Create multiple random time-based splits
    splits = robust_time_split_cv(ref_dates, n_splits=n_iterations,
                                    test_size_months=2, gap_months=1)

    if len(splits) < n_iterations:
        n_iterations = len(splits)
        print(f"Warning: Only {n_iterations} splits available")

    # Track importance for each feature across folds
    importance_tracker = defaultdict(list)

    for i, split in enumerate(splits[:n_iterations]):
        print(f"  Fold {i+1}/{n_iterations}...", end=' ')

        train_idx = split['train_idx']
        val_idx = split['test_idx']

        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Quick LightGBM model
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'binary', 'metric': 'auc',
            'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'verbose': -1, 'n_jobs': -1
        }

        model = lgb.train(params, train_data, num_boost_round=500,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(period=0)])

        # Get importance
        importance = model.feature_importance(importance_type='gain')
        importance_norm = importance / (importance.sum() + 1e-9)

        for feat_name, imp in zip(X_train.columns, importance_norm):
            importance_tracker[feat_name].append(imp)

        auc = roc_auc_score(y_val, model.predict(X_val))
        print(f"AUC={auc:.4f}")

    # Analyze stability
    importance_df = pd.DataFrame({
        'feature': list(importance_tracker.keys()),
        'importance_mean': [np.mean(imps) for imps in importance_tracker.values()],
        'importance_std': [np.std(imps) for imps in importance_tracker.values()],
        'importance_cv': [np.std(imps) / (np.mean(imps) + 1e-9)
                          for imps in importance_tracker.values()],
        'presence_rate': [np.mean([imp > importance_threshold for imp in imps])
                          for imps in importance_tracker.values()]
    })

    # Select stable features
    stable_features = importance_df[
        (importance_df['presence_rate'] >= stability_threshold) &
        (importance_df['importance_cv'] < 1.0) &  # Not too volatile
        (importance_df['importance_mean'] > importance_threshold)
    ]['feature'].tolist()

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total features:              {len(X_train.columns)}")
    print(f"Stable features selected:    {len(stable_features)}")
    print(f"Removed unstable:            {len(X_train.columns) - len(stable_features)}")
    print(f"\nTop 20 most stable features:")
    top_stable = importance_df.sort_values('importance_cv').head(20)
    for _, row in top_stable.iterrows():
        print(f"  {row['feature'][:40]:40s}  mean={row['importance_mean']:.6f}  "
              f"cv={row['importance_cv']:.4f}  presence={row['presence_rate']:.2f}")
    print(f"\nTop 10 most UNSTABLE features (consider removing):")
    unstable = importance_df[importance_df['importance_mean'] > importance_threshold].sort_values('importance_cv', ascending=False).head(10)
    for _, row in unstable.iterrows():
        print(f"  {row['feature'][:40]:40s}  mean={row['importance_mean']:.6f}  "
              f"cv={row['importance_cv']:.4f}  ⚠")
    print(f"{'='*60}\n")

    return stable_features, importance_df


def get_conservative_blend_weights(models_oof: Dict[str, np.ndarray],
                                    method: str = 'equal') -> Dict[str, float]:
    """
    Get blend weights that DON'T overfit to validation set.

    CRITICAL: Optimizing weights on validation leads to overfitting.
    Use conservative approaches instead.

    Parameters
    ----------
    models_oof : Dict[str, np.ndarray]
        Dictionary mapping model names to OOF predictions
    method : str
        'equal': Simple average (1/N for each model)
        'diversity': Weight by prediction diversity

    Returns
    -------
    Dict[str, float]
        Dictionary mapping model names to weights (sum to 1.0)
    """
    n_models = len(models_oof)

    if method == 'equal':
        weights = {name: 1.0 / n_models for name in models_oof.keys()}

    elif method == 'diversity':
        # Calculate pairwise correlation between model predictions
        preds = np.column_stack([models_oof[name] for name in models_oof.keys()])
        corr_matrix = np.corrcoef(preds.T)

        # Lower average correlation → higher weight (more diverse)
        avg_corr = corr_matrix.mean(axis=1)
        diversity_scores = 1.0 - avg_corr
        diversity_scores = np.clip(diversity_scores, 0.1, 1.0)

        # Normalize to sum to 1
        weights_array = diversity_scores / diversity_scores.sum()
        weights = {name: w for name, w in zip(models_oof.keys(), weights_array)}

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"\nConservative blend weights (method={method}):")
    for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name:15s}: {weight:.3f}")

    return weights


class RobustnessTracker:
    """
    Track model robustness metrics across CV folds.

    Use this to monitor stability WITHOUT relying on public leaderboard.
    """

    def __init__(self):
        self.metrics = {
            'fold_scores': [],
            'fold_std': None,
            'score_trend': [],
            'stability_score': None
        }

    def track_fold(self, fold_num: int, fold_score: float, fold_month: str,
                   y_true: np.ndarray, y_pred: np.ndarray):
        """Track metrics for a single fold."""
        self.metrics['fold_scores'].append({
            'fold': fold_num,
            'month': fold_month,
            'score': fold_score,
            'auc': roc_auc_score(y_true, y_pred),
            'recall_at_10': self._recall_at_k(y_true, y_pred, k=0.1),
            'n_samples': len(y_true)
        })

    def _recall_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.1) -> float:
        """Calculate recall at top-k%."""
        n = len(y_true)
        top_k = int(k * n)
        top_idx = np.argsort(-y_pred)[:top_k]
        return y_true[top_idx].sum() / (y_true.sum() + 1e-9)

    def compute_stability_metrics(self):
        """Compute overall stability metrics after all folds."""
        scores = [f['score'] for f in self.metrics['fold_scores']]

        self.metrics['fold_std'] = np.std(scores)
        self.metrics['score_trend'] = self._compute_trend(scores)

        # Stability score: inverse of CV, bounded [0, 1]
        mean_score = np.mean(scores)
        cv = self.metrics['fold_std'] / (mean_score + 1e-9)
        self.metrics['stability_score'] = 1.0 / (1.0 + cv)

    def _compute_trend(self, scores: List[float]) -> float:
        """Fit linear trend to scores over time."""
        if len(scores) < 2:
            return 0.0
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        return slope

    def get_robustness_report(self) -> str:
        """Generate comprehensive robustness report."""
        self.compute_stability_metrics()

        scores = [f['score'] for f in self.metrics['fold_scores']]

        report = f"""
{'='*70}
MODEL ROBUSTNESS REPORT
{'='*70}

SCORE STATISTICS:
  Mean Score:              {np.mean(scores):.6f}
  Std Dev:                 {self.metrics['fold_std']:.6f}
  Min/Max:                 {np.min(scores):.6f} / {np.max(scores):.6f}
  Coefficient of Variation: {self.metrics['fold_std'] / (np.mean(scores) + 1e-9):.4f}

STABILITY INDICATORS:
  Stability Score (0-1):   {self.metrics['stability_score']:.4f}  {'✓ STABLE' if self.metrics['stability_score'] > 0.9 else '⚠ UNSTABLE'}
  Score Trend (slope):     {self.metrics['score_trend']:.6f}  {'⚠ DEGRADING' if self.metrics['score_trend'] < -0.001 else '✓ STABLE'}

PER-FOLD BREAKDOWN:
"""

        for f in self.metrics['fold_scores']:
            report += f"  Fold {f['fold']} ({f['month']}): {f['score']:.6f}  (AUC={f['auc']:.4f}, Recall@10={f['recall_at_10']:.4f}, n={f['n_samples']})\n"

        report += f"\n{'='*70}\n"

        # Add recommendations
        if self.metrics['stability_score'] < 0.9:
            report += "⚠ RECOMMENDATION: Model is UNSTABLE. Consider:\n"
            report += "  - Reducing feature complexity\n"
            report += "  - Using more conservative hyperparameters\n"
            report += "  - Increasing regularization\n"
            report += "  - Removing unstable features (use stable_feature_selection)\n"
        elif self.metrics['score_trend'] < -0.001:
            report += "⚠ RECOMMENDATION: Performance DEGRADING over time. Consider:\n"
            report += "  - Checking for temporal drift in features\n"
            report += "  - Using time-based weighting in training\n"
            report += "  - Validating feature distributions across time\n"
        else:
            report += "✓ Model appears ROBUST and STABLE for deployment.\n"

        report += f"{'='*70}\n"

        return report


def comprehensive_leakage_check(train_features: pd.DataFrame,
                                test_features: pd.DataFrame,
                                y_train: pd.Series) -> List[str]:
    """
    Multi-layer leakage detection.

    Detects:
    1. Features too correlated with target (>0.9)
    2. Features with large train-test distribution shifts
    3. Suspicious feature names
    4. Different null rates in train vs test

    Parameters
    ----------
    train_features : pd.DataFrame
        Training features
    test_features : pd.DataFrame
        Test features
    y_train : pd.Series
        Training labels

    Returns
    -------
    List[str]
        List of warning messages
    """
    warnings_list = []

    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE LEAKAGE CHECK")
    print(f"{'='*60}")

    # 1. Check for suspiciously high correlations with target
    print("1. Checking target correlations...")
    for col in train_features.columns:
        if col in ['cust_id', 'ref_date', 'churn']:
            continue

        if not pd.api.types.is_numeric_dtype(train_features[col]):
            continue

        try:
            corr = train_features[col].corr(y_train)
            if abs(corr) > 0.9:
                warnings_list.append(f"⚠ HIGH TARGET CORRELATION: {col} (r={corr:.4f})")
        except:
            pass

    # 2. Check for features that differ drastically between train and test
    print("2. Checking train-test distribution shifts...")
    common_cols = [c for c in train_features.columns if c in test_features.columns]
    for col in common_cols:
        if not pd.api.types.is_numeric_dtype(train_features[col]):
            continue

        train_mean = train_features[col].mean()
        test_mean = test_features[col].mean()
        train_std = train_features[col].std()

        if train_std > 1e-6:
            shift = abs(test_mean - train_mean) / train_std
            if shift > 3.0:  # More than 3 std devs different
                warnings_list.append(f"⚠ LARGE TRAIN-TEST SHIFT: {col} "
                                f"(shift={shift:.2f} stds)")

    # 3. Check for temporal leakage in feature names
    print("3. Checking for suspicious feature names...")
    suspicious_patterns = ['_future_', '_next_', '_after_', 'forward_', '_ahead_']
    for col in train_features.columns:
        for pattern in suspicious_patterns:
            if pattern in col.lower():
                warnings_list.append(f"⚠ SUSPICIOUS FEATURE NAME: {col} (contains '{pattern}')")

    # 4. Check if any features have different nullity rates in train vs test
    print("4. Checking null rate differences...")
    for col in common_cols:
        train_null_rate = train_features[col].isnull().mean()
        test_null_rate = test_features[col].isnull().mean()

        if abs(train_null_rate - test_null_rate) > 0.3:
            warnings_list.append(f"⚠ DIFFERENT NULL RATES: {col} "
                           f"(train={train_null_rate:.2%}, test={test_null_rate:.2%})")

    # Report
    print(f"\n{'='*60}")
    if warnings_list:
        print(f"⚠ {len(warnings_list)} POTENTIAL ISSUES FOUND")
        print(f"{'='*60}")
        for w in warnings_list[:50]:  # Limit output
            print(w)
        if len(warnings_list) > 50:
            print(f"... and {len(warnings_list) - 50} more warnings")
    else:
        print(f"✓ NO LEAKAGE DETECTED")
    print(f"{'='*60}\n")

    return warnings_list


if __name__ == "__main__":
    print(__doc__)
    print("\nAvailable functions:")
    print("  - robust_time_split_cv(): Better CV that prevents overfitting")
    print("  - check_model_stability(): Detect unstable models")
    print("  - stable_feature_selection(): Remove unreliable features")
    print("  - get_conservative_blend_weights(): Avoid overfitting ensemble weights")
    print("  - RobustnessTracker: Monitor stability across folds")
    print("  - comprehensive_leakage_check(): Find data leakage")
