import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import rankdata
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
import warnings
warnings.filterwarnings('ignore')

def month_folds(ref_dates, last_n=6, gap=1):
    """Yield train/valid indices based on monthly time splits with optional gap.

    For each of the last_n unique months in ref_dates, use that month as
    validation and earlier months as training, enforcing a gap of `gap` months
    immediately before the validation month to prevent look-ahead.

    - gap=1 means exclude the month right before validation (vm-1) from train.
    Returns tuples (train_idx, valid_idx, month_label)
    """
    ref_ts = pd.to_datetime(ref_dates)
    m = ref_ts.dt.to_period("M")
    uniq = sorted(m.unique())
    # Precompute numeric month codes for fast diffs
    m_codes = (ref_ts.dt.year.values * 12 + ref_ts.dt.month.values).astype(int)
    for vm in uniq[-last_n:]:
        va = np.where(m.values == vm)[0]
        # distance in months via numeric codes
        vm_code = int(vm.year * 12 + vm.month)
        dist = vm_code - m_codes
        tr = np.where(dist >= (gap + 1))[0]
        if len(tr) and len(va):
            yield tr, va, str(vm)


def _compute_time_decay_weights(ref_dates, vm_period, train_idx, gap=1, lam: float | None = None):
    """Compute per-row time decay weights for the training subset of a fold.

    w_time = exp(-lam * (vm - m)) for training rows where (vm - m) >= gap+1.
    We normalize weights to mean 1 to keep scale stable. If lam is None, return ones.
    """
    n_tr = len(train_idx)
    if lam is None or lam <= 0:
        return np.ones(n_tr, dtype=float)
    ref_ts = pd.to_datetime(ref_dates)
    m_codes = (ref_ts.dt.year.values * 12 + ref_ts.dt.month.values).astype(int)
    vm_code = int(vm_period.year * 12 + vm_period.month)
    dist_all = vm_code - m_codes
    dist_tr = dist_all[train_idx]
    w = np.exp(-float(lam) * dist_tr)
    mean_w = w.mean() if np.isfinite(w).all() and w.size > 0 else 1.0
    if mean_w <= 0:
        mean_w = 1.0
    return (w / mean_w).astype(float)


def recall_at_k(y_true, y_prob, k=0.1):
    """Calculate recall at top k%"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    tp_at_k = y_true[top].sum()
    P = y_true.sum()
    return float(tp_at_k / P) if P > 0 else 0.0


def lift_at_k(y_true, y_prob, k=0.1):
    """Calculate lift at top k%"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    tp_at_k = y_true[top].sum()
    precision_at_k = tp_at_k / m
    prevalence = y_true.mean()
    return float(precision_at_k / prevalence) if prevalence > 0 else 0.0


def ing_hubs_datathon_metric(y_true, y_prob):
    """
    Competition metric: weighted combination of Gini, Recall@10%, Lift@10%
    """
    score_weights = {
        "gini": 0.4,
        "recall_at_10perc": 0.3,
        "lift_at_10perc": 0.3,
    }
    
    baseline_scores = {
        "roc_auc": 0.6925726757936908,
        "recall_at_10perc": 0.18469015795868773,
        "lift_at_10perc": 1.847159286784029,
    }
    
    roc_auc = roc_auc_score(y_true, y_prob)
    recall_at_10perc = recall_at_k(y_true, y_prob, k=0.1)
    lift_at_10perc = lift_at_k(y_true, y_prob, k=0.1)
    
    baseline_gini = 2 * baseline_scores["roc_auc"] - 1
    new_gini = 2 * roc_auc - 1
    
    final_gini_score = new_gini / baseline_gini
    final_recall_score = recall_at_10perc / baseline_scores["recall_at_10perc"]
    final_lift_score = lift_at_10perc / baseline_scores["lift_at_10perc"]
    
    final_score = (
        final_gini_score * score_weights["gini"] +
        final_recall_score * score_weights["recall_at_10perc"] + 
        final_lift_score * score_weights["lift_at_10perc"]
    )
    
    return final_score, {
        'gini': new_gini,
        'recall@10': recall_at_10perc,
        'lift@10': lift_at_10perc,
        'auc': roc_auc
    }


def oof_composite_monthwise(y_true, y_prob, ref_dates=None, last_n_months=6):
    """
    Compute Overall OOF Score consistent with the per-month table:
    composite = 0.4*(gini/base_gini) + 0.3*(recall10/base_recall10) + 0.3*(lift10/base_lift10)

    Behavior:
    - If ref_dates is provided: compute the metric per month (last_n_months) with per-month flip-guard,
      then return the mean of monthly composites. This mirrors the per-month table aggregation.
    - Else: compute on the full vector with a global flip-guard.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    # No ref_dates: return single-vector composite
    if ref_dates is None:
        score, _ = ing_hubs_datathon_metric(y_true, y_prob)
        return float(score)

    # Month-wise aggregation (mean of monthly composites)
    m = pd.to_datetime(ref_dates).dt.to_period('M')
    months = sorted(m.unique())[-last_n_months:]
    scores = []
    for vm in months:
        mask = (m.values == vm)
        if not np.any(mask):
            continue
        y_v = y_true[mask]
        p_v = y_prob[mask].astype(float)
        # Per-month flip-guard
        try:
            auc = roc_auc_score(y_v, p_v)
            if auc < 0.5:
                p_v = 1.0 - p_v
        except Exception:
            pass
        s, _ = ing_hubs_datathon_metric(y_v, p_v)
        scores.append(s)

    # Fallback if no months collected
    if not scores:
        score, _ = ing_hubs_datathon_metric(y_true, y_prob)
        return float(score)

    return float(np.mean(scores))


class ChurnModelingPipeline:
    """
    Advanced modeling pipeline optimized for ING competition metric
    """
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = []
        self.lgb_models = []
        self.xgb_models = []
        self.feature_importance = None
        # Two-stage head state
        self.stageA_models = []
        self.stageB_models = []
        self.stageB_feature_names = None
        # CatBoost head state
        self.cat_models = []  # list of models across folds (may contain multiple seeds per fold)
        
    def train_lightgbm(self, X, y, params=None, ref_dates=None, last_n_months=6, sample_weight=None, gap_months: int = 1, time_decay_lambda: float | None = 0.2, seeds=None):
        """
        Train LightGBM with optimized parameters for competition metric
        """
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 63,  # keep trees moderate (63-127)
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_child_samples': 120,  # ensure ≥100 to reduce overfitting
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'verbose': -1,
                'n_jobs': -1
            }
        
        if seeds is None:
            seeds = [1, 11, 111, 1111, 11111]
        oof_predictions = np.zeros(len(X))
        models = []
        scores = []
        
        # Build folds: month-based if ref_dates provided, else stratified k-fold
        if ref_dates is not None:
            fold_iter = month_folds(ref_dates, last_n=last_n_months, gap=gap_months)
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            fold_iter = ((tr, va, f"KF-{i+1}") for i, (tr, va) in enumerate(skf.split(X, y)))

        for fold, (train_idx, val_idx, month_label) in enumerate(fold_iter):
            print(f"\n--- Fold {fold + 1} | Val month: {month_label} ---")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Combine provided sample weights with time-decay weights
            if sample_weight is not None:
                sw_base = np.asarray(sample_weight)[train_idx].astype(float)
            else:
                sw_base = np.ones(len(train_idx), dtype=float)
            w_time = _compute_time_decay_weights(ref_dates, pd.Period(month_label), train_idx, gap=gap_months, lam=time_decay_lambda)
            sw_tr = (sw_base * w_time).astype(float)
            train_data = lgb.Dataset(X_train, label=y_train, weight=sw_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Seed-bagging: train multiple seeds and average
            seed_preds = []
            for sd in seeds:
                p_sd = dict(params)
                p_sd['random_state'] = int(sd)
                model = lgb.train(
                    p_sd,
                    train_data,
                    num_boost_round=1800,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'valid'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100, verbose=False),
                        lgb.log_evaluation(period=100)
                    ]
                )
                vp = model.predict(X_val, num_iteration=model.best_iteration)
                vp = np.asarray(vp, dtype=float)
                # Fold-level flip-guard per seed
                try:
                    auc = roc_auc_score(np.asarray(y_val), vp)
                    if auc < 0.5:
                        vp = 1.0 - vp
                except Exception:
                    pass
                seed_preds.append(vp)
                models.append(model)

            val_preds = np.mean(np.vstack(seed_preds), axis=0)
            # Store averaged per-fold preds
            oof_predictions[val_idx] = val_preds

            # Calculate competition metric
            fold_score, metrics = ing_hubs_datathon_metric(y_val, val_preds)
            scores.append(fold_score)
            
            print(f"Fold {fold + 1} Score: {fold_score:.6f}")
            print(f"  Gini: {metrics['gini']:.4f}")
            print(f"  Recall@10%: {metrics['recall@10']:.4f}")
            print(f"  Lift@10%: {metrics['lift@10']:.4f}")


        # Overall OOF score (month-wise aggregation if ref_dates provided)
        print(f"\n{'='*50}")
        overall_score = oof_composite_monthwise(y, oof_predictions, ref_dates=ref_dates, last_n_months=last_n_months)
        print(f"Overall OOF Composite (month-wise mean): {overall_score:.6f}")
        print(f"{'='*50}\n")

        self.models = models
        self.lgb_models = models
        return oof_predictions, overall_score
    
    def train_xgboost(self, X, y, params=None, ref_dates=None, last_n_months=6, sample_weight=None, gap_months: int = 1, time_decay_lambda: float | None = 0.2, seeds=None):
        """
        Train XGBoost as alternative/ensemble component
        """
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'random_state': self.random_state,
                'tree_method': 'hist',
                'n_jobs': -1
            }
        
        if seeds is None:
            seeds = [1, 11, 111, 1111, 11111]
        oof_predictions = np.zeros(len(X))
        models = []
        scores = []
        
        if ref_dates is not None:
            fold_iter = month_folds(ref_dates, last_n=last_n_months, gap=gap_months)
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            fold_iter = ((tr, va, f"KF-{i+1}") for i, (tr, va) in enumerate(skf.split(X, y)))

        for fold, (train_idx, val_idx, month_label) in enumerate(fold_iter):
            print(f"\n--- Fold {fold + 1} | Val month: {month_label} ---")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Prepare safe sample weights (avoid NaNs/zeros)
            if sample_weight is not None:
                sw_base = np.asarray(sample_weight)[train_idx].astype(float)
            else:
                sw_base = np.ones(len(train_idx), dtype=float)
            # Time decay weights
            w_time = _compute_time_decay_weights(ref_dates, pd.Period(month_label), train_idx, gap=gap_months, lam=time_decay_lambda)
            sw_tr = (sw_base * w_time).astype(float)
            # Replace NaN/inf and floor to tiny positive to avoid zero total weight
            sw_tr = np.nan_to_num(sw_tr, nan=1.0, posinf=1.0, neginf=1.0)
            sw_tr = np.clip(sw_tr, 1e-6, None)

            # Compute a robust base_score from (weighted) prevalence; fallback=0.5
            try:
                if sw_tr is not None and np.isfinite(sw_tr).all() and sw_tr.sum() > 0:
                    base_score_fold = float(np.average(np.asarray(y_train, dtype=float), weights=sw_tr))
                else:
                    base_score_fold = float(np.mean(np.asarray(y_train, dtype=float)))
                if not np.isfinite(base_score_fold):
                    base_score_fold = 0.5
            except Exception:
                base_score_fold = 0.5

            # If validation fold is single-class, avoid AUC for early stopping
            params_fold = dict(params)
            if getattr(y_val, 'nunique', None) and y_val.nunique() < 2:
                params_fold['eval_metric'] = 'logloss'
            else:
                params_fold['eval_metric'] = params.get('eval_metric', 'auc')

            seed_preds = []
            for sd in seeds:
                pf = dict(params_fold)
                pf['random_state'] = int(sd)
                model = xgb.XGBClassifier(
                    **pf,
                    n_estimators=1200,
                    early_stopping_rounds=100,
                    base_score=base_score_fold,
                    verbose=False
                )

                model.fit(
                    X_train, y_train,
                    sample_weight=sw_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=100
                )
                vp = model.predict_proba(X_val)[:, 1]
                vp = np.asarray(vp, dtype=float)
                try:
                    auc = roc_auc_score(np.asarray(y_val), vp)
                    if auc < 0.5:
                        vp = 1.0 - vp
                except Exception:
                    pass
                seed_preds.append(vp)
                models.append(model)

            val_preds = np.mean(np.vstack(seed_preds), axis=0)
            oof_predictions[val_idx] = val_preds

            fold_score, metrics = ing_hubs_datathon_metric(y_val, val_preds)
            scores.append(fold_score)
            print(f"Fold {fold + 1} Score: {fold_score:.6f}")

        overall_score = oof_composite_monthwise(y, oof_predictions, ref_dates=ref_dates, last_n_months=last_n_months)
        print(f"\n{'='*50}")
        print(f"Overall OOF Composite (month-wise mean): {overall_score:.6f}")
        print(f"{'='*50}\n")
        
        self.xgb_models = models
        return oof_predictions, overall_score, models

    def train_catboost_timecv(self, X, y, params=None, ref_dates=None, last_n_months=6, seeds=None, sample_weight=None, gap_months: int = 1, time_decay_lambda: float | None = 0.2):
        """
        Train CatBoost with time-ordered month folds. Per fold, train multiple seeds and
        average their predictions. Returns OOF predictions, overall score, and list of models.

        Parameters:
        - X, y: training features/labels (pandas DataFrame/Series)
        - params: CatBoost parameters dict
        - ref_dates: reference dates aligned with X for month folds
        - last_n_months: number of last months to use for CV
        - seeds: list of seeds to use per fold; defaults to 5 seeds derived from random_state
        - sample_weight: optional array-like of weights per row (train-only)
        """
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed but required for train_catboost_timecv")

        # Defaults
        if params is None:
            params = {
                'iterations': 2000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1,
                'random_strength': 1,
                'od_type': 'Iter',
                'od_wait': 100,
                'task_type': 'CPU',
                'verbose': 100,
                'random_seed': self.random_state,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'thread_count': -1,
            }
        # Sanitize incompatible params
        p = dict(params)
        if p.get('bootstrap_type', '').lower() == 'bayesian' and 'subsample' in p:
            p.pop('subsample', None)

        # Seeds
        if seeds is None:
            seeds = [1, 11, 111, 1111, 11111]

        # Folds
        if ref_dates is None:
            raise ValueError("ref_dates are required for time-based folds in CatBoost")

        oof = np.zeros(len(X), dtype=float)
        self.cat_models = []

        for fold, (train_idx, val_idx, month_label) in enumerate(month_folds(ref_dates, last_n=last_n_months, gap=gap_months)):
            print("\n" + "="*60)
            print(f"CatBoost Fold {fold+1} | Val month: {month_label}")
            print("="*60)

            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            if sample_weight is not None:
                sw_base = np.asarray(sample_weight)[train_idx].astype(float)
            else:
                sw_base = np.ones(len(train_idx), dtype=float)
            w_time = _compute_time_decay_weights(ref_dates, pd.Period(month_label), train_idx, gap=gap_months, lam=time_decay_lambda)
            sw_tr = (sw_base * w_time).astype(float)
            sw_tr = np.nan_to_num(sw_tr, nan=1.0, posinf=1.0, neginf=1.0)
            sw_tr = np.clip(sw_tr, 1e-6, None)

            fold_models = []
            preds_seed_stack = []

            for si, seed in enumerate(seeds):
                p_seed = dict(p)
                p_seed['random_seed'] = int(seed)
                model = CatBoostClassifier(**p_seed)
                model.fit(
                    X_tr, y_tr,
                    sample_weight=sw_tr,
                    eval_set=(X_va, y_va),
                    verbose=p_seed.get('verbose', 100)
                )
                pv = model.predict_proba(X_va)[:, 1].astype(float)
                preds_seed_stack.append(pv)
                fold_models.append(model)

            # Average across seeds for this fold
            pv_mean = np.mean(np.vstack(preds_seed_stack), axis=0)
            # Flip-guard per fold
            try:
                auc = roc_auc_score(np.asarray(y_va), pv_mean)
                if auc < 0.5:
                    pv_mean = 1.0 - pv_mean
            except Exception:
                pass
            oof[val_idx] = pv_mean
            self.cat_models.extend(fold_models)

            s, m = ing_hubs_datathon_metric(y_va, pv_mean)
            print(f"Fold Score: {s:.6f} | AUC: {m['auc']:.4f} | Recall@10: {m['recall@10']:.4f} | Lift@10: {m['lift@10']:.4f}")

        overall_score = oof_composite_monthwise(y, oof, ref_dates=ref_dates, last_n_months=last_n_months)
        print(f"\n{'='*50}")
        print(f"CatBoost OOF Composite (month-wise mean): {overall_score:.6f}")
        print(f"{'='*50}\n")

        return oof, overall_score, self.cat_models

    def predict_catboost(self, X):
        """Predict by averaging probabilities across all stored CatBoost models."""
        if not self.cat_models:
            raise ValueError("CatBoost models not trained. Call train_catboost_timecv first.")
        preds = []
        for m in self.cat_models:
            preds.append(m.predict_proba(X)[:, 1].astype(float))
        return np.mean(np.vstack(preds), axis=0)

    def train_two_stage_timecv(self, X, y, ref_dates, last_n_months=6, params_A=None, params_B=None, sample_weight=None, gap_months: int = 1, time_decay_lambda: float | None = 0.2):
        """
        Two-stage training with time-ordered month folds.

        Stage-A: recall-oriented (heavier positive weight via scale_pos_weight), produce OOF pA.
        Stage-B: refine ranking using features [X, pA_oof, rank_pA, rank_boost], produce OOF pB.

        Returns: oof_pB, overall_score_B, oof_pA, overall_score_A
        """
        # Defaults
        if params_A is None:
            params_A = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 6,
                'min_child_samples': 200,
                'scale_pos_weight': 8.0,
                'reg_alpha': 0.0,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'verbose': -1,
                'n_jobs': -1
            }
        if params_B is None:
            params_B = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_child_samples': 20,
                'reg_alpha': 0.0,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'verbose': -1,
                'n_jobs': -1
            }

        # Stage-A: OOF predictions
        oof_pA = np.zeros(len(X), dtype=float)
        self.stageA_models = []

        print("\n" + "="*50)
        print("Stage-A (Recall-oriented) - Time CV")
        print("="*50)

        for fold, (train_idx, val_idx, month_label) in enumerate(month_folds(ref_dates, last_n=last_n_months, gap=gap_months)):
            print(f"\n--- Stage-A Fold {fold+1} | Val month: {month_label} ---")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Base sample weights (if provided) times time decay
            if sample_weight is not None:
                sw_base = np.asarray(sample_weight)[train_idx].astype(float)
            else:
                sw_base = np.ones(len(train_idx), dtype=float)
            w_time = _compute_time_decay_weights(ref_dates, pd.Period(month_label), train_idx, gap=gap_months, lam=time_decay_lambda)
            sw_tr = (sw_base * w_time).astype(float)

            dtrain = lgb.Dataset(X_train, label=y_train, weight=sw_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            modelA = lgb.train(
                params_A,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dtrain, dval],
                valid_names=['train','valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=100)
                ]
            )

            pA_val = modelA.predict(X_val, num_iteration=modelA.best_iteration)
            pA_val = np.asarray(pA_val, dtype=float)
            # Flip-guard per fold
            try:
                auc = roc_auc_score(np.asarray(y_val), pA_val)
                if auc < 0.5:
                    pA_val = 1.0 - pA_val
            except Exception:
                pass
            oof_pA[val_idx] = pA_val

            sA, mA = ing_hubs_datathon_metric(y_val, pA_val)
            print(f"Stage-A Fold Score: {sA:.6f} | AUC: {mA['auc']:.4f} | Recall@10: {mA['recall@10']:.4f} | Lift@10: {mA['lift@10']:.4f}")

            self.stageA_models.append(modelA)

        score_A = oof_composite_monthwise(y, oof_pA, ref_dates=ref_dates, last_n_months=last_n_months)
        print(f"\nStage-A OOF Composite (month-wise mean): {score_A:.6f}")

        # Prepare rank-based features from OOF pA
        ranks = rankdata(oof_pA, method='average')
        rank_pA = (ranks - 1) / max(1, (len(oof_pA) - 1))
        rank_boost = ranks / float(len(oof_pA))

        # Stage-B: train on [X, pA_oof, rank features]
        oof_pB = np.zeros(len(X), dtype=float)
        self.stageB_models = []

        print("\n" + "="*50)
        print("Stage-B (Ranking refinement) - Time CV")
        print("="*50)

        feat_plus = list(X.columns) + ['pA', 'rank_pA', 'rank_boost']
        self.stageB_feature_names = feat_plus

        for fold, (train_idx, val_idx, month_label) in enumerate(month_folds(ref_dates, last_n=last_n_months, gap=gap_months)):
            print(f"\n--- Stage-B Fold {fold+1} | Val month: {month_label} ---")
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            # Append OOF-based features
            X_train['pA'] = oof_pA[train_idx]
            X_val['pA'] = oof_pA[val_idx]
            X_train['rank_pA'] = rank_pA[train_idx]
            X_val['rank_pA'] = rank_pA[val_idx]
            X_train['rank_boost'] = rank_boost[train_idx]
            X_val['rank_boost'] = rank_boost[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if sample_weight is not None:
                sw_base = np.asarray(sample_weight)[train_idx].astype(float)
            else:
                sw_base = np.ones(len(train_idx), dtype=float)
            w_time = _compute_time_decay_weights(ref_dates, pd.Period(month_label), train_idx, gap=gap_months, lam=time_decay_lambda)
            sw_tr = (sw_base * w_time).astype(float)
            dtrain = lgb.Dataset(X_train[feat_plus], label=y_train, feature_name=feat_plus, weight=sw_tr)
            dval = lgb.Dataset(X_val[feat_plus], label=y_val, reference=dtrain, feature_name=feat_plus)

            modelB = lgb.train(
                params_B,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dtrain, dval],
                valid_names=['train','valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=100)
                ]
            )

            pB_val = modelB.predict(X_val[feat_plus], num_iteration=modelB.best_iteration)
            pB_val = np.asarray(pB_val, dtype=float)
            try:
                auc = roc_auc_score(np.asarray(y_val), pB_val)
                if auc < 0.5:
                    pB_val = 1.0 - pB_val
            except Exception:
                pass
            oof_pB[val_idx] = pB_val

            sB, mB = ing_hubs_datathon_metric(y_val, pB_val)
            print(f"Stage-B Fold Score: {sB:.6f} | AUC: {mB['auc']:.4f} | Recall@10: {mB['recall@10']:.4f} | Lift@10: {mB['lift@10']:.4f}")

            self.stageB_models.append(modelB)

        score_B = oof_composite_monthwise(y, oof_pB, ref_dates=ref_dates, last_n_months=last_n_months)
        print(f"\nStage-B OOF Composite (month-wise mean): {score_B:.6f}")

        return oof_pB, score_B, oof_pA, score_A

    def predict_two_stage(self, X_test):
        """
        Predict with two-stage head: average Stage-A across folds to get pA_test,
        then apply Stage-B models on [X_test, pA_test] and average.
        """
        if not self.stageA_models or not self.stageB_models:
            raise ValueError("Two-stage models not trained. Call train_two_stage_timecv first.")

        # Stage-A test: average across folds
        pA_stack = []
        for m in self.stageA_models:
            p = m.predict(X_test, num_iteration=m.best_iteration)
            pA_stack.append(np.asarray(p, dtype=float))
        pA_test = np.mean(np.vstack(pA_stack), axis=0)

        # Stage-B test: append pA_test
        X_te_ext = X_test.copy()
        X_te_ext['pA'] = pA_test
        # rank_pA for test: rank within test set
        ranks_te = rankdata(pA_test, method='average')
        X_te_ext['rank_pA'] = (ranks_te - 1) / max(1, (len(pA_test) - 1))

        pB_stack = []
        feat_plus = self.stageB_feature_names if self.stageB_feature_names is not None else list(X_test.columns) + ['pA', 'rank_pA']
        for m in self.stageB_models:
            p = m.predict(X_te_ext[feat_plus], num_iteration=m.best_iteration)
            pB_stack.append(np.asarray(p, dtype=float))
        pB_test = np.mean(np.vstack(pB_stack), axis=0)

        return pB_test, pA_test
    
    def predict(self, X):
        """
        Generate predictions using trained models
        """
        predictions = np.zeros(len(X))
        
        for model in self.models:
            if isinstance(model, lgb.Booster):
                fold_pred = np.asarray(model.predict(X, num_iteration=model.best_iteration), dtype=float)
                predictions += fold_pred
            else:  # XGBoost
                fold_pred = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
                predictions += fold_pred
        
        predictions /= len(self.models)
        return predictions
    
    def calibrate_predictions(self, y_true, y_pred, X_test):
        """
        Calibrate probabilities to improve Lift@10% and Recall@10%
        Uses isotonic regression for better tail behavior
        """
        from sklearn.isotonic import IsotonicRegression
        
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(y_pred, y_true)
        
        # Apply to test predictions
        test_pred = self.predict(X_test)
        calibrated_pred = iso_reg.transform(test_pred)
        
        return calibrated_pred
    
    def get_feature_importance(self, feature_names=None):
        """
        Get average feature importance across folds.

        Robust behavior:
        - If provided feature_names length doesn't match model's features, fall back to the
          model's own feature names (LightGBM: Booster.feature_name(); XGBoost: feature_names_in_ or f0..fn).
        - Merge importance across folds with an outer join to include features that appear
          in some folds but not others, filling missing importances with 0 before aggregating.
        """
        if not self.models:
            return None

        importance_df = None

        for i, model in enumerate(self.models):
            if isinstance(model, lgb.Booster):
                imp_vals = model.feature_importance(importance_type='gain')
                # Prefer model's feature names if lengths mismatch or names not provided
                if feature_names is None or len(feature_names) != len(imp_vals):
                    names = model.feature_name()
                else:
                    names = list(feature_names)
                imp = pd.DataFrame({
                    'feature': names,
                    f'importance_fold_{i}': imp_vals
                })
            else:  # XGBoost (sklearn wrapper)
                imp_vals = getattr(model, 'feature_importances_', None)
                if imp_vals is None:
                    continue
                # Determine names robustly
                if feature_names is not None and len(feature_names) == len(imp_vals):
                    names = list(feature_names)
                elif hasattr(model, 'feature_names_in_'):
                    names = list(model.feature_names_in_)
                else:
                    names = [f'f{j}' for j in range(len(imp_vals))]
                imp = pd.DataFrame({
                    'feature': names,
                    f'importance_fold_{i}': imp_vals
                })

            if importance_df is None:
                importance_df = imp
            else:
                # Outer merge to retain all features seen in any fold
                importance_df = importance_df.merge(imp, on='feature', how='outer')

        if importance_df is None or importance_df.empty:
            return None

        # Calculate mean/std importance across folds; fill NaNs as 0 for missing features per fold
        importance_df = importance_df.fillna(0.0)
        importance_cols = [c for c in importance_df.columns if 'importance_fold_' in c]
        if not importance_cols:
            return importance_df
        importance_df['importance_mean'] = importance_df[importance_cols].mean(axis=1)
        importance_df['importance_std'] = importance_df[importance_cols].std(axis=1)

        return importance_df.sort_values('importance_mean', ascending=False)


# Example usage:
"""
pipeline = ChurnModelingPipeline(n_folds=5)

# Train LightGBM
oof_lgb, score_lgb = pipeline.train_lightgbm(X_train, y_train)

# Train XGBoost  
oof_xgb, score_xgb, xgb_models = pipeline.train_xgboost(X_train, y_train)

# Ensemble predictions (weighted average)
oof_ensemble = 0.6 * oof_lgb + 0.4 * oof_xgb
final_score, _ = ing_hubs_datathon_metric(y_train, oof_ensemble)

# Make test predictions
test_pred_lgb = pipeline.predict(X_test)
test_pred_xgb = np.mean([m.predict_proba(X_test)[:, 1] for m in xgb_models], axis=0)
test_pred_ensemble = 0.6 * test_pred_lgb + 0.4 * test_pred_xgb

# Calibrate
calibrated_pred = pipeline.calibrate_predictions(y_train, oof_ensemble, X_test)
"""