"""
ING Hubs Türkiye Datathon - Complete Workflow
Optimized for competition metric: 40% Gini + 30% Recall@10% + 30% Lift@10%

This notebook provides a complete end-to-end solution
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import argparse
from sklearn.metrics import roc_auc_score

# Import Optuna-optimized parameters (after running hyperparameter_optimizer.py)
# If optimized_params.py doesn't exist, comment out this line and use default params below
try:
    # Expecting keys LGB_BEST and XGB_BEST in optimized_params.py
    from optimized_params import LGB_BEST, XGB_BEST
    OPT_AVAILABLE = True
    USE_OPTIMIZED_PARAMS = True
    print("✓ Loaded Optuna-optimized hyperparameters")
except ImportError:
    OPT_AVAILABLE = False
    USE_OPTIMIZED_PARAMS = False
    print("⚠ optimized_params.py not found - using default parameters")
    print("  Run hyperparameter_optimizer.py first to generate optimized parameters")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

parser = argparse.ArgumentParser(description="ING Hubs Türkiye Datathon - Complete Workflow")
parser.add_argument('--adv-filter', action='store_true', help='Enable adversarial feature filtering (train vs test)')
parser.add_argument('--adv-drop-k', type=int, default=40, help='Number of highest drift features to drop (e.g., 30-50)')
parser.add_argument('--adv-auc-thresh', type=float, default=0.70, help='Target upper bound for domain AUC after filtering')
parser.add_argument('--adv-w-low', type=float, default=0.8, help='Lower clip for sample weights')
parser.add_argument('--adv-w-high', type=float, default=1.2, help='Upper clip for sample weights')
parser.add_argument('--adv-diag-failfast', action='store_true', default=False, help='Fail the run if adversarial diagnostics domain AUC exceeds threshold (uses --adv-auc-thresh)')
parser.add_argument('--cat-seeds', type=int, default=5, help='Number of seeds to average per CatBoost fold')
parser.add_argument('--cat-depth', type=int, default=None, help='Override CatBoost tree depth')
parser.add_argument('--with-stacker', action='store_true', help='Enable level-2 stacker using base OOF predictions')
parser.add_argument('--force-meta', action='store_true', help='Force including meta stacker stream even if it underperforms gating threshold')
parser.add_argument('--with-ftt', action='store_true', help='Enable FT-Transformer (PyTorch) head and include in blend')
parser.add_argument('--pseudo-label', action='store_true', help='Enable pseudo-labeling retrain for CatBoost and XGBoost')
parser.add_argument('--pl-pos-top', type=float, default=0.025, help='Top fraction for positive pseudo-labels (on calibrated test preds)')
parser.add_argument('--pl-neg-bottom', type=float, default=0.40, help='Bottom fraction for negative pseudo-labels (on calibrated test preds)')
parser.add_argument('--pl-improve-min', type=float, default=0.003, help='Minimum composite lift to accept pseudo-labeled model')
# Compatibility flags requested by users
parser.add_argument('--models', nargs='+', default=['cat', 'lgb', 'xgb', 'two'],
                    help="Models to train/use in blend. Choices include: lgb xgb cat two ftt meta")
parser.add_argument('--scaler', choices=['none', 'standard', 'minmax'], default='none',
                    help="Feature scaler to use (tree models ignore; meta may standardize internally).")
parser.add_argument('--last-n', dest='last_n', type=int, default=6,
                    help='Number of most recent months to use for time-based CV and metrics')
parser.add_argument('--calib', choices=['auto', 'isotonic', 'beta', 'none'], default='auto',
                    help='Calibration strategy: auto compares isotonic vs beta; none disables calibration')
parser.add_argument('--gamma-grid', type=str, default='0.85,0.90,0.95,1.00,1.05',
                    help='Comma-separated gamma values for calibration power adjustment')
parser.add_argument('--no-interactions', action='store_true',
                    help='Skip feature interaction engineering step')
parser.add_argument('--iter-adv', action='store_true',
                    help='Enable iterative adversarial filtering loop to push domain AUC ≤ target before modeling')
parser.add_argument('--iter-adv-k', type=int, default=5,
                    help='Number of top drift features to try dropping each step (K); halves if OOF drop too high')
parser.add_argument('--iter-adv-target', type=float, default=0.75,
                    help='Target upper bound for domain AUC for iterative adversarial loop')
parser.add_argument('--iter-adv-max-drop', type=float, default=0.02,
                    help='Maximum allowed decrease in OOF composite during iterative adversarial loop')
parser.add_argument('--use-optimized-params', dest='use_optimized_params', action='store_true',
                    help='Force using optimized_params.py hyperparameters if available')
parser.add_argument('--no-optimized-params', dest='use_optimized_params', action='store_false',
                    help='Force ignore optimized_params.py and use built-in defaults')
parser.set_defaults(use_optimized_params=None)
args = parser.parse_args()

print("Loading data...")

# Apply CLI overrides and compatibility mappings
if args.use_optimized_params is not None:
    if args.use_optimized_params and not OPT_AVAILABLE:
        print("⚠ --use-optimized-params set but optimized_params.py not available; continuing with defaults")
        USE_OPTIMIZED_PARAMS = False
    else:
        USE_OPTIMIZED_PARAMS = bool(args.use_optimized_params)

# Map --models to optional heads
if args.models:
    # convenience: enable flags if listed in models
    if ('meta' in args.models) and (not args.with_stacker):
        args.with_stacker = True
    if ('ftt' in args.models) and (not args.with_ftt):
        args.with_ftt = True

# Parse gamma grid for calibration
try:
    _gamma_vals = tuple(float(x.strip()) for x in args.gamma_grid.split(',') if x.strip())
    if len(_gamma_vals) > 0:
        _CALIB_GAMMA_GRID = _gamma_vals
    else:
        _CALIB_GAMMA_GRID = (0.85, 0.90, 0.95, 1.00, 1.05)
except Exception:
    _CALIB_GAMMA_GRID = (0.85, 0.90, 0.95, 1.00, 1.05)

LN = int(max(1, args.last_n))
if args.scaler != 'none':
    print(f"Scaler requested: {args.scaler} (note: tree models ignore; stacker scales internally if needed)")

def _resolve_data_path(name: str) -> str:
    candidates = [
        name,
        os.path.join('data', 'raw', name),
        os.path.join('data', name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {name}. Looked in: {', '.join(candidates)}")

customer_history = pd.read_csv(_resolve_data_path('customer_history.csv'))
customers = pd.read_csv(_resolve_data_path('customers.csv'))
reference_data = pd.read_csv(_resolve_data_path('reference_data.csv'))
reference_data_test = pd.read_csv(_resolve_data_path('reference_data_test.csv'))

print(f"Customer history shape: {customer_history.shape}")
print(f"Customers shape: {customers.shape}")
print(f"Reference train shape: {reference_data.shape}")
print(f"Reference test shape: {reference_data_test.shape}")

# Check churn distribution
print(f"\nChurn rate: {reference_data['churn'].mean():.4f}")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

# Import the feature engineering classes
from src.features.feature_engineering import ChurnFeatureEngineering
from src.features.advanced_features import AdvancedFeatureEngineering

fe = ChurnFeatureEngineering()
afe = AdvancedFeatureEngineering()

# Process each reference date separately
train_features_list = []

print("\n" + "="*60)
print("CREATING TRAINING FEATURES")
print("="*60)

for ref_date in reference_data['ref_date'].unique():
    print(f"\nProcessing ref_date: {ref_date}")
    
    # Get customers for this ref_date
    ref_customers = reference_data[reference_data['ref_date'] == ref_date]['cust_id'].unique()
    
    # Filter history up to ref_date
    ref_date_dt = pd.to_datetime(ref_date)
    history_subset = customer_history[
        (customer_history['cust_id'].isin(ref_customers)) &
        (pd.to_datetime(customer_history['date']) <= ref_date_dt)
    ].copy()
    
    customers_subset = customers[customers['cust_id'].isin(ref_customers)].copy()

    # Create basic features
    features = fe.create_all_features(history_subset, customers_subset, ref_date)

    # Create advanced features (RFM, behavioral change, lifecycle, time-based)
    advanced_features = afe.create_all_advanced_features(history_subset, customers_subset, ref_date)

    # Merge basic and advanced features
    features = features.merge(advanced_features, on='cust_id', how='left')
    features['ref_date'] = ref_date

    train_features_list.append(features)

# Combine all training features
train_features = pd.concat(train_features_list, axis=0, ignore_index=True)
print(f"\nTotal training features shape: {train_features.shape}")

# Exclude any religion-related features from modeling inputs
drop_cols_train = [c for c in train_features.columns if 'religion' in c.lower()]
if drop_cols_train:
    train_features = train_features.drop(columns=drop_cols_train, errors='ignore')
    print(f"Dropped religion columns from train features: {len(drop_cols_train)}")

# Count and display advanced features added
advanced_feature_cols = [col for col in train_features.columns
                         if any(prefix in col for prefix in ['rfm_', 'behavior_', 'lifecycle_', 'time_'])]
print(f"Advanced features added: {len(advanced_feature_cols)}")
print(f"  - RFM features: {len([c for c in advanced_feature_cols if c.startswith('rfm_')])}")
print(f"  - Behavioral features: {len([c for c in advanced_feature_cols if c.startswith('behavior_')])}")
print(f"  - Lifecycle features: {len([c for c in advanced_feature_cols if c.startswith('lifecycle_')])}")
print(f"  - Time-based features: {len([c for c in advanced_feature_cols if c.startswith('time_')])}")

# Merge with labels
train_data = train_features.merge(
    reference_data[['cust_id', 'ref_date', 'churn']], 
    on=['cust_id', 'ref_date'], 
    how='left'
)

print(f"Training data with labels shape: {train_data.shape}")
print(f"Missing labels: {train_data['churn'].isna().sum()}")

# ============================================================================
# STEP 3: CREATE TEST FEATURES
# ============================================================================

print("\n" + "="*60)
print("CREATING TEST FEATURES")
print("="*60)

test_features_list = []

for ref_date in reference_data_test['ref_date'].unique():
    print(f"\nProcessing test ref_date: {ref_date}")

    ref_customers = reference_data_test[reference_data_test['ref_date'] == ref_date]['cust_id'].unique()

    ref_date_dt = pd.to_datetime(ref_date)
    history_subset = customer_history[
        (customer_history['cust_id'].isin(ref_customers)) &
        (pd.to_datetime(customer_history['date']) <= ref_date_dt)
    ].copy()

    customers_subset = customers[customers['cust_id'].isin(ref_customers)].copy()

    # Create basic features
    features = fe.create_all_features(history_subset, customers_subset, ref_date)

    # Create advanced features (RFM, behavioral change, lifecycle, time-based)
    advanced_features = afe.create_all_advanced_features(history_subset, customers_subset, ref_date)

    # Merge basic and advanced features
    features = features.merge(advanced_features, on='cust_id', how='left')
    features['ref_date'] = ref_date

    test_features_list.append(features)

test_features = pd.concat(test_features_list, axis=0, ignore_index=True)
print(f"\nTotal test features shape: {test_features.shape}")

# Exclude any religion-related features from test inputs as well
drop_cols_test = [c for c in test_features.columns if 'religion' in c.lower()]
if drop_cols_test:
    test_features = test_features.drop(columns=drop_cols_test, errors='ignore')
    print(f"Dropped religion columns from test features: {len(drop_cols_test)}")

# Verify advanced features are present in test set
advanced_feature_cols_test = [col for col in test_features.columns
                              if any(prefix in col for prefix in ['rfm_', 'behavior_', 'lifecycle_', 'time_'])]
print(f"Advanced features in test set: {len(advanced_feature_cols_test)}")

# ============================================================================
# STEP 4: PREPARE DATA FOR MODELING
# ============================================================================

print("\n" + "="*60)
print("PREPARING DATA FOR MODELING")
print("="*60)

# Separate features and target (keep only numeric dtypes to avoid object columns like 'rfm_segment')
non_feature_cols = {'cust_id', 'ref_date', 'churn'}
numeric_cols = [
    col for col in train_data.columns
    if col not in non_feature_cols and pd.api.types.is_numeric_dtype(train_data[col])
]
feature_cols = numeric_cols

X_train = train_data[feature_cols]
y_train = train_data['churn']

X_test = test_features[feature_cols]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
non_numeric_cols = [c for c in train_data.columns if c not in non_feature_cols and not pd.api.types.is_numeric_dtype(train_data[c])]
if non_numeric_cols:
    print(f"Non-numeric columns excluded from modeling: {non_numeric_cols}")

# (removed here; moved after feature interactions to ensure exact modeling feature set)

# Tracking for adversarial AUCs to report later
adv_pre_before = None
adv_pre_after = None
adv_post_before = None
adv_post_after = None
domain_auc_summary = None

# --------------------------------------------------------------------------
# OPTIONAL: FAST ADVERSARIAL FILTER (train vs test drift removal) BEFORE MODELING
if args.adv_filter:
    try:
        from adversarial_filter import run_fast_adversarial_filter  # repo-root module
    except Exception as _e:
        print(f"[ADV] Unable to import adversarial_filter utility: {_e}")
    else:
        print("\n" + "="*60)
        print("ADVERSARIAL FILTER (train vs test) BEFORE MODELING")
        print("="*60)
        kdrop = int(max(0, args.adv_drop_k))
        try:
            adv_res = run_fast_adversarial_filter(X_train, X_test, k=kdrop)
            # Safely extract metrics
            _before = adv_res.get('domain_auc_before', None)
            _after = adv_res.get('domain_auc_after', None)
            before = float(_before) if isinstance(_before, (int, float, np.floating)) else float('nan')
            after = float(_after) if isinstance(_after, (int, float, np.floating)) else float('nan')
            _dropped = adv_res.get('dropped_features', [])
            dropped = list(_dropped) if isinstance(_dropped, (list, tuple, pd.Series, np.ndarray)) else []
            print(f"[ADV] Domain AUC: {before:.3f} → {after:.3f}; dropped={len(dropped)}")
            try:
                adv_pre_before = float(before)
            except Exception:
                adv_pre_before = None
            try:
                adv_pre_after = float(after)
            except Exception:
                adv_pre_after = None

            # Replace X frames with reduced versions by dropping listed columns
            if dropped:
                X_train = X_train.drop(columns=dropped, errors='ignore')
                X_test = X_test.drop(columns=dropped, errors='ignore')
                print(f"[ADV] Reduced feature set: X_train={X_train.shape}, X_test={X_test.shape}")

            # Save report CSV
            try:
                os.makedirs(os.path.join('outputs', 'reports'), exist_ok=True)
                imp_df = adv_res.get('importances')
                if isinstance(imp_df, pd.DataFrame) and not imp_df.empty:
                    imp_df = imp_df.copy()
                    imp_df['rank'] = np.arange(1, len(imp_df) + 1)
                    imp_df['dropped_flag'] = imp_df['feature'].isin(dropped).astype(int)
                    cols = ['rank', 'feature', 'gain_mean', 'gain_std', 'dropped_flag']
                    # Ensure all required columns exist
                    for col in cols:
                        if col not in imp_df.columns:
                            imp_df[col] = 0
                    imp_df[cols].to_csv(os.path.join('outputs', 'reports', 'adversarial_drop.csv'), index=False)
                    print("[ADV] Wrote outputs/reports/adversarial_drop.csv")
            except Exception as _e2:
                print(f"[ADV] Failed to write adversarial report: {_e2}")
        except Exception as _e3:
            print(f"[ADV] Adversarial filter failed: {_e3}")

# ============================================================================
# FEATURE INTERACTION ENGINEERING
# ============================================================================

def add_feature_interactions(X_train, X_test):
    """
    Create interaction features between top predictive features.

    Generates multiplicative and ratio-based interactions for the top 5 features:
    - active_product_category_nbr_mean_12m
    - cc_transaction_all_cnt_sum_1m
    - mobile_eft_all_amt_trend_mean
    - age
    - tenure

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features

    Returns:
    --------
    X_train, X_test : pd.DataFrame
        DataFrames with added interaction features
    """
    print("\n" + "="*60)
    print("CREATING FEATURE INTERACTIONS")
    print("="*60)

    # Top 5 features to create interactions from
    top_features = [
        'active_product_category_nbr_mean_12m',
        'cc_transaction_all_cnt_sum_1m',
        'mobile_eft_all_amt_trend_mean',
        'age',
        'tenure'
    ]

    # Filter to only features that exist in the dataframe
    existing_features = [f for f in top_features if f in X_train.columns]

    print(f"Found {len(existing_features)} out of {len(top_features)} features in dataset")
    print(f"Creating interactions for: {existing_features}")

    interaction_count = 0

    # Create pairwise interactions
    for i in range(len(existing_features)):
        for j in range(i + 1, len(existing_features)):
            feat1 = existing_features[i]
            feat2 = existing_features[j]

            # Multiplication interaction: feat1 * feat2
            interaction_name_mult = f"{feat1}_X_{feat2}"
            X_train[interaction_name_mult] = X_train[feat1] * X_train[feat2]
            X_test[interaction_name_mult] = X_test[feat1] * X_test[feat2]
            interaction_count += 1

            # Ratio interaction: feat1 / (feat2 + 1)
            interaction_name_ratio1 = f"{feat1}_DIV_{feat2}"
            X_train[interaction_name_ratio1] = X_train[feat1] / (X_train[feat2] + 1)
            X_test[interaction_name_ratio1] = X_test[feat1] / (X_test[feat2] + 1)
            interaction_count += 1

            # Ratio interaction: feat2 / (feat1 + 1)
            interaction_name_ratio2 = f"{feat2}_DIV_{feat1}"
            X_train[interaction_name_ratio2] = X_train[feat2] / (X_train[feat1] + 1)
            X_test[interaction_name_ratio2] = X_test[feat2] / (X_test[feat1] + 1)
            interaction_count += 1

            print(f"  Created interactions for {feat1} x {feat2}")

    # Handle any infinity values created by interactions
    X_train = X_train.replace([np.inf, -np.inf], -999)
    X_test = X_test.replace([np.inf, -np.inf], -999)

    print(f"\nTotal interactions created: {interaction_count}")
    print(f"New X_train shape: {X_train.shape}")
    print(f"New X_test shape: {X_test.shape}")

    return X_train, X_test

# Apply feature interactions
if getattr(args, 'no_interactions', False):
    print("\n-- Skipping feature interaction engineering due to --no-interactions --")
else:
    X_train, X_test = add_feature_interactions(X_train, X_test)

# Update feature columns list to include all columns
feature_cols = list(X_train.columns)

# --------------------------------------------------------------------------
# Adversarial diagnostics on final feature space (train vs test after interactions)
from src.models.modeling_pipeline import adversarial_diagnostics
print("\n" + "="*60)
print("ADVERSARIAL DIAGNOSTICS (train vs test) AFTER INTERACTIONS")
print("="*60)
try:
    adversarial_diagnostics(X_train, X_test, threshold=float(args.adv_auc_thresh))
except RuntimeError as e:
    msg = str(e)
    if args.adv_diag_failfast:
        # Respect fail-fast only if explicitly requested
        print(msg)
        raise
    else:
        print(msg)
        print("Proceeding despite high domain AUC; consider enabling --adv-filter and/or increasing --adv-drop-k.")

# Optional: Iterative adversarial feature dropping to reach target domain AUC
if getattr(args, 'iter_adv', False):
    print("\n" + "="*60)
    print("ITERATIVE ADVERSARIAL FILTERING (push domain AUC ≤ target)")
    print("="*60)
    try:
        from src.features.adversarial_validate import iterative_adversarial_filter as _iter_adv
        iter_res = _iter_adv(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            ref_dates=pd.Series(train_data['ref_date']).astype(str),
            last_n=LN,
            base_models=None,
            K_to_drop_each_step=int(args.iter_adv_k),
            target_auc=float(args.iter_adv_target),
            max_oof_drop=float(args.iter_adv_max_drop),
            seed=42,
        )
        keep_cols = list(iter_res.get('keep_columns', []))
        dropped_cols = list(iter_res.get('dropped_features', []))
        print(f"[IterADV] Dropped {len(dropped_cols)} features; keeping {len(keep_cols)}")
        if keep_cols:
            X_train = X_train[keep_cols].copy()
            X_test = X_test[keep_cols].copy()
            print(f"[IterADV] Reduced feature set: X_train={X_train.shape}, X_test={X_test.shape}")
        # Save report
        try:
            os.makedirs(os.path.join('outputs', 'reports'), exist_ok=True)
            import json as _json
            with open(os.path.join('outputs', 'reports', 'iter_adv_report.json'), 'w', encoding='utf-8') as _f:
                _json.dump(iter_res, _f, indent=2)
            print("[IterADV] Wrote outputs/reports/iter_adv_report.json")
        except Exception as _e:
            print(f"[IterADV] Warning: could not save iter_adv_report.json: {_e}")
        # Track post-iterative domain AUC for final summary
        try:
            _val = iter_res.get('domain_auc_final', None)
            if _val is not None:
                adv_post_after = float(_val)
        except Exception:
            pass
    except Exception as _e:
        print(f"[IterADV] Failed to run: {_e}")

# ============================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as _roc_auc

print("\n" + "="*60)
print("COMPUTING DOMAIN WEIGHTS (train≈test) ON FINAL FEATURES")
print("="*60)

# Ensure ref_date alignment and year split (train_data aligns with X_train rows)
ref_train_dt = pd.to_datetime(train_data['ref_date'])
train_years = ref_train_dt.dt.year.values
mask_1718 = (train_years == 2017) | (train_years == 2018)

# Concatenate without target; 0 for train (2017-2018), 1 for test (2019)
X_dom_train = X_train.loc[mask_1718]
y_dom_train = np.zeros(len(X_dom_train), dtype=int)
X_dom_test = X_test
y_dom_test = np.ones(len(X_dom_test), dtype=int)

X_dom = pd.concat([X_dom_train, X_dom_test], axis=0)
y_dom = np.concatenate([y_dom_train, y_dom_test], axis=0)

# Adversarial filter path (optional)
if args.adv_filter:
    print("Adversarial filter: training LightGBM domain classifier and dropping drift features...")
    try:
        import lightgbm as lgb
    except Exception as e:
        raise RuntimeError("LightGBM is required for --adv-filter path") from e

    # Holdout AUC before
    try:
        X_trd, X_vad, y_trd, y_vad = train_test_split(X_dom, y_dom, test_size=0.2, random_state=42, stratify=y_dom)
        dtr = lgb.Dataset(X_trd, label=y_trd)
        dva = lgb.Dataset(X_vad, label=y_vad, reference=dtr)
        params = {
            'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
            'num_leaves': 63, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 1, 'min_data_in_leaf': 50, 'verbose': -1
        }
        m0 = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva], valid_names=['valid'], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        # Prefer tracked best AUC; fallback to computing AUC from predictions
        if 'valid' in m0.best_score and 'auc' in m0.best_score['valid']:
            auc_before = float(m0.best_score['valid']['auc'])
        else:
            _p = np.asarray(m0.predict(X_vad), dtype=float)
            auc_before = _roc_auc(y_vad, _p)  # type: ignore[arg-type]
    except Exception:
        auc_before = float('nan')
    print(f"Domain AUC BEFORE filtering: {auc_before:.4f}")
    try:
        adv_post_before = float(auc_before)
    except Exception:
        adv_post_before = None

    # Train on all and drop top-K by gain importance
    d_all = lgb.Dataset(X_dom, label=y_dom)
    m_all = lgb.train(params, d_all, num_boost_round=600)
    gain = m_all.feature_importance(importance_type='gain')
    feat_names = np.asarray(X_dom.columns)
    order = np.argsort(-gain)
    k = int(max(0, args.adv_drop_k))
    drop_cols = feat_names[order[:k]].tolist()
    if drop_cols and not getattr(args, 'iter_adv', False):
        print(f"Dropping top-{k} drift features (by gain):")
        for c in drop_cols[:min(k, 30)]:
            print("  -", c)
        X_train = X_train.drop(columns=drop_cols, errors='ignore')
        X_test = X_test.drop(columns=drop_cols, errors='ignore')
        X_dom_train = X_dom_train.drop(columns=drop_cols, errors='ignore')
        X_dom_test = X_dom_test.drop(columns=drop_cols, errors='ignore')
        X_dom = pd.concat([X_dom_train, X_dom_test], axis=0)

    # Refit domain model post-filter and compute AUC
    X_trd, X_vad, y_trd, y_vad = train_test_split(X_dom, y_dom, test_size=0.2, random_state=42, stratify=y_dom)
    dtr = lgb.Dataset(X_trd, label=y_trd)
    dva = lgb.Dataset(X_vad, label=y_vad, reference=dtr)
    m1 = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva], valid_names=['valid'], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    if 'valid' in m1.best_score and 'auc' in m1.best_score['valid']:
        auc_after = float(m1.best_score['valid']['auc'])
    else:
        _p2 = np.asarray(m1.predict(X_vad), dtype=float)
        auc_after = _roc_auc(y_vad, _p2)  # type: ignore[arg-type]
    print(f"Domain AUC AFTER filtering:  {auc_after:.4f} (target ≤ {args.adv_auc_thresh:.2f})")
    try:
        adv_post_after = float(auc_after)
    except Exception:
        adv_post_after = None
    try:
        domain_auc_summary = float(auc_after)
    except Exception:
        pass
    if not np.isnan(auc_after) and auc_after > args.adv_auc_thresh:
        print("Warning: Domain AUC remains above threshold; consider increasing --adv-drop-k.")

    # Compute training sample weights from domain probs on training rows (proba of is_test)
    from scipy.special import expit
    p_train = np.asarray(m1.predict(X_train), dtype=float)
    w_train = np.clip(expit(-2.0 * (p_train - 0.5)), args.adv_w_low, args.adv_w_high)
    print(f"Adversarial sample weights: min={w_train.min():.4f}, max={w_train.max():.4f}, mean={w_train.mean():.4f}")

else:
    # Baseline logistic regression-based weights (previous behavior)
    from sklearn.linear_model import LogisticRegression
    try:
        X_trd, X_vad, y_trd, y_vad = train_test_split(
            X_dom, y_dom, test_size=0.2, random_state=42, stratify=y_dom
        )
        dom_probe = LogisticRegression(max_iter=1000)
        dom_probe.fit(X_trd, y_trd)
        auc_dom = _roc_auc(y_vad, dom_probe.predict_proba(X_vad)[:, 1])
        print(f"Domain classifier holdout AUC (train vs test): {auc_dom:.4f} (closer to 0.5 is better)")
        try:
            domain_auc_summary = float(auc_dom)
        except Exception:
            pass
    except Exception as _e:
        print(f"Could not compute domain holdout AUC: {_e}")

    dom_clf = LogisticRegression(max_iter=1000)
    dom_clf.fit(X_dom, y_dom)
    w_prob = dom_clf.predict_proba(X_train)[:, 1].astype(float)
    prob_safe = np.clip(w_prob, 1e-6, 1.0 - 1e-6)
    odds = prob_safe / (1.0 - prob_safe)
    scale = odds.mean() if np.isfinite(odds).all() and odds.size > 0 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    w_train = np.clip(odds / scale, 0.5, 2.0)

    print(f"Domain weights: min={w_train.min():.4f}, max={w_train.max():.4f}, mean={w_train.mean():.4f}")
    if np.allclose(w_train, w_train.mean(), atol=1e-6) or (w_train.max() - w_train.min() < 1e-3):
        try:
            coef_series = pd.Series(dom_clf.coef_[0], index=X_train.columns)
            top = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)[:15]
            print("Domain model seems flat; top |coef| features:")
            for fname, val in top.items():
                print(f"  {fname}: {val:.6f}")
        except Exception:
            print("Domain model seems flat; could not extract coefficients.")

# ============================================================================
# STEP 5: TRAIN MODELS
# ============================================================================

from src.models.modeling_pipeline import ChurnModelingPipeline
from src.models.modeling_pipeline import oof_composite_monthwise
from src.models.modeling_pipeline import ing_hubs_datathon_metric as _metric_fn

print("\n" + "="*60)
print("TRAINING LIGHTGBM MODEL")
print("="*60)

# Initialize pipeline
pipeline_lgb = ChurnModelingPipeline(n_folds=5, random_state=42)

# Placeholders for OOF and scores
oof_lgb = None; score_lgb = None
oof_xgb = None; score_xgb = None; xgb_models = []
oof_cat = None; score_cat = None; cat_models = []
oof_two_stage = None; score_two_stage = None; oof_two_stage_B = None; oof_two_stage_A = None

# LightGBM parameters - Use Optuna-optimized if available, otherwise use defaults
if 'lgb' in args.models:
    if USE_OPTIMIZED_PARAMS:
        lgb_params = LGB_BEST.copy()
        print("Using Optuna-optimized LightGBM parameters")
    else:
        # Default LightGBM parameters
        lgb_params = {
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
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        print("Using default LightGBM parameters")

    # Train LightGBM (month-based folds on last N months)
    oof_lgb, score_lgb = pipeline_lgb.train_lightgbm(
        X_train, y_train, lgb_params, ref_dates=train_data['ref_date'], last_n_months=LN, sample_weight=w_train
    )

    # Flip-guard on LGB OOF
    from sklearn.metrics import roc_auc_score
    auc_lgb = roc_auc_score(y_train, oof_lgb)
    if auc_lgb < 0.5:
        print("Flip-guard: Inverting LightGBM OOF predictions (AUC < 0.5)")
        oof_lgb = 1.0 - oof_lgb
        auc_lgb = roc_auc_score(y_train, oof_lgb)
    # Recompute LGB score post flip-guard
    score_lgb = oof_composite_monthwise(y_train, oof_lgb, ref_dates=train_data['ref_date'], last_n_months=LN)

# Helper: print month-wise metrics for drift/flip detection
def _print_monthwise_metrics(label, y, oof, ref_dates, last_n=6):
    m = pd.to_datetime(ref_dates).dt.to_period('M')
    months = sorted(m.unique())[-last_n:]
    print(f"\n{label} - per-month validation metrics:")
    print("month    |   AUC   | Recall@10 | Lift@10 | Composite")
    for vm in months:
        mask = (m.values == vm)
        if not np.any(mask):
            continue
        y_v = np.asarray(y)[mask]
        p_v = np.asarray(oof, dtype=float)[mask]
        # Flip-guard per month
        try:
            auc = roc_auc_score(y_v, p_v)
            if auc < 0.5:
                p_v = 1.0 - p_v
        except Exception:
            pass
        s, met = _metric_fn(y_v, p_v)
        print(f"{str(vm):<8} | {met['auc']:.4f} |   {met['recall@10']:.4f}   |  {met['lift@10']:.4f} | {s:.6f}")

if oof_lgb is not None:
    _print_monthwise_metrics("LightGBM", y_train, oof_lgb, train_data['ref_date'], last_n=LN)

print("\n" + "="*60)
print("TRAINING XGBOOST MODEL")
print("="*60)

# XGBoost parameters - Use Optuna-optimized if available, otherwise use defaults
if 'xgb' in args.models:
    if USE_OPTIMIZED_PARAMS:
        xgb_params = XGB_BEST.copy()
        print("Using Optuna-optimized XGBoost parameters")
    else:
        # Default XGBoost parameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'tree_method': 'hist'
        }
        print("Using default XGBoost parameters")

    # Train XGBoost (month-based folds on last N months)
    oof_xgb, score_xgb, xgb_models = pipeline_lgb.train_xgboost(
        X_train, y_train, xgb_params, ref_dates=train_data['ref_date'], last_n_months=LN, sample_weight=w_train
    )

    # Flip-guard on XGB OOF
    auc_xgb = roc_auc_score(y_train, oof_xgb)
    if auc_xgb < 0.5:
        print("Flip-guard: Inverting XGBoost OOF predictions (AUC < 0.5)")
        oof_xgb = 1.0 - oof_xgb
        auc_xgb = roc_auc_score(y_train, oof_xgb)
    # Recompute XGB score post flip-guard using month-wise composite
    score_xgb = oof_composite_monthwise(y_train, oof_xgb, ref_dates=train_data['ref_date'], last_n_months=LN)
    _print_monthwise_metrics("XGBoost", y_train, oof_xgb, train_data['ref_date'], last_n=LN)

print("\n" + "="*60)
print("TRAINING CATBOOST MODEL (time-CV, seeds/fold)")
print("="*60)

if 'cat' in args.models:
    # Prepare CatBoost parameters with overrides; multi-seed rank averaging
    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'depth': 6 if args.cat_depth is None else int(args.cat_depth),
        'learning_rate': 0.06,
        'l2_leaf_reg': 6,
        'iterations': 4000,
        'od_type': 'Iter',
        'od_wait': 200,
        'rsm': 0.8,
        'border_count': 128,
        'thread_count': -1,
        'verbose': 100,
        'task_type': 'CPU',
    }

    # Train CatBoost multi-seed with time folds; rank-average within fold
    from src.models.modeling_pipeline import ChurnModelingPipeline as _CMP
    cat_res = pipeline_lgb.train_catboost_timecv_multi_seed(
        X_train, y_train,
        ref_dates=train_data['ref_date'],
        last_n=LN,
        seeds=int(args.cat_seeds),
        params=cat_params,
        X_test=X_test,
        sample_weight=w_train,
        gap_months=1,
        time_decay_lambda=0.2,
    )
    oof_cat = np.asarray(cat_res['oof_pred'], dtype=float)
    score_cat = float(cat_res['oof_score'])
    cat_models = cat_res.get('models', [])
    # Per-month table already printed inside; ensure header mentions seeds
    print(f"CatBoost multi-seed complete (seeds={args.cat_seeds})")

# ============================================================================
# OPTIONAL: FT-TRANSFORMER (PyTorch)
# ============================================================================
if args.with_ftt and ('ftt' in args.models):
    print("\n" + "="*60)
    print("TRAINING FT-TRANSFORMER (rtdl-style) WITH TIME-CV")
    print("="*60)
    try:
        from models.ft_transformer import train_ft_transformer_timecv
        oof_ftt, score_ftt, test_pred_ftt = train_ft_transformer_timecv(
            X_train, y_train, ref_dates=train_data['ref_date'], last_n_months=LN, gap_months=1, compute_permutation=True, X_test=X_test
        )
        # Flip-guard on FTT OOF
        from sklearn.metrics import roc_auc_score as _AUC
        _auc_ftt = _AUC(y_train, oof_ftt)
        if _auc_ftt < 0.5:
            print("Flip-guard: Inverting FT-Transformer OOF predictions (AUC < 0.5)")
            oof_ftt = 1.0 - oof_ftt
        score_ftt = oof_composite_monthwise(y_train, oof_ftt, ref_dates=train_data['ref_date'], last_n_months=LN)
        _print_monthwise_metrics("FT-Transformer", y_train, oof_ftt, train_data['ref_date'], last_n=LN)
        ftt_ok = True
    except Exception as _e:
        print(f"⚠ FT-Transformer training failed or unavailable: {_e}")
        oof_ftt = None
        test_pred_ftt = None
        ftt_ok = False
else:
    oof_ftt = None
    test_pred_ftt = None
    ftt_ok = False

# ============================================================================
# STEP 5.7: TRAIN TWO-STAGE HEAD (STAGE-A/B)
# ============================================================================

print("\n" + "="*60)
print("TRAINING TWO-STAGE HEAD (Stage-A recall → Stage-B refine)")
print("="*60)

if 'two' in args.models:
    # Train two-stage head on time-ordered month folds
    oof_two_stage_B, score_two_stage_B, oof_two_stage_A, score_two_stage_A = pipeline_lgb.train_two_stage_timecv(
        X_train, y_train, ref_dates=train_data['ref_date'], last_n_months=LN, sample_weight=w_train
    )

    # Flip-guard on Stage-B OOF
    auc_two_stage_B = roc_auc_score(y_train, oof_two_stage_B)
    if auc_two_stage_B < 0.5:
            print("Flip-guard: Inverting Two-Stage Stage-B OOF predictions (AUC < 0.5)")
            oof_two_stage_B = 1.0 - oof_two_stage_B
            auc_two_stage_B = roc_auc_score(y_train, oof_two_stage_B)
    # Recompute Two-Stage Stage-B score post flip-guard using month-wise composite
    score_two_stage_B = oof_composite_monthwise(y_train, oof_two_stage_B, ref_dates=train_data['ref_date'], last_n_months=LN)

    # Alias for ensemble compatibility
    oof_two_stage = oof_two_stage_B
    score_two_stage = score_two_stage_B
    _print_monthwise_metrics("Two-Stage B", y_train, oof_two_stage, train_data['ref_date'], last_n=LN)

# ============================================================================
# STEP 5.8: LEVEL-2 STACKER (optional)
# ============================================================================
print("\n" + "="*60)
print("TRAINING ROBUST STACKER (LogisticRegressionCV on base OOF only)")
print("="*60)
from scripts.repair_modules import train_robust_stacker as _robust_stack
base_oof_inputs = {
    'lgb': oof_lgb,
    'xgb': oof_xgb,
    'cat': oof_cat,
    'two_stageB': oof_two_stage,
}
base_oof_inputs = {k: v for k, v in base_oof_inputs.items() if v is not None}
if len(base_oof_inputs) >= 2:
    try:
        oof_meta, score_meta, lr_meta_model = _robust_stack(base_oof_inputs, np.asarray(y_train, dtype=float), n_splits=5)
        # Print per-month diagnostics for meta
        _print_monthwise_metrics("Robust Stacker", y_train, oof_meta, train_data['ref_date'], last_n=LN)
        # Remember order of base keys for test-time matrix
        _stack_order = [k for k in ['lgb', 'xgb', 'cat', 'two_stageB'] if k in base_oof_inputs]
    except Exception as _e:
        print(f"[STACKER] Robust stacker failed: {_e}")
        oof_meta = None; score_meta = None; lr_meta_model = None; _stack_order = []
else:
    print("[STACKER] Not enough base OOF streams for stacking; skipping.")
    oof_meta = None; score_meta = None; lr_meta_model = None; _stack_order = []

# ============================================================================
# STEP 6: ENSEMBLE AND CALIBRATION
# ============================================================================

print("\n" + "="*60)
print("SEARCHING MONTH-SPECIFIC BLEND WEIGHTS (OOF)")
print("="*60)

"""
Prepare candidate models based on availability and user selection
"""
models_oof = {}
if oof_lgb is not None:
    models_oof['lgb'] = oof_lgb
if oof_xgb is not None:
    models_oof['xgb'] = oof_xgb
if oof_two_stage is not None:
    models_oof['two_stageB'] = oof_two_stage
if oof_cat is not None:
    models_oof['cat'] = oof_cat
if oof_meta is not None:
    models_oof['meta'] = oof_meta
if args.with_ftt and (oof_ftt is not None):
    models_oof['ftt'] = oof_ftt

names = list(models_oof.keys())
step = 0.05
grid = np.linspace(0, 1, int(1/step) + 1)

def _score_mix_for_mask(weights, mask):
    mix = np.zeros_like(next(iter(models_oof.values())))
    for w, n in zip(weights, names):
        mix = mix + w * models_oof[n]
    # score on only the masked rows (single month)
    y_v = np.asarray(y_train)[mask]
    p_v = np.asarray(mix, dtype=float)[mask]
    # Flip-guard per month
    try:
        auc = roc_auc_score(y_v, p_v)
        if auc < 0.5:
            p_v = 1.0 - p_v
    except Exception:
        pass
    s, _ = _metric_fn(y_v, p_v)
    return s

def _optimize_month(mask):
    best_w = None
    best_s = -1
    if len(names) == 3:
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - (w0 + w1)
                if w2 < -1e-9:
                    continue
                weights = [w0, w1, max(0.0, w2)]
                s = _score_mix_for_mask(weights, mask)
                if s > best_s:
                    best_s = s
                    best_w = dict(zip(names, weights))
    elif len(names) == 4:
        for w0 in grid:
            for w1 in grid:
                for w2 in grid:
                    w3 = 1.0 - (w0 + w1 + w2)
                    if w3 < -1e-9:
                        continue
                    weights = [w0, w1, w2, max(0.0, w3)]
                    s = _score_mix_for_mask(weights, mask)
                    if s > best_s:
                        best_s = s
                        best_w = dict(zip(names, weights))
    else:
        # For 5+ models, use a quick random Dirichlet search for simplex weights
        rng = np.random.default_rng(42)
        trials = 500
        for _ in range(trials):
            w = rng.dirichlet(np.ones(len(names))).astype(float)
            s = _score_mix_for_mask(w, mask)
            if s > best_s:
                best_s = s
                best_w = dict(zip(names, w.tolist()))
        if best_w is None:
            eq = [1.0 / len(names)] * len(names)
            best_w = dict(zip(names, eq))
    return best_w

# Build month masks
m = pd.to_datetime(train_data['ref_date']).dt.to_period('M')
target_months = ['2018-09', '2018-11', '2018-12']
month_masks = {str(vm): (m.values == vm) for vm in sorted(m.unique())}

per_month_weights = {}
for tm in target_months:
    if tm in month_masks:
        print(f"Optimizing weights for month {tm}...")
        per_month_weights[tm] = _optimize_month(month_masks[tm])
        print("  Weights:")
        for n in names:
            print(f"    {n}: {per_month_weights[tm][n]:.2f}")
    else:
        print(f"Skipping month {tm} (not present)")

# Aggregate weights with emphasis on 2018-11/12
coeffs = {'2018-09': 0.2, '2018-11': 0.3, '2018-12': 0.5}
present = [k for k in coeffs.keys() if k in per_month_weights]
if not present:
    # Fallback: equal weights
    final_weights = {n: 1.0 / len(names) for n in names}
else:
    total = sum(coeffs[k] for k in present)
    final_weights = {n: 0.0 for n in names}
    for k in present:
        alpha = coeffs[k] / total
        for n in names:
            final_weights[n] += alpha * per_month_weights[k][n]

# Normalize final weights to ensure they sum to 1.0 exactly
sum_w = sum(final_weights.values())
if sum_w > 0:
    for n in list(final_weights.keys()):
        final_weights[n] = final_weights[n] / sum_w

print("\nAggregated blend weights (sum≈1):")
for n in names:
    print(f"  {n}: {final_weights[n]:.3f}")

# Build OOF ensemble with aggregated weights
oof_ensemble = np.zeros_like(next(iter(models_oof.values())))
for n, w in final_weights.items():
    oof_ensemble = oof_ensemble + w * models_oof[n]

# Flip-guard on blended OOF as safety
auc_ens = roc_auc_score(y_train, oof_ensemble)
if auc_ens < 0.5:
    print("Flip-guard: Inverting blended OOF predictions (AUC < 0.5)")
    oof_ensemble = 1.0 - oof_ensemble
    auc_ens = roc_auc_score(y_train, oof_ensemble)

# Evaluate blended ensemble
ensemble_score = oof_composite_monthwise(y_train, oof_ensemble, ref_dates=train_data['ref_date'], last_n_months=LN)

print(f"\nBlended Ensemble OOF Score: {ensemble_score:.6f}")
print("  Scores reflect month-wise composite (normalized Gini/Recall/Lift)")
_print_monthwise_metrics("Blended Ensemble", y_train, oof_ensemble, train_data['ref_date'], last_n=LN)

# ============================================================================
# STEP 7: GENERATE TEST PREDICTIONS
# ============================================================================

print("\n" + "="*60)
print("GENERATING TEST PREDICTIONS")
print("="*60)

test_pred_lgb = None
test_pred_xgb = None
test_pred_cat = None
test_pred_two_stage = None
test_pred_two_stage_A = None
test_pred_meta = None

# LightGBM predictions
if 'lgb' in models_oof:
    test_pred_lgb = pipeline_lgb.predict(X_test)
    print(f"✓ LightGBM predictions complete")

if 'xgb' in models_oof and xgb_models:
    # XGBoost predictions
    test_pred_xgb = np.mean([m.predict_proba(X_test)[:, 1] for m in xgb_models], axis=0)
    print(f"✓ XGBoost predictions complete")

# CatBoost predictions (optional)
cat_ok = False
if 'cat' in models_oof and cat_models:
    try:
        # Prefer test prediction returned by multi-seed trainer if available
        test_pred_cat = None
        if 'cat_res' in globals():
            test_pred_cat = cat_res.get('test_pred', None)
        if test_pred_cat is None:
            # Fallback: average ranks across seeds/models for CatBoost predictions
            _cat_preds = []
            for m in cat_models:
                _p = m.predict_proba(X_test)[:, 1].astype(float)
                _cat_preds.append(_p)
            if len(_cat_preds) > 0:
                arr = np.vstack(_cat_preds)
                ranks = np.apply_along_axis(lambda v: (v.argsort().argsort().astype(float) / max(1, len(v)-1)), 1, arr)
                avg_rank = ranks.mean(axis=0)
                test_pred_cat = avg_rank
        if test_pred_cat is not None:
            cat_ok = True
            print(f"✓ CatBoost predictions complete (rank-averaged; seeds={args.cat_seeds})")
        else:
            test_pred_cat = None
            cat_ok = False
            print(f"⚠ CatBoost models empty; continuing without cat in blend")
    except Exception:
        test_pred_cat = None
        cat_ok = False
        print(f"⚠ CatBoost predictions unavailable; continuing without cat in blend")

if 'two_stageB' in models_oof:
    # Two-Stage head predictions
    test_pred_two_stage, test_pred_two_stage_A = pipeline_lgb.predict_two_stage(X_test)
    print(f"✓ Two-Stage head predictions complete")

# Level-2 stacker predictions (optional)
if lr_meta_model is not None:
    # Build test meta matrix in the same order used for training
    _meta_cols = []
    if 'lgb' in _stack_order and (test_pred_lgb is not None):
        _meta_cols.append(np.asarray(test_pred_lgb, dtype=float))
    if 'xgb' in _stack_order and (test_pred_xgb is not None):
        _meta_cols.append(np.asarray(test_pred_xgb, dtype=float))
    if 'cat' in _stack_order and cat_ok and (test_pred_cat is not None):
        _meta_cols.append(np.asarray(test_pred_cat, dtype=float))
    if 'two_stageB' in _stack_order and (test_pred_two_stage is not None):
        _meta_cols.append(np.asarray(test_pred_two_stage, dtype=float))
    if len(_meta_cols) == len(_stack_order) and len(_meta_cols) > 0:
        _X_meta_test = np.vstack(_meta_cols).T
        test_pred_meta = lr_meta_model.predict_proba(_X_meta_test)[:, 1].astype(float)
        print("✓ Robust stacker predictions complete")
        try:
            import pandas as _pd
            os.makedirs(os.path.join('outputs', 'test_preds'), exist_ok=True)
            _pd.DataFrame({'test_meta': np.asarray(test_pred_meta, dtype=float)}).to_parquet(
                os.path.join('outputs', 'test_preds', 'meta_stacker_test.parquet'), index=False
            )
        except Exception as _e:
            print(f"[META] Warning: failed to save test parquet: {_e}")
else:
    test_pred_meta = None

# FT-Transformer predictions already computed in training block; nothing to do here

base_len = None
for arr in [test_pred_lgb, test_pred_xgb, test_pred_cat, test_pred_two_stage, test_pred_meta]:
    if arr is not None:
        base_len = len(arr)
        break
if base_len is None:
    raise RuntimeError("No model predictions available to blend; check --models selection.")

# Blended predictions using best OOF weights
test_pred_ensemble = np.zeros(base_len, dtype=float)
for n, w in final_weights.items():
    if n == 'lgb' and (test_pred_lgb is not None):
        test_pred_ensemble = test_pred_ensemble + w * test_pred_lgb
    elif n == 'xgb' and (test_pred_xgb is not None):
        test_pred_ensemble = test_pred_ensemble + w * test_pred_xgb
    elif n == 'cat' and cat_ok and (test_pred_cat is not None):
        test_pred_ensemble = test_pred_ensemble + w * test_pred_cat
    elif n == 'two_stageB' and (test_pred_two_stage is not None):
        test_pred_ensemble = test_pred_ensemble + w * test_pred_two_stage
    elif n == 'meta' and (test_pred_meta is not None):
        test_pred_ensemble = test_pred_ensemble + w * test_pred_meta
    elif n == 'ftt' and (test_pred_ftt is not None):
        test_pred_ensemble = test_pred_ensemble + w * test_pred_ftt
print(f"\n✓ Blended ensemble predictions complete")

# ============================================================================
# STEP 7.1: ADVANCED PROBABILITY CALIBRATION (modular)
# ============================================================================

print("\n" + "="*60)
print("CALIBRATING PREDICTIONS (K-fold Isotonic/Beta; Gamma grid)")
print("="*60)

from src.utils.calibration import CalibrationConfig, kfold_calibrate_with_gamma

seg_values = None
if 'tenure' in train_data.columns:
    seg_values = np.asarray(train_data['tenure'].values, dtype=float)

cfg_cal = CalibrationConfig(segment_col='tenure' if seg_values is not None else None,
                            gamma_grid=_CALIB_GAMMA_GRID)

# Apply repaired K-fold calibration only if it improves OOF
from scripts.repair_modules import train_kfold_calibrator as _kcal
try:
    oof_cal, score_cal, final_cal = _kcal(np.asarray(oof_ensemble, dtype=float), np.asarray(y_train, dtype=float), n_splits=6)
    print(f"[CALIB] OOF pre={ensemble_score:.6f} vs k-fold isotonic post={score_cal:.6f}")
    if float(score_cal) >= float(ensemble_score) - 1e-12:
        calibrated_pred = final_cal(np.asarray(test_pred_ensemble, dtype=float))
        chosen_name = 'isotonic_kfold'
        best_gamma = 1.0
        chosen_oof = oof_cal
        print("[CALIB] Applied k-fold isotonic calibrator (improved or equal OOF)")
    else:
        calibrated_pred = test_pred_ensemble
        chosen_name = 'none'
        best_gamma = 1.0
        chosen_oof = oof_ensemble
        print("[CALIB] Skipped calibration (no improvement)")
except Exception as _e:
    print(f"[CALIB] Calibration failed or unavailable: {_e}")
    calibrated_pred = test_pred_ensemble
    chosen_name = 'none'
    best_gamma = 1.0
    chosen_oof = oof_ensemble

# ============================================================================
# STEP 7.2: OPTIONAL PSEUDO-LABELING (CatBoost/XGBoost)
# ============================================================================

if args.pseudo_label:
    print("\n" + "="*60)
    print("PSEUDO-LABELING: augment train with confident test predictions and retrain CatBoost/XGBoost")
    print("="*60)

    # Select confident pseudo-labels from calibrated test predictions
    p_test = np.asarray(calibrated_pred, dtype=float)
    n_te = len(p_test)
    k_pos = max(1, int(round(args.pl_pos_top * n_te)))
    k_neg = max(1, int(round(args.pl_neg_bottom * n_te)))
    order = np.argsort(p_test)
    neg_idx = order[:k_neg]
    pos_idx = order[-k_pos:]

    X_te_sel = pd.concat([X_test.iloc[neg_idx], X_test.iloc[pos_idx]], axis=0)
    y_te_sel = pd.Series(np.concatenate([np.zeros(len(neg_idx), dtype=int), np.ones(len(pos_idx), dtype=int)]))
    ref_te_sel = pd.concat([test_features['ref_date'].iloc[neg_idx], test_features['ref_date'].iloc[pos_idx]], axis=0)

    # Assemble augmented train
    X_aug = pd.concat([X_train, X_te_sel], axis=0, ignore_index=True)
    y_aug = pd.concat([y_train.reset_index(drop=True), y_te_sel.reset_index(drop=True)], axis=0, ignore_index=True)
    ref_aug = pd.concat([pd.Series(train_data['ref_date']).astype(str).reset_index(drop=True), pd.Series(ref_te_sel).astype(str).reset_index(drop=True)], axis=0, ignore_index=True)

    # Retrain XGBoost on augmented data
    if (oof_xgb is not None) and (score_xgb is not None):
        print("\n[PL] Retraining XGBoost on augmented data...")
        oof_xgb_pl_all, score_xgb_pl_all, xgb_models_pl = pipeline_lgb.train_xgboost(
            X_aug, y_aug, xgb_params, ref_dates=ref_aug, last_n_months=LN, sample_weight=None
        )
        # Evaluate composite only on original train rows
        oof_xgb_pl = oof_xgb_pl_all[:len(X_train)]
        score_xgb_pl = oof_composite_monthwise(y_train, oof_xgb_pl, ref_dates=train_data['ref_date'], last_n_months=LN)
        delta_xgb = (score_xgb_pl - float(score_xgb)) if (score_xgb is not None) else 0.0
        print(f"[PL] XGB original-train OOF composite: {score_xgb_pl:.6f} (baseline {score_xgb:.6f}, Δ={delta_xgb:+.6f})")

        accept_xgb_pl = delta_xgb >= float(args.pl_improve_min)
    else:
        print("[PL] Skipping XGB pseudo-labeling (baseline XGB not trained)")
        oof_xgb_pl = None
        test_pred_xgb_pl = None
        accept_xgb_pl = False
    if accept_xgb_pl:
        test_pred_xgb_pl = np.mean([m.predict_proba(X_test)[:, 1] for m in xgb_models_pl], axis=0)
        print("[PL] Accepted XGB pseudo-labeled model.")
    else:
        test_pred_xgb_pl = None
        print("[PL] Rejected XGB pseudo-labeled model (insufficient lift).")

    # Retrain CatBoost on augmented data
    if (oof_cat is not None) and (score_cat is not None):
        print("\n[PL] Retraining CatBoost on augmented data...")
        try:
            oof_cat_pl_all, score_cat_pl_all, cat_models_pl = pipeline_lgb.train_catboost_timecv(
                X_aug, y_aug, params=cat_params, ref_dates=ref_aug, last_n_months=LN, seeds=list(range(1, args.cat_seeds + 1)), sample_weight=None
            )
            oof_cat_pl = oof_cat_pl_all[:len(X_train)]
            score_cat_pl = oof_composite_monthwise(y_train, oof_cat_pl, ref_dates=train_data['ref_date'], last_n_months=LN)
            delta_cat = (score_cat_pl - float(score_cat)) if (score_cat is not None) else 0.0
            print(f"[PL] CAT original-train OOF composite: {score_cat_pl:.6f} (baseline {score_cat:.6f}, Δ={delta_cat:+.6f})")
            accept_cat_pl = delta_cat >= float(args.pl_improve_min)
            if accept_cat_pl:
                _cat_preds_pl = []
                for m in cat_models_pl:
                    _p = m.predict_proba(X_test)[:, 1].astype(float)
                    _cat_preds_pl.append(_p)
                if len(_cat_preds_pl) > 0:
                    arr = np.vstack(_cat_preds_pl)
                    ranks = np.apply_along_axis(lambda v: (v.argsort().argsort().astype(float) / max(1, len(v)-1)), 1, arr)
                    test_pred_cat_pl = ranks.mean(axis=0)
                else:
                    test_pred_cat_pl = None
                print("[PL] Accepted CatBoost pseudo-labeled model.")
            else:
                test_pred_cat_pl = None
                print("[PL] Rejected CatBoost pseudo-labeled model (insufficient lift).")
        except Exception as _e:
            print(f"[PL] CatBoost pseudo-label retrain failed: {_e}")
            oof_cat_pl = None
            test_pred_cat_pl = None
            accept_cat_pl = False
    else:
        print("[PL] Skipping CatBoost pseudo-labeling (baseline CatBoost not trained)")
        oof_cat_pl = None
        test_pred_cat_pl = None
        accept_cat_pl = False

else:
    oof_xgb_pl = None
    test_pred_xgb_pl = None
    accept_xgb_pl = False
    oof_cat_pl = None
    test_pred_cat_pl = None
    accept_cat_pl = False
# STEP 8: CREATE SUBMISSION FILE
# ============================================================================

print("\n" + "="*60)
print("CREATING SUBMISSION")
print("="*60)

# Prepare submission dataframe
submission = pd.DataFrame({
    'cust_id': test_features['cust_id'],
    'churn': calibrated_pred  # Use calibrated predictions
})

# Save submission
os.makedirs(os.path.join('data', 'submissions'), exist_ok=True)
sub_path = os.path.join('data', 'submissions', 'submission.csv')
submission.to_csv(sub_path, index=False)
print(f"Submission saved! Shape: {submission.shape} -> {sub_path}")
print(f"Sample predictions:\n{submission.head(10)}")

# Write/update single last-update marker
try:
    update_path = os.path.join('data', 'submissions', 'last_update.txt')
    with open(update_path, 'w', encoding='utf-8') as _u:
        _u.write(f"updated_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | file=submission.csv | rows={len(submission)}\n")
    print(f"Updated {update_path}")
except Exception as _e:
    print(f"Could not write last_update.txt: {_e}")

# Also export predictions bundle for portfolio generation (OOF + Test)
try:
    import pickle as _pkl
    print("\nExporting predictions bundle (for make_portfolio.py)...")
    bundle = {
        'y_train': np.asarray(y_train, dtype=float),
        'ref_dates': np.asarray(train_data['ref_date']).astype(str),
        'oof_lgb': np.asarray(oof_lgb, dtype=float),
        'oof_xgb': np.asarray(oof_xgb, dtype=float),
        'oof_two_stage_B': np.asarray(oof_two_stage_B, dtype=float),
        'oof_two_stage_A': np.asarray(oof_two_stage_A, dtype=float),
        'test_lgb': np.asarray(test_pred_lgb, dtype=float),
        'test_xgb': np.asarray(test_pred_xgb, dtype=float),
        'test_two_stage_B': np.asarray(test_pred_two_stage, dtype=float),
        'test_two_stage_A': np.asarray(test_pred_two_stage_A, dtype=float),
        'oof_ensemble': np.asarray(oof_ensemble, dtype=float),
        'test_ensemble_raw': np.asarray(test_pred_ensemble, dtype=float),
        'calibrated_test': np.asarray(calibrated_pred, dtype=float),
        'final_weights': {k: float(v) for k, v in final_weights.items()},
        'chosen_calibration': str(chosen_name),
        'best_gamma': float(best_gamma)
    }
    # Optional CatBoost
    try:
        if 'oof_cat' in globals() and oof_cat is not None:
            bundle['oof_cat'] = np.asarray(oof_cat, dtype=float)
        if 'test_pred_cat' in globals() and test_pred_cat is not None:
            bundle['test_cat'] = np.asarray(test_pred_cat, dtype=float)
        if oof_meta is not None:
            bundle['oof_meta'] = np.asarray(oof_meta, dtype=float)
        if test_pred_meta is not None:
            bundle['test_meta'] = np.asarray(test_pred_meta, dtype=float)
        if args.with_ftt and (oof_ftt is not None):
            bundle['oof_ftt'] = np.asarray(oof_ftt, dtype=float)
        if args.with_ftt and (test_pred_ftt is not None):
            bundle['test_ftt'] = np.asarray(test_pred_ftt, dtype=float)
        # Pseudo-labeled artifacts (accepted only)
        if args.pseudo_label and (test_pred_xgb_pl is not None):
            bundle['oof_xgb_pl'] = np.asarray(oof_xgb_pl, dtype=float)
            bundle['test_xgb_pl'] = np.asarray(test_pred_xgb_pl, dtype=float)
        if args.pseudo_label and (test_pred_cat_pl is not None):
            bundle['oof_cat_pl'] = np.asarray(oof_cat_pl, dtype=float)
            bundle['test_cat_pl'] = np.asarray(test_pred_cat_pl, dtype=float)
    except Exception:
        pass

    os.makedirs(os.path.join('outputs', 'predictions'), exist_ok=True)
    bundle_path = os.path.join('outputs', 'predictions', 'predictions_bundle.pkl')
    with open(bundle_path, 'wb') as _f:
        _pkl.dump(bundle, _f)
    print(f"✓ Saved {bundle_path}")
except Exception as e:
    print(f"⚠ Could not export predictions bundle: {e}")

# ============================================================================
# STEP 9: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*60)

feature_importance = None
try:
    feature_importance = pipeline_lgb.get_feature_importance(feature_cols)
except Exception:
    feature_importance = None

if feature_importance is None:
    # Fallback: try CatBoost feature importance if available
    try:
        if 'cat_res' in globals() and isinstance(cat_res, dict):
            feature_importance = cat_res.get('feature_importance', None)
    except Exception:
        feature_importance = None

if feature_importance is not None:
    print(feature_importance[['feature', 'importance_mean']].head(20))
    # Save feature importance
    os.makedirs(os.path.join('outputs', 'reports'), exist_ok=True)
    # Choose filename based on source
    fi_filename = 'feature_importance.csv'
    try:
        if 'cat_res' in globals() and isinstance(cat_res, dict) and feature_importance is cat_res.get('feature_importance'):
            fi_filename = 'catboost_feature_importance.csv'
    except Exception:
        pass
    fi_path = os.path.join('outputs', 'reports', fi_filename)
    feature_importance.to_csv(fi_path, index=False)
else:
    print("Feature importance not available.")

# ============================================================================
# STEP 10: ADVANCED OPTIMIZATION STRATEGIES (OPTIONAL)
# ============================================================================

print("\n" + "="*60)
print("OPTIMIZATION TIPS FOR EVEN BETTER SCORES")
print("="*60)

print("""
1. STACKING:
   - Use OOF predictions from LGB/XGB as meta-features
   - Train logistic regression or neural network on top
   
2. ADVERSARIAL VALIDATION:
   - Check if train/test distributions match
   - Adjust features or reweight samples if needed
   
3. PSEUDO-LABELING:
   - Use high-confidence test predictions as additional training data
   - Be careful with validation strategy
   
4. HYPERPARAMETER TUNING:
   - Use Optuna or similar for systematic search
   - Optimize directly for competition metric
   
5. FEATURE SELECTION:
   - Remove low-importance features
   - Try different feature subsets
   
6. MORE FEATURES:
   - RFM (Recency, Frequency, Monetary) scores
   - Customer lifecycle stage
   - Seasonality patterns
   - Interaction features between top features
   
7. CUSTOM OBJECTIVES:
   - Implement custom LightGBM objective for Recall@10%
   - This can significantly boost your score
""")

print("\n" + "="*60)
print("WORKFLOW COMPLETE!")
print("="*60)
print(f"\nIndividual Model Scores:")
print(f"  LightGBM:             {score_lgb:.6f}" if score_lgb is not None else "  LightGBM:             n/a")
print(f"  XGBoost:              {score_xgb:.6f}" if score_xgb is not None else "  XGBoost:              n/a")
print(f"  CatBoost:             {score_cat:.6f}" if score_cat is not None else "  CatBoost:             n/a")
print(f"  Two-Stage (Advanced): {score_two_stage:.6f}" if score_two_stage is not None else "  Two-Stage (Advanced): n/a")
print(f"\nEnsemble Score:")
print(f"  Blended Ensemble:     {ensemble_score:.6f}")
print(f"\nModel Weights:")
try:
    for n, w in final_weights.items():
        print(f"  {n}: {w:.3f}")
except Exception:
    print("  Weights not available")
print(f"\nSubmission file: submission.csv")

# ============================================================================
# FINAL CONSOLIDATED SUMMARY
# ============================================================================
try:
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    # 1) Global OOF composite: raw vs calibrated
    post_calib_score = None
    try:
        post_calib_score = oof_composite_monthwise(y_train, np.asarray(chosen_oof, dtype=float),
                                                   ref_dates=train_data['ref_date'], last_n_months=LN)
    except Exception:
        post_calib_score = None
    raw_val = float(ensemble_score) if 'ensemble_score' in globals() and ensemble_score is not None else float('nan')
    cal_val = float(post_calib_score) if post_calib_score is not None else float('nan')
    print(f"- Global OOF composite: raw={raw_val:.6f} vs calibrated={cal_val:.6f}  (method={str(chosen_name)}, gamma={float(best_gamma):.2f})")

    # 2) Per-month table for final blended OOF (calibrated)
    print(f"- Per-month validation table for final blend (calibrated):")
    try:
        _print_monthwise_metrics("Final Blended (calibrated)", y_train, np.asarray(chosen_oof, dtype=float), train_data['ref_date'], last_n=LN)
    except Exception as _e:
        print(f"  (Could not print per-month table: {_e})")

    # 2b) Blended ensemble OOF summary
    try:
        print(f"- Blended Ensemble OOF Score: {float(ensemble_score):.6f}")
    except Exception:
        pass

    # 3) Domain AUC if adversarial filtering enabled
    if True:
        def _fmt(v):
            try:
                return f"{float(v):.4f}"
            except Exception:
                return "n/a"
        if bool(getattr(args, 'adv_filter', False)):
            print(f"- Domain AUC (train vs test): pre-interactions {_fmt(adv_pre_before)} → {_fmt(adv_pre_after)}; post-filter {_fmt(adv_post_before)} → {_fmt(adv_post_after)}")
        if domain_auc_summary is not None:
            print(f"- Domain AUC (final features): {_fmt(domain_auc_summary)}")

    # 4) Final blend weights (per-month and global)
    print("- Final blend weights (per-month + global):")
    try:
        # Per-month
        if 'per_month_weights' in globals() and isinstance(per_month_weights, dict) and per_month_weights:
            for mname, wdict in per_month_weights.items():
                parts = ", ".join([f"{k}={wdict.get(k, 0.0):.3f}" for k in sorted(wdict.keys())])
                print(f"  • {mname}: {parts}")
        # Global
        g_parts = ", ".join([f"{k}={final_weights.get(k, 0.0):.3f}" for k in sorted(final_weights.keys())])
        print(f"  • global: {g_parts}")
    except Exception as _e:
        print(f"  (Could not print weights: {_e})")

    # 5) Submission path
    try:
        print(f"- Submission saved to: {sub_path}")
    except Exception:
        print("- Submission saved to: data/submissions/submission.csv")
except Exception as _e:
    print(f"[SUMMARY] Warning: could not print final summary: {_e}")