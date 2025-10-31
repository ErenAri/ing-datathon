"""
Hyperparameter Tuning with Optuna - Extended CV Strategy

Optimizes LightGBM, XGBoost, and CatBoost hyperparameters using:
- Extended CV (2017+2018 validation months)
- Stable objective (consistency across time)
- Conservative parameter ranges (avoid overfitting)

Target: +0.01-0.02 performance while maintaining small gap
"""

import optuna
import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("HYPERPARAMETER TUNING - EXTENDED CV STRATEGY")
print("="*70)

# Load data
print("\nLoading data...")
customer_history = pd.read_csv('data/raw/customer_history.csv')
customers = pd.read_csv('data/raw/customers.csv')
reference_data = pd.read_csv('data/raw/reference_data.csv')

print(f"Customer history: {customer_history.shape}")
print(f"Customers: {customers.shape}")
print(f"Reference data: {reference_data.shape}")

# Feature engineering
print("\nCreating features...")
from src.features.feature_engineering import ChurnFeatureEngineering
from src.features.advanced_features import AdvancedFeatureEngineering
from src.features.performance_features import PerformanceFeatureEngineering

fe = ChurnFeatureEngineering()
afe = AdvancedFeatureEngineering()
pfe = PerformanceFeatureEngineering()

train_features_list = []
for ref_date in reference_data['ref_date'].unique():
    print(f"  Processing {ref_date}...")
    ref_customers = reference_data[reference_data['ref_date'] == ref_date]['cust_id'].unique()
    ref_date_dt = pd.to_datetime(ref_date)

    history_subset = customer_history[
        (customer_history['cust_id'].isin(ref_customers)) &
        (pd.to_datetime(customer_history['date']) <= ref_date_dt)
    ].copy()

    customers_subset = customers[customers['cust_id'].isin(ref_customers)].copy()

    features = fe.create_all_features(history_subset, customers_subset, ref_date)
    advanced = afe.create_all_advanced_features(history_subset, customers_subset, ref_date)
    performance = pfe.create_all_performance_features(history_subset, customers_subset, ref_date)

    features = features.merge(advanced, on='cust_id', how='left')
    features = features.merge(performance, on='cust_id', how='left')
    features['ref_date'] = ref_date

    train_features_list.append(features)

train_features = pd.concat(train_features_list, axis=0, ignore_index=True)
train_data = train_features.merge(
    reference_data[['cust_id', 'ref_date', 'churn']],
    on=['cust_id', 'ref_date'],
    how='left'
)

# Prepare data
non_feature_cols = {'cust_id', 'ref_date', 'churn'}
numeric_cols = [col for col in train_data.columns
                if col not in non_feature_cols and pd.api.types.is_numeric_dtype(train_data[col])]

X_train = train_data[numeric_cols].fillna(-999)
y_train = train_data['churn'].values
ref_dates = train_data['ref_date'].values

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Extended CV months (same as main pipeline)
target_months = ['2017-09', '2017-11', '2017-12', '2018-09', '2018-11', '2018-12']

from src.models.modeling_pipeline import ing_hubs_datathon_metric

def create_cv_folds(ref_dates, target_months):
    """Create CV folds based on target validation months."""
    m = pd.to_datetime(ref_dates).to_period('M')
    folds = []
    for tm in target_months:
        val_mask = (m == tm)
        train_mask = ~val_mask
        if val_mask.sum() > 0 and train_mask.sum() > 0:
            folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return folds

folds = create_cv_folds(ref_dates, target_months)
print(f"\nCreated {len(folds)} CV folds")

# ============================================================================
# LIGHTGBM TUNING
# ============================================================================

print("\n" + "="*70)
print("TUNING LIGHTGBM")
print("="*70)

import lightgbm as lgb

def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 5,
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'verbose': -1
    }

    scores = []
    for train_idx, val_idx in folds:
        dtrain = lgb.Dataset(X_train.iloc[train_idx], label=y_train[train_idx])
        dval = lgb.Dataset(X_train.iloc[val_idx], label=y_train[val_idx])

        model = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        preds = model.predict(X_train.iloc[val_idx])
        score, _ = ing_hubs_datathon_metric(y_train[val_idx], preds)
        scores.append(score)

    # Return mean score (stability across time)
    return np.mean(scores)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=True)

print(f"\nBest LGB score: {study_lgb.best_value:.6f}")
print("Best params:")
for k, v in study_lgb.best_params.items():
    print(f"  {k}: {v}")

# ============================================================================
# XGBOOST TUNING
# ============================================================================

print("\n" + "="*70)
print("TUNING XGBOOST")
print("="*70)

import xgboost as xgb

def objective_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'tree_method': 'hist'
    }

    scores = []
    for train_idx, val_idx in folds:
        dtrain = xgb.DMatrix(X_train.iloc[train_idx], label=y_train[train_idx])
        dval = xgb.DMatrix(X_train.iloc[val_idx], label=y_train[val_idx])

        model = xgb.train(
            params, dtrain, num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        preds = model.predict(dval)
        score, _ = ing_hubs_datathon_metric(y_train[val_idx], preds)
        scores.append(score)

    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)

print(f"\nBest XGB score: {study_xgb.best_value:.6f}")
print("Best params:")
for k, v in study_xgb.best_params.items():
    print(f"  {k}: {v}")

# ============================================================================
# SAVE BEST PARAMETERS
# ============================================================================

print("\n" + "="*70)
print("SAVING OPTIMIZED PARAMETERS")
print("="*70)

best_lgb = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': 42,
    **study_lgb.best_params
}

best_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'tree_method': 'hist',
    **study_xgb.best_params
}

# Save to optimized_params.py
with open('optimized_params.py', 'w') as f:
    f.write('# Optuna-optimized hyperparameters (Extended CV strategy)\n\n')
    f.write('LGB_BEST = {\n')
    for k, v in best_lgb.items():
        f.write(f"    '{k}': {repr(v)},\n")
    f.write('}\n\n')

    f.write('XGB_BEST = {\n')
    for k, v in best_xgb.items():
        f.write(f"    '{k}': {repr(v)},\n")
    f.write('}\n\n')

    f.write('USE_OPTIMIZED_PARAMS = True\n')

print("✅ Saved to optimized_params.py")
print("\nRun main pipeline to use these parameters:")
print("  python -m src.main --models lgb xgb cat two meta --cat-seeds 5 --with-stacker")
print("\n" + "="*70)
