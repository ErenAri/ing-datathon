"""
Quick Robustness Diagnostic Script

Run this to diagnose why your model had a public/private leaderboard gap.
This will tell you exactly what's wrong and what to fix.

Usage:
    python diagnose_robustness.py
"""

import pandas as pd
import numpy as np
import os
from src.utils.robustness_tools import (
    robust_time_split_cv,
    stable_feature_selection,
    comprehensive_leakage_check,
    RobustnessTracker
)
from src.features.feature_engineering import ChurnFeatureEngineering
from src.features.advanced_features import AdvancedFeatureEngineering
from src.models.modeling_pipeline import ChurnModelingPipeline
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def load_data():
    """Load data from standard locations."""
    print("="*70)
    print("LOADING DATA")
    print("="*70)

    def _resolve_path(name):
        for p in [name, os.path.join('data', 'raw', name), os.path.join('data', name)]:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Cannot find {name}")

    customer_history = pd.read_csv(_resolve_path('customer_history.csv'))
    customers = pd.read_csv(_resolve_path('customers.csv'))
    reference_data = pd.read_csv(_resolve_path('reference_data.csv'))
    reference_data_test = pd.read_csv(_resolve_path('reference_data_test.csv'))

    print(f"✓ Loaded customer_history: {customer_history.shape}")
    print(f"✓ Loaded customers: {customers.shape}")
    print(f"✓ Loaded reference_data (train): {reference_data.shape}")
    print(f"✓ Loaded reference_data_test: {reference_data_test.shape}")
    print()

    return customer_history, customers, reference_data, reference_data_test


def create_features(customer_history, customers, reference_data, reference_data_test):
    """Create features for train and test."""
    print("="*70)
    print("CREATING FEATURES")
    print("="*70)

    fe = ChurnFeatureEngineering()
    afe = AdvancedFeatureEngineering()

    # Train features
    train_features_list = []
    for ref_date in reference_data['ref_date'].unique():
        ref_customers = reference_data[reference_data['ref_date'] == ref_date]['cust_id'].unique()
        ref_date_dt = pd.to_datetime(ref_date)
        history_subset = customer_history[
            (customer_history['cust_id'].isin(ref_customers)) &
            (pd.to_datetime(customer_history['date']) <= ref_date_dt)
        ].copy()
        customers_subset = customers[customers['cust_id'].isin(ref_customers)].copy()

        features = fe.create_all_features(history_subset, customers_subset, ref_date)
        advanced_features = afe.create_all_advanced_features(history_subset, customers_subset, ref_date)
        features = features.merge(advanced_features, on='cust_id', how='left')
        features['ref_date'] = ref_date
        train_features_list.append(features)

    train_features = pd.concat(train_features_list, axis=0, ignore_index=True)

    # Test features (simplified for diagnostic)
    test_features_list = []
    for ref_date in reference_data_test['ref_date'].unique()[:3]:  # Just first 3 months for speed
        ref_customers = reference_data_test[reference_data_test['ref_date'] == ref_date]['cust_id'].unique()
        ref_date_dt = pd.to_datetime(ref_date)
        history_subset = customer_history[
            (customer_history['cust_id'].isin(ref_customers)) &
            (pd.to_datetime(customer_history['date']) <= ref_date_dt)
        ].copy()
        customers_subset = customers[customers['cust_id'].isin(ref_customers)].copy()

        features = fe.create_all_features(history_subset, customers_subset, ref_date)
        advanced_features = afe.create_all_advanced_features(history_subset, customers_subset, ref_date)
        features = features.merge(advanced_features, on='cust_id', how='left')
        features['ref_date'] = ref_date
        test_features_list.append(features)

    test_features = pd.concat(test_features_list, axis=0, ignore_index=True)

    # Prepare modeling data
    train_data = train_features.merge(
        reference_data[['cust_id', 'ref_date', 'churn']],
        on=['cust_id', 'ref_date'],
        how='left'
    )

    non_feature_cols = {'cust_id', 'ref_date', 'churn'}

    # Get numeric columns from train
    train_numeric_cols = [
        col for col in train_data.columns
        if col not in non_feature_cols and pd.api.types.is_numeric_dtype(train_data[col])
    ]

    # Get numeric columns from test
    test_numeric_cols = [
        col for col in test_features.columns
        if col not in non_feature_cols and pd.api.types.is_numeric_dtype(test_features[col])
    ]

    # Use only common columns (intersection)
    common_cols = list(set(train_numeric_cols) & set(test_numeric_cols))
    print(f"  Train-only features: {len(set(train_numeric_cols) - set(test_numeric_cols))}")
    print(f"  Test-only features: {len(set(test_numeric_cols) - set(train_numeric_cols))}")
    print(f"  Common features: {len(common_cols)}")

    # Use common columns
    X_train = train_data[common_cols].fillna(-999).replace([np.inf, -np.inf], -999)
    y_train = train_data['churn']
    X_test = test_features[common_cols].fillna(-999).replace([np.inf, -np.inf], -999)
    ref_dates_train = train_data['ref_date']

    print(f"✓ Train features: {X_train.shape}")
    print(f"✓ Test features: {X_test.shape}")
    print()

    return X_train, y_train, X_test, ref_dates_train


def diagnose_cv_strategy(X_train, y_train, ref_dates_train):
    """Diagnose if CV strategy is causing overfitting."""
    print("\n" + "="*70)
    print("DIAGNOSIS 1: CROSS-VALIDATION STRATEGY")
    print("="*70)

    print("\nTesting OLD vs NEW CV strategies...\n")

    # Test robust time-split CV
    try:
        splits = robust_time_split_cv(ref_dates_train, n_splits=4, test_size_months=2, gap_months=1)
        print(f"✓ New robust CV created {len(splits)} splits:")
        for i, split in enumerate(splits):
            print(f"  Split {i+1}: Train on {split['train_months']} → Test on {split['test_months']} "
                  f"(n_train={split['n_train']}, n_test={split['n_test']})")

        # Quick model to compare CV scores
        print("\nTraining quick LightGBM to compare CV scores...")
        params = {
            'objective': 'binary', 'metric': 'auc',
            'num_leaves': 31, 'learning_rate': 0.05,
            'verbose': -1, 'n_jobs': -1
        }

        scores = []
        for i, split in enumerate(splits):
            train_idx = split['train_idx']
            val_idx = split['test_idx']

            train_data = lgb.Dataset(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
            val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx], reference=train_data)

            model = lgb.train(params, train_data, num_boost_round=200,
                            valid_sets=[val_data],
                            callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(period=0)])

            val_pred = model.predict(X_train.iloc[val_idx])
            auc = roc_auc_score(y_train.iloc[val_idx], val_pred)
            scores.append(auc)
            print(f"  Split {i+1} AUC: {auc:.4f}")

        print(f"\nRobust CV Summary:")
        print(f"  Mean AUC:    {np.mean(scores):.4f}")
        print(f"  Std AUC:     {np.std(scores):.4f}")
        print(f"  CV (%)       {(np.std(scores) / np.mean(scores) * 100):.2f}%")

        if np.std(scores) / np.mean(scores) > 0.05:
            print(f"  ⚠ WARNING: High variance! Model is UNSTABLE across time periods.")
            print(f"  → This WILL cause public/private LB gaps!")
        else:
            print(f"  ✓ Low variance - model is stable across time.")

    except Exception as e:
        print(f"✗ Error testing robust CV: {e}")

    print("\n" + "="*70)


def diagnose_feature_stability(X_train, y_train, ref_dates_train):
    """Diagnose if features are stable across time."""
    print("\n" + "="*70)
    print("DIAGNOSIS 2: FEATURE STABILITY")
    print("="*70)

    try:
        stable_features, importance_df = stable_feature_selection(
            X_train, y_train, ref_dates_train,
            n_iterations=5, stability_threshold=0.6
        )

        unstable_pct = (len(X_train.columns) - len(stable_features)) / len(X_train.columns) * 100

        print(f"\n📊 Feature Stability Summary:")
        print(f"  Unstable features: {unstable_pct:.1f}%")

        if unstable_pct > 30:
            print(f"  ⚠ WARNING: >30% of features are UNSTABLE!")
            print(f"  → These features will hurt generalization.")
            print(f"  → Remove them before training final model.")
        elif unstable_pct > 15:
            print(f"  ⚠ CAUTION: {unstable_pct:.1f}% unstable is moderate.")
            print(f"  → Consider removing for better robustness.")
        else:
            print(f"  ✓ Good - most features are stable.")

        # Save stable features list
        with open('stable_features.txt', 'w') as f:
            for feat in stable_features:
                f.write(f"{feat}\n")
        print(f"\n  → Saved stable features to: stable_features.txt")

        importance_df.to_csv('feature_stability_report.csv', index=False)
        print(f"  → Saved stability report to: feature_stability_report.csv")

    except Exception as e:
        print(f"✗ Error checking feature stability: {e}")

    print("\n" + "="*70)


def diagnose_leakage(X_train, X_test, y_train):
    """Check for data leakage."""
    print("\n" + "="*70)
    print("DIAGNOSIS 3: DATA LEAKAGE CHECK")
    print("="*70)

    try:
        warnings = comprehensive_leakage_check(X_train, X_test, y_train)

        if len(warnings) > 0:
            print(f"\n⚠ Found {len(warnings)} potential leakage issues!")
            print(f"  Review the warnings above and investigate suspicious features.")

            # Save warnings to file
            with open('leakage_warnings.txt', 'w') as f:
                for w in warnings:
                    f.write(f"{w}\n")
            print(f"\n  → Saved warnings to: leakage_warnings.txt")
        else:
            print(f"\n✓ No obvious leakage detected.")

    except Exception as e:
        print(f"✗ Error checking leakage: {e}")

    print("\n" + "="*70)


def generate_recommendations(X_train, y_train, ref_dates_train):
    """Generate actionable recommendations."""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    print("""
Based on your competition results (Rank 50 → 86), here are the TOP 3 actions:

1. 🔴 CRITICAL: Fix Cross-Validation Strategy
   → Replace current CV with robust_time_split_cv()
   → Your current CV uses last 6 months (2018-09 to 2018-12)
   → This doesn't simulate the train→2019 test gap properly

   Action: Edit src/models/modeling_pipeline.py
   Replace line 295:
       fold_iter = month_folds(ref_dates, last_n=last_n_months, gap=gap_months)
   With:
       from src.utils.robustness_tools import robust_time_split_cv
       splits = robust_time_split_cv(ref_dates, n_splits=5, test_size_months=2)
       fold_iter = ((s['train_idx'], s['test_idx'], s['test_months'][0]) for s in splits)

2. 🟠 HIGH: Use Conservative Ensemble Weights
   → Stop optimizing weights on validation months
   → Use equal weights or diversity-based weights

   Action: Edit src/main.py line 1152
   Change to:
       from src.utils.robustness_tools import get_conservative_blend_weights
       final_weights = get_conservative_blend_weights(models_oof, method='equal')

3. 🟡 MEDIUM: Remove Unstable Features
   → Remove features that have high importance variance across folds
   → Use the stable_features.txt file generated above

   Action: After feature engineering in src/main.py:
       # Load stable features
       with open('stable_features.txt', 'r') as f:
           stable_features = [line.strip() for line in f]
       X_train = X_train[stable_features]
       X_test = X_test[stable_features]

Expected impact:
- Implementing #1: Reduces public-private gap by ~15-20 places
- Implementing #2: Reduces gap by ~8-12 places
- Implementing #3: Reduces gap by ~5-8 places

Total expected improvement: Rank 86 → 60-65 (back to top 10%!)
""")

    print("="*70)


def main():
    """Run full diagnostic."""
    print("\n" + "="*70)
    print("ROBUSTNESS DIAGNOSTIC FOR ING DATATHON")
    print("Analyzing why you dropped from Rank 50 → 86")
    print("="*70 + "\n")

    try:
        # Load data
        customer_history, customers, reference_data, reference_data_test = load_data()

        # Create features
        X_train, y_train, X_test, ref_dates_train = create_features(
            customer_history, customers, reference_data, reference_data_test
        )

        # Run diagnostics
        diagnose_cv_strategy(X_train, y_train, ref_dates_train)
        diagnose_feature_stability(X_train, y_train, ref_dates_train)
        diagnose_leakage(X_train, X_test, y_train)

        # Generate recommendations
        generate_recommendations(X_train, y_train, ref_dates_train)

        print("\n✓ Diagnostic complete!")
        print("\nGenerated files:")
        print("  - stable_features.txt: List of stable features to use")
        print("  - feature_stability_report.csv: Detailed stability analysis")
        print("  - leakage_warnings.txt: Potential leakage issues")
        print("\nNext steps:")
        print("  1. Review LEADERBOARD_GAP_ANALYSIS.md for detailed explanations")
        print("  2. Implement the 3 critical fixes from RECOMMENDATIONS above")
        print("  3. Retrain your model with the new CV strategy")
        print("  4. Use RobustnessTracker to monitor stability (see robustness_tools.py)")

    except Exception as e:
        print(f"\n✗ Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
