"""
Save Training Data for Hyperparameter Optimization

This script loads all data, performs feature engineering, and saves
the processed training data as pickle files. This allows hyperparameter
optimization to run quickly without re-doing expensive feature engineering.

Outputs:
- X_train.pkl: Training features
- y_train.pkl: Training labels
- X_test.pkl: Test features
- feature_cols.pkl: List of feature names
- ref_dates.pkl: Pandas Series of ref_date aligned with X_train rows (for time-based CV)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

def _resolve_data_path(filename: str) -> str:
    """Resolve CSV path from common locations: data/raw, data, or project root."""
    candidates = [
        f"data/raw/{filename}",
        f"data/{filename}",
        filename,
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(filename)


def load_data():
    """Load all CSV files"""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    try:
        customer_history = pd.read_csv(_resolve_data_path('customer_history.csv'))
        customers = pd.read_csv(_resolve_data_path('customers.csv'))
        reference_data = pd.read_csv(_resolve_data_path('reference_data.csv'))
        reference_data_test = pd.read_csv(_resolve_data_path('reference_data_test.csv'))

        print(f"✓ Customer history shape: {customer_history.shape}")
        print(f"✓ Customers shape: {customers.shape}")
        print(f"✓ Reference train shape: {reference_data.shape}")
        print(f"✓ Reference test shape: {reference_data_test.shape}")
        print(f"✓ Churn rate: {reference_data['churn'].mean():.4f}")

        return customer_history, customers, reference_data, reference_data_test

    except FileNotFoundError as e:
        print(f"\n✗ ERROR: Could not find required CSV file: {e}")
        print("  Please ensure all CSV files are available under one of:")
        print("  - data/raw/")
        print("  - data/")
        print("  - project root (current directory)")
        print("  - customer_history.csv")
        print("  - customers.csv")
        print("  - reference_data.csv")
        print("  - reference_data_test.csv")
        raise
    except Exception as e:
        print(f"\n✗ ERROR loading data: {e}")
        raise

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_training_features(customer_history, customers, reference_data):
    """Create features for all training reference dates"""
    print("\n" + "="*60)
    print("CREATING TRAINING FEATURES")
    print("="*60)

    try:
        from src.features.feature_engineering import ChurnFeatureEngineering

        fe = ChurnFeatureEngineering()
        train_features_list = []

        unique_dates = reference_data['ref_date'].unique()
        print(f"Processing {len(unique_dates)} reference dates...")

        for idx, ref_date in enumerate(unique_dates, 1):
            print(f"\n[{idx}/{len(unique_dates)}] Processing ref_date: {ref_date}")

            # Get customers for this ref_date
            ref_customers = reference_data[reference_data['ref_date'] == ref_date]['cust_id'].unique()
            print(f"  Customers: {len(ref_customers)}")

            # Filter history up to ref_date
            ref_date_dt = pd.to_datetime(ref_date)
            history_subset = customer_history[
                (customer_history['cust_id'].isin(ref_customers)) &
                (pd.to_datetime(customer_history['date']) <= ref_date_dt)
            ].copy()
            print(f"  History records: {len(history_subset)}")

            customers_subset = customers[customers['cust_id'].isin(ref_customers)].copy()

            # Create features
            features = fe.create_all_features(history_subset, customers_subset, ref_date)
            features['ref_date'] = ref_date

            train_features_list.append(features)
            print(f"  ✓ Created {features.shape[1]} features for {features.shape[0]} customers")

        # Combine all training features
        train_features = pd.concat(train_features_list, axis=0, ignore_index=True)
        print(f"\n✓ Combined training features shape: {train_features.shape}")

        # Merge with labels
        train_data = train_features.merge(
            reference_data[['cust_id', 'ref_date', 'churn']],
            on=['cust_id', 'ref_date'],
            how='left'
        )

        print(f"✓ Training data with labels shape: {train_data.shape}")

        missing_labels = train_data['churn'].isna().sum()
        if missing_labels > 0:
            print(f"⚠ WARNING: {missing_labels} missing labels")
        else:
            print(f"✓ No missing labels")

        return train_data

    except ImportError:
        print("\n✗ ERROR: Could not import ChurnFeatureEngineering")
        print("  Please ensure feature_engineering.py is in the current directory")
        raise
    except Exception as e:
        print(f"\n✗ ERROR creating training features: {e}")
        raise

def create_test_features(customer_history, customers, reference_data_test):
    """Create features for all test reference dates"""
    print("\n" + "="*60)
    print("CREATING TEST FEATURES")
    print("="*60)

    try:
        from src.features.feature_engineering import ChurnFeatureEngineering

        fe = ChurnFeatureEngineering()
        test_features_list = []

        unique_dates = reference_data_test['ref_date'].unique()
        print(f"Processing {len(unique_dates)} test reference dates...")

        for idx, ref_date in enumerate(unique_dates, 1):
            print(f"\n[{idx}/{len(unique_dates)}] Processing test ref_date: {ref_date}")

            ref_customers = reference_data_test[reference_data_test['ref_date'] == ref_date]['cust_id'].unique()
            print(f"  Customers: {len(ref_customers)}")

            ref_date_dt = pd.to_datetime(ref_date)
            history_subset = customer_history[
                (customer_history['cust_id'].isin(ref_customers)) &
                (pd.to_datetime(customer_history['date']) <= ref_date_dt)
            ].copy()
            print(f"  History records: {len(history_subset)}")

            customers_subset = customers[customers['cust_id'].isin(ref_customers)].copy()

            features = fe.create_all_features(history_subset, customers_subset, ref_date)
            features['ref_date'] = ref_date

            test_features_list.append(features)
            print(f"  ✓ Created {features.shape[1]} features for {features.shape[0]} customers")

        test_features = pd.concat(test_features_list, axis=0, ignore_index=True)
        print(f"\n✓ Combined test features shape: {test_features.shape}")

        return test_features

    except Exception as e:
        print(f"\n✗ ERROR creating test features: {e}")
        raise

# ============================================================================
# PREPARE DATA FOR MODELING
# ============================================================================

def prepare_modeling_data(train_data, test_features):
    """Prepare X_train, y_train, X_test with all preprocessing, and ref_dates aligned to X_train"""
    print("\n" + "="*60)
    print("PREPARING DATA FOR MODELING")
    print("="*60)

    try:
        # Separate features and target
        feature_cols = [col for col in train_data.columns
                        if col not in ['cust_id', 'ref_date', 'churn']]

        X_train = train_data[feature_cols].copy()
        y_train = train_data['churn'].copy()
        ref_dates = train_data['ref_date'].copy()
        X_test = test_features[feature_cols].copy()

        print(f"✓ X_train shape: {X_train.shape}")
        print(f"✓ y_train shape: {y_train.shape}")
        print(f"✓ X_test shape: {X_test.shape}")

        # Handle infinity values
        print("\nHandling infinity values...")
        inf_count_train = np.isinf(X_train.values).sum()
        inf_count_test = np.isinf(X_test.values).sum()

        if inf_count_train > 0 or inf_count_test > 0:
            print(f"  Found {inf_count_train} inf values in train, {inf_count_test} in test")
            X_train = X_train.replace([np.inf, -np.inf], -999)
            X_test = X_test.replace([np.inf, -np.inf], -999)
            print("  ✓ Replaced with -999")
        else:
            print("  ✓ No infinity values found")

        return X_train, y_train, X_test, feature_cols, ref_dates

    except Exception as e:
        print(f"\n✗ ERROR preparing modeling data: {e}")
        raise

# ============================================================================
# FEATURE INTERACTIONS
# ============================================================================

def add_feature_interactions(X_train, X_test):
    """Create interaction features between top predictive features"""
    print("\n" + "="*60)
    print("CREATING FEATURE INTERACTIONS")
    print("="*60)

    try:
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

        print(f"  ✓ Created {len(existing_features) * (len(existing_features) - 1) // 2} feature pairs")

        # Handle any infinity values created by interactions
        inf_count_train = np.isinf(X_train.values).sum()
        inf_count_test = np.isinf(X_test.values).sum()

        if inf_count_train > 0 or inf_count_test > 0:
            print(f"\n  Handling {inf_count_train} inf values in train, {inf_count_test} in test")
            X_train = X_train.replace([np.inf, -np.inf], -999)
            X_test = X_test.replace([np.inf, -np.inf], -999)

        print(f"\n✓ Total interactions created: {interaction_count}")
        print(f"✓ New X_train shape: {X_train.shape}")
        print(f"✓ New X_test shape: {X_test.shape}")

        return X_train, X_test

    except Exception as e:
        print(f"\n✗ ERROR creating feature interactions: {e}")
        raise

# ============================================================================
# SAVE PICKLE FILES
# ============================================================================

def save_pickle_files(X_train, y_train, X_test, feature_cols, ref_dates):
    """Save all data as pickle files, including ref_dates aligned to X_train"""
    print("\n" + "="*60)
    print("SAVING PICKLE FILES")
    print("="*60)

    try:
        # Save X_train
        print("\nSaving X_train.pkl...")
        with open('X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        print(f"  ✓ Saved X_train.pkl ({X_train.shape[0]} rows, {X_train.shape[1]} columns)")

        # Save y_train
        print("\nSaving y_train.pkl...")
        with open('y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        print(f"  ✓ Saved y_train.pkl ({len(y_train)} labels)")

        # Save X_test
        print("\nSaving X_test.pkl...")
        with open('X_test.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        print(f"  ✓ Saved X_test.pkl ({X_test.shape[0]} rows, {X_test.shape[1]} columns)")

        # Save feature_cols
        print("\nSaving feature_cols.pkl...")
        with open('feature_cols.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        print(f"  ✓ Saved feature_cols.pkl ({len(feature_cols)} feature names)")

        # Save ref_dates
        print("\nSaving ref_dates.pkl...")
        with open('ref_dates.pkl', 'wb') as f:
            pickle.dump(ref_dates, f)
        print(f"  ✓ Saved ref_dates.pkl ({len(ref_dates)} dates, aligned to X_train)")

        # Print summary
        print("\n" + "="*60)
        print("SAVE COMPLETE!")
        print("="*60)
        print("\nSaved files:")
        print("  - X_train.pkl")
        print("  - y_train.pkl")
        print("  - X_test.pkl")
        print("  - feature_cols.pkl")
        print("  - ref_dates.pkl")

        print("\nTo load these files later:")
        print("  import pickle")
        print("  with open('X_train.pkl', 'rb') as f:")
        print("      X_train = pickle.load(f)")

    except Exception as e:
        print(f"\n✗ ERROR saving pickle files: {e}")
        raise

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    start_time = datetime.now()

    print("\n" + "="*60)
    print("ING DATATHON - SAVE TRAINING DATA")
    print("="*60)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Step 1: Load data
        customer_history, customers, reference_data, reference_data_test = load_data()

        # Step 2: Create training features
        train_data = create_training_features(customer_history, customers, reference_data)

        # Step 3: Create test features
        test_features = create_test_features(customer_history, customers, reference_data_test)

        # Step 4: Prepare modeling data
        X_train, y_train, X_test, feature_cols, ref_dates = prepare_modeling_data(train_data, test_features)

        # Step 5: Add feature interactions
        X_train, X_test = add_feature_interactions(X_train, X_test)

        # Update feature_cols to include interactions
        feature_cols = list(X_train.columns)

        # Step 6: Save pickle files
        save_pickle_files(X_train, y_train, X_test, feature_cols, ref_dates)

        # Final summary
        end_time = datetime.now()
        elapsed = end_time - start_time

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {elapsed}")
        print("\nYou can now run hyperparameter optimization without")
        print("re-running expensive feature engineering!")

    except Exception as e:
        print("\n" + "="*60)
        print("FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
