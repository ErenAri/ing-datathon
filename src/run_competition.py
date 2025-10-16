"""
ING Hubs Türkiye Datathon - Competition Runner
Production-ready script with error handling, validation, and beautiful output
"""

import sys
import os
import traceback
from datetime import datetime
import pandas as pd
import numpy as np


def print_header(text, char="=", width=80):
    """Print a formatted header"""
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def print_section(text, char="-", width=80):
    """Print a formatted section header"""
    print("\n" + char * width)
    print(text)
    print(char * width)


def validate_input_files():
    """Validate that all required input files exist"""
    required_files = [
        'customer_history.csv',
        'customers.csv',
        'reference_data.csv',
        'reference_data_test.csv',
        'feature_engineering.py',
        'modeling_pipeline.py',
        'main.py'
    ]

    print_section("Validating Input Files")
    missing_files = []

    for file in required_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"  ✓ {file:<35} ({file_size:.2f} MB)")
        else:
            print(f"  ✗ {file:<35} MISSING")
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ ERROR: Missing {len(missing_files)} required file(s)")
        return False

    print(f"\n✓ All {len(required_files)} required files found")
    return True


def run_main_pipeline():
    """Run the main competition pipeline with error handling"""
    print_section("Running Competition Pipeline")

    try:
        # Capture important variables before running main
        import src.main as main

        # Main script runs automatically on import
        print("\n✓ Pipeline executed successfully")

        return {
            'success': True,
            'error': None
        }

    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def validate_submission():
    """Validate the submission file"""
    print_section("Validating Submission File")

    if not os.path.exists('submission.csv'):
        print("  ❌ submission.csv not found")
        return False

    try:
        submission = pd.read_csv('submission.csv')

        # Check structure
        required_columns = ['cust_id', 'churn']
        missing_cols = [col for col in required_columns if col not in submission.columns]

        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            return False

        # Check for missing values
        if submission.isnull().any().any():
            null_counts = submission.isnull().sum()
            print(f"  ⚠️  Warning: Found null values:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"      {col}: {count} nulls")
            return False

        # Check prediction range
        churn_min = submission['churn'].min()
        churn_max = submission['churn'].max()

        if churn_min < 0 or churn_max > 1:
            print(f"  ⚠️  Warning: Predictions outside [0, 1] range")
            print(f"      Range: [{churn_min:.4f}, {churn_max:.4f}]")

        # Validation passed
        print(f"  ✓ File structure valid")
        print(f"  ✓ Shape: {submission.shape}")
        print(f"  ✓ No missing values")
        print(f"  ✓ Prediction range: [{churn_min:.6f}, {churn_max:.6f}]")

        return True, submission

    except Exception as e:
        print(f"  ❌ Error reading submission: {str(e)}")
        return False, None


def show_submission_preview(submission, n=10):
    """Display a nicely formatted preview of submission"""
    print_section("Submission Preview (Top Predictions)")

    # Get top predictions
    top_pred = submission.nlargest(n, 'churn')

    print(f"\n  {'Rank':<6} {'Customer ID':<15} {'Churn Probability':<20}")
    print(f"  {'-'*6} {'-'*15} {'-'*20}")

    for idx, (i, row) in enumerate(top_pred.iterrows(), 1):
        prob = row['churn']
        bar_length = int(prob * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"  {idx:<6} {row['cust_id']:<15} {prob:.6f} |{bar}|")

    print(f"\n  Statistics:")
    print(f"    Mean:   {submission['churn'].mean():.6f}")
    print(f"    Median: {submission['churn'].median():.6f}")
    print(f"    Std:    {submission['churn'].std():.6f}")


def show_output_files():
    """Show all generated output files"""
    print_section("Generated Output Files")

    output_files = [
        'submission.csv',
        'feature_importance.csv'
    ]

    for file in output_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / 1024  # Size in KB
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"  ✓ {file:<30} {file_size:>8.2f} KB  [{mod_time.strftime('%Y-%m-%d %H:%M:%S')}]")
        else:
            print(f"  ✗ {file:<30} NOT CREATED")


def show_feature_importance(top_n=15):
    """Display top features if available"""
    if not os.path.exists('feature_importance.csv'):
        return

    print_section(f"Top {top_n} Most Important Features")

    try:
        feat_imp = pd.read_csv('feature_importance.csv')
        top_features = feat_imp.head(top_n)

        print(f"\n  {'Rank':<6} {'Feature Name':<50} {'Importance':<12}")
        print(f"  {'-'*6} {'-'*50} {'-'*12}")

        max_importance = top_features['importance_mean'].max()

        for idx, (i, row) in enumerate(top_features.iterrows(), 1):
            feat_name = row['feature'][:48]  # Truncate long names
            importance = row['importance_mean']
            bar_length = int((importance / max_importance) * 30)
            bar = '█' * bar_length
            print(f"  {idx:<6} {feat_name:<50} {importance:>10.2f} {bar}")

    except Exception as e:
        print(f"  ⚠️  Could not load feature importance: {str(e)}")


def print_final_summary():
    """Print beautiful final summary"""
    print_header("🏆 COMPETITION PIPELINE COMPLETE 🏆", "=", 80)

    print("\n" + "┌" + "─" * 78 + "┐")
    print("│" + " SUBMISSION READY FOR UPLOAD ".center(78) + "│")
    print("└" + "─" * 78 + "┘")

    print("\n📁 Next Steps:")
    print("   1. Review your submission.csv file")
    print("   2. Upload submission.csv to the competition platform")
    print("   3. Check your leaderboard score")
    print("   4. Iterate and improve!")

    print("\n💡 Tips to Improve Your Score:")
    print("   • Tune hyperparameters with Optuna")
    print("   • Try different ensemble weights")
    print("   • Add more domain-specific features")
    print("   • Experiment with different calibration strategies")
    print("   • Use pseudo-labeling for semi-supervised learning")

    print("\n" + "="*80)
    print(f"  Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def main():
    """Main execution function"""
    start_time = datetime.now()

    print_header("🚀 ING HUBS TÜRKİYE DATATHON 🚀", "=", 80)
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Validate input files
    if not validate_input_files():
        print("\n❌ FAILED: Missing required input files")
        sys.exit(1)

    # Step 2: Run main pipeline
    result = run_main_pipeline()

    if not result['success']:
        print("\n❌ FAILED: Pipeline execution error")
        print("\nTraceback:")
        print(result['traceback'])
        sys.exit(1)

    # Step 3: Validate submission
    validation_result = validate_submission()

    if isinstance(validation_result, tuple):
        is_valid, submission = validation_result
    else:
        is_valid = validation_result
        submission = None

    if not is_valid:
        print("\n❌ FAILED: Submission validation error")
        sys.exit(1)

    # Step 4: Show results
    if submission is not None:
        show_submission_preview(submission, n=10)

    show_output_files()
    show_feature_importance(top_n=15)

    # Calculate execution time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n⏱️  Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # Print final summary
    print_final_summary()

    print("✅ SUCCESS: All steps completed successfully!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
