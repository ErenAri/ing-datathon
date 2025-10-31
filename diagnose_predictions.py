"""
Quick diagnostic for prediction issues

Checks:
1. Prediction scale and distribution
2. AUC direction
3. Correlation between models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys

def diagnose_predictions(oof_dict, y_true):
    """Diagnose prediction issues."""
    print("\n" + "="*70)
    print("PREDICTION DIAGNOSTICS")
    print("="*70)

    for name, preds in oof_dict.items():
        preds = np.asarray(preds, dtype=float)

        print(f"\n{name.upper()}:")
        print(f"  Shape: {preds.shape}")
        print(f"  Range: [{preds.min():.6f}, {preds.max():.6f}]")
        print(f"  Mean: {preds.mean():.6f}")
        print(f"  Std: {preds.std():.6f}")
        print(f"  NaN count: {np.isnan(preds).sum()}")
        print(f"  Inf count: {np.isinf(preds).sum()}")

        # Check AUC
        try:
            auc = roc_auc_score(y_true, preds)
            print(f"  AUC: {auc:.6f} {'✓' if auc > 0.5 else '⚠ INVERTED'}")
        except Exception as e:
            print(f"  AUC: ERROR - {e}")

        # Check distribution
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(preds, bins=bins)
        print(f"  Distribution:")
        for i in range(len(bins)-1):
            pct = hist[i] / len(preds) * 100
            bar = '█' * int(pct / 2)
            print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {pct:5.1f}% {bar}")

    # Correlation between models
    print(f"\n{'='*70}")
    print("MODEL CORRELATION MATRIX")
    print(f"{'='*70}")

    pred_df = pd.DataFrame({name: np.asarray(preds, dtype=float)
                            for name, preds in oof_dict.items()})
    corr = pred_df.corr()
    print(corr.round(3))

    # Check average
    print(f"\n{'='*70}")
    print("SIMPLE AVERAGE TEST")
    print(f"{'='*70}")

    avg = pred_df.mean(axis=1).values
    print(f"Average predictions:")
    print(f"  Range: [{avg.min():.6f}, {avg.max():.6f}]")
    print(f"  Mean: {avg.mean():.6f}")

    try:
        auc_avg = roc_auc_score(y_true, avg)
        print(f"  AUC: {auc_avg:.6f}")

        from src.models.modeling_pipeline import ing_hubs_datathon_metric
        score_avg, metrics = ing_hubs_datathon_metric(y_true, avg)
        print(f"  Competition Score: {score_avg:.6f}")
        print(f"    Gini: {metrics['gini']:.4f}")
        print(f"    Recall@10: {metrics['recall@10']:.4f}")
        print(f"    Lift@10: {metrics['lift@10']:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    print("This script is meant to be imported, not run directly.")
    print("Use it in your main script like:")
    print("  from diagnose_predictions import diagnose_predictions")
    print("  diagnose_predictions(oof_predictions, y_train)")
