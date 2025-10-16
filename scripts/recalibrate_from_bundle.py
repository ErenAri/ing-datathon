"""
Recalibrate predictions from predictions_bundle.pkl and write new submissions.

Usage examples:
  # Recalibrate ensemble with isotonic and a gamma sweep
  python scripts/recalibrate_from_bundle.py --method isotonic --gamma-grid 0.85,0.90,0.95,1.00,1.05

  # Beta calibration with a different grid
  python scripts/recalibrate_from_bundle.py --method beta --gamma-grid 0.90,0.95,1.00,1.05,1.10

  # Save with a custom name
  python scripts/recalibrate_from_bundle.py --out-name submission_cal_iso.csv

Notes:
  - Reads OOF/test probs from outputs/predictions/predictions_bundle.pkl
  - Defaults to using ensemble probs (oof_ensemble, test_ensemble_raw)
  - Uses reference_data_test.csv for customer ids order; falls back to existing submission.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def _load_bundle(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _resolve_test_ids() -> pd.Series:
    # Prefer reference_data_test.csv at repo root
    ref_test = 'reference_data_test.csv'
    sub_csv = os.path.join('data', 'submissions', 'submission.csv')
    if os.path.exists(ref_test):
        df = pd.read_csv(ref_test)
        if 'cust_id' in df.columns:
            return df['cust_id']
        # Fallback to first column if cust_id missing
        return df.iloc[:, 0]
    if os.path.exists(sub_csv):
        df = pd.read_csv(sub_csv)
        if 'cust_id' in df.columns:
            return df['cust_id']
    raise FileNotFoundError(
        "Could not resolve test customer ids. Ensure reference_data_test.csv or data/submissions/submission.csv exists."
    )


def _parse_gamma_grid(s: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return tuple(float(p) for p in parts)


def _monthwise_composite(y: np.ndarray, p: np.ndarray, ref_dates: np.ndarray) -> float:
    # Minimal local reimplementation to avoid heavy imports
    # Composite = 0.4 * gini + 0.3 * recall@10 + 0.3 * lift@10, averaged per-month
    import numpy as np
    import pandas as pd

    def _gini(y_true, y_prob):
        # Normalized Gini from AUC: Gini = 2*AUC - 1; approx from roc_auc_score
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            return 0.0
        return 2.0 * auc - 1.0

    def _recall_at_k(y_true, y_prob, k=0.10):
        n = len(y_true)
        k_n = max(1, int(np.ceil(k * n)))
        idx = np.argsort(-y_prob)[:k_n]
        tp = float(np.sum(y_true[idx]))
        pos = float(np.sum(y_true))
        return (tp / pos) if pos > 0 else 0.0

    def _lift_at_k(y_true, y_prob, k=0.10):
        rec = _recall_at_k(y_true, y_prob, k)
        return rec / k if k > 0 else 0.0

    m = pd.to_datetime(pd.Series(ref_dates)).dt.to_period('M').astype(str).values
    months = sorted(set(m))
    scores = []
    for mo in months:
        mask = (m == mo)
        yt = y[mask]
        pt = p[mask]
        if yt.size == 0:
            continue
        comp = 0.4 * _gini(yt, pt) + 0.3 * _recall_at_k(yt, pt, 0.10) + 0.3 * _lift_at_k(yt, pt, 0.10)
        scores.append(comp)
    return float(np.mean(scores)) if scores else 0.0


def recalibrate(
    bundle_path: str,
    method: str,
    gamma_grid: Tuple[float, ...],
    source: str = 'ensemble',
    out_name: Optional[str] = None,
) -> str:
    from src.utils.calibration import kfold_calibrate_with_gamma, CalibrationConfig

    bundle = _load_bundle(bundle_path)
    y = np.asarray(bundle['y_train'], dtype=float)
    ref_dates = np.asarray(bundle['ref_dates'])

    key_map = {
        'ensemble': ('oof_ensemble', 'test_ensemble_raw'),
        'meta': ('oof_meta', 'test_meta'),
        'cat': ('oof_cat', 'test_cat'),
        'lgb': ('oof_lgb', 'test_lgb'),
        'xgb': ('oof_xgb', 'test_xgb'),
        'two': ('oof_two_stage_B', 'test_two_stage_B'),
    }
    if source not in key_map:
        raise ValueError(f"Unknown source='{source}'. Choose from {list(key_map)}")
    oof_key, test_key = key_map[source]
    if oof_key not in bundle or test_key not in bundle:
        raise KeyError(f"Bundle missing keys for source '{source}': {oof_key}, {test_key}")

    oof = np.asarray(bundle[oof_key], dtype=float)
    test = np.asarray(bundle[test_key], dtype=float)

    # Pre-score on OOF for reference
    pre_score = _monthwise_composite(y, oof, ref_dates)

    if method == 'none':
        # Just gamma power sweep without calibrator
        best_gamma = 1.0
        best_score = -1.0
        best_test = test.copy()
        eps = 1e-6
        for g in gamma_grid:
            oof_adj = np.clip(np.power(oof, g), eps, 1.0 - eps)
            sc = _monthwise_composite(y, oof_adj, ref_dates)
            if sc > best_score:
                best_score = sc
                best_gamma = g
        test_calibrated = np.clip(np.power(test, best_gamma), eps, 1.0 - eps)
        post_score = best_score
    else:
        cfg = CalibrationConfig(
            n_folds=5,
            gamma_grid=tuple(gamma_grid),
            raw_blend=0.2,
        )
        result = kfold_calibrate_with_gamma(
            y=y,
            oof=oof,
            ref_dates=pd.Series(ref_dates),
            method=method,  # type: ignore[arg-type]
            folds=None,
            segment_values=None,
            cfg=cfg,
        )
        post_score = float(result['post_score'])
        best_gamma = float(result['best_gamma'])
        # Apply to test
        test_calibrated = result['test_cal_fn'](test)

    # Save submission
    cust_ids = _resolve_test_ids()
    if len(cust_ids) != len(test):
        raise ValueError(f"Length mismatch: cust_ids={len(cust_ids)} vs test_preds={len(test)}")
    sub = pd.DataFrame({'cust_id': cust_ids, 'churn': test_calibrated})
    os.makedirs(os.path.join('data', 'submissions'), exist_ok=True)
    if out_name is None:
        out_name = f"submission_cal_{source}_{method}_g{str(best_gamma).replace('.', '')}.csv"
    out_path = os.path.join('data', 'submissions', out_name)
    sub.to_csv(out_path, index=False)

    print("Recalibration summary:")
    print(f"  Source:        {source}")
    print(f"  Method:        {method}")
    print(f"  Gamma grid:    {gamma_grid}")
    print(f"  Pre OOF score: {pre_score:.6f}")
    print(f"  Post OOF:      {post_score:.6f}")
    print(f"  Best gamma:    {best_gamma}")
    print(f"  Saved:         {out_path}")

    # Update last_update marker
    try:
        update_path = os.path.join('data', 'submissions', 'last_update.txt')
        with open(update_path, 'a', encoding='utf-8') as _u:
            _u.write(
                f"updated_at={pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | file={os.path.basename(out_path)} | rows={len(sub)}\n"
            )
    except Exception:
        pass

    return out_path


def main():
    ap = argparse.ArgumentParser(description='Recalibrate predictions from predictions_bundle.pkl')
    ap.add_argument('--bundle', type=str, default=os.path.join('outputs', 'predictions', 'predictions_bundle.pkl'))
    ap.add_argument('--method', type=str, choices=['isotonic', 'beta', 'none'], default='isotonic')
    ap.add_argument('--gamma-grid', type=str, default='0.85,0.90,0.95,1.00,1.05')
    ap.add_argument('--source', type=str, choices=['ensemble', 'meta', 'cat', 'lgb', 'xgb', 'two'], default='ensemble')
    ap.add_argument('--out-name', type=str, default=None)
    args = ap.parse_args()

    gamma = _parse_gamma_grid(args.gamma_grid)
    _ = recalibrate(
        bundle_path=args.bundle,
        method=args.method,
        gamma_grid=gamma,
        source=args.source,
        out_name=args.out_name,
    )


if __name__ == '__main__':
    main()
