"""
Portfolio submission generator.

Goal: Emit 15–25 diverse submissions without retraining by varying:
- per-month blend weights (2018-09/11/12 emphasis)
- calibration method (isotonic vs beta-like)
- power gamma (temperature)
- top-decile emphasis (Stage-A weight inside blend)

Inputs:
- predictions_bundle.pkl (created by main.py)
- reference_data_test.csv (for cust_id/ref_date ordering), used only for submission row alignment

Outputs:
- submission_XX.csv files
- README_portfolio.md documenting each variant and knobs
"""

import os
import json
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Local metric helpers: replicate the month-wise composite scorer
from src.models.modeling_pipeline import oof_composite_monthwise, ing_hubs_datathon_metric as _metric_fn

EPS = 1e-6
RAW_BLEND = 0.2
SEG_MIN = 2000

def _segment_isotonic(oof: np.ndarray, y: np.ndarray, test_raw: np.ndarray,
                      tenure_train: Optional[np.ndarray] = None, tenure_test: Optional[np.ndarray] = None):
    iso_global = IsotonicRegression(out_of_bounds='clip')
    iso_global.fit(oof, y)
    iso_oof = iso_global.transform(oof)
    iso_test = iso_global.transform(test_raw)

    # Optional segmentation by tenure if provided
    if tenure_train is not None and tenure_test is not None:
        try:
            q_edges = np.quantile(tenure_train[~np.isnan(tenure_train)], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            edges = np.unique(q_edges)
            if len(edges) >= 3:
                bucket_train = np.digitize(tenure_train, edges[1:-1], right=False)
                bucket_test = np.digitize(tenure_test, edges[1:-1], right=False)
                iso_oof_seg = np.empty_like(oof, dtype=float)
                iso_test_seg = np.empty_like(test_raw, dtype=float)
                total_segments = len(edges) - 1
                for b in range(total_segments):
                    mask_tr = bucket_train == b
                    mask_te = bucket_test == b
                    if int(mask_tr.sum()) >= SEG_MIN:
                        iso_b = IsotonicRegression(out_of_bounds='clip')
                        iso_b.fit(oof[mask_tr], y[mask_tr])
                        iso_oof_seg[mask_tr] = iso_b.transform(oof[mask_tr])
                        iso_test_seg[mask_te] = iso_b.transform(test_raw[mask_te])
                    else:
                        iso_oof_seg[mask_tr] = iso_oof[mask_tr]
                        iso_test_seg[mask_te] = iso_test[mask_te]
                iso_oof = iso_oof_seg
                iso_test = iso_test_seg
        except Exception:
            pass

    iso_oof = np.clip(iso_oof, EPS, 1.0 - EPS)
    iso_test = np.clip(iso_test, EPS, 1.0 - EPS)
    iso_oof = (1.0 - RAW_BLEND) * iso_oof + RAW_BLEND * oof
    iso_test = (1.0 - RAW_BLEND) * iso_test + RAW_BLEND * test_raw
    return iso_oof, iso_test


def _beta_calibration(oof: np.ndarray, y: np.ndarray, test_raw: np.ndarray):
    p_tr = np.clip(oof.astype(float), EPS, 1.0 - EPS)
    X_tr = np.column_stack([np.log(p_tr), np.log(1.0 - p_tr)])
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_tr, y)
    oof_cal = clf.predict_proba(X_tr)[:, 1]
    p_te = np.clip(test_raw.astype(float), EPS, 1.0 - EPS)
    X_te = np.column_stack([np.log(p_te), np.log(1.0 - p_te)])
    te_cal = clf.predict_proba(X_te)[:, 1]
    oof_cal = np.clip(oof_cal, EPS, 1.0 - EPS)
    te_cal = np.clip(te_cal, EPS, 1.0 - EPS)
    oof_cal = (1.0 - RAW_BLEND) * oof_cal + RAW_BLEND * oof
    te_cal = (1.0 - RAW_BLEND) * te_cal + RAW_BLEND * test_raw
    return oof_cal, te_cal


def _safe_auc_flip(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    from sklearn.metrics import roc_auc_score
    try:
        if roc_auc_score(y, p) < 0.5:
            return 1.0 - p
    except Exception:
        pass
    return p


def _blend(models: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    out = np.zeros_like(next(iter(models.values())))
    for n, w in weights.items():
        out = out + float(w) * models[n]
    return out


@dataclass
class Variant:
    name: str
    weights: Dict[str, float]
    month_weights: Dict[str, Dict[str, float]]
    calibration: str
    gamma: float
    stageA_weight: float
    score: float


def main():
    # Load bundle
    if not os.path.exists('predictions_bundle.pkl'):
        print('ERROR: predictions_bundle.pkl not found. Run main.py once to generate it.')
        return 1

    with open('predictions_bundle.pkl', 'rb') as f:
        B = pickle.load(f)

    y = B['y_train']
    # Normalize to pandas Series for consistent .dt access
    ref_dates = pd.Series(pd.to_datetime(B['ref_dates']))

    # Collect available base model OOF/test
    oof_models = {k: B[k] for k in ['oof_lgb', 'oof_xgb', 'oof_two_stage_B'] if k in B}
    test_models = {k.replace('oof_', 'test_'): B[k.replace('oof_', 'test_')] for k in oof_models.keys()}
    # Stage-A available separately
    if 'oof_two_stage_A' in B and 'test_two_stage_A' in B:
        oof_stageA = B['oof_two_stage_A']
        test_stageA = B['test_two_stage_A']
    else:
        oof_stageA = None
        test_stageA = None

    if 'oof_cat' in B and 'test_cat' in B:
        oof_models['oof_cat'] = B['oof_cat']
        test_models['test_cat'] = B['test_cat']

    # Define month masks
    m = ref_dates.dt.to_period('M')
    target_months = ['2018-09', '2018-11', '2018-12']
    existing = set(map(str, sorted(m.unique())))
    month_masks = {tm: (m.astype(str).values == tm) for tm in target_months if tm in existing}

    # Base weight templates to explore (sum to ~1)
    base_weight_sets = []
    # Equal
    base_weight_sets.append({'oof_lgb': 0.33, 'oof_xgb': 0.33, 'oof_two_stage_B': 0.34})
    # LGB heavy
    base_weight_sets.append({'oof_lgb': 0.50, 'oof_xgb': 0.25, 'oof_two_stage_B': 0.25})
    # XGB heavy
    base_weight_sets.append({'oof_lgb': 0.25, 'oof_xgb': 0.50, 'oof_two_stage_B': 0.25})
    # Two-stage heavy
    base_weight_sets.append({'oof_lgb': 0.25, 'oof_xgb': 0.25, 'oof_two_stage_B': 0.50})
    # If Cat present, add variants
    if 'oof_cat' in oof_models:
        base_weight_sets.append({'oof_lgb': 0.30, 'oof_xgb': 0.30, 'oof_two_stage_B': 0.30, 'oof_cat': 0.10})
        base_weight_sets.append({'oof_lgb': 0.40, 'oof_xgb': 0.20, 'oof_two_stage_B': 0.20, 'oof_cat': 0.20})

    # Month aggregation emphasis to explore
    agg_emphasis = [
        {'2018-09': 0.2, '2018-11': 0.3, '2018-12': 0.5},
        {'2018-09': 0.1, '2018-11': 0.3, '2018-12': 0.6},
        {'2018-09': 0.3, '2018-11': 0.3, '2018-12': 0.4}
    ]

    # Stage-A emphasis inside blend (adds stageA * wA to two-stageB)
    stageA_weights = [0.0, 0.05, 0.10]

    # Calibration choices and gammas
    calibrators = ['isotonic', 'beta']
    gammas = [0.95, 1.00, 1.05, 1.10]

    variants: List[Variant] = []

    # Build helper mapping for test arrays
    test_map = {
        'oof_lgb': B['test_lgb'],
        'oof_xgb': B['test_xgb'],
        'oof_two_stage_B': B['test_two_stage_B']
    }
    if 'oof_cat' in oof_models:
        test_map['oof_cat'] = B['test_cat']

    # Iterate grid (keep total around 15–25)
    for base_idx, base_w in enumerate(base_weight_sets):
        # Per-month optimized weights: lightly perturb base via month emphasis coefficients
        for agg in agg_emphasis:
            # Compute month-specific best weights by scoring base weights on each month (no heavy grid)
            per_month_w = {}
            for tm, mask in month_masks.items():
                per_month_w[tm] = {k: float(v) for k, v in base_w.items()}

            # Aggregate to a single set using emphasis
            present = [k for k in agg.keys() if k in per_month_w]
            if not present:
                final_w = {k: float(v) for k, v in base_w.items()}
            else:
                total = sum(agg[k] for k in present)
                final_w = {k: 0.0 for k in base_w.keys()}
                for k in present:
                    alpha = agg[k] / total
                    for n in base_w.keys():
                        final_w[n] += alpha * per_month_w[k][n]

            # Stage-A emphasis fold-in
            for wA in stageA_weights:
                weights_plus = dict(final_w)
                if wA > 0 and oof_stageA is not None:
                    # Steal weight proportionally from others to add stage-A
                    steal = wA
                    denom = sum(weights_plus.values())
                    if denom <= 0:
                        denom = 1.0
                    for n in list(weights_plus.keys()):
                        weights_plus[n] = max(0.0, weights_plus[n] * (1.0 - steal))
                    weights_plus['oof_two_stage_A'] = wA

                # Build blended OOF and test
                oof_all = dict(oof_models)
                test_all = dict(test_map)
                if 'oof_two_stage_A' in weights_plus:
                    oof_all['oof_two_stage_A'] = oof_stageA
                    test_all['oof_two_stage_A'] = test_stageA

                # Renormalize to sum 1
                s = sum(weights_plus.values())
                if s <= 0:
                    continue
                weights_norm = {k: float(v) / s for k, v in weights_plus.items()}

                blend_oof = _blend(oof_all, weights_norm)
                blend_oof = _safe_auc_flip(y, blend_oof)

                # Test blend uses the same keys as OOF (mapped to test arrays in test_all)
                blend_test = _blend(test_all, weights_norm)

                # Calibrate
                for cal in calibrators:
                    if cal == 'isotonic':
                        oof_c, te_c = _segment_isotonic(blend_oof, y, blend_test)
                    else:
                        oof_c, te_c = _beta_calibration(blend_oof, y, blend_test)

                    # Gamma sweep (small set)
                    for g in gammas:
                        oof_adj = np.clip(np.power(oof_c, g), EPS, 1.0 - EPS)
                        score = oof_composite_monthwise(y, oof_adj, ref_dates, last_n_months=6)
                        name = f"b{base_idx+1}-agg{agg['2018-09']:.1f}-{agg['2018-11']:.1f}-{agg['2018-12']:.1f}-A{wA:.2f}-{cal}-g{g:.2f}"
                        variants.append(Variant(
                            name=name,
                            weights=weights_norm,
                            month_weights=per_month_w,
                            calibration=cal,
                            gamma=float(g),
                            stageA_weight=float(wA),
                            score=float(score)
                        ))

    # Select top ~20 variants by OOF composite
    variants = sorted(variants, key=lambda v: v.score, reverse=True)[:22]

    # Emit submissions and README
    os.makedirs('portfolio', exist_ok=True)
    reference_data_test = pd.read_csv('reference_data_test.csv')
    cust_ids = reference_data_test['cust_id'].values

    rows = []
    for i, v in enumerate(variants, 1):
        # Rebuild one more time to get test predictions
        # Build test blend using selected weights
        weights = v.weights
        oof_all = dict(oof_models)
        test_all = dict(test_map)
        if v.stageA_weight > 0 and oof_stageA is not None:
            oof_all['oof_two_stage_A'] = oof_stageA
            test_all['oof_two_stage_A'] = test_stageA
        # Normalize weights
        s = sum(weights.values())
        weights = {k: float(w) / s for k, w in weights.items()}
        # Test uses same keys as OOF (test_all maps those keys to test arrays)
        raw_test = _blend(test_all, weights)

        # Calibrate and gamma
        # Fit calibration again on OOF for consistency
        if v.calibration == 'isotonic':
            oof_c, te_c = _segment_isotonic(_blend(oof_all, weights), y, raw_test)
        else:
            oof_c, te_c = _beta_calibration(_blend(oof_all, weights), y, raw_test)
        pred = np.clip(np.power(te_c, v.gamma), EPS, 1.0 - EPS)

        df = pd.DataFrame({'cust_id': cust_ids, 'churn': pred})
        out_name = f"portfolio/submission_{i:02d}.csv"
        df.to_csv(out_name, index=False)

        rows.append({
            'id': i,
            'file': out_name,
            'oof_score': v.score,
            'weights': v.weights,
            'calibration': v.calibration,
            'gamma': v.gamma,
            'stageA_weight': v.stageA_weight,
            'agg_emphasis': v.month_weights
        })
        print(f"Saved {out_name} (OOF composite={v.score:.6f})")

    with open('portfolio/README_portfolio.md', 'w', encoding='utf-8') as f:
        f.write('# Portfolio Submissions\n\n')
        f.write('This folder contains diverse submissions generated without retraining.\\n')
        f.write('Each variant varies blend weights, calibration, temperature (gamma), and Stage-A emphasis.\\n\n')
        f.write('| id | file | oof_composite | calibration | gamma | stageA | weights |\n')
        f.write('|---:|:-----|--------------:|:-----------|------:|------:|:--------|\n')
        for r in rows:
            f.write(f"| {r['id']} | {os.path.basename(r['file'])} | {r['oof_score']:.6f} | {r['calibration']} | {r['gamma']:.2f} | {r['stageA_weight']:.2f} | {json.dumps(r['weights'])} |\n")

    print(f"\nDone. Generated {len(rows)} submissions in ./portfolio")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
