"""
Build portfolio/portfolio_meta.json with per-month metrics for each submission variant.

Reads:
- outputs/predictions/predictions_bundle.pkl (OOF/test streams + y_train, ref_dates)
- portfolio/README_portfolio.md (table with file, calibration, gamma, weights)

Writes:
- portfolio/portfolio_meta.json: { filename: { 'oof_composite': float,
                                               'recall@10': { '2018-11': float, '2018-12': float },
                                               'last_n': 6,
                                               'calibration': str,
                                               'gamma': float } }
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd

# Prefer canonical metric utilities if available
try:
    from src.models.modeling_pipeline import oof_composite_monthwise  # type: ignore
except Exception:  # fallback
    from sklearn.metrics import roc_auc_score

    def oof_composite_monthwise(y_true, y_score, ref_dates=None, last_n_months: int = 6):
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        auc = roc_auc_score(y_true, y_score)
        k = max(1, int(round(0.10 * len(y_true))))
        idx = np.argsort(-y_score)[:k]
        rec = float(y_true[idx].sum()) / max(1, int(y_true.sum())) if int(y_true.sum()) > 0 else 0.0
        lift = (float(y_true[idx].sum()) / k) / max(1e-12, (float(y_true.sum()) / len(y_true)))
        comp = (auc - 0.5) * 2.0 + 0.5 * rec + 0.5 * (lift / 2.0)
        return float(comp)


def _parse_readme_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith('|'):
                continue
            if set(line.replace('|', '').strip()) <= set('-: '):
                continue
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) < 7:
                continue
            rid, fname, score, calibration, gamma, stageA, weights_json = parts[:7]
            try:
                rows.append({
                    'id': int(rid),
                    'file': fname,
                    'oof_composite': float(score),
                    'calibration': calibration,
                    'gamma': float(gamma),
                    'stageA_weight': float(stageA),
                    'weights': weights_json,
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


def _reconstruct_oof(y: np.ndarray, ref_dates: pd.Series, bundle: Dict[str, Any], row: pd.Series) -> np.ndarray:
    import json as _json
    # Build OOF streams dict from weights keys
    weights = _json.loads(row['weights'])
    oof_map: Dict[str, np.ndarray] = {}
    for k in weights.keys():
        if k not in bundle:
            # keys are like 'oof_lgb'; ensure present
            if k.startswith('oof_') and k in bundle:
                pass
        oof_map[k] = np.asarray(bundle[k], dtype=float)
    # Optional Stage-A
    if 'oof_two_stage_A' in weights and 'oof_two_stage_A' in bundle:
        oof_map['oof_two_stage_A'] = np.asarray(bundle['oof_two_stage_A'], dtype=float)
    # Blend
    out = np.zeros_like(y, dtype=float)
    s = sum(float(v) for v in weights.values())
    if s <= 0:
        s = 1.0
    for name, w in weights.items():
        out += (float(w) / s) * oof_map[name]
    # Calibration
    cal = str(row.get('calibration', 'isotonic'))
    g = float(row.get('gamma', 1.0))
    if cal == 'beta':
        p = np.clip(out, 1e-6, 1 - 1e-6)
        X = np.column_stack([np.log(p), np.log(1.0 - p)])
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=5000)
        lr.fit(X, y)
        out = lr.predict_proba(X)[:, 1]
    else:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(out, y)
        out = iso.transform(out)
    out = np.clip(out, 1e-6, 1 - 1e-6)
    if g != 1.0:
        out = np.clip(out ** g, 1e-6, 1 - 1e-6)
    return out


def _recall_at_10_by_month(y: np.ndarray, p: np.ndarray, ref_dates: pd.Series) -> Dict[str, float]:
    ref = pd.to_datetime(ref_dates).dt.to_period('M').astype(str)
    months = sorted(ref.unique())
    out: Dict[str, float] = {}
    for m in months:
        mask = (ref.values == m)
        if not np.any(mask):
            continue
        y_m = y[mask]
        p_m = p[mask]
        k = max(1, int(round(0.10 * len(p_m))))
        idx = np.argsort(-p_m)[:k]
        pos = int(y_m.sum())
        rec = float(y_m[idx].sum()) / max(1, pos) if pos > 0 else 0.0
        out[m] = float(rec)
    return out


def main() -> int:
    bundle_path = os.path.join('outputs', 'predictions', 'predictions_bundle.pkl')
    readme_path = os.path.join('portfolio', 'README_portfolio.md')
    out_path = os.path.join('portfolio', 'portfolio_meta.json')

    if not os.path.exists(bundle_path):
        print(f"Missing {bundle_path}; run main pipeline first.")
        return 1
    if not os.path.exists(readme_path):
        print(f"Missing {readme_path}; generate portfolio first.")
        return 1

    with open(bundle_path, 'rb') as f:
        B = pickle.load(f)
    y = np.asarray(B['y_train'], dtype=int)
    ref_dates = pd.Series(pd.to_datetime(B['ref_dates']))

    df = _parse_readme_table(readme_path)
    if df.empty:
        print("README table appears empty; nothing to do.")
        return 1

    meta: Dict[str, Any] = {}
    for _, row in df.iterrows():
        try:
            p_oof = _reconstruct_oof(y, ref_dates, B, row)
            comp = float(oof_composite_monthwise(y, p_oof, ref_dates=ref_dates, last_n_months=6))
            recm = _recall_at_10_by_month(y, p_oof, ref_dates)
            meta[row['file'].split('/')[-1]] = {
                'oof_composite': comp,
                'recall@10': recm,
                'last_n': 6,
                'calibration': row.get('calibration'),
                'gamma': float(row.get('gamma', 1.0)),
            }
        except Exception as e:
            # Skip problematic row but continue
            continue

    os.makedirs('portfolio', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_path} with {len(meta)} entries.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
