"""
Portfolio Runner: one-stop CLI to (1) cluster reps, (2) propose today's 5 files, and (3) update README with LB results.

Usage:
  # 1) Cluster portfolio submissions and pick representatives (writes clusters.json, representatives.json)
  python portfolio_runner.py --cluster

  # 2) Print today's recommended 5 files (uses representatives + metadata)
  python portfolio_runner.py --plan

  # 3) Append a Leaderboard row after you paste an LB score line
  python portfolio_runner.py --update "submission_07.csv: 1.246, rank +12"

Environment overrides (optional):
  CLUSTER_CORR_THRESHOLD, SUBMISSION_GLOB, PORTFOLIO_OUT
"""

import argparse
import os
import sys
import json
import subprocess
from typing import List, Dict, Tuple
import glob
import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:
    _spearmanr = None

from src.utils.calibration import kfold_calibrate_with_gamma, CalibrationConfig

# Internal tools
from portfolio_tools.cluster_submissions import main as cluster_main
from portfolio_tools.plan_schedule import main as plan_main
from portfolio_tools.update_readme import main as update_main


def _run_cluster() -> int:
    return int(cluster_main([]) if callable(cluster_main) else cluster_main())


def _run_plan() -> int:
    return int(plan_main() if callable(plan_main) else plan_main)


def _run_update(arg_line: str | None) -> int:
    # Forward the argument line to update_readme's main via sys.argv emulation
    if arg_line is None or not arg_line.strip():
        print("Provide an input like: --update \"submission_07.csv: 1.246, rank +12\"")
        return 1
    # Temporarily patch sys.argv for the imported script
    argv_bak = sys.argv[:]
    sys.argv = ["update_readme.py", arg_line]
    try:
        return int(update_main())
    finally:
        sys.argv = argv_bak


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portfolio Runner")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cluster", action="store_true", help="Cluster submissions and pick representatives")
    g.add_argument("--plan", action="store_true", help="Print today's recommended 5 files")
    g.add_argument("--update", type=str, metavar='LINE', help="Append LB result line to README (e.g., 'submission_07.csv: 1.246, rank +12')")
    g.add_argument("--generate", action="store_true", help="Generate diversified submissions from predictions bundle")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    if ns.cluster:
        return _run_cluster()
    if ns.plan:
        return _run_plan()
    if ns.update is not None:
        return _run_update(ns.update)
    if ns.generate:
        return _run_generate()
    return 0


# =========================
# Generate diversified submissions
# =========================

def _load_bundle(path: str) -> Dict[str, np.ndarray]:
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def _rank01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    order = v.argsort().argsort().astype(float)
    n = max(1, len(v) - 1)
    return order / n


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if _spearmanr is not None:
        res = _spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        # SciPy can return (corr,p) tuple or an object with .correlation
        if isinstance(res, tuple):
            r_val = res[0]
        else:
            r_val = getattr(res, 'correlation', None)
            if r_val is None:
                # Some versions may return ndarray-like; coerce
                r_val = np.asarray(res, dtype=float)[0] if np.size(res) else 0.0
    return float(np.asarray(r_val, dtype=float))
    # fallback: Pearson over ranks
    ra = _rank01(a)
    rb = _rank01(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _existing_submissions(sub_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(sub_dir, '*.csv')))


def _load_cust_ids(sub_dir: str) -> np.ndarray:
    base_sub = os.path.join(sub_dir, 'submission.csv')
    if not os.path.exists(base_sub):
        raise FileNotFoundError("submission.csv not found in data/submissions. Run main pipeline first.")
    df = pd.read_csv(base_sub)
    if 'cust_id' not in df.columns:
        raise ValueError("submission.csv must contain 'cust_id' column")
    return np.asarray(df['cust_id'].values)


def _check_corr_and_save(name: str, probs: np.ndarray, cust_ids: np.ndarray, sub_dir: str, threshold: float = 0.98) -> Tuple[bool, str]:
    # Compare against existing submissions
    for fpath in _existing_submissions(sub_dir):
        try:
            ex = pd.read_csv(fpath)
            if 'cust_id' in ex.columns and 'churn' in ex.columns:
                merged = pd.merge(pd.DataFrame({'cust_id': cust_ids, 'p': probs}), ex[['cust_id', 'churn']], on='cust_id', how='inner')
                if len(merged) == 0:
                    continue
                rho = _spearman(np.asarray(merged['p'].values, dtype=float), np.asarray(merged['churn'].values, dtype=float))
                if rho >= threshold:
                    return False, f"Refusing to save {name}: Spearman={rho:.4f} with existing {os.path.basename(fpath)} ≥ {threshold}"
        except Exception:
            continue
    # Save
    out_path = os.path.join(sub_dir, name)
    pd.DataFrame({'cust_id': cust_ids, 'churn': probs}).to_csv(out_path, index=False)
    return True, out_path


def _run_generate() -> int:
    sub_dir = os.path.join('data', 'submissions')
    os.makedirs(sub_dir, exist_ok=True)

    bundle_path = os.path.join('outputs', 'predictions', 'predictions_bundle.pkl')
    if not os.path.exists(bundle_path):
        print(f"Bundle not found at {bundle_path}. Run src/main.py first.")
        return 1

    bundle = _load_bundle(bundle_path)
    cust_ids = _load_cust_ids(sub_dir)

    # Collect available base predictions
    test_heads = {}
    oof_heads = {}
    for base in ['lgb', 'xgb', 'cat', 'two_stage_B', 'meta']:
        tk = f'test_{base}' if base != 'two_stage_B' else 'test_two_stage_B'
        ok = f'oof_{base}' if base != 'two_stage_B' else 'oof_two_stage_B'
        if tk in bundle:
            test_heads[base] = np.asarray(bundle[tk], dtype=float)
        if ok in bundle:
            oof_heads[base] = np.asarray(bundle[ok], dtype=float)

    # Ensemble from bundle
    test_ensemble_raw = np.asarray(bundle.get('test_ensemble_raw', None), dtype=float) if 'test_ensemble_raw' in bundle else None
    oof_ensemble = np.asarray(bundle.get('oof_ensemble', None), dtype=float) if 'oof_ensemble' in bundle else None
    y_train = np.asarray(bundle.get('y_train', None), dtype=float) if 'y_train' in bundle else None
    ref_dates = pd.Series(bundle.get('ref_dates', None)) if 'ref_dates' in bundle else None

    generated: List[Tuple[str, np.ndarray]] = []

    # cat_only
    if 'cat' in test_heads:
        generated.append(('cat_only.csv', test_heads['cat']))

    # xgb_only, lgb_only
    if 'xgb' in test_heads:
        generated.append(('xgb_only.csv', test_heads['xgb']))
    if 'lgb' in test_heads:
        generated.append(('lgb_only.csv', test_heads['lgb']))

    # blend_monthwise via final_weights if available; else equal over available
    if 'final_weights' in bundle and isinstance(bundle['final_weights'], dict):
        weights = {k: float(v) for k, v in bundle['final_weights'].items()}
        # Filter to available heads
        weights = {k: v for k, v in weights.items() if k in test_heads}
        sw = sum(weights.values())
        if sw <= 0:
            weights = {k: 1.0 / len(test_heads) for k in test_heads}
        else:
            weights = {k: v / sw for k, v in weights.items()}
    else:
        weights = {k: 1.0 / len(test_heads) for k in test_heads}

    if weights:
        mix = np.zeros_like(next(iter(test_heads.values())))
        for k, w in weights.items():
            mix = mix + w * test_heads[k]
        generated.append(('blend_monthwise.csv', mix))

    # blend_uniform over available heads
    if test_heads:
        uni = np.mean(np.vstack([v for v in test_heads.values()]), axis=0)
        generated.append(('blend_uniform.csv', uni))

    # blend_rankavg over available heads
    if test_heads:
        ranks = np.vstack([_rank01(v) for v in test_heads.values()])
        generated.append(('blend_rankavg.csv', ranks.mean(axis=0)))

    # blend_isotonic_g090 & g105: fit isotonic on OOF ensemble
    if (oof_ensemble is not None) and (y_train is not None) and (test_ensemble_raw is not None) and (ref_dates is not None):
        try:
            cfg = CalibrationConfig()
            # We'll use method='isotonic' and then force gamma 0.90 / 1.05
            res_iso = kfold_calibrate_with_gamma(y_train, oof_ensemble, pd.Series(ref_dates).astype(str), method='isotonic', cfg=cfg)
            # Refit test fn is already returned; apply and then re-apply gamma tweak
            base_test_cal = res_iso['test_cal_fn'](test_ensemble_raw)
            for g, fname in [(0.90, 'blend_isotonic_g090.csv'), (1.05, 'blend_isotonic_g105.csv')]:
                p = np.clip(np.power(base_test_cal, g), 1e-6, 1.0 - 1e-6)
                generated.append((fname, p))
        except Exception:
            pass

    # blend_beta: best beta calibrator with its own gamma
    if (oof_ensemble is not None) and (y_train is not None) and (test_ensemble_raw is not None) and (ref_dates is not None):
        try:
            res_beta = kfold_calibrate_with_gamma(y_train, oof_ensemble, pd.Series(ref_dates).astype(str), method='beta', cfg=CalibrationConfig())
            p = res_beta['test_cal_fn'](test_ensemble_raw)
            generated.append(('blend_beta.csv', p))
        except Exception:
            pass

    # blend_drop_top_leak: heuristic drop top-3 most drifted heads by OOF vs Test distribution distance
    if len(test_heads) >= 2 and oof_heads:
        try:
            # compute simple Wasserstein-like proxy: L1 distance between binned histograms
            def drift_score(oof_v, test_v) -> float:
                oof_v = np.asarray(oof_v, dtype=float)
                test_v = np.asarray(test_v, dtype=float)
                bins = np.linspace(0, 1, 51)
                h1, _ = np.histogram(np.clip(oof_v, 0, 1), bins=bins, density=True)
                h2, _ = np.histogram(np.clip(test_v, 0, 1), bins=bins, density=True)
                return float(np.abs(h1 - h2).sum())

            scores = []
            for k in test_heads:
                if k in oof_heads:
                    scores.append((k, drift_score(oof_heads[k], test_heads[k])))
            scores.sort(key=lambda x: x[1], reverse=True)
            drop = [k for k, _ in scores[:3]]
            kept = [k for k in test_heads.keys() if k not in drop]
            if kept:
                mix = np.mean(np.vstack([test_heads[k] for k in kept]), axis=0)
                generated.append(('blend_drop_top_leak.csv', mix))
        except Exception:
            pass

    # Now enforce Spearman correlation threshold against existing submissions and among new ones
    saved = []
    for name, pred in generated:
        # Check against already-saved new ones too
        too_similar = False
        for _, prev in saved:
            rho = _spearman(pred, prev)
            if rho >= 0.98:
                print(f"Refusing to create {name}: Spearman={rho:.4f} vs another new file ≥ 0.98")
                too_similar = True
                break
        if too_similar:
            continue
        ok, msg = _check_corr_and_save(name, pred, cust_ids, sub_dir, threshold=0.98)
        if ok:
            print(f"Saved {msg}")
            saved.append((name, pred))
        else:
            print(msg)

    print(f"Generated {len(saved)} diversified submissions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
