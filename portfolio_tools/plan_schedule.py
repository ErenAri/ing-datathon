"""
Plan a Day 1 submission schedule using representatives.json and per-file metadata.

- Loads portfolio/representatives.json to get candidate representative filenames
- Loads portfolio/README_portfolio.md (optional) and per-file sidecar JSONs (optional)
- Loads portfolio/portfolio_meta.json (optional) with per-month OOF metrics
- Builds an aligned churn matrix for candidate files and computes Spearman correlations
- Clusters candidates with edges where corr >= --corr-threshold and saves clusters.json
- In --plan mode, greedily selects up to 5 files prioritizing:
    1) Lower correlation to the current set (and skips any with max corr >= threshold)
    2) Higher offline OOF composite (if found in a sidecar JSON next to each submission)
    3) Diversity in calibration family, gamma center, and last-n
- Prints an ordered list with rationale and writes schedule_day1.txt
"""

import os
import re
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


OUT_DIR = 'portfolio'
REPS_PATH = os.path.join(OUT_DIR, 'representatives.json')
README_PATH = os.path.join(OUT_DIR, 'README_portfolio.md')
META_JSON = os.path.join(OUT_DIR, 'portfolio_meta.json')  # optional richer metadata
SCHEDULE_OUT = os.path.join(OUT_DIR, 'schedule_day1.txt')
CLUSTERS_OUT = os.path.join(OUT_DIR, 'clusters.json')

SUBMIT_GLOB = os.environ.get('SUBMISSION_GLOB', os.path.join(OUT_DIR, 'submission_*.csv'))

MONTH_KEYS = ['2018-11', '2018-12']


def _parse_readme_table(path: str) -> pd.DataFrame:
    """Parse README_portfolio.md table into a DataFrame.

    Expected columns (by position): id | file | oof_composite | calibration | gamma | stageA_weight | weights
    If the file is missing or malformed, returns an empty DataFrame.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith('|'):
                continue
            # skip header separators
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
                # If parsing/conversion fails, skip row
                continue
    return pd.DataFrame(rows)


def _load_meta_json(path: str) -> Dict[str, Dict]:
    """Load optional metadata JSON mapping filename -> metrics dict (including per-month R@10)."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _infer_from_filename(fname: str) -> Tuple[Optional[str], Optional[float]]:
    """Infer calibration and gamma from a filename pattern like
    'submission_cal_ensemble_isotonic_g085.csv' or '..._beta_g09.csv'.

    Returns (calibration, gamma) or (None, None) if not parsable.
    """
    try:
        m = re.search(r"cal_ensemble_(isotonic|beta|none)_g(\d+)", fname)
        if not m:
            return None, None
        cal = m.group(1)
        gd = m.group(2)
        # Interpret digits as hundredths if 3 digits (085 -> 0.85), tenths if 2 digits (09 -> 0.9)
        if len(gd) == 3:
            gamma = int(gd) / 100.0
        elif len(gd) == 2:
            gamma = int(gd) / 10.0
        else:
            gamma = int(gd) / 100.0
        return cal, float(gamma)
    except Exception:
        return None, None


def _priority_score(meta: Dict) -> float:
    """Compute priority = R@10(2018-12) + 0.5 * R@10(2018-11). If missing, return -inf to deprioritize."""
    try:
        r12 = float(meta.get('recall@10', {}).get('2018-12', np.nan))
        r11 = float(meta.get('recall@10', {}).get('2018-11', np.nan))
        if np.isnan(r12) or np.isnan(r11):
            return float('-inf')
        return r12 + 0.5 * r11
    except Exception:
        return float('-inf')


def _pick_schedule(candidates: List[Dict], k: int = 5) -> List[Dict]:
    """Greedy pick with diversity: avoid same calibration and gamma back-to-back."""
    picks: List[Dict] = []
    last_cal: Optional[str] = None
    last_g: Optional[float] = None

    remaining = sorted(candidates, key=lambda d: d['priority'], reverse=True)
    while remaining and len(picks) < k:
        # choose best that doesn't repeat both cal and gamma back-to-back
        chosen_idx = None
        for i, d in enumerate(remaining):
            if last_cal is not None and last_g is not None:
                try:
                    same_combo = (d.get('calibration') == last_cal and abs(float(d.get('gamma', 0.0)) - float(last_g)) < 1e-9)
                except Exception:
                    same_combo = False
                if same_combo:
                    continue
            chosen_idx = i
            break
        if chosen_idx is None:
            chosen_idx = 0
        chosen = remaining.pop(chosen_idx)
        picks.append(chosen)
        last_cal = chosen.get('calibration')
        try:
            last_g = float(chosen.get('gamma', 0.0))
        except Exception:
            last_g = None
    return picks


def _read_submissions(paths: List[str]) -> Tuple[Optional[pd.Index], Optional[pd.DataFrame]]:
    """Read submissions, align by cust_id, and return churn matrix (n_customers x n_files)."""
    dfs = []
    names: List[str] = []
    index: Optional[pd.Index] = None
    for p in paths:
        try:
            df = pd.read_csv(p, dtype={'cust_id': str})
            if 'cust_id' not in df.columns or 'churn' not in df.columns:
                continue
            df = df[['cust_id', 'churn']].copy()
            df['cust_id'] = df['cust_id'].astype(str)
            df = df.sort_values('cust_id').reset_index(drop=True)
            if index is None:
                index = pd.Index(df['cust_id'].astype(str).values, name='cust_id')
            else:
                if not index.equals(pd.Index(df['cust_id'].astype(str).values)):
                    idx_df = pd.DataFrame({'cust_id': index.values})
                    df = idx_df.merge(df, on='cust_id', how='left')
            dfs.append(df['churn'].astype(float))
            names.append(os.path.basename(p))
        except Exception:
            continue
    if not dfs or index is None:
        # Graceful: signal no matrix available
        return None, None
    mat = pd.concat(dfs, axis=1)
    mat.columns = names
    assert isinstance(index, pd.Index)
    return index, mat


def _spearman_corr_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr
    arr = mat.values.astype(float)
    arr[~np.isfinite(arr)] = np.nan
    corr, _ = spearmanr(arr, axis=0, nan_policy='omit')
    n = mat.shape[1]
    corr = np.asarray(corr)
    if corr.shape != (n, n):
        corr = corr[:n, :n]
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=mat.columns, columns=mat.columns).astype(float)


def _clusters_from_threshold(corr: pd.DataFrame, thr: float) -> List[List[str]]:
    names = list(corr.index)
    n = len(names)
    corr_v = corr.values.astype(float)
    adj: Dict[str, List[str]] = {name: [] for name in names}
    for i in range(n):
        for j in range(i + 1, n):
            if corr_v[i, j] >= float(thr):
                a, b = names[i], names[j]
                adj[a].append(b)
                adj[b].append(a)
    seen = set()
    comps: List[List[str]] = []
    for v in names:
        if v in seen:
            continue
        stack = [v]
        comp: List[str] = []
        seen.add(v)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(sorted(comp))
    comps.sort(key=lambda x: (-len(x), x))
    return comps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plan Day 1 schedule from representatives with correlation gating")
    parser.add_argument('--corr-threshold', type=float, default=0.98, help='Max allowed Spearman corr to current set; skip if >= threshold')
    parser.add_argument('--plan', action='store_true', help='Produce schedule_day1.txt based on correlation gating and priorities')
    parser.add_argument('--glob', type=str, default=SUBMIT_GLOB, help='Glob for submission CSVs (for correlation matrix)')
    args = parser.parse_args(argv)

    if not os.path.exists(REPS_PATH):
        print(f"Missing {REPS_PATH}. Run clustering first.")
        return 1

    with open(REPS_PATH, 'r', encoding='utf-8') as f:
        reps_map = json.load(f)

    reps = sorted(reps_map.keys())
    if not reps:
        print("No representatives found.")
        return 1

    readme_df = _parse_readme_table(README_PATH)
    meta = _load_meta_json(META_JSON)

    # Map representative -> full path for correlation analysis
    all_paths = sorted(glob.glob(args.glob))
    rep_set = set(reps)
    rep_paths = {os.path.basename(p): p for p in all_paths if os.path.basename(p) in rep_set}
    # Fallback searches for any missing reps
    if len(rep_paths) < len(reps):
        for r in reps:
            if r in rep_paths:
                continue
            # Try typical locations
            for cand in [os.path.join(OUT_DIR, r), os.path.join('data', 'submissions', r), r]:
                if os.path.exists(cand):
                    rep_paths[r] = cand
                    break
    if len(rep_paths) < len(reps):
        missing = [r for r in reps if r not in rep_paths]
        if missing:
            print(f"Warning: {len(missing)} representative files not found for correlation: {missing[:5]}{'...' if len(missing)>5 else ''}")

    # Build aligned matrix and correlation for available representatives
    present_files = sorted(rep_paths.keys())
    index, mat = _read_submissions([rep_paths[n] for n in present_files])
    corr_unavailable = (mat is None or index is None)
    if not corr_unavailable:
        assert mat is not None
        corr = _spearman_corr_matrix(mat)
    else:
        corr = pd.DataFrame()

    # Derive clusters on representatives with the provided threshold
    thr = float(args.corr_threshold)
    if not corr_unavailable and not corr.empty:
        clusters = _clusters_from_threshold(corr.loc[present_files, present_files], thr)
    else:
        # No correlation available: treat each representative as its own singleton cluster
        clusters = [[r] for r in reps]
        print("Warning: correlation matrix unavailable; saving singleton clusters.")
    # Save clusters.json
    os.makedirs(OUT_DIR, exist_ok=True)
    clusters_json = [{"members": comp, "size": len(comp)} for comp in clusters]
    with open(CLUSTERS_OUT, 'w', encoding='utf-8') as f:
        json.dump(clusters_json, f, indent=2)

    candidates: List[Dict] = []
    for rep in reps:
        calib: Optional[str] = None
        gamma: Optional[float] = None
        if not readme_df.empty and 'file' in readme_df.columns:
            try:
                row = readme_df[readme_df['file'] == rep]
                if not row.empty:
                    calib = row.iloc[0].get('calibration')
                    gamma = row.iloc[0].get('gamma')
            except Exception:
                pass
        if calib is None or gamma is None:
            inf_cal, inf_gamma = _infer_from_filename(rep)
            calib = calib or inf_cal
            gamma = gamma if gamma is not None else inf_gamma
        # Sidecar JSON next to each submission for offline OOF and params
        sidecar = {}
        try:
            stem = os.path.splitext(rep_paths.get(rep, os.path.join(OUT_DIR, rep)))[0]
            sc_path = stem + '.json'
            if os.path.exists(sc_path):
                with open(sc_path, 'r', encoding='utf-8') as sf:
                    sidecar = json.load(sf)
            # Pull overrides from sidecar
            if sidecar:
                if calib is None:
                    calib = sidecar.get('calibration', calib)
                if gamma is None:
                    g = sidecar.get('gamma', gamma)
                    try:
                        gamma = float(g) if g is not None else gamma
                    except Exception:
                        pass
        except Exception:
            sidecar = {}

        m = meta.get(rep, {})
        prio = _priority_score(m)
        candidates.append({
            'file': rep,
            'calibration': calib,
            'gamma': gamma,
            'priority': prio,
            'meta': m,
            'sidecar': sidecar,
        })

    # If all priorities missing, fallback to OOF composite order from README
    if candidates and all(np.isneginf(c['priority']) for c in candidates) and not readme_df.empty:
        try:
            if 'oof_composite' in readme_df.columns and 'file' in readme_df.columns:
                sorted_files = readme_df.sort_values('oof_composite', ascending=False)['file'].tolist()
                rank = {fname: i for i, fname in enumerate(sorted_files)}
                for c in candidates:
                    c['priority'] = float(len(rank) - rank.get(c['file'], len(rank)))
        except Exception:
            pass

    # Enrich candidates with OOF composite and last_n if available
    for c in candidates:
        oof_comp = None
        last_n = None
        try:
            if isinstance(c.get('sidecar'), dict):
                oof_comp = c['sidecar'].get('oof_composite', None)
                last_n = c['sidecar'].get('last_n', None)
        except Exception:
            pass
        c['oof_composite'] = float(oof_comp) if oof_comp is not None else None
        c['last_n'] = int(last_n) if isinstance(last_n, (int, float)) else None

    # Build quick lookup for corr rows
    corr_map = corr.to_dict(orient='index') if not corr.empty else {}

    # Greedy selection with correlation gating and diversity
    k = min(5, len(candidates)) if candidates else 0
    picks: List[Dict] = []
    chosen_names: List[str] = []
    while candidates and len(picks) < k:
        eligible: List[Dict[str, Any]] = []
        # Track selected meta for diversity
        chosen_cals = {p.get('calibration') for p in picks if p.get('calibration') is not None}
        chosen_gammas: List[float] = []
        for p in picks:
            g_val = p.get('gamma')
            if g_val is None:
                continue
            try:
                chosen_gammas.append(float(g_val))
            except Exception:
                continue
        chosen_lastn: set[int] = set()
        for p in picks:
            ln_val = p.get('last_n')
            if ln_val is None:
                continue
            try:
                chosen_lastn.add(int(ln_val))
            except Exception:
                continue

        for c in candidates:
            name = c['file']
            # Correlation to already selected
            if not chosen_names:
                max_corr = 0.0  # no constraint for first pick
            else:
                if name in corr_map:
                    try:
                        vals = [float(corr_map[name].get(ch, 0.0)) for ch in chosen_names if ch in corr_map[name]]
                        max_corr = max(vals) if vals else 0.0
                    except Exception:
                        max_corr = 0.0
                else:
                    # No corr info for this file; treat as dissimilar for gating
                    max_corr = 0.0
            # Gating: skip if max corr >= threshold
            if chosen_names and (max_corr >= thr - 1e-12):
                continue

            # Diversity penalty: prefer new calibration, gamma far from chosen, and new last_n
            pen = 0
            cal = c.get('calibration')
            if (cal is not None) and (cal in chosen_cals):
                pen += 1
            cg = None
            g_cand = c.get('gamma')
            if g_cand is not None:
                try:
                    cg = float(g_cand)
                except Exception:
                    cg = None
            if cg is not None and chosen_gammas:
                mind = min(abs(cg - g) for g in chosen_gammas)
                if mind < 0.03:  # within 0.03 considered same center
                    pen += 1
            ln = c.get('last_n')
            if (ln is not None) and (ln in chosen_lastn):
                pen += 1

            # OOF composite (descending preferred); fallback to priority if missing
            oof = c.get('oof_composite')
            # Normalize OOF and priority to finite floats
            if oof is None or (isinstance(oof, float) and (np.isnan(oof) or not np.isfinite(oof))):
                prio_val = c.get('priority', float('-inf'))
                if isinstance(prio_val, float) and (np.isnan(prio_val) or np.isneginf(prio_val)):
                    prio_val = -1e9
                try:
                    oof_val = float(prio_val)
                except Exception:
                    oof_val = -1e9
            else:
                try:
                    oof_val = float(oof)
                    if not np.isfinite(oof_val):
                        oof_val = -1e9
                except Exception:
                    oof_val = -1e9

            # Record candidate with fields for sorting
            eligible.append({
                'max_corr': float(max_corr),
                'pen': int(pen),
                'neg_oof': -float(oof_val),
                'name': str(name),
                'cand': c,
            })

        if not eligible:
            break
        eligible.sort(key=lambda e: (e['max_corr'], e['pen'], e['neg_oof'], e['name']))
        chosen = eligible[0]['cand']
        picks.append(chosen)
        chosen_names.append(chosen['file'])
        # Remove chosen from candidates list
        candidates = [c for c in candidates if c['file'] != chosen['file']]

    # Print schedule with rationale
    print(f"Day 1 Submission Schedule ({len(picks)} files):")
    lines: List[str] = []
    # Validate pairwise corr < threshold
    ok_names: List[str] = []
    for i, p in enumerate(picks, 1):
        name = p['file']
        # Compute max corr to previously kept ok_names for display and validation
        if name in corr_map:
            try:
                prev_corrs = [float(corr_map[name].get(o, 0.0)) for o in ok_names]
                max_corr_prev = (max(prev_corrs) if prev_corrs else 0.0)
            except Exception:
                max_corr_prev = 0.0
        else:
            max_corr_prev = 0.0
        if ok_names and (max_corr_prev >= thr - 1e-12):
            # Skip violating pick from output to guarantee acceptance criteria
            continue
        ok_names.append(name)
        r12 = p['meta'].get('recall@10', {}).get('2018-12', None)
        r11 = p['meta'].get('recall@10', {}).get('2018-11', None)
        oofc = p.get('oof_composite')
        rationale = (
            f"max_corr_prev={max_corr_prev:.4f}; oof={oofc if oofc is not None else 'n/a'}; "
            f"priority={p['priority']:.4f}; R@10[12]={r12}; R@10[11]={r11}; cal={p['calibration']}; gamma={p['gamma']}; last_n={p.get('last_n')}"
        )
        line = f"{len(ok_names)}. {p['file']}  |  {rationale}"
        print(line)
        lines.append(line)

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Write schedule
    with open(SCHEDULE_OUT, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\nSaved {SCHEDULE_OUT}")

    print(f"Saved {CLUSTERS_OUT}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
