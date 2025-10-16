"""
Cluster submission files by Spearman correlation and select medoid representatives.

- Globs files: portfolio/submission_*.csv (adjust path if needed)
- Aligns by cust_id and uses the churn column only
- Computes pairwise Spearman correlation matrix
- Builds an undirected graph with edges when corr >= threshold (default 0.98)
- Finds connected components as clusters
- For each cluster, selects a medoid: file with lowest average (1 - corr) to others
- Saves clusters.json and representatives.json and prints a concise summary
"""

import os
import sys
import json
import glob
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

THRESHOLD = float(os.environ.get('CLUSTER_CORR_THRESHOLD', 0.98))
SUBMIT_GLOB = os.environ.get('SUBMISSION_GLOB', 'portfolio/submission_*.csv')
OUT_DIR = os.environ.get('PORTFOLIO_OUT', 'portfolio')


def _read_submissions(paths: List[str]) -> Tuple[pd.Index, pd.DataFrame]:
    """Read all submissions, align by cust_id, and return churn matrix (n_customers x n_files)."""
    dfs = []
    names = []
    index: pd.Index | None = None
    for p in paths:
        try:
            df = pd.read_csv(p, dtype={'cust_id': str})
            if 'cust_id' not in df.columns or 'churn' not in df.columns:
                print(f"Skipping {p}: missing required columns")
                continue
            df = df[['cust_id', 'churn']].copy()
            df['cust_id'] = df['cust_id'].astype(str)
            df = df.sort_values('cust_id').reset_index(drop=True)
            if index is None:
                index = pd.Index(df['cust_id'].astype(str).values, name='cust_id')
            else:
                # Ensure same set/order of customers
                if not index.equals(pd.Index(df['cust_id'].astype(str).values)):
                    # Align by merge to index
                    idx_df = pd.DataFrame({'cust_id': index.values})
                    df = idx_df.merge(df, on='cust_id', how='left')
            dfs.append(df['churn'].astype(float))
            names.append(os.path.basename(p))
        except Exception as e:
            print(f"Error reading {p}: {e}")
    if not dfs or index is None:
        raise SystemExit("No valid submission files found.")
    mat = pd.concat(dfs, axis=1)
    mat.columns = names
    assert isinstance(index, pd.Index)
    return index, mat


def _spearman_corr_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation matrix for columns of mat."""
    # scipy.spearmanr handles NaNs by pairwise deletion; but ensure finite
    arr = mat.values.astype(float)
    # Replace inf with nan
    arr[~np.isfinite(arr)] = np.nan
    corr, _ = spearmanr(arr, axis=0, nan_policy='omit')
    # spearmanr returns a 2D array; shape should be (n_files, n_files)
    n = mat.shape[1]
    corr = np.asarray(corr)
    if corr.shape != (n, n):
        corr = corr[:n, :n]
    # Diagonal to 1
    np.fill_diagonal(corr, 1.0)
    df = pd.DataFrame(corr, index=mat.columns, columns=mat.columns)
    # Ensure float dtype
    return df.astype(float)


def _clusters_from_threshold(corr: pd.DataFrame, thr: float) -> List[List[str]]:
    """Build graph edges where corr >= thr and return connected components."""
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
    # Find connected components via DFS
    seen = set()
    comps = []
    for v in names:
        if v in seen:
            continue
        stack = [v]
        comp = []
        seen.add(v)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(sorted(comp))
    # Sort clusters by size desc, then lexicographically
    comps.sort(key=lambda x: (-len(x), x))
    return comps


def _pick_medoid(cluster: List[str], corr: pd.DataFrame) -> str:
    if len(cluster) == 1:
        return cluster[0]
    sub = corr.loc[cluster, cluster].copy()
    # Distance is (1 - corr); average distance per file
    dist = 1.0 - sub
    avg_dist = dist.mean(axis=1)
    rep = avg_dist.idxmin()
    return str(rep)


def main(argv=None) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(SUBMIT_GLOB))
    if not paths:
        print(f"No files matched pattern: {SUBMIT_GLOB}")
        return 1

    print(f"Found {len(paths)} submission files. Reading and aligning by cust_id...")
    index, mat = _read_submissions(paths)
    print(f"Aligned matrix shape: {mat.shape} (customers x files: {mat.shape[0]} x {mat.shape[1]})")

    print("Computing Spearman correlation matrix...")
    corr = _spearman_corr_matrix(mat)

    print(f"Clustering with threshold corr >= {THRESHOLD:.3f}...")
    clusters = _clusters_from_threshold(corr, THRESHOLD)

    reps: Dict[str, str] = {}
    for comp in clusters:
        rep = _pick_medoid(comp, corr)
        reps[rep] = rep  # mark representative

    # Prepare JSON outputs
    clusters_json = [{"members": comp, "size": len(comp), "representative": _pick_medoid(comp, corr)} for comp in clusters]
    reps_json = {comp[0] if len(comp) == 1 else _pick_medoid(comp, corr): comp for comp in clusters}

    # Save
    clusters_path = os.path.join(OUT_DIR, 'clusters.json')
    reps_path = os.path.join(OUT_DIR, 'representatives.json')
    with open(clusters_path, 'w', encoding='utf-8') as f:
        json.dump(clusters_json, f, indent=2)
    with open(reps_path, 'w', encoding='utf-8') as f:
        json.dump(reps_json, f, indent=2)

    # CLI summary
    print("\nClusters summary:")
    for comp in clusters:
        rep = _pick_medoid(comp, corr)
        print(f"  - size={len(comp):2d} | rep={rep} | members={', '.join(comp)}")

    print(f"\nSaved {clusters_path} and {reps_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
