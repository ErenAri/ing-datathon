import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_sub(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    # ensure cust_id and churn present
    if 'cust_id' not in cols:
        # try to find id-like column
        for c in df.columns:
            if c.lower() in ('id', 'customer_id'):
                df = df.rename(columns={c: 'cust_id'})
                break
    if 'churn' not in cols:
        # assume second column is prediction
        pred_cols = [c for c in df.columns if c != 'cust_id']
        if pred_cols:
            df = df.rename(columns={pred_cols[0]: 'churn'})
    return df[['cust_id', 'churn']].copy()


def main():
    sub_dir = Path('data/submissions')
    candidates = [
        ('baseline', sub_dir / 'submission.csv'),
        ('stacking', sub_dir / 'submission_stacking.csv'),
        ('segmentblend', sub_dir / 'submission_segmentblend.csv'),
    ]

    subs = []
    for name, p in candidates:
        if p.exists():
            df = load_sub(p)
            df = df.rename(columns={'churn': f'churn_{name}'})
            subs.append(df)

    if not subs:
        print('No submissions found under data/submissions')
        return

    merged = subs[0]
    for df in subs[1:]:
        merged = merged.merge(df, on='cust_id', how='inner')

    pred_cols = [c for c in merged.columns if c.startswith('churn_')]
    stats = {}
    for c in pred_cols:
        s = merged[c]
        stats[c] = {
            'min': float(np.min(s)),
            'max': float(np.max(s)),
            'mean': float(np.mean(s)),
            'std': float(np.std(s)),
            'p1': float(np.percentile(s, 1)),
            'p50': float(np.percentile(s, 50)),
            'p99': float(np.percentile(s, 99)),
        }

    # correlations
    pearson = merged[pred_cols].corr(method='pearson').round(6).to_dict()
    spearman = merged[pred_cols].corr(method='spearman').round(6).to_dict()

    report = {
        'rows_compared': int(len(merged)),
        'columns': pred_cols,
        'stats': stats,
        'pearson_corr': pearson,
        'spearman_corr': spearman,
    }

    out_dir = Path('outputs/reports')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'submission_comparison.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('\nComparison report saved to', out_path)
    print('\nSummary:')
    for c, st in stats.items():
        print(f"  {c} -> mean={st['mean']:.4f}, std={st['std']:.4f}, min={st['min']:.4f}, max={st['max']:.4f}")
    print('\nPearson correlations:')
    for a in pred_cols:
        row = report['pearson_corr'][a]
        print(' ', a, ' '.join([f"{b}={row[b]:.3f}" for b in pred_cols]))


if __name__ == '__main__':
    main()
