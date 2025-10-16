"""
Run main.py with predefined presets and save submissions with informative suffixes.

Usage:
  python scripts/run_variants.py --preset bestbet

Presets:
  - bestbet:   --adv-filter --adv-drop-k 60 --last-n 6 --calib isotonic --gamma-grid 0.90,0.95,1.00 --cat-seeds 7 --use-optimized-params
  - longwindow: same as bestbet but --last-n 8
  - sigmoidcal: same as bestbet but --calib sigmoid --gamma-grid 0.95,1.00,1.05
  - nocall:     same as bestbet but --calib none --gamma-grid 0.85,0.90,0.95
  - catheavy:   same as bestbet but --cat-seeds 12

The script prints the composed command, runs it, and then renames
data/submissions/submission.csv to an informative name with the preset suffix.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime


def build_command(preset: str) -> list[str]:
    base = [
        sys.executable,
        os.path.join('.', 'main.py'),
        '--models', 'cat', 'lgb', 'xgb', 'two',
        '--with-stacker',
        '--adv-filter', '--adv-drop-k', '60',
        '--last-n', '6',
        '--calib', 'isotonic', '--gamma-grid', '0.90,0.95,1.00',
        '--cat-seeds', '7',
        '--use-optimized-params',
    ]

    if preset == 'bestbet':
        return base
    elif preset == 'longwindow':
        cmd = base.copy()
        # last-n 8
        i = cmd.index('--last-n')
        cmd[i + 1] = '8'
        return cmd
    elif preset == 'sigmoidcal':
        cmd = base.copy()
        # calib sigmoid, gamma-grid 0.95,1.00,1.05
        i = cmd.index('--calib'); cmd[i + 1] = 'sigmoid'
        j = cmd.index('--gamma-grid'); cmd[j + 1] = '0.95,1.00,1.05'
        return cmd
    elif preset == 'nocall':
        cmd = base.copy()
        # calib none, gamma-grid 0.85,0.90,0.95
        i = cmd.index('--calib'); cmd[i + 1] = 'none'
        j = cmd.index('--gamma-grid'); cmd[j + 1] = '0.85,0.90,0.95'
        return cmd
    elif preset == 'catheavy':
        cmd = base.copy()
        # cat-seeds 12
        i = cmd.index('--cat-seeds'); cmd[i + 1] = '12'
        return cmd
    else:
        raise SystemExit(f"Unknown preset: {preset}")


def suffix_for(preset: str) -> str:
    return {
        'bestbet': '_bestbet',
        'longwindow': '_longwindow',
        'sigmoidcal': '_sigmoidcal',
        'nocall': '_nocall',
        'catheavy': '_catheavy',
    }.get(preset, f'_{preset}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run main.py with preset flags and rename submission')
    parser.add_argument('--preset', required=True, choices=['bestbet', 'longwindow', 'sigmoidcal', 'nocall', 'catheavy'])
    args = parser.parse_args(argv)

    cmd = build_command(args.preset)
    print('Executing command:')
    print(' '.join(cmd))

    # Ensure PYTHONPATH includes project root when launch from scripts/
    env = os.environ.copy()
    if 'PYTHONPATH' not in env:
        env['PYTHONPATH'] = os.getcwd()

    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"main.py exited with code {proc.returncode}")
        return proc.returncode

    # Rename submission to include preset and timestamp to avoid overwrite
    sub_dir = os.path.join('data', 'submissions')
    src = os.path.join(sub_dir, 'submission.csv')
    if not os.path.exists(src):
        print(f"Submission file not found at {src}")
        return 1
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    dst = os.path.join(sub_dir, f"submission{suffix_for(args.preset)}_{ts}.csv")
    try:
        shutil.copyfile(src, dst)
        print(f"Saved {dst}")
    except Exception as e:
        print(f"Failed to write {dst}: {e}")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
