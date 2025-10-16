"""
Update README_portfolio.md with a new Leaderboard row when you paste an LB score.

Usage examples:
  python portfolio_tools/update_readme.py "submission_07.csv: 1.246, rank +12"
  python portfolio_tools/update_readme.py  # then paste the line when prompted

It will:
- Parse: filename: score, rank +/-N
- Look up calibration, gamma, and blend weights from README_portfolio.md
- Optionally load per-month OOF metrics from portfolio/portfolio_meta.json
- Append a Markdown table row with columns:
  file | calib | gamma | blend | OOF_2018-12_R10 | OOF_2018-11_L10 | LB | Δrank | verdict(✓/✗)
- Maintain a "Best so far" marker (updates if improved by >= 0.002)
"""

import os
import re
import json
import sys
from typing import Dict, List, Optional, Tuple

README_PATH = os.path.join('portfolio', 'README_portfolio.md')
META_PATH = os.path.join('portfolio', 'portfolio_meta.json')  # optional


def parse_input_line(s: str) -> Tuple[str, float, str]:
    s = s.strip()
    # Example formats: "submission_07.csv: 1.246, rank +12" or "submission_07.csv 1.246 +12"
    m = re.match(r"^\s*([^:]+?)\s*:\s*([0-9]*\.?[0-9]+)\s*,\s*rank\s*([+\-−]\s*\d+).*$", s, re.IGNORECASE)
    if not m:
        # try simpler
        m2 = re.match(r"^\s*([^\s]+)\s+([0-9]*\.?[0-9]+)\s+([+\-−]?\d+)\s*$", s)
        if not m2:
            raise ValueError("Could not parse input. Expected 'file: score, rank +/-N'")
        fname, score, dr = m2.group(1), float(m2.group(2)), m2.group(3)
        return fname.strip(), float(score), dr.replace('−', '-')
    fname = m.group(1).strip()
    score = float(m.group(2))
    delta_rank = m.group(3).replace('−', '-')
    return fname, score, delta_rank


def parse_readme_variants(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('|'):
                continue
            # header separator row has only - :
            if set(line.replace('|', '').strip()) <= set('-: '):
                continue
            parts = [p.strip() for p in line.split('|')[1:-1]]
            # Expect the first large table format: id | file | oof_composite | calibration | gamma | stageA | weights
            if len(parts) >= 7 and parts[0].isdigit():
                try:
                    rows.append({
                        'id': int(parts[0]),
                        'file': parts[1],
                        'oof_composite': float(parts[2]),
                        'calibration': parts[3],
                        'gamma': float(parts[4]),
                        'stageA_weight': float(parts[5]),
                        'weights': parts[6],
                    })
                except Exception:
                    continue
    return rows


def ensure_lb_section(content: str) -> str:
    if '## Leaderboard Log' in content:
        return content
    section = []
    section.append('\n## Leaderboard Log\n')
    section.append('Best so far: N/A\n\n')
    section.append('| file | calib | gamma | blend | OOF_2018-12_R10 | OOF_2018-11_L10 | LB | Δrank | verdict |\n')
    section.append('|:-----|:------|------:|:------|------------------:|------------------:|----:|:------:|:--------:|\n')
    return content.rstrip() + '\n\n' + ''.join(section)


def load_meta(path: str) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def find_best_so_far(lines: List[str]) -> Optional[Tuple[float, str]]:
    # Scan after '## Leaderboard Log' for 'Best so far: X (file)'
    best_val = None
    best_file = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Best so far:'):
            # extract number
            m = re.search(r"([0-9]*\.?[0-9]+)", line)
            if m:
                try:
                    best_val = float(m.group(1))
                except Exception:
                    best_val = None
            m2 = re.search(r"\(([^\)]+)\)", line)
            if m2:
                best_file = m2.group(1).strip()
            break
    if best_val is not None:
        return best_val, best_file or ''
    return None


def update_best_marker(lines: List[str], new_best: Tuple[float, str]) -> List[str]:
    out = []
    updated = False
    for i, line in enumerate(lines):
        if not updated and line.strip().startswith('Best so far:'):
            out.append(f"Best so far: {new_best[0]:.3f} ({new_best[1]})\n")
            updated = True
        else:
            out.append(line)
    if not updated:
        out.append(f"Best so far: {new_best[0]:.3f} ({new_best[1]})\n")
    return out


def main() -> int:
    if len(sys.argv) >= 2:
        input_line = ' '.join(sys.argv[1:])
    else:
        print("Paste LB line (e.g., 'submission_07.csv: 1.246, rank +12'):")
        input_line = sys.stdin.readline()
    try:
        fname, lb_score, delta_rank = parse_input_line(input_line)
    except Exception as e:
        print(f"Parse error: {e}")
        return 1

    # Read README and ensure section
    content = ''
    if os.path.exists(README_PATH):
        with open(README_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
    content = ensure_lb_section(content)

    # Gather variant metadata
    variants = parse_readme_variants(README_PATH)
    meta = load_meta(META_PATH)
    calib = ''
    gamma = ''
    blend = ''
    for v in variants:
        if v.get('file') == fname:
            calib = v.get('calibration', '')
            gamma = v.get('gamma', '')
            blend = v.get('weights', '')
            break

    # Monthly metrics (optional)
    m = meta.get(fname, {}) if isinstance(meta, dict) else {}
    r12 = m.get('recall@10', {}).get('2018-12', '')
    l11 = m.get('lift@10', {}).get('2018-11', '')

    # Locate Leaderboard section lines
    lines = content.splitlines(keepends=True)
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == '## Leaderboard Log')
    except StopIteration:
        # Should not happen, ensure_lb_section adds it
        start = len(lines)
        lines.append('\n## Leaderboard Log\n')
        lines.append('Best so far: N/A\n\n')
        lines.append('| file | calib | gamma | blend | OOF_2018-12_R10 | OOF_2018-11_L10 | LB | Δrank | verdict |\n')
        lines.append('|:-----|:------|------:|:------|------------------:|------------------:|----:|:------:|:--------:|\n')

    # Find best so far before appending
    # Search the line after header for Best so far
    tail = lines[start+1:]
    prev_best = find_best_so_far(tail)
    improved = False
    if prev_best is None:
        improved = True
        prev_best_val = float('-inf')
    else:
        prev_best_val = prev_best[0]
        improved = (lb_score >= prev_best_val + 0.002)

    verdict = '✓' if improved else '✗'

    # Append row at end of file
    new_row = f"| {fname} | {calib} | {gamma} | {blend} | {r12} | {l11} | {lb_score:.3f} | {delta_rank} | {verdict} |\n"
    # Insert new_row just before EOF (we simply append)
    lines.append(new_row)

    # Update Best so far marker if improved
    if improved:
        # Update within the Leaderboard section
        head = lines[:start+1]
        section_lines = lines[start+1:]
        section_lines = update_best_marker(section_lines, (lb_score, fname))
        lines = head + section_lines

    # Write back
    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))

    print(f"Appended LB row for {fname}. {'New BEST' if improved else 'No improvement'}.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
