"""
Setup project folder structure for a data science workflow without breaking imports.

This script:
- Creates a conventional directory layout (data/, outputs/, configs/, notebooks/, scripts/)
- Adds __init__.py files to src/* subpackages
- Optionally moves common artifacts/outputs into the right places (dry-run by default)

Usage (Windows PowerShell):
    # Dry-run (print what would happen)
    python ./setup_structure.py

    # Apply changes (perform moves safely)
    python ./setup_structure.py --apply

Safe-moving rules:
- Never overwrite: if destination exists, a numeric suffix is added
- Only moves non-code artifacts by default (csv/pkl/json and known folders)
- Source code files remain where they are to avoid breaking imports
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple


ROOT = os.path.dirname(os.path.abspath(__file__))


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        full = os.path.join(ROOT, p)
        os.makedirs(full, exist_ok=True)


def ensure_init_py(package_dirs: List[str]) -> None:
    for d in package_dirs:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        init_path = os.path.join(full, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w", encoding="utf-8") as f:
                f.write("# package\n")


def unique_dest_path(dest_path: str) -> str:
    if not os.path.exists(dest_path):
        return dest_path
    base, ext = os.path.splitext(dest_path)
    i = 1
    while True:
        candidate = f"{base}.{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def safe_move(src: str, dest: str, apply: bool) -> Tuple[str, str]:
    dest_unique = unique_dest_path(dest)
    if apply:
        os.makedirs(os.path.dirname(dest_unique), exist_ok=True)
        shutil.move(src, dest_unique)
    return src, dest_unique


@dataclass
class MovePlan:
    src: str
    dest: str


def discover_artifact_moves() -> List[MovePlan]:
    """Plan moves for common artifacts/outputs only. Source code stays put.

    Returns a list of (src, dest) relative paths from ROOT.
    """
    plans: List[MovePlan] = []

    # Files we commonly want to move
    candidates = [
        ("submission.csv", "data/submissions/submission.csv"),
        ("predictions_bundle.pkl", "outputs/predictions/predictions_bundle.pkl"),
        ("feature_importance.csv", "outputs/reports/feature_importance.csv"),
        ("ref_dates.pkl", "data/processed/ref_dates.pkl"),
        ("y_train.pkl", "data/processed/y_train.pkl"),
        ("tuned_params.json", "configs/tuned_params.json"),
    ]

    for src_rel, dest_rel in candidates:
        src_abs = os.path.join(ROOT, src_rel)
        if os.path.exists(src_abs):
            plans.append(MovePlan(src_rel, dest_rel))

    # Folders we commonly want to move
    folder_candidates = [
        ("portfolio", "data/portfolio"),
        ("catboost_info", "outputs/catboost_info"),
        ("models", "models"),  # keep as-is but ensure it exists
    ]

    for src_rel, dest_rel in folder_candidates:
        src_abs = os.path.join(ROOT, src_rel)
        if os.path.isdir(src_abs):
            # Skip moving src==dest (models)
            if src_rel == dest_rel:
                continue
            plans.append(MovePlan(src_rel, dest_rel))

    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup/organize project folders safely")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply planned moves (default: dry-run)",
    )
    args = parser.parse_args()

    # 1) Ensure directory layout
    ensure_dirs([
        "data/raw",
        "data/processed",
        "data/submissions",
        "data/portfolio",
        "outputs/predictions",
        "outputs/reports",
        "configs",
        "notebooks",
        "scripts",
        "models",
    ])

    # 2) Ensure src subpackages have __init__.py
    ensure_init_py([
        "src",
        "src/features",
        "src/models",
        "src/ensemble",
        "src/utils",
    ])

    # 3) Plan artifact moves (non-code only)
    plans = discover_artifact_moves()

    if not plans:
        print("No artifact moves planned. Structure ensured.")
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] Planned artifact moves:")
    for p in plans:
        print(f" - {p.src} -> {p.dest}")

    if args.apply:
        print("\nApplying moves...")
        for p in plans:
            src_abs = os.path.join(ROOT, p.src)
            dest_abs = os.path.join(ROOT, p.dest)
            s, d = safe_move(src_abs, dest_abs, apply=True)
            print(f" Moved: {os.path.relpath(s, ROOT)} -> {os.path.relpath(d, ROOT)}")
        print("\nDone.")
    else:
        print("\nDry-run complete. Re-run with --apply to perform moves.")


if __name__ == "__main__":
    main()
