r"""
Metric demo: Computes the Datathon composite metric using ONLY ing_hubs_datathon_metric.

Usage examples (PowerShell):

1) From predictions bundle (preferred)
    $Env:PYTHONPATH = "$PWD"; .\venv\Scripts\python.exe .\scripts\metric_demo.py `
        --bundle .\outputs\predictions\predictions_bundle.pkl --key oof_ensemble

2) From CSVs (columns y and p)
    $Env:PYTHONPATH = "$PWD"; .\venv\Scripts\python.exe .\scripts\metric_demo.py `
        --y-csv .\path\to\y.csv --p-csv .\path\to\p.csv --y-col y --p-col p
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any, Tuple

import numpy as np


def _import_metric():
    """Prefer root-level ing_hubs_datathon_metric.py; fallback to pipeline metric.

    Returns the callable ing_hubs_datathon_metric.
    """
    try:
        import ing_hubs_datathon_metric as root_metric
        if hasattr(root_metric, "ing_hubs_datathon_metric"):
            return root_metric.ing_hubs_datathon_metric
    except Exception:
        pass
    # Fallback to pipeline implementation (returns (score, details))
    from src.models.modeling_pipeline import ing_hubs_datathon_metric as pipe_metric
    return pipe_metric


def _load_from_bundle(path: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        b = pickle.load(f)
    y = np.asarray(b["y_train"], dtype=float)
    if key not in b:
        # Common aliases
        aliases = {
            "oof_two_stage": "oof_two_stage_B",
            "oof_meta": "oof_meta",
            "oof_lgb": "oof_lgb",
            "oof_xgb": "oof_xgb",
            "oof_cat": "oof_cat",
            "oof_ensemble": "oof_ensemble",
        }
        if key in aliases and aliases[key] in b:
            key = aliases[key]
        else:
            available = ", ".join(sorted(k for k in b.keys() if k.startswith("oof_")))
            raise KeyError(f"Key '{key}' not found in bundle. Available OOF keys: {available}")
    p = np.asarray(b[key], dtype=float)
    return y, p


def _load_from_csv(y_csv: str, p_csv: str, y_col: str, p_col: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    y = pd.read_csv(y_csv)[y_col].to_numpy(dtype=float)
    p = pd.read_csv(p_csv)[p_col].to_numpy(dtype=float)
    if len(y) != len(p):
        raise ValueError(f"Length mismatch: y={len(y)} vs p={len(p)}")
    return y, p


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute Datathon metric using ing_hubs_datathon_metric only")
    ap.add_argument("--bundle", default=os.path.join("outputs", "predictions", "predictions_bundle.pkl"), help="Path to predictions bundle .pkl")
    ap.add_argument("--key", default="oof_ensemble", help="OOF key in bundle (e.g., oof_ensemble, oof_lgb, oof_xgb, oof_cat, oof_two_stage_B, oof_meta)")
    ap.add_argument("--y-csv", default=None, help="Optional CSV file with y_true column")
    ap.add_argument("--p-csv", default=None, help="Optional CSV file with y_prob column")
    ap.add_argument("--y-col", default="y", help="Column name for y in y-csv")
    ap.add_argument("--p-col", default="p", help="Column name for p in p-csv")
    args = ap.parse_args()

    metric_fn = _import_metric()

    if args.y_csv and args.p_csv:
        y, p = _load_from_csv(args.y_csv, args.p_csv, args.y_col, args.p_col)
    else:
        y, p = _load_from_bundle(args.bundle, args.key)

    # Call the function (supports either float or (float, details))
    out = metric_fn(y, p)
    if isinstance(out, tuple) and len(out) >= 1:
        score = float(out[0])
        details: Any = out[1] if len(out) > 1 else None
    else:
        # If the function returns a bare float-like, coerce safely
        try:
            score = float(out)  # type: ignore[arg-type]
        except Exception:
            raise TypeError("ing_hubs_datathon_metric returned an unexpected type; expected float or (float, details)")
        details = None

    print(f"Composite Score: {score:.6f}")
    if isinstance(details, dict):
        auc = details.get("auc")
        rec = details.get("recall@10")
        lift = details.get("lift@10")
        if auc is not None:
            print(f"  AUC: {float(auc):.4f}")
        if rec is not None:
            print(f"  Recall@10%: {float(rec):.4f}")
        if lift is not None:
            print(f"  Lift@10%: {float(lift):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
