"""
Repair utilities for stacking and calibration.

Part 1: train_robust_stacker(oof_predictions_dict, y_train)
  - Build a meta-dataset from base OOF predictions only (LGB, XGB, Cat, Two-Stage)
  - Train LogisticRegressionCV with time-based folds; return oof_meta, score_meta, model

Part 2: train_kfold_calibrator(oof_predictions, y_train, n_splits=6)
  - StratifiedKFold K-1 fit with IsotonicRegression; fold-calibrated OOF and final test calibrator factory
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.isotonic import IsotonicRegression

from src.models.modeling_pipeline import ing_hubs_datathon_metric as _metric_fn


def _safe_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return np.clip(arr, 1e-9, 1 - 1e-9)


def train_robust_stacker(oof_predictions_dict: Dict[str, np.ndarray], y_train: np.ndarray, n_splits: int = 5) -> Tuple[np.ndarray, float, LogisticRegressionCV]:
    """Train a simple LogisticRegressionCV stacker using only base OOF predictions.

    Returns: (oof_meta, score_meta, model)
    """
    # Filter only allowed bases and stack columns in a fixed order
    base_order = [k for k in ['lgb', 'xgb', 'cat', 'two_stageB'] if k in oof_predictions_dict]
    if not base_order:
        raise ValueError("No base OOF predictions provided for stacker.")
    X_meta = np.vstack([_safe_array(oof_predictions_dict[k]) for k in base_order]).T
    y = np.asarray(y_train, dtype=int).reshape(-1)  # FIX: Must be int for classifier

    # Time-based CV for LR hyperparameters
    # We use TimeSeriesSplit as a simple proxy (assuming y is already time-ordered); otherwise set cv=5
    try:
        cv = TimeSeriesSplit(n_splits=n_splits)
    except Exception:
        cv = n_splits
    lr = LogisticRegressionCV(Cs=20, cv=cv, solver='lbfgs', max_iter=2000, n_jobs=-1)
    lr.fit(X_meta, y)

    # OOF for meta equals the LR decision function on OOF inputs
    oof_meta = lr.predict_proba(X_meta)[:, 1].astype(float)
    score_meta, _ = _metric_fn(y, oof_meta)
    return oof_meta, float(score_meta), lr


def train_kfold_calibrator(oof_predictions: np.ndarray, y_train: np.ndarray, n_splits: int = 6) -> Tuple[np.ndarray, float, Callable[[np.ndarray], np.ndarray]]:
    """K-fold calibration using IsotonicRegression.

    Returns calibrated OOF predictions, the composite score, and a function that calibrates new arrays.
    """
    p = _safe_array(oof_predictions)
    y = _safe_array(y_train)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    p_cal = np.zeros_like(p, dtype=float)
    calibrators: List[IsotonicRegression] = []
    for tr_idx, va_idx in skf.split(p.reshape(-1, 1), y.astype(int)):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(p[tr_idx], y[tr_idx])
        p_cal[va_idx] = ir.transform(p[va_idx])
        calibrators.append(ir)

    score_cal, _ = _metric_fn(y, p_cal)

    def final_calibrator(new_scores: np.ndarray) -> np.ndarray:
        # Fit a final calibrator on full OOF for test-time transformation
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(p, y)
        return ir.transform(_safe_array(new_scores))

    return p_cal, float(score_cal), final_calibrator


if __name__ == '__main__':
    # Tiny smoke (optional): users can import these in notebooks or other scripts
    pass
