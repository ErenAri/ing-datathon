import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.modeling_pipeline import oof_composite_monthwise

Method = Literal['isotonic', 'beta']


@dataclass
class CalibrationConfig:
    n_folds: int = 5
    segment_col: Optional[str] = None  # e.g., 'tenure'
    min_segment_rows: int = 2000
    gamma_grid: Tuple[float, ...] = (0.85, 0.90, 0.95, 1.00, 1.05)
    eps: float = 1e-6
    raw_blend: float = 0.2


def _segment_bins(values: np.ndarray, min_rows: int) -> Optional[np.ndarray]:
    vals = np.asarray(values, dtype=float)
    valid = vals[~np.isnan(vals)]
    if valid.size < min_rows:
        return None
    q_edges = np.quantile(valid, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges = np.unique(q_edges)
    if len(edges) < 3:
        return None
    bins = np.digitize(vals, edges[1:-1], right=False)
    return bins


def _fit_isotonic_oof(y: np.ndarray, oof: np.ndarray, folds: np.ndarray, segment: Optional[np.ndarray], cfg: CalibrationConfig) -> Tuple[np.ndarray, Dict]:
    eps = cfg.eps
    y = np.asarray(y)
    oof = np.clip(np.asarray(oof, dtype=float), eps, 1.0 - eps)
    cal = np.zeros_like(oof, dtype=float)
    info = {'segments': 0}

    if segment is None:
        # K-fold isotonic: fit on out-of-fold logic using provided fold assignments
        for k in np.unique(folds):
            mask_va = (folds == k)
            mask_tr = ~mask_va
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(oof[mask_tr], y[mask_tr])
            cal[mask_va] = iso.transform(oof[mask_va])
    else:
        # Segment-aware: per-segment isotonic within each fold
        for k in np.unique(folds):
            mask_va = (folds == k)
            mask_tr = ~mask_va
            seg_tr = segment[mask_tr]
            seg_va = segment[mask_va]
            cal_va = np.empty(np.sum(mask_va), dtype=float)
            used = 0
            for s in np.unique(segment):
                mtr = mask_tr.copy()
                mva = mask_va.copy()
                mtr &= (segment == s)
                mva &= (segment == s)
                n_tr = int(mtr.sum())
                if n_tr >= cfg.min_segment_rows:
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(oof[mtr], y[mtr])
                    cal[mva] = iso.transform(oof[mva])
                    used += 1
                else:
                    # fallback global
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(oof[mask_tr], y[mask_tr])
                    cal[mva] = iso.transform(oof[mva])
            info['segments'] = max(info['segments'], used)

    # Stabilize and blend
    cal = np.clip(cal, eps, 1.0 - eps)
    cal = (1.0 - cfg.raw_blend) * cal + cfg.raw_blend * oof
    return cal, info


def _fit_beta_oof(y: np.ndarray, oof: np.ndarray, folds: np.ndarray, segment: Optional[np.ndarray], cfg: CalibrationConfig) -> Tuple[np.ndarray, Dict]:
    eps = cfg.eps
    y = np.asarray(y)
    oof = np.clip(np.asarray(oof, dtype=float), eps, 1.0 - eps)
    cal = np.zeros_like(oof, dtype=float)
    info = {'segments': 0}

    def _beta_fit_predict(x_tr, y_tr, x_va):
        p = np.clip(x_tr, eps, 1.0 - eps)
        X = np.column_stack([np.log(p), np.log(1.0 - p)])
        lr = LogisticRegression(max_iter=5000)
        lr.fit(X, y_tr)
        pva = np.clip(x_va, eps, 1.0 - eps)
        Xv = np.column_stack([np.log(pva), np.log(1.0 - pva)])
        return lr.predict_proba(Xv)[:, 1]

    if segment is None:
        for k in np.unique(folds):
            mask_va = (folds == k)
            mask_tr = ~mask_va
            cal[mask_va] = _beta_fit_predict(oof[mask_tr], y[mask_tr], oof[mask_va])
    else:
        for k in np.unique(folds):
            mask_va = (folds == k)
            mask_tr = ~mask_va
            used = 0
            for s in np.unique(segment):
                mtr = mask_tr.copy()
                mva = mask_va.copy()
                mtr &= (segment == s)
                mva &= (segment == s)
                n_tr = int(mtr.sum())
                if n_tr >= cfg.min_segment_rows:
                    cal[mva] = _beta_fit_predict(oof[mtr], y[mtr], oof[mva])
                    used += 1
                else:
                    cal[mva] = _beta_fit_predict(oof[mask_tr], y[mask_tr], oof[mva])
            info['segments'] = max(info['segments'], used)

    cal = np.clip(cal, eps, 1.0 - eps)
    cal = (1.0 - cfg.raw_blend) * cal + cfg.raw_blend * oof
    return cal, info


def kfold_calibrate_with_gamma(
    y: np.ndarray,
    oof: np.ndarray,
    ref_dates: pd.Series,
    method: Method,
    folds: Optional[np.ndarray] = None,
    segment_values: Optional[np.ndarray] = None,
    cfg: Optional[CalibrationConfig] = None,
) -> Dict:
    """Calibrate OOF with K-fold calibrators (isotonic or beta), then grid search gamma on OOF
    to maximize month-wise composite. Also refits a final calibrator on full data for test use.

    Returns dict with keys: oof_cal, test_cal_fn, best_gamma, method, pre_score, post_score, ranges.
    test_cal_fn: a function that takes raw_test_probs and returns calibrated+gamma-adjusted probs.
    """
    if cfg is None:
        cfg = CalibrationConfig()

    y = np.asarray(y)
    oof = np.asarray(oof, dtype=float)
    pre_score = oof_composite_monthwise(y, oof, ref_dates=ref_dates, last_n_months=6)

    # Build folds if not provided: use month folds as group ids
    if folds is None:
        m = pd.to_datetime(ref_dates).dt.to_period('M')
        months = m.astype(str).values
        # Assign fold id by month label
        uniq = {mo: i for i, mo in enumerate(sorted(set(months)))}
        folds = np.array([uniq[mo] for mo in months], dtype=int)

    seg = None
    if cfg.segment_col is not None and segment_values is not None:
        seg = _segment_bins(np.asarray(segment_values, dtype=float), cfg.min_segment_rows)

    if method == 'isotonic':
        oof_cal, info = _fit_isotonic_oof(y, oof, folds, seg, cfg)
        # Full-data refit for test
        def _fit_full_iso():
            if seg is None:
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(oof, y)
                return lambda t: iso.transform(np.clip(t, cfg.eps, 1.0 - cfg.eps))
            else:
                # segment-aware: fit per segment else global fallback
                iso_full = IsotonicRegression(out_of_bounds='clip')
                iso_full.fit(oof, y)
                def _apply(t, seg_vals=None):
                    tt = np.clip(np.asarray(t, dtype=float), cfg.eps, 1.0 - cfg.eps)
                    return iso_full.transform(tt)
                return _apply
        test_fn = _fit_full_iso()
    else:
        oof_cal, info = _fit_beta_oof(y, oof, folds, seg, cfg)
        def _fit_full_beta():
            p = np.clip(np.asarray(oof, dtype=float), cfg.eps, 1.0 - cfg.eps)
            X = np.column_stack([np.log(p), np.log(1.0 - p)])
            lr = LogisticRegression(max_iter=5000)
            lr.fit(X, y)
            def _apply(t):
                pt = np.clip(np.asarray(t, dtype=float), cfg.eps, 1.0 - cfg.eps)
                Xt = np.column_stack([np.log(pt), np.log(1.0 - pt)])
                return lr.predict_proba(Xt)[:, 1]
            return _apply
        test_fn = _fit_full_beta()

    # Gamma grid-search
    best_gamma = 1.0
    best_score = -1.0
    for g in cfg.gamma_grid:
        p_adj = np.clip(np.power(oof_cal, g), cfg.eps, 1.0 - cfg.eps)
        sc = oof_composite_monthwise(y, p_adj, ref_dates=ref_dates, last_n_months=6)
        if sc > best_score:
            best_score = sc
            best_gamma = g

    post_score = best_score
    ranges = {
        'pre_min': float(np.min(oof)), 'pre_max': float(np.max(oof)),
        'post_min': float(np.min(np.power(oof_cal, best_gamma))),
        'post_max': float(np.max(np.power(oof_cal, best_gamma)))
    }

    def test_cal_fn(raw_test: np.ndarray, segment_values_test: Optional[np.ndarray] = None) -> np.ndarray:
        calibrated = test_fn(raw_test)
        calibrated = (1.0 - cfg.raw_blend) * np.asarray(calibrated, dtype=float) + cfg.raw_blend * np.asarray(raw_test, dtype=float)
        calibrated = np.clip(calibrated, cfg.eps, 1.0 - cfg.eps)
        return np.clip(np.power(calibrated, best_gamma), cfg.eps, 1.0 - cfg.eps)

    return {
        'oof_cal': np.clip(np.power(oof_cal, best_gamma), cfg.eps, 1.0 - cfg.eps),
        'test_cal_fn': test_cal_fn,
        'best_gamma': best_gamma,
        'method': method,
        'pre_score': pre_score,
        'post_score': post_score,
        'ranges': ranges,
        'segments_used': info.get('segments', 0)
    }
