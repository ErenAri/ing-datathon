import argparse
import os
import pickle
from typing import List, Optional

import numpy as np


def recall_at_k(y_true, y_prob, k=0.1):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    tp_at_k = y_true[top].sum()
    P = y_true.sum()
    return float(tp_at_k / P) if P > 0 else 0.0


def lift_at_k(y_true, y_prob, k=0.1):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    tp_at_k = y_true[top].sum()
    precision_at_k = tp_at_k / m
    prevalence = y_true.mean()
    return float(precision_at_k / prevalence) if prevalence > 0 else 0.0


def convert_auc_to_gini(auc: float) -> float:
    return 2 * float(auc) - 1


def ing_hubs_datathon_metric(y_true, y_prob):
    # final metrik için ağırlıklar
    score_weights = {
        "gini": 0.4,
        "recall_at_10perc": 0.3,
        "lift_at_10perc": 0.3,
    }

    # baseline modelin her bir metrik için değerleri
    baseline_scores = {
        "roc_auc": 0.6925726757936908,
        "recall_at_10perc": 0.18469015795868773,
        "lift_at_10perc": 1.847159286784029,
    }

    from sklearn.metrics import roc_auc_score

    roc_auc = roc_auc_score(y_true, y_prob)
    recall10 = recall_at_k(y_true, y_prob, k=0.1)
    lift10 = lift_at_k(y_true, y_prob, k=0.1)

    baseline_gini = convert_auc_to_gini(baseline_scores["roc_auc"])
    new_gini = convert_auc_to_gini(roc_auc)

    final_gini_score = new_gini / baseline_gini
    final_recall_score = recall10 / baseline_scores["recall_at_10perc"]
    final_lift_score = lift10 / baseline_scores["lift_at_10perc"]

    final_score = (
        final_gini_score * score_weights["gini"]
        + final_recall_score * score_weights["recall_at_10perc"]
        + final_lift_score * score_weights["lift_at_10perc"]
    )

    details = {
        "auc": float(roc_auc),
        "gini": float(new_gini),
        "recall@10": float(recall10),
        "lift@10": float(lift10),
    }
    return float(final_score), details


def monthwise_scores(y_true, y_prob, ref_dates, last_n: int = 6):
    import pandas as pd
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    # Robust conversion to PeriodIndex by month for arrays, Series, or DatetimeIndex
    try:
        if hasattr(ref_dates, "to_period"):
            m = ref_dates.to_period("M")
        else:
            m = pd.to_datetime(ref_dates).to_period("M")
    except Exception:
        m = pd.to_datetime(ref_dates).to_period("M")
    months = list(sorted(m.unique()))[-last_n:]
    rows = []
    vals = []
    for vm in months:
        mask = (m.values == vm)
        if not np.any(mask):
            continue
        y_v = y_true[mask]
        p_v = y_prob[mask].astype(float)
        # Flip guard per month
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_v, p_v)
            if auc < 0.5:
                p_v = 1.0 - p_v
        except Exception:
            pass
        s, det = ing_hubs_datathon_metric(y_v, p_v)
        rows.append((str(vm), s, det))
        vals.append(s)
    avg = float(np.mean(vals)) if vals else None
    return rows, avg


def build_pred_vector(bundle: dict, pred_key: Optional[str], keys: Optional[List[str]]):
    import numpy as _np
    if pred_key and pred_key in bundle:
        return _np.asarray(bundle[pred_key], dtype=float)
    if keys:
        parts = [k for k in keys if k in bundle]
        if parts:
            return _np.mean(_np.vstack([_np.asarray(bundle[k], dtype=float) for k in parts]), axis=0)
    # Fallbacks
    if 'oof_ensemble' in bundle:
        return _np.asarray(bundle['oof_ensemble'], dtype=float)
    parts = [k for k in ['oof_lgb', 'oof_xgb', 'oof_two_stage_B', 'oof_cat'] if k in bundle]
    if parts:
        return _np.mean(_np.vstack([_np.asarray(bundle[k], dtype=float) for k in parts]), axis=0)
    raise KeyError("No suitable OOF prediction vector found in bundle.")


def main():
    ap = argparse.ArgumentParser(description="Score OOF predictions using ING Hubs Datathon metric")
    ap.add_argument("--bundle", default=os.path.join("outputs", "predictions", "predictions_bundle.pkl"), help="Path to predictions bundle pickle")
    ap.add_argument("--pred-key", default=None, help="Prediction key in bundle to score (e.g., oof_ensemble)")
    ap.add_argument("--keys", default=None, help="Comma-separated keys to average if --pred-key is not provided")
    ap.add_argument("--monthwise", action="store_true", help="Also compute month-wise scores (requires ref_dates in bundle)")
    ap.add_argument("--last-n", type=int, default=6, help="Number of last months to include for month-wise averaging")
    args = ap.parse_args()

    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)

    y = bundle.get("y_train")
    if y is None:
        raise KeyError("Bundle missing 'y_train' for OOF ground truth.")

    keys_list = [k.strip() for k in args.keys.split(",") if k.strip()] if args.keys else None
    p = build_pred_vector(bundle, args.pred_key, keys_list)

    score, details = ing_hubs_datathon_metric(y, p)
    print(f"OOF composite: {score:.6f}")
    print(f"Details: {details}")

    if args.monthwise:
        ref_dates = bundle.get("ref_dates")
        if ref_dates is None:
            print("Bundle missing 'ref_dates'; skipping month-wise breakdown.")
        else:
            rows, avg = monthwise_scores(y, p, ref_dates, last_n=args.last_n)
            print("\nMonth-wise (last {}):".format(args.last_n))
            for vm, s, det in rows:
                print(f"  {vm}: {s:.6f} | {{'auc': {det['auc']:.4f}, 'gini': {det['gini']:.4f}, 'recall@10': {det['recall@10']:.4f}, 'lift@10': {det['lift@10']:.4f}}}")
            if avg is not None:
                print(f"Mean of month-wise composites: {avg:.6f}")


if __name__ == "__main__":
    main()
