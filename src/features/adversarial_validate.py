import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any

import lightgbm as lgb

from src.models.modeling_pipeline import ChurnModelingPipeline, oof_composite_monthwise


def load_cached() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_tr = pd.read_pickle('X_train.pkl')
    y_tr = pd.read_pickle('y_train.pkl')
    X_te = pd.read_pickle('X_test.pkl')
    ref = pd.read_pickle('ref_dates.pkl')
    if not isinstance(ref, pd.Series):
        ref = pd.Series(ref)
    return X_tr, pd.Series(y_tr), X_te, ref


def train_domain_auc(X_train: pd.DataFrame, X_test: pd.DataFrame, seed: int = 42) -> tuple[float, List[str]]:
    # Align columns
    common = [c for c in X_train.columns if c in set(X_test.columns)]
    Xt = X_train[common].copy()
    Xs = X_test[common].copy()
    y_dom = np.concatenate([np.zeros(len(Xt), dtype=int), np.ones(len(Xs), dtype=int)])
    X_dom = pd.concat([Xt, Xs], axis=0, ignore_index=True)

    # Quick holdout AUC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    X_tr, X_va, y_tr, y_va = train_test_split(X_dom, y_dom, test_size=0.2, random_state=seed, stratify=y_dom)
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
        'num_leaves': 63, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'min_data_in_leaf': 50, 'verbose': -1, 'random_state': seed
    }
    m = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva], valid_names=['valid'], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    if 'valid' in m.best_score and 'auc' in m.best_score['valid']:
        auc = float(m.best_score['valid']['auc'])
    else:
        auc = float(roc_auc_score(y_va, np.asarray(m.predict(X_va), dtype=float)))

    # Feature ranking via SHAP if available, else gain
    try:
        import shap  # type: ignore
        expl = shap.TreeExplainer(m)
        sample_n = min(3000, len(X_dom))
        idx = np.random.default_rng(seed).choice(len(X_dom), size=sample_n, replace=False)
        sv = expl.shap_values(X_dom.iloc[idx])
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        shap_mean = np.mean(np.abs(sv), axis=0)
        imp = pd.Series(shap_mean, index=X_dom.columns).sort_values(ascending=False)
        ranked = imp.index.tolist()
    except Exception:
        gain = m.feature_importance(importance_type='gain')
        imp = pd.Series(gain, index=X_dom.columns).sort_values(ascending=False)
        ranked = imp.index.tolist()

    return auc, ranked


def evaluate_main_metric(X: pd.DataFrame, y: pd.Series, ref_dates: pd.Series, drop_cols: List[str]) -> float:
    # Keep intersection only
    cols = [c for c in X.columns if c not in set(drop_cols)]
    Xr = X[cols]
    pipe = ChurnModelingPipeline(n_folds=5, random_state=42)
    # Quick LGB with defaults; use month-based folds on last 6 months
    oof, score = pipe.train_lightgbm(Xr, y, params=None, ref_dates=ref_dates, last_n_months=6, sample_weight=None)
    # The train_lightgbm already prints and computes; return month-wise composite
    return float(score)


def iterative_adversarial_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    ref_dates: pd.Series,
    last_n: int = 6,
    base_models: Dict[str, Any] | None = None,
    K_to_drop_each_step: int = 5,
    target_auc: float = 0.75,
    max_oof_drop: float = 0.02,
    seed: int = 42,
) -> Dict[str, Any]:
    """Iteratively drop top drift features to push domain AUC ≤ target.

    Allows month-wise composite to degrade by up to max_oof_drop from the best-so-far baseline.

    Returns a dict with:
      - keep_columns: List[str]
      - dropped_features: List[str]
      - history: List[dict(step, domain_auc, oof_composite, dropped)]
      - domain_auc_final: float
      - oof_composite_final: float
    """
    # Align and numeric-only frames
    common = [c for c in X_train.columns if c in set(X_test.columns)]
    X_tr = X_train[common].select_dtypes(include=[np.number]).copy()
    X_te = X_test[common].select_dtypes(include=[np.number]).copy()

    # Baseline composite
    pipe = ChurnModelingPipeline(n_folds=5, random_state=seed)
    oof_base, score_base = pipe.train_lightgbm(X_tr, y_train, params=None, ref_dates=ref_dates, last_n_months=last_n, sample_weight=None)
    best_score = float(score_base)

    # Helper to train domain and rank features
    def _domain_rank(cols: List[str]) -> tuple[float, List[str]]:
        # Train/valid split on domain task
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score as _AUC
        Xt = X_tr[cols]
        Xs = X_te[cols]
        y_dom = np.concatenate([np.zeros(len(Xt), dtype=int), np.ones(len(Xs), dtype=int)])
        X_dom = pd.concat([Xt, Xs], axis=0, ignore_index=True)
        X_trd, X_vad, y_trd, y_vad = train_test_split(X_dom, y_dom, test_size=0.2, random_state=seed, stratify=y_dom)
        dtr = lgb.Dataset(X_trd, label=y_trd)
        dva = lgb.Dataset(X_vad, label=y_vad, reference=dtr)
        params = {
            'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
            'num_leaves': 63, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 1, 'min_data_in_leaf': 50, 'verbose': -1, 'random_state': seed
        }
        m = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva], valid_names=['valid'], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        if 'valid' in m.best_score and 'auc' in m.best_score['valid']:
            auc = float(m.best_score['valid']['auc'])
        else:
            auc = float(_AUC(y_vad, np.asarray(m.predict(X_vad), dtype=float)))
        # Rank features by SHAP if available; fallback to gain
        try:
            import shap  # type: ignore
            expl = shap.TreeExplainer(m)
            sample_n = min(3000, len(X_dom))
            idx = np.random.default_rng(seed).choice(len(X_dom), size=sample_n, replace=False)
            sv = expl.shap_values(X_dom.iloc[idx])
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            shap_mean = np.mean(np.abs(sv), axis=0)
            imp = pd.Series(shap_mean, index=cols).sort_values(ascending=False)
            ranked = imp.index.tolist()
        except Exception:
            gain = m.feature_importance(importance_type='gain')
            imp = pd.Series(gain, index=cols).sort_values(ascending=False)
            ranked = imp.index.tolist()
        return auc, ranked

    keep_cols = X_tr.columns.tolist()
    dropped: List[str] = []
    history: List[Dict[str, Any]] = []
    step = 0

    while True:
        step += 1
        dom_auc, ranked = _domain_rank(keep_cols)
        print(f"[IterADV] Step {step}: domain AUC={dom_auc:.4f} | keep_features={len(keep_cols)}")
        history.append({'step': step, 'domain_auc': float(dom_auc), 'oof_composite': float(best_score), 'dropped': []})
        if dom_auc <= float(target_auc):
            print(f"[IterADV] Target achieved: domain AUC ≤ {target_auc:.3f}")
            break

        # Propose dropping top-K not yet dropped
        K = int(max(1, K_to_drop_each_step))
        ranked_in = [c for c in ranked if c in keep_cols]
        if not ranked_in:
            print("[IterADV] No rankable features left; stopping.")
            break
        # Iteratively try smaller K if necessary to satisfy oof constraint
        accepted = False
        while K >= 1 and not accepted:
            cand_drop = [c for c in ranked_in[:K] if c in keep_cols]
            trial_cols = [c for c in keep_cols if c not in set(cand_drop)]
            # Re-evaluate composite on trial set
            oof_trial, score_trial = pipe.train_lightgbm(pd.DataFrame(X_tr[trial_cols]), y_train, params=None, ref_dates=ref_dates, last_n_months=last_n, sample_weight=None)
            score_trial = float(score_trial)
            delta = score_trial - best_score
            print(f"[IterADV] Try drop K={K}: Δoof={delta:+.6f} (allow ≥ {-float(max_oof_drop):.6f})")
            if delta >= -float(max_oof_drop):
                # Accept drop
                keep_cols = trial_cols
                dropped.extend(cand_drop)
                best_score = score_trial
                history[-1]['dropped'] = cand_drop
                history[-1]['oof_composite'] = best_score
                accepted = True
            else:
                K = K // 2
        if not accepted:
            print("[IterADV] No acceptable drop found under OOF constraint; stopping.")
            break

    # Final domain AUC on keep set
    final_auc, _ = _domain_rank(keep_cols)
    return {
        'keep_columns': keep_cols,
        'dropped_features': dropped,
        'history': history,
        'domain_auc_final': float(final_auc),
        'oof_composite_final': float(best_score),
        'target_auc': float(target_auc),
        'max_oof_drop': float(max_oof_drop),
    }


def main():
    ap = argparse.ArgumentParser(description='Adversarial validation and feature dropper')
    ap.add_argument('--max-k', type=int, default=5, help='Maximum number of top domain features to consider (iteratively)')
    ap.add_argument('--tol', type=float, default=0.0, help='Allowable decrease in composite (<= tol is accepted)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    X_tr, y_tr, X_te, ref = load_cached()

    # Baseline domain AUC and ranking
    auc_before, ranked = train_domain_auc(X_tr, X_te, seed=args.seed)
    print(f"[Adversarial] Domain AUC before: {auc_before:.4f}")
    print("Top-15 domain features:")
    for i, f in enumerate(ranked[:15], 1):
        print(f"  {i:2d}. {f}")

    # Baseline main composite
    base_score = evaluate_main_metric(X_tr, y_tr, ref, drop_cols=[])
    print(f"Baseline month-wise composite: {base_score:.6f}")

    accepted: List[str] = []
    tried: List[Dict[str, Any]] = []

    for i in range(min(args.max_k, len(ranked))):
        cand = ranked[i]
        to_drop = accepted + [cand]
        print(f"\nTrying to drop feature: {cand}")
        sc = evaluate_main_metric(X_tr, y_tr, ref, drop_cols=to_drop)
        delta = sc - base_score
        accept = (delta >= -abs(args.tol))
        tried.append({'feature': cand, 'score': sc, 'delta': delta, 'accepted': accept})
        print(f"  New composite: {sc:.6f} (delta {delta:+.6f}) -> {'ACCEPT' if accept else 'REJECT'}")
        if accept:
            accepted.append(cand)
            base_score = sc

    # Report and re-train domain AUC after accepted drops
    if accepted:
        X_tr_d = X_tr.drop(columns=accepted, errors='ignore')
        X_te_d = X_te.drop(columns=accepted, errors='ignore')
        auc_after, _ = train_domain_auc(X_tr_d, X_te_d, seed=args.seed)
    else:
        auc_after = auc_before

    report = {
        'accepted_features': accepted,
        'tried': tried,
        'domain_auc_before': float(auc_before),
        'domain_auc_after': float(auc_after),
        'final_composite': float(base_score)
    }

    os.makedirs(os.path.join('outputs', 'reports'), exist_ok=True)
    out_path = os.path.join('outputs', 'reports', 'adversarial_validate_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {out_path}")
    if accepted:
        print("Accepted drops (in order):")
        for f in accepted:
            print("  -", f)
    print(f"Domain AUC before: {auc_before:.4f} -> after: {auc_after:.4f}")


if __name__ == '__main__':
    main()
