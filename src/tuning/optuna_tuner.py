"""Optuna-based hyperparameter tuner for LightGBM / XGBoost using time-based CV and ING composite metric.

Usage examples (PowerShell):
  python src/tuning/optuna_tuner.py --model lgb --trials 100 --last-n 6 --timeout 7200
  python src/tuning/optuna_tuner.py --model xgb --trials 100 --last-n 6 --timeout 7200

Key steps:
  1. Load processed training matrices (X_train.pkl, y_train.pkl, ref_dates.pkl) from data/processed
  2. Build time folds: each of the last N months held out once (training strictly before month)
  3. For each trial, train per-fold model (early stopping) and compute ING composite on validation folds
  4. Average composite across folds as objective (maximize)
  5. Persist best params to models/tuned_params.json and models/optimized_params.py (also root optimized_params.py for main pipeline)
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# Project metric: re-use existing implementation in modeling_pipeline
from src.models.modeling_pipeline import ing_hubs_datathon_metric

# Optional paths helper (fallback if module not present)
try:
    from src.utils.paths import PROC, MODELS  # type: ignore
except Exception:
    PROC = ROOT / 'data' / 'processed'
    MODELS = ROOT / 'models'

SEED = 42


def load_data() -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    X_train = pd.read_pickle(PROC / 'X_train.pkl')
    y_train = pd.read_pickle(PROC / 'y_train.pkl')
    ref_dates = pd.read_pickle(PROC / 'ref_dates.pkl')
    y = np.asarray(y_train).ravel().astype(int)
    return X_train, y, pd.to_datetime(ref_dates)


def build_time_folds(ref_dates: pd.Series, last_n: int) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    months = ref_dates.dt.to_period('M')
    # months.unique() returns an array; convert to pandas Index for sorting
    uniq = pd.Index(months.unique()).sort_values()
    val_months = uniq[-last_n:]
    folds: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for m in val_months:
        va_idx = np.where(months == m)[0]
        tr_idx = np.where(months < m)[0]
        if len(va_idx) < 500 or len(tr_idx) < 500:
            continue
        folds.append((tr_idx, va_idx, str(m)))
    return folds


def suggest_params_lgb(trial: optuna.Trial) -> Dict[str, Any]:
    return dict(
        num_leaves=trial.suggest_int('num_leaves', 24, 256),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        feature_fraction=trial.suggest_float('feature_fraction', 0.6, 0.9),
        bagging_fraction=trial.suggest_float('bagging_fraction', 0.6, 0.9),
        bagging_freq=trial.suggest_int('bagging_freq', 1, 10),
        min_child_samples=trial.suggest_int('min_child_samples', 50, 250),
        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 10.0),
        reg_lambda=trial.suggest_float('reg_lambda', 0.0, 10.0),
        min_split_gain=trial.suggest_float('min_split_gain', 0.0, 1.0),
    )


def suggest_params_xgb(trial: optuna.Trial) -> Dict[str, Any]:
    return dict(
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        subsample=trial.suggest_float('subsample', 0.6, 0.9),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 0.9),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 12),
        gamma=trial.suggest_float('gamma', 0.0, 2.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 10.0),
        reg_lambda=trial.suggest_float('reg_lambda', 0.0, 10.0),
    )


def eval_lgb(params: Dict[str, Any], X: pd.DataFrame, y: np.ndarray, folds: List[Tuple[np.ndarray, np.ndarray, str]]) -> float:
    import lightgbm as lgb
    scores = []
    for tr_idx, va_idx, m in folds:
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        dvalid = lgb.Dataset(X.iloc[va_idx], label=y[va_idx])
        num_threads = os.cpu_count() or 4
        cfg = dict(
            objective='binary',
            metric='auc',
            verbosity=-1,
            seed=SEED,
            num_threads=max(1, num_threads // 2),
            feature_pre_filter=False,
            **params
        )
        model = lgb.train(
            cfg,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
            ]
        )
        best_iter = getattr(model, 'best_iteration', None)
        p = model.predict(X.iloc[va_idx], num_iteration=best_iter)
        comp = ing_hubs_datathon_metric(y[va_idx], p)
        if isinstance(comp, tuple):
            comp = comp[0]
        scores.append(float(comp))
    return float(np.mean(scores))


def eval_xgb(params: Dict[str, Any], X: pd.DataFrame, y: np.ndarray, folds: List[Tuple[np.ndarray, np.ndarray, str]]) -> float:
    import xgboost as xgb
    scores = []
    for tr_idx, va_idx, m in folds:
        dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx])
        dvalid = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx])
        nthread = os.cpu_count() or 4
        cfg = dict(
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            seed=SEED,
            nthread=max(1, nthread // 2),
            **params
        )
        model = xgb.train(
            cfg,
            dtrain,
            num_boost_round=5000,
            evals=[(dvalid, 'val')],
            early_stopping_rounds=200,
            verbose_eval=False
        )
        # XGBoost 2.x: use iteration_range instead of ntree_limit
        best_iter = getattr(model, 'best_iteration', None)
        if best_iter is None:
            try:
                # fallback to full boosted rounds
                best_iter = model.num_boosted_rounds()
            except Exception:
                best_iter = 0
        it_range = (0, int(best_iter) + 1) if best_iter is not None else None
        p = model.predict(xgb.DMatrix(X.iloc[va_idx]), iteration_range=it_range)
        comp = ing_hubs_datathon_metric(y[va_idx], p)
        if isinstance(comp, tuple):
            comp = comp[0]
        scores.append(float(comp))
    return float(np.mean(scores))


def objective(trial: optuna.Trial, model_kind: str, X: pd.DataFrame, y: np.ndarray, folds):
    if model_kind == 'lgb':
        params = suggest_params_lgb(trial)
        score = eval_lgb(params, X, y, folds)
    else:
        params = suggest_params_xgb(trial)
        score = eval_xgb(params, X, y, folds)
    trial.set_user_attr('params', params)
    return score


def save_best(model_kind: str, study: optuna.Study):
    MODELS.mkdir(parents=True, exist_ok=True)
    tuned_path = MODELS / 'tuned_params.json'
    try:
        current = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    except Exception:
        current = {}
    current[model_kind] = study.best_trial.user_attrs['params']
    tuned_path.write_text(json.dumps(current, indent=2))
    print(f"✓ Saved best {model_kind} params to {tuned_path}")

    # Also update models/optimized_params.py and root optimized_params.py for main pipeline compatibility
    for target in [MODELS / 'optimized_params.py', ROOT / 'optimized_params.py']:
        try:
            lines = ["# auto-generated by optuna_tuner.py\n"]
            for k, v in current.items():
                lines.append(f"{k.upper()}_BEST = {json.dumps(v, indent=2)}\n")
            target.write_text("".join(lines))
            print(f"✓ Updated {target}")
        except Exception as e:
            print(f"⚠ Failed to update {target}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['lgb','xgb'], help='Model to tune')
    ap.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    ap.add_argument('--last-n', type=int, default=6, help='Last N months to use for CV')
    ap.add_argument('--timeout', type=int, default=None, help='Timeout seconds (optional)')
    args = ap.parse_args()

    X, y, ref_dates = load_data()
    folds = build_time_folds(ref_dates, args.last_n)
    if not folds:
        raise RuntimeError('No valid folds constructed. Check ref_dates and last-n.')

    print(f"Model={args.model} | trials={args.trials} | last_n={args.last_n} | folds={len(folds)}")
    study = optuna.create_study(
        study_name=f"{args.model}_timecv",
        direction='maximize',
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_warmup_steps=10)
    )
    start = time.time()
    try:
        study.optimize(
            lambda t: objective(t, args.model, X, y, folds),
            n_trials=args.trials,
            timeout=args.timeout,
            show_progress_bar=True,
            gc_after_trial=True,
            n_jobs=1,
        )
    finally:
        dur = time.time() - start
        completed = [t for t in study.trials if t.state.name == 'COMPLETE']
        print(f"Completed in {dur/60:.1f} min | Trials completed: {len(completed)}/{len(study.trials)}")
        if completed:
            print(f"Best value: {study.best_value:.6f}")
            print('Best params:')
            for k,v in study.best_trial.user_attrs['params'].items():
                print(f"  {k}: {v}")
            save_best(args.model, study)
        else:
            print("No trials completed successfully; nothing to save.")


if __name__ == '__main__':
    main()
