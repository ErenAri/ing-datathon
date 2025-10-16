"""
Hyperparameter tuning for LightGBM and XGBoost optimized for ING metric
- Uses month-based folds (last_n months) from saved ref_dates.pkl
- Runs Optuna with objective functions that train per-fold and return composite metric
- Saves tuned parameters to tuned_params.json and optimized_params.py

Usage (Windows PowerShell):
  # Precompute and cache features
  python .\save_training_data.py

  # Run tuning (adjust --model and --trials as needed)
  python .\tune_params.py --model lgb --trials 80 --last-n 6
  python .\tune_params.py --model xgb --trials 100 --last-n 6

Afterwards, run main.py which auto-loads optimized_params.py if present.
"""

import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb

from src.models.modeling_pipeline import ing_hubs_datathon_metric, month_folds


def load_cached_data():
    required = [
        'X_train.pkl', 'y_train.pkl', 'ref_dates.pkl'
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached files: {missing}. Run save_training_data.py first.")
    with open('X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('ref_dates.pkl', 'rb') as f:
        ref_dates = pickle.load(f)
    return X_train, y_train, ref_dates


def suggest_lgb_params(trial):
    return {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', -1, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }


def suggest_xgb_params(trial):
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }


def cv_score_lgb(params, X, y, ref_dates, last_n):
    oof = np.zeros(len(X), dtype=float)
    for tr_idx, va_idx, month_label in month_folds(ref_dates, last_n=last_n):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        model = lgb.train(
            params,
            dtr,
            num_boost_round=2000,
            valid_sets=[dtr, dva],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        p = model.predict(X_va, num_iteration=model.best_iteration)
        # flip-guard per fold
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_va, p)
            if auc < 0.5:
                p = 1.0 - p
        except Exception:
            pass
        oof[va_idx] = p
    score, _ = ing_hubs_datathon_metric(y, oof)
    return score


def cv_score_xgb(params, X, y, ref_dates, last_n):
    oof = np.zeros(len(X), dtype=float)
    for tr_idx, va_idx, month_label in month_folds(ref_dates, last_n=last_n):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        # base_score for stability
        base_score = float(np.clip(y_tr.mean(), 1e-6, 1-1e-6)) if len(y_tr) else 0.5
        model = xgb.XGBClassifier(
            **params,
            n_estimators=2000,
            early_stopping_rounds=100,
            base_score=base_score,
            verbosity=0
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p = model.predict_proba(X_va)[:, 1]
        # flip-guard per fold
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_va, p)
            if auc < 0.5:
                p = 1.0 - p
        except Exception:
            pass
        oof[va_idx] = p
    score, _ = ing_hubs_datathon_metric(y, oof)
    return score


def run_tuning(model_name: str, trials: int, last_n: int, timeout: int | None = None):
    X, y, ref_dates = load_cached_data()

    if model_name == 'lgb':
        def objective(trial: optuna.Trial):
            params = suggest_lgb_params(trial)
            return cv_score_lgb(params, X, y, ref_dates, last_n)
        direction = 'maximize'
        study = optuna.create_study(direction=direction, study_name='lgb_timecv',
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
        study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=True)
        best_params = study.best_params
        best_score = study.best_value
        tag = 'lgb'
    elif model_name == 'xgb':
        def objective(trial: optuna.Trial):
            params = suggest_xgb_params(trial)
            return cv_score_xgb(params, X, y, ref_dates, last_n)
        direction = 'maximize'
        study = optuna.create_study(direction=direction, study_name='xgb_timecv',
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
        study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=True)
        best_params = study.best_params
        best_score = study.best_value
        tag = 'xgb'
    else:
        raise ValueError("model must be one of: 'lgb', 'xgb'")

    # Save tuned params as JSON and also write optimized_params.py for main.py
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    json_path = 'tuned_params.json'
    payload = {
        'timestamp': ts,
        'model': tag,
        'last_n_months': last_n,
        'best_score': best_score,
        'best_params': best_params,
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n✓ Saved tuned params to {json_path}")

    # Update or create optimized_params.py
    opt_py = Path('optimized_params.py')
    if opt_py.exists():
        # Load existing and update just the section for this model
        with open(opt_py, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = (
            '"""\nOptuna-Optimized Hyperparameters (auto-generated)\n"""\n\n'
            'OPTIMIZED_LGB_PARAMS = None\n\nOPTIMIZED_XGB_PARAMS = None\n'
        )

    def to_block(name, params, best_score):
        lines = [f"# Best Score: {best_score:.6f}", f"{name} = {{"]
        # Inject fixed keys
        if name == 'OPTIMIZED_LGB_PARAMS':
            fixed = {
                'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                'verbose': -1, 'random_state': 42, 'n_jobs': -1
            }
        else:
            fixed = {
                'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
                'random_state': 42, 'n_jobs': -1
            }
        all_params = {**fixed, **params}
        for k, v in all_params.items():
            if isinstance(v, str):
                lines.append(f"    '{k}': '{v}',")
            else:
                lines.append(f"    '{k}': {v},")
        lines.append("}\n")
        return "\n".join(lines)

    block = to_block('OPTIMIZED_LGB_PARAMS' if tag == 'lgb' else 'OPTIMIZED_XGB_PARAMS', best_params, best_score)

    # Replace or append block in file
    if tag == 'lgb':
        import re
        if 'OPTIMIZED_LGB_PARAMS' in content:
            content = re.sub(r"# Best Score: .*?\nOPTIMIZED_LGB_PARAMS = \{[\s\S]*?\}\n",
                             block, content, count=1)
        else:
            content += "\n\n" + block
    else:
        import re
        if 'OPTIMIZED_XGB_PARAMS' in content:
            content = re.sub(r"# Best Score: .*?\nOPTIMIZED_XGB_PARAMS = \{[\s\S]*?\}\n",
                             block, content, count=1)
        else:
            content += "\n\n" + block

    with open(opt_py, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Updated {opt_py} with best {tag.upper()} params")

    print(f"\nBest {tag.upper()} score: {best_score:.6f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['lgb', 'xgb'], required=True, help='Model to tune')
    ap.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    ap.add_argument('--last-n', type=int, default=6, help='Last N months to use for CV')
    ap.add_argument('--timeout', type=int, default=None, help='Optional timeout in seconds')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_tuning(args.model, args.trials, args.last_n, args.timeout)
