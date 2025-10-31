# Optuna-optimized hyperparameters (Extended CV strategy)

LGB_BEST = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': 42,
    'num_leaves': 52,
    'learning_rate': 0.046721016026456376,
    'feature_fraction': 0.8933519463197295,
    'bagging_fraction': 0.6712428466350675,
    'max_depth': 4,
    'min_child_samples': 20,
    'reg_alpha': 0.557066133544035,
    'reg_lambda': 5.480515264997663,
}

XGB_BEST = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'tree_method': 'hist',
    'max_depth': 6,
    'learning_rate': 0.04727214601413622,
    'subsample': 0.879214350091299,
    'colsample_bytree': 0.7314332805931403,
    'min_child_weight': 11,
    'reg_alpha': 0.5473119659763015,
    'reg_lambda': 5.591121995268725,
}

USE_OPTIMIZED_PARAMS = True
