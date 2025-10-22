"""
Hyperparameter Optimizer for ING Datathon

Uses Optuna to find optimal hyperparameters for LightGBM and XGBoost
models that maximize the competition metric (40% Gini + 30% Recall@10% + 30% Lift@10%).

Usage:
    import pickle

    # Load preprocessed data
    with open('X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    # Optimize
    optimizer = HyperparameterOptimizer(X_train, y_train, n_folds=5)
    best_lgb_params = optimizer.optimize_lgb(n_trials=50)
    best_xgb_params = optimizer.optimize_xgb(n_trials=50)

    # Save best params
    optimizer.save_params('best_params.py')
"""

import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
import pickle
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Import competition metric from modeling_pipeline
try:
    from src.models.modeling_pipeline import ing_hubs_datathon_metric
except ImportError as e:
    print(f"ERROR: Could not import ing_hubs_datathon_metric from modeling_pipeline.py")
    print(f"Please ensure modeling_pipeline.py is in the current directory")
    raise


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna for ING Datathon

    Optimizes LightGBM and XGBoost models to maximize the competition metric:
    40% Gini + 30% Recall@10% + 30% Lift@10%
    """

    def __init__(self, X_train, y_train, n_folds=5, random_state=42):
        """
        Initialize optimizer

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        n_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_folds = n_folds
        self.random_state = random_state

        # Best parameters found
        self.best_lgb_params = None
        self.best_xgb_params = None
        self.best_lgb_score = None
        self.best_xgb_score = None

        # Optuna studies
        self.lgb_study = None
        self.xgb_study = None

        print("="*60)
        print("HYPERPARAMETER OPTIMIZER INITIALIZED")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Churn rate: {y_train.mean():.4f}")
        print(f"Cross-validation folds: {n_folds}")
        print(f"Random state: {random_state}")
        print("="*60)

    def objective_lgb(self, trial):
        """
        Optuna objective function for LightGBM

        Suggests hyperparameters and evaluates them using cross-validation
        with the competition metric.

        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object

        Returns:
        --------
        float : Mean competition score across folds
        """
        # Suggest hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_jobs': -1,

            # Tunable parameters
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        }

        # Cross-validation
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # Create datasets
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=2000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0)  # Silent training
                ]
            )

            # Predict on validation set
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)

            # Calculate competition metric
            fold_score, _ = ing_hubs_datathon_metric(y_val, val_preds)
            fold_scores.append(fold_score)

            # Report intermediate value for pruning
            trial.report(fold_score, fold)

            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = np.mean(fold_scores)
        return mean_score

    def objective_xgb(self, trial):
        """
        Optuna objective function for XGBoost

        Suggests hyperparameters and evaluates them using cross-validation
        with the competition metric.

        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object

        Returns:
        --------
        float : Mean competition score across folds
        """
        # Suggest hyperparameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': self.random_state,
            'n_jobs': -1,

            # Tunable parameters
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        }

        # Cross-validation
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # Train model
            model = xgb.XGBClassifier(
                **params,
                n_estimators=2000,
                early_stopping_rounds=100,
                verbose=0
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=0
            )

            # Predict on validation set
            val_preds = model.predict_proba(X_val)[:, 1]

            # Calculate competition metric
            fold_score, _ = ing_hubs_datathon_metric(y_val, val_preds)
            fold_scores.append(fold_score)

            # Report intermediate value for pruning
            trial.report(fold_score, fold)

            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = np.mean(fold_scores)
        return mean_score

    def optimize_lgb(self, n_trials=50, timeout=None):
        """
        Optimize LightGBM hyperparameters using Optuna

        Parameters:
        -----------
        n_trials : int
            Number of optimization trials
        timeout : int, optional
            Timeout in seconds (None = no timeout)

        Returns:
        --------
        dict : Best hyperparameters found
        """
        print("\n" + "="*60)
        print("OPTIMIZING LIGHTGBM HYPERPARAMETERS")
        print("="*60)
        print(f"Trials: {n_trials}")
        print(f"Timeout: {timeout if timeout else 'None'}")
        print("="*60)

        start_time = datetime.now()

        try:
            # Create Optuna study
            self.lgb_study = optuna.create_study(
                direction='maximize',
                study_name='lgb_optimization',
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=3
                )
            )

            # Optimize
            self.lgb_study.optimize(
                self.objective_lgb,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
                callbacks=[
                    lambda study, trial: print(
                        f"\nTrial {trial.number + 1}/{n_trials} completed | "
                        f"Score: {trial.value:.6f} | "
                        f"Best: {study.best_value:.6f}"
                    )
                ]
            )

            # Get best parameters
            self.best_lgb_params = self.lgb_study.best_params
            self.best_lgb_score = self.lgb_study.best_value

            elapsed = datetime.now() - start_time

            print("\n" + "="*60)
            print("LIGHTGBM OPTIMIZATION COMPLETE")
            print("="*60)
            print(f"Best score: {self.best_lgb_score:.6f}")
            print(f"Time elapsed: {elapsed}")
            print("\nBest parameters:")
            for param, value in self.best_lgb_params.items():
                print(f"  {param}: {value}")
            print("="*60)

            return self.best_lgb_params

        except KeyboardInterrupt:
            print("\n\nOptimization interrupted by user")
            if self.lgb_study and self.lgb_study.best_trial:
                print("Returning best parameters found so far...")
                self.best_lgb_params = self.lgb_study.best_params
                self.best_lgb_score = self.lgb_study.best_value
                return self.best_lgb_params
            else:
                print("No trials completed")
                return None
        except Exception as e:
            print(f"\nERROR during LightGBM optimization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def optimize_xgb(self, n_trials=50, timeout=None):
        """
        Optimize XGBoost hyperparameters using Optuna

        Parameters:
        -----------
        n_trials : int
            Number of optimization trials
        timeout : int, optional
            Timeout in seconds (None = no timeout)

        Returns:
        --------
        dict : Best hyperparameters found
        """
        print("\n" + "="*60)
        print("OPTIMIZING XGBOOST HYPERPARAMETERS")
        print("="*60)
        print(f"Trials: {n_trials}")
        print(f"Timeout: {timeout if timeout else 'None'}")
        print("="*60)

        start_time = datetime.now()

        try:
            # Create Optuna study
            self.xgb_study = optuna.create_study(
                direction='maximize',
                study_name='xgb_optimization',
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=3
                )
            )

            # Optimize
            self.xgb_study.optimize(
                self.objective_xgb,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
                callbacks=[
                    lambda study, trial: print(
                        f"\nTrial {trial.number + 1}/{n_trials} completed | "
                        f"Score: {trial.value:.6f} | "
                        f"Best: {study.best_value:.6f}"
                    )
                ]
            )

            # Get best parameters
            self.best_xgb_params = self.xgb_study.best_params
            self.best_xgb_score = self.xgb_study.best_value

            elapsed = datetime.now() - start_time

            print("\n" + "="*60)
            print("XGBOOST OPTIMIZATION COMPLETE")
            print("="*60)
            print(f"Best score: {self.best_xgb_score:.6f}")
            print(f"Time elapsed: {elapsed}")
            print("\nBest parameters:")
            for param, value in self.best_xgb_params.items():
                print(f"  {param}: {value}")
            print("="*60)

            return self.best_xgb_params

        except KeyboardInterrupt:
            print("\n\nOptimization interrupted by user")
            if self.xgb_study and self.xgb_study.best_trial:
                print("Returning best parameters found so far...")
                self.best_xgb_params = self.xgb_study.best_params
                self.best_xgb_score = self.xgb_study.best_value
                return self.best_xgb_params
            else:
                print("No trials completed")
                return None
        except Exception as e:
            print(f"\nERROR during XGBoost optimization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_params(self, filename='optimized_params.py'):
        """
        Save best parameters to a Python file

        Parameters:
        -----------
        filename : str
            Output filename (should end with .py)
        """
        print("\n" + "="*60)
        print("SAVING BEST PARAMETERS")
        print("="*60)

        try:
            with open(filename, 'w') as f:
                f.write('"""\n')
                f.write('Best Hyperparameters from Optuna Optimization\n')
                f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write('"""\n\n')

                # LightGBM parameters
                if self.best_lgb_params:
                    f.write('# LightGBM Optimized Parameters\n')
                    f.write(f'# Best Score: {self.best_lgb_score:.6f}\n')
                    f.write('OPTIMIZED_LGB_PARAMS = {\n')
                    f.write("    'objective': 'binary',\n")
                    f.write("    'metric': 'auc',\n")
                    f.write("    'boosting_type': 'gbdt',\n")
                    f.write("    'verbose': -1,\n")
                    f.write("    'random_state': 42,\n")
                    f.write("    'n_jobs': -1,\n")
                    for param, value in self.best_lgb_params.items():
                        if isinstance(value, str):
                            f.write(f"    '{param}': '{value}',\n")
                        else:
                            f.write(f"    '{param}': {value},\n")
                    f.write('}\n\n')
                else:
                    f.write('# LightGBM parameters not optimized\n')
                    f.write('OPTIMIZED_LGB_PARAMS = None\n\n')

                # XGBoost parameters
                if self.best_xgb_params:
                    f.write('# XGBoost Optimized Parameters\n')
                    f.write(f'# Best Score: {self.best_xgb_score:.6f}\n')
                    f.write('OPTIMIZED_XGB_PARAMS = {\n')
                    f.write("    'objective': 'binary:logistic',\n")
                    f.write("    'eval_metric': 'auc',\n")
                    f.write("    'tree_method': 'hist',\n")
                    f.write("    'random_state': 42,\n")
                    f.write("    'n_jobs': -1,\n")
                    for param, value in self.best_xgb_params.items():
                        if isinstance(value, str):
                            f.write(f"    '{param}': '{value}',\n")
                        else:
                            f.write(f"    '{param}': {value},\n")
                    f.write('}\n\n')
                else:
                    f.write('# XGBoost parameters not optimized\n')
                    f.write('OPTIMIZED_XGB_PARAMS = None\n\n')

                # Usage example
                f.write('# Usage:\n')
                f.write('# from optimized_params import OPTIMIZED_LGB_PARAMS, OPTIMIZED_XGB_PARAMS\n')
                f.write('# \n')
                f.write('# # In main.py, these are automatically imported and used\n')
                f.write('# model = lgb.train(OPTIMIZED_LGB_PARAMS, train_data, ...)\n')
                f.write('# or\n')
                f.write('# model = xgb.XGBClassifier(**OPTIMIZED_XGB_PARAMS)\n')

            print(f"[OK] Parameters saved to: {filename}")
            print("="*60)

        except Exception as e:
            print(f"ERROR saving parameters: {e}")
            import traceback
            traceback.print_exc()

    def plot_optimization_history(self):
        """Plot optimization history for both studies"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # LightGBM history
            if self.lgb_study:
                trials_df = self.lgb_study.trials_dataframe()
                axes[0].plot(trials_df['number'], trials_df['value'], marker='o')
                axes[0].axhline(y=self.best_lgb_score, color='r', linestyle='--',
                               label=f'Best: {self.best_lgb_score:.6f}')
                axes[0].set_xlabel('Trial')
                axes[0].set_ylabel('Competition Score')
                axes[0].set_title('LightGBM Optimization History')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

            # XGBoost history
            if self.xgb_study:
                trials_df = self.xgb_study.trials_dataframe()
                axes[1].plot(trials_df['number'], trials_df['value'], marker='o', color='orange')
                axes[1].axhline(y=self.best_xgb_score, color='r', linestyle='--',
                               label=f'Best: {self.best_xgb_score:.6f}')
                axes[1].set_xlabel('Trial')
                axes[1].set_ylabel('Competition Score')
                axes[1].set_title('XGBoost Optimization History')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('optimization_history.png', dpi=300, bbox_inches='tight')
            print("\n[OK] Optimization history plot saved to: optimization_history.png")

        except ImportError:
            print("\nNote: matplotlib not available for plotting")
        except Exception as e:
            print(f"\nWarning: Could not create plot: {e}")


def main():
    """
    Main function demonstrating usage
    """
    print("\n" + "="*60)
    print("ING DATATHON - HYPERPARAMETER OPTIMIZER")
    print("="*60)

    try:
        # Load preprocessed training data
        print("\nLoading preprocessed data...")

        if not Path('X_train.pkl').exists():
            print("\nERROR: X_train.pkl not found!")
            print("Please run save_training_data.py first to create preprocessed data files:")
            print("  python save_training_data.py")
            return 1

        with open('X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open('y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)

        print(f"[OK] Loaded X_train: {X_train.shape}")
        print(f"[OK] Loaded y_train: {len(y_train)}")

        # Initialize optimizer
        optimizer = HyperparameterOptimizer(X_train, y_train, n_folds=5)

        # Optimize LightGBM (adjust n_trials as needed)
        print("\n" + "="*60)
        print("Starting LightGBM optimization...")
        print("This may take a while. You can interrupt with Ctrl+C")
        print("="*60)

        best_lgb = optimizer.optimize_lgb(n_trials=50)

        # Optimize XGBoost
        print("\n" + "="*60)
        print("Starting XGBoost optimization...")
        print("="*60)

        best_xgb = optimizer.optimize_xgb(n_trials=50)

        # Save best parameters
        optimizer.save_params('optimized_params.py')

        # Plot optimization history
        optimizer.plot_optimization_history()

        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review optimized_params.py for optimized hyperparameters")
        print("2. Run main.py - it will automatically use these optimized parameters")
        print("3. Consider running more trials for further improvement")

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: Required file not found: {e}")
        print("Please ensure all required files are in the current directory")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
