# ING Datathon Project

This repository contains feature engineering, modeling, and ensembling code for the ING Datathon.

## Quick start (Windows PowerShell)

1) Create/activate a virtual environment (optional but recommended):

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Run the full pipeline (training + inference + submission):

```powershell
python .\main.py
```

Outputs will be written under `data/submissions/` and `outputs/` (details below).

## Folder structure

- `src/` — source code
  - `features/` — time-safe feature generation modules
  - `models/` — CV, training heads (LightGBM/XGBoost/CatBoost/Two-Stage)
  - `ensemble/` — blending, stacking, calibration utilities
  - `utils/` — helpers: evaluation, saving data, tuning
- `data/` — datasets and submissions
  - `raw/` — raw inputs (not tracked)
  - `processed/` — processed intermediates (not tracked)
  - `submissions/` — generated submissions (not tracked)
  - `portfolio/` — portfolio-related files (not tracked)
- `outputs/` — predictions, reports, and logs (not tracked)
  - `predictions/` — prediction bundles and OOFs
  - `reports/` — feature importances, diagnostics
  - `catboost_info/` — CatBoost training logs
- `configs/` — tuned parameters and configuration files
- `models/` — trained model artifacts (not tracked)
- `notebooks/` — exploratory notebooks
- `scripts/` — helper scripts and CLI tools

Project-specific entry points:
- `src/main.py` or repository `main.py` — end-to-end training/inference
- `portfolio_runner.py` — cluster submissions, plan schedule, and update README

## Data flow diagram

```mermaid
flowchart LR
  A[Raw data CSVs\n(data/raw/*.csv)] --> B[Load & Preprocess\n(main.py)]
  B --> C[Feature Engineering\nfeatures.basic + features.advanced]
  C --> D[Cache Matrices\nutils/save_training_data.py\noutputs/predictions/*.pkl]
  C --> E[Models\nLGB / XGB / CatBoost / Two-Stage\nTime-based CV]
  E --> F[OOF + Test Predictions]
  F --> G[Ensemble\nstacking.py + blend_submissions.py]
  G --> H[Calibration\nisotonic / beta + gamma sweep]
  H --> I[Submission CSV\n(data/submissions/submission*.csv)]
  E --> J[Reports\nFeature importances -> outputs/reports]
  E --> K[CatBoost logs -> outputs/catboost_info]
```

## Cross-validation scheme (time-based)

- Chronological, month-based folds; no shuffling.
- For validation month t, we train on all data strictly before month t.
- Typical setup uses the last 5–6 months as validation folds.

Example (6 months):

```
Fold 1: Train [M1..M4] -> Validate M5
Fold 2: Train [M1..M5] -> Validate M6
...
```

This ensures leakage-safe evaluation and mimics the leaderboard month.

## Leakage checks

- All features are computed using dates strictly <= `ref_date` and per-fold training cutoffs.
- Trend windows and period filters assert end dates are before the cutoff (see `src/features/feature_engineering.py`).
- A utility `validate_no_future_leakage(df, ref_date)` is provided in `src/features/advanced_features.py`:
  - Detects date/period columns (e.g., `window_end`, `date`, `month`) and raises if any value is >= `ref_date`.
  - Use it after constructing any rolling-window or period-based dataframe.
- Outlier control for trend/ratio features: winsorization at [1, 99] percentiles is applied in advanced features to stabilize training without peeking into the future.

Minimal example:

```python
from src.features.advanced_features import validate_no_future_leakage
validate_no_future_leakage(recent_data, ref_date)
```

## What happens when you run `main.py`

- Loads input CSVs from `data/raw/` (falls back to `data/` or project root if needed).
- Trains LightGBM, XGBoost, CatBoost, and Two-Stage head with time-based CV.
- Blends models and calibrates predictions.
- Saves submission to `data/submissions/submission.csv`.
- Writes `data/submissions/last_update.txt` (timestamp + row count) each run.
- Exports a predictions bundle for portfolio tooling to `outputs/predictions/predictions_bundle.pkl`.
- Saves feature importance to `outputs/reports/feature_importance.csv`.
- Writes CatBoost logs under `outputs/catboost_info/`.

## How to reproduce

1) Create a virtual environment and install dependencies (Windows PowerShell):

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Place input CSVs under `data/raw/` (or `data/`/project root as fallback):
  - `customers.csv`
  - `customer_history.csv`
  - `reference_data.csv` and `reference_data_test.csv` (if applicable)

3) Run the end-to-end pipeline:

```powershell
python .\main.py
```

4) Optional extras:
  - Save cached matrices for faster iteration:

```powershell
python -m src.utils.save_training_data
```

  - Train a stacking meta-learner on base OOF predictions:

```powershell
python -m src.ensemble.stacking --help
```

  - Run segment-wise blending with non-negative L2 weights:

```powershell
python -m src.ensemble.blend_submissions --segments tenure_bin,avg_active_products_bin --l2 1e-3
```

  - Hyperparameter tuning with Optuna:

```powershell
python -m src.tuning.optuna_tuner --model lgb --trials 200 --last-n 6
python -m src.tuning.optuna_tuner --model xgb --trials 200 --last-n 6
```

5) Find outputs:
  - Submissions: `data/submissions/submission*.csv`
  - Prediction bundles and OOFs: `outputs/predictions/`
  - Reports (feature importance, blending weights): `outputs/reports/`
  - CatBoost logs: `outputs/catboost_info/`

## Setup structure script

A helper script is available to create the directories above and (optionally) move artifacts safely.

Dry-run (prints planned moves):

```powershell
python ./setup_structure.py
```

Apply moves:

```powershell
python ./setup_structure.py --apply
```

The script only moves common non-code artifacts and never overwrites existing files. If a target exists, a numeric suffix is added.

## Notes

- Large files and generated artifacts are ignored via `.gitignore`.
- Keep raw data outside version control. Place them under `data/raw/` locally.
- If you reorganize code within `src/`, ensure `__init__.py` files exist so imports continue to work.

## Portfolio tools

You can manage submissions and planning helpers via `portfolio_runner.py`:

```powershell
python .\portfolio_runner.py --help
```

Typical sub-commands wire up utilities in `portfolio_tools/` (e.g., clustering submissions, planning schedules, or updating README notes).