# ING Hubs Türkiye Datathon 2024 - Churn Prediction Solution

**Final Rank: 50th / 700 teams (Top 7%)**  
**Final Score: 1.23107**  
**Competition:** ING Hubs Türkiye Datathon - Customer Churn Prediction

## 📊 Competition Overview

### Objective
Predict customer churn for a banking dataset using a composite metric combining:
- **Gini coefficient** (40% weight)
- **Recall@10%** (30% weight)
- **Lift@10%** (30% weight)

### Evaluation Metric
```
Composite Score = 0.4 × (Gini/0.38515) + 0.3 × (Recall@10/0.18469) + 0.3 × (Lift@10/1.84715)
```

---

## 🏆 Solution Highlights

### Model Architecture
**5-Model Ensemble:**
1. **LightGBM** - Gradient boosting with Optuna-tuned hyperparameters
2. **XGBoost** - Extreme gradient boosting
3. **CatBoost** - Multi-seed (5 seeds) with rank averaging
4. **Two-Stage Model** - Custom model targeting Recall@10%
5. **Meta Stacker** - Level-2 stacker on base OOF predictions

### Key Innovations

#### 1. Two-Stage Top-Decile Optimization
Custom model designed to maximize Recall@10% and Lift@10% (60% of score):
- Stage A: Logistic regression targeting top 10%
- Stage B: Refinement layer
- **28.21%** ensemble weight

#### 2. Multi-Seed CatBoost with Rank Averaging
- 5 different random seeds
- Rank-transform and average predictions
- **38.84%** ensemble weight (highest)

#### 3. Advanced Feature Engineering (220+ features)
- **RFM Analysis:** Recency, Frequency, Monetary segmentation
- **Behavioral Features:** Transaction patterns, trend analysis
- **Lifecycle Features:** Customer tenure, engagement metrics
- **Time-Based Features:** Activity decay, inactivity periods
- **Feature Interactions:** Top-8 importance crossed

#### 4. Time-Based Cross-Validation
- Month-based folds with gap to prevent leakage
- Validates on last 6 months
- Month-wise weight aggregation

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/ing_datathon.git
cd ing_datathon
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
```

### Training
```bash
# Full pipeline
python -m src.main --models lgb xgb cat two meta --cat-seeds 5 --with-stacker

# Quick run
python -m src.main --models cat two --cat-seeds 3 --last-n 3
```

---

## 📈 Performance

### Results
- **Leaderboard:** 1.23107 (50th / 700)
- **Baseline:** 1.0000

### Ensemble Weights
```
CatBoost:     38.84%  (strongest)
Two-Stage:    28.21%  (top-decile specialist)
XGBoost:      11.73%
LightGBM:     10.97%
Meta:         10.25%
```

---

## 🗂️ Project Structure
```
ing_datathon/
├── src/
│   ├── main.py                    # Main pipeline
│   ├── models/                    # Model training
│   ├── features/                  # Feature engineering
│   ├── ensemble/                  # Blending & stacking
│   └── utils/                     # Calibration, tuning
├── portfolio_tools/               # Submission variants
├── scripts/                       # Utilities
└── outputs/                       # Results & logs
```

---

## 🎓 Key Learnings

### What Worked
✅ Two-Stage model explicitly optimizing Recall@10  
✅ Multi-seed CatBoost with rank averaging  
✅ RFM analysis for banking domain  
✅ Time-based cross-validation  
✅ Ensemble diversity

### What Didn't
❌ Extreme month weighting (>70% one month)  
❌ Too many correlated features  
❌ Over-aggressive adversarial filtering

---

## 📜 License
MIT License

---

**Final Rank: 50th / 700 teams (Top 7%)** 🏆
