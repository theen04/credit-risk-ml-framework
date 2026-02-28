# Credit Risk Machine Learning Framework

A credit risk ML framework built on real Lending Club data. The banking_ml package provides reusable preprocessing, evaluation, and modeling utilities — demonstrated across three notebooks covering default classification, risk-based pricing, and neural network scoring.

---

## Project Structure
```
banking-ml/
│
├── LICENSE
├── README.md
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── setup.py
├── pytest.ini
│
├── data/
│   ├── raw/                         # Source datasets (not tracked in git)
│   └── processed/                   # Intermediate processed data
│
├── artifacts/
│   ├── classification/
│   │   ├── eda/
│   │   └── models/
│   ├── pricing/
│   │   ├── eda/
│   │   └── models/
│   └── neural_network/
│       ├── eda/
│       └── models/
│
├── notebooks/
│   ├── 01_credit_risk_classification.ipynb
│   ├── 02_risk_based_pricing.ipynb
│   └── 03_neural_risk_scoring.ipynb
│
├── src/
│   └── banking_ml/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── feature_engineering.py
│       ├── pipelines.py
│       ├── models.py
│       ├── evaluation.py
│       ├── explainability.py
│       ├── visualization.py
│       └── utils.py
│
└── tests/
    ├── conftest.py
    ├── test_preprocessing.py
    ├── test_feature_engineering.py
    ├── test_pipelines.py
    ├── test_models.py
    └── test_explainability.py
```

---

## The Trilogy

All three notebooks share the same Lending Club dataset and the same `banking_ml` package, creating a coherent analytical narrative across three problems.

| Notebook | Problem | Target | Best Model | AUC / R² |
|---|---|---|---|---|
| 01 | Credit Risk Classification | Default (binary) | XGBoost (Tuned) | AUC 0.7366 |
| 02 | Risk-Based Pricing | Interest Rate (continuous) | XGBoost (Tuned) | R² 0.4838 |
| 03 | Neural Risk Scoring | Default (binary) | XGBoost Benchmark | AUC 0.7325 |

The progression is deliberate — establish what default looks like, price for it, then test whether a neural network can find signal that XGBoost missed.

### The Cross-Notebook Finding

The most interesting result of the trilogy isn't any single model's performance — it's what the comparison reveals about feature selection and architecture:

- **XGBoost performs best with 33 curated features** — expanding to 92 features reduces AUC by 0.0105. Deliberate feature selection is a meaningful modeling decision for tree-based architectures.
- **Neural networks perform best with all 92 features** — reducing to 33 costs 0.0024 AUC. Neural networks learn to suppress irrelevant features internally and benefit from broader input.
- **The two architectures have fundamentally different relationships with feature selection** — and that finding is more interesting than the AUC gap between them.

---

## The `banking_ml` Package

The shared library contains financially-tailored ML utilities reused across all three notebooks.

**`utils.py`** — Data loading and inspection
- `load_data` — Compressed dataset loader with dtype handling
- `inspect_dataframe` — Structured summary of shape, dtypes, missingness
- `save_data_summary_tables` — Persist EDA snapshots to CSV
- `filter_by_vintage` — Filter loans by origination year range
- `report_skewness` — Numeric feature skewness assessment
- `assess_features` — Redundancy, variance, and correlation reporting
- `clean_raw_data` — Remove % symbols and convert rate columns to float

**`preprocessing.py`** — Financial data preprocessing
- `drop_high_missingness` — Configurable null threshold filtering
- `inspect_target` — Target variable distribution analysis
- `encode_binary_target` — Multi-class to binary encoding with ambiguous class removal
- `drop_leakage_columns` — Post-origination and administrative column removal
- `split_data` — Stratified train/test splitting
- `select_core_features` — 33-feature curated set for classification
- `select_core_features_regression` — 29-feature set for pricing (leakage-aware)
- `select_core_features_neural` — Full 92-feature set for neural network

**`feature_engineering.py`** — Financial feature engineering
- `engineer_features` — Domain-specific feature construction (14 engineered features)

**`pipelines.py`** — Sklearn-compatible pipeline construction
- `build_preprocessor` — ColumnTransformer with numeric, ordinal, and nominal branches
- `build_pipeline` — Full preprocessing + model pipeline

**`evaluation.py`** — Standard and financial evaluation metrics
- `evaluate_classifier` — AUC, Gini coefficient, KS statistic, Brier score
- `evaluate_regressor` — RMSE, MAE, R², Max Error
- `compare_classifiers` — Multi-model classification summary table
- `compare_regressors` — Multi-model regression summary table

**`visualization.py`** — Reusable EDA and model evaluation plots
- `plot_class_distribution`
- `plot_numeric_distributions`
- `plot_default_rate_by_category` — Handles binary and continuous targets
- `plot_correlation_heatmap`
- `plot_missingness`
- `plot_feature_target_correlation` — Point-biserial or Pearson depending on target
- `plot_roc_curves`
- `plot_roc_curves_from_probs` — ROC comparison from precomputed probability arrays
- `plot_target_distribution` — Continuous target histogram with mean/median markers
- `plot_residuals` — Predicted vs residuals scatter
- `plot_predicted_vs_actual` — Predicted vs actual with reference line
- `plot_training_history` — Neural network training and validation loss curves

**`explainability.py`** — SHAP-based model interpretation
- `get_shap_values` — Extracts SHAP values from fitted sklearn pipelines (classifiers and regressors)
- `plot_shap_summary` — Mean absolute feature importance bar chart
- `plot_shap_beeswarm` — Feature impact direction and magnitude

**`models.py`** — Model definitions and training utilities
- `tune_model` — RandomizedSearchCV wrapper with sample weight support
- `get_xgb_param_grid` — XGBoost classification hyperparameter search space
- `get_xgb_regression_param_grid` — XGBoost regression hyperparameter search space
- `display_best_params` — Formatted best parameter display
- `NeuralCreditScorer` — PyTorch feedforward network for binary credit classification
- `NeuralClassifierWrapper` — Sklearn-compatible wrapper for PyTorch models
- `train_neural_model` — PyTorch training loop with BCEWithLogitsLoss, class weighting, and early stopping

---

## Testing

The package ships with a pytest test suite covering all five core modules.
```bash
# Run full suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/banking_ml --cov-report=term-missing

# Run a single module
pytest tests/test_preprocessing.py -v

# Skip slower neural network training tests
pytest tests/ -v -k "not train_neural"
```

**Coverage:**

| Module | Tests | What's Covered |
|---|---|---|
| `preprocessing.py` | 28 | missingness filtering, target encoding, leakage removal, feature selection, train/test split |
| `feature_engineering.py` | 9 | covered via integration through preprocessing fixtures |
| `evaluation.py` | 11 | classifier and regressor metrics, model comparison tables |
| `models.py` | 13 | forward pass shapes, wrapper interface, training loop, early stopping, reproducibility |
| `explainability.py` | 4 | SHAP value extraction, output shapes, feature name types |

Tests use synthetic DataFrames — no dependency on the raw dataset. All 59 tests
pass on Python 3.11 with the dependencies pinned in `environment.yml`.

---

## Notebook 1: Credit Risk Classification

**Problem:** Predict whether a resolved loan will default  
**Target:** `loan_status` → binary (Charged Off / Default = 1, Fully Paid = 0)  
**Vintage:** 2017–2019 (1.86M resolved loans)  
**Features:** 33 curated features after engineering and selection

### Key Decisions

**Grade exclusion** — Lending Club's proprietary `grade` and `sub_grade` features are excluded. A lender building their own risk model wouldn't have access to a competitor's internal scoring. Removing them costs less than 0.001 AUC — the signal is recoverable from raw borrower fundamentals.

**Two-pass missingness filtering** — A first pass at 50% removes structurally empty columns. A second pass at 40% after target encoding reveals additional missingness only visible once unresolved loans are excluded.

**Leakage removal** — Post-origination columns (payment history, recovery amounts, last FICO pulls) are explicitly removed. These features are only observable after a loan outcome is known.

### Results

| Model | ROC-AUC | Gini | KS Statistic | Brier Score |
|---|---|---|---|---|
| Logistic Regression | 0.7141 | 0.4282 | 0.3088 | 0.2161 |
| Gradient Boosting | 0.7239 | 0.4478 | 0.3250 | 0.2132 |
| XGBoost | 0.7239 | 0.4477 | 0.3236 | 0.2133 |
| XGBoost (Tuned) | 0.7366 | 0.4732 | 0.3430 | 0.2054 |

### Key Finding

Removing grade and sub_grade costs less than 0.001 AUC. FICO score, rate_to_dti, and payment_to_income — all available at origination — carry enough signal to match the performance of a model that relies on Lending Club's internal grading.

---

## Notebook 2: Risk-Based Pricing

**Problem:** Predict the interest rate assigned to a loan at origination  
**Target:** `int_rate` (continuous, 6–31%)  
**Vintage:** 2017–2019 (same as notebook 1)  
**Features:** 29 curated features (rate-derived features excluded as leakage)

### Key Decisions

**Leakage-aware feature selection** — `payment_to_income` and `rate_to_dti` are excluded from the regression feature set since both are mathematically derived from `int_rate`. A single missingness pass is sufficient — no binary encoding step changes the missingness profile.

**Discrete rate tiers** — Lending Club assigned rates in discrete tiers rather than as a continuous function. This creates a structural ceiling on R² for any regression model — the predicted vs actual plot shows characteristic vertical striping at each tier boundary.

### Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 3.9494 | 2.9971 | 0.4014 |
| Gradient Boosting | 3.7604 | 2.8176 | 0.4573 |
| XGBoost | 3.7584 | 2.8161 | 0.4579 |
| XGBoost (Tuned) | 3.6590 | 2.7351 | 0.4862 |


### Key Finding

The model explains 48% of variance in interest rates without proprietary grades. FICO score is the strongest pricing factor. The remaining unexplained variance is largely structural — discrete rate tiers create a ceiling that no regression model can fully overcome without access to the internal tier-assignment formula.

---

## Notebook 3: Neural Risk Scoring

**Problem:** Test whether a neural network can find nonlinear signal that XGBoost missed  
**Target:** `loan_status` → binary (same as notebook 1)  
**Vintage:** 2017–2019 (same as notebooks 1 and 2)  
**Features:** 92 features (full set — no reduction applied)

### Architecture
```
Input (160) → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
            → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
            → Linear(64)  → BatchNorm → ReLU → Dropout(0.2)
            → Linear(1)
```

BCEWithLogitsLoss with positive class weighting for 80/20 class imbalance.  
Early stopping with patience of 10 epochs. Adam optimizer at lr=1e-4.

### Results

| Model | ROC-AUC | Gini | KS Statistic | Brier Score |
|---|---|---|---|---|
| XGBoost (Benchmark) | 0.7366 | 0.4732 | 0.3430 | 0.2054 |
| Neural Network | 0.7325 | 0.4651 | 0.3371 | 0.2096 |


### Key Findings

**The neural network doesn't beat XGBoost** — but the investigation reveals something more interesting. The two architectures have fundamentally different relationships with feature selection:

- XGBoost: 33 features → AUC 0.7362 | 90 features → AUC 0.7257 (-0.0105)
- Neural Network: 33 features → AUC 0.7302 | 90 features → AUC 0.7325 (+0.0023)

**The cross-notebook SHAP comparison** reveals that term and loan amount are universal credit risk signals — equally important for default prediction and rate pricing. Beyond those, the two problems diverge: default prediction is driven by credit history and behavioral features, rate pricing by debt burden and recent credit activity.

---

## Data Source

**Dataset:** Lending Club 2007–2020Q3  
**Author:** Ethon0426  
**URL:** https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1  
**Vintage used:** 2017–2019 (post-crisis, pre-COVID resolved loans)

Original data published by LendingClub Corporation.  
All analysis uses public loan-level data for educational purposes only.

---

## Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/credit-risk-ml-framework.git
cd credit-risk-ml-framework

# Create environment
conda env create -f environment.yml
conda activate fintech-applied-ml

# Install core
pip install -e .

# Install dev dependencies
pip install -e .[dev]

# Install test dependencies
pip install -e .[test]

# Install neural extras
pip install -e .[neural]

# Or both dev + neural
pip install -e .[dev,neural]
```

## Data

Download the Lending Club dataset from Kaggle before running any notebooks.

**Option 1 — Manual download (recommended)**  
1. Go to https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1
2. Click Download
3. Place `Loan_status_2007-2020Q3.gzip` in `data/raw/`

**Option 2 — Kaggle API**  
Requires a Kaggle account and API token configured at `~/.kaggle/kaggle.json`.
```python
import kagglehub

path = kagglehub.dataset_download("ethon0426/lending-club-20072020q1")
print("Path to dataset files:", path)
```

Then move the downloaded file to `data/raw/`.

The dataset is not tracked in this repository. All notebooks expect the file at:
```
data/raw/Loan_status_2007-2020Q3.gzip
```
---

## Dependencies

Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `torch`, `matplotlib`, `seaborn`, `shap`, `joblib`

See `requirements.txt` or `environment.yml` for the full pinned dependency list.

---