# House Price Prediction — California Housing

End-to-end regression project predicting median house values for California census block groups, covering EDA, preprocessing, model comparison, and hyperparameter tuning.

## Overview

This project predicts `MedHouseVal` (median house value) using 8 features describing a California census block group — income, housing age, room/bedroom averages, population, occupancy, and geographic coordinates.

**Champion model:** Random Forest Regressor (`n_estimators=300`, `max_features='sqrt'`)

| Metric | Score |
|---|---|
| RMSE | 0.4744 |
| MAE | 0.3136 |
| R² | 0.8293 |

## Dataset

**Source:** [Kaggle Learn — "Clustering with K-Means" notebook by Ryan Holbrook](https://www.kaggle.com/code/ryanholbrook/clustering-with-k-means/data?select=housing.csv), manually downloaded (not via Kaggle API).

This is the **sklearn-derived version** of California Housing (`fetch_california_housing`), not the classic camnugent Kaggle dataset — it has no missing values and no categorical columns, but does carry two known capping/censoring quirks (see Limitations below).

| Column | Description |
|---|---|
| `MedInc` | Median income (tens of thousands of $) |
| `HouseAge` | Median house age (years) |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` / `Longitude` | Geographic coordinates |
| `MedHouseVal` | **Target** — median house value (hundreds of thousands of $) |

## Project Structure

```
house-price-prediction/
├── 01_eda.ipynb                  # Exploratory data analysis
├── 02_data_preprocessing.ipynb   # Cleaning, feature engineering, train/test split
├── 03_model_training.ipynb       # 6-model comparison (Linear/Ridge/Lasso/KNN/RF/GB)
├── 04_model_tuning.ipynb         # 8 Random Forest tuning variants
├── train_final_model.py          # Standalone end-to-end pipeline script
├── housing.csv                   # Raw dataset
├── requirements.txt
├── outputs/                      # Saved plots, metrics, model artifacts (model .pkl files gitignored — see Usage)
└── README.md
```

## Usage

### 1. Clone the repo
```powershell
git clone git@github.com:HanosThePlacid/house-price-prediction.git
cd house-price-prediction
```

### 2. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
If PowerShell blocks script execution:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the project

**Option A — step through the notebooks** (recommended for following the full analysis):
```powershell
jupyter notebook
```
Run in order: `01_eda.ipynb` → `02_data_preprocessing.ipynb` → `03_model_training.ipynb` → `04_model_tuning.ipynb`

**Option B — run the full pipeline as a script** (fastest way to reproduce the final model):
```powershell
python train_final_model.py
```
This loads `housing.csv`, applies all cleaning/feature-engineering steps, trains the champion Random Forest, evaluates it, and saves the model plus train/test artifacts to `outputs/`.

> Note: large `.pkl` model artifacts in `outputs/` are excluded from version control (see `.gitignore`) since they're fully regenerable by running the script or notebooks above.

## Key Findings & Decisions

**Outlier removal** — `AveOccup > 10` and `AveRooms > 20` rows (105 total, ~0.5%) were dropped after investigation showed they were calculation artifacts from block groups with very small implied household counts (e.g. institutional/group housing), not genuine residential data.

**Feature engineering:**
- `bedroom_ratio = AveBedrms / AveRooms` replaces `AveBedrms` (was 0.85-correlated with `AveRooms`) — preserves both a size signal (`AveRooms`) and a layout signal (the ratio) without redundancy.
- `is_max_house_age` flags the censored `HouseAge` value (52) so the model can treat "52-or-older, unknown" differently from a literal age of exactly 52.
- `log_MedInc` / `log_Population` were added as extra columns (not replacements) for linear models; unused by the tree-based champion.

**Model comparison** (6 models, default hyperparameters):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | 0.4896 | 0.3193 | 0.8182 |
| Gradient Boosting | 0.5160 | 0.3561 | 0.7980 |
| KNN | 0.5969 | 0.4057 | 0.7297 |
| Linear Regression | 0.6805 | 0.5094 | 0.6488 |
| Ridge | 0.6805 | 0.5094 | 0.6488 |
| Lasso | 0.7778 | 0.5909 | 0.5411 |

**Hyperparameter tuning** — 8 Random Forest variants were cross-validated (5-fold, on training data only, to avoid overfitting to the test set via repeated model selection). The winner (`max_features='sqrt'`, `n_estimators=300`) outperformed even a 30-iteration `RandomizedSearchCV` search — likely because 30 samples covers only ~1% of the ~2,880-combination search space. Improvement over the untuned baseline was modest (RMSE 0.4896 → 0.4744), suggesting Random Forest is fairly robust to these hyperparameters on this dataset.

## Known Limitations

`MedHouseVal` is capped at 5.00001 ($500,001) for 4.81% of rows — a known property of the original census-derived data, not a data quality issue. This cap was kept (rather than dropped) since removing it would make the model blind to high-value areas entirely. The consequence is measurable: the final model's RMSE on capped rows is **1.0339**, roughly **2.4x worse** than its RMSE of **0.4289** on non-capped rows. **In practice, this model will systematically underpredict prices for genuinely high-value properties ($500k+).**

`HouseAge` is similarly capped at 52 (6.17% of rows) — handled via the `is_max_house_age` flag rather than dropping or ignoring it.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter

