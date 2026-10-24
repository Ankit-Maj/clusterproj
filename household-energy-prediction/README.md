# INT315 — Household Energy Consumption Prediction using Apache Spark MLlib

## Project Overview

Predict household appliance energy consumption (Wh) from environmental sensor readings using Apache Spark MLlib. This is an academic project demonstrating end-to-end ML pipeline design using distributed computing.

---

## Dataset Description

- **File:** `energydata_complete.csv` (located at `C:\Users\ASUS\Desktop\clusterproj\energydata_complete.csv`)
- **Size:** 19,735 rows, 29 columns, 10-minute interval readings
- **Date range:** January 11 – May 27, 2016
- **Target variable:** `Appliances` (Wh, appliance energy consumption)
- **Features:**
  - 9 indoor temperature sensors: `T1`–`T9`
  - 9 indoor humidity sensors: `RH_1`–`RH_9`
  - Outdoor weather: `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`
  - Lighting energy: `lights`
- **Excluded:** `rv1`, `rv2` (random variables, non-predictive)
- **Engineered features:** `hour`, `day_of_week`, `month`, `is_weekend` (derived from `date`)

---

## Project Structure

```
household-energy-prediction/
├── app.py                    # Streamlit dashboard (7 sections)
├── requirements.txt          # Pinned Python dependencies
├── README.md                 # This file
├── .gitignore
├── src/
│   ├── data_loader.py        # CSV loading and date parsing
│   ├── preprocessing.py      # Cleaning, feature engineering, splitting
│   ├── train.py              # LR and RF model training/saving/loading
│   ├── evaluate.py           # RMSE/MAE/R² computation, output persistence
│   ├── predict.py            # Single-record inference from both models
│   └── visualization.py      # Plotly chart generation functions
├── tests/
│   ├── test_data_loader.py   # Data loading tests
│   ├── test_preprocessing.py # Preprocessing and splitting tests
│   └── test_evaluation.py    # Metric correctness tests
├── data/                     # (dataset reference — original CSV not duplicated)
├── models/                   # Saved Spark ML models (generated at runtime)
└── outputs/                  # metrics.json and predictions.csv (generated at runtime)
```

---

## Prerequisites

- **Python 3.9 or later**
- **Java 8 or later** (required for Apache Spark)
  - Check: `java -version`
  - Windows: Download from https://adoptium.net/
  - Set the `JAVA_HOME` environment variable
- **PySpark 3.5.0** (installed via `requirements.txt`)
- **Windows note:** Spark 3.5 works in local mode without Hadoop on Windows for local execution. `HADOOP_HOME` is not required unless you encounter `winutils.exe` errors (see Troubleshooting).

---

## Installation

### 1. Set up a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Spark works

```bash
python -c "from pyspark.sql import SparkSession; s=SparkSession.builder.master('local[1]').getOrCreate(); print('Spark OK'); s.stop()"
```

---

## Usage

### Train the models

```bash
python src/train.py
```

This will:
1. Load `energydata_complete.csv`
2. Preprocess: drop `rv1`/`rv2`, deduplicate, add time features, chronological 80/20 split
3. Train Linear Regression (scaled features) and Random Forest (`numTrees=100`, `maxDepth=10`, `seed=42`)
4. Save models to `models/lr_model`, `models/rf_model`, and `models/scaler_model`

To force retraining (deletes existing saved models):

```bash
python src/train.py --retrain
```

### Evaluate the models

```bash
python src/evaluate.py
```

Computes RMSE, MAE, and R² on the chronological test set and writes:
- `outputs/metrics.json`
- `outputs/predictions.csv`

### Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501

### Run automated tests

```bash
pytest tests/ -v
```

---

## Module Descriptions

### `src/data_loader.py`

Loads `energydata_complete.csv` from its absolute path using PySpark. Parses `date` as `TimestampType`. Raises a descriptive `FileNotFoundError` if the file is missing. Does not drop any columns at load time — all 29 original columns are returned.

### `src/preprocessing.py`

Implements the full preprocessing pipeline: drops `rv1`/`rv2` (random variables with no predictive value), removes duplicate rows, drops any null rows, extracts `hour`/`day_of_week`/`month`/`is_weekend` from the timestamp, assembles a 29-feature vector using `VectorAssembler`, and applies a `StandardScaler` (for Linear Regression only). Uses a Window-based chronological 80/20 split: the first 80% of rows sorted by `date` form the training set; the remaining 20% form the test set. All pipeline stages are fit exclusively on the training set to prevent data leakage.

### `src/train.py`

Trains and persists two Spark MLlib models. `LinearRegression` is trained on scaled features; `RandomForestRegressor` (`numTrees=100`, `maxDepth=10`, `seed=42`) on unscaled features. Models are saved to `models/` using Spark's native model persistence. On subsequent runs, existing models are loaded automatically unless `--retrain` is passed. The fitted `StandardScaler` model is saved separately to `models/scaler_model` for use at inference time.

### `src/evaluate.py`

Computes RMSE, MAE, and R² on the chronological test set using `pyspark.ml.evaluation.RegressionEvaluator`. Saves all metrics to `outputs/metrics.json` and saves combined actual/predicted values to `outputs/predictions.csv`. Never evaluates on the training set.

### `src/predict.py`

Loads saved models and the saved scaler, then produces predictions for a single input record from both LR and RF models. Raises an informative `RuntimeError` if models are missing. Accepts a dictionary of feature values and applies the same preprocessing pipeline used during training.

### `src/visualization.py`

Provides Plotly chart functions used by the Streamlit dashboard: time-series of consumption, distribution histogram, actual vs. predicted scatter plots (with perfect-prediction line), residual distribution, predictions over time, error by consumption level, correlation heatmap, and feature distribution subplots.

---

## Preprocessing Decisions

| Decision | Rationale |
|---|---|
| Drop `rv1`, `rv2` | Random variables — correlation ≈ 0 with target, inflate feature dimensionality |
| Deduplicate rows | Prevent model from seeing identical observations multiple times |
| Drop null rows | Dataset has no nulls in practice; guard against schema changes |
| Chronological 80/20 split | Prevents temporal leakage: model never trains on future observations |
| StandardScaler for LR only | RF is scale-invariant; LR benefits from normalised features |
| Fit scaler on train only | Prevents test-set statistics from influencing training |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error (Wh) — penalises large errors more heavily |
| MAE | Mean Absolute Error (Wh) — average absolute prediction error |
| R² | Coefficient of determination — proportion of variance explained (1.0 = perfect) |

---

## Troubleshooting

### Java not found

```
JAVA_HOME is not set
```

Install Java 8+ and set `JAVA_HOME`. On Windows:

```bash
set JAVA_HOME=C:\Program Files\Java\jdk-11
set PATH=%JAVA_HOME%\bin;%PATH%
```

### Spark fails to start on Windows

Some Windows machines require `winutils.exe` for Hadoop. For local-mode-only use:

```bash
set HADOOP_HOME=C:\hadoop
# Create C:\hadoop\bin\winutils.exe (download from https://github.com/steveloughran/winutils)
```

### PySpark version mismatch

Ensure `pyspark==3.5.0` is installed:

```bash
pip show pyspark
```

### Dataset not found

Ensure `energydata_complete.csv` is at `C:\Users\ASUS\Desktop\clusterproj\energydata_complete.csv`. The path is configured in `src/data_loader.py`.

### Models not found when running app

Run `python src/train.py` first to generate the saved models. Then run `python src/evaluate.py` to generate `outputs/metrics.json` and `outputs/predictions.csv`.

---

## Limitations

- Dataset covers only one household over ~4.5 months (Jan–May 2016)
- No cross-validation or hyperparameter tuning was performed
- Models may not generalise to different households, climates, or seasons
- `rv1`/`rv2` excluded as they are random variables with no physical meaning
- Linear Regression assumes linear relationships which may not capture all patterns

---

## Reproducibility Notes

- Random Forest uses `seed=42` for deterministic tree construction
- Chronological split is deterministic (based on sorted timestamps, not random)
- All model artifacts are saved to `models/` and reloaded on subsequent runs
- `outputs/` and `models/` are gitignored — regenerate by running `train.py` then `evaluate.py`
