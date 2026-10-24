# Design Document — Household Energy Consumption Prediction

## Overview

The system is a PySpark-based machine learning pipeline that predicts household appliance energy consumption (`Appliances`, Wh) from a set of indoor/outdoor environmental sensor readings. Two Spark MLlib regression models — Linear Regression and Random Forest — are trained on a chronological 80/20 split of the dataset and evaluated using standard regression metrics. An interactive Streamlit dashboard presents the full pipeline — from raw data exploration through model comparison and interactive single-record prediction.

All ML training and evaluation is performed exclusively via the Spark ML Pipeline API (`pyspark.ml`). No scikit-learn, XGBoost, or TensorFlow components are used.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit)                     │
│   Section A │ B │ C │ D │ E │ F │ G                          │
└──────┬───────┴──┬┴──┬┴──┬┴──┬┴──┬┘                          │
       │          │   │   │   │   │                              │
       ▼          ▼   ▼   ▼   ▼   ▼                              │
  DataLoader  Preprocessor  Trainer  Evaluator  Predictor  VisualizationEngine
       │          │               │        │
       ▼          ▼               ▼        ▼
  Spark DataFrame  Pipeline  models/  outputs/
                   Stages
```

### Module Responsibilities

| Module | File | Responsibility |
|---|---|---|
| DataLoader | `src/data_loader.py` | Load CSV, parse `date`, return raw Spark DataFrame |
| Preprocessor | `src/preprocessing.py` | Drop junk, extract time features, assemble FeatureVector, scale, split |
| Trainer | `src/train.py` | Train/save/load LR and RF models |
| Evaluator | `src/evaluate.py` | Compute RMSE/MAE/R², persist metrics and predictions |
| Predictor | `src/predict.py` | Load models, preprocess single record, return predictions |
| VisualizationEngine | `src/visualization.py` | Return Plotly figure objects for dashboard |
| Dashboard | `app.py` | Streamlit app with seven named sections |

---

## Component Design

### DataLoader (`src/data_loader.py`)

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType
from pyspark.sql import functions as F
import os

DATA_PATH = r"C:\Users\ASUS\Desktop\clusterproj\energydata_complete.csv"

def get_data_path() -> str:
    """Return absolute path to dataset. Raises FileNotFoundError if absent."""
    path = os.path.abspath(DATA_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found at resolved path: {path}")
    return path

def load_data(spark: SparkSession) -> "DataFrame":
    path = get_data_path()
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn("date", F.to_timestamp(F.col("date"), "yyyy-MM-dd HH:mm:ss"))
    return df
```

**Key decisions:**
- Path is defined as a module-level constant (absolute Windows path matching the project environment).
- `inferSchema=True` avoids manual schema specification while retaining correct numeric types.
- `date` is cast to `TimestampType` immediately after load.
- No column is dropped at load time; dropping `rv1`/`rv2` is the Preprocessor's responsibility.

---

### Preprocessor (`src/preprocessing.py`)

**Pipeline for Linear Regression:**
```
DataFrame → drop(rv1, rv2) → deduplicate → drop nulls
         → add time features (hour, day_of_week, month, is_weekend)
         → sort by date → chronological split (80/20)
         → VectorAssembler (fit on train) → StandardScaler (fit on train)
         → (train_df_scaled, test_df_scaled)
```

**Pipeline for Random Forest:**
```
Same as above but StandardScaler step is omitted.
→ (train_df, test_df) with raw feature vector
```

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import functions as F

EXCLUDE_COLS = {"date", "Appliances", "rv1", "rv2"}

FEATURE_COLS = [
    "lights",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9",
    "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
    "hour", "day_of_week", "month", "is_weekend",
]

def clean(df):
    df = df.drop("rv1", "rv2")
    df = df.dropDuplicates()
    df = df.dropna()
    return df

def add_time_features(df):
    df = df.withColumn("hour", F.hour("date"))
    df = df.withColumn("day_of_week", F.dayofweek("date") - 1)  # 0=Sun..6=Sat
    df = df.withColumn("month", F.month("date"))
    df = df.withColumn("is_weekend",
        F.when(F.dayofweek("date").isin([1, 7]), 1).otherwise(0))
    return df

def chronological_split(df, train_ratio=0.8):
    df_sorted = df.orderBy("date")
    n = df_sorted.count()
    train_n = int(n * train_ratio)
    # Use monotonically_increasing_id after sort to index rows
    from pyspark.sql.window import Window
    w = Window.orderBy("date")
    df_indexed = df_sorted.withColumn("_row_id", F.row_number().over(w))
    train_df = df_indexed.filter(F.col("_row_id") <= train_n).drop("_row_id")
    test_df  = df_indexed.filter(F.col("_row_id") >  train_n).drop("_row_id")
    return train_df, test_df

def build_lr_pipeline(train_df):
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True)
    train_assembled = assembler.transform(train_df)
    scaler_model = scaler.fit(train_assembled)
    return assembler, scaler_model

def build_rf_pipeline(train_df):
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    assembler.transform(train_df)  # validate
    return assembler

def preprocess(df):
    df = clean(df)
    df = add_time_features(df)
    train_df, test_df = chronological_split(df)
    assembler_lr, scaler_model = build_lr_pipeline(train_df)
    assembler_rf = build_rf_pipeline(train_df)
    # LR data
    train_lr = scaler_model.transform(assembler_lr.transform(train_df))
    test_lr  = scaler_model.transform(assembler_lr.transform(test_df))
    # RF data
    train_rf = assembler_rf.transform(train_df)
    test_rf  = assembler_rf.transform(test_df)
    return {
        "train_lr": train_lr, "test_lr": test_lr,
        "train_rf": train_rf, "test_rf": test_rf,
        "assembler_lr": assembler_lr, "scaler_model": scaler_model,
        "assembler_rf": assembler_rf,
    }
```

---

### Trainer (`src/train.py`)

```python
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel
import os, shutil

LR_MODEL_PATH  = os.path.abspath("models/lr_model")
RF_MODEL_PATH  = os.path.abspath("models/rf_model")

def train_lr(train_df, retrain=False):
    if os.path.exists(LR_MODEL_PATH) and not retrain:
        return LinearRegressionModel.load(LR_MODEL_PATH)
    if os.path.exists(LR_MODEL_PATH):
        shutil.rmtree(LR_MODEL_PATH)
    lr = LinearRegression(featuresCol="features", labelCol="Appliances",
                          maxIter=100, regParam=0.1, elasticNetParam=0.0)
    model = lr.fit(train_df)
    model.save(LR_MODEL_PATH)
    return model

def train_rf(train_df, retrain=False):
    if os.path.exists(RF_MODEL_PATH) and not retrain:
        return RandomForestRegressionModel.load(RF_MODEL_PATH)
    if os.path.exists(RF_MODEL_PATH):
        shutil.rmtree(RF_MODEL_PATH)
    rf = RandomForestRegressor(featuresCol="features", labelCol="Appliances",
                               numTrees=100, maxDepth=10, seed=42)
    model = rf.fit(train_df)
    model.save(RF_MODEL_PATH)
    return model
```

**CLI entrypoint:**
```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    # ... initialise Spark, load data, preprocess, train
```

---

### Evaluator (`src/evaluate.py`)

```python
from pyspark.ml.evaluation import RegressionEvaluator
import json, os

OUTPUTS_DIR = os.path.abspath("outputs")

def evaluate_model(model, test_df, model_name):
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(labelCol="Appliances", predictionCol="prediction")
    rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    mae  = evaluator.setMetricName("mae").evaluate(predictions)
    r2   = evaluator.setMetricName("r2").evaluate(predictions)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}, predictions

def save_metrics(metrics_dict):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

def save_predictions(lr_preds, rf_preds):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    lr_pd = lr_preds.select("date", "Appliances", "prediction") \
                    .toPandas().rename(columns={"prediction": "LR_prediction"})
    rf_pd = rf_preds.select("date", "Appliances", "prediction") \
                    .toPandas().rename(columns={"prediction": "RF_prediction"})
    merged = lr_pd.merge(rf_pd, on=["date", "Appliances"])
    merged.to_csv(os.path.join(OUTPUTS_DIR, "predictions.csv"), index=False)
```

---

### Predictor (`src/predict.py`)

```python
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel
import os

def predict(spark: SparkSession, feature_dict: dict) -> dict:
    lr_path = os.path.abspath("models/lr_model")
    rf_path = os.path.abspath("models/rf_model")
    if not os.path.exists(lr_path) or not os.path.exists(rf_path):
        raise RuntimeError(
            "Trained models not found in models/. Run python src/train.py first.")
    lr_model = LinearRegressionModel.load(lr_path)
    rf_model = RandomForestRegressionModel.load(rf_path)
    row_df = spark.createDataFrame([feature_dict])
    # Apply preprocessing pipeline (assembler + scaler loaded from artefacts)
    from src.preprocessing import FEATURE_COLS
    from pyspark.ml.feature import VectorAssembler
    assembler_lr = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
    # StandardScaler model must be loaded (saved alongside LR model as pipeline)
    lr_pred = lr_model.transform(assembler_lr.transform(row_df)) \
                      .select("prediction").collect()[0][0]
    assembler_rf = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    rf_pred = rf_model.transform(assembler_rf.transform(row_df)) \
                      .select("prediction").collect()[0][0]
    return {"LinearRegression": lr_pred, "RandomForest": rf_pred}
```

> **Implementation note:** For production use, the fitted `StandardScaler` model should be serialised alongside the LR model (e.g., saved as part of a `Pipeline` model) so that the same scaling parameters are applied at prediction time. The design records this as the recommended approach for the implementation task.

---

### VisualizationEngine (`src/visualization.py`)

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_timeseries(df_pd: pd.DataFrame):
    """Time-series line chart of Appliances over date."""
    return px.line(df_pd, x="date", y="Appliances",
                   title="Appliance Energy Consumption Over Time")

def plot_metrics_comparison(metrics: dict):
    """Bar chart comparing RMSE, MAE, R2 across models."""
    rows = [{"Model": m, "Metric": k, "Value": v}
            for m, vals in metrics.items() for k, v in vals.items()]
    df = pd.DataFrame(rows)
    return px.bar(df, x="Metric", y="Value", color="Model", barmode="group",
                  title="Model Performance Comparison")

def plot_actual_vs_predicted(preds_pd: pd.DataFrame, model_name: str):
    """Scatter plot of actual vs. predicted Appliances."""
    return px.scatter(preds_pd, x="Appliances", y="prediction",
                      title=f"{model_name}: Actual vs. Predicted",
                      labels={"Appliances": "Actual (Wh)", "prediction": "Predicted (Wh)"})

def plot_correlation_heatmap(df_pd: pd.DataFrame):
    """Correlation heatmap of numeric features."""
    corr = df_pd.corr(numeric_only=True)
    return go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu", zmid=0))

def plot_distribution(df_pd: pd.DataFrame):
    """Histogram of Appliances distribution."""
    return px.histogram(df_pd, x="Appliances",
                        title="Distribution of Appliance Energy Consumption (Wh)")
```

---

### Dashboard (`app.py`)

Sections are rendered in a sidebar-navigated layout using `st.sidebar.radio`.

```python
import streamlit as st
from pyspark.sql import SparkSession

@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("HouseholdEnergyPrediction") \
        .getOrCreate()
```

Each section is wrapped in a `try/except` block; `st.error(str(e))` is called on any exception.

| Section | Content |
|---|---|
| A – Project Overview | Dataset summary, objective, technology stack description |
| B – Data Explorer | Schema, row count, `describe()`, time-series + distribution charts |
| C – Data Preparation | Preprocessing steps description, train/test split sizes |
| D – Model Comparison | Metrics table (`st.dataframe`), metrics bar chart |
| E – Prediction Analysis | Actual vs. predicted scatter plots for LR and RF |
| F – Interactive Prediction | `st.number_input` for each feature, predict button, display both model results |
| G – Findings and Limitations | Static markdown: model performance notes, limitations |

---

## Data Flow

```
energydata_complete.csv
        │
        ▼
   DataLoader.load_data()          → raw Spark DataFrame (29 cols)
        │
        ▼
   Preprocessor.clean()            → drop rv1/rv2, dedup, drop nulls
        │
   Preprocessor.add_time_features() → +hour, +day_of_week, +month, +is_weekend
        │
   Preprocessor.chronological_split() → train_df (80%), test_df (20%)
        │
   Preprocessor.build_lr_pipeline()   → assembler_lr, scaler_model (fit on train)
   Preprocessor.build_rf_pipeline()   → assembler_rf (no scaler)
        │
   ┌────┴────┐
   ▼         ▼
train_lr   train_rf
test_lr    test_rf
   │         │
   ▼         ▼
Trainer.train_lr()   Trainer.train_rf()
   │         │
   ▼         ▼
lr_model   rf_model  → saved to models/
   │         │
   ▼         ▼
Evaluator.evaluate_model()
   │
   ▼
metrics.json + predictions.csv → saved to outputs/
```

---

## Directory Structure

```
c:\Users\ASUS\Desktop\clusterproj\household-energy-prediction\
  app.py                      # Streamlit dashboard
  requirements.txt            # Pinned dependencies
  README.md                   # Project documentation
  .gitignore
  src/
    __init__.py
    data_loader.py
    preprocessing.py
    train.py
    evaluate.py
    predict.py
    visualization.py
  tests/
    __init__.py
    test_data_loader.py
    test_preprocessing.py
    test_evaluation.py
  data/                       # Reference to dataset (no copy)
  models/                     # Runtime-generated (gitignored)
  outputs/                    # Runtime-generated (gitignored)
```

---

## Technology Stack

| Library | Version | Role |
|---|---|---|
| pyspark | 3.5.0 | Spark engine, ML Pipeline, DataFrame API |
| streamlit | latest | Dashboard UI |
| pandas | latest | Helper DataFrames for Plotly, CSV output |
| numpy | latest | Numeric utilities |
| plotly | latest | Charts |
| pytest | latest | Test runner |

---

## Error Handling Strategy

| Scenario | Handler |
|---|---|
| CSV not found | `FileNotFoundError` raised in `DataLoader` with resolved path |
| Models not found at predict time | `RuntimeError` raised in `Predictor` |
| `outputs/` missing | `os.makedirs(exist_ok=True)` in Evaluator |
| Any section failure in Dashboard | `st.error(str(e))` wrapping `try/except` |
| Empty DataFrame after cleaning | Log warning; downstream code handles zero-row DataFrames gracefully |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

---

### Property 1: FeatureVector excludes forbidden columns

For any DataFrame that passes through the Preprocessor, the assembled FeatureVector SHALL NOT contain columns `rv1`, `rv2`, `Appliances`, or `date`, and SHALL contain all of: `lights`, `T1`–`T9`, `RH_1`–`RH_9`, `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`, `hour`, `day_of_week`, `month`, and `is_weekend`.

**Validates: Requirements 3.1, 3.5, 3.6**

---

### Property 2: Deduplication removes all duplicate rows

For any DataFrame with one or more duplicate rows, after applying the `clean()` function, the resulting DataFrame SHALL contain no duplicate rows (i.e., `df.count() == df.dropDuplicates().count()`).

**Validates: Requirements 3.2**

---

### Property 3: Null-dropping produces a null-free DataFrame

For any DataFrame that contains null values in any column, after applying the `clean()` function, the resulting DataFrame SHALL contain zero null values across all columns.

**Validates: Requirements 3.3**

---

### Property 4: Time feature range invariants

For any Spark DataFrame row produced by `add_time_features()`, the derived columns SHALL satisfy: `hour` ∈ [0, 23], `day_of_week` ∈ [0, 6], `month` ∈ [1, 12], and `is_weekend` ∈ {0, 1}.

**Validates: Requirements 3.4**

---

### Property 5: Chronological split preserves temporal ordering

For any DataFrame sorted ascending by `date`, after applying `chronological_split()`, the maximum `date` value in the training set SHALL be strictly less than or equal to the minimum `date` value in the test set — guaranteeing no temporal leakage from future observations into the training set.

**Validates: Requirements 3.7**

---

### Property 6: No data leakage — scaler fit on training set only

For any training/test split, the `StandardScaler` model's mean and standard deviation statistics SHALL equal the statistics computed from the training set alone. The scaler SHALL be fitted by calling `scaler.fit(train_df)` and only then used to transform both `train_df` and `test_df`.

**Validates: Requirements 3.8, 3.9**

---

### Property 7: Metric computation matches mathematical definition

For any Spark DataFrame containing known `Appliances` (actual) and `prediction` (predicted) columns, the RMSE computed by `RegressionEvaluator` SHALL equal `sqrt(mean((actual - predicted)²))`, the MAE SHALL equal `mean(|actual - predicted|)`, and R² SHALL equal `1 - SS_res / SS_tot`, all within a floating-point tolerance of 0.001.

**Validates: Requirements 5.1**

---

### Property 8: Metrics JSON round-trip preserves values

For any metrics dictionary containing float-valued RMSE, MAE, and R² for one or more model names, serialising the dictionary to `metrics.json` using `json.dump` and then deserialising it with `json.load` SHALL produce a dictionary that is equal to the original, with each numeric value matching within floating-point tolerance of 1e-9.

**Validates: Requirements 5.2**

---

### Property 9: Predictor returns numeric predictions for any valid input

For any dictionary of feature values where all required feature columns are present and all values are finite floats, the `Predictor.predict()` function SHALL return a dictionary containing numeric (float) predictions under both keys `"LinearRegression"` and `"RandomForest"`, without raising an exception.

**Validates: Requirements 6.1**
