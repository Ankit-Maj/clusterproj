# Requirements Document

## Introduction

This document specifies the requirements for the INT315 academic project: **Household Energy Consumption Prediction using Apache Spark MLlib**. The system loads a real-world household appliance energy dataset, performs data preprocessing and feature engineering using PySpark, trains two regression models (Linear Regression and Random Forest) via Spark MLlib, evaluates them using standard regression metrics, and presents results through an interactive Streamlit dashboard.

The dataset (`energydata_complete.csv`) contains 19,735 rows of 10-minute interval sensor readings (January–May 2016) with 29 columns including temperature sensors (T1–T9), humidity sensors (RH_1–RH_9), outdoor weather features, and the target variable `Appliances` (Wh).

---

## Glossary

- **System**: The household energy consumption prediction application as a whole.
- **DataLoader**: The module responsible for reading the CSV dataset and returning a Spark DataFrame.
- **Preprocessor**: The module responsible for cleaning, feature engineering, assembling feature vectors, scaling, and splitting data.
- **Trainer**: The module responsible for training, saving, and loading Spark MLlib regression models.
- **Evaluator**: The module responsible for computing regression metrics and persisting evaluation outputs.
- **Predictor**: The module responsible for loading trained models and producing a prediction for a single input record.
- **Dashboard**: The Streamlit web application presenting all sections to the user.
- **VisualizationEngine**: The module containing Plotly chart-generation functions used by the Dashboard.
- **FeatureVector**: The assembled numerical feature vector produced by VectorAssembler, excluding `rv1`, `rv2`, and `date`.
- **Appliances**: The target variable representing appliance energy consumption in watt-hours (Wh).
- **ChronologicalSplit**: A train/test split that preserves temporal order — the first 80% of rows (sorted ascending by `date`) form the training set; the remaining 20% form the test set.
- **RegressionEvaluator**: The `pyspark.ml.evaluation.RegressionEvaluator` class used to compute RMSE, MAE, and R².
- **MLlibModel**: Any model trained exclusively through `pyspark.ml` (Spark ML Pipeline API), specifically `LinearRegression` or `RandomForestRegressor`.
- **PySpark**: Apache Spark's Python API (`pyspark==3.5.0`).

---

## Requirements

### Requirement 1 — Project Structure and Environment Setup

**User Story:** As a developer, I want a well-organized project scaffold with pinned dependencies, so that the project is reproducible and easy to navigate.

#### Acceptance Criteria

1. THE System SHALL provide a directory layout matching: `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `src/` (containing `data_loader.py`, `preprocessing.py`, `train.py`, `evaluate.py`, `predict.py`, `visualization.py`), `tests/` (containing `test_data_loader.py`, `test_preprocessing.py`, `test_evaluation.py`), `data/`, `models/`, and `outputs/`.
2. THE System SHALL list `pyspark==3.5.0`, `streamlit`, `pandas`, `numpy`, `plotly`, and `pytest` as pinned dependencies in `requirements.txt`.
3. THE `requirements.txt` SHALL NOT include `scikit-learn`, `xgboost`, `tensorflow`, or `networkx`.
4. THE `.gitignore` SHALL exclude `models/`, `outputs/`, `__pycache__/`, `*.pyc`, `.env`, and `.DS_Store`.

---

### Requirement 2 — Data Loading

**User Story:** As a data scientist, I want the system to load the CSV dataset reliably from an absolute path, so that the application is not dependent on the current working directory.

#### Acceptance Criteria

1. THE DataLoader SHALL resolve the path to `energydata_complete.csv` using an absolute path strategy that does not rely on the process's current working directory.
2. WHEN the DataLoader reads the CSV, THE DataLoader SHALL infer the schema automatically and retain all 29 original columns.
3. WHEN the CSV is loaded, THE DataLoader SHALL parse the `date` column as a `TimestampType` using the format `"yyyy-MM-dd HH:mm:ss"`.
4. WHEN the loaded DataFrame is inspected, THE DataLoader SHALL return a DataFrame containing exactly the columns present in `energydata_complete.csv` without adding or removing any columns at load time.
5. IF the CSV file is not found at the resolved path, THEN THE DataLoader SHALL raise a descriptive `FileNotFoundError` that includes the resolved path in its message.

---

### Requirement 3 — Data Preprocessing and Feature Engineering

**User Story:** As a data scientist, I want the dataset cleaned, enriched with time-based features, and split chronologically, so that models are trained on sound, leak-free data.

#### Acceptance Criteria

1. THE Preprocessor SHALL drop columns `rv1` and `rv2` from the DataFrame before any feature engineering or model training.
2. THE Preprocessor SHALL deduplicate rows using all columns as the deduplication key.
3. IF any null values are present in any column, THEN THE Preprocessor SHALL drop those rows.
4. THE Preprocessor SHALL extract the following time-derived features from the `date` column: `hour` (integer 0–23), `day_of_week` (integer 0–6), `month` (integer 1–12), and `is_weekend` (integer 0 or 1, where Saturday=1 and Sunday=1).
5. THE Preprocessor SHALL assemble a FeatureVector using VectorAssembler from all numeric columns except `Appliances`, `rv1`, `rv2`, and `date`.
6. WHEN assembling the FeatureVector, THE Preprocessor SHALL include `lights`, `T1` through `T9`, `RH_1` through `RH_9`, `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`, `hour`, `day_of_week`, `month`, and `is_weekend`.
7. THE Preprocessor SHALL apply a ChronologicalSplit by first sorting the DataFrame ascending by `date`, then assigning the first 80% of rows to the training set and the remaining 20% to the test set.
8. WHEN preparing features for `LinearRegression`, THE Preprocessor SHALL apply `StandardScaler` to the FeatureVector; the `StandardScaler` SHALL be fit exclusively on the training set.
9. THE Preprocessor SHALL fit all pipeline stages (VectorAssembler, StandardScaler) exclusively on the training set and transform both training and test sets using the fitted pipeline.

---

### Requirement 4 — Model Training

**User Story:** As a data scientist, I want two Spark MLlib regression models trained, saved, and reloadable, so that results are reproducible and retraining is not mandatory on every run.

#### Acceptance Criteria

1. THE Trainer SHALL train Model A as a `pyspark.ml.regression.LinearRegression` with the scaled FeatureVector as input and `Appliances` as the label column.
2. THE Trainer SHALL train Model B as a `pyspark.ml.regression.RandomForestRegressor` with `numTrees=100`, `maxDepth=10`, `seed=42`, using the unscaled FeatureVector as input and `Appliances` as the label column.
3. WHEN training is complete, THE Trainer SHALL save each fitted model to the `models/` directory using Spark's native model persistence (`model.save(path)`).
4. WHEN the `models/` directory contains a saved model for a given algorithm, THE Trainer SHALL load the existing model instead of retraining, unless the `--retrain` flag is explicitly passed.
5. WHERE the `--retrain` flag is present, THE Trainer SHALL delete existing saved models and retrain from scratch.
6. THE Trainer SHALL NOT use any model class from `scikit-learn`, `xgboost`, or `tensorflow`.

---

### Requirement 5 — Model Evaluation

**User Story:** As a data scientist, I want evaluation metrics computed on the chronological test set and persisted, so that I can compare models and document results.

#### Acceptance Criteria

1. THE Evaluator SHALL compute RMSE, MAE, and R² for each trained model using `pyspark.ml.evaluation.RegressionEvaluator` applied exclusively to the test set.
2. THE Evaluator SHALL save all computed metrics to `outputs/metrics.json` as a JSON object keyed by model name (e.g., `"LinearRegression"`, `"RandomForest"`), each containing keys `"RMSE"`, `"MAE"`, and `"R2"`.
3. THE Evaluator SHALL save the combined test-set predictions (actual `Appliances`, predicted value, and `date`) from both models to `outputs/predictions.csv`.
4. WHEN the `outputs/` directory does not exist, THE Evaluator SHALL create it before writing any files.
5. THE Evaluator SHALL NOT evaluate any model on the training set.

---

### Requirement 6 — Single-Record Prediction

**User Story:** As a user, I want to input sensor readings and receive an energy consumption prediction from both models, so that I can interactively explore the models' behaviour.

#### Acceptance Criteria

1. THE Predictor SHALL accept a dictionary of feature values corresponding to all feature columns (excluding `date`, `Appliances`, `rv1`, `rv2`) and return numeric predictions from both the LinearRegression model and the RandomForest model.
2. WHEN a model is requested for prediction and no saved model exists in `models/`, THEN THE Predictor SHALL raise an informative `RuntimeError` indicating that training must be run first.
3. THE Predictor SHALL apply the same preprocessing pipeline (VectorAssembler and, for LinearRegression, StandardScaler) to the input record before passing it to the model.

---

### Requirement 7 — Visualization

**User Story:** As a user, I want Plotly-based charts rendered in the Dashboard, so that I can visually understand the data and model performance.

#### Acceptance Criteria

1. THE VisualizationEngine SHALL provide a function that returns a Plotly time-series line chart of `Appliances` consumption over `date`.
2. THE VisualizationEngine SHALL provide a function that returns a Plotly bar chart comparing RMSE, MAE, and R² values across both models.
3. THE VisualizationEngine SHALL provide a function that returns a Plotly scatter plot of actual vs. predicted `Appliances` values for a given model's prediction DataFrame.
4. THE VisualizationEngine SHALL provide a function that returns a Plotly correlation heatmap of numeric features.
5. THE VisualizationEngine SHALL provide a function that returns a Plotly histogram of the `Appliances` distribution.

---

### Requirement 8 — Streamlit Dashboard

**User Story:** As an academic evaluator, I want a multi-section Streamlit dashboard, so that all project components are accessible and demonstrable in one place.

#### Acceptance Criteria

1. THE Dashboard SHALL render Section A (Project Overview) describing the dataset, objective, and technology stack.
2. THE Dashboard SHALL render Section B (Data Explorer) displaying the raw DataFrame schema, row count, descriptive statistics, and the `Appliances` time-series and distribution charts.
3. THE Dashboard SHALL render Section C (Data Preparation) describing preprocessing steps applied and showing the chronological split sizes.
4. THE Dashboard SHALL render Section D (Model Comparison) displaying the metrics table and bar chart comparing both models.
5. THE Dashboard SHALL render Section E (Prediction Analysis) displaying the actual vs. predicted scatter plots for both models.
6. THE Dashboard SHALL render Section F (Interactive Prediction) providing input widgets for all feature values and displaying predictions from both models upon submission.
7. THE Dashboard SHALL render Section G (Findings and Limitations) summarising model performance observations and known limitations (including the exclusion of `rv1`/`rv2` and the absence of cross-validation).
8. WHEN the Dashboard starts, THE Dashboard SHALL initialise a `SparkSession` with `appName="HouseholdEnergyPrediction"` and reuse it across all sections.
9. THE Dashboard SHALL NOT expose raw Python exceptions to the end user; IF an unhandled error occurs in any section, THEN THE Dashboard SHALL display a descriptive error message using `st.error`.

---

### Requirement 9 — Automated Tests

**User Story:** As a developer, I want pytest-based unit tests covering data loading, preprocessing, and evaluation, so that regressions are caught automatically.

#### Acceptance Criteria

1. THE test suite `test_data_loader.py` SHALL verify that the CSV loads successfully, that the resulting DataFrame contains the expected 29 columns, that the row count equals 19,735, and that neither `rv1` nor `rv2` appears in the columns returned by the DataLoader (after drop).
2. THE test suite `test_preprocessing.py` SHALL verify that the FeatureVector does not contain `rv1`, `rv2`, `date`, or `Appliances`; that the ChronologicalSplit produces a training set size of approximately 80% of total rows (within ±1 row due to integer rounding); and that sorting by `date` is preserved (the last `date` in the training set is earlier than the first `date` in the test set).
3. THE test suite `test_evaluation.py` SHALL verify that RMSE, MAE, and R² are computed correctly by constructing a mock Spark DataFrame with known actual and predicted values, running the Evaluator's metric functions, and asserting the results match hand-calculated expected values within a tolerance of 0.001.
4. WHEN all tests are run with `pytest`, THE test suite SHALL complete without errors on a machine with PySpark 3.5.0 installed.

---

### Requirement 10 — Documentation

**User Story:** As a developer or evaluator, I want a complete README.md, so that the project can be set up and run without prior knowledge of its internals.

#### Acceptance Criteria

1. THE `README.md` SHALL include: project title and description, prerequisites (Python 3.9+, Java 8+, PySpark 3.5.0), installation steps (`pip install -r requirements.txt`), instructions to run training (`python src/train.py`), instructions to run evaluation (`python src/evaluate.py`), instructions to launch the dashboard (`streamlit run app.py`), instructions to run tests (`pytest tests/`), and a description of the project directory structure.
2. THE `README.md` SHALL document the dataset source and describe each major module (`data_loader`, `preprocessing`, `train`, `evaluate`, `predict`, `visualization`).
