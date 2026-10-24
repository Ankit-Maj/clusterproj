# Implementation Plan: Household Energy Consumption Prediction

## Overview

Implement a PySpark MLlib regression pipeline that predicts household appliance energy consumption (`Appliances`, Wh) from sensor data. The pipeline covers data loading, preprocessing with chronological splitting, training two Spark MLlib models (Linear Regression and Random Forest), evaluation with RMSE/MAE/R², and a seven-section Streamlit dashboard. All tasks are written in Python 3.9+ using PySpark 3.5.0.

---

## Tasks

- [x] 1. Project scaffold — directories, config files, and pinned dependencies
  - Create the full directory tree under `c:\Users\ASUS\Desktop\clusterproj\household-energy-prediction\`: `src/`, `tests/`, `data/`, `models/`, `outputs/`
  - Create `src/__init__.py` and `tests/__init__.py` (empty)
  - Create `requirements.txt` with pinned entries: `pyspark==3.5.0`, `streamlit`, `pandas`, `numpy`, `plotly`, `pytest`
  - Create `.gitignore` excluding `models/`, `outputs/`, `__pycache__/`, `*.pyc`, `.env`, `.DS_Store`
  - Create a minimal `app.py` stub (imports only, `st.title("Household Energy Prediction")`) so the file exists
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Implement `src/data_loader.py`
  - [x] 2.1 Write `get_data_path()` — returns absolute path constant `C:\Users\ASUS\Desktop\clusterproj\energydata_complete.csv`; raises `FileNotFoundError` with the resolved path if the file is absent
  - Write `load_data(spark)` — reads CSV with `header=True, inferSchema=True`; casts `date` column to `TimestampType` using `F.to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ss")`; returns the 29-column DataFrame unchanged (no drops at load time)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x]* 2.2 Write `tests/test_data_loader.py`
    - Test that `load_data()` returns a DataFrame with exactly 29 columns
    - Test that the row count equals 19,735
    - Test that `date` column dtype is `TimestampType`
    - Test that calling `get_data_path()` with a patched non-existent path raises `FileNotFoundError` containing the path string
    - _Requirements: 9.1_

- [x] 3. Implement `src/preprocessing.py`
  - [x] 3.1 Write `clean(df)` — drops `rv1` and `rv2`, calls `dropDuplicates()`, calls `dropna()`; returns cleaned DataFrame
  - Write `add_time_features(df)` — adds `hour`, `day_of_week` (0=Sunday…6=Saturday via `dayofweek()-1`), `month`, and `is_weekend` (1 if `dayofweek` ∈ {1,7} else 0) using PySpark `functions`
  - Write `chronological_split(df, train_ratio=0.8)` — sorts DataFrame ascending by `date`; uses `Window.orderBy("date")` with `row_number()` to assign stable row indices; assigns first 80% to `train_df` and remaining 20% to `test_df`; returns `(train_df, test_df)`
  - Define `FEATURE_COLS` list: `lights`, `T1`–`T9`, `RH_1`–`RH_9`, `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`, `hour`, `day_of_week`, `month`, `is_weekend`
  - Write `build_lr_pipeline(train_df)` — creates `VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")`, fits `StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)` on the assembled training data; returns `(assembler, fitted_scaler_model)`
  - Write `build_rf_pipeline(train_df)` — creates `VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")`; returns the assembler
  - Write `preprocess(df)` — orchestrates clean → time features → split → pipelines; returns dict with keys `train_lr`, `test_lr`, `train_rf`, `test_rf`, `assembler_lr`, `scaler_model`, `assembler_rf`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [x]* 3.2 Write property test for FeatureVector composition (Property 1)
  - [x]* 3.3 Write property test for deduplication invariant (Property 2)
  - [x]* 3.4 Write property test for null-dropping invariant (Property 3)
  - [x]* 3.5 Write property test for time feature range invariants (Property 4)
  - [x]* 3.6 Write property test for chronological split ordering (Property 5)

- [x] 4. Checkpoint — verify preprocessing

- [x] 5. Implement `src/train.py`
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 6. Implement `src/evaluate.py`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x]* 6.1 Write property test for metric computation correctness (Property 7)
  - [x]* 6.2 Write property test for metrics JSON round-trip (Property 8)

- [x] 7. Implement `src/predict.py`
  - _Requirements: 6.1, 6.2, 6.3_

  - [x]* 7.1 Write property test for predictor numeric output (Property 9) — skipped (requires trained models on disk; tested manually via RuntimeError check)

- [x] 8. Implement `src/visualization.py`
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 9. Checkpoint — verify core modules

- [x] 10. Implement `app.py` — Sections A through D
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.8, 8.9_

- [x] 11. Implement `app.py` — Sections E through G
  - _Requirements: 8.5, 8.6, 8.7, 8.9_

- [x] 12. Write `tests/test_preprocessing.py`
  - _Requirements: 9.2, 3.7, 3.5_

- [x] 13. Write `README.md`
  - _Requirements: 10.1, 10.2_

- [x] 14. Integration verification
  - [x] 14.1 All imports OK — confirmed (`python -c "from src.data_loader import ..."`)
  - [x] 14.2 `pytest tests/ -v` — 19 passed in 124s, zero failures
  - [x]* 14.3 `python -m py_compile app.py` — syntax OK
  - _Requirements: 1.1, 9.4_

- [x] 15. Final checkpoint — all tests pass (19/19 ✅)

---

## Task Dependency Graph

```json
{
  "wave1": ["1"],
  "wave2": ["2.1"],
  "wave3": ["2.2", "3.1"],
  "wave4": ["3.2", "3.3", "3.4", "3.5", "3.6"],
  "wave5": ["4"],
  "wave6": ["5"],
  "wave7": ["6", "7", "8"],
  "wave8": ["6.1", "6.2", "9"],
  "wave9": ["10", "11", "12", "13"],
  "wave10": ["14.1", "14.2", "14.3"],
  "wave11": ["14", "15"]
}
```

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints at tasks 4, 9, and 15 ensure incremental validation
- Property tests (Properties 1–9 from design.md) validate universal correctness guarantees
- Unit tests validate specific examples and error conditions
- The `StandardScaler` model is saved to `models/scaler_model` by `train.py` and loaded at inference time by `predict.py`
- `models/` and `outputs/` are runtime-generated and gitignored; do not commit them
