"""
Predictor module: loads trained Spark MLlib models and produces a prediction
for a single input record from both Linear Regression and Random Forest models.

The StandardScaler model is loaded from disk so LR prediction uses the same
scaling statistics computed during training.
"""
import os
import sys

# Resolve project root relative to this file's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel

from src.preprocessing import FEATURE_COLS
from src.train import LR_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH


def _check_models():
    """Raise RuntimeError if any required model artifact is missing."""
    missing = []
    if not os.path.exists(LR_MODEL_PATH):
        missing.append(f"LR model at {LR_MODEL_PATH}")
    if not os.path.exists(RF_MODEL_PATH):
        missing.append(f"RF model at {RF_MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        missing.append(f"Scaler at {SCALER_PATH}")
    if missing:
        raise RuntimeError(
            "The following model artifacts were not found:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nRun `python src/train.py` to train and save all models."
        )


def predict(spark: SparkSession, feature_dict: dict) -> dict:
    """Produce energy consumption predictions from both trained models.
    
    Applies the same preprocessing pipeline used during training:
    - LR: VectorAssembler (features_raw) → StandardScaler → LinearRegression
    - RF: VectorAssembler (features) → RandomForestRegressor
    
    Args:
        spark: Active SparkSession.
        feature_dict: Dict mapping feature names (from FEATURE_COLS) to numeric values.
                      Must contain all 29 feature columns.
    
    Returns:
        Dict: {"LinearRegression": float, "RandomForest": float}
    
    Raises:
        RuntimeError: If any model artifact is missing from models/.
        ValueError: If feature_dict is missing required columns.
    """
    _check_models()
    
    # Validate that all required features are present
    missing_features = [f for f in FEATURE_COLS if f not in feature_dict]
    if missing_features:
        raise ValueError(
            f"feature_dict is missing required feature columns: {missing_features}"
        )
    
    # Ensure all values are floats
    clean_dict = {k: float(v) for k, v in feature_dict.items() if k in FEATURE_COLS}
    
    # Create a single-row DataFrame
    row_df = spark.createDataFrame([clean_dict])
    
    # ── Linear Regression prediction ──────────────────────────────────────────
    assembler_lr = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    scaler_model = StandardScalerModel.load(SCALER_PATH)
    lr_model     = LinearRegressionModel.load(LR_MODEL_PATH)
    
    lr_assembled = assembler_lr.transform(row_df)
    lr_scaled    = scaler_model.transform(lr_assembled)
    lr_result    = lr_model.transform(lr_scaled)
    lr_pred      = lr_result.select("prediction").collect()[0][0]
    
    # ── Random Forest prediction ───────────────────────────────────────────────
    assembler_rf = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features",
        handleInvalid="skip"
    )
    rf_model  = RandomForestRegressionModel.load(RF_MODEL_PATH)
    rf_result = rf_model.transform(assembler_rf.transform(row_df))
    rf_pred   = rf_result.select("prediction").collect()[0][0]
    
    return {
        "LinearRegression": float(lr_pred),
        "RandomForest": float(rf_pred),
    }


def get_default_feature_values() -> dict:
    """Return a dict of representative (mean-like) feature values for the UI.
    
    These defaults are based on approximate dataset means and are used as
    initial values in the Streamlit interactive prediction form.
    
    Returns:
        Dict mapping each feature column name to a default float value.
    """
    return {
        "lights": 3.8,
        "T1": 20.86, "T2": 20.32, "T3": 22.27, "T4": 20.96,
        "T5": 19.59, "T6": 7.91,  "T7": 20.26, "T8": 22.03, "T9": 19.49,
        "RH_1": 42.37, "RH_2": 40.42, "RH_3": 39.24, "RH_4": 36.83,
        "RH_5": 50.95, "RH_6": 54.61, "RH_7": 35.39, "RH_8": 47.66, "RH_9": 41.11,
        "T_out": 7.41, "Press_mm_hg": 755.52, "RH_out": 79.75,
        "Windspeed": 4.04, "Visibility": 38.33, "Tdewpoint": 3.76,
        "hour": 14, "day_of_week": 1, "month": 2, "is_weekend": 0,
    }
