"""
Trainer module: trains, saves, and loads Spark MLlib regression models.

Linear Regression uses scaled features; Random Forest uses unscaled features.
Models are saved to models/ under the project root and loaded on subsequent
runs to avoid expensive retraining.
"""
import os
import sys
import shutil
import argparse

# Resolve project root relative to this file's location (not CWD)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
LR_MODEL_PATH  = os.path.join(MODELS_DIR, "lr_model")
RF_MODEL_PATH  = os.path.join(MODELS_DIR, "rf_model")
SCALER_PATH    = os.path.join(MODELS_DIR, "scaler_model")

from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel
from pyspark.ml.feature import StandardScalerModel


def train_lr(train_df, retrain: bool = False):
    """Train or load a Linear Regression model.
    
    If the model already exists and retrain=False, loads and returns the
    saved model. Otherwise trains from scratch, saves to LR_MODEL_PATH,
    and returns the fitted model.
    
    Args:
        train_df: Scaled training DataFrame with 'features' and 'Appliances' columns.
        retrain: If True, delete existing model and retrain.
    
    Returns:
        Fitted LinearRegressionModel.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(LR_MODEL_PATH) and not retrain:
        print(f"Loading existing LR model from {LR_MODEL_PATH}")
        return LinearRegressionModel.load(LR_MODEL_PATH)
    if os.path.exists(LR_MODEL_PATH):
        shutil.rmtree(LR_MODEL_PATH)
        print(f"Removed existing LR model for retraining.")
    print("Training Linear Regression model...")
    lr = LinearRegression(
        featuresCol="features",
        labelCol="Appliances",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.0,
        standardization=False  # data already scaled by StandardScaler
    )
    model = lr.fit(train_df)
    model.save(LR_MODEL_PATH)
    print(f"LR model saved to {LR_MODEL_PATH}")
    return model


def train_rf(train_df, retrain: bool = False):
    """Train or load a Random Forest Regression model.
    
    If the model already exists and retrain=False, loads and returns the
    saved model. Otherwise trains from scratch, saves to RF_MODEL_PATH,
    and returns the fitted model.
    
    Args:
        train_df: Unscaled training DataFrame with 'features' and 'Appliances' columns.
        retrain: If True, delete existing model and retrain.
    
    Returns:
        Fitted RandomForestRegressionModel.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(RF_MODEL_PATH) and not retrain:
        print(f"Loading existing RF model from {RF_MODEL_PATH}")
        return RandomForestRegressionModel.load(RF_MODEL_PATH)
    if os.path.exists(RF_MODEL_PATH):
        shutil.rmtree(RF_MODEL_PATH)
        print(f"Removed existing RF model for retraining.")
    print("Training Random Forest Regression model (this may take a few minutes)...")
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="Appliances",
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    model = rf.fit(train_df)
    model.save(RF_MODEL_PATH)
    print(f"RF model saved to {RF_MODEL_PATH}")
    return model


def save_scaler(scaler_model):
    """Persist the fitted StandardScaler model alongside the LR model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(SCALER_PATH):
        shutil.rmtree(SCALER_PATH)
    scaler_model.save(SCALER_PATH)
    print(f"Scaler model saved to {SCALER_PATH}")


def load_scaler():
    """Load the fitted StandardScaler model from disk."""
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            f"Scaler model not found at {SCALER_PATH}. "
            "Run python src/train.py to train and save all models."
        )
    return StandardScalerModel.load(SCALER_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Spark MLlib regression models for household energy prediction."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if saved models already exist."
    )
    args = parser.parse_args()

    # Add src/ to path so imports work when running as script
    sys.path.insert(0, PROJECT_ROOT)

    from pyspark.sql import SparkSession
    from src.data_loader import load_data
    from src.preprocessing import preprocess

    spark = SparkSession.builder \
        .appName("HouseholdEnergyPrediction-Train") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("Loading dataset...")
        df = load_data(spark)
        
        print("Preprocessing...")
        result = preprocess(df)
        
        print(f"Train set: {result['train_count']} rows "
              f"({result['train_date_min']} → {result['train_date_max']})")
        print(f"Test set:  {result['test_count']} rows "
              f"({result['test_date_min']} → {result['test_date_max']})")
        
        # Save scaler so predict.py can apply it at inference time
        save_scaler(result['scaler_model'])
        
        # Train models
        lr_model = train_lr(result['train_lr'], retrain=args.retrain)
        rf_model = train_rf(result['train_rf'], retrain=args.retrain)
        
        print("\nTraining complete. Models saved to models/")
        print(f"  LR model : {LR_MODEL_PATH}")
        print(f"  RF model : {RF_MODEL_PATH}")
        print(f"  Scaler   : {SCALER_PATH}")
        
    finally:
        spark.stop()
