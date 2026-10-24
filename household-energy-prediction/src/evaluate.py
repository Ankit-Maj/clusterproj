"""
Evaluator module: computes regression metrics for trained Spark MLlib models
and persists metrics and predictions to the outputs/ directory.
"""
import os
import json
import math

# Absolute paths derived from this file's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
import pandas as pd


def evaluate_model(model, test_df: DataFrame, model_name: str) -> tuple:
    """Compute RMSE, MAE, and R² for a fitted model on the test set.
    
    Args:
        model: Fitted Spark MLlib regression model.
        test_df: Test DataFrame with 'features' and 'Appliances' columns.
        model_name: Display name used in logging.
    
    Returns:
        (metrics_dict, predictions_df) where:
          metrics_dict = {"RMSE": float, "MAE": float, "R2": float}
          predictions_df = Spark DataFrame with prediction column added
    """
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(
        labelCol="Appliances",
        predictionCol="prediction"
    )
    rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    mae  = evaluator.setMetricName("mae").evaluate(predictions)
    r2   = evaluator.setMetricName("r2").evaluate(predictions)
    
    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"{model_name} — RMSE: {rmse:.2f} Wh | MAE: {mae:.2f} Wh | R²: {r2:.4f}")
    return metrics, predictions


def save_metrics(metrics_dict: dict):
    """Save all model metrics to outputs/metrics.json.
    
    Args:
        metrics_dict: {"ModelName": {"RMSE": float, "MAE": float, "R2": float}, ...}
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {path}")


def load_metrics() -> dict:
    """Load metrics from outputs/metrics.json.
    
    Returns:
        Metrics dict or empty dict if file does not exist.
    """
    path = os.path.join(OUTPUTS_DIR, "metrics.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_predictions(lr_preds: DataFrame, rf_preds: DataFrame):
    """Save combined test predictions to outputs/predictions.csv.
    
    Selects date, Appliances (actual), LR prediction, and RF prediction,
    merges on date+Appliances, and writes to CSV.
    
    Args:
        lr_preds: Spark DataFrame from LR model.transform(test_df)
        rf_preds: Spark DataFrame from RF model.transform(test_df)
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, "predictions.csv")
    
    lr_pd = lr_preds.select("date", "Appliances", "prediction") \
                    .toPandas() \
                    .rename(columns={"prediction": "LR_prediction"})
    rf_pd = rf_preds.select("date", "Appliances", "prediction") \
                    .toPandas() \
                    .rename(columns={"prediction": "RF_prediction"})
    
    merged = lr_pd.merge(rf_pd, on=["date", "Appliances"])
    merged["date"] = merged["date"].astype(str)
    merged.to_csv(path, index=False)
    print(f"Predictions saved to {path} ({len(merged)} rows)")


def load_predictions() -> pd.DataFrame:
    """Load saved predictions from outputs/predictions.csv.
    
    Returns:
        Pandas DataFrame or empty DataFrame if file does not exist.
    """
    path = os.path.join(OUTPUTS_DIR, "predictions.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    
    from pyspark.sql import SparkSession
    from src.data_loader import load_data
    from src.preprocessing import preprocess
    from src.train import train_lr, train_rf, load_scaler, LR_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH
    from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel
    from pyspark.ml.feature import StandardScalerModel
    from src.preprocessing import FEATURE_COLS, apply_lr_pipeline, apply_rf_pipeline
    
    spark = SparkSession.builder \
        .appName("HouseholdEnergyPrediction-Evaluate") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        print("Loading data and preprocessing...")
        df = load_data(spark)
        result = preprocess(df)
        
        print("Loading trained models...")
        lr_model = LinearRegressionModel.load(LR_MODEL_PATH)
        rf_model  = RandomForestRegressionModel.load(RF_MODEL_PATH)
        
        print("Evaluating models on test set...")
        lr_metrics, lr_preds = evaluate_model(lr_model, result["test_lr"], "LinearRegression")
        rf_metrics, rf_preds = evaluate_model(rf_model, result["test_rf"], "RandomForest")
        
        all_metrics = {
            "LinearRegression": lr_metrics,
            "RandomForest": rf_metrics,
        }
        save_metrics(all_metrics)
        save_predictions(lr_preds, rf_preds)
        
        print("\nEvaluation complete.")
        print(f"  outputs/metrics.json written")
        print(f"  outputs/predictions.csv written")
        
    finally:
        spark.stop()
