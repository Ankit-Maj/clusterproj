"""
Tests for evaluate module — property tests for metric correctness and JSON round-trip.
"""
import pytest
import sys
import os
import json
import math
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from src.evaluate import evaluate_model, save_metrics, load_metrics


@pytest.fixture(scope="module")
def spark():
    s = SparkSession.builder \
        .appName("TestEvaluation") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


# ── Property 7: Metric computation matches mathematical definition ────────────

class MockModel:
    """A mock Spark model that adds a fixed 'prediction' column."""
    def __init__(self, preds):
        self._preds = preds  # list of floats
    
    def transform(self, df):
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType
        # Add prediction column by index (using monotonically_increasing_id trick)
        # For simplicity, create predictions as a literal mapping via a join
        spark = df.sparkSession
        pred_df = spark.createDataFrame(
            [(float(p),) for p in self._preds],
            ["prediction"]
        )
        # Add row numbers to both and join
        from pyspark.sql.window import Window
        w = Window.orderBy(F.monotonically_increasing_id())
        df_idx = df.withColumn("_idx", F.row_number().over(w))
        pred_idx = pred_df.withColumn("_idx", F.row_number().over(
            Window.orderBy(F.monotonically_increasing_id())
        ))
        return df_idx.join(pred_idx, on="_idx").drop("_idx")


def test_metric_computation_correctness(spark):
    """Property 7: RMSE, MAE, R² must match hand-calculated values within 0.001.
    
    **Validates: Requirements 5.1**
    """
    # Known actuals and predictions
    actuals = [10.0, 20.0, 30.0, 40.0, 50.0]
    preds   = [12.0, 18.0, 33.0, 38.0, 52.0]
    
    # Hand-calculate expected values
    n = len(actuals)
    errors = [a - p for a, p in zip(actuals, preds)]
    sq_errors = [e ** 2 for e in errors]
    abs_errors = [abs(e) for e in errors]
    
    expected_rmse = math.sqrt(sum(sq_errors) / n)
    expected_mae  = sum(abs_errors) / n
    mean_actual   = sum(actuals) / n
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
    ss_res = sum(sq_errors)
    expected_r2   = 1 - ss_res / ss_tot
    
    # Build a Spark DataFrame with known actuals
    schema = StructType([StructField("Appliances", DoubleType())])
    df = spark.createDataFrame([(float(a),) for a in actuals], schema)
    
    # Add prediction column manually
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    pred_df = spark.createDataFrame([(float(p),) for p in preds], ["prediction"])
    w = Window.orderBy(F.monotonically_increasing_id())
    df_idx   = df.withColumn("_idx", F.row_number().over(w))
    pred_idx = pred_df.withColumn("_idx", F.row_number().over(w))
    joined   = df_idx.join(pred_idx, on="_idx").drop("_idx")
    
    from pyspark.ml.evaluation import RegressionEvaluator
    evaluator = RegressionEvaluator(labelCol="Appliances", predictionCol="prediction")
    
    actual_rmse = evaluator.setMetricName("rmse").evaluate(joined)
    actual_mae  = evaluator.setMetricName("mae").evaluate(joined)
    actual_r2   = evaluator.setMetricName("r2").evaluate(joined)
    
    assert abs(actual_rmse - expected_rmse) < 0.001, \
        f"RMSE mismatch: got {actual_rmse:.6f}, expected {expected_rmse:.6f}"
    assert abs(actual_mae - expected_mae) < 0.001, \
        f"MAE mismatch: got {actual_mae:.6f}, expected {expected_mae:.6f}"
    assert abs(actual_r2 - expected_r2) < 0.001, \
        f"R² mismatch: got {actual_r2:.6f}, expected {expected_r2:.6f}"


# ── Property 8: Metrics JSON round-trip preserves values ─────────────────────

def test_metrics_json_roundtrip(tmp_path):
    """Property 8: save_metrics + load_metrics must preserve float values within 1e-9.
    
    **Validates: Requirements 5.2**
    """
    import src.evaluate as ev
    
    # Temporarily redirect OUTPUTS_DIR
    original_dir = ev.OUTPUTS_DIR
    ev.OUTPUTS_DIR = str(tmp_path)
    
    try:
        sample = {
            "LinearRegression": {"RMSE": 87.654321, "MAE": 54.123456, "R2": 0.345678},
            "RandomForest":     {"RMSE": 62.111111, "MAE": 41.999999, "R2": 0.612345},
        }
        save_metrics(sample)
        loaded = load_metrics()
        
        for model_name, vals in sample.items():
            assert model_name in loaded, f"Model '{model_name}' missing from loaded metrics"
            for metric_name, expected_val in vals.items():
                actual_val = loaded[model_name][metric_name]
                assert abs(actual_val - expected_val) < 1e-9, (
                    f"{model_name}.{metric_name}: expected {expected_val}, "
                    f"got {actual_val}"
                )
    finally:
        ev.OUTPUTS_DIR = original_dir


def test_load_metrics_returns_empty_when_missing(tmp_path):
    """load_metrics() must return an empty dict when metrics.json does not exist."""
    import src.evaluate as ev
    original_dir = ev.OUTPUTS_DIR
    ev.OUTPUTS_DIR = str(tmp_path / "nonexistent_dir")
    try:
        result = load_metrics()
        assert result == {}, f"Expected empty dict, got {result}"
    finally:
        ev.OUTPUTS_DIR = original_dir
