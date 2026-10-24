"""
Preprocessing module: cleans data, engineers time features, builds feature vectors,
applies scaling, and performs a chronological 80/20 train/test split.

All preprocessing stages are fit on the training set only to prevent data leakage.
"""
import os
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline, PipelineModel

# Columns to exclude from the feature vector
EXCLUDE_COLS = {"date", "Appliances", "rv1", "rv2"}

# Ordered feature column list (28 features + 4 time features = 29 total,
# minus date and Appliances = 27 original features, but rv1/rv2 are dropped
# leaving 25 original + 4 time = 29 features total)
FEATURE_COLS = [
    "lights",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9",
    "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
    "hour", "day_of_week", "month", "is_weekend",
]

# Path for saving the scaler model alongside the LR model
SCALER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "scaler_model"
)


def clean(df: DataFrame) -> DataFrame:
    """Drop non-predictive columns, remove duplicates and null rows.

    Drops rv1 and rv2 (random variables with no predictive value).
    Removes exact duplicate rows.
    Drops rows with any null values.
    """
    df = df.drop("rv1", "rv2")
    df = df.dropDuplicates()
    df = df.dropna()
    return df


def add_time_features(df: DataFrame) -> DataFrame:
    """Extract time-based features from the date column.

    Added columns:
    - hour: integer 0-23
    - day_of_week: integer 0-6 (0=Sunday, 6=Saturday) via dayofweek()-1
    - month: integer 1-12
    - is_weekend: 1 if Saturday or Sunday, else 0
    """
    df = df.withColumn("hour", F.hour(F.col("date")))
    df = df.withColumn("day_of_week", F.dayofweek(F.col("date")) - 1)
    df = df.withColumn("month", F.month(F.col("date")))
    df = df.withColumn(
        "is_weekend",
        F.when(F.dayofweek(F.col("date")).isin([1, 7]), 1).otherwise(0)
    )
    return df


def chronological_split(df: DataFrame, train_ratio: float = 0.8):
    """Split the DataFrame chronologically into train and test sets.

    Sorts ascending by date, then assigns the first train_ratio fraction
    of rows to training and the remainder to testing. Uses a Window-based
    row_number() to guarantee a stable, reproducible split.

    Args:
        df: Input DataFrame containing a 'date' column.
        train_ratio: Fraction of rows for training (default 0.8).

    Returns:
        (train_df, test_df): Two DataFrames with no overlap.
    """
    df_sorted = df.orderBy("date")
    n = df_sorted.count()
    train_n = int(n * train_ratio)

    w = Window.orderBy("date")
    df_indexed = df_sorted.withColumn("_row_id", F.row_number().over(w))
    train_df = df_indexed.filter(F.col("_row_id") <= train_n).drop("_row_id")
    test_df  = df_indexed.filter(F.col("_row_id") >  train_n).drop("_row_id")
    return train_df, test_df


def build_lr_pipeline(train_df: DataFrame):
    """Build and fit the Linear Regression feature pipeline on training data.

    Creates a VectorAssembler → StandardScaler pipeline fit exclusively on
    train_df. Returns the fitted assembler and fitted scaler model so they
    can be applied to both train and test sets.

    Args:
        train_df: Training DataFrame (must contain all FEATURE_COLS).

    Returns:
        (assembler, fitted_scaler_model)
    """
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    train_assembled = assembler.transform(train_df)
    scaler_model = scaler.fit(train_assembled)
    return assembler, scaler_model


def build_rf_pipeline(train_df: DataFrame):
    """Build the Random Forest feature pipeline (no scaling needed).

    Returns a VectorAssembler that maps FEATURE_COLS → 'features'.
    """
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features",
        handleInvalid="skip"
    )
    return assembler


def apply_lr_pipeline(assembler, scaler_model, df: DataFrame) -> DataFrame:
    """Apply the LR pipeline (assemble + scale) to a DataFrame."""
    assembled = assembler.transform(df)
    scaled = scaler_model.transform(assembled)
    return scaled


def apply_rf_pipeline(assembler, df: DataFrame) -> DataFrame:
    """Apply the RF pipeline (assemble only) to a DataFrame."""
    return assembler.transform(df)


def preprocess(df: DataFrame) -> dict:
    """Full preprocessing pipeline.

    1. Clean: drop rv1/rv2, dedup, drop nulls
    2. Add time features
    3. Chronological 80/20 split
    4. Build LR pipeline (fit on train only)
    5. Build RF pipeline
    6. Transform train and test sets for both models

    Returns:
        dict with keys:
          train_lr, test_lr   — scaled DataFrames for Linear Regression
          train_rf, test_rf   — unscaled DataFrames for Random Forest
          assembler_lr, scaler_model, assembler_rf — fitted pipeline objects
          train_count, test_count — row counts
          train_date_min, train_date_max, test_date_min, test_date_max — date ranges
    """
    df = clean(df)
    df = add_time_features(df)
    train_df, test_df = chronological_split(df)

    # Build pipelines (fit on training data ONLY)
    assembler_lr, scaler_model = build_lr_pipeline(train_df)
    assembler_rf = build_rf_pipeline(train_df)

    # Apply to both splits
    train_lr = apply_lr_pipeline(assembler_lr, scaler_model, train_df)
    test_lr  = apply_lr_pipeline(assembler_lr, scaler_model, test_df)
    train_rf = apply_rf_pipeline(assembler_rf, train_df)
    test_rf  = apply_rf_pipeline(assembler_rf, test_df)

    # Collect date range stats
    train_dates = train_df.agg(
        F.min("date").alias("dmin"), F.max("date").alias("dmax")
    ).collect()[0]
    test_dates = test_df.agg(
        F.min("date").alias("dmin"), F.max("date").alias("dmax")
    ).collect()[0]

    return {
        "train_lr": train_lr,
        "test_lr":  test_lr,
        "train_rf": train_rf,
        "test_rf":  test_rf,
        "assembler_lr":  assembler_lr,
        "scaler_model":  scaler_model,
        "assembler_rf":  assembler_rf,
        "train_count":   train_df.count(),
        "test_count":    test_df.count(),
        "train_date_min": str(train_dates["dmin"]),
        "train_date_max": str(train_dates["dmax"]),
        "test_date_min":  str(test_dates["dmin"]),
        "test_date_max":  str(test_dates["dmax"]),
    }
