"""Tests for preprocessing module — property tests and unit tests."""
import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType
)
from src.preprocessing import (
    FEATURE_COLS, EXCLUDE_COLS, clean, add_time_features,
    chronological_split, build_lr_pipeline, build_rf_pipeline, preprocess
)


@pytest.fixture(scope="module")
def spark():
    s = SparkSession.builder \
        .appName("TestPreprocessing") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def make_sample_df(spark, n=20):
    """Create a small synthetic DataFrame matching the dataset schema."""
    base_time = datetime(2016, 1, 11, 17, 0, 0)
    rows = []
    for i in range(n):
        ts = base_time + timedelta(minutes=10 * i)
        rows.append((
            ts, float(50 + i * 5), float(10),
            20.0, 45.0, 19.5, 44.0, 20.0, 44.5, 19.0, 45.0, 17.0,
            55.0, 7.0, 84.0, 17.2, 41.5, 18.2, 48.9, 17.0, 45.5,
            6.5, 733.5, 92.0, 7.0, 63.0, 5.3, 13.27, 13.27
        ))

    schema = StructType([
        StructField("date", TimestampType()),
        StructField("Appliances", DoubleType()),
        StructField("lights", DoubleType()),
        StructField("T1", DoubleType()), StructField("RH_1", DoubleType()),
        StructField("T2", DoubleType()), StructField("RH_2", DoubleType()),
        StructField("T3", DoubleType()), StructField("RH_3", DoubleType()),
        StructField("T4", DoubleType()), StructField("RH_4", DoubleType()),
        StructField("T5", DoubleType()), StructField("RH_5", DoubleType()),
        StructField("T6", DoubleType()), StructField("RH_6", DoubleType()),
        StructField("T7", DoubleType()), StructField("RH_7", DoubleType()),
        StructField("T8", DoubleType()), StructField("RH_8", DoubleType()),
        StructField("T9", DoubleType()), StructField("RH_9", DoubleType()),
        StructField("T_out", DoubleType()),
        StructField("Press_mm_hg", DoubleType()),
        StructField("RH_out", DoubleType()),
        StructField("Windspeed", DoubleType()),
        StructField("Visibility", DoubleType()),
        StructField("Tdewpoint", DoubleType()),
        StructField("rv1", DoubleType()),
        StructField("rv2", DoubleType()),
    ])
    return spark.createDataFrame(rows, schema)


# ── Property 1: FeatureVector excludes forbidden columns ─────────────────────

def test_feature_cols_exclude_forbidden():
    """FEATURE_COLS must not contain rv1, rv2, Appliances, or date."""
    for forbidden in ["rv1", "rv2", "Appliances", "date"]:
        assert forbidden not in FEATURE_COLS, \
            f"Forbidden column '{forbidden}' found in FEATURE_COLS"


def test_feature_cols_include_required():
    """FEATURE_COLS must contain all expected sensor and time features."""
    required = [
        "lights",
        "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
        "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9",
        "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
        "hour", "day_of_week", "month", "is_weekend",
    ]
    for col in required:
        assert col in FEATURE_COLS, f"Expected feature '{col}' missing from FEATURE_COLS"


# ── Property 2: Deduplication removes all duplicate rows ─────────────────────

def test_clean_removes_duplicates(spark):
    """After clean(), no duplicate rows should remain."""
    df = make_sample_df(spark, n=10)
    # Duplicate the first 3 rows
    df_with_dupes = df.union(df.limit(3))
    assert df_with_dupes.count() == 13, "Pre-condition: should have 13 rows"
    cleaned = clean(df_with_dupes)
    assert cleaned.count() == cleaned.dropDuplicates().count(), \
        "clean() should remove all duplicate rows"


# ── Property 3: Null-dropping produces a null-free DataFrame ─────────────────

def test_clean_removes_nulls(spark):
    """After clean(), no null values should remain in any column."""
    df = make_sample_df(spark, n=10)
    # Inject nulls
    df_nulls = df.withColumn("T1", F.when(F.col("T1") > 20.5, None).otherwise(F.col("T1")))
    cleaned = clean(df_nulls)
    for col_name in cleaned.columns:
        null_count = cleaned.filter(F.col(col_name).isNull()).count()
        assert null_count == 0, f"Column '{col_name}' still has nulls after clean()"


# ── Property 4: Time feature range invariants ─────────────────────────────────

def test_time_features_range(spark):
    """Time features must stay within valid ranges after add_time_features()."""
    df = make_sample_df(spark, n=20)
    df_clean = clean(df)
    df_tf = add_time_features(df_clean)

    rows = df_tf.select("hour", "day_of_week", "month", "is_weekend").collect()
    for row in rows:
        assert 0 <= row["hour"] <= 23, f"hour out of range: {row['hour']}"
        assert 0 <= row["day_of_week"] <= 6, f"day_of_week out of range: {row['day_of_week']}"
        assert 1 <= row["month"] <= 12, f"month out of range: {row['month']}"
        assert row["is_weekend"] in (0, 1), f"is_weekend must be 0 or 1: {row['is_weekend']}"


# ── Property 5: Chronological split preserves temporal ordering ───────────────

def test_chronological_split_ordering(spark):
    """Max date in train must be <= min date in test (no temporal leakage)."""
    df = make_sample_df(spark, n=20)
    df_clean = clean(df)
    df_tf = add_time_features(df_clean)
    train_df, test_df = chronological_split(df_tf, train_ratio=0.8)

    train_max = train_df.agg(F.max("date")).collect()[0][0]
    test_min  = test_df.agg(F.min("date")).collect()[0][0]
    assert train_max <= test_min, \
        f"Temporal leakage: train max ({train_max}) > test min ({test_min})"


def test_chronological_split_ratio(spark):
    """Train/test split must approximate the 80/20 ratio."""
    df = make_sample_df(spark, n=100)
    df_clean = clean(df)
    df_tf = add_time_features(df_clean)
    train_df, test_df = chronological_split(df_tf, train_ratio=0.8)

    total = train_df.count() + test_df.count()
    train_ratio = train_df.count() / total
    assert abs(train_ratio - 0.8) <= 0.02, \
        f"Train ratio {train_ratio:.3f} is too far from 0.8"


def test_chronological_split_no_overlap(spark):
    """Train and test sets must have no overlapping timestamps."""
    df = make_sample_df(spark, n=20)
    df_clean = clean(df)
    df_tf = add_time_features(df_clean)
    train_df, test_df = chronological_split(df_tf, train_ratio=0.8)

    train_dates = set(row[0] for row in train_df.select("date").collect())
    test_dates  = set(row[0] for row in test_df.select("date").collect())
    overlap = train_dates.intersection(test_dates)
    assert len(overlap) == 0, f"Train/test overlap detected: {overlap}"


# ── Unit test: clean drops rv1 and rv2 ───────────────────────────────────────

def test_clean_drops_rv_columns(spark):
    """clean() must remove rv1 and rv2 from the DataFrame."""
    df = make_sample_df(spark, n=5)
    assert "rv1" in df.columns
    assert "rv2" in df.columns
    cleaned = clean(df)
    assert "rv1" not in cleaned.columns, "rv1 should be dropped by clean()"
    assert "rv2" not in cleaned.columns, "rv2 should be dropped by clean()"
