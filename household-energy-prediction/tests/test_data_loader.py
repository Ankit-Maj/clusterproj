"""Tests for data_loader module."""
import pytest
import sys
import os

# Add project root to path so src imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_data_path, load_data
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType


@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    spark = SparkSession.builder \
        .appName("TestDataLoader") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def raw_df(spark):
    """Load the raw dataset once for all tests."""
    return load_data(spark)


def test_data_path_resolves():
    """get_data_path() should return a non-empty string pointing to an existing file."""
    path = get_data_path()
    assert isinstance(path, str)
    assert len(path) > 0
    assert os.path.isfile(path)


def test_data_path_missing_raises():
    """get_data_path() should raise FileNotFoundError for a non-existent path."""
    original = __import__("src.data_loader", fromlist=["data_loader"]).DATA_PATH
    import src.data_loader as dl
    dl.DATA_PATH = r"C:\nonexistent\path\missing.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        dl.get_data_path()
    assert "missing.csv" in str(exc_info.value) or "nonexistent" in str(exc_info.value)
    dl.DATA_PATH = original  # restore


def test_column_count(raw_df):
    """DataFrame should have exactly 29 columns."""
    assert len(raw_df.columns) == 29


def test_row_count(raw_df):
    """DataFrame should have exactly 19,735 rows."""
    assert raw_df.count() == 19735


def test_date_column_is_timestamp(raw_df):
    """The date column must be parsed as TimestampType."""
    date_field = [f for f in raw_df.schema.fields if f.name == "date"][0]
    assert isinstance(date_field.dataType, TimestampType)


def test_target_column_present(raw_df):
    """The Appliances target column must be present."""
    assert "Appliances" in raw_df.columns


def test_rv_columns_present_in_raw(raw_df):
    """rv1 and rv2 should still be present in the raw loaded data (dropped in preprocessing)."""
    assert "rv1" in raw_df.columns
    assert "rv2" in raw_df.columns
