"""
DataLoader module: loads energydata_complete.csv from its absolute path,
parses the date column as TimestampType, and returns the raw Spark DataFrame.
"""
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Absolute path to the dataset — does not depend on CWD
DATA_PATH = r"C:\Users\ASUS\Desktop\clusterproj\energydata_complete.csv"


def get_data_path() -> str:
    """Return absolute path to the dataset CSV.

    Raises:
        FileNotFoundError: If the file is not found at the resolved path.
    """
    path = os.path.abspath(DATA_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found at resolved path: {path}. "
            "Ensure energydata_complete.csv is present at "
            r"C:\Users\ASUS\Desktop\clusterproj\energydata_complete.csv"
        )
    return path


def load_data(spark: SparkSession) -> DataFrame:
    """Load the CSV dataset and parse the date column.

    Args:
        spark: Active SparkSession.

    Returns:
        Spark DataFrame with 29 columns. The 'date' column is cast to
        TimestampType. No columns are dropped (rv1/rv2 are dropped later
        in preprocessing).

    Raises:
        FileNotFoundError: If the CSV file cannot be found.
    """
    path = get_data_path()
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn("date", F.to_timestamp(F.col("date"), "yyyy-MM-dd HH:mm:ss"))
    return df


def get_dataset_info(spark: SparkSession) -> dict:
    """Return a summary dict of dataset characteristics.

    Returns dict with keys: row_count, column_count, columns, date_min, date_max.
    """
    df = load_data(spark)
    row_count = df.count()
    date_stats = df.agg(
        F.min("date").alias("date_min"),
        F.max("date").alias("date_max")
    ).collect()[0]
    return {
        "row_count": row_count,
        "column_count": len(df.columns),
        "columns": df.columns,
        "date_min": str(date_stats["date_min"]),
        "date_max": str(date_stats["date_max"]),
    }
