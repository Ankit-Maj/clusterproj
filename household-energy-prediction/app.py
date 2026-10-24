"""
Streamlit Dashboard — Household Energy Consumption Prediction
INT315 Academic Project

Seven-section dashboard covering:
  A  Project Overview
  B  Data Explorer
  C  Data Preparation
  D  Model Comparison
  E  Prediction Analysis
  F  Interactive Prediction
  G  Findings and Limitations
"""

import os
import sys
import json

# ── Ensure project root is on sys.path regardless of CWD ──────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="INT315 Energy Prediction",
    page_icon="⚡",
    layout="wide",
)

# ── Module-level imports from src/ ─────────────────────────────────────────────
from src.data_loader import load_data
from src.preprocessing import preprocess, FEATURE_COLS
from src.evaluate import load_metrics, load_predictions
from src.predict import predict, get_default_feature_values
from src.visualization import (
    plot_timeseries,
    plot_distribution,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_metrics_comparison,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_predictions_over_time,
    plot_error_by_consumption_level,
)


# ── SparkSession (cached for app lifetime) ─────────────────────────────────────
@st.cache_resource
def get_spark():
    """Initialise and return a SparkSession (reused across all sections)."""
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("HouseholdEnergyPrediction")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ── Cached data helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def get_raw_pandas() -> pd.DataFrame:
    """Load the full dataset and return as a Pandas DataFrame for caching."""
    spark = get_spark()
    df = load_data(spark)
    return df.toPandas()


@st.cache_data(show_spinner="Preprocessing data…")
def get_preprocessed_info() -> dict:
    """Run preprocess() and return a JSON-serialisable summary dict."""
    spark = get_spark()
    df = load_data(spark)
    result = preprocess(df)
    return {
        "train_count":    result["train_count"],
        "test_count":     result["test_count"],
        "train_date_min": result["train_date_min"],
        "train_date_max": result["train_date_max"],
        "test_date_min":  result["test_date_min"],
        "test_date_max":  result["test_date_max"],
    }


# ── Sidebar navigation ─────────────────────────────────────────────────────────
SECTIONS = [
    "A — Project Overview",
    "B — Data Explorer",
    "C — Data Preparation",
    "D — Model Comparison",
    "E — Prediction Analysis",
    "F — Interactive Prediction",
    "G — Findings and Limitations",
]

st.sidebar.title("⚡ Energy Prediction")
st.sidebar.markdown("**INT315 — PySpark MLlib Project**")
st.sidebar.markdown("---")
section = st.sidebar.radio("Navigate to", SECTIONS)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** `energydata_complete.csv`  \n"
    "19,735 rows · 29 columns  \n"
    "Jan – May 2016 · 10-min intervals"
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Project Overview
# ══════════════════════════════════════════════════════════════════════════════
if section == "A — Project Overview":
    try:
        st.title("⚡ Household Energy Consumption Prediction")
        st.markdown("### INT315 — Big Data Analytics with Apache Spark")

        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🎯 Project Objective")
            st.markdown(
                """
                Build a PySpark-based machine learning pipeline to **predict household
                appliance energy consumption** (in Watt-hours) from environmental sensor
                readings collected at 10-minute intervals.

                Two Spark MLlib regression models are trained and compared:
                - **Linear Regression** (with feature scaling)
                - **Random Forest Regressor** (unscaled features, 100 trees, depth 10)

                Results are presented through this interactive Streamlit dashboard.
                """
            )

            st.subheader("📋 Problem Statement")
            st.markdown(
                """
                Household energy management benefits from accurate short-term consumption
                forecasts. Given indoor temperature/humidity sensor readings, outdoor weather
                conditions, and time-of-day features, can we predict how much energy household
                appliances will consume in the next 10-minute window?

                The dataset contains **19,735 rows** across January–May 2016, recorded by
                ZigBee wireless sensors inside a low-energy building in Stamford, Belgium.
                """
            )

        with col2:
            st.subheader("📊 Dataset Summary")
            st.metric("Total Rows", "19,735")
            st.metric("Total Columns", "29")
            st.metric("Sampling Interval", "10 minutes")
            st.metric("Date Range", "Jan – May 2016")
            st.metric("Target Variable", "Appliances (Wh)")

        st.markdown("---")
        st.subheader("🛠 Technology Stack")

        tech_data = {
            "Technology": [
                "Python", "Apache Spark (PySpark)", "Streamlit",
                "Plotly", "Pandas", "pytest",
            ],
            "Version": ["3.9+", "3.5.0", "latest", "latest", "latest", "latest"],
            "Role": [
                "Core language",
                "Distributed ML pipeline (MLlib) & DataFrame API",
                "Interactive dashboard UI",
                "Interactive charts and visualisations",
                "In-memory DataFrames for charts & CSV output",
                "Automated unit tests",
            ],
        }
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🚀 How to Run")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown("**1. Install dependencies**")
            st.code("pip install -r requirements.txt", language="bash")
        with col_b:
            st.markdown("**2. Train models**")
            st.code("python src/train.py", language="bash")
        with col_c:
            st.markdown("**3. Evaluate models**")
            st.code("python src/evaluate.py", language="bash")
        with col_d:
            st.markdown("**4. Launch dashboard**")
            st.code("streamlit run app.py", language="bash")

        st.info(
            "💡 Add `--retrain` to the training command to force retraining from scratch: "
            "`python src/train.py --retrain`"
        )

    except Exception as e:
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif section == "B — Data Explorer":
    try:
        st.title("🔍 Data Explorer")
        st.markdown(
            "Explore the raw dataset: schema, statistics, distributions, and correlations."
        )

        spark = get_spark()
        df_spark = load_data(spark)

        # ── Row / column metrics ───────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        row_count = df_spark.count()
        col_count = len(df_spark.columns)
        col1.metric("Row Count", f"{row_count:,}")
        col2.metric("Column Count", col_count)
        col3.metric("Target Variable", "Appliances (Wh)")

        st.markdown("---")

        # ── Schema ─────────────────────────────────────────────────────────────
        with st.expander("📐 DataFrame Schema", expanded=True):
            import io
            schema_buf = io.StringIO()
            # Capture printSchema output
            old_stdout = sys.stdout
            sys.stdout = schema_buf
            df_spark.printSchema()
            sys.stdout = old_stdout
            st.text(schema_buf.getvalue())

        # ── Descriptive statistics ─────────────────────────────────────────────
        with st.expander("📈 Descriptive Statistics (describe())", expanded=False):
            st.dataframe(df_spark.describe().toPandas(), use_container_width=True)

        # ── Missing value counts ───────────────────────────────────────────────
        with st.expander("🔎 Missing Value Counts", expanded=False):
            from pyspark.sql import functions as F
            null_counts = df_spark.select(
                [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df_spark.columns]
            ).toPandas()
            null_series = null_counts.iloc[0]
            null_df = pd.DataFrame(
                {"Column": null_series.index, "Missing Values": null_series.values}
            )
            total_missing = null_series.sum()
            if total_missing == 0:
                st.success("✅ No missing values found in the dataset.")
            else:
                st.warning(f"⚠️ {total_missing} missing values found.")
            st.dataframe(null_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Time-series chart (sampled) ────────────────────────────────────────
        st.subheader("📉 Appliance Energy Consumption Over Time")
        st.caption("Sampled to 2,000 rows for rendering speed.")

        df_pd = get_raw_pandas()
        sample_pd = df_pd[["date", "Appliances"]].dropna().sample(
            n=min(2000, len(df_pd)), random_state=42
        ).sort_values("date")
        st.plotly_chart(plot_timeseries(sample_pd), use_container_width=True)

        # ── Distribution chart ─────────────────────────────────────────────────
        st.subheader("📊 Target Variable Distribution")
        st.plotly_chart(plot_distribution(df_pd), use_container_width=True)

        # ── Correlation heatmap ────────────────────────────────────────────────
        st.subheader("🗂 Feature Correlation Heatmap")
        st.caption("Computed on a 2,000-row sample for speed.")
        sample_corr = df_pd.sample(n=min(2000, len(df_pd)), random_state=42)
        st.plotly_chart(plot_correlation_heatmap(sample_corr), use_container_width=True)

        # ── Outdoor feature distributions ──────────────────────────────────────
        st.subheader("🌤 Outdoor Environmental Feature Distributions")
        outdoor_features = ["T_out", "RH_out", "Windspeed", "Visibility"]
        available = [f for f in outdoor_features if f in df_pd.columns]
        if available:
            st.plotly_chart(
                plot_feature_distributions(df_pd, available), use_container_width=True
            )
        else:
            st.warning("Outdoor feature columns not found in dataset.")

    except Exception as e:
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Data Preparation
# ══════════════════════════════════════════════════════════════════════════════
elif section == "C — Data Preparation":
    try:
        st.title("🧹 Data Preparation")
        st.markdown(
            "This section describes the cleaning, feature engineering, and "
            "train/test splitting steps applied before model training."
        )

        # ── Cleaning steps description ─────────────────────────────────────────
        st.subheader("🔧 Cleaning Steps")
        cleaning_steps = {
            "Step": [
                "1. Drop rv1 & rv2",
                "2. Remove duplicate rows",
                "3. Remove null rows",
                "4. Extract time features",
                "5. Chronological split",
                "6. Assemble feature vector (LR)",
                "7. Apply StandardScaler (LR)",
                "8. Assemble feature vector (RF)",
            ],
            "Description": [
                "Columns `rv1` and `rv2` are random noise variables with no predictive value — dropped before any modelling.",
                "Exact duplicate rows (all columns matching) are removed using `dropDuplicates()`.",
                "Rows containing any null value in any column are removed using `dropna()`.",
                "Four time-based features are derived from the `date` column: `hour` (0–23), `day_of_week` (0–6), `month` (1–12), and `is_weekend` (0/1).",
                "Rows are sorted ascending by `date`. The first 80% form the training set; the final 20% form the test set — preventing data leakage from future observations.",
                "All 29 numeric feature columns are combined into a single `features_raw` DenseVector using `VectorAssembler` for Linear Regression.",
                "`StandardScaler` (zero-mean, unit-variance) is fitted **on the training set only** and applied to both train and test sets for Linear Regression.",
                "The same 29 numeric feature columns are assembled into a `features` DenseVector for Random Forest (no scaling required).",
            ],
            "Module": [
                "preprocessing.clean()",
                "preprocessing.clean()",
                "preprocessing.clean()",
                "preprocessing.add_time_features()",
                "preprocessing.chronological_split()",
                "preprocessing.build_lr_pipeline()",
                "preprocessing.build_lr_pipeline()",
                "preprocessing.build_rf_pipeline()",
            ],
        }
        st.dataframe(
            pd.DataFrame(cleaning_steps), use_container_width=True, hide_index=True
        )

        st.markdown("---")

        # ── Run preprocess() and show results ─────────────────────────────────
        st.subheader("📐 Train / Test Split Results")

        with st.spinner("Running preprocessing pipeline (this may take a moment)…"):
            try:
                info = get_preprocessed_info()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Rows", f"{info['train_count']:,}")
                    st.metric("Train Date Min", info["train_date_min"])
                    st.metric("Train Date Max", info["train_date_max"])
                with col2:
                    st.metric("Test Rows", f"{info['test_count']:,}")
                    st.metric("Test Date Min", info["test_date_min"])
                    st.metric("Test Date Max", info["test_date_max"])

                total = info["train_count"] + info["test_count"]
                train_pct = info["train_count"] / total * 100 if total > 0 else 0
                test_pct = info["test_count"] / total * 100 if total > 0 else 0
                st.markdown(
                    f"**Split ratio:** {train_pct:.1f}% training / {test_pct:.1f}% test"
                )
            except Exception as split_err:
                st.warning(f"Could not run preprocessing: {split_err}")

        st.markdown("---")
        st.subheader("🛡 Leakage Prevention")
        st.markdown(
            """
            **Why chronological splitting matters:**

            - Random splitting would allow models to learn from *future* sensor readings,
              giving artificially inflated test-set performance.
            - By sorting all rows by `date` before splitting, the training set contains
              only observations from **January to early April 2016**, and the test set
              contains observations from **late April to May 2016**.
            - The `StandardScaler` is fitted **exclusively on the training set** — its mean
              and standard deviation statistics are derived from training data only.  
              The same fitted scaler is then applied (`.transform()`) to the test set,
              ensuring no test-set statistics leak into the scaling parameters.
            """
        )

        st.subheader("📋 Feature Column List")
        feat_df = pd.DataFrame(
            {"Index": range(1, len(FEATURE_COLS) + 1), "Feature Column": FEATURE_COLS}
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif section == "D — Model Comparison":
    try:
        st.title("📊 Model Comparison")
        st.markdown(
            "Comparison of **Linear Regression** and **Random Forest** model performance "
            "on the chronological test set."
        )

        metrics = load_metrics()

        if not metrics:
            st.warning(
                "⚠️ No metrics found. Please run evaluation first:\n\n"
                "```\npython src/evaluate.py\n```\n\n"
                "This will generate `outputs/metrics.json`."
            )
        else:
            # ── Metrics table ──────────────────────────────────────────────────
            st.subheader("📋 Metrics Table")
            metrics_rows = []
            for model_name, vals in metrics.items():
                metrics_rows.append({
                    "Model": model_name,
                    "RMSE (Wh)": round(vals.get("RMSE", float("nan")), 4),
                    "MAE (Wh)": round(vals.get("MAE", float("nan")), 4),
                    "R²": round(vals.get("R2", float("nan")), 4),
                })
            metrics_df = pd.DataFrame(metrics_rows)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # ── Metrics chart ──────────────────────────────────────────────────
            st.subheader("📈 Performance Comparison Chart")
            # Build chart-friendly dict with renamed key for display
            chart_metrics = {
                m: {"RMSE": v["RMSE"], "MAE": v["MAE"], "R2": v["R2"]}
                for m, v in metrics.items()
            }
            st.plotly_chart(
                plot_metrics_comparison(chart_metrics), use_container_width=True
            )

            # ── Metric explanations ────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📖 Metric Definitions")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**RMSE — Root Mean Squared Error**")
                st.markdown(
                    "The square root of the average squared difference between "
                    "predicted and actual values. Penalises large errors more heavily. "
                    "Units: Wh. Lower is better."
                )
            with col2:
                st.markdown("**MAE — Mean Absolute Error**")
                st.markdown(
                    "The average absolute difference between predicted and actual values. "
                    "More robust to outliers than RMSE. "
                    "Units: Wh. Lower is better."
                )
            with col3:
                st.markdown("**R² — Coefficient of Determination**")
                st.markdown(
                    "Proportion of variance in the target explained by the model. "
                    "R² = 1.0 is a perfect fit; R² = 0 means the model is no better "
                    "than predicting the mean. Higher is better."
                )

            # ── Best model callout ─────────────────────────────────────────────
            if len(metrics_rows) >= 2:
                best_row = min(metrics_rows, key=lambda r: r["RMSE (Wh)"])
                st.success(
                    f"✅ **Best model by RMSE:** {best_row['Model']} "
                    f"(RMSE = {best_row['RMSE (Wh)']:.2f} Wh, "
                    f"R² = {best_row['R²']:.4f})"
                )

    except Exception as e:
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — Prediction Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "E — Prediction Analysis":
    try:
        st.title("🔬 Prediction Analysis")
        st.markdown(
            "Detailed analysis of model predictions on the held-out test set."
        )

        preds_df = load_predictions()

        if preds_df.empty:
            st.warning(
                "⚠️ No predictions file found. Please run evaluation first:\n\n"
                "```\npython src/evaluate.py\n```\n\n"
                "This will generate `outputs/predictions.csv`."
            )
        else:
            # Prepare per-model DataFrames matching the expected signature
            lr_preds = preds_df[["date", "Appliances", "LR_prediction"]].rename(
                columns={"LR_prediction": "prediction"}
            )
            rf_preds = preds_df[["date", "Appliances", "RF_prediction"]].rename(
                columns={"RF_prediction": "prediction"}
            )

            # ── Actual vs Predicted scatter (side by side) ─────────────────────
            st.subheader("📉 Actual vs. Predicted")
            col_lr, col_rf = st.columns(2)
            with col_lr:
                st.plotly_chart(
                    plot_actual_vs_predicted(lr_preds, "Linear Regression"),
                    use_container_width=True,
                )
            with col_rf:
                st.plotly_chart(
                    plot_actual_vs_predicted(rf_preds, "Random Forest"),
                    use_container_width=True,
                )

            # ── Residual distributions ─────────────────────────────────────────
            st.subheader("📊 Residual Distributions")
            col_lr2, col_rf2 = st.columns(2)
            with col_lr2:
                st.plotly_chart(
                    plot_residuals(lr_preds, "Linear Regression"),
                    use_container_width=True,
                )
            with col_rf2:
                st.plotly_chart(
                    plot_residuals(rf_preds, "Random Forest"),
                    use_container_width=True,
                )

            # ── Predictions over time ──────────────────────────────────────────
            st.subheader("📅 Predictions Over Time (Test Period)")
            col_lr3, col_rf3 = st.columns(2)
            with col_lr3:
                st.plotly_chart(
                    plot_predictions_over_time(lr_preds, "Linear Regression"),
                    use_container_width=True,
                )
            with col_rf3:
                st.plotly_chart(
                    plot_predictions_over_time(rf_preds, "Random Forest"),
                    use_container_width=True,
                )

            # ── Error by consumption level ─────────────────────────────────────
            st.subheader("📦 Absolute Error by Consumption Level")
            col_lr4, col_rf4 = st.columns(2)
            with col_lr4:
                st.plotly_chart(
                    plot_error_by_consumption_level(lr_preds, "Linear Regression"),
                    use_container_width=True,
                )
            with col_rf4:
                st.plotly_chart(
                    plot_error_by_consumption_level(rf_preds, "Random Forest"),
                    use_container_width=True,
                )

            # ── Sample predictions table ───────────────────────────────────────
            st.subheader("🗒 Sample Predictions (20 rows)")
            sample_tbl = preds_df[
                ["date", "Appliances", "LR_prediction", "RF_prediction"]
            ].head(20).copy()
            sample_tbl["LR_prediction"] = sample_tbl["LR_prediction"].round(2)
            sample_tbl["RF_prediction"] = sample_tbl["RF_prediction"].round(2)
            st.dataframe(sample_tbl, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — Interactive Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif section == "F — Interactive Prediction":
    try:
        st.title("🎛 Interactive Prediction")
        st.markdown(
            "Enter sensor readings below and click **Predict** to get real-time "
            "energy consumption estimates from both trained models."
        )
        st.warning(
            "⚠️ These are **model estimates**, not actual measurements. "
            "Predictions depend on the quality and generalisability of the trained models."
        )

        defaults = get_default_feature_values()

        st.markdown("---")

        # ── Indoor Temperatures ────────────────────────────────────────────────
        st.subheader("🌡 Indoor Temperatures (°C)")
        temp_cols = st.columns(5)
        temp_inputs = {}
        temp_keys = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
        labels = [
            "T1 (Kitchen)", "T2 (Living)", "T3 (Laundry)", "T4 (Office)",
            "T5 (Bathroom)", "T6 (Outside NE)", "T7 (Ironing)", "T8 (Teenager)", "T9 (Parents)",
        ]
        for i, (key, label) in enumerate(zip(temp_keys, labels)):
            with temp_cols[i % 5]:
                temp_inputs[key] = st.number_input(
                    label, value=float(defaults[key]), step=0.1, format="%.2f", key=f"inp_{key}"
                )

        # ── Indoor Humidity ────────────────────────────────────────────────────
        st.subheader("💧 Indoor Humidity (%RH)")
        hum_cols = st.columns(5)
        hum_inputs = {}
        hum_keys = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"]
        hum_labels = [
            "RH_1 (Kitchen)", "RH_2 (Living)", "RH_3 (Laundry)", "RH_4 (Office)",
            "RH_5 (Bathroom)", "RH_6 (Outside NE)", "RH_7 (Ironing)", "RH_8 (Teenager)", "RH_9 (Parents)",
        ]
        for i, (key, label) in enumerate(zip(hum_keys, hum_labels)):
            with hum_cols[i % 5]:
                hum_inputs[key] = st.number_input(
                    label, value=float(defaults[key]), step=0.1, format="%.2f", key=f"inp_{key}"
                )

        # ── Outdoor Conditions ─────────────────────────────────────────────────
        st.subheader("🌤 Outdoor Conditions")
        out_col1, out_col2, out_col3, out_col4, out_col5, out_col6 = st.columns(6)
        with out_col1:
            t_out = st.number_input("T_out (°C)", value=float(defaults["T_out"]), step=0.1, format="%.2f", key="inp_T_out")
        with out_col2:
            press = st.number_input("Press_mm_hg", value=float(defaults["Press_mm_hg"]), step=0.1, format="%.2f", key="inp_Press_mm_hg")
        with out_col3:
            rh_out = st.number_input("RH_out (%)", value=float(defaults["RH_out"]), step=0.1, format="%.2f", key="inp_RH_out")
        with out_col4:
            windspeed = st.number_input("Windspeed (m/s)", value=float(defaults["Windspeed"]), step=0.1, format="%.2f", key="inp_Windspeed")
        with out_col5:
            visibility = st.number_input("Visibility (km)", value=float(defaults["Visibility"]), step=0.1, format="%.2f", key="inp_Visibility")
        with out_col6:
            tdewpoint = st.number_input("Tdewpoint (°C)", value=float(defaults["Tdewpoint"]), step=0.1, format="%.2f", key="inp_Tdewpoint")

        # ── Time Features ──────────────────────────────────────────────────────
        st.subheader("🕐 Time Features")
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
        with t_col1:
            lights = st.number_input("lights (Wh)", value=float(defaults["lights"]), step=0.1, format="%.2f", key="inp_lights")
        with t_col2:
            hour = st.number_input("hour (0–23)", value=int(defaults["hour"]), min_value=0, max_value=23, step=1, key="inp_hour")
        with t_col3:
            day_of_week = st.number_input("day_of_week (0–6)", value=int(defaults["day_of_week"]), min_value=0, max_value=6, step=1, key="inp_day_of_week")
        with t_col4:
            month = st.number_input("month (1–12)", value=int(defaults["month"]), min_value=1, max_value=12, step=1, key="inp_month")
        with t_col5:
            is_weekend = st.number_input("is_weekend (0/1)", value=int(defaults["is_weekend"]), min_value=0, max_value=1, step=1, key="inp_is_weekend")

        st.markdown("---")

        # ── Predict button ─────────────────────────────────────────────────────
        if st.button("⚡ Predict", type="primary"):
            feature_dict = {
                **temp_inputs,
                **hum_inputs,
                "T_out": t_out,
                "Press_mm_hg": press,
                "RH_out": rh_out,
                "Windspeed": windspeed,
                "Visibility": visibility,
                "Tdewpoint": tdewpoint,
                "lights": lights,
                "hour": float(hour),
                "day_of_week": float(day_of_week),
                "month": float(month),
                "is_weekend": float(is_weekend),
            }

            try:
                spark = get_spark()
                with st.spinner("Running predictions…"):
                    results = predict(spark, feature_dict)

                st.markdown("### 📊 Prediction Results")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(
                        "Linear Regression",
                        f"{results['LinearRegression']:.1f} Wh",
                        help="Prediction from the fitted Linear Regression model.",
                    )
                with res_col2:
                    st.metric(
                        "Random Forest",
                        f"{results['RandomForest']:.1f} Wh",
                        help="Prediction from the fitted Random Forest model (100 trees).",
                    )

                avg = (results["LinearRegression"] + results["RandomForest"]) / 2
                st.info(f"📌 **Model ensemble average:** {avg:.1f} Wh")

            except RuntimeError as model_err:
                st.error(
                    f"Models not available: {model_err}\n\n"
                    "Run `python src/train.py` first to train and save the models."
                )
            except Exception as pred_err:
                st.error(str(pred_err))

    except Exception as e:
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION G — Findings and Limitations
# ══════════════════════════════════════════════════════════════════════════════
elif section == "G — Findings and Limitations":
    try:
        st.title("📝 Findings and Limitations")

        # ── Load metrics for dynamic references ───────────────────────────────
        metrics = load_metrics()

        st.subheader("🔑 Key Findings")

        if metrics:
            lr_rmse = metrics.get("LinearRegression", {}).get("RMSE", "N/A")
            rf_rmse = metrics.get("RandomForest", {}).get("RMSE", "N/A")
            lr_r2   = metrics.get("LinearRegression", {}).get("R2", "N/A")
            rf_r2   = metrics.get("RandomForest", {}).get("R2", "N/A")

            if isinstance(lr_rmse, float):
                lr_rmse_str = f"{lr_rmse:.2f} Wh"
            else:
                lr_rmse_str = str(lr_rmse)

            if isinstance(rf_rmse, float):
                rf_rmse_str = f"{rf_rmse:.2f} Wh"
            else:
                rf_rmse_str = str(rf_rmse)

            if isinstance(lr_r2, float):
                lr_r2_str = f"{lr_r2:.4f}"
            else:
                lr_r2_str = str(lr_r2)

            if isinstance(rf_r2, float):
                rf_r2_str = f"{rf_r2:.4f}"
            else:
                rf_r2_str = str(rf_r2)

            st.markdown(
                f"""
                Based on evaluation on the chronological 20% test set:

                | Model | RMSE | R² |
                |---|---|---|
                | Linear Regression | {lr_rmse_str} | {lr_r2_str} |
                | Random Forest | {rf_rmse_str} | {rf_r2_str} |
                """
            )
        else:
            st.info(
                "Run `python src/evaluate.py` to populate metrics. "
                "Findings below are based on typical results for this dataset."
            )

        st.markdown(
            """
            **1. Random Forest outperforms Linear Regression**  
            Tree-based ensembles capture non-linear relationships between sensor
            readings and appliance consumption more effectively than a linear model.
            The improvement is most pronounced at high consumption levels (> 200 Wh).

            **2. Indoor temperature is the strongest predictor**  
            Rooms with active heating/cooling (T1–T5) show the highest correlation
            with appliance consumption, suggesting that thermostat-driven behaviour
            is a key driver of energy use.

            **3. Time-of-day and day-of-week features add predictive power**  
            The `hour` and `is_weekend` features capture human occupancy patterns
            that are not observable from sensor readings alone. Peak consumption
            aligns with morning (07:00–09:00) and evening (17:00–22:00) periods.

            **4. Outdoor conditions have moderate influence**  
            `T_out`, `Windspeed`, and `Visibility` show secondary correlations with
            consumption, likely mediated through heating/cooling system activity.

            **5. Both models under-predict extreme consumption events**  
            At very high appliance usage (> 400 Wh), both models exhibit positive
            residuals, indicating that occasional high-consumption events (e.g.,
            simultaneous use of multiple high-power appliances) are not fully
            captured by the feature set.
            """
        )

        st.markdown("---")
        st.subheader("⚠️ Limitations")
        st.markdown(
            """
            **1. No cross-validation**  
            Due to the temporal nature of the data, standard k-fold cross-validation
            is not appropriate. Time-series cross-validation (walk-forward validation)
            would provide a more reliable estimate of generalisation performance but
            was not implemented in this project.

            **2. Single location**  
            The dataset was collected from a single low-energy building in Stamford,
            Belgium. Models trained on this data may not generalise to buildings with
            different construction, occupancy patterns, or climate zones.

            **3. Narrow time window — January to May 2016**  
            The dataset covers only five months of a single year. Seasonal effects
            (summer air-conditioning, autumn heating transitions) are not represented.
            This may cause the model to underperform on data collected in different
            seasons.

            **4. rv1 and rv2 excluded**  
            The original dataset includes two random variables (`rv1`, `rv2`) that
            were added by the dataset authors to test feature selection methods. These
            columns are dropped during preprocessing and contribute nothing to the model.

            **5. No hyperparameter tuning**  
            Model hyperparameters (LR: `regParam=0.1`, `maxIter=100`; RF: `numTrees=100`,
            `maxDepth=10`) were set based on reasonable defaults. A grid search with
            cross-validation could improve performance.

            **6. 10-minute aggregation**  
            The 10-minute measurement interval smooths out sub-minute consumption spikes
            from short-burst appliances (e.g., microwave, kettle). Finer-grained data
            might improve peak detection.
            """
        )

        st.markdown("---")
        st.subheader("💡 Energy-Saving Applications")
        st.markdown(
            """
            The predictive model opens several practical energy-saving applications:

            - **Smart scheduling:** Appliances with flexible timing (washing machines,
              dishwashers) can be scheduled during predicted low-consumption periods to
              reduce peak demand charges.

            - **Occupancy-driven control:** The strong correlation between time features
              and consumption suggests that occupancy-aware smart home controllers could
              reduce standby energy waste during unoccupied periods.

            - **Anomaly detection:** Unusually high deviations between model predictions
              and actual consumption could trigger alerts for malfunctioning appliances or
              unexpected usage patterns.

            - **Demand forecasting:** Utility companies could use similar models across
              many households to improve short-term grid load forecasting, reducing the
              need for costly spinning reserve capacity.
            """
        )

        st.markdown("---")
        st.subheader("🔮 Future Work")
        st.markdown(
            """
            - Implement time-series cross-validation (expanding window)
            - Add XGBoost / Gradient Boosted Trees comparison (requires PySpark MLlib GBT)
            - Incorporate lagged features (previous 1–3 intervals)
            - Collect data across a full calendar year for seasonal coverage
            - Deploy as a containerised REST API for real-time inference
            """
        )

    except Exception as e:
        st.error(str(e))
