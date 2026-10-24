"""
VisualizationEngine: Plotly chart functions for the Streamlit dashboard.

All functions accept pandas DataFrames or plain Python dicts and return
Plotly figure objects ready for st.plotly_chart().
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_timeseries(df_pd: pd.DataFrame) -> go.Figure:
    """Line chart of appliance energy consumption over time.
    
    Args:
        df_pd: Pandas DataFrame with 'date' and 'Appliances' columns.
    
    Returns:
        Plotly line figure.
    """
    fig = px.line(
        df_pd,
        x="date",
        y="Appliances",
        title="Appliance Energy Consumption Over Time",
        labels={"date": "Date", "Appliances": "Energy Consumption (Wh)"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Energy Consumption (Wh)",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_distribution(df_pd: pd.DataFrame) -> go.Figure:
    """Histogram of the target variable (Appliances) distribution.
    
    Args:
        df_pd: Pandas DataFrame with 'Appliances' column.
    
    Returns:
        Plotly histogram figure.
    """
    fig = px.histogram(
        df_pd,
        x="Appliances",
        nbins=50,
        title="Distribution of Appliance Energy Consumption",
        labels={"Appliances": "Energy Consumption (Wh)", "count": "Frequency"},
        color_discrete_sequence=["#2ca02c"],
    )
    fig.update_layout(
        xaxis_title="Energy Consumption (Wh)",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    return fig


def plot_metrics_comparison(metrics: dict) -> go.Figure:
    """Grouped bar chart comparing RMSE, MAE, and R² across models.
    
    Args:
        metrics: Dict of form {"ModelName": {"RMSE": float, "MAE": float, "R2": float}, ...}
    
    Returns:
        Plotly bar figure.
    """
    rows = []
    for model_name, vals in metrics.items():
        for metric_name, value in vals.items():
            rows.append({"Model": model_name, "Metric": metric_name, "Value": value})
    df = pd.DataFrame(rows)
    
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="Model Performance Comparison (RMSE, MAE, R²)",
        labels={"Value": "Metric Value", "Metric": "Evaluation Metric"},
        color_discrete_map={
            "LinearRegression": "#1f77b4",
            "RandomForest": "#ff7f0e",
        },
    )
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Model",
    )
    return fig


def plot_actual_vs_predicted(preds_pd: pd.DataFrame, model_name: str) -> go.Figure:
    """Scatter plot of actual vs. predicted energy consumption for one model.
    
    Includes a perfect-prediction reference line (y=x).
    
    Args:
        preds_pd: Pandas DataFrame with 'Appliances' (actual) and 'prediction' columns.
        model_name: Display name for the chart title.
    
    Returns:
        Plotly scatter figure.
    """
    mn = preds_pd["Appliances"].min()
    mx = preds_pd["Appliances"].max()
    
    fig = px.scatter(
        preds_pd,
        x="Appliances",
        y="prediction",
        opacity=0.5,
        title=f"{model_name}: Actual vs. Predicted Energy Consumption",
        labels={
            "Appliances": "Actual Consumption (Wh)",
            "prediction": "Predicted Consumption (Wh)",
        },
        color_discrete_sequence=["#9467bd"],
    )
    # Add perfect prediction line
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="red", dash="dash"),
    ))
    fig.update_layout(template="plotly_white", legend_title_text="")
    return fig


def plot_residuals(preds_pd: pd.DataFrame, model_name: str) -> go.Figure:
    """Histogram of prediction residuals (actual − predicted).
    
    Args:
        preds_pd: Pandas DataFrame with 'Appliances' and 'prediction' columns.
        model_name: Display name for the chart title.
    
    Returns:
        Plotly histogram figure.
    """
    preds_pd = preds_pd.copy()
    preds_pd["residual"] = preds_pd["Appliances"] - preds_pd["prediction"]
    fig = px.histogram(
        preds_pd,
        x="residual",
        nbins=60,
        title=f"{model_name}: Residual Distribution (Actual − Predicted)",
        labels={"residual": "Residual (Wh)", "count": "Frequency"},
        color_discrete_sequence=["#d62728"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Zero")
    fig.update_layout(
        xaxis_title="Residual (Wh)",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    return fig


def plot_predictions_over_time(preds_pd: pd.DataFrame, model_name: str) -> go.Figure:
    """Line chart comparing actual and predicted values over the test period.
    
    Args:
        preds_pd: Pandas DataFrame with 'date', 'Appliances', and 'prediction' columns.
        model_name: Display name for the chart title.
    
    Returns:
        Plotly line figure.
    """
    preds_sorted = preds_pd.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=preds_sorted["date"], y=preds_sorted["Appliances"],
        mode="lines", name="Actual", line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=preds_sorted["date"], y=preds_sorted["prediction"],
        mode="lines", name="Predicted", line=dict(color="#ff7f0e", dash="dot"),
    ))
    fig.update_layout(
        title=f"{model_name}: Actual vs. Predicted Consumption Over Test Period",
        xaxis_title="Date",
        yaxis_title="Energy Consumption (Wh)",
        legend_title_text="",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_error_by_consumption_level(preds_pd: pd.DataFrame, model_name: str) -> go.Figure:
    """Box plot of absolute prediction error grouped by consumption level.
    
    Groups test observations into Low / Medium / High consumption tiers
    (by tertile of actual values) and shows error distribution per tier.
    
    Args:
        preds_pd: Pandas DataFrame with 'Appliances' and 'prediction' columns.
        model_name: Display name for the chart title.
    
    Returns:
        Plotly box figure.
    """
    preds_pd = preds_pd.copy()
    preds_pd["abs_error"] = (preds_pd["Appliances"] - preds_pd["prediction"]).abs()
    preds_pd["consumption_level"] = pd.qcut(
        preds_pd["Appliances"], q=3, labels=["Low", "Medium", "High"]
    )
    fig = px.box(
        preds_pd,
        x="consumption_level",
        y="abs_error",
        title=f"{model_name}: Absolute Error by Consumption Level",
        labels={
            "consumption_level": "Consumption Level",
            "abs_error": "Absolute Error (Wh)",
        },
        color="consumption_level",
        color_discrete_map={"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"},
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_correlation_heatmap(df_pd: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of numeric features.
    
    Args:
        df_pd: Pandas DataFrame with numeric columns.
    
    Returns:
        Plotly heatmap figure.
    """
    # Select numeric columns only, drop non-feature columns
    drop_cols = [c for c in ["rv1", "rv2"] if c in df_pd.columns]
    numeric_df = df_pd.drop(columns=drop_cols).select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        template="plotly_white",
        height=700,
    )
    return fig


def plot_feature_distributions(df_pd: pd.DataFrame, features: list) -> go.Figure:
    """Small-multiple histograms for selected environmental features.
    
    Args:
        df_pd: Pandas DataFrame.
        features: List of column names to plot (max ~6 for readability).
    
    Returns:
        Plotly figure with subplots.
    """
    n = len(features)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features)
    
    for i, feat in enumerate(features):
        r = i // cols + 1
        c = i % cols + 1
        fig.add_trace(
            go.Histogram(x=df_pd[feat].dropna(), name=feat, showlegend=False,
                         marker_color="#1f77b4"),
            row=r, col=c
        )
    
    fig.update_layout(
        title="Environmental Feature Distributions",
        template="plotly_white",
        height=300 * rows,
    )
    return fig
