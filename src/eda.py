"""
eda.py
------
Exploratory Data Analysis (EDA) and model-evaluation plots.

Each function draws one figure. Pass `save_dir` to save it as a PNG and/or
`show=True` to display it (in a notebook). If neither is given the figure
is simply closed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from src.preprocessing import BASE_NUMERIC_FEATURES, TARGET_GROWTH

sns.set_theme(style="whitegrid", context="notebook")

# Show prices in lakh (1 lakh = 1,00,000) on chart axes
LAKH = FuncFormatter(lambda x, _: f"{x / 1e5:,.0f}")
PRICE_LABEL = "Price (Rs. lakh)"


def _finish(fig, filename, save_dir, show):
    """Save / show / close a figure."""
    fig.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / filename, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------- EDA plots
def plot_price_distribution(df, save_dir=None, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df["current_price"], bins=50, kde=True, ax=axes[0], color="#3b7dd8")
    axes[0].xaxis.set_major_formatter(LAKH)
    axes[0].set_xlabel(PRICE_LABEL)
    axes[0].set_title("Distribution of current house prices (right-skewed)")
    sns.histplot(df[TARGET_GROWTH], bins=50, kde=True, ax=axes[1], color="#2a9d8f")
    axes[1].set_xlabel("Annual price growth (%)")
    axes[1].set_title("Distribution of annual price growth")
    _finish(fig, "01_price_distribution.png", save_dir, show)


def plot_area_vs_price(df, save_dir=None, show=False):
    sample = df.dropna(subset=["area_sqft", "property_type"]).sample(min(3000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=sample, x="area_sqft", y="current_price", hue="property_type",
                    alpha=0.5, s=18, ax=ax)
    ax.yaxis.set_major_formatter(LAKH)
    ax.set_ylabel(PRICE_LABEL)
    ax.set_xlabel("Area (sqft)")
    ax.set_title("Area vs price")
    _finish(fig, "02_area_vs_price.png", save_dir, show)


def plot_location_vs_price(df, save_dir=None, show=False):
    data = df.dropna(subset=["location_score"]).copy()
    data["location_bucket"] = data["location_score"].round().astype(int)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=data, x="location_bucket", y="current_price", ax=ax,
                color="#8ecae6", showfliers=False)
    ax.yaxis.set_major_formatter(LAKH)
    ax.set_ylabel(PRICE_LABEL)
    ax.set_xlabel("Location score (rounded)")
    ax.set_title("Location score vs price")
    _finish(fig, "03_location_vs_price.png", save_dir, show)


def plot_age_vs_price(df, save_dir=None, show=False):
    data = df.dropna(subset=["age_years"]).copy()
    sample = data.sample(min(3000, len(data)), random_state=42)
    data["age_bin"] = (data["age_years"] // 5) * 5
    trend = data.groupby("age_bin")["current_price"].median().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(sample["age_years"], sample["current_price"], s=10, alpha=0.3, color="gray")
    ax.plot(trend["age_bin"] + 2.5, trend["current_price"], color="crimson", lw=2.5,
            marker="o", label="Median price per 5-year band")
    ax.yaxis.set_major_formatter(LAKH)
    ax.set_ylabel(PRICE_LABEL)
    ax.set_xlabel("Property age (years)")
    ax.set_title("Age vs price")
    ax.legend()
    _finish(fig, "04_age_vs_price.png", save_dir, show)


def plot_correlation_heatmap(df, save_dir=None, show=False):
    cols = BASE_NUMERIC_FEATURES + ["current_price", TARGET_GROWTH]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.4,
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation heatmap (features and targets)")
    _finish(fig, "05_correlation_heatmap.png", save_dir, show)


def plot_feature_distributions(df, save_dir=None, show=False):
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.ravel()
    for ax, col in zip(axes, BASE_NUMERIC_FEATURES):
        sns.histplot(df[col].dropna(), bins=30, ax=ax, color="#457b9d")
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes[len(BASE_NUMERIC_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Feature distributions (note the very different scales)", y=1.0, fontsize=14)
    _finish(fig, "06_feature_distributions.png", save_dir, show)


def plot_growth_drivers(df, save_dir=None, show=False):
    sample = df.sample(min(2500, len(df)), random_state=42)
    drivers = [("economic_growth_rate", "Economic growth rate (%)"),
               ("location_score", "Location score"),
               ("distance_to_metro_km", "Distance to metro (km)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (col, label) in zip(axes, drivers):
        sns.regplot(data=sample, x=col, y=TARGET_GROWTH, ax=ax, scatter_kws={"alpha": 0.25, "s": 10},
                    line_kws={"color": "crimson"})
        ax.set_xlabel(label)
        ax.set_ylabel("Annual price growth (%)")
    fig.suptitle("What drives future price appreciation?", y=1.02)
    _finish(fig, "07_growth_drivers.png", save_dir, show)


def plot_historical_trend(hist_df, projected_rate_pct, years=5, save_dir=None, show=False):
    """Historical price index + an ILLUSTRATIVE projection at the model's average growth rate."""
    last_year = int(hist_df["year"].iloc[-1])
    last_price = float(hist_df["avg_price_per_sqft"].iloc[-1])
    future_years = np.arange(last_year, last_year + years + 1)
    future_prices = last_price * (1 + projected_rate_pct / 100) ** np.arange(0, years + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_df["year"], hist_df["avg_price_per_sqft"], marker="o", lw=2.5, label="Historical (synthetic)")
    ax.plot(future_years, future_prices, marker="o", ls="--", lw=2.5, color="crimson",
            label=f"Projected at {projected_rate_pct:.1f}% / year (model average)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average price per sqft (Rs.)")
    ax.set_title("Historical prices and projected trend")
    ax.legend()
    _finish(fig, "08_historical_and_projected_trend.png", save_dir, show)


def run_eda(df, hist_df=None, save_dir=None, show=False):
    """Draw every EDA figure."""
    plot_price_distribution(df, save_dir, show)
    plot_area_vs_price(df, save_dir, show)
    plot_location_vs_price(df, save_dir, show)
    plot_age_vs_price(df, save_dir, show)
    plot_correlation_heatmap(df, save_dir, show)
    plot_feature_distributions(df, save_dir, show)
    plot_growth_drivers(df, save_dir, show)
    if hist_df is not None:
        plot_historical_trend(hist_df, df[TARGET_GROWTH].mean(), save_dir=save_dir, show=show)


# ----------------------------------------------------- model-evaluation plots
def plot_actual_vs_predicted(y_true, y_pred, title, unit="price", filename="actual_vs_predicted.png",
                             save_dir=None, show=False):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.4, color="#3b7dd8")
    ax.plot([lo, hi], [lo, hi], color="crimson", lw=2, label="Perfect prediction")
    if unit == "price":
        ax.xaxis.set_major_formatter(LAKH)
        ax.yaxis.set_major_formatter(LAKH)
        ax.set_xlabel("Actual price (Rs. lakh)")
        ax.set_ylabel("Predicted price (Rs. lakh)")
    else:
        ax.set_xlabel("Actual annual growth (%)")
        ax.set_ylabel("Predicted annual growth (%)")
    ax.set_title(title)
    ax.legend()
    _finish(fig, filename, save_dir, show)


def plot_residuals(y_true, y_pred, title, unit="price", filename="residuals.png", save_dir=None, show=False):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].scatter(y_pred, residuals, s=10, alpha=0.4, color="#2a9d8f")
    axes[0].axhline(0, color="crimson", lw=2)
    axes[0].set_title("Residuals vs predicted (should be a random cloud around 0)")
    axes[0].set_ylabel("Error = actual - predicted")
    if unit == "price":
        axes[0].xaxis.set_major_formatter(LAKH)
        axes[0].yaxis.set_major_formatter(LAKH)
        axes[0].set_xlabel("Predicted price (Rs. lakh)")
        axes[0].set_ylabel("Error (Rs. lakh)")
        axes[1].xaxis.set_major_formatter(LAKH)
        axes[1].set_xlabel("Error (Rs. lakh)")
    else:
        axes[0].set_xlabel("Predicted annual growth (%)")
        axes[1].set_xlabel("Error (percentage points)")
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color="#e9c46a")
    axes[1].set_title("Distribution of errors")
    fig.suptitle(title, y=1.02)
    _finish(fig, filename, save_dir, show)


def plot_model_comparison(results, title, filename, save_dir=None, show=False):
    """Bar charts of RMSE and R2 for every model (results = comparison DataFrame)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.barplot(data=results, x="Model", y="RMSE", ax=axes[0], color="#e76f51")
    axes[0].set_title("RMSE (lower is better)")
    sns.barplot(data=results, x="Model", y="R2", ax=axes[1], color="#2a9d8f")
    axes[1].set_title("R2 (higher is better)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.set_xlabel("")
    fig.suptitle(title, y=1.02)
    _finish(fig, filename, save_dir, show)


def plot_feature_importance(fitted_pipeline, feature_names, title, filename, top_n=12, save_dir=None, show=False):
    """Bar chart of the most important features (tree models) or |coefficients| (linear model)."""
    model = fitted_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)
    else:
        return
    imp = pd.Series(values, index=feature_names).sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    imp.plot.barh(ax=ax, color="#457b9d")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    _finish(fig, filename, save_dir, show)
