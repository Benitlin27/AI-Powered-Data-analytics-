import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def generate_charts(df: pd.DataFrame, eda_stats: dict) -> dict:
    """
    Auto-generate charts based on data types.
    Returns dict of {chart_title: PNG_bytes}.
    """
    charts = {}
    numeric_cols = eda_stats.get("numeric_cols", [])

    # 1. Histograms for numeric columns (up to 6)
    cols_to_plot = numeric_cols[:6]
    if cols_to_plot:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for i, col in enumerate(cols_to_plot):
            axes[i].hist(df[col].dropna(), bins=30, color="#4F8EF7", edgecolor="white", alpha=0.85)
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
        for j in range(len(cols_to_plot), 6):
            axes[j].set_visible(False)
        plt.suptitle("Distributions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        charts["Distributions"] = _fig_to_bytes(fig)

    # 2. Correlation heatmap
    corr = eda_stats.get("correlation")
    if corr is not None and not corr.empty and len(corr) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        charts["Correlation Matrix"] = _fig_to_bytes(fig)

    # 3. Boxplots
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(12, 5))
        df[numeric_cols[:8]].boxplot(ax=ax, vert=True)
        ax.set_title("Boxplots — Numeric Features", fontsize=14, fontweight="bold")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Value")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        charts["Boxplots"] = _fig_to_bytes(fig)

    # 4. Missing values bar chart
    missing = {k: v for k, v in eda_stats.get("missing", {}).items() if v > 0}
    if missing:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(missing.keys(), missing.values(), color="#E05C5C", edgecolor="white")
        ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        charts["Missing Values"] = _fig_to_bytes(fig)

    # 5. Pairplot (top 4 numeric cols only)
    if len(numeric_cols) >= 3:
        sample = df[numeric_cols[:4]].sample(min(500, len(df)), random_state=42)
        fig = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.5}).fig
        fig.suptitle("Pair Plot", y=1.02, fontsize=14, fontweight="bold")
        charts["Pair Plot"] = _fig_to_bytes(fig)

    return charts


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
