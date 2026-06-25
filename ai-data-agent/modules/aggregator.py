import pandas as pd
import numpy as np


def aggregate_results(eda_stats: dict, model_results: dict, df: pd.DataFrame) -> dict:
    """
    Bundle EDA stats + model results into one clean payload
    to pass into the LLM prompt.
    """
    shape = eda_stats.get("shape", (0, 0))
    numeric_cols = eda_stats.get("numeric_cols", [])
    missing = eda_stats.get("missing", {})
    total_missing = sum(missing.values())

    skewness = eda_stats.get("skewness", {})
    highly_skewed = [k for k, v in skewness.items() if abs(v) > 1.0]

    top_corr = eda_stats.get("top_corr_pairs", [])[:5]

    # Model summary
    task = model_results.get("task", "unknown")
    best_model = model_results.get("best_model", "N/A")
    scores = model_results.get("scores", {})
    fi = model_results.get("feature_importance")

    top_features = []
    if fi is not None and not fi.empty:
        top_features = fi.head(5)[["feature", "importance"]].to_dict(orient="records")

    return {
        "dataset_shape": {"rows": shape[0], "cols": shape[1]},
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "total_missing_values": total_missing,
        "highly_skewed_features": highly_skewed,
        "top_correlations": top_corr,
        "task_type": task,
        "best_model": best_model,
        "model_scores": scores,
        "top_features": top_features,
    }
