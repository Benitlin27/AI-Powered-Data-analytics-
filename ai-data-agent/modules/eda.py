import pandas as pd
import numpy as np


def run_eda(df: pd.DataFrame) -> dict:
    """
    Compute:
      - Descriptive stats
      - Correlation matrix
      - Missing value counts
      - Skewness & kurtosis
      - Top correlated feature pairs
    Returns a dict of all results.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    describe = df.describe(include="all").fillna("").round(4)
    correlation = numeric_df.corr().round(4) if not numeric_df.empty else pd.DataFrame()
    missing = df.isna().sum().to_dict()
    skewness = numeric_df.skew().round(4).to_dict() if not numeric_df.empty else {}
    kurtosis = numeric_df.kurtosis().round(4).to_dict() if not numeric_df.empty else {}

    # Top correlated pairs (exclude self-correlation)
    top_corr_pairs = []
    if not correlation.empty:
        corr_unstacked = correlation.abs().unstack()
        corr_unstacked = corr_unstacked[corr_unstacked < 1.0].sort_values(ascending=False)
        seen = set()
        for (c1, c2), val in corr_unstacked.items():
            pair = tuple(sorted([c1, c2]))
            if pair not in seen:
                seen.add(pair)
                top_corr_pairs.append({"feature_1": c1, "feature_2": c2, "correlation": round(float(val), 4)})
            if len(top_corr_pairs) >= 10:
                break

    return {
        "describe": describe,
        "correlation": correlation,
        "missing": missing,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "top_corr_pairs": top_corr_pairs,
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": list(numeric_df.columns),
        "all_cols": list(df.columns),
    }
