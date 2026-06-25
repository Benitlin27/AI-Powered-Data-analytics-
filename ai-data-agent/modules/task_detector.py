import pandas as pd
import numpy as np


def detect_task(df: pd.DataFrame, target_col: str | None) -> str:
    """
    Detect ML task type:
      - 'classification'  → target is categorical or low-cardinality integer
      - 'regression'      → target is continuous numeric
      - 'clustering'      → no target column provided
    """
    if target_col is None:
        return "clustering"

    target = df[target_col]

    # If dtype is object/bool → classification
    if target.dtype == object or target.dtype == bool:
        return "classification"

    # If dtype is numeric, check cardinality
    n_unique = target.nunique()
    n_rows = len(target)

    # Low cardinality → classification
    if n_unique <= 20 or (n_unique / n_rows < 0.05):
        return "classification"

    # High cardinality continuous → regression
    return "regression"
