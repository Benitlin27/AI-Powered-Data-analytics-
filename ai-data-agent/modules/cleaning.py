import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame):
    """
    Clean the DataFrame:
      - Drop columns that are >80% null
      - Fill numeric nulls with median
      - Fill categorical nulls with mode
      - Remove outliers (IQR method) on numeric columns
      - Encode categorical columns with LabelEncoder
    Returns (clean_df, log_list).
    """
    df = df.copy()
    log = []

    # 1. Drop mostly-null columns
    threshold = 0.8
    mostly_null = [c for c in df.columns if df[c].isna().mean() > threshold]
    if mostly_null:
        df.drop(columns=mostly_null, inplace=True)
        log.append(f"Dropped {len(mostly_null)} column(s) with >80% nulls: {mostly_null}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 2. Fill numeric nulls with median
    for col in numeric_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            log.append(f"Filled {n_null} nulls in '{col}' with median ({median_val:.2f})")

    # 3. Fill categorical nulls with mode
    for col in cat_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            log.append(f"Filled {n_null} nulls in '{col}' with mode ('{mode_val}')")

    # 4. Remove outliers (IQR method) on numeric cols with reasonable range
    rows_before = len(df)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            log.append(f"Removed {outliers} outliers from '{col}' (IQR × 3 method)")
    if len(df) < rows_before:
        log.append(f"Row count after outlier removal: {rows_before} → {len(df)}")

    # 5. Encode categoricals with LabelEncoder
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            log.append(f"Label-encoded '{col}' ({df[col].nunique()} unique values)")

    # 6. Drop duplicate rows
    dups = df.duplicated().sum()
    if dups > 0:
        df.drop_duplicates(inplace=True)
        log.append(f"Removed {dups} duplicate rows")

    df.reset_index(drop=True, inplace=True)

    if not log:
        log.append("Data was already clean — no changes needed.")

    return df, log
