import pandas as pd
import json


def load_dataset(uploaded_file):
    """
    Parse CSV, Excel, or JSON upload.
    Returns (DataFrame, schema_list).
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        content = json.load(uploaded_file)
        if isinstance(content, list):
            df = pd.DataFrame(content)
        else:
            df = pd.DataFrame([content])
    else:
        raise ValueError(f"Unsupported file type: {name}")

    # Try to parse obvious datetime columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    schema = _build_schema(df)
    return df, schema


def _build_schema(df):
    schema = []
    for col in df.columns:
        schema.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "unique_values": int(df[col].nunique()),
            "sample": str(df[col].dropna().iloc[0]) if df[col].notna().any() else "N/A",
        })
    return schema
