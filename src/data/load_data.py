from __future__ import annotations

import pandas as pd

from ..config import DATE_COL, PRODUCT_COL, PRODUCT_VALUE, TARGET_COL

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load raw dataset from CSV and parse datetime column."""
    df = pd.read_csv(csv_path)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df

def filter_product(df: pd.DataFrame, product: str = PRODUCT_VALUE) -> pd.DataFrame:
    """Filter dataframe by product value."""
    if PRODUCT_COL not in df.columns:
        raise ValueError(f"Missing column: {PRODUCT_COL}")
    return df[df[PRODUCT_COL] == product].copy()

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning shared across notebooks."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    out = df[df[TARGET_COL].notna()].copy()
    out = out[out[TARGET_COL] >= 0]

    if DATE_COL in out.columns:
        out = out.sort_values(DATE_COL)

    return out

def load_amst_clean(csv_path: str) -> pd.DataFrame:
    """Convenience helper: load → filter AMST → basic cleaning."""
    df = load_dataset(csv_path)
    df = filter_product(df, product=PRODUCT_VALUE)
    df = basic_cleaning(df)
    return df
