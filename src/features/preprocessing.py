from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-inspired features used in this project."""
    out = df.copy()

    roast_amt = "Roast amount (kg)"
    m1_amt = "1st (base) malt amount (kg)"
    m2_amt = "2nd (base) malt amount (kg)"
    mt_temp = "MT – Temperature (°C)"
    mt_time = "MT – Time (s)"
    wk_temp = "WK – Temperature (°C)"
    wk_time = "WK – Time (s)"

    if all(c in out.columns for c in [roast_amt, m1_amt, m2_amt]):
        out["total_malt"] = out[roast_amt] + out[m1_amt] + out[m2_amt]
        out["roast_pct"] = np.where(out["total_malt"] > 0, out[roast_amt] / out["total_malt"], 0.0)

    if mt_temp in out.columns and mt_time in out.columns:
        out["mt_energy"] = out[mt_temp] * out[mt_time]

    if wk_temp in out.columns and wk_time in out.columns:
        out["wk_energy"] = out[wk_temp] * out[wk_time]

    return out

def build_preprocess_pipeline(feature_cols: List[str]) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric features."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
    )

def temporal_train_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Temporal split using the dataframe order (assumes df is already time-sorted)."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    n = len(df)
    split_idx = int(n * (1 - test_size))

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    return (
        train[feature_cols],
        test[feature_cols],
        train[target_col],
        test[target_col],
    )
