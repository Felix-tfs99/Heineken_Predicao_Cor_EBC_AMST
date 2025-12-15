from __future__ import annotations

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def save_model(model: Pipeline, path: str) -> None:
    """Save trained model to disk using joblib."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo salvo em: {path}")

def load_model(path: str) -> Pipeline:
    """Load a saved model."""
    return joblib.load(path)

def predict_color_for_batch(model: Pipeline, batch_df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Predict Color (EBC) for new batches."""
    X = batch_df[feature_cols]
    return model.predict(X)
