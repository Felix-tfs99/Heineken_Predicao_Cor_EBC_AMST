from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .evaluation import eval_regression
from .pipelines import build_dummy_pipeline, build_ridge_pipeline, build_gbr_pipeline


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocess: ColumnTransformer
) -> Dict[str, Any]:
    """
    Treina o baseline DummyRegressor e avalia no conjunto de teste.

    Retorna
    -------
    Dict com:
    - model: pipeline treinado
    - metrics: métricas no teste
    """
    model = build_dummy_pipeline(preprocess)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = eval_regression(y_test, y_pred, label="Dummy (mean)")
    return {"model": model, "metrics": metrics}


def train_ridge_default(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer
) -> Pipeline:
    """
    Treina um Ridge Regression com hiperparâmetros default.
    """
    model = build_ridge_pipeline(preprocess)
    model.fit(X_train, y_train)
    return model


def train_gbr_default(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer,
    random_state: int
) -> Pipeline:
    """
    Treina um GradientBoostingRegressor com hiperparâmetros default.
    """
    model = build_gbr_pipeline(preprocess, random_state=random_state)
    model.fit(X_train, y_train)
    return model
