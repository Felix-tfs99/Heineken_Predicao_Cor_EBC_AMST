from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from .pipelines import build_rf_pipeline, build_xgb_pipeline, build_lgbm_pipeline


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer,
    random_state: int,
    n_iter: int = 40,
    verbose: int = 0
) -> RandomizedSearchCV:
    """
    Ajusta hiperparâmetros do RandomForest via RandomizedSearchCV com validação temporal.

    Métrica otimizada: RMSE (negativo) via `neg_root_mean_squared_error`.

    Parâmetros
    ----------
    verbose : int
        Verbose do RandomizedSearchCV (0 silencioso, 1/2+ mais logs).
        Atenção: scikit-learn NÃO aceita verbose=-1.
    """
    rf_pipe = build_rf_pipeline(preprocess, random_state)

    param_distributions = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 4, 6, 8, 10],
        "model__min_samples_leaf": [1, 2, 3, 4],
        "model__max_features": ["sqrt", "log2", 0.8],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        random_state=random_state,
        verbose=verbose,
    )

    search.fit(X_train, y_train)
    return search


def tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer,
    random_state: int,
    n_iter: int = 40,
    verbose: int = 0
) -> RandomizedSearchCV:
    """
    Ajusta hiperparâmetros do XGBoost via RandomizedSearchCV com validação temporal.
    """
    xgb_pipe = build_xgb_pipeline(preprocess, random_state)

    param_distributions = {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        xgb_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        random_state=random_state,
        verbose=verbose,
    )

    search.fit(X_train, y_train)
    return search


def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer,
    random_state: int,
    n_iter: int = 40,
    verbose: int = 0,
    lgbm_verbosity: Optional[int] = -1
) -> RandomizedSearchCV:
    """
    Ajusta hiperparâmetros do LightGBM via RandomizedSearchCV com validação temporal.

    Parâmetros
    ----------
    verbose : int
        Verbose do RandomizedSearchCV (0 silencioso).
    lgbm_verbosity : int | None
        Controle de logs do LightGBM (quando suportado pelo wrapper).
        Use -1 para silenciar. Se sua versão não suportar, defina None e use `lightgbm.set_config`.
    """
    lgbm_pipe = build_lgbm_pipeline(preprocess, random_state)

    if lgbm_verbosity is not None:
        # garante que o log do LGBM seja reduzido mesmo durante CV
        try:
            lgbm_pipe.set_params(model__verbosity=lgbm_verbosity)
        except Exception:
            pass

    param_distributions = {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [-1, 4, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__num_leaves": [31, 63, 127],
        "model__min_child_samples": [3, 5, 10, 20, 30],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        lgbm_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        random_state=random_state,
        verbose=verbose,
    )

    search.fit(X_train, y_train)
    return search
