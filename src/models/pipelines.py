from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def build_dummy_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    """Cria pipeline baseline com DummyRegressor (prediz a média do target)."""
    return Pipeline([
        ("preprocess", preprocess),
        ("model", DummyRegressor(strategy="mean")),
    ])


def build_ridge_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    """
    Cria pipeline com Ridge Regression.
    Observação: Ridge é determinístico e não utiliza random_state.
    """
    return Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge()),
    ])


def build_rf_pipeline(preprocess: ColumnTransformer, random_state: int) -> Pipeline:
    """Cria pipeline com RandomForestRegressor."""
    return Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1
        )),
    ])


def build_gbr_pipeline(preprocess: ColumnTransformer, random_state: int) -> Pipeline:
    """Cria pipeline com GradientBoostingRegressor."""
    return Pipeline([
        ("preprocess", preprocess),
        ("model", GradientBoostingRegressor(random_state=random_state)),
    ])


def build_xgb_pipeline(preprocess: ColumnTransformer, random_state: int) -> Pipeline:
    """
    Cria pipeline com XGBoost (se disponível).
    """
    try:
        from xgboost import XGBRegressor
    except Exception as e:  # pragma: no cover
        raise ImportError("xgboost não está instalado. Instale com: pip install xgboost") from e

    return Pipeline([
        ("preprocess", preprocess),
        ("model", XGBRegressor(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="reg:squarederror",
            n_jobs=-1
        )),
    ])


def build_lgbm_pipeline(preprocess: ColumnTransformer, random_state: int) -> Pipeline:
    """
    Cria pipeline com LightGBM (se disponível).

    Observação:
    - logs do LightGBM podem ser controlados via `verbosity=-1` (quando suportado).
    """
    try:
        from lightgbm import LGBMRegressor
    except Exception as e:  # pragma: no cover
        raise ImportError("lightgbm não está instalado. Instale com: pip install lightgbm") from e

    return Pipeline([
        ("preprocess", preprocess),
        ("model", LGBMRegressor(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbosity=-1,  # se sua versão suportar
        )),
    ])
