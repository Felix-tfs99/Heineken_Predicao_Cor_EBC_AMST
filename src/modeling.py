from typing import Dict, List

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import joblib


# =======================================================================
# MÉTRICAS E PRÉ-PROCESSAMENTO
# =======================================================================

def eval_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "model"
) -> Dict[str, float]:
    """
    Calcula métricas de regressão padrão e imprime de forma amigável.

    MAE  = mean absolute error
    RMSE = root mean squared error (raiz do MSE)
    R²   = coeficiente de determinação
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"{label:<25} -> MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | R²: {r2:6.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def build_preprocess_pipeline(feature_cols: List[str]) -> ColumnTransformer:
    """
    Cria o ColumnTransformer para tratar as variáveis numéricas:
    - Imputação (mediana)
    - Padronização (StandardScaler)

    Se no futuro houver variáveis categóricas, é só estender aqui.
    """
    numeric_features = feature_cols

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="drop"  # Garante que só as features definidas sejam usadas
    )

    return preprocess


# =======================================================================
# MODELOS - CONSTRUÇÃO DOS PIPELINES
# =======================================================================

def build_dummy_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    """
    Pipeline de baseline com DummyRegressor (prediz a média do target).
    """
    return Pipeline([
        ("preprocess", preprocess),
        ("model", DummyRegressor(strategy="mean"))
    ])


def build_ridge_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    """
    Pipeline com Ridge Regression (modelo linear regularizado).
    Ridge é determinístico e não utiliza random_state.
    """
    return Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge())
    ])


def build_rf_pipeline(preprocess: ColumnTransformer, random_state: int) -> Pipeline:
    """
    Pipeline com RandomForestRegressor.
    """
    return Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1
        ))
    ])


def build_gbr_pipeline(preprocess: ColumnTransformer, random_state: int) -> Pipeline:
    """
    Pipeline com GradientBoostingRegressor.
    """
    return Pipeline([
        ("preprocess", preprocess),
        ("model", GradientBoostingRegressor(random_state=random_state))
    ])


def build_xgb_pipeline(preprocess: ColumnTransformer, random_state: int) -> Pipeline:
    """
    Pipeline com XGBRegressor.
    Usa configuração base; hiperparâmetros serão tunados depois.
    """
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
        ))
    ])


def build_lgbm_pipeline(preprocess: ColumnTransformer, random_state: int) -> Pipeline:
    """
    Pipeline com LGBMRegressor.
    """
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LGBMRegressor(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        ))
    ])


# =======================================================================
# TREINO, TUNING E COMPARAÇÃO
# =======================================================================

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocess: ColumnTransformer
) -> Dict[str, object]:
    """
    Treina o baseline DummyRegressor e avalia no teste.
    Retorna o modelo treinado e suas métricas.
    """
    dummy_pipe = build_dummy_pipeline(preprocess)
    dummy_pipe.fit(X_train, y_train)
    y_pred = dummy_pipe.predict(X_test)
    metrics = eval_regression(y_test, y_pred, label="Dummy (mean)")
    return {"model": dummy_pipe, "metrics": metrics}


def train_ridge_default(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer
) -> Pipeline:
    """
    Treina um Ridge Regression simples (default).
    """
    ridge_pipe = build_ridge_pipeline(preprocess)
    ridge_pipe.fit(X_train, y_train)
    return ridge_pipe


def train_gbr_default(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer,
    random_state: int
) -> Pipeline:
    """
    Treina um GradientBoostingRegressor com hiperparâmetros default.
    (Se houver tempo, pode fazer um RandomizedSearch também.)
    """
    gbr_pipe = build_gbr_pipeline(preprocess, random_state)
    gbr_pipe.fit(X_train, y_train)
    return gbr_pipe


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer,
    random_state: int,
    n_iter: int = 40
) -> RandomizedSearchCV:
    """
    Realiza RandomizedSearchCV para RandomForest com validação temporal.
    Usa RMSE (negativo) como métrica de otimização.
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
        verbose=2
    )

    search.fit(X_train, y_train)

    print("Melhores hiperparâmetros RF:", search.best_params_)
    print("Melhor RMSE (CV):", -search.best_score_)

    return search


def tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer,
    random_state: int,
    n_iter: int = 40
) -> RandomizedSearchCV:
    """
    RandomizedSearchCV para XGBRegressor com validação temporal.
    Otimiza RMSE (negativo).
    """
    xgb_pipe = build_xgb_pipeline(preprocess, random_state)

    param_distributions = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5]
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
        verbose=2
    )

    search.fit(X_train, y_train)

    print("Melhores hiperparâmetros XGB:", search.best_params_)
    print("Melhor RMSE (CV):", -search.best_score_)

    return search


def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocess: ColumnTransformer,
    random_state: int,
    n_iter: int = 40
) -> RandomizedSearchCV:
    """
    RandomizedSearchCV para LGBMRegressor com validação temporal.
    Otimiza RMSE (negativo).
    """
    lgbm_pipe = build_lgbm_pipeline(preprocess, random_state)

    param_distributions = {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [-1, 4, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__num_leaves": [31, 63, 127],
        "model__min_child_samples": [10, 20, 30]
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
        verbose=2
    )

    search.fit(X_train, y_train)

    print("Melhores hiperparâmetros LGBM:", search.best_params_)
    print("Melhor RMSE (CV):", -search.best_score_)

    return search


# =======================================================================
# SELEÇÃO, INTERPRETAÇÃO E SALVAMENTO
# =======================================================================

def select_best_model(results_df: pd.DataFrame, metric: str = "rmse") -> str:
    """
    Seleciona o melhor modelo com base em uma métrica (quanto menor, melhor).
    """
    best_model_name = results_df[metric].idxmin()
    print(f"Melhor modelo segundo {metric.upper()}: {best_model_name}")
    return best_model_name


def plot_feature_importance(
    model: Pipeline,
    feature_cols: List[str],
    top_n: int = 10
) -> None:
    """
    Plota as top_n features mais importantes para modelos baseados em árvore.
    """
    final_model = model.named_steps["model"]

    if not hasattr(final_model, "feature_importances_"):
        print("O modelo não possui atributo feature_importances_.")
        return

    importances = final_model.feature_importances_
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    fi.head(top_n).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Importância das features (top {top_n})")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.show()


def save_model(model: Pipeline, path: str) -> None:
    """
    Salva o modelo treinado em disco usando joblib.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo salvo em: {path}")


def load_model(path: str) -> Pipeline:
    """
    Carrega um modelo salvo.
    """
    model = joblib.load(path)
    return model


def predict_color_for_batch(
    model: Pipeline,
    batch_df: pd.DataFrame,
    feature_cols: List[str]
) -> np.ndarray:
    """
    Recebe um DataFrame de novos lotes com as mesmas colunas de features
    e retorna as previsões de Color (EBC).

    IMPORTANTE: batch_df deve ter as mesmas colunas usadas no treino
    (incluindo as features derivadas, se você não encapsular isso em uma função
    de pré-processamento de dados crus).
    """
    X = batch_df[feature_cols]
    preds = model.predict(X)
    return preds
