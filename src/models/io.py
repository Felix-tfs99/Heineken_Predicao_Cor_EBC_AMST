from __future__ import annotations

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def save_model(model: Pipeline, path: str) -> None:
    """
    Salva um modelo treinado em disco usando joblib.

    Parâmetros
    ----------
    model : Pipeline
        Pipeline treinado.
    path : str
        Caminho de saída (ex.: "artifacts/models/best_model.pkl").
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(model, path)
    print(f"Modelo salvo em: {path}")


def load_model(path: str) -> Pipeline:
    """
    Carrega um modelo salvo com joblib.

    Parâmetros
    ----------
    path : str
        Caminho do artefato.

    Retorna
    -------
    Pipeline
        Modelo carregado.
    """
    return joblib.load(path)


def predict_color_for_batch(
    model: Pipeline,
    batch_df: pd.DataFrame,
    feature_cols: List[str]
) -> np.ndarray:
    """
    Gera predição de Color (EBC) para novos lotes.

    Parâmetros
    ----------
    model : Pipeline
        Pipeline treinado (com preprocess + modelo).
    batch_df : pd.DataFrame
        DataFrame com lotes novos.
    feature_cols : List[str]
        Colunas usadas como features no treino.

    Retorna
    -------
    np.ndarray
        Predições do target.
    """
    X = batch_df[feature_cols]
    return model.predict(X)
