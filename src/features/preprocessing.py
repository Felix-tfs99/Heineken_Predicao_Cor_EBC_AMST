from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas orientadas pelo processo (quando as colunas existirem).

    Features geradas (se possível):
    - total_malt: soma das massas de malte
    - roast_pct: proporção de malte torrado no total (evita divisão por zero)
    - mt_energy: proxy simples de energia térmica no Malt Cooker (Temp * Time)
    - wk_energy: proxy simples de energia térmica no Wort Cooker (Temp * Time)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorna
    -------
    pd.DataFrame
        DataFrame com novas features (cópia).
    """
    out = df.copy()

    roast = "Roast amount (kg)"
    m1 = "1st (base) malt amount (kg)"
    m2 = "2nd (base) malt amount (kg)"

    if all(c in out.columns for c in [roast, m1, m2]):
        out["total_malt"] = out[roast].fillna(0) + out[m1].fillna(0) + out[m2].fillna(0)
        denom = out["total_malt"].replace(0, np.nan)
        out["roast_pct"] = (out[roast].fillna(0) / denom).fillna(0.0)

    mt_t = "MT – Temperature (°C)"
    mt_time = "MT – Time (s)"
    if all(c in out.columns for c in [mt_t, mt_time]):
        out["mt_energy"] = out[mt_t].astype(float) * out[mt_time].astype(float)

    wk_t = "WK – Temperature (°C)"
    wk_time = "WK – Time (s)"
    if all(c in out.columns for c in [wk_t, wk_time]):
        out["wk_energy"] = out[wk_t].astype(float) * out[wk_time].astype(float)

    return out


def build_preprocess_pipeline(feature_cols: List[str]) -> ColumnTransformer:
    """
    Cria um ColumnTransformer para variáveis numéricas:
    - imputação por mediana
    - padronização (StandardScaler)

    Parâmetros
    ----------
    feature_cols : List[str]
        Lista de colunas numéricas a serem usadas.

    Retorna
    -------
    ColumnTransformer
        Transformador pronto para ser usado em pipelines.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocess = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop"
    )
    return preprocess


def temporal_train_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide o dataset em treino/teste preservando ordem temporal (sem embaralhar).

    Importante: assume que o DataFrame já está ordenado por tempo.
    Caso não esteja, ordene antes no notebook (ou no load/clean).

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame já ordenado.
    feature_cols : List[str]
        Colunas de entrada (X).
    target_col : str
        Coluna alvo (y).
    test_size : float, default=0.2
        Fração do final da série que será usada como teste (0 < test_size < 1).

    Retorna
    -------
    (X_train, X_test, y_train, y_test)
    """
    if not (0 < test_size < 1):
        raise ValueError("test_size deve estar no intervalo (0, 1).")

    n = len(df)
    if n < 5:
        raise ValueError("Dataset muito pequeno para split temporal com segurança.")

    split_idx = int(np.floor((1 - test_size) * n))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Split inválido. Ajuste test_size ou verifique o tamanho do dataset.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test
