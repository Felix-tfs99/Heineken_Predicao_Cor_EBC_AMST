from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eval_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "modelo",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calcula métricas padrão de regressão.

    Métricas:
    - MAE  (mean absolute error)
    - RMSE (raiz do MSE)
    - R²   (coeficiente de determinação)

    Parâmetros
    ----------
    y_true, y_pred : array-like
        Valores reais e preditos.
    label : str
        Nome do modelo (apenas para impressão).
    verbose : bool
        Se True, imprime métricas formatadas.

    Retorna
    -------
    Dict[str, float]
        {"mae": ..., "rmse": ..., "r2": ...}
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    out = {"mae": float(mae), "rmse": rmse, "r2": r2}

    if verbose:
        print(f"{label:<25} -> MAE: {out['mae']:.3f} | RMSE: {out['rmse']:.3f} | R²: {out['r2']:.3f}")

    return out


def evaluate_by_bins(
    y_true: pd.Series,
    y_pred: np.ndarray,
    q_low: float,
    q_high: float,
    label: str = "modelo"
) -> pd.DataFrame:
    """
    Avalia o erro por faixas do target (low / mid / high) usando thresholds baseados em quantis.

    Parâmetros
    ----------
    y_true : pd.Series
        Target real.
    y_pred : np.ndarray
        Predição do modelo.
    q_low, q_high : float
        Limiares (ex.: 10% e 90% do y_train).
    label : str
        Nome do modelo.

    Retorna
    -------
    pd.DataFrame
        Tabela com n, MAE, RMSE e média do target por faixa.
    """
    df_eval = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred})
    df_eval["abs_error"] = np.abs(df_eval["y_true"] - df_eval["y_pred"])
    df_eval["sq_error"] = (df_eval["y_true"] - df_eval["y_pred"]) ** 2

    df_eval["bin"] = np.where(
        df_eval["y_true"] <= q_low, "low",
        np.where(df_eval["y_true"] >= q_high, "high", "mid")
    )

    out = df_eval.groupby("bin", as_index=False).agg(
        n=("y_true", "size"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        y_mean=("y_true", "mean"),
    )
    out["model"] = label
    return out
