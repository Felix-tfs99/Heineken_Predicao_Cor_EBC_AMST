from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def eval_regression(y_true, y_pred, label: str = "model", verbose: bool = True) -> Dict[str, float]:
    """Compute MAE, RMSE and R² for regression."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = r2_score(y_true_arr, y_pred_arr)

    if verbose:
        print(f"{label:<25} -> MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | R²: {r2:6.3f}")

    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}

def evaluate_by_bins(y_true, y_pred, q_low: float, q_high: float, label: str = "model") -> pd.DataFrame:
    """Evaluate metrics by low/mid/high bins defined by target quantiles."""
    y_true_s = pd.Series(y_true).reset_index(drop=True)
    y_pred_a = np.asarray(y_pred)

    df_eval = pd.DataFrame({"y_true": y_true_s, "y_pred": y_pred_a})
    df_eval["abs_error"] = (df_eval["y_true"] - df_eval["y_pred"]).abs()
    df_eval["sq_error"] = (df_eval["y_true"] - df_eval["y_pred"]) ** 2

    df_eval["bin"] = np.where(
        df_eval["y_true"] <= q_low, "low",
        np.where(df_eval["y_true"] >= q_high, "high", "mid")
    )

    out = df_eval.groupby("bin").agg(
        n=("y_true", "size"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        y_mean=("y_true", "mean"),
    )
    out["model"] = label
    return out.reset_index()
