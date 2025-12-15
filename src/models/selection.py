from __future__ import annotations
import pandas as pd

def select_best_model(results_df: pd.DataFrame, metric: str = "rmse") -> str:
    """
    Seleciona o melhor modelo com base na métrica (menor = melhor).
    Aceita results_df com:
      - modelos no índice, ou
      - coluna 'model'
    """
    if metric not in results_df.columns:
        raise ValueError(f"Métrica '{metric}' não encontrada. Colunas disponíveis: {list(results_df.columns)}")

    if "model" in results_df.columns:
        return results_df.sort_values(metric).iloc[0]["model"]

    # caso modelos estejam no índice
    return results_df[metric].idxmin()
