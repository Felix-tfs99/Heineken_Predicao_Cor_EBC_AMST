from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


def plot_feature_importance(
    model: Pipeline,
    feature_cols: List[str],
    top_n: int = 15,
    title: Optional[str] = None,
) -> None:
    """
    Plota as top_n features mais importantes para modelos baseados em árvore.
    Funciona para: RandomForest, GradientBoosting, XGBoost, LightGBM (quando possuem feature_importances_).
    """
    final_model = model.named_steps.get("model", model)

    if not hasattr(final_model, "feature_importances_"):
        raise AttributeError("O modelo não possui atributo feature_importances_ (não é baseado em árvore ou não expõe importâncias).")

    importances = np.asarray(final_model.feature_importances_, dtype=float)
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(9, 6))
    fi.sort_values().plot(kind="barh")
    plt.title(title or f"Feature importance (Top {top_n})")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.show()
