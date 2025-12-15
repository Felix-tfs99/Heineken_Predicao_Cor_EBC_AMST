import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date/Time"] = pd.to_datetime(
        df["Date/Time"],
        format="%m/%d/%Y %H:%M",
        errors="coerce"
    )
    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("==== DIMENSÕES ====")
    print(df.shape)
    print("\n==== TIPOS ====")
    print(df.dtypes)
    print("\n==== VALORES FALTANTES ====")
    print(df.isna().sum())
    print("\n==== ESTATÍSTICAS DESCRITIVAS ====")
    display(df.describe())
    print("\n==== PRIMEIRAS LINHAS ====")
    display(df.head())


def plot_target_distribution(df: pd.DataFrame, target_col: str = "Color") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df[target_col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"Distribuição de {target_col}")
    sns.boxplot(x=df[target_col].dropna(), ax=axes[1])
    axes[1].set_title(f"Boxplot de {target_col}")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, max_cols: int = 20) -> None:
    """
    Plota um mapa de calor de correlação para até max_cols variáveis numéricas.
    """
    num_df = df.select_dtypes(include=[np.number])
    
    # Se tiver muitas colunas, pegar só as primeiras max_cols
    if num_df.shape[1] > max_cols:
        num_df = num_df.iloc[:, :max_cols]
    
    plt.figure(figsize=(10, 8))
    corr = num_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Matriz de correlação (amostra de variáveis numéricas)")
    plt.tight_layout()
    plt.show()


def filter_scope(df: pd.DataFrame, product: str = "AMST") -> pd.DataFrame:
    df = df.copy()
    df = df[df["Product"].notna()]
    df = df[df["Product"] == product]
    return df


def clean_and_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df = df[df["Color"].notna()]
    df = df[df["Color"] >= 0]
    df = df.sort_values("Date/Time")

    df["total_malt"] = (
        df["Roast amount (kg)"]
        + df["1st malt amount (kg)"]
        + df["2nd malt amount (kg)"]
    )
    df["roast_pct"] = np.where(
        df["total_malt"] > 0,
        df["Roast amount (kg)"] / df["total_malt"],
        0.0
    )

    drop_cols = ["Unnamed: 0", "Product", "Job ID", "Date/Time"]
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols + ["Color"]
    ]
    return df, feature_cols


def temporal_train_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float
):
    n_rows = df.shape[0]
    split_idx = int(n_rows * (1 - test_size))

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    print(f"Total de linhas: {n_rows}")
    print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test
