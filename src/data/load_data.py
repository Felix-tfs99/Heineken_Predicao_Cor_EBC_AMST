from __future__ import annotations
import pandas as pd
from ..config import DATE_COL, PRODUCT_COL, PRODUCT_VALUE, TARGET_COL


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Carrega o dataset bruto a partir de um CSV e faz parse da coluna de data/hora.

    Parâmetros
    ----------
    csv_path : str
        Caminho para o arquivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados carregados. Se DATE_COL existir, é convertida para datetime
        (valores inválidos viram NaT).
    """
    df = pd.read_csv(csv_path)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def filter_product(df: pd.DataFrame, product: str = PRODUCT_VALUE) -> pd.DataFrame:
    """
    Filtra o DataFrame para manter apenas registros do produto especificado.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    product : str, default=PRODUCT_VALUE
        Valor do produto (ex.: "AMST").

    Retorna
    -------
    pd.DataFrame
        DataFrame filtrado (cópia).

    Levanta
    -------
    ValueError
        Se PRODUCT_COL não existir.
    """
    if PRODUCT_COL not in df.columns:
        raise ValueError(f"Coluna ausente no dataset: {PRODUCT_COL}")
    return df[df[PRODUCT_COL] == product].copy()


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza básica padronizada:
    - remove target nulo
    - remove target negativo (quando não faz sentido)
    - ordena pelo tempo (se DATE_COL existir)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpo e ordenado (quando aplicável).

    Levanta
    -------
    ValueError
        Se TARGET_COL não existir.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Coluna alvo ausente no dataset: {TARGET_COL}")

    out = df[df[TARGET_COL].notna()].copy()
    out = out[out[TARGET_COL] >= 0]

    if DATE_COL in out.columns:
        out = out.sort_values(DATE_COL)

    return out


def load_amst_clean(csv_path: str) -> pd.DataFrame:
    """
    Função utilitária: carrega CSV → filtra AMST → aplica limpeza básica.

    Parâmetros
    ----------
    csv_path : str
        Caminho do arquivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame do produto AMST pronto para EDA/modelagem.
    """
    df = load_dataset(csv_path)
    df = filter_product(df, product=PRODUCT_VALUE)
    df = basic_cleaning(df)
    return df
