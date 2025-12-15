from __future__ import annotations
import pandas as pd
from ..config import DATE_COL, PRODUCT_COL, PRODUCT_VALUE, TARGET_COL


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Carrega o dataset bruto a partir de um arquivo CSV e faz o parse da coluna de data/hora.

    Parâmetros
    ----------
    csv_path : str
        Caminho para o arquivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados carregados. Se a coluna DATE_COL existir, será convertida
        para datetime (valores inválidos viram NaT).
    """
    df = pd.read_csv(csv_path)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def filter_product(df: pd.DataFrame, product: str = PRODUCT_VALUE) -> pd.DataFrame:
    """
    Filtra o DataFrame para manter apenas os registros do produto especificado.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    product : str, default=PRODUCT_VALUE
        Valor do produto a ser filtrado (ex.: "AMST").

    Retorna
    -------
    pd.DataFrame
        DataFrame filtrado (cópia), contendo apenas o produto selecionado.

    Levanta
    -------
    ValueError
        Se a coluna de produto (PRODUCT_COL) não existir no DataFrame.
    """
    if PRODUCT_COL not in df.columns:
        raise ValueError(f"Coluna ausente no dataset: {PRODUCT_COL}")
    return df[df[PRODUCT_COL] == product].copy()


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa uma limpeza básica e padronizada, reaproveitada entre notebooks.

    Regras aplicadas:
    - remove linhas com target nulo
    - remove valores negativos do target (quando não fazem sentido)
    - ordena temporalmente pela coluna de data/hora (se existir)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpo e, quando aplicável, ordenado por data/hora.

    Levanta
    -------
    ValueError
        Se a coluna alvo (TARGET_COL) não existir no DataFrame.
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
    Função utilitária (conveniência): carrega o CSV, filtra AMST e aplica limpeza básica.

    Fluxo:
    1) load_dataset
    2) filter_product (AMST)
    3) basic_cleaning

    Parâmetros
    ----------
    csv_path : str
        Caminho para o arquivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame do produto AMST já limpo e preparado para EDA/modelagem.
    """
    df = load_dataset(csv_path)
    df = filter_product(df, product=PRODUCT_VALUE)
    df = basic_cleaning(df)
    return df
