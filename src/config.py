from __future__ import annotations
from dataclasses import dataclass

# ============================
# Colunas do dataset
# ============================
DATE_COL: str = "Date/Time"
PRODUCT_COL: str = "Product"
TARGET_COL: str = "Color (EBC)"

# Produto alvo (case)
PRODUCT_VALUE: str = "AMST"

# ============================
# Configuração de experimento
# ============================
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2  # divisão temporal (parte final vira teste)

# Quantis para análise por faixa (extremos)
Q_LOW: float = 0.10
Q_HIGH: float = 0.90

# Peso para extremos (quando usar sample_weight)
EXTREME_WEIGHT: float = 3.0
