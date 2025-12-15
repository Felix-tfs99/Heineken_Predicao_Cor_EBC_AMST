# Heineken — Predição de Cor do Mosto (EBC) | Produto AMST

Este repositório apresenta uma solução completa de **Data Science** para predição da **cor (EBC)** do mosto frio do produto **AMST**, utilizando dados de processo em nível de lote (*batch-level*).

O projeto foi estruturado com foco em:
- clareza de raciocínio analítico,
- boas práticas de modelagem e validação,
- organização de código reutilizável,
- qualidade e reprodutibilidade,
- e narrativa técnica adequada.

---

## 1) Contexto do Problema

A cor (EBC) é um dos principais atributos de qualidade do produto final.  
Desvios fora da especificação podem gerar **retrabalho, perdas operacionais ou descarte de lotes**.

O desafio consiste em **prever a cor do mosto frio antes do final do processo**, utilizando informações disponíveis ao longo da produção, apoiando decisões operacionais e controle de qualidade.

---

## 2) Visão Geral dos Dados

Cada linha do dataset representa um lote produtivo e inclui:

- **Composição de maltes**
  - malte torrado
  - maltes base (1º e 2º)
- **Parâmetros térmicos**
  - temperaturas
  - tempos
  - vapor
- **Propriedades físico-químicas**
  - pH
  - extrato (°P)
- **Variável alvo**
  - **Color (EBC)**

> Observação: os maltes base podem vir de lotes distintos, mas são misturados durante a moagem.

---

## 3) Metodologia

### Etapa 1 — Análise Exploratória (EDA)
- verificação de qualidade dos dados
- análise temporal
- distribuição do target
- correlações e interações entre variáveis
- identificação de não linearidades
- hipóteses orientadas por conhecimento do processo

### Etapa 2 — Modelagem Baseline
- Dummy Regressor (baseline mínimo)
- Ridge Regression (modelo linear)
- Random Forest com hiperparâmetros default
- divisão temporal dos dados (evitando *data leakage*)

### Etapa 3 — Modelagem Avançada
- ajuste de hiperparâmetros com **TimeSeriesSplit**
- comparação entre:
  - Random Forest otimizado
  - XGBoost
  - LightGBM
- seleção do melhor modelo global com base em RMSE e análises gráficas

### Etapa 4 — Análise de Erros e Trade-offs
- avaliação por faixas do target (low / mid / high EBC)
- identificação de limitações nos extremos
- exploração de abordagens complementares:
  - `sample_weight` em modelos de boosting
- análise de trade-offs entre estabilidade global e sensibilidade a extremos

### Etapa 5 — Qualidade e Testes
- testes automatizados com `pytest` para:
  - garantir split temporal correto
  - validar robustez do feature engineering
  - assegurar consistência das métricas

---

## 4) Estrutura do Repositório

```text
heineken_case/
├─ notebooks/
│  ├─ 01_eda_heineken.ipynb
│  ├─ 02_modeling_baseline.ipynb
│  ├─ 03_modeling_advanced.ipynb
│  ├─ 04_error_analysis_and_improvements.ipynb
│  └─ 05_testing_and_quality_checks.ipynb
│
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  │
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ load_data.py
│  │
│  ├─ features/
│  │  ├─ __init__.py
│  │  └─ preprocessing.py
│  │
│  └─ models/
│     ├─ __init__.py
│     ├─ evaluation.py
│     ├─ pipelines.py
│     ├─ training.py
│     ├─ tuning.py
│     └─ io.py
│
├─ tests/
│  ├─ __init__.py
│  ├─ test_split.py
│  ├─ test_features.py
│  └─ test_evaluation.py
│
├─ models/
├─ requirements.txt
└─ README.md
