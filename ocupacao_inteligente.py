
# %% [markdown]
# # PROJETO DE INTEGRAÇÃO DE DADOS COM IA
# ## Ocupação Inteligente — Indústria 4.0
#
# Autor: Petras Ruben Carvalho
# Professor: Felipe Rodrigues Souza
# Instituição: SENAI / SCTEC
#
# ### BLOCO 00 — Identificação e contextualização do projeto
#
# ### Problema de negócio
#
# **Contexto do enunciado:** prever se salas técnicas e escritórios estão ocupados ou vazios a partir de sensores
# ambientais IoT, para automatizar o desligamento de climatização (HVAC) e iluminação e reduzir custos energéticos.
#
# **Pontos a apresentar:** problema a ser resolvido, benefício esperado e caráter experimental do projeto.
#
# *[Contextualização a ser redigida pelo estudante]*
#
# ### Variável alvo
#
# | Variável | Tipo | Valores |
# |---|---|---|
# | `Occupancy` | Binária (`int64`) | 0 = desocupado; 1 = ocupado |
#
# **Pontos a apresentar:** tipo de problema de aprendizado (classificação binária) e modelos previstos (KNN e Árvore de Decisão).
#
# *[Descrição a ser redigida pelo estudante]*
#
# ### Base de dados, fonte e licença
#
# | Item | Informação |
# |---|---|
# | Dataset | Occupancy Detection — Luis Candanedo (2016) |
# | Fonte | UCI Machine Learning Repository — DOI 10.24432/C5X01N |
# | Licença | Creative Commons Attribution 4.0 (CC BY 4.0) |
# | Arquivos utilizados | `datatest.txt` (2.665) e `datatest2.txt` (9.752) |
# | Arquivo preservado e não utilizado | `datatraining.txt` (8.143) |
# | Base consolidada | 12.417 registros e 7 colunas |
# | Atribuição e hashes SHA-256 | `dados/README.md` |
#
# *[Descrição a ser redigida pelo estudante]*

# %% [markdown]
# ### BLOCO 01 — Importação das bibliotecas
#
# **Evidências:**
#
# | Grupo | Bibliotecas |
# |---|---|
# | Padrão do Python | `os`, `zipfile`, `urllib.request` |
# | Análise e manipulação | `pandas`, `numpy` |
# | Visualização | `matplotlib.pyplot`, `seaborn` |
# | Machine Learning | `train_test_split`, `StandardScaler`, `Pipeline`, `DecisionTreeClassifier`, `KNeighborsClassifier`, métricas |
#
# Versões no ambiente `.venv`: Python 3.14.4, pandas 3.0.2, NumPy 2.4.4, Matplotlib 3.10.9, Seaborn 0.13.2, scikit-learn 1.8.0.
#
# **Perguntas orientadoras:** por que centralizar as importações? Qual a função de cada grupo? Por que os recursos de
# Machine Learning já são importados, se serão usados apenas nas Fases 4 a 7?
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 01 — Importação das bibliotecas

# Bibliotecas padrão do Python
import os
import zipfile
import urllib.request

# Análise e manipulação de dados
import pandas as pd
import numpy as np

# Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# %% [markdown]
# ### BLOCO 02 — Configurações visuais
#
# **Evidências:**
#
# | Configuração | Valor |
# |---|---|
# | Estilo do Seaborn | `whitegrid` |
# | Tamanho padrão das figuras | `(10, 5)` polegadas |
#
# Mesma configuração do exemplo didático do enunciado (p. 3).
#
# **Perguntas orientadoras:** qual o efeito de definir as configurações visuais uma única vez no início? Por que o padrão
# visual é importante para a leitura dos gráficos?
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 02 — Configurações visuais

# Configurando o estilo dos gráficos
sns.set_theme(style="whitegrid")

# Definindo o tamanho padrão dos gráficos
plt.rcParams["figure.figsize"] = (10, 5)

print(plt.rcParams["figure.figsize"])


# %% [markdown]
# ### BLOCO 03 — Caminhos relativos e diretórios
#
# **Evidências:**
#
# | Variável | Caminho |
# |---|---|
# | `zip_path` | `dados/brutos/occupancy_data.zip` |
# | `caminho_datatest` | `dados/brutos/datatest.txt` |
# | `caminho_datatest2` | `dados/brutos/datatest2.txt` |
# | `caminho_csv` | `dados/consolidados/occupancy.csv` |
# | `caminho_tratado` | `dados/tratados/occupancy_tratado.csv` (gerado na Fase 2) |
# | `pasta_graficos` / `pasta_metricas` | `resultados/graficos` / `resultados/metricas` |
#
# Conceitos: o item 5.1 do enunciado proíbe caminhos absolutos; `os.makedirs(..., exist_ok=True)` cria as pastas sem erro
# quando já existem; o Git não versiona pastas vazias.
#
# **Perguntas orientadoras:** por que caminhos relativos garantem a execução pelo avaliador? Por que criar as pastas no próprio código?
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 03 — Caminhos relativos e diretórios

# Definindo os diretórios do projeto
pasta_brutos = "dados/brutos"
pasta_consolidados = "dados/consolidados"
pasta_tratados = "dados/tratados"
pasta_graficos = "resultados/graficos"
pasta_metricas = "resultados/metricas"

# Criando os diretórios, caso não existam
os.makedirs(pasta_brutos, exist_ok=True)
os.makedirs(pasta_consolidados, exist_ok=True)
os.makedirs(pasta_tratados, exist_ok=True)
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_metricas, exist_ok=True)

# Definindo os caminhos dos arquivos brutos
zip_path = "dados/brutos/occupancy_data.zip"
caminho_datatest = "dados/brutos/datatest.txt"
caminho_datatest2 = "dados/brutos/datatest2.txt"

# Definindo o caminho do arquivo consolidado
caminho_csv = "dados/consolidados/occupancy.csv"

# Definindo o caminho do arquivo tratado
caminho_tratado = "dados/tratados/occupancy_tratado.csv"

# Validando o diretório de trabalho
print("Diretório atual:", os.getcwd())

# Verificando a existência dos arquivos
print("Arquivo ZIP:", os.path.exists(zip_path))
print("Arquivo datatest:", os.path.exists(caminho_datatest))
print("Arquivo datatest2:", os.path.exists(caminho_datatest2))
print("Arquivo consolidado:", os.path.exists(caminho_csv))
print("Arquivo tratado:", os.path.exists(caminho_tratado))


# %% [markdown]
# ### BLOCO 04 — Download condicional e extração dos dados
#
# **Evidências:**
#
# | Item | Evidência |
# |---|---|
# | Condição do download | Ausência de `datatest.txt` ou `datatest2.txt` em `dados/brutos/` |
# | Extração | Somente dos arquivos ausentes (`extract`), sem sobrescrever originais |
# | Origem | URL oficial da UCI indicada no enunciado |
# | Última execução | "Arquivos originais já disponíveis!" |
# | Integridade | SHA-256 dos arquivos brutos idênticos aos registrados em `dados/README.md` |
#
# **Perguntas orientadoras:** por que verificar os arquivos brutos, e não o CSV consolidado? Por que extrair apenas os
# arquivos ausentes? Qual a diferença em relação ao exemplo do professor?
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 04 — Download condicional e extração dos dados

# Endereço oficial do dataset da UCI
url_uci = "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"

# Verificando se os arquivos brutos estão disponíveis
if not os.path.exists(caminho_datatest) or not os.path.exists(caminho_datatest2):

    # Baixando o ZIP somente se não estiver disponível
    if not os.path.exists(zip_path):
        print("Baixando o dataset oficial da UCI...")
        urllib.request.urlretrieve(url_uci, zip_path)
        print("Download concluído com sucesso!")
    else:
        print("Arquivo ZIP já disponível!")

    # Extraindo somente os arquivos ausentes
    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        if not os.path.exists(caminho_datatest):
            zip_ref.extract("datatest.txt", pasta_brutos)

        if not os.path.exists(caminho_datatest2):
            zip_ref.extract("datatest2.txt", pasta_brutos)

    print("Extração concluída com sucesso!")

else:
    print("Arquivos originais já disponíveis!")


# %% [markdown]
# ### BLOCO 05 — Leitura dos arquivos brutos
#
# **Evidências:**
#
# | Arquivo | Registros | Situação |
# |---|---:|---|
# | `datatest.txt` | 2.665 | Utilizado |
# | `datatest2.txt` | 9.752 | Utilizado |
# | `datatraining.txt` | 8.143 | Preservado, fora da consolidação |
#
# Conceito: a primeira coluna dos arquivos `.txt` é uma numeração sem nome no cabeçalho, que o pandas utiliza
# automaticamente como índice.
#
# **Perguntas orientadoras:** por que utilizar `datatest.txt` e `datatest2.txt`? Por que manter `datatraining.txt` fora da
# consolidação? Qual a diferença em relação ao exemplo do enunciado, que lê apenas `datatest.txt`?
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 05 — Leitura dos arquivos brutos

# Carregando os datasets originais
df1 = pd.read_csv(caminho_datatest)
df2 = pd.read_csv(caminho_datatest2)

print("Arquivo datatest.txt carregado com sucesso!")
print("Arquivo datatest2.txt carregado com sucesso!")


# %% [markdown]
# ### BLOCO 06 — Inspeção individual dos dados brutos
#
# **Evidências:**
#
# | Arquivo | Dimensões | Colunas |
# |---|---|---|
# | `datatest.txt` | (2665, 7) | `date`, `Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio`, `Occupancy` |
# | `datatest2.txt` | (9752, 7) | Mesmas colunas, na mesma ordem |
#
# **Perguntas orientadoras:** por que inspecionar cada base antes de uni-las? O que a compatibilidade das colunas garante
# para a consolidação?
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 06 — Inspeção individual dos dados brutos

# Inspecionando o primeiro dataset
print("DATASET 01 - datatest.txt")
print("Dimensões:", df1.shape)
print("Colunas:", df1.columns)
print(df1.head())

print("=" * 80)

# Inspecionando o segundo dataset
print("DATASET 02 - datatest2.txt")
print("Dimensões:", df2.shape)
print("Colunas:", df2.columns)
print(df2.head())


# %% [markdown]
# ### BLOCO 07 — Consolidação dos DataFrames
#
# **Evidências:**
#
# | Item | Evidência |
# |---|---|
# | Método | `pd.concat([df1, df2], ignore_index=True)` |
# | Ordem | Registros de `datatest.txt` seguidos dos de `datatest2.txt` |
# | Índice resultante | 0 a 12.416 |
#
# Conceito: os índices herdados dos arquivos `.txt` se repetem entre as bases; `ignore_index=True` gera um índice novo e único.
#
# **Perguntas orientadoras:** o que é uma concatenação vertical? Que problema o `ignore_index=True` evita?
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 07 — Consolidação dos DataFrames

# Concatenando os dois datasets em um único DataFrame
df = pd.concat([df1, df2], ignore_index=True)

# Visualizando as primeiras linhas do DataFrame consolidado
print("DATASET CONSOLIDADO")
print(df.head())


# %% [markdown]
# ### BLOCO 08 — Validação da consolidação
#
# **Evidências:**
#
# | Validação | Resultado |
# |---|---|
# | Registros: 2.665 + 9.752 = 12.417 | True |
# | Compatibilidade das colunas | True |
# | Dimensões esperadas (12417, 7) | True |
#
# Conceito: `assert` interrompe a execução quando a condição é falsa, impedindo a exportação de uma base inconsistente.
#
# **Perguntas orientadoras:** o que cada validação garante? Por que validar antes de exportar o CSV?
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 08 — Validação da consolidação

# Verificando as dimensões do DataFrame consolidado
print("Dimensões da base consolidada:", df.shape)

# Verificando as colunas
print("Colunas:", df.columns)

# Verificando a quantidade de registros
print("Registros do dataset 01:", len(df1))
print("Registros do dataset 02:", len(df2))
print("Total de registros consolidados:", len(df))

# Validando a quantidade total de registros
assert len(df) == len(df1) + len(df2), (
    "O número total de registros não corresponde à soma "
    "dos registros dos datasets individuais."
)

# Validando a compatibilidade das colunas
assert df1.columns.equals(df2.columns), (
    "Os datasets apresentam estruturas de colunas diferentes."
)

# Validando as dimensões esperadas
assert df.shape == (12417, 7), (
    "A base consolidada não apresenta as dimensões esperadas."
)

# Apresentando os resultados das validações
print("Quantidade de registros válida:", len(df) == len(df1) + len(df2))
print("Compatibilidade das colunas:", df1.columns.equals(df2.columns))
print("Dimensões esperadas:", df.shape == (12417, 7))

print("Validação da consolidação concluída com sucesso!")


# %% [markdown]
# ### BLOCO 09 — Exportação e carregamento do CSV
#
# **Evidências:**
#
# | Item | Evidência |
# |---|---|
# | Arquivo | `dados/consolidados/occupancy.csv` |
# | Exportação | `to_csv(..., index=False)`, sem formatação de casas decimais |
# | Recarga | `df_consolidado`, com dimensões (12417, 7) |
# | Equivalência | `pd.testing.assert_frame_equal(..., check_exact=True)` aprovado |
# | Reprodutibilidade | CSV idêntico ao versionado (mesmo SHA-256) em sessão Python nova |
#
# Conceito: `check_exact=True` compara os valores sem tolerância numérica, comprovando a preservação da precisão original.
#
# **Perguntas orientadoras:** por que `index=False`? Por que recarregar o arquivo exportado? O que o `check_exact=True` comprova?
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 09 — Exportação e carregamento do CSV

# Exportando o DataFrame consolidado
df.to_csv(caminho_csv, index=False)

print("Arquivo CSV consolidado exportado com sucesso!")
print("Arquivo:", caminho_csv)

# Carregando a base consolidada
df_consolidado = pd.read_csv(caminho_csv)

# Inspecionando o arquivo exportado
print("Dimensões:", df_consolidado.shape)
print("Colunas:", df_consolidado.columns)

# Visualizando os primeiros registros
print(df_consolidado.head())

# Validando a equivalência dos DataFrames
pd.testing.assert_frame_equal(df, df_consolidado, check_exact=True)

# Confirmando a conclusão das validações
print("Validação concluída: os DataFrames são equivalentes!")
print("ETL de consolidação inicial concluído com sucesso!")


###############################################################################################################################################################
# %% [markdown]
# # FASE 1 — Análise Exploratória dos Dados (EDA)
# ## BLOCO 10 — Contextualização da análise exploratória
#
# ### Objetivo da análise exploratória
#
# - Compreender as características dos dados consolidados.
# - Investigar a distribuição das variáveis ambientais.
# - Identificar possíveis problemas de qualidade dos dados.
# - Investigar as relações entre as variáveis e a variável alvo `Occupancy`.
#
# *[Contextualização a ser redigida pelo estudante]*
#
# ### Caracterização da base de dados
#
# | Item | Informação |
# |---|---|
# | Fonte | UCI Machine Learning Repository — Occupancy Detection (CC BY 4.0) |
# | Arquivos consolidados | `datatest.txt` (2.665) + `datatest2.txt` (9.752) |
# | Registros | 12.417 |
# | Colunas | 7 |
# | Variáveis ambientais | 5 — `Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio` |
# | Variável temporal | 1 — `date` |
# | Variável alvo | 1 — `Occupancy` (binária: 0 = desocupado, 1 = ocupado) |
# | Valores ausentes | Nenhum identificado |
#
# ### Estratégia de análise
#
# | Bloco | Método |
# |---|---|
# | 11 | Dimensões e estrutura (`shape`, `columns`, `head`) |
# | 12 | Tipos de dados e valores não nulos (`info`, `dtypes`) |
# | 13 | Estatísticas descritivas (`describe`) |
# | 14 | Histogramas das cinco variáveis ambientais |
# | 15 | Distribuição da variável alvo (quantidade e percentual) |
# | 16 | Matriz de correlação de Pearson |
# | 16.1 | Avaliação exploratória da coluna `date` |
#
# ### Contribuição para a modelagem preditiva
#
# Conceitos a serem fundamentados pelo estudante:
#
# - Escalonamento das variáveis para o KNN.
# - Divisão estratificada entre treino e teste.
# - Diferenças de escala entre os sensores.
# - Correlação entre as variáveis preditoras.
# - Limitações decorrentes da natureza temporal dos dados.
# - Preparação para a Fase 2 — Data Prep.
#
# *[Justificativa a ser redigida pelo estudante]*

# %%
# BLOCO 11 — Dimensões e estrutura dos dados
print("Dimensões:", df_consolidado.shape)
print("Colunas:", df_consolidado.columns)
print(df_consolidado.head())

# %%
# BLOCO 12 — Inspeção dos tipos de dados
df_consolidado.info()
print(df_consolidado.dtypes)

# %%
# BLOCO 13 — Estatística descritiva
print(df_consolidado.describe())

# %%
# BLOCO 14 — Histogramas das variáveis numéricas

# Seleção das variáveis numéricas
colunas_numericas = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio"
]

unidades = {
    "Temperature": "Temperatura (°C)",
    "Humidity": "Umidade (%)",
    "Light": "Luminosidade (lux)",
    "CO2": "CO2 (ppm)",
    "HumidityRatio": "Razão de umidade (kg/kg)"
}

# Construção dos histogramas
eixos = df_consolidado[colunas_numericas].hist(
    bins=30,
    figsize=(15, 10),
    edgecolor="black",
    layout=(2, 3)
)

# Personalização dos eixos dos histogramas
for eixo in eixos.flatten():
    nome = eixo.get_title()

    if nome in unidades:
        eixo.set_xlabel(unidades[nome])
        eixo.set_ylabel("Frequência")

# Título geral
plt.suptitle(
    "Distribuição das variáveis numéricas",
    fontsize=16
)

# Ajuste do espaçamento
plt.tight_layout()

# Exportação dos histogramas
caminho_figura = f"{pasta_graficos}/eda_01_histogramas.png"

plt.savefig(
    caminho_figura,
    dpi=300,
    bbox_inches="tight"
)

# Exibição dos histogramas
plt.show()

# %%
# BLOCO 15 — Distribuição da variável alvo

# Contagem absoluta das classes
contagem_classes = df_consolidado["Occupancy"].value_counts().sort_index()

print("Quantidade de registros por classe:")
print(contagem_classes)

# Distribuição percentual das classes
percentual_classes = (
    df_consolidado["Occupancy"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("\nDistribuição percentual das classes:")
print(percentual_classes.round(2))

# Configuração da figura
plt.figure(figsize=(10, 5))

# Gráfico de distribuição das classes
ax = sns.countplot(
    data=df_consolidado,
    x="Occupancy",
    hue="Occupancy",
    palette="Set2",
    legend=False
)

# Rótulos personalizados nas barras: quantidade + percentual
for i, barra in enumerate(ax.patches):
    quantidade = contagem_classes.iloc[i]
    percentual = percentual_classes.iloc[i]

    # Quantidade acima da barra
    ax.annotate(
        f"{quantidade:,}".replace(",", "."),  # formato brasileiro: 9.396
        (barra.get_x() + barra.get_width() / 2, barra.get_height()),
        ha="center",
        va="bottom",
        fontsize=11,
        xytext=(0, 3),
        textcoords="offset points"
    )

    # Percentual dentro da barra
    ax.annotate(
        f"({percentual:.2f}%)".replace(".", ","),  # formato brasileiro: 75,67%
        (
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() * 0.7
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="black"
    )

# Personalização dos eixos
ax.set_title("Distribuição da variável alvo", fontsize=14)
ax.set_xlabel("Classe", fontsize=12)
ax.set_ylabel("Quantidade de registros", fontsize=12)
ax.set_xticks(
    [0, 1],
    ["0 — Desocupado", "1 — Ocupado"]
)

plt.tight_layout()

# Exportação do gráfico de distribuição da variável alvo
caminho_figura = f"{pasta_graficos}/eda_02_distribuicao_occupancy.png"

plt.savefig(
    caminho_figura,
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# %%
# BLOCO 16 — Matriz de correlação de Pearson

# Seleção das variáveis numéricas
dados_numericos = df_consolidado.select_dtypes(include="number")

# Cálculo da matriz de correlação de Pearson
matriz_correlacao = dados_numericos.corr(method="pearson")

# Exibição da matriz no terminal
print("Matriz de correlação de Pearson:")
print(matriz_correlacao.round(4))

# Configuração da figura
plt.figure(figsize=(12, 8))

# Construção do mapa de calor
sns.heatmap(
    matriz_correlacao,
    annot=True,
    fmt=".4f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "Coeficiente de correlação"}
)

# Título do gráfico
plt.title("Matriz de correlação de Pearson", fontsize=14)

# Ajuste do espaçamento
plt.tight_layout()

# Exportação do gráfico de correlação
caminho_figura = f"{pasta_graficos}/eda_03_correlacao_pearson.png"

plt.savefig(
    caminho_figura,
    dpi=300,
    bbox_inches="tight"
)

# Exibição do gráfico
plt.show()

# %%
# BLOCO 16.1 — Avaliação exploratória da coluna date
# Verificação do tipo original
print("Tipo original:", df_consolidado["date"].dtype)

# Conversão temporária para datetime
datas = pd.to_datetime(
    df_consolidado["date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

# Verificação de datas inválidas
print("\nDatas inválidas:", datas.isna().sum())

# Período das observações
print("\nData inicial:", datas.min())
print("Data final:", datas.max())

# Verificação da ordem cronológica
print("\nOrdem cronológica:", datas.is_monotonic_increasing)

# Intervalos entre observações consecutivas
intervalos = datas.diff()

# Distribuição dos intervalos
print("\nIntervalos mais frequentes:")
print(intervalos.value_counts().head(10))

# Identificação da maior descontinuidade temporal
print("\nMaior intervalo:", intervalos.max())
print("Índice do maior intervalo:", intervalos.idxmax())

# Confirmação da preservação do tipo original
print("\nTipo original preservado:")
print(df_consolidado["date"].dtype)

# %% [markdown]
# ## BLOCO 17 — Interpretação dos resultados da EDA
#
# ### 17.1 — Estrutura e qualidade inicial dos dados
#
# **Evidências (BLOCOS 11 e 12):**
#
# | Item | Resultado |
# |---|---|
# | Registros | 12.417 |
# | Colunas | 7 |
# | Valores ausentes | 0 (12.417 valores não nulos em todas as colunas) |
# | `date` | `str` |
# | `Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio` | `float64` |
# | `Occupancy` | `int64` |
#
# **Pontos para interpretação:** os tipos são adequados para a análise? Que tratamento a coluna `date` exige antes da modelagem?
#
# *[Interpretação a ser redigida pelo estudante]*
#
# ### 17.2 — Estatísticas descritivas
#
# **Evidências (BLOCO 13 — `describe()`):**
#
# | Variável | Média | Mediana (50%) | Desvio padrão | Mínimo | Máximo |
# |---|---:|---:|---:|---:|---:|
# | Temperature (°C) | 21,094510 | 20,790000 | 1,037511 | 19,500000 | 24,408333 |
# | Humidity (%) | 28,917947 | 28,600000 | 4,125142 | 21,865000 | 39,500000 |
# | Light (lux) | 138,125947 | 0,000000 | 219,801233 | 0,000000 | 1.697,250000 |
# | CO2 (ppm) | 745,644624 | 632,000000 | 296,497310 | 427,500000 | 2.076,500000 |
# | HumidityRatio (kg/kg) | 0,0044682086 | 0,0045113541 | 0,0005954957 | 0,0032747640 | 0,0057686083 |
#
# Valores apresentados com seis casas decimais, conforme a saída do `describe()`; `HumidityRatio` com dez casas para preservar a precisão. Os dados não foram arredondados.
#
# **Pontos para interpretação:**
#
# - Diferença de escala e amplitude entre as variáveis (ex.: `HumidityRatio` na ordem de 0,004 e `CO2` na ordem de centenas a milhares).
# - Relação entre média e mediana em cada variável, especialmente em `Light` e `CO2`.
# - Significado de `Light` apresentar mediana igual a 0 lux.
#
# *[Interpretação a ser redigida pelo estudante]*
#
# ### 17.3 — Distribuição das variáveis ambientais
#
# **Evidência:** `resultados/graficos/eda_01_histogramas.png` (BLOCO 14 — cinco histogramas com 30 intervalos).
#
# **Pontos para interpretação em cada histograma:**
#
# - `Temperature`: formato da distribuição e região de maior concentração.
# - `Humidity`: quantidade de picos e amplitude.
# - `Light`: concentração de registros próximos de 0 lux e existência de um segundo agrupamento de valores.
# - `CO2`: simetria da distribuição e presença de valores elevados.
# - `HumidityRatio`: formato da distribuição em comparação com `Humidity`.
# - Indícios de valores discrepantes a serem verificados com boxplots na Fase 2.
#
# *[Interpretação a ser redigida pelo estudante]*
#
# ### 17.4 — Distribuição da variável alvo
#
# **Evidências (BLOCO 15 — `resultados/graficos/eda_02_distribuicao_occupancy.png`):**
#
# | Classe | Quantidade | Percentual |
# |---|---:|---:|
# | 0 — Desocupado | 9.396 | 75,67% |
# | 1 — Ocupado | 3.021 | 24,33% |
# | Total | 12.417 | 100% |
#
# **Pontos para interpretação:**
#
# - As classes estão balanceadas?
# - Qual seria a acurácia de referência de um modelo que sempre previsse a classe majoritária?
# - Como esse resultado se relaciona com o uso de `stratify=y` na Fase 4?
#
# *[Interpretação a ser redigida pelo estudante]*
#
# ### 17.5 — Correlação entre as variáveis
#
# **Evidências (BLOCO 16 — `resultados/graficos/eda_03_correlacao_pearson.png`):**
#
# | Variável | Correlação com Occupancy |
# |---|---:|
# | Light | 0,9197 |
# | Temperature | 0,5764 |
# | CO2 | 0,3710 |
# | HumidityRatio | 0,2420 |
# | Humidity | -0,0427 |
#
# Correlações entre preditoras:
#
# - `Humidity` × `HumidityRatio`: 0,8919.
# - `Temperature` × `Light`: 0,7248.
#
# **Pontos para interpretação:**
#
# - Intensidade e sentido da associação linear de cada variável com `Occupancy`.
# - Redundância entre preditoras fortemente correlacionadas.
# - Limitações da correlação de Pearson: mede apenas associação linear e não indica causalidade.
#
# *[Interpretação a ser redigida pelo estudante]*
#
# ### 17.6 — Avaliação da variável temporal
#
# **Evidências (BLOCO 16.1):**
#
# | Item | Resultado |
# |---|---|
# | Tipo original | `str` |
# | Datas inválidas | 0 |
# | Data inicial | 02/02/2015 14:19:00 |
# | Data final | 18/02/2015 09:19:00 |
# | Ordem cronológica | True |
# | Intervalos de 59 segundos | 2.485 |
# | Intervalos de 60 segundos | 7.445 |
# | Intervalos de 61 segundos | 2.485 |
# | Maior intervalo | 7 dias, 4 horas e 5 minutos |
# | Índice da descontinuidade | 2665 (início dos registros de `datatest2.txt`) |
#
# A coluna `date` foi preservada em sua forma original; a conversão para `datetime` foi realizada apenas em uma variável temporária.
#
# **Pontos para interpretação:**
#
# - Decisão sobre a utilização ou a exclusão de `date` entre os preditores, com justificativa.
# - Dependência entre observações consecutivas registradas em intervalos de aproximadamente um minuto.
# - Efeito da descontinuidade entre os períodos dos dois arquivos.
#
# *[Interpretação a ser redigida pelo estudante]*
#
# ### 17.7 — Implicações para a modelagem preditiva
#
# Pontos técnicos a serem fundamentados:
#
# - **StandardScaler para o KNN:** relação com as diferenças de escala registradas em 17.2.
# - **Árvore de Decisão sem escalonamento:** motivo da insensibilidade à escala.
# - **Divisão estratificada:** relação com a proporção das classes registrada em 17.4.
# - **Desbalanceamento:** impacto na interpretação dos resultados.
# - **Redundância entre preditoras:** correlações registradas em 17.5.
# - **Dependência temporal:** efeito da divisão aleatória sobre registros consecutivos (17.6).
# - **Acurácia como métrica isolada:** limitações diante do desbalanceamento.
#
# *[Interpretação a ser redigida pelo estudante]*
#
# ### 17.8 — Conclusão da análise exploratória
#
# Resultados que deverão fundamentar a conclusão:
#
# - Base com 12.417 registros, 7 colunas e nenhum valor ausente (17.1).
# - Diferenças de escala entre as variáveis ambientais (17.2).
# - Padrões das distribuições observados nos histogramas (17.3).
# - Proporção de 75,67% e 24,33% entre as classes (17.4).
# - `Light` com a maior correlação com `Occupancy` (0,9197) (17.5).
# - Coluna `date` válida, contínua dentro de cada arquivo e com uma descontinuidade entre os períodos (17.6).
# - Pontos a serem verificados na Fase 2 — Data Prep: duplicidades, valores ausentes e outliers.
#
# *[Conclusão a ser redigida pelo estudante]*