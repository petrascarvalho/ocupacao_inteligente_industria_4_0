
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
# O projeto tem como objetivo investigar a utilização de dados provenientes de sensores ambientais IoT para identificar
# a ocupação de salas técnicas e escritórios.
#
# A identificação da presença ou ausência de pessoas poderá subsidiar estratégias de automação predial, permitindo
# avaliar possibilidades de acionamento e desligamento de sistemas de iluminação e climatização de acordo com a
# utilização dos ambientes.
#
# A proposta está inserida no contexto da Indústria 4.0, utilizando técnicas de Ciência de Dados e Machine Learning para
# apoiar decisões relacionadas à eficiência energética.
#
# O desenvolvimento possui caráter experimental e não contempla, nesta etapa, integração física com sistemas de
# automação ou comprovação de redução efetiva do consumo energético.
#
# ### Variável alvo
#
# | Variável | Tipo | Valores |
# |---|---|---|
# | `Occupancy` | Binária (`int64`) | 0 = desocupado; 1 = ocupado |
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

# %% [markdown]
# ### BLOCO 01 — Importação das bibliotecas
#
# **Evidências:**
#
# | Grupo | Bibliotecas |
# |---|---|
# | Padrão do Python | `os`, `zipfile`, `urllib.request`, `hashlib` |
# | Análise e manipulação | `pandas`, `numpy` |
# | Visualização | `matplotlib.pyplot`, `seaborn` |
# | Machine Learning | `train_test_split`, `StandardScaler`, `Pipeline`, `DecisionTreeClassifier`, `KNeighborsClassifier`, métricas |
#
# Versões no ambiente `.venv`: Python 3.14.4, pandas 3.0.2, NumPy 2.4.4, Matplotlib 3.10.9, Seaborn 0.13.2, scikit-learn 1.8.0.

# %%
# BLOCO 01 — Importação das bibliotecas

# Bibliotecas padrão do Python
import os
import zipfile
import urllib.request
import hashlib

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
# **Justificativa técnica:**
#
# Para a construção da base consolidada, foram utilizados os arquivos `datatest.txt` e `datatest2.txt`, que apresentam
# estruturas compatíveis e totalizam 12.417 registros.
#
# A utilização dos dois arquivos amplia a quantidade de observações em relação ao exemplo inicial do enunciado,
# permitindo uma análise exploratória sobre uma amostra maior.
#
# O arquivo `datatraining.txt` foi preservado em sua estrutura original, sem inclusão na consolidação adotada neste
# projeto.
#
# A denominação original dos arquivos não determina a divisão de treinamento e teste utilizada nesta aplicação, que será
# realizada posteriormente sobre a base consolidada, conforme a metodologia exigida no enunciado.

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
# **Justificativa técnica:**
#
# A consolidação foi realizada por meio da concatenação vertical dos DataFrames, reunindo os registros dos dois arquivos
# em uma única estrutura de dados.
#
# A utilização do parâmetro `ignore_index=True` permite gerar um novo índice sequencial, evitando a manutenção dos
# índices repetidos provenientes dos arquivos individuais.
#
# Esse procedimento preserva as observações originais e facilita as etapas posteriores de inspeção, análise
# exploratória, tratamento e modelagem.
#
# A compatibilidade estrutural dos arquivos e a quantidade final de registros são verificadas no bloco seguinte.

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
# **Justificativa técnica:**
#
# A exportação da base consolidada para CSV permite disponibilizar uma estrutura única e reproduzível para as etapas
# posteriores do projeto.
#
# O parâmetro `index=False` evita que o índice gerado durante a concatenação seja armazenado como uma coluna adicional
# no arquivo.
#
# Após a exportação, a base é novamente carregada e comparada com o DataFrame consolidado por meio do método
# `assert_frame_equal`, utilizando o parâmetro `check_exact=True`.
#
# Essa validação verifica a equivalência exata dos DataFrames conforme os tipos e valores representados no ambiente de
# execução, contribuindo para a identificação de alterações indevidas durante o processo de exportação e carregamento.
#
# A precisão numérica é preservada, sem aplicação de arredondamento intencional aos dados armazenados.

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
# A análise exploratória constitui uma etapa fundamental para compreender as características da base de dados antes da
# implementação dos modelos preditivos.
#
# A investigação das distribuições, das escalas das variáveis, da proporção das classes e das correlações permite
# identificar aspectos que deverão ser considerados durante a preparação dos dados e a escolha das estratégias de
# modelagem.
#
# Essas informações contribuirão para fundamentar a divisão estratificada dos dados, a aplicação do escalonamento ao KNN
# e a interpretação dos resultados dos modelos.
#
# A análise também permite identificar limitações iniciais e orientar as verificações que serão realizadas na etapa de
# Data Prep.

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

# Verificação da unicidade dos registros temporais
print("\nUnicidade dos registros temporais:")
print("Total de registros:", len(datas))
print("Timestamps distintos:", datas.nunique())
print("Todos os timestamps são únicos:", datas.is_unique)
print("Timestamps repetidos:", datas.duplicated().sum())

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
# **Interpretação:**
#
# A base consolidada apresenta 12.417 registros distribuídos em sete colunas, sem valores ausentes identificados. As
# cinco variáveis ambientais possuem formato numérico `float64`, enquanto a variável alvo `Occupancy` apresenta valores
# inteiros, representando as classes de ocupação.
#
# A variável `date` encontra-se originalmente em formato textual, sendo necessária a avaliação de sua estrutura temporal
# antes da modelagem.
#
# A ausência de valores nulos indica uma condição inicial favorável, porém não elimina a necessidade de investigar
# duplicidades, inconsistências e possíveis valores discrepantes durante a etapa de tratamento dos dados.
#
# ### 17.2 — Estatísticas descritivas
#
# **Evidências (BLOCO 13 — `describe()`):**
#
# | Variável | Média | Q1 (25%) | Mediana (50%) | Q3 (75%) | Desvio padrão | Mínimo | Máximo |
# |---|---:|---:|---:|---:|---:|---:|---:|
# | Temperature (°C) | 21,094510 | 20,390000 | 20,790000 | 21,675000 | 1,037511 | 19,500000 | 24,408333 |
# | Humidity (%) | 28,917947 | 25,390000 | 28,600000 | 31,790000 | 4,125142 | 21,865000 | 39,500000 |
# | Light (lux) | 138,125947 | 0,000000 | 0,000000 | 392,000000 | 219,801233 | 0,000000 | 1.697,250000 |
# | CO2 (ppm) | 745,644624 | 527,500000 | 632,000000 | 857,000000 | 296,497310 | 427,500000 | 2.076,500000 |
# | HumidityRatio (kg/kg) | 0,0044682086 | 0,0039880226 | 0,0045113541 | 0,0049417554 | 0,0005954957 | 0,0032747640 | 0,0057686083 |
#
# Valores apresentados com seis casas decimais, conforme a saída do `describe()`; `HumidityRatio` com dez casas para preservar a precisão. Os dados não foram arredondados.
#
# **Interpretação:**
#
# As estatísticas descritivas demonstram diferenças relevantes nas escalas e na dispersão das variáveis ambientais.
#
# A temperatura apresenta média de 21,09 °C e desvio padrão de 1,04 °C, indicando uma concentração relativamente elevada
# das observações. A umidade relativa apresenta média de 28,92%, próxima à mediana de 28,60%.
#
# A luminosidade possui mediana igual a zero e média de 138,13 lux, indicando que pelo menos metade das observações
# apresenta luminosidade igual a zero, enquanto os valores mais elevados influenciam a média.
#
# O CO2 apresenta média de 745,64 ppm e mediana de 632 ppm, sugerindo uma distribuição assimétrica à direita.
#
# A razão de umidade apresenta valores na ordem dos milésimos, enquanto luminosidade e CO2 alcançam valores superiores a
# mil unidades em suas respectivas escalas.
#
# Essas diferenças justificam a aplicação do escalonamento das variáveis destinadas ao KNN, evitando que determinadas
# características exerçam influência desproporcional no cálculo das distâncias.
#
# ### 17.3 — Distribuição das variáveis ambientais
#
# **Evidência:** `resultados/graficos/eda_01_histogramas.png` (BLOCO 14 — cinco histogramas com 30 intervalos).
#
# **Interpretação:**
#
# A distribuição das variáveis ambientais apresenta comportamentos distintos, refletindo diferentes condições
# registradas pelos sensores:
#
# - `Temperature`: valores concentrados entre aproximadamente 20 °C e 21 °C, com cauda mais longa em direção às temperaturas mais altas (até cerca de 24,4 °C), caracterizando uma leve assimetria à direita.
# - `Humidity`: distribuição dispersa entre aproximadamente 22% e 39,5%, com mais de um pico de frequência (próximos de 25%, 30% e 35%), sem uma concentração única predominante.
# - `Light`: elevada concentração de registros próximos de zero e um agrupamento adicional de observações entre aproximadamente 400 e 550 lux, com poucas ocorrências acima de 800 lux, possivelmente associado a diferentes condições de utilização do ambiente.
# - `CO2`: indícios de assimetria positiva, com maior concentração entre aproximadamente 450 e 600 ppm, agrupamentos menores próximos de 750 ppm e 1.500 ppm e ocorrência esporádica de concentrações mais elevadas, até cerca de 2.076 ppm.
# - `HumidityRatio`: distribuição irregular, com mais de um pico de frequência, comportamento que deve ser analisado considerando sua relação com `Humidity`.
#
# Os valores extremos identificados não devem ser classificados automaticamente como erros, pois podem representar
# condições ambientais legítimas. A investigação de possíveis outliers será aprofundada na etapa de Data Prep,
# utilizando boxplots e o intervalo interquartil.
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
# **Interpretação:**
#
# A distribuição da variável alvo apresenta desbalanceamento entre as classes. Dos 12.417 registros, 9.396 correspondem
# a ambientes desocupados (75,67%), enquanto 3.021 correspondem a ambientes ocupados (24,33%).
#
# Um classificador que previsse exclusivamente a classe majoritária alcançaria uma acurácia de referência de
# aproximadamente 75,67%, sem identificar corretamente nenhuma observação da classe ocupada.
#
# Esse cenário demonstra a importância da utilização da divisão estratificada entre treino e teste, preservando
# aproximadamente as proporções originais das classes.
#
# Além da acurácia, serão consideradas a matriz de confusão, a precisão, o recall e o F1-score, permitindo uma avaliação
# mais abrangente do desempenho dos modelos, especialmente em relação à classe minoritária.
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
# **Interpretação:**
#
# A matriz de correlação de Pearson evidencia que a luminosidade apresenta a maior associação linear positiva com a
# variável `Occupancy`, com coeficiente de 0,9197.
#
# A temperatura apresenta correlação positiva de 0,5764, seguida pelo CO2, com 0,3710, e pela razão de umidade, com
# 0,2420. A umidade relativa apresenta uma associação linear negativa próxima de zero (-0,0427).
#
# Entre as variáveis preditoras, destacam-se as correlações entre `Humidity` e `HumidityRatio` (0,8919) e entre
# `Temperature` e `Light` (0,7248), indicando possível redundância de informações.
#
# Entretanto, a correlação de Pearson não estabelece relações de causalidade e não determina isoladamente a importância
# de uma variável para os modelos preditivos.
#
# Dessa forma, as cinco variáveis ambientais serão inicialmente preservadas, permitindo que seu desempenho conjunto seja
# avaliado durante a modelagem.
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
# | Timestamps distintos | 12.417 |
# | Todos os timestamps são únicos | True |
# | Timestamps repetidos | 0 |
#
# A coluna `date` foi preservada em sua forma original; a conversão para `datetime` foi realizada apenas em uma variável temporária.
#
# **Interpretação:**
#
# A avaliação exploratória da coluna `date` identificou registros compreendidos entre 2 e 18 de fevereiro de 2015,
# organizados em ordem cronológica e sem datas inválidas.
#
# As observações apresentam intervalos predominantemente próximos de um minuto, com uma descontinuidade de sete dias,
# quatro horas e cinco minutos entre os períodos correspondentes aos dois arquivos consolidados.
#
# A verificação de unicidade confirmou que os 12.417 timestamps são distintos, sem repetições temporais. Esse resultado
# não descarta, por si só, a existência de registros com medições idênticas em instantes diferentes, aspecto a ser
# considerado na verificação de duplicidades da etapa de Data Prep.
#
# A coluna temporal será preservada para fins de rastreabilidade e identificação das observações, mas não será utilizada
# diretamente como variável preditora nos modelos numéricos.
#
# Considerando a proximidade temporal entre registros consecutivos, existe a possibilidade de dependência entre as
# observações. A divisão aleatória dos dados pode, portanto, produzir estimativas otimistas de desempenho.
#
# Será mantida a divisão estratificada de 80% para treinamento e 20% para teste, conforme a metodologia estabelecida no
# projeto, registrando-se a dependência temporal como uma limitação potencial da avaliação.
#
# ### 17.7 — Implicações para a modelagem preditiva
#
# Os resultados da análise exploratória direcionam as estratégias de preparação dos dados e modelagem preditiva.
#
# As diferenças de escala entre as variáveis ambientais justificam a utilização do `StandardScaler` para o KNN, uma vez
# que esse algoritmo utiliza distâncias entre observações para realizar a classificação.
#
# A Árvore de Decisão será treinada sem escalonamento, pois sua estrutura utiliza divisões baseadas em limiares
# individuais das variáveis.
#
# A divisão estratificada será utilizada para preservar aproximadamente a distribuição das classes nos conjuntos de
# treinamento e teste.
#
# O escalonamento será ajustado exclusivamente sobre os dados de treinamento e posteriormente aplicado aos dados de
# teste, evitando o vazamento de informações (Data Leakage) associado ao cálculo dos parâmetros de escala.
#
# O desempenho dos modelos será investigado por meio da comparação entre diferentes configurações de hiperparâmetros,
# analisando as acurácias de treinamento e teste para identificar possíveis ocorrências de overfitting.
#
# As métricas complementares permitirão avaliar a capacidade de classificação das duas classes, considerando o
# desbalanceamento identificado.
#
# A possível redundância entre preditoras e a dependência temporal serão consideradas na interpretação dos resultados e
# das limitações dos modelos.
#
# ### 17.8 — Conclusão da análise exploratória
#
# A análise exploratória permitiu compreender a estrutura da base consolidada, as características estatísticas dos
# sensores ambientais e suas associações com a variável alvo.
#
# Foram identificadas diferenças relevantes de escala, desbalanceamento entre as classes e uma forte associação linear
# entre luminosidade e ocupação.
#
# A base não apresentou valores ausentes na inspeção inicial, porém ainda necessita de verificações relacionadas à
# duplicidade, à presença de valores discrepantes e à adequação dos tipos de dados.
#
# Os resultados obtidos estabelecem a fundamentação técnica para as próximas etapas de tratamento, engenharia de
# atributos, escalonamento e modelagem preditiva.
#
# A próxima fase será dedicada à preparação dos dados (Data Prep), buscando preservar a integridade das informações e
# garantir condições adequadas para o treinamento dos modelos de Machine Learning.


###############################################################################################################################################################
# %% [markdown]
# # FASE 2 — Limpeza e Tratamento de Dados (Data Prep)
# ## BLOCO 18 — Contextualização da preparação dos dados
#
# ### Objetivo
#
# A preparação dos dados verifica e assegura a qualidade da base antes da modelagem: identifica duplicidades, valores
# ausentes, tipos inadequados e valores discrepantes, documentando cada decisão e preservando as fontes originais.
#
# ### Data Prep e qualidade dos modelos
#
# O princípio *Garbage In, Garbage Out* indica que modelos treinados com dados inconsistentes produzem resultados
# pouco confiáveis, independentemente do algoritmo utilizado. A preparação dos dados reduz esse risco e torna o
# pipeline reproduzível.
#
# ### Relação com os resultados da EDA
#
# | Resultado da EDA | Verificação nesta fase |
# |---|---|
# | Nenhum valor ausente na inspeção inicial | Confirmar ausentes e infinitos na base de trabalho (BLOCO 21) |
# | 12.417 timestamps distintos | Confirmar a unicidade temporal e investigar duplicidades (BLOCO 20) |
# | Diferenças de escala entre os sensores | Avaliar tipos e precisão; o escalonamento ocorrerá após a divisão treino/teste (BLOCOS 22 e 23) |
# | Indícios de valores extremos em `Light` e `CO2` | Construir boxplots e investigar potenciais outliers pelo IQR (BLOCOS 24 e 25) |
#
# ### Preservação das fontes
#
# Os arquivos de `dados/brutos/` e a base consolidada `dados/consolidados/occupancy.csv` não são alterados. O
# tratamento é realizado sobre uma cópia de trabalho e exportado para `dados/tratados/occupancy_tratado.csv`.

# %% [markdown]
# ### BLOCO 19 — Criação da base de trabalho
#
# **Justificativa técnica:** a cópia independente (`copy(deep=True)`) garante que nenhuma operação desta fase altere
# `df_consolidado`, que permanece como referência para comparação e rastreabilidade.

# %%
# BLOCO 19 — Criação da base de trabalho

# Cópia independente da base consolidada
df_tratado = df_consolidado.copy(deep=True)

# Registro das dimensões iniciais para comparação posterior
registros_iniciais = len(df_tratado)
colunas_iniciais = df_tratado.shape[1]

print("Dimensões iniciais:", df_tratado.shape)
print("Colunas:", list(df_tratado.columns))
print(df_tratado.head())

# Confirmação da equivalência inicial entre as bases
print("\nBase de trabalho equivalente à base consolidada:", df_tratado.equals(df_consolidado))

# %% [markdown]
# ### BLOCO 20 — Identificação e tratamento de duplicidades
#
# A investigação considera três critérios distintos, porque cada um responde a uma pergunta diferente:
#
# | Critério | Pergunta |
# |---|---|
# | 20.1 — Todas as colunas | Existem linhas integralmente repetidas? |
# | 20.2 — Coluna `date` | Existe mais de uma observação registrada no mesmo instante? |
# | 20.3 — Somente os sensores | As mesmas medições ambientais ocorrem em instantes diferentes? |

# %%
# BLOCO 20.1 — Duplicidades considerando todas as colunas

# Registro dos indicadores antes do tratamento
registros_antes_duplicidades = len(df_tratado)
duplicatas_integrais_antes = df_tratado.duplicated().sum()
timestamps_repetidos_antes = df_tratado["date"].duplicated().sum()
classes_antes = df_tratado["Occupancy"].value_counts().sort_index()

# Quantidade de linhas envolvidas em grupos duplicados (todas as ocorrências)
registros_em_grupos_duplicados = df_tratado.duplicated(keep=False).sum()

print("Total de linhas:", registros_antes_duplicidades)
print("Repetições integrais:", duplicatas_integrais_antes)
print("Registros envolvidos em grupos duplicados:", registros_em_grupos_duplicados)

# Remoção somente se existirem duplicatas integrais confirmadas
if duplicatas_integrais_antes > 0:
    print("\nExemplos de registros duplicados:")
    print(df_tratado[df_tratado.duplicated(keep=False)].head(10))
    df_tratado = df_tratado.drop_duplicates().reset_index(drop=True)
    print("Linhas removidas:", registros_antes_duplicidades - len(df_tratado))
else:
    print("\nNenhuma duplicata integral encontrada. Nenhum registro removido.")

# %%
# BLOCO 20.2 — Duplicidades considerando a coluna temporal

timestamps_distintos = df_tratado["date"].nunique()
timestamps_repetidos = df_tratado["date"].duplicated().sum()

# Registros com o mesmo timestamp e valores diferentes (conflitos)
mesmo_instante = df_tratado[df_tratado["date"].duplicated(keep=False)]
conflitos_temporais = mesmo_instante.drop_duplicates()["date"].duplicated().sum()

print("Timestamps distintos:", timestamps_distintos)
print("Timestamps repetidos:", timestamps_repetidos)
print("Conflitos (mesmo instante com valores diferentes):", conflitos_temporais)

# %%
# BLOCO 20.3 — Repetições considerando somente os sensores

# Combinações repetidas das cinco variáveis ambientais (sem date)
repeticoes_sensores = df_tratado.duplicated(subset=colunas_numericas).sum()
envolvidos_sensores = df_tratado.duplicated(subset=colunas_numericas, keep=False).sum()

# Combinações repetidas dos sensores incluindo Occupancy (sem date)
colunas_sensores_classe = colunas_numericas + ["Occupancy"]
repeticoes_sensores_classe = df_tratado.duplicated(subset=colunas_sensores_classe).sum()
envolvidos_sensores_classe = df_tratado.duplicated(subset=colunas_sensores_classe, keep=False).sum()

# Mesmas medições ambientais associadas a classes diferentes
classes_por_medicao = df_tratado.groupby(colunas_numericas)["Occupancy"].nunique()
medicoes_com_classes_diferentes = (classes_por_medicao > 1).sum()

print("Somente sensores — repetições:", repeticoes_sensores)
print("Somente sensores — registros envolvidos:", envolvidos_sensores)
print("Sensores + Occupancy — repetições:", repeticoes_sensores_classe)
print("Sensores + Occupancy — registros envolvidos:", envolvidos_sensores_classe)
print("Medições iguais associadas a classes diferentes:", medicoes_com_classes_diferentes)

# Exemplos de medições iguais registradas em instantes diferentes
print("\nExemplos de medições iguais em horários diferentes:")
exemplos = df_tratado[df_tratado.duplicated(subset=colunas_numericas, keep=False)]
print(exemplos.sort_values(colunas_numericas).head(6))

# %%
# BLOCO 20.4 — Validação após o tratamento de duplicidades

classes_depois = df_tratado["Occupancy"].value_counts().sort_index()

tabela_duplicidades = pd.DataFrame({
    "Antes": [
        registros_antes_duplicidades,
        duplicatas_integrais_antes,
        timestamps_repetidos_antes,
        classes_antes[0],
        classes_antes[1],
    ],
    "Depois": [
        len(df_tratado),
        df_tratado.duplicated().sum(),
        df_tratado["date"].duplicated().sum(),
        classes_depois[0],
        classes_depois[1],
    ],
}, index=["Registros", "Duplicatas integrais", "Timestamps repetidos", "Classe desocupado", "Classe ocupado"])

print(tabela_duplicidades)

# %% [markdown]
# ### BLOCO 21 — Verificação e tratamento de valores ausentes
#
# A verificação considera valores ausentes (`NaN`) em todas as colunas e valores infinitos nas colunas numéricas.
# Valores iguais a zero não são tratados como ausentes: em `Light`, zero representa uma leitura válida do sensor
# (ambiente sem iluminação).

# %%
# BLOCO 21 — Verificação de valores ausentes e infinitos

ausentes_por_coluna = df_tratado.isna().sum()
percentual_ausentes = (ausentes_por_coluna / len(df_tratado) * 100).round(4)

tabela_ausentes = pd.DataFrame({
    "Ausentes": ausentes_por_coluna,
    "Percentual (%)": percentual_ausentes,
})
print(tabela_ausentes)

total_ausentes = ausentes_por_coluna.sum()
registros_com_ausentes = df_tratado.isna().any(axis=1).sum()
infinitos_por_coluna = np.isinf(df_tratado[colunas_numericas]).sum()

print("\nTotal de valores ausentes:", total_ausentes)
print("Registros afetados:", registros_com_ausentes)
print("Valores infinitos nas colunas numéricas:", infinitos_por_coluna.sum())
print("Leituras de Light iguais a zero (valores válidos):", (df_tratado["Light"] == 0).sum())

# %%
# BLOCO 21.1 — Tratamento da base real

if total_ausentes == 0 and infinitos_por_coluna.sum() == 0:
    print("A base real não apresenta valores ausentes nem infinitos.")
    print("Nenhuma imputação foi aplicada; os dados foram integralmente preservados.")
else:
    print("ATENÇÃO: valores ausentes ou infinitos encontrados. Investigar a origem antes de qualquer tratamento.")
    print(df_tratado[df_tratado.isna().any(axis=1)].head(10))

# %% [markdown]
# ### BLOCO 21.2 — Demonstração isolada de imputação (simulação didática)
#
# **Esta seção é uma simulação.** Como a base real não possui valores ausentes, a imputação é demonstrada em uma cópia
# temporária e independente (`df_simulacao`), sem qualquer efeito sobre `df_tratado` ou sobre o CSV exportado.
#
# **Critério de escolha:** a média é adequada para distribuições aproximadamente simétricas; a mediana é preferível em
# distribuições assimétricas, por ser resistente a valores extremos. A assimetria (*skewness*) calculada abaixo
# fundamenta a escolha das variáveis.
#
# Em uma situação real de modelagem, a estatística de imputação deve ser calculada exclusivamente sobre o conjunto de
# treinamento e depois aplicada ao conjunto de teste, evitando Data Leakage.

# %%
# BLOCO 21.2 — Demonstração isolada de imputação (simulação didática)

# Assimetria das variáveis ambientais (fundamenta a escolha entre média e mediana)
assimetria = df_tratado[colunas_numericas].skew()
print("Assimetria (skewness) das variáveis ambientais:")
print(assimetria.round(4))

# Cópia temporária e independente para a simulação
df_simulacao = df_tratado[["Humidity", "CO2"]].copy(deep=True)
ausentes_iniciais_simulacao = df_simulacao.isna().sum().sum()

# Simulação reproduzível de valores ausentes (semente fixa)
gerador = np.random.default_rng(42)
quantidade_simulada = 50
indices_humidity = gerador.choice(df_simulacao.index, size=quantidade_simulada, replace=False)
indices_co2 = gerador.choice(df_simulacao.index, size=quantidade_simulada, replace=False)

valores_reais_humidity = df_simulacao.loc[indices_humidity, "Humidity"].copy()
valores_reais_co2 = df_simulacao.loc[indices_co2, "CO2"].copy()

df_simulacao.loc[indices_humidity, "Humidity"] = np.nan
df_simulacao.loc[indices_co2, "CO2"] = np.nan

print("\nAusentes antes da simulação:", ausentes_iniciais_simulacao)
print("Ausentes após a simulação:", df_simulacao.isna().sum().to_dict())

# Estatísticas calculadas sem os valores simulados como ausentes
media_humidity = df_simulacao["Humidity"].mean()
mediana_co2 = df_simulacao["CO2"].median()

print("\nHumidity — imputação pela média:", round(media_humidity, 6))
print("CO2 — imputação pela mediana:", round(mediana_co2, 6))

# Imputação
df_simulacao["Humidity"] = df_simulacao["Humidity"].fillna(media_humidity)
df_simulacao["CO2"] = df_simulacao["CO2"].fillna(mediana_co2)

print("\nAusentes após a imputação:", df_simulacao.isna().sum().to_dict())

# Diferença média entre os valores imputados e os valores reais ocultados
erro_humidity = (df_simulacao.loc[indices_humidity, "Humidity"] - valores_reais_humidity).abs().mean()
erro_co2 = (df_simulacao.loc[indices_co2, "CO2"] - valores_reais_co2).abs().mean()
print("Erro absoluto médio — Humidity:", round(erro_humidity, 4))
print("Erro absoluto médio — CO2:", round(erro_co2, 4))

# Confirmação de que a base de trabalho não foi alterada
print("\nValores ausentes em df_tratado após a simulação:", df_tratado.isna().sum().sum())

# %% [markdown]
# ### BLOCO 22 — Avaliação e adequação dos tipos de dados
#
# **Justificativa técnica:** a coluna `date` é convertida para `datetime` apenas na base de trabalho, permitindo
# validar o formato, a ordem e a unicidade temporal. A coluna é mantida para rastreabilidade e será excluída das
# variáveis preditoras somente na separação entre X e y (Fase 4). Os tipos numéricos originais são preservados.

# %%
# BLOCO 22 — Avaliação e adequação dos tipos de dados

tipos_antes = df_tratado.dtypes.astype(str)

# Conversão da coluna date somente na base de trabalho
df_tratado["date"] = pd.to_datetime(df_tratado["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

tipos_depois = df_tratado.dtypes.astype(str)
print(pd.DataFrame({"Antes": tipos_antes, "Depois": tipos_depois}))

print("\nDatas inválidas após a conversão:", df_tratado["date"].isna().sum())
print("Timestamps preservados:", df_tratado["date"].notna().sum(), "de", len(df_tratado))
print("Ordem cronológica:", df_tratado["date"].is_monotonic_increasing)
print("Timestamps únicos:", df_tratado["date"].is_unique)
print("Classes presentes em Occupancy:", sorted(df_tratado["Occupancy"].unique().tolist()))
print("Tipo de date em df_consolidado (preservado):", df_consolidado["date"].dtype)

# %% [markdown]
# ### BLOCO 23 — Avaliação da precisão numérica
#
# A precisão armazenada é diferente da formatação exibida: o pandas mostra apenas algumas casas decimais, mas o valor
# em memória mantém a precisão completa do `float64`. Nenhum arredondamento é aplicado aos dados. A conversão para um
# tipo de menor precisão (`float32`) é testada apenas para demonstrar a perda que ela provocaria.

# %%
# BLOCO 23 — Avaliação da precisão numérica

print("Tipos das variáveis ambientais:")
print(df_tratado[colunas_numericas].dtypes)

# Diferença entre o valor exibido e o valor armazenado
exemplo = df_tratado.loc[0, "HumidityRatio"]
print("\nHumidityRatio (registro 0) — valor armazenado:", repr(float(exemplo)))
print("HumidityRatio (registro 0) — exibido com 6 casas:", f"{exemplo:.6f}")

# Perda de precisão caso os dados fossem convertidos para float32
erro_relativo_float32 = (
    (df_tratado[colunas_numericas].astype("float32").astype("float64") - df_tratado[colunas_numericas]).abs()
    / df_tratado[colunas_numericas].abs().replace(0, np.nan)
).max()
print("\nErro relativo máximo se os dados fossem convertidos para float32:")
print(erro_relativo_float32)

# Confirmação de que os valores numéricos não foram alterados
print("\nValores numéricos idênticos à base consolidada:",
      df_tratado[colunas_numericas].equals(df_consolidado[colunas_numericas]))

# %% [markdown]
# ### BLOCO 24 — Construção dos boxplots
#
# Cada variável é apresentada em um eixo próprio, pois as unidades e magnitudes são diferentes. No boxplot, a caixa
# vai de Q1 a Q3, a linha interna é a mediana, as hastes terminam nas observações mais extremas dentro dos limites
# de 1,5 × IQR e os pontos isolados são potenciais outliers.

# %%
# BLOCO 24 — Construção dos boxplots

fig, eixos_boxplot = plt.subplots(2, 3, figsize=(15, 10))
eixos_boxplot = eixos_boxplot.flatten()

for eixo, coluna in zip(eixos_boxplot, colunas_numericas):
    eixo.boxplot(
        df_tratado[coluna],
        widths=0.5,
        whis=1.5,
        patch_artist=True,
        boxprops={"facecolor": "#4C72B0", "alpha": 0.8},
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    eixo.set_title(coluna)
    eixo.set_ylabel(unidades[coluna])
    eixo.set_xticks([])

# Ocultação do eixo excedente
eixos_boxplot[-1].set_visible(False)

plt.suptitle("Boxplots das variáveis ambientais", fontsize=16)
plt.tight_layout()

# Exportação antes da exibição
caminho_figura = f"{pasta_graficos}/dp_01_boxplots.png"
plt.savefig(caminho_figura, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_figura)

plt.show()

# %% [markdown]
# ### BLOCO 25 — Investigação de outliers pelo IQR
#
# Para cada variável: IQR = Q3 − Q1; limite inferior = Q1 − 1,5 × IQR; limite superior = Q3 + 1,5 × IQR. São
# potenciais outliers os valores estritamente abaixo do limite inferior ou acima do limite superior. A coluna
# "Ocupado (%)" indica a proporção da classe 1 entre os potenciais outliers, para comparação com a proporção geral
# da base (24,33%).

# %%
# BLOCO 25 — Investigação de outliers pelo IQR

linhas_iqr = []
for coluna in colunas_numericas:
    q1 = df_tratado[coluna].quantile(0.25)
    q3 = df_tratado[coluna].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    abaixo = df_tratado[coluna] < limite_inferior
    acima = df_tratado[coluna] > limite_superior
    total = (abaixo | acima).sum()

    ocupado = df_tratado.loc[abaixo | acima, "Occupancy"].mean() * 100 if total > 0 else np.nan

    linhas_iqr.append({
        "Variável": coluna,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Limite inferior": limite_inferior,
        "Limite superior": limite_superior,
        "Abaixo": abaixo.sum(),
        "Acima": acima.sum(),
        "Total": total,
        "Percentual (%)": round(total / len(df_tratado) * 100, 2),
        "Ocupado (%)": round(ocupado, 2),
    })

tabela_iqr = pd.DataFrame(linhas_iqr).set_index("Variável")
print(tabela_iqr.to_string())
print("\nProporção geral da classe ocupado (%):", round(df_tratado["Occupancy"].mean() * 100, 2))

# %%
# BLOCO 25.2 — Plausibilidade física

verificacoes_fisicas = pd.Series({
    "Light negativo": (df_tratado["Light"] < 0).sum(),
    "Humidity fora de 0% a 100%": ((df_tratado["Humidity"] < 0) | (df_tratado["Humidity"] > 100)).sum(),
    "CO2 negativo": (df_tratado["CO2"] < 0).sum(),
    "HumidityRatio negativo": (df_tratado["HumidityRatio"] < 0).sum(),
    "Valores infinitos": np.isinf(df_tratado[colunas_numericas]).sum().sum(),
    "Colunas ambientais não numéricas": sum(
        not pd.api.types.is_float_dtype(df_tratado[coluna]) for coluna in colunas_numericas
    ),
})
print("Verificações de plausibilidade física (quantidade de ocorrências):")
print(verificacoes_fisicas)

print("\nMínimos e máximos observados:")
print(df_tratado[colunas_numericas].agg(["min", "max"]).T)

# %%
# BLOCO 25.3 — Decisão sobre o tratamento de outliers

# Nenhuma observação é removida ou transformada: os valores sinalizados pelo IQR são fisicamente plausíveis
print("Registros antes da decisão:", len(df_tratado))
print("Observações removidas por outliers: 0")
print("Valores numéricos preservados:",
      df_tratado[colunas_numericas].equals(df_consolidado[colunas_numericas]))

# %% [markdown]
# ### BLOCO 26 — Validação da base tratada
#
# As verificações com `assert` interrompem a execução caso a base tratada apresente alguma inconsistência, impedindo a
# exportação de um arquivo inválido.

# %%
# BLOCO 26 — Validação da base tratada

tabela_validacao = pd.DataFrame({
    "Base consolidada": [
        len(df_consolidado),
        df_consolidado.shape[1],
        df_consolidado.isna().sum().sum(),
        df_consolidado.duplicated().sum(),
        df_consolidado["date"].nunique(),
        (df_consolidado["Occupancy"] == 0).sum(),
        (df_consolidado["Occupancy"] == 1).sum(),
    ],
    "Base tratada": [
        len(df_tratado),
        df_tratado.shape[1],
        df_tratado.isna().sum().sum(),
        df_tratado.duplicated().sum(),
        df_tratado["date"].nunique(),
        (df_tratado["Occupancy"] == 0).sum(),
        (df_tratado["Occupancy"] == 1).sum(),
    ],
}, index=["Registros", "Colunas", "Valores ausentes", "Duplicatas integrais",
          "Timestamps distintos", "Classe 0", "Classe 1"])
tabela_validacao["Diferença"] = tabela_validacao["Base tratada"] - tabela_validacao["Base consolidada"]
print(tabela_validacao)

# Verificações que impedem a exportação de uma base inconsistente
assert list(df_tratado.columns) == list(df_consolidado.columns), "Colunas divergentes."
assert df_tratado.isna().sum().sum() == 0, "Existem valores ausentes."
assert np.isinf(df_tratado[colunas_numericas]).sum().sum() == 0, "Existem valores infinitos."
assert df_tratado.duplicated().sum() == 0, "Existem duplicatas integrais."
assert df_tratado["date"].is_unique, "Existem timestamps repetidos."
assert set(df_tratado["Occupancy"].unique()) <= {0, 1}, "Occupancy possui valores fora de {0, 1}."
assert pd.api.types.is_datetime64_any_dtype(df_tratado["date"]), "date não está em formato datetime."

# Preservação dos valores originais quando nenhum registro foi removido
if len(df_tratado) == len(df_consolidado):
    pd.testing.assert_frame_equal(
        df_tratado[colunas_sensores_classe], df_consolidado[colunas_sensores_classe], check_exact=True
    )
    print("\nValores dos sensores e da variável alvo idênticos à base consolidada.")

print("Validação da base tratada concluída com sucesso!")

# %% [markdown]
# ### BLOCO 27 — Exportação e validação da base tratada
#
# A exportação utiliza `index=False` e preserva os segundos da coluna `date` (`%Y-%m-%d %H:%M:%S`), sem formatação
# das casas decimais. Após a recarga, `date` é convertida novamente para `datetime` e a equivalência é verificada com
# `check_exact=True`.

# %%
# BLOCO 27 — Exportação e validação da base tratada

df_tratado.to_csv(caminho_tratado, index=False, date_format="%Y-%m-%d %H:%M:%S")
print("Base tratada exportada:", caminho_tratado)

# Recarga e conversão da coluna temporal
df_tratado_recarregado = pd.read_csv(caminho_tratado)
df_tratado_recarregado["date"] = pd.to_datetime(
    df_tratado_recarregado["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)

print("Dimensões:", df_tratado_recarregado.shape)
print(df_tratado_recarregado.dtypes)
print("Valores ausentes:", df_tratado_recarregado.isna().sum().sum())
print(df_tratado_recarregado["Occupancy"].value_counts().sort_index())

# Equivalência exata entre a base tratada e a base recarregada
pd.testing.assert_frame_equal(df_tratado, df_tratado_recarregado, check_exact=True)
print("\nBase recarregada idêntica à base tratada.")

# Tamanho e SHA-256 do arquivo exportado
with open(caminho_tratado, "rb") as arquivo:
    hash_tratado = hashlib.sha256(arquivo.read()).hexdigest()

print("Tamanho do arquivo (bytes):", os.path.getsize(caminho_tratado))
print("SHA-256:", hash_tratado)

# %% [markdown]
# ## BLOCO 28 — Interpretação e conclusão do Data Prep
#
# ### 28.1 — Qualidade inicial da base
#
# A base de trabalho foi criada como cópia independente da base consolidada, com 12.417 registros e 7 colunas, e
# confirmada como equivalente a ela (`True`). Todas as verificações desta fase partiram, portanto, dos mesmos dados
# analisados na EDA, enquanto `df_consolidado` permaneceu inalterado como referência.
#
# ### 28.2 — Investigação de duplicidades
#
# | Critério | Repetições | Registros envolvidos |
# |---|---:|---:|
# | Todas as colunas (inclui `date`) | 0 | 0 |
# | Coluna `date` | 0 | — |
# | Somente os cinco sensores | 571 | 1.032 |
# | Sensores + `Occupancy` (sem `date`) | 571 | 1.032 |
#
# | Indicador | Antes | Depois |
# |---|---:|---:|
# | Registros | 12.417 | 12.417 |
# | Duplicatas integrais | 0 | 0 |
# | Timestamps repetidos | 0 | 0 |
# | Classe desocupado | 9.396 | 9.396 |
# | Classe ocupado | 3.021 | 3.021 |
#
# Não existem linhas integralmente duplicadas nem observações registradas no mesmo instante; por isso, nenhum registro
# foi removido. As 571 repetições encontradas ao desconsiderar `date` correspondem a medições ambientais iguais em
# instantes diferentes (por exemplo, leituras idênticas registradas com cerca de 20 minutos de diferença na madrugada de
# 14/02/2015, com o ambiente desocupado). Em nenhum caso a mesma combinação de medições aparece associada a classes
# diferentes. Essas repetições são compatíveis com um ambiente estável e com a resolução dos sensores, representando
# observações físicas legítimas; removê-las reduziria artificialmente a amostra e alteraria a frequência real dessas
# condições.
#
# ### 28.3 — Tratamento de valores ausentes
#
# **Base real:** 0 valores ausentes em todas as colunas, 0 registros afetados e 0 valores infinitos. Nenhuma imputação
# foi aplicada. As 7.612 leituras de `Light` iguais a zero (61,30% da base) foram mantidas, pois representam ambientes
# sem iluminação e não ausência de medição.
#
# **Simulação didática (cópia temporária `df_simulacao`):**
#
# | Variável | Assimetria | Ausentes simulados | Estratégia | Valor imputado | Ausentes após | Erro absoluto médio |
# |---|---:|---:|---|---:|---:|---:|
# | Humidity | 0,3008 | 50 | Média | 28,918288 % | 0 | 3,1983 % |
# | CO2 | 1,3851 | 50 | Mediana | 632,0 ppm | 0 | 192,6317 ppm |
#
# A média foi aplicada a `Humidity`, cuja assimetria próxima de zero indica distribuição aproximadamente simétrica. A
# mediana foi aplicada a `CO2`, cuja assimetria positiva elevada faria a média ser deslocada pelos valores altos.
# `Temperature`, inicialmente considerada para a média, apresentou assimetria de 1,1041 e, por isso, não foi utilizada
# como exemplo de distribuição simétrica. A simulação utilizou semente fixa (42) e não alterou `df_tratado` (0 ausentes
# após a demonstração). Em um cenário real, a estatística de imputação deve ser calculada apenas sobre o conjunto de
# treinamento.
#
# ### 28.4 — Adequação dos tipos de dados
#
# | Coluna | Antes | Depois |
# |---|---|---|
# | `date` | `str` | `datetime64[us]` |
# | Cinco variáveis ambientais | `float64` | `float64` |
# | `Occupancy` | `int64` | `int64` |
#
# A conversão de `date` não gerou datas inválidas, preservou os 12.417 timestamps, manteve a ordem cronológica e a
# unicidade. `Occupancy` contém apenas as classes 0 e 1. A coluna `date` de `df_consolidado` permaneceu como `str`. A
# coluna temporal será mantida para rastreabilidade e excluída dos preditores na separação entre X e y.
#
# ### 28.5 — Preservação da precisão
#
# Os valores armazenados mantêm a precisão completa do `float64`: o primeiro registro de `HumidityRatio`, exibido como
# 0,004764, está armazenado como 0,0047641630241641. A conversão para `float32` introduziria erros relativos de até
# aproximadamente 5,9 × 10⁻⁸; como não há justificativa de desempenho para essa redução, os tipos `float64` foram
# preservados. Nenhum arredondamento foi aplicado e os valores numéricos permanecem idênticos aos da base consolidada.
# As diferenças de escala serão tratadas pelo `StandardScaler`, exclusivamente para o KNN e somente após a divisão
# treino/teste.
#
# ### 28.6 — Análise dos boxplots e outliers
#
# Evidência visual: `resultados/graficos/dp_01_boxplots.png`.
#
# | Variável | Q1 | Q3 | IQR | Limite inferior | Limite superior | Abaixo | Acima | Total | Percentual | Ocupado entre os outliers |
# |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | Temperature (°C) | 20,390000 | 21,675000 | 1,285000 | 18,462500 | 23,602500 | 0 | 461 | 461 | 3,71% | 44,69% |
# | Humidity (%) | 25,390000 | 31,790000 | 6,400000 | 15,790000 | 41,390000 | 0 | 0 | 0 | 0,00% | — |
# | Light (lux) | 0,000000 | 392,000000 | 392,000000 | -588,000000 | 980,000000 | 0 | 7 | 7 | 0,06% | 100,00% |
# | CO2 (ppm) | 527,500000 | 857,000000 | 329,500000 | 33,250000 | 1.351,250000 | 0 | 992 | 992 | 7,99% | 30,95% |
# | HumidityRatio (kg/kg) | 0,0039880226 | 0,0049417554 | 0,0009537328 | 0,0025574233 | 0,0063723546 | 0 | 0 | 0 | 0,00% | — |
#
# - `Temperature`: 461 potenciais outliers, todos acima de 23,60 °C, coerentes com a cauda à direita observada no
#   histograma. Entre eles, 44,69% são registros ocupados, proporção superior aos 24,33% da base.
# - `Humidity` e `HumidityRatio`: nenhum valor fora dos limites, em concordância com as distribuições sem caudas
#   longas observadas na EDA.
# - `Light`: apenas 7 valores acima de 980 lux (máximo de 1.697,25 lux), todos em registros ocupados. O limite inferior
#   negativo resulta de Q1 = 0 e não tem significado físico; a concentração de leituras em zero aparece como a mediana
#   coincidente com a base da caixa.
# - `CO2`: 992 potenciais outliers (7,99%), todos acima de 1.351,25 ppm, refletindo a assimetria positiva. Entre eles,
#   30,95% são registros ocupados.
#
# **Plausibilidade física:** não há luminosidade, CO2 ou razão de umidade negativos, umidade relativa fora de 0% a
# 100% nem valores infinitos; as cinco variáveis permanecem numéricas. Os valores extremos encontram-se em faixas
# compatíveis com ambientes internos (luz natural intensa, maior concentração de pessoas ou ventilação reduzida).
#
# **Decisão:** nenhuma observação foi removida, truncada ou substituída. Os potenciais outliers são fisicamente
# plausíveis e concentram proporções de ocupação superiores à média da base, indicando que carregam informação
# relevante para a classificação. Os limites IQR foram calculados com finalidade exploratória; caso venham a ser usados
# em alguma transformação do modelo, deverão ser ajustados exclusivamente sobre o conjunto de treinamento.
#
# ### 28.7 — Validação da base tratada
#
# | Indicador | Base consolidada | Base tratada | Diferença |
# |---|---:|---:|---:|
# | Registros | 12.417 | 12.417 | 0 |
# | Colunas | 7 | 7 | 0 |
# | Valores ausentes | 0 | 0 | 0 |
# | Duplicatas integrais | 0 | 0 | 0 |
# | Timestamps distintos | 12.417 | 12.417 | 0 |
# | Classe 0 | 9.396 | 9.396 | 0 |
# | Classe 1 | 3.021 | 3.021 | 0 |
#
# Todas as verificações `assert` foram aprovadas, e os valores dos sensores e da variável alvo são idênticos aos da
# base consolidada (`check_exact=True`). A base foi exportada para `dados/tratados/occupancy_tratado.csv` (846.701
# bytes) e a recarga, com `date` convertida para `datetime`, reproduziu exatamente a base tratada. Como nenhum valor
# foi alterado e a data foi exportada no mesmo formato textual, o arquivo tratado é idêntico, byte a byte, ao CSV
# consolidado (mesmo SHA-256); a diferença entre as bases está na validação documentada e no tipo `datetime` adotado
# em memória para a coluna temporal.
#
# ### 28.8 — Conclusão do Data Prep
#
# A preparação dos dados confirmou que a base consolidada não apresenta duplicatas integrais, timestamps repetidos,
# valores ausentes, valores infinitos ou valores fisicamente implausíveis. As repetições de medições em instantes
# diferentes e os potenciais outliers identificados pelo IQR foram investigados e mantidos, por representarem
# condições ambientais legítimas e informativas para a ocupação. A base tratada preserva os 12.417 registros, a
# proporção das classes (75,67% e 24,33%) e a precisão original dos dados.
#
# A próxima etapa é a Fase 3 — Feature Engineering, que criará uma nova variável numérica a partir de combinações
# matemáticas das variáveis existentes. O escalonamento, a divisão treino/teste e a modelagem serão realizados nas
# fases seguintes, respeitando a ordem que evita o vazamento de informações.
