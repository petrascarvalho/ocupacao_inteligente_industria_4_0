
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
# | Padrão do Python | `os`, `zipfile`, `urllib.request` |
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