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
# # PREPARAÇÃO INICIAL E ETL (BLOCOS 01 A 09)
#
# Esta seção prepara o ambiente e consolida a base de dados. Ela antecede as sete fases avaliativas do enunciado
# (EDA, Data Prep, Feature Engineering, divisão dos dados, escalonamento, hiperparâmetros e avaliação final) e não
# altera sua numeração.
#
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
# | Exibição dos DataFrames | Todas as colunas, largura de 200 caracteres, sem quebra da tabela |
#
# Mesma configuração gráfica do exemplo didático do enunciado (p. 3). As opções de exibição do pandas evitam que
# tabelas como a do `describe()` sejam truncadas com `...` na execução em terminal; elas alteram somente a apresentação,
# não a precisão armazenada nem os valores dos dados.

# %%
# BLOCO 02 — Configurações visuais

# Configurando o estilo dos gráficos
sns.set_theme(style="whitegrid")

# Definindo o tamanho padrão dos gráficos
plt.rcParams["figure.figsize"] = (10, 5)

# Configurando a exibição dos DataFrames (somente apresentação)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.expand_frame_repr", False)

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
# | `caminho_datatraining` | `dados/brutos/datatraining.txt` (somente leitura no BLOCO 06.1; fora da consolidação) |
# | `caminho_csv` | `dados/consolidados/occupancy.csv` |
# | `caminho_tratado` | `dados/tratados/occupancy_tratado.csv` (gerado na Fase 2) |
# | `pasta_features` | `dados/features` |
# | `caminho_features` | `dados/features/occupancy_features.csv` (gerado na Fase 3) |
# | `caminho_grafico_split` | `resultados/graficos/split_01_divisao_treino_teste.png` (gerado na Fase 4) |
# | `caminho_grafico_scaler` | `resultados/graficos/scaler_01_padronizacao_atributos.png` (gerado na Fase 5) |
# | `pasta_graficos` / `pasta_metricas` | `resultados/graficos` / `resultados/metricas` |
#
# Conceitos: o item 5.1 do enunciado proíbe caminhos absolutos; `os.makedirs(..., exist_ok=True)` cria as pastas sem erro
# quando já existem; o Git não versiona pastas vazias. O arquivo de features só existe após a primeira execução do
# BLOCO 35; antes disso, sua verificação retorna `False`.

# %%
# BLOCO 03 — Caminhos relativos e diretórios

# Definindo os diretórios do projeto
pasta_brutos = "dados/brutos"
pasta_consolidados = "dados/consolidados"
pasta_tratados = "dados/tratados"
pasta_features = "dados/features"
pasta_graficos = "resultados/graficos"
pasta_metricas = "resultados/metricas"

# Criando os diretórios, caso não existam
os.makedirs(pasta_brutos, exist_ok=True)
os.makedirs(pasta_consolidados, exist_ok=True)
os.makedirs(pasta_tratados, exist_ok=True)
os.makedirs(pasta_features, exist_ok=True)
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_metricas, exist_ok=True)

# Definindo os caminhos dos arquivos brutos
zip_path = "dados/brutos/occupancy_data.zip"
caminho_datatest = "dados/brutos/datatest.txt"
caminho_datatest2 = "dados/brutos/datatest2.txt"
caminho_datatraining = "dados/brutos/datatraining.txt"

# Definindo o caminho do arquivo consolidado
caminho_csv = "dados/consolidados/occupancy.csv"

# Definindo o caminho do arquivo tratado
caminho_tratado = "dados/tratados/occupancy_tratado.csv"

# Definindo o caminho do arquivo com a nova variável (Feature Engineering)
caminho_features = "dados/features/occupancy_features.csv"

# Definindo o caminho do gráfico da divisão treino/teste (Fase 4)
caminho_grafico_split = f"{pasta_graficos}/split_01_divisao_treino_teste.png"

# Definindo o caminho do gráfico da padronização dos atributos (Fase 5)
caminho_grafico_scaler = f"{pasta_graficos}/scaler_01_padronizacao_atributos.png"

# Validando o diretório de trabalho
print("Diretório atual:", os.getcwd())

# Verificando a existência dos arquivos
print("Arquivo ZIP:", os.path.exists(zip_path))
print("Arquivo datatest:", os.path.exists(caminho_datatest))
print("Arquivo datatest2:", os.path.exists(caminho_datatest2))
print("Arquivo datatraining:", os.path.exists(caminho_datatraining))
print("Arquivo consolidado:", os.path.exists(caminho_csv))
print("Arquivo tratado:", os.path.exists(caminho_tratado))
print("Arquivo de features:", os.path.exists(caminho_features))


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
# **Composição da base — fatos, escolhas e inferências:**
#
# | Tipo | Conteúdo |
# |---|---|
# | Fato documentado (UCI) | O pacote contém três arquivos, 20.560 registros no total, descritos como conjuntos "for training and testing", sem atribuição explícita de papel a cada arquivo |
# | Fato documentado (enunciado, p. 3) | O exemplo do professor lê somente `datatest.txt`; a Fase 4 exige a divisão 80/20 estratificada da base utilizada |
# | Escolha metodológica deste projeto | Consolidar `datatest.txt` e `datatest2.txt` (12.417 registros) e manter `datatraining.txt` fora da base |
# | Inferência pelos nomes dos arquivos | No estudo original, `datatraining` teria sido o conjunto de treino e os dois `datatest` os conjuntos de teste |
#
# A consolidação dos dois arquivos `datatest` é uma escolha deste projeto, distinta do protocolo original de avaliação
# do conjunto de dados; ela não é uma exigência do enunciado. O critério que levou a manter `datatraining.txt` fora da
# consolidação não foi registrado quando a decisão foi tomada e, por isso, não é reconstruído aqui.
#
# A ampliação em relação ao exemplo do enunciado aumenta a quantidade de observações disponíveis para a análise. Os
# nomes originais dos arquivos não determinam a divisão treino/teste desta aplicação, que será realizada na Fase 4 sobre
# a base consolidada.
#
# O arquivo `datatraining.txt` permanece preservado e não é incorporado à base. O BLOCO 06.1 apenas lê seus períodos e
# proporções de classe para documentar a composição; ele poderá ser usado, ao final, em uma avaliação complementar
# claramente identificada, sem substituir a divisão exigida.

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
# ### BLOCO 06.1 — Inventário temporal dos três arquivos originais
#
# Leitura somente para documentação da composição da base. `datatraining.txt` não é concatenado, tratado ou utilizado
# em nenhuma análise de modelagem.

# %%
# BLOCO 06.1 — Inventário temporal dos três arquivos originais

# Leitura somente para inventário (não incorporado à base consolidada)
df_inventario_training = pd.read_csv(caminho_datatraining)

arquivos_originais = {
    "datatest.txt": df1,
    "datatraining.txt": df_inventario_training,
    "datatest2.txt": df2,
}

linhas_inventario = []
for nome, dados in arquivos_originais.items():
    datas_arquivo = pd.to_datetime(dados["date"], format="%Y-%m-%d %H:%M:%S")
    linhas_inventario.append({
        "Arquivo": nome,
        "Registros": len(dados),
        "Início": datas_arquivo.min(),
        "Fim": datas_arquivo.max(),
        "Ocupados": int(dados["Occupancy"].sum()),
        "Ocupados (%)": round(dados["Occupancy"].mean() * 100, 2),
        "Na base consolidada": nome != "datatraining.txt",
    })

tabela_inventario = pd.DataFrame(linhas_inventario).set_index("Arquivo")
print(tabela_inventario)
print("\nTotal do pacote original:", tabela_inventario["Registros"].sum())
print("Total da base consolidada:", tabela_inventario.loc[tabela_inventario["Na base consolidada"], "Registros"].sum())

# Intervalos entre o fim de um arquivo e o início do seguinte (ordem cronológica)
print("\nFim de datatest.txt -> início de datatraining.txt:",
      tabela_inventario.loc["datatraining.txt", "Início"] - tabela_inventario.loc["datatest.txt", "Fim"])
print("Fim de datatraining.txt -> início de datatest2.txt:",
      tabela_inventario.loc["datatest2.txt", "Início"] - tabela_inventario.loc["datatraining.txt", "Fim"])


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
# | 16.2 | Dependência temporal entre observações consecutivas |
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

# %% [markdown]
# ### BLOCO 11 — Dimensões e estrutura dos dados
#
# Apresenta a quantidade de linhas e colunas (`shape`), os nomes das colunas e os primeiros registros da base consolidada.

# %%
# BLOCO 11 — Dimensões e estrutura dos dados
print("Dimensões:", df_consolidado.shape)
print("Colunas:", df_consolidado.columns)
print(df_consolidado.head())

# %% [markdown]
# ### BLOCO 12 — Inspeção dos tipos de dados
#
# `info()` mostra a quantidade de valores não nulos e o tipo de cada coluna; `dtypes` confirma os tipos usados nas análises.

# %%
# BLOCO 12 — Inspeção dos tipos de dados
df_consolidado.info()
print(df_consolidado.dtypes)

# %% [markdown]
# ### BLOCO 13 — Estatística descritiva
#
# `describe()` resume as colunas numéricas: contagem, média, desvio padrão, mínimo, quartis e máximo.

# %%
# BLOCO 13 — Estatística descritiva
print(df_consolidado.describe())

# %% [markdown]
# ### BLOCO 14 — Histogramas das variáveis ambientais
#
# Cinco histogramas com 30 intervalos mostram a forma da distribuição de cada sensor, com as unidades nos eixos.

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

# %% [markdown]
# ### BLOCO 15 — Distribuição da variável alvo
#
# Gráfico de barras com a quantidade e o percentual de cada classe de `Occupancy`, para avaliar o desbalanceamento.

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

# %% [markdown]
# ### BLOCO 16 — Matriz de correlação de Pearson
#
# Mapa de calor da correlação linear entre as variáveis numéricas, incluindo `Occupancy`, na escala de −1 a 1.

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

# %% [markdown]
# ### BLOCO 16.1 — Avaliação exploratória da coluna `date`
#
# A coluna é convertida para `datetime` somente em uma variável temporária, para verificar formato, período, ordem,
# intervalos e unicidade, sem alterar `df_consolidado`.

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
# ### BLOCO 16.2 — Dependência temporal entre observações consecutivas
#
# A base é dividida em segmentos temporais contínuos, separados por interrupções superiores a 2 minutos. Dentro de cada
# segmento, os intervalos regulares são de 59 a 61 segundos; assim, o lag 1 por posição corresponde a aproximadamente
# um minuto. Na base completa, o lag 1 por posição também compara o último registro de `datatest.txt` com o primeiro de
# `datatest2.txt`, separados por mais de sete dias; por isso, as medidas são apresentadas também restritas às transições
# dentro dos segmentos. A análise é descritiva e utiliza `Occupancy` na base completa, sem orientar a escolha de
# atributos ou de hiperparâmetros.

# %%
# BLOCO 16.2 — Dependência temporal entre observações consecutivas

# Interrupções na sequência temporal (intervalos acima de 2 minutos)
limite_continuidade = pd.Timedelta(minutes=2)
interrupcoes = intervalos[intervalos > limite_continuidade]
print("Interrupções acima de 2 minutos:", len(interrupcoes))
print(interrupcoes)

# Segmentos temporais contínuos e transições dentro do mesmo segmento
segmentos = (intervalos > limite_continuidade).cumsum()
transicao_mesmo_segmento = segmentos.eq(segmentos.shift())

resumo_segmentos = pd.DataFrame({
    "Registros": segmentos.value_counts().sort_index(),
    "Início": datas.groupby(segmentos).min(),
    "Fim": datas.groupby(segmentos).max(),
})
print("\nSegmentos temporais contínuos:")
print(resumo_segmentos)

# Autocorrelação de lag 1 das variáveis ambientais
autocorrelacoes = pd.DataFrame({
    "Lag 1 por posição (base completa)": [
        df_consolidado[coluna].autocorr(lag=1) for coluna in colunas_numericas
    ],
    "Lag 1 dentro dos segmentos": [
        df_consolidado[coluna][transicao_mesmo_segmento].corr(
            df_consolidado[coluna].shift(1)[transicao_mesmo_segmento]
        )
        for coluna in colunas_numericas
    ],
}, index=colunas_numericas)
print("\nAutocorrelação de lag 1:")
print(autocorrelacoes.round(4))

# Permanência da classe entre registros consecutivos
ocupacao = df_consolidado["Occupancy"]
mudanca_classe = ocupacao.ne(ocupacao.shift())
transicoes_totais = len(df_consolidado) - 1
transicoes_validas = int(transicao_mesmo_segmento.sum())
mudancas_validas = int((mudanca_classe & transicao_mesmo_segmento).sum())

print("\nTransições entre registros consecutivos (base completa):", transicoes_totais)
print("Transições dentro dos segmentos contínuos:", transicoes_validas)
print("Mudanças de classe na base completa:", int(mudanca_classe.iloc[1:].sum()))
print("Mudanças de classe dentro dos segmentos:", mudancas_validas)
print("Transições dentro dos segmentos com a mesma classe (%):",
      round((transicoes_validas - mudancas_validas) / transicoes_validas * 100, 2))
print("\nMudanças de classe por segmento:")
print((mudanca_classe & transicao_mesmo_segmento).groupby(segmentos).sum())

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
# - `Humidity`: distribuição dispersa entre aproximadamente 22% e 39,5%, com vários agrupamentos; os picos mais evidentes ficam em torno de 25%, entre 30% e 31% e em torno de 36% (intervalos de aproximadamente 0,6 ponto percentual), sem uma concentração única predominante.
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
# **Composição temporal (BLOCO 06.1):**
#
# | Arquivo | Registros | Período (2015) | Ocupados | Na base |
# |---|---:|---|---:|---|
# | `datatest.txt` | 2.665 | 02/02 14:19 a 04/02 10:43 | 972 (36,47%) | Sim |
# | `datatraining.txt` | 8.143 | 04/02 17:51 a 10/02 09:33 | 1.729 (21,23%) | Não |
# | `datatest2.txt` | 9.752 | 11/02 14:48 a 18/02 09:19 | 2.049 (21,01%) | Sim |
#
# A descontinuidade de 7 dias, 4 horas e 5 minutos entre os dois arquivos consolidados contém o período de
# observações de `datatraining.txt`, que não faz parte da base. Esse período não ocupa todo o intervalo: há 7 horas e
# 8 minutos entre o fim de `datatest.txt` e o início de `datatraining.txt`, e 1 dia, 5 horas e 15 minutos entre o fim
# de `datatraining.txt` e o início de `datatest2.txt`. A proporção de ocupados também difere entre os arquivos
# consolidados (36,47% e 21,01%).
#
# **Dependência temporal (BLOCO 16.2):**
#
# | Indicador | Resultado |
# |---|---|
# | Interrupções acima de 2 minutos | 1 (fronteira entre os arquivos) |
# | Segmentos contínuos | 2 (2.665 e 9.752 registros) |
# | Autocorrelação de lag 1 dentro dos segmentos | 0,9952 (`Light`) a 0,9995 (`Temperature`) |
# | Autocorrelação de lag 1 por posição, base completa | 0,9951 a 0,9993 |
# | Transições dentro dos segmentos | 12.415 de 12.416 |
# | Mudanças de classe dentro dos segmentos | 74 (26 e 48) |
# | Transições com a mesma classe | 99,40% |
#
# As medições de cada sensor variam muito pouco de um minuto para o seguinte, e a classe permanece a mesma em 99,40%
# das transições. Registros vizinhos são, portanto, quase redundantes. Como o lag 1 por posição na base completa inclui
# a fronteira de mais de sete dias, a medida restrita aos segmentos é a referência adequada; os dois cálculos produzem
# valores muito próximos porque essa fronteira corresponde a uma única transição.
#
# Em uma divisão aleatória, a maior parte dos registros de teste tende a ter vizinhos temporais no treino. Como
# aproximação simplificada, supondo que cada registro tenha dois vizinhos elegíveis e que cada um seja enviado ao teste
# de forma independente com probabilidade de 20%, a chance de pelo menos um vizinho ficar no treino seria
# 1 − 0,2² = 0,96. Esse valor é uma estimativa teórica, não uma medição; o percentual real será calculado após a divisão
# da Fase 4. Essa proximidade pode produzir estimativas otimistas de desempenho.
#
# Será mantida a divisão estratificada de 80% para treinamento e 20% para teste, conforme o enunciado, registrando-se a
# dependência temporal como limitação da avaliação e complementando-a nas fases seguintes.
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

# Caracterização dos registros envolvidos nas repetições
print("\nRegistros envolvidos por classe de Occupancy:")
print(exemplos["Occupancy"].value_counts().sort_index())
print("Registros envolvidos com Light igual a zero:", (exemplos["Light"] == 0).sum())

# Intervalo entre a primeira e a última ocorrência de cada medição repetida
intervalo_repeticoes = (
    pd.to_datetime(exemplos["date"], format="%Y-%m-%d %H:%M:%S")
    .groupby([exemplos[coluna] for coluna in colunas_numericas])
    .agg(lambda serie: serie.max() - serie.min())
)
print("\nIntervalos nos três grupos dos exemplos acima:")
print(intervalo_repeticoes.loc[exemplos.sort_values(colunas_numericas).head(6)
                               .set_index(colunas_numericas).index.unique()].to_string())
print("\nIntervalo entre ocorrências em todos os", len(intervalo_repeticoes), "grupos repetidos:")
print(intervalo_repeticoes.describe()[["min", "50%", "max"]])

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
# **Critério de escolha:** a média é adequada quando a distribuição não tem cauda longa e a média fica próxima da
# mediana; a mediana é preferível em distribuições assimétricas, por ser resistente a valores extremos. A escolha
# considera em conjunto a assimetria (*skewness*), a diferença entre média e mediana (também expressa em desvios padrão)
# e a forma dos histogramas do BLOCO 14. Nenhum desses indicadores é usado isoladamente como critério absoluto.
#
# Em uma situação real de modelagem, a estatística de imputação deve ser calculada exclusivamente sobre o conjunto de
# treinamento e depois aplicada ao conjunto de teste, evitando Data Leakage.

# %%
# BLOCO 21.2 — Demonstração isolada de imputação (simulação didática)

# Assimetria das variáveis ambientais (fundamenta a escolha entre média e mediana)
assimetria = df_tratado[colunas_numericas].skew()
print("Assimetria (skewness) das variáveis ambientais:")
print(assimetria.round(4))

# Comparação entre média e mediana (sensibilidade a valores extremos)
tabela_tendencia_central = pd.DataFrame({
    "Média": df_tratado[colunas_numericas].mean(),
    "Mediana": df_tratado[colunas_numericas].median(),
})
tabela_tendencia_central["Média − mediana"] = (
    tabela_tendencia_central["Média"] - tabela_tendencia_central["Mediana"]
)
tabela_tendencia_central["Diferença em desvios padrão"] = (
    tabela_tendencia_central["Média − mediana"] / df_tratado[colunas_numericas].std()
)
tabela_tendencia_central["Assimetria"] = assimetria
print("\nMédia, mediana e assimetria:")
print(tabela_tendencia_central.to_string())

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
# instantes diferentes, distribuídas em 461 grupos de medições idênticas que envolvem 1.032 registros.
#
# | Característica dos 1.032 registros envolvidos | Resultado |
# |---|---:|
# | Desocupados (`Occupancy = 0`) | 1.028 |
# | Ocupados (`Occupancy = 1`) | 4 |
# | Com `Light` igual a zero | 987 |
# | Intervalo entre a primeira e a última ocorrência de cada grupo | de 59 segundos a 9 dias e 6 horas (mediana de 6 minutos) |
#
# Os três grupos exibidos como exemplo ocorreram na manhã de 14/02/2015, com o ambiente desocupado, com intervalos de
# 8 minutos e 59 segundos, 13 minutos e 20 minutos e 1 segundo entre as leituras iguais; esses valores descrevem apenas
# os exemplos e não o conjunto das repetições. Em nenhum caso a mesma combinação de medições aparece associada a classes
# diferentes.
#
# As repetições concentram-se em registros desocupados e sem iluminação, situação em que as condições do ambiente
# variam pouco. Isso é compatível com um ambiente estável e com a resolução de registro dos sensores, mas não comprova,
# por si só, o funcionamento correto dos equipamentos. Como cada registro corresponde a um instante distinto, as
# repetições foram mantidas: removê-las reduziria a amostra e alteraria a frequência real dessas condições.
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
# | Variável | Média | Mediana | Média − mediana | Diferença em desvios padrão | Assimetria |
# |---|---:|---:|---:|---:|---:|
# | Temperature | 21,094510 | 20,790000 | 0,304510 | 0,2935 | 1,1041 |
# | Humidity | 28,917947 | 28,600000 | 0,317947 | 0,0771 | 0,3008 |
# | Light | 138,125947 | 0,000000 | 138,125947 | 0,6284 | 1,3173 |
# | CO2 | 745,644624 | 632,000000 | 113,644624 | 0,3833 | 1,3851 |
# | HumidityRatio | 0,0044682086 | 0,0045113541 | −0,0000431455 | −0,0725 | −0,1458 |
#
# **Humidity (média):** a média e a mediana diferem em apenas 0,08 desvio padrão, a assimetria é baixa (0,3008) e o
# histograma não apresenta cauda longa nem valores isolados. A distribuição, porém, não é normal nem perfeitamente
# simétrica: apresenta vários agrupamentos (em torno de 25%, entre 30% e 31% e em torno de 36%). Nessas condições, a
# média e a mediana produzem valores próximos (28,92% e 28,60%), e a média foi adotada para demonstrar didaticamente
# essa estratégia.
#
# **CO2 (mediana):** a média supera a mediana em 113,64 ppm (0,38 desvio padrão), a assimetria é a maior entre as
# variáveis (1,3851) e o histograma mostra cauda longa à direita, com 992 potenciais outliers acima de 1.351,25 ppm
# (BLOCO 25). A média é deslocada por esses valores altos; a mediana, resistente a eles, representa melhor o valor
# típico.
#
# `HumidityRatio` tem a menor assimetria absoluta (−0,1458), mas é uma grandeza derivada da temperatura e da umidade
# relativa; `Humidity` foi escolhida por ser uma medição direta do sensor. `Temperature`, considerada inicialmente para
# a média, foi descartada como exemplo por sua assimetria de 1,1041. `Light` não foi usada na simulação, pois 61,30% dos
# seus valores são zero e a mediana coincide com o mínimo.
#
# Os erros absolutos médios medem a distância entre o valor imputado e os valores reais ocultados nesta simulação
# específica; eles não demonstram que a estratégia escolhida seja a melhor em qualquer cenário. A simulação utilizou
# semente fixa (42) e não alterou `df_tratado` (0 ausentes após a demonstração). Em um cenário real, a estatística de
# imputação deve ser calculada apenas sobre o conjunto de treinamento.
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
# **Decisão:** nenhuma observação foi removida, truncada ou substituída. A decisão se fundamenta em quatro pontos:
#
# 1. Nenhum valor violou os critérios de plausibilidade física aplicados no BLOCO 25.2.
# 2. Não há evidência suficiente de erro de medição: em `Temperature` e `CO2`, os valores sinalizados formam caudas
#    contínuas das distribuições; em `Light`, os 7 valores acima de 980 lux são poucos e afastados, mas compatíveis com
#    luz natural intensa.
# 3. Os valores extremos podem representar condições ambientais legítimas, como luz natural intensa, maior número de
#    pessoas ou ventilação reduzida.
# 4. Remover essas observações eliminaria variações reais do ambiente que o modelo precisa representar.
#
# A plausibilidade física indica apenas que os valores são possíveis; ela não comprova a exatidão dos sensores.
#
# Como evidência descritiva secundária, os registros sinalizados apresentam proporções de ocupação superiores à da
# base (44,69% em `Temperature`, 100% nos 7 casos de `Light` e 30,95% em `CO2`, contra 24,33%). Essa associação foi
# observada na base completa, antes da divisão treino/teste, e não foi usada como fundamento da decisão. Os limites IQR
# foram calculados com finalidade exploratória; caso venham a ser usados em alguma transformação do modelo, deverão ser
# ajustados exclusivamente sobre o conjunto de treinamento.
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


# %% [markdown]
# # FASE 3 — Engenharia de Atributos (Feature Engineering)
# ## BLOCO 29 — Contextualização da engenharia de atributos
#
# ### Conceito
#
# Feature Engineering é a criação ou transformação de atributos a partir das variáveis existentes, com o objetivo de
# representar os dados de uma forma que facilite o aprendizado dos modelos. No material didático MUS01, a Feature
# Engineering aparece como uma etapa do "Pipeline básico de Machine Learning" (slide 49: criar novas variáveis a partir
# das existentes), posterior à limpeza e ao pré-processamento (slide 48) e anterior à divisão do dataset em treino e
# teste (slide 50).
#
# ### Objetivo no projeto
#
# O enunciado exige a criação de pelo menos uma nova coluna numérica, obtida por operação matemática entre colunas
# existentes, com verificação de valores ausentes antes do cálculo e validação da execução. Nesta fase é criada uma
# única variável derivada, que será avaliada posteriormente pelos modelos KNN e Árvore de Decisão.
#
# ### Variável selecionada
#
# $$\text{CO2\_Light\_Ratio} = \frac{\text{CO2}}{\text{Light} + 1}$$
#
# | Critério | Justificativa |
# |---|---|
# | Requisito do enunciado | É uma das combinações sugeridas no documento oficial do projeto |
# | Sensores envolvidos | Combina a concentração de CO2 (ppm) com a luminosidade (lux) em um único índice |
# | Constante `+1` | Evita divisão por zero nas 7.612 leituras de `Light` iguais a zero |
# | Ausência do alvo | `Occupancy` não participa do cálculo |
# | Cálculo por registro | Cada valor depende apenas do próprio registro, sem estatísticas globais da base |
#
# A variável é tratada como um **índice experimental**: não possui significado físico validado e não há garantia
# antecipada de ganho preditivo.
#
# ### Nova representação, não nova informação
#
# A nova coluna é uma função determinística de `CO2` e `Light`, que continuam disponíveis na base. Ela não acrescenta
# medições independentes; apenas reorganiza a informação existente em uma forma que pode, ou não, ser mais útil para
# os classificadores. Por isso, a fase investiga também a possível redundância com os sensores originais.
#
# ### Limitações conceituais
#
# - A constante `+1` é arbitrária e influencia principalmente as leituras de baixa luminosidade.
# - A razão é muito sensível a variações de `Light` próximas de zero.
# - Valores elevados do índice não comprovam ocupação; associação não implica causalidade.
# - A comparação entre as candidatas foi realizada na base completa antes da divisão treino/teste, sem uso do alvo.

# %% [markdown]
# ### BLOCO 30 — Criação da base de Feature Engineering
#
# **Justificativa técnica:** a cópia independente (`copy(deep=True)`) mantém `df_tratado` e `df_consolidado`
# inalterados, preservando as bases das fases anteriores como referência para a validação da integridade.

# %%
# BLOCO 30 — Criação da base de Feature Engineering

# Cópia independente da base tratada
df_features = df_tratado.copy(deep=True)

print("Dimensões iniciais:", df_features.shape)
print("Registros:", len(df_features))
print("Colunas:", df_features.shape[1])
print(df_features.dtypes)

# Igualdade inicial dos valores
pd.testing.assert_frame_equal(df_features, df_tratado, check_exact=True)
print("\nBase de Feature Engineering idêntica à base tratada:", df_features.equals(df_tratado))

# Independência dos objetos (não compartilham a mesma memória)
print("Objetos distintos:", df_features is not df_tratado)
print("Memória compartilhada em CO2:",
      np.shares_memory(df_features["CO2"].to_numpy(), df_tratado["CO2"].to_numpy()))

# %% [markdown]
# ### BLOCO 31 — Criação da variável `CO2_Light_Ratio`
#
# As verificações com `assert` são executadas **antes** do cálculo e interrompem a execução caso as colunas da fórmula
# apresentem alguma inconsistência.
#
# **Comportamento da fórmula** (contínua para `Light ≥ 0`):
#
# | Situação | Denominador | Efeito no índice |
# |---|---|---|
# | `Light = 0` | 1 | O índice é exatamente igual ao valor de `CO2` |
# | `Light` próxima de zero (0 a poucos lux) | Entre 1 e alguns poucos | Queda acentuada: com `Light = 1`, o índice já corresponde à metade de `CO2` |
# | `Light` elevada (centenas de lux) | Aproximadamente `Light` | Índice pequeno, aproximadamente `CO2 / Light` |
#
# A taxa de variação em relação à luminosidade é −CO2 / (Light + 1)², de maior intensidade em `Light = 0` e cada vez
# menor à medida que a luminosidade aumenta. Não há descontinuidade: a função é definida e contínua em todo o domínio
# observado.

# %%
# BLOCO 31 — Criação da variável CO2_Light_Ratio

colunas_formula = ["CO2", "Light"]

# Verificações anteriores ao cálculo
verificacoes_formula = pd.DataFrame({
    "Ausentes": df_features[colunas_formula].isna().sum(),
    "Infinitos": np.isinf(df_features[colunas_formula]).sum(),
    "Negativos": (df_features[colunas_formula] < 0).sum(),
    "Tipo": df_features[colunas_formula].dtypes.astype(str),
})
print("Verificações das colunas da fórmula:")
print(verificacoes_formula)

denominador = df_features["Light"] + 1
print("\nLeituras de Light iguais a zero:", (df_features["Light"] == 0).sum())
print("Denominadores iguais a zero:", (denominador == 0).sum())
print("Menor denominador:", denominador.min())

# Condições bloqueantes
assert set(colunas_formula).issubset(df_features.columns), "CO2 ou Light ausente na base."
assert all(pd.api.types.is_float_dtype(df_features[coluna]) for coluna in colunas_formula), "Tipo não numérico."
assert df_features[colunas_formula].isna().sum().sum() == 0, "Existem valores ausentes em CO2 ou Light."
assert np.isinf(df_features[colunas_formula]).sum().sum() == 0, "Existem valores infinitos em CO2 ou Light."
assert (df_features["Light"] >= 0).all(), "Existem leituras negativas de Light."
assert (denominador != 0).all(), "Existem denominadores iguais a zero."
print("\nVerificações anteriores ao cálculo aprovadas.")

# Criação da nova variável por operação vetorizada
df_features["CO2_Light_Ratio"] = df_features["CO2"] / denominador

# Confirmações após o cálculo
colunas_originais = list(df_tratado.columns)
print("\nTipo da nova variável:", df_features["CO2_Light_Ratio"].dtype)
print("Dimensões após a criação:", df_features.shape)
print("Ausentes na nova variável:", df_features["CO2_Light_Ratio"].isna().sum())
print("Infinitos na nova variável:", np.isinf(df_features["CO2_Light_Ratio"]).sum())
print("Colunas originais inalteradas:", df_features[colunas_originais].equals(df_tratado))
print("Primeiro valor armazenado:", repr(float(df_features.loc[0, "CO2_Light_Ratio"])))
print(df_features[["CO2", "Light", "CO2_Light_Ratio"]].head())

# Demonstração ilustrativa do efeito da luminosidade (CO2 hipotético de 600 ppm)
luminosidades_exemplo = pd.Series([0, 0.5, 1, 5, 50, 500], name="Light")
print("\nIlustração com CO2 = 600 ppm (valores hipotéticos, fora da base):")
print(pd.DataFrame({"Light": luminosidades_exemplo, "CO2_Light_Ratio": 600 / (luminosidades_exemplo + 1)}))

# %% [markdown]
# ### BLOCO 32 — Inspeção estatística da nova variável
#
# A inspeção apresenta as estatísticas descritivas da nova variável, compara os registros com luminosidade igual a
# zero e positiva e mede a associação com as variáveis de origem. **`Occupancy` não é utilizada nesta comparação.**

# %%
# BLOCO 32.1 — Estatísticas descritivas da nova variável

nova_variavel = df_features["CO2_Light_Ratio"]

resumo_nova_variavel = pd.Series({
    "Tipo": str(nova_variavel.dtype),
    "Observações": nova_variavel.count(),
    "Média": nova_variavel.mean(),
    "Desvio padrão": nova_variavel.std(),
    "Mínimo": nova_variavel.min(),
    "Primeiro quartil (Q1)": nova_variavel.quantile(0.25),
    "Mediana": nova_variavel.median(),
    "Terceiro quartil (Q3)": nova_variavel.quantile(0.75),
    "Máximo": nova_variavel.max(),
    "Amplitude": nova_variavel.max() - nova_variavel.min(),
    "Assimetria": nova_variavel.skew(),
    "Zeros": (nova_variavel == 0).sum(),
    "Valores distintos": nova_variavel.nunique(),
    "Ausentes": nova_variavel.isna().sum(),
    "Infinitos": np.isinf(nova_variavel).sum(),
})
print(resumo_nova_variavel.to_string())

# %%
# BLOCO 32.2 — Comparação entre luminosidade igual a zero e positiva

mascara_light_zero = df_features["Light"] == 0
mascara_light_positiva = df_features["Light"] > 0
assert mascara_light_zero.sum() + mascara_light_positiva.sum() == len(df_features), "Grupos incompletos."

tabela_grupos_light = pd.DataFrame({
    "Light = 0": nova_variavel[mascara_light_zero].describe(),
    "Light > 0": nova_variavel[mascara_light_positiva].describe(),
}).T
tabela_grupos_light.insert(1, "Percentual (%)", [
    round(mascara_light_zero.mean() * 100, 2),
    round(mascara_light_positiva.mean() * 100, 2),
])
print(tabela_grupos_light.to_string())

# Comparação exata: com Light = 0, o denominador é 1 e o índice deve ser igual ao CO2
pd.testing.assert_series_equal(
    nova_variavel[mascara_light_zero],
    df_features.loc[mascara_light_zero, "CO2"],
    check_exact=True,
    check_names=False,
)
print("\nCom Light = 0, CO2_Light_Ratio é exatamente igual a CO2:",
      (nova_variavel[mascara_light_zero] == df_features.loc[mascara_light_zero, "CO2"]).all())

# %%
# BLOCO 32.3 — Correlação com as variáveis de origem (sem Occupancy)

tabela_correlacoes = pd.DataFrame({
    "Pearson com CO2_Light_Ratio": [
        nova_variavel.corr(df_features[coluna], method="pearson") for coluna in colunas_formula
    ],
    "Spearman com CO2_Light_Ratio": [
        nova_variavel.corr(df_features[coluna], method="spearman") for coluna in colunas_formula
    ],
}, index=pd.Index(colunas_formula, name="Variável original"))
print(tabela_correlacoes.round(4))

# %% [markdown]
# ### BLOCO 33 — Visualização da nova variável
#
# Os dois histogramas utilizam 50 intervalos. No gráfico logarítmico, os 51 limites são espaçados geometricamente entre
# o mínimo e o máximo exatos da variável. A escala logarítmica é aplicada **somente à visualização**: a coluna
# `CO2_Light_Ratio` não é transformada.

# %%
# BLOCO 33 — Histogramas da nova variável

valor_minimo = nova_variavel.min()
valor_maximo = nova_variavel.max()

# A escala logarítmica exige valores estritamente positivos
assert (nova_variavel > 0).all(), "Existem valores não positivos; a escala logarítmica não é aplicável."

# Limites dos intervalos (50 intervalos = 51 limites)
limites_lineares = np.linspace(valor_minimo, valor_maximo, 51)
limites_log = np.logspace(np.log10(valor_minimo), np.log10(valor_maximo), 51)
limites_log[0] = valor_minimo
limites_log[-1] = valor_maximo

fig, (eixo_linear, eixo_log) = plt.subplots(1, 2, figsize=(15, 5))

frequencias_lineares, _, _ = eixo_linear.hist(nova_variavel, bins=limites_lineares, edgecolor="black")
eixo_linear.set_title("Escala linear")
eixo_linear.set_xlabel("Índice experimental CO2/Luz — CO2 / (Light + 1)")
eixo_linear.set_ylabel("Frequência")

frequencias_log, _, _ = eixo_log.hist(nova_variavel, bins=limites_log, edgecolor="black")
eixo_log.set_xscale("log")
eixo_log.set_title("Escala logarítmica no eixo horizontal (somente visualização)")
eixo_log.set_xlabel("Índice experimental CO2/Luz — CO2 / (Light + 1)")
eixo_log.set_ylabel("Frequência")

plt.suptitle("Distribuição da nova variável CO2_Light_Ratio", fontsize=16)
plt.tight_layout()

# Exportação antes da exibição
caminho_figura = f"{pasta_graficos}/fe_01_distribuicao_nova_variavel.png"
plt.savefig(caminho_figura, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_figura)

plt.show()

# Conferência dos limites e das frequências
assert len(limites_lineares) == 51 and len(limites_log) == 51, "Quantidade de limites incorreta."
assert limites_lineares[0] == valor_minimo and limites_lineares[-1] == valor_maximo, "Extremos lineares incorretos."
assert limites_log[0] == valor_minimo and limites_log[-1] == valor_maximo, "Extremos logarítmicos incorretos."
assert frequencias_lineares.sum() == len(df_features), "O histograma linear não representa todos os registros."
assert frequencias_log.sum() == len(df_features), "O histograma logarítmico não representa todos os registros."

tabela_histogramas = pd.DataFrame({
    "Escala linear": [
        len(limites_lineares) - 1,
        len(limites_lineares),
        int(frequencias_lineares.sum()),
        int((frequencias_lineares == 0).sum()),
        int(frequencias_lineares.max()),
    ],
    "Escala logarítmica": [
        len(limites_log) - 1,
        len(limites_log),
        int(frequencias_log.sum()),
        int((frequencias_log == 0).sum()),
        int(frequencias_log.max()),
    ],
}, index=["Intervalos", "Limites", "Soma das frequências", "Intervalos vazios", "Maior frequência"])
print(tabela_histogramas)

# Frequências por intervalo da escala logarítmica
tabela_frequencias_log = pd.DataFrame({
    "Limite inferior": limites_log[:-1],
    "Limite superior": limites_log[1:],
    "Frequência": frequencias_log.astype(int),
})
print("\nFrequências por intervalo (escala logarítmica):")
print(tabela_frequencias_log.to_string())

# %% [markdown]
# ### BLOCO 34 — Validação da integridade da base de Feature Engineering
#
# As verificações com `assert` impedem a exportação caso a nova base apresente alguma inconsistência em relação à base
# tratada. A coluna derivada é recalculada de forma independente e comparada com `check_exact=True`.

# %%
# BLOCO 34 — Validação da integridade

colunas_numericas_features = colunas_numericas + ["CO2_Light_Ratio"]

tabela_integridade = pd.DataFrame({
    "Base tratada": [
        len(df_tratado),
        df_tratado.shape[1],
        df_tratado.isna().sum().sum(),
        np.isinf(df_tratado[colunas_numericas]).sum().sum(),
        df_tratado["date"].nunique(),
        (df_tratado["Occupancy"] == 0).sum(),
        (df_tratado["Occupancy"] == 1).sum(),
    ],
    "Base de Feature Engineering": [
        len(df_features),
        df_features.shape[1],
        df_features.isna().sum().sum(),
        np.isinf(df_features[colunas_numericas_features]).sum().sum(),
        df_features["date"].nunique(),
        (df_features["Occupancy"] == 0).sum(),
        (df_features["Occupancy"] == 1).sum(),
    ],
}, index=["Registros", "Colunas", "Valores ausentes", "Valores infinitos",
          "Timestamps distintos", "Classe 0", "Classe 1"])
tabela_integridade["Diferença"] = (
    tabela_integridade["Base de Feature Engineering"] - tabela_integridade["Base tratada"]
)
print(tabela_integridade)

# Dimensões e colunas
assert len(df_tratado) == 12417 and len(df_features) == 12417, "Quantidade de registros divergente."
assert df_tratado.shape[1] == 7, "A base tratada não possui sete colunas."
assert df_features.shape[1] == 8, "A base de Feature Engineering não possui oito colunas."
colunas_acrescentadas = [coluna for coluna in df_features.columns if coluna not in df_tratado.columns]
assert colunas_acrescentadas == ["CO2_Light_Ratio"], "Colunas acrescentadas divergentes."
assert list(df_features.columns[:7]) == colunas_originais, "Ordem das colunas originais alterada."

# Preservação integral das sete colunas originais, dos timestamps e do alvo
pd.testing.assert_frame_equal(df_features[colunas_originais], df_tratado, check_exact=True)
pd.testing.assert_series_equal(df_features["date"], df_tratado["date"], check_exact=True)
pd.testing.assert_series_equal(df_features["Occupancy"], df_tratado["Occupancy"], check_exact=True)
assert (df_features["Occupancy"] == 0).sum() == 9396, "Quantidade de desocupados divergente."
assert (df_features["Occupancy"] == 1).sum() == 3021, "Quantidade de ocupados divergente."

# Ausentes, infinitos e tipo da variável derivada
assert df_features.isna().sum().sum() == 0, "Existem valores ausentes."
assert np.isinf(df_features[colunas_numericas_features]).sum().sum() == 0, "Existem valores infinitos."
assert df_features["CO2_Light_Ratio"].dtype == "float64", "A variável derivada não é float64."

# Igualdade exata com o recálculo independente da fórmula
recalculo_formula = df_tratado["CO2"] / (df_tratado["Light"] + 1)
pd.testing.assert_series_equal(df_features["CO2_Light_Ratio"], recalculo_formula, check_exact=True, check_names=False)

# Preservação da precisão numérica em relação à base consolidada
pd.testing.assert_frame_equal(
    df_features[colunas_numericas], df_consolidado[colunas_numericas], check_exact=True
)

# Preservação das bases anteriores
pd.testing.assert_frame_equal(df_tratado, df_tratado_recarregado, check_exact=True)
pd.testing.assert_frame_equal(df_consolidado, pd.read_csv(caminho_csv), check_exact=True)
assert df_tratado.shape == (12417, 7) and df_consolidado.shape == (12417, 7), "Bases anteriores alteradas."

print("\nValidação da integridade concluída com sucesso!")

# %% [markdown]
# ### BLOCO 35 — Exportação e validação da base de Feature Engineering
#
# A exportação segue o padrão da Fase 2: `index=False`, coluna `date` no formato `%Y-%m-%d %H:%M:%S` e nenhuma
# formatação das casas decimais. Após a recarga, `date` é convertida novamente para `datetime` e a equivalência é
# verificada com `check_exact=True`.
#
# **Precisão na leitura:** a divisão produz valores com até 17 algarismos significativos. O pandas grava cada valor com
# a representação textual mais curta que reconstrói exatamente o `float64`, mas o conversor padrão de `read_csv`
# (`float_precision="high"`) pode diferir do valor original no último bit. Por isso, a recarga utiliza
# `float_precision="round_trip"`, que reconstrói exatamente os valores gravados. Nenhum arredondamento é aplicado e a
# comparação permanece exata.

# %%
# BLOCO 35 — Exportação e validação da base de Feature Engineering

df_features.to_csv(caminho_features, index=False, date_format="%Y-%m-%d %H:%M:%S")
print("Base de Feature Engineering exportada:", caminho_features)

# Diagnóstico: divergências da nova variável com o conversor padrão de ponto flutuante
recarga_padrao = pd.read_csv(caminho_features)
divergencias_padrao = recarga_padrao["CO2_Light_Ratio"] != df_features["CO2_Light_Ratio"]
diferenca_relativa_padrao = (
    (recarga_padrao["CO2_Light_Ratio"] - df_features["CO2_Light_Ratio"]).abs() / df_features["CO2_Light_Ratio"]
).max()
print("Valores divergentes com o conversor padrão:", divergencias_padrao.sum())
print("Maior diferença relativa com o conversor padrão:", diferenca_relativa_padrao)

# Recarga com reconstrução exata dos valores e conversão da coluna temporal
df_features_recarregado = pd.read_csv(caminho_features, float_precision="round_trip")
df_features_recarregado["date"] = pd.to_datetime(
    df_features_recarregado["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)

print("Dimensões:", df_features_recarregado.shape)
print(df_features_recarregado.dtypes)
print("Valores ausentes:", df_features_recarregado.isna().sum().sum())
print(df_features_recarregado["Occupancy"].value_counts().sort_index())

# Equivalência exata entre a base em memória e a base recarregada
pd.testing.assert_frame_equal(df_features, df_features_recarregado, check_exact=True)
print("\nBase recarregada idêntica à base de Feature Engineering.")

# Tamanho e SHA-256 do arquivo exportado
with open(caminho_features, "rb") as arquivo:
    hash_features = hashlib.sha256(arquivo.read()).hexdigest()

print("Tamanho do arquivo (bytes):", os.path.getsize(caminho_features))
print("SHA-256:", hash_features)

# %% [markdown]
# ## BLOCO 36 — Interpretação e conclusão da Feature Engineering
#
# ### 36.1 — Justificativa da variável escolhida
#
# Foi criada a variável `CO2_Light_Ratio = CO2 / (Light + 1)`, uma das combinações sugeridas no enunciado. Ela reúne,
# em um único índice, dois sensores que respondem à presença de pessoas por mecanismos diferentes: a concentração de
# CO2 acumula-se com a respiração dos ocupantes, enquanto a luminosidade depende da iluminação artificial e da luz
# natural. A razão entre eles produz uma representação em que leituras de CO2 com ambiente escuro resultam em valores
# altos e leituras com ambiente iluminado resultam em valores baixos. A construção não utiliza `Occupancy`, é calculada
# registro a registro e não depende de estatísticas globais. A escolha não pressupõe superioridade preditiva em
# relação às demais candidatas investigadas.
#
# ### 36.2 — Comportamento matemático
#
# A constante `+1` torna o denominador estritamente positivo (menor denominador observado: 1,0), eliminando a divisão
# por zero sem excluir registros. Quando `Light = 0`, o índice é exatamente igual a `CO2`; à medida que a luminosidade
# aumenta, o índice diminui de forma contínua, com variação mais intensa nos primeiros lux. A ilustração do BLOCO 31,
# com CO2 hipotético de 600 ppm, mostra essa sensibilidade: o índice passa de 600 (0 lux) para 300 (1 lux), 100 (5
# lux) e cerca de 1,20 (500 lux).
#
# Na base real, porém, não há leituras de `Light` entre 0 e 4,67 lux: a menor luminosidade positiva é 4,67 lux.
# Assim, a faixa de maior sensibilidade da fórmula não é ocupada por observações reais, e o salto do índice ocorre
# entre o grupo `Light = 0` e as leituras positivas. Para `Light > 0`, o denominador é de pelo menos 5,67, e a
# constante `+1` altera o índice, no máximo, em cerca de 18% em relação a `CO2 / Light`, influência que se reduz nas
# luminosidades elevadas. A constante permanece arbitrária: outro valor modificaria a escala do índice nas leituras de
# baixa luminosidade.
#
# ### 36.3 — Estatísticas descritivas e distribuição
#
# | Estatística | Valor |
# |---|---:|
# | Tipo | `float64` |
# | Observações | 12.417 |
# | Média | 426,134753 |
# | Desvio padrão | 403,140840 |
# | Mínimo | 0,560430 |
# | Primeiro quartil (Q1) | 2,999126 |
# | Mediana | 503,000000 |
# | Terceiro quartil (Q3) | 596,333333 |
# | Máximo | 2.076,500000 |
# | Amplitude | 2.075,939570 |
# | Assimetria | 0,822262 |
# | Zeros / distintos / ausentes / infinitos | 0 / 6.989 / 0 / 0 |
#
# Evidência visual: `resultados/graficos/fe_01_distribuicao_nova_variavel.png`.
#
# A distribuição não é contínua ao longo da amplitude: os valores formam dois agrupamentos separados. Os registros com
# luz ocupam a faixa de 0,56 a 263,29, e os registros sem luz ocupam a faixa de 427,50 a 2.076,50. Não há nenhuma
# observação entre 263,29 e 427,50. Por isso, a média (426,13) cai justamente nessa faixa vazia e não representa um
# valor típico. O salto entre Q1 (3,00) e a mediana (503,00) indica a mesma separação.
#
# - **Escala linear:** os 50 intervalos somam 12.417 registros; 8 intervalos ficaram vazios. O primeiro intervalo (0,56
#   a 42,08) concentra 4.558 registros, comprimindo quase todo o grupo com luz em uma única barra. O grupo sem luz
#   reproduz a forma da distribuição de `CO2`, com maior concentração entre aproximadamente 500 e 600.
# - **Escala logarítmica (somente visualização):** os 50 intervalos também somam 12.417 registros; 3 intervalos
#   ficaram vazios, todos entre 208,00 e 401,39. O agrupamento com luz fica visível, com maior concentração entre 1,77
#   e 2,46 (1.430 registros em dois intervalos). O intervalo de maior frequência (473,09 a 557,60, com 2.527 registros)
#   pertence ao grupo sem luz.
#
# A existência de dois agrupamentos decorre da estrutura da fórmula combinada com o excesso de zeros em `Light`, e
# não de uma propriedade física independente. A assimetria positiva (0,8223) resume uma distribuição heterogênea e
# não deve ser lida isoladamente.
#
# ### 36.4 — Influência das leituras de luminosidade igual a zero
#
# | Grupo | Registros | Percentual | Média | Mediana | Mínimo | Máximo |
# |---|---:|---:|---:|---:|---:|---:|
# | `Light = 0` | 7.612 | 61,30% | 689,649896 | 569,333333 | 427,500000 | 2.076,500000 |
# | `Light > 0` | 4.805 | 38,70% | 8,678507 | 2,299843 | 0,560430 | 263,294118 |
#
# Nos 7.612 registros com `Light = 0`, a comparação exata confirmou que `CO2_Light_Ratio` é igual a `CO2` (`True`).
# Nesses casos, a nova variável não acrescenta nenhuma diferença em relação ao sensor original: ela repete o valor
# de CO2. A transformação só modifica efetivamente os 4.805 registros com luminosidade positiva, cujo índice é baixo
# (mediana de 2,30; 75% abaixo de 4,55). Como o menor valor do grupo sem luz (427,50) é maior que o maior valor do
# grupo com luz (263,29), o índice separa completamente os dois grupos de luminosidade. Essa separação, porém, já está
# disponível na própria variável `Light`.
#
# ### 36.5 — Correlação e possível redundância
#
# | Variável original | Pearson com `CO2_Light_Ratio` | Spearman com `CO2_Light_Ratio` |
# |---|---:|---:|
# | CO2 | 0,2419 | 0,0686 |
# | Light | −0,6607 | −0,8633 |
#
# Pearson mede associação linear; Spearman mede associação monotônica, calculada sobre as posições (postos) dos
# valores. A associação com `Light` é negativa e forte em Spearman (−0,8633), maior em módulo que em Pearson
# (−0,6607): quanto maior a luminosidade, menor tende a ser o índice, mas essa relação não é linear, coerente com a
# forma da divisão. Com `CO2`, as duas correlações globais são fracas (0,2419 e 0,0686). Esse resultado não significa
# independência, pois o índice é exatamente igual a CO2 em 61,30% dos registros; a associação global é reduzida
# porque, nos registros com luz, o índice tende a acompanhar mais a variação de `Light` do que a de `CO2`.
#
# Há, portanto, redundância parcial: o índice reproduz `CO2` no grupo sem luz e acompanha `Light` de forma monotônica
# inversa no conjunto da base. Nenhum dos coeficientes é próximo de ±1, de modo que a variável não é uma cópia de um
# único sensor. As correlações foram calculadas sem `Occupancy` e não indicam causalidade nem importância preditiva;
# o efeito sobre a classificação só poderá ser avaliado nos modelos.
#
# ### 36.6 — Implicações para KNN e Árvore de Decisão
#
# - **Escala no KNN:** o KNN classifica pela distância entre registros. O índice tem desvio padrão de 403,14, enquanto
#   `Temperature` tem 1,04 e `HumidityRatio` tem 0,0006 (BLOCO 13). Sem padronização, o índice, `CO2` e `Light`
#   dominariam a distância euclidiana, e as demais variáveis teriam influência desprezível.
# - **StandardScaler:** na configuração prevista para o KNN, o `StandardScaler` será ajustado somente no conjunto de
#   treinamento (`fit_transform`) e aplicado ao teste (`transform`), colocando todos os atributos na mesma escala.
# - **Redundância no KNN:** como o índice repete parte da informação de `CO2` e `Light`, incluí-lo junto aos sensores
#   originais aumenta o peso dessa informação no cálculo da distância. Isso pode ajudar ou prejudicar a classificação.
# - **Árvore de Decisão:** a árvore divide os dados por limiares em um atributo de cada vez, e os limiares não dependem
#   da escala. Por isso, o escalonamento não é necessário. Qualquer limiar entre 263,29 e 427,50 no índice separa os
#   mesmos registros que um limiar em `Light` entre 0 e 4,67 lux, o que mostra que, nesse ponto, a árvore não ganha
#   informação nova.
# - **Sem garantia de melhoria:** a contribuição da variável será verificada apenas pelo desempenho dos modelos nas
#   fases seguintes.
#
# ### 36.7 — Prevenção de Data Leakage
#
# | Tipo de operação | Aprende parâmetros dos dados? | Momento adequado |
# |---|---|---|
# | Transformação determinística por registro (`CO2 / (Light + 1)`) | Não | Pode ser aplicada antes da divisão |
# | Transformações com parâmetros (padronização, imputação, limites IQR) | Sim | Ajuste somente no treino |
# | Seleção de atributos pelo desempenho | Sim, indiretamente | Somente com dados de treino ou validação |
# | Ajuste de hiperparâmetros | Sim, indiretamente | Somente com dados de treino ou validação |
#
# A nova variável é calculada com os valores do próprio registro, sem média, desvio padrão ou qualquer informação de
# outros registros, e sem `Occupancy`. Por isso, criá-la antes da divisão treino/teste não transfere informações do
# teste para o treino.
#
# Há, contudo, limitações metodológicas a registrar. As três candidatas avaliadas foram comparadas na base completa
# antes da divisão, ainda que sem o alvo. A EDA também analisou as associações entre os sensores e `Occupancy` na base
# completa. Essas análises orientaram decisões do projeto com dados que, depois, farão parte do conjunto de teste.
# Além disso, o split aleatório exigido aproxima registros temporalmente vizinhos entre treino e teste: no BLOCO 16.2,
# a classe permanece a mesma em 99,40% das transições entre registros consecutivos. Por esses motivos, não é possível
# afirmar que o processo esteja completamente livre de vieses de avaliação.
#
# Quanto à composição da base, a consolidação dos dois arquivos `datatest` é uma escolha deste projeto, distinta do
# protocolo original de avaliação do conjunto de dados (BLOCO 05). O arquivo `datatraining.txt` não participou da EDA,
# do tratamento, da Feature Engineering nem da escolha da variável; apenas seus períodos e proporções de classe foram
# lidos no BLOCO 06.1. Ele poderá servir a uma avaliação complementar ao final, entendida como avaliação em outro
# período do mesmo conjunto de dados e do mesmo ambiente, e não como validação externa em ambiente independente.
#
# ### 36.8 — Integridade, exportação e limitações
#
# | Indicador | Base tratada | Base de Feature Engineering | Diferença |
# |---|---:|---:|---:|
# | Registros | 12.417 | 12.417 | 0 |
# | Colunas | 7 | 8 | 1 |
# | Valores ausentes | 0 | 0 | 0 |
# | Valores infinitos | 0 | 0 | 0 |
# | Timestamps distintos | 12.417 | 12.417 | 0 |
# | Classe 0 | 9.396 | 9.396 | 0 |
# | Classe 1 | 3.021 | 3.021 | 0 |
#
# Todas as verificações `assert` foram aprovadas. Foi acrescentada somente a coluna `CO2_Light_Ratio` (`float64`). As
# sete colunas originais, os timestamps e `Occupancy` permaneceram idênticos aos da base tratada (`check_exact=True`).
# A coluna derivada coincide exatamente com o recálculo independente da fórmula, e `df_tratado` e `df_consolidado`
# foram preservados.
#
# A base foi exportada para `dados/features/occupancy_features.csv` (990.635 bytes). Na recarga com o conversor padrão
# do pandas, 850 valores da nova variável divergiram no último bit (diferença relativa máxima de 2,43 × 10⁻¹⁶), embora
# o texto gravado no arquivo represente exatamente cada valor. A leitura com `float_precision="round_trip"` reconstruiu
# todos os valores, e a comparação com `check_exact=True` foi aprovada sem arredondamento e sem tolerância. Nenhum
# registro foi excluído.
#
# **Limitações:** a constante `+1` é arbitrária; o índice é muito sensível à luminosidade próxima de zero, embora essa
# faixa não esteja ocupada na base atual; há redundância parcial com `CO2` e `Light`; o índice não tem interpretação
# física validada; e o ganho preditivo ainda não foi comprovado. A divisão treino/teste, o escalonamento e o
# treinamento dos modelos serão realizados nas fases seguintes.

# %% [markdown]
# # FASE 4 — Train/Test Split (Divisão dos Dados)
# ## BLOCO 37 — Contextualização da divisão dos dados
#
# ### Aprendizado supervisionado
#
# No aprendizado supervisionado, o modelo aprende a partir de exemplos rotulados: cada registro reúne os **atributos
# preditores** (`X`), que descrevem a situação observada pelos sensores, e a **variável alvo** (`y`), que é a resposta
# conhecida — aqui, `Occupancy`. O modelo procura uma regra que relacione `X` a `y` e que funcione em registros que
# não viu durante o aprendizado (MUS01, slides 39, 40 e 50; MUS02, slides 39 a 42).
#
# ### Treinamento e teste
#
# | Conjunto | Proporção | Finalidade |
# |---|---:|---|
# | Treinamento | 80% | Dados usados para ajustar os modelos e todas as transformações que aprendem parâmetros |
# | Teste | 20% | Dados reservados para a avaliação final, simulando registros novos que o modelo nunca viu |
#
# A proporção 80/20 é a exigida pelo enunciado (Fase 4, p. 5) e está na faixa apresentada no MUS01 (slide 50: treino
# de 70% a 80% e teste de 20% a 30%). Ela mantém a maior parte dos dados para o aprendizado e reserva 2.484 registros
# para a avaliação.
#
# | Parâmetro | Valor | Efeito |
# |---|---|---|
# | `test_size` | `0.2` | 20% dos registros vão para o teste |
# | `stratify` | `y` | A proporção de ocupados e desocupados é preservada nos dois conjuntos |
# | `random_state` | `42` | O sorteio é reproduzido exatamente a cada execução, nas versões testadas |
#
# ### Por que dividir antes de qualquer transformação de escala (Data Leakage)
#
# O `StandardScaler` aprende a média e o desvio padrão dos dados. Se ele fosse ajustado na base completa, essas
# estatísticas incluiriam informações dos registros de teste, e o modelo seria avaliado com dados que, indiretamente,
# já influenciaram sua preparação. Esse vazamento de dados (*Data Leakage*) produz avaliações otimistas, que não se
# repetem com dados realmente novos. Por isso, a divisão é feita **antes** de qualquer transformação estatística: o
# escalonamento será ajustado somente no treinamento (Fase 5) e apenas aplicado ao teste. O mesmo vale para qualquer
# outra decisão que aprenda com os dados, como a escolha de hiperparâmetros.
#
# ### Registros desta fase
#
# - Nenhum modelo é treinado nesta fase, e nenhuma transformação de escala é aplicada.
# - O conjunto de teste não deverá orientar a escolha de hiperparâmetros nem de atributos.
# - O título do enunciado menciona "Balanceamento", mas o requisito da fase é a estratificação. A estratificação
#   preserva a proporção original das classes; ela não as equilibra. Nenhuma técnica de balanceamento é aplicada.
# - O split aleatório exigido pelo enunciado é preservado. Como os registros são temporalmente dependentes (BLOCO
#   16.2), registros vizinhos tendem a ficar em conjuntos diferentes; essa proximidade é medida no BLOCO 44, e uma
#   validação temporal complementar será realizada posteriormente.

# %% [markdown]
# ### BLOCO 38 — Preparação da matriz X e do vetor y
#
# Os seis atributos preditores são as cinco medições dos sensores e a variável criada na Fase 3. `date` não entra em
# `X`: o modelo deve reconhecer a ocupação a partir das condições do ambiente, e não da data ou do horário dos
# registros, que identificam cada observação e não descrevem o ambiente. A coluna fica em uma estrutura separada
# (`rastreio_temporal`), com o mesmo índice, para a auditoria temporal do BLOCO 44. `Occupancy` é separada como `y`.

# %%
# BLOCO 38 — Preparação da matriz X e do vetor y

# Referência para confirmar, ao final da fase, que df_features não foi alterada
df_features_referencia = df_features.copy(deep=True)

# Atributos preditores, na ordem utilizada em todas as fases seguintes
atributos_preditores = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
    "CO2_Light_Ratio",
]
variavel_alvo = "Occupancy"

# Verificações anteriores à separação
assert len(atributos_preditores) == len(set(atributos_preditores)), "Atributos preditores duplicados."
assert set(atributos_preditores + [variavel_alvo, "date"]).issubset(df_features.columns), "Coluna ausente."
assert df_features.columns.is_unique, "Nomes de colunas duplicados em df_features."
assert "date" not in atributos_preditores and variavel_alvo not in atributos_preditores, "Coluna indevida em X."

# Separação de X, y e da estrutura de rastreabilidade temporal (índice original preservado)
X = df_features[atributos_preditores].copy()
y = df_features[variavel_alvo].copy()
rastreio_temporal = df_features[["date"]].copy()
rastreio_temporal["segmento"] = (rastreio_temporal["date"].diff() > limite_continuidade).cumsum()

# Validações
assert X.shape == (12417, 6), "X não possui as dimensões esperadas."
assert y.shape == (12417,), "y não possui a quantidade esperada de observações."
assert X.index.equals(y.index) and X.index.equals(rastreio_temporal.index), "Índices divergentes."
assert all(pd.api.types.is_float_dtype(X[coluna]) for coluna in atributos_preditores), "Atributo não float64."
assert X.isna().sum().sum() == 0 and y.isna().sum() == 0, "Existem valores ausentes."
assert np.isinf(X).sum().sum() == 0, "Existem valores infinitos."
assert set(y.unique()) == {0, 1}, "y possui classes diferentes de 0 e 1."
assert (rastreio_temporal["segmento"].to_numpy() == segmentos.to_numpy()).all(), "Segmentos divergentes do BLOCO 16.2."
pd.testing.assert_frame_equal(X, df_features[atributos_preditores], check_exact=True)

tabela_colunas_modelagem = pd.DataFrame({
    "Coluna": atributos_preditores + [variavel_alvo, "date"],
    "Tipo": [str(X[coluna].dtype) for coluna in atributos_preditores]
            + [str(y.dtype), str(rastreio_temporal["date"].dtype)],
    "Função": ["Preditor (sensor)"] * 5 + ["Preditor (variável derivada)", "Alvo (y)",
                                           "Rastreabilidade temporal (fora de X)"],
}).set_index("Coluna")
print(tabela_colunas_modelagem)
print("\nX:", X.shape, "| y:", y.shape, "| rastreio_temporal:", rastreio_temporal.shape)
print("Índices de 0 a", X.index.max(), "| únicos:", X.index.is_unique)
print("Registros por segmento temporal:", rastreio_temporal["segmento"].value_counts().sort_index().to_dict())

# %% [markdown]
# ### BLOCO 39 — Distribuição original das classes
#
# A acurácia de um classificador que previsse sempre a classe majoritária é calculada como referência descritiva: ela
# indica o patamar que qualquer modelo útil deverá superar, mas não é o resultado de um modelo treinado. O
# `DummyClassifier` será implementado na avaliação dos modelos. Nenhuma técnica de balanceamento (*oversampling*,
# *undersampling* ou SMOTE) é aplicada.

# %%
# BLOCO 39 — Distribuição original das classes

contagem_y = y.value_counts().sort_index()
percentual_y = y.value_counts(normalize=True).sort_index() * 100

tabela_classes_original = pd.DataFrame({
    "Registros": [contagem_y[0], contagem_y[1], contagem_y.sum()],
    "Percentual (%)": [percentual_y[0], percentual_y[1], percentual_y.sum()],
}, index=pd.Index(["0 — Desocupado", "1 — Ocupado", "Total"], name="Occupancy"))
print(tabela_classes_original.round(2))

classe_majoritaria = contagem_y.idxmax()
classe_minoritaria = contagem_y.idxmin()
razao_classes = contagem_y[classe_majoritaria] / contagem_y[classe_minoritaria]
acuracia_referencia = contagem_y[classe_majoritaria] / contagem_y.sum()

print("\nClasse majoritária:", classe_majoritaria, "| classe minoritária:", classe_minoritaria)
print(f"Razão entre as classes: {razao_classes:.4f} desocupados para cada ocupado")
print(f"Acurácia de referência (sempre prever a classe {classe_majoritaria}): {acuracia_referencia:.4%}")

# %% [markdown]
# ### BLOCO 40 — Divisão entre treinamento e teste
#
# `train_test_split` já está importado no BLOCO 01. Com o comportamento padrão (`shuffle=True`), os registros são
# embaralhados antes da divisão; com `stratify=y`, o sorteio é feito separadamente dentro de cada classe. Os índices
# originais de `X` e `y` são preservados nos quatro conjuntos, o que permite rastrear cada registro até `df_features`
# e até sua data. Nenhum escalonamento é aplicado nesta etapa, e a divisão é executada uma única vez, sem testar
# sementes alternativas.

# %%
# BLOCO 40 — Divisão entre treinamento e teste

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Quantidade esperada no teste: o scikit-learn arredonda 20% de 12.417 para cima
registros_teste_esperados = int(np.ceil(0.2 * len(X)))
assert len(X_test) == registros_teste_esperados and len(X_train) == len(X) - registros_teste_esperados

print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_test: ", X_test.shape, "| y_test: ", y_test.shape)
print(f"Proporção efetiva: treino {len(X_train) / len(X):.4%} | teste {len(X_test) / len(X):.4%}")
print("Primeiros índices originais do treino:", X_train.index[:5].tolist())
print("Primeiros índices originais do teste: ", X_test.index[:5].tolist())

# %% [markdown]
# ### BLOCO 41 — Verificação da estratificação
#
# A diferença entre a proporção de uma classe em um conjunto e na base completa é expressa em **pontos percentuais**
# (p.p.), isto é, pela subtração direta dos percentuais. Como as contagens são números inteiros, a igualdade exata das
# proporções não é possível; a referência de tolerância é o peso de um único registro no menor conjunto
# (100 / 2.484 ≈ 0,04 p.p.).

# %%
# BLOCO 41 — Verificação da estratificação

def resumir_classes(serie_alvo, nome_conjunto):
    contagem = serie_alvo.value_counts().sort_index()
    return {
        "Conjunto": nome_conjunto,
        "Classe 0": contagem[0],
        "Classe 1": contagem[1],
        "Total": len(serie_alvo),
        "% Classe 0": contagem[0] / len(serie_alvo) * 100,
        "% Classe 1": contagem[1] / len(serie_alvo) * 100,
        "Razão 0/1": contagem[0] / contagem[1],
    }

tabela_estratificacao = pd.DataFrame([
    resumir_classes(y, "Base completa"),
    resumir_classes(y_train, "Treinamento"),
    resumir_classes(y_test, "Teste"),
]).set_index("Conjunto")
tabela_estratificacao["Diferença % Classe 1 (p.p.)"] = (
    tabela_estratificacao["% Classe 1"] - tabela_estratificacao.loc["Base completa", "% Classe 1"]
)
tabela_estratificacao["Proporção da base (%)"] = tabela_estratificacao["Total"] / len(y) * 100
print(tabela_estratificacao.round(4).to_string())

desvio_maximo_pp = tabela_estratificacao["Diferença % Classe 1 (p.p.)"].abs().max()
tolerancia_pp = 100 / len(y_test)
print(f"\nDesvio máximo da proporção das classes: {desvio_maximo_pp:.4f} p.p.")
print(f"Peso de um registro no conjunto de teste: {tolerancia_pp:.4f} p.p.")

assert tabela_estratificacao.loc["Treinamento", "Classe 0"] + tabela_estratificacao.loc["Teste", "Classe 0"] \
    == tabela_estratificacao.loc["Base completa", "Classe 0"], "Classe 0 não se conserva."
assert tabela_estratificacao.loc["Treinamento", "Classe 1"] + tabela_estratificacao.loc["Teste", "Classe 1"] \
    == tabela_estratificacao.loc["Base completa", "Classe 1"], "Classe 1 não se conserva."
assert desvio_maximo_pp <= tolerancia_pp, "A estratificação desviou mais do que um registro."
print("Estratificação verificada.")

# %% [markdown]
# ### BLOCO 42 — Validação da integridade da divisão
#
# As verificações confirmam que cada registro foi para exatamente um dos conjuntos e que nenhum valor foi alterado. A
# ausência de registros em comum prova apenas que treino e teste não compartilham **linhas**; ela não significa
# independência estatística, pois registros vizinhos no tempo são muito semelhantes (BLOCO 44).

# %%
# BLOCO 42 — Validação da integridade da divisão

indices_treino = set(X_train.index)
indices_teste = set(X_test.index)

assert X_train.index.equals(y_train.index), "Índices de X_train e y_train divergentes."
assert X_test.index.equals(y_test.index), "Índices de X_test e y_test divergentes."
assert X_train.index.is_unique and X_test.index.is_unique, "Índices repetidos."
assert indices_treino.isdisjoint(indices_teste), "Registros presentes no treino e no teste."
assert indices_treino | indices_teste == set(X.index), "A união dos conjuntos não reproduz a base."
assert len(X_train) + len(X_test) == len(X) == 12417, "Quantidade total de registros divergente."
assert (y_train.value_counts() + y_test.value_counts()).sort_index().equals(contagem_y), "Classes divergentes."
assert list(X_train.columns) == list(X_test.columns) == atributos_preditores, "Atributos ou ordem divergentes."
assert "date" not in X_train.columns and variavel_alvo not in X_train.columns, "Coluna indevida nos preditores."
assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0, "Existem valores ausentes."
assert np.isinf(X_train).sum().sum() == 0 and np.isinf(X_test).sum().sum() == 0, "Existem valores infinitos."
assert (X_train.dtypes == X.dtypes).all() and (X_test.dtypes == X.dtypes).all(), "Tipos alterados."

# Reconstrução temporária da base a partir dos dois conjuntos, ordenada pelo índice original
X_reconstruido = pd.concat([X_train, X_test]).sort_index()
y_reconstruido = pd.concat([y_train, y_test]).sort_index()
pd.testing.assert_frame_equal(X_reconstruido, X, check_exact=True)
pd.testing.assert_series_equal(y_reconstruido, y, check_exact=True)

# Base de Feature Engineering preservada
pd.testing.assert_frame_equal(df_features, df_features_referencia, check_exact=True)

print("Registros no treino:", len(indices_treino), "| no teste:", len(indices_teste))
print("Registros em comum entre treino e teste:", len(indices_treino & indices_teste))
print("Reconstrução de X e y idêntica à base original (check_exact=True): True")
print("df_features inalterada: True")
print("Validação da integridade da divisão concluída com sucesso!")

# %% [markdown]
# ### BLOCO 43 — Validação da reprodutibilidade
#
# A divisão é repetida em variáveis temporárias, com os mesmos parâmetros, e comparada à divisão oficial. Os conjuntos
# temporários não substituem os oficiais. A igualdade comprova que a semente fixa reproduz a divisão nas condições e
# versões testadas (scikit-learn 1.8.0); ela não elimina vieses e não garante o mesmo sorteio em outras versões das
# bibliotecas.

# %%
# BLOCO 43 — Validação da reprodutibilidade

X_train_repeticao, X_test_repeticao, y_train_repeticao, y_test_repeticao = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

pd.testing.assert_frame_equal(X_train_repeticao, X_train, check_exact=True)
pd.testing.assert_frame_equal(X_test_repeticao, X_test, check_exact=True)
pd.testing.assert_series_equal(y_train_repeticao, y_train, check_exact=True)
pd.testing.assert_series_equal(y_test_repeticao, y_test, check_exact=True)
print("Repetição com test_size=0.2, random_state=42 e stratify=y: índices e valores idênticos nos quatro conjuntos.")

del X_train_repeticao, X_test_repeticao, y_train_repeticao, y_test_repeticao

# %% [markdown]
# ### BLOCO 44 — Auditoria da proximidade temporal entre treino e teste
#
# Análise complementar ao enunciado, motivada pela dependência temporal medida no BLOCO 16.2. Três medidas
# diferentes são calculadas para cada registro de teste, sempre dentro do seu segmento temporal contínuo (registros
# do outro segmento não são considerados vizinhos):
#
# | Medida | Definição |
# |---|---|
# | Vizinhança por posição | O registro imediatamente anterior ou posterior, na ordem das linhas do segmento, está no treino |
# | Vizinhança por diferença temporal | Existe um registro de treino a até 60, 61 ou 120 segundos |
# | Distância ao treino mais próximo | Menor diferença absoluta de tempo até qualquer registro de treino do segmento |
#
# O limite de 61 segundos acompanha os intervalos reais entre registros consecutivos (59, 60 ou 61 segundos, BLOCO
# 16.1): um vizinho imediato pode estar a 61 segundos, e o limite de 60 segundos não o contaria. Como dois passos
# consecutivos somam pelo menos 118 segundos, uma distância de até 61 segundos só ocorre com o vizinho imediato. A
# distância mais próxima é obtida por busca binária (`np.searchsorted`) nos tempos ordenados do treino, sem calcular
# a matriz completa de distâncias. `Occupancy` não é usada, e os conjuntos não são alterados após esta análise.

# %%
# BLOCO 44 — Auditoria da proximidade temporal entre treino e teste

no_treino = pd.Series(X.index.isin(X_train.index), index=X.index)
datas_teste = rastreio_temporal.loc[X_test.index]

distancia_treino_s = pd.Series(np.nan, index=X_test.index, name="distancia_s")
vizinho_posicao_treino = pd.Series(False, index=X_test.index)
vizinhos_elegiveis = pd.Series(0, index=X_test.index)

for segmento_id, indices_segmento in rastreio_temporal.groupby("segmento").groups.items():
    tempos_segmento = rastreio_temporal.loc[indices_segmento, "date"].to_numpy()
    assert (np.diff(tempos_segmento) > np.timedelta64(0, "s")).all(), "Tempos fora de ordem no segmento."
    marcador_treino = no_treino.loc[indices_segmento].to_numpy()
    tempos_treino = tempos_segmento[marcador_treino]
    posicoes_teste = np.flatnonzero(~marcador_treino)
    tempos_teste = tempos_segmento[posicoes_teste]

    # Distância ao registro de treino mais próximo (busca binária nos tempos ordenados)
    posicao_insercao = np.searchsorted(tempos_treino, tempos_teste)
    anterior = tempos_treino[np.clip(posicao_insercao - 1, 0, len(tempos_treino) - 1)]
    posterior = tempos_treino[np.clip(posicao_insercao, 0, len(tempos_treino) - 1)]
    distancia_anterior = np.where(posicao_insercao > 0, (tempos_teste - anterior) / np.timedelta64(1, "s"), np.inf)
    distancia_posterior = np.where(
        posicao_insercao < len(tempos_treino), (posterior - tempos_teste) / np.timedelta64(1, "s"), np.inf
    )
    rotulos_teste = indices_segmento[posicoes_teste]
    distancia_treino_s.loc[rotulos_teste] = np.minimum(distancia_anterior, distancia_posterior)

    # Vizinhança por posição (registros das extremidades têm apenas um vizinho elegível)
    tem_anterior = posicoes_teste - 1 >= 0
    tem_posterior = posicoes_teste + 1 < len(indices_segmento)
    anterior_no_treino = tem_anterior & marcador_treino[np.clip(posicoes_teste - 1, 0, None)]
    posterior_no_treino = tem_posterior & marcador_treino[np.clip(posicoes_teste + 1, None, len(indices_segmento) - 1)]
    vizinho_posicao_treino.loc[rotulos_teste] = anterior_no_treino | posterior_no_treino
    vizinhos_elegiveis.loc[rotulos_teste] = tem_anterior.astype(int) + tem_posterior.astype(int)

assert distancia_treino_s.notna().all() and np.isfinite(distancia_treino_s).all(), "Distância não calculada."

# Validação independente em uma amostra: menor distância calculada por força bruta
amostra_validacao = X_test.index.to_series().sample(n=100, random_state=42)
for indice in amostra_validacao:
    segmento_registro = rastreio_temporal.loc[indice, "segmento"]
    candidatos = X_train.index[rastreio_temporal.loc[X_train.index, "segmento"].eq(segmento_registro)]
    menor = (rastreio_temporal.loc[candidatos, "date"] - rastreio_temporal.loc[indice, "date"]).abs().min()
    assert menor / pd.Timedelta(seconds=1) == distancia_treino_s.loc[indice], f"Divergência no índice {indice}."
print("Validação por força bruta em 100 registros de teste: aprovada.")

# Estatísticas da menor distância ao treino
estatisticas_distancia = distancia_treino_s.describe()[["count", "min", "25%", "50%", "mean", "75%", "max"]]
print("\nMenor distância (s) entre cada registro de teste e o treino do mesmo segmento:")
print(estatisticas_distancia.round(2).to_string())
print("\nFrequência das distâncias mais comuns (s):")
print(distancia_treino_s.value_counts().head(8).to_string())

# Resumo geral e por segmento
def resumir_proximidade(mascara):
    distancias = distancia_treino_s[mascara]
    return {
        "Registros de teste": int(mascara.sum()),
        "Com vizinho de treino elegível": int(np.isfinite(distancias).sum()),
        "Nas extremidades do segmento": int((vizinhos_elegiveis[mascara] == 1).sum()),
        "Mediana (s)": distancias.median(),
        "Até 60 s (%)": (distancias <= 60).mean() * 100,
        "Até 61 s (%)": (distancias <= 61).mean() * 100,
        "Até 120 s (%)": (distancias <= 120).mean() * 100,
        "Vizinho por posição no treino (%)": vizinho_posicao_treino[mascara].mean() * 100,
    }

segmento_teste = datas_teste["segmento"]
tabela_proximidade = pd.DataFrame({
    "Base completa": resumir_proximidade(pd.Series(True, index=X_test.index)),
    "Segmento 0": resumir_proximidade(segmento_teste.eq(0)),
    "Segmento 1": resumir_proximidade(segmento_teste.eq(1)),
}).T
print("\nProximidade temporal entre teste e treino:")
print(tabela_proximidade.round(2).to_string())

# Comparação com a aproximação teórica (dois vizinhos elegíveis, atribuições independentes)
estimativa_teorica = 1 - 0.2 ** 2
medicao_posicao = vizinho_posicao_treino.mean()
print(f"\nEstimativa teórica (1 − 0,2²): {estimativa_teorica:.2%}")
print(f"Medição por posição após o split: {medicao_posicao:.4%}")
print("Vizinhança por posição equivale ao critério de até 61 s:",
      vizinho_posicao_treino.equals(distancia_treino_s <= 61))

# %% [markdown]
# ### BLOCO 45 — Visualização da divisão dos dados
#
# A figura reúne a quantidade de registros, a proporção das classes em cada conjunto e a distribuição acumulada da
# menor distância temporal entre cada registro de teste e o treino. Os pontos destacados na curva indicam a fração de
# registros de teste a até 60, 61 e 120 segundos de um registro de treino.

# %%
# BLOCO 45 — Visualização da divisão dos dados

nomes_conjuntos = ["Base completa", "Treinamento", "Teste"]
cores_classes = sns.color_palette("Set2", 2)

def formatar_inteiro(valor):
    return f"{int(valor):,}".replace(",", ".")

def formatar_percentual(valor):
    return f"{valor:.2f}%".replace(".", ",")

fig, (eixo_quantidade, eixo_classes, eixo_distancia) = plt.subplots(1, 3, figsize=(19, 6))

# Painel 1 — Quantidade de registros
quantidades = tabela_estratificacao.loc[nomes_conjuntos, "Total"]
barras = eixo_quantidade.bar(nomes_conjuntos, quantidades, color="#4C72B0", edgecolor="black")
for barra, quantidade in zip(barras, quantidades):
    eixo_quantidade.annotate(
        f"{formatar_inteiro(quantidade)}\n({formatar_percentual(quantidade / len(y) * 100)})",
        (barra.get_x() + barra.get_width() / 2, barra.get_height()),
        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=10,
    )
eixo_quantidade.set_ylim(0, len(y) * 1.18)
eixo_quantidade.set_title("Quantidade de registros")
eixo_quantidade.set_ylabel("Registros")

# Painel 2 — Proporção das classes (barras empilhadas em percentual)
base_empilhada = np.zeros(len(nomes_conjuntos))
for classe, cor, rotulo in zip([0, 1], cores_classes, ["0 — Desocupado", "1 — Ocupado"]):
    percentuais = tabela_estratificacao.loc[nomes_conjuntos, f"% Classe {classe}"].to_numpy()
    contagens = tabela_estratificacao.loc[nomes_conjuntos, f"Classe {classe}"].to_numpy()
    eixo_classes.bar(nomes_conjuntos, percentuais, bottom=base_empilhada, color=cor, edgecolor="black", label=rotulo)
    for posicao, (percentual, contagem, base) in enumerate(zip(percentuais, contagens, base_empilhada)):
        eixo_classes.text(posicao, base + percentual / 2,
                          f"{formatar_inteiro(contagem)}\n({formatar_percentual(percentual)})",
                          ha="center", va="center", fontsize=10)
    base_empilhada += percentuais
eixo_classes.set_ylim(0, 100)
eixo_classes.set_title("Proporção das classes de Occupancy")
eixo_classes.set_ylabel("Percentual dos registros (%)")
eixo_classes.legend(title="Classe", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

# Painel 3 — Distribuição acumulada da menor distância ao treino
distancias_ordenadas = np.sort(distancia_treino_s.to_numpy())
fracao_acumulada = np.arange(1, len(distancias_ordenadas) + 1) / len(distancias_ordenadas) * 100
eixo_distancia.step(distancias_ordenadas, fracao_acumulada, where="post", color="#4C72B0", linewidth=2)
# Rótulos em área livre do gráfico, ligados aos pontos por setas (evita sobreposição com a curva)
posicoes_rotulo = {60: (110, 62), 61: (110, 74), 120: (160, 86)}
for limite, posicao_texto in posicoes_rotulo.items():
    fracao = (distancia_treino_s <= limite).mean() * 100
    eixo_distancia.plot(limite, fracao, marker="o", color="#C44E52", markersize=6, zorder=3)
    eixo_distancia.annotate(f"até {limite} s: {formatar_percentual(fracao)}", (limite, fracao),
                            xytext=posicao_texto, textcoords="data", fontsize=10,
                            arrowprops={"arrowstyle": "->", "color": "#C44E52", "linewidth": 1})
eixo_distancia.set_xlim(50, distancias_ordenadas.max() + 10)
eixo_distancia.set_ylim(0, 102)
eixo_distancia.set_title("Menor distância de cada registro de teste ao treino")
eixo_distancia.set_xlabel("Distância temporal (segundos), dentro do mesmo segmento")
eixo_distancia.set_ylabel("Registros de teste acumulados (%)")

plt.suptitle("Divisão treino/teste (80/20, estratificada, random_state=42)", fontsize=16)
plt.tight_layout()

# Exportação antes da exibição
plt.savefig(caminho_grafico_split, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_split)

plt.show()

# %% [markdown]
# ### BLOCO 46 — Resumo consolidado da Fase 4
#
# Os conjuntos permanecem em memória para as fases seguintes e são reproduzidos pela execução sequencial do script.
# Nenhum arquivo CSV de treino ou teste é exportado.

# %%
# BLOCO 46 — Resumo consolidado da Fase 4

tabela_dimensoes = pd.DataFrame({
    "Linhas": [X_train.shape[0], X_test.shape[0], y_train.shape[0], y_test.shape[0]],
    "Colunas": [X_train.shape[1], X_test.shape[1], 1, 1],
}, index=["X_train", "X_test", "y_train", "y_test"])
print("Dimensões dos conjuntos:")
print(tabela_dimensoes)

print("\nAtributos preditores:", atributos_preditores)

tabela_resumo_fase4 = pd.Series({
    "Registros (treino / teste)": f"{len(X_train)} / {len(X_test)}",
    "Classe 1 na base / treino / teste (%)": " / ".join(
        f"{valor:.2f}" for valor in tabela_estratificacao["% Classe 1"]
    ),
    "Desvio máximo de proporção (p.p.)": f"{desvio_maximo_pp:.4f}",
    "Registros em comum entre treino e teste": len(indices_treino & indices_teste),
    "Reconstrução exata de X e y": True,
    "Divisão reproduzida com a mesma semente": True,
    "Mediana da distância ao treino (s)": distancia_treino_s.median(),
    "Teste com vizinho imediato no treino (%)": round(medicao_posicao * 100, 2),
    "df_features preservada": df_features.equals(df_features_referencia),
})
print("\nResumo da Fase 4:")
print(tabela_resumo_fase4.to_string())

# %% [markdown]
# ## BLOCO 47 — Interpretação e conclusão da divisão dos dados
#
# ### 47.1 — Seleção das variáveis
#
# `X` reúne seis atributos `float64`, na ordem fixa `Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio` e
# `CO2_Light_Ratio`: as cinco medições dos sensores e a variável derivada da Fase 3. `y` contém apenas `Occupancy`
# (`int64`, classes 0 e 1). A coluna `date` ficou fora de `X` porque identifica o momento da observação e não descreve
# as condições do ambiente; usá-la, ou atributos derivados dela, faria o modelo depender do período específico desta
# coleta. Ela permanece em `rastreio_temporal`, com o mesmo índice de `X` e `y`, apenas para a auditoria temporal. As
# validações confirmaram 12.417 × 6 em `X`, 12.417 observações em `y`, índices idênticos, nenhum valor ausente ou
# infinito e valores iguais aos de `df_features` (`check_exact=True`).
#
# ### 47.2 — Divisão 80/20
#
# | Conjunto | X | y |
# |---|---|---|
# | Treinamento | 9.933 × 6 | 9.933 |
# | Teste | 2.484 × 6 | 2.484 |
#
# A proporção efetiva é de 79,9952% para treino e 20,0048% para teste. A pequena diferença em relação a 80/20 vem do
# arredondamento: 20% de 12.417 é 2.483,4, e o scikit-learn arredonda o tamanho do teste para cima (2.484 registros).
#
# ### 47.3 — Estratificação e desequilíbrio
#
# | Conjunto | Classe 0 | Classe 1 | % Classe 1 | Diferença (p.p.) |
# |---|---:|---:|---:|---:|
# | Base completa | 9.396 | 3.021 | 24,3295% | — |
# | Treinamento | 7.516 | 2.417 | 24,3330% | +0,0035 |
# | Teste | 1.880 | 604 | 24,3156% | −0,0139 |
#
# O maior desvio (0,0139 p.p.) é menor que o peso de um único registro no teste (0,0403 p.p.): a distribuição foi
# preservada tão de perto quanto as contagens inteiras permitem. A estratificação reproduz o desequilíbrio original
# (cerca de 3,11 desocupados para cada ocupado) em ambos os conjuntos; ela evita que o teste tenha, por acaso, uma
# proporção diferente da base, mas não equilibra as classes. A acurácia de referência de quem previsse sempre
# "desocupado" é 75,67% na base — valor descritivo que os modelos deverão superar, e não o resultado de um modelo.
#
# ### 47.4 — Integridade e reprodutibilidade
#
# Os índices de `X_train` e `y_train`, e de `X_test` e `y_test`, são idênticos e únicos; nenhum registro aparece nos
# dois conjuntos, e sua união reproduz os 12.417 índices da base. As classes se conservam (7.516 + 1.880 = 9.396 e
# 2.417 + 604 = 3.021), os seis atributos mantêm nome, ordem e tipo, e `date` e `Occupancy` não estão entre os
# preditores. A concatenação dos dois conjuntos, reordenada pelo índice original, reconstruiu `X` e `y` exatamente
# (`check_exact=True`), e `df_features` permaneceu inalterada. A repetição da divisão com os mesmos parâmetros
# produziu os mesmos índices e valores nos quatro conjuntos, nas versões testadas.
#
# ### 47.5 — Dependência temporal entre os conjuntos
#
# | Medida (registros de teste, mesmo segmento) | Base | Segmento 0 | Segmento 1 |
# |---|---:|---:|---:|
# | Registros de teste | 2.484 | 535 | 1.949 |
# | Mediana da distância ao treino | 60 s | 60 s | 60 s |
# | Até 60 s | 90,10% | 90,47% | 89,99% |
# | Até 61 s | 96,10% | 96,26% | 96,05% |
# | Até 120 s | 99,52% | 100,00% | 99,38% |
# | Vizinho imediato (por posição) no treino | 96,10% | 96,26% | 96,05% |
#
# Todos os 2.484 registros de teste têm algum registro de treino no mesmo segmento. A menor distância varia de 59 a
# 240 segundos (média de 62,25 s); apenas 1 registro de teste está na extremidade de um segmento. As três medidas
# respondem a perguntas diferentes, mas, nesta base, a vizinhança por posição coincide exatamente com o critério de
# até 61 segundos, porque registros consecutivos distam de 59 a 61 segundos. O limite de 60 segundos subestima a
# proximidade (90,10%), pois exclui os vizinhos registrados a 61 segundos.
#
# A medição (96,10%) ficou muito próxima da aproximação teórica 1 − 0,2² = 96%, calculada supondo dois vizinhos
# elegíveis e atribuições independentes. A pequena diferença é compatível com essas hipóteses aproximadas: a
# estratificação torna o sorteio dependente da classe, e um registro de extremidade tem apenas um vizinho.
#
# Na prática, para quase todo registro de teste existe, no treino, uma medição feita cerca de um minuto antes ou
# depois, com valores dos sensores quase idênticos (autocorrelação de lag 1 acima de 0,99, BLOCO 16.2). Um modelo pode
# acertar esses registros por semelhança com o vizinho, e não por ter aprendido um padrão que se mantenha em outros
# períodos. Por isso, a acurácia no teste tende a ser otimista como estimativa do desempenho em dados de dias ou
# ambientes diferentes. Isso não invalida os resultados: o teste continua medindo o desempenho em registros não usados
# no ajuste, mas sua interpretação deve considerar essa proximidade, e a avaliação temporal complementar ajudará a
# dimensioná-la.
#
# ### 47.6 — Prevenção de Data Leakage
#
# - O conjunto de teste foi separado antes de qualquer transformação que aprenda parâmetros e não será usado para
#   ajustar o escalonamento, escolher atributos ou escolher hiperparâmetros; ele fica reservado para a avaliação.
# - A EDA, a investigação de outliers e a comparação das variáveis candidatas da Fase 3 foram feitas na base completa,
#   antes desta divisão. Nenhum parâmetro de modelo foi ajustado com esses dados, mas essas decisões observaram
#   registros que agora estão no teste — limitação registrada desde a Fase 3.
# - A ausência de registros em comum garante que nenhuma linha é usada nos dois papéis, mas não garante independência
#   estatística: registros vizinhos no tempo carregam praticamente a mesma informação.
# - A seleção de hiperparâmetros será feita por validação cruzada apenas no treinamento.
# - Esta fase caracteriza a proximidade temporal e o equilíbrio das classes; nenhuma métrica de desempenho no teste foi
#   calculada, e nenhuma decisão foi tomada com base nele.
#
# ### 47.7 — Preparação para o StandardScaler
#
# Os atributos têm escalas muito diferentes (por exemplo, desvio padrão de cerca de 1 °C em `Temperature` e de cerca de
# 400 em `CO2_Light_Ratio`), e o KNN calcula distâncias entre registros. Na Fase 5, o `StandardScaler` será ajustado
# somente em `X_train` (`fit_transform`), e `X_test` será transformado com a média e o desvio padrão aprendidos no
# treino (`transform`), sem novo ajuste. A Árvore de Decisão usará os atributos sem escalonamento, pois suas divisões
# dependem apenas da ordem dos valores. Na validação cruzada do KNN, o scaler deverá estar dentro de um `Pipeline`,
# para que cada divisão interna seja escalonada apenas com os dados da sua parte de treino.
#
# ### 47.8 — Limitações e próximos passos
#
# - A dependência temporal é elevada, e 96,10% dos registros de teste têm um vizinho imediato no treino.
# - As classes continuam desequilibradas (24,33% de ocupados); a acurácia deverá ser acompanhada de métricas por
#   classe e comparada com a referência da classe majoritária.
# - Há redundância parcial entre atributos (`Humidity` × `HumidityRatio`, `Temperature` × `Light` e
#   `CO2_Light_Ratio` × `Light`), cujo efeito será examinado com e sem a variável derivada, dentro do treino.
# - A base consolida dois arquivos com uma descontinuidade de sete dias e proporções de ocupação diferentes.
# - Uma avaliação temporal complementar e, possivelmente, uma avaliação em `datatraining.txt` — período intermediário
#   do mesmo conjunto de dados, não usado no ajuste dos modelos — serão realizadas ao final, sem substituir o split
#   exigido.
# - Nenhum resultado preditivo foi produzido nesta fase. A próxima etapa é a Fase 5 — escalonamento com
#   `StandardScaler`.

# %% [markdown]
# # FASE 5 — StandardScaler (Padronização dos Dados)
# ## BLOCO 48 — Contextualização do escalonamento
#
# ### Requisitos do enunciado
#
# | Fonte | Exigência |
# |---|---|
# | Enunciado, Fase 5 (p. 5) | Aplicar o `StandardScaler` nas variáveis numéricas destinadas ao KNN, usando obrigatoriamente `fit_transform()` apenas no treino e `transform()` no teste |
# | Enunciado, Fase 5 (p. 5) | Manter sem escalonamento os dados da Árvore de Decisão e justificar no código por que algoritmos baseados em árvores são imunes à diferença de escala entre os sensores (ex.: Lux de 0 a mais de 1.000 e Temperatura de 19 a 24) |
# | Rubrica, critério 8 (p. 8 e 9) | Pontuação total: `StandardScaler` apenas no KNN (`fit_transform` no treino e `transform` no teste) e justificativa da ausência na Árvore |
#
# Os materiais didáticos apresentam o mesmo procedimento: o MUS01 (slide 48) inclui a normalização/padronização no
# pré-processamento, para não comparar diretamente grandezas de escalas diferentes; o notebook `knn.ipynb` aplica
# `fit_transform` no treino e `transform` no teste e explica que, sem padronização, uma variável de valores grandes
# domina o cálculo da distância; os notebooks `01_pipeline_exemplo.ipynb` e `03_pipeline_exemplo.ipynb` mostram o mesmo
# par de chamadas e o `StandardScaler` dentro de um `Pipeline`, ajustado apenas na parte de treino de cada divisão da
# validação cruzada. A inspeção das escalas e dos parâmetros, a validação matemática, a transformação inversa e a
# análise das distâncias desta fase são **complementos metodológicos** deste projeto, e não exigências do enunciado.
#
# ### Escalonamento
#
# Escalonar é expressar atributos medidos em unidades e ordens de grandeza diferentes em escalas numéricas comparáveis,
# sem alterar a ordem dos valores dentro de cada atributo.
#
# | Técnica | Fórmula | Resultado no conjunto de ajuste |
# |---|---|---|
# | Padronização (`StandardScaler`, usada nesta fase) | `z = (x − μ) / σ` | Média 0 e desvio padrão 1; valores sem limites fixos |
# | Normalização Min-Max (`MinMaxScaler`, apenas para comparação) | `x' = (x − mínimo) / (máximo − mínimo)` | Valores entre 0 e 1; depende diretamente dos dois valores extremos |
#
# ### Fórmula do StandardScaler
#
# Para cada atributo `j`, o scaler aprende no treinamento (`n` = 9.933 registros):
#
# - a média `μ_j = (1/n) · Σ x_ij`;
# - a variância populacional `σ²_j = (1/n) · Σ (x_ij − μ_j)²`, com divisor `n` (`ddof=0`), e não `n − 1`;
# - o desvio padrão populacional `σ_j = √σ²_j`.
#
# Cada valor é transformado em `z_ij = (x_ij − μ_j) / σ_j`, isto é, na distância até a média do treino medida em
# desvios padrão do treino. Os parâmetros ficam em `scaler.mean_` (μ), `scaler.var_` (σ²) e `scaler.scale_` (σ). Se
# um atributo tivesse variância zero, o scikit-learn usaria fator de escala 1 para evitar divisão por zero.
#
# | Método | O que faz | Uso nesta fase |
# |---|---|---|
# | `fit` | Aprende μ e σ de cada atributo | Executado dentro de `fit_transform` |
# | `transform` | Aplica a fórmula com os parâmetros **já aprendidos** | `X_test` |
# | `fit_transform` | `fit` seguido de `transform` no mesmo conjunto | `X_train` |
#
# ### Escala e distância
#
# O KNN classifica um registro pelos vizinhos mais próximos, usando por padrão a distância euclidiana
# `d(a, b) = √Σ_j (x_aj − x_bj)²`. As diferenças de todos os atributos são somadas como números puros: uma diferença de
# 100 ppm de `CO2` pesa 10.000 vezes mais no quadrado da distância que uma diferença de 1 °C de `Temperature`, apenas
# por causa das unidades. Após a padronização, cada diferença é medida em desvios padrão do treino do próprio atributo.
#
# ### Data Leakage
#
# O scaler é ajustado depois da divisão da Fase 4 e somente com `X_train`. O teste é transformado com a média e o
# desvio padrão do treino, como aconteceria com registros novos em produção. Ajustar o scaler na base completa ou no
# teste faria estatísticas do teste participarem da preparação dos dados (BLOCO 37).
#
# ### Por que a Árvore de Decisão não precisa de escalonamento
#
# Cada divisão de uma árvore compara **um único atributo** com um limiar (na forma `Light ≤ limiar`) e escolhe a
# divisão que melhor separa as classes. O resultado depende apenas da **ordem** dos valores daquele atributo, e as
# diferenças de atributos distintos nunca são somadas. A padronização é uma transformação crescente (σ > 0): preserva
# a ordem dos valores e, portanto, as mesmas separações possíveis dos registros — só o valor numérico do limiar muda.
# Por isso, a diferença de escala entre `Light` (0 a mais de 1.000 lux) e `Temperature` (19 a 24 °C) não afeta a
# árvore, e os dados destinados a ela permanecem **sem escalonamento**.
#
# ### Registros desta fase
#
# - Apenas os atributos de `X` são padronizados; a variável alvo `y` não é transformada.
# - O conjunto de teste não participa do ajuste.
# - O `StandardScaler` muda a posição e a escala de cada atributo, mas não a forma da distribuição: não torna os dados
#   normais, não remove outliers e não altera as correlações entre os atributos.
# - Nenhum classificador é instanciado ou treinado nesta fase.

# %% [markdown]
# ### BLOCO 49 — Inspeção das escalas originais
#
# As estatísticas são calculadas **somente em `X_train`**, o conjunto em que o scaler será ajustado. A variância e o
# desvio padrão são populacionais (`ddof=0`), como no `StandardScaler`; a assimetria é a do pandas (`skew`, ajustada
# para o tamanho da amostra), a mesma usada nas fases anteriores. Os valores armazenados não são arredondados; o
# arredondamento é aplicado apenas na exibição.
#
# Os atributos têm unidades diferentes, e por isso seus desvios padrão não são comparáveis como grandezas físicas. A
# comparação abaixo trata somente da **ordem de grandeza numérica**, que é o que a distância euclidiana enxerga, e não
# indica importância preditiva.

# %%
# BLOCO 49 — Inspeção das escalas originais (somente treinamento)

unidades_atributos = {
    "Temperature": "°C",
    "Humidity": "% (umidade relativa)",
    "Light": "lux",
    "CO2": "ppm",
    "HumidityRatio": "kg/kg (razão de umidade)",
    "CO2_Light_Ratio": "índice experimental",
}

tabela_escalas = pd.DataFrame({
    "Unidade": pd.Series(unidades_atributos),
    "Tipo": X_train.dtypes.astype(str),
    "Observações": X_train.count(),
    "Mínimo": X_train.min(),
    "Máximo": X_train.max(),
    "Amplitude": X_train.max() - X_train.min(),
    "Média": X_train.mean(),
    "Mediana": X_train.median(),
    "Variância (ddof=0)": X_train.var(ddof=0),
    "Desvio padrão (ddof=0)": X_train.std(ddof=0),
    "Assimetria": X_train.skew(),
}).loc[atributos_preditores]
tabela_escalas.index.name = "Atributo"
print(tabela_escalas.to_string(float_format=lambda valor: f"{valor:.6g}"))

# Ordem de grandeza numérica dos desvios padrão (unidades diferentes)
desvios_originais = tabela_escalas["Desvio padrão (ddof=0)"]
atributo_maior_escala = desvios_originais.idxmax()
atributo_menor_escala = desvios_originais.idxmin()
print("\nAtributos ordenados pelo desvio padrão numérico:")
print(desvios_originais.sort_values(ascending=False).map(lambda valor: f"{valor:.6g}").to_string())
print(f"\nMaior desvio padrão numérico: {atributo_maior_escala} ({desvios_originais.max():.6g})")
print(f"Menor desvio padrão numérico: {atributo_menor_escala} ({desvios_originais.min():.6g})")
print(f"Razão entre o maior e o menor desvio padrão numérico: {desvios_originais.max() / desvios_originais.min():.6g}")

assert (tabela_escalas["Observações"] == len(X_train)).all(), "Observações ausentes no treinamento."

# %% [markdown]
# ### BLOCO 50 — Preservação dos conjuntos originais
#
# Os conjuntos padronizados serão criados com **nomes novos** (`X_train_scaled` e `X_test_scaled`). Os conjuntos
# originais continuam disponíveis para a Árvore de Decisão e para o `Pipeline` do KNN na Fase 6. Cópias independentes
# são guardadas como referência para confirmar, ao final da fase, que nenhum valor original foi alterado.

# %%
# BLOCO 50 — Preservação dos conjuntos originais

# Referências independentes (cópias profundas) para a verificação final
X_train_referencia = X_train.copy(deep=True)
X_test_referencia = X_test.copy(deep=True)
y_train_referencia = y_train.copy(deep=True)
y_test_referencia = y_test.copy(deep=True)
X_referencia = X.copy(deep=True)
y_referencia = y.copy(deep=True)
rastreio_temporal_referencia = rastreio_temporal.copy(deep=True)
# df_features_referencia foi criada no BLOCO 38

# Validações dos conjuntos que serão usados no escalonamento
assert X_train.shape == (9933, 6) and X_test.shape == (2484, 6), "Dimensões de X_train ou X_test divergentes."
assert y_train.shape == (9933,) and y_test.shape == (2484,), "Dimensões de y_train ou y_test divergentes."
assert X_train.index.is_unique and X_test.index.is_unique, "Índices repetidos."
assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index), "Índices de X e y divergentes."
assert set(X_train.index).isdisjoint(X_test.index), "Registros presentes no treino e no teste."
assert list(X_train.columns) == list(X_test.columns) == atributos_preditores, "Atributos ou ordem divergentes."
assert (X_train.dtypes == "float64").all() and (X_test.dtypes == "float64").all(), "Atributo não float64."
assert "date" not in X_train.columns and variavel_alvo not in X_train.columns, "Coluna indevida nos preditores."
assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0, "Existem valores ausentes."
assert np.isinf(X_train).sum().sum() == 0 and np.isinf(X_test).sum().sum() == 0, "Existem valores infinitos."
pd.testing.assert_frame_equal(df_features, df_features_referencia, check_exact=True)

tabela_conjuntos_originais = pd.DataFrame({
    "Dimensões": [str(conjunto.shape) for conjunto in [X_train, X_test, y_train, y_test]],
    "Índices únicos": [conjunto.index.is_unique for conjunto in [X_train, X_test, y_train, y_test]],
}, index=["X_train", "X_test", "y_train", "y_test"])
print(tabela_conjuntos_originais)
print("\nConjuntos originais validados e copiados para referência.")

# %% [markdown]
# ### BLOCO 51 — Ajuste do StandardScaler
#
# `StandardScaler` já está importado no BLOCO 01. Um único scaler é instanciado nesta fase. As chamadas seguem a
# exigência do enunciado: `fit_transform` **somente** em `X_train` e `transform` em `X_test`. As saídas do scikit-learn
# são matrizes NumPy; elas são convertidas em DataFrames com os nomes, a ordem e os índices originais, para que cada
# registro padronizado continue rastreável até `df_features`. Os parâmetros são copiados logo após o ajuste, para
# confirmar que a transformação do teste não os altera.

# %%
# BLOCO 51 — Ajuste do StandardScaler

scaler = StandardScaler()

# Ajuste (média e desvio padrão) e transformação somente no treinamento
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

# Cópia dos parâmetros logo após o ajuste
parametros_apos_ajuste = {
    "mean_": scaler.mean_.copy(),
    "var_": scaler.var_.copy(),
    "scale_": scaler.scale_.copy(),
    "n_samples_seen_": int(scaler.n_samples_seen_),
}

# Transformação do teste com os parâmetros aprendidos no treinamento (sem novo ajuste)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

# Validações
for nome_parametro in ["mean_", "var_", "scale_"]:
    assert np.array_equal(getattr(scaler, nome_parametro), parametros_apos_ajuste[nome_parametro]), \
        f"Parâmetro {nome_parametro} alterado pela transformação do teste."
assert X_train_scaled.shape == (9933, 6) and X_test_scaled.shape == (2484, 6), "Dimensões padronizadas divergentes."
assert X_train_scaled.index.equals(X_train.index) and X_test_scaled.index.equals(X_test.index), "Índices alterados."
assert list(X_train_scaled.columns) == list(X_test_scaled.columns) == atributos_preditores, "Atributos divergentes."
assert (X_train_scaled.dtypes == "float64").all() and (X_test_scaled.dtypes == "float64").all(), "Tipo não float64."
assert np.isfinite(X_train_scaled.to_numpy()).all() and np.isfinite(X_test_scaled.to_numpy()).all(), \
    "Valores ausentes ou infinitos após a padronização."
pd.testing.assert_frame_equal(X_train, X_train_referencia, check_exact=True)
pd.testing.assert_frame_equal(X_test, X_test_referencia, check_exact=True)

print("X_train_scaled:", X_train_scaled.shape, "| X_test_scaled:", X_test_scaled.shape)
print("Observações usadas no ajuste (n_samples_seen_):", parametros_apos_ajuste["n_samples_seen_"])
print("\nPrimeiros registros do treinamento padronizado (índices originais):")
print(X_train_scaled.head(3).round(4))

# %% [markdown]
# ### BLOCO 52 — Parâmetros aprendidos pelo StandardScaler
#
# Os parâmetros do scaler são comparados com um cálculo independente feito com o pandas sobre `X_train` (`ddof=0`). Os
# dois algoritmos são matematicamente equivalentes, mas podem somar os valores em ordens diferentes; por isso, a
# comparação usa tolerância relativa `rtol = 1e-10` e `atol = 0`, e não igualdade binária. A tolerância absoluta é zero
# porque nenhum parâmetro é nulo e a variância de `HumidityRatio` é da ordem de 10⁻⁷: uma tolerância absoluta de
# `1e-12` equivaleria, para ela, a um critério relativo muito mais frouxo. Diferenças relativas da ordem de 10⁻¹⁶ são
# arredondamentos de ponto flutuante; uma diferença acima de 10⁻¹⁰ indicaria erro de cálculo, e não arredondamento.

# %%
# BLOCO 52 — Parâmetros aprendidos pelo StandardScaler

tabela_parametros = pd.DataFrame({
    "Média aprendida (mean_)": scaler.mean_,
    "Variância aprendida (var_)": scaler.var_,
    "Fator de escala (scale_)": scaler.scale_,
}, index=pd.Index(scaler.feature_names_in_, name="Atributo"))
print(tabela_parametros.to_string(float_format=lambda valor: f"{valor:.10g}"))

print("\nn_features_in_:", scaler.n_features_in_)
print("n_samples_seen_:", int(scaler.n_samples_seen_))
print("feature_names_in_:", list(scaler.feature_names_in_))

assert scaler.n_features_in_ == 6, "Quantidade de atributos do ajuste divergente."
assert int(scaler.n_samples_seen_) == len(X_train) == 9933, "Quantidade de observações do ajuste divergente."
assert list(scaler.feature_names_in_) == atributos_preditores, "Atributos ou ordem do ajuste divergentes."
assert np.isfinite(tabela_parametros.to_numpy()).all(), "Parâmetros ausentes ou infinitos."
assert (scaler.scale_ > 0).all(), "Fator de escala não positivo."

# Atributos constantes no treinamento (variância zero)
atributos_constantes = [atributo for atributo, variancia in zip(atributos_preditores, scaler.var_) if variancia == 0]
print("Atributos com variância zero no treinamento:", atributos_constantes if atributos_constantes else "nenhum")

# Cálculo independente, somente no treinamento (ddof=0)
media_independente = X_train.mean()
variancia_independente = X_train.var(ddof=0)
desvio_independente = X_train.std(ddof=0)

def diferenca_relativa(valor_scaler, valor_independente):
    return np.abs(np.asarray(valor_scaler) - np.asarray(valor_independente)) / np.abs(np.asarray(valor_independente))

tabela_comparacao_parametros = pd.DataFrame({
    "Média (pandas)": media_independente,
    "Dif. relativa da média": diferenca_relativa(scaler.mean_, media_independente),
    "Variância (pandas)": variancia_independente,
    "Dif. relativa da variância": diferenca_relativa(scaler.var_, variancia_independente),
    "Desvio padrão (pandas)": desvio_independente,
    "Dif. relativa do desvio": diferenca_relativa(scaler.scale_, desvio_independente),
})
tabela_comparacao_parametros.index.name = "Atributo"
print("\nComparação com o cálculo independente (pandas, ddof=0):")
print(tabela_comparacao_parametros.to_string(float_format=lambda valor: f"{valor:.6g}"))

tolerancia_parametros = 1e-10
maior_diferenca_parametros = tabela_comparacao_parametros.filter(like="Dif. relativa").to_numpy().max()
print(f"\nMaior diferença relativa: {maior_diferenca_parametros:.3e} (tolerância: {tolerancia_parametros:.0e})")
print("Parâmetros binariamente iguais ao cálculo independente:",
      int((scaler.mean_ == media_independente.to_numpy()).sum()), "médias,",
      int((scaler.var_ == variancia_independente.to_numpy()).sum()), "variâncias,",
      int((scaler.scale_ == desvio_independente.to_numpy()).sum()), "desvios (de 6)")

assert np.allclose(scaler.mean_, media_independente, rtol=tolerancia_parametros, atol=0), "Médias divergentes."
assert np.allclose(scaler.var_, variancia_independente, rtol=tolerancia_parametros, atol=0), "Variâncias divergentes."
assert np.allclose(scaler.scale_, desvio_independente, rtol=tolerancia_parametros, atol=0), "Desvios divergentes."
assert np.allclose(scaler.scale_, np.sqrt(scaler.var_), rtol=tolerancia_parametros, atol=0), "scale_ ≠ √var_."
print("Parâmetros do scaler conferidos com o cálculo independente.")

# %% [markdown]
# ### BLOCO 53 — Validação matemática da transformação
#
# A fórmula `(x − mean_) / scale_` é recalculada com o pandas, para o treino e para o teste, sempre com os parâmetros
# do treino, e comparada aos DataFrames padronizados. Os valores padronizados são adimensionais e da ordem de 1; a
# tolerância `rtol = atol = 1e-12` é, portanto, milhares de vezes maior que o arredondamento de um `float64` nessa faixa
# (cerca de 2 × 10⁻¹⁶) e ainda muito menor que qualquer erro de fórmula. Em seguida, o cálculo é mostrado passo a passo
# para `CO2` e `HumidityRatio` — os atributos de maior e de menor ordem de grandeza entre as medições dos sensores — nos
# dois primeiros registros reais de `X_train`.

# %%
# BLOCO 53 — Validação matemática da transformação

X_train_manual = (X_train - scaler.mean_) / scaler.scale_
X_test_manual = (X_test - scaler.mean_) / scaler.scale_

for nome_conjunto, manual, escalonado in [
    ("Treinamento", X_train_manual, X_train_scaled),
    ("Teste", X_test_manual, X_test_scaled),
]:
    assert manual.shape == escalonado.shape, "Dimensões divergentes."
    assert manual.index.equals(escalonado.index), "Índices divergentes."
    assert list(manual.columns) == list(escalonado.columns) == atributos_preditores, "Atributos divergentes."
    assert (manual.dtypes == "float64").all(), "Tipo não float64."
    assert np.isfinite(manual.to_numpy()).all(), "Valores ausentes ou infinitos."
    pd.testing.assert_frame_equal(manual, escalonado, check_exact=False, rtol=1e-12, atol=1e-12)
    diferenca_absoluta = (manual - escalonado).abs()
    print(f"{nome_conjunto}: {int((diferenca_absoluta == 0).sum().sum())} de {manual.size} valores idênticos; "
          f"maior diferença absoluta = {diferenca_absoluta.max().max():.3e}")

# Exemplos reais: dois primeiros registros do treinamento
linhas_exemplo = []
for indice in X_train.index[:2]:
    for atributo in ["CO2", "HumidityRatio"]:
        posicao = atributos_preditores.index(atributo)
        valor_original = X_train.loc[indice, atributo]
        resultado_manual = (valor_original - scaler.mean_[posicao]) / scaler.scale_[posicao]
        resultado_scaler = X_train_scaled.loc[indice, atributo]
        linhas_exemplo.append({
            "Índice": indice,
            "Data": rastreio_temporal.loc[indice, "date"],
            "Atributo": atributo,
            "Valor original": valor_original,
            "Média aprendida": scaler.mean_[posicao],
            "Fator de escala": scaler.scale_[posicao],
            "Resultado manual": resultado_manual,
            "StandardScaler": resultado_scaler,
            "Diferença": resultado_manual - resultado_scaler,
        })
tabela_exemplos_transformacao = pd.DataFrame(linhas_exemplo)
print("\nExemplos de cálculo com registros reais do treinamento:")
print(tabela_exemplos_transformacao.to_string(index=False, float_format=lambda valor: f"{valor:.10g}"))

del X_train_manual, X_test_manual

# %% [markdown]
# ### BLOCO 54 — Estatísticas após a padronização
#
# No treinamento, os atributos não constantes devem ter média 0 e desvio padrão populacional 1, a menos de
# arredondamentos de ponto flutuante (tolerância de `1e-10`). No teste, a média e o desvio padrão **não precisam** ser
# 0 e 1: o teste é transformado com os parâmetros do treino, e suas estatísticas refletem a diferença entre as duas
# amostras. O scaler não é reajustado para forçar esses valores, pois isso seria ajustar o scaler no teste.

# %%
# BLOCO 54 — Estatísticas após a padronização

tabela_estatisticas_padronizadas = pd.DataFrame({
    "Média treino original": X_train.mean(),
    "Desvio treino original": X_train.std(ddof=0),
    "Média treino escalonado": X_train_scaled.mean(),
    "Desvio treino escalonado": X_train_scaled.std(ddof=0),
    "Média teste escalonado": X_test_scaled.mean(),
    "Desvio teste escalonado": X_test_scaled.std(ddof=0),
})
tabela_estatisticas_padronizadas.index.name = "Atributo"
print(tabela_estatisticas_padronizadas.to_string(float_format=lambda valor: f"{valor:.6g}"))

atributos_nao_constantes = [atributo for atributo in atributos_preditores if atributo not in atributos_constantes]
media_abs_treino = X_train_scaled[atributos_nao_constantes].mean().abs()
desvio_unitario_treino = (X_train_scaled[atributos_nao_constantes].std(ddof=0) - 1).abs()
media_abs_teste = X_test_scaled.mean().abs()
desvio_unitario_teste = (X_test_scaled.std(ddof=0) - 1).abs()

tabela_desvios_numericos = pd.DataFrame({
    "Treinamento": [
        f"{media_abs_treino.min():.3e} ({media_abs_treino.idxmin()})",
        f"{media_abs_treino.max():.3e} ({media_abs_treino.idxmax()})",
        f"{desvio_unitario_treino.max():.3e} ({desvio_unitario_treino.idxmax()})",
    ],
    "Teste": [
        f"{media_abs_teste.min():.4f} ({media_abs_teste.idxmin()})",
        f"{media_abs_teste.max():.4f} ({media_abs_teste.idxmax()})",
        f"{desvio_unitario_teste.max():.4f} ({desvio_unitario_teste.idxmax()})",
    ],
}, index=["Menor |média|", "Maior |média| (desvio máximo em relação a 0)", "Maior |desvio padrão − 1|"])
print("\nDesvios em relação a média 0 e desvio padrão 1:")
print(tabela_desvios_numericos.to_string())

tolerancia_padronizacao = 1e-10
assert media_abs_treino.max() < tolerancia_padronizacao, "Média do treino padronizado diferente de 0."
assert desvio_unitario_treino.max() < tolerancia_padronizacao, "Desvio do treino padronizado diferente de 1."
print("\nTreinamento padronizado: média 0 e desvio padrão 1 (tolerância de 1e-10).")

# %% [markdown]
# ### BLOCO 55 — Validação da transformação inversa
#
# `inverse_transform` calcula `z · scale_ + mean_` e deve reconstruir os valores originais. A subtração, a divisão, a
# multiplicação e a soma introduzem arredondamentos legítimos, por isso a igualdade binária não é exigida. A tolerância
# de cada atributo é `1e-12` vezes o maior valor absoluto daquele atributo no treino: um critério relativo à magnitude da
# coluna, que acomoda poucos arredondamentos e ainda detectaria qualquer erro real. O erro relativo é calculado apenas
# para valores originais diferentes de zero (`Light` tem muitos zeros). Os DataFrames reconstruídos são temporários e
# descartados ao final; os dados originais não são arredondados nem substituídos.

# %%
# BLOCO 55 — Validação da transformação inversa

X_train_reconstruido = pd.DataFrame(
    scaler.inverse_transform(X_train_scaled), columns=X_train.columns, index=X_train.index
)
X_test_reconstruido = pd.DataFrame(
    scaler.inverse_transform(X_test_scaled), columns=X_test.columns, index=X_test.index
)

tolerancia_inversa = 1e-12 * X_train.abs().max()

def medir_inversa(original, reconstruido):
    erro_absoluto = (reconstruido - original).abs()
    valores_nao_nulos = original != 0
    erro_relativo = (erro_absoluto / original.abs()).where(valores_nao_nulos)
    return {
        "Valores": original.size,
        "Iguais exatamente": int((erro_absoluto == 0).sum().sum()),
        "Valores zero (sem erro relativo)": int((~valores_nao_nulos).sum().sum()),
        "Maior erro absoluto": erro_absoluto.max().max(),
        "Maior erro relativo": erro_relativo.max().max(),
        "Dentro da tolerância": bool(erro_absoluto.le(tolerancia_inversa, axis=1).all().all()),
    }, erro_absoluto.max(), erro_relativo.max()

resumo_inversa_treino, erro_por_atributo_treino, erro_relativo_por_atributo_treino = medir_inversa(
    X_train, X_train_reconstruido
)
resumo_inversa_teste, erro_por_atributo_teste, erro_relativo_por_atributo_teste = medir_inversa(
    X_test, X_test_reconstruido
)

def formatar_resumo_inversa(resumo):
    return {chave: (f"{valor:.3e}" if isinstance(valor, float) else valor) for chave, valor in resumo.items()}

tabela_inversa = pd.DataFrame({
    "Treinamento": formatar_resumo_inversa(resumo_inversa_treino),
    "Teste": formatar_resumo_inversa(resumo_inversa_teste),
})
print(tabela_inversa.to_string())

tabela_erro_inversa = pd.DataFrame({
    "Tolerância": tolerancia_inversa,
    "Maior erro absoluto (treino)": erro_por_atributo_treino,
    "Maior erro absoluto (teste)": erro_por_atributo_teste,
    "Maior erro relativo (treino)": erro_relativo_por_atributo_treino,
    "Maior erro relativo (teste)": erro_relativo_por_atributo_teste,
})
print("\nMaiores erros por atributo:")
print(tabela_erro_inversa.to_string(float_format=lambda valor: f"{valor:.3e}"))

assert resumo_inversa_treino["Dentro da tolerância"] and resumo_inversa_teste["Dentro da tolerância"], \
    "A transformação inversa não reconstruiu os dados originais."
assert X_train_reconstruido.index.equals(X_train.index) and X_test_reconstruido.index.equals(X_test.index)
print("\nTransformação inversa validada dentro da tolerância.")

del X_train_reconstruido, X_test_reconstruido

# %% [markdown]
# ### BLOCO 56 — Influência da escala nas distâncias
#
# Análise complementar, **sem treinar classificadores**, feita somente com `X_train` e sem usar `y_train`. São
# sorteados 1.000 pares de registros distintos do treino (gerador NumPy com semente 42). Para cada par, calculam-se a
# distância euclidiana antes e depois da padronização e a contribuição de cada atributo para o quadrado da distância:
#
# `contribuição_j = (x_aj − x_bj)² / Σ_k (x_ak − x_bk)² × 100`
#
# As contribuições de um par somam 100%. Pares com distância zero (medições idênticas nos seis atributos, possíveis
# pelas repetições identificadas no BLOCO 20.3) não têm contribuição definida e são excluídos **apenas** dessa média,
# sem nenhuma divisão por zero; nenhum registro é removido. As distâncias originais e padronizadas estão em escalas
# diferentes e não devem ser comparadas em valor absoluto: o objetivo é medir quais atributos determinam a distância.

# %%
# BLOCO 56 — Influência da escala nas distâncias

quantidade_pares = 1000
gerador_pares = np.random.default_rng(42)
posicao_a = gerador_pares.integers(0, len(X_train), size=quantidade_pares)
posicao_b = gerador_pares.integers(0, len(X_train) - 1, size=quantidade_pares)
posicao_b = posicao_b + (posicao_b >= posicao_a)  # garante dois registros distintos em cada par

assert (posicao_a != posicao_b).all(), "Par com o mesmo registro."
assert len(np.unique(np.sort(np.column_stack([posicao_a, posicao_b]), axis=1), axis=0)) == quantidade_pares, \
    "Pares repetidos."

def decompor_distancias(matriz):
    diferencas_quadraticas = (matriz[posicao_a] - matriz[posicao_b]) ** 2
    soma_quadrados = diferencas_quadraticas.sum(axis=1)
    validos = soma_quadrados > 0
    contribuicoes = diferencas_quadraticas[validos] / soma_quadrados[validos, None] * 100
    return np.sqrt(soma_quadrados), validos, contribuicoes

distancias_originais, pares_validos_originais, contribuicoes_originais = decompor_distancias(X_train.to_numpy())
distancias_padronizadas, pares_validos_padronizados, contribuicoes_padronizadas = decompor_distancias(
    X_train_scaled.to_numpy()
)

assert np.array_equal(pares_validos_originais, pares_validos_padronizados), "Pares válidos divergentes."
assert np.allclose(contribuicoes_originais.sum(axis=1), 100, rtol=0, atol=1e-9), "Contribuições não somam 100%."
assert np.allclose(contribuicoes_padronizadas.sum(axis=1), 100, rtol=0, atol=1e-9), "Contribuições não somam 100%."

pares_distancia_zero = int((~pares_validos_originais).sum())
pares_validos = int(pares_validos_originais.sum())
print("Pares sorteados:", quantidade_pares, "| com distância zero (excluídos das contribuições):",
      pares_distancia_zero, "| válidos:", pares_validos)
print("Maior desvio da soma das contribuições em relação a 100%:",
      f"{max(np.abs(contribuicoes_originais.sum(axis=1) - 100).max(), np.abs(contribuicoes_padronizadas.sum(axis=1) - 100).max()):.3e}")

tabela_distancias = pd.DataFrame({
    "Distância original (unidades mistas)": pd.Series(distancias_originais).describe(),
    "Distância padronizada": pd.Series(distancias_padronizadas).describe(),
})
print("\nEstatísticas das distâncias euclidianas dos pares:")
print(tabela_distancias.to_string(float_format=lambda valor: f"{valor:.4f}"))

tabela_contribuicoes = pd.DataFrame({
    "Antes (%)": contribuicoes_originais.mean(axis=0),
    "Depois (%)": contribuicoes_padronizadas.mean(axis=0),
    "Maior contribuição do par — antes": np.bincount(contribuicoes_originais.argmax(axis=1), minlength=6),
    "Maior contribuição do par — depois": np.bincount(contribuicoes_padronizadas.argmax(axis=1), minlength=6),
}, index=pd.Index(atributos_preditores, name="Atributo"))
tabela_contribuicoes["Variação (p.p.)"] = tabela_contribuicoes["Depois (%)"] - tabela_contribuicoes["Antes (%)"]
print("\nContribuição média de cada atributo para o quadrado da distância:")
print(tabela_contribuicoes.round(4).to_string())

contribuicao_dois_maiores_antes = tabela_contribuicoes["Antes (%)"].nlargest(2)
print(f"\nDois atributos com maior contribuição antes: {list(contribuicao_dois_maiores_antes.index)} "
      f"({contribuicao_dois_maiores_antes.sum():.2f}% somados)")

# %% [markdown]
# ### BLOCO 57 — Visualização da padronização
#
# A figura resume a fase em quatro painéis: (1) o desvio padrão original de cada atributo no treino, em escala
# logarítmica apenas no eixo, pois as unidades e as ordens de grandeza são diferentes; (2) os boxplots dos atributos
# padronizados do treino, com os outliers preservados; (3) a contribuição média de cada atributo para a distância
# euclidiana, antes e depois da padronização (pares do BLOCO 56); e (4) a média e o desvio padrão populacional dos
# conjuntos padronizados de treino e teste.

# %%
# BLOCO 57 — Visualização da padronização

cor_antes, cor_depois = "#4C72B0", "#DD8452"
unidades_curtas = {"Temperature": "°C", "Humidity": "%", "Light": "lux", "CO2": "ppm",
                   "HumidityRatio": "kg/kg", "CO2_Light_Ratio": "índice"}
posicoes_atributos = np.arange(len(atributos_preditores))

def formatar_decimal(valor, casas=2):
    return f"{valor:.{casas}f}".replace(".", ",")

fig, eixos = plt.subplots(2, 2, figsize=(18, 13))
eixo_desvio, eixo_box, eixo_contribuicao, eixo_estatisticas = eixos.ravel()

# Painel 1 — Desvio padrão original (eixo logarítmico)
eixo_desvio.barh(posicoes_atributos, desvios_originais.loc[atributos_preditores], color=cor_antes, edgecolor="black")
for posicao, atributo in enumerate(atributos_preditores):
    valor = desvios_originais[atributo]
    texto_valor = f"{valor:.2e}".replace(".", ",") if valor < 0.01 else formatar_decimal(valor)
    eixo_desvio.text(valor * 1.25, posicao, f"{texto_valor} {unidades_curtas[atributo]}", va="center", fontsize=10)
eixo_desvio.set_xscale("log")
eixo_desvio.set_xlim(desvios_originais.min() / 3, desvios_originais.max() * 40)
eixo_desvio.set_yticks(posicoes_atributos, atributos_preditores)
eixo_desvio.invert_yaxis()
eixo_desvio.set_title("Desvio padrão original no treino (unidades diferentes)")
eixo_desvio.set_xlabel("Desvio padrão populacional (eixo em escala logarítmica)")

# Painel 2 — Boxplots dos atributos padronizados do treino
eixo_box.boxplot(
    [X_train_scaled[atributo] for atributo in atributos_preditores],
    orientation="horizontal",
    tick_labels=atributos_preditores,
    patch_artist=True,
    widths=0.6,
    boxprops={"facecolor": "#8DA0CB", "edgecolor": "black"},
    medianprops={"color": "black", "linewidth": 1.5},
    flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "#4C72B0", "markeredgecolor": "#4C72B0", "alpha": 0.5},
)
eixo_box.invert_yaxis()
eixo_box.axvline(0, color="#C44E52", linestyle="--", linewidth=1.5, label="Média do treino (0)")
eixo_box.set_title("Atributos padronizados no treino")
eixo_box.set_xlabel("Valor padronizado (desvios padrão do treino em relação à média)")
eixo_box.legend(loc="lower right")

# Painel 3 — Contribuição média para o quadrado da distância
altura_barra = 0.38
for deslocamento, coluna, cor, rotulo in [(-altura_barra / 2, "Antes (%)", cor_antes, "Dados originais"),
                                          (altura_barra / 2, "Depois (%)", cor_depois, "Dados padronizados")]:
    valores = tabela_contribuicoes[coluna]
    eixo_contribuicao.barh(posicoes_atributos + deslocamento, valores, height=altura_barra, color=cor,
                           edgecolor="black", label=rotulo)
    for posicao, valor in enumerate(valores):
        eixo_contribuicao.text(valor + 0.8, posicao + deslocamento, f"{formatar_decimal(valor)}%", va="center",
                               fontsize=9)
eixo_contribuicao.axvline(100 / 6, color="gray", linestyle=":", linewidth=1.5, label="Participação igual (16,67%)")
eixo_contribuicao.set_yticks(posicoes_atributos, atributos_preditores)
eixo_contribuicao.invert_yaxis()
eixo_contribuicao.set_xlim(0, tabela_contribuicoes[["Antes (%)", "Depois (%)"]].to_numpy().max() * 1.2)
eixo_contribuicao.set_title(
    f"Contribuição média para a distância euclidiana ({formatar_inteiro(pares_validos)} pares do treino)"
)
eixo_contribuicao.set_xlabel("Participação no quadrado da distância (%)")
eixo_contribuicao.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

# Painel 4 — Média e desvio padrão após a padronização (treino e teste)
estatisticas_painel = [
    ("Média treino escalonado", "o", cor_antes, -0.15, "Média — treino"),
    ("Média teste escalonado", "o", cor_depois, 0.15, "Média — teste"),
    ("Desvio treino escalonado", "s", cor_antes, -0.15, "Desvio padrão — treino"),
    ("Desvio teste escalonado", "s", cor_depois, 0.15, "Desvio padrão — teste"),
]
for coluna, marcador, cor, deslocamento, rotulo in estatisticas_painel:
    preenchimento = cor if marcador == "o" else "white"
    eixo_estatisticas.scatter(tabela_estatisticas_padronizadas[coluna], posicoes_atributos + deslocamento,
                              marker=marcador, s=70, color=preenchimento, edgecolor=cor, linewidth=2, label=rotulo,
                              zorder=3)
eixo_estatisticas.axvline(0, color="gray", linestyle="--", linewidth=1)
eixo_estatisticas.axvline(1, color="gray", linestyle="--", linewidth=1)
eixo_estatisticas.set_yticks(posicoes_atributos, atributos_preditores)
eixo_estatisticas.invert_yaxis()
eixo_estatisticas.set_xlim(-0.25, 1.25)
texto_desvios = (
    f"Treino: |média| ≤ {media_abs_treino.max():.1e} e |desvio − 1| ≤ {desvio_unitario_treino.max():.1e}".replace(".", ",")
    + f"\nTeste: |média| ≤ {formatar_decimal(media_abs_teste.max(), 4)} e "
    + f"|desvio − 1| ≤ {formatar_decimal(desvio_unitario_teste.max(), 4)}"
)
eixo_estatisticas.text(
    0.5, 0.5, texto_desvios,
    transform=eixo_estatisticas.transAxes, ha="center", va="center", fontsize=10,
    bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray"},
)
eixo_estatisticas.set_title("Média e desvio padrão (ddof=0) após a padronização")
eixo_estatisticas.set_xlabel("Valor (referências: média 0 e desvio padrão 1)")
eixo_estatisticas.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

plt.suptitle("Padronização com StandardScaler (ajuste somente no treinamento: 9.933 registros)", fontsize=16)
plt.tight_layout()

# Exportação antes da exibição
plt.savefig(caminho_grafico_scaler, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_scaler)

plt.show()

# %% [markdown]
# ### BLOCO 58 — Validação contra Data Leakage
#
# A quantidade de observações do ajuste, sozinha, não prova a ausência de vazamento: um ajuste com 9.933 registros
# quaisquer teria o mesmo `n_samples_seen_`. Por isso, a verificação combina três evidências:
#
# 1. **Fluxo das chamadas no código:** `fit_transform` aparece uma única vez, no BLOCO 51, aplicado a `X_train`; `X_test`
#    recebe somente `transform` (BLOCO 51 e a reaplicação abaixo) e `inverse_transform` (BLOCO 55), que não aprendem
#    parâmetros. Não há `fit` em `X_test`, em `X` nem em treino e teste concatenados.
# 2. **Parâmetros:** a média e o desvio padrão aprendidos coincidem com os de `X_train` (BLOCO 52), diferem dos da base
#    completa e dos do teste — calculados aqui com o pandas, sem ajustar nenhum scaler — e não mudaram desde o ajuste.
# 3. **Dados:** os conjuntos originais, os alvos e os índices continuam idênticos às referências do BLOCO 50.

# %%
# BLOCO 58 — Validação contra Data Leakage

# Parâmetros inalterados desde o ajuste (após o transform do teste e todas as validações)
for nome_parametro in ["mean_", "var_", "scale_"]:
    assert np.array_equal(getattr(scaler, nome_parametro), parametros_apos_ajuste[nome_parametro]), \
        f"Parâmetro {nome_parametro} alterado após o ajuste."
assert int(scaler.n_samples_seen_) == parametros_apos_ajuste["n_samples_seen_"] == len(X_train) == 9933

# Médias e desvios do treino, da base completa e do teste (pandas, sem ajustar scaler)
tabela_origem_parametros = pd.DataFrame({
    "Média do scaler": scaler.mean_,
    "Média do treino": X_train.mean().to_numpy(),
    "Média da base completa": X.mean().to_numpy(),
    "Média do teste": X_test.mean().to_numpy(),
    "Desvio do scaler": scaler.scale_,
    "Desvio da base completa": X.std(ddof=0).to_numpy(),
    "Desvio do teste": X_test.std(ddof=0).to_numpy(),
}, index=pd.Index(atributos_preditores, name="Atributo"))
print(tabela_origem_parametros.to_string(float_format=lambda valor: f"{valor:.6g}"))

assert np.allclose(scaler.mean_, X_train.mean(), rtol=tolerancia_parametros, atol=0), "Média diferente do treino."
assert np.allclose(scaler.scale_, X_train.std(ddof=0), rtol=tolerancia_parametros, atol=0), "Desvio diferente do treino."
assert not np.allclose(scaler.mean_, X.mean(), rtol=tolerancia_parametros, atol=0), "Média igual à da base completa."
assert not np.allclose(scaler.mean_, X_test.mean(), rtol=tolerancia_parametros, atol=0), "Média igual à do teste."

# Reaplicação do transform (sem ajuste): o teste padronizado é reproduzido exatamente com os parâmetros do treino
pd.testing.assert_frame_equal(
    pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index), X_test_scaled, check_exact=True
)

# Conjuntos originais, alvos e índices preservados
pd.testing.assert_frame_equal(X_train, X_train_referencia, check_exact=True)
pd.testing.assert_frame_equal(X_test, X_test_referencia, check_exact=True)
pd.testing.assert_series_equal(y_train, y_train_referencia, check_exact=True)
pd.testing.assert_series_equal(y_test, y_test_referencia, check_exact=True)
pd.testing.assert_frame_equal(X, X_referencia, check_exact=True)
pd.testing.assert_series_equal(y, y_referencia, check_exact=True)
pd.testing.assert_frame_equal(rastreio_temporal, rastreio_temporal_referencia, check_exact=True)
pd.testing.assert_frame_equal(df_features, df_features_referencia, check_exact=True)
assert X_train_scaled.index.equals(y_train.index) and X_test_scaled.index.equals(y_test.index), "Índices divergentes."
assert set(y_train.unique()) == set(y_test.unique()) == {0, 1} and y_train.dtype == y_test.dtype == "int64", \
    "Variável alvo alterada."

tabela_validacao_leakage = pd.Series({
    "Observações do ajuste (n_samples_seen_)": int(scaler.n_samples_seen_),
    "Parâmetros iguais às estatísticas do treino": True,
    "Parâmetros diferentes da base completa": True,
    "Parâmetros diferentes do teste": True,
    "Parâmetros inalterados após o transform do teste": True,
    "Teste reproduzido apenas com transform": True,
    "X_train, X_test, X, rastreio_temporal e df_features inalterados": True,
    "y_train, y_test e y inalterados (não escalonados)": True,
})
print("\nVerificações contra Data Leakage:")
print(tabela_validacao_leakage.to_string())

# %% [markdown]
# ### BLOCO 59 — Resumo consolidado da Fase 5
#
# Os conjuntos permanecem em memória e são reproduzidos pela execução sequencial do script. Nenhum conjunto padronizado
# é exportado em CSV, e o objeto `scaler` não é salvo em arquivo (`pickle` ou `joblib`).
#
# **Regras para a Fase 6:**
#
# - **KNN:** usar um `Pipeline` com `StandardScaler` e `KNeighborsClassifier`, que recebe `X_train` **original**. Na
#   validação cruzada, o scaler interno é reajustado na parte de treino de cada divisão (*fold*), sem ver a parte de
#   validação. Por isso, `X_train_scaled` e `X_test_scaled` **não** devem ser fornecidos a um `Pipeline` que já tenha
#   `StandardScaler` (escalonamento duplicado), e o `scaler` desta fase não deve ser reaproveitado nos *folds*. O valor
#   de K será escolhido dentro do treino, antes de consultar o teste.
# - **Árvore de Decisão:** usar `X_train` e `X_test` originais, sem escalonamento; os valores de `max_depth` exigidos
#   pelo enunciado serão avaliados com a seleção feita dentro do treino.
# - **Atributos:** a comparação com e sem `CO2_Light_Ratio` será feita dentro do treino, sem usar o teste para escolher
#   a combinação.
# - **Dependência temporal:** o split oficial é preservado; a validação cruzada aleatória não equivale a independência
#   temporal, e uma validação temporal complementar será planejada.

# %%
# BLOCO 59 — Resumo consolidado da Fase 5

tabela_conjuntos_fase5 = pd.DataFrame({
    "Dimensões": [str(conjunto.shape) for conjunto in
                  [X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test]],
    "Escalonamento": ["Não", "Não", "Sim", "Sim", "Não", "Não"],
    "Finalidade": [
        "Árvore de Decisão e entrada do Pipeline do KNN",
        "Árvore de Decisão e entrada do Pipeline do KNN",
        "Demonstração do escalonamento e KNN direto, quando pertinente",
        "Demonstração do escalonamento e KNN direto, quando pertinente",
        "Alvo de treinamento",
        "Alvo de avaliação",
    ],
}, index=pd.Index(["X_train", "X_test", "X_train_scaled", "X_test_scaled", "y_train", "y_test"], name="Estrutura"))
print(tabela_conjuntos_fase5.to_string())

tabela_resumo_fase5 = pd.Series({
    "Observações usadas no ajuste": int(scaler.n_samples_seen_),
    "Maior |média| do treino padronizado": f"{media_abs_treino.max():.3e}",
    "Maior |desvio − 1| do treino padronizado": f"{desvio_unitario_treino.max():.3e}",
    "Maior |média| do teste padronizado": f"{media_abs_teste.max():.4f}",
    "Maior |desvio − 1| do teste padronizado": f"{desvio_unitario_teste.max():.4f}",
    "Maior erro absoluto da inversa (treino / teste)":
        f"{resumo_inversa_treino['Maior erro absoluto']:.3e} / {resumo_inversa_teste['Maior erro absoluto']:.3e}",
    "Pares válidos na análise de distâncias": pares_validos,
    "Conjuntos originais preservados": X_train.equals(X_train_referencia) and X_test.equals(X_test_referencia),
})
print("\nResumo da Fase 5:")
print(tabela_resumo_fase5.to_string())

# %% [markdown]
# ## BLOCO 60 — Interpretação e conclusão do escalonamento
#
# ### 60.1 — Diferenças de escala dos atributos
#
# | Atributo (treino, 9.933 registros) | Unidade | Mínimo | Máximo | Média | Desvio padrão (ddof=0) |
# |---|---|---:|---:|---:|---:|
# | Temperature | °C | 19,50 | 24,41 | 21,09 | 1,04 |
# | Humidity | % | 21,87 | 39,50 | 28,92 | 4,13 |
# | Light | lux | 0,00 | 1.581,00 | 137,87 | 219,17 |
# | CO2 | ppm | 427,50 | 2.076,50 | 746,50 | 297,49 |
# | HumidityRatio | kg/kg | 0,00327 | 0,00577 | 0,00447 | 0,000595 |
# | CO2_Light_Ratio | índice | 0,56 | 2.076,50 | 426,77 | 404,49 |
#
# Numericamente, o maior desvio padrão (`CO2_Light_Ratio`, 404,49) é cerca de 679 mil vezes o menor
# (`HumidityRatio`, 0,000595). Essa razão não compara grandezas físicas — as unidades são diferentes — e não indica
# importância preditiva; ela mostra apenas quanto cada atributo varia como número. Como a distância euclidiana soma
# os quadrados das diferenças como números puros, os atributos de maior dispersão numérica determinam a distância, e
# os de menor dispersão praticamente não participam, independentemente da informação que carregam.
#
# ### 60.2 — Funcionamento matemático do StandardScaler
#
# O scaler foi ajustado com as 9.933 observações de `X_train` e aprendeu, para cada atributo, a média (`mean_`), a
# variância populacional (`var_`, divisor `n`) e o fator de escala (`scale_`), que é a raiz quadrada da variância — o
# desvio padrão populacional. A variância está na unidade ao quadrado (por exemplo, 88.497,51 ppm² em `CO2`), e o
# desvio padrão, na unidade original (297,49 ppm); por isso a divisão usa o desvio padrão. Nenhum atributo tem
# variância zero.
#
# Os parâmetros foram comparados com um cálculo independente do pandas (`ddof=0`): as 18 quantidades (6 médias, 6
# variâncias e 6 desvios) foram **binariamente iguais** nas versões testadas, bem abaixo da tolerância relativa de
# `1e-10`. Exemplos reais do treinamento:
#
# | Registro | Atributo | Valor original | Média aprendida | Fator de escala | Valor padronizado |
# |---|---|---:|---:|---:|---:|
# | 10164 (16/02/2015 19:46:59) | CO2 | 760,6666667 | 746,49934 | 297,4853061 | 0,0476236 |
# | 10164 | HumidityRatio | 0,0045977054 | 0,0044683535 | 0,0005953432 | 0,2172728 |
# | 986 (03/02/2015 06:45:00) | CO2 | 436,8 | 746,49934 | 297,4853061 | −1,0410576 |
# | 986 | HumidityRatio | 0,0033523233 | 0,0044683535 | 0,0005953432 | −1,8745996 |
#
# O cálculo manual coincidiu exatamente com o `StandardScaler` nos quatro casos (diferença 0), e a fórmula recalculada
# para todos os valores reproduziu os 59.598 valores do treino e os 14.904 do teste sem nenhuma diferença. O registro
# 986 tem `CO2` 1,04 desvio padrão abaixo da média e `HumidityRatio` 1,87 abaixo: depois da padronização, as duas
# medições, antes separadas por seis ordens de grandeza, passam a ser expressas na mesma escala.
#
# ### 60.3 — Resultados do treinamento padronizado
#
# Nos seis atributos, a média do treino padronizado ficou entre 3,0 × 10⁻¹⁷ (`Light`) e 2,5 × 10⁻¹⁵ em valor absoluto
# (`Temperature`), e o desvio padrão populacional diferiu de 1 em no máximo 2,2 × 10⁻¹⁶ (`CO2`). Esses resíduos não são
# zero porque a média é calculada com somas de números `float64`, cuja precisão relativa é de cerca de 2,2 × 10⁻¹⁶;
# somar quase 10 mil valores acumula erros dessa ordem. Eles estão muito abaixo da tolerância de `1e-10` e confirmam
# média 0 e desvio padrão 1, como esperado para o conjunto em que o scaler foi ajustado.
#
# ### 60.4 — Resultados do teste padronizado
#
# | Atributo | Média do teste padronizado | Desvio padrão do teste padronizado |
# |---|---:|---:|
# | Temperature | 0,0001 | 1,0027 |
# | Humidity | −0,0014 | 0,9988 |
# | Light | 0,0059 | 1,0141 |
# | CO2 | −0,0144 | 0,9830 |
# | HumidityRatio | −0,0012 | 1,0011 |
# | CO2_Light_Ratio | −0,0079 | 0,9830 |
#
# O teste foi transformado com a média e o desvio padrão do **treino**; por isso, suas estatísticas mostram quanto a
# amostra de teste difere da de treino, em desvios padrão do treino. A maior diferença de média ocorre em `CO2`
# (−0,0144, isto é, a média do teste, 742,23 ppm, está 4,27 ppm abaixo da do treino), e a maior diferença de desvio,
# em `CO2_Light_Ratio` (0,9830, desvio 1,70% menor). As diferenças são pequenas, o que é coerente com a divisão
# aleatória estratificada de uma mesma base. Forçar média 0 e desvio 1 no teste exigiria ajustar o scaler no teste, o
# que seria vazamento de dados.
#
# ### 60.5 — Influência nas distâncias
#
# | Atributo | Contribuição média antes | Contribuição média depois | Pares em que domina (antes → depois) |
# |---|---:|---:|---:|
# | Temperature | 0,02% | 16,63% | 0 → 165 |
# | Humidity | 1,00% | 20,76% | 10 → 235 |
# | Light | 15,95% | 13,63% | 109 → 192 |
# | CO2 | 34,31% | 14,27% | 509 → 153 |
# | HumidityRatio | < 0,0001% | 20,08% | 0 → 154 |
# | CO2_Light_Ratio | 48,72% | 14,64% | 372 → 101 |
#
# Nos 1.000 pares de treino sorteados (nenhum com distância zero, portanto nenhum excluído das contribuições; as
# contribuições de cada par somaram 100% com desvio máximo de 4,3 × 10⁻¹⁴), `CO2_Light_Ratio` e `CO2` respondiam
# juntos por 83,03% do quadrado da distância antes da padronização, e `CO2` ou `CO2_Light_Ratio` era o maior termo em
# 881 pares. `Temperature` (0,02%) e `HumidityRatio` (menos de 0,0001%) praticamente não participavam: na prática, um
# KNN sem escalonamento ignoraria esses atributos. Depois da padronização, as contribuições médias ficaram entre
# 13,63% e 20,76%, e todos os atributos passaram a dominar uma parte dos pares.
#
# As contribuições não ficaram iguais (16,67% cada): a padronização iguala a variância, e não a importância. `Humidity`
# e `HumidityRatio`, fortemente correlacionadas (0,89 na EDA), somam 40,84% das distâncias padronizadas, e a umidade
# passa a ter um peso conjunto maior. As distâncias absolutas (média de 603,13 em unidades mistas e de 3,15 em desvios
# padrão) pertencem a escalas diferentes e não são comparáveis. Esta análise não mede desempenho: nenhum KNN foi
# treinado, e o efeito na acurácia será avaliado na Fase 6.
#
# ### 60.6 — Prevenção de Data Leakage
#
# A divisão da Fase 4 foi feita antes do ajuste, e o scaler aprendeu seus parâmetros somente em `X_train`. As
# verificações do BLOCO 58 combinaram a inspeção do fluxo das chamadas (`fit_transform` apenas em `X_train`; somente
# `transform` e `inverse_transform` em `X_test`), a coincidência dos parâmetros com as estatísticas do treino, a
# diferença em relação à base completa (média de `CO2`: 746,50 ppm no scaler e 745,64 ppm na base) e ao teste
# (742,23 ppm), a estabilidade dos parâmetros após o `transform` do teste e a preservação dos conjuntos originais.
#
# Este ajuste é uma **demonstração** da exigência do enunciado. Na validação cruzada da Fase 6, o `StandardScaler`
# estará dentro de um `Pipeline` e será reajustado na parte de treino de cada divisão, sem ver a parte de validação;
# o scaler desta fase não será reaproveitado nos *folds*. Os riscos registrados anteriormente continuam: a EDA e as
# decisões das Fases 2 e 3 usaram a base completa, e a dependência temporal entre treino e teste (96,10% dos registros
# de teste com vizinho imediato no treino) permanece.
#
# ### 60.7 — KNN versus Árvore de Decisão
#
# | Aspecto | KNN | Árvore de Decisão |
# |---|---|---|
# | Como decide | Pelos vizinhos mais próximos, segundo uma distância que soma as diferenças de todos os atributos | Por perguntas sucessivas sobre um atributo de cada vez (`atributo ≤ limiar`) |
# | Efeito da escala | Decisivo: sem padronização, `CO2_Light_Ratio` e `CO2` responderam por 83,03% das distâncias | Nenhum: a divisão depende só da ordem dos valores, que a padronização preserva |
# | Dados na Fase 6 | `Pipeline` com `StandardScaler` e `KNeighborsClassifier`, recebendo `X_train` original | `X_train` e `X_test` originais, sem escalonamento |
#
# A árvore compara `Light` apenas com limiares de `Light` e `Temperature` apenas com limiares de `Temperature`; a
# diferença entre 0 a 1.581 lux e 19,50 a 24,41 °C não interfere, pois os atributos nunca são somados.
#
# ### 60.8 — Limitações do StandardScaler
#
# - **Valores extremos:** a média e o desvio padrão são sensíveis a outliers. `Light` chega a 6,58 desvios padrão
#   acima da média e `CO2` a 4,47, e esses valores continuam extremos após a padronização (painel 2 da figura).
# - **Forma e correlações:** a transformação é linear; não torna as distribuições normais, não remove a assimetria e
#   preserva as correlações entre os atributos.
# - **Redundância:** atributos correlacionados continuam somando informação semelhante na distância (60.5).
# - **Classes:** o escalonamento não altera `y`; o desequilíbrio de 24,33% de ocupados permanece.
# - **Desempenho:** a padronização equilibra a escala, mas não garante melhor classificação; o ganho será medido.
# - **Dependência temporal:** o escalonamento não altera a proximidade temporal entre treino e teste (Fase 4).
#
# ### 60.9 — Integridade e próximos passos
#
# `X_train`, `X_test`, `y_train`, `y_test`, `X`, `y`, `rastreio_temporal` e `df_features` permaneceram idênticos às
# referências (`check_exact=True`). `X_train_scaled` (9.933 × 6) e `X_test_scaled` (2.484 × 6) mantêm os índices, a
# ordem e o tipo `float64` dos atributos, sem ausentes nem infinitos. A transformação inversa reconstruiu 93,01% dos
# valores do treino e 93,06% dos do teste de forma binariamente idêntica; nos demais, o maior erro absoluto foi de
# 4,5 × 10⁻¹³ e o maior erro relativo, de 1,5 × 10⁻¹³, ambos em `CO2_Light_Ratio`, cujos menores valores (0,56)
# convivem com média de 426,77 — diferenças de arredondamento, dentro da tolerância relativa à magnitude de cada
# atributo.
#
# Os conjuntos estão prontos para a Fase 6, na qual serão treinados o KNN (com `Pipeline`) e a Árvore de Decisão,
# seguindo as regras do BLOCO 59. Nenhum classificador foi instanciado ou treinado nesta fase.
