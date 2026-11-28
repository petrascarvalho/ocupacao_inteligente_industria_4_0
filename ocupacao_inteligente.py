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
# | Base consolidada | 12.417 registros e 7 colunas, com a precisão original das fontes |
# | Base tratada e base de Feature Engineering | 12.417 registros, com precisão decimal padronizada na Fase 2 (BLOCO 19.1) |
# | Atribuição e hashes SHA-256 | `dados/README.md` |
#
# ### Revisão metodológica da precisão decimal
#
# Após a conclusão da primeira versão das sete fases, o estudante solicitou a padronização da quantidade de casas
# decimais dos atributos (`Temperature` 1, `Humidity` 2, `Light` 1, `CO2` 1 e `HumidityRatio` 5). A regra é uma decisão
# metodológica deste projeto, e não uma exigência do enunciado. Ela foi incorporada à Fase 2, e todas as fases seguintes
# foram reprocessadas. Como o teste oficial e o `datatraining.txt` já haviam sido avaliados na versão anterior, a
# comparação entre as versões (BLOCOS 103 a 108) é uma análise de sensibilidade posterior, e não uma avaliação
# independente. Os resultados da versão anterior estão preservados em
# `resultados/referencia_historica/resultados_versao_precisao_integral.csv`.

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
# | Padrão do Python | `os`, `zipfile`, `urllib.request`, `hashlib`, `time`, `decimal` (análise dos empates de arredondamento, Fase 2) |
# | Análise e manipulação | `pandas`, `numpy` |
# | Visualização | `matplotlib.pyplot`, `matplotlib.dates`, `seaborn` |
# | Machine Learning | `train_test_split`, `StratifiedKFold`, `TimeSeriesSplit`, `cross_validate`, `StandardScaler`, `Pipeline`, `DecisionTreeClassifier`, `export_text`, `KNeighborsClassifier`, `DummyClassifier`, métricas, `check_is_fitted` |
# | Estatística | `scipy.stats.binomtest` (teste exato de McNemar, Fase 7) |
#
# Versões no ambiente `.venv`: Python 3.14.4, pandas 3.0.2, NumPy 2.4.4, Matplotlib 3.10.9, Seaborn 0.13.2, scikit-learn 1.8.0.

# %%
# BLOCO 01 — Importação das bibliotecas

# Bibliotecas padrão do Python
import os
import zipfile
import urllib.request
import hashlib
import time
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP

# Análise e manipulação de dados
import pandas as pd
import numpy as np

# Visualização de dados
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.utils.validation import check_is_fitted

# Estatística
from scipy.stats import binomtest


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
# | `caminho_grafico_hiperparametros` | `resultados/graficos/modelagem_01_comparacao_hiperparametros.png` (Fase 6) |
# | `caminho_grafico_feature_engineering` | `resultados/graficos/modelagem_02_impacto_feature_engineering.png` (Fase 6) |
# | `caminho_grafico_validacao_temporal` | `resultados/graficos/modelagem_03_validacao_temporal.png` (Fase 6) |
# | `caminho_cv_hiperparametros` / `caminho_cv_feature_engineering` / `caminho_cv_temporal` | `resultados/modelagem/cv_hiperparametros.csv`, `cv_feature_engineering.csv` e `cv_temporal.csv` (Fase 6) |
# | `caminho_grafico_matrizes` / `caminho_grafico_metricas` / `caminho_grafico_erros` / `caminho_grafico_generalizacao` | `resultados/graficos/avaliacao_01_matrizes_confusao.png` a `avaliacao_04_generalizacao_modelos.png` (Fase 7) |
# | `caminho_metricas_avaliacao_final` / `caminho_previsoes_teste` / `caminho_analise_erros_teste` | `resultados/modelagem/metricas_avaliacao_final.csv`, `previsoes_teste.csv` e `analise_erros_teste.csv` (Fase 7) |
# | `caminho_metricas_avaliacao_complementar` / `caminho_previsoes_avaliacao_complementar` | `resultados/modelagem/metricas_avaliacao_complementar.csv` e `previsoes_avaliacao_complementar.csv` (Fase 7) |
# | `caminho_grafico_precisao` / `caminho_auditoria_precisao` | `resultados/graficos/prep_02_padronizacao_precisao_decimal.png` e `resultados/modelagem/auditoria_precisao_decimal.csv` (Fase 2) |
# | `caminho_referencia_historica` | `resultados/referencia_historica/resultados_versao_precisao_integral.csv` (somente leitura; resultados preservados da versão anterior à regra decimal) |
# | `caminho_grafico_revisao` / `caminho_comparacao_precisao` | `resultados/graficos/revisao_01_impacto_precisao_decimal.png` e `resultados/modelagem/comparacao_precisao_decimal.csv` (revisão, BLOCOS 103 a 108) |
# | `pasta_graficos` / `pasta_metricas` / `pasta_modelagem` | `resultados/graficos` / `resultados/metricas` / `resultados/modelagem` |
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
pasta_modelagem = "resultados/modelagem"

# Criando os diretórios, caso não existam
os.makedirs(pasta_brutos, exist_ok=True)
os.makedirs(pasta_consolidados, exist_ok=True)
os.makedirs(pasta_tratados, exist_ok=True)
os.makedirs(pasta_features, exist_ok=True)
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_metricas, exist_ok=True)
os.makedirs(pasta_modelagem, exist_ok=True)

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

# Definindo os caminhos dos gráficos e dos resultados tabulares da modelagem (Fase 6)
caminho_grafico_hiperparametros = f"{pasta_graficos}/modelagem_01_comparacao_hiperparametros.png"
caminho_grafico_feature_engineering = f"{pasta_graficos}/modelagem_02_impacto_feature_engineering.png"
caminho_grafico_validacao_temporal = f"{pasta_graficos}/modelagem_03_validacao_temporal.png"
caminho_cv_hiperparametros = f"{pasta_modelagem}/cv_hiperparametros.csv"
caminho_cv_feature_engineering = f"{pasta_modelagem}/cv_feature_engineering.csv"
caminho_cv_temporal = f"{pasta_modelagem}/cv_temporal.csv"

# Definindo os caminhos dos gráficos e dos resultados da avaliação final (Fase 7)
caminho_grafico_matrizes = f"{pasta_graficos}/avaliacao_01_matrizes_confusao.png"
caminho_grafico_metricas = f"{pasta_graficos}/avaliacao_02_comparacao_metricas.png"
caminho_grafico_erros = f"{pasta_graficos}/avaliacao_03_analise_erros.png"
caminho_grafico_generalizacao = f"{pasta_graficos}/avaliacao_04_generalizacao_modelos.png"
caminho_metricas_avaliacao_final = f"{pasta_modelagem}/metricas_avaliacao_final.csv"
caminho_previsoes_teste = f"{pasta_modelagem}/previsoes_teste.csv"
caminho_analise_erros_teste = f"{pasta_modelagem}/analise_erros_teste.csv"
caminho_metricas_avaliacao_complementar = f"{pasta_modelagem}/metricas_avaliacao_complementar.csv"
caminho_previsoes_avaliacao_complementar = f"{pasta_modelagem}/previsoes_avaliacao_complementar.csv"

# Definindo os caminhos da padronização da precisão decimal (Fase 2) e da revisão metodológica
caminho_grafico_precisao = f"{pasta_graficos}/prep_02_padronizacao_precisao_decimal.png"
caminho_auditoria_precisao = f"{pasta_modelagem}/auditoria_precisao_decimal.csv"
caminho_referencia_historica = "resultados/referencia_historica/resultados_versao_precisao_integral.csv"
caminho_grafico_revisao = f"{pasta_graficos}/revisao_01_impacto_precisao_decimal.png"
caminho_comparacao_precisao = f"{pasta_modelagem}/comparacao_precisao_decimal.csv"

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
print("Referência histórica da versão anterior:", os.path.exists(caminho_referencia_historica))


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
# **Camada analisada:** todas as estatísticas, gráficos e correlações desta fase descrevem a **base consolidada**, com a
# precisão original das fontes. Elas foram preservadas como diagnóstico dessa camada. A padronização decimal ocorre na
# Fase 2 (BLOCO 19.1), e a comparação entre a base consolidada e a base tratada está no BLOCO 25.4: as médias, os
# desvios padrão e as correlações mudam de forma residual (por exemplo, a correlação de `Light` com `Occupancy`
# permanece 0,9197), de modo que as interpretações abaixo continuam válidas para a base tratada.
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
# | Quantidades muito diferentes de casas decimais entre os atributos | Padronizar a precisão decimal por uma regra explícita e auditada (BLOCO 19.1) |
# | Diferenças de escala entre os sensores | Avaliar tipos e precisão; o escalonamento ocorrerá após a divisão treino/teste (BLOCOS 22 e 23) |
# | Indícios de valores extremos em `Light` e `CO2` | Construir boxplots e investigar potenciais outliers pelo IQR (BLOCOS 24 e 25) |
#
# ### Preservação das fontes
#
# Os arquivos de `dados/brutos/` e a base consolidada `dados/consolidados/occupancy.csv` não são alterados e mantêm a
# precisão original. O tratamento, incluindo a padronização decimal, é realizado sobre uma cópia de trabalho e
# exportado para `dados/tratados/occupancy_tratado.csv`.

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
# ### BLOCO 19.1 — Padronização da precisão decimal dos atributos numéricos
#
# **Origem da regra:** a padronização foi solicitada pelo estudante depois da inspeção visual de
# `dados/consolidados/occupancy.csv` e `dados/features/occupancy_features.csv`, nos quais os atributos apareciam com
# quantidades muito diferentes de casas decimais (por exemplo, `Light` = 572.666666666667 e `HumidityRatio` =
# 0.0047641630241641). **A regra é uma decisão metodológica deste projeto**: o enunciado, a rubrica e os materiais
# MUS01 e MUS02 tratam da limpeza, da verificação de tipos, de ausentes, de duplicidades e de outliers e da padronização
# de escala (`StandardScaler`), mas não definem quantidades de casas decimais para os sensores.
#
# | Atributo | Unidade | Casas decimais adotadas | Maior alteração possível por valor |
# |---|---|---:|---:|
# | `Temperature` | °C | 1 | 0,05 °C |
# | `Humidity` | % | 2 | 0,005 p.p. |
# | `Light` | lux | 1 | 0,05 lux |
# | `CO2` | ppm | 1 | 0,05 ppm |
# | `HumidityRatio` | kg/kg | 5 | 0,000005 kg/kg |
#
# `Occupancy` (inteiro) e `date` não são alterados, assim como os índices, a ordem das observações e os tipos
# (`float64`). A regra é aplicada somente às cinco colunas listadas, nunca a todas as colunas da base.
#
# **Justificativas:**
#
# | Aspecto | Justificativa |
# |---|---|
# | Consistência | Cada atributo passa a ser representado com uma quantidade fixa de casas decimais, independentemente de como o valor foi registrado na fonte |
# | Legibilidade | Tabelas, CSVs e relatórios ficam mais fáceis de ler e comparar |
# | Governança | A transformação é explícita (dicionário `casas_decimais`), uniforme e reproduzível |
# | Rastreabilidade | Os arquivos brutos e o CSV consolidado mantêm a precisão original, permitindo comparar e auditar a transformação |
# | Precisão numérica | Evita a apresentação de dígitos que sugerem uma precisão física não demonstrada (valores como 572.666666666667 resultam de médias calculadas na fonte, e não de uma resolução de 10⁻¹² lux) |
#
# **Quatro noções diferentes de precisão:**
#
# | Noção | Significado neste projeto |
# |---|---|
# | Precisão textual | Quantidade de dígitos com que o valor foi escrito no arquivo de origem (inspecionada abaixo) |
# | Precisão de armazenamento | O `float64` guarda cerca de 15 a 17 algarismos significativos; números como 0.1 não têm representação binária exata |
# | Precisão física dos sensores | **Não documentada** na página da UCI consultada; por isso, a regra não é apresentada como a resolução real dos equipamentos |
# | Casas exibidas | Formatação de apresentação; não altera o valor armazenado |
#
# **Arredondamento não é formatação:** `df_tratado.round(casas_decimais)` substitui cada valor pelo `float64` mais
# próximo do número decimal arredondado. Todas as fases seguintes usam esses valores. A formatação com zeros à direita
# (por exemplo, `26.30`) é aplicada somente na exportação do CSV (BLOCO 27), em uma cópia textual.
#
# **Empates e representação binária:** o pandas usa `numpy.round`, que multiplica o valor por 10^casas, arredonda para o
# inteiro par mais próximo em caso de empate exato (*round half to even*) e divide novamente. Como o valor armazenado é
# binário, um empate do texto decimal (por exemplo, 20.25 com uma casa) pode não ser um empate exato do `float64`, e o
# resultado depende da representação binária. Os empates textuais são contados e conferidos abaixo com o módulo
# `decimal`.
#
# **Diferença em relação ao StandardScaler:** o arredondamento reduz a resolução de cada valor, registro a registro, sem
# aprender nenhum parâmetro dos dados; por isso pode ser aplicado antes da divisão treino/teste sem Data Leakage. O
# `StandardScaler` (Fase 5) muda a escala dos atributos com média e desvio aprendidos no treino e não reduz a precisão.
#
# **Limitações:** o arredondamento é uma transformação com perda potencial de informação. Pode aumentar a quantidade de
# valores repetidos, alterar estatísticas, correlações, limites de outliers, distâncias do KNN, limiares da árvore e os
# hiperparâmetros selecionados, e pode melhorar ou piorar o desempenho observado. Não se afirma, antecipadamente,
# nenhum ganho preditivo. A regra foi definida **depois** da primeira avaliação completa dos modelos (BLOCO 103) e
# permanece fixa, independentemente do efeito nas métricas.
#
# **Camadas de dados:**
#
# | Camada | Arquivo | Precisão |
# |---|---|---|
# | Dados brutos | `dados/brutos/*.txt` e ZIP da UCI | Original |
# | Dados consolidados | `dados/consolidados/occupancy.csv` | Original (referência rastreável das fontes) |
# | Dados tratados | `dados/tratados/occupancy_tratado.csv` | Padronizada por esta regra |
# | Feature Engineering | `dados/features/occupancy_features.csv` | Atributos tratados; `CO2_Light_Ratio` recalculada com precisão integral |
# | Modelagem | Conjuntos em memória (Fases 4 a 7) | Atributos tratados |
#
# O CSV consolidado **não** contém os valores tratados: ele permanece como a cópia fiel das fontes.

# %%
# BLOCO 19.1 — Inspeção da precisão original (conteúdo textual das fontes)

# Regra decimal da revisão (decisão metodológica do estudante; aplicada somente a estes cinco atributos)
casas_decimais = {"Temperature": 1, "Humidity": 2, "Light": 1, "CO2": 1, "HumidityRatio": 5}
assert list(casas_decimais) == colunas_numericas, "A regra deve cobrir exatamente os cinco atributos ambientais."

# Leitura textual das duas fontes consolidadas, na mesma ordem da consolidação
texto_fontes = pd.concat(
    [pd.read_csv(caminho_datatest, dtype=str), pd.read_csv(caminho_datatest2, dtype=str)], ignore_index=True
)
assert len(texto_fontes) == len(df_tratado), "Quantidade de registros textuais divergente."

# O texto convertido de forma exata (float do Python) é comparado com a base consolidada. O conversor padrão do
# read_csv (BLOCO 05) não garante o float64 mais próximo de textos com 17 algarismos significativos (mesmo efeito
# documentado no BLOCO 35); as diferenças são medidas, e não ignoradas
linhas_conversao = []
for coluna in colunas_numericas:
    convertido_exato = texto_fontes[coluna].map(float).to_numpy()
    valores_consolidados = df_consolidado[coluna].to_numpy()
    diferentes = convertido_exato != valores_consolidados
    nao_nulos = valores_consolidados != 0
    diferenca_relativa_conversao = np.abs(convertido_exato - valores_consolidados)[nao_nulos] / valores_consolidados[nao_nulos]
    linhas_conversao.append({"Atributo": coluna, "Valores diferentes da conversão exata": int(diferentes.sum()),
                             "Maior diferença relativa": diferenca_relativa_conversao.max()})
    assert diferenca_relativa_conversao.max() < 1e-13, f"O texto de {coluna} não reproduz a base consolidada."
tabela_conversao_texto = pd.DataFrame(linhas_conversao).set_index("Atributo")
print("Base consolidada × conversão exata do texto das fontes:")
print(tabela_conversao_texto.to_string(float_format=lambda valor: f"{valor:.3e}"))


def casas_no_texto(serie_texto):
    """Quantidade de dígitos após o ponto decimal, como escrito no arquivo de origem."""
    return serie_texto.str.partition(".")[2].str.len()


linhas_precisao_original = []
for coluna, casas in casas_decimais.items():
    casas_texto = casas_no_texto(texto_fontes[coluna])
    posicao_exemplo = casas_texto.idxmax()
    linhas_precisao_original.append({
        "Atributo": coluna,
        "Casas adotadas": casas,
        "Valores distintos": df_consolidado[coluna].nunique(),
        "Mínimo": df_consolidado[coluna].min(),
        "Máximo": df_consolidado[coluna].max(),
        "Casas no texto (mín.)": int(casas_texto.min()),
        "Casas no texto (mediana)": float(casas_texto.median()),
        "Casas no texto (máx.)": int(casas_texto.max()),
        "Textos com até as casas adotadas": int((casas_texto <= casas).sum()),
        "Textos com mais casas": int((casas_texto > casas).sum()),
        "Exemplo com mais casas": texto_fontes.loc[posicao_exemplo, coluna],
    })
tabela_precisao_original = pd.DataFrame(linhas_precisao_original).set_index("Atributo")
print("Precisão textual das fontes (datatest.txt + datatest2.txt):")
print(tabela_precisao_original.to_string())

print("\nDistribuição da quantidade de casas decimais no texto das fontes (registros):")
distribuicao_casas = pd.DataFrame({
    coluna: casas_no_texto(texto_fontes[coluna]).value_counts() for coluna in colunas_numericas
}).fillna(0).astype(int).sort_index()
distribuicao_casas.index.name = "Casas no texto"
print(distribuicao_casas.to_string())

# Empates textuais: o texto termina exatamente na metade da última casa adotada (ex.: 20.25 com 1 casa)
linhas_empates = []
for coluna, casas in casas_decimais.items():
    quantum = Decimal(1).scaleb(-casas)
    decimais = texto_fontes[coluna].map(Decimal)
    empate = decimais.map(lambda valor: valor.scaleb(casas) % 1 == Decimal("0.5"))
    arredondado_numpy = df_consolidado[coluna].round(casas)
    meio_par = decimais.map(lambda valor: float(valor.quantize(quantum, rounding=ROUND_HALF_EVEN)))
    meio_cima = decimais.map(lambda valor: float(valor.quantize(quantum, rounding=ROUND_HALF_UP)))
    linhas_empates.append({
        "Atributo": coluna,
        "Empates textuais": int(empate.sum()),
        "Empates iguais ao meio-par decimal": int((arredondado_numpy[empate] == meio_par[empate]).sum()),
        "Empates iguais ao meio-para-cima decimal": int((arredondado_numpy[empate] == meio_cima[empate]).sum()),
        "Divergências fora de empates (numpy × decimal)": int((arredondado_numpy[~empate] != meio_par[~empate]).sum()),
    })
tabela_empates = pd.DataFrame(linhas_empates).set_index("Atributo")
print("\nEmpates de arredondamento no texto das fontes e resultado do numpy.round:")
print(tabela_empates.to_string())

# %%
# BLOCO 19.2 — Aplicação da regra decimal na base de trabalho

# Referência da base de trabalho antes da regra (idêntica à base consolidada)
df_tratado_antes_arredondamento = df_tratado.copy(deep=True)

# Arredondamento somente das cinco colunas autorizadas; date e Occupancy não participam
df_tratado = df_tratado.round(casas_decimais)

print("Regra aplicada:", casas_decimais)
print("Tipos após a regra:")
print(df_tratado.dtypes.to_string())
print("\nPrimeiro registro antes e depois:")
print(pd.DataFrame({
    "Antes": df_tratado_antes_arredondamento.loc[0, colunas_numericas],
    "Depois": df_tratado.loc[0, colunas_numericas],
}).to_string(float_format=lambda valor: repr(float(valor))))

# %%
# BLOCO 19.3 — Validação matemática do arredondamento

linhas_auditoria = []
exemplos_maior_alteracao = []
for coluna, casas in casas_decimais.items():
    antes = df_tratado_antes_arredondamento[coluna]
    depois = df_tratado[coluna]
    diferenca = depois - antes
    diferenca_abs = diferenca.abs()
    limite_teorico = 0.5 * 10.0 ** (-casas)
    # Tolerância para a representação binária: poucos ulps da maior magnitude da coluna
    tolerancia_binaria = 4 * np.finfo(np.float64).eps * antes.abs().max()
    modificados = depois != antes
    nao_nulos = antes != 0
    erro_relativo = diferenca_abs[nao_nulos] / antes[nao_nulos].abs()
    linhas_auditoria.append({
        "Atributo": coluna,
        "Casas decimais": casas,
        "Limite teórico": limite_teorico,
        "Registros": len(antes),
        "Modificados": int(modificados.sum()),
        "Modificados (%)": modificados.mean() * 100,
        "Inalterados": int((~modificados).sum()),
        "Diferença média com sinal": diferenca.mean(),
        "Diferença absoluta média": diferenca_abs.mean(),
        "Mediana da diferença absoluta": diferenca_abs.median(),
        "Maior diferença absoluta": diferenca_abs.max(),
        "Erro quadrático médio": (diferenca ** 2).mean(),
        "Erro relativo médio (valores ≠ 0)": erro_relativo.mean(),
        "Maior erro relativo (valores ≠ 0)": erro_relativo.max(),
        "Dentro do limite": bool((diferenca_abs <= limite_teorico + tolerancia_binaria).all()),
        "Valores distintos antes": antes.nunique(),
        "Valores distintos depois": depois.nunique(),
        "Zeros antes": int((antes == 0).sum()),
        "Zeros depois": int((depois == 0).sum()),
    })
    posicao = diferenca_abs.idxmax()
    exemplos_maior_alteracao.append({"Atributo": coluna, "Índice": posicao,
                                     "Texto na fonte": texto_fontes.loc[posicao, coluna],
                                     "Antes (float64)": antes[posicao], "Depois": depois[posicao],
                                     "Diferença": diferenca[posicao]})
tabela_auditoria_arredondamento = pd.DataFrame(linhas_auditoria).set_index("Atributo")
print("Auditoria do arredondamento por atributo:")
print(tabela_auditoria_arredondamento.T.to_string(float_format=lambda valor: f"{valor:.6g}"))

print("\nRegistros com a maior alteração absoluta em cada atributo:")
print(pd.DataFrame(exemplos_maior_alteracao).to_string(index=False, float_format=lambda valor: f"{valor:.10g}"))

# Validações bloqueantes
assert tabela_auditoria_arredondamento["Dentro do limite"].all(), "Alteração acima do limite teórico."
assert len(df_tratado) == len(df_tratado_antes_arredondamento) == 12417, "Quantidade de registros alterada."
assert df_tratado.index.equals(df_tratado_antes_arredondamento.index), "Índices ou ordem alterados."
assert list(df_tratado.columns) == list(df_consolidado.columns), "Colunas alteradas."
assert (df_tratado[colunas_numericas].dtypes == "float64").all(), "Tipo diferente de float64."
assert df_tratado.isna().sum().sum() == 0, "Valores ausentes introduzidos."
assert np.isfinite(df_tratado[colunas_numericas].to_numpy()).all(), "Valores infinitos introduzidos."
pd.testing.assert_series_equal(df_tratado["Occupancy"], df_consolidado["Occupancy"], check_exact=True)
pd.testing.assert_series_equal(df_tratado["date"], df_consolidado["date"], check_exact=True)
assert df_tratado["Occupancy"].dtype == "int64", "Occupancy deixou de ser inteiro."

# Idempotência: aplicar a mesma regra novamente não altera nenhum valor
pd.testing.assert_frame_equal(df_tratado.round(casas_decimais), df_tratado, check_exact=True)

# Cada valor é exatamente o float64 do número decimal com as casas adotadas
for coluna, casas in casas_decimais.items():
    reconstruido = df_tratado[coluna].map(lambda valor: float(f"{valor:.{casas}f}"))
    assert np.array_equal(reconstruido.to_numpy(), df_tratado[coluna].to_numpy()), f"{coluna}: valor não decimal."

print("\nValidações aprovadas: limites teóricos, 12.417 registros, índices e ordem, tipos float64, nenhum ausente ou")
print("infinito, Occupancy e date idênticos à base consolidada, idempotência e representação decimal exata.")

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
# em memória é um `float64`. Desde o BLOCO 19.2, os atributos ambientais contêm os valores arredondados pela regra
# decimal; este bloco confirma que eles correspondem exatamente à base consolidada arredondada e que nenhuma outra
# perda de precisão foi introduzida. A conversão para um tipo de menor precisão (`float32`) é testada apenas para
# demonstrar a perda adicional que ela provocaria; o tipo `float64` é mantido.

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

# Confirmação de que a única alteração numérica é a regra decimal do BLOCO 19.2
print("\nValores numéricos idênticos à base consolidada arredondada pela regra decimal:",
      df_tratado[colunas_numericas].equals(df_consolidado[colunas_numericas].round(casas_decimais)))
print("Valores numéricos idênticos à base consolidada sem arredondamento:",
      df_tratado[colunas_numericas].equals(df_consolidado[colunas_numericas]))
print("HumidityRatio (registro 0) — valor original na base consolidada:",
      repr(float(df_consolidado.loc[0, "HumidityRatio"])))

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
print("Valores numéricos preservados (iguais à base consolidada com a regra decimal):",
      df_tratado[colunas_numericas].equals(df_consolidado[colunas_numericas].round(casas_decimais)))

# %% [markdown]
# ### BLOCO 25.4 — Impacto da padronização decimal na qualidade dos dados
#
# A base consolidada (precisão original) é comparada com a base tratada (precisão padronizada) nas mesmas 12.417
# observações. As estatísticas da EDA (Fase 1) continuam descrevendo a camada consolidada; as dos BLOCOS 20 a 25
# descrevem a camada tratada. São comparadas a distribuição de cada atributo, a cardinalidade, os novos zeros, as
# repetições de vetores de sensores, as correlações com `Occupancy` (Pearson e Spearman) e os potenciais outliers pelo
# mesmo critério IQR do BLOCO 25. Uma diferença numérica não significa melhoria nem piora: ela apenas mede o efeito da
# transformação. O resultado é salvo em `resultados/modelagem/auditoria_precisao_decimal.csv`, com precisão integral, e
# resumido na figura `resultados/graficos/prep_02_padronizacao_precisao_decimal.png`.

# %%
# BLOCO 25.4 — Comparação entre a base consolidada e a base tratada


def estatisticas_atributo(serie, alvo):
    """Estatísticas descritivas, cardinalidade, correlações com o alvo e outliers pelo IQR (critério do BLOCO 25)."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return {
        "Média": serie.mean(),
        "Mediana": serie.median(),
        "Variância": serie.var(),
        "Desvio padrão": serie.std(),
        "Mínimo": serie.min(),
        "Primeiro quartil (Q1)": q1,
        "Terceiro quartil (Q3)": q3,
        "Máximo": serie.max(),
        "Amplitude": serie.max() - serie.min(),
        "Assimetria": serie.skew(),
        "Valores distintos": serie.nunique(),
        "Zeros": int((serie == 0).sum()),
        "Pearson com Occupancy": serie.corr(alvo, method="pearson"),
        "Spearman com Occupancy": serie.corr(alvo, method="spearman"),
        "Limite inferior IQR": q1 - 1.5 * iqr,
        "Limite superior IQR": q3 + 1.5 * iqr,
        "Potenciais outliers IQR": int(((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).sum()),
    }


linhas_comparacao_dados = []
for coluna in colunas_numericas:
    antes = estatisticas_atributo(df_consolidado[coluna], df_consolidado["Occupancy"])
    depois = estatisticas_atributo(df_tratado[coluna], df_tratado["Occupancy"])
    for indicador in antes:
        linhas_comparacao_dados.append({"Atributo": coluna, "Indicador": indicador, "Antes": antes[indicador],
                                        "Depois": depois[indicador], "Diferença": depois[indicador] - antes[indicador]})
tabela_comparacao_dados = pd.DataFrame(linhas_comparacao_dados)
for coluna in colunas_numericas:
    print(f"\n{coluna} — base consolidada (antes) × base tratada (depois):")
    print(tabela_comparacao_dados.query("Atributo == @coluna").set_index("Indicador")[["Antes", "Depois", "Diferença"]]
          .to_string(float_format=lambda valor: f"{valor:.10g}"))


def resumir_repeticoes(base):
    """Duplicidades integrais, timestamps repetidos e repetições de vetores dos cinco sensores."""
    repetidos = base.duplicated(subset=colunas_numericas, keep=False)
    classes_por_vetor = base.groupby(colunas_numericas)["Occupancy"].nunique()
    return {
        "Linhas integralmente duplicadas": int(base.duplicated().sum()),
        "Timestamps repetidos": int(base["date"].duplicated().sum()),
        "Vetores de sensores repetidos (ocorrências excedentes)": int(base.duplicated(subset=colunas_numericas).sum()),
        "Grupos de vetores repetidos": int(base.loc[repetidos].groupby(colunas_numericas).ngroups),
        "Registros envolvidos em repetições": int(repetidos.sum()),
        "Vetores associados a classes diferentes": int((classes_por_vetor > 1).sum()),
        "Vetores distintos": int(len(classes_por_vetor)),
    }


tabela_repeticoes_precisao = pd.DataFrame({
    "Antes": resumir_repeticoes(df_consolidado),
    "Depois": resumir_repeticoes(df_tratado),
})
tabela_repeticoes_precisao["Diferença"] = tabela_repeticoes_precisao["Depois"] - tabela_repeticoes_precisao["Antes"]
print("\nRepetições e duplicidades:")
print(tabela_repeticoes_precisao.to_string())

# Coincidências criadas pelo arredondamento: vetores originalmente distintos que passaram a ser iguais
vetores_originais_distintos = df_consolidado[colunas_numericas].drop_duplicates()
vetores_originais_por_grupo = vetores_originais_distintos.round(casas_decimais).value_counts()
grupos_com_fusao = vetores_originais_por_grupo[vetores_originais_por_grupo > 1]
print("\nVetores arredondados que reúnem dois ou mais vetores originalmente distintos:", len(grupos_com_fusao))
print("Vetores originais distintos envolvidos nessas fusões:", int(grupos_com_fusao.sum()))

# Maior alteração nas correlações de Pearson entre todas as variáveis numéricas (inclui Occupancy)
colunas_correlacao = colunas_numericas + ["Occupancy"]
diferenca_correlacoes = (df_tratado[colunas_correlacao].corr() - df_consolidado[colunas_correlacao].corr()).abs()
print(f"Maior alteração absoluta na matriz de correlação de Pearson: {diferenca_correlacoes.to_numpy().max():.3e}")

# Exportação da auditoria (precisão integral, formato longo)
linhas_exportacao_precisao = []
for coluna, linha in tabela_precisao_original.iterrows():
    for indicador in ["Valores distintos", "Casas no texto (mín.)", "Casas no texto (mediana)",
                      "Casas no texto (máx.)", "Textos com até as casas adotadas", "Textos com mais casas"]:
        linhas_exportacao_precisao.append({"grupo": "precisao_textual_original", "atributo": coluna,
                                           "indicador": indicador, "antes": np.nan, "depois": np.nan,
                                           "diferenca": np.nan, "valor": float(linha[indicador])})
for tabela_origem, grupo in [(tabela_empates, "empates_arredondamento"),
                             (tabela_auditoria_arredondamento.drop(columns=["Dentro do limite"]), "auditoria_arredondamento")]:
    for coluna, linha in tabela_origem.iterrows():
        for indicador, valor in linha.items():
            linhas_exportacao_precisao.append({"grupo": grupo, "atributo": coluna, "indicador": indicador,
                                               "antes": np.nan, "depois": np.nan, "diferenca": np.nan,
                                               "valor": float(valor)})
for coluna, linha in tabela_auditoria_arredondamento.iterrows():
    linhas_exportacao_precisao.append({"grupo": "auditoria_arredondamento", "atributo": coluna,
                                       "indicador": "Dentro do limite", "antes": np.nan, "depois": np.nan,
                                       "diferenca": np.nan, "valor": float(linha["Dentro do limite"])})
for _, linha in tabela_comparacao_dados.iterrows():
    linhas_exportacao_precisao.append({"grupo": "estatisticas_antes_depois", "atributo": linha["Atributo"],
                                       "indicador": linha["Indicador"], "antes": float(linha["Antes"]),
                                       "depois": float(linha["Depois"]), "diferenca": float(linha["Diferença"]),
                                       "valor": np.nan})
for indicador, linha in tabela_repeticoes_precisao.iterrows():
    linhas_exportacao_precisao.append({"grupo": "repeticoes_antes_depois", "atributo": "cinco sensores",
                                       "indicador": indicador, "antes": float(linha["Antes"]),
                                       "depois": float(linha["Depois"]), "diferenca": float(linha["Diferença"]),
                                       "valor": np.nan})
linhas_exportacao_precisao.append({"grupo": "repeticoes_antes_depois", "atributo": "cinco sensores",
                                   "indicador": "Vetores arredondados com fusão de vetores originais",
                                   "antes": np.nan, "depois": np.nan, "diferenca": np.nan,
                                   "valor": float(len(grupos_com_fusao))})
linhas_exportacao_precisao.append({"grupo": "correlacoes", "atributo": "cinco sensores e Occupancy",
                                   "indicador": "Maior alteração absoluta na matriz de Pearson", "antes": np.nan,
                                   "depois": np.nan, "diferenca": np.nan,
                                   "valor": float(diferenca_correlacoes.to_numpy().max())})
auditoria_precisao_exportacao = pd.DataFrame(linhas_exportacao_precisao)
assert not auditoria_precisao_exportacao.duplicated(subset=["grupo", "atributo", "indicador"]).any()
auditoria_precisao_exportacao.to_csv(caminho_auditoria_precisao, index=False)
pd.testing.assert_frame_equal(pd.read_csv(caminho_auditoria_precisao, float_precision="round_trip"),
                              auditoria_precisao_exportacao, check_exact=True, check_dtype=False)
print("\nAuditoria exportada:", caminho_auditoria_precisao, auditoria_precisao_exportacao.shape)

# %%
# BLOCO 25.5 — Visualização da padronização decimal

fig, eixos_precisao = plt.subplots(2, 3, figsize=(20, 11))
(eixo_exemplos, eixo_modificados, eixo_alteracoes), (eixo_cardinalidade, eixo_repeticoes, eixo_outliers) = eixos_precisao
posicoes_precisao = np.arange(len(colunas_numericas))
rotulos_precisao = [f"{coluna}\n({casas} casa{'s' if casas > 1 else ''})" for coluna, casas in casas_decimais.items()]

# Painel 1 — Exemplos reais (registro 0 da base) antes e depois
eixo_exemplos.axis("off")
linhas_tabela_exemplos = [[coluna, texto_fontes.loc[0, coluna], f"{df_tratado.loc[0, coluna]:.{casas}f}"]
                          for coluna, casas in casas_decimais.items()]
tabela_figura = eixo_exemplos.table(cellText=linhas_tabela_exemplos,
                                    colLabels=["Atributo", "Fonte (texto)", "Tratado"],
                                    loc="center", cellLoc="center", colWidths=[0.32, 0.42, 0.26])
tabela_figura.auto_set_font_size(False)
tabela_figura.set_fontsize(11)
tabela_figura.scale(1, 2.2)
eixo_exemplos.set_title(f"Exemplo real: registro 0 ({df_consolidado.loc[0, 'date']})")

# Painel 2 — Percentual de valores modificados
percentual_modificados = tabela_auditoria_arredondamento["Modificados (%)"]
barras = eixo_modificados.bar(posicoes_precisao, percentual_modificados, color="#4C72B0", edgecolor="black")
for barra, valor in zip(barras, percentual_modificados):
    eixo_modificados.annotate(f"{valor:.2f}%".replace(".", ","), (barra.get_x() + barra.get_width() / 2, valor),
                              ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=10)
eixo_modificados.set_xticks(posicoes_precisao, rotulos_precisao, fontsize=9)
eixo_modificados.set_ylim(0, 110)
eixo_modificados.set_ylabel("Valores modificados (%)")
eixo_modificados.set_title("Valores alterados pela regra decimal")

# Painel 3 — Alterações média e máxima relativas ao limite teórico de cada atributo
media_relativa = tabela_auditoria_arredondamento["Diferença absoluta média"] / tabela_auditoria_arredondamento["Limite teórico"]
maxima_relativa = tabela_auditoria_arredondamento["Maior diferença absoluta"] / tabela_auditoria_arredondamento["Limite teórico"]
eixo_alteracoes.bar(posicoes_precisao - 0.18, media_relativa * 100, width=0.36, color="#55A868", edgecolor="black",
                    label="Alteração absoluta média")
eixo_alteracoes.bar(posicoes_precisao + 0.18, maxima_relativa * 100, width=0.36, color="#DD8452", edgecolor="black",
                    label="Maior alteração absoluta")
eixo_alteracoes.axhline(100, color="#C44E52", linestyle="--", linewidth=1.2, label="Limite teórico (meia unidade)")
eixo_alteracoes.set_xticks(posicoes_precisao, rotulos_precisao, fontsize=9)
eixo_alteracoes.set_ylim(0, 125)
eixo_alteracoes.set_ylabel("Percentual do limite teórico (%)")
eixo_alteracoes.set_title("Alterações em relação ao limite de cada atributo")
eixo_alteracoes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)

# Painel 4 — Cardinalidade antes e depois
distintos_antes = tabela_auditoria_arredondamento["Valores distintos antes"]
distintos_depois = tabela_auditoria_arredondamento["Valores distintos depois"]
for deslocamento, valores, cor, rotulo in [(-0.18, distintos_antes, "#4C72B0", "Base consolidada"),
                                           (0.18, distintos_depois, "#DD8452", "Base tratada")]:
    barras = eixo_cardinalidade.bar(posicoes_precisao + deslocamento, valores, width=0.36, color=cor,
                                    edgecolor="black", label=rotulo)
    for barra, valor in zip(barras, valores):
        eixo_cardinalidade.annotate(f"{int(valor):,}".replace(",", "."), (barra.get_x() + barra.get_width() / 2, valor),
                                    ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=8,
                                    rotation=90)
eixo_cardinalidade.set_xticks(posicoes_precisao, rotulos_precisao, fontsize=9)
eixo_cardinalidade.set_ylim(0, max(distintos_antes.max(), distintos_depois.max()) * 1.3)
eixo_cardinalidade.set_ylabel("Valores distintos")
eixo_cardinalidade.set_title("Cardinalidade de cada atributo")
eixo_cardinalidade.legend(loc="upper left", fontsize=9)

# Painel 5 — Repetições de vetores dos cinco sensores
indicadores_repeticao = ["Vetores de sensores repetidos (ocorrências excedentes)", "Grupos de vetores repetidos",
                         "Registros envolvidos em repetições"]
rotulos_repeticao = ["Ocorrências\nexcedentes", "Grupos de\nvetores repetidos", "Registros\nenvolvidos"]
posicoes_repeticao = np.arange(len(indicadores_repeticao))
for deslocamento, coluna_tabela, cor, rotulo in [(-0.18, "Antes", "#4C72B0", "Base consolidada"),
                                                 (0.18, "Depois", "#DD8452", "Base tratada")]:
    valores = tabela_repeticoes_precisao.loc[indicadores_repeticao, coluna_tabela]
    barras = eixo_repeticoes.bar(posicoes_repeticao + deslocamento, valores, width=0.36, color=cor,
                                 edgecolor="black", label=rotulo)
    for barra, valor in zip(barras, valores):
        eixo_repeticoes.annotate(f"{int(valor):,}".replace(",", "."), (barra.get_x() + barra.get_width() / 2, valor),
                                 ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=9)
eixo_repeticoes.set_xticks(posicoes_repeticao, rotulos_repeticao, fontsize=9)
eixo_repeticoes.set_ylim(0, tabela_repeticoes_precisao.loc[indicadores_repeticao].to_numpy().max() * 1.2)
eixo_repeticoes.set_ylabel("Quantidade")
eixo_repeticoes.set_title("Vetores dos cinco sensores repetidos (sem date)")
eixo_repeticoes.legend(loc="upper left", fontsize=9)

# Painel 6 — Potenciais outliers pelo IQR antes e depois
outliers_antes = tabela_comparacao_dados.query("Indicador == 'Potenciais outliers IQR'").set_index("Atributo")["Antes"]
outliers_depois = tabela_comparacao_dados.query("Indicador == 'Potenciais outliers IQR'").set_index("Atributo")["Depois"]
for deslocamento, valores, cor, rotulo in [(-0.18, outliers_antes, "#4C72B0", "Base consolidada"),
                                           (0.18, outliers_depois, "#DD8452", "Base tratada")]:
    barras = eixo_outliers.bar(posicoes_precisao + deslocamento, valores.loc[colunas_numericas], width=0.36, color=cor,
                               edgecolor="black", label=rotulo)
    for barra, valor in zip(barras, valores.loc[colunas_numericas]):
        eixo_outliers.annotate(f"{int(valor):,}".replace(",", "."), (barra.get_x() + barra.get_width() / 2, valor),
                               ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=9)
eixo_outliers.set_xticks(posicoes_precisao, rotulos_precisao, fontsize=9)
eixo_outliers.set_ylim(0, max(outliers_antes.max(), outliers_depois.max()) * 1.2 + 1)
eixo_outliers.set_ylabel("Registros fora dos limites 1,5 × IQR")
eixo_outliers.set_title("Potenciais outliers (critério do BLOCO 25)")
eixo_outliers.legend(loc="upper left", fontsize=9)

plt.suptitle("Padronização da precisão decimal: base consolidada (original) × base tratada", fontsize=16)
plt.tight_layout()

# Exportação antes da exibição
plt.savefig(caminho_grafico_precisao, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_precisao)

plt.show()

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

# Correspondência com a base consolidada quando nenhum registro foi removido
if len(df_tratado) == len(df_consolidado):
    pd.testing.assert_frame_equal(
        df_tratado[colunas_numericas], df_consolidado[colunas_numericas].round(casas_decimais), check_exact=True
    )
    pd.testing.assert_series_equal(df_tratado["Occupancy"], df_consolidado["Occupancy"], check_exact=True)
    pd.testing.assert_series_equal(
        df_tratado["date"], pd.to_datetime(df_consolidado["date"], format="%Y-%m-%d %H:%M:%S"), check_exact=True
    )
    print("\nSensores idênticos à base consolidada arredondada pela regra decimal; Occupancy e date idênticos à base "
          "consolidada.")

print("Validação da base tratada concluída com sucesso!")

# %% [markdown]
# ### BLOCO 27 — Exportação e validação da base tratada
#
# A exportação utiliza `index=False` e preserva os segundos da coluna `date` (`%Y-%m-%d %H:%M:%S`). Para que o arquivo
# mostre exatamente a quantidade de casas decimais da regra (por exemplo, `26.30` em `Humidity`), cada atributo é
# convertido em texto com a sua própria quantidade de casas em uma **cópia exclusiva para exportação**; não se usa um
# `float_format` global, que imporia a mesma precisão a todas as colunas. O DataFrame analítico `df_tratado` continua
# numérico. O conteúdo textual do arquivo é conferido coluna a coluna, e a recarga com `float_precision="round_trip"`,
# com `date` convertida novamente para `datetime`, deve reproduzir a base tratada com `check_exact=True`: a formatação
# não introduz nenhuma perda além do arredondamento já aplicado. O visualizador de CSV do VS Code apenas exibe o texto
# gravado; o valor usado na análise é o `float64` recarregado.

# %%
# BLOCO 27 — Exportação e validação da base tratada


def formatar_para_exportacao(base, regra_casas):
    """Cópia exclusiva para exportação: cada atributo da regra vira texto com a sua quantidade fixa de casas."""
    copia = base.copy(deep=True)
    for coluna, casas in regra_casas.items():
        copia[coluna] = base[coluna].map(lambda valor: f"{valor:.{casas}f}")
    return copia


def conferir_texto_exportado(caminho, regra_casas):
    """Confere, no texto gravado, a quantidade exata de casas de cada atributo e o inteiro de Occupancy."""
    texto = pd.read_csv(caminho, dtype=str)
    resultado = {}
    for coluna, casas in regra_casas.items():
        resultado[coluna] = bool(texto[coluna].str.fullmatch(rf"\d+\.\d{{{casas}}}").all())
    resultado["Occupancy"] = bool(texto["Occupancy"].str.fullmatch(r"[01]").all())
    resultado["date"] = bool(texto["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all())
    return pd.Series(resultado, name="Formato textual correto"), texto


formatar_para_exportacao(df_tratado, casas_decimais).to_csv(
    caminho_tratado, index=False, date_format="%Y-%m-%d %H:%M:%S"
)
print("Base tratada exportada:", caminho_tratado)

# Conferência do conteúdo textual gravado
formato_texto_tratado, texto_tratado = conferir_texto_exportado(caminho_tratado, casas_decimais)
print("\nFormato textual por coluna:")
print(formato_texto_tratado.to_string())
assert formato_texto_tratado.all(), "O CSV tratado não segue a quantidade de casas definida."
print("\nPrimeiras linhas do texto gravado:")
print(texto_tratado.head(3).to_string(index=False))

# Recarga numérica exata e conversão da coluna temporal
df_tratado_recarregado = pd.read_csv(caminho_tratado, float_precision="round_trip")
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
# confirmada como equivalente a ela (`True`). Em seguida, recebeu a padronização da precisão decimal (BLOCO 19.2);
# todas as verificações dos BLOCOS 20 a 27 descrevem, portanto, a **base tratada**, enquanto `df_consolidado`
# permaneceu inalterado como referência da precisão original e a EDA (Fase 1) continua descrevendo a camada
# consolidada.
#
# ### 28.2 — Padronização da precisão decimal
#
# **Precisão textual das fontes.** As fontes registram os atributos com quantidades muito variáveis de casas
# decimais: de 0 a 13 em `Temperature` e `Humidity`, de 0 a 14 em `Light`, de 0 a 12 em `CO2` e de 13 a 17 em
# `HumidityRatio` (11.158 dos 12.417 textos com 17 casas). Os valores com muitas casas (por exemplo, `Light` =
# 9.33333333333333 e `CO2` = 769.666666666667) são compatíveis com médias calculadas na origem, e não com uma
# resolução física dos sensores, que não está documentada na página da UCI consultada.
#
# | Atributo | Casas adotadas | Modificados | Inalterados | Diferença absoluta média | Maior diferença | Distintos antes → depois |
# |---|---:|---:|---:|---:|---:|---:|
# | `Temperature` | 1 | 7.045 (56,74%) | 5.372 | 0,0118 °C | 0,05 °C | 444 → 50 |
# | `Humidity` | 2 | 3.708 (29,86%) | 8.709 | 0,00116 % | 0,005 % | 1.815 → 1.134 |
# | `Light` | 1 | 855 (6,89%) | 11.562 | 0,00286 lux | 0,05 lux | 1.439 → 1.424 |
# | `CO2` | 1 | 3.376 (27,19%) | 9.041 | 0,0111 ppm | 0,05 ppm | 4.262 → 4.114 |
# | `HumidityRatio` | 5 | 12.417 (100%) | 0 | 2,49 × 10⁻⁶ | 5,0 × 10⁻⁶ | 6.220 → 250 |
#
# Exemplos reais: no registro 0 (02/02/2015 14:19), `Humidity` passou de 26.272 para 26.27 e `HumidityRatio` de
# 0.00476416302416414 (texto da fonte) para 0.00476; `Temperature` (23.7), `Light` (585.2) e `CO2` (749.2) já tinham
# as casas adotadas. As maiores alterações ocorreram em empates exatos, como `Temperature` 23.15 → 23.2 e `CO2`
# 2065.75 → 2065.8. Nenhuma alteração ultrapassou o limite teórico de meia unidade da última casa (`Dentro do limite`
# em todos os atributos), e as diferenças médias com sinal são praticamente nulas (de −0,00019 a +0,0017 nas unidades
# de cada atributo), sem viés sistemático relevante. O maior erro relativo foi de 0,71% em uma leitura baixa de `Light`;
# nos demais atributos, ficou abaixo de 0,26%.
#
# **Empates e representação binária.** Havia 417 empates textuais em `Temperature`, 1.684 em `Humidity`, 428 em
# `Light`, 1.491 em `CO2` e nenhum em `HumidityRatio`. O `numpy.round` coincidiu com o arredondamento decimal
# "meio-par" em todos os empates, exceto em 89 de `Humidity`, nos quais o `float64` armazenado não é um empate exato e
# o resultado seguiu o valor binário; fora dos empates, não houve nenhuma divergência entre o arredondamento binário
# e o decimal. A conversão exata do texto das fontes difere da base consolidada apenas em `HumidityRatio` (11.158
# valores, diferença relativa máxima de 2,7 × 10⁻¹⁴), porque o conversor padrão do `read_csv` não garante o `float64`
# mais próximo de textos com 17 algarismos; essa diferença é muitas ordens de grandeza menor que a regra de cinco casas.
#
# **Validações:** 12.417 registros, índices, ordem e colunas preservados; tipos `float64`; nenhum ausente ou infinito
# introduzido; `Occupancy` e `date` idênticos à base consolidada; a regra é idempotente (aplicá-la novamente não altera
# nenhum valor); e cada valor é exatamente o `float64` do número decimal com as casas adotadas. Nenhum zero novo foi
# criado (`Light` mantém 7.612 zeros, e os demais atributos continuam sem zeros).
#
# **Impacto na qualidade dos dados (BLOCO 25.4):** as médias variaram no máximo 0,0017 °C (`Temperature`), os desvios
# padrão no máximo 0,0004 °C, e a maior alteração na matriz de correlação de Pearson foi de 9,8 × 10⁻⁴. As correlações
# com `Occupancy` ficaram praticamente iguais (`Light`: 0,9197 nas duas versões; `Temperature`: 0,5764 → 0,5765). A
# mudança mais visível está na cardinalidade, sobretudo de `HumidityRatio` (6.220 → 250 valores distintos, redução de
# 95,98%) e `Temperature` (444 → 50, redução de 88,74%); `Humidity` reduziu 37,52%, `CO2` 3,47% e `Light` 1,04%. Menos
# valores distintos não significam melhor qualidade preditiva, e a resolução física dos sensores não foi confirmada. A
# mudança também aparece no limite superior IQR
# de `Temperature`, que passou de 23,6025 para 23,65 °C (461 → 447 potenciais outliers). O arredondamento fundiu 64
# vetores de sensores originalmente distintos em 32 vetores, o que elevou as repetições de 571 para 603. Essas
# variações descrevem o efeito da transformação, e não uma melhoria da qualidade dos dados.
#
# ### 28.3 — Investigação de duplicidades
#
# | Critério | Repetições | Registros envolvidos |
# |---|---:|---:|
# | Todas as colunas (inclui `date`) | 0 | 0 |
# | Coluna `date` | 0 | — |
# | Somente os cinco sensores | 603 | 1.090 |
# | Sensores + `Occupancy` (sem `date`) | 603 | 1.090 |
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
# foi removido. As 603 repetições encontradas ao desconsiderar `date` correspondem a medições ambientais iguais em
# instantes diferentes, distribuídas em 487 grupos de medições idênticas que envolvem 1.090 registros. Na precisão
# original eram 571 repetições, 461 grupos e 1.032 registros; o acréscimo vem das 32 fusões de vetores causadas pelo
# arredondamento (28.2).
#
# | Característica dos 1.090 registros envolvidos | Resultado |
# |---|---:|
# | Desocupados (`Occupancy = 0`) | 1.084 |
# | Ocupados (`Occupancy = 1`) | 6 |
# | Com `Light` igual a zero | 1.043 |
# | Intervalo entre a primeira e a última ocorrência de cada grupo | de 59 segundos a 9 dias, 6 horas e 37 minutos (mediana de 6 minutos) |
#
# Os três grupos exibidos como exemplo ocorreram na madrugada e no início da manhã de 14/02/2015, com o ambiente
# desocupado, com intervalos de 20 minutos e 1 segundo, 13 minutos e 8 minutos e 59 segundos entre as leituras iguais;
# esses valores descrevem apenas os exemplos e não o conjunto das repetições. Em nenhum caso a mesma combinação de
# medições aparece associada a classes diferentes, nem antes nem depois do arredondamento.
#
# As repetições concentram-se em registros desocupados e sem iluminação, situação em que as condições do ambiente
# variam pouco. Isso é compatível com um ambiente estável e com a resolução de registro dos sensores, mas não comprova,
# por si só, o funcionamento correto dos equipamentos. Como cada registro corresponde a um instante distinto, as
# repetições foram mantidas: removê-las reduziria a amostra e alteraria a frequência real dessas condições.
#
# ### 28.4 — Tratamento de valores ausentes
#
# **Base real:** 0 valores ausentes em todas as colunas, 0 registros afetados e 0 valores infinitos. Nenhuma imputação
# foi aplicada. As 7.612 leituras de `Light` iguais a zero (61,30% da base) foram mantidas, pois representam ambientes
# sem iluminação e não ausência de medição.
#
# **Simulação didática (cópia temporária `df_simulacao`, calculada sobre a base tratada):**
#
# | Variável | Assimetria | Ausentes simulados | Estratégia | Valor imputado | Ausentes após | Erro absoluto médio |
# |---|---:|---:|---|---:|---:|---:|
# | Humidity | 0,3008 | 50 | Média | 28,918101 % | 0 | 3,1982 % |
# | CO2 | 1,3851 | 50 | Mediana | 632,0 ppm | 0 | 192,63 ppm |
#
# | Variável | Média | Mediana | Média − mediana | Diferença em desvios padrão | Assimetria |
# |---|---:|---:|---:|---:|---:|
# | Temperature | 21,096183 | 20,800000 | 0,296183 | 0,2854 | 1,1015 |
# | Humidity | 28,917759 | 28,600000 | 0,317759 | 0,0770 | 0,3008 |
# | Light | 138,125900 | 0,000000 | 138,125900 | 0,6284 | 1,3173 |
# | CO2 | 745,644455 | 632,000000 | 113,644455 | 0,3833 | 1,3851 |
# | HumidityRatio | 0,0044682476 | 0,0045100000 | −0,0000417524 | −0,0701 | −0,1450 |
#
# A simulação usa as mesmas sementes e posições da versão anterior (sobre a precisão original: média de 28,918288 %,
# erro de 3,1983 % e 192,6317 ppm); as diferenças decorrem apenas do arredondamento e não alteram a escolha das
# estratégias.
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
# `HumidityRatio` tem a menor assimetria absoluta (−0,1450), mas é uma grandeza derivada da temperatura e da umidade
# relativa; `Humidity` foi escolhida por ser uma medição direta do sensor. `Temperature`, considerada inicialmente para
# a média, foi descartada como exemplo por sua assimetria de 1,1015. `Light` não foi usada na simulação, pois 61,30% dos
# seus valores são zero e a mediana coincide com o mínimo.
#
# Os erros absolutos médios medem a distância entre o valor imputado e os valores reais ocultados nesta simulação
# específica; eles não demonstram que a estratégia escolhida seja a melhor em qualquer cenário. A simulação utilizou
# semente fixa (42) e não alterou `df_tratado` (0 ausentes após a demonstração). Em um cenário real, a estatística de
# imputação deve ser calculada apenas sobre o conjunto de treinamento.
#
# ### 28.5 — Adequação dos tipos de dados
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
# ### 28.6 — Precisão armazenada
#
# Desde o BLOCO 19.2, os atributos armazenam os valores arredondados pela regra decimal: o primeiro registro de
# `HumidityRatio`, que na base consolidada é 0,0047641630241641, está armazenado como 0,00476. O BLOCO 23 confirmou que
# os valores tratados são exatamente a base consolidada arredondada (`True`) e, portanto, diferentes da base sem
# arredondamento (`False`): nenhuma outra perda de precisão foi introduzida. Os tipos `float64` foram mantidos; a
# conversão para `float32` acrescentaria erros relativos de até aproximadamente 5,7 × 10⁻⁸, sem justificativa de
# desempenho. As diferenças de escala serão tratadas pelo `StandardScaler`, exclusivamente para o KNN e somente após a
# divisão treino/teste.
#
# ### 28.7 — Análise dos boxplots e outliers
#
# Evidência visual: `resultados/graficos/dp_01_boxplots.png` (base tratada).
#
# | Variável | Q1 | Q3 | IQR | Limite inferior | Limite superior | Abaixo | Acima | Total | Percentual | Ocupado entre os outliers |
# |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | Temperature (°C) | 20,40 | 21,70 | 1,30 | 18,45 | 23,65 | 0 | 447 | 447 | 3,60% | 42,95% |
# | Humidity (%) | 25,39 | 31,79 | 6,40 | 15,79 | 41,39 | 0 | 0 | 0 | 0,00% | — |
# | Light (lux) | 0,0 | 392,0 | 392,0 | −588,0 | 980,0 | 0 | 7 | 7 | 0,06% | 100,00% |
# | CO2 (ppm) | 527,5 | 857,0 | 329,5 | 33,25 | 1.351,25 | 0 | 992 | 992 | 7,99% | 30,95% |
# | HumidityRatio (kg/kg) | 0,00399 | 0,00494 | 0,00095 | 0,002565 | 0,006365 | 0 | 0 | 0 | 0,00% | — |
#
# Na precisão original, `Temperature` tinha Q1 = 20,39, Q3 = 21,675, limite superior de 23,6025 °C e 461 potenciais
# outliers (44,69% ocupados); os demais atributos tinham as mesmas contagens. A redução para 447 decorre do
# arredondamento dos quartis e dos valores próximos ao limite, e não de uma mudança no ambiente.
#
# - `Temperature`: 447 potenciais outliers, todos acima de 23,65 °C, coerentes com a cauda à direita observada no
#   histograma. Entre eles, 42,95% são registros ocupados, proporção superior aos 24,33% da base.
# - `Humidity` e `HumidityRatio`: nenhum valor fora dos limites, em concordância com as distribuições sem caudas
#   longas observadas na EDA.
# - `Light`: apenas 7 valores acima de 980 lux (máximo de 1.697,2 lux), todos em registros ocupados. O limite inferior
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
# base (42,95% em `Temperature`, 100% nos 7 casos de `Light` e 30,95% em `CO2`, contra 24,33%). Essa associação foi
# observada na base completa, antes da divisão treino/teste, e não foi usada como fundamento da decisão. Os limites IQR
# foram calculados com finalidade exploratória; caso venham a ser usados em alguma transformação do modelo, deverão ser
# ajustados exclusivamente sobre o conjunto de treinamento.
#
# ### 28.8 — Validação da base tratada
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
# Todas as verificações `assert` foram aprovadas: os sensores são idênticos à base consolidada arredondada pela regra
# decimal, e `Occupancy` e `date` são idênticos à base consolidada (`check_exact=True`). A base foi exportada para
# `dados/tratados/occupancy_tratado.csv` (656.251 bytes, CRLF) a partir de uma cópia textual com a quantidade fixa de
# casas de cada atributo (por exemplo, `23.7`, `26.27`, `585.2`, `749.2` e `0.00476` no primeiro registro). O texto
# gravado foi conferido coluna a coluna, e a recarga com `float_precision="round_trip"`, com `date` convertida para
# `datetime`, reproduziu exatamente a base tratada. **O arquivo tratado deixou de ser idêntico ao CSV consolidado**,
# que permanece com a precisão original e SHA-256 inalterado como referência rastreável das fontes.
#
# ### 28.9 — Conclusão do Data Prep
#
# A preparação dos dados confirmou que a base consolidada não apresenta duplicatas integrais, timestamps repetidos,
# valores ausentes, valores infinitos ou valores fisicamente implausíveis. A precisão decimal dos cinco atributos foi
# padronizada por uma regra explícita, idempotente e auditada, com alterações sempre dentro de meia unidade da última
# casa adotada e efeito pequeno nas estatísticas e correlações. As repetições de medições em instantes diferentes e os
# potenciais outliers identificados pelo IQR foram investigados e mantidos, por representarem condições ambientais
# legítimas e informativas para a ocupação. A base tratada preserva os 12.417 registros, a ordem, os timestamps e a
# proporção das classes (75,67% e 24,33%); a precisão original permanece disponível na base consolidada.
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

# Atributos de origem com a regra decimal da Fase 2 (base consolidada arredondada)
pd.testing.assert_frame_equal(
    df_features[colunas_numericas], df_consolidado[colunas_numericas].round(casas_decimais), check_exact=True
)
# A variável derivada não é arredondada: precisão integral do float64 sobre os atributos tratados
assert not df_features["CO2_Light_Ratio"].equals(df_features["CO2_Light_Ratio"].round(5)), \
    "CO2_Light_Ratio não deveria estar arredondada."

# Preservação das bases anteriores
pd.testing.assert_frame_equal(df_tratado, df_tratado_recarregado, check_exact=True)
pd.testing.assert_frame_equal(df_consolidado, pd.read_csv(caminho_csv), check_exact=True)
assert df_tratado.shape == (12417, 7) and df_consolidado.shape == (12417, 7), "Bases anteriores alteradas."

print("\nValidação da integridade concluída com sucesso!")

# %% [markdown]
# ### BLOCO 35 — Exportação e validação da base de Feature Engineering
#
# A exportação segue o padrão da Fase 2: `index=False`, coluna `date` no formato `%Y-%m-%d %H:%M:%S` e os cinco
# atributos tratados gravados com a quantidade fixa de casas da regra decimal, em uma cópia exclusiva para exportação.
# `CO2_Light_Ratio` **não** segue a regra dos sensores: é gravada com a representação textual mais curta que reconstrói
# exatamente o `float64` (precisão integral), porque é o valor usado pela modelagem; cinco casas decimais são usadas
# apenas em tabelas de apresentação. Após a recarga, `date` é convertida novamente para `datetime` e a equivalência é
# verificada com `check_exact=True`.
#
# **Precisão na leitura:** a divisão produz valores com até 17 algarismos significativos. O pandas grava cada valor com
# a representação textual mais curta que reconstrói exatamente o `float64`, mas o conversor padrão de `read_csv`
# (`float_precision="high"`) pode diferir do valor original no último bit. Por isso, a recarga utiliza
# `float_precision="round_trip"`, que reconstrói exatamente os valores gravados. A variável derivada não é arredondada
# e a comparação permanece exata.

# %%
# BLOCO 35 — Exportação e validação da base de Feature Engineering

formatar_para_exportacao(df_features, casas_decimais).to_csv(
    caminho_features, index=False, date_format="%Y-%m-%d %H:%M:%S"
)
print("Base de Feature Engineering exportada:", caminho_features)

# Conferência do conteúdo textual: atributos com casas fixas; variável derivada com precisão integral
formato_texto_features, texto_features = conferir_texto_exportado(caminho_features, casas_decimais)
formato_texto_features["CO2_Light_Ratio (reconstrução exata do texto)"] = bool(
    (texto_features["CO2_Light_Ratio"].map(float).to_numpy() == df_features["CO2_Light_Ratio"].to_numpy()).all()
)
print("Formato textual por coluna:")
print(formato_texto_features.to_string())
assert formato_texto_features.all(), "O CSV de Feature Engineering não segue o formato definido."
print("\nPrimeiras linhas do texto gravado:")
print(texto_features.head(3).to_string(index=False))

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
# Na base tratada, porém, não há leituras de `Light` entre 0 e 4,7 lux: a menor luminosidade positiva é 4,7 lux (4,67
# na precisão original). Assim, a faixa de maior sensibilidade da fórmula não é ocupada por observações reais, e o
# salto do índice ocorre entre o grupo `Light = 0` e as leituras positivas. Para `Light > 0`, o denominador é de pelo
# menos 5,7, e a
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
# | Média | 426,134639 |
# | Desvio padrão | 403,140847 |
# | Mínimo | 0,560476 |
# | Primeiro quartil (Q1) | 2,999126 |
# | Mediana | 503,000000 |
# | Terceiro quartil (Q3) | 596,300000 |
# | Máximo | 2.076,500000 |
# | Amplitude | 2.075,939524 |
# | Assimetria | 0,822266 |
# | Zeros / distintos / ausentes / infinitos | 0 / 6.928 / 0 / 0 |
#
# A variável foi recalculada com `CO2` e `Light` tratados, mantendo a precisão integral do `float64` (cinco casas
# decimais são usadas somente em tabelas de apresentação). Na precisão original, a média era 426,134753, o Q3 596,333333,
# o máximo do grupo com luz 263,294118 e havia 6.989 valores distintos; as diferenças são residuais e não mudam a
# interpretação abaixo.
#
# Evidência visual: `resultados/graficos/fe_01_distribuicao_nova_variavel.png`.
#
# A distribuição não é contínua ao longo da amplitude: os valores formam dois agrupamentos separados. Os registros com
# luz ocupam a faixa de 0,56 a 261,75, e os registros sem luz ocupam a faixa de 427,50 a 2.076,50. Não há nenhuma
# observação entre 261,75 e 427,50. Por isso, a média (426,13) cai justamente nessa faixa vazia e não representa um
# valor típico. O salto entre Q1 (3,00) e a mediana (503,00) indica a mesma separação.
#
# - **Escala linear:** os 50 intervalos somam 12.417 registros; 8 intervalos ficaram vazios. O primeiro intervalo (0,56
#   a 42,08) concentra 4.558 registros, comprimindo quase todo o grupo com luz em uma única barra. O grupo sem luz
#   reproduz a forma da distribuição de `CO2`, com maior concentração entre aproximadamente 500 e 600.
# - **Escala logarítmica (somente visualização):** os 50 intervalos também somam 12.417 registros; 3 intervalos
#   ficaram vazios, todos entre 208,01 e 401,40. O agrupamento com luz fica visível, com maior concentração entre 1,77
#   e 2,46 (1.430 registros em dois intervalos). O intervalo de maior frequência (473,10 a 557,61, com 2.527 registros)
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
# | `Light = 0` | 7.612 | 61,30% | 689,649764 | 569,300000 | 427,500000 | 2.076,500000 |
# | `Light > 0` | 4.805 | 38,70% | 8,678422 | 2,299771 | 0,560476 | 261,754386 |
#
# Nos 7.612 registros com `Light = 0`, a comparação exata confirmou que `CO2_Light_Ratio` é igual a `CO2` (`True`).
# Nesses casos, a nova variável não acrescenta nenhuma diferença em relação ao sensor original: ela repete o valor
# de CO2. A transformação só modifica efetivamente os 4.805 registros com luminosidade positiva, cujo índice é baixo
# (mediana de 2,30; 75% abaixo de 4,55). Como o menor valor do grupo sem luz (427,50) é maior que o maior valor do
# grupo com luz (261,75), o índice separa completamente os dois grupos de luminosidade. Essa separação, porém, já está
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
#   da escala. Por isso, o escalonamento não é necessário. Qualquer limiar entre 261,75 e 427,50 no índice separa os
#   mesmos registros que um limiar em `Light` entre 0 e 4,7 lux, o que mostra que, nesse ponto, a árvore não ganha
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
# Os cinco atributos de origem coincidem exatamente com a base consolidada arredondada pela regra decimal, e a variável
# derivada não foi arredondada.
#
# A base foi exportada para `dados/features/occupancy_features.csv` (787.706 bytes, CRLF), com os cinco atributos no
# formato textual de casas fixas da Fase 2 e `CO2_Light_Ratio` com a representação mais curta que reconstrói exatamente
# cada `float64`; o formato de todas as colunas foi conferido no texto gravado. Na recarga com o conversor padrão do
# pandas, 801 valores da nova variável divergiram no último bit (diferença relativa máxima de 2,45 × 10⁻¹⁶), embora o
# texto gravado represente exatamente cada valor. A leitura com `float_precision="round_trip"` reconstruiu todos os
# valores, e a comparação com `check_exact=True` foi aprovada sem tolerância. Nenhum registro foi excluído.
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

# Comparação com os índices da versão anterior à regra decimal (referência histórica preservada no repositório)
referencia_historica = pd.read_csv(caminho_referencia_historica, float_precision="round_trip")
assert set(referencia_historica["versao"]) == {"anterior_precisao_integral"}, "Referência histórica inesperada."


def sha256_indices(valores):
    """Assinatura SHA-256 de uma sequência de índices ou posições (ordem preservada)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(valores, dtype=np.int64)).tobytes()).hexdigest()


def valor_texto_historico(secao, metrica, configuracao=None):
    filtro = (referencia_historica["secao"] == secao) & (referencia_historica["metrica"] == metrica)
    if configuracao is not None:
        filtro &= referencia_historica["configuracao"] == configuracao
    selecionado = referencia_historica.loc[filtro, "valor_texto"]
    assert len(selecionado) == 1, f"Referência histórica ausente ou duplicada: {secao}/{metrica}/{configuracao}."
    return selecionado.iloc[0]


comparacao_indices_split = pd.DataFrame({
    "SHA-256 atual (16 caracteres)": [sha256_indices(X_train.index)[:16], sha256_indices(X_test.index)[:16]],
    "Idêntico à versão anterior": [
        sha256_indices(X_train.index) == valor_texto_historico("indices", "sha256_indices_treino"),
        sha256_indices(X_test.index) == valor_texto_historico("indices", "sha256_indices_teste"),
    ],
}, index=["Índices de treino (ordem do split)", "Índices de teste (ordem do split)"])
print("\nComparação com o split da versão anterior à padronização decimal:")
print(comparacao_indices_split.to_string())
assert comparacao_indices_split["Idêntico à versão anterior"].all(), "O split difere da versão anterior."
print("O split depende apenas de y, do tamanho da base e da semente; a regra decimal não o alterou.")

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
# **Revisão da precisão decimal:** os atributos de `X` passaram a conter os valores tratados da Fase 2, mas a divisão
# estratificada depende apenas de `y`, do tamanho da base e da semente. As assinaturas SHA-256 dos índices de treino e
# de teste, na ordem gerada pelo split, são idênticas às da versão anterior (`True`, BLOCO 43); as distribuições das
# classes e toda a auditoria temporal do BLOCO 44 permanecem iguais, pois timestamps, índices e classes não mudaram.
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
# | Temperature | °C | 19,50 | 24,40 | 21,10 | 1,04 |
# | Humidity | % | 21,86 | 39,50 | 28,92 | 4,13 |
# | Light | lux | 0,00 | 1.581,00 | 137,87 | 219,17 |
# | CO2 | ppm | 427,50 | 2.076,50 | 746,50 | 297,49 |
# | HumidityRatio | kg/kg | 0,00327 | 0,00577 | 0,00447 | 0,000595 |
# | CO2_Light_Ratio | índice | 0,56 | 2.076,50 | 426,77 | 404,49 |
#
# Os valores descrevem os atributos tratados (regra decimal da Fase 2). Com duas casas, as estatísticas coincidem com
# as da precisão original, exceto o máximo de `Temperature` (24,41 → 24,40), o mínimo de `Humidity` (21,87 → 21,86) e a
# média de `Temperature` (21,09 → 21,10).
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
# | 10164 (16/02/2015 19:46:59) | CO2 | 760,7 | 746,49908 | 297,4853080 | 0,0477365 |
# | 10164 | HumidityRatio | 0,0046 | 0,0044684265 | 0,0005952746 | 0,2210300 |
# | 986 (03/02/2015 06:45:00) | CO2 | 436,8 | 746,49908 | 297,4853080 | −1,0410567 |
# | 986 | HumidityRatio | 0,00335 | 0,0044684265 | 0,0005952746 | −1,8788411 |
#
# Na precisão original, a média aprendida de `CO2` era 746,49934 e o fator de escala 297,4853061; o `HumidityRatio` do
# registro 10164 era 0,0045977054 e passou a 0,0046 (0.00460 no CSV tratado). O cálculo manual coincidiu exatamente com
# o `StandardScaler` nos quatro casos (diferença 0), e a fórmula recalculada para todos os valores reproduziu os 59.598
# valores do treino e os 14.904 do teste sem nenhuma diferença. O registro
# 986 tem `CO2` 1,04 desvio padrão abaixo da média e `HumidityRatio` 1,88 abaixo: depois da padronização, as duas
# medições, antes separadas por seis ordens de grandeza, passam a ser expressas na mesma escala.
#
# ### 60.3 — Resultados do treinamento padronizado
#
# Nos seis atributos, a média do treino padronizado ficou entre 2,4 × 10⁻¹⁷ (`Light`) e 1,1 × 10⁻¹⁵ em valor absoluto
# (`HumidityRatio`), e o desvio padrão populacional não diferiu de 1 na precisão calculada. Esses resíduos não são
# zero porque a média é calculada com somas de números `float64`, cuja precisão relativa é de cerca de 2,2 × 10⁻¹⁶;
# somar quase 10 mil valores acumula erros dessa ordem. Eles estão muito abaixo da tolerância de `1e-10` e confirmam
# média 0 e desvio padrão 1, como esperado para o conjunto em que o scaler foi ajustado.
#
# ### 60.4 — Resultados do teste padronizado
#
# | Atributo | Média do teste padronizado | Desvio padrão do teste padronizado |
# |---|---:|---:|
# | Temperature | −0,0008 | 1,0027 |
# | Humidity | −0,0014 | 0,9988 |
# | Light | 0,0059 | 1,0141 |
# | CO2 | −0,0144 | 0,9830 |
# | HumidityRatio | −0,0015 | 1,0012 |
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
# | Temperature | 0,02% | 16,70% | 0 → 164 |
# | Humidity | 1,00% | 20,73% | 10 → 237 |
# | Light | 15,95% | 13,63% | 109 → 192 |
# | CO2 | 34,31% | 14,24% | 509 → 152 |
# | HumidityRatio | < 0,0001% | 20,07% | 0 → 153 |
# | CO2_Light_Ratio | 48,72% | 14,63% | 372 → 102 |
#
# Os pares sorteados são os mesmos da versão anterior (mesmas posições em `X_train` e semente 42). As contribuições
# variaram no máximo 0,07 p.p. em relação à precisão original (por exemplo, `Temperature` depois da padronização:
# 16,63% → 16,70%); a conclusão não muda.
#
# Nos 1.000 pares de treino sorteados (nenhum com distância zero, portanto nenhum excluído das contribuições; as
# contribuições de cada par somaram 100% com desvio máximo de 4,3 × 10⁻¹⁴), `CO2_Light_Ratio` e `CO2` respondiam
# juntos por 83,03% do quadrado da distância antes da padronização, e `CO2` ou `CO2_Light_Ratio` era o maior termo em
# 881 pares. `Temperature` (0,02%) e `HumidityRatio` (menos de 0,0001%) praticamente não participavam: na prática, um
# KNN sem escalonamento ignoraria esses atributos. Depois da padronização, as contribuições médias ficaram entre
# 13,63% e 20,73%, e todos os atributos passaram a dominar uma parte dos pares.
#
# As contribuições não ficaram iguais (16,67% cada): a padronização iguala a variância, e não a importância. `Humidity`
# e `HumidityRatio`, fortemente correlacionadas (0,89 na EDA), somam 40,80% das distâncias padronizadas, e a umidade
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
# diferença entre 0 a 1.581 lux e 19,50 a 24,40 °C não interfere, pois os atributos nunca são somados.
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
# ordem e o tipo `float64` dos atributos, sem ausentes nem infinitos. A transformação inversa reconstruiu 92,82% dos
# valores do treino e 92,91% dos do teste de forma **binariamente idêntica** (igualdade exata de bits; 100% em
# `Temperature` e `Humidity`, cerca de 62% em `CO2_Light_Ratio`); **100% dos valores** foram reconstruídos dentro da
# tolerância do BLOCO 55, e o maior erro equivale a menos de 0,6 ulp da maior magnitude de cada atributo, isto é,
# arredondamento de ponto flutuante nas operações de subtração, divisão, multiplicação e soma, e não um erro relevante.
# Nos valores não idênticos, o maior erro absoluto foi de
# 2,3 × 10⁻¹³ (`CO2` e `CO2_Light_Ratio`) e o maior erro relativo, de 1,1 × 10⁻¹³, em `CO2_Light_Ratio`, cujos menores valores (0,56)
# convivem com média de 426,77 — diferenças de arredondamento, dentro da tolerância relativa à magnitude de cada
# atributo.
#
# Os conjuntos estão prontos para a Fase 6, na qual serão treinados o KNN (com `Pipeline`) e a Árvore de Decisão,
# seguindo as regras do BLOCO 59. Nenhum classificador foi instanciado ou treinado nesta fase.

# %% [markdown]
# # FASE 6 — Hiperparâmetros e Pipeline
# ## BLOCO 61 — Contextualização do ajuste de hiperparâmetros
#
# ### Requisitos do enunciado e materiais do curso
#
# | Fonte | Exigência ou conteúdo |
# |---|---|
# | Enunciado, Fase 6 (p. 5) | KNN: variar `n_neighbors` (K) em, no mínimo, 3 valores ímpares (ex.: K = 3, 5, 7) e registrar a acurácia **no treino e no teste** |
# | Enunciado, Fase 6 (p. 5) | Árvore de Decisão: variar `max_depth` em, no mínimo, 3 limites (ex.: 3, 5 e None) e registrar a acurácia **no treino e no teste** |
# | Enunciado, Fase 6 (p. 5) | Célula de texto identificando em quais configurações ocorreu overfitting (alta acurácia no treino acompanhada de queda no teste) e qual configuração garantiu maior estabilidade |
# | Rubrica, critério 9 (p. 9) | Pontuação total: ao menos 3 variações de parâmetros para cada modelo e os pontos de sobreajuste apontados no texto; avaliar apenas com os dados de treino é pontuação parcial |
# | Enunciado, Fase 7 (p. 5) e rubrica, critério 10 (p. 9) | Acurácia final do melhor KNN e da melhor Árvore exclusivamente no teste, comparação e conclusão (fase seguinte, não implementada aqui) |
# | `knn.ipynb` | Compara K = 1, 5 e 25: K muito pequeno gera fronteiras irregulares (overfitting) e K muito grande gera fronteiras genéricas (underfitting); recomenda K ímpar em problemas binários e o `StandardScaler` antes do KNN |
# | `ArvoreDecisao_RandomForest (1).ipynb` | `max_depth` limita a profundidade para evitar overfitting; `random_state` garante reprodutibilidade |
# | `03_pipeline_exemplo.ipynb` | `Pipeline(StandardScaler, KNeighborsClassifier(n_neighbors=5))` avaliado com `cross_val_score` e `KFold(n_splits=5, shuffle=True, random_state=42)`; o scaler aprende apenas com o treino de cada fold |
# | MUS01, slides 62 e 63 | Underfitting (viés): modelo simplista demais; overfitting (variância): modelo sensível demais a ruídos e pequenas variações |
# | MUS01, slides 52, 95 e 96 | Acurácia ou matriz de confusão na classificação; em bases desbalanceadas a acurácia pode ser ilusória (paradoxo da acurácia) |
#
# O enunciado indica valores **de exemplo**, não valores obrigatórios; todos eles estão incluídos na busca. São
# **complementos metodológicos deste projeto**, e não exigências do professor: a seleção por validação cruzada
# estratificada no treino, o F1-score da classe ocupada e a balanced accuracy, a comparação com e sem
# `CO2_Light_Ratio`, a medição da proximidade temporal nos folds e a validação temporal complementar.
#
# ### Conceitos
#
# - **Aprendizado supervisionado e classificação binária:** o modelo aprende com registros rotulados (`X_train` e
#   `y_train`) a prever uma classe entre duas possíveis: 0 (desocupado) ou 1 (ocupado).
# - **Parâmetros × hiperparâmetros:** parâmetros são aprendidos no ajuste (a média e o desvio do scaler, os atributos
#   e limiares de cada divisão da árvore); hiperparâmetros são definidos antes do ajuste e controlam a complexidade
#   do modelo (K no KNN, `max_depth` na árvore).
# - **KNN:** armazena os registros de treino e classifica um novo registro pela votação majoritária dos K vizinhos
#   mais próximos, segundo a distância euclidiana `d(a, b) = √Σ_j (x_aj − x_bj)²` calculada sobre os atributos
#   padronizados (BLOCO 56). K pequeno acompanha detalhes locais e é mais sensível a ruídos; K grande suaviza a
#   fronteira e pode ignorar padrões locais. Valores ímpares evitam empate na votação entre duas classes.
# - **Árvore de Decisão:** divide os dados por perguntas sucessivas do tipo `atributo ≤ limiar`, escolhendo em cada
#   nó a divisão que mais reduz a impureza de Gini. `max_depth` limita o número de perguntas do caminho mais longo;
#   com `None`, a árvore cresce até separar completamente os registros de treino (ou até não haver divisão possível).
# - **Generalização, overfitting e underfitting:** generalizar é acertar registros não usados no ajuste. Overfitting
#   é um desempenho muito melhor no treino do que em dados novos (o modelo memoriza particularidades); underfitting é
#   um desempenho insuficiente já no treino (o modelo é simples demais). Uma pequena diferença entre treino e
#   avaliação é esperada e não caracteriza, sozinha, overfitting.
# - **Validação cruzada:** o treino é dividido em 5 partes (*folds*); cada parte serve uma vez de validação enquanto
#   as outras quatro ajustam o modelo. A média e a variabilidade dos 5 resultados estimam o desempenho sem usar o teste.
# - **Pipeline e Data Leakage:** o `Pipeline` encadeia `StandardScaler` e `KNeighborsClassifier`; a cada ajuste, o
#   scaler aprende a média e o desvio somente com os dados recebidos no `fit` (o treino de cada fold), e a validação
#   é apenas transformada. Escalonar antes da validação cruzada faria estatísticas da validação participarem da
#   preparação (vazamento de dados).
#
# ### Protocolo desta fase
#
# 1. Hiperparâmetros e atributos são escolhidos **somente com o treino oficial**, pela validação cruzada.
# 2. A seleção é registrada e congelada antes de qualquer consulta ao teste.
# 3. Como o enunciado exige a acurácia de teste de cada configuração, o teste é consultado depois, no BLOCO 73, como
#    resultado descritivo que não altera a seleção. Por isso, a Fase 7 não será a primeira consulta ao teste.
#
# **Limitação da validação cruzada aleatória nesta base:** os registros são medições consecutivas, a cada minuto,
# quase idênticas às vizinhas (BLOCO 16.2). Ao sortear os folds, a maioria dos registros de validação terá um vizinho
# temporal imediato na parte de ajuste, e a validação cruzada herda o otimismo já medido para o teste (BLOCO 44). Os
# BLOCOS 70 e 71 medem esse efeito e o comparam com uma validação que respeita a ordem do tempo.

# %% [markdown]
# ### BLOCO 62 — Verificação dos conjuntos oficiais
#
# São usados os conjuntos da Fase 4 (nenhum novo split): `X_train` e `X_test` não escalonados, que alimentam o
# `Pipeline` do KNN e a Árvore, e os conjuntos padronizados da Fase 5, que não são usados no treinamento desta fase. As
# assinaturas SHA-256 calculadas aqui (valores e índices) servem de referência para conferir, no BLOCO 77, que nenhum
# conjunto foi alterado. O marcador `teste_consultado_fase6` registra se o teste já foi consultado.

# %%
# BLOCO 62 — Verificação dos conjuntos oficiais

# Conjuntos oficiais da Fase 4: nenhum novo split é realizado nesta fase
assert X_train.shape == (9933, 6) and X_test.shape == (2484, 6), "Dimensões de X_train/X_test divergentes."
assert y_train.shape == (9933,) and y_test.shape == (2484,), "Dimensões de y_train/y_test divergentes."
assert list(X_train.columns) == list(X_test.columns) == atributos_preditores, "Ordem dos atributos alterada."
assert (X_train.dtypes == "float64").all() and (X_test.dtypes == "float64").all(), "Atributo diferente de float64."
assert y_train.value_counts().sort_index().tolist() == [7516, 2417], "Classes do treino alteradas."
assert y_test.value_counts().sort_index().tolist() == [1880, 604], "Classes do teste alteradas."
assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index), "Índices de X e y divergentes."
assert X_train.index.intersection(X_test.index).empty, "Registros em comum entre treino e teste."
assert "date" not in X_train.columns and variavel_alvo not in X_train.columns, "Coluna indevida nos atributos."

# Conjuntos idênticos às referências do BLOCO 50 (dados não escalonados e alvos)
pd.testing.assert_frame_equal(X_train, X_train_referencia, check_exact=True)
pd.testing.assert_frame_equal(X_test, X_test_referencia, check_exact=True)
pd.testing.assert_series_equal(y_train, y_train_referencia, check_exact=True)
pd.testing.assert_series_equal(y_test, y_test_referencia, check_exact=True)
pd.testing.assert_frame_equal(rastreio_temporal, rastreio_temporal_referencia, check_exact=True)
pd.testing.assert_frame_equal(df_features, df_features_referencia, check_exact=True)

# Conjuntos escalonados da Fase 5 reproduzidos pelo scaler oficial (somente transform, sem novo ajuste)
pd.testing.assert_frame_equal(
    pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index), X_train_scaled,
    check_exact=True,
)
pd.testing.assert_frame_equal(
    pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index), X_test_scaled,
    check_exact=True,
)
assert rastreio_temporal.index.equals(X.index) and set(X_train.index) | set(X_test.index) == set(X.index), \
    "Rastreabilidade temporal incompleta."


def assinatura_integridade(objeto):
    """SHA-256 dos valores e do índice (pd.util.hash_pandas_object), usado como referência de integridade."""
    valores_hash = pd.util.hash_pandas_object(objeto, index=True).to_numpy()
    return hashlib.sha256(valores_hash.tobytes()).hexdigest()


estruturas_fase6 = {
    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
    "rastreio_temporal": rastreio_temporal, "df_features": df_features,
}
assinaturas_fase6 = {nome: assinatura_integridade(estrutura) for nome, estrutura in estruturas_fase6.items()}

# Marcador de controle: passa a True somente no BLOCO 73, depois do registro da seleção
teste_consultado_fase6 = False

tabela_conjuntos_fase6 = pd.DataFrame({
    "Dimensões": [str(estrutura.shape) for estrutura in estruturas_fase6.values()],
    "Escalonado": ["Não", "Não", "—", "—", "Sim", "Sim", "—", "—"],
    "Assinatura SHA-256 (16 primeiros caracteres)": [assinatura[:16] for assinatura in assinaturas_fase6.values()],
}, index=pd.Index(estruturas_fase6.keys(), name="Estrutura"))
print(tabela_conjuntos_fase6.to_string())

tabela_classes_fase6 = pd.DataFrame({
    "Classe 0": [int((y_train == 0).sum()), int((y_test == 0).sum())],
    "Classe 1": [int((y_train == 1).sum()), int((y_test == 1).sum())],
}, index=pd.Index(["Treinamento", "Teste"], name="Conjunto"))
tabela_classes_fase6["% Classe 1"] = tabela_classes_fase6["Classe 1"] / tabela_classes_fase6.sum(axis=1) * 100
print("\nDistribuição das classes:")
print(tabela_classes_fase6.round(4).to_string())
print("\nAtributos (ordem oficial):", atributos_preditores)

# %% [markdown]
# ### BLOCO 63 — Configuração dos hiperparâmetros
#
# | Modelo | Hiperparâmetro investigado | Valores | Justificativa |
# |---|---|---|---|
# | KNN | `n_neighbors` | 3, 5, 7, 9, 11 | Cinco valores ímpares, incluindo os exemplos do enunciado (3, 5, 7) e dois valores maiores para observar a suavização |
# | Árvore de Decisão | `max_depth` | 3, 5, 10, None | Inclui os exemplos do enunciado (3, 5, None) e um limite intermediário (10) entre a árvore rasa e a árvore sem limite |
#
# Parâmetros fixos: no KNN, distância euclidiana (`metric="euclidean"`) e votos com pesos iguais (`weights="uniform"`),
# sempre dentro do `Pipeline` com `StandardScaler`; na árvore, critério de Gini, `random_state=42` e demais parâmetros
# no padrão do scikit-learn, sem escalonamento. A busca é deliberadamente pequena: o objetivo é cumprir e interpretar a
# comparação exigida, sem uma otimização extensa que aumentaria o risco de selecionar ruído. A execução é sequencial
# (um processo), sem paralelismo.

# %%
# BLOCO 63 — Configuração dos hiperparâmetros

# Hiperparâmetros investigados (incluem os exemplos do enunciado: K = 3, 5, 7 e max_depth = 3, 5, None)
valores_k = [3, 5, 7, 9, 11]
valores_max_depth = [3, 5, 10, None]

# Parâmetros fixos (não investigados)
parametros_fixos_knn = {"weights": "uniform", "metric": "euclidean"}
parametros_fixos_arvore = {"criterion": "gini", "random_state": 42}

# Protocolo de validação cruzada e métricas (F1 da classe 1 = ocupado)
quantidade_folds = 5
semente_validacao = 42
metricas_validacao = {"acuracia": "accuracy", "f1_ocupado": "f1", "balanced_accuracy": "balanced_accuracy"}

assert len(valores_k) >= 3 and all(k % 2 == 1 for k in valores_k), "K deve ter ao menos 3 valores ímpares."
assert len(set(valores_k)) == len(valores_k) and {3, 5, 7}.issubset(valores_k), "Valores de K incompletos."
assert len(valores_max_depth) >= 3 and {3, 5, None}.issubset(set(valores_max_depth)), "max_depth incompleto."


def rotulo_profundidade(profundidade):
    return "None" if profundidade is None else str(profundidade)


def criar_pipeline_knn(k):
    """Pipeline do KNN: o StandardScaler é ajustado novamente a cada chamada de fit."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, **parametros_fixos_knn)),
    ])


def criar_arvore(profundidade):
    """Árvore de Decisão sem escalonamento, com semente fixa."""
    return DecisionTreeClassifier(max_depth=profundidade, **parametros_fixos_arvore)


tabela_hiperparametros = pd.DataFrame([
    {"Modelo": "KNN (Pipeline)", "Investigado": "n_neighbors", "Valores": ", ".join(map(str, valores_k)),
     "Parâmetros fixos": "StandardScaler no Pipeline; weights='uniform'; metric='euclidean'"},
    {"Modelo": "Árvore de Decisão", "Investigado": "max_depth",
     "Valores": ", ".join(rotulo_profundidade(valor) for valor in valores_max_depth),
     "Parâmetros fixos": "criterion='gini'; random_state=42; demais parâmetros no padrão; sem escalonamento"},
]).set_index("Modelo")
print(tabela_hiperparametros.to_string())
print(f"\nValidação cruzada: StratifiedKFold(n_splits={quantidade_folds}, shuffle=True, "
      f"random_state={semente_validacao}) | métricas: {list(metricas_validacao)}")
print("Execução sequencial (n_jobs não informado = 1 processo).")

# %% [markdown]
# ### BLOCO 64 — Validação cruzada estratificada
#
# `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` divide **apenas** `X_train` e `y_train`, preservando
# aproximadamente a proporção de ocupados em cada fold. Os índices dos folds são materializados uma única vez
# (`folds_cv`), de modo que todas as configurações dos dois modelos, com 5 ou 6 atributos, são avaliadas exatamente nas
# mesmas divisões. A estratificação controla a proporção das classes, mas não a dependência temporal: os folds continuam
# sorteados entre registros vizinhos no tempo (BLOCO 70).

# %%
# BLOCO 64 — Validação cruzada estratificada

validacao_estratificada = StratifiedKFold(n_splits=quantidade_folds, shuffle=True, random_state=semente_validacao)
folds_cv = list(validacao_estratificada.split(X_train, y_train))

# Reprodutibilidade: a mesma configuração gera exatamente as mesmas divisões
folds_repetidos = list(
    StratifiedKFold(n_splits=quantidade_folds, shuffle=True, random_state=semente_validacao).split(X_train, y_train)
)
assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(folds_cv, folds_repetidos)), \
    "Folds não reproduzíveis."
del folds_repetidos

linhas_folds = []
for numero_fold, (posicoes_treino, posicoes_validacao) in enumerate(folds_cv, start=1):
    assert np.intersect1d(posicoes_treino, posicoes_validacao).size == 0, "Sobreposição dentro do fold."
    assert len(posicoes_treino) + len(posicoes_validacao) == len(X_train), "Fold incompleto."
    y_treino_fold = y_train.iloc[posicoes_treino]
    y_validacao_fold = y_train.iloc[posicoes_validacao]
    linhas_folds.append({
        "Fold": numero_fold,
        "Treino do fold": len(posicoes_treino),
        "Treino classe 0": int((y_treino_fold == 0).sum()),
        "Treino classe 1": int((y_treino_fold == 1).sum()),
        "Treino % classe 1": y_treino_fold.mean() * 100,
        "Validação": len(posicoes_validacao),
        "Validação classe 0": int((y_validacao_fold == 0).sum()),
        "Validação classe 1": int((y_validacao_fold == 1).sum()),
        "Validação % classe 1": y_validacao_fold.mean() * 100,
    })
tabela_folds = pd.DataFrame(linhas_folds).set_index("Fold")

# Cobertura: cada registro do treino oficial aparece exatamente uma vez como validação
todas_validacoes = np.sort(np.concatenate([posicoes_validacao for _, posicoes_validacao in folds_cv]))
assert np.array_equal(todas_validacoes, np.arange(len(X_train))), "Validações não cobrem o treino integralmente."

# Nenhum registro do teste oficial participa dos folds (posições referem-se apenas a X_train)
rotulos_folds = set(X_train.index[todas_validacoes])
assert rotulos_folds == set(X_train.index) and rotulos_folds.isdisjoint(X_test.index), "Teste presente nos folds."

print(tabela_folds.round(4).to_string())

# Os folds devem ser exatamente os mesmos da versão anterior à padronização decimal
folds_identicos_historico = [
    sha256_indices(posicoes_validacao) == valor_texto_historico("indices", "sha256_posicoes_validacao",
                                                                f"fold {numero_fold}")
    for numero_fold, (_, posicoes_validacao) in enumerate(folds_cv, start=1)
]
print("\nFolds idênticos aos da versão anterior (posições de validação, SHA-256):", folds_identicos_historico)
assert all(folds_identicos_historico), "Os folds diferem da versão anterior."
print(f"\nMaior desvio da % de classe 1 na validação em relação ao treino oficial: "
      f"{(tabela_folds['Validação % classe 1'] - y_train.mean() * 100).abs().max():.4f} p.p.")
print("Registros de validação somados:", int(tabela_folds["Validação"].sum()), "| registros do teste nos folds: 0")

# %% [markdown]
# ### BLOCO 65 — Pipeline do KNN
#
# O `Pipeline` recebe `X_train` **original** (não escalonado). Em cada fold, o `fit` do Pipeline ajusta um novo
# `StandardScaler` somente com a parte de treino do fold e transforma a validação com esses parâmetros. `X_train_scaled`
# não é fornecido ao Pipeline (seria um escalonamento duplicado), e o `scaler` da Fase 5 não é reutilizado. A
# demonstração abaixo ajusta o Pipeline no fold 1 e compara a média aprendida pelo scaler interno com as médias do
# treino e da validação do fold e com o scaler da Fase 5.

# %%
# BLOCO 65 — Pipeline do KNN

pipeline_knn_estrutura = criar_pipeline_knn(valores_k[0])
print(pipeline_knn_estrutura)
assert list(pipeline_knn_estrutura.named_steps) == ["scaler", "knn"], "Etapas do Pipeline divergentes."
assert isinstance(pipeline_knn_estrutura.named_steps["scaler"], StandardScaler)
assert isinstance(pipeline_knn_estrutura.named_steps["knn"], KNeighborsClassifier)
assert pipeline_knn_estrutura.named_steps["scaler"] is not scaler, "Scaler da Fase 5 reutilizado."

# Demonstração no fold 1: o scaler interno aprende somente com a parte de treino do fold
posicoes_treino_demo, posicoes_validacao_demo = folds_cv[0]
pipeline_demonstracao = criar_pipeline_knn(valores_k[0]).fit(
    X_train.iloc[posicoes_treino_demo], y_train.iloc[posicoes_treino_demo]
)
scaler_interno_demo = pipeline_demonstracao.named_steps["scaler"]

tabela_scaler_fold = pd.DataFrame({
    "Média do scaler interno (fold 1)": scaler_interno_demo.mean_,
    "Média do treino do fold 1": X_train.iloc[posicoes_treino_demo].mean().to_numpy(),
    "Média da validação do fold 1": X_train.iloc[posicoes_validacao_demo].mean().to_numpy(),
    "Média do scaler da Fase 5 (treino oficial)": scaler.mean_,
}, index=pd.Index(atributos_preditores, name="Atributo"))
print("\nMédias aprendidas pelo scaler interno no fold 1:")
print(tabela_scaler_fold.to_string(float_format=lambda valor: f"{valor:.6f}"))

assert int(scaler_interno_demo.n_samples_seen_) == len(posicoes_treino_demo), "Scaler interno viu outros registros."
assert np.allclose(scaler_interno_demo.mean_, X_train.iloc[posicoes_treino_demo].mean(),
                   rtol=tolerancia_parametros, atol=0), "Média do scaler interno difere do treino do fold."
assert np.allclose(scaler_interno_demo.scale_, X_train.iloc[posicoes_treino_demo].std(ddof=0),
                   rtol=tolerancia_parametros, atol=0), "Desvio do scaler interno difere do treino do fold."
assert not np.allclose(scaler_interno_demo.mean_, X_train.mean(), rtol=tolerancia_parametros, atol=0)
assert not np.allclose(scaler_interno_demo.mean_, X_train.iloc[posicoes_validacao_demo].mean(),
                       rtol=tolerancia_parametros, atol=0)

# O scaler oficial da Fase 5 permanece inalterado
for nome_parametro in ["mean_", "var_", "scale_"]:
    assert np.array_equal(getattr(scaler, nome_parametro), parametros_apos_ajuste[nome_parametro]), \
        f"Scaler da Fase 5 alterado ({nome_parametro})."
print("\nObservações vistas pelo scaler interno:", int(scaler_interno_demo.n_samples_seen_),
      "| registros de validação do fold:", len(posicoes_validacao_demo))
print("Scaler interno = treino do fold: True | scaler da Fase 5 inalterado: True")

# %% [markdown]
# ### BLOCO 66 — Comparação dos valores de K
#
# `cross_validate` avalia cada K nos mesmos 5 folds, com `return_train_score=True` (métricas na parte de ajuste e na
# validação de cada fold) e `return_estimator=True` (os Pipelines ajustados são conferidos no BLOCO 77). As métricas
# são a acurácia, o F1-score da classe ocupada (classe 1) e a balanced accuracy (média do recall das duas classes).
# Médias e desvios padrão são calculados entre os 5 folds (`ddof=0`). Nenhuma métrica de teste é calculada. No KNN, a
# acurácia "de treino" inclui o próprio registro entre os vizinhos, o que a torna naturalmente mais alta para K pequeno.
# Os tempos são informativos e variam entre execuções; por isso não são exportados nos CSVs.

# %%
# BLOCO 66 — Comparação dos valores de K (validação cruzada)

rotulo_protocolo_cv = f"StratifiedKFold(n_splits={quantidade_folds}, shuffle=True, random_state={semente_validacao})"


def avaliar_validacao_cruzada(fabrica, valores, nome_modelo, nome_hiperparametro, colunas):
    """Avalia cada valor do hiperparâmetro nos mesmos folds; devolve registros por fold, tempos e estimadores."""
    registros, tempos, estimadores = [], [], {}
    X_avaliacao = X_train[colunas]
    for valor in valores:
        resultado = cross_validate(
            fabrica(valor), X_avaliacao, y_train, cv=folds_cv, scoring=metricas_validacao,
            return_train_score=True, return_estimator=True,
        )
        rotulo = rotulo_profundidade(valor)
        estimadores[rotulo] = resultado["estimator"]
        tempos.append({"Modelo": nome_modelo, "Valor": rotulo, "Ajuste médio (s)": resultado["fit_time"].mean(),
                       "Predição média (s)": resultado["score_time"].mean()})
        for indice_fold, (posicoes_treino, posicoes_validacao) in enumerate(folds_cv):
            for conjunto, prefixo, quantidade in [("treino", "train", len(posicoes_treino)),
                                                  ("validacao", "test", len(posicoes_validacao))]:
                for metrica in metricas_validacao:
                    registros.append({
                        "protocolo": rotulo_protocolo_cv,
                        "modelo": nome_modelo,
                        "hiperparametro": nome_hiperparametro,
                        "valor_hiperparametro": rotulo,
                        "configuracao_atributos": f"{len(colunas)} atributos",
                        "quantidade_atributos": len(colunas),
                        "fold": indice_fold + 1,
                        "conjunto": conjunto,
                        "metrica": metrica,
                        "resultado": resultado[f"{prefixo}_{metrica}"][indice_fold],
                        "observacoes_avaliadas": quantidade,
                        "observacoes_ajuste": len(posicoes_treino),
                        "semente_folds": semente_validacao,
                    })
    return pd.DataFrame(registros), pd.DataFrame(tempos), estimadores


def resumir_validacao_cruzada(registros):
    """Médias e desvios padrão (ddof=0, entre os 5 folds) por configuração."""
    agrupado = registros.groupby(["modelo", "valor_hiperparametro", "conjunto", "metrica"], sort=False)["resultado"]
    estatisticas = agrupado.agg(media="mean", desvio=lambda serie: serie.std(ddof=0)).unstack(["conjunto", "metrica"])
    resumo = pd.DataFrame({
        "Acurácia treino (média)": estatisticas[("media", "treino", "acuracia")],
        "Acurácia treino (dp)": estatisticas[("desvio", "treino", "acuracia")],
        "Acurácia validação (média)": estatisticas[("media", "validacao", "acuracia")],
        "Acurácia validação (dp)": estatisticas[("desvio", "validacao", "acuracia")],
        "F1 ocupado validação (média)": estatisticas[("media", "validacao", "f1_ocupado")],
        "F1 ocupado validação (dp)": estatisticas[("desvio", "validacao", "f1_ocupado")],
        "Balanced accuracy validação (média)": estatisticas[("media", "validacao", "balanced_accuracy")],
        "Balanced accuracy validação (dp)": estatisticas[("desvio", "validacao", "balanced_accuracy")],
    })
    resumo["Diferença treino − validação (p.p.)"] = (
        resumo["Acurácia treino (média)"] - resumo["Acurácia validação (média)"]
    ) * 100
    return resumo


registros_knn, tempos_knn, estimadores_knn = avaliar_validacao_cruzada(
    criar_pipeline_knn, valores_k, "KNN", "n_neighbors", atributos_preditores
)
resumo_knn = resumir_validacao_cruzada(registros_knn).droplevel("modelo")
resumo_knn.index.name = "K"

print("Validação cruzada — KNN (Pipeline com StandardScaler), 6 atributos:")
print(resumo_knn.to_string(float_format=lambda valor: f"{valor:.6f}"))

acuracia_validacao_knn_folds = (
    registros_knn.query("conjunto == 'validacao' and metrica == 'acuracia'")
    .pivot(index="valor_hiperparametro", columns="fold", values="resultado")
    .loc[resumo_knn.index]
)
print("\nAcurácia de validação por fold:")
print(acuracia_validacao_knn_folds.to_string(float_format=lambda valor: f"{valor:.6f}"))

# Erros médios por fold (a validação tem cerca de 1.987 registros: 1 erro ≈ 0,05 p.p.)
media_validacao_fold = tabela_folds["Validação"].mean()
erros_knn = (1 - resumo_knn["Acurácia validação (média)"]) * media_validacao_fold
print("\nErros médios de validação por fold:", erros_knn.round(2).to_dict())
print("\nTempos médios (informativos, variam entre execuções):")
print(tempos_knn.set_index("Valor").drop(columns="Modelo").to_string(float_format=lambda valor: f"{valor:.4f}"))

# %% [markdown]
# ### BLOCO 67 — Árvore de Decisão
#
# A árvore usa `X_train` original, sem escalonamento, e os mesmos folds do KNN. Além das métricas, registra-se a
# profundidade efetivamente alcançada: `max_depth` é apenas um limite máximo, e a árvore pode parar antes se não houver
# divisão útil. Com `max_depth=None`, não há limite: a árvore cresce até que as folhas sejam puras ou não possam mais
# ser divididas.

# %%
# BLOCO 67 — Árvore de Decisão (validação cruzada nos mesmos folds)

registros_arvore, tempos_arvore, estimadores_arvore = avaliar_validacao_cruzada(
    criar_arvore, valores_max_depth, "Árvore de Decisão", "max_depth", atributos_preditores
)
resumo_arvore = resumir_validacao_cruzada(registros_arvore).droplevel("modelo")
resumo_arvore.index.name = "max_depth"

# Profundidade efetivamente alcançada e tamanho das árvores em cada fold
estrutura_arvores_folds = pd.DataFrame({
    rotulo: {
        "Profundidade efetiva (mín.)": min(arvore.get_depth() for arvore in arvores),
        "Profundidade efetiva (máx.)": max(arvore.get_depth() for arvore in arvores),
        "Folhas (média)": np.mean([arvore.get_n_leaves() for arvore in arvores]),
        "Nós (média)": np.mean([arvore.tree_.node_count for arvore in arvores]),
    }
    for rotulo, arvores in estimadores_arvore.items()
}).T
estrutura_arvores_folds.index.name = "max_depth"

print("Validação cruzada — Árvore de Decisão (dados originais, sem escalonamento), 6 atributos:")
print(resumo_arvore.to_string(float_format=lambda valor: f"{valor:.6f}"))
print("\nEstrutura das árvores ajustadas nos 5 folds:")
print(estrutura_arvores_folds.to_string(float_format=lambda valor: f"{valor:.1f}"))

acuracia_validacao_arvore_folds = (
    registros_arvore.query("conjunto == 'validacao' and metrica == 'acuracia'")
    .pivot(index="valor_hiperparametro", columns="fold", values="resultado")
    .loc[resumo_arvore.index]
)
print("\nAcurácia de validação por fold:")
print(acuracia_validacao_arvore_folds.to_string(float_format=lambda valor: f"{valor:.6f}"))
erros_arvore = (1 - resumo_arvore["Acurácia validação (média)"]) * media_validacao_fold
print("\nErros médios de validação por fold:", erros_arvore.round(2).to_dict())
print("\nTempos médios (informativos, variam entre execuções):")
print(tempos_arvore.set_index("Valor").drop(columns="Modelo").to_string(float_format=lambda valor: f"{valor:.4f}"))

# %% [markdown]
# ### BLOCO 68 — Critério de seleção
#
# O critério foi definido antes de qualquer métrica de teste. Como o enunciado escolhe o modelo final pela acurácia, o
# critério principal é a **maior acurácia média na validação cruzada**. Empates numéricos efetivos (diferença menor que
# 10⁻¹²) são resolvidos, nesta ordem, pela maior balanced accuracy média, pelo maior F1 médio da classe ocupada e,
# por fim, pela configuração mais simples: no KNN, o maior K (vizinhança mais ampla, fronteira mais suave); na árvore,
# a menor profundidade (`None` é a menos restrita). O desvio padrão entre folds e a diferença entre treino e validação
# são registrados como informação complementar, sem entrar no critério. K e `max_depth` são selecionados
# independentemente. A acurácia de treino não é usada para escolher.

# %%
# BLOCO 68 — Critério de seleção (definido antes de qualquer métrica de teste)

assert not teste_consultado_fase6, "O teste não pode ser consultado antes da seleção."
tolerancia_empate = 1e-12
criterios_selecao = ["Acurácia validação (média)", "Balanced accuracy validação (média)",
                     "F1 ocupado validação (média)"]


def selecionar_configuracao(resumo, ordem_simplicidade):
    """Aplica os critérios em sequência; empates numéricos passam ao critério seguinte e, por fim, à simplicidade."""
    candidatos = resumo.copy()
    etapas = []
    for criterio in criterios_selecao:
        melhor = candidatos[criterio].max()
        candidatos = candidatos[candidatos[criterio] >= melhor - tolerancia_empate]
        etapas.append(f"{criterio}: {len(candidatos)} candidata(s)")
    escolhido = min(candidatos.index, key=ordem_simplicidade)
    return escolhido, etapas


# Simplicidade: no KNN, maior K (vizinhança mais ampla); na árvore, menor profundidade (None é a menos restrita)
k_selecionado_rotulo, etapas_knn = selecionar_configuracao(resumo_knn, lambda rotulo: -int(rotulo))
profundidade_selecionada_rotulo, etapas_arvore = selecionar_configuracao(
    resumo_arvore, lambda rotulo: np.inf if rotulo == "None" else int(rotulo)
)
k_selecionado = int(k_selecionado_rotulo)
profundidade_selecionada = None if profundidade_selecionada_rotulo == "None" else int(profundidade_selecionada_rotulo)

# Registro da seleção: congelado antes de qualquer consulta ao teste
selecao_hiperparametros = {"n_neighbors": k_selecionado, "max_depth": profundidade_selecionada,
                           "atributos": tuple(atributos_preditores)}
registro_selecao = repr(selecao_hiperparametros)

for nome, resumo, etapas, escolhido in [("KNN", resumo_knn, etapas_knn, k_selecionado_rotulo),
                                        ("Árvore de Decisão", resumo_arvore, etapas_arvore,
                                         profundidade_selecionada_rotulo)]:
    ordenado = resumo["Acurácia validação (média)"].sort_values(ascending=False)
    segunda = ordenado.drop(escolhido).iloc[0] if ordenado.index[0] == escolhido else ordenado.iloc[0]
    print(f"{nome}: selecionado = {escolhido} | " + " → ".join(etapas))
    print(f"  Acurácia média de validação: {resumo.loc[escolhido, 'Acurácia validação (média)']:.6f} | "
          f"diferença para a próxima configuração: "
          f"{(resumo.loc[escolhido, 'Acurácia validação (média)'] - segunda) * 100:.4f} p.p. "
          f"(≈ {(resumo.loc[escolhido, 'Acurácia validação (média)'] - segunda) * media_validacao_fold:.2f} "
          f"registros por fold)")
print("\nSeleção registrada:", registro_selecao)

# %% [markdown]
# ### BLOCO 69 — Avaliação da variável derivada
#
# Comparação feita **somente no treino**, com os mesmos folds, os mesmos hiperparâmetros e o mesmo protocolo:
#
# - **Configuração A:** 5 atributos originais (`Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio`).
# - **Configuração B:** os 5 atributos mais `CO2_Light_Ratio` (resultados dos BLOCOS 66 e 67).
#
# No KNN, o Pipeline ajusta um scaler próprio em cada fold para cada configuração; a árvore usa dados originais. As
# diferenças são pareadas por fold (B − A), pois as duas configurações são avaliadas nos mesmos registros. Os modelos
# principais continuam com 6 atributos, para manter a variável da Fase 3 no fluxo exigido; a configuração A é um
# experimento complementar e a coluna não é removida da base. Esta comparação usa as mesmas validações que selecionaram
# os hiperparâmetros e, portanto, não é uma estimativa independente do desempenho.

# %%
# BLOCO 69 — Avaliação da variável derivada (5 × 6 atributos, somente no treino)

assert not teste_consultado_fase6, "O teste não pode ser consultado na comparação de atributos."
atributos_originais = [atributo for atributo in atributos_preditores if atributo != "CO2_Light_Ratio"]
assert atributos_originais == ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

registros_knn_5, _, _ = avaliar_validacao_cruzada(criar_pipeline_knn, valores_k, "KNN", "n_neighbors",
                                                  atributos_originais)
registros_arvore_5, _, _ = avaliar_validacao_cruzada(criar_arvore, valores_max_depth, "Árvore de Decisão",
                                                     "max_depth", atributos_originais)
registros_feature_engineering = pd.concat(
    [registros_knn_5, registros_knn, registros_arvore_5, registros_arvore], ignore_index=True
)

# Diferenças pareadas por fold (6 atributos − 5 atributos), somente na validação
chaves = ["modelo", "valor_hiperparametro", "fold", "metrica"]
validacao_fe = registros_feature_engineering.query("conjunto == 'validacao'")
pareado = (
    validacao_fe.pivot_table(index=chaves, columns="quantidade_atributos", values="resultado", aggfunc="first")
    .rename(columns={5: "5 atributos", 6: "6 atributos"})
)
pareado["Diferença (6 − 5)"] = pareado["6 atributos"] - pareado["5 atributos"]

resumo_feature_engineering = pareado.groupby(["modelo", "valor_hiperparametro", "metrica"]).agg(
    media_5=("5 atributos", "mean"),
    media_6=("6 atributos", "mean"),
    diferenca_media=("Diferença (6 − 5)", "mean"),
    diferenca_dp=("Diferença (6 − 5)", lambda serie: serie.std(ddof=0)),
    folds_6_melhor=("Diferença (6 − 5)", lambda serie: int((serie > 0).sum())),
    folds_5_melhor=("Diferença (6 − 5)", lambda serie: int((serie < 0).sum())),
    folds_empate=("Diferença (6 − 5)", lambda serie: int((serie == 0).sum())),
).reset_index()
ordem_configuracoes = (
    [("KNN", str(k), metrica) for k in valores_k for metrica in metricas_validacao]
    + [("Árvore de Decisão", rotulo_profundidade(valor), metrica) for valor in valores_max_depth
       for metrica in metricas_validacao]
)
resumo_feature_engineering = resumo_feature_engineering.set_index(
    ["modelo", "valor_hiperparametro", "metrica"]
).loc[ordem_configuracoes]
resumo_feature_engineering[["diferenca_media", "diferenca_dp"]] *= 100

print("Comparação na validação cruzada: 5 atributos (sem CO2_Light_Ratio) × 6 atributos")
print("(médias como proporção; diferença e desvio da diferença em pontos percentuais)")
print(resumo_feature_engineering.to_string(float_format=lambda valor: f"{valor:.4f}"))

# KNN no fold 1 (K selecionado): a nova coluna altera vizinhos, mas é preciso verificar se altera os votos
posicoes_treino_f1, posicoes_validacao_f1 = folds_cv[0]
vizinhos_por_configuracao, previsoes_por_configuracao = {}, {}
for quantidade, colunas in [(6, atributos_preditores), (5, atributos_originais)]:
    pipeline_comparacao = criar_pipeline_knn(k_selecionado).fit(
        X_train.iloc[posicoes_treino_f1][colunas], y_train.iloc[posicoes_treino_f1]
    )
    validacao_comparacao = X_train.iloc[posicoes_validacao_f1][colunas]
    vizinhos_por_configuracao[quantidade] = pipeline_comparacao.named_steps["knn"].kneighbors(
        pipeline_comparacao.named_steps["scaler"].transform(validacao_comparacao), return_distance=False
    )
    previsoes_por_configuracao[quantidade] = pipeline_comparacao.predict(validacao_comparacao)
vizinhos_diferentes = np.mean([
    set(vizinhos_6) != set(vizinhos_5)
    for vizinhos_6, vizinhos_5 in zip(vizinhos_por_configuracao[6], vizinhos_por_configuracao[5])
]) * 100
previsoes_diferentes = int((previsoes_por_configuracao[6] != previsoes_por_configuracao[5]).sum())
print(f"\nKNN (K={k_selecionado}), fold 1: registros de validação com conjunto de vizinhos diferente = "
      f"{vizinhos_diferentes:.2f}% | previsões diferentes = {previsoes_diferentes} de {len(posicoes_validacao_f1)}")

for modelo, rotulo in [("KNN", k_selecionado_rotulo), ("Árvore de Decisão", profundidade_selecionada_rotulo)]:
    print(f"\n{modelo} — configuração selecionada ({rotulo}), diferenças por fold (p.p.):")
    print((pareado.loc[(modelo, rotulo)]["Diferença (6 − 5)"].unstack("metrica") * 100).to_string(
        float_format=lambda valor: f"{valor:.4f}"))

# %% [markdown]
# ### BLOCO 70 — Proximidade temporal entre treino e validação nos folds
#
# Para cada registro de validação de cada fold, calcula-se a menor distância temporal até um registro da parte de
# ajuste do mesmo fold, dentro do mesmo segmento contínuo (mesma função de busca binária do BLOCO 44, conferida contra
# aquela auditoria e por força bruta em 100 registros). Diferentemente da Fase 4, a parte de ajuste de cada fold
# contém cerca de 64% da base (80% × 80%) e os registros do teste oficial não participam, de modo que o vizinho
# imediato pode estar no teste. Por isso, a comparação usa limites de tempo (61 e 120 segundos). Os folds não são
# alterados após esta medição.

# %%
# BLOCO 70 — Proximidade temporal entre treino e validação nos folds


def distancia_temporal_minima(rotulos_referencia, rotulos_avaliados):
    """Menor distância (s) de cada registro avaliado a um registro de referência do mesmo segmento."""
    distancias = pd.Series(np.inf, index=pd.Index(rotulos_avaliados), dtype="float64")
    datas = rastreio_temporal["date"]
    segmentos_registro = rastreio_temporal["segmento"]
    referencia = pd.Index(rotulos_referencia)
    for segmento_id in segmentos_registro.loc[distancias.index].unique():
        avaliados_seg = distancias.index[segmentos_registro.loc[distancias.index].eq(segmento_id).to_numpy()]
        referencia_seg = referencia[segmentos_registro.loc[referencia].eq(segmento_id).to_numpy()]
        if len(referencia_seg) == 0:
            continue
        tempos_ref = np.sort(datas.loc[referencia_seg].to_numpy())
        tempos_av = datas.loc[avaliados_seg].to_numpy()
        insercao = np.searchsorted(tempos_ref, tempos_av)
        anterior = np.where(insercao > 0,
                            (tempos_av - tempos_ref[np.clip(insercao - 1, 0, None)]) / np.timedelta64(1, "s"),
                            np.inf)
        posterior = np.where(insercao < len(tempos_ref),
                             (tempos_ref[np.clip(insercao, None, len(tempos_ref) - 1)] - tempos_av)
                             / np.timedelta64(1, "s"), np.inf)
        distancias.loc[avaliados_seg] = np.minimum(anterior, posterior)
    return distancias


def resumir_distancias(distancias):
    finitas = distancias[np.isfinite(distancias)]
    return {
        "Registros avaliados": len(distancias),
        "Com vizinho elegível": len(finitas),
        "Até 61 s (%)": (distancias <= 61).mean() * 100,
        "Até 120 s (%)": (distancias <= 120).mean() * 100,
        "Mediana (s)": finitas.median(),
        "Média (s)": finitas.mean(),
        "Máximo (s)": finitas.max(),
    }


# Conferência da função com a auditoria da Fase 4 (teste × treino oficial)
assert distancia_temporal_minima(X_train.index, X_test.index).equals(distancia_treino_s.rename(None)), \
    "Função divergente da auditoria do BLOCO 44."

distancias_folds = {}
linhas_proximidade_folds = []
for numero_fold, (posicoes_treino, posicoes_validacao) in enumerate(folds_cv, start=1):
    rotulos_treino_fold = X_train.index[posicoes_treino]
    rotulos_validacao_fold = X_train.index[posicoes_validacao]
    distancias_folds[numero_fold] = distancia_temporal_minima(rotulos_treino_fold, rotulos_validacao_fold)
    linhas_proximidade_folds.append({"Comparação": f"Fold {numero_fold}: validação × treino do fold",
                                     **resumir_distancias(distancias_folds[numero_fold])})

# Validação por força bruta em 100 registros de validação do fold 1
amostra_fold = pd.Series(X_train.index[folds_cv[0][1]]).sample(n=100, random_state=42)
rotulos_treino_fold1 = X_train.index[folds_cv[0][0]]
for indice in amostra_fold:
    candidatos = rotulos_treino_fold1[
        rastreio_temporal.loc[rotulos_treino_fold1, "segmento"].eq(rastreio_temporal.loc[indice, "segmento"]).to_numpy()
    ]
    menor = (rastreio_temporal.loc[candidatos, "date"] - rastreio_temporal.loc[indice, "date"]).abs().min()
    assert menor / pd.Timedelta(seconds=1) == distancias_folds[1].loc[indice], f"Divergência no índice {indice}."

distancias_todos_folds = pd.concat(distancias_folds.values())
linhas_proximidade_folds.append({"Comparação": "Cinco folds (todas as validações)",
                                 **resumir_distancias(distancias_todos_folds)})
linhas_proximidade_folds.append({"Comparação": "Fase 4: teste oficial × treino oficial",
                                 **resumir_distancias(distancia_treino_s)})
tabela_proximidade_folds = pd.DataFrame(linhas_proximidade_folds).set_index("Comparação")
print("Validação por força bruta em 100 registros do fold 1: aprovada.")
print(tabela_proximidade_folds.to_string(float_format=lambda valor: f"{valor:.2f}"))

# %% [markdown]
# ### BLOCO 71 — Validação temporal complementar
#
# Protocolo definido antes da avaliação, usando **somente os 9.933 registros do treino oficial**:
#
# | Item | Definição |
# |---|---|
# | Ordenação | Cronológica, pela coluna `date` de `rastreio_temporal` (equivale à ordem dos índices) |
# | Divisões | `TimeSeriesSplit(n_splits=5, gap=60)`: janelas expansivas, em que o treino de cada divisão contém todos os registros anteriores e a validação é o bloco seguinte de 1.655 registros |
# | Intervalo | 60 **registros** do treino oficial descartados entre ajuste e validação. Como o teste oficial foi retirado da sequência, 60 registros correspondem a cerca de 75 minutos reais; a duração efetiva de cada intervalo é registrada na tabela |
# | Lacuna entre segmentos | Os 7 dias sem registros entre os dois arquivos não são tratados como observações; o fold 1 atravessa essa lacuna, e sua duração aparece nas datas, não na contagem de registros |
# | Classes | O treino de cada divisão deve conter as duas classes (verificação bloqueante). Se uma janela de validação tiver uma única classe, a acurácia é calculada, e o F1 e a balanced accuracy são registrados como **não calculáveis**, sem excluir a divisão |
# | Modelos | Somente as configurações selecionadas no BLOCO 68, com 6 atributos, sem alterá-las após os resultados |
#
# O objetivo é comparar a avaliação aleatória com uma avaliação que respeita a anterioridade temporal, sem substituir
# o protocolo acadêmico. `X_test` e `y_test` não são usados. Como referência descritiva, calcula-se a acurácia de
# prever sempre a classe majoritária do treino de cada divisão.

# %%
# BLOCO 71 — Validação temporal complementar (somente com o treino oficial)

assert not teste_consultado_fase6, "O teste não pode ser consultado na validação temporal."

# Ordem cronológica dos registros do treino oficial (os índices seguem a ordem temporal da base)
ordem_cronologica = rastreio_temporal.loc[X_train.index, "date"].sort_values(kind="stable").index
assert ordem_cronologica.equals(X_train.index.sort_values()), "Ordem temporal diferente da ordem dos índices."
X_train_cronologico = X_train.loc[ordem_cronologica]
y_train_cronologico = y_train.loc[ordem_cronologica]
datas_treino_cronologico = rastreio_temporal.loc[ordem_cronologica, "date"]
segmentos_treino_cronologico = rastreio_temporal.loc[ordem_cronologica, "segmento"]

# Protocolo definido antes da avaliação
quantidade_divisoes_temporais = 5
intervalo_registros = 60
divisoes_temporais = TimeSeriesSplit(n_splits=quantidade_divisoes_temporais, gap=intervalo_registros)
folds_temporais = list(divisoes_temporais.split(X_train_cronologico))


def formatar_data(data):
    return pd.Timestamp(data).strftime("%d/%m %H:%M")


nomes_dias = {0: "seg", 1: "ter", 2: "qua", 3: "qui", 4: "sex", 5: "sáb", 6: "dom"}


def dias_semana(datas):
    """Dias da semana presentes, na ordem cronológica de ocorrência."""
    return "-".join(dict.fromkeys(nomes_dias[dia] for dia in datas.dt.dayofweek))


linhas_temporais = []
for numero_fold, (posicoes_treino, posicoes_validacao) in enumerate(folds_temporais, start=1):
    assert posicoes_treino.max() + intervalo_registros < posicoes_validacao.min(), "Anterioridade violada."
    datas_t = datas_treino_cronologico.iloc[posicoes_treino]
    datas_v = datas_treino_cronologico.iloc[posicoes_validacao]
    y_t = y_train_cronologico.iloc[posicoes_treino]
    y_v = y_train_cronologico.iloc[posicoes_validacao]
    linhas_temporais.append({
        "Fold": numero_fold,
        "Treino (registros)": len(posicoes_treino),
        "Treino: início": formatar_data(datas_t.iloc[0]),
        "Treino: fim": formatar_data(datas_t.iloc[-1]),
        "Treino % classe 1": y_t.mean() * 100,
        "Treino: registros de fim de semana": int((datas_t.dt.dayofweek >= 5).sum()),
        "Intervalo (registros)": intervalo_registros,
        "Intervalo real": str(datas_v.iloc[0] - datas_t.iloc[-1]),
        "Validação (registros)": len(posicoes_validacao),
        "Validação: início": formatar_data(datas_v.iloc[0]),
        "Validação: fim": formatar_data(datas_v.iloc[-1]),
        "Dias na validação": dias_semana(datas_v),
        "Validação classe 0": int((y_v == 0).sum()),
        "Validação classe 1": int((y_v == 1).sum()),
        "Validação % classe 1": y_v.mean() * 100,
        "Segmentos na validação": ", ".join(map(str, sorted(segmentos_treino_cronologico.iloc[posicoes_validacao]
                                                           .unique()))),
        "Validação até 61 s do treino (%)": (distancia_temporal_minima(
            datas_t.index, datas_v.index) <= 61).mean() * 100,
    })
tabela_folds_temporais = pd.DataFrame(linhas_temporais).set_index("Fold")
print(f"TimeSeriesSplit(n_splits={quantidade_divisoes_temporais}, gap={intervalo_registros}) sobre os "
      f"{len(X_train_cronologico)} registros do treino oficial em ordem cronológica:")
print(tabela_folds_temporais.to_string(float_format=lambda valor: f"{valor:.2f}"))

# Avaliação das configurações selecionadas no BLOCO 68 (sem alterá-las)
modelos_temporais = {
    "KNN": (lambda: criar_pipeline_knn(k_selecionado), "n_neighbors", k_selecionado_rotulo),
    "Árvore de Decisão": (lambda: criar_arvore(profundidade_selecionada), "max_depth",
                          profundidade_selecionada_rotulo),
}
rotulo_protocolo_temporal = (f"TimeSeriesSplit(n_splits={quantidade_divisoes_temporais}, "
                             f"gap={intervalo_registros}) cronológico no treino oficial")
registros_temporais = []
diagnostico_classe_unica = []
for numero_fold, (posicoes_treino, posicoes_validacao) in enumerate(folds_temporais, start=1):
    X_t, y_t = X_train_cronologico.iloc[posicoes_treino], y_train_cronologico.iloc[posicoes_treino]
    X_v, y_v = X_train_cronologico.iloc[posicoes_validacao], y_train_cronologico.iloc[posicoes_validacao]
    assert set(y_t.unique()) == {0, 1}, f"Treino do fold temporal {numero_fold} sem as duas classes."
    for modelo, (fabrica, hiperparametro, rotulo) in modelos_temporais.items():
        estimador = fabrica().fit(X_t, y_t)
        for conjunto, X_av, y_av in [("treino", X_t, y_t), ("validacao", X_v, y_v)]:
            previsto = estimador.predict(X_av)
            if conjunto == "validacao" and y_av.nunique() == 1:
                falsos = X_av.loc[previsto != y_av.to_numpy(), "Light"]
                diagnostico_classe_unica.append({
                    "Fold": numero_fold, "Modelo": modelo, "Classe presente": int(y_av.iloc[0]),
                    "Registros": len(y_av), "Previsões erradas": len(falsos),
                    "Light > 0 na janela (%)": (X_av["Light"] > 0).mean() * 100,
                    "Light dos erros: mediana (lux)": falsos.median(),
                    "Light dos erros: máximo (lux)": falsos.max(),
                })
            calculavel = set(y_av.unique()) == {0, 1}
            valores_metricas = {
                "acuracia": accuracy_score(y_av, previsto),
                "f1_ocupado": f1_score(y_av, previsto, zero_division=0.0) if calculavel else np.nan,
                "balanced_accuracy": balanced_accuracy_score(y_av, previsto) if calculavel else np.nan,
            }
            for metrica, valor in valores_metricas.items():
                registros_temporais.append({
                    "protocolo": rotulo_protocolo_temporal,
                    "modelo": modelo,
                    "hiperparametro": hiperparametro,
                    "valor_hiperparametro": rotulo,
                    "configuracao_atributos": "6 atributos",
                    "quantidade_atributos": 6,
                    "fold": numero_fold,
                    "conjunto": conjunto,
                    "metrica": metrica,
                    "resultado": valor,
                    "observacoes_avaliadas": len(y_av),
                    "observacoes_ajuste": len(y_t),
                    "intervalo_registros": intervalo_registros,
                    "intervalo_real_s": (datas_treino_cronologico.iloc[posicoes_validacao[0]]
                                         - datas_treino_cronologico.iloc[posicoes_treino[-1]]).total_seconds(),
                    "inicio_validacao": datas_treino_cronologico.iloc[posicoes_validacao[0]],
                    "fim_validacao": datas_treino_cronologico.iloc[posicoes_validacao[-1]],
                    "validacao_classe_1": int(y_v.sum()),
                    "observacao": "calculável" if calculavel else "não calculável: avaliação sem as duas classes",
                })
registros_temporais = pd.DataFrame(registros_temporais)
tabela_diagnostico_classe_unica = pd.DataFrame(diagnostico_classe_unica)

resultados_temporais = registros_temporais.query("conjunto == 'validacao'").pivot_table(
    index="fold", columns=["modelo", "metrica"], values="resultado", aggfunc="first", dropna=False
)
print("\nResultados na validação temporal (por fold):")
print(resultados_temporais.to_string(float_format=lambda valor: f"{valor:.6f}"))

# Referência descritiva: proporção da classe majoritária do treino em cada janela de validação
referencia_majoritaria_temporal = pd.Series({
    numero_fold: (y_train_cronologico.iloc[posicoes_validacao] == y_train_cronologico.iloc[posicoes_treino].mode()[0])
    .mean()
    for numero_fold, (posicoes_treino, posicoes_validacao) in enumerate(folds_temporais, start=1)
}, name="Acurácia de prever sempre a classe majoritária do treino")
print("\n" + referencia_majoritaria_temporal.to_frame().to_string(float_format=lambda valor: f"{valor:.6f}"))

if len(tabela_diagnostico_classe_unica):
    print("\nJanelas de validação com uma única classe (F1 e balanced accuracy não calculáveis):")
    print(tabela_diagnostico_classe_unica.set_index(["Fold", "Modelo"]).to_string(
        float_format=lambda valor: f"{valor:.2f}"))

# Comparação descritiva entre protocolos (configurações selecionadas)
linhas_comparacao_protocolos = []
for modelo, rotulo, resumo in [("KNN", k_selecionado_rotulo, resumo_knn),
                               ("Árvore de Decisão", profundidade_selecionada_rotulo, resumo_arvore)]:
    validacao_temporal = registros_temporais.query("modelo == @modelo and conjunto == 'validacao'")
    for metrica, coluna in [("acuracia", "Acurácia validação"), ("f1_ocupado", "F1 ocupado validação"),
                            ("balanced_accuracy", "Balanced accuracy validação")]:
        valores_temporais = validacao_temporal.query("metrica == @metrica")["resultado"]
        linhas_comparacao_protocolos.append({
            "Modelo": modelo, "Métrica": metrica,
            "Aleatória estratificada (média)": resumo.loc[rotulo, f"{coluna} (média)"],
            "Aleatória estratificada (dp)": resumo.loc[rotulo, f"{coluna} (dp)"],
            "Temporal (média dos folds calculáveis)": valores_temporais.mean(),
            "Temporal (dp)": valores_temporais.std(ddof=0),
            "Temporal (mínimo)": valores_temporais.min(),
            "Folds temporais calculáveis": int(valores_temporais.notna().sum()),
        })
tabela_comparacao_protocolos = pd.DataFrame(linhas_comparacao_protocolos).set_index(["Modelo", "Métrica"])
print("\nComparação descritiva entre protocolos:")
print(tabela_comparacao_protocolos.to_string(float_format=lambda valor: f"{valor:.6f}"))

# %% [markdown]
# ### BLOCO 72 — Consolidação e ajuste dos modelos selecionados
#
# Os modelos principais são instanciados com os hiperparâmetros congelados no BLOCO 68 e ajustados **somente** em
# `X_train` e `y_train`: `modelo_knn_final` (Pipeline com `StandardScaler` e `KNeighborsClassifier`) e
# `modelo_arvore_final` (`DecisionTreeClassifier` com `random_state=42`, dados não escalonados). Os objetos ficam em
# memória para a Fase 7, sem serialização. Como o scaler interno é ajustado nos mesmos 9.933 registros do scaler da
# Fase 5, seus parâmetros devem coincidir com os daquele scaler, embora sejam objetos diferentes.

# %%
# BLOCO 72 — Consolidação e ajuste dos modelos selecionados

assert repr(selecao_hiperparametros) == registro_selecao, "Seleção alterada após o registro."
assert not teste_consultado_fase6, "Os modelos finais devem ser definidos antes da consulta ao teste."

modelo_knn_final = criar_pipeline_knn(k_selecionado).fit(X_train, y_train)
modelo_arvore_final = criar_arvore(profundidade_selecionada).fit(X_train, y_train)

check_is_fitted(modelo_knn_final)
check_is_fitted(modelo_arvore_final)
scaler_knn_final = modelo_knn_final.named_steps["scaler"]
knn_final = modelo_knn_final.named_steps["knn"]
assert int(scaler_knn_final.n_samples_seen_) == knn_final.n_samples_fit_ == len(X_train) == 9933
assert list(modelo_knn_final.feature_names_in_) == list(modelo_arvore_final.feature_names_in_) == atributos_preditores
assert scaler_knn_final is not scaler, "Scaler da Fase 5 reutilizado no modelo final."

tabela_knn_final = pd.Series({
    "n_neighbors": knn_final.n_neighbors,
    "weights": knn_final.weights,
    "Métrica de distância": knn_final.effective_metric_,
    "Amostras de treinamento (n_samples_fit_)": knn_final.n_samples_fit_,
    "Scaler interno: observações": int(scaler_knn_final.n_samples_seen_),
    "Scaler interno igual ao da Fase 5 (mesmos dados)": bool(
        np.allclose(scaler_knn_final.mean_, scaler.mean_, rtol=tolerancia_parametros, atol=0)
        and np.allclose(scaler_knn_final.scale_, scaler.scale_, rtol=tolerancia_parametros, atol=0)
    ),
})
tabela_arvore_final = pd.Series({
    "max_depth configurado": rotulo_profundidade(modelo_arvore_final.max_depth),
    "Profundidade efetiva": modelo_arvore_final.get_depth(),
    "Nós": modelo_arvore_final.tree_.node_count,
    "Folhas": modelo_arvore_final.get_n_leaves(),
    "criterion": modelo_arvore_final.criterion,
    "random_state": modelo_arvore_final.random_state,
    "Amostras de treinamento": int(modelo_arvore_final.tree_.n_node_samples[0]),
})
print("Modelo KNN final (Pipeline):")
print(tabela_knn_final.to_string())
print("\nParâmetros do scaler interno:")
print(pd.DataFrame({"mean_": scaler_knn_final.mean_, "scale_": scaler_knn_final.scale_},
                   index=atributos_preditores).to_string(float_format=lambda valor: f"{valor:.6g}"))
print("\nÁrvore de Decisão final:")
print(tabela_arvore_final.to_string())

importancias_arvore = pd.Series(modelo_arvore_final.feature_importances_, index=atributos_preditores,
                                name="Importância (redução de impureza)").sort_values(ascending=False)
print("\n" + importancias_arvore.to_frame().to_string(float_format=lambda valor: f"{valor:.6f}"))

# %% [markdown]
# ### BLOCO 73 — Tabela acadêmica de acurácia no treino e no teste
#
# O enunciado exige registrar a acurácia **no treino e no teste** de cada configuração. Para atender a exigência sem
# comprometer a seleção:
#
# 1. a seleção do BLOCO 68 foi registrada (`registro_selecao`) antes desta célula e é conferida antes e depois dela;
# 2. cada configuração é ajustada somente em `X_train` e avaliada uma vez em `X_train` e em `X_test`;
# 3. as acurácias de teste são **resultados descritivos posteriores à seleção**: nenhum hiperparâmetro, atributo ou
#    semente é alterado depois desta consulta, e a maior acurácia de teste não é critério de escolha;
# 4. a partir daqui, o teste foi consultado (`teste_consultado_fase6 = True`); a Fase 7 não será a primeira consulta.
#
# Apenas a acurácia é calculada no teste, como pede a Fase 6; as demais métricas de teste ficam para a Fase 7.

# %%
# BLOCO 73 — Tabela acadêmica de acurácia no treino e no teste (exigência do enunciado)

# A seleção do BLOCO 68 permanece congelada: as métricas abaixo são descritivas e posteriores à seleção
assert repr(selecao_hiperparametros) == registro_selecao, "Seleção alterada antes da consulta ao teste."
teste_consultado_fase6 = True  # a partir daqui o teste foi consultado (exigência do enunciado)

linhas_treino_teste = []
for modelo, fabrica, valores, hiperparametro, selecionado in [
    ("KNN", criar_pipeline_knn, valores_k, "n_neighbors", k_selecionado_rotulo),
    ("Árvore de Decisão", criar_arvore, valores_max_depth, "max_depth", profundidade_selecionada_rotulo),
]:
    resumo = resumo_knn if modelo == "KNN" else resumo_arvore
    for valor in valores:
        rotulo = rotulo_profundidade(valor)
        estimador = fabrica(valor).fit(X_train, y_train)
        acuracia_treino = accuracy_score(y_train, estimador.predict(X_train))
        acuracia_teste = accuracy_score(y_test, estimador.predict(X_test))
        linhas_treino_teste.append({
            "Modelo": modelo, "Hiperparâmetro": f"{hiperparametro}={rotulo}",
            "Acurácia treino": acuracia_treino,
            "Acurácia teste": acuracia_teste,
            "Diferença treino − teste (p.p.)": (acuracia_treino - acuracia_teste) * 100,
            "Acurácia validação cruzada (média)": resumo.loc[rotulo, "Acurácia validação (média)"],
            "Selecionado pela validação cruzada": rotulo == selecionado,
            "Profundidade efetiva": estimador.get_depth() if modelo != "KNN" else np.nan,
        })
tabela_treino_teste = pd.DataFrame(linhas_treino_teste).set_index(["Modelo", "Hiperparâmetro"])
print(tabela_treino_teste.to_string(float_format=lambda valor: f"{valor:.6f}"))

# Os modelos selecionados reproduzem os modelos finais do BLOCO 72
assert np.array_equal(modelo_knn_final.predict(X_test),
                      criar_pipeline_knn(k_selecionado).fit(X_train, y_train).predict(X_test))
assert np.array_equal(modelo_arvore_final.predict(X_test),
                      criar_arvore(profundidade_selecionada).fit(X_train, y_train).predict(X_test))
assert repr(selecao_hiperparametros) == registro_selecao, "Seleção alterada após a consulta ao teste."
print("\nSeleção inalterada após a consulta ao teste:", registro_selecao)

# %% [markdown]
# ### BLOCO 74 — Visualização da comparação dos hiperparâmetros
#
# Quatro painéis: (1) e (2) acurácia média de treino e de validação cruzada por K e por `max_depth`, com o desvio
# padrão entre folds, os resultados individuais dos folds e, em linha tracejada, a acurácia de teste do BLOCO 73
# (descritiva, posterior à seleção); (3) diferença entre treino e validação cruzada e entre treino e teste; (4) desvio
# padrão das três métricas de validação entre os folds. Os eixos dos painéis 1 e 2 foram ampliados para mostrar
# diferenças pequenas, e a faixa de cada eixo está indicada no rótulo.

# %%
# BLOCO 74 — Visualização da comparação dos hiperparâmetros

cor_treino, cor_validacao, cor_teste = "#4C72B0", "#DD8452", "#55A868"


def formatar_eixo_percentual(eixo, casas=1):
    eixo.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda valor, _: f"{valor:.{casas}f}%".replace(".", ","))
    )


def painel_curvas(eixo, resumo, folds_validacao, rotulos, selecionado, titulo, rotulo_x, nome_modelo):
    posicoes = np.arange(len(rotulos))
    eixo.errorbar(posicoes - 0.06, resumo["Acurácia treino (média)"] * 100, yerr=resumo["Acurácia treino (dp)"] * 100,
                  marker="o", color=cor_treino, capsize=4, linewidth=2, label="Treino (média ± dp dos folds)")
    eixo.errorbar(posicoes + 0.06, resumo["Acurácia validação (média)"] * 100,
                  yerr=resumo["Acurácia validação (dp)"] * 100, marker="s", color=cor_validacao, capsize=4,
                  linewidth=2, label="Validação cruzada (média ± dp)")
    for coluna_fold in folds_validacao.columns:
        eixo.scatter(posicoes + 0.18, folds_validacao[coluna_fold] * 100, color=cor_validacao, s=12, alpha=0.5,
                     zorder=3, label="Validação: cada fold" if coluna_fold == 1 else None)
    teste = tabela_treino_teste.loc[nome_modelo, "Acurácia teste"].to_numpy() * 100
    eixo.plot(posicoes, teste, linestyle="--", marker="x", color=cor_teste, linewidth=1.5,
              label="Teste (descritivo, após a seleção)")
    posicao_selecionada = rotulos.index(selecionado)
    eixo.axvspan(posicao_selecionada - 0.35, posicao_selecionada + 0.35, color="gold", alpha=0.25,
                 label="Selecionado pela validação cruzada")
    limite_inferior = np.floor(min(resumo["Acurácia validação (média)"].min() * 100 - resumo[
        "Acurácia validação (dp)"].max() * 100, folds_validacao.min().min() * 100, teste.min()) * 2) / 2 - 0.25
    eixo.set_ylim(limite_inferior, 100.15)
    eixo.set_xticks(posicoes, rotulos)
    eixo.set_xlabel(rotulo_x)
    eixo.set_ylabel(f"Acurácia (eixo de {formatar_decimal(limite_inferior, 2)}% a 100%)")
    formatar_eixo_percentual(eixo, 2)
    eixo.set_title(titulo)
    eixo.legend(loc="lower left", fontsize=8.5)


rotulos_k = [str(k) for k in valores_k]
rotulos_profundidade = [rotulo_profundidade(valor) for valor in valores_max_depth]
rotulos_profundidade_grafico = [rotulo if rotulo != "None" else "None\n(sem limite)" for rotulo in rotulos_profundidade]

fig, eixos = plt.subplots(2, 2, figsize=(18, 13))
eixo_knn, eixo_arvore, eixo_diferenca, eixo_variabilidade = eixos.ravel()

painel_curvas(eixo_knn, resumo_knn, acuracia_validacao_knn_folds, rotulos_k, k_selecionado_rotulo,
              "KNN (Pipeline com StandardScaler): acurácia por K", "Número de vizinhos (K)", "KNN")
painel_curvas(eixo_arvore, resumo_arvore, acuracia_validacao_arvore_folds, rotulos_profundidade,
              profundidade_selecionada_rotulo, "Árvore de Decisão: acurácia por max_depth", "max_depth",
              "Árvore de Decisão")
eixo_arvore.set_xticks(np.arange(len(rotulos_profundidade)), rotulos_profundidade_grafico)

# Painel 3 — Diferença entre treino e validação (seleção) e entre treino e teste (descritivo)
rotulos_configuracoes = [f"K={rotulo}" for rotulo in rotulos_k] + [f"prof.={rotulo}" for rotulo in rotulos_profundidade]
diferencas_cv = np.concatenate([resumo_knn["Diferença treino − validação (p.p.)"],
                                resumo_arvore["Diferença treino − validação (p.p.)"]])
diferencas_teste = tabela_treino_teste["Diferença treino − teste (p.p.)"].to_numpy()
posicoes_configuracoes = np.arange(len(rotulos_configuracoes))
selecionados = [f"K={k_selecionado_rotulo}", f"prof.={profundidade_selecionada_rotulo}"]
barras = eixo_diferenca.bar(posicoes_configuracoes, diferencas_cv, color=cor_validacao, edgecolor="black",
                            label="Treino − validação cruzada (médias dos folds)")
for barra, rotulo in zip(barras, rotulos_configuracoes):
    if rotulo in selecionados:
        barra.set_linewidth(2.5)
        barra.set_hatch("//")
eixo_diferenca.scatter(posicoes_configuracoes + 0.3, diferencas_teste, marker="x", s=70, color=cor_teste, zorder=3,
                       label="Treino − teste (descritivo, após a seleção)")
for posicao, valor in zip(posicoes_configuracoes, diferencas_cv):
    eixo_diferenca.text(posicao - 0.05, valor + 0.02, formatar_decimal(valor, 2), ha="center", va="bottom", fontsize=9)
eixo_diferenca.axvline(len(valores_k) - 0.5, color="gray", linewidth=1)
eixo_diferenca.set_xticks(posicoes_configuracoes, rotulos_configuracoes, rotation=30)
eixo_diferenca.set_ylabel("Diferença de acurácia (pontos percentuais)")
eixo_diferenca.yaxis.set_major_formatter(plt.FuncFormatter(lambda valor, _: formatar_decimal(valor, 1)))
eixo_diferenca.set_ylim(0, max(diferencas_cv.max(), diferencas_teste.max()) * 1.2)
eixo_diferenca.set_title("Diferença entre treino e avaliação (hachurado = selecionado)")
eixo_diferenca.legend(loc="upper left", fontsize=9)

# Painel 4 — Variabilidade entre os folds (desvio padrão das métricas de validação)
largura = 0.26
for deslocamento, metrica, cor, rotulo in [(-largura, "Acurácia validação (dp)", cor_validacao, "Acurácia"),
                                           (0, "F1 ocupado validação (dp)", "#8172B3", "F1 da classe ocupada"),
                                           (largura, "Balanced accuracy validação (dp)", "#937860",
                                            "Balanced accuracy")]:
    valores_dp = np.concatenate([resumo_knn[metrica], resumo_arvore[metrica]]) * 100
    eixo_variabilidade.bar(posicoes_configuracoes + deslocamento, valores_dp, width=largura, color=cor,
                           edgecolor="black", label=rotulo)
eixo_variabilidade.axvline(len(valores_k) - 0.5, color="gray", linewidth=1)
eixo_variabilidade.set_xticks(posicoes_configuracoes, rotulos_configuracoes, rotation=30)
eixo_variabilidade.set_ylabel("Desvio padrão entre os 5 folds (pontos percentuais)")
eixo_variabilidade.yaxis.set_major_formatter(plt.FuncFormatter(lambda valor, _: formatar_decimal(valor, 1)))
eixo_variabilidade.set_title("Variabilidade das métricas de validação entre os folds")
eixo_variabilidade.legend(loc="upper left", fontsize=9)

plt.suptitle("Fase 6 — Comparação dos hiperparâmetros (validação cruzada estratificada de 5 folds no treino)",
             fontsize=16)
plt.tight_layout()
plt.savefig(caminho_grafico_hiperparametros, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_hiperparametros)
plt.show()

# %% [markdown]
# ### BLOCO 75 — Visualizações complementares: variável derivada e validação temporal
#
# A primeira figura compara 5 e 6 atributos: acurácia média de validação de cada configuração e diferenças pareadas por
# fold (6 − 5) na acurácia, no F1 da classe ocupada e na balanced accuracy. A segunda figura resume a validação
# temporal: a organização das divisões no tempo real (com a lacuna entre os segmentos e os fins de semana), as métricas
# por divisão comparadas com a média da validação aleatória, a proximidade temporal entre registros avaliados e de
# ajuste nos dois protocolos e na Fase 4, e a proporção de ocupados em cada janela.

# %%
# BLOCO 75 — Visualizações complementares: variável derivada e validação temporal

# Figura 2 — Impacto da CO2_Light_Ratio (5 × 6 atributos)
metricas_grafico = [("acuracia", "Acurácia"), ("f1_ocupado", "F1 da classe ocupada"),
                    ("balanced_accuracy", "Balanced accuracy")]
fig, eixos = plt.subplots(2, 2, figsize=(18, 12))
eixo_absoluto = eixos[0, 0]
eixos_diferencas = [eixos[0, 1], eixos[1, 0], eixos[1, 1]]

medias_acuracia = resumo_feature_engineering.xs("acuracia", level="metrica")
largura = 0.38
eixo_absoluto.bar(posicoes_configuracoes - largura / 2, medias_acuracia["media_5"] * 100, width=largura,
                  color="#8DA0CB", edgecolor="black", label="5 atributos (sem CO2_Light_Ratio)")
eixo_absoluto.bar(posicoes_configuracoes + largura / 2, medias_acuracia["media_6"] * 100, width=largura,
                  color=cor_validacao, edgecolor="black", label="6 atributos (com CO2_Light_Ratio)")
limite_absoluto = np.floor(medias_acuracia[["media_5", "media_6"]].min().min() * 100 * 4) / 4 - 0.25
eixo_absoluto.set_ylim(limite_absoluto, 100)
eixo_absoluto.axvline(len(valores_k) - 0.5, color="gray", linewidth=1)
eixo_absoluto.set_xticks(posicoes_configuracoes, rotulos_configuracoes, rotation=30)
eixo_absoluto.set_ylabel(f"Acurácia média de validação (eixo de {formatar_decimal(limite_absoluto, 2)}% a 100%)")
formatar_eixo_percentual(eixo_absoluto, 2)
eixo_absoluto.set_title("Acurácia média na validação cruzada")
eixo_absoluto.legend(loc="upper right", fontsize=9)

for eixo, (metrica, nome_metrica) in zip(eixos_diferencas, metricas_grafico):
    diferencas_folds = (
        pareado.xs(metrica, level="metrica")["Diferença (6 − 5)"].unstack("fold")
        .loc[[(modelo, valor) for modelo, valor, _ in ordem_configuracoes[::len(metricas_validacao)]]] * 100
    )
    for posicao, (_, linha) in enumerate(diferencas_folds.iterrows()):
        eixo.scatter(np.full(len(linha), posicao) + np.linspace(-0.15, 0.15, len(linha)), linha, s=22,
                     color="gray", alpha=0.7, zorder=3, label="Cada fold" if posicao == 0 else None)
    eixo.scatter(posicoes_configuracoes, diferencas_folds.mean(axis=1), marker="D", s=60, color="#C44E52",
                 zorder=4, label="Média dos 5 folds")
    eixo.axhline(0, color="black", linewidth=1)
    eixo.axvline(len(valores_k) - 0.5, color="gray", linewidth=1)
    amplitude = max(np.abs(diferencas_folds.to_numpy()).max(), 0.05) * 1.3
    eixo.set_ylim(-amplitude, amplitude)
    eixo.yaxis.set_major_formatter(plt.FuncFormatter(lambda valor, _: formatar_decimal(valor, 2)))
    eixo.set_xticks(posicoes_configuracoes, rotulos_configuracoes, rotation=30)
    eixo.set_ylabel("Diferença (6 − 5 atributos), pontos percentuais")
    eixo.set_title(f"{nome_metrica}: diferença pareada por fold")
    eixo.legend(loc="lower left", fontsize=9)

plt.suptitle("Fase 6 — Contribuição da CO2_Light_Ratio na validação cruzada (somente treino, mesmos folds)",
             fontsize=16)
plt.tight_layout()
plt.savefig(caminho_grafico_feature_engineering, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_feature_engineering)
plt.show()

# Figura 3 — Validação temporal complementar
fig, eixos = plt.subplots(2, 2, figsize=(18, 12))
eixo_linha_tempo, eixo_metricas_temporais, eixo_proximidade, eixo_composicao = eixos.ravel()

# Painel 1 — Organização das divisões cronológicas no tempo real
inicio_lacuna = rastreio_temporal.loc[rastreio_temporal["segmento"].eq(0), "date"].max()
fim_lacuna = rastreio_temporal.loc[rastreio_temporal["segmento"].eq(1), "date"].min()
eixo_linha_tempo.axvspan(inicio_lacuna, fim_lacuna, color="lightgray", alpha=0.6, label="Sem registros na base")
for dia in pd.date_range("2015-02-07", "2015-02-15"):
    if dia.dayofweek >= 5:
        eixo_linha_tempo.axvspan(dia, dia + pd.Timedelta(days=1), color="#C44E52", alpha=0.08,
                                 label="Fim de semana" if dia.dayofweek == 5 and dia.day == 7 else None)
for numero_fold, (posicoes_treino, posicoes_validacao) in enumerate(folds_temporais, start=1):
    inicio_t = datas_treino_cronologico.iloc[posicoes_treino[0]]
    fim_t = datas_treino_cronologico.iloc[posicoes_treino[-1]]
    inicio_v = datas_treino_cronologico.iloc[posicoes_validacao[0]]
    fim_v = datas_treino_cronologico.iloc[posicoes_validacao[-1]]
    eixo_linha_tempo.barh(numero_fold, fim_t - inicio_t, left=inicio_t, height=0.5, color=cor_treino,
                          label="Treino do fold" if numero_fold == 1 else None)
    eixo_linha_tempo.barh(numero_fold, inicio_v - fim_t, left=fim_t, height=0.5, color="black",
                          label="Intervalo de 60 registros" if numero_fold == 1 else None)
    eixo_linha_tempo.barh(numero_fold, fim_v - inicio_v, left=inicio_v, height=0.5, color=cor_validacao,
                          label="Validação do fold" if numero_fold == 1 else None)
eixo_linha_tempo.set_yticks(range(1, quantidade_divisoes_temporais + 1),
                            [f"Fold {numero}" for numero in range(1, quantidade_divisoes_temporais + 1)])
eixo_linha_tempo.invert_yaxis()
eixo_linha_tempo.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
eixo_linha_tempo.set_xlabel("Data (fevereiro de 2015)")
eixo_linha_tempo.set_title("Divisões cronológicas expansivas no treino oficial")
eixo_linha_tempo.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9, frameon=False)

# Painel 2 — Métricas por fold temporal × média da validação aleatória
posicoes_temporais = np.arange(1, quantidade_divisoes_temporais + 1)
for modelo, cor, deslocamento, rotulo, resumo in [("KNN", cor_treino, -0.1, k_selecionado_rotulo, resumo_knn),
                                                  ("Árvore de Decisão", "#55A868", 0.1, profundidade_selecionada_rotulo,
                                                   resumo_arvore)]:
    nome_curto = f"KNN (K={rotulo})" if modelo == "KNN" else f"Árvore (max_depth={rotulo})"
    acuracia_temporal = resultados_temporais[(modelo, "acuracia")] * 100
    balanceada_temporal = resultados_temporais[(modelo, "balanced_accuracy")] * 100
    eixo_metricas_temporais.plot(posicoes_temporais + deslocamento, acuracia_temporal, marker="o", color=cor,
                                 linewidth=2, label=f"{nome_curto}: acurácia temporal")
    eixo_metricas_temporais.scatter(posicoes_temporais + deslocamento, balanceada_temporal, marker="^", s=60,
                                    facecolor="white", edgecolor=cor, linewidth=2, zorder=3,
                                    label=f"{nome_curto}: balanced accuracy temporal")
    eixo_metricas_temporais.axhline(resumo.loc[rotulo, "Acurácia validação (média)"] * 100, color=cor,
                                    linestyle="--", linewidth=1.2,
                                    label=f"{nome_curto}: acurácia média na validação aleatória")
    # Rótulo apenas nos pontos com queda acentuada (os demais valores estão nas tabelas do BLOCO 71)
    for posicao, valor in zip(posicoes_temporais + deslocamento, acuracia_temporal):
        if valor < 90:
            eixo_metricas_temporais.annotate(f"{formatar_decimal(valor, 1)}%", (posicao, valor), xytext=(0, -16),
                                             textcoords="offset points", ha="center", fontsize=9, color=cor)
fold_sem_classe = [numero for numero, linha in tabela_folds_temporais.iterrows() if linha["Validação classe 1"] == 0]
for numero in fold_sem_classe:
    eixo_metricas_temporais.annotate("sem registros ocupados:\nF1 e balanced accuracy\nnão calculáveis",
                                     (numero, 60), ha="center", fontsize=9,
                                     bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray"})
eixo_metricas_temporais.set_ylim(40, 102)
eixo_metricas_temporais.set_xticks(posicoes_temporais, [f"Fold {numero}" for numero in posicoes_temporais])
eixo_metricas_temporais.set_ylabel("Métrica na validação (eixo de 40% a 102%)")
formatar_eixo_percentual(eixo_metricas_temporais, 0)
eixo_metricas_temporais.set_title("Validação temporal × validação aleatória (configurações selecionadas)")
eixo_metricas_temporais.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5,
                               frameon=False)

# Painel 3 — Proximidade temporal entre avaliação e ajuste
rotulos_proximidade = ([f"Aleatória\nfold {numero}" for numero in range(1, quantidade_folds + 1)]
                       + ["Fase 4\nteste"] + [f"Temporal\nfold {numero}" for numero in posicoes_temporais])
valores_61 = (list(tabela_proximidade_folds["Até 61 s (%)"].iloc[:quantidade_folds])
              + [tabela_proximidade_folds.loc["Fase 4: teste oficial × treino oficial", "Até 61 s (%)"]]
              + list(tabela_folds_temporais["Validação até 61 s do treino (%)"]))
cores_proximidade = [cor_validacao] * quantidade_folds + [cor_teste] + [cor_treino] * quantidade_divisoes_temporais
barras = eixo_proximidade.bar(np.arange(len(rotulos_proximidade)), valores_61, color=cores_proximidade,
                              edgecolor="black")
for barra, valor in zip(barras, valores_61):
    eixo_proximidade.text(barra.get_x() + barra.get_width() / 2, valor + 1, formatar_percentual(valor),
                          ha="center", fontsize=9)
eixo_proximidade.set_xticks(np.arange(len(rotulos_proximidade)), rotulos_proximidade, fontsize=9)
eixo_proximidade.set_ylim(0, 110)
eixo_proximidade.set_ylabel("Registros avaliados com vizinho de ajuste a até 61 s (%)")
formatar_eixo_percentual(eixo_proximidade, 0)
eixo_proximidade.set_title("Proximidade temporal entre os registros avaliados e os de ajuste")

# Painel 4 — Composição das janelas temporais
eixo_composicao.bar(posicoes_temporais - 0.2, tabela_folds_temporais["Treino % classe 1"], width=0.4,
                    color=cor_treino, edgecolor="black", label="Treino do fold")
eixo_composicao.bar(posicoes_temporais + 0.2, tabela_folds_temporais["Validação % classe 1"], width=0.4,
                    color=cor_validacao, edgecolor="black", label="Validação do fold")
for posicao, (_, linha) in zip(posicoes_temporais, tabela_folds_temporais.iterrows()):
    eixo_composicao.text(posicao - 0.2, linha["Treino % classe 1"] - 1.5,
                         formatar_inteiro(linha["Treino (registros)"]), ha="center", va="top", fontsize=8.5,
                         color="white", rotation=90)
    eixo_composicao.text(posicao + 0.2, linha["Validação % classe 1"] + 0.8, linha["Dias na validação"],
                         ha="center", va="bottom", fontsize=8.5, rotation=90)
eixo_composicao.axhline(y_train.mean() * 100, color="gray", linestyle=":", linewidth=1.5,
                        label="Treino oficial completo")
eixo_composicao.set_xticks(posicoes_temporais, [f"Fold {numero}" for numero in posicoes_temporais])
eixo_composicao.set_ylim(0, 50)
eixo_composicao.set_ylabel("Registros ocupados (%)")
formatar_eixo_percentual(eixo_composicao, 0)
eixo_composicao.set_title("Ocupação por janela (rótulos: registros de treino e dias da validação)")
eixo_composicao.legend(loc="upper right", fontsize=9)

fig.text(0.5, -0.045,
         "Limitação: os protocolos usam treinos de tamanhos diferentes (7.946–7.947 registros na validação aleatória; "
         "1.598 a 8.218 na temporal) e janelas com composições distintas;\nas diferenças não podem ser atribuídas "
         "exclusivamente à ordem temporal.", ha="center", fontsize=11)
plt.suptitle("Fase 6 — Validação temporal complementar no treino oficial (TimeSeriesSplit, 5 divisões, "
             "intervalo de 60 registros)", fontsize=16)
plt.tight_layout()
plt.savefig(caminho_grafico_validacao_temporal, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_validacao_temporal)
plt.show()

# %% [markdown]
# ### BLOCO 76 — Exportação dos resultados tabulares
#
# Os resultados por fold são exportados em formato longo (uma linha por modelo, valor do hiperparâmetro, configuração de
# atributos, fold, conjunto e métrica), sem arredondamento e sem tempos de execução, para que os arquivos sejam
# reproduzíveis byte a byte:
#
# | Arquivo | Conteúdo |
# |---|---|
# | `resultados/modelagem/cv_hiperparametros.csv` | Validação cruzada dos 5 valores de K e dos 4 de `max_depth` (6 atributos) |
# | `resultados/modelagem/cv_feature_engineering.csv` | Mesmas avaliações com 5 e com 6 atributos; as linhas de 6 atributos repetem, de forma intencional, as do arquivo anterior, para que a comparação fique completa em um único arquivo |
# | `resultados/modelagem/cv_temporal.csv` | Validação temporal das configurações selecionadas, com datas das janelas e duração real do intervalo; métricas não calculáveis ficam vazias e são identificadas na coluna `observacao` |
#
# A recarga usa `float_precision="round_trip"` e é comparada com os resultados em memória (`check_exact=True`).

# %%
# BLOCO 76 — Exportação dos resultados tabulares

colunas_registro = ["protocolo", "modelo", "hiperparametro", "valor_hiperparametro", "configuracao_atributos",
                    "quantidade_atributos", "fold", "conjunto", "metrica", "resultado", "observacoes_avaliadas",
                    "observacoes_ajuste"]
cv_hiperparametros = pd.concat([registros_knn, registros_arvore], ignore_index=True)[
    colunas_registro + ["semente_folds"]]
cv_feature_engineering = registros_feature_engineering[colunas_registro + ["semente_folds"]].reset_index(drop=True)
cv_temporal = registros_temporais.copy()
for coluna in ["inicio_validacao", "fim_validacao"]:
    cv_temporal[coluna] = cv_temporal[coluna].dt.strftime("%Y-%m-%d %H:%M:%S")

exportacoes = {
    caminho_cv_hiperparametros: cv_hiperparametros,
    caminho_cv_feature_engineering: cv_feature_engineering,
    caminho_cv_temporal: cv_temporal,
}
chaves_unicas = ["protocolo", "modelo", "valor_hiperparametro", "quantidade_atributos", "fold", "conjunto", "metrica"]
linhas_exportacao = []
for caminho, tabela in exportacoes.items():
    assert not tabela.duplicated(subset=chaves_unicas).any(), f"Linhas duplicadas em {caminho}."
    tabela.to_csv(caminho, index=False)
    recarregada = pd.read_csv(caminho, float_precision="round_trip", dtype={"valor_hiperparametro": str},
                              keep_default_na=False, na_values=[""])
    pd.testing.assert_frame_equal(recarregada, tabela, check_exact=True, check_dtype=False)
    with open(caminho, "rb") as arquivo:
        linhas_exportacao.append({
            "Arquivo": caminho, "Linhas": len(tabela), "Colunas": tabela.shape[1],
            "SHA-256": hashlib.sha256(arquivo.read()).hexdigest(),
        })

# O bloco de 6 atributos da comparação de atributos é idêntico aos resultados do BLOCO 66/67
pd.testing.assert_frame_equal(
    cv_feature_engineering.query("quantidade_atributos == 6").sort_values(chaves_unicas).reset_index(drop=True),
    cv_hiperparametros.sort_values(chaves_unicas).reset_index(drop=True), check_exact=True,
)
tabela_exportacao = pd.DataFrame(linhas_exportacao).set_index("Arquivo")
print(tabela_exportacao.to_string())
print("\nRecarga com float_precision='round_trip' idêntica aos resultados em memória: True")

# %% [markdown]
# ### BLOCO 77 — Auditoria contra Data Leakage
#
# A auditoria reúne evidências verificáveis no código e nos objetos ajustados: assinaturas dos conjuntos, composição dos
# folds, parâmetros de todos os scalers internos, escala dos limiares da árvore, atributos dos modelos, quantidade e
# composição dos dados de ajuste dos modelos finais e o momento da consulta ao teste. Essas verificações reduzem, mas
# não eliminam, os riscos metodológicos registrados no BLOCO 79.9.

# %%
# BLOCO 77 — Auditoria contra Data Leakage

# 1. Conjuntos oficiais inalterados desde o BLOCO 62
for nome, estrutura in estruturas_fase6.items():
    assert assinatura_integridade(estrutura) == assinaturas_fase6[nome], f"{nome} alterado durante a Fase 6."

# 2. Scaler interno de cada Pipeline ajustado somente no treino do respectivo fold (5 valores de K × 5 folds)
pipelines_conferidos = 0
for pipelines_fold in estimadores_knn.values():
    for pipeline_fold, (posicoes_treino, _) in zip(pipelines_fold, folds_cv):
        scaler_fold = pipeline_fold.named_steps["scaler"]
        assert int(scaler_fold.n_samples_seen_) == len(posicoes_treino)
        assert np.allclose(scaler_fold.mean_, X_train.iloc[posicoes_treino].mean(), rtol=tolerancia_parametros, atol=0)
        assert scaler_fold is not scaler
        pipelines_conferidos += 1

# 3. Sem escalonamento duplicado: o Pipeline recebeu dados na escala original (médias em unidades físicas)
assert (np.abs(estimadores_knn[k_selecionado_rotulo][0].named_steps["scaler"].mean_) > 1e-3).all()

# 4. Scaler oficial da Fase 5 inalterado
for nome_parametro in ["mean_", "var_", "scale_"]:
    assert np.array_equal(getattr(scaler, nome_parametro), parametros_apos_ajuste[nome_parametro])

# 5. Árvore treinada com atributos originais: limiares dentro da faixa original de cada atributo no treino
arvore_tree = modelo_arvore_final.tree_
for posicao_no, indice_atributo in enumerate(arvore_tree.feature):
    if indice_atributo >= 0:
        atributo = atributos_preditores[indice_atributo]
        assert X_train[atributo].min() <= arvore_tree.threshold[posicao_no] <= X_train[atributo].max(), \
            f"Limiar fora da escala original em {atributo}."

# 6. Atributos dos modelos: sem a variável alvo e sem a data
for modelo_final in [modelo_knn_final, modelo_arvore_final]:
    assert list(modelo_final.feature_names_in_) == atributos_preditores
    assert variavel_alvo not in modelo_final.feature_names_in_ and "date" not in modelo_final.feature_names_in_

# 7. Modelos finais ajustados somente no treino oficial (quantidade e composição das classes)
contagem_raiz = arvore_tree.value[0, 0] * arvore_tree.weighted_n_node_samples[0]
assert np.allclose(contagem_raiz, y_train.value_counts().sort_index().to_numpy()), "Árvore ajustada em outros dados."
assert modelo_knn_final.named_steps["knn"].n_samples_fit_ == len(X_train)
assert np.array_equal(modelo_knn_final.named_steps["knn"].classes_, np.array([0, 1]))

# 8. Seleção registrada antes da consulta ao teste e inalterada depois dela
assert teste_consultado_fase6 and repr(selecao_hiperparametros) == registro_selecao

# 9. Variáveis originais preservadas
pd.testing.assert_series_equal(df_features["CO2_Light_Ratio"], df_features_referencia["CO2_Light_Ratio"],
                               check_exact=True)
pd.testing.assert_series_equal(y, y_referencia, check_exact=True)

tabela_auditoria_fase6 = pd.Series({
    "Split oficial e conjuntos escalonados inalterados (assinaturas do BLOCO 62)": True,
    "Folds formados somente por registros do treino oficial": True,
    f"Scalers internos conferidos (K × fold): treino do próprio fold": pipelines_conferidos,
    "Pipeline alimentado com dados não escalonados (sem escala duplicada)": True,
    "Scaler da Fase 5 inalterado e não reutilizado": True,
    "Limiares da árvore na escala original dos atributos": True,
    "Sem Occupancy e sem date entre os atributos": True,
    "Modelos finais ajustados somente com X_train e y_train": True,
    "Hiperparâmetros e atributos escolhidos antes da consulta ao teste": True,
    "Teste consultado apenas no BLOCO 73 (exigência do enunciado)": True,
    "CO2_Light_Ratio, Occupancy e df_features preservados": True,
})
print(tabela_auditoria_fase6.to_string())

# %% [markdown]
# ### BLOCO 78 — Resumo consolidado da Fase 6
#
# Os resultados de validação (cruzada e temporal), obtidos somente no treino, ficam separados dos resultados descritivos
# no teste do BLOCO 73. Nenhum modelo é declarado vencedor definitivo nesta fase: a comparação final, com as métricas
# complementares, é atribuição da Fase 7.

# %%
# BLOCO 78 — Resumo consolidado da Fase 6

linhas_resumo_fase6 = []
for modelo, rotulo, resumo, hiperparametro in [("KNN (Pipeline)", k_selecionado_rotulo, resumo_knn, "n_neighbors"),
                                                ("Árvore de Decisão", profundidade_selecionada_rotulo, resumo_arvore,
                                                 "max_depth")]:
    chave_modelo = "KNN" if modelo.startswith("KNN") else modelo
    temporal = registros_temporais.query("modelo == @chave_modelo and conjunto == 'validacao'")
    linhas_resumo_fase6.append({
        "Modelo": modelo,
        "Hiperparâmetro": f"{hiperparametro}={rotulo}",
        "Atributos": len(atributos_preditores),
        "CV: acurácia treino": resumo.loc[rotulo, "Acurácia treino (média)"],
        "CV: acurácia validação": resumo.loc[rotulo, "Acurácia validação (média)"],
        "CV: dp validação": resumo.loc[rotulo, "Acurácia validação (dp)"],
        "CV: F1 ocupado": resumo.loc[rotulo, "F1 ocupado validação (média)"],
        "CV: balanced accuracy": resumo.loc[rotulo, "Balanced accuracy validação (média)"],
        "CV: treino − validação (p.p.)": resumo.loc[rotulo, "Diferença treino − validação (p.p.)"],
        "Temporal: acurácia média (5 folds)": temporal.query("metrica == 'acuracia'")["resultado"].mean(),
        "Temporal: acurácia mínima": temporal.query("metrica == 'acuracia'")["resultado"].min(),
        "Temporal: balanced accuracy (4 folds)": temporal.query("metrica == 'balanced_accuracy'")["resultado"].mean(),
    })
tabela_resumo_fase6 = pd.DataFrame(linhas_resumo_fase6).set_index("Modelo")
print("Resultados no treino (validação cruzada e validação temporal):")
print(tabela_resumo_fase6.T.to_string(float_format=lambda valor: f"{valor:.6f}"))

print("\nResultados descritivos no teste (BLOCO 73, calculados após a seleção):")
print(tabela_treino_teste.loc[[("KNN", f"n_neighbors={k_selecionado_rotulo}"),
                               ("Árvore de Decisão", f"max_depth={profundidade_selecionada_rotulo}")],
                              ["Acurácia treino", "Acurácia teste", "Diferença treino − teste (p.p.)"]]
      .to_string(float_format=lambda valor: f"{valor:.6f}"))
print("\nModelos em memória para a Fase 7: modelo_knn_final e modelo_arvore_final")

# %% [markdown]
# ## BLOCO 79 — Interpretação e conclusão do ajuste de hiperparâmetros
#
# Os resultados abaixo referem-se aos dados com a precisão decimal padronizada na Fase 2. A comparação com a versão
# anterior (precisão integral) está nos BLOCOS 103 a 108.
#
# ### 79.1 — Funcionamento do KNN
#
# Todas as configurações ficaram entre 99,08% e 99,17% de acurácia média de validação, isto é, entre 16,4 e 18,2 erros
# por fold de cerca de 1.987 registros. As diferenças entre os valores de K (no máximo 0,09 p.p., menos de 2 registros
# por fold) são menores ou da ordem do desvio padrão entre folds (0,18 a 0,22 p.p.) e não demonstram superioridade de
# nenhum K. O padrão esperado aparece na acurácia de treino: ela cai de 99,54% (K = 3) para 99,14% (K = 11), porque com
# poucos vizinhos o próprio registro e seus vizinhos imediatos pesam mais na votação. Com K maior, a diferença entre
# treino e validação diminui (0,37 p.p. com K = 3; 0,05 p.p. com K = 9 e 11), mas a validação não melhora. Não houve
# sinal de underfitting: mesmo K = 11 mantém 99,08% na validação.
#
# ### 79.2 — Funcionamento da Árvore de Decisão
#
# A profundidade controla a complexidade de forma visível. Com `max_depth` = 3 e 5, as árvores atingem exatamente o
# limite (8,0 e 19,8 folhas em média) e ficam entre 99,05% e 99,07% de validação. Com 10 e None, a acurácia de treino
# sobe para 99,73% e 100%, enquanto a validação cai para 98,99% e 98,96% (cerca de 20 a 21 erros por fold, contra 18
# com profundidade 5). Sem limite, as árvores alcançaram profundidade efetiva de 15 a 18 nos folds (92 folhas em média):
# o limite `None` não significa profundidade infinita, mas crescimento até separar completamente os registros de treino.
# Na árvore final de profundidade 5, a importância de `Light` por redução de impureza (`feature_importances_`, que soma
# 100% entre os seis atributos) é de 98,92%: quase toda a redução da impureza de Gini nas divisões desta árvore ocorre
# em divisões por luminosidade. Essa medida descreve a estrutura da árvore ajustada neste treino; não é uma
# probabilidade de ocupação, não é o percentual de acertos devido ao atributo, não indica relação causal e não vale
# automaticamente para outros ambientes. Ela é coerente com a forte associação entre `Light` e ocupação (BLOCO 17).
#
# ### 79.3 — Pipeline e prevenção de Data Leakage
#
# No fold 1, o scaler interno aprendeu exatamente a média dos 7.946 registros de ajuste do fold (por exemplo,
# `CO2_Light_Ratio` = 424,43), diferente da média da validação do fold (436,15) e da média do treino oficial usada na
# Fase 5 (426,77). O BLOCO 77 repetiu a verificação para os 25 Pipelines da validação cruzada (5 valores de K × 5 folds):
# cada scaler viu apenas a parte de ajuste do seu fold. O Pipeline sempre recebeu dados na escala original, sem
# escalonamento duplicado, e o scaler da Fase 5 não foi reutilizado nem alterado. Os folds são idênticos aos da versão
# anterior (assinaturas SHA-256 conferidas no BLOCO 64).
#
# ### 79.4 — Seleção dos hiperparâmetros
#
# Pelo critério definido no BLOCO 68, foram selecionados **K = 3** (99,17% de acurácia média de validação, F1 da classe
# ocupada de 98,31% e balanced accuracy de 98,98%) e **`max_depth` = 5** (99,07%, 98,11% e 98,97%). Não houve empate
# numérico. As margens são muito pequenas: K = 3 supera K = 5 em 0,01 p.p. Nos cinco folds, K = 3 acertou 1.976, 1.965,
# 1.975, 1.968 e 1.967 registros (9.851 de 9.933) e K = 5 acertou 1.976, 1.969, 1.970, 1.964 e 1.971 (9.850): **a
# diferença é de exatamente um registro**. O critério usa a média não ponderada das acurácias dos folds (99,174448% contra
# 99,164382%), que difere ligeiramente da acurácia agregada de todas as previsões (99,174469% e 99,164401%), porque os
# folds têm 1.986 ou 1.987 registros; as duas formas levam à mesma ordem. A diferença não demonstra superioridade
# robusta de K = 3. `max_depth` = 5 supera `max_depth` = 3 em 0,02 p.p. (cerca de 0,4 registro
# por fold). As métricas complementares não apontam na mesma direção: a maior balanced accuracy do KNN foi a de K = 7
# (99,16%), e K = 3 teve a menor balanced accuracy entre os valores de K; na árvore, a maior foi a de `max_depth` = 3
# (99,11%). As configurações selecionadas são, portanto, escolhas válidas pelo critério pré-definido, mas não
# configurações comprovadamente superiores às vizinhas.
#
# **Mudança em relação à versão anterior:** com a precisão integral, o mesmo critério havia selecionado K = 5 (99,1644%
# contra 99,1443% de K = 3). Com os dados arredondados, K = 5 manteve exatamente a mesma acurácia média (99,1644%), e K = 3
# passou a 99,1744%, uma diferença equivalente a um registro em 9.933. A mudança decorre da regra decimal, aplicada sem
# consultar o teste, e não altera a natureza da conclusão: os valores de K são praticamente equivalentes.
#
# ### 79.5 — Overfitting, underfitting e estabilidade
#
# **Overfitting** (alta acurácia no treino acompanhada de queda na avaliação), considerando a validação cruzada e a
# tabela de treino e teste exigida pelo enunciado (BLOCO 73):
#
# - **Árvore com `max_depth=None`:** 100% no treino, com queda para 98,96% na validação cruzada (1,04 p.p.) e para
#   99,24% no teste (0,76 p.p.). É o caso mais claro de memorização: a árvore separa todos os registros de treino.
# - **Árvore com `max_depth=10`:** 99,66% no treino oficial, 98,99% na validação (0,73 p.p.) e 99,11% no teste
#   (0,54 p.p.); mesmo padrão, um pouco menos intenso.
# - **KNN com K = 3:** diferença de 0,37 p.p. entre treino e validação cruzada, a maior entre os KNN, mas sem queda no
#   teste (99,57% no treino oficial e 99,56% no teste); o sinal de sobreajuste é leve e restrito à validação cruzada.
#
# As quedas são pequenas em valor absoluto porque a tarefa é quase separável pela luminosidade. No teste, a árvore sem
# limite obteve acurácia maior que a de profundidade 5 (99,24% contra 99,03%, uma diferença de 5 registros), o oposto da
# validação cruzada; K = 3 e K = 5 empataram no teste (99,56%). Esses resultados não mudam a seleção. Uma interpretação
# possível, não demonstrada isoladamente, envolve a proximidade temporal: 96,10% dos registros de teste têm um registro
# de treino a até 61 segundos, o que limita a leitura do teste como estimativa de generalização e pode beneficiar modelos
# que exploram padrões locais. A árvore com `max_depth=3` teve a menor acurácia de teste (98,63%) e a menor acurácia de treino entre as
# árvores, mas a diferença entre treino e validação foi de apenas 0,13 p.p.: é o indício mais próximo de underfitting,
# ainda que muito discreto. Nenhuma configuração apresentou underfitting acentuado.
#
# **Estabilidade:** entre as configurações selecionadas, o **KNN com K = 3** foi o mais estável entre treino e teste
# (99,57% e 99,56%, diferença de 0,01 p.p.) e degradou menos na validação temporal (BLOCO 79.8). Na validação cruzada,
# as menores diferenças entre treino e validação foram as do KNN com K = 9 e 11 (0,05 p.p.) e da árvore com
# profundidade 3 (0,13 p.p.); o menor desvio padrão de acurácia entre folds foi o da árvore com profundidade 5
# (0,14 p.p.).
#
# ### 79.6 — Avaliação da CO2_Light_Ratio
#
# No KNN, as métricas com 5 e com 6 atributos foram **idênticas em todos os folds e em todos os valores de K**. A nova
# coluna altera o conjunto de vizinhos de parte dos registros, mas, no recorte verificado (somente o fold 1, com K = 3),
# isso ocorreu em 10,92% das validações sem mudar nenhum voto majoritário (0 previsões diferentes em 1.987); nos demais
# folds e valores de K, a igualdade das métricas foi verificada, mas não a dos vizinhos. Na árvore, as diferenças médias
# foram de 0,00 p.p. com profundidade 3, +0,01 p.p. com 5 (favorável aos 6 atributos), −0,01 p.p. com 10 e −0,06 p.p.
# com None, todas da ordem de um registro por fold ou menos e menores que a variação entre folds. A importância da
# variável por redução de impureza na árvore final é de 0,11%.
#
# **A inclusão da variável derivada não demonstrou ganho preditivo nas configurações e nos protocolos avaliados:** a
# única diferença favorável (+0,01 p.p. na árvore de profundidade 5) equivale a um registro em todos os folds e não é
# distinguível do ruído entre folds. Uma explicação compatível com os resultados é a redundância com `Light` e `CO2`
# (com `Light = 0`, o índice é igual a `CO2`; BLOCO 36), mas isso não foi testado isoladamente, e o resultado não se
# generaliza para outros modelos, ambientes ou conjuntos de dados. A variável permanece nos modelos principais por
# exigência do fluxo do projeto, não por desempenho comprovado.
#
# ### 79.7 — Dependência temporal na validação cruzada
#
# Em média, 87,18% dos registros de validação dos folds (86,11% a 87,81% por fold) têm um registro da parte de ajuste a
# até 61 segundos, e 96,74% a até 120 segundos (mediana de 60 segundos). A proximidade é um pouco menor que a do teste
# oficial (96,10% a até 61 segundos), porque a parte de ajuste de cada fold contém menos registros, mas continua muito
# alta. Esses valores dependem apenas dos índices e dos timestamps e são iguais aos da versão anterior. A validação
# cruzada aleatória, portanto, mede principalmente a capacidade de reconhecer condições quase idênticas às já vistas
# cerca de um minuto antes ou depois, e não a capacidade de generalizar para outros períodos.
#
# ### 79.8 — Validação temporal complementar
#
# Com janelas cronológicas e um intervalo de 60 registros (72 a 82 minutos reais), nenhum registro de validação teve
# registro de ajuste a até 61 segundos. Os resultados das configurações selecionadas foram:
#
# | Fold | Validação | KNN (K = 3): acurácia | Árvore (profundidade 5): acurácia |
# |---|---|---:|---:|
# | 1 | 04/02 a 12/02 (atravessa a lacuna) | 96,92% | 93,35% |
# | 2 | 12/02 a 14/02 | 98,37% | 99,15% |
# | 3 | 14/02 a 15/02 (sábado e domingo, sem ocupação) | 80,06% | 45,86% |
# | 4 | 15/02 a 16/02 | 99,46% | 98,43% |
# | 5 | 16/02 a 18/02 | 98,31% | 99,58% |
#
# A média dos 5 folds foi de 94,62% (KNN) e 87,27% (árvore). Nas quatro janelas com as duas classes, a balanced accuracy
# média foi de 98,16% (KNN) e 96,71% (árvore), abaixo da validação aleatória (98,98% e 98,97%). A diferença decisiva
# está no fold 3: a janela de validação cobre o fim de semana, quando ninguém ocupou a sala, e o treino daquela divisão
# tinha apenas 55 registros de fim de semana (madrugada de sábado). Dos 1.655 registros da janela, 982 têm `Light = 0` e
# 673 (40,66%) têm luminosidade positiva sem ocupação — possivelmente luz natural, hipótese não verificável com os
# dados. Como a janela não tem registros ocupados, todos os erros são previsões incorretas de ocupação:
#
# | Modelo | Erros | Com `Light = 0` | Com `Light > 0` | Acurácia na janela |
# |---|---:|---:|---:|---:|
# | KNN (K = 3) | 330 | 0 | 330 (49,03% dos 673) | 80,06% |
# | Árvore (profundidade 5) | 896 | 405 | 491 (72,96% dos 673) | 45,86% |
#
# Os erros do KNN concentram-se nos registros iluminados (mediana de 161,8 lux). Os da árvore ocorrem nos dois grupos:
# 405 registros escuros também foram classificados como ocupados, por divisões que, abaixo do limiar de `Light`, usam
# outros atributos (a mediana de 7 lux refere-se ao conjunto dos 896 erros). Prever sempre "desocupado" teria acertado
# 100% da janela. Quando o treino passa a incluir o fim de semana (folds 4 e 5), as duas acurácias voltam a ficar
# acima de 98%. Em relação à versão anterior, só o fold 1 mudou de forma perceptível (KNN 98,13% → 96,92%, também por
# causa da troca de K = 5 para K = 3; árvore 95,53% → 93,35%), e o fold 3 passou de 327 para 330 erros no KNN; a
# árvore manteve exatamente os 896 erros.
#
# A comparação é descritiva: os protocolos usam treinos de tamanhos diferentes (1.598 a 8.218 registros na temporal,
# contra 7.946 na aleatória), janelas com composições distintas e uma única sequência temporal; a diferença não pode ser
# atribuída exclusivamente à ordem do tempo. Ainda assim, ela indica que o desempenho de 99% depende de o treino conter
# condições semelhantes às avaliadas, e que o KNN selecionado degradou menos que a árvore selecionada na condição
# inédita. As configurações principais não foram alteradas por esses resultados.
#
# ### 79.9 — Limitações metodológicas
#
# - **Dependência temporal:** a validação cruzada e o teste oficial têm registros quase idênticos na parte de ajuste;
#   as acurácias próximas de 99% são otimistas como estimativa para outros períodos (BLOCOS 44, 70 e 71).
# - **Seleção pela validação cruzada:** as mesmas validações escolheram os hiperparâmetros e compararam os atributos;
#   o valor da configuração escolhida tende a ser ligeiramente otimista, e as diferenças entre configurações são da
#   ordem de um registro por fold. A troca de K = 5 por K = 3 na revisão ilustra essa fragilidade.
# - **Consulta ao teste:** exigida pelo enunciado na Fase 6; a seleção foi congelada antes, mas o teste deixa de ser
#   inédito para a Fase 7. Além disso, o teste já havia sido consultado na versão anterior à regra decimal.
# - **Desequilíbrio:** 24,33% de ocupados; a acurácia sozinha pode esconder erros na classe minoritária, por isso F1 e
#   balanced accuracy foram acompanhados.
# - **Redundância e dependência de `Light`:** a árvore concentra em `Light` quase toda a redução de impureza (98,92%),
#   e a variável derivada não demonstrou ganho preditivo nas configurações e nos protocolos avaliados. Em automação
#   predial, a luz natural e o próprio acionamento da iluminação afetam o sensor, com risco de realimentação (BLOCO 17).
# - **Composição da base:** a EDA e a escolha da variável derivada usaram a base completa; a base tem dois segmentos,
#   um único fim de semana (14 e 15/02) e não inclui o período de `datatraining.txt`, que contém outro fim de semana.
#
# ### 79.10 — Preparação para a Fase 7
#
# Ficam em memória `modelo_knn_final` (Pipeline com `StandardScaler` e KNN com K = 3, ajustado nos 9.933 registros do
# treino) e `modelo_arvore_final` (Árvore de Decisão com `max_depth=5`, profundidade efetiva 5, 37 nós e 19 folhas). A
# Fase 7 deverá apresentar a acurácia final dos dois modelos no teste, compará-los e justificar a escolha, complementando
# a análise com o baseline da classe majoritária (`DummyClassifier`), a matriz de confusão, a precision, o recall, o
# F1-score e a balanced accuracy, sem alterar os hiperparâmetros selecionados.

# %% [markdown]
# # FASE 7 — Avaliação dos Modelos e Veredito Final
# ## BLOCO 80 — Contextualização da avaliação
#
# ### Requisitos do enunciado e materiais do curso
#
# | Fonte | Exigência ou conteúdo |
# |---|---|
# | Enunciado, Fase 7 (p. 5) | Calcular e exibir a acurácia final do melhor KNN e da melhor Árvore de Decisão **exclusivamente com os dados de teste** |
# | Enunciado, Fase 7 (p. 5) | Comparar os resultados numéricos e escrever uma conclusão técnica justificando qual modelo apresentou desempenho superior e deve ser adotado pelo sistema de automação predial |
# | Rubrica, critério 10 (p. 9) | Pontuação total: calcular as acurácias no teste, comparar os modelos e justificar a decisão final; calcular sem comparar é pontuação parcial |
# | Enunciado, contexto (p. 2) | Automação predial na Indústria 4.0: desligamento automático de iluminação e climatização em salas desocupadas |
# | MUS01, slides 52 e 80 a 86 | Acurácia e matriz de confusão na avaliação de classificadores |
# | MUS01, slides 90 e 91 | Precision e recall e a escolha do erro mais custoso para o negócio |
# | MUS01, slides 95 e 96 | Paradoxo da acurácia em bases desbalanceadas |
# | Atividade semanal (`trabalho_semanal_analise_preditiva.py`) | Importa `accuracy_score`, `classification_report` e `confusion_matrix` para a avaliação |
#
# São **complementos metodológicos deste projeto**, e não exigências do enunciado: o baseline `DummyClassifier`, as
# métricas além da acurácia, a comparação pareada com o teste de McNemar, a investigação dos erros, os cenários de
# custo, a revisão temporal e a avaliação no `datatraining.txt`.
#
# ### Conceitos
#
# - **Finalidade:** estimar como os modelos selecionados se comportam em registros não usados no ajuste e decidir qual
#   é mais adequado à aplicação.
# - **Treino, validação cruzada e teste:** o treino ajusta o modelo; a validação cruzada (Fase 6) comparou
#   configurações dentro do treino; o teste avalia os modelos finais em registros separados desde a Fase 4.
# - **Matriz de confusão (classificação binária, classe positiva = ocupado):** TN (desocupado previsto como
#   desocupado), FP (desocupado previsto como ocupado), FN (ocupado previsto como desocupado) e TP (ocupado previsto
#   como ocupado).
# - **Métricas:** acurácia = (TP + TN) / total; precision = TP / (TP + FP); recall = TP / (TP + FN); especificidade =
#   TN / (TN + FP); F1 = 2 × precision × recall / (precision + recall); balanced accuracy = (recall + especificidade) / 2.
# - **Baseline e limitação da acurácia:** com 75,67% de registros desocupados, prever sempre "desocupado" já alcança
#   uma acurácia alta sem detectar nenhuma ocupação. Por isso, os erros de cada classe são analisados separadamente.
# - **Erros na automação:** um falso positivo mantém luz e climatização ligadas sem ninguém (desperdício); um falso
#   negativo desliga o ambiente com pessoas presentes (desconforto ou risco).
# - **Generalização:** a dependência temporal medida nas Fases 4 e 6 torna o teste otimista; por isso, os resultados
#   são confrontados com a validação temporal e com o `datatraining.txt`.
#
# ### Condições desta fase
#
# - O conjunto de teste **já foi consultado na Fase 6** (BLOCO 73), por exigência do enunciado; a seleção de K = 3 e
#   `max_depth` = 5 foi registrada antes dessa consulta. Além disso, o teste e o `datatraining.txt` já haviam sido
#   avaliados na versão anterior à padronização decimal (BLOCO 103).
# - As configurações não são alteradas nesta fase, e os modelos principais não são reajustados: `modelo_knn_final` e
#   `modelo_arvore_final` vêm da execução sequencial da Fase 6.
# - A avaliação aprofunda os resultados dos modelos já selecionados; nenhuma métrica desta fase é usada para
#   otimizar novamente os classificadores.

# %% [markdown]
# ### BLOCO 81 — Verificação dos modelos selecionados e dos conjuntos
#
# Confere os hiperparâmetros, os atributos e o estado ajustado dos dois modelos e as assinaturas dos oito conjuntos de
# referência da Fase 6. A função `assinatura_modelo` calcula o SHA-256 dos hiperparâmetros e dos atributos aprendidos
# (parâmetros do scaler e dados de referência do KNN; estrutura e limiares da árvore), sem serializar os objetos. Essas
# assinaturas são conferidas depois de cada etapa de previsão.

# %%
# BLOCO 81 — Verificação dos modelos selecionados e dos conjuntos

# Modelos produzidos pela execução sequencial da Fase 6 (nenhum novo ajuste nesta fase)
check_is_fitted(modelo_knn_final)
check_is_fitted(modelo_arvore_final)
knn_avaliado = modelo_knn_final.named_steps["knn"]
scaler_avaliado = modelo_knn_final.named_steps["scaler"]
assert list(modelo_knn_final.named_steps) == ["scaler", "knn"], "Etapas do Pipeline alteradas."
assert (knn_avaliado.n_neighbors, knn_avaliado.weights, knn_avaliado.effective_metric_) == (k_selecionado, "uniform",
                                                                                       "euclidean")
assert (modelo_arvore_final.max_depth, modelo_arvore_final.criterion, modelo_arvore_final.random_state) == (
    profundidade_selecionada, "gini", 42)
assert repr(selecao_hiperparametros) == registro_selecao, "Seleção da Fase 6 alterada."
for modelo_final in [modelo_knn_final, modelo_arvore_final]:
    assert list(modelo_final.feature_names_in_) == atributos_preditores, "Atributos dos modelos alterados."

# Conjuntos oficiais (sem novo split) e assinaturas da Fase 6
assert X_train.shape == (9933, 6) and X_test.shape == (2484, 6) and len(y_train) == 9933 and len(y_test) == 2484
assert y_train.value_counts().sort_index().tolist() == [7516, 2417]
assert y_test.value_counts().sort_index().tolist() == [1880, 604]
for nome, estrutura in estruturas_fase6.items():
    assert assinatura_integridade(estrutura) == assinaturas_fase6[nome], f"{nome} alterado antes da Fase 7."


def assinatura_modelo(modelo):
    """SHA-256 dos hiperparâmetros e dos atributos aprendidos (sem serializar o objeto)."""
    if isinstance(modelo, Pipeline):
        escala, vizinhos = modelo.named_steps["scaler"], modelo.named_steps["knn"]
        partes = [escala.mean_, escala.var_, escala.scale_, np.asarray(escala.n_samples_seen_),
                  vizinhos._fit_X, vizinhos._y, vizinhos.classes_]
    elif isinstance(modelo, DecisionTreeClassifier):
        estrutura = modelo.tree_
        partes = [estrutura.feature, estrutura.threshold, estrutura.value, estrutura.children_left,
                  estrutura.children_right, modelo.classes_]
    else:
        partes = [modelo.classes_, np.asarray(modelo.class_prior_)]
    resumo = hashlib.sha256(repr(sorted(modelo.get_params().items())).encode("utf-8"))
    for parte in partes:
        resumo.update(np.ascontiguousarray(parte).tobytes())
    return resumo.hexdigest()


assinaturas_modelos_fase7 = {"KNN": assinatura_modelo(modelo_knn_final),
                             "Árvore de Decisão": assinatura_modelo(modelo_arvore_final)}

tabela_modelos_fase7 = pd.DataFrame({
    "Configuração": [f"Pipeline(StandardScaler, KNN): n_neighbors={knn_avaliado.n_neighbors}, "
                     f"weights='{knn_avaliado.weights}', metric='{knn_avaliado.effective_metric_}'",
                     f"DecisionTreeClassifier: max_depth={modelo_arvore_final.max_depth}, "
                     f"criterion='{modelo_arvore_final.criterion}', random_state={modelo_arvore_final.random_state}"],
    "Registros de ajuste": [knn_avaliado.n_samples_fit_, int(modelo_arvore_final.tree_.n_node_samples[0])],
    "Escalonamento": ["Dentro do Pipeline (ajustado no treino)", "Nenhum"],
    "Assinatura (16 caracteres)": [assinatura[:16] for assinatura in assinaturas_modelos_fase7.values()],
}, index=pd.Index(assinaturas_modelos_fase7.keys(), name="Modelo"))
print(tabela_modelos_fase7.to_string())
print("\nConjuntos oficiais e modelos verificados; nenhum ajuste será feito nos modelos principais nesta fase.")

# %% [markdown]
# ### BLOCO 82 — Baseline com DummyClassifier
#
# `DummyClassifier(strategy="most_frequent")` é ajustado somente com `X_train` e `y_train` e prevê sempre a classe
# majoritária do treino. Sua acurácia no teste é um resultado calculado neste conjunto e não se confunde com a
# referência descritiva de 75,6705% do BLOCO 39, que é a proporção de desocupados na base completa.

# %%
# BLOCO 82 — Baseline com DummyClassifier

modelo_baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
assert modelo_baseline.classes_[np.argmax(modelo_baseline.class_prior_)] == 0, "Classe majoritária diferente de 0."
assinaturas_modelos_fase7["DummyClassifier"] = assinatura_modelo(modelo_baseline)

y_pred_baseline = pd.Series(modelo_baseline.predict(X_test), index=X_test.index, name="previsao_baseline")
acuracia_treino_baseline = accuracy_score(y_train, modelo_baseline.predict(X_train))
acuracia_teste_baseline = accuracy_score(y_test, y_pred_baseline)

print("Classe prevista para todos os registros:", int(np.unique(y_pred_baseline)[0]), "(0 = desocupado)")
print("Proporções aprendidas no treino (class_prior_):", dict(zip(modelo_baseline.classes_.tolist(),
                                                                  modelo_baseline.class_prior_.round(6))))
print(f"Acurácia no treino: {acuracia_treino_baseline:.6%} | no teste: {acuracia_teste_baseline:.6%}")
print(f"Referência descritiva da base completa (BLOCO 39, não é resultado de modelo): {acuracia_referencia:.4%}")

# %% [markdown]
# ### BLOCO 83 — Previsões dos classificadores no teste oficial
#
# `modelo_knn_final.predict(X_test)` recebe `X_test` original: o `StandardScaler` interno do Pipeline aplica apenas
# `transform` com os parâmetros aprendidos no treino. A Árvore usa os mesmos atributos sem escalonamento. As acurácias
# devem reproduzir exatamente as do BLOCO 73.

# %%
# BLOCO 83 — Previsões dos classificadores no teste oficial

# O Pipeline recebe X_test original: o scaler interno aplica apenas transform com os parâmetros do treino
y_pred_knn = pd.Series(modelo_knn_final.predict(X_test), index=X_test.index, name="previsao_knn")
y_pred_arvore = pd.Series(modelo_arvore_final.predict(X_test), index=X_test.index, name="previsao_arvore")

for previsoes in [y_pred_baseline, y_pred_knn, y_pred_arvore]:
    assert len(previsoes) == 2484 and previsoes.index.equals(y_test.index), "Previsões desalinhadas do teste."
    assert previsoes.notna().all() and set(previsoes.unique()).issubset({0, 1}), "Previsão inválida."
pd.testing.assert_series_equal(y_test, y_test_referencia, check_exact=True)

# Correspondência com as acurácias da Fase 6 (BLOCO 73)
acuracia_teste_knn = accuracy_score(y_test, y_pred_knn)
acuracia_teste_arvore = accuracy_score(y_test, y_pred_arvore)
assert acuracia_teste_knn == tabela_treino_teste.loc[("KNN", f"n_neighbors={k_selecionado}"), "Acurácia teste"]
assert acuracia_teste_arvore == tabela_treino_teste.loc[
    ("Árvore de Decisão", f"max_depth={profundidade_selecionada}"), "Acurácia teste"]
for nome, modelo_final in [("KNN", modelo_knn_final), ("Árvore de Decisão", modelo_arvore_final)]:
    assert assinatura_modelo(modelo_final) == assinaturas_modelos_fase7[nome], f"{nome} alterado pela previsão."

print(f"Acurácia no teste — KNN: {acuracia_teste_knn!r} | Árvore: {acuracia_teste_arvore!r}")
print("Idênticas às acurácias registradas na Fase 6: True | modelos inalterados: True")

# %% [markdown]
# ### BLOCO 84 — Matrizes de confusão
#
# Matrizes com ordem explícita `labels=[0, 1]`: linhas = classe real, colunas = classe prevista. Cada matriz é
# conferida por contagem direta e pela relação entre a diagonal principal e a acurácia.

# %%
# BLOCO 84 — Matrizes de confusão

previsoes_teste_modelos = {"DummyClassifier": y_pred_baseline, "KNN": y_pred_knn, "Árvore de Decisão": y_pred_arvore}
matrizes_confusao = {}
for nome, previsoes in previsoes_teste_modelos.items():
    matriz = confusion_matrix(y_test, previsoes, labels=[0, 1])
    # Conferência independente por contagem direta
    contagem = np.array([[((y_test == real) & (previsoes == prevista)).sum() for prevista in (0, 1)]
                         for real in (0, 1)])
    assert np.array_equal(matriz, contagem), f"Matriz divergente da contagem direta ({nome})."
    assert matriz.sum() == 2484 and matriz[0].sum() == 1880 and matriz[1].sum() == 604
    assert np.trace(matriz) / matriz.sum() == accuracy_score(y_test, previsoes)
    matrizes_confusao[nome] = matriz
    print(f"\n{nome}:")
    print(pd.DataFrame(matriz, index=["Realmente desocupado (0)", "Realmente ocupado (1)"],
                       columns=["Previsto desocupado (0)", "Previsto ocupado (1)"]).to_string())

tabela_acertos = pd.DataFrame({
    nome: {"TN": int(matriz[0, 0]), "FP": int(matriz[0, 1]), "FN": int(matriz[1, 0]), "TP": int(matriz[1, 1]),
           "Acertos": int(np.trace(matriz)), "Erros": int(matriz.sum() - np.trace(matriz))}
    for nome, matriz in matrizes_confusao.items()
}).T
print("\n" + tabela_acertos.to_string())

# %% [markdown]
# ### BLOCO 85 — Métricas completas de classificação
#
# Métricas do scikit-learn com classe positiva 1 (ocupado), conferidas pelas fórmulas da matriz de confusão. Quando um
# denominador é zero, a métrica é **indefinida** e aparece como `NaN` (`zero_division=np.nan`), e não como um resultado
# medido. Observações sobre o baseline: a precision da classe ocupada é indefinida (TP + FP = 0); o F1 é 0, pois o
# scikit-learn o calcula como 2TP / (2TP + FP + FN), forma equivalente à média harmônica quando precision e recall estão
# definidas; e as médias macro de precision excluem a classe cujo valor é indefinido.

# %%
# BLOCO 85 — Métricas completas de classificação


def razao(numerador, denominador):
    """Divisão com denominador zero tratado explicitamente como indefinido (NaN)."""
    return numerador / denominador if denominador > 0 else np.nan


def calcular_metricas(y_real, y_previsto):
    """Métricas do scikit-learn (classe positiva = 1) conferidas pelas fórmulas da matriz de confusão."""
    tn, fp, fn, tp = confusion_matrix(y_real, y_previsto, labels=[0, 1]).ravel()
    metricas = {
        "accuracy": accuracy_score(y_real, y_previsto),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_previsto),
        "precision_ocupado": precision_score(y_real, y_previsto, pos_label=1, zero_division=np.nan),
        "recall_ocupado": recall_score(y_real, y_previsto, pos_label=1, zero_division=np.nan),
        "f1_ocupado": f1_score(y_real, y_previsto, pos_label=1, zero_division=np.nan),
        "especificidade": recall_score(y_real, y_previsto, pos_label=0, zero_division=np.nan),
        "taxa_falsos_positivos": razao(fp, fp + tn),
        "taxa_falsos_negativos": razao(fn, fn + tp),
        "precision_macro": precision_score(y_real, y_previsto, average="macro", zero_division=np.nan),
        "recall_macro": recall_score(y_real, y_previsto, average="macro", zero_division=np.nan),
        "f1_macro": f1_score(y_real, y_previsto, average="macro", zero_division=np.nan),
        "f1_ponderado": f1_score(y_real, y_previsto, average="weighted", zero_division=np.nan),
    }
    # Conferência pelas fórmulas (NaN apenas quando o denominador é zero)
    precisao = razao(tp, tp + fp)
    sensibilidade = razao(tp, tp + fn)
    especificidade = razao(tn, tn + fp)
    formulas = {
        "accuracy": razao(tp + tn, tp + tn + fp + fn),
        "precision_ocupado": precisao,
        "recall_ocupado": sensibilidade,
        "especificidade": especificidade,
        # Forma equivalente a 2 × P × R / (P + R) quando P e R estão definidas; permanece definida se TP + FP = 0
        "f1_ocupado": razao(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sensibilidade + especificidade) / 2,
    }
    if not (np.isnan(precisao) or np.isnan(sensibilidade)) and precisao + sensibilidade > 0:
        assert np.isclose(2 * precisao * sensibilidade / (precisao + sensibilidade), formulas["f1_ocupado"],
                          rtol=1e-12, atol=0), "F1 diferente da média harmônica."
    for chave, valor in formulas.items():
        assert (np.isnan(valor) and np.isnan(metricas[chave])) or np.isclose(valor, metricas[chave], rtol=1e-12,
                                                                             atol=0), f"Fórmula divergente: {chave}."
    return {**metricas, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


metricas_teste = {nome: calcular_metricas(y_test, previsoes) for nome, previsoes in previsoes_teste_modelos.items()}
tabela_metricas_teste = pd.DataFrame(metricas_teste)
colunas_percentuais = [indice for indice in tabela_metricas_teste.index if indice not in ("TN", "FP", "FN", "TP")]
print("Métricas no teste oficial (classe positiva = 1, ocupado; NaN = indefinida por denominador zero):")
print(tabela_metricas_teste.loc[colunas_percentuais].map(
    lambda valor: "indefinida" if pd.isna(valor) else f"{valor:.6%}").to_string())
print(tabela_metricas_teste.loc[["TN", "FP", "FN", "TP"]].astype(int).to_string())

diferenca_knn_arvore = (tabela_metricas_teste.loc[colunas_percentuais, "KNN"]
                        - tabela_metricas_teste.loc[colunas_percentuais, "Árvore de Decisão"]) * 100
print("\nDiferença KNN − Árvore (pontos percentuais):")
print(diferenca_knn_arvore.round(4).to_string())

print("\nObservação: no DummyClassifier, a precision da classe ocupada é indefinida (nenhuma previsão positiva:")
print("TP + FP = 0); o F1 é 0, pois o scikit-learn o calcula como 2TP / (2TP + FP + FN) = 0 / 604. As médias macro")
print("de precision do scikit-learn com zero_division=np.nan excluem a classe cuja precision é indefinida.")

for nome, previsoes in previsoes_teste_modelos.items():
    print(f"\nRelatório de classificação — {nome}:")
    print(classification_report(y_test, previsoes, labels=[0, 1], target_names=["0 — Desocupado", "1 — Ocupado"],
                                digits=4, zero_division=np.nan))

# %% [markdown]
# ### BLOCO 86 — Comparação pareada dos erros
#
# Os dois modelos são avaliados nos mesmos 2.484 registros. A tabela de contingência separa os registros que ambos
# acertam, que ambos erram e que só um dos modelos acerta. O teste exato de McNemar (binomial sobre os pares
# discordantes, `scipy.stats.binomtest`) avalia a hipótese de que os dois modelos erram com a mesma frequência. Como os
# registros do teste são vizinhos temporais dependentes entre si, o pressuposto de independência não se sustenta, e o
# p-valor é apenas **exploratório**.

# %%
# BLOCO 86 — Comparação pareada dos erros (KNN × Árvore)

acerto_knn = y_pred_knn.eq(y_test)
acerto_arvore = y_pred_arvore.eq(y_test)
tabela_pareada = pd.crosstab(acerto_knn.map({True: "KNN acerta", False: "KNN erra"}),
                             acerto_arvore.map({True: "Árvore acerta", False: "Árvore erra"}))
tabela_pareada = tabela_pareada.loc[["KNN acerta", "KNN erra"], ["Árvore acerta", "Árvore erra"]]
assert int(tabela_pareada.to_numpy().sum()) == 2484, "Contingência não soma 2.484."
print(tabela_pareada.to_string())

ambos_acertam = int((acerto_knn & acerto_arvore).sum())
ambos_erram = int((~acerto_knn & ~acerto_arvore).sum())
somente_knn_acerta = int((acerto_knn & ~acerto_arvore).sum())
somente_arvore_acerta = int((~acerto_knn & acerto_arvore).sum())
pares_discordantes = somente_knn_acerta + somente_arvore_acerta

# Teste exato de McNemar (binomial sobre os pares discordantes; H0: os dois modelos erram com a mesma frequência)
resultado_mcnemar = binomtest(somente_knn_acerta, n=pares_discordantes, p=0.5, alternative="two-sided")
print(f"\nAmbos acertam: {ambos_acertam} | ambos erram: {ambos_erram} | só o KNN acerta: {somente_knn_acerta} | "
      f"só a Árvore acerta: {somente_arvore_acerta}")
print(f"Diferença de acertos: {somente_knn_acerta - somente_arvore_acerta} registros "
      f"({(acuracia_teste_knn - acuracia_teste_arvore) * 100:.4f} p.p.)")
print(f"McNemar exato (exploratório): {pares_discordantes} pares discordantes, p-valor = {resultado_mcnemar.pvalue:.6f}")

# %% [markdown]
# ### BLOCO 87 — Investigação descritiva dos erros no teste
#
# Associa cada previsão incorreta ao índice original, à data e aos atributos. A proximidade de transições de ocupação é
# uma análise **exploratória posterior à avaliação**: o intervalo de 10 minutos foi definido antes da contagem; as
# transições são as 74 mudanças de classe entre registros consecutivos do mesmo segmento (BLOCO 16.2), sem atravessar a
# lacuna entre os segmentos. Os resultados descrevem associações e não são usados para criar regras ou alterar modelos.

# %%
# BLOCO 87 — Investigação descritiva dos erros no teste


def tipo_erro(real, previsto):
    return np.select([(real == 0) & (previsto == 1), (real == 1) & (previsto == 0)],
                     ["Falso positivo", "Falso negativo"], default="Acerto")


# Transições de ocupação na base completa, dentro de cada segmento (uso exploratório, após a avaliação)
janela_transicao_min = 10  # definida antes da contagem: até 10 minutos de uma mudança de classe
datas_base = rastreio_temporal["date"]
mudanca_ocupacao = y.ne(y.shift()) & rastreio_temporal["segmento"].eq(rastreio_temporal["segmento"].shift())
transicoes_ocupacao = rastreio_temporal.loc[mudanca_ocupacao]
assert len(transicoes_ocupacao) == mudancas_validas, "Transições diferentes das registradas no BLOCO 16.2."

distancia_transicao_min = pd.Series(np.inf, index=X_test.index)
for segmento_id in rastreio_temporal["segmento"].unique():
    tempos_transicao = np.sort(
        transicoes_ocupacao.loc[transicoes_ocupacao["segmento"].eq(segmento_id), "date"].to_numpy()
    )
    rotulos_segmento = X_test.index[rastreio_temporal.loc[X_test.index, "segmento"].eq(segmento_id).to_numpy()]
    if len(tempos_transicao) == 0:
        continue
    tempos = datas_base.loc[rotulos_segmento].to_numpy()
    insercao = np.searchsorted(tempos_transicao, tempos)
    anterior = np.where(insercao > 0, (tempos - tempos_transicao[np.clip(insercao - 1, 0, None)])
                        / np.timedelta64(1, "m"), np.inf)
    posterior = np.where(insercao < len(tempos_transicao),
                         (tempos_transicao[np.clip(insercao, None, len(tempos_transicao) - 1)] - tempos)
                         / np.timedelta64(1, "m"), np.inf)
    distancia_transicao_min.loc[rotulos_segmento] = np.minimum(np.abs(anterior), np.abs(posterior))

tabela_previsoes_teste = pd.DataFrame({
    "indice_original": X_test.index,
    "date": datas_base.loc[X_test.index].to_numpy(),
    **{atributo: X_test[atributo].to_numpy() for atributo in atributos_preditores},
    "occupancy_real": y_test.to_numpy(),
    "previsao_baseline": y_pred_baseline.to_numpy(),
    "previsao_knn": y_pred_knn.to_numpy(),
    "previsao_arvore": y_pred_arvore.to_numpy(),
}, index=X_test.index)
tabela_previsoes_teste["erro_knn"] = tipo_erro(y_test.to_numpy(), y_pred_knn.to_numpy())
tabela_previsoes_teste["erro_arvore"] = tipo_erro(y_test.to_numpy(), y_pred_arvore.to_numpy())
tabela_previsoes_teste["minutos_ate_transicao"] = distancia_transicao_min.to_numpy()
tabela_previsoes_teste = tabela_previsoes_teste.sort_values("indice_original").reset_index(drop=True)

tabela_erros_teste = tabela_previsoes_teste.query("erro_knn != 'Acerto' or erro_arvore != 'Acerto'").copy()
tabela_erros_teste["errado_em"] = np.select(
    [tabela_erros_teste["erro_knn"].ne("Acerto") & tabela_erros_teste["erro_arvore"].ne("Acerto"),
     tabela_erros_teste["erro_knn"].ne("Acerto")], ["Ambos", "Somente KNN"], default="Somente Árvore")
assert len(tabela_erros_teste) == ambos_erram + somente_knn_acerta + somente_arvore_acerta

print(f"Registros com erro em pelo menos um modelo: {len(tabela_erros_teste)}")
print(tabela_erros_teste[["indice_original", "date", "Light", "CO2", "Temperature", "occupancy_real", "erro_knn",
                          "erro_arvore", "minutos_ate_transicao"]].to_string(index=False))

linhas_erros = []
for nome, coluna in [("KNN", "erro_knn"), ("Árvore de Decisão", "erro_arvore")]:
    erros = tabela_previsoes_teste[tabela_previsoes_teste[coluna].ne("Acerto")]
    linhas_erros.append({
        "Modelo": nome, "Erros": len(erros),
        "Falsos positivos": int(erros[coluna].eq("Falso positivo").sum()),
        "Falsos negativos": int(erros[coluna].eq("Falso negativo").sum()),
        "Com Light = 0": int(erros["Light"].eq(0).sum()),
        "Com Light > 0": int(erros["Light"].gt(0).sum()),
        f"Até {janela_transicao_min} min de uma transição": int((erros["minutos_ate_transicao"]
                                                                <= janela_transicao_min).sum()),
        "Em fim de semana": int((erros["date"].dt.dayofweek >= 5).sum()),
        "CO2 mediano (ppm)": erros["CO2"].median(),
    })
tabela_resumo_erros = pd.DataFrame(linhas_erros).set_index("Modelo")
print("\n" + tabela_resumo_erros.to_string(float_format=lambda valor: f"{valor:.2f}"))

proporcao_perto_transicao_teste = (tabela_previsoes_teste["minutos_ate_transicao"] <= janela_transicao_min).mean()
print(f"\nReferência: registros do teste a até {janela_transicao_min} min de uma transição: "
      f"{proporcao_perto_transicao_teste:.2%} (todos os 2.484)")
print("Hora dos erros (KNN ∪ Árvore):", tabela_erros_teste["date"].dt.hour.value_counts().sort_index().to_dict())
print("Dia dos erros:", tabela_erros_teste["date"].dt.strftime("%d/%m").value_counts().sort_index().to_dict())

comparacao_fp_fn = tabela_erros_teste.assign(
    tipo=np.where(tabela_erros_teste["occupancy_real"].eq(0), "Falso positivo (real 0)", "Falso negativo (real 1)")
).groupby("tipo")[["Light", "CO2", "Temperature", "minutos_ate_transicao"]].median()
print("\nMedianas por tipo de erro (registros com erro em ao menos um modelo):")
print(comparacao_fp_fn.to_string(float_format=lambda valor: f"{valor:.2f}"))

# %% [markdown]
# ### BLOCO 88 — Impacto operacional e cenários hipotéticos de custo
#
# | Erro | Situação real | Consequências possíveis em uma aplicação hipotética |
# |---|---|---|
# | Falso positivo | Sala desocupada prevista como ocupada | Iluminação e climatização ligadas sem necessidade; consumo evitável |
# | Falso negativo | Sala ocupada prevista como desocupada | Desligamento indevido de luz ou climatização; desconforto; riscos conforme o ambiente |
#
# Os cenários usam `Custo = FP × custo_FP + FN × custo_FN` em **unidades arbitrárias**. Eles mostram como a preferência
# entre os modelos muda conforme o peso de cada erro, e não representam energia, kWh ou valores monetários.

# %%
# BLOCO 88 — Impacto operacional e cenários hipotéticos de custo

cenarios_custo = {"Custos iguais (FP = 1, FN = 1)": (1, 1), "FN com peso 2 (FP = 1, FN = 2)": (1, 2),
                  "FN com peso 5 (FP = 1, FN = 5)": (1, 5), "FP com peso 2 (FP = 2, FN = 1)": (2, 1)}
tabela_custos = pd.DataFrame({
    nome: {cenario: metricas_teste[nome]["FP"] * custo_fp + metricas_teste[nome]["FN"] * custo_fn
           for cenario, (custo_fp, custo_fn) in cenarios_custo.items()}
    for nome in ["DummyClassifier", "KNN", "Árvore de Decisão"]
})
print("Falsos positivos e falsos negativos no teste:")
print(tabela_acertos.loc[:, ["FP", "FN"]].to_string())
print("\nCusto relativo hipotético (unidades arbitrárias; não representa energia nem valores monetários):")
print(tabela_custos.to_string())

# %% [markdown]
# ### BLOCO 89 — Revisão da generalização temporal
#
# Recupera os resultados da Fase 6 sem alterar o protocolo. Para detalhar o fold temporal 3 por grupo de luminosidade,
# os modelos temporários daquele fold são reproduzidos com o mesmo protocolo da Fase 6 (modelos novos, ajustados apenas
# na janela de treino do fold); os modelos finais não são usados nem alterados, e as acurácias devem coincidir com as do
# BLOCO 71.

# %%
# BLOCO 89 — Revisão da generalização temporal (resultados da Fase 6)

proximidade_teste_61 = tabela_proximidade.loc["Base completa", "Até 61 s (%)"]
proximidade_cv = tabela_proximidade_folds.loc["Cinco folds (todas as validações)"]
print(f"Teste oficial com vizinho de treino a até 61 s: {proximidade_teste_61:.4f}%")
print(f"Validação cruzada: até 61 s = {proximidade_cv['Até 61 s (%)']:.4f}% | até 120 s = "
      f"{proximidade_cv['Até 120 s (%)']:.4f}% | mediana = {proximidade_cv['Mediana (s)']:.0f} s")
print("\nAcurácia por divisão temporal (BLOCO 71):")
print(resultados_temporais.xs("acuracia", axis=1, level="metrica").to_string(float_format=lambda v: f"{v:.6f}"))

# Reprodução do fold 3 com o protocolo da Fase 6 (modelos temporários do fold; os modelos finais não são usados)
posicoes_treino_f3, posicoes_validacao_f3 = folds_temporais[2]
X_validacao_f3 = X_train_cronologico.iloc[posicoes_validacao_f3]
y_validacao_f3 = y_train_cronologico.iloc[posicoes_validacao_f3]
luz_zero_f3 = X_validacao_f3["Light"].eq(0)
linhas_fold3 = {"Registros": {"Light = 0": int(luz_zero_f3.sum()), "Light > 0": int((~luz_zero_f3).sum()),
                              "Total": len(y_validacao_f3)}}
for nome, fabrica in [("KNN", lambda: criar_pipeline_knn(k_selecionado)),
                      ("Árvore de Decisão", lambda: criar_arvore(profundidade_selecionada))]:
    previsto_f3 = fabrica().fit(X_train_cronologico.iloc[posicoes_treino_f3],
                                y_train_cronologico.iloc[posicoes_treino_f3]).predict(X_validacao_f3)
    erro_f3 = previsto_f3 != y_validacao_f3.to_numpy()
    assert accuracy_score(y_validacao_f3, previsto_f3) == resultados_temporais.loc[3, (nome, "acuracia")]
    assert (previsto_f3[erro_f3] == 1).all(), "Erro diferente de falso positivo na janela sem ocupação."
    linhas_fold3[f"Erros {nome}"] = {"Light = 0": int((erro_f3 & luz_zero_f3.to_numpy()).sum()),
                                     "Light > 0": int((erro_f3 & ~luz_zero_f3.to_numpy()).sum()),
                                     "Total": int(erro_f3.sum())}
tabela_fold3 = pd.DataFrame(linhas_fold3)
assert int(y_validacao_f3.sum()) == 0, "A janela do fold 3 contém registros ocupados."
print("\nFold temporal 3 (14 e 15/02, sem registros ocupados; todos os erros são falsos positivos):")
print(tabela_fold3.to_string())

# %% [markdown]
# ### BLOCO 90 — Preparação da avaliação complementar no datatraining.txt
#
# `dados/brutos/datatraining.txt` é lido do disco (sem novo download), com a mesma leitura do BLOCO 05, e conferido pelo
# SHA-256 documentado em `dados/README.md`. O arquivo **pertence à mesma coleta da UCI** e cobre um período
# intermediário (04/02 a 10/02) entre os dois arquivos consolidados; não é um edifício independente, uma fonte externa
# nem um período exclusivamente futuro. A inspeção diferencia linhas duplicadas (data, sensores e classe iguais) de
# repetições do vetor de sensores em instantes diferentes, e nenhum registro é removido. A **mesma regra decimal** da
# Fase 2 (`casas_decimais`, BLOCO 19.2) é aplicada aos cinco sensores antes de qualquer cálculo, como aconteceria com
# registros novos em produção; o arquivo bruto não é alterado. O bloco verifica se o arredondamento aumenta as
# coincidências de vetores de sensores com a base principal e com o treino oficial. A variável derivada é calculada com
# a mesma fórmula da Fase 3, sobre os sensores arredondados, e apenas os seis atributos, na ordem oficial, são usados
# como preditores. Nenhum scaler ou modelo é ajustado com essa base, e ela não orienta a escolha da regra decimal nem
# dos hiperparâmetros.

# %%
# BLOCO 90 — Preparação da avaliação complementar no datatraining.txt

sha256_datatraining_documentado = "b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56"  # dados/README.md
with open(caminho_datatraining, "rb") as arquivo:
    assert hashlib.sha256(arquivo.read()).hexdigest() == sha256_datatraining_documentado, "datatraining.txt alterado."

# Mesma leitura do BLOCO 05: a primeira coluna (identificador original da linha) vira o índice
df_complementar = pd.read_csv(caminho_datatraining)
df_complementar.index.name = "id_original"
assert df_complementar.shape == (8143, 7), "datatraining.txt não tem 8.143 registros e 7 colunas."
assert list(df_complementar.columns) == list(df1.columns), "Esquema diferente dos arquivos consolidados."
df_complementar["date"] = pd.to_datetime(df_complementar["date"], format="%Y-%m-%d %H:%M:%S")

colunas_sensores = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
assert all(pd.api.types.is_float_dtype(df_complementar[coluna]) for coluna in colunas_sensores)
assert df_complementar.isna().sum().sum() == 0 and np.isinf(df_complementar[colunas_sensores]).sum().sum() == 0
assert set(df_complementar["Occupancy"].unique()) == {0, 1} and (df_complementar["Light"] >= 0).all()
assert df_complementar["date"].is_monotonic_increasing and df_complementar["date"].is_unique

# Mesma regra decimal da Fase 2, aplicada somente aos cinco sensores (date, Occupancy e índices inalterados)
df_complementar_original = df_complementar.copy(deep=True)
df_complementar = df_complementar.round(casas_decimais)
assert df_complementar.shape == (8143, 7) and df_complementar.index.equals(df_complementar_original.index)
pd.testing.assert_series_equal(df_complementar["Occupancy"], df_complementar_original["Occupancy"], check_exact=True)
pd.testing.assert_series_equal(df_complementar["date"], df_complementar_original["date"], check_exact=True)
assert (df_complementar[colunas_sensores].dtypes == "float64").all()
pd.testing.assert_frame_equal(df_complementar.round(casas_decimais), df_complementar, check_exact=True)
for coluna, casas in casas_decimais.items():
    alteracao = (df_complementar[coluna] - df_complementar_original[coluna]).abs()
    assert (alteracao <= 0.5 * 10.0 ** (-casas) + 4 * np.finfo(np.float64).eps
            * df_complementar_original[coluna].abs().max()).all(), f"Alteração acima do limite em {coluna}."
print("Regra decimal aplicada ao datatraining.txt; valores modificados por atributo (%):",
      {coluna: round((df_complementar[coluna] != df_complementar_original[coluna]).mean() * 100, 2)
       for coluna in colunas_sensores})


def contar_coincidencias(complementar, principal, treino):
    """Repetições internas e vetores de sensores do datatraining.txt presentes na base principal e no treino."""
    vetores_principal = set(map(tuple, principal[colunas_sensores].to_numpy()))
    vetores_treino_oficial = set(map(tuple, treino[colunas_sensores].to_numpy()))
    vetores = list(map(tuple, complementar[colunas_sensores].to_numpy()))
    return {
        "Repetições do vetor de sensores no arquivo": int(complementar.duplicated(subset=colunas_sensores).sum()),
        "Vetores de sensores presentes na base principal": int(sum(v in vetores_principal for v in vetores)),
        "Vetores de sensores presentes no treino oficial": int(sum(v in vetores_treino_oficial for v in vetores)),
    }


tabela_coincidencias_precisao = pd.DataFrame({
    "Antes (precisão original)": contar_coincidencias(
        df_complementar_original, df_consolidado, df_consolidado.loc[X_train.index]),
    "Depois (regra decimal)": contar_coincidencias(df_complementar, df_features, X_train),
})
tabela_coincidencias_precisao["Diferença"] = (
    tabela_coincidencias_precisao["Depois (regra decimal)"] - tabela_coincidencias_precisao["Antes (precisão original)"]
)
print("\nCoincidências de vetores de sensores antes e depois da regra decimal:")
print(tabela_coincidencias_precisao.to_string())

# Sobreposições com a base principal (datatest.txt + datatest2.txt), após a regra decimal
chaves_linha = ["date"] + colunas_sensores + ["Occupancy"]
datas_principais = set(df_features["date"])
vetores_principais = set(map(tuple, df_features[colunas_sensores].to_numpy()))
vetores_treino = set(map(tuple, X_train[colunas_sensores].to_numpy()))
vetores_complementares = list(map(tuple, df_complementar[colunas_sensores].to_numpy()))
linhas_principais = set(map(tuple, df_features[chaves_linha].astype(str).to_numpy()))

intervalos_complementar = df_complementar["date"].diff().dropna()
tabela_inspecao_complementar = pd.Series({
    "Registros": len(df_complementar),
    "Início": df_complementar["date"].min(),
    "Fim": df_complementar["date"].max(),
    "Maior intervalo entre registros": intervalos_complementar.max(),
    "Ausentes / infinitos": 0,
    "Classe 0 / Classe 1": f"{int((df_complementar['Occupancy'] == 0).sum())} / "
                           f"{int((df_complementar['Occupancy'] == 1).sum())}",
    "Ocupados (%)": round(df_complementar["Occupancy"].mean() * 100, 4),
    "Registros em fim de semana": int((df_complementar["date"].dt.dayofweek >= 5).sum()),
    "Linhas completas duplicadas no arquivo": int(df_complementar.duplicated(subset=chaves_linha).sum()),
    "Repetições do vetor de sensores no arquivo": int(df_complementar.duplicated(subset=colunas_sensores).sum()),
    "Timestamps em comum com a base principal": int(df_complementar["date"].isin(datas_principais).sum()),
    "Linhas completas em comum com a base principal": int(sum(
        linha in linhas_principais for linha in map(tuple, df_complementar[chaves_linha].astype(str).to_numpy()))),
    "Vetores de sensores presentes na base principal": int(sum(v in vetores_principais for v in vetores_complementares)),
    "Vetores de sensores presentes no treino oficial": int(sum(v in vetores_treino for v in vetores_complementares)),
})
print(tabela_inspecao_complementar.to_string())
print("\nTipos:", df_complementar.dtypes.astype(str).to_dict())

# Mesma engenharia de atributos da Fase 3 (fórmula inalterada) e mesma ordem de colunas
df_complementar["CO2_Light_Ratio"] = df_complementar["CO2"] / (df_complementar["Light"] + 1)
X_complementar = df_complementar[atributos_preditores].copy()
y_complementar = df_complementar["Occupancy"].copy()
assert list(X_complementar.columns) == atributos_preditores and X_complementar.notna().all().all()
assert np.isfinite(X_complementar.to_numpy()).all()
X_complementar_referencia = X_complementar.copy(deep=True)

# Diferenças de distribuição em relação ao treino oficial (descritivas; nenhum ajuste é feito)
tabela_distribuicao_complementar = pd.DataFrame({
    "Treino oficial (média)": X_train.mean(), "datatraining.txt (média)": X_complementar.mean(),
    "Treino oficial (mediana)": X_train.median(), "datatraining.txt (mediana)": X_complementar.median(),
})
print("\n" + tabela_distribuicao_complementar.to_string(float_format=lambda valor: f"{valor:.4f}"))
luz_desocupado_treino = (X_train["Light"].gt(0) & y_train.eq(0)).sum() / y_train.eq(0).sum()
luz_desocupado_complementar = (X_complementar["Light"].gt(0) & y_complementar.eq(0)).sum() / y_complementar.eq(0).sum()
print(f"\nDesocupados com Light > 0 — treino oficial: {luz_desocupado_treino:.2%} | "
      f"datatraining.txt: {luz_desocupado_complementar:.2%}")
print(f"Ocupados — treino oficial: {y_train.mean():.2%} | datatraining.txt: {y_complementar.mean():.2%}")

# %% [markdown]
# ### BLOCO 91 — Avaliação complementar dos modelos
#
# Os três modelos já ajustados recebem os dados originais do `datatraining.txt` apenas com `predict` (nenhum `fit` ou
# `fit_transform`). Calculam-se as mesmas métricas do teste, conferidas pelas matrizes de confusão, e um resumo
# descritivo por dia. Essa avaliação não seleciona novamente K, `max_depth` ou atributos e não é uma validação
# prospectiva.

# %%
# BLOCO 91 — Avaliação complementar dos modelos (sem nenhum ajuste nessa base)

previsoes_complementares = {
    "DummyClassifier": pd.Series(modelo_baseline.predict(X_complementar), index=X_complementar.index),
    "KNN": pd.Series(modelo_knn_final.predict(X_complementar), index=X_complementar.index),
    "Árvore de Decisão": pd.Series(modelo_arvore_final.predict(X_complementar), index=X_complementar.index),
}
for nome, modelo_avaliado in [("KNN", modelo_knn_final), ("Árvore de Decisão", modelo_arvore_final),
                              ("DummyClassifier", modelo_baseline)]:
    assert assinatura_modelo(modelo_avaliado) == assinaturas_modelos_fase7[nome], f"{nome} alterado."
pd.testing.assert_frame_equal(X_complementar, X_complementar_referencia, check_exact=True)

metricas_complementares = {nome: calcular_metricas(y_complementar, previsoes)
                           for nome, previsoes in previsoes_complementares.items()}
tabela_metricas_complementar = pd.DataFrame(metricas_complementares)
for nome, previsoes in previsoes_complementares.items():
    matriz = confusion_matrix(y_complementar, previsoes, labels=[0, 1])
    assert matriz.sum() == 8143 and np.trace(matriz) / 8143 == metricas_complementares[nome]["accuracy"]
    print(f"\n{nome} — matriz de confusão no datatraining.txt:")
    print(pd.DataFrame(matriz, index=["Real 0", "Real 1"], columns=["Previsto 0", "Previsto 1"]).to_string())
print("\nMétricas no datatraining.txt (8.143 registros; NaN = indefinida):")
print(tabela_metricas_complementar.loc[colunas_percentuais].map(
    lambda valor: "indefinida" if pd.isna(valor) else f"{valor:.6%}").to_string())
print(tabela_metricas_complementar.loc[["TN", "FP", "FN", "TP"]].astype(int).to_string())

# Resultados por dia (descritivo)
dia_complementar = (df_complementar["date"].dt.strftime("%d/%m") + " ("
                    + df_complementar["date"].dt.dayofweek.map(nomes_dias) + ")")
linhas_dia = []
for dia, indices_dia in df_complementar.groupby(dia_complementar, sort=False).groups.items():
    y_dia = y_complementar.loc[indices_dia]
    linha = {"Dia": dia, "Registros": len(indices_dia), "Ocupados": int(y_dia.sum()),
             "Light > 0 (%)": X_complementar.loc[indices_dia, "Light"].gt(0).mean() * 100}
    for nome in ["KNN", "Árvore de Decisão"]:
        previsto_dia = previsoes_complementares[nome].loc[indices_dia]
        linha[f"Acurácia {nome}"] = accuracy_score(y_dia, previsto_dia)
        linha[f"FP {nome}"] = int(((previsto_dia == 1) & (y_dia == 0)).sum())
        linha[f"FN {nome}"] = int(((previsto_dia == 0) & (y_dia == 1)).sum())
    linhas_dia.append(linha)
tabela_dias_complementar = pd.DataFrame(linhas_dia).set_index("Dia")
print("\nResultados por dia no datatraining.txt:")
print(tabela_dias_complementar.to_string(float_format=lambda valor: f"{valor:.4f}"))

# Erros por grupo de luminosidade
for nome in ["KNN", "Árvore de Decisão"]:
    erro = previsoes_complementares[nome].ne(y_complementar)
    print(f"{nome}: erros com Light = 0: {int((erro & X_complementar['Light'].eq(0)).sum())} | "
          f"com Light > 0: {int((erro & X_complementar['Light'].gt(0)).sum())}")

# %% [markdown]
# ### BLOCO 92 — Comparação entre protocolos de avaliação
#
# Os cinco protocolos têm naturezas diferentes: o treino completo avalia os modelos nos próprios dados de ajuste; a
# validação cruzada e a temporal são médias de modelos ajustados em partições do treino; o teste oficial e o
# `datatraining.txt` avaliam os modelos finais em conjuntos completos. Os tamanhos, as composições e a proximidade
# temporal diferem, e nenhum intervalo de confiança é calculado.

# %%
# BLOCO 92 — Comparação entre protocolos de avaliação

metricas_treino_completo = {nome: calcular_metricas(y_train, modelo_avaliado.predict(X_train))
                            for nome, modelo_avaliado in [("KNN", modelo_knn_final),
                                                          ("Árvore de Decisão", modelo_arvore_final)]}
linhas_protocolos = []
for nome, rotulo, resumo in [("KNN", k_selecionado_rotulo, resumo_knn),
                             ("Árvore de Decisão", profundidade_selecionada_rotulo, resumo_arvore)]:
    temporal = registros_temporais.query("modelo == @nome and conjunto == 'validacao'")
    protocolos_modelo = [
        ("1. Treino completo", "Modelo final avaliado nos próprios dados de ajuste", "9.933 (os mesmos do ajuste)",
         metricas_treino_completo[nome]["accuracy"], metricas_treino_completo[nome]["balanced_accuracy"],
         metricas_treino_completo[nome]["f1_ocupado"]),
        ("2. Validação cruzada estratificada", "Média de 5 folds; 5 modelos ajustados em partições do treino",
         "5 × 1.986–1.987", resumo.loc[rotulo, "Acurácia validação (média)"],
         resumo.loc[rotulo, "Balanced accuracy validação (média)"], resumo.loc[rotulo, "F1 ocupado validação (média)"]),
        ("3. Teste oficial", "Modelo final; 20% sorteados da mesma sequência (96,10% com vizinho a até 61 s)",
         "2.484", metricas_teste[nome]["accuracy"], metricas_teste[nome]["balanced_accuracy"],
         metricas_teste[nome]["f1_ocupado"]),
        ("4. Validação temporal complementar", "Média de 5 janelas cronológicas; treinos de 1.598 a 8.218 registros",
         "5 × 1.655", temporal.query("metrica == 'acuracia'")["resultado"].mean(),
         temporal.query("metrica == 'balanced_accuracy'")["resultado"].mean(),
         temporal.query("metrica == 'f1_ocupado'")["resultado"].mean()),
        ("5. datatraining.txt", "Modelo final; período intermediário da mesma coleta, fora da base consolidada",
         "8.143", metricas_complementares[nome]["accuracy"], metricas_complementares[nome]["balanced_accuracy"],
         metricas_complementares[nome]["f1_ocupado"]),
    ]
    for protocolo, natureza, registros, acuracia, balanceada, f1_ocupado in protocolos_modelo:
        linhas_protocolos.append({"Modelo": nome, "Protocolo": protocolo, "Natureza": natureza,
                                  "Registros avaliados": registros, "Acurácia": acuracia,
                                  "Balanced accuracy": balanceada, "F1 ocupado": f1_ocupado})
tabela_protocolos = pd.DataFrame(linhas_protocolos).set_index(["Modelo", "Protocolo"])
print(tabela_protocolos.to_string(float_format=lambda valor: f"{valor:.4%}"))
print("\nNa validação temporal, a acurácia é a média dos 5 folds; a balanced accuracy e o F1 são médias dos 4 folds")
print("calculáveis (o fold 3 não tem registros ocupados). Os protocolos não são equivalentes nem independentes.")

# %% [markdown]
# ### BLOCO 93 — Variável derivada e importância dos atributos
#
# Recupera, sem novos treinamentos, a comparação de 5 e 6 atributos da Fase 6 e a importância por redução de impureza
# (`feature_importances_`) da Árvore final.

# %%
# BLOCO 93 — Variável derivada e importância dos atributos (resultados da Fase 6)

print("Comparação 5 × 6 atributos na validação cruzada (diferença média 6 − 5, em p.p., na acurácia):")
print(resumo_feature_engineering.xs("acuracia", level="metrica")[["media_5", "media_6", "diferenca_media",
                                                                  "folds_6_melhor", "folds_5_melhor", "folds_empate"]]
      .to_string(float_format=lambda valor: f"{valor:.4f}"))
print(f"\nKNN (K={k_selecionado}), recorte do fold 1: vizinhos diferentes em {vizinhos_diferentes:.2f}% das validações; "
      f"previsões diferentes: {previsoes_diferentes}")
print("\nImportância por redução de impureza na Árvore final (soma = 100%):")
print((importancias_arvore * 100).round(4).to_string())

# %% [markdown]
# ### BLOCO 94 — Elementos do veredito técnico
#
# Reúne as evidências usadas no veredito: métricas no teste e no `datatraining.txt`, pior janela temporal,
# armazenamento, tempo de previsão (informativo, varia entre execuções), interpretabilidade e necessidade de
# padronização. As regras da Árvore final são exibidas com os limiares na escala original.

# %%
# BLOCO 94 — Elementos do veredito técnico

# Tempo de previsão no datatraining.txt (informativo; varia entre execuções e não é exportado)
tempos_previsao = {}
for nome, modelo_avaliado in [("KNN", modelo_knn_final), ("Árvore de Decisão", modelo_arvore_final)]:
    inicio = time.perf_counter()
    modelo_avaliado.predict(X_complementar)
    tempos_previsao[nome] = time.perf_counter() - inicio

tabela_veredito = pd.DataFrame({
    f"KNN (K={k_selecionado})": {
        "Teste: acurácia": metricas_teste["KNN"]["accuracy"],
        "Teste: balanced accuracy": metricas_teste["KNN"]["balanced_accuracy"],
        "Teste: precision / recall (ocupado)":
            f"{metricas_teste['KNN']['precision_ocupado']:.4%} / {metricas_teste['KNN']['recall_ocupado']:.4%}",
        "Teste: FP / FN": f"{metricas_teste['KNN']['FP']} / {metricas_teste['KNN']['FN']}",
        "datatraining.txt: acurácia": metricas_complementares["KNN"]["accuracy"],
        "datatraining.txt: recall (ocupado)": metricas_complementares["KNN"]["recall_ocupado"],
        "datatraining.txt: FP / FN":
            f"{metricas_complementares['KNN']['FP']} / {metricas_complementares['KNN']['FN']}",
        "Temporal: acurácia mínima (fold 3)": resultados_temporais.loc[3, ("KNN", "acuracia")],
        "Armazenamento": f"{knn_avaliado.n_samples_fit_} registros × 6 atributos + 12 parâmetros do scaler",
        "Previsão de 8.143 registros (s, informativo)": f"{tempos_previsao['KNN']:.4f}",
        "Interpretabilidade": f"Baixa: decisão por votação de {k_selecionado} vizinhos",
        "Padronização": "Obrigatória (Pipeline)",
    },
    f"Árvore (max_depth={profundidade_selecionada_rotulo})": {
        "Teste: acurácia": metricas_teste["Árvore de Decisão"]["accuracy"],
        "Teste: balanced accuracy": metricas_teste["Árvore de Decisão"]["balanced_accuracy"],
        "Teste: precision / recall (ocupado)":
            f"{metricas_teste['Árvore de Decisão']['precision_ocupado']:.4%} / "
            f"{metricas_teste['Árvore de Decisão']['recall_ocupado']:.4%}",
        "Teste: FP / FN": f"{metricas_teste['Árvore de Decisão']['FP']} / {metricas_teste['Árvore de Decisão']['FN']}",
        "datatraining.txt: acurácia": metricas_complementares["Árvore de Decisão"]["accuracy"],
        "datatraining.txt: recall (ocupado)": metricas_complementares["Árvore de Decisão"]["recall_ocupado"],
        "datatraining.txt: FP / FN": f"{metricas_complementares['Árvore de Decisão']['FP']} / "
                                     f"{metricas_complementares['Árvore de Decisão']['FN']}",
        "Temporal: acurácia mínima (fold 3)": resultados_temporais.loc[3, ("Árvore de Decisão", "acuracia")],
        "Armazenamento": f"{modelo_arvore_final.tree_.node_count} nós ({modelo_arvore_final.get_n_leaves()} folhas)",
        "Previsão de 8.143 registros (s, informativo)": f"{tempos_previsao['Árvore de Decisão']:.4f}",
        "Interpretabilidade": "Alta: regras explícitas do tipo atributo ≤ limiar",
        "Padronização": "Desnecessária",
    },
})
print(tabela_veredito.map(lambda valor: f"{valor:.4%}" if isinstance(valor, float) else valor).to_string())
print("\nRegras da Árvore de Decisão final (limiares na escala original):")
print(export_text(modelo_arvore_final, feature_names=atributos_preditores, decimals=3))

# %% [markdown]
# ### BLOCO 95 — Visualização das matrizes de confusão
#
# Cada célula mostra o tipo (TN, FP, FN, TP), a contagem absoluta e o percentual em relação à **classe real** (cada
# linha soma 100%: denominadores de 1.880 desocupados e 604 ocupados). A mesma escala de cores, de 0% a 100%, é usada
# nos três painéis.

# %%
# BLOCO 95 — Visualização das matrizes de confusão

nomes_modelos = ["DummyClassifier", "KNN", "Árvore de Decisão"]
titulos_modelos = {"DummyClassifier": "DummyClassifier (classe majoritária)", "KNN": f"KNN (K = {k_selecionado}, Pipeline)",
                   "Árvore de Decisão": f"Árvore de Decisão (max_depth = {profundidade_selecionada_rotulo})"}
rotulos_celulas = [["TN", "FP"], ["FN", "TP"]]

fig, eixos = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={"wspace": 0.12})
for eixo, nome in zip(eixos, nomes_modelos):
    matriz = matrizes_confusao[nome]
    percentual_linha = matriz / matriz.sum(axis=1, keepdims=True) * 100
    imagem = eixo.imshow(percentual_linha, cmap="Blues", vmin=0, vmax=100)
    for linha in range(2):
        for coluna in range(2):
            cor_texto = "white" if percentual_linha[linha, coluna] > 60 else "black"
            eixo.text(coluna, linha,
                      f"{rotulos_celulas[linha][coluna]}\n{formatar_inteiro(matriz[linha, coluna])}\n"
                      f"({formatar_percentual(percentual_linha[linha, coluna])})",
                      ha="center", va="center", fontsize=13, color=cor_texto)
    eixo.set_xticks([0, 1], ["Previsto\ndesocupado (0)", "Previsto\nocupado (1)"])
    eixo.set_yticks([0, 1], ["Realmente\ndesocupado (0)", "Realmente\nocupado (1)"] if eixo is eixos[0] else [])
    eixo.grid(False)
    acertos = int(np.trace(matriz))
    eixo.set_title(f"{titulos_modelos[nome]}\n{formatar_inteiro(acertos)} acertos e "
                   f"{formatar_inteiro(matriz.sum() - acertos)} erros em 2.484")
barra_cor = fig.colorbar(imagem, ax=eixos, fraction=0.02, pad=0.02)
barra_cor.set_label("Percentual da classe real (cada linha soma 100%)")
fig.suptitle("Fase 7 — Matrizes de confusão no teste oficial (contagens e percentual por classe real: "
             "1.880 desocupados e 604 ocupados)", fontsize=15)
plt.savefig(caminho_grafico_matrizes, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_matrizes)
plt.show()

# %% [markdown]
# ### BLOCO 96 — Visualização comparativa das métricas
#
# O primeiro painel usa a escala completa (0% a 100%) e identifica a precision indefinida do baseline. O segundo amplia o
# eixo para 95% a 100% para mostrar as diferenças pequenas entre KNN e Árvore. O terceiro mostra falsos positivos e
# falsos negativos, com o eixo cortado em 30 registros (os 604 falsos negativos do baseline aparecem indicados).

# %%
# BLOCO 96 — Visualização comparativa das métricas

metricas_grafico_final = [("accuracy", "Acurácia"), ("balanced_accuracy", "Balanced\naccuracy"),
                          ("precision_ocupado", "Precision\n(ocupado)"), ("recall_ocupado", "Recall\n(ocupado)"),
                          ("f1_ocupado", "F1\n(ocupado)"), ("especificidade", "Especificidade")]
cores_modelos = {"DummyClassifier": "#BBBBBB", "KNN": "#4C72B0", "Árvore de Decisão": "#55A868"}

fig, (eixo_geral, eixo_ampliado, eixo_erros) = plt.subplots(
    1, 3, figsize=(21, 6.8), gridspec_kw={"width_ratios": [1.5, 1.1, 0.8]})
posicoes_metricas = np.arange(len(metricas_grafico_final))
largura = 0.27
for deslocamento, nome in zip([-largura, 0, largura], nomes_modelos):
    valores = np.array([metricas_teste[nome][chave] for chave, _ in metricas_grafico_final]) * 100
    barras = eixo_geral.bar(posicoes_metricas + deslocamento, np.nan_to_num(valores), width=largura,
                            color=cores_modelos[nome], edgecolor="black", label=nome)
    for posicao, valor in zip(posicoes_metricas + deslocamento, valores):
        if np.isnan(valor):
            eixo_geral.text(posicao, 3, "indefinida\n(TP + FP = 0)", ha="center", va="bottom", fontsize=8,
                            rotation=90)
        elif valor == 0:
            eixo_geral.text(posicao, 1, "0%", ha="center", va="bottom", fontsize=8)
eixo_geral.set_xticks(posicoes_metricas, [rotulo for _, rotulo in metricas_grafico_final])
eixo_geral.set_ylim(0, 105)
eixo_geral.set_ylabel("Valor no teste oficial (%)")
formatar_eixo_percentual(eixo_geral, 0)
eixo_geral.set_title("Métricas dos três modelos (escala completa, 0% a 100%)")
eixo_geral.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)

# Painel ampliado: somente KNN e Árvore, eixo de 95% a 100%
for deslocamento, nome in zip([-0.18, 0.18], ["KNN", "Árvore de Decisão"]):
    valores = np.array([metricas_teste[nome][chave] for chave, _ in metricas_grafico_final]) * 100
    eixo_ampliado.bar(posicoes_metricas + deslocamento, valores, width=0.36, color=cores_modelos[nome],
                      edgecolor="black", label=nome)
    for posicao, valor in zip(posicoes_metricas + deslocamento, valores):
        eixo_ampliado.text(posicao, valor + 0.05, formatar_decimal(valor, 2), ha="center", va="bottom", fontsize=8,
                           rotation=90)
eixo_ampliado.set_xticks(posicoes_metricas, [rotulo for _, rotulo in metricas_grafico_final], fontsize=9)
eixo_ampliado.set_ylim(95, 100.9)
eixo_ampliado.set_ylabel("Valor no teste oficial (eixo ampliado: 95% a 100%)")
formatar_eixo_percentual(eixo_ampliado, 0)
eixo_ampliado.set_title("KNN × Árvore (eixo ampliado)")
eixo_ampliado.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

# Painel de erros: falsos positivos e falsos negativos
posicoes_tipos = np.arange(2)
limite_erros = 30
for deslocamento, nome in zip([-largura, 0, largura], nomes_modelos):
    valores = [metricas_teste[nome]["FP"], metricas_teste[nome]["FN"]]
    eixo_erros.bar(posicoes_tipos + deslocamento, np.minimum(valores, limite_erros), width=largura,
                   color=cores_modelos[nome], edgecolor="black", label=nome)
    for posicao, valor in zip(posicoes_tipos + deslocamento, valores):
        texto = f"{formatar_inteiro(valor)}\n(fora da\nescala)" if valor > limite_erros else formatar_inteiro(valor)
        eixo_erros.text(posicao, min(valor, limite_erros) + 0.4, texto, ha="center", va="bottom", fontsize=9)
eixo_erros.set_xticks(posicoes_tipos, ["Falsos positivos\n(desocupado → ocupado)", "Falsos negativos\n(ocupado → desocupado)"])
eixo_erros.set_ylim(0, limite_erros + 9)
eixo_erros.set_ylabel(f"Registros (eixo cortado em {limite_erros})")
eixo_erros.set_title("Erros no teste oficial")
eixo_erros.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=9)

fig.suptitle("Fase 7 — Comparação das métricas no teste oficial (2.484 registros; classe positiva = ocupado)",
             fontsize=15)
plt.tight_layout()
plt.savefig(caminho_grafico_metricas, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_metricas)
plt.show()

# %% [markdown]
# ### BLOCO 97 — Visualização dos erros
#
# Quatro painéis descritivos: erros compartilhados e exclusivos, tipos de erro, posição dos erros no espaço Light × CO2
# e proporção de registros próximos de transições de ocupação. As associações não indicam causalidade.

# %%
# BLOCO 97 — Visualização dos erros

fig, eixos = plt.subplots(2, 2, figsize=(18, 12.5))
eixo_categorias, eixo_tipos, eixo_dispersao, eixo_transicao = eixos.ravel()

# Painel 1 — Erros compartilhados e exclusivos
categorias_erro = {"Ambos erram": ambos_erram, "Só o KNN erra": somente_arvore_acerta,
                   "Só a Árvore erra": somente_knn_acerta}
barras = eixo_categorias.bar(list(categorias_erro), list(categorias_erro.values()),
                             color=["#C44E52", cores_modelos["KNN"], cores_modelos["Árvore de Decisão"]],
                             edgecolor="black")
for barra, valor in zip(barras, categorias_erro.values()):
    eixo_categorias.text(barra.get_x() + barra.get_width() / 2, valor + 0.2, formatar_inteiro(valor), ha="center",
                         fontsize=11)
eixo_categorias.set_ylabel("Registros do teste")
eixo_categorias.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
eixo_categorias.set_ylim(0, max(categorias_erro.values()) * 1.25)
eixo_categorias.set_title(f"Erros compartilhados e exclusivos ({formatar_inteiro(ambos_acertam)} registros "
                          f"acertados pelos dois)")

# Painel 2 — Tipos de erro por modelo
for posicao, (nome, coluna) in enumerate([("KNN", "erro_knn"), ("Árvore de Decisão", "erro_arvore")]):
    falsos_positivos = tabela_resumo_erros.loc[nome, "Falsos positivos"]
    falsos_negativos = tabela_resumo_erros.loc[nome, "Falsos negativos"]
    eixo_tipos.bar(posicao, falsos_positivos, color="#DD8452", edgecolor="black",
                   label="Falso positivo" if posicao == 0 else None)
    eixo_tipos.bar(posicao, falsos_negativos, bottom=falsos_positivos, color="#8172B3", edgecolor="black",
                   label="Falso negativo" if posicao == 0 else None)
    eixo_tipos.text(posicao, falsos_positivos / 2, formatar_inteiro(falsos_positivos), ha="center", va="center",
                    fontsize=11)
    eixo_tipos.text(posicao, falsos_positivos + falsos_negativos / 2, formatar_inteiro(falsos_negativos),
                    ha="center", va="center", fontsize=11, color="white")
eixo_tipos.set_xticks([0, 1], ["KNN", "Árvore de Decisão"])
eixo_tipos.set_ylabel("Erros no teste oficial")
eixo_tipos.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
eixo_tipos.set_title("Tipos de erro (nenhum erro com Light = 0 em nenhum modelo)")
eixo_tipos.legend(loc="upper left")

# Painel 3 — Luminosidade e CO2 dos registros do teste, com os erros destacados
eixo_dispersao.scatter(tabela_previsoes_teste["Light"], tabela_previsoes_teste["CO2"], s=6, color="lightgray",
                       label="Registros do teste (acertos dos dois modelos)")
estilos_erro = {"Ambos": ("X", "#C44E52"), "Somente KNN": ("o", cores_modelos["KNN"]),
                "Somente Árvore": ("s", cores_modelos["Árvore de Decisão"])}
for categoria, (marcador, cor) in estilos_erro.items():
    selecao = tabela_erros_teste[tabela_erros_teste["errado_em"].eq(categoria)]
    for real, preenchimento in [(0, cor), (1, "white")]:
        parte = selecao[selecao["occupancy_real"].eq(real)]
        eixo_dispersao.scatter(parte["Light"], parte["CO2"], marker=marcador, s=70, facecolor=preenchimento,
                               edgecolor=cor, linewidth=1.8, zorder=3,
                               label=f"{categoria} erra — real {'desocupado (FP)' if real == 0 else 'ocupado (FN)'}"
                               if len(parte) else None)
eixo_dispersao.set_xlabel("Light (lux)")
eixo_dispersao.set_ylabel("CO2 (ppm)")
eixo_dispersao.set_title("Erros no espaço Light × CO2 (preenchido = FP; vazado = FN)")
eixo_dispersao.legend(loc="upper right", fontsize=8.5)

# Painel 4 — Proximidade de transições de ocupação
proporcoes_transicao = {
    "Todos os registros\ndo teste": proporcao_perto_transicao_teste * 100,
    "Erros do KNN": tabela_resumo_erros.loc["KNN", f"Até {janela_transicao_min} min de uma transição"]
    / tabela_resumo_erros.loc["KNN", "Erros"] * 100,
    "Erros da Árvore": tabela_resumo_erros.loc["Árvore de Decisão", f"Até {janela_transicao_min} min de uma transição"]
    / tabela_resumo_erros.loc["Árvore de Decisão", "Erros"] * 100,
}
barras = eixo_transicao.bar(list(proporcoes_transicao), list(proporcoes_transicao.values()),
                            color=["lightgray", cores_modelos["KNN"], cores_modelos["Árvore de Decisão"]],
                            edgecolor="black")
for barra, valor in zip(barras, proporcoes_transicao.values()):
    eixo_transicao.text(barra.get_x() + barra.get_width() / 2, valor + 1, formatar_percentual(valor), ha="center",
                        fontsize=11)
eixo_transicao.set_ylim(0, 105)
eixo_transicao.set_ylabel(f"Registros a até {janela_transicao_min} minutos de uma mudança de classe (%)")
formatar_eixo_percentual(eixo_transicao, 0)
eixo_transicao.set_title("Proximidade de transições de ocupação (análise exploratória)")

fig.suptitle("Fase 7 — Investigação descritiva dos erros no teste oficial", fontsize=16)
plt.tight_layout()
plt.savefig(caminho_grafico_erros, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_erros)
plt.show()

# %% [markdown]
# ### BLOCO 98 — Visualização da generalização
#
# Compara a acurácia nos cinco protocolos, as janelas temporais (com destaque para o fold 3), os resultados diários no
# `datatraining.txt` e o desempenho por classe no teste e no `datatraining.txt`. A média temporal inclui o fold 3.

# %%
# BLOCO 98 — Visualização da generalização

fig, eixos = plt.subplots(2, 2, figsize=(18, 12.5))
eixo_protocolos, eixo_folds, eixo_dias, eixo_classes = eixos.ravel()

# Painel 1 — Acurácia por protocolo
rotulos_protocolos = ["Treino\ncompleto", "Validação\ncruzada (média)", "Teste\noficial",
                      "Temporal\n(média de 5)", "datatraining.txt"]
posicoes_protocolos = np.arange(len(rotulos_protocolos))
for deslocamento, nome in zip([-0.18, 0.18], ["KNN", "Árvore de Decisão"]):
    valores = tabela_protocolos.loc[nome, "Acurácia"].to_numpy() * 100
    eixo_protocolos.bar(posicoes_protocolos + deslocamento, valores, width=0.36, color=cores_modelos[nome],
                        edgecolor="black", label=nome)
    for posicao, valor in zip(posicoes_protocolos + deslocamento, valores):
        eixo_protocolos.text(posicao, valor + 0.3, formatar_decimal(valor, 2), ha="center", fontsize=9, rotation=90,
                             va="bottom")
eixo_protocolos.axhline(acuracia_teste_baseline * 100, color="gray", linestyle=":", linewidth=1.5,
                        label="DummyClassifier no teste (75,68%)")
eixo_protocolos.set_xticks(posicoes_protocolos, rotulos_protocolos)
eixo_protocolos.set_ylim(70, 103)
eixo_protocolos.set_ylabel("Acurácia (eixo de 70% a 103%)")
formatar_eixo_percentual(eixo_protocolos, 0)
eixo_protocolos.set_title("Acurácia por protocolo (protocolos não equivalentes)")
eixo_protocolos.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9, frameon=False)

# Painel 2 — Janelas temporais, com destaque para o fold 3
for deslocamento, nome in zip([-0.18, 0.18], ["KNN", "Árvore de Decisão"]):
    valores = resultados_temporais[(nome, "acuracia")].to_numpy() * 100
    eixo_folds.bar(posicoes_temporais + deslocamento, valores, width=0.36, color=cores_modelos[nome],
                   edgecolor="black", label=nome)
    for posicao, valor in zip(posicoes_temporais + deslocamento, valores):
        eixo_folds.text(posicao, valor + 1, formatar_decimal(valor, 1), ha="center", fontsize=9)
eixo_folds.axvspan(2.55, 3.45, color="#C44E52", alpha=0.12, label="Fold 3: fim de semana sem ocupação")
eixo_folds.set_xticks(posicoes_temporais, [f"Fold {numero}" for numero in posicoes_temporais])
eixo_folds.set_ylim(0, 110)
eixo_folds.set_ylabel("Acurácia na janela de validação (%)")
formatar_eixo_percentual(eixo_folds, 0)
eixo_folds.set_title(f"Validação temporal (Fase 6): fold 3 — KNN {int(tabela_fold3.loc['Total', 'Erros KNN'])} e "
                     f"Árvore {int(tabela_fold3.loc['Total', 'Erros Árvore de Decisão'])} falsos positivos")
eixo_folds.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9, frameon=False)

# Painel 3 — datatraining.txt por dia
posicoes_dias = np.arange(len(tabela_dias_complementar))
for deslocamento, nome in zip([-0.18, 0.18], ["KNN", "Árvore de Decisão"]):
    valores = tabela_dias_complementar[f"Acurácia {nome}"].to_numpy() * 100
    eixo_dias.bar(posicoes_dias + deslocamento, valores, width=0.36, color=cores_modelos[nome], edgecolor="black",
                  label=nome)
    for posicao, valor in zip(posicoes_dias + deslocamento, valores):
        eixo_dias.text(posicao, valor + 0.3, formatar_decimal(valor, 1), ha="center", fontsize=8.5, rotation=90,
                       va="bottom")
eixo_dias.set_xticks(posicoes_dias, [f"{dia}\n{formatar_inteiro(registros)} reg."
                                     for dia, registros in tabela_dias_complementar["Registros"].items()], fontsize=9)
eixo_dias.set_ylim(80, 103)
eixo_dias.set_ylabel("Acurácia no dia (eixo de 80% a 103%)")
formatar_eixo_percentual(eixo_dias, 0)
eixo_dias.set_title("datatraining.txt por dia (07 e 08/02: fim de semana sem ocupação)")
eixo_dias.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9, frameon=False)

# Painel 4 — Recall e especificidade: teste oficial × datatraining.txt
rotulos_classes = ["Recall (ocupado)\nteste", "Recall (ocupado)\ndatatraining", "Especificidade\nteste",
                   "Especificidade\ndatatraining"]
posicoes_classes = np.arange(len(rotulos_classes))
for deslocamento, nome in zip([-0.18, 0.18], ["KNN", "Árvore de Decisão"]):
    valores = np.array([metricas_teste[nome]["recall_ocupado"], metricas_complementares[nome]["recall_ocupado"],
                        metricas_teste[nome]["especificidade"], metricas_complementares[nome]["especificidade"]]) * 100
    eixo_classes.bar(posicoes_classes + deslocamento, valores, width=0.36, color=cores_modelos[nome],
                     edgecolor="black", label=nome)
    for posicao, valor in zip(posicoes_classes + deslocamento, valores):
        eixo_classes.text(posicao, valor + 0.2, formatar_decimal(valor, 2), ha="center", fontsize=9, rotation=90,
                          va="bottom")
eixo_classes.set_xticks(posicoes_classes, rotulos_classes, fontsize=9)
eixo_classes.set_ylim(90, 102.5)
eixo_classes.set_ylabel("Valor (eixo de 90% a 102,5%)")
formatar_eixo_percentual(eixo_classes, 0)
eixo_classes.set_title("Desempenho por classe: teste oficial × datatraining.txt")
eixo_classes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9, frameon=False)

fig.text(0.5, -0.035, "Validação cruzada e temporal: médias de modelos ajustados em partições do treino. Teste, treino "
         "completo e datatraining.txt: modelos finais. A média temporal inclui o fold 3; datatraining.txt é um período "
         "intermediário da mesma coleta, não uma validação externa.", ha="center", fontsize=10.5, wrap=True)
fig.suptitle("Fase 7 — Generalização dos modelos selecionados em diferentes protocolos", fontsize=16)
plt.tight_layout()
plt.savefig(caminho_grafico_generalizacao, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_generalizacao)
plt.show()

# %% [markdown]
# ### BLOCO 99 — Exportação dos resultados finais
#
# | Arquivo | Conteúdo |
# |---|---|
# | `metricas_avaliacao_final.csv` | Métricas dos três modelos no teste oficial, em formato longo, com observação das métricas indefinidas |
# | `previsoes_teste.csv` | As 2.484 observações do teste: índice original, data, atributos, classe real, previsões, tipo de erro e distância até a transição mais próxima |
# | `analise_erros_teste.csv` | Registros errados por pelo menos um dos modelos principais |
# | `metricas_avaliacao_complementar.csv` | Métricas dos três modelos no `datatraining.txt` |
# | `previsoes_avaliacao_complementar.csv` | As 8.143 observações do `datatraining.txt`, com o identificador original da linha |
#
# Sem arredondamento e sem tempos de execução; a recarga com `float_precision="round_trip"` é comparada com os dados em
# memória.

# %%
# BLOCO 99 — Exportação dos resultados finais


def tabela_metricas_longa(metricas_por_modelo, conjunto, registros):
    linhas = []
    for nome in nomes_modelos:
        for metrica, valor in metricas_por_modelo[nome].items():
            observacao = "medida"
            if pd.isna(valor):
                observacao = "indefinida: denominador zero (TP + FP = 0, nenhuma previsão positiva)"
            elif nome == "DummyClassifier" and metrica == "precision_macro":
                observacao = "média apenas da classe com precision definida (classe 1 indefinida)"
            linhas.append({"conjunto": conjunto, "modelo": nome, "metrica": metrica, "valor": valor,
                           "observacoes_avaliadas": registros, "observacao": observacao})
    return pd.DataFrame(linhas)


metricas_avaliacao_final = tabela_metricas_longa(metricas_teste, "teste_oficial", len(y_test))
metricas_avaliacao_complementar = tabela_metricas_longa(metricas_complementares, "datatraining.txt",
                                                        len(y_complementar))

previsoes_teste_exportacao = tabela_previsoes_teste.copy()
assert np.isfinite(previsoes_teste_exportacao["minutos_ate_transicao"]).all()
previsoes_teste_exportacao["date"] = previsoes_teste_exportacao["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
analise_erros_exportacao = tabela_erros_teste.copy()
analise_erros_exportacao["date"] = analise_erros_exportacao["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

previsoes_complementar_exportacao = pd.DataFrame({
    "id_original": X_complementar.index,
    "date": df_complementar["date"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
    **{atributo: X_complementar[atributo].to_numpy() for atributo in atributos_preditores},
    "occupancy_real": y_complementar.to_numpy(),
    "previsao_baseline": previsoes_complementares["DummyClassifier"].to_numpy(),
    "previsao_knn": previsoes_complementares["KNN"].to_numpy(),
    "previsao_arvore": previsoes_complementares["Árvore de Decisão"].to_numpy(),
    "erro_knn": tipo_erro(y_complementar.to_numpy(), previsoes_complementares["KNN"].to_numpy()),
    "erro_arvore": tipo_erro(y_complementar.to_numpy(), previsoes_complementares["Árvore de Decisão"].to_numpy()),
})

exportacoes_fase7 = {
    caminho_metricas_avaliacao_final: (metricas_avaliacao_final, ["modelo", "metrica"]),
    caminho_previsoes_teste: (previsoes_teste_exportacao, ["indice_original"]),
    caminho_analise_erros_teste: (analise_erros_exportacao, ["indice_original"]),
    caminho_metricas_avaliacao_complementar: (metricas_avaliacao_complementar, ["modelo", "metrica"]),
    caminho_previsoes_avaliacao_complementar: (previsoes_complementar_exportacao, ["id_original"]),
}
linhas_exportacao_fase7 = []
for caminho, (tabela, chaves) in exportacoes_fase7.items():
    assert not tabela.duplicated(subset=chaves).any(), f"Duplicidade em {caminho}."
    tabela.to_csv(caminho, index=False)
    recarregada = pd.read_csv(caminho, float_precision="round_trip", keep_default_na=False, na_values=[""])
    pd.testing.assert_frame_equal(recarregada, tabela.reset_index(drop=True), check_exact=True, check_dtype=False)
    with open(caminho, "rb") as arquivo:
        linhas_exportacao_fase7.append({"Arquivo": caminho, "Linhas": len(tabela), "Colunas": tabela.shape[1],
                                        "SHA-256": hashlib.sha256(arquivo.read()).hexdigest()})
assert len(previsoes_teste_exportacao) == 2484 and len(previsoes_complementar_exportacao) == 8143
tabela_exportacao_fase7 = pd.DataFrame(linhas_exportacao_fase7).set_index("Arquivo")
print(tabela_exportacao_fase7.to_string())
print("\nRecarga com float_precision='round_trip' idêntica aos resultados em memória: True")

# %% [markdown]
# ### BLOCO 100 — Auditoria final contra Data Leakage
#
# As verificações abaixo reduzem, mas não eliminam, os riscos metodológicos. Permanecem: a EDA feita na base completa; a
# escolha da variável derivada após a exploração dos dados; a dependência temporal entre treino e teste; a consulta ao
# teste na Fase 6 por exigência acadêmica; a seleção de hiperparâmetros pela própria validação cruzada; a avaliação
# complementar proveniente da mesma coleta; e a ausência de validação em outros edifícios.

# %%
# BLOCO 100 — Auditoria final contra Data Leakage

# Conjuntos, split e seleção
for nome, estrutura in estruturas_fase6.items():
    assert assinatura_integridade(estrutura) == assinaturas_fase6[nome], f"{nome} alterado na Fase 7."
assert repr(selecao_hiperparametros) == registro_selecao
pd.testing.assert_frame_equal(X, X_referencia, check_exact=True)
pd.testing.assert_series_equal(y, y_referencia, check_exact=True)

# Modelos principais e baseline: parâmetros e atributos aprendidos inalterados após todas as previsões
for nome, modelo_avaliado in [("KNN", modelo_knn_final), ("Árvore de Decisão", modelo_arvore_final),
                              ("DummyClassifier", modelo_baseline)]:
    assert assinatura_modelo(modelo_avaliado) == assinaturas_modelos_fase7[nome], f"{nome} alterado."
assert int(modelo_knn_final.named_steps["scaler"].n_samples_seen_) == len(X_train)
assert modelo_baseline.n_features_in_ == 6 and np.allclose(modelo_baseline.class_prior_, y_train.value_counts(
    normalize=True).sort_index().to_numpy())

# Scaler da Fase 5 inalterado
for nome_parametro in ["mean_", "var_", "scale_"]:
    assert np.array_equal(getattr(scaler, nome_parametro), parametros_apos_ajuste[nome_parametro])

# Previsões reproduzíveis e consistentes com as acurácias da Fase 6
assert np.array_equal(modelo_knn_final.predict(X_test), y_pred_knn.to_numpy())
assert np.array_equal(modelo_arvore_final.predict(X_test), y_pred_arvore.to_numpy())
assert accuracy_score(y_test, y_pred_knn) == tabela_treino_teste.loc[("KNN", f"n_neighbors={k_selecionado}"),
                                                                       "Acurácia teste"]

# Base complementar sem ajuste e sem alteração
pd.testing.assert_frame_equal(X_complementar, X_complementar_referencia, check_exact=True)
with open(caminho_datatraining, "rb") as arquivo:
    assert hashlib.sha256(arquivo.read()).hexdigest() == sha256_datatraining_documentado

tabela_auditoria_fase7 = pd.Series({
    "Split, folds, sementes e oito conjuntos de referência preservados": True,
    f"K = {k_selecionado}, max_depth = {profundidade_selecionada_rotulo} e seis atributos preservados (registro_selecao)": True,
    "Modelos principais sem novo ajuste na Fase 7 (assinaturas inalteradas)": True,
    "DummyClassifier ajustado somente com X_train e y_train": True,
    "Nenhum ajuste no datatraining.txt; arquivo bruto inalterado": True,
    "Scaler da Fase 5 inalterado; Pipeline final com 9.933 registros de ajuste": True,
    "Occupancy e date fora dos preditores": True,
    "Erros e métricas da Fase 7 não usados para alterar modelos": True,
    "Previsões reproduzem as acurácias da Fase 6": True,
})
print(tabela_auditoria_fase7.to_string())

# %% [markdown]
# ### BLOCO 101 — Resumo consolidado das sete fases
#
# Tabela construída a partir das variáveis calculadas ao longo do script.

# %%
# BLOCO 101 — Resumo consolidado das sete fases

permanencia_classe = (1 - mudancas_validas / transicoes_validas) * 100
correlacao_light = df_consolidado[["Light", "Occupancy"]].corr().iloc[0, 1]
valores_conferidos_scaler = X_train_scaled.size + X_test_scaled.size
tabela_sete_fases = pd.DataFrame([
    ("Fase 1 — EDA", f"{formatar_inteiro(len(df_consolidado))} registros e 7 colunas; correlação de Light com "
     f"Occupancy de {formatar_decimal(correlacao_light, 4)}; {formatar_percentual(permanencia_classe)} das "
     "transições mantêm a classe (dependência temporal)", "eda_01 a eda_03"),
    ("Fase 2 — Data Prep", f"{formatar_inteiro(len(df_tratado))} registros preservados; precisão decimal padronizada "
     f"({', '.join(f'{coluna} {casas}' for coluna, casas in casas_decimais.items())}); 0 ausentes; imputação apenas "
     "simulada; outliers mantidos por plausibilidade física",
     "dp_01; prep_02; occupancy_tratado.csv; auditoria_precisao_decimal.csv"),
    ("Fase 3 — Feature Engineering", "CO2_Light_Ratio = CO2 / (Light + 1); sem ausentes nem infinitos; índice "
     "experimental, com redundância parcial com Light e CO2", "fe_01; occupancy_features.csv"),
    ("Fase 4 — Train/Test Split", f"80/20 estratificado (random_state=42): {formatar_inteiro(len(X_train))} treino e "
     f"{formatar_inteiro(len(X_test))} teste; {formatar_percentual(proximidade_teste_61)} do teste com vizinho de "
     "treino a até 61 s", "split_01"),
    ("Fase 5 — StandardScaler", f"Ajuste somente no treino; {formatar_inteiro(valores_conferidos_scaler)} valores "
     f"conferidos; CO2 e CO2_Light_Ratio somavam {formatar_percentual(contribuicao_dois_maiores_antes.sum())} da "
     "distância antes da padronização", "scaler_01"),
    ("Fase 6 — Hiperparâmetros e Pipeline", f"K = {k_selecionado} e max_depth = {profundidade_selecionada} pela "
     "validação cruzada; CO2_Light_Ratio sem ganho demonstrado; validação temporal com queda no fim de semana",
     "modelagem_01 a 03; 3 CSVs"),
    ("Fase 7 — Avaliação e veredito", f"Teste: KNN {formatar_percentual(acuracia_teste_knn * 100)}, Árvore "
     f"{formatar_percentual(acuracia_teste_arvore * 100)}, baseline {formatar_percentual(acuracia_teste_baseline * 100)}; "
     f"datatraining.txt: KNN {formatar_percentual(metricas_complementares['KNN']['accuracy'] * 100)}, Árvore "
     f"{formatar_percentual(metricas_complementares['Árvore de Decisão']['accuracy'] * 100)}",
     "avaliacao_01 a 04; 5 CSVs"),
], columns=["Fase", "Principais resultados", "Artefatos"]).set_index("Fase")
print(tabela_sete_fases.to_string())

# %% [markdown]
# ## BLOCO 102 — Interpretações e conclusões finais
#
# Todos os resultados desta seção foram recalculados com a precisão decimal padronizada na Fase 2. A comparação com a
# versão anterior está nos BLOCOS 103 a 108.
#
# ### 102.1 — Atendimento ao objetivo do projeto
#
# Nas condições avaliadas, os dois classificadores identificam a ocupação com alta concordância: no teste oficial, o
# KNN (K = 3) acertou 2.473 de 2.484 registros (99,56%) e a Árvore (`max_depth` = 5), 2.460 (99,03%), com recall da
# classe ocupada de 99,50% e 99,34%. Esses valores, porém, são obtidos em registros sorteados da mesma sequência
# temporal do treino (96,10% com vizinho a até 61 segundos) e não equivalem ao desempenho esperado em outros períodos
# ou ambientes.
#
# ### 102.2 — Comparação com o baseline
#
# O `DummyClassifier` obteve 75,68% de acurácia no teste (75,67% no treino) sem detectar nenhuma ocupação: recall de 0%,
# F1 de 0 e precision indefinida, com os 604 registros ocupados classificados como desocupados. Os dois modelos superam o
# baseline em cerca de 23,3 a 23,9 p.p. de acurácia, mas o ganho relevante está na classe ocupada: o recall sobe de 0%
# para mais de 99% e a balanced accuracy, de 50% para 99,54% (KNN) e 99,14% (Árvore). O valor de 75,68% é calculado no
# teste e não se confunde com a referência descritiva de 75,6705% da base completa (BLOCO 39). O baseline não depende
# dos atributos e é idêntico nas duas versões.
#
# ### 102.3 — Comparação dos classificadores no teste
#
# | Métrica (teste) | KNN (K = 3) | Árvore (`max_depth` = 5) | Diferença (KNN − Árvore) |
# |---|---:|---:|---:|
# | Acurácia | 99,56% | 99,03% | +0,52 p.p. |
# | Balanced accuracy | 99,54% | 99,14% | +0,40 p.p. |
# | Precision (ocupado) | 98,69% | 96,77% | +1,91 p.p. |
# | Recall (ocupado) | 99,50% | 99,34% | +0,17 p.p. |
# | F1 (ocupado) | 99,09% | 98,04% | +1,05 p.p. |
# | Especificidade | 99,57% | 98,94% | +0,64 p.p. |
# | Falsos positivos / falsos negativos | 8 / 3 | 20 / 4 | — |
#
# O KNN erra menos no total (11 contra 24), com menos da metade dos falsos positivos e um falso negativo a menos. Na
# comparação pareada, os dois acertam 2.456 registros e erram 7 em comum; só o KNN acerta 17 registros, e só a Árvore
# acerta 4. O teste de McNemar na variante **exata** (teste binomial bilateral sobre os 21 pares discordantes, 17 contra
# 4, com p = 0,5; `scipy.stats.binomtest`) resultou em p = 0,0072. O valor é **exploratório**:
# o teste supõe registros independentes, o que a dependência temporal viola, e o teste já havia sido consultado na
# versão anterior (p = 0,0784). Ele indica uma vantagem consistente do KNN neste protocolo, e não uma superioridade
# geral.
#
# Os erros do teste têm um padrão claro: **nenhum ocorre com `Light = 0`**, e a maioria está a até 10 minutos de uma
# mudança de ocupação (os 11 erros do KNN e 20 dos 24 da Árvore, contra 7,09% dos registros do teste nessa condição).
# Dos 28 registros com erro de pelo menos um modelo, 14 ocorrem entre 7 h e 9 h e 5 às 17 h, isto é, na chegada e na
# saída. São, portanto, sobretudo erros de transição, em que luminosidade e CO2 ainda refletem a condição anterior, e não
# erros em condições estáveis. Dois falsos positivos da Árvore ocorreram no fim de semana de 14 e 15/02.
#
# ### 102.4 — Generalização temporal
#
# A validação cruzada aleatória e o teste oficial compartilham vizinhos temporais com o treino (87,18% e 96,10% a até 61
# segundos). Na validação temporal da Fase 6, sem essa proximidade, a acurácia média caiu para 94,62% (KNN) e 87,27%
# (Árvore), principalmente por causa do fold 3: a janela de 14 e 15/02 não tem nenhum registro ocupado, e o treino
# daquela divisão tinha apenas 55 registros de fim de semana.
#
# | Fold 3 (1.655 registros, todos desocupados) | Registros | Erros do KNN | Erros da Árvore |
# |---|---:|---:|---:|
# | `Light = 0` | 982 | 0 | 405 |
# | `Light > 0` | 673 | 330 | 491 |
# | Total | 1.655 | 330 (acurácia de 80,06%) | 896 (acurácia de 45,86%) |
#
# Todos esses erros são falsos positivos. A diferença entre os protocolos não pode ser atribuída apenas à ordem
# temporal, porque os tamanhos de treino e as composições das janelas também diferem; ainda assim, ela mostra que o
# desempenho depende de o treino conter condições semelhantes às avaliadas.
#
# ### 102.5 — Avaliação complementar no datatraining.txt
#
# O `datatraining.txt` tem 8.143 registros de 04/02 a 10/02 (1.729 ocupados, 21,23%), incluindo o fim de semana de 07 e
# 08/02, e recebeu a mesma regra decimal da Fase 2 antes da previsão. Não há timestamps nem linhas completas em comum
# com a base principal; apenas 3 vetores de sensores coincidem com registros do treino oficial (os mesmos 3 da precisão
# original: o arredondamento não criou novas coincidências com o treino), e 880 vetores se repetem dentro do próprio
# arquivo em instantes diferentes (867 na precisão original). A distribuição difere da do treino (CO2 médio de
# 606,5 ppm contra 746,5 ppm; umidade média de 25,7% contra 28,9%).
#
# | Métrica (datatraining.txt) | DummyClassifier | KNN (K = 3) | Árvore (`max_depth` = 5) |
# |---|---:|---:|---:|
# | Acurácia | 78,77% | 95,71% | 98,13% |
# | Balanced accuracy | 50,00% | 94,87% | 98,43% |
# | Precision / recall (ocupado) | indefinida / 0% | 87,30% / 93,41% | 92,74% / 98,96% |
# | Falsos positivos / falsos negativos | 0 / 1.729 | 235 / 114 | 134 / 18 |
#
# Nessa base, a ordem dos modelos se inverte: a Árvore erra 152 registros e o KNN, 349. A diferença é maior nos falsos
# negativos (18 contra 114) e aparece em dias específicos: no sábado (07/02), o KNN acertou 88,68% (163 falsos
# positivos) e a Árvore, 97,15% (41); na terça-feira (10/02), o KNN teve 43 falsos negativos, e a Árvore, nenhum. Também
# aqui todos os erros ocorrem com luminosidade positiva. Como o arquivo pertence à mesma coleta e a um período
# intermediário, já havia sido avaliado na versão anterior e não orientou nenhuma escolha, esse resultado é uma
# avaliação complementar fora da base consolidada, e não uma validação externa independente.
#
# ### 102.6 — Feature Engineering
#
# A inclusão da variável derivada não demonstrou ganho preditivo nas configurações e nos protocolos avaliados. Na Fase
# 6, o KNN teve métricas idênticas com 5 e 6 atributos, e a Árvore, diferenças de −0,06 a +0,01 p.p., todas da ordem de
# um registro por fold ou menos; no recorte do fold 1 com K = 3, a variável alterou os vizinhos de 10,92% das validações
# sem mudar nenhuma previsão. Na Árvore final, a importância por redução de impureza de `Light` é de 98,92%, e a de
# `CO2_Light_Ratio`, de 0,11%. Essa importância descreve a estrutura da árvore ajustada: não é causal, não é
# probabilidade de ocupação, não corresponde ao percentual de acertos, depende da estrutura dos atributos e do modelo e
# não necessariamente se mantém em outros ambientes.
#
# ### 102.7 — Aplicação na Indústria 4.0
#
# Um classificador de ocupação alimentado por sensores ambientais poderia apoiar a automação da iluminação e da
# climatização de salas técnicas e escritórios, o monitoramento do uso dos ambientes, a programação de limpeza e
# manutenção e a gestão de recursos energéticos, integrado a um sistema de gestão predial. O projeto, porém, **não mediu
# consumo de energia nem implantou o sistema**: não há estimativa de economia em kWh ou em valores monetários, e os
# cenários de custo do BLOCO 88 usam unidades arbitrárias.
#
# ### 102.8 — Riscos operacionais
#
# Um **falso positivo** é um ambiente desocupado classificado como ocupado; um **falso negativo** é um ambiente ocupado
# classificado como desocupado. Dependendo das regras de automação, falsos positivos podem manter iluminação e
# climatização funcionando sem necessidade, e falsos negativos podem desligá-las com pessoas presentes, prejudicando
# conforto, iluminação, climatização e, em alguns ambientes industriais, a segurança. Essas consequências são
# **potenciais**: não foram medidas neste projeto, que não estima economia de energia, consumo em kWh, valores
# financeiros, retorno sobre investimento nem redução de emissões. Nos
# cenários hipotéticos do BLOCO 88, o KNN tem custo menor no teste em todos os pesos avaliados (11 contra 24 com custos
# iguais; 23 contra 40 com falso negativo de peso 5), mas, no `datatraining.txt`, a Árvore tem custo menor com qualquer
# um desses pesos, pois erra menos nos dois tipos.
#
# Há ainda um **ciclo de retroalimentação**: `Light` é a principal entrada dos modelos (a raiz da árvore é
# `Light ≤ 388,15 lux`; 388,167 lux na precisão original), e a previsão controlaria a própria iluminação. Se o sistema
# desligar a luz, a leitura cai e o modelo tende a continuar prevendo "desocupado", mesmo que alguém entre na sala; se a
# luz permanecer ligada, o modelo tende a continuar prevendo "ocupado". A ação do sistema modificaria o atributo usado
# para inferir a ocupação, e a luz natural (fim de semana com luminosidade positiva e sem ocupação) também interfere no
# sensor. Esse efeito não está presente nos dados históricos, coletados sem controle automático, e exige validação em
# operação antes de uma implantação real.
#
# ### 102.9 — Veredito final
#
# O enunciado (Fase 7, p. 5) pede a acurácia final do melhor KNN e da melhor Árvore **exclusivamente nos dados de teste**
# e uma conclusão técnica sobre qual modelo apresentou desempenho superior e deve ser adotado pelo sistema de automação
# predial. O veredito separa esse resultado acadêmico das considerações operacionais complementares.
#
# **Veredito acadêmico: KNN com K = 3 (Pipeline com `StandardScaler`).**
#
# | Evidência | KNN (K = 3) | Árvore (`max_depth` = 5) |
# |---|---:|---:|
# | Acurácia no teste oficial (critério do enunciado) | **99,56%** | 99,03% |
# | Erros no teste oficial (FP / FN) | **11 (8 / 3)** | 24 (20 / 4) |
# | Acurácia média na validação cruzada (seleção) | **99,17%** | 99,07% |
# | Acurácia média na validação temporal (5 folds) | **94,62%** | 87,27% |
#
# O KNN foi superior no critério exigido e também nos dois protocolos internos ao treino, com menos falsos positivos e
# menos falsos negativos no teste. As margens, porém, são pequenas: 13 registros em 2.484 no teste, e a escolha de K = 3
# em vez de K = 5 decorreu de um único registro na validação cruzada. O McNemar exploratório (p = 0,0072) é coerente com
# a vantagem no teste, mas não é confirmatório, pela dependência temporal e porque o teste já havia sido consultado.
# **O resultado não indica superioridade universal do KNN**: ele vale para este conjunto de dados, este protocolo e
# estas configurações.
#
# **Considerações operacionais complementares.** A Árvore de Decisão com `max_depth` = 5 permanece como **candidata
# alternativa para avaliação operacional supervisionada**, e não como opção superior:
#
# - **A favor da Árvore:** no `datatraining.txt` (avaliação complementar de um período intermediário da mesma fonte, não
#   usado no ajuste, mas já consultado na versão anterior), teve acurácia de 98,13% contra 95,71% e 18 contra 114 falsos
#   negativos; suas regras são explícitas e auditáveis, sem padronização; e suas previsões no teste e no
#   `datatraining.txt` foram as mesmas nas duas versões dos dados. Essa estabilidade em dois conjuntos não prova
#   robustez em geral.
# - **Contra a Árvore:** acurácia menor no teste oficial, na validação cruzada e na validação temporal; no fold 3 (fim de
#   semana sem ocupação e com luminosidade positiva), errou 896 de 1.655 registros, contra 330 do KNN.
# - **Interpretação dos protocolos:** a alta proximidade temporal entre treino e teste (96,10% dos registros de teste
#   com vizinho de treino a até 61 segundos) limita a interpretação do teste como estimativa de generalização e pode
#   beneficiar modelos que exploram padrões locais, como o KNN; essa proximidade, porém, não demonstra isoladamente um
#   efeito causal sobre o desempenho de nenhum dos modelos. O `datatraining.txt`, por sua vez, é um único período, e
#   seus resultados também não se generalizam automaticamente.
#
# **Nenhum dos modelos está suficientemente validado para implantação autônoma.** Os dois exigem validação operacional
# adicional em outros períodos, ambientes e estações, com o sistema atuando sobre a iluminação (retroalimentação de
# `Light`) e com medição real do consumo. Uma implantação piloto deveria funcionar como recomendação ou com confirmação
# humana, monitoramento dos erros e regras de segurança (por exemplo, atrasos antes de desligar), comparando os dois
# candidatos nas mesmas condições.
#
# ### 102.10 — Limitações
#
# - Base de uma única sala, de um único edifício, em fevereiro de 2015.
# - Dependência temporal entre registros: o teste e a validação cruzada são otimistas.
# - Desequilíbrio das classes (24,33% de ocupados na base consolidada).
# - Redundância entre atributos e forte dependência de `Light`.
# - Possível sensibilidade à luminosidade natural, sem variável que a identifique.
# - Apenas um fim de semana na base consolidada (14 e 15/02).
# - Seleção e comparação de atributos feitas com as mesmas validações; teste consultado na Fase 6.
# - Regra decimal definida após a primeira avaliação completa: o teste oficial e o `datatraining.txt` já tinham sido
#   consultados, e a revisão é uma análise de sensibilidade, não uma avaliação independente.
# - Resolução física dos sensores não documentada; a regra decimal não representa a precisão real dos equipamentos.
# - Avaliação complementar da mesma coleta; nenhum teste em outros edifícios.
# - Nenhuma medição de energia e nenhuma implantação operacional.
#
# ### 102.11 — Trabalhos futuros
#
# Sem implementação neste projeto: coletar dados em outros ambientes e estações; validar em períodos futuros, com
# separação temporal mais rigorosa entre treino e teste; investigar modelos sem `Light` ou com sensores adicionais
# (movimento, acionamento de portas, consumo elétrico); testar a robustez a falhas e ruídos de sensores; obter a
# resolução documentada dos sensores para fundamentar a precisão decimal; estimar os custos reais de falsos positivos e
# negativos com os responsáveis pela operação; medir o consumo de energia antes e depois de uma implantação piloto; e
# monitorar mudanças na distribuição dos dados ao longo do tempo.
#
# ### 102.12 — Encerramento do projeto analítico
#
# As sete fases analíticas estão implementadas e validadas no script: EDA, Data Prep (com a padronização da precisão
# decimal), Feature Engineering, divisão treino/teste, padronização, ajuste de hiperparâmetros e avaliação final com
# veredito. A revisão metodológica (BLOCOS 103 a 108) documenta o efeito da regra decimal. Permanecem como atividades de
# entrega, fora desta fase: o versionamento da revisão, a geração do notebook único executado, a publicação na `main` e a
# gravação do vídeo de apresentação.

# %% [markdown]
# # REVISÃO METODOLÓGICA — Sensibilidade à padronização da precisão decimal
# ## BLOCO 103 — Contextualização da revisão
#
# A primeira versão completa das sete fases utilizou os atributos com a precisão original das fontes. Depois da
# avaliação final, o estudante solicitou a padronização das casas decimais (BLOCO 19.1), e todas as fases foram
# reprocessadas a partir dos dados brutos. Esta seção compara as duas versões.
#
# | Protocolo da revisão | Situação |
# |---|---|
# | Dados brutos, CSV consolidado, fórmula da variável derivada, split e folds | Preservados (índices conferidos por SHA-256 nos BLOCOS 43 e 64) |
# | Grade de hiperparâmetros, critério de seleção e desempates | Os mesmos da Fase 6 (BLOCOS 63 e 68), sem nenhuma nova busca |
# | Seleção dos hiperparâmetros | Refeita somente com a validação cruzada no treino revisado; podia mudar |
# | Teste oficial e `datatraining.txt` | Reavaliados com a mesma transformação, apenas com `predict` |
#
# **Limitação central:** a regra decimal foi solicitada **depois** da primeira avaliação. O teste oficial e o
# `datatraining.txt` já tinham sido consultados na versão anterior; por isso, a comparação abaixo é uma **análise de
# sensibilidade posterior à avaliação**, e não uma nova evidência independente. A nova seleção ocorreu somente no
# treino, mas isso não torna os conjuntos já consultados intocados. A regra não foi ajustada para favorecer nenhuma
# métrica e permanece fixa, independentemente do efeito observado.
#
# **Origem dos valores anteriores:** `resultados/referencia_historica/resultados_versao_precisao_integral.csv`, um
# conjunto compacto gerado uma única vez a partir da reexecução isolada do script anterior (SHA-256 `1209aab0…`, que
# reproduziu byte a byte os 25 artefatos derivados daquela versão) e dos CSVs de resultados daquela versão; cada linha
# traz a origem e o SHA-256 do artefato de origem. O script não depende de backups pessoais nem de caminhos absolutos.
# Os valores anteriores são **históricos preservados**; os revisados são **recalculados nesta execução**; as diferenças
# são calculadas aqui (revisado − anterior).
#
# Para separar o efeito do arredondamento do efeito de uma eventual mudança de hiperparâmetros, as configurações fixas
# da versão anterior (K = 5 e `max_depth` = 5) também são reavaliadas nos dados revisados (BLOCO 105). Essa reavaliação
# é descritiva: não é uma busca de hiperparâmetros e não altera a seleção.

# %%
# BLOCO 104 — Referência histórica da versão anterior

referencia_historica_texto = referencia_historica.fillna({"modelo": "", "configuracao": "", "valor_texto": ""})
print("Referência histórica:", caminho_referencia_historica, referencia_historica.shape)
print("\nLinhas por seção:")
print(referencia_historica_texto.groupby("secao").size().to_string())
print("\nOrigens e SHA-256 dos artefatos de origem:")
print(referencia_historica_texto.groupby(["origem", "sha256_origem"]).size().rename("linhas").to_string())


def valor_historico(secao, metrica, modelo="", configuracao=""):
    """Valor numérico preservado da versão anterior (exige exatamente uma linha correspondente)."""
    filtro = ((referencia_historica_texto["secao"] == secao) & (referencia_historica_texto["metrica"] == metrica)
              & (referencia_historica_texto["modelo"] == modelo)
              & (referencia_historica_texto["configuracao"] == configuracao))
    selecionado = referencia_historica_texto.loc[filtro]
    assert len(selecionado) == 1, f"Referência ausente ou duplicada: {secao}/{modelo}/{configuracao}/{metrica}."
    return float(selecionado["valor"].iloc[0]), selecionado["origem"].iloc[0]


k_anterior = int(valor_historico("selecao", "n_neighbors_selecionado", "KNN")[0])
profundidade_anterior = int(valor_historico("selecao", "max_depth_selecionado", "Árvore de Decisão")[0])
print(f"\nHiperparâmetros selecionados — versão anterior: K = {k_anterior}, max_depth = {profundidade_anterior} | "
      f"versão revisada: K = {k_selecionado}, max_depth = {profundidade_selecionada_rotulo}")

# %% [markdown]
# ### BLOCO 105 — Reavaliação descritiva das configurações fixas da versão anterior
#
# As configurações K = 5 (Pipeline com `StandardScaler`) e `max_depth` = 5 são ajustadas no treino oficial revisado e
# avaliadas no teste oficial, no `datatraining.txt` e na validação temporal (mesmos folds do BLOCO 71). Quando a
# configuração revisada selecionada coincide com a anterior, o resultado deve ser idêntico ao das Fases 6 e 7, o que é
# conferido. Nenhum resultado deste bloco participa da seleção.

# %%
# BLOCO 105 — Configurações fixas da versão anterior avaliadas nos dados revisados

configuracoes_fixas = {
    "KNN": (lambda: criar_pipeline_knn(k_anterior), f"n_neighbors={k_anterior}"),
    "Árvore de Decisão": (lambda: criar_arvore(profundidade_anterior), f"max_depth={profundidade_anterior}"),
}
resultados_fixos = {}
for nome, (fabrica, rotulo_configuracao) in configuracoes_fixas.items():
    modelo_fixo = fabrica().fit(X_train, y_train)
    previsto_teste = modelo_fixo.predict(X_test)
    previsto_complementar = modelo_fixo.predict(X_complementar)
    acuracias_temporais, erros_fold3 = {}, {}
    for numero_fold, (posicoes_treino, posicoes_validacao) in enumerate(folds_temporais, start=1):
        estimador = fabrica().fit(X_train_cronologico.iloc[posicoes_treino], y_train_cronologico.iloc[posicoes_treino])
        X_v = X_train_cronologico.iloc[posicoes_validacao]
        y_v = y_train_cronologico.iloc[posicoes_validacao]
        previsto_v = estimador.predict(X_v)
        acuracias_temporais[numero_fold] = accuracy_score(y_v, previsto_v)
        if numero_fold == 3:
            erro_v = previsto_v != y_v.to_numpy()
            erros_fold3 = {"Light = 0": int((erro_v & X_v["Light"].eq(0).to_numpy()).sum()),
                           "Light > 0": int((erro_v & X_v["Light"].gt(0).to_numpy()).sum())}
    resultados_fixos[nome] = {
        "configuracao": rotulo_configuracao,
        "teste": calcular_metricas(y_test, pd.Series(previsto_teste, index=y_test.index)),
        "datatraining": calcular_metricas(y_complementar, pd.Series(previsto_complementar,
                                                                    index=y_complementar.index)),
        "temporal": pd.Series(acuracias_temporais),
        "fold3": erros_fold3,
    }
    # Coerência com a tabela acadêmica do BLOCO 73 (mesma configuração e mesmos dados)
    assert resultados_fixos[nome]["teste"]["accuracy"] == tabela_treino_teste.loc[(nome, rotulo_configuracao),
                                                                                  "Acurácia teste"]

# Quando a configuração fixa coincide com a selecionada, os resultados devem ser idênticos aos das Fases 6 e 7
for nome, selecionado_igual in [("KNN", k_anterior == k_selecionado),
                                ("Árvore de Decisão", profundidade_anterior == profundidade_selecionada)]:
    if selecionado_igual:
        assert resultados_fixos[nome]["teste"]["accuracy"] == metricas_teste[nome]["accuracy"]
        assert resultados_fixos[nome]["datatraining"]["accuracy"] == metricas_complementares[nome]["accuracy"]
        assert np.allclose(resultados_fixos[nome]["temporal"].to_numpy(),
                           resultados_temporais[(nome, "acuracia")].to_numpy(), rtol=0, atol=0)

tabela_configuracoes_fixas = pd.DataFrame({
    f"{nome} ({resultado['configuracao']})": {
        "Teste: acurácia": resultado["teste"]["accuracy"],
        "Teste: FP / FN": f"{int(resultado['teste']['FP'])} / {int(resultado['teste']['FN'])}",
        "datatraining.txt: acurácia": resultado["datatraining"]["accuracy"],
        "datatraining.txt: FP / FN": f"{int(resultado['datatraining']['FP'])} / {int(resultado['datatraining']['FN'])}",
        "Temporal: acurácia média (5 folds)": resultado["temporal"].mean(),
        "Temporal: fold 3": resultado["temporal"][3],
        "Fold 3: erros com Light = 0 / Light > 0": f"{resultado['fold3']['Light = 0']} / {resultado['fold3']['Light > 0']}",
    }
    for nome, resultado in resultados_fixos.items()
})
print("Configurações fixas da versão anterior, avaliadas nos dados revisados (descritivo, sem seleção):")
print(tabela_configuracoes_fixas.to_string())

# %% [markdown]
# ### BLOCO 106 — Comparação integral entre a versão anterior e a revisada
#
# Cada linha compara um valor histórico preservado com o valor recalculado nesta execução. Proporções (acurácias,
# taxas, F1 etc.) também têm a diferença expressa em pontos percentuais. São comparados: seleção, validação cruzada de
# todas as configurações, comparação de 5 e 6 atributos, acurácias de treino e teste por configuração (BLOCO 73),
# validação temporal, fold temporal 3, métricas finais no teste e no `datatraining.txt`, comparação pareada, parâmetros
# do `StandardScaler`, contribuições para as distâncias, importâncias da árvore, configurações fixas e estatísticas da
# variável derivada. O resultado é exportado em `resultados/modelagem/comparacao_precisao_decimal.csv`.

# %%
# BLOCO 106 — Comparação integral entre a versão anterior e a revisada

linhas_comparacao_versoes = []


def registrar_comparacao(secao, protocolo, modelo, configuracao, metrica, anterior, revisado, proporcao, origem,
                         observacao=""):
    diferenca = revisado - anterior if np.isfinite(anterior) and np.isfinite(revisado) else np.nan
    linhas_comparacao_versoes.append({
        "secao": secao, "protocolo": protocolo, "modelo": modelo, "configuracao": configuracao, "metrica": metrica,
        "valor_anterior": anterior, "valor_revisado": revisado, "diferenca": diferenca,
        "diferenca_pp": diferenca * 100 if proporcao else np.nan,
        "origem_anterior": origem, "observacao": observacao,
    })


# Seleção de hiperparâmetros
registrar_comparacao("selecao", rotulo_protocolo_cv, "KNN", "", "n_neighbors_selecionado", k_anterior,
                     k_selecionado, False, valor_historico("selecao", "n_neighbors_selecionado", "KNN")[1])
registrar_comparacao("selecao", rotulo_protocolo_cv, "Árvore de Decisão", "", "max_depth_selecionado",
                     profundidade_anterior, profundidade_selecionada, False,
                     valor_historico("selecao", "max_depth_selecionado", "Árvore de Decisão")[1])

# Validação cruzada de todas as configurações (mesmos folds, mesma grade)
mapa_metricas_cv = {
    "Acurácia treino (média)": "acuracia_treino_media", "Acurácia treino (dp)": "acuracia_treino_dp",
    "Acurácia validação (média)": "acuracia_validacao_media", "Acurácia validação (dp)": "acuracia_validacao_dp",
    "F1 ocupado validação (média)": "f1_ocupado_validacao_media", "F1 ocupado validação (dp)": "f1_ocupado_validacao_dp",
    "Balanced accuracy validação (média)": "balanced_accuracy_validacao_media",
    "Balanced accuracy validação (dp)": "balanced_accuracy_validacao_dp",
}
for nome, resumo, hiperparametro in [("KNN", resumo_knn, "n_neighbors"),
                                     ("Árvore de Decisão", resumo_arvore, "max_depth")]:
    for rotulo, linha in resumo.iterrows():
        for coluna, metrica in mapa_metricas_cv.items():
            anterior, origem = valor_historico("validacao_cruzada", metrica, nome, f"{hiperparametro}={rotulo}")
            registrar_comparacao("validacao_cruzada", rotulo_protocolo_cv, nome, f"{hiperparametro}={rotulo}", metrica,
                                 anterior, linha[coluna], True, origem)

# Comparação de 5 e 6 atributos (acurácia média de validação)
for (nome, rotulo), linha in medias_acuracia.iterrows():
    hiperparametro = "n_neighbors" if nome == "KNN" else "max_depth"
    for coluna, metrica in [("media_5", "acuracia_validacao_media_5_atributos"),
                            ("media_6", "acuracia_validacao_media_6_atributos")]:
        anterior, origem = valor_historico("feature_engineering", metrica, nome, f"{hiperparametro}={rotulo}")
        registrar_comparacao("feature_engineering", rotulo_protocolo_cv, nome, f"{hiperparametro}={rotulo}", metrica,
                             anterior, linha[coluna], True, origem)

# Acurácia de treino completo e de teste por configuração (BLOCO 73)
for (nome, configuracao), linha in tabela_treino_teste.iterrows():
    for coluna, metrica in [("Acurácia treino", "acuracia_treino"), ("Acurácia teste", "acuracia_teste")]:
        anterior, origem = valor_historico("treino_teste_por_configuracao", metrica, nome, configuracao)
        registrar_comparacao("treino_teste_por_configuracao", "treino oficial completo / teste oficial", nome,
                             configuracao, metrica, anterior, linha[coluna], True, origem,
                             "teste já consultado na versão anterior; descritivo")

# Validação temporal das configurações selecionadas em cada versão
for nome, hiperparametro, rotulo_anterior, rotulo_revisado in [
        ("KNN", "n_neighbors", k_anterior, k_selecionado_rotulo),
        ("Árvore de Decisão", "max_depth", profundidade_anterior, profundidade_selecionada_rotulo)]:
    for numero_fold in resultados_temporais.index:
        for metrica in ["acuracia", "f1_ocupado", "balanced_accuracy"]:
            anterior, origem = valor_historico("validacao_temporal", metrica, nome,
                                               f"{hiperparametro}={rotulo_anterior}; fold {numero_fold}")
            registrar_comparacao("validacao_temporal", rotulo_protocolo_temporal, nome, f"fold {numero_fold}", metrica,
                                 anterior, resultados_temporais.loc[numero_fold, (nome, metrica)], True, origem,
                                 f"anterior: {hiperparametro}={rotulo_anterior}; revisado: {hiperparametro}={rotulo_revisado}")
    acuracias_anteriores = [valor_historico("validacao_temporal", "acuracia", nome,
                                            f"{hiperparametro}={rotulo_anterior}; fold {numero_fold}")[0]
                            for numero_fold in resultados_temporais.index]
    registrar_comparacao("validacao_temporal", rotulo_protocolo_temporal, nome, "média dos 5 folds", "acuracia",
                         float(np.mean(acuracias_anteriores)), resultados_temporais[(nome, "acuracia")].mean(), True,
                         origem, f"anterior: {hiperparametro}={rotulo_anterior}; revisado: {hiperparametro}={rotulo_revisado}")

# Fold temporal 3 por grupo de luminosidade
for grupo, linha in tabela_fold3.iterrows():
    for coluna in tabela_fold3.columns:
        anterior, origem = valor_historico("fold3_temporal", coluna, "", grupo)
        registrar_comparacao("fold3_temporal", rotulo_protocolo_temporal, "", grupo, coluna, anterior, linha[coluna],
                             False, origem)

# Métricas finais dos modelos selecionados: teste oficial e datatraining.txt
for secao, metricas_revisadas in [("teste_oficial", metricas_teste), ("datatraining", metricas_complementares)]:
    for nome, metricas_modelo in metricas_revisadas.items():
        for metrica, revisado in metricas_modelo.items():
            anterior, origem = valor_historico(secao, metrica, nome, "selecionado")
            observacao = ("indefinida nas duas versões (TP + FP = 0)" if np.isnan(anterior) and np.isnan(revisado)
                          else "")
            if nome == "KNN":
                observacao = (observacao + "; " if observacao else "") + \
                    f"anterior: K={k_anterior}; revisado: K={k_selecionado}"
            registrar_comparacao(secao, secao, nome, "selecionado", metrica, anterior, revisado,
                                 metrica not in ("TN", "FP", "FN", "TP"), origem, observacao)

# Comparação pareada no teste e McNemar exploratório
for (linha_rotulo, coluna_rotulo), revisado in tabela_pareada.stack().items():
    anterior, origem = valor_historico("comparacao_pareada", "registros", "KNN × Árvore",
                                       f"{linha_rotulo} / {coluna_rotulo}")
    registrar_comparacao("comparacao_pareada", "teste oficial", "KNN × Árvore", f"{linha_rotulo} / {coluna_rotulo}",
                         "registros", anterior, revisado, False, origem)
anterior, origem = valor_historico("comparacao_pareada", "p_valor", "KNN × Árvore", "McNemar exato")
registrar_comparacao("comparacao_pareada", "teste oficial", "KNN × Árvore", "McNemar exato", "p_valor", anterior,
                     resultado_mcnemar.pvalue, False, origem, "exploratório: dependência temporal entre registros")

# StandardScaler da Fase 5 e contribuições para as distâncias (mesmos 1.000 pares)
for posicao, atributo in enumerate(atributos_preditores):
    for nome_parametro in ["mean_", "var_", "scale_"]:
        anterior, origem = valor_historico("standard_scaler", nome_parametro, "StandardScaler", atributo)
        registrar_comparacao("standard_scaler", "fit_transform(X_train)", "StandardScaler", atributo, nome_parametro,
                             anterior, parametros_apos_ajuste[nome_parametro][posicao], False, origem)
for atributo, linha in tabela_contribuicoes.iterrows():
    for coluna, metrica in [("Antes (%)", "contribuicao_media_antes_pct"), ("Depois (%)", "contribuicao_media_depois_pct")]:
        anterior, origem = valor_historico("distancias", metrica, "", atributo)
        registrar_comparacao("distancias", "1.000 pares de X_train (semente 42)", "", atributo, metrica, anterior,
                             linha[coluna], False, origem, "valores em %")

# Árvore final: importâncias e limiar da raiz
for atributo, importancia in zip(atributos_preditores, modelo_arvore_final.feature_importances_):
    anterior, origem = valor_historico("arvore_final", "importancia_impureza", "Árvore de Decisão", atributo)
    registrar_comparacao("arvore_final", "treino oficial completo", "Árvore de Decisão", atributo,
                         "importancia_impureza", anterior, importancia, True, origem)
anterior, origem = valor_historico("arvore_final", "limiar", "Árvore de Decisão", "raiz")
registrar_comparacao("arvore_final", "treino oficial completo", "Árvore de Decisão", "raiz", "limiar", anterior,
                     modelo_arvore_final.tree_.threshold[0], False, origem,
                     f"atributo da raiz revisado: {atributos_preditores[modelo_arvore_final.tree_.feature[0]]}")

# Configurações fixas da versão anterior (K = 5 e max_depth = 5) nos dados revisados
for nome, resultado in resultados_fixos.items():
    for secao_origem, chave in [("teste_oficial", "teste"), ("datatraining", "datatraining")]:
        for metrica in ["accuracy", "balanced_accuracy", "recall_ocupado", "precision_ocupado", "f1_ocupado",
                        "FP", "FN"]:
            anterior, origem = valor_historico(secao_origem, metrica, nome, "selecionado")
            registrar_comparacao("configuracao_fixa", secao_origem, nome, resultado["configuracao"], metrica,
                                 anterior, resultado[chave][metrica], metrica not in ("FP", "FN"), origem,
                                 "mesma configuração nas duas versões; efeito isolado da regra decimal")
    hiperparametro, valor = resultado["configuracao"].split("=")
    for numero_fold, revisado in resultado["temporal"].items():
        anterior, origem = valor_historico("validacao_temporal", "acuracia", nome,
                                           f"{hiperparametro}={valor}; fold {numero_fold}")
        registrar_comparacao("configuracao_fixa", rotulo_protocolo_temporal, nome, resultado["configuracao"],
                             f"acuracia fold {numero_fold}", anterior, revisado, True, origem,
                             "mesma configuração nas duas versões; efeito isolado da regra decimal")
    for grupo, revisado in resultado["fold3"].items():
        anterior, origem = valor_historico("fold3_temporal", f"Erros {nome}", "", grupo)
        registrar_comparacao("configuracao_fixa", rotulo_protocolo_temporal, nome, resultado["configuracao"],
                             f"erros fold 3 ({grupo})", anterior, revisado, False, origem,
                             "mesma configuração nas duas versões; efeito isolado da regra decimal")

# Estatísticas da variável derivada: recalculadas sobre a base consolidada (precisão original) e sobre a tratada
razao_original = df_consolidado["CO2"] / (df_consolidado["Light"] + 1)
for metrica, funcao in [("media", "mean"), ("mediana", "median"), ("variancia", "var"), ("desvio_padrao", "std"),
                        ("minimo", "min"), ("maximo", "max"), ("assimetria", "skew"), ("valores_distintos", "nunique")]:
    registrar_comparacao("feature_engineering_estatisticas", "base completa (12.417 registros)", "", "CO2_Light_Ratio",
                         metrica, float(getattr(razao_original, funcao)()),
                         float(getattr(df_features["CO2_Light_Ratio"], funcao)()), False,
                         "recalculado da base consolidada nesta execução")
for coluna in ["CO2", "Light"]:
    for metodo in ["pearson", "spearman"]:
        registrar_comparacao("feature_engineering_estatisticas", "base completa (12.417 registros)", "",
                             "CO2_Light_Ratio", f"{metodo}_com_{coluna}",
                             razao_original.corr(df_consolidado[coluna], method=metodo),
                             df_features["CO2_Light_Ratio"].corr(df_features[coluna], method=metodo), False,
                             "recalculado da base consolidada nesta execução")

comparacao_precisao = pd.DataFrame(linhas_comparacao_versoes)
assert not comparacao_precisao.duplicated(subset=["secao", "protocolo", "modelo", "configuracao", "metrica"]).any()
comparacao_precisao.to_csv(caminho_comparacao_precisao, index=False)
pd.testing.assert_frame_equal(pd.read_csv(caminho_comparacao_precisao, float_precision="round_trip").fillna(
    {"modelo": "", "configuracao": "", "observacao": ""}), comparacao_precisao.fillna(
    {"modelo": "", "configuracao": "", "observacao": ""}), check_exact=True, check_dtype=False)
print("Comparação exportada:", caminho_comparacao_precisao, comparacao_precisao.shape)
print("\nLinhas por seção:")
print(comparacao_precisao.groupby("secao", sort=False).size().to_string())

# Resumos para leitura
colunas_exibicao = ["valor_anterior", "valor_revisado", "diferenca_pp"]
print("\nValidação cruzada — acurácia média de validação (%):")
print((comparacao_precisao.query("secao == 'validacao_cruzada' and metrica == 'acuracia_validacao_media'")
       .set_index(["modelo", "configuracao"])[["valor_anterior", "valor_revisado"]] * 100)
      .assign(diferenca_pp=lambda tabela: tabela["valor_revisado"] - tabela["valor_anterior"])
      .to_string(float_format=lambda valor: f"{valor:.4f}"))
print("\nModelos selecionados — teste oficial e datatraining.txt:")
print(comparacao_precisao.query("secao in ['teste_oficial', 'datatraining'] and modelo != 'DummyClassifier' "
                                "and metrica in ['accuracy', 'balanced_accuracy', 'recall_ocupado', "
                                "'precision_ocupado', 'FP', 'FN']")
      .set_index(["secao", "modelo", "metrica"])[["valor_anterior", "valor_revisado", "diferenca", "diferenca_pp"]]
      .to_string(float_format=lambda valor: f"{valor:.6g}"))
print("\nValidação temporal — acurácia por fold:")
print(comparacao_precisao.query("secao == 'validacao_temporal' and metrica == 'acuracia'")
      .set_index(["modelo", "configuracao"])[colunas_exibicao].to_string(float_format=lambda valor: f"{valor:.6g}"))
print("\nConfigurações fixas (K = 5 e max_depth = 5) — efeito isolado da regra decimal:")
print(comparacao_precisao.query("secao == 'configuracao_fixa'")
      .set_index(["modelo", "protocolo", "metrica"])[["valor_anterior", "valor_revisado", "diferenca"]]
      .to_string(float_format=lambda valor: f"{valor:.6g}"))
print(f"\nMcNemar exploratório — p anterior: {valor_historico('comparacao_pareada', 'p_valor', 'KNN × Árvore', 'McNemar exato')[0]:.6f}"
      f" | p revisado: {resultado_mcnemar.pvalue:.6f}")

# %% [markdown]
# ### BLOCO 107 — Visualização do impacto da precisão decimal
#
# A figura reúne: (1) a acurácia média de validação cruzada de todas as configurações; (2) a acurácia no teste oficial
# por configuração (descritiva, BLOCO 73); (3) a acurácia por fold temporal dos modelos selecionados em cada versão;
# (4) os falsos positivos e falsos negativos dos modelos selecionados no teste e no `datatraining.txt`; (5) as
# diferenças das configurações fixas (efeito isolado da regra) e dos modelos selecionados em cada protocolo; e (6) a
# diferença de acurácia entre KNN e Árvore em cada protocolo, nas duas versões. Os eixos ampliados informam a faixa.

# %%
# BLOCO 107 — Visualização do impacto da precisão decimal

cor_anterior, cor_revisada = "#4C72B0", "#DD8452"
fig, eixos_revisao = plt.subplots(2, 3, figsize=(22, 12))
(eixo_cv, eixo_teste_config, eixo_temporal), (eixo_erros, eixo_efeito, eixo_diferenca) = eixos_revisao

# Painéis 1 e 2 — Acurácia por configuração (validação cruzada e teste)
configuracoes_ordem = [f"n_neighbors={k}" for k in resumo_knn.index] + [f"max_depth={p}" for p in resumo_arvore.index]
rotulos_configuracoes = [f"K={k}" for k in resumo_knn.index] + [f"prof.={p}" for p in resumo_arvore.index]
posicoes_configuracoes = np.arange(len(configuracoes_ordem))
for eixo, secao, metrica, titulo in [
        (eixo_cv, "validacao_cruzada", "acuracia_validacao_media", "Validação cruzada: acurácia média de validação"),
        (eixo_teste_config, "treino_teste_por_configuracao", "acuracia_teste",
         "Teste oficial por configuração (descritivo)")]:
    dados_painel = comparacao_precisao.query("secao == @secao and metrica == @metrica").set_index("configuracao")
    dados_painel = dados_painel.loc[configuracoes_ordem]
    eixo.plot(posicoes_configuracoes, dados_painel["valor_anterior"] * 100, marker="o", color=cor_anterior,
              linestyle="none", markersize=8, label="Versão anterior (precisão integral)")
    eixo.plot(posicoes_configuracoes, dados_painel["valor_revisado"] * 100, marker="D", color=cor_revisada,
              linestyle="none", markersize=7, label="Versão revisada (regra decimal)")
    eixo.axvline(len(resumo_knn) - 0.5, color="gray", linewidth=0.8)
    eixo.set_xticks(posicoes_configuracoes, rotulos_configuracoes, rotation=45)
    valores_painel = np.concatenate([dados_painel["valor_anterior"], dados_painel["valor_revisado"]]) * 100
    eixo.set_ylim(valores_painel.min() - 0.15, valores_painel.max() + 0.15)
    eixo.set_ylabel(f"Acurácia (%) — eixo ampliado de {valores_painel.min() - 0.15:.2f} a "
                    f"{valores_painel.max() + 0.15:.2f}".replace(".", ","))
    eixo.set_title(titulo)
    eixo.legend(loc="lower left", fontsize=9)
for posicao, rotulo in enumerate(configuracoes_ordem):
    if rotulo in (f"n_neighbors={k_selecionado_rotulo}", f"max_depth={profundidade_selecionada_rotulo}"):
        eixo_cv.annotate("selecionado\n(revisado)", (posicao, comparacao_precisao.query(
            "secao == 'validacao_cruzada' and metrica == 'acuracia_validacao_media' and configuracao == @rotulo")
            ["valor_revisado"].iloc[0] * 100), xytext=(0, -38), textcoords="offset points", ha="center", fontsize=8,
            color=cor_revisada, arrowprops={"arrowstyle": "->", "color": cor_revisada})
    if rotulo in (f"n_neighbors={k_anterior}", f"max_depth={profundidade_anterior}"):
        eixo_cv.annotate("selecionado\n(anterior)", (posicao, comparacao_precisao.query(
            "secao == 'validacao_cruzada' and metrica == 'acuracia_validacao_media' and configuracao == @rotulo")
            ["valor_anterior"].iloc[0] * 100), xytext=(0, 22), textcoords="offset points", ha="center", fontsize=8,
            color=cor_anterior, arrowprops={"arrowstyle": "->", "color": cor_anterior})

# Painel 3 — Validação temporal por fold (modelos selecionados em cada versão)
temporal_comparado = comparacao_precisao.query("secao == 'validacao_temporal' and metrica == 'acuracia' "
                                               "and configuracao != 'média dos 5 folds'")
for nome, marcador in [("KNN", "o"), ("Árvore de Decisão", "s")]:
    dados_modelo = temporal_comparado.query("modelo == @nome")
    folds_numeros = dados_modelo["configuracao"].str.replace("fold ", "").astype(int)
    rotulo_anterior = f"K={k_anterior}" if nome == "KNN" else f"prof.={profundidade_anterior}"
    rotulo_revisado = f"K={k_selecionado}" if nome == "KNN" else f"prof.={profundidade_selecionada_rotulo}"
    eixo_temporal.plot(folds_numeros, dados_modelo["valor_anterior"] * 100, marker=marcador, color=cor_anterior,
                       linestyle="-" if nome == "KNN" else "--", label=f"{nome} anterior ({rotulo_anterior})")
    eixo_temporal.plot(folds_numeros, dados_modelo["valor_revisado"] * 100, marker=marcador, color=cor_revisada,
                       linestyle="-" if nome == "KNN" else "--", label=f"{nome} revisado ({rotulo_revisado})")
eixo_temporal.set_xticks(range(1, 6))
eixo_temporal.set_xlabel("Fold temporal (TimeSeriesSplit, gap = 60)")
eixo_temporal.set_ylabel("Acurácia de validação (%)")
eixo_temporal.set_ylim(40, 103)
eixo_temporal.set_title("Validação temporal dos modelos selecionados")
eixo_temporal.legend(loc="lower left", fontsize=8)

# Painel 4 — Falsos positivos e falsos negativos dos modelos selecionados
categorias_erros = [("teste_oficial", "KNN"), ("teste_oficial", "Árvore de Decisão"), ("datatraining", "KNN"),
                    ("datatraining", "Árvore de Decisão")]
rotulos_erros = ["Teste\nKNN", "Teste\nÁrvore", "datatraining\nKNN", "datatraining\nÁrvore"]
posicoes_erros = np.arange(len(categorias_erros))
for deslocamento, metrica, coluna_valor, cor, hachura, rotulo in [
        (-0.3, "FP", "valor_anterior", cor_anterior, "", "FP anterior"),
        (-0.1, "FP", "valor_revisado", cor_revisada, "", "FP revisado"),
        (0.1, "FN", "valor_anterior", cor_anterior, "//", "FN anterior"),
        (0.3, "FN", "valor_revisado", cor_revisada, "//", "FN revisado")]:
    valores = [comparacao_precisao.query("secao == @secao and modelo == @nome and metrica == @metrica")
               [coluna_valor].iloc[0] for secao, nome in categorias_erros]
    barras = eixo_erros.bar(posicoes_erros + deslocamento, valores, width=0.2, color=cor, edgecolor="black",
                            hatch=hachura, label=rotulo)
    for barra, valor in zip(barras, valores):
        eixo_erros.annotate(f"{int(valor)}", (barra.get_x() + barra.get_width() / 2, valor), ha="center",
                            va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=8)
eixo_erros.set_xticks(posicoes_erros, rotulos_erros)
eixo_erros.set_ylabel("Registros")
eixo_erros.set_title(f"Erros dos modelos selecionados (anterior: K={k_anterior}; revisado: K={k_selecionado})")
eixo_erros.legend(loc="upper left", fontsize=9, ncol=2)
eixo_erros.set_ylim(0, comparacao_precisao.query("secao in ['teste_oficial', 'datatraining'] and modelo != "
                                                 "'DummyClassifier' and metrica in ['FP', 'FN']")
                    [["valor_anterior", "valor_revisado"]].to_numpy().max() * 1.2)

# Painel 5 — Diferenças de acurácia (revisado − anterior, p.p.) por protocolo
protocolos_efeito = ["Validação cruzada", "Teste oficial", "Temporal (média)", "datatraining.txt"]


def diferencas_protocolo(nome, fixa):
    if fixa:
        configuracao = resultados_fixos[nome]["configuracao"]
        cv = comparacao_precisao.query("secao == 'validacao_cruzada' and modelo == @nome and configuracao == "
                                       "@configuracao and metrica == 'acuracia_validacao_media'")["diferenca_pp"].iloc[0]
        teste = comparacao_precisao.query("secao == 'configuracao_fixa' and modelo == @nome and protocolo == "
                                          "'teste_oficial' and metrica == 'accuracy'")["diferenca_pp"].iloc[0]
        complementar = comparacao_precisao.query("secao == 'configuracao_fixa' and modelo == @nome and protocolo == "
                                                 "'datatraining' and metrica == 'accuracy'")["diferenca_pp"].iloc[0]
        mascara_temporal = ((comparacao_precisao["secao"] == "configuracao_fixa")
                            & (comparacao_precisao["modelo"] == nome)
                            & comparacao_precisao["metrica"].str.startswith("acuracia fold"))
        temporal = comparacao_precisao.loc[mascara_temporal, "diferenca_pp"].mean()
    else:
        cv_anterior = comparacao_precisao.query(
            "secao == 'validacao_cruzada' and modelo == @nome and metrica == 'acuracia_validacao_media'")
        rotulo_anterior = f"n_neighbors={k_anterior}" if nome == "KNN" else f"max_depth={profundidade_anterior}"
        rotulo_revisado = (f"n_neighbors={k_selecionado_rotulo}" if nome == "KNN"
                           else f"max_depth={profundidade_selecionada_rotulo}")
        cv = (cv_anterior.set_index("configuracao").loc[rotulo_revisado, "valor_revisado"]
              - cv_anterior.set_index("configuracao").loc[rotulo_anterior, "valor_anterior"]) * 100
        teste = comparacao_precisao.query("secao == 'teste_oficial' and modelo == @nome and metrica == 'accuracy'"
                                          )["diferenca_pp"].iloc[0]
        complementar = comparacao_precisao.query("secao == 'datatraining' and modelo == @nome and metrica == "
                                                 "'accuracy'")["diferenca_pp"].iloc[0]
        temporal = comparacao_precisao.query("secao == 'validacao_temporal' and modelo == @nome and configuracao == "
                                             "'média dos 5 folds'")["diferenca_pp"].iloc[0]
    return [cv, teste, temporal, complementar]


series_efeito = [
    (f"KNN fixo (K={k_anterior})", diferencas_protocolo("KNN", True), "#4C72B0", ""),
    (f"KNN selecionado (K={k_anterior} → K={k_selecionado})", diferencas_protocolo("KNN", False), "#8172B3", "//"),
    (f"Árvore fixa (prof.={profundidade_anterior})", diferencas_protocolo("Árvore de Decisão", True), "#55A868", ""),
]
posicoes_efeito = np.arange(len(protocolos_efeito))
for indice_serie, (rotulo, valores, cor, hachura) in enumerate(series_efeito):
    barras = eixo_efeito.bar(posicoes_efeito + (indice_serie - 1) * 0.26, valores, width=0.26, color=cor,
                             edgecolor="black", hatch=hachura, label=rotulo)
    for barra, valor in zip(barras, valores):
        eixo_efeito.annotate(f"{round(valor, 2) + 0.0:+.2f}".replace(".", ","),
                             (barra.get_x() + barra.get_width() / 2, valor),
                             ha="center", va="bottom" if valor >= 0 else "top",
                             xytext=(0, 2 if valor >= 0 else -2), textcoords="offset points", fontsize=8)
eixo_efeito.axhline(0, color="black", linewidth=0.8)
eixo_efeito.set_xticks(posicoes_efeito, protocolos_efeito)
limite_efeito = max(abs(valor) for _, valores, _, _ in series_efeito for valor in valores) * 1.35 + 0.05
eixo_efeito.set_ylim(-limite_efeito, limite_efeito)
eixo_efeito.set_ylabel("Revisado − anterior (p.p. de acurácia)")
eixo_efeito.set_title("Efeito da revisão: configuração fixa × seleção revisada")
eixo_efeito.legend(loc="upper left", fontsize=8)

# Painel 6 — Diferença de acurácia KNN − Árvore por protocolo, nas duas versões
def diferenca_classificadores(coluna_valor):
    cv_rotulos = ({"valor_anterior": (f"n_neighbors={k_anterior}", f"max_depth={profundidade_anterior}"),
                   "valor_revisado": (f"n_neighbors={k_selecionado_rotulo}",
                                      f"max_depth={profundidade_selecionada_rotulo}")}[coluna_valor])
    cv = comparacao_precisao.query("secao == 'validacao_cruzada' and metrica == 'acuracia_validacao_media'"
                                   ).set_index("configuracao")[coluna_valor]
    resultado = [cv.loc[cv_rotulos[0]] - cv.loc[cv_rotulos[1]]]
    for consulta in ["secao == 'teste_oficial' and metrica == 'accuracy'",
                     "secao == 'validacao_temporal' and metrica == 'acuracia' and configuracao == 'média dos 5 folds'",
                     "secao == 'datatraining' and metrica == 'accuracy'"]:
        tabela = comparacao_precisao.query(consulta).set_index("modelo")[coluna_valor]
        resultado.append(tabela.loc["KNN"] - tabela.loc["Árvore de Decisão"])
    return np.array(resultado) * 100


for deslocamento, coluna_valor, cor, rotulo in [(-0.18, "valor_anterior", cor_anterior, "Versão anterior"),
                                                (0.18, "valor_revisado", cor_revisada, "Versão revisada")]:
    valores = diferenca_classificadores(coluna_valor)
    barras = eixo_diferenca.bar(posicoes_efeito + deslocamento, valores, width=0.36, color=cor, edgecolor="black",
                                label=rotulo)
    for barra, valor in zip(barras, valores):
        eixo_diferenca.annotate(f"{valor:+.2f}".replace(".", ","), (barra.get_x() + barra.get_width() / 2, valor),
                                ha="center", va="bottom" if valor >= 0 else "top",
                                xytext=(0, 2 if valor >= 0 else -2), textcoords="offset points", fontsize=9)
eixo_diferenca.axhline(0, color="black", linewidth=0.8)
eixo_diferenca.set_xticks(posicoes_efeito, protocolos_efeito)
limite_diferenca = np.abs(np.concatenate([diferenca_classificadores("valor_anterior"),
                                          diferenca_classificadores("valor_revisado")])).max() * 1.25
eixo_diferenca.set_ylim(-limite_diferenca, limite_diferenca)
eixo_diferenca.set_ylabel("KNN − Árvore (p.p. de acurácia)")
eixo_diferenca.set_title("Diferença entre os classificadores selecionados")
eixo_diferenca.legend(loc="upper left", fontsize=9)

plt.suptitle("Revisão da precisão decimal: versão anterior (precisão integral) × versão revisada", fontsize=16)
plt.tight_layout()

# Exportação antes da exibição
plt.savefig(caminho_grafico_revisao, dpi=300, bbox_inches="tight")
print("Figura salva em:", caminho_grafico_revisao)

plt.show()

# %% [markdown]
# ## BLOCO 108 — Interpretação da revisão da precisão decimal
#
# ### 108.1 — O que foi preservado
#
# Os dados brutos, o ZIP da UCI e o CSV consolidado (precisão original) permaneceram inalterados. Os 12.417 registros,
# a ordem, os índices, os timestamps e as classes foram preservados; o split (BLOCO 43) e os cinco folds da validação
# cruzada (BLOCO 64) têm assinaturas SHA-256 idênticas às da versão anterior; as janelas temporais, a fórmula da
# variável derivada, a grade de hiperparâmetros e o critério de seleção não mudaram. A proximidade temporal entre treino
# e teste (96,10% a até 61 segundos) também é a mesma, porque depende apenas dos índices e dos timestamps.
#
# ### 108.2 — Efeito nos dados
#
# A regra alterou 56,74% dos valores de `Temperature`, 29,86% de `Humidity`, 6,89% de `Light`, 27,19% de `CO2` e 100% de
# `HumidityRatio`, sempre dentro de meia unidade da última casa adotada. As médias, os desvios padrão, as assimetrias e
# as correlações com `Occupancy` mudaram de forma residual (maior alteração na matriz de Pearson: 9,8 × 10⁻⁴). As
# mudanças visíveis foram a cardinalidade (por exemplo, `HumidityRatio` de 6.220 para 250 valores distintos), 32 fusões
# de vetores de sensores (571 → 603 repetições) e o limite IQR de `Temperature` (461 → 447 potenciais outliers). Na
# variável derivada, a média passou de 426,134753 para 426,134639 e os valores distintos de 6.989 para 6.928. Os
# parâmetros do `StandardScaler` variaram, no máximo, cerca de 0,002 °C na média de `Temperature`, e as contribuições
# para as distâncias, no máximo 0,07 p.p.
#
# ### 108.3 — Efeito isolado da regra (configurações fixas K = 5 e `max_depth` = 5)
#
# | Protocolo | KNN K = 5: anterior → revisado | Árvore `max_depth` = 5: anterior → revisado |
# |---|---:|---:|
# | Validação cruzada (acurácia média) | 99,1644% → 99,1644% | 99,0939% → 99,0738% |
# | Teste oficial (acurácia; FP / FN) | 99,40% (10 / 5) → 99,56% (8 / 3) | 99,03% (20 / 4) → 99,03% (20 / 4) |
# | Validação temporal (média dos 5 folds) | 94,90% → 94,89% | 87,71% → 87,27% |
# | `datatraining.txt` (acurácia; FP / FN) | 95,80% (237 / 105) → 95,94% (230 / 101) | 98,13% (134 / 18) → 98,13% (134 / 18) |
#
# Com os hiperparâmetros fixos, o arredondamento teve efeito pequeno: a Árvore produziu exatamente as mesmas previsões no
# teste e no `datatraining.txt` (só mudou o fold temporal 1, de 95,53% para 93,35%, e −0,02 p.p. na validação cruzada),
# e o KNN com K = 5 errou 4 registros a menos no teste (15 → 11) e 11 a menos no `datatraining.txt` (342 → 331).
# Esses efeitos são da ordem de poucos
# registros e não demonstram que o arredondamento melhore ou piore a qualidade preditiva em geral.
#
# ### 108.4 — Efeito da nova seleção
#
# A seleção do KNN mudou de K = 5 para K = 3 porque a acurácia média de validação de K = 3 subiu de 99,1443% para
# 99,1744%, enquanto K = 5 ficou em 99,1644%: uma diferença de um registro entre 9.933. No teste oficial, K = 3 e K = 5
# revisados empataram (99,56%; 8 falsos positivos e 3 falsos negativos). Fora do teste, a troca foi ligeiramente
# desfavorável: no `datatraining.txt`, K = 3 teve 95,71% (235 / 114) contra 95,94% (230 / 101) de K = 5; na validação
# temporal, 94,62% contra 94,89%. Ou seja, a nova seleção não trouxe ganho fora do teste; ela mostra a instabilidade de
# escolher entre configurações praticamente equivalentes. A Árvore manteve `max_depth` = 5.
#
# ### 108.5 — Efeito no veredito
#
# | Aspecto | Versão anterior | Versão revisada |
# |---|---|---|
# | Hiperparâmetros selecionados | K = 5; `max_depth` = 5 | K = 3; `max_depth` = 5 |
# | Teste oficial: acurácia KNN / Árvore | 99,40% / 99,03% (+9 registros para o KNN) | 99,56% / 99,03% (+13 registros para o KNN) |
# | Teste oficial: FN KNN / Árvore | 5 / 4 | 3 / 4 |
# | McNemar exploratório | p = 0,0784 | p = 0,0072 |
# | `datatraining.txt`: acurácia KNN / Árvore | 95,80% / 98,13% | 95,71% / 98,13% |
# | `datatraining.txt`: FN KNN / Árvore | 105 / 18 | 114 / 18 |
# | Temporal: média KNN / Árvore | 94,90% / 87,71% | 94,62% / 87,27% |
# | Fold 3: acurácia KNN / Árvore | 80,24% / 45,86% | 80,06% / 45,86% |
# | Veredito acadêmico (critério do enunciado) | KNN (K = 5) superior no teste, mas Árvore indicada para adoção | **KNN com K = 3** |
# | Consideração operacional | Árvore `max_depth` = 5, implantação supervisionada | Árvore `max_depth` = 5 como candidata alternativa para avaliação operacional supervisionada; ambos exigem validação adicional |
#
# A revisão **reforçou a vantagem do KNN no teste oficial** e manteve a vantagem da Árvore no `datatraining.txt`. A
# conclusão foi reformulada (BLOCO 102.9): pelo critério do enunciado — acurácia no teste, com o KNN também à frente na
# validação cruzada e na validação temporal — o vencedor acadêmico é o KNN com K = 3. A Árvore permanece apenas como
# candidata alternativa para avaliação operacional supervisionada, pelo resultado no `datatraining.txt`, pelos menos
# falsos negativos nesse período e pela interpretabilidade, sem ser apresentada como operacionalmente superior.
# Nenhum dos modelos está suficientemente validado para operação autônoma.
#
# ### 108.6 — Limitações da revisão
#
# - A regra decimal foi solicitada após a primeira avaliação completa; o teste oficial e o `datatraining.txt` já tinham
#   sido consultados. A comparação é uma análise de sensibilidade posterior, e não uma evidência independente.
# - A regra não foi ajustada às métricas e permaneceu fixa; nenhuma outra quantidade de casas foi testada.
# - A resolução física dos sensores não está documentada; a regra representa uma convenção de apresentação e
#   governança dos dados, e não a precisão real dos equipamentos.
# - O arredondamento é uma transformação com perda de informação; seus efeitos aqui foram pequenos, mas não
#   necessariamente se repetem em outros conjuntos de dados ou modelos.
# - As diferenças observadas entre as versões são da ordem de poucos registros e estão dentro da variabilidade entre
#   folds; não devem ser lidas como relações causais entre a precisão decimal e o desempenho.
