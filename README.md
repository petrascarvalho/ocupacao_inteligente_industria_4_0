# Ocupação Inteligente — Indústria 4.0

Pipeline de Ciência de Dados e Machine Learning para identificar a ocupação de ambientes a partir de sensores ambientais IoT, aplicado à automação predial e à eficiência energética.

> **Status:** em desenvolvimento. As Fases 1 a 5 (EDA, Data Prep, Feature Engineering, divisão treino/teste e escalonamento) estão concluídas e publicadas na branch `develop`. As Fases 6 e 7 (ajuste de hiperparâmetros e avaliação final) ainda não foram implementadas; não existem modelos treinados nem métricas de desempenho. Esta documentação será complementada com os resultados finais.

| Item | Informação |
|---|---|
| Autor | Petras Ruben Carvalho |
| Curso | SCTEC Aperfeiçoar — Integração de IA com Dados (SENAI/SC) |
| Professor | Felipe Rodrigues Souza |
| Entrega | AVA, até 30/11/2026 às 22h |

## 1. Problema e objetivo

Um complexo industrial equipado com sensores ambientais precisa saber se salas técnicas e escritórios estão ocupados, para automatizar o desligamento de iluminação e climatização (HVAC). O projeto treinará e comparará dois classificadores supervisionados — **KNN (K-Nearest Neighbors)** e **Árvore de Decisão** — para prever a variável binária `Occupancy` (0 = desocupado, 1 = ocupado), usando a acurácia como métrica principal.

O trabalho é experimental: não inclui integração com equipamentos reais nem estimativa de economia de energia.

## 2. Base de dados

**Occupancy Detection** — Luis Candanedo (2016), UCI Machine Learning Repository. DOI [10.24432/C5X01N](https://doi.org/10.24432/C5X01N). Licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Página oficial: [archive.ics.uci.edu/dataset/357](https://archive.ics.uci.edu/dataset/357/occupancy-detection).

| Coluna | Descrição | Unidade |
|---|---|---|
| `date` | Data e hora da observação | `AAAA-MM-DD hh:mm:ss` |
| `Temperature` | Temperatura do ambiente | °C |
| `Humidity` | Umidade relativa do ar | % |
| `Light` | Luminosidade | lux |
| `CO2` | Concentração de dióxido de carbono | ppm |
| `HumidityRatio` | Razão de umidade, derivada da temperatura e da umidade relativa | kg de vapor / kg de ar |
| `Occupancy` | Variável alvo | 0 = desocupado, 1 = ocupado |

### Composição da base

| Arquivo | Registros | Período (2015) | Ocupados | Uso |
|---|---:|---|---:|---|
| `datatest.txt` | 2.665 | 02/02 14:19 a 04/02 10:43 | 36,47% | Consolidado |
| `datatraining.txt` | 8.143 | 04/02 17:51 a 10/02 09:33 | 21,23% | Preservado, fora da consolidação |
| `datatest2.txt` | 9.752 | 11/02 14:48 a 18/02 09:19 | 21,01% | Consolidado |
| **Base utilizada** | **12.417** | dois segmentos contínuos | **24,33%** | Todas as fases |

A consolidação dos dois arquivos `datatest` é uma escolha metodológica deste projeto, distinta do protocolo original de avaliação do conjunto de dados (a UCI descreve os arquivos apenas como conjuntos "for training and testing"). O exemplo do enunciado lê somente `datatest.txt`. O arquivo `datatraining.txt` é preservado sem alterações e não participa das análises. Atribuição, hashes SHA-256 e rastreabilidade das bases derivadas: [dados/README.md](dados/README.md).

## 3. Tecnologias

| Tecnologia | Versão testada | Uso |
|---|---|---|
| Python | 3.14.4 | Linguagem |
| pandas | 3.0.2 | Manipulação de dados |
| NumPy | 2.4.4 | Operações numéricas |
| Matplotlib | 3.10.9 | Gráficos |
| Seaborn | 0.13.2 | Gráficos estatísticos |
| scikit-learn | 1.8.0 | Modelagem e avaliação |
| SciPy | 1.18.1 | Dependência do scikit-learn e da correlação de Spearman no pandas |
| ipykernel | 7.2.0 | Execução do notebook |

Ferramentas: VS Code com células `# %%`, Jupyter, Git e GitHub.

As versões acima estão fixadas no [`requirements.txt`](requirements.txt). O projeto foi testado apenas em **Python 3.14.4 no Windows**. Os metadados das dependências selecionadas exigem Python 3.12 ou superior (`scipy==1.18.1`); não há garantia de compatibilidade com versões não testadas.

O `requirements.txt` contém somente as dependências de execução do projeto. As ferramentas usadas para converter o script em notebook (`jupytext` e `nbconvert`) são instaladas à parte, apenas no momento da exportação.

## 4. Estrutura do repositório

```text
ocupacao_inteligente_industria_4_0/
├── dados/
│   ├── brutos/            # pacote original da UCI e os três arquivos .txt (sem alterações)
│   ├── consolidados/      # occupancy.csv — 12.417 × 7
│   ├── tratados/          # occupancy_tratado.csv — 12.417 × 7 (Fase 2)
│   ├── features/          # occupancy_features.csv — 12.417 × 8 (Fase 3)
│   └── README.md          # origem, licença, hashes e rastreabilidade
├── resultados/
│   └── graficos/          # 7 gráficos exportados em 300 DPI (Fases 1 a 5)
├── ocupacao_inteligente.py  # desenvolvimento em células # %% (fonte do notebook)
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

O notebook único de entrega (`ocupacao_inteligente.ipynb`) ainda não está no repositório: será gerado a partir do script e executado ao final das sete fases.

## 5. Como executar

Nesta versão, o código está na branch `develop`; a `main` contém apenas a estrutura inicial e receberá o projeto completo ao final.

```powershell
git clone https://github.com/petrascarvalho/ocupacao_inteligente_industria_4_0.git
cd ocupacao_inteligente_industria_4_0
git switch develop
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No VS Code, selecione o interpretador `.venv` e execute as células do arquivo [`ocupacao_inteligente.py`](ocupacao_inteligente.py) em ordem. Também é possível executar o script inteiro pelo terminal, a partir da raiz do repositório:

```powershell
.\.venv\Scripts\python.exe ocupacao_inteligente.py
```

Na execução pelo terminal, cada gráfico abre em uma janela, e a execução continua depois que a janela é fechada.

Todos os caminhos são relativos à raiz do repositório. Se os arquivos de `dados/brutos/` não existirem, o BLOCO 04 baixa o pacote oficial da UCI e extrai os arquivos necessários. A execução regrava as bases de `dados/` e os gráficos de `resultados/graficos/` com conteúdo idêntico ao publicado.

Estas instruções foram testadas em um ambiente virtual novo criado a partir do `requirements.txt` (`pip check` sem conflitos), com execução integral do script, então até a Fase 3, sem erros ou avisos. A versão atual, até a Fase 5, foi executada integralmente a partir dos dados brutos, em cópia isolada com as mesmas versões de bibliotecas, sem erros ou avisos e com reprodução idêntica (mesmo SHA-256) das 7 bases de dados e dos 7 gráficos. As instruções serão revalidadas após a conclusão das Fases 6 e 7, quando passarão a indicar a `main` e o notebook final.

## 6. Organização em sete fases

| Etapa | Conteúdo | Situação |
|---|---|---|
| Preparação e ETL (Blocos 00–09) | Ambiente, caminhos, download condicional, leitura, inventário dos arquivos originais, consolidação e validação | Concluída |
| Fase 1 — EDA (Blocos 10–17) | Estrutura, tipos, estatísticas, histogramas, classes, correlação, análise e dependência temporal | Concluída |
| Fase 2 — Data Prep (Blocos 18–28) | Duplicidades, ausentes, imputação simulada, tipos, precisão, boxplots, IQR | Concluída |
| Fase 3 — Feature Engineering (Blocos 29–36) | Criação e validação de `CO2_Light_Ratio` | Concluída |
| Fase 4 — Divisão dos dados (Blocos 37–47) | X e y; treino 80% e teste 20% com `stratify=y` e `random_state=42`; integridade e proximidade temporal | Concluída |
| Fase 5 — Escalonamento (Blocos 48–60) | `StandardScaler` ajustado só no treino, apenas para o KNN; validação e análise das distâncias | Concluída |
| Fase 6 — Hiperparâmetros | K ímpares no KNN e `max_depth` na Árvore; acurácia de treino e teste | Pendente |
| Fase 7 — Avaliação final | Acurácia no teste, comparação e escolha do modelo | Pendente |

## 7. Resultados das fases concluídas

### Fase 1 — Análise exploratória

- 12.417 registros, 7 colunas, nenhum valor ausente; `date` em texto, sensores em `float64`, `Occupancy` em `int64`.
- Classes: 9.396 desocupados (75,67%) e 3.021 ocupados (24,33%). A acurácia de um classificador que sempre prevê "desocupado" seria 75,67%.
- Correlação de Pearson com `Occupancy`: `Light` 0,9197; `Temperature` 0,5764; `CO2` 0,3710; `HumidityRatio` 0,2420; `Humidity` −0,0427. Entre preditores: `Humidity` × `HumidityRatio` 0,8919 e `Temperature` × `Light` 0,7248.
- `Light` tem mediana zero (61,30% das leituras são zero); `CO2` é assimétrica à direita.
- Dependência temporal (valores medidos): observações a cada 59–61 segundos, em dois segmentos contínuos (2.665 e 9.752 registros) separados por 7 dias, 4 horas e 5 minutos. A autocorrelação de lag 1 dos sensores varia de 0,9952 a 0,9995 dentro dos segmentos. Entre as 12.415 transições válidas de um registro para o seguinte, há 74 mudanças de classe, ou seja, a classe se mantém em 99,40% delas.
- Estimativa teórica: em uma divisão aleatória de 80/20, cerca de 96% dos registros de teste teriam algum vizinho temporal imediato no treino (1 − 0,2², supondo dois vizinhos independentes). O valor medido após a divisão da Fase 4 foi de 96,10%.

![Histogramas](resultados/graficos/eda_01_histogramas.png)
![Distribuição da variável alvo](resultados/graficos/eda_02_distribuicao_occupancy.png)
![Correlação de Pearson](resultados/graficos/eda_03_correlacao_pearson.png)

### Fase 2 — Limpeza e tratamento

- 0 duplicatas integrais e 0 timestamps repetidos. 571 repetições apenas das medições dos sensores, em instantes diferentes (1.032 registros, 1.028 deles desocupados), sem conflito de classe — mantidas.
- 0 ausentes e 0 infinitos na base real. Imputação demonstrada em cópia isolada com 50 ausentes simulados por variável (semente 42): média para `Humidity` (média 28,92 e mediana 28,60) e mediana para `CO2` (média 745,64 e mediana 632,00, com cauda à direita).
- Boxplots e IQR: 461 potenciais outliers em `Temperature`, 7 em `Light` e 992 em `CO2`. Nenhum viola os critérios de plausibilidade física aplicados; todos foram mantidos.
- Nenhum registro removido, nenhum valor arredondado; `date` convertida para `datetime` apenas em memória.
- Base tratada (`dados/tratados/occupancy_tratado.csv`): 12.417 × 7, com 9.396 desocupados e 3.021 ocupados; recarga idêntica à base em memória (`check_exact=True`).

![Boxplots](resultados/graficos/dp_01_boxplots.png)

### Fase 3 — Feature Engineering

Nova variável sugerida no enunciado: **`CO2_Light_Ratio = CO2 / (Light + 1)`**, calculada registro a registro, sem usar `Occupancy`.

- Verificações antes do cálculo: sem ausentes, infinitos ou luminosidade negativa; menor denominador igual a 1.
- Resultado: `float64`, 0 ausentes, 0 infinitos, 6.989 valores distintos; mediana 503,0, mínimo 0,56 e máximo 2.076,5.
- Com `Light = 0` (7.612 registros), o índice é exatamente igual a `CO2`; com `Light > 0` (4.805 registros), fica entre 0,56 e 263,29.
- Correlação com as variáveis de origem (sem o alvo): `CO2` — Pearson 0,2419 e Spearman 0,0686; `Light` — Pearson −0,6607 e Spearman −0,8633. Há redundância parcial com os sensores originais; o ganho preditivo ainda não foi avaliado.
- Base de Feature Engineering (`dados/features/occupancy_features.csv`): 12.417 × 8. As sete colunas originais e `Occupancy` são idênticas às da base tratada, e a nova coluna coincide exatamente com o recálculo da fórmula.
- Precisão: com o conversor padrão do `read_csv`, 850 valores da nova coluna diferem no último bit; a leitura com `float_precision="round_trip"` reconstrói todos os valores exatamente, e a recarga é validada com `check_exact=True`.

![Distribuição da nova variável](resultados/graficos/fe_01_distribuicao_nova_variavel.png)

### Fase 4 — Divisão treino/teste

- Preditores `X` com seis atributos, nesta ordem: `Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio` e `CO2_Light_Ratio`. Alvo `y = Occupancy`. A coluna `date` fica fora de `X` e é mantida separadamente, com o mesmo índice, apenas para rastreabilidade temporal.
- Divisão executada antes de qualquer transformação de escala, para evitar Data Leakage: `train_test_split` com `test_size=0.2`, `stratify=y` e `random_state=42`.
- Treino: 9.933 registros (7.516 desocupados e 2.417 ocupados). Teste: 2.484 registros (1.880 e 604). A proporção efetiva é de 79,9952% e 20,0048%. A maior diferença entre a proporção de ocupados de um conjunto e a da base foi de 0,0139 ponto percentual.
- Integridade: os índices originais são preservados, há 0 registros em comum entre treino e teste, e a união dos dois reconstrói exatamente `X` e `y` (`check_exact=True`). Repetir a divisão com os mesmos parâmetros produziu conjuntos idênticos.
- Dependência temporal medida: **96,10%** dos registros de teste têm um vizinho de treino a até 61 segundos, o que coincide com ter o vizinho imediato no treino. Com limite de 60 s são 90,10%, e com 120 s, 99,52%; a mediana da menor distância é de 60 s, e o máximo, de 240 s. Para quase todo registro de teste existe no treino uma medição feita cerca de um minuto antes ou depois, o que deve ser considerado ao interpretar a acurácia de teste.
- Nenhum CSV de treino ou teste foi exportado: os conjuntos ficam em memória.

![Divisão treino/teste](resultados/graficos/split_01_divisao_treino_teste.png)

### Fase 5 — Escalonamento com StandardScaler

- Um único `StandardScaler`, ajustado com `fit_transform` somente em `X_train`. `X_test` recebe apenas `transform`, com a média e o desvio padrão aprendidos no treino, sem novo ajuste. A variável alvo não é escalonada.
- Parâmetros aprendidos com as 9.933 observações de treino (`n_samples_seen_`) e os seis atributos na ordem de `X`. Nenhum atributo tem variância zero:

| Atributo | Média (`mean_`) | Desvio padrão (`scale_`) |
|---|---:|---:|
| Temperature | 21,0945 | 1,03692 |
| Humidity | 28,9191 | 4,12600 |
| Light | 137,866 | 219,168 |
| CO2 | 746,499 | 297,485 |
| HumidityRatio | 0,00446835 | 0,000595343 |
| CO2_Light_Ratio | 426,774 | 404,489 |

- Validação matemática: as 18 quantidades aprendidas (médias, variâncias e desvios, com `ddof=0`) coincidem com o cálculo independente feito no treino, e os 74.502 valores transformados (treino e teste) são idênticos aos da fórmula `z = (x − μ) / σ`. No treino padronizado, a maior média absoluta é de 2,5 × 10⁻¹⁵, e o desvio padrão difere de 1 em no máximo 2,2 × 10⁻¹⁶. No teste, as médias e os desvios ficam próximos, mas não iguais, a 0 e 1, como esperado. A transformação inversa reconstrói os valores originais com erro absoluto máximo de 4,5 × 10⁻¹³.
- Prevenção de Data Leakage: os parâmetros coincidem com as estatísticas do treino e diferem das da base completa e das do teste (média de `CO2`: 746,50 no scaler, 745,64 na base e 742,23 no teste). Os conjuntos originais `X_train`, `X_test`, `y_train` e `y_test` permanecem inalterados.
- Influência na distância usada pelo KNN, em análise complementar com 1.000 pares de registros do treino (semente 42, sem usar o alvo nem o teste). A tabela mostra a contribuição média de cada atributo para o quadrado da distância euclidiana:

| Atributo | Antes da padronização | Depois da padronização |
|---|---:|---:|
| Temperature | 0,02% | 16,63% |
| Humidity | 1,00% | 20,76% |
| Light | 15,95% | 13,63% |
| CO2 | 34,31% | 14,27% |
| HumidityRatio | < 0,0001% | 20,08% |
| CO2_Light_Ratio | 48,72% | 14,64% |

Sem escala, `CO2_Light_Ratio` e `CO2` respondiam por 83,03% da distância apenas por causa das unidades. Após a padronização, todos os atributos participam. Essas contribuições não medem importância preditiva, e o efeito sobre a acurácia do KNN só será avaliado na Fase 6.

- Árvore de Decisão sem escalonamento: cada divisão da árvore compara um único atributo com um limiar e depende apenas da ordem dos valores, que a padronização preserva. Por isso, a diferença de escala entre `Light` (0 a mais de 1.000 lux) e `Temperature` (19 a 24 °C) não afeta a árvore, que usa os dados originais.

![Padronização dos atributos](resultados/graficos/scaler_01_padronizacao_atributos.png)

### Revisão metodológica (PR #5)

Uma auditoria geral das Fases 1 a 3, realizada antes da Fase 4, acrescentou ao script: o inventário temporal dos três arquivos originais (BLOCO 06.1), a medição da dependência temporal (BLOCO 16.2), a comparação entre média e mediana na justificativa da imputação, a fundamentação da preservação dos outliers pela plausibilidade física, a caracterização das medições repetidas e a reorganização das células Markdown. Nenhum dado, gráfico ou resultado anterior foi alterado: os 12 arquivos de dados e gráficos mantiveram os mesmos SHA-256.

## 8. Limitações metodológicas

- **Dependência temporal e split aleatório:** registros vizinhos são quase idênticos (autocorrelação de lag 1 acima de 0,99 e 99,40% de permanência da classe). Na divisão aleatória exigida no enunciado, 96,10% dos registros de teste têm o vizinho temporal imediato no treino, o que pode produzir estimativas otimistas de desempenho. A divisão é mantida, e o efeito será complementado por uma avaliação temporal.
- **Composição da base:** os dois arquivos `datatest` foram consolidados por escolha do projeto; a base tem uma descontinuidade de sete dias e proporções de ocupação diferentes entre os segmentos.
- **Análises na base completa:** a EDA, a investigação de outliers e a escolha da variável derivada foram feitas antes da divisão treino/teste. Nenhum parâmetro foi ajustado com esses dados, mas as decisões foram tomadas observando a base inteira.
- **Desbalanceamento:** a estratificação preserva a proporção das classes, mas não as balanceia. Serão apresentadas métricas complementares à acurácia.
- **`Light` em automação predial:** a luminosidade depende das próprias luzes do ambiente; um sistema que desligue a iluminação com base nela pode criar um ciclo de realimentação. Esse ponto será discutido na avaliação final.
- **Variável derivada:** a constante `+1` é arbitrária e o índice não tem significado físico validado.
- **Padronização:** o `StandardScaler` usa média e desvio padrão, que são sensíveis a valores extremos (`Light` chega a 6,58 desvios padrão acima da média do treino). Ele não torna as distribuições normais nem altera as correlações entre os atributos, e atributos correlacionados, como `Humidity` e `HumidityRatio`, continuam somando informação semelhante na distância. O escalonamento equilibra as escalas, mas não garante melhor classificação.

## 9. Próximas etapas

- Fase 6: comparação de pelo menos três valores ímpares de K no KNN e três valores de `max_depth` na Árvore de Decisão, com acurácia de treino e teste. O KNN usará um `Pipeline` com `StandardScaler` e `KNeighborsClassifier` recebendo os dados originais, sem escalonamento duplicado; a Árvore usará os dados sem escala.
- Fase 7: acurácia final no teste, comparação dos modelos e veredito.
- Complementos previstos: validação cruzada dentro do treino, comparação com e sem `CO2_Light_Ratio` (somente no treino), baseline da classe majoritária, matriz de confusão, precision, recall, F1-score, balanced accuracy e uma avaliação temporal complementar.
- Notebook único executado, atualização deste README com os resultados e merge final na `main`.

Melhorias futuras fora do escopo: validação em outros ambientes e períodos, monitoramento do modelo em produção e integração com sistemas prediais.

## 10. Versionamento

Fluxo: `main` ← `develop` ← branches `feature/*`, com pull requests para `develop` e branches preservadas após o merge.

| Branch | Objetivo | Situação |
|---|---|---|
| `feature/00-base-dados` | Aquisição e preservação dos dados originais | PR #1 mesclado |
| `feature/01-eda` | ETL e Fase 1 | PR #2 mesclado |
| `feature/02-data-prep` | Fase 2 | PR #3 mesclado |
| `feature/03-feature-engineering` | Fase 3 | PR #4 mesclado |
| `feature/09-revisao-geral` | Correções da auditoria geral das Fases 1 a 3 | PR #5 mesclado |
| `feature/08-documentacao` | README e requirements | PR #6 mesclado; atualizada após a Fase 5; será atualizada ao final |
| `feature/04-train-test-split` | Fase 4 | PR #7 mesclado |
| `feature/05-standard-scaler` | Fase 5 | PR #8 mesclado |
| `feature/06-hiperparametros` e `feature/07-avaliacao-modelos` | Fases 6 e 7 | Pendentes |

## 11. Referências

- Candanedo, L. (2016). *Occupancy Detection* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5X01N>
- Candanedo, L.; Feldheim, V. (2016). Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. *Energy and Buildings*, 112.
- Documento oficial do Projeto Avaliativo e materiais didáticos MUS01 e MUS02 do curso (uso local, não redistribuídos).
- [scikit-learn — documentação](https://scikit-learn.org/stable/); repositórios didáticos [mapa-hard-skills](https://github.com/Desenvolvimento-IA-Analise-Preditiva-T1/mapa-hard-skills) e [scikit-learn-pipeline](https://github.com/Desenvolvimento-IA-Analise-Preditiva-T1/scikit-learn-pipeline).
