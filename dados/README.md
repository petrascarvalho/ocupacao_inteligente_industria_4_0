# Dados — Occupancy Detection

Esta pasta contém a base de dados original utilizada no projeto **Ocupação Inteligente — Indústria 4.0**.

## Identificação e atribuição

| Campo | Informação |
|---|---|
| Dataset | Occupancy Detection |
| Autor | Luis Candanedo |
| Ano | 2016 |
| Fonte | UCI Machine Learning Repository |
| DOI | [10.24432/C5X01N](https://doi.org/10.24432/C5X01N) |
| Página oficial | https://archive.ics.uci.edu/dataset/357/occupancy-detection |
| Licença | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Link da licença | https://creativecommons.org/licenses/by/4.0/ |

**Citação:** Candanedo, L. (2016). *Occupancy Detection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X01N

Os arquivos são redistribuídos neste repositório nos termos da licença CC BY 4.0, com a atribuição acima. Nenhuma alteração foi feita nos arquivos originais.

## Aquisição

| Campo | Informação |
|---|---|
| URL utilizada para download | https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip |
| Data de aquisição | 25/11/2026, 21:01 (UTC−03:00) |
| Conferência adicional | O pacote é idêntico, byte a byte, ao disponibilizado na página oficial (`https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip`) |

## Arquivos originais (`dados/brutos/`)

| Arquivo | Tamanho (bytes) | Registros | SHA-256 |
|---|---:|---:|---|
| `occupancy_data.zip` | 335.713 | — | `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a` |
| `datatest.txt` | 200.766 | 2.665 | `1b92c7c1b2838963464fa891a610cf3c5db4becb7189189b29b330107a584c7f` |
| `datatest2.txt` | 699.664 | 9.752 | `d026d1bd5aeccd4aff4f3b3710d48e40613bd5fc370db7e61bbdcaa50d985095` |
| `datatraining.txt` | 596.674 | 8.143 | `b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56` |

- Os três arquivos `.txt` foram extraídos de `occupancy_data.zip` sem qualquer modificação.
- A quantidade de registros desconsidera a linha de cabeçalho de cada arquivo. O total (20.560) coincide com o número de instâncias informado pela UCI.
- Colunas dos arquivos: `date`, `Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio`, `Occupancy`.

## Preservação

- Os arquivos de `dados/brutos/` são mantidos exatamente como foram distribuídos pela fonte e não devem ser editados.
- O arquivo `.gitattributes` da raiz do repositório desativa a conversão de quebras de linha nesta pasta (`dados/brutos/** -text`) e trata arquivos `.zip` como binários, de modo que os hashes SHA-256 acima permaneçam válidos em qualquer clone.
- Para conferir a integridade de um arquivo: `sha256sum dados/brutos/<arquivo>` (Linux/Git Bash) ou `Get-FileHash dados\brutos\<arquivo> -Algorithm SHA256` (PowerShell).

## Composição da base utilizada

| Arquivo | Registros | Período (2015) | Ocupados | Uso no projeto |
|---|---:|---|---:|---|
| `datatest.txt` | 2.665 | 02/02 14:19 a 04/02 10:43 | 972 (36,47%) | Consolidado |
| `datatraining.txt` | 8.143 | 04/02 17:51 a 10/02 09:33 | 1.729 (21,23%) | Preservado, fora da consolidação |
| `datatest2.txt` | 9.752 | 11/02 14:48 a 18/02 09:19 | 2.049 (21,01%) | Consolidado |

A consolidação de `datatest.txt` e `datatest2.txt` (12.417 registros) é uma escolha metodológica deste projeto, distinta do protocolo original de avaliação do conjunto de dados. A UCI descreve os três arquivos apenas como conjuntos "for training and testing"; a atribuição de papéis a cada arquivo é inferida pelos seus nomes. O período de `datatraining.txt` fica dentro do intervalo entre os dois arquivos consolidados, que permanece como uma descontinuidade temporal na base.

## Bases derivadas

| Etapa | Arquivo | Dimensões | Finalidade | Gerado por |
|---|---|---|---|---|
| ETL | `consolidados/occupancy.csv` | 12.417 × 7 | Concatenação vertical de `datatest.txt` e `datatest2.txt`, com índice reiniciado | BLOCO 09 |
| Fase 2 — Data Prep | `tratados/occupancy_tratado.csv` | 12.417 × 7 | Base validada: sem duplicatas integrais, ausentes ou infinitos; nenhum registro removido nem valor alterado | BLOCO 27 |
| Fase 3 — Feature Engineering | `features/occupancy_features.csv` | 12.417 × 8 | Base tratada acrescida de `CO2_Light_Ratio = CO2 / (Light + 1)` | BLOCO 35 |

Rastreabilidade: `brutos` → `consolidados` → `tratados` → `features`. Cada etapa parte de uma cópia independente da anterior e é validada com `pd.testing.assert_frame_equal(..., check_exact=True)`. As sete colunas originais e `Occupancy` (9.396 desocupados e 3.021 ocupados) são idênticas nas três bases. Como o tratamento não alterou nenhum valor e a coluna `date` é exportada no mesmo formato (`%Y-%m-%d %H:%M:%S`), o CSV tratado é idêntico, byte a byte, ao consolidado.

## Precisão numérica

- Nenhuma base derivada é arredondada; os arquivos são gravados sem `float_format`, com a representação textual que reconstrói exatamente cada `float64`.
- Para ler `features/occupancy_features.csv` sem perda no último bit, use `pd.read_csv(caminho, float_precision="round_trip")`. Com o conversor padrão do pandas, 850 valores de `CO2_Light_Ratio` diferem do original em até 2,43 × 10⁻¹⁶ (diferença relativa), embora o texto do arquivo esteja exato.
- Após a leitura, converta `date` com `pd.to_datetime(..., format="%Y-%m-%d %H:%M:%S")` para reproduzir os tipos usados no projeto.

## Finais de linha e hashes das bases derivadas

Os arquivos de `dados/brutos/` não sofrem conversão (`-text` no `.gitattributes`). Os CSVs derivados são armazenados no repositório com quebras de linha LF; em um clone no Windows com `core.autocrlf=true`, eles aparecem com CRLF. O conteúdo é o mesmo, mas o tamanho e o SHA-256 dependem do final de linha:

| Arquivo | LF (repositório) | CRLF (cópia de trabalho no Windows) |
|---|---|---|
| `consolidados/occupancy.csv` | 834.283 bytes — `dc3f3b9a13fa4c196607e6f4b0e493a4f0c4da143405c720ad836585cafe9cf1` | 846.701 bytes — `f8bcd8125b4276c38012986cf9991689b61cd3ab6014864fc7c122cfd5e4eb10` |
| `tratados/occupancy_tratado.csv` | 834.283 bytes — `dc3f3b9a13fa4c196607e6f4b0e493a4f0c4da143405c720ad836585cafe9cf1` | 846.701 bytes — `f8bcd8125b4276c38012986cf9991689b61cd3ab6014864fc7c122cfd5e4eb10` |
| `features/occupancy_features.csv` | 978.217 bytes — `05b455f852176b995099d20c843d9ac38a5265199a6528c5ec6d7cc2e4aed994` | 990.635 bytes — `e72b654e9aac84d9e31a0b9cba5217d6ce2454657b060e8be511d98240302f14` |

Os valores lidos pelo pandas são idênticos nas duas formas.
