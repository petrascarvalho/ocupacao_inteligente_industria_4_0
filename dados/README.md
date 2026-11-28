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
| `datatraining.txt` | 8.143 | 04/02 17:51 a 10/02 09:33 | 1.729 (21,23%) | Preservado, fora da consolidação; usado apenas na avaliação complementar da Fase 7, com a mesma regra decimal e sem ajuste de modelos |
| `datatest2.txt` | 9.752 | 11/02 14:48 a 18/02 09:19 | 2.049 (21,01%) | Consolidado |

A consolidação de `datatest.txt` e `datatest2.txt` (12.417 registros) é uma escolha metodológica deste projeto, distinta do protocolo original de avaliação do conjunto de dados. A UCI descreve os três arquivos apenas como conjuntos "for training and testing"; a atribuição de papéis a cada arquivo é inferida pelos seus nomes. O período de `datatraining.txt` fica dentro do intervalo entre os dois arquivos consolidados, que permanece como uma descontinuidade temporal na base.

## Bases derivadas

| Etapa | Arquivo | Dimensões | Finalidade | Gerado por |
|---|---|---|---|---|
| ETL | `consolidados/occupancy.csv` | 12.417 × 7 | Concatenação vertical de `datatest.txt` e `datatest2.txt`, com índice reiniciado | BLOCO 09 |
| Fase 2 — Data Prep | `tratados/occupancy_tratado.csv` | 12.417 × 7 | Base validada (sem duplicatas integrais, ausentes ou infinitos; nenhum registro removido) e com a precisão decimal padronizada | BLOCOS 19.1 a 19.3 e 27 |
| Fase 3 — Feature Engineering | `features/occupancy_features.csv` | 12.417 × 8 | Base tratada acrescida de `CO2_Light_Ratio = CO2 / (Light + 1)`, calculada com os atributos tratados | BLOCO 35 |

Rastreabilidade: `brutos` → `consolidados` → `tratados` → `features`. Cada etapa parte de uma cópia independente da anterior e é validada com `pd.testing.assert_frame_equal(..., check_exact=True)`. Os 12.417 registros, a ordem, os índices, a coluna `date` e `Occupancy` (9.396 desocupados e 3.021 ocupados) são idênticos nas três bases. Os cinco sensores da base tratada e da base de features são exatamente os da base consolidada arredondados pela regra abaixo; por isso, **o CSV tratado não é mais idêntico ao consolidado**, que permanece como cópia rastreável da precisão original das fontes.

## Precisão numérica

### Regra decimal da Fase 2

| Atributo | Casas decimais | Maior alteração possível |
|---|---:|---:|
| `Temperature` | 1 | 0,05 °C |
| `Humidity` | 2 | 0,005 % |
| `Light` | 1 | 0,05 lux |
| `CO2` | 1 | 0,05 ppm |
| `HumidityRatio` | 5 | 0,000005 kg/kg |

- A regra é uma decisão metodológica do projeto (não uma exigência do enunciado), aplicada com `df.round(casas_decimais)` somente às cinco colunas. `date` e `Occupancy` não são alterados. Os arquivos brutos e o CSV consolidado mantêm a precisão original.
- O arredondamento tem perda potencial de informação. A resolução física dos sensores não está documentada pela fonte, e a regra não a representa. Os empates seguem o `numpy.round` (meio-par sobre o valor binário armazenado).
- Validação: todas as alterações ficaram dentro dos limites acima; a regra é idempotente; nenhum ausente, infinito ou zero novo foi criado. Valores modificados: `Temperature` 56,74%, `Humidity` 29,86%, `Light` 6,89%, `CO2` 27,19% e `HumidityRatio` 100%. A auditoria completa está em `resultados/modelagem/auditoria_precisao_decimal.csv`.
- Cardinalidade (valores distintos, base consolidada → base tratada): `Temperature` 444 → 50 (−88,74%), `Humidity` 1.815 → 1.134 (−37,52%), `Light` 1.439 → 1.424 (−1,04%), `CO2` 4.262 → 4.114 (−3,47%) e `HumidityRatio` 6.220 → 250 (−95,98%). As repetições de vetores dos cinco sensores passaram de 571 para 603, sem nenhum vetor associado a classes diferentes, e nenhum registro foi excluído.

### Gravação e leitura

- `occupancy_tratado.csv` e `occupancy_features.csv` gravam cada sensor com a quantidade fixa de casas da regra (por exemplo, `23.7`, `26.30`, `585.2`, `749.2` e `0.00476`), a partir de uma cópia textual exclusiva para exportação. O formato de todas as colunas é conferido após a gravação.
- `CO2_Light_Ratio` não é arredondada: é gravada com a representação mais curta que reconstrói exatamente cada `float64`.
- Para ler os CSVs derivados sem perda no último bit, use `pd.read_csv(caminho, float_precision="round_trip")`. Com o conversor padrão do pandas, 801 valores de `CO2_Light_Ratio` diferem do original em até 2,45 × 10⁻¹⁶ (diferença relativa), embora o texto do arquivo esteja exato. O mesmo conversor padrão é a causa de 11.158 valores de `HumidityRatio` da base consolidada diferirem da conversão exata do texto das fontes em até 2,7 × 10⁻¹⁴ (diferença relativa); essa diferença não afeta a regra de cinco casas.
- Após a leitura, converta `date` com `pd.to_datetime(..., format="%Y-%m-%d %H:%M:%S")` para reproduzir os tipos usados no projeto.

## Finais de linha e hashes das bases derivadas

Os arquivos de `dados/brutos/` não sofrem conversão (`-text` no `.gitattributes`). Os CSVs derivados são armazenados no repositório com quebras de linha LF; em um clone no Windows com `core.autocrlf=true`, eles aparecem com CRLF. O conteúdo é o mesmo, mas o tamanho e o SHA-256 dependem do final de linha:

| Arquivo | LF (repositório) | CRLF (cópia de trabalho no Windows) |
|---|---|---|
| `consolidados/occupancy.csv` | 834.283 bytes — `dc3f3b9a13fa4c196607e6f4b0e493a4f0c4da143405c720ad836585cafe9cf1` | 846.701 bytes — `f8bcd8125b4276c38012986cf9991689b61cd3ab6014864fc7c122cfd5e4eb10` |
| `tratados/occupancy_tratado.csv` | 643.833 bytes — `652b798caab41321240517c1ec362c666a4db5646e68b371d129ceedc3b5a4db` | 656.251 bytes — `6b9200ba3aac68b308147bb8f1eb6d2eda28002cab0cd286e082120af125d167` |
| `features/occupancy_features.csv` | 775.288 bytes — `20c7cdf9fa1d984cebce9f9d054a62cb22f5435ced6d82dce6d64202bfda801d` | 787.706 bytes — `17737744d759981c2cbaa020e2617ea1b65ddfcb3fc90c085751c0b44f06e718` |

Os valores lidos pelo pandas são idênticos nas duas formas.
