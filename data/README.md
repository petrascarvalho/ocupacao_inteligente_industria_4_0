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

## Arquivos originais (`data/raw/`)

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

- Os arquivos de `data/raw/` são mantidos exatamente como foram distribuídos pela fonte e não devem ser editados.
- O arquivo `.gitattributes` da raiz do repositório desativa a conversão de quebras de linha nesta pasta (`data/raw/** -text`) e trata arquivos `.zip` como binários, de modo que os hashes SHA-256 acima permaneçam válidos em qualquer clone.
- Para conferir a integridade de um arquivo: `sha256sum data/raw/<arquivo>` (Linux/Git Bash) ou `Get-FileHash data\raw\<arquivo> -Algorithm SHA256` (PowerShell).
