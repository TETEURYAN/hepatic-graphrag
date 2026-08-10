# Log de Operações de Código

Registro cronológico das principais operações de código feitas no projeto,
construído a partir do histórico real do `git` (datas de commit) e das datas
de modificação dos artefatos gerados. Para cada fase, indica-se se o código
correspondente segue em uso hoje ou se ficou obsoleto ao longo do projeto.

## 2025-09-22 — RAG-01: Inicialização do projeto
Estrutura inicial do repositório com GraphRAG.

## 2025-10-02 a 2025-10-07 — RAG-02: Busca e remoção de nomes
Heurística de substring para validar nomes próprios a anonimizar.

## 2025-10-13 a 2025-10-16 — RAG-03/RAG-04: Núcleo de anonimização, pré-processamento e config do GraphRAG
- Criação de `src/core/anonymization` (carregamento de CSV, limpeza de entidades
  com spaCy, validação e relatório de anonimização) e `src/core/preprocess`
  (`GraphRAGPreprocessor`, gera `input/texts/*.txt` a partir do dataframe
  anonimizado).
- Criação de `ragtest/` (config mínima do GraphRAG) e `prompts/`.
- **Status atual**: sem commits desde 2025-10-16. `main.py`, único ponto que
  encadeava `anonymization` → `preprocess`, também parado desde então — o
  `input/` que ele geraria contém só um `metadata.csv` de 2025-10-15, sem
  `texts/`. O pipeline ativo hoje (`experimentos_graphrag.py`) lê de um CSV
  bruto fora de `data/`, sem passar por este núcleo.

## 2025-12-01 — RAG-05: Documentação inicial
`docs/arquitetura.md`, `docs/guia-graphrag.md`, `docs/metodologia.md`,
`docs/gitflow.md`.

## 2025-12-06 a 2026-01-22 — RAG-05: Módulo de análise (`src/core/analysis`)
`GraphAnalyzer`, `CommunityReportsAnalyzer` e `PersonAnalyzer` sobre os
`.parquet` gerados pelo GraphRAG.
- **Status atual**: último commit 2026-03-21 (só documentação). Nenhum script
  ativo importa esses módulos hoje; o único chamador nominal
  (`scripts/extract_insight.py`) tem `sys.path` e caminho de dados quebrados
  e não roda.

## 2026-03-21 — RAG-06: Prompts estruturados
Reorganização de `prompts/query/` por doença e saída em JSON estruturado.

## 2026-04-10 a 2026-04-23 — Comparação baseline vs. prompt-tuning
Scripts para cruzar prontuário × doença a partir dos `text_units` do GraphRAG
(`create_dataset_doc_sick.py`, `create_dataset_record_sick[_nopt].py`) e
runner genérico de experimentos comparando índice base vs. tuned
(`run_experiments.py`).
- **Status atual**: em uso — é a base do experimento `prompt_tuning_comparison`.

## 2026-08-10 — Correção do parser de doença e separação execução/análise
Ao rodar o grid de 8 sistemas (`basic`/`local`/`global`/`drift` × 2
community-levels, 141 prontuários), o parser de doença classificava **100%**
das respostas como `nao_identificado`, apesar de `resposta_raw` conter o
diagnóstico correto (ex.: `"Esteatose Hepática. [Data(123123)]"`).

**Causa raiz**: o parser comparava termos em MAIÚSCULO contra a resposta já
normalizada para minúsculo — `in` nunca encontra correspondência nesse caso.

**Correção aplicada**:
- `parser_doencas.py`: módulo único, compartilhado entre execução e análise;
  normaliza acento/caixa dos dois lados e usa correspondência por limite de
  palavra (leftmost match).
- `experimentos_graphrag.py`: passa a só executar as queries (progresso
  incremental, resumível); parsing e análise saem daqui.
- `analise_resultados_grid.py`: script novo, só leitura — reprocessa os
  `progresso_grid/*.json`/`resultados_grid/*.csv` já existentes com o parser
  corrigido e recalcula consolidação, métricas, testes estatísticos
  (Friedman + Wilcoxon post-hoc) e visualizações, sem custo de API.

**Resultado obtido** (reprocessamento dos 8 sistemas do grid, 141 prontuários,
ground truth em `progresso/ground_truth_anotado.csv`, excluindo a classe
`nenhuma` da avaliação):

| Sistema | Modo | Level | Accuracy | F1 macro | F1 weighted | Cohen's Kappa | MCC |
|---|---|---|---|---|---|---|---|
| S4 | local | 2 | 0.587 | **0.363** | 0.593 | 0.482 | 0.488 |
| S3 | local | 1 | 0.578 | 0.338 | 0.589 | 0.471 | 0.479 |
| S6 | global | 2 | 0.569 | 0.328 | 0.558 | 0.457 | 0.464 |
| S7 | drift | 1 | 0.587 | 0.318 | 0.571 | 0.480 | 0.492 |
| S8 | drift | 2 | 0.560 | 0.317 | 0.545 | 0.446 | 0.457 |
| S5 | global | 1 | 0.578 | 0.299 | 0.561 | 0.468 | 0.478 |
| S2 | basic | 2 | 0.541 | 0.298 | 0.551 | 0.428 | 0.434 |
| S1 | basic | 1 | 0.541 | 0.293 | 0.554 | 0.434 | 0.441 |

Melhor sistema por F1 macro: **S4 (local, community-level 2)**, com F1 macro
0.363 e Kappa 0.482. Antes da correção, essas métricas eram degeneradas — a
coluna `doenca_prevista` era `nao_identificado` em praticamente 100% dos
registros, o que tornava qualquer classificador equivalente a chute.

## 2026-08-10 — Refatoração do repositório

O repositório local estava desalinhado do GitHub havia meses: ~50 arquivos
soltos na raiz nunca commitados, misturando pipeline ativo, scaffolding já
consumido e scripts de debug pontual; dados sensíveis (`.env`,
`progresso*/`, `resultados_*/`, um `patients.txt` com iniciais reais de
paciente) sem proteção no `.gitignore`; `prompts/query/insight/output/` com
nome real de paciente no filename dentro de uma pasta versionada.

**Removido/arquivado** (`src/core` antigo — órfão desde RAG-03/RAG-05, ver
seções acima): `anonymization/` → `archive/`; `preprocess/` e `analysis/`
apagados; `main.py` e alvo `run` do `makefile` removidos.

**Novo `src/core`** — infraestrutura de fato compartilhada entre os dois
experimentos do projeto: `patches/` (hotfixes do GraphRAG/litellm),
`query/runner.py` + `progress.py` (unificam duas implementações duplicadas
de chamada ao `graphrag query`, uma em `experimentos_graphrag.py` outra em
`run_experiments.py` — a segunda ganhou suporte a `--community-level` e
progresso resumível, que não tinha), `dataset/split_dataset.py` (novo — o
CSV bruto nunca tinha sido dividido em um documento por prontuário),
`ground_truth/annotate.py` (ex-`note_record.py`).

**`experiments/`** — separa as duas baterias de query que existiam
misturadas na raiz: `grid_search/` (método × community-level, mesmo
índice) e `prompt_tuning_comparison/` (índice com vs. sem prompt-tuning),
cada um com `README.md` de reprodução. `build_datasets.py` unifica três
scripts quase duplicados de montagem de dataset; `evaluate.py` venceu a
reconciliação com um script concorrente (`evaluate_classifier.py`) por ter
bootstrap CI, McNemar e separação de falsos-positivos-puros.

**Descoberta durante a reconciliação**: `python_query/analyse.py`, não
rastreado, já tinha rodado uma comparação de métodos completa em
10/07/2026 (`results/rag_comparison/resumo_comparativo_metodos.csv`),
antes e independente do trabalho desta sessão — mantido em
`archive/python_query/` como referência histórica; `grid_search` é a
implementação de referência daqui pra frente.

**Resultado**: `git status` limpo, nenhum arquivo sensível descoberto,
árvore de trabalho reduzida a código ativo + `archive/` para histórico.
