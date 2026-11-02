# grid_search

Compara 8 configurações do GraphRAG (4 métodos de busca × 2 community-levels)
como classificador multirrótulo de oito doenças hepáticas por prontuário,
contra o ground truth anotado manualmente
(`progresso/ground_truth_segunda_opiniao.csv`, 537 prontuários).

| Sistema | Método | Community Level |
|---|---|---|
| S1 | basic | 1 |
| S2 | basic | 2 |
| S3 | local | 1 |
| S4 | local | 2 |
| S5 | global | 1 |
| S6 | global | 2 |
| S7 | drift | 1 |
| S8 | drift | 2 |

## Motivação

O `community_level` controla o quão agregada é a vizinhança do grafo usada
como contexto — nível mais alto agrupa comunidades maiores, aproximando nós
mais distantes entre si antes de responder. A hipótese testada aqui é se
recuperar contexto de nós mais distantes (community-level mais alto) produz
respostas mais satisfatórias do que uma vizinhança mais restrita, na linha
do que Wu et al. (2024, Medical Graph RAG) propõem com o U-Retrieval —
combinar recuperação top-down (mais ampla, hierárquica) com refinamento
bottom-up (mais preciso, local) para equilibrar abrangência de contexto com
indexação precisa. Cruzar isso com os 4 métodos de busca do GraphRAG separa
o efeito da profundidade/abrangência da vizinhança do efeito do algoritmo de
busca em si.

## Pré-requisitos

- Repositório clonado, dependências instaladas (`uv sync` ou `bash setup.sh`).
- Um índice GraphRAG já construído (`graphrag index`) — por padrão em
  `experimentos/local copy/` (ajustável via env vars `GRAPHRAG_ROOT` /
  `GRAPHRAG_DATA`, já que `experimentos/` é local e nunca versionado).
- `progresso/ground_truth_segunda_opiniao.csv` com a coluna `prontuario_id`
  e uma coluna binária (`0`/`1`) por doença em `parser_doencas.CLASSES`.
- `.env` com `GRAPHRAG_API_KEY` (ver `.env.example`).

## Como rodar

1. **Executar o grid** (`run.py`, célula a célula num Jupyter/VS Code
   interactive window, ou `uv run python experiments/grid_search/run.py`):
   roda as queries de cada sistema contra o índice, salvando progresso
   incremental em `progresso_grid/progresso_S{n}.json` — se interrompido
   (rate limit, timeout, queda de conexão), rodar de novo só refaz o que
   falta. `SISTEMAS_RODAR` e `MODO_TESTE` no topo da célula 5 controlam
   quais sistemas rodar e se é um piloto rápido (3 prontuários).

2. **Analisar os resultados** (`analyze.py`): lê só os `.json`/`.csv` já
   gerados por `run.py` — não roda nenhuma query nova, então pode ser
   reexecutado a qualquer momento (ex.: após um ajuste em
   `parser_doencas.py`) sem custo de API. Gera:
   - `resultados_grid/resultados_consolidados.csv`, `pivot_sistemas.csv`
   - `resultados_grid/metricas_grid.csv` (accuracy, F1 macro/weighted,
     Cohen's Kappa, MCC, IC 95% via bootstrap)
   - `resultados_grid/wilcoxon_posthoc.csv` (Friedman + Wilcoxon com
     correção Bonferroni)
   - `resultados_grid/fig_grid_metricas.png`, `fig_heatmap_acertos.png`
   - `resultados_grid/analise_concordancia.csv` (Inter-Mode Agreement)

Nenhum desses diretórios de saída é versionado (contêm trecho de texto de
prontuário real) — resultados finais para publicação vão para `figures/`
na raiz do repositório.

## Módulos

- `run.py`: execução do grid. Usa `src/core/query/runner.py` (chamada ao
  `graphrag query`) e `src/core/query/progress.py` (progresso resumível).
- `analyze.py`: consolidação, parsing, métricas, testes estatísticos,
  visualizações.
- `parser_doencas.py`: extrai a classe de doença da resposta em texto livre
  do LLM. Compartilhado entre `run.py` e `analyze.py` para garantir que a
  mesma lógica seja aplicada dos dois lados.
- `gera_tabela_metricas.py`, `gera_matrizes_confusao.py`,
  `plota_acertos_erros_drift.py`, `plota_subgrafo_paciente.py`,
  `gera_todos_subgrafos.py`: geram as figuras finais usadas no relatório, a
  partir dos resultados já salvos — não fazem nenhuma *query* nova.
- `adhoc/`: scripts de investigação pontual (não fazem parte do caminho de
  reprodução dos resultados principais) — ver `adhoc/README.md`.

## Nota histórica

Uma implementação anterior e independente desta mesma comparação de métodos
(sem o eixo `community_level`, com cache por resposta e paralelismo) existe
em `archive/python_query/analyse.py`, com resultados já publicados em
`results/rag_comparison/` (10/07/2026). Foi mantida como referência
histórica, mas `grid_search` é a implementação de referência a partir de
agora — é a que teve o bug do parser corrigido e é ativamente mantida.

**Nota de reprodutibilidade:** `run.py`/`analyze.py` ainda fazem extração
*single-label* (pedem 1 doença por prontuário), enquanto
`standalone_grid_multilabel.py` já é multirrótulo de verdade e foi o
script que de fato gerou os resultados publicados no TCC — ou seja, o
caminho "oficial" documentado acima não reproduz hoje os números
publicados. A migração de `run.py`/`analyze.py` para multirrótulo está
em andamento (ver PR dedicado).
