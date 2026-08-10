# prompt_tuning_comparison

Compara o GraphRAG **com** e **sem** prompt-tuning dos passos de indexação
(extração de entidades, relatórios de comunidade) como classificador da
doença hepática principal de cada prontuário, contra o ground truth anotado
manualmente (`progresso/ground_truth_anotado.csv`).

Diferente de `grid_search` (que varia o *método de busca* sobre o **mesmo**
índice), aqui os dois índices são construídos de formas diferentes — um com
os prompts de indexação padrão do GraphRAG, outro com os prompts ajustados
para hepatologia em `prompts/index/prompt-tuning/` — e a pergunta é se o
tuning da indexação melhora a extração.

## Pré-requisitos

- Dois índices GraphRAG já construídos (`graphrag index`): um baseline
  (prompts padrão / `prompts/index/no-prompt-tuning/`) e um tuned
  (`prompts/index/prompt-tuning/`).
- `progresso/ground_truth_anotado.csv`.
- `.env` com `GRAPHRAG_API_KEY`, se for rodar `run.py` (queries livres, não
  a classificação por prontuário).

## Como rodar

1. **Construir os datasets prontuário × doença** a partir dos parquets de
   cada índice (`build_datasets.py`):

   ```bash
   # índice tuned (anonimizado, entidades tipadas)
   uv run python experiments/prompt_tuning_comparison/build_datasets.py \
     --parquet-dir "experimentos/local copy/output" \
     --csv data/coluna-evolucoes_anonimizadas.csv \
     --matcher fingerprint --type prompt-tuning \
     --output-dir results/dataset

   # índice baseline (CSV bruto, entidades genéricas)
   uv run python experiments/prompt_tuning_comparison/build_datasets.py \
     --parquet-dir experimentos/local/output \
     --csv "experimentos/local/input/data-hepato-sem-anonimizacao.csv" \
     --matcher regex --no-filter-disease-types --type no-prompt-tuning \
     --output-dir results/dataset
   ```

   `--granularity document` gera uma variante agregada por documento em vez
   de por prontuário (não precisa de `--csv`/`--matcher`).

2. **Avaliar contra o ground truth** (`evaluate.py`, ex-`comaprison.py`):
   ajuste `DATASET_TUNED_CSV`/`DATASET_BASE_CSV` no topo do arquivo para os
   `.csv` gerados no passo 1 (o nome inclui timestamp) e rode. Calcula
   accuracy, F1 macro/weighted, Cohen's Kappa, IC 95% via bootstrap, teste de
   McNemar e Wilcoxon, e separa falsos-positivos-puros (ground truth =
   "nenhuma"). Saída em `resultados_avaliacao/`.

3. **(Opcional) Rodar queries livres nos dois índices** (`run.py`, ex-
   `run_experiments.py`): compara respostas a perguntas abertas (não
   classificação) usando métricas de geração de texto (FactScore,
   Faithfulness, Recall@k, RadGraph-F1 — em `metrics.py`). Usa
   `src/core/query/runner.py` e `progress.py` (mesmos módulos do
   `grid_search`) — resumível se interrompido.

   ```bash
   uv run python experiments/prompt_tuning_comparison/run.py \
     --root-base experimentos/local \
     --root-tuned "experimentos/local copy" \
     --queries-dir prompts/query/illness \
     --results-dir results/prompt_tuning_comparison
   ```

## Módulos

- `build_datasets.py`: unifica os três scripts que existiam quase
  duplicados para montar o dataset prontuário/documento × doença a partir
  dos parquets do GraphRAG.
- `evaluate.py`: avaliação estatística baseline vs. tuned como classificador.
- `run.py` + `config.py` + `utils.py` + `metrics.py`: comparação por queries
  livres (métricas de geração de texto, não classificação).
