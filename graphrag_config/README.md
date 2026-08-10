# graphrag_config

`settings.yaml`: config atual do GraphRAG (schema `models:`/`litellm:`, via
OpenRouter) usada como base para os índices em `experimentos/*` (pasta
gitignored — cada índice tem sua própria cópia, apontando pra própria
`input`/`output`/`cache`). Requer `GRAPHRAG_API_KEY` no `.env`.

Os prompts referenciados (`prompts/extract_graph.txt`,
`prompts/community_report_graph.txt` etc.) **não ficam aqui** — vêm de
`prompts/index/prompt-tuning/` ou `prompts/index/no-prompt-tuning/`,
dependendo do experimento (ver `experiments/prompt_tuning_comparison/`).
Para inicializar um novo índice:

```bash
mkdir -p meu_indice
cp graphrag_config/settings.yaml meu_indice/settings.yaml
cp prompts/index/prompt-tuning/*.txt meu_indice/prompts/   # ou no-prompt-tuning/
# colocar os .txt de entrada em meu_indice/input/ (ver src/core/dataset/split_dataset.py)
uv run graphrag index --root meu_indice
```
