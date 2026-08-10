# figures/

Figuras de resultado prontas para publicação (relatório, TCC). Só imagens
`.png` e tabelas-resumo `.csv` pequenas — nunca dado bruto de prontuário.

- `fig_grid_metricas.png`, `fig_heatmap_acertos.png`: `experiments/grid_search`
  (comparação dos 8 sistemas method × community-level).
- `fig1_estatisticas_gerais.png` … `fig6_venn.png`,
  `fig_falsos_positivos_puros.png`, `fig_ground_truth.png`,
  `fig_matriz_confusao_tuned.png`, `fig_metricas_por_classe.png`,
  `matriz_confusao_tuned.csv`, `metricas_por_classe_tuned.csv`:
  `experiments/prompt_tuning_comparison` (baseline vs. prompt-tuning).

Para atualizar, rode o experimento correspondente e copie a figura nova pra
cá — os diretórios de saída dos experimentos (`resultados_grid/`,
`resultados_avaliacao/`) não são versionados.
