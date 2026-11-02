# adhoc/

Scripts de investigação pontual, não fazem parte do caminho de reprodução
dos resultados principais (esse caminho é `run.py` ou
`standalone_grid_multilabel.py` + `analyze.py`, na raiz de
`experiments/grid_search/`). Cada um aqui responde a uma pergunta específica
levantada durante a análise dos resultados do grid, com prontuários e
hipóteses fixados no próprio código — não foram pensados para reuso direto,
mas ficam versionados para rastreabilidade dos achados reportados no
trabalho (em especial a investigação do modo `drift`, Seção 4.9.1/5.4 da
monografia).

| Script | Pergunta que responde |
|---|---|
| `testa_hipoteses_drift.py` | Por que o `drift` acerta onde `basic`/`local` erram? (H1–H4, teste estatístico) |
| `testa_hipoteses_drift_erro.py` | Por que o `drift` erra onde `basic`/`local` acertam? (H1–H4, teste estatístico) |
| `testa_fix_hda_drift.py` | Um *prompt* reforçado corrige os falsos positivos de HDA do `drift`? |
| `testa_fix_alucinacao_hepb_hepc.py` | Idem, para os falsos positivos de hepatite B/C |
| `teste_max_level_zero.py` | `dynamic_search_max_level: 0` resolve os timeouts do modo `global`? |
| `teste_custo_global_drift_tuned.py` | Qual o custo real da config de redução de custo vs. baseline, em `global`/`drift`? |
| `teste_custo_multilabel.py` | Qual o custo real de 1 consulta multilabel por sistema? |
| `validar_prompt_multilabel.py` | O *prompt* multilabel funciona num subconjunto pequeno, antes do grid inteiro? |
| `run_s1_s6_ling30.py` | Tentativa (incompleta) de rodar S1–S6 com `ling-3.0-flash` em vez do modelo final |
| `inspect_confusions.py` | Inspeção manual dos prontuários mais confundidos pelo grid |
| `export_confusion_texts.py` | Exporta o texto desses prontuários confundidos pra leitura fora do terminal |
| `export_prontuarios_erros_drift.py` | Exporta texto + contexto dos erros específicos de S7/S8 |
| `exporta_casos_exemplo.py` | Exporta exemplos qualitativos (falso positivo binário / acerto de inferência) com a resposta do LLM anexada |

Todos assumem que `GRAPHRAG_ROOT`/`GRAPHRAG_DATA` (ver `src/core/paths.py`)
já apontam para um índice construído — nenhum constrói o índice sozinho.
