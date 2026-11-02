# Documentação de Arquitetura

## Arquitetura do sistema

Este documento descreve a arquitetura **atual** do código de suporte à
avaliação do GraphRAG como classificador de doença hepática a partir de
prontuários eletrônicos.

> **Nota histórica:** uma versão anterior deste documento descrevia um
> pipeline de anonimização e pré-processamento (`src/core/anonymization`,
> `src/core/preprocess`, um `main.py` de linha de comando). Esse código foi
> movido para [`archive/anonymization/`](../archive/anonymization/) e não é
> mais usado pelo pipeline atual — fica preservado só por referência
> histórica. O conteúdo abaixo descreve o que está em uso hoje.

## Índice

1. [Visão Geral](#visão-geral)
2. [`src/core/` — biblioteca compartilhada](#srccore--biblioteca-compartilhada)
3. [`experiments/` — baterias de avaliação](#experiments--baterias-de-avaliação)
4. [Fluxo de dados ponta a ponta](#fluxo-de-dados-ponta-a-ponta)
5. [Dependências e Tecnologias](#dependências-e-tecnologias)

---

## Visão Geral

O sistema é composto por dois núcleos:

1. **`src/core/`**: biblioteca compartilhada — caminhos configuráveis,
   normalização/carregamento de texto clínico, construção do *ground
   truth*, execução de *queries* no GraphRAG e hotfixes (`patches/`) para o
   pacote `graphrag` de terceiros. É importada por todos os experimentos,
   nunca o contrário.
2. **`experiments/`**: baterias de avaliação independentes entre si, cada
   uma com seu próprio `README.md`. Usam `src/core/` para evitar duplicar
   lógica de normalização de dados e execução de *query*, mas não dependem
   umas das outras.

Não há um único `main.py` ou interface de linha de comando unificada — cada
experimento é um ponto de entrada próprio, documentado no seu `README.md`.

---

## `src/core/` — biblioteca compartilhada

```
src/core/
├── paths.py                      # Caminhos padrão, sobrescrevíveis por .env
├── dataset/
│   ├── clinical_text.py          # Normalização de ID + agregação de texto por prontuário
│   └── split_dataset.py          # CSV bruto (1 linha/atendimento) -> 1 .txt por prontuário (input/ do GraphRAG)
├── ground_truth/
│   └── annotate.py               # Anotação interativa (segunda opinião humana)
├── query/
│   ├── runner.py                 # `graphrag query` via subprocess (timeout, kill de órfão, --config, --dynamic-community-selection)
│   └── progress.py               # Cache de progresso resumível (bateria longa de queries)
└── patches/
    ├── litelm_patch.py                   # Silencia warning de cleanup assíncrono do litellm
    ├── graphrag_json_patch.py            # Remove response_format=json_object forçado (map/primer), sem tocar no resto
    ├── reasoning_effort_patch.py         # Desliga de fato o raciocínio do modelo (settings.yaml não tem efeito via OpenRouter)
    ├── batch_report_embeddings_patch.py  # 1 consulta em lote ao LanceDB em vez de N sequenciais (drift)
    ├── run_graphrag.py                   # Entry point que aplica os patches acima antes de `graphrag query`
    └── run_graphrag_debug.py             # Mesma coisa + log verboso do litellm + dump periódico de pilha (diagnóstico)
```

**`paths.py`** centraliza todo caminho de índice/dataset/saída usado pelos
experimentos — nenhum script deve hardcodar um caminho absoluto. Os
defaults assumem a convenção de pastas deste repositório (ver
`experimentos/` nas instruções de configuração), mas qualquer um pode ser
sobrescrito via `.env`.

**`dataset/clinical_text.py`** resolve uma inconsistência da fonte de dados:
o CSV bruto tem dois formatos de ID para o mesmo prontuário (`"10001436"` e
`"PACIENTE - 10001436"`). `carrega_texto_por_prontuario()` normaliza o ID e
concatena cronologicamente todos os atendimentos de um mesmo paciente num
único texto — usado por todo experimento que precisa do texto completo de
um prontuário.

**`query/runner.py`** é o único ponto que chama `graphrag query` via
subprocess em todo o repositório. Resolve dois problemas reais encontrados
durante o trabalho: `uv run` não substitui o processo (faz fork), então
matar só o processo pai no timeout deixa o neto (o `graphrag`/`python` que
de fato está esperando a API) órfão rodando para sempre — por isso roda tudo
num grupo de processo novo e mata o grupo inteiro no timeout.

**`patches/`** existe porque o pacote `graphrag` (dependência externa,
`graphrag>=2.7.0`) tem bugs e incompatibilidades reais com o roteamento via
OpenRouter e com modelos de raciocínio híbrido, documentados no docstring de
cada patch. `run_graphrag.py` é o *entry point* que os aplica antes de
delegar para `graphrag.__main__`; qualquer script que precise desses
hotfixes chama `run_graphrag_query(..., use_patched_runner=True)` em vez de
`graphrag query` diretamente.

---

## `experiments/` — baterias de avaliação

```
experiments/
├── grid_search/                    # Comparação dos 8 sistemas (4 modos x 2 community-levels)
│   ├── run.py                      # Roda a bateria completa (depende do resto do repo)
│   ├── standalone_grid_multilabel.py  # Mesma bateria, script único e autocontido (PEP 723, roda sem clonar o repo)
│   ├── analyze.py                  # Lê os resultados salvos e calcula métricas
│   ├── parser_doencas.py           # Parser determinístico resposta-livre -> classes (multilabel)
│   ├── gera_tabela_metricas.py     # Figura final: tabela de métricas por sistema
│   ├── gera_matrizes_confusao.py   # Figura final: matrizes de confusão por doença
│   ├── plota_subgrafo_paciente.py  # Subgrafo de entidades/relações de 1 paciente
│   ├── gera_todos_subgrafos.py     # Idem, em lote pra todos os prontuários + análise agregada
│   ├── plota_acertos_erros_drift.py  # Figura final: dispersão acerto/erro do drift por tamanho de texto
│   └── adhoc/                      # Scripts de investigação pontual -- ver adhoc/README.md
├── ground_truth_expansion/         # Funil de triagem textual + triagem por LLM (amplia a amostra anotável)
├── ground_truth_second_opinion/    # Segunda opinião via LLM (anotador cego, comparação com a anotação manual)
└── prompt_tuning_comparison/       # Compara baseline vs. prompt-tuning na extração de entidades
```

Cada subpasta tem seu próprio `README.md` com o passo a passo de reprodução.
A distinção entre a raiz de `grid_search/` e `grid_search/adhoc/` é: a raiz
é o caminho que reproduz os resultados reportados (a bateria dos 8 sistemas
e as figuras finais); `adhoc/` são scripts que responderam perguntas
pontuais levantadas durante a análise (por que o `drift` erra/acerta em tal
caso, se tal ajuste de parâmetro resolve tal problema de custo/timeout) —
não fazem parte do caminho de reprodução principal, mas ficam versionados
para rastreabilidade.

---

## Fluxo de dados ponta a ponta

```
data/coluna-evolucoes_anonimizadas.csv  (1 linha por atendimento)
        │
        ▼  src/core/dataset/split_dataset.py
input/*.txt  (1 documento por prontuário)
        │
        ▼  graphrag index  (fora deste repositório; ver graphrag_config/)
experimentos/local copy/output/*.parquet  (grafo, comunidades, resumos)
        │
        ▼  experiments/grid_search/run.py  (ou standalone_grid_multilabel.py)
        │     para cada prontuário x sistema: monta o prompt, chama
        │     src/core/query/runner.py (graphrag query), salva a resposta bruta
        │
        ▼  experiments/grid_search/parser_doencas.py
        │     resposta em texto livre -> conjunto de classes positivas
        │
        ▼  experiments/grid_search/analyze.py
métricas por sistema/classe, matrizes de confusão, tabelas e gráficos finais
```

---

## Dependências e Tecnologias

Gerenciadas por `uv` a partir de `pyproject.toml`/`uv.lock` (não usar
`pip install` nem `scripts/requirements.txt`, que está desatualizado e
cobre só o utilitário de visualização em `scripts/`).

| Biblioteca | Uso |
|---|---|
| `graphrag` | Implementação oficial Microsoft do GraphRAG (dependência avaliada, não escrita por este projeto) |
| `pandas`, `numpy` | Manipulação de dados tabulares |
| `scipy` | Testes estatísticos (McNemar, Mann-Whitney, qui-quadrado) |
| `matplotlib`, `networkx` | Figuras e visualização de grafo |
| `python-dotenv` | Carregamento de `.env` |

---

**Última atualização:** 2026-11-02
