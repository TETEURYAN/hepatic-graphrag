<div align="center">
<h1>graphrag para prontuários hepatológicos</h1>
</div>

<div align="center" style="max-width: 850px; margin: 0 auto;">
<table>
  <tr>
    <td valign="top" style="padding-right: 16px; width: 150px;">
      <img src="https://user-images.githubusercontent.com/91018438/204195385-acc6fcd4-05a7-4f25-87d1-cb7d5cc5c852.png" alt="animated" width="140" />
    </td>
    <td valign="top" style="padding-left: 16px;">
      <p>Pipeline que avalia o <strong>GraphRAG</strong> como classificador da doença hepática principal de um paciente a partir de prontuários eletrônicos anonimizados. Cada prontuário vira um documento indexado num grafo de conhecimento; a doença é obtida perguntando ao índice e extraindo o diagnóstico da resposta em texto livre do LLM.</p>
      <p>Dois eixos são avaliados contra um ground truth anotado manualmente (141 prontuários): o <strong>método de busca</strong> do GraphRAG (<code>basic</code>/<code>local</code>/<code>global</code>/<code>drift</code>, em dois níveis de comunidade) e o efeito do <strong>prompt-tuning</strong> na etapa de indexação (extração de entidades e relatórios de comunidade).</p>
    </td>
  </tr>
</table>
</div>

Este repositório é o código de suporte de um Trabalho de Conclusão de Curso
(TCC). A pergunta de pesquisa é se — e sob qual configuração — o GraphRAG
consegue recuperar corretamente o diagnóstico principal de um paciente a
partir do texto clínico bruto, sem que isso lhe seja dito diretamente. As
duas baterias de experimento em [`experiments/`](experiments/) existem para
isolar essa pergunta em duas comparações controladas — método de busca de um
lado, prompt-tuning de indexação do outro — em vez de uma única configuração
ad-hoc. Veja [docs/metodologia.md](docs/metodologia.md) para o desenho
experimental completo e [docs/log_operacoes.md](docs/log_operacoes.md) para
o histórico de como o código chegou nessa forma, o que foi corrigido ao
longo do caminho e os resultados obtidos até agora.

## Documentação

- [docs/metodologia.md](docs/metodologia.md): desenho experimental, hipóteses e critérios de avaliação.
- [docs/arquitetura.md](docs/arquitetura.md): como o código está organizado (`src/core` vs. `experiments/`).
- [docs/guia-graphrag.md](docs/guia-graphrag.md): referência de uso do GraphRAG (indexação, query, prompt-tuning).
- [docs/arquivos_parquet.md](docs/arquivos_parquet.md): o que cada `.parquet` gerado pelo índice contém.
- [docs/log_operacoes.md](docs/log_operacoes.md): log cronológico do que foi implementado, corrigido e removido, com os resultados de cada etapa.
- [graphrag_config/README.md](graphrag_config/README.md): como montar um novo índice GraphRAG a partir da config e dos prompts deste repositório.
- [experiments/grid_search/README.md](experiments/grid_search/README.md) e [experiments/prompt_tuning_comparison/README.md](experiments/prompt_tuning_comparison/README.md): passo a passo de reprodução de cada experimento.

## Pré-requisitos

- Python 3.11 (fixado em `.python-version`).
- [uv](https://docs.astral.sh/uv/) — gerencia o ambiente virtual e as dependências.
- Uma API key do [OpenRouter](https://openrouter.ai/keys), necessária para as chamadas ao LLM (`GRAPHRAG_API_KEY`).
- O dataset bruto de prontuários em `data/` — **não incluído no repositório** por conter dado clínico sensível; ver a seção de dados sensíveis do [log de operações](docs/log_operacoes.md).

## Instalação

```bash
bash setup.sh
```

Isso instala o `uv` (se necessário), cria o `.venv` e instala as dependências
do projeto. Depois, configure a API key:

```bash
cp .env.example .env
# edite .env e preencha GRAPHRAG_API_KEY
```

## Como rodar

A reprodução completa tem três etapas — dividir o dataset em documentos,
indexar com o GraphRAG e rodar um dos dois experimentos:

```bash
# 1. divide o CSV bruto (várias linhas por prontuário) em um documento por paciente
uv run python src/core/dataset/split_dataset.py \
  --csv data/coluna-evolucoes_anonimizadas.csv \
  --output-dir input/texts

# 2. indexa (ver graphrag_config/README.md para montar a pasta do índice)
uv run graphrag index --root <seu_indice>

# 3a. compara métodos de busca (ver experiments/grid_search/README.md)
uv run python experiments/grid_search/run.py

# 3b. ou compara com/sem prompt-tuning na indexação (ver
#     experiments/prompt_tuning_comparison/README.md)
uv run python experiments/prompt_tuning_comparison/run.py --help
```

Cada experimento salva progresso incrementalmente (`src/core/query/progress.py`)
— se uma bateria for interrompida por rate limit ou timeout, rodar de novo só
refaz o que falta. Os resultados brutos (`resultados_grid/`,
`resultados_avaliacao/`) não são versionados; as figuras escolhidas para o
relatório final do TCC ficam em [`figures/`](figures/).

## Fora de escopo

Sem interface web, API REST, banco de dados dedicado, autenticação, Docker ou
orquestração (Airflow etc.). É um conjunto de scripts/notebooks de pesquisa,
pensado para ser rodado localmente e reproduzido por quem revisar o TCC.
