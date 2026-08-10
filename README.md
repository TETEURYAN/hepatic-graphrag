# Análise Longitudinal de Prontuários Eletrônicos com GraphRAG

<p align="center">
  <img src="https://user-images.githubusercontent.com/91018438/204195385-acc6fcd4-05a7-4f25-87d1-cb7d5cc5c852.png" alt="animated" />
</p>

Projeto de TCC que avalia o **GraphRAG** como classificador da doença hepática
principal de um paciente a partir de prontuários eletrônicos anonimizados,
comparando métodos de busca (`basic`/`local`/`global`/`drift`) e o efeito de
prompt-tuning na indexação.

---

## Estrutura do projeto

```
.
├── setup.sh                  # instala uv + ambiente virtual
├── graphrag_config/           # settings.yaml de referência do GraphRAG
├── prompts/                   # prompts de indexação e de query (por doença,
│                               #   meta-prompts, extração de conhecimento)
├── src/core/                  # biblioteca compartilhada entre os experimentos
│   ├── patches/                #   hotfixes necessários antes de `graphrag query/index`
│   ├── query/                  #   execução de query (runner.py) e progresso resumível
│   ├── dataset/                #   split_dataset.py — divide o CSV bruto em documentos
│   └── ground_truth/           #   anotação interativa do ground truth
├── experiments/
│   ├── grid_search/            # 8 sistemas (método × community-level) vs. ground truth
│   └── prompt_tuning_comparison/  # índice com vs. sem prompt-tuning na indexação
├── figures/                   # PNGs/CSVs de resultado prontos para publicação
├── docs/                      # arquitetura, metodologia, guia de uso do GraphRAG
└── archive/                   # código/notebooks/docs supersedidos, mantidos como
                                #   referência histórica (não fazem parte do pipeline ativo)
```

**Nunca versionados** (`.gitignore`): `data/` (dataset bruto), `experimentos/`
(índices GraphRAG locais), `input/`, `results/`, `progresso*/`,
`resultados_*/` — tudo derivado do dataset ou contendo texto de prontuário
real. Ver `docs/log_operacoes.md` para o histórico de por que cada coisa está
onde está.

## Pré-requisitos

- [uv](https://github.com/astral-sh/uv)
- Uma chave de API compatível com OpenRouter (`GRAPHRAG_API_KEY`) — copie
  `.env.example` para `.env` e preencha.
- O dataset bruto em `data/` (não incluído no repositório — dado sensível).

## Setup

```bash
bash setup.sh          # instala uv + cria .venv + instala dependências
cp .env.example .env   # preencha GRAPHRAG_API_KEY
```

## Fluxo de reprodução

1. **Dividir o dataset por documento** (o CSV bruto tem várias linhas por
   prontuário, não um documento por arquivo):
   ```bash
   uv run python src/core/dataset/split_dataset.py \
     --csv data/coluna-evolucoes_anonimizadas.csv \
     --output-dir input/texts
   ```
2. **Indexar** com o GraphRAG (ver `graphrag_config/README.md` para montar o
   `settings.yaml` + prompts de um novo índice):
   ```bash
   uv run graphrag index --root <seu_indice>
   ```
3. **Rodar um experimento** — cada um em `experiments/<nome>/` tem seu
   próprio `README.md` com o passo a passo específico:
   - [`experiments/grid_search/`](experiments/grid_search/README.md):
     compara os 4 métodos de busca × 2 community-levels no mesmo índice.
   - [`experiments/prompt_tuning_comparison/`](experiments/prompt_tuning_comparison/README.md):
     compara indexação com vs. sem prompt-tuning.

Resultados de cada experimento (métricas, testes estatísticos, figuras) vão
para diretórios não versionados (`resultados_grid/`, `resultados_avaliacao/`)
— as figuras finais escolhidas para publicação são copiadas manualmente para
`figures/`.

## Documentação

- `docs/arquitetura.md`, `docs/metodologia.md`, `docs/guia-graphrag.md`: guias gerais.
- `docs/log_operacoes.md`: log cronológico do que foi feito no código, o que
  segue em uso e o que ficou obsoleto.
- `archive/`: código, notebooks e docs de fases anteriores do projeto,
  preservados por valor histórico/metodológico mas fora do pipeline ativo.

## Licença

Distribuído sob a licença MIT.
