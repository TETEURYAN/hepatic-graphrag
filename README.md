<div align="center">
<h1>GraphrRAG para prontuários hepatológicos</h1>
</div>

<div align="center" style="max-width: 850px; margin: 0 auto;">
<table>
  <tr>
    <td valign="center" style="padding-right: 20px; width: 200px;">
      <img src="https://user-images.githubusercontent.com/91018438/204195385-acc6fcd4-05a7-4f25-87d1-cb7d5cc5c852.png" alt="animated" width="230" />
    </td>
    <td valign="center" style="padding-left: 20px;">
      <p>Pipeline que avalia o <a href="https://github.com/microsoft/graphrag"><strong>GraphRAG</strong></a> (implementação oficial da Microsoft) como classificador da doença hepática principal de um paciente a partir de prontuários eletrônicos anonimizados. Cada prontuário vira um documento indexado num grafo de conhecimento; a doença é obtida perguntando ao índice e extraindo o diagnóstico da resposta em texto livre do LLM.</p>
      <p>Dois eixos são avaliados contra um ground truth anotado manualmente (141 prontuários): o <strong>método de busca</strong> do GraphRAG (<code>basic</code>/<code>local</code>/<code>global</code>/<code>drift</code>, em dois níveis de comunidade) e o efeito do <strong>prompt-tuning</strong> na etapa de indexação (extração de entidades e relatórios de comunidade).</p>
    </td>
  </tr>
</table>
</div>

Este repositório é o código de suporte de uma Tese de Láurea. A pergunta de pesquisa é se — e sob qual configuração — o GraphRAG
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
- Uma API key do [OpenRouter](https://openrouter.ai/keys), necessária para as chamadas ao LLM (`GRAPHRAG_API_KEY`) — **ou** um LLM rodando localmente (ver "Privacidade" abaixo).
- O dataset bruto de prontuários em `data/` — **não incluído no repositório** por conter dado clínico sensível; ver a seção de dados sensíveis do [log de operações](docs/log_operacoes.md).

## Privacidade

As queries do GraphRAG (indexação e busca) não precisam de um provedor
externo — o `settings.yaml` pode apontar `api_base` para um LLM rodando
localmente (Ollama, LM Studio etc.) em vez da OpenRouter, garantindo que
nenhum trecho de prontuário saia da máquina em nenhuma etapa do pipeline.
Essa é a opção recomendada quando o dataset real (não anonimizado ou
anonimizado de forma imperfeita) está em uso. Ver
[graphrag_config/README.md](graphrag_config/README.md) para como trocar o
`api_base`.

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

## Referências

- Alkhalaf, M., Yu, P., Yin, M., & Deng, C. (2024). Applying generative AI with retrieval augmented generation to summarize and extract key clinical information from electronic health records. *Journal of Biomedical Informatics*, 156, 104662. https://doi.org/10.1016/j.jbi.2024.104662
- Gadotti, A., Rocher, L., Houssiau, F., Creţu, A.-M., & de Montjoye, Y.-A. Anonymization: The imperfect science of using data while preserving privacy. *Science Advances*, 10(29), eadn7053. https://doi.org/10.1126/sciadv.adn7053
- Ge, J., Chen, I. Y., Pletcher, M. J., & Lai, J. C. (2024). How I Approach It: Prompt Engineering for Generative Artificial Intelligence (GAI) in Gastroenterology and Hepatology. *The American Journal of Gastroenterology*.
- Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T. J., Hao, S., Moody, B., Gow, B., Lehman, L.-w. H., Celi, L. A., & Mark, R. G. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10(1), 1. https://doi.org/10.1038/s41597-022-01899-x
- Khanbhai, M., Anyadi, P., Symons, J., Flott, K., Darzi, A., & Mayer, E. (2021). Applying natural language processing and machine learning techniques to patient experience feedback: a systematic review. *BMJ Health & Care Informatics*, 28(1), e100262. https://doi.org/10.1136/bmjhci-2020-100262
- Lahat, A., Shachar, E., Avidan, B., Glicksberg, B., & Klang, E. (2023). Evaluating the Utility of a Large Language Model in Answering Common Patients' Gastrointestinal Health-Related Questions: Are We There Yet? *Diagnostics*, 13(11), 1950. https://doi.org/10.3390/diagnostics13111950
- Lee, E. B., Heo, G. E., Choi, C. M., & Song, M. (2022). MLM-based typographical error correction of unstructured medical texts for named entity recognition. *BMC Bioinformatics*, 23(1), 486. https://doi.org/10.1186/s12859-022-05035-9
- Miao, J., Thongprayoon, C., Suppadungsuk, S., Garcia Valencia, O. A., & Cheungpasitporn, W. (2024). Integrating Retrieval-Augmented Generation with Large Language Models in Nephrology: Advancing Practical Applications. *Medicina*, 60(3), 445. https://doi.org/10.3390/medicina60030445
- Nascimento, M. R. S., Silva, V. A. F., Souza, G., Lima, K. G. S., Araújo, E. T. M., Souza, E. R., Turet, J., & Carvalho, V. D. H. (2025). Extração de Notícias sobre Segurança Pública para Desenvolvimento de Corpora em português: uma análise preliminar em cidades do nordeste brasileiro. *Anais do Workshop sobre as Implicações da Computação na Sociedade (WICS)*. https://sol.sbc.org.br/index.php/wics/article/view/35954
- Neha, F., Bhati, D., & Shukla, D. K. (2025). Retrieval-Augmented Generation (RAG) in Healthcare: A Comprehensive Review. *AI*, 6(9), 226. https://doi.org/10.3390/ai6090226
- Sarcinelli, J. L. L. L., Teixeira, M. L. G., Paiva, J. B., & Silva, D. F. (2025). MariNER: A Dataset for Historical Brazilian Portuguese Named Entity Recognition. *Anais da Brazilian Conference on Intelligent Systems (BRACIS)*. https://sol.sbc.org.br/index.php/bracis/article/view/40897
- Stammers, M., Ramgopal, B., Owusu Nimako, A., Vyas, A., Nouraei, R., Metcalf, C., Batchelor, J., Shepherd, J., & Gwiggner, M. (2025). A foundation systematic review of natural language processing applied to gastroenterology & hepatology. *BMC Gastroenterology*, 25(1), 58. https://doi.org/10.1186/s12876-025-03608-5
- Wang, M., Shen, Y., Zhao, B., Zhou, X., Sun, L., & Liu, X. (2025). Enhancing LLM-based clinical reasoning in anesthesiology via graph-augmented retrieval and explainable generation. *Health Information Science and Systems*, 13(1), 62. https://doi.org/10.1007/s13755-025-00379-x
- Wu, J., Zhu, J., Qi, Y., Chen, J., Xu, M., Menolascina, F., & Grau, V. (2024). Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation. *arXiv:2408.04187*. https://doi.org/10.48550/ARXIV.2408.04187
- Yang, R., Ning, Y., Keppo, E., Liu, M., Hong, C., Bitterman, D. S., Ong, J. C. L., Ting, D. S. W., & Liu, N. (2025). Retrieval-augmented generation for generative artificial intelligence in health care. *npj Health Systems*, 2(1), 2. https://doi.org/10.1038/s44401-024-00004-1
