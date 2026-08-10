# Visualizador de Resultados GraphRAG

Este script Python gera visualizações interativas dos resultados do GraphRAG a partir dos arquivos .parquet na pasta `datatest/output`.

## Instalação

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Uso

Execute o script:

```bash
python visualize_graphrag.py
```

O script irá:
1. Carregar todos os arquivos .parquet da pasta `datatest/output`
2. Gerar visualizações interativas em HTML na pasta `visualizacoes/`
3. Criar um relatório resumo

## Visualizações Geradas

O script gera os seguintes arquivos HTML na pasta `visualizacoes/`:

- **entidades.html**: Análise de entidades (top entidades, distribuição de tipos, estatísticas)
- **relacionamentos.html**: Grafo interativo de relacionamentos entre entidades
- **estatisticas_grafo.html**: Estatísticas do grafo (distribuição de graus, centralidade, etc.)
- **comunidades.html**: Análise de comunidades (tamanhos, distribuições)
- **documentos.html**: Informações sobre documentos processados
- **text_units.html**: Análise de unidades de texto
- **resumo_geral.html**: Visão geral de todos os componentes

## Estrutura dos Dados

O script espera os seguintes arquivos .parquet:
- `entities.parquet`: Entidades extraídas
- `relationships.parquet`: Relacionamentos entre entidades
- `communities.parquet`: Comunidades identificadas
- `documents.parquet`: Documentos processados
- `text_units.parquet`: Unidades de texto
- `community_reports.parquet`: Relatórios de comunidades

## Notas

- Os gráficos são interativos (use zoom, pan, hover para explorar)
- O grafo de relacionamentos pode ser grande - use zoom para navegar
- Todas as visualizações são salvas como HTML e podem ser abertas em qualquer navegador







