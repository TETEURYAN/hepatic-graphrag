# Documentação dos Arquivos Parquet - GraphRAG Output

Este documento descreve cada arquivo `.parquet` gerado pelo GraphRAG no diretório `datatest/output/`.

## Visão Geral

O GraphRAG (Graph Retrieval-Augmented Generation) processa documentos de texto e extrai um grafo de conhecimento estruturado. Cada arquivo `.parquet` representa uma camada diferente dessa estrutura de conhecimento.

---

## 1. `relationships.parquet` (Relacionamentos)

**Propósito:** Armazena as arestas (edges) do grafo de conhecimento - os relacionamentos entre entidades.

**Estrutura:**
- `id`: ID único do relacionamento
- `human_readable_id`: ID numérico legível
- `source` ou `subject_id`: ID da entidade origem do relacionamento
- `target` ou `object_id`: ID da entidade destino do relacionamento
- `description`: Descrição textual do relacionamento (ex: "Patient attended hepatology consultation at...")
- `weight`: Peso/importância do relacionamento (opcional)
- `combined_degree`: Grau combinado do relacionamento
- `text_unit_ids`: IDs das unidades de texto que mencionam este relacionamento
- `type`: Tipo do relacionamento (geralmente "RELATED")

**Exemplo de uso:** 
- Identificar como entidades se conectam
- Analisar padrões de relacionamento
- Encontrar nós centrais (hubs) no grafo

**Tamanho típico:** ~14MB (113.115 relacionamentos no seu caso)

---

## 2. `entities.parquet` (Entidades)

**Propósito:** Armazena os nós (nodes) do grafo - as entidades extraídas dos documentos.

**Estrutura:**
- `id`: ID único da entidade
- `human_readable_id`: ID numérico legível
- `title`: Título/nome da entidade (ex: "HEPATOLOGIA", "MACEIÓ", "PACIENTE")
- `type`: Tipo da entidade (ex: "organization", "person", "geo", "event")
- `description`: Descrição detalhada da entidade
- `text_unit_ids`: IDs das unidades de texto que mencionam esta entidade

**Exemplo de uso:**
- Listar todas as entidades identificadas
- Filtrar por tipo de entidade
- Enriquecer análises de relacionamentos com informações das entidades

**Tamanho típico:** ~15MB (55.519 entidades no seu caso)

---

## 3. `documents.parquet` (Documentos)

**Propósito:** Armazena os documentos originais processados pelo GraphRAG.

**Estrutura:**
- `id`: ID único do documento
- `human_readable_id`: ID numérico legível
- `title`: Título do documento
- `text`: Texto completo do documento
- `text_unit_ids`: IDs das unidades de texto que compõem este documento

**Exemplo de uso:**
- Recuperar o texto original de um documento
- Rastrear a origem de uma entidade ou relacionamento
- Analisar a distribuição de conteúdo entre documentos

**Tamanho típico:** ~9.3MB (116.299 documentos no seu caso)

---

## 4. `text_units.parquet` (Unidades de Texto)

**Propósito:** Armazena os chunks/fragmentos de texto em que os documentos foram divididos para processamento.

**Estrutura:**
- `id`: ID único da unidade de texto
- `human_readable_id`: ID numérico legível
- `text`: Texto da unidade
- `n_tokens`: Número de tokens na unidade
- `document_ids`: IDs dos documentos que contêm esta unidade
- `entity_ids`: IDs das entidades mencionadas nesta unidade
- `relationship_ids`: IDs dos relacionamentos extraídos desta unidade

**Exemplo de uso:**
- Entender como os documentos foram segmentados
- Rastrear contexto específico de uma entidade
- Analisar densidade de informações por unidade

**Tamanho típico:** ~16MB

---

## 5. `communities.parquet` (Comunidades)

**Propósito:** Armazena as comunidades detectadas no grafo - grupos de entidades relacionadas.

**Estrutura:**
- `id`: ID único da comunidade
- `human_readable_id`: ID numérico legível
- `community`: Número da comunidade
- `size`: Tamanho da comunidade (número de nós)
- Outros campos específicos de comunidade

**Exemplo de uso:**
- Identificar grupos temáticos no grafo
- Analisar estrutura modular do conhecimento
- Encontrar clusters de entidades relacionadas

**Tamanho típico:** ~17MB

---

## 6. `community_reports.parquet` (Relatórios de Comunidades)

**Propósito:** Armazena relatórios gerados por LLM sobre cada comunidade, fornecendo resumos e insights.

**Estrutura:**
- `id`: ID único do relatório
- `human_readable_id`: ID numérico legível
- `community`: Número da comunidade
- `parent`: ID da comunidade pai (se hierárquico)
- `level`: Nível hierárquico
- `title`: Título do relatório
- `summary`: Resumo da comunidade
- `full_content`: Conteúdo completo do relatório
- `rank`: Ranking da comunidade
- `rank_explanation`: Explicação do ranking
- `findings`: Achados principais
- `full_content_json`: Conteúdo em formato JSON
- `period`: Período temporal (se aplicável)
- `size`: Tamanho da comunidade

**Exemplo de uso:**
- Obter insights gerados automaticamente sobre grupos de entidades
- Entender o contexto e significado de comunidades
- Usar em sistemas de busca e recuperação de informação

**Tamanho típico:** ~48MB (maior arquivo, contém muito texto gerado)

---

## Relacionamentos entre Arquivos

```
documents.parquet
    ↓ (dividido em)
text_units.parquet
    ↓ (extrai)
entities.parquet ←→ relationships.parquet
    ↓ (agrupa em)
communities.parquet
    ↓ (gera relatórios)
community_reports.parquet
```

## Fluxo de Dados

1. **Documentos** são carregados e divididos em **unidades de texto**
2. **Entidades** são extraídas das unidades de texto
3. **Relacionamentos** são identificados entre entidades
4. **Comunidades** são detectadas no grafo de relacionamentos
5. **Relatórios** são gerados por LLM para cada comunidade

## Estatísticas do Seu Dataset

Baseado no arquivo `stats.json`:
- **Total de documentos:** 1 documento grande processado
- **Tempo total de processamento:** ~91.450 segundos (~25 horas)
- **Workflow mais demorado:** `extract_graph` (81.459s) - extração do grafo
- **Segundo mais demorado:** `create_community_reports` (8.951s) - geração de relatórios

## Casos de Uso por Arquivo

| Arquivo | Caso de Uso Principal |
|---------|----------------------|
| `relationships.parquet` | Análise de grafos, encontrar nós centrais, padrões de conexão |
| `entities.parquet` | Listagem de entidades, filtragem por tipo, enriquecimento de dados |
| `documents.parquet` | Recuperação de texto original, rastreamento de origem |
| `text_units.parquet` | Análise de segmentação, contexto específico |
| `communities.parquet` | Análise de estrutura modular, clusters temáticos |
| `community_reports.parquet` | Insights gerados por LLM, busca semântica, resumos |

## Notas Importantes

- Todos os arquivos usam o formato **Parquet** (eficiente para dados colunares)
- Os IDs são consistentes entre arquivos (ex: `text_unit_ids` em entities referencia `text_units.parquet`)
- O arquivo `relationships.parquet` é o mais importante para análise de grafos
- O arquivo `community_reports.parquet` contém insights de alto nível gerados por LLM
- Os arquivos podem ser lidos com `pandas.read_parquet()` ou ferramentas de big data

