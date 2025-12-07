# Análise de Grafos, Comunidades e Pessoas

Módulo para análise de grafos de relacionamentos, relatórios de comunidades e evolução clínica de pessoas, focado em identificar nós com mais arestas, comunidades importantes e extrair insights sobre padrões de conexão e trajetórias clínicas.

## Requisitos
Para o uso efetivo dessas features é necessário os arquivos de extensão `.parquet` que são disponibilizados após a execução do `GraphRAG`.

## Funcionalidades

### Análise de Grafos (Relationships)
- **Análise de graus de nós**: Calcula o número de arestas (relacionamentos) de cada nó, incluindo grau de entrada (in-degree) e grau de saída (out-degree)
- **Identificação de nós centrais**: Identifica os nós com mais conexões no grafo
- **Extração de insights**: Analisa padrões de relacionamento, tipos mais comuns e pesos médios
- **Geração de relatórios**: Gera relatórios em JSON e CSV com análises detalhadas

### Análise de Comunidades (Community Reports)
- **Análise de rankings**: Analisa a importância das comunidades baseado em rankings gerados por LLM
- **Análise hierárquica**: Examina a estrutura hierárquica das comunidades (níveis, parent-child)
- **Análise de findings**: Extrai e analisa os findings (achados) de cada comunidade
- **Análise de padrões textuais**: Identifica palavras-chave e padrões nos títulos e summaries
- **Distribuição de tamanhos**: Analisa a distribuição estatística dos tamanhos das comunidades
- **Geração de relatórios completos**: Gera relatórios detalhados em JSON e CSV

### Análise de Pessoas (Person Analysis)
- **Identificação de pessoas**: Filtra e identifica entidades do tipo "PERSON" no grafo
- **Análise de relacionamentos**: Calcula o número de relacionamentos (vértices) de cada pessoa
- **Categorização clínica**: Categoriza relacionamentos em tipos clínicos (consultas, exames, procedimentos, diagnósticos, tratamentos, medicamentos, hospitalizações, localizações)
- **Análise de evolução clínica**: Extrai insights sobre a trajetória clínica de uma pessoa específica
- **Extração de padrões temporais**: Identifica datas e eventos temporais nas descrições
- **Geração de resumo clínico**: Gera resumo automático da evolução clínica baseado nas categorias identificadas
- **Relatórios detalhados**: Gera relatórios completos em JSON com análise clínica estruturada

## Estrutura

```
src/core/analysis/
├── __init__.py                      # Exportações do módulo
├── models/
│   ├── graph_analyzer.py            # Classe principal de análise de grafos
│   ├── community_analyzer.py        # Classe principal de análise de comunidades
│   └── person_analyzer.py           # Classe principal de análise de pessoas
├── facade.py                        # Interface simplificada
├── run_usage.py                     # Exemplo de uso (análise de grafos)
├── run_community_analysis.py        # Exemplo de uso (análise de comunidades)
├── run_person_analysis.py           # Exemplo de uso (análise de pessoas)
└── README.md                        # Esta documentação
```

## Uso Básico

### Usando a Facade (Recomendado)

#### Análise de Grafos

```python
from src.core.analysis import GraphAnalysisFeature

# Inicializa a feature
analyzer = GraphAnalysisFeature(
    relationships_path="datatest/output/relationships.parquet",
    entities_path="datatest/output/entities.parquet",  # Opcional
    community_reports_path="datatest/output/community_reports.parquet"  # Opcional
)

# Analisa os top 20 nós
top_nodes = analyzer.analyze(top_n=20)

# Obtém insights detalhados de um nó específico
insights = analyzer.get_insights(node_id="algum_node_id")

# Gera relatório completo
report_path = analyzer.generate_report(
    output_path="output/graph_analysis_report.json",
    top_n=20
)
```

#### Análise de Comunidades

```python
# Analisa rankings das comunidades
rankings = analyzer.analyze_communities(top_n=20)

# Obtém top comunidades
top_communities = analyzer.get_top_communities(top_n=10, by='rank')

# Analisa findings
findings = analyzer.analyze_community_findings(top_n=20)

# Gera relatório completo de comunidades
report_path = analyzer.generate_community_report(
    output_path="output/community_analysis_report.json",
    top_n=20
)
```

#### Análise de Pessoas

```python
# Obtém pessoas com mais relacionamentos
top_persons = analyzer.get_top_persons(top_n=10)

# Analisa evolução clínica de uma pessoa específica
clinical_analysis = analyzer.analyze_person_clinical_evolution(
    person_id="MARIA"  # Pode ser ID ou nome
)

# Gera relatório completo de análise de pessoa
report_path = analyzer.generate_person_report(
    person_id="MARIA",  # Opcional: se None, usa a pessoa com mais relacionamentos
    output_path="output/person_analysis_report.json"
)
```

### Usando o Analisador Diretamente

#### Análise de Grafos

```python
from src.core.analysis.models import GraphAnalyzer

# Inicializa o analisador
analyzer = GraphAnalyzer(
    relationships_path="datatest/output/relationships.parquet",
    entities_path="datatest/output/entities.parquet"
)

# Carrega os dados
analyzer.load_data()

# Analisa os nós
top_nodes_df = analyzer.analyze_top_nodes(top_n=20)

# Obtém insights de um nó
insights = analyzer.get_node_insights(node_id="algum_node_id")

# Gera relatório
report_path = analyzer.generate_report("output/report.json", top_n=20)
```

#### Análise de Pessoas

```python
from src.core.analysis.models import PersonAnalyzer

# Inicializa o analisador
person_analyzer = PersonAnalyzer(
    relationships_path="datatest/output/relationships.parquet",
    entities_path="datatest/output/entities.parquet"
)

# Carrega os dados
person_analyzer.load_data()

# Obtém pessoas com mais relacionamentos
top_persons = person_analyzer.get_top_persons(top_n=10)

# Analisa evolução clínica
clinical_analysis = person_analyzer.analyze_clinical_evolution(person_id="MARIA")

# Gera relatório completo
report_path = person_analyzer.generate_person_report(
    person_id="MARIA",
    output_path="output/person_analysis_report.json"
)
```

## Formato dos Dados

### Arquivo relationships.parquet

O arquivo deve conter as seguintes colunas:
- `source` ou `subject_id`: ID do nó origem
- `target` ou `object_id`: ID do nó destino
- `description`: Descrição do relacionamento (opcional)
- `weight`: Peso do relacionamento (opcional)
- `id`: ID único do relacionamento

### Arquivo entities.parquet (Opcional)

Se fornecido, enriquece os resultados com:
- `id`: ID da entidade (deve corresponder aos IDs em relationships)
- `title`: Título da entidade
- `type`: Tipo da entidade
- `description`: Descrição da entidade

### Arquivo community_reports.parquet (Opcional)

Para análise de comunidades, o arquivo deve conter:
- `id`: ID único do relatório
- `community`: Número da comunidade
- `title`: Título da comunidade
- `summary`: Resumo executivo
- `rank`: Ranking de importância (0-10)
- `rank_explanation`: Explicação do ranking
- `findings`: Lista de findings (achados)
- `size`: Tamanho da comunidade (número de entidades)
- `level`: Nível hierárquico
- `parent`: ID da comunidade pai (se houver)
- `period`: Período temporal (se aplicável)

### Requisitos para Análise de Pessoas

Para análise de pessoas, é necessário:
- **entities.parquet**: Deve conter entidades com `type = "PERSON"` (case-insensitive)
- **relationships.parquet**: A coluna `source` deve conter nomes/títulos das pessoas (não apenas IDs)
- As entidades do tipo "PERSON" devem ter a coluna `title` com o nome da pessoa
- Os relacionamentos devem ter a coluna `description` com descrições detalhadas para categorização clínica

## Saída

### DataFrame de Análise

O método `analyze_top_nodes()` retorna um DataFrame com as seguintes colunas:

- `node_id`: ID do nó
- `title`: Título do nó (se disponível)
- `type`: Tipo do nó (se disponível)
- `description`: Descrição do nó (se disponível)
- `total_degree`: Número total de arestas
- `out_degree`: Número de arestas saindo
- `in_degree`: Número de arestas entrando
- `avg_weight`: Peso médio das arestas
- `top_relationship_types`: Tipos de relacionamento mais comuns
- `human_readable_id`: ID legível (se disponível)

### Relatório JSON (Análise de Grafos)

O relatório gerado contém:

```json
{
  "summary": {
    "total_relationships": 60755,
    "total_nodes_analyzed": 1234,
    "top_n_nodes": 20
  },
  "top_nodes": [...],
  "detailed_insights": {
    "node_id": {
      "node_id": "...",
      "title": "...",
      "type": "...",
      "degrees": {
        "total": 150,
        "out": 75,
        "in": 75
      },
      "outgoing_relationships": {
        "top_descriptions": {...},
        "top_targets": {...}
      },
      "incoming_relationships": {
        "top_descriptions": {...},
        "top_sources": {...}
      }
    }
  }
}
```

### DataFrame de Análise de Comunidades

O método `analyze_communities()` retorna um DataFrame com:
- `community`: Número da comunidade
- `title`: Título da comunidade
- `rank`: Ranking de importância
- `rank_explanation`: Explicação do ranking
- `size`: Tamanho da comunidade
- `level`: Nível hierárquico
- `findings_count`: Número de findings
- `summary_length`: Comprimento do resumo
- `has_parent`: Se tem comunidade pai
- `period`: Período temporal

### Relatório JSON (Análise de Comunidades)

O relatório de comunidades contém:

```json
{
  "summary": {
    "total_communities": 500,
    "total_reports_analyzed": 20,
    "analysis_date": "2024-01-01T00:00:00"
  },
  "rankings": {
    "top_communities": [...],
    "rankings_analysis": [...]
  },
  "hierarchy": {
    "total_communities": 500,
    "levels": {
      "count_by_level": {...},
      "avg_size_by_level": {...},
      "avg_rank_by_level": {...}
    },
    "hierarchy": {
      "with_parent": 300,
      "without_parent": 200,
      "root_communities_ratio": 0.4
    }
  },
  "size_distribution": {...},
  "text_patterns": {...},
  "findings_analysis": [...]
}
```

### DataFrame de Análise de Pessoas

O método `get_top_persons()` retorna um DataFrame com:
- `person_id`: ID da pessoa
- `person_name`: Nome da pessoa (chave usada para matching)
- `title`: Título da entidade
- `description`: Descrição da pessoa (truncada a 200 caracteres)
- `total_degree`: Número total de relacionamentos
- `out_degree`: Número de relacionamentos saindo (source)
- `in_degree`: Número de relacionamentos entrando (target)
- `human_readable_id`: ID legível (se disponível)

### Relatório JSON (Análise de Pessoas)

O relatório de análise de pessoa contém:

```json
{
  "analysis_date": "2025-12-06T21:11:49.046000",
  "person_analysis": {
    "person_id": "MARIA",
    "person_info": {
      "title": "MARIA",
      "description": "Descrição completa da pessoa...",
      "total_relationships": 302
    },
    "relationship_categories": {
      "consultas": {
        "count": 34,
        "relationships": [...]
      },
      "exames": {
        "count": 102,
        "relationships": [...]
      },
      "procedimentos": {
        "count": 14,
        "relationships": [...]
      },
      "diagnosticos": {
        "count": 12,
        "relationships": [...]
      },
      "tratamentos": {
        "count": 4,
        "relationships": [...]
      },
      "hospitalizacoes": {
        "count": 11,
        "relationships": [...]
      },
      "localizacoes": {
        "count": 90,
        "relationships": [...]
      },
      "outros": {
        "count": 35,
        "relationships": [...]
      }
    },
    "top_relationship_types": [
      {
        "description": "Descrição do relacionamento...",
        "count": 1
      }
    ],
    "connected_entities": {
      "total": 302,
      "sample": [...]
    },
    "temporal_patterns": {
      "dates_found": 34,
      "sample_dates": ["02/10/2024", "01/07/2025", ...]
    },
    "clinical_summary": "Realizou 34 consultas médicas. Submeteu-se a 102 exames..."
  },
  "summary": {
    "total_relationships": 302,
    "categories_identified": 8,
    "connected_entities": 302
  }
}
```

#### Categorias Clínicas

As descrições dos relacionamentos são categorizadas automaticamente usando palavras-chave bilíngues (português e inglês):

- **consultas**: consultation, consulta, atendimento, appointment, visit
- **exames**: exam, exame, test, teste, ultrasound, ultrassom, endoscopy, endoscopia, lab, laboratório
- **procedimentos**: procedure, procedimento, surgery, cirurgia, cholecystectomy, colecistectomia, splenectomy, esplenectomia
- **diagnosticos**: diagnosis, diagnóstico, diagnosed, diagnosticado, disease, doença
- **tratamentos**: treatment, tratamento, therapy, terapia
- **medicamentos**: medication, medicamento, drug, medicina
- **hospitalizacoes**: hospitalization, hospitalização, hospital, hospitalizado
- **localizacoes**: resides, reside, native, nativo, location, localização, at, em
- **outros**: Relacionamentos que não se encaixam nas categorias acima

## Exemplo de Execução

### Análise de Grafos

Execute o exemplo incluído:

```bash
python -m src.core.analysis.example_usage
```

Ou use diretamente:

```bash
python src/core/analysis/example_usage.py
```

### Análise de Comunidades

Execute o exemplo de análise de comunidades:

```bash
python src/core/analysis/run_community_analysis.py
```

### Análise de Pessoas

Execute o exemplo de análise de pessoas:

```bash
python src/core/analysis/run_person_analysis.py
```

## Insights Extraídos

### Análise de Grafos

A análise identifica:

1. **Nós centrais**: Nós com maior número de conexões (hubs do grafo)
2. **Padrões de relacionamento**: Tipos de relacionamento mais frequentes
3. **Estrutura de conexões**: Diferença entre nós que recebem muitas conexões vs. nós que fazem muitas conexões
4. **Pesos médios**: Importância média das conexões de cada nó
5. **Top conexões**: Principais nós conectados a cada nó central

### Análise de Pessoas

A análise de pessoas identifica:

1. **Pessoas com mais relacionamentos**: Identifica as pessoas mais conectadas no grafo
2. **Categorização clínica**: Classifica relacionamentos em categorias clínicas (consultas, exames, procedimentos, etc.)
3. **Evolução clínica**: Traça a trajetória clínica de uma pessoa ao longo do tempo
4. **Padrões temporais**: Extrai datas e eventos temporais das descrições
5. **Entidades conectadas**: Identifica hospitais, especialidades, condições e procedimentos relacionados
6. **Resumo clínico**: Gera resumo automático da evolução clínica em linguagem natural
7. **Tipos de relacionamento mais comuns**: Identifica os relacionamentos mais frequentes para cada pessoa

## Casos de Uso

### Análise de Grafos
- Identificar entidades mais importantes em um grafo de conhecimento
- Encontrar padrões de relacionamento em dados médicos/anamnese
- Analisar centralidade de nós em grafos de documentos
- Extrair insights sobre conectividade em redes de informações

### Análise de Comunidades
- Identificar comunidades mais importantes baseado em rankings de LLM
- Analisar estrutura hierárquica de conhecimento
- Extrair findings e insights gerados automaticamente
- Identificar padrões textuais e palavras-chave em relatórios
- Analisar distribuição e tamanho de comunidades temáticas

### Análise de Pessoas
- Identificar pacientes/pessoas com mais interações clínicas
- Analisar evolução clínica de pacientes específicos
- Categorizar eventos clínicos (consultas, exames, procedimentos, diagnósticos)
- Extrair insights sobre trajetória de tratamento
- Identificar padrões temporais em históricos clínicos
- Gerar resumos clínicos automáticos para análise médica
- Analisar relacionamentos entre pacientes, hospitais, especialidades e condições

