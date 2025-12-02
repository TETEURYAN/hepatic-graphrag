# Documentação de Arquitetura
## Arquitetura do sistema

Este documento descreve a arquitetura do sistema de anonimização e pré-processamento de prontuários eletrônicos para análise com GraphRAG.

## Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura de Alto Nível](#arquitetura-de-alto-nível)
3. [Componentes Principais](#componentes-principais)
4. [Padrões de Design](#padrões-de-design)
5. [Fluxo de Dados](#fluxo-de-dados)
6. [Estrutura de Diretórios](#estrutura-de-diretórios)
7. [Dependências e Tecnologias](#dependências-e-tecnologias)
8. [Exemplos de Uso](#exemplos-de-uso)

---

## Visão Geral

O sistema é composto por dois núcleos principais:

1. **Núcleo de Anonimização** (`src/core/anonymization`): Responsável por carregar, validar e anonimizar dados de prontuários eletrônicos
2. **Núcleo de Pré-processamento** (`src/core/preprocess`): Transforma dados anonimizados em formato compatível com GraphRAG

A arquitetura segue princípios de **Separação de Responsabilidades (SRP)** e utiliza o padrão **Facade** para simplificar a interface de uso.

---

## Arquitetura de Alto Nível

```
┌─────────────────────────────────────────────────────────────┐
│                      main.py                                 │
│              (Interface de Linha de Comando)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌─────────▼──────────┐
│  Anonymization │          │   Preprocessing    │
│    Feature     │          │   (GraphRAG)       │
│   (Facade)     │          │                    │
└───────┬────────┘          └─────────┬──────────┘
        │                             │
        │                             │
┌───────▼─────────────────────────────▼──────────┐
│         Componentes Especializados             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Loader  │  │ Validator│  │   NLP    │    │
│  └──────────┘  └──────────┘  └──────────┘    │
│  ┌──────────┐  ┌──────────┐                  │
│  │ Explorer │  │ Reporter │                  │
│  └──────────┘  └──────────┘                  │
└───────────────────────────────────────────────┘
```

---

## Componentes Principais

### 1. Núcleo de Anonimização (`src/core/anonymization`)

#### 1.1. Facade (`facade.py`)

**Responsabilidade:** Interface principal que orquestra todas as operações de anonimização.

**Classe Principal:** `AnonymizationFeature`

**Métodos Principais:**
- `load_datasets()`: Carrega todos os CSVs do diretório especificado
- `validate_descriptions()`: Valida se descrições foram anonimizadas corretamente
- `validate_anonymization()`: Validação profunda com detecção de entidades e padrões sensíveis
- `generate_report()`: Gera relatórios em formato TXT

**Padrão:** Facade Pattern - simplifica a interface complexa do subsistema

```python
class AnonymizationFeature:
    def __init__(self, data_dir: str = "data")
    def load_datasets(self) -> dict
    def validate_descriptions(self, dataset_name: str) -> tuple
    def validate_anonymization(self, dataset_name: str) -> pd.DataFrame
    def generate_report(self, df: pd.DataFrame, filename: str) -> str
```

---

#### 1.2. Loader (`loader/data_loader.py`)

**Responsabilidade:** Carregamento e normalização de arquivos CSV.

**Função Principal:** `load_all_csvs()`

**Funcionalidades:**
- Detecção automática de separadores (`,` ou `;`)
- Normalização de nomes de colunas (lowercase, substituição de espaços por `_`)
- Suporte a múltiplos arquivos CSV
- Tratamento de erros de encoding

**Características:**
- Normalização automática de colunas (remove caracteres especiais, converte para lowercase)
- Detecção inteligente de separadores
- Retorna dicionário `{nome_arquivo: DataFrame}` ou DataFrame concatenado

```python
def load_all_csvs(
    data_dir: str = "data",
    concat: bool = False,
    normalize_cols: bool = True
) -> dict | pd.DataFrame
```

---

#### 1.3. NLP (`nlp/entity_cleaner.py`)

**Responsabilidade:** Remoção de nomes próprios usando Processamento de Linguagem Natural.

**Classe Principal:** `EntityCleaner`

**Tecnologia:** spaCy com modelo `pt_core_news_sm` (NER - Named Entity Recognition)

**Funcionalidades:**
- Identificação de entidades do tipo `PER` (Pessoa)
- Remoção de nomes próprios de textos
- Processamento de colunas individuais ou múltiplos datasets

**Métodos:**
- `_remove_person_names(text: str) -> str`: Remove nomes próprios de um texto
- `clean_column(df, column: str) -> pd.DataFrame`: Limpa uma coluna específica
- `clean_multiple(datasets: dict, column: str) -> dict`: Processa múltiplos datasets

```python
class EntityCleaner:
    def __init__(self, model: str = "pt_core_news_sm")
    def clean_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame
    def clean_multiple(self, datasets: dict, column: str) -> dict
```

---

#### 1.4. Validator (`validator/`)

##### 1.4.1. `description_matcher.py`

**Responsabilidade:** Validação básica de anonimização verificando se a coluna `descricao_anonimizada` contém o padrão `<nome anonimizado>`.

**Classe:** `DescriptionMatcher`

**Métodos:**
- `find_mismatches(df) -> List[int]`: Retorna índices de linhas inválidas
- `sample_mismatches(df, n: int) -> pd.DataFrame`: Retorna amostras de problemas

**Lógica:** Uma linha é inválida se `descricao_anonimizada` não contém `<nome anonimizado>`

---

##### 1.4.2. `anonymization_validator.py`

**Responsabilidade:** Validação profunda de anonimização com detecção avançada.

**Classe:** `AnonymizationValidator`

**Funcionalidades:**
- **Fuzzy Matching:** Detecta nomes mesmo com variações (ex: "Maria" vs "M a r i a")
- **Detecção de Entidades:** Identifica entidades residuais (PER, LOC, ORG) usando spaCy
- **Padrões Sensíveis:** Detecta CPF, telefones, datas usando regex
- **Normalização:** Remove acentos e pontuação para comparação

**Métodos:**
- `normalize(text: str) -> str`: Normaliza texto para comparação
- `fuzzy_match(nome: str, texto: str, threshold: float) -> tuple`: Busca fuzzy de nomes
- `contains_sensitive_patterns(text: str) -> bool`: Verifica padrões sensíveis
- `validate(df: pd.DataFrame) -> pd.DataFrame`: Executa validação completa

**Padrões Sensíveis Detectados:**
- Datas: `\d{2}/\d{2}/\d{4}`
- CPF: `\d{3}\.\d{3}\.\d{3}-\d{2}`
- Telefones: `\d{2}\s?\d{4,5}-\d{4}`
- Números longos: `\d{6,}`

```python
class AnonymizationValidator:
    def __init__(self, cleaner: EntityCleaner)
    def validate(self, df: pd.DataFrame) -> pd.DataFrame
    def fuzzy_match(self, nome: str, texto: str, threshold: float = 0.85) -> tuple
    def contains_sensitive_patterns(self, text: str) -> bool
```

---

#### 1.5. Explorer (`explorer/explorer.py`)

**Responsabilidade:** Exploração e visualização de dados.

**Classe:** `DatasetExplorer`

**Funcionalidades:**
- Exibição de informações básicas do dataset (número de registros, colunas)
- Amostragem de dados para análise rápida

**Métodos:**
- `show_samples(df, column: str, n: int, label: str)`: Exibe amostras de uma coluna
- `show_info(df, name: str)`: Exibe informações básicas do dataset

---

#### 1.6. Report (`report/report_generator.py`)

**Responsabilidade:** Geração de relatórios em formato TXT.

**Classe:** `ReportGenerator`

**Métodos:**
- `save_logs(logs: list, filename: str) -> str`: Salva lista de logs em TXT
- `save_dataframe(df: pd.DataFrame, filename: str) -> str`: Salva DataFrame em formato TSV (TXT)

**Formato de Saída:** Arquivos `.txt` com separador tabular (TSV)

---

#### 1.7. Exceptions (`exceptions.py`)

**Responsabilidade:** Definição de exceções customizadas.

**Hierarquia de Exceções:**
```
AnonymizationError (base)
├── MissingColumnError
└── EmptyDatasetError
```

**Uso:** Permite tratamento de erros específicos e informativos

---

### 2. Núcleo de Pré-processamento (`src/core/preprocess`)

#### 2.1. Preprocessor (`preprocess.py`)

**Responsabilidade:** Transformar dados anonimizados em formato compatível com GraphRAG.

**Classe:** `GraphRAGPreprocessor`

**Funcionalidades:**
- Gera arquivos de texto individuais a partir da coluna `descricao_anonimizada`
- Cria estrutura de diretórios esperada pelo GraphRAG (`input/texts/`)
- Gera arquivo de metadados (`metadata.csv`) com informações sobre os documentos

**Estrutura Gerada:**
```
input/
├── texts/
│   ├── doc_00001.txt
│   ├── doc_00002.txt
│   └── ...
└── metadata.csv
```

**Métodos:**
- `__init__(data_dir: str = None, output_dir: str = "graphrag/input", dataframe: pd.DataFrame = None)`
- `prepare() -> Path`: Processa os dados e gera os arquivos

**Inicialização:**
- Pode receber um `data_dir` (usa `AnonymizationFeature` para carregar)
- Ou receber um `dataframe` diretamente
- Cria automaticamente a estrutura de diretórios

```python
class GraphRAGPreprocessor:
    def __init__(
        self,
        data_dir: str = None,
        output_dir: str = "graphrag/input",
        dataframe: pd.DataFrame = None
    )
    def prepare(self) -> Path
```

---

## Fluxo de Dados

### Fluxo 1: Validação de Anonimização

```
1. main.py
   │
   ├─> AnonymizationFeature.load_datasets()
   │   └─> DataLoader.load_all_csvs()
   │       └─> Retorna: dict {nome: DataFrame}
   │
   ├─> AnonymizationFeature.validate_anonymization()
   │   ├─> AnonymizationValidator.validate()
   │   │   ├─> EntityCleaner (para detecção de entidades)
   │   │   ├─> fuzzy_match() (detecção de nomes)
   │   │   └─> contains_sensitive_patterns() (regex)
   │   │
   │   └─> Retorna: DataFrame com problemas encontrados
   │
   └─> AnonymizationFeature.generate_report()
       └─> ReportGenerator.save_dataframe()
           └─> Arquivo .txt gerado
```

### Fluxo 2: Pré-processamento para GraphRAG

```
1. main.py
   │
   ├─> AnonymizationFeature.load_datasets()
   │   └─> DataLoader.load_all_csvs()
   │       └─> Retorna: dict {nome: DataFrame}
   │
   ├─> GraphRAGPreprocessor(dataframe=df)
   │   └─> GraphRAGPreprocessor.prepare()
   │       ├─> Para cada linha do DataFrame:
   │       │   ├─> Extrai 'descricao_anonimizada'
   │       │   └─> Salva em input/texts/doc_XXXXX.txt
   │       │
   │       └─> Gera metadata.csv
   │
   └─> Estrutura pronta para graphrag index
```

---

## Estrutura de Diretórios

```
src/
└── core/
    ├── anonymization/
    │   ├── __init__.py
    │   ├── facade.py              # Interface principal (Facade)
    │   ├── exceptions.py          # Exceções customizadas
    │   ├── README.md              # Documentação do módulo
    │   │
    │   ├── loader/
    │   │   ├── __init__.py
    │   │   └── data_loader.py     # Carregamento de CSVs
    │   │
    │   ├── nlp/
    │   │   ├── __init__.py
    │   │   └── entity_cleaner.py  # Limpeza de entidades (spaCy)
    │   │
    │   ├── validator/
    │   │   ├── __init__.py
    │   │   ├── description_matcher.py      # Validação simples
    │   │   └── anonymization_validator.py  # Validação avançada
    │   │
    │   ├── explorer/
    │   │   ├── __init__.py
    │   │   └── explorer.py        # Exploração de dados
    │   │
    │   └── report/
    │       ├── __init__.py
    │       └── report_generator.py # Geração de relatórios
    │
    └── preprocess/
        ├── __init__.py
        └── preprocess.py          # Pré-processamento para GraphRAG
```

---

## Dependências e Tecnologias

### Dependências Principais

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| `pandas` | - | Manipulação de dados tabulares |
| `spacy` | >=3.8.7 | Processamento de linguagem natural |
| `pt_core_news_sm` | - | Modelo spaCy para português (NER) |

### Dependências Internas

```
AnonymizationFeature
├── DataLoader
├── DescriptionMatcher
├── AnonymizationValidator
│   └── EntityCleaner (spaCy)
├── DatasetExplorer
└── ReportGenerator

GraphRAGPreprocessor
└── AnonymizationFeature (opcional, se usar data_dir)
```

### Tecnologias Utilizadas

- **Python 3.10+**: Linguagem principal
- **spaCy**: Framework de NLP para detecção de entidades
- **pandas**: Manipulação e análise de dados
- **GraphRAG**: Sistema de RAG baseado em grafos (uso externo)

---

## Diagrama de Sequência - Validação

```
main.py          AnonymizationFeature    DataLoader    AnonymizationValidator    EntityCleaner
  │                        │                 │                  │                      │
  │──load_datasets()──────>│                 │                  │                      │
  │                        │──load_all_csvs()│                  │                      │
  │                        │<───DataFrames───│                  │                      │
  │<───datasets────────────│                 │                  │                      │
  │                        │                 │                  │                      │
  │──validate_anonymization()───────────────>│                  │                      │
  │                        │                 │                  │                      │
  │                        │                 │                  │──validate()──────────>│
  │                        │                 │                  │                      │
  │                        │                 │                  │<──nlp_model───────────│
  │                        │                 │                  │                      │
  │                        │                 │                  │──fuzzy_match()───────>│
  │                        │                 │                  │                      │
  │                        │                 │                  │──contains_patterns()─>│
  │                        │                 │                  │                      │
  │                        │                 │                  │<───results────────────│
  │<───results─────────────│                 │                  │                      │
  │                        │                 │                  │                      │
  │──generate_report()────>│                 │                  │                      │
  │                        │──save_dataframe()                  │                      │
  │<───report_path─────────│                 │                  │                      │
```


---

**Última atualização:** 2025-01-12

