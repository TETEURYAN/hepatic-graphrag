# Guia de Uso do GraphRAG para Hepatologia

Este guia fornece instruções práticas para usar o GraphRAG em projetos de análise de prontuários eletrônicos de hepatologia, utilizando o gerenciador de variáveis `uv`.

##  Índice

1. [Instalação e Configuração](#instalação-e-configuração)
2. [Comandos Principais](#comandos-principais)
3. [Fluxo de Trabalho Completo](#fluxo-de-trabalho-completo)
4. [Configuração para Hepatologia](#configuração-para-hepatologia)
5. [Exemplos Práticos](#exemplos-práticos)
6. [Troubleshooting](#troubleshooting)

---

## Instalação e Configuração

### Pré-requisitos

- Python 3.10 ou superior
- `uv` instalado ([instruções de instalação](https://github.com/astral-sh/uv))

### Instalação do GraphRAG

```bash
# Ativar o ambiente virtual (se ainda não estiver ativo)
uv venv

# Instalar o GraphRAG e dependências
uv pip install graphrag>=2.7.0
```

### Configuração Inicial

1. **Inicializar um projeto GraphRAG:**

```bash
# Navegar até o diretório do projeto
cd datatest  # ou o diretório onde você quer trabalhar

# Inicializar o projeto (gera o arquivo settings.yaml padrão)
uv run graphrag init --root .
```

2. **Configurar variáveis de ambiente:**

Crie um arquivo `.env` na raiz do projeto com sua chave de API:

```bash
GRAPHRAG_API_KEY=sua_chave_api_aqui
```

3. **Verificar a configuração:**

O arquivo `settings.yaml` gerado contém todas as configurações necessárias. Para hepatologia, você pode precisar ajustar:

- Modelos de LLM e embedding
- Tipos de entidades a extrair
- Prompts personalizados

---

## Comandos Principais

### 1. `graphrag init` - Inicializar Projeto

Gera um arquivo de configuração padrão (`settings.yaml`).

```bash
# Inicializar no diretório atual
uv run graphrag init --root .

# Forçar inicialização mesmo se o projeto já existir
uv run graphrag init --root . --force
```

**Uso típico:** Execute uma vez no início do projeto para criar a estrutura base.

---

### 2. `graphrag index` - Construir Índice do Grafo

Este é o comando mais importante. Ele processa seus documentos e constrói o grafo de conhecimento.

```bash
# Indexação padrão (método standard)
uv run graphrag index --root . --config settings.yaml

# Indexação rápida (método fast - mais rápido, menos preciso)
uv run graphrag index --root . --config settings.yaml --method fast

# Com logs verbosos (recomendado para debug)
uv run graphrag index --root . --config settings.yaml --verbose

# Dry-run (valida configuração sem executar)
uv run graphrag index --root . --config settings.yaml --dry-run

# Sem cache (útil para testar mudanças)
uv run graphrag index --root . --config settings.yaml --no-cache

# Especificar diretório de saída customizado
uv run graphrag index --root . --config settings.yaml --output minha_saida
```

**Parâmetros importantes:**
- `--root`: Diretório raiz do projeto (onde está o `settings.yaml`)
- `--config`: Caminho para o arquivo de configuração
- `--method`: `standard` (padrão, mais completo) ou `fast` (mais rápido)
- `--verbose`: Mostra logs detalhados durante a execução
- `--dry-run`: Valida a configuração sem executar (útil para testar)
- `--output`: Sobrescreve o diretório de saída definido no config

**Tempo estimado:** Depende do volume de dados. Para prontuários de hepatologia:
- 100 documentos: ~10-30 minutos
- 1000 documentos: ~2-4 horas
- 10000 documentos: ~1-2 dias

---

### 3. `graphrag query` - Consultar o Grafo

Após indexar, você pode fazer consultas ao grafo de conhecimento.

```bash
# Busca local (recomendado para perguntas específicas)
uv run graphrag query \
  --root . \
  --config settings.yaml \
  --method local \
  --query "Quais são os principais sintomas de hepatite C?"

# Busca global (recomendado para perguntas amplas)
uv run graphrag query \
  --root . \
  --config settings.yaml \
  --method global \
  --query "Como os pacientes com cirrose evoluem ao longo do tempo?"

# Busca drift (para explorar tópicos relacionados)
uv run graphrag query \
  --root . \
  --config settings.yaml \
  --method drift \
  --query "Tratamento de ascite em pacientes hepáticos"

# Busca básica (mais simples, sem agregação)
uv run graphrag query \
  --root . \
  --config settings.yaml \
  --method basic \
  --query "Quais medicamentos são mais prescritos?"

# Com streaming (resposta aparece em tempo real)
uv run graphrag query \
  --root . \
  --config settings.yaml \
  --method local \
  --query "Sintomas de icterícia" \
  --streaming

# Especificar formato de resposta
uv run graphrag query \
  --root . \
  --config settings.yaml \
  --method global \
  --query "Padrões de evolução clínica" \
  --response-type "Lista de 5-7 pontos principais"

# Especificar diretório de dados (se diferente do padrão)
uv run graphrag query \
  --root . \
  --config settings.yaml \
  --method local \
  --query "Sua pergunta aqui" \
  --data output
```

**Métodos de busca:**
- **`local`**: Melhor para perguntas específicas sobre entidades ou conceitos conhecidos
- **`global`**: Melhor para perguntas amplas que requerem síntese de múltiplas comunidades
- **`drift`**: Útil para explorar tópicos relacionados e descobrir conexões inesperadas
- **`basic`**: Busca simples sem agregação complexa

**Parâmetros importantes:**
- `--method`: **Obrigatório**. Escolha entre `local`, `global`, `drift`, ou `basic`
- `--query`: **Obrigatório**. Sua pergunta em texto livre
- `--data`: Diretório onde estão os arquivos parquet gerados pela indexação
- `--streaming`: Mostra a resposta em tempo real (útil para respostas longas)
- `--response-type`: Formato desejado da resposta (ex: "Lista de 3-7 pontos")

---

### 4. `graphrag update` - Atualizar Índice Existente

Atualiza um índice existente com novos documentos ou configurações.

```bash
# Atualização padrão
uv run graphrag update --root . --config settings.yaml

# Atualização rápida
uv run graphrag update --root . --config settings.yaml --method fast-update

# Com logs verbosos
uv run graphrag update --root . --config settings.yaml --verbose
```

**Quando usar:**
- Adicionar novos prontuários ao índice existente
- Atualizar configurações e reindexar
- Corrigir erros na indexação anterior

**Nota:** Por padrão, salva em `update_output/`. Use `--output` para customizar.

---

### 5. `graphrag prompt-tune` - Personalizar Prompts

Gera prompts customizados baseados nos seus dados, otimizados para o domínio de hepatologia.

```bash
# Tuning básico (domínio inferido automaticamente)
uv run graphrag prompt-tune \
  --root . \
  --config settings.yaml \
  --domain "hepatologia"

# Com seleção automática de exemplos
uv run graphrag prompt-tune \
  --root . \
  --config settings.yaml \
  --domain "hepatologia" \
  --selection-method auto \
  --n-subset-max 300 \
  --k 15

# Seleção aleatória de exemplos
uv run graphrag prompt-tune \
  --root . \
  --config settings.yaml \
  --domain "hepatologia" \
  --selection-method random \
  --limit 20

# Especificar idioma
uv run graphrag prompt-tune \
  --root . \
  --config settings.yaml \
  --domain "hepatologia" \
  --language "português"

# Desabilitar descoberta automática de tipos de entidades
uv run graphrag prompt-tune \
  --root . \
  --config settings.yaml \
  --domain "hepatologia" \
  --no-discover-entity-types
```

**Parâmetros importantes:**
- `--domain`: Domínio dos dados (ex: "hepatologia", "medicina clínica")
- `--selection-method`: `all`, `random`, `top`, ou `auto` (recomendado)
- `--language`: Idioma dos dados (ex: "português", "português brasileiro")
- `--output`: Diretório para salvar os prompts (padrão: `prompts/`)

**Quando usar:**
- Primeira vez configurando o projeto
- Mudança significativa no tipo de dados
- Melhorar qualidade da extração de entidades

---

## Fluxo de Trabalho Completo

### Passo 1: Preparar os Dados

Certifique-se de que seus prontuários anonimizados estão na pasta `input/`:

```
datatest/
├── input/
│   ├── documento1.txt
│   ├── documento2.txt
│   └── ...
├── settings.yaml
└── prompts/
```

### Passo 2: Configurar o Settings.yaml

Edite o `settings.yaml` para seu contexto de hepatologia (veja seção abaixo).

### Passo 3: (Opcional) Ajustar Prompts

Se necessário, execute o tuning de prompts:

```bash
uv run graphrag prompt-tune \
  --root datatest \
  --config settings.yaml \
  --domain "hepatologia" \
  --language "português"
```

### Passo 4: Indexar os Dados

```bash
uv run graphrag index \
  --root datatest \
  --config settings.yaml \
  --verbose
```

**Aguarde a conclusão.** Isso pode levar várias horas dependendo do volume.

### Passo 5: Consultar o Grafo

```bash
uv run graphrag query \
  --root datatest \
  --config settings.yaml \
  --method local \
  --query "Quais são os principais sintomas relatados em pacientes com hepatite?"
```

### Passo 6: Analisar Resultados

Os resultados da indexação ficam em:
- `output/entities.parquet` - Entidades extraídas
- `output/relationships.parquet` - Relacionamentos
- `output/community_reports.parquet` - Relatórios de comunidades
- `output/documents.parquet` - Documentos processados

---

## Configuração para Hepatologia

### Tipos de Entidades Recomendados

No `settings.yaml`, ajuste a seção `extract_graph`:

```yaml
extract_graph:
  model_id: default_chat_model
  prompt: "prompts/extract_graph.txt"
  entity_types: 
    - person          # Pacientes, médicos
    - organization    # Hospitais, clínicas
    - geo             # Localizações
    - event           # Eventos clínicos
    - condition       # Condições médicas (hepatite, cirrose, etc.)
    - medication      # Medicamentos
    - procedure       # Procedimentos médicos
    - symptom         # Sintomas
    - lab_result      # Resultados de exames
  max_gleanings: 1
```

### Configuração de Chunks

Para prontuários médicos, recomenda-se:

```yaml
chunks:
  size: 1200          # Tamanho do chunk (caracteres)
  overlap: 100        # Sobreposição entre chunks
  group_by_columns: [id]  # Agrupar por ID do paciente (se aplicável)
```

### Configuração de Modelos

Para português, use modelos multilíngues:

```yaml
models:
  default_embedding_model:
    type: embedding
    model_provider: deepinfra/fp32
    model: intfloat/multilingual-e5-large  # Bom para português
    # ... outras configurações
```

### Prompts Personalizados

Os prompts em `prompts/` podem ser ajustados para o contexto médico:

- `extract_graph.txt`: Define como extrair entidades e relacionamentos
- `summarize_descriptions.txt`: Como resumir descrições de entidades
- `local_search_system_prompt.txt`: Como responder perguntas locais
- `global_search_*.txt`: Como fazer buscas globais

---

## Exemplos Práticos

### Exemplo 1: Indexação Inicial Completa

```bash
# 1. Validar configuração
uv run graphrag index --root datatest --config settings.yaml --dry-run

# 2. Indexar com logs
uv run graphrag index --root datatest --config settings.yaml --verbose

# 3. Verificar resultados
ls -la datatest/output/
```

### Exemplo 2: Consultas Típicas em Hepatologia

```bash
# Pergunta sobre sintomas
uv run graphrag query \
  --root datatest \
  --config settings.yaml \
  --method local \
  --query "Quais sintomas são mais comuns em pacientes com cirrose hepática?"

# Pergunta sobre evolução temporal
uv run graphrag query \
  --root datatest \
  --config settings.yaml \
  --method global \
  --query "Como os pacientes com hepatite C evoluem ao longo do tempo de tratamento?"

# Pergunta sobre tratamentos
uv run graphrag query \
  --root datatest \
  --config settings.yaml \
  --method drift \
  --query "Quais são os tratamentos mais eficazes para ascite em pacientes hepáticos?"

# Pergunta sobre exames
uv run graphrag query \
  --root datatest \
  --config settings.yaml \
  --method local \
  --query "Quais valores de bilirrubina são associados a pior prognóstico?"
```

### Exemplo 3: Atualização Incremental

```bash
# Adicionar novos prontuários à pasta input/
# Depois atualizar o índice
uv run graphrag update \
  --root datatest \
  --config settings.yaml \
  --method fast-update \
  --verbose
```

### Exemplo 4: Tuning de Prompts para Hepatologia

```bash
uv run graphrag prompt-tune \
  --root datatest \
  --config settings.yaml \
  --domain "hepatologia" \
  --language "português" \
  --selection-method auto \
  --n-subset-max 200 \
  --k 10 \
  --verbose
```

---

## Troubleshooting

### Problema: "API key not found"

**Solução:**
```bash
# Verificar se o arquivo .env existe e tem a chave
cat .env | grep GRAPHRAG_API_KEY

# Ou definir diretamente no terminal (Windows PowerShell)
$env:GRAPHRAG_API_KEY="sua_chave_aqui"

# Ou definir diretamente no terminal (Linux/Mac)
export GRAPHRAG_API_KEY="sua_chave_aqui"
```

### Problema: "No input files found"

**Solução:**
- Verifique se os arquivos estão em `input/` (ou o diretório configurado)
- Verifique o `file_type` no `settings.yaml` (deve ser `text`, `csv`, ou `json`)
- Para CSV, certifique-se de que a coluna de texto está configurada corretamente

### Problema: Indexação muito lenta

**Soluções:**
- Use `--method fast` para indexação mais rápida (menos precisa)
- Reduza `concurrent_requests` no `settings.yaml`
- Use cache: `--cache` (padrão)
- Considere processar em lotes menores

### Problema: Respostas de baixa qualidade

**Soluções:**
- Execute `prompt-tune` com seus dados
- Ajuste os prompts manualmente em `prompts/`
- Use `--method global` para perguntas amplas
- Verifique se os tipos de entidades estão corretos

### Problema: Erro de memória

**Soluções:**
- Reduza `chunks.size` no `settings.yaml`
- Processe menos documentos por vez
- Use `--memprofile` para identificar gargalos
- Aumente a memória disponível

### Problema: Modelo não responde em português

**Solução:**
- Use modelos multilíngues (ex: `multilingual-e5-large` para embeddings)
- Configure `--language "português"` no `prompt-tune`
- Ajuste os prompts para incluir instruções em português

---

## Dicas e Boas Práticas

1. **Sempre valide antes de indexar:**
   ```bash
   uv run graphrag index --dry-run
   ```

2. **Use `--verbose` durante desenvolvimento** para entender o que está acontecendo.

3. **Mantenha backups** dos arquivos `output/` - a indexação pode levar muito tempo.

4. **Teste com subconjunto pequeno primeiro** antes de processar todo o dataset.

5. **Documente suas perguntas** - mantenha um log das consultas que funcionam bem.

6. **Ajuste os prompts** para seu contexto específico - prompts genéricos podem não funcionar bem para dados médicos.

7. **Use cache durante desenvolvimento** - desabilite apenas quando quiser forçar reprocessamento.

---

## Recursos Adicionais

- [Documentação oficial do GraphRAG](https://microsoft.github.io/graphrag/)
- [Configuração completa do settings.yaml](https://microsoft.github.io/graphrag/config/yaml/)
- [Guia de prompts](https://microsoft.github.io/graphrag/prompts/)

---

## Suporte

Para problemas específicos do projeto, consulte:
- `README.md` na raiz do projeto
- Logs em `datatest/logs/`
- Arquivos de configuração em `datatest/settings.yaml`

---

**Última atualização:** 2025-01-12

