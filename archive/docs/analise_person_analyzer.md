# Análise do Person Analyzer - Passo a Passo

Este documento explica como o `PersonAnalyzer` gera o relatório de análise de evolução clínica, usando o exemplo de **MARIA** do arquivo `person_analysis_report.json`.

---

## 📊 Exemplo Analisado: MARIA

**Resultado Final:**
- **Total de relacionamentos:** 302
- **Categorias identificadas:** 8
- **Datas encontradas:** 34
- **Resumo clínico:** "Realizou 34 consultas médicas. Submeteu-se a 102 exames. Passou por 14 procedimentos. Recebeu 12 diagnósticos. Teve 11 hospitalizações."

---

## 🔄 Fluxo Completo de Geração do Relatório

### **FASE 1: Carregamento de Dados** (`load_data()`)

```python
# Linha 36-84 do person_analyzer.py
```

**Passo 1.1:** Carrega `relationships.parquet`
- Lê todos os relacionamentos do grafo
- Exemplo: 113.115 relacionamentos carregados

**Passo 1.2:** Carrega `entities.parquet`
- Lê todas as entidades do grafo
- Exemplo: 55.519 entidades carregadas

**Passo 1.3:** Filtra entidades do tipo "PERSON"
- Converte `type` para maiúsculas e compara com "PERSON"
- Exemplo: Encontra entidades como "MARIA", "JULIANA BRASIL", etc.
- **Resultado:** Lista de pessoas identificadas

---

### **FASE 2: Identificação de Pessoas com Mais Relacionamentos** (`get_top_persons()`)

```python
# Linha 264-316 do person_analyzer.py
```

**Passo 2.1:** Chama `_calculate_person_degrees()`
- Obtém nomes das pessoas via `_get_person_names()`
- Compara coluna `source` dos relacionamentos com nomes das pessoas

**Passo 2.2:** Filtra relacionamentos onde `source` é uma pessoa
- Para cada relacionamento, verifica se `source` corresponde a um nome de pessoa
- Usa comparação case-insensitive
- **Exemplo para MARIA:**
  - Encontra relacionamentos onde `source = "MARIA"`
  - Total encontrado: 302 relacionamentos

**Passo 2.3:** Agrupa relacionamentos por pessoa
- Cria dicionário `person_stats` com:
  - `out_degree`: número de relacionamentos saindo
  - `out_relationships`: lista de relacionamentos
  - `all_relationships`: todos os relacionamentos

**Passo 2.4:** Enriquece com informações das entidades (`_enrich_person_info()`)
- Busca título, descrição, tipo da entidade
- Adiciona ao `person_stats`
- **Para MARIA:** Adiciona descrição completa (498 palavras)

**Passo 2.5:** Ordena por número de relacionamentos
- Ordena pessoas por `total_degree` (decrescente)
- **Resultado:** MARIA aparece em primeiro lugar com 302 relacionamentos

---

### **FASE 3: Análise de Evolução Clínica** (`analyze_clinical_evolution()`)

```python
# Linha 318-484 do person_analyzer.py
```

**Passo 3.1:** Localiza a pessoa nas estatísticas
- Busca "MARIA" no dicionário `person_stats`
- Obtém todos os 302 relacionamentos

**Passo 3.2:** Categoriza relacionamentos por tipo clínico

O algoritmo percorre cada relacionamento e analisa a `description` procurando palavras-chave:

```python
keywords = {
    'consultas': ['consultation', 'consulta', 'atendimento', ...],
    'exames': ['exam', 'exame', 'test', 'ultrasound', ...],
    'procedimentos': ['procedure', 'surgery', 'cirurgia', ...],
    'diagnosticos': ['diagnosis', 'diagnóstico', ...],
    'tratamentos': ['treatment', 'tratamento', ...],
    'medicamentos': ['medication', 'medicamento', ...],
    'hospitalizacoes': ['hospitalization', 'hospital', ...],
    'localizacoes': ['resides', 'native', 'at', 'em', ...]
}
```

**Exemplo de categorização:**

1. **Relacionamento:** `"Maria accompanied the patient to his first hepatology consultation..."`
   - Palavra-chave encontrada: "consultation"
   - **Categoria:** `consultas` ✅

2. **Relacionamento:** `"Endoscopia was performed on Maria revealing esophagitis..."`
   - Palavra-chave encontrada: "endoscopia"
   - **Categoria:** `exames` ✅

3. **Relacionamento:** `"Maria underwent cholecystectomy surgery at Hospital Veredas..."`
   - Palavra-chave encontrada: "surgery"
   - **Categoria:** `procedimentos` ✅

**Resultado para MARIA:**
- `consultas`: 34 relacionamentos
- `exames`: 102 relacionamentos
- `procedimentos`: 14 relacionamentos
- `diagnosticos`: 12 relacionamentos
- `tratamentos`: 4 relacionamentos
- `hospitalizacoes`: 11 relacionamentos
- `localizacoes`: 90 relacionamentos
- `outros`: 35 relacionamentos

**Passo 3.3:** Extrai entidades conectadas
- Coleta todos os `target` dos relacionamentos
- Identifica entidades únicas conectadas a MARIA
- **Resultado:** 302 entidades únicas conectadas

**Passo 3.4:** Extrai padrões temporais
- Procura por datas nas descrições usando regex:
  - Padrões: `YYYY-MM-DD`, `DD/MM/YYYY`, `DD.MM.YYYY`
- **Exemplo encontrado:** "07/12/21", "July 18, 2024", etc.
- **Resultado:** 34 datas encontradas

**Passo 3.5:** Identifica top tipos de relacionamento
- Conta frequência de cada descrição
- Retorna os 10 mais comuns
- **Exemplo:** "Maria accompanied the patient..." aparece 1 vez

**Passo 3.6:** Gera resumo clínico (`_generate_clinical_summary()`)
- Combina contagens das categorias em texto natural
- **Resultado:** "Realizou 34 consultas médicas. Submeteu-se a 102 exames..."

---

### **FASE 4: Compilação do Relatório** (`generate_person_report()`)

```python
# Linha 510-559 do person_analyzer.py
```

**Passo 4.1:** Se não especificou pessoa, identifica a com mais relacionamentos
- Chama `get_top_persons(top_n=1)`
- Seleciona "MARIA" (302 relacionamentos)

**Passo 4.2:** Executa análise clínica completa
- Chama `analyze_clinical_evolution("MARIA")`
- Obtém todos os dados categorizados

**Passo 4.3:** Compila estrutura JSON do relatório

```json
{
  "analysis_date": "2025-12-06T21:11:49.046000",
  "person_analysis": {
    "person_id": "MARIA",
    "person_info": {...},
    "relationship_categories": {...},
    "top_relationship_types": [...],
    "connected_entities": {...},
    "temporal_patterns": {...},
    "clinical_summary": "..."
  },
  "summary": {
    "total_relationships": 302,
    "categories_identified": 8,
    "connected_entities": 302
  }
}
```

**Passo 4.4:** Salva arquivo JSON
- Cria diretório se não existir
- Salva com encoding UTF-8
- **Arquivo gerado:** `person_analysis_report.json`

---

## 📈 Exemplo Detalhado: Categorização de um Relacionamento

Vamos seguir um relacionamento específico de MARIA:

### **Relacionamento Original:**
```json
{
  "source": "MARIA",
  "target": "AMBULATÓRIO DE ESPECIALIDADES GERAIS 1",
  "description": "Maria, a patient at Ambulatório de Especialidades Gerais 1, has received comprehensive hepatology care, including initial consultations, treatment planning, follow-up visits...",
  "weight": 84.0
}
```

### **Processo de Categorização:**

1. **Análise da descrição:**
   - Texto: "Maria, a patient at Ambulatório... has received comprehensive hepatology care, including initial **consultations**..."
   - Palavra-chave encontrada: "consultations" ✅

2. **Classificação:**
   - Verifica lista de keywords de `consultas`
   - Encontra match: "consultation" está na lista
   - **Categoria atribuída:** `consultas`

3. **Armazenamento:**
   ```python
   relationship_categories['consultas'].append({
       'target': "AMBULATÓRIO DE ESPECIALIDADES GERAIS 1",
       'description': "...",
       'weight': 84.0,
       'id': "...",
       'type': 'outgoing'
   })
   ```

4. **No relatório final:**
   - Contador de `consultas` incrementado
   - Relacionamento adicionado à lista (top 5 mostrados)

---

## 🔍 Insights Extraídos para MARIA

### **1. Perfil Clínico:**
- **Idade:** 74-86 anos
- **Condições principais:** 
  - Cirrose descompensada (Child-Pugh B)
  - Diabetes tipo 2
  - Hipertensão arterial
  - Esteatose hepática

### **2. Trajetória de Cuidados:**
- **34 consultas médicas** (principalmente hepatologia)
- **102 exames realizados** (ultrassons, endoscopias, laboratoriais)
- **14 procedimentos** (paracentese, colecistectomia, etc.)
- **11 hospitalizações**

### **3. Padrões Temporais:**
- Datas encontradas: 34 eventos datados
- Período: 2017-2025
- Exemplos: "07/12/21", "July 18, 2024", "19/12/2022"

### **4. Entidades Conectadas:**
- Hospitais: HGE, HUPAA, Hospital Veredas, Santa Casa
- Especialidades: Hepatologia, Cardiologia, Nefrologia
- Condições: Diabetes, Hipertensão, Hepatite B
- Procedimentos: Paracentese, Colecistectomia, Endoscopia

---

## 🎯 Resumo do Algoritmo

```
1. CARREGA DADOS
   ├── relationships.parquet (113k relacionamentos)
   └── entities.parquet (55k entidades)

2. IDENTIFICA PESSOAS
   ├── Filtra type = "PERSON"
   └── Obtém nomes/títulos

3. FILTRA RELACIONAMENTOS
   ├── Compara source com nomes de pessoas
   └── Agrupa por pessoa

4. CALCULA ESTATÍSTICAS
   ├── Conta relacionamentos por pessoa
   └── Ordena por total_degree

5. ANALISA EVOLUÇÃO CLÍNICA
   ├── Categoriza relacionamentos (consultas, exames, etc.)
   ├── Extrai entidades conectadas
   ├── Identifica padrões temporais
   └── Gera resumo clínico

6. GERA RELATÓRIO
   ├── Compila estrutura JSON
   └── Salva arquivo
```

---

## 💡 Pontos Importantes

1. **Comparação por Nome:** O algoritmo compara a coluna `source` com os **nomes/títulos** das pessoas, não com IDs
2. **Case-Insensitive:** A comparação ignora maiúsculas/minúsculas
3. **Categorização Inteligente:** Usa palavras-chave em português e inglês para categorizar
4. **Top 5 por Categoria:** Mostra apenas os 5 principais relacionamentos de cada categoria no relatório
5. **Extração de Datas:** Usa regex para encontrar padrões de data em diferentes formatos
6. **Resumo Automático:** Gera texto natural a partir das categorias identificadas

---

## 📝 Estrutura Final do Relatório

O relatório JSON contém:

1. **`analysis_date`:** Data/hora da análise
2. **`person_analysis`:** Análise completa da pessoa
   - `person_info`: Informações básicas
   - `relationship_categories`: Relacionamentos categorizados
   - `top_relationship_types`: Tipos mais comuns
   - `connected_entities`: Entidades conectadas
   - `temporal_patterns`: Datas encontradas
   - `clinical_summary`: Resumo em texto
3. **`summary`:** Estatísticas gerais

Este processo transforma dados brutos do grafo em insights clínicos estruturados e acionáveis! 🎉

