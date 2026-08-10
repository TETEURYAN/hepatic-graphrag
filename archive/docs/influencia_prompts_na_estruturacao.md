# Influência dos Prompts na Estruturação das Descrições

Este documento explica como os prompts do GraphRAG influenciam a qualidade e estruturação das descrições dos relacionamentos, permitindo a categorização eficiente no `PersonAnalyzer`.

---

## 🎯 Resposta Direta

**SIM, a boa estruturação das keywords existe principalmente por causa dos prompts**, mas há uma combinação de fatores:

1. **Prompts do GraphRAG** (principal fator) - definem o formato e qualidade das descrições
2. **Código de categorização** (fator secundário) - precisa ser bem projetado para aproveitar essa estruturação
3. **Qualidade do LLM** - influencia a consistência das descrições geradas

---

## 📝 Como os Prompts Influenciam

### **Prompt `extract_graph.txt` - Linha 16**

```text
- relationship_description: explanation as to why you think the source entity 
  and the target entity are related to each other
```

**Instrução chave:** O prompt pede uma **explicação** do relacionamento, não apenas um rótulo.

**Exemplo do prompt (linha 39):**
```
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION
{tuple_delimiter}Martin Smith is the Chair of the Central Institution and will 
answer questions at a press conference{tuple_delimiter}9)
```

**Impacto:** O LLM gera descrições descritivas e contextuais, como:
- ✅ "Maria accompanied the patient to his first **hepatology consultation**..."
- ✅ "Endoscopia was **performed on** Maria revealing esophagitis..."
- ✅ "Maria **underwent cholecystectomy surgery** at Hospital Veredas..."

### **Prompt `extract_graph.txt` - Linha 20**

```text
3. Return output in English as a single list...
```

**Instrução:** Retornar em inglês, mas o LLM pode usar termos técnicos em português quando apropriado.

**Resultado:** Descrições bilíngues que facilitam a categorização:
- Inglês: "consultation", "exam", "surgery", "diagnosis"
- Português: "consulta", "exame", "cirurgia", "diagnóstico"

---

## 🔍 Análise das Descrições Geradas

### **Exemplos Reais do Relatório de MARIA:**

#### **1. Consultas (34 relacionamentos)**
```
"Maria accompanied the patient to his first hepatology consultation..."
```
- ✅ Palavra-chave: "consultation" (inglês)
- ✅ Categorização funciona porque o código busca "consultation" E "consulta"

#### **2. Exames (102 relacionamentos)**
```
"Endoscopia was performed on Maria revealing esophagitis..."
```
- ✅ Palavra-chave: "performed" + contexto de exame
- ✅ "PCR HBV-DNA test performed on Maria..."
- ✅ Categorização funciona porque busca "exam", "test", "endoscopy"

#### **3. Procedimentos (14 relacionamentos)**
```
"Maria underwent cholecystectomy surgery at Hospital Veredas..."
```
- ✅ Palavra-chave: "surgery", "underwent"
- ✅ Categorização funciona porque busca "surgery", "procedure", "cirurgia"

---

## 🎨 Estruturação vs. Categorização

### **O que os Prompts Fazem:**

1. **Geram descrições descritivas** (não apenas rótulos)
   - ❌ Ruim: "related_to"
   - ✅ Bom: "Maria underwent cholecystectomy surgery..."

2. **Usam vocabulário médico consistente**
   - Termos técnicos em inglês e português
   - Verbos de ação claros: "performed", "underwent", "attended"

3. **Incluem contexto temporal e espacial**
   - "on July 18, 2024"
   - "at Hospital Veredas"

### **O que o Código de Categorização Faz:**

1. **Define palavras-chave bilíngues** (português + inglês)
   ```python
   keywords = {
       'consultas': ['consultation', 'consulta', 'atendimento', ...],
       'exames': ['exam', 'exame', 'test', 'ultrasound', ...],
       ...
   }
   ```

2. **Busca substrings nas descrições**
   - Não precisa de match exato
   - Funciona com variações e contextos

3. **Categoriza de forma hierárquica**
   - Primeira categoria que faz match
   - Fallback para "outros"

---

## 📊 Comparação: Com vs. Sem Boa Estruturação

### **Cenário 1: Prompts Bem Estruturados (Atual)**

**Descrição gerada:**
```
"Maria underwent cholecystectomy surgery at Hospital Veredas on 19/12/2022"
```

**Categorização:**
- ✅ Encontra "surgery" → categoria `procedimentos`
- ✅ Encontra data "19/12/2022" → padrão temporal
- ✅ Identifica hospital "Hospital Veredas" → entidade conectada

### **Cenário 2: Prompts Mal Estruturados (Hipotético)**

**Descrição gerada:**
```
"Maria related to Hospital Veredas"
```

**Categorização:**
- ❌ Não encontra palavras-chave específicas
- ❌ Vai para categoria "outros"
- ❌ Perde informações clínicas importantes

---

## 🔧 Como os Prompts Garantem Boa Estruturação

### **1. Instruções Explícitas (extract_graph.txt)**

```text
- relationship_description: explanation as to why you think the source entity 
  and the target entity are related to each other
```

**Efeito:** O LLM é instruído a **explicar**, não apenas rotular.

### **2. Exemplos no Prompt**

O prompt inclui exemplos que mostram o formato esperado:
```
("relationship"...MARTIN SMITH...CENTRAL INSTITUTION...
Martin Smith is the Chair of the Central Institution and will answer questions 
at a press conference...)
```

**Efeito:** O LLM aprende o padrão e replica em descrições similares.

### **3. Idioma Consistente**

```text
3. Return output in English...
```

**Efeito:** Descrições principalmente em inglês, mas com termos técnicos em português quando necessário.

---

## 💡 Por Que a Categorização Funciona Bem

### **Fator 1: Descrições Ricas dos Prompts**

Os prompts geram descrições que naturalmente contêm palavras-chave:
- "consultation" aparece porque o LLM descreve ações
- "surgery" aparece porque o LLM menciona procedimentos
- "diagnosis" aparece porque o LLM explica condições

### **Fator 2: Vocabulário Médico Consistente**

O LLM usa vocabulário médico padrão:
- Inglês: consultation, examination, procedure, diagnosis
- Português: consulta, exame, procedimento, diagnóstico

### **Fator 3: Código de Categorização Bilíngue**

O `PersonAnalyzer` busca palavras-chave em ambos idiomas:
```python
'consultas': ['consultation', 'consulta', 'atendimento', ...]
'exames': ['exam', 'exame', 'test', 'ultrasound', ...]
```

---

## 🎯 Conclusão

### **A estruturação existe principalmente por causa dos prompts porque:**

1. ✅ **Prompts definem o formato:** Pedem explicações descritivas, não rótulos
2. ✅ **Prompts fornecem exemplos:** Mostram o padrão esperado
3. ✅ **Prompts especificam idioma:** Inglês com termos técnicos quando necessário
4. ✅ **LLM segue instruções:** Gera descrições consistentes e ricas em contexto

### **Mas o código de categorização também é importante:**

1. ✅ **Define palavras-chave bilíngues:** Captura termos em português e inglês
2. ✅ **Busca flexível:** Usa substring matching, não exato
3. ✅ **Hierarquia de categorias:** Prioriza categorias mais específicas

### **Sem os prompts bem estruturados:**

- ❌ Descrições seriam genéricas ("related_to", "connected")
- ❌ Categorização falharia (não encontraria palavras-chave)
- ❌ Insights clínicos seriam perdidos

### **Sem o código de categorização bem projetado:**

- ⚠️ Mesmo com boas descrições, categorização seria limitada
- ⚠️ Não capturaria termos em português
- ⚠️ Perderia nuances entre categorias

---

## 📈 Recomendações

### **Para Melhorar Ainda Mais:**

1. **Ajustar prompts** para incluir mais termos médicos específicos
2. **Expandir palavras-chave** no código de categorização
3. **Adicionar categorias** específicas (ex: "medicamentos", "sintomas")
4. **Usar NLP** para melhorar categorização (ex: spaCy para NER médico)

---

**Em resumo:** Os prompts são o **fator principal** que garante descrições estruturadas e ricas em contexto, mas o código de categorização precisa ser bem projetado para aproveitar essa estruturação ao máximo! 🎯

