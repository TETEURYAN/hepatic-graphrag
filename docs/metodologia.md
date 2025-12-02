# Documentação do Processo de Pesquisa

Este documento descreve a metodologia utilizada no desenvolvimento deste Trabalho de Conclusão de Curso, focando em análise de prontuários eletrônicos de hepatologia usando GraphRAG.

## Índice

1. [Visão Geral](#visão-geral)
2. [Metodologia Ágil Adaptada](#metodologia)
3. [Milestones do Projeto](#milestones-do-projeto)
4. [Gestão de Issues](#gestão-de-issues)
5. [Fluxo de Trabalho](#fluxo-de-trabalho)
6. [Ferramentas e Práticas](#ferramentas-utilizadas)

---

## Visão Geral

Este TCC utiliza uma **metodologia ágil** que combina princípios de desenvolvimento ágil (como Scrum/Kanban) com as necessidades específicas de um trabalho acadêmico. A metodologia é organizada em **9 milestones principais** (M1 a M9), cada um representando uma fase crítica do desenvolvimento do trabalho.

---

## Metodologia 

### Estrutura

A metodologia é baseada em **Milestones** (marcos) que representam fases principais do TCC, e **Issues** (tarefas) que quebram cada milestone em atividades concretas e executáveis.

```
TCC
└── Milestone 1 (M1)
    ├── Issue 1.1
    ├── Issue 1.2
    └── Issue 1.3
└── Milestone 2 (M2)
    ├── Issue 2.1
    └── ...
```

---

## Milestones do Projeto

### M1 – Definição de Escopo & Proposta

**Objetivo:** Formalizar tema, problema de pesquisa, objetivos, hipóteses e plano de trabalho.

**Entregas:**
- Documento de proposta do TCC
- Definição clara do problema de pesquisa
- Objetivos gerais e específicos
- Hipóteses de trabalho
- Plano de trabalho com cronograma

**Critérios de Aceitação:**
- [ ] Tema definido e aprovado pelo orientador
- [ ] Problema de pesquisa claramente formulado
- [ ] Objetivos alinhados com o problema
- [ ] Hipóteses testáveis definidas
- [ ] Plano de trabalho com datas realistas

---

### M2 – Revisão Bibliográfica e Fundamentação Teórica

**Objetivo:** Levantar literatura sobre grafos de conhecimento em saúde, hepatologia, extração de dados, anonimização e GraphRAG.

**Entregas:**
- Revisão bibliográfica estruturada
- Fundamentação teórica sobre:
  - Grafos de conhecimento em saúde
  - Hepatologia e terminologias médicas
  - Extração e processamento de dados clínicos
  - Técnicas de anonimização de prontuários
  - GraphRAG e sistemas de RAG baseados em grafos
- Mapeamento de trabalhos relacionados

**Critérios de Aceitação:**
- [ ] Mínimo de 30-50 referências relevantes coletadas
- [ ] Revisão crítica da literatura realizada
- [ ] Fundamentação teórica escrita
- [ ] Trabalhos relacionados mapeados e comparados
- [ ] Lacunas de pesquisa identificadas

---

### M3 – Preparação e Tratamento dos Dados

**Objetivo:** Organizar prontuários, anonimizar dados, converter em formato compatível e gerar inputs para GraphRAG.

**Entregas:**
- Prontuários organizados e estruturados
- Pipeline de anonimização implementado e testado
- Dados convertidos para formato compatível com GraphRAG
- Estrutura de diretórios `input/` criada
- Validação da qualidade dos dados anonimizados

**Critérios de Aceitação:**
- [ ] Prontuários coletados e organizados
- [ ] Sistema de anonimização funcionando (spaCy, entity cleaning)
- [ ] Dados convertidos para `.txt` compatível com GraphRAG
- [ ] Validação de anonimização executada e documentada
- [ ] Estrutura `input/texts/` pronta para indexação

**Artefatos Técnicos:**
- Código em `src/core/anonymization/`
- Código em `src/core/preprocess/`
- Relatórios de validação de anonimização
- Estrutura de diretórios `input/`

---

### M4 – Indexação + Construção do Grafo com GraphRAG

**Objetivo:** Executar pipeline do GraphRAG, gerar nós/arestas, snapshots, exports (parquet, graphml, gexf).

**Entregas:**
- Configuração do GraphRAG (`settings.yaml`)
- Grafo de conhecimento indexado
- Arquivos de saída:
  - `entities.parquet` - Entidades extraídas
  - `relationships.parquet` - Relacionamentos
  - `documents.parquet` - Documentos processados
  - `community_reports.parquet` - Relatórios de comunidades
  - Snapshots em GraphML/GEXF (se configurado)
- Logs de indexação e validação

**Critérios de Aceitação:**
- [ ] `settings.yaml` configurado com modelos apropriados
- [ ] Comando `graphrag index` executado com sucesso
- [ ] Grafo gerado sem erros críticos
- [ ] Arquivos Parquet validados e acessíveis
- [ ] Estatísticas do grafo documentadas (número de nós, arestas, etc.)

**Comandos Principais:**
```bash
# Configurar
uv run graphrag init --root datatest

# Indexar
uv run graphrag index --root datatest --config settings.yaml --verbose

# Validar saída
ls -la datatest/output/
```
---

### M5 – Exploração e Visualização do Grafo

**Objetivo:** Converter e exportar grafo para visualização (ex: Gephi), revisar estrutura, comunidades, centralidades iniciais.

**Entregas:**
- Scripts de conversão Parquet → GEXF/GraphML
- Grafo importado e visualizado no Gephi (ou ferramenta similar)
- Análise inicial de:
  - Estrutura do grafo (densidade, componentes conectados)
  - Comunidades detectadas
  - Centralidades (grau, betweenness, closeness)
  - Layouts e visualizações iniciais
- Documentação das descobertas iniciais

**Critérios de Aceitação:**
- [ ] Script de conversão funcionando
- [ ] Grafo visualizado no Gephi
- [ ] Análise de comunidades realizada
- [ ] Métricas de centralidade calculadas
- [ ] Visualizações iniciais geradas e documentadas

**Artefatos Técnicos:**
- Scripts de conversão em `scripts/`
- Arquivos GEXF/GraphML exportados
- Visualizações (imagens, HTML)
- Relatório de análise inicial

---

### M6 – Extração e Análise de Relações / Insight Clínico

**Objetivo:** Filtrar subgrafos relevantes (ex: hepatologia), extrair relações e descritivos, cruzar com literatura médica.

**Entregas:**
- Subgrafo de hepatologia extraído
- Análise de relações clínicas relevantes
- Tabelas e descrições de entidades e relacionamentos
- Comparação com evidências da literatura médica
- Insights clínicos identificados

**Critérios de Aceitação:**
- [ ] Subgrafo de hepatologia filtrado e extraído
- [ ] Relações relevantes identificadas e documentadas
- [ ] Tabelas CSV/Excel com relações principais
- [ ] Comparação com literatura realizada
- [ ] Insights clínicos documentados

**Duração Estimada:** 3-4 semanas

**Análises Principais:**
- Filtrar entidades relacionadas a hepatologia
- Extrair relacionamentos entre condições, sintomas, tratamentos
- Identificar padrões clínicos
- Validar descobertas com literatura médica

---

### M7 – Escrita dos Resultados e Discussão

**Objetivo:** Redigir resultados, tabelas, gráficos, análise crítica e implicações para a hepatologia.

**Entregas:**
- Capítulo de Resultados completo
- Tabelas e gráficos formatados
- Análise crítica dos resultados
- Discussão dos achados
- Implicações para hepatologia e pesquisa futura
- Limitações do estudo

**Critérios de Aceitação:**
- [ ] Seção de resultados escrita e revisada
- [ ] Tabelas e gráficos incluídos e formatados
- [ ] Análise crítica realizada
- [ ] Discussão alinhada com objetivos
- [ ] Limitações e implicações documentadas

**Conteúdo Esperado:**
- Estatísticas descritivas do grafo
- Análise de comunidades e centralidades
- Relações clínicas identificadas
- Comparação com literatura
- Discussão crítica

---

### M8 – Escrita do TCC + Documentação Técnica do Código

**Objetivo:** Escrever introdução, metodologia, conclusão, anexos; limpar e documentar código e scripts.

**Entregas:**
- TCC completo com todos os capítulos:
  - Introdução
  - Revisão Bibliográfica
  - Metodologia
  - Resultados
  - Discussão
  - Conclusão
- Documentação técnica completa:
  - README atualizado
  - Documentação de arquitetura
  - Guias de uso
  - Comentários no código
- Código limpo e organizado
- Anexos 

**Critérios de Aceitação:**
- [ ] Todos os capítulos do TCC escritos
- [ ] Metodologia detalhada e reprodutível
- [ ] Código documentado e limpo
- [ ] README com instruções de reprodução
- [ ] Anexos incluídos 

**Documentação Técnica:**
- `README.md` - Visão geral do projeto
- `docs/arquitetura.md` - Arquitetura do sistema
- `docs/guia-graphrag.md` - Guia de uso do GraphRAG
- `docs/metodologia-agil-tcc.md` - Este documento
- Comentários e docstrings no código

---

### M9 – Revisão Final, Formatação e Defesa / Entrega

**Objetivo:** Revisar texto completo, formatar conforme normas, preparar apresentação, relatório final.

**Entregas:**
- TCC revisado e formatado (ABNT ou normas da instituição)
- PDF final gerado
- Apresentação preparada (slides)
- Material de defesa pronto
- Repositório finalizado e documentado
- Relatório final (se necessário)

**Critérios de Aceitação:**
- [ ] Revisão ortográfica e gramatical completa
- [ ] Formatação conforme normas da instituição
- [ ] Referências formatadas corretamente
- [ ] Apresentação preparada e revisada
- [ ] PDF final gerado
- [ ] Repositório atualizado e documentado

**Checklist Final:**
- [ ] Revisão completa do texto
- [ ] Formatação ABNT/instituição
- [ ] Referências validadas
- [ ] Figuras e tabelas numeradas
- [ ] Sumário atualizado
- [ ] Apresentação preparada
- [ ] Material de defesa pronto

---

## Gestão de Issues

### Estrutura de Issues

Cada milestone é quebrado em **issues** (tarefas) específicas e executáveis. As issues seguem o padrão:

```
[M#] Título da Issue

Descrição: O que precisa ser feito
Critérios de Aceitação: Como saber se está completo
Labels: milestone, tipo (documentação, código, análise, etc.)
```

### Exemplos de Issues por Milestone

#### M1 – Definição de Escopo & Proposta

**Issue 1.1:** Definir pergunta de pesquisa + hipóteses
- **Descrição:** Formular pergunta de pesquisa clara e hipóteses testáveis relacionadas ao uso de GraphRAG em análise de prontuários de hepatologia.
- **Critérios de Aceitação:**
  - [ ] Pergunta de pesquisa escrita e aprovada
  - [ ] Hipóteses formuladas e testáveis
  - [ ] Alinhamento com orientador

**Issue 1.2:** Escrever objetivos gerais e específicos
- **Descrição:** Definir objetivos gerais e específicos alinhados com a pergunta de pesquisa.
- **Critérios de Aceitação:**
  - [ ] Objetivo geral definido
  - [ ] 3-5 objetivos específicos definidos
  - [ ] Objetivos mensuráveis e alcançáveis

**Issue 1.3:** Montar plano de trabalho (fases + datas)
- **Descrição:** Criar cronograma detalhado com fases do trabalho e datas estimadas.
- **Critérios de Aceitação:**
  - [ ] Cronograma com todas as fases
  - [ ] Datas realistas definidas
  - [ ] Marcos (milestones) identificados

---

#### M2 – Revisão Bibliográfica e Fundamentação Teórica

**Issue 2.1:** Buscar artigos sobre grafos de conhecimento em biomedicina / saúde
- **Descrição:** Realizar busca sistemática em bases de dados (PubMed, IEEE, ACM) sobre grafos de conhecimento aplicados à saúde.
- **Critérios de Aceitação:**
  - [ ] Mínimo de 15 artigos relevantes coletados
  - [ ] Resumos revisados
  - [ ] Artigos principais selecionados

**Issue 2.2:** Revisar literatura sobre ontologias em hepatologia
- **Descrição:** Mapear ontologias e terminologias médicas específicas de hepatologia (SNOMED CT, ICD-10, etc.).
- **Critérios de Aceitação:**
  - [ ] Ontologias relevantes identificadas
  - [ ] Terminologias mapeadas
  - [ ] Relevância para o projeto documentada

**Issue 2.3:** Mapear trabalhos de anonimização de prontuários + ética de dados
- **Descrição:** Revisar literatura sobre técnicas de anonimização de dados clínicos e aspectos éticos.
- **Critérios de Aceitação:**
  - [ ] Técnicas de anonimização revisadas
  - [ ] Aspectos éticos documentados
  - [ ] Conformidade com LGPD/GDPR considerada

---

#### M3 – Preparação e Tratamento dos Dados

**Issue 3.1:** Coletar prontuários originais (CSV/TXT)
- **Descrição:** Obter acesso aos prontuários eletrônicos e organizá-los em formato estruturado.
- **Critérios de Aceitação:**
  - [ ] Prontuários coletados
  - [ ] Formato padronizado
  - [ ] Estrutura de diretórios criada

**Issue 3.2:** Implementar anonimização automática (spaCy, entity cleaning)
- **Descrição:** Desenvolver pipeline de anonimização usando spaCy para remoção de nomes próprios e entidades sensíveis.
- **Critérios de Aceitação:**
  - [ ] Código de anonimização implementado
  - [ ] Validação de anonimização funcionando
  - [ ] Relatórios de validação gerados

**Issue 3.3:** Converter os dados para .txt / input compatível GraphRAG
- **Descrição:** Transformar dados anonimizados em formato de texto compatível com GraphRAG.
- **Critérios de Aceitação:**
  - [ ] Estrutura `input/texts/` criada
  - [ ] Arquivos `.txt` gerados
  - [ ] Metadados criados (se necessário)

---

#### M4 – Indexação + Construção do Grafo

**Issue 4.1:** Configurar settings.yaml com modelos Ollama / embedding
- **Descrição:** Configurar arquivo de configuração do GraphRAG com modelos apropriados para português.
- **Critérios de Aceitação:**
  - [ ] `settings.yaml` configurado
  - [ ] Modelos de LLM e embedding definidos
  - [ ] Configuração testada

**Issue 4.2:** Executar graphrag index e validar logs
- **Descrição:** Executar pipeline de indexação e validar que o processo completou sem erros críticos.
- **Critérios de Aceitação:**
  - [ ] Indexação executada com sucesso
  - [ ] Logs revisados
  - [ ] Erros identificados e corrigidos (se houver)

**Issue 4.3:** Exportar grafo (graphml/gexf + parquet) para análise
- **Descrição:** Exportar grafo em formatos adequados para análise e visualização.
- **Critérios de Aceitação:**
  - [ ] Arquivos Parquet gerados
  - [ ] Snapshots em GraphML/GEXF (se configurado)
  - [ ] Arquivos validados e acessíveis

---

#### M5 – Exploração e Visualização do Grafo

**Issue 5.1:** Converter parquet → GEXF (script Python + NetworkX)
- **Descrição:** Desenvolver script para converter dados Parquet em formato GEXF para visualização.
- **Critérios de Aceitação:**
  - [ ] Script de conversão funcionando
  - [ ] Arquivo GEXF gerado
  - [ ] Validação do formato

**Issue 5.2:** Importar grafo no Gephi e rodar layout + modularidade
- **Descrição:** Importar grafo no Gephi, aplicar layouts e calcular modularidade (comunidades).
- **Critérios de Aceitação:**
  - [ ] Grafo importado no Gephi
  - [ ] Layout aplicado
  - [ ] Modularidade calculada

**Issue 5.3:** Identificar comunidades / clusters relevantes (hepatologia, complicações, comorbidades)
- **Descrição:** Analisar comunidades detectadas e identificar aquelas relevantes para hepatologia.
- **Critérios de Aceitação:**
  - [ ] Comunidades identificadas
  - [ ] Comunidades relevantes documentadas
  - [ ] Análise inicial realizada

---

#### M6 – Extração e Análise de Relações / Insight Clínico

**Issue 6.1:** Filtrar arestas com 'hepat' / 'fígado' / termos hepáticos
- **Descrição:** Extrair subgrafo contendo apenas entidades e relações relacionadas a hepatologia.
- **Critérios de Aceitação:**
  - [ ] Subgrafo de hepatologia extraído
  - [ ] Filtros aplicados corretamente
  - [ ] Subgrafo validado

**Issue 6.2:** Extrair subgrafo de hepatologia + gerar tabela CSV de relações + descrições
- **Descrição:** Gerar tabelas estruturadas com relações e descrições do subgrafo de hepatologia.
- **Critérios de Aceitação:**
  - [ ] Tabelas CSV geradas
  - [ ] Relações documentadas
  - [ ] Descrições incluídas

**Issue 6.3:** Comparar essas relações com evidências da literatura médica
- **Descrição:** Validar relações encontradas comparando com evidências da literatura médica.
- **Critérios de Aceitação:**
  - [ ] Comparação realizada
  - [ ] Relações validadas/invalidadas documentadas
  - [ ] Insights clínicos identificados

---

#### M7 – Escrita dos Resultados e Discussão

**Issue 7.1:** Escrever seção de resultados com estatísticas do grafo (nós, densidade, centralidade, comunidades)
- **Descrição:** Redigir seção de resultados incluindo estatísticas descritivas do grafo.
- **Critérios de Aceitação:**
  - [ ] Estatísticas calculadas
  - [ ] Seção escrita
  - [ ] Tabelas e gráficos incluídos

**Issue 7.2:** Discutir achados clínicos à luz da literatura
- **Descrição:** Discutir os achados do grafo comparando com literatura médica relevante.
- **Critérios de Aceitação:**
  - [ ] Discussão escrita
  - [ ] Literatura citada
  - [ ] Análise crítica realizada

**Issue 7.3:** Elaborar limitações e implicações para prática médica / pesquisa futura
- **Descrição:** Documentar limitações do estudo e implicações para prática e pesquisa.
- **Critérios de Aceitação:**
  - [ ] Limitações identificadas e documentadas
  - [ ] Implicações discutidas
  - [ ] Pesquisa futura sugerida

---

#### M8 – Escrita do TCC + Documentação Técnica do Código

**Issue 8.1:** Escrever capítulo de metodologia: pipeline GraphRAG + anonimização + análise
- **Descrição:** Redigir capítulo de metodologia detalhando todo o pipeline de processamento.
- **Critérios de Aceitação:**
  - [ ] Metodologia escrita
  - [ ] Pipeline documentado
  - [ ] Reproductibilidade garantida

**Issue 8.2:** Documentar scripts de processamento e visualização no repositório
- **Descrição:** Adicionar documentação técnica completa no repositório.
- **Critérios de Aceitação:**
  - [ ] README atualizado
  - [ ] Documentação de arquitetura criada
  - [ ] Guias de uso escritos

**Issue 8.3:** Incluir README com guia de reprodução + dependências (conda, Ollama, etc.)
- **Descrição:** Criar README completo com instruções de reprodução do trabalho.
- **Critérios de Aceitação:**
  - [ ] README completo
  - [ ] Dependências listadas
  - [ ] Instruções de instalação e execução

---

#### M9 – Revisão Final, Formatação e Defesa / Entrega

**Issue 9.1:** Revisar ortografia, formatação e normas ABNT / da instituição
- **Descrição:** Realizar revisão completa do texto, formatação e normas.
- **Critérios de Aceitação:**
  - [ ] Revisão ortográfica completa
  - [ ] Formatação conforme normas
  - [ ] Referências formatadas

**Issue 9.2:** Gerar PDF final + anexos de dados (se permitido) / código (GitHub)
- **Descrição:** Preparar versão final do TCC em PDF e organizar anexos.
- **Critérios de Aceitação:**
  - [ ] PDF gerado
  - [ ] Anexos organizados
  - [ ] Repositório finalizado

**Issue 9.3:** Preparar apresentação / slides / defesa
- **Descrição:** Criar apresentação para defesa do TCC.
- **Critérios de Aceitação:**
  - [ ] Slides preparados
  - [ ] Apresentação revisada
  - [ ] Material de apoio pronto

---

## Fluxo de Trabalho

### Ciclo de Desenvolvimento

```
1. Planejamento
   └── Selecionar milestone atual
       └── Criar/issues para o milestone

2. Execução
   └── Trabalhar nas issues
       └── Commits das atualizações feitas
           └── Atualizar progresso

3. Revisão
   └── Validar critérios de aceitação
       └── Fechar issues completas

4. Entrega
   └── Marcar milestone como completo
       └── Iniciar próximo milestone
```

---

## Ferramentas utilizadas

### Ferramentas Recomendadas

1. **Controle de Versão**
   - Github
   - Branches por issue de milestone
   - Tags para releases importantes

2. **Gestão de Issues**
   - GitHub Issues / GitLab Issues
   - Labels: `milestone`, `documentação`, `código`, `análise`, `bloqueado`
   - Milestones para agrupar issues

3. **Documentação**
   - Markdown para documentação
   - LaTeX para escrita do TCC
   - Diagramas: Mermaid, PlantUML, ou ferramentas visuais

4. **Análise e Visualização**
   - Gephi para visualização de grafos
   - Python para análise
   - Jupyter Notebooks para análises exploratórias

---

**Última atualização:** 2025-01-12

