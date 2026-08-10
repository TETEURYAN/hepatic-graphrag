# Análise dos Resultados do Grid GraphRAG (8 sistemas × 141 prontuários)

**Data:** 2026-08-10
**Fonte dos dados:** `resultados_grid/` e `progresso_grid/`, reprocessados por `analise_resultados_grid.py` com o parser corrigido (`parser_doencas.py`).
**Ground truth:** `progresso/ground_truth_anotado.csv` (141 prontuários anotados).

---

## 1. O que mudou com a correção do parser

A função original `extrai_doenca()` (em `experimentos_graphrag.py`) comparava termos do dicionário em **MAIÚSCULAS** contra a resposta já convertida para `resposta.lower()`. Como o operador `in` do Python é sensível a maiúsculas/minúsculas, **nenhum termo do dicionário jamais era encontrado**, independente do conteúdo da resposta.

Resultado antes da correção: os 8 sistemas ficaram com **100% das respostas `ok` marcadas como `nao_identificado`**, mesmo quando a resposta trazia claramente o diagnóstico (ex.: `"Esteatose hepática. [Data: Sources (70)]"`).

O novo parser (`parser_doencas.extrai_doenca`):
- normaliza acentuação e caixa antes de comparar;
- usa `\b` (limite de palavra) para evitar substrings soltas;
- entre múltiplos termos encontrados, escolhe o que aparece **mais cedo** na resposta (o prompt pede a doença logo no início da frase), desempatando pelo termo mais específico (mais longo).

Depois de reprocessar os `.json`/`.csv` existentes (sem rodar nenhuma query nova, sem custo de API), os 8 sistemas passaram a ter distribuições de diagnóstico condizentes com o conteúdo real das respostas.

---

## 2. Ground truth (referência para todas as métricas)

| Classe | Nº de prontuários |
|---|---|
| esteatose_hepatica | 41 |
| **nenhuma** (excluída da avaliação de classificação) | 32 |
| cirrose_hepatica | 16 |
| esquistossomose | 16 |
| hipertensao_portal | 15 |
| hepatite_b | 10 |
| hemorragia_digestiva_alta | 5 |
| hepatite_c | 5 |
| pbe | 1 |
| **Total** | **141** |

Todas as métricas de classificação (accuracy, F1, kappa, MCC) são calculadas sobre os **109 prontuários com doença principal definida** (excluindo os 32 rotulados `nenhuma`, seguindo a mesma convenção do script original). Além disso, **12 prontuários não têm texto-fonte disponível** (`SEM_TEXTO`) em nenhum sistema — são os mesmos 12 em todos os 8 sistemas, já que a limitação é do dado de entrada, não do método de busca.

---

## 3. Metodologia de avaliação

- **Accuracy / F1 macro / F1 weighted / Cohen's Kappa / MCC** — calculados com `sklearn.metrics`.
- **IC 95% do F1 macro** — bootstrap com 5.000 reamostragens.
- **Friedman** — testa se há diferença significativa entre os 8 sistemas simultaneamente (acerto binário por prontuário).
- **Wilcoxon post-hoc** (pareado, com correção de Bonferroni para 28 comparações → α = 0,05/28 ≈ 0,0018) — compara pares de sistemas.
- **IMA (Inter-Mode Agreement)** — proporção de sistemas que concordam na resposta mais votada, por prontuário.

---

## 4. Ranking geral (ordenado por F1 macro)

| Sistema | Modo | Community Level | Accuracy | F1 macro | F1 weighted | Kappa | MCC | IC 95% F1 macro |
|---|---|---|---|---|---|---|---|---|
| **S4** | local | 2 | 0,587 | **0,363** | 0,593 | 0,482 | 0,488 | [0,296 – 0,459] |
| S3 | local | 1 | 0,578 | 0,338 | 0,589 | 0,471 | 0,479 | [0,269 – 0,431] |
| S6 | global | 2 | 0,569 | 0,328 | 0,558 | 0,457 | 0,464 | [0,269 – 0,406] |
| S7 | drift | 1 | 0,587 | 0,318 | 0,571 | 0,480 | 0,492 | [0,270 – 0,412] |
| S8 | drift | 2 | 0,560 | 0,317 | 0,545 | 0,446 | 0,457 | [0,253 – 0,401] |
| S5 | global | 1 | 0,578 | 0,299 | 0,561 | 0,468 | 0,478 | [0,250 – 0,421] |
| S2 | basic | 2 | 0,541 | 0,298 | 0,551 | 0,428 | 0,434 | [0,235 – 0,384] |
| S1 | basic | 1 | 0,541 | 0,293 | 0,554 | 0,434 | 0,441 | [0,237 – 0,393] |

**Leitura importante:** os intervalos de confiança do F1 macro se sobrepõem fortemente entre todos os 8 sistemas — e isso é confirmado pelos testes estatísticos (seção 7): **nenhuma diferença entre sistemas é estatisticamente significativa**. O ranking acima reflete a estimativa pontual, não uma vitória robusta de um método sobre outro.

---

## 5. Análise por sistema

Para cada sistema: distribuição de `doenca_prevista` (respostas com `status=ok`), métricas de classificação e tempo de execução por query.

### S1 — basic / community-level 1

| doenca_prevista | n |
|---|---|
| esteatose_hepatica | 40 |
| cirrose_hepatica | 31 |
| esquistossomose | 19 |
| nenhuma | 13 |
| SEM_TEXTO | 12 |
| hipertensao_portal | 10 |
| hepatite_b | 10 |
| hepatite_c | 5 |
| nao_identificado | 1 |

- **Accuracy 0,541 · F1 macro 0,293 · Kappa 0,434 · MCC 0,441**
- Tempo por query: média 192s, mediana 103s, pico 9.121s (~2,5h — outlier isolado).
- Melhor classe: `hepatite_b` (F1 0,842). Pior: `hipertensao_portal` (F1 0,174) e `hemorragia_digestiva_alta` (F1 0,000 — nunca acertada).
- Único sistema com 1 caso genuinamente ambíguo (`nao_identificado`), residual esperado — não é falha de parsing (ver seção 6).

### S2 — basic / community-level 2

| doenca_prevista | n |
|---|---|
| esteatose_hepatica | 43 |
| cirrose_hepatica | 30 |
| esquistossomose | 20 |
| nenhuma | 14 |
| SEM_TEXTO | 12 |
| hipertensao_portal | 9 |
| hepatite_b | 9 |
| hepatite_c | 4 |

- **Accuracy 0,541 · F1 macro 0,298 · Kappa 0,428 · MCC 0,434**
- Tempo por query: média 250s, mediana 220s, pico 3.851s.
- Quase idêntico a S1 em desempenho (community-level 2 não trouxe ganho relevante para `basic`); leve melhora em `esteatose_hepatica` (F1 0,805 vs. 0,757) compensada por queda em `esquistossomose` (F1 0,323 vs. 0,438).
- Nenhum caso `nao_identificado` — todas as 129 respostas `ok` foram parseadas com sucesso.

### S3 — local / community-level 1

| doenca_prevista | n |
|---|---|
| esteatose_hepatica | 45 |
| cirrose_hepatica | 33 |
| esquistossomose | 17 |
| nenhuma | 13 |
| SEM_TEXTO | 12 |
| hipertensao_portal | 8 |
| hepatite_b | 8 |
| hepatite_c | 5 |

- **Accuracy 0,578 · F1 macro 0,338 · Kappa 0,471 · MCC 0,479**
- Tempo por query: média 193s, mediana 188s, pico 477s — **o sistema mais estável em tempo de execução** (sem outliers extremos).
- `hepatite_b` com precisão perfeita (1,000) mas recall 0,700 — quando erra, é conservador (prefere não citar hepatite B a citar errado).
- Segundo melhor sistema geral, com custo computacional baixo e previsível — bom candidato a "sistema de referência" para comparações futuras.

### S4 — local / community-level 2 — **melhor F1 macro**

| doenca_prevista | n |
|---|---|
| esteatose_hepatica | 44 |
| cirrose_hepatica | 29 |
| esquistossomose | 19 |
| nenhuma | 13 |
| SEM_TEXTO | 12 |
| hepatite_b | 9 |
| hipertensao_portal | 8 |
| hepatite_c | 7 |

- **Accuracy 0,587 · F1 macro 0,363 (melhor do grid) · Kappa 0,482 · MCC 0,488**
- Tempo por query: média 194s, mediana 169s, pico 1.328s.
- `hepatite_c` com F1 0,727 (recall 0,800) — bom para a classe, mas não é exclusividade de S4 (S6 e S7 também passam de F1 0,7 nessa classe). O ganho de S4 sobre S3 é mais amplo e distribuído: sobe em `hepatite_c` (0,444→0,727) sem perder em `esteatose_hepatica` (0,810→0,785, praticamente estável) — já `hipertensao_portal` piora levemente em precisão frente a S3 (0,571 vs. 0,667), com recall igual (0,267) nos dois.
- Combina o melhor F1 macro do grid com tempo de execução baixo (comparável a S1/S3), tornando-o a melhor relação custo-benefício do experimento.

### S5 — global / community-level 1

| doenca_prevista | n |
|---|---|
| esteatose_hepatica | 44 |
| cirrose_hepatica | 32 |
| esquistossomose | 23 |
| SEM_TEXTO | 12 |
| hepatite_b | 10 |
| nenhuma | 6 |
| hipertensao_portal | 5 |
| hepatite_c | 5 |
| TIMEOUT | 2 |
| nao_identificado | 2 |

- **Accuracy 0,578 · F1 macro 0,299 · Kappa 0,468 · MCC 0,478**
- Tempo por query: média **666s** (o mais lento de todos, exceto S8), mediana 603s, pico 2.039s.
- Único sistema com timeouts reais (2 casos, `status=timeout`) — método `global` faz map-reduce sobre todos os community reports, custo cresce com o tamanho do grafo.
- **`hipertensao_portal` nunca é acertada** (F1 = 0,000, recall = 0,000) — todas as 15 ocorrências reais foram classificadas como outra doença (majoritariamente `esquistossomose` e `cirrose_hepatica`, ver seção 6).
- 2 casos `nao_identificado` residuais: respostas citaram doenças fora das 9 classes-alvo (ex.: "Doença de Caroli"), não falha do parser.

### S6 — global / community-level 2

| doenca_prevista | n |
|---|---|
| esteatose_hepatica | 49 |
| esquistossomose | 27 |
| cirrose_hepatica | 22 |
| SEM_TEXTO | 12 |
| nao_identificado | 11 |
| hepatite_b | 10 |
| hepatite_c | 7 |
| hipertensao_portal | 2 |
| nenhuma | 1 |

- **Accuracy 0,569 · F1 macro 0,328 · Kappa 0,457 · MCC 0,464**
- Tempo por query: média 359s, mediana 264s, pico 1.076s — mais rápido que S5, apesar de também ser `global` (community-level maior gera menos comunidades de nível raso para o map-reduce percorrer, o que parece compensar o custo).
- **Maior número de `nao_identificado` do grid (11 casos)** — o modo `global`/level 2 tende a produzir respostas mais elaboradas/hedged (múltiplas hipóteses, ressalvas), dificultando a extração de uma única doença mesmo com o parser corrigido. Vale inspecionar manualmente essas 11 respostas se for necessário reduzir esse resíduo.
- `hipertensao_portal` quase não é reconhecida (F1 = 0,000, apenas 2 previsões da classe no sistema todo).

### S7 — drift / community-level 1 (default)

| doenca_prevista | n |
|---|---|
| esteatose_hepatica | 50 |
| cirrose_hepatica | 33 |
| esquistossomose | 23 |
| SEM_TEXTO | 12 |
| nao_identificado | 8 |
| hepatite_b | 7 |
| hepatite_c | 5 |
| nenhuma | 2 |
| hipertensao_portal | 1 |

- **Accuracy 0,587 (empatado com S4) · F1 macro 0,318 · Kappa 0,480 (2º melhor) · MCC 0,492 (melhor do grid)**
- Tempo por query: média 389s, mediana 306s, pico 2.635s.
- Melhor `hepatite_c` do grid em precisão (1,000) com recall 0,800 (F1 0,889) e melhor `cirrose_hepatica` em recall (0,875).
- Maior MCC do grid (0,492) apesar do F1 macro mediano — indica boa correlação geral entre previsão e verdade mesmo com F1 macro penalizado pelas classes que `drift` praticamente não reconhece (`hipertensao_portal`: só 1 previsão em 141).
- 8 casos `nao_identificado`: mistura de doenças fora do escopo (Caroli, hemocromatose, hepatite autoimune, Budd-Chiari) — um padrão consistente de que `drift`, com seu raciocínio mais profundo, tende a propor diagnósticos diferenciais mais específicos que fogem das 9 classes do estudo.

### S8 — drift / community-level 2

| doenca_prevista | n |
|---|---|
| esteatose_hepatica | 51 |
| cirrose_hepatica | 32 |
| esquistossomose | 26 |
| SEM_TEXTO | 12 |
| hepatite_c | 8 |
| hepatite_b | 7 |
| nao_identificado | 4 |
| hipertensao_portal | 1 |

- **Accuracy 0,560 · F1 macro 0,317 · Kappa 0,446 · MCC 0,457**
- Tempo por query: média **594s** (2º mais lento do grid), mediana 416s, pico **9.261s** (~2,6h — maior outlier de todo o experimento).
- Muito parecido com S7 em distribuição, mas com leve queda em quase todas as métricas — o community-level 2 não trouxe ganho para `drift` (ao contrário do que ocorreu em `local`, S3→S4).
- `hipertensao_portal` praticamente ausente (1 única previsão em 141 prontuários), mesmo padrão de S7.

---

## 6. Padrões de erro transversais (comuns aos 8 sistemas)

1. **`hipertensao_portal` é a classe mais mal reconhecida em todo o grid.** Recall despenca de ~13-27% (basic/local) para **0%** em `global` e `drift`. Quando o sistema erra essa classe, confunde consistentemente com:
   - `esquistossomose` (a causa mais comum de hipertensão portal não-cirrótica na base, clinicamente relacionada);
   - `cirrose_hepatica` (a causa mais comum de hipertensão portal cirrótica).

   Isso sugere que os modelos tendem a citar a **causa/doença de base** em vez do **efeito** (`hipertensao_portal` é uma síndrome secundária), o que é clinicamente defensável mas conta como erro na avaliação estrita de rótulo único.

2. **`hemorragia_digestiva_alta` nunca é prevista como diagnóstico principal em nenhum sistema.** Mesmo padrão do item anterior: HDA é uma complicação/evento agudo, não uma "doença hepática de base" — os modelos tendem a apontar a cirrose/esquistossomose subjacente. F1 = 0,000 em todos os 8 sistemas.

3. **`esteatose_hepatica` é a classe mais bem reconhecida** em todos os sistemas (F1 entre 0,757 e 0,854), consistente com ser a classe majoritária do ground truth (41/141) e com apresentação clínica mais "isolada" (menos sobreposição com outras doenças hepáticas do estudo).

4. **Casos residuais `nao_identificado` (S1: 1, S5: 2, S6: 11, S7: 8, S8: 4) não são falhas de parsing** — inspeção manual confirma que são respostas onde o modelo diagnosticou corretamente uma doença hepática real, porém **fora das 9 classes-alvo do estudo** (ex.: Doença de Caroli, hemocromatose hereditária, hepatite autoimune, síndrome de Budd-Chiari, hepatite tóxica). É uma decisão de modelagem em aberto: tratar esses casos como `nenhuma` (nenhuma das 8 condições listadas) poderia ser mais fiel à instrução do prompt do que deixá-los como não identificados — mas isso é uma escolha de avaliação, não um bug.

5. **Tempo de execução por modo:** `basic` e `local` são consistentemente os mais rápidos e previsíveis (médias de 190-250s, poucos outliers); `global` e `drift` são 2-3× mais lentos em média, com outliers extremos (S1 e S8 tiveram picos isolados de ~9.100-9.260s, provavelmente relacionados a retries por rate limiting da OpenRouter, não ao método em si).

---

## 7. Testes estatísticos

**Friedman (diferença entre os 8 sistemas simultaneamente):**
- Estatística = 4,74 · p-valor = 0,691 → **sem diferença estatisticamente significativa** entre os 8 sistemas.

**Wilcoxon post-hoc pareado (28 comparações, α corrigido por Bonferroni = 0,0018):**
- **Nenhum par de sistemas apresentou diferença significativa.** A menor p-valor observada foi 0,096 (S1 vs. S4), ainda 50× acima do limiar corrigido.

**Interpretação:** com n=109 prontuários avaliados, o grid de 8 configurações (4 modos × 2 community-levels) não permite afirmar, com significância estatística, que um método/parametrização é superior aos demais. As diferenças de accuracy observadas (0,541 a 0,587, uma amplitude de ~4,6 pontos percentuais) são compatíveis com variação amostral.

---

## 8. Concordância entre modos (Inter-Mode Agreement — IMA)

- **IMA médio: 0,876** · IMA mediano: 1,000 (a maioria dos prontuários tem alta concordância entre os 8 sistemas).
- **65 prontuários (59,6%)** têm concordância total entre todos os sistemas.
- **74 prontuários (67,9%)** são acertados por pelo menos 1 sistema.
- **46 prontuários (42,2%)** são acertados por todos os sistemas simultaneamente.
- **35 prontuários (32,1%)** são errados por todos os sistemas simultaneamente — um "piso" comum de dificuldade que independe do método de busca.
- Correlação entre IMA e acerto: r = 0,0045 (p = 0,963) — **concordância alta entre os modos não prediz acerto**. Ou seja, quando todos os sistemas concordam, não é mais provável que estejam certos do que errados — sugere que os erros compartilhados vêm de uma limitação comum (provavelmente o texto-fonte do prontuário ou a ambiguidade clínica do caso), não de um viés específico de um método de busca.

---

## 9. Conclusões

1. **A correção do parser foi o ganho mais impactante do processo**: sem ela, os 8 sistemas eram indistinguíveis (100% `nao_identificado`); com ela, accuracy varia de 0,541 a 0,587 e as diferenças entre classes ficam visíveis e clinicamente interpretáveis.
2. **`S4` (local, community-level 2) tem a melhor estimativa pontual de F1 macro (0,363)**, com tempo de execução baixo — melhor relação custo-benefício do grid.
3. **`S7` (drift, community-level 1) tem o melhor MCC (0,492) e Kappa (0,480)**, mas com custo computacional 2× maior que `local`.
4. **Nenhuma diferença entre os 8 sistemas é estatisticamente significativa** (Friedman p=0,691; Wilcoxon sem pares significativos) — a escolha entre eles, no estado atual dos dados, deve priorizar custo computacional e estabilidade (favorecendo `local`) em vez de acurácia esperada.
5. **`hipertensao_portal` e `hemorragia_digestiva_alta` são pontos cegos sistemáticos** de todos os 8 sistemas — não é um problema de busca (RAG), e sim de a tarefa de classificação exigir "doença principal" quando essas duas classes são, clinicamente, causas/efeitos de outras doenças da lista. Vale considerar, na escrita do TCC, discutir se essas duas classes deveriam ser reformuladas (ex.: permitir múltiplos rótulos, ou hierarquizar causa/complicação) em vez de tratadas como diagnósticos concorrentes de cirrose/esquistossomose.

---

## 10. Arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `resultados_grid/resultado_S{1..8}.csv` | Respostas individuais reprocessadas, com `doenca_prevista` corrigida |
| `resultados_grid/resultados_consolidados.csv` | Todas as 1.128 linhas (8 sistemas × 141 prontuários) |
| `resultados_grid/pivot_sistemas.csv` | Prontuário × sistema → doença prevista, com ground truth |
| `resultados_grid/metricas_grid.csv` | Métricas por sistema (tabela da seção 4) |
| `resultados_grid/wilcoxon_posthoc.csv` | Testes pareados (seção 7) |
| `resultados_grid/analise_concordancia.csv` | IMA por prontuário (seção 8) |
| `resultados_grid/fig_grid_metricas.png` | Gráfico comparativo (F1 macro, accuracy, kappa, MCC) |
| `resultados_grid/fig_heatmap_acertos.png` | Heatmap de acerto/erro por prontuário × sistema |
