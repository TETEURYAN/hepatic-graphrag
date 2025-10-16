# Anonimização e Análise Longitudinal de Prontuários Eletrônicos | GraphRag

<p align="center">
  <img src="https://user-images.githubusercontent.com/91018438/204195385-acc6fcd4-05a7-4f25-87d1-cb7d5cc5c852.png" alt="animated" />
</p>

Este projeto implementa um sistema completo para **anonimização de prontuários eletrônicos** e **análise longitudinal de evoluções clínicas**, integrando um pipeline de pré-processamento compatível com o **GraphRAG**.

O sistema foi projetado para garantir a **privacidade de dados sensíveis** em contextos clínicos, permitindo a **construção de grafos de conhecimento** sobre textos anonimizados, com o objetivo de estudar padrões e trajetórias clínicas ao longo do tempo.

---

## Núcleos Principais 

### 1. Núcleo de Anonimização (`src/core/anonymization`)
Responsável por todo o fluxo de **carregamento, limpeza e validação** dos dados clínicos.

#### Etapas realizadas:
1. **Leitura dos datasets** em formato `.csv` (pasta `data/`);
2. **Normalização das colunas** e detecção automática de separadores;
3. **Remoção de nomes próprios** e entidades sensíveis com o modelo spaCy (`pt_core_news_sm`);
4. **Validação da anonimização**, verificando:
   - presença residual de nomes ou entidades (`PER`, `LOC`, `ORG`);
   - padrões sensíveis (datas, CPFs, telefones);
   - consistência entre colunas `descricao` e `descricao_anonimizada`.

#### Resultados:
- Relatórios `.txt` com inconsistências e exemplos de falhas de anonimização;
- Log estruturado de entidades detectadas e padrões sensíveis.

---

### 2. Núcleo de Pré-Processamento para GraphRAG (`src/core/preprocess`)
Após a anonimização, este núcleo transforma os textos clínicos em **entradas compatíveis com o GraphRAG**, permitindo análise contextual via grafos semânticos.

#### Funções principais:
- Geração de diretórios `input/` com os textos anonimizados;
- Criação da estrutura esperada pelo **indexador do GraphRAG**, incluindo:
  - `input/texts/*.txt` → textos clínicos processados;
  - `settings.yaml` → configurações do pipeline;
  - `prompts/` → templates de extração (podem ser adaptados por domínio).

#### Execução do indexador:
```bash
graphrag index --root graphrag
```

Após a indexação, o grafo resultante poderá ser consultado via:
```bash
graphrag query --root graphrag
```

---

## 🧱 Estrutura do Projeto

```
hepatic-graphrag/
├── data/                     # Datasets clínicos em csv
├── input/                    # Textos prontos para indexação no GraphRAG
├── src/
│   └── core/
│       ├── anonymization/    # Núcleo de anonimização e validação
│       │   ├── facade.py
│       │   ├── validator/
│       │   ├── nlp/
│       │   ├── explorer/
│       │   ├── loader/
│       │   └── report/
│       └── preprocess/       # Núcleo de pré-processamento dos inputs
├── main.py                   # Interface de linha de comando
├── Makefile                  # Automação com uv
├── pyproject.toml            # Configuração do ambiente Python
└── README.md
```

---

## Como Executar o Sistema

O projeto utiliza o **[uv](https://github.com/astral-sh/uv)** como gerenciador de ambientes.  
Você pode configurar e rodar tudo automaticamente via `Makefile`.

### Instalação

1. Crie e ative o ambiente virtual:
```bash
make setup
```

2. Atualize as variáveis
```bash
uv venv
uv pip install -r pyproject.toml
```

### ▶Execução

Para rodar o sistema interativo:
```bash
make run
```

ou manualmente:
```bash
uv run python main.py
```

### Limpeza
Remove o ambiente e caches temporários:
```bash
make clean
```

---

## Fluxo de Uso

1. **O dataset deverá estar** na pasta `data/`.
2. Execute o sistema:
   ```bash
   make run
   ```
3. Escolha:
   - **[1]** Validar anonimização;
   - **[2]** Gerar inputs compatíveis com GraphRAG.
4. Verifique os resultados:
   - Arquivos `.txt` com relatórios em `output/`;
   - Estrutura `input/` pronta para o comando `graphrag index`.

---

## Tecnologias e Dependências

- **Python 3.10+**
- **uv** — ambiente e instalação de dependências
- **spaCy** com modelo `pt_core_news_sm`
- **pandas** para manipulação de dados tabulares
- **GraphRAG** (para indexação e consultas em grafos semânticos)

---

## Objetivo Científico

Este projeto faz parte de um estudo sobre **PLN à saúde**, com foco em:
- Avaliar o progresso clínico longitudinal;
- Analisar padrões de evolução textual em prontuários;
- Correlacionar trajetórias de pacientes por meio de **grafos de conhecimento**.

---

## Licença
Distribuído sob a licença MIT.  
Sinta-se livre para adaptar, expandir e integrar o sistema em fluxos de pesquisa ou aplicações clínicas.
