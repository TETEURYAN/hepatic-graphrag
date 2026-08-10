# Anonymization Feature - Projeto de Validação de Dados Sensíveis

Este projeto é um módulo de **anonimização e validação de datasets**, projetado para ser integrado como uma **feature** do sistema de anonimização do `GraphRAG`. Ele permite carregar dados de anamnese, validar se a anonimização está correta e gerar relatórios detalhados.

---

## Funcionalidades

1. **Carregamento de datasets**
   - Suporte a múltiplos arquivos CSV.
   - Normalização automática de colunas.
   - Detecção automática de separadores (`,` ou `;`).
   
2. **Exploração de datasets**
   - Exibição de informações básicas: colunas, número de registros.
   - Amostras de dados para análise rápida.

3. **Anonimização de dados**
   - Remoção de nomes próprios utilizando NLP com SpaCy (`pt_core_news_sm`).
   - Validação se a coluna `descricao_anonimizada` contém o padrão `<nome anonimizado>`.

4. **Validação de anonimização**
   - Verificação de linhas inválidas.
   - Exibição de exemplos problemáticos.
   - Detecção de padrões sensíveis (CPF, telefone, datas).

5. **Geração de relatórios**
   - Exporta relatórios TXT ou CSV tabulado com resultados da validação.

6. **Arquitetura modular**
   - Implementado usando **Facade Pattern** (`AnonymizationFeature`).
   - Tratamento de exceções especializadas: `AnonymizationError`, `EmptyDatasetError`.
   - Classes separadas por responsabilidade (SRP).

---

## 🔹 Estrutura do Projeto
``` bash
src/
└─ core/
└── anonymization/
├── facade.py # Fachada principal da feature
├── exceptions.py # Exceções customizadas
├── loader/
│ └── data_loader.py # Carregamento e normalização de CSVs
├── validator/
│ ├── description_matcher.py # Validação de descrições
│ └── anonymization_validator.py # Validação detalhada de anonimização
├── nlp/
│ └── entity_cleaner.py # Limpeza de nomes próprios
├── explorer/
│ └── explorer.py # Exibição de dados e amostras
└── report/
└── report_generator.py # Geração de relatórios TXT/CSV
```

---

## Dependências

- Python >= 3.10
- Pandas
- SpaCy
- Modelo SpaCy: `pt_core_news_sm`

## Caso de uso 

``` python

from src.core.anonymization.facade import AnonymizationFeature
from src.core.anonymization.exceptions import AnonymizationError, EmptyDatasetError

def main():
    feature = AnonymizationFeature(data_dir="data")
    
    # Carrega datasets
    datasets = feature.load_datasets()
    for i, name in enumerate(datasets.keys(), 1):
        print(f"{i}. {name}")

    choice = int(input("Escolha o número do dataset: "))
    dataset_name = list(datasets.keys())[choice - 1]

    try:
        mismatches, examples = feature.validate_descriptions(dataset_name)
        print(f"Total de linhas inválidas: {len(mismatches)}")
        if not examples.empty:
            print("Exemplos de problemas encontrados:")
            print(examples.to_string(index=True))
        else:
            print("Nenhum problema encontrado nas colunas de descrição.")
    except (AnonymizationError, EmptyDatasetError) as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()

```

