from src.core.anonymization.facade import AnonymizationFeature
from src.core.anonymization.exceptions import AnonymizationError

def main():
    print(" Iniciando módulo de anonimização...")
    feature = AnonymizationFeature()

    try:
        datasets = feature.load_datasets()
    except Exception as e:
        print(f"Erro ao carregar datasets: {e}")
        return

    names = list(datasets.keys())
    print("\nDatasets disponíveis:")
    for i, name in enumerate(names, 1):
        print(f"   {i}. {name}")

    choice = input("\nEscolha o número do dataset: ")
    try:
        idx = int(choice) - 1
        selected = names[idx]
    except Exception:
        print("Escolha inválida.")
        return

    print(f"\nAnalisando '{selected}'...")
    try:
        examples = feature.validate_anonymization(selected)
        print(f"Linhas com problemas: {len(examples)}")
        if not examples.empty:
            print("\nExemplos de problemas:")
            print(examples.to_string(index=False))
        else:
            print("Nenhum problema encontrado.")

    except AnonymizationError as e:
        print(f"\nErro de anonimização: {e}")
    except Exception as e:
        print(f"\nErro inesperado: {e}")

if __name__ == '__main__':
    main()