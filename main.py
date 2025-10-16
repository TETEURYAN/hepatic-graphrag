from src.core.anonymization.facade import AnonymizationFeature
from src.core.anonymization.exceptions import AnonymizationError
from src.core.preprocess.preprocess import GraphRAGPreprocessor

def main():
    print("=== Sistema de Anonimização de Prontuários Eletrônicos ===")

    feature = AnonymizationFeature()
    try:
        datasets = feature.load_datasets()
    except Exception as e:
        print(f"Erro ao carregar datasets: {e}")
        return

    names = list(datasets.keys())
    if not names:
        print("Nenhum dataset disponível. Encerrando.")
        return

    # Seleção do dataset
    print("\nDatasets disponíveis:")
    for i, name in enumerate(names, 1):
        print(f"   {i}. {name}")

    choice = input("\nEscolha o número do dataset: ").strip()
    try:
        idx = int(choice) - 1
        selected = names[idx]
        df = datasets[selected]
    except Exception:
        print("Escolha inválida. Encerrando.")
        return

    print("\nServiços disponíveis para o dataset selecionado:")
    print("1. Executar anonimização e validação")
    print("2. Gerar inputs para o GraphRAG")
    service_choice = input("Escolha uma opção (1 ou 2): ").strip()

    if service_choice == "1":
        run_anonymization(feature, selected)
    elif service_choice == "2":
        run_preprocess(df)
    else:
        print("Opção inválida. Encerrando.")

def run_anonymization(feature: AnonymizationFeature, dataset_name: str):
    print(f"\nAnalisando anonimização do dataset '{dataset_name}'...")
    try:
        examples = feature.validate_anonymization(dataset_name)
        print(f"\nLinhas com possíveis problemas: {len(examples)}")
        if not examples.empty:
            print("\nExemplos de problemas detectados:")
            print(examples.to_string(index=False))
        else:
            print("Nenhum problema encontrado.")
    except AnonymizationError as e:
        print(f"\nErro de anonimização: {e}")
    except Exception as e:
        print(f"\nErro inesperado: {e}")

def run_preprocess(df):
    print("\nIniciando pré-processamento para GraphRAG...")
    try:
        pre = GraphRAGPreprocessor(dataframe=df, output_dir="input")
        output_dir = pre.prepare()
        print(f"\nDados prontos para o GraphRAG em: {output_dir}")
        print("Para indexar, execute:")
        print("   graphrag index --root graphrag")
    except Exception as e:
        print(f"Erro durante o pré-processamento: {e}")

if __name__ == "__main__":
    main()
