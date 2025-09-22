from src.data_loader import load_all_csvs
from src.nlp.entity_cleaner import EntityCleaner
from src.explorer.explorer import DatasetExplorer
from src.validator.description_matcher import DescriptionMatcher
import pandas as pd


def main():
    print("-> Carregando datasets de anamnese...")
    datasets = load_all_csvs(concat=False)

    explorer = DatasetExplorer()

    dataset_names = list(datasets.keys())
    print("\n-> Datasets disponíveis:")
    for i, name in enumerate(dataset_names, start=1):
        print(f"   {i}. {name}")

    choice = input("\n-> Escolha o número do dataset que deseja analisar: ")

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(dataset_names):
            raise ValueError
        selected_name = dataset_names[idx]
    except ValueError:
        print("⚠️ Opção inválida, saindo...")
        return

    df = datasets[selected_name]

    if isinstance(df, pd.Series):
        df = df.to_frame()

    print(f"\n📂 Analisando dataset: {selected_name}")
    print(f"   Registros: {len(df)}")
    print(f"   Colunas: {list(df.columns)}")

    if df.empty:
        print("⚠️ Dataset vazio.")
        return

    matcher = DescriptionMatcher()
    try:
        mismatches = matcher.find_mismatches(df)
    except ValueError as e:
        print(f"⚠️ Erro: {e}")
        return

    num_invalid = len(mismatches)

    if num_invalid > 0:
        print(f"\n❌ Número de linhas inválidas: {num_invalid}")
        print("Linhas com problemas de match:")
        print(mismatches)

        print("\n-> Exemplos de problemas encontrados:")
        examples = matcher.sample_mismatches(df, n=5)
        print(examples.to_string(index=True))
    else:
        print("\n✅ Todas as linhas possuem match válido nas colunas.")


if __name__ == "__main__":
    main()
