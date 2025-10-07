from src.data_loader import load_all_csvs
from src.nlp.entity_cleaner import EntityCleaner
from src.explorer.explorer import DatasetExplorer
from src.validator.anonymization_validator import AnonymizationValidator
from src.report.report_generator import ReportGenerator
import pandas as pd

def main():
    logs = []

    logs.append("-> Carregando datasets de anamnese...")
    datasets = load_all_csvs(concat=False)

    explorer = DatasetExplorer()
    dataset_names = list(datasets.keys())

    logs.append("\n-> Datasets disponíveis:")
    for i, name in enumerate(dataset_names, start=1):
        logs.append(f"   {i}. {name}")

    choice = input("\n-> Escolha o número do dataset que deseja analisar: ")

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(dataset_names):
            raise ValueError
        selected_name = dataset_names[idx]
    except ValueError:
        logs.append("! Opção inválida, saindo...")
        print("\n".join(logs))
        return

    df = datasets[selected_name]
    if isinstance(df, pd.Series):
        df = df.to_frame()

    explorer.show_info(df, selected_name)

    logs.append("\n🔍 Validando anonimização...")
    cleaner = EntityCleaner()
    validator = AnonymizationValidator(cleaner)
    problemas = validator.validate(df)

    if not problemas.empty:
        logs.append(f"\n⚠️ {len(problemas)} registros possuem nome presente na descrição anonimizadas:")
        for _, row in problemas.iterrows():
            logs.append(f"- Index {row['index']}, Nome: {row['nome_detectado']}")
            logs.append(f"  Entidades restantes: {row['entidades_restantes']}")
            logs.append(f"  Regex sensível: {row['regex_sensivel']}")
            # logs.append(f"  Descrição anonimizada: {row['descricao_anonimizada'][:200]}...")
            logs.append(f"  Palavra encontrada anonimizada: {row['palavra_encontrada']}...")
    else:
        logs.append("\n✅ Nenhum registro com nome presente detectado.")

    report_path = ReportGenerator.save_logs(logs, f"relatorio_{selected_name}")
    print("\n".join(logs))
    print(f"\n📑 Relatório salvo em: {report_path}")

if __name__ == "__main__":
    main()
