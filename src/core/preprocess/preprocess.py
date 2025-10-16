import pandas as pd
from pathlib import Path

class GraphRAGPreprocessor:
    """
    Prepara datasets anonimizados para ingestão no GraphRAG.
    Gera arquivos de texto no formato esperado em input/texts/.
    Aceita dataframe direto ou diretório de dados.
    """

    def __init__(self, data_dir: str = None, output_dir: str = "graphrag/input", dataframe: pd.DataFrame = None):
        self.output_dir = Path(output_dir)
        self.text_dir = self.output_dir / "texts"
        self.metadata_file = self.output_dir / "metadata.csv"
        self.text_dir.mkdir(parents=True, exist_ok=True)

        if dataframe is not None:
            self.datasets = {"custom_df": dataframe}
        elif data_dir:
            from src.core.anonymization.facade import AnonymizationFeature
            self.datasets = AnonymizationFeature(data_dir=data_dir).load_datasets()
        else:
            raise ValueError("Forneça 'data_dir' ou 'dataframe' para inicializar.")

    def prepare(self):
        """
        Gera arquivos de texto para GraphRAG a partir da coluna 'descricao_anonimizada'.
        """
        records = []
        counter = 1

        for name, df in self.datasets.items():
            if "descricao_anonimizada" not in df.columns:
                print(f"⚠️ Coluna 'descricao_anonimizada' ausente em {name}, ignorando.")
                continue

            for idx, row in df.iterrows():
                text = str(row["descricao_anonimizada"]).strip()
                if not text:
                    continue

                file_name = f"doc_{counter:05d}.txt"
                file_path = self.text_dir / file_name

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)

                records.append({
                    "file_name": file_name,
                    "source_dataset": name,
                    "row_index": idx,
                    "char_count": len(text)
                })
                counter += 1

        pd.DataFrame(records).to_csv(self.metadata_file, index=False)
        print(f"Gerados {counter - 1} arquivos em {self.text_dir}")
        print(f"Metadados: {self.metadata_file}")

        return self.output_dir
