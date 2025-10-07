import pandas as pd

class DatasetExplorer:
    """Serviço para explorar e exibir informações de datasets."""

    def show_samples(self, df: pd.DataFrame, column: str, n: int = 5, label: str = ""):
        """Exibe amostras não nulas de uma coluna."""
        if column not in df.columns:
            print(f"Coluna '{column}' não existe no DataFrame.")
            return

        samples = df[column].dropna().head(n).to_list()
        if label:
            print(label)
        for i, s in enumerate(samples, 1):
            print(f"   {i}. {s}")

    def show_info(self, df: pd.DataFrame, name: str):
        """Exibe informações básicas do dataset."""
        print(f"\n📂 Dataset: {name}")
        print(f"   Registros: {len(df)}")
        print(f"   Colunas: {list(df.columns)}")
