import os
import pandas as pd

class ReportGenerator:
    """Gera relatório em arquivo TXT a partir de logs ou DataFrames."""

    @staticmethod
    def save_logs(logs: list, filename: str) -> str:
        path = f"{filename}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        return os.path.abspath(path)

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filename: str) -> str:
        path = f"{filename}.txt"
        df.to_csv(path, sep="\t", index=False)
        return os.path.abspath(path)
