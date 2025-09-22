import pandas as pd
from pathlib import Path

def load_all_csvs(data_dir: str = "data", concat: bool = False, normalize_cols: bool = True):
    """
    Lê todos os arquivos CSV dentro do diretório especificado.
    Detecta separador automaticamente e tenta alinhar colunas.
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {data_dir}")

    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {data_dir}")

    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.replace(" ", "_")
            .str.replace(r"[^\w_]", "", regex=True)
            .str.lower()
        )
        return df

    def _safe_read(csv: Path) -> pd.DataFrame:
        # tenta detectar separador automaticamente
        try:
            df = pd.read_csv(csv, sep=None, engine="python", encoding="utf-8", header=0)
        except Exception as e:
            print(f"⚠️ Erro ao ler {csv.name}: {e}")
            return None

        # se só tiver 1 coluna, pode ser que o separador real seja ";"
        if df.shape[1] == 1:
            df = pd.read_csv(csv, sep=";", engine="python", encoding="utf-8", header=0)

        return df

    dfs = []
    results = {}

    for csv in csv_files:
        df = _safe_read(csv)
        if df is None:
            continue

        if normalize_cols:
            df = _normalize_columns(df)

        if concat:
            dfs.append(df)
        else:
            results[csv.name] = df

    if concat:
        return pd.concat(dfs, ignore_index=True)
    return results
