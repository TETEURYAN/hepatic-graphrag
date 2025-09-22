import pandas as pd
from typing import List

class DescriptionMatcher:
    """
    Classe responsável por validar se colunas de descrição estão devidamente anonimizadas.
    Uma linha é inválida se 'descricao_anonimizada' NÃO contiver o nome anonimizado.
    """

    def find_mismatches(self, df: pd.DataFrame) -> List[int]:
        """
        Retorna os índices das linhas em que 'descricao_anonimizada' não contém
        o nome anonimizado indicado em 'descricao'.
        """
        if "descricao" not in df.columns or "descricao_anonimizada" not in df.columns:
            raise ValueError("Colunas 'descricao' e 'descricao_anonimizada' são obrigatórias.")

        def is_invalid(row):
            anon = str(row["descricao_anonimizada"]) if pd.notna(row["descricao_anonimizada"]) else ""
            return "<nome anonimizado>" not in anon

        mask_invalid = df.apply(is_invalid, axis=1)
        mismatches = df.index[mask_invalid].tolist()
        return mismatches

    def sample_mismatches(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Retorna até 'n' exemplos das linhas problemáticas,
        mostrando as duas colunas lado a lado.
        """
        mismatches = self.find_mismatches(df)
        if not mismatches:
            return pd.DataFrame(columns=["descricao", "descricao_anonimizada"])

        return df.loc[mismatches, ["descricao", "descricao_anonimizada"]].head(n)
