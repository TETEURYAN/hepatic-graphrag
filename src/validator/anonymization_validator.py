import pandas as pd
import re
from src.nlp.entity_cleaner import EntityCleaner

class AnonymizationValidator:
    """Serviço responsável por validar anonimização de datasets."""

    SENSITIVE_REGEX = [
        r"\d{2}/\d{2}/\d{4}",           # datas dd/mm/yyyy
        r"\d{3}\.\d{3}\.\d{3}-\d{2}",   # CPF
        r"\d{2}\s?\d{4,5}-\d{4}",       # telefones BR
        r"\d{6,}",                       # números longos
    ]

    def __init__(self, cleaner: EntityCleaner):
        self.cleaner = cleaner

    def contains_sensitive_patterns(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        return any(re.search(p, text) for p in self.SENSITIVE_REGEX)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retorna apenas linhas com nome_presente=True, desconsiderando nomes vazios."""
        if "nome" not in df.columns or "descricao_anonimizada" not in df.columns:
            raise ValueError("Colunas 'nome' ou 'descricao_anonimizada' não encontradas.")

        df_valid = df[df["nome"].notna() & (df["nome"].str.strip() != "")]
        results = []

        for idx, row in df_valid.iterrows():
            desc_anon = str(row.get("descricao_anonimizada", ""))
            nome = str(row.get("nome", "")).strip()
            nome_presente = nome.lower() in desc_anon.lower() if nome else False
            if nome_presente:
                doc = self.cleaner._nlp(desc_anon)
                ents = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PER", "LOC", "ORG"]]
                regex_flag = self.contains_sensitive_patterns(desc_anon)
                results.append({
                    "index": idx,
                    "nome_presente": nome_presente,
                    "nome_detectado": nome,
                    "entidades_restantes": ents,
                    "regex_sensivel": regex_flag,
                    "descricao_anonimizada": desc_anon
                })
        return pd.DataFrame(results)
