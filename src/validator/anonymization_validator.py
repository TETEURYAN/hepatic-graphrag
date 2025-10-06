import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher
from src.nlp.entity_cleaner import EntityCleaner


class AnonymizationValidator:
    """Serviço responsável por validar anonimização de datasets."""

    SENSITIVE_REGEX = [
        r"\d{2}/\d{2}/\d{4}",           # datas dd/mm/yyyy
        r"\d{3}\.\d{3}\.\d{3}-\d{2}",   # CPF
        r"\d{2}\s?\d{4,5}-\d{4}",       # telefones BR
        r"\d{6,}",                      # números longos
    ]

    def __init__(self, cleaner: EntityCleaner):
        self.cleaner = cleaner

    # -------------------------
    # 🔤 Normalização
    # -------------------------
    def normalize(self, text: str) -> str:
        """Remove acentos, pontuação e transforma em minúsculas."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = ''.join(
            c for c in unicodedata.normalize('NFKD', text)
            if not unicodedata.combining(c)
        )
        return re.sub(r'[^a-z0-9\s]', '', text)

    # -------------------------
    # 🔍 Busca fuzzy parcial (com retorno da palavra encontrada)
    # -------------------------
    def fuzzy_match(self, nome: str, texto: str, threshold: float = 0.85):
        """
        Verifica se o nome (ou parte dele) aparece no texto com variações leves.
        Retorna (True, palavra_encontrada) ou (False, None).
        """
        nome_norm = self.normalize(nome)
        texto_norm = self.normalize(texto)
        if not nome_norm or not texto_norm:
            return False, None

        tokens = nome_norm.split()
        texto_tokens = texto_norm.split()

        for token in tokens:
            if len(token) < 3:
                continue

            # Permite fragmentos espaçados (ex: M a r i a -> "M\s*a\s*r\s*i\s*a")
            spaced_pattern = r"".join([f"{ch}\\s*" for ch in token])
            match = re.search(spaced_pattern, texto_norm)
            if match:
                return True, match.group(0)

            # Busca fuzzy token a token
            for word in texto_tokens:
                if SequenceMatcher(None, token, word).ratio() > threshold:
                    return True, word

        return False, None

    # -------------------------
    # ⚠️ Padrões sensíveis
    # -------------------------
    def contains_sensitive_patterns(self, text: str) -> bool:
        """Verifica padrões como CPF, data ou telefone."""
        if not isinstance(text, str):
            return False
        return any(re.search(p, text) for p in self.SENSITIVE_REGEX)

    # -------------------------
    # 🧩 Validação principal
    # -------------------------
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retorna linhas com nome ou fragmento de nome detectado."""
        if "nome" not in df.columns or "descricao_anonimizada" not in df.columns:
            raise ValueError("Colunas 'nome' ou 'descricao_anonimizada' não encontradas.")

        df_valid = df[df["nome"].notna() & (df["nome"].str.strip() != "")]
        results = []

        for idx, row in df_valid.iterrows():
            desc_anon = str(row.get("descricao_anonimizada", ""))
            nome = str(row.get("nome", "")).strip()

            nome_presente, palavra_encontrada = self.fuzzy_match(nome, desc_anon)
            if nome_presente:
                doc = self.cleaner._nlp(desc_anon)
                ents = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PER", "LOC", "ORG"]]
                regex_flag = self.contains_sensitive_patterns(desc_anon)

                results.append({
                    "index": idx,
                    "nome_presente": True,
                    "nome_detectado": nome,
                    "palavra_encontrada": palavra_encontrada,
                    "entidades_restantes": ents,
                    "regex_sensivel": regex_flag,
                    "descricao_anonimizada": desc_anon
                })

        return pd.DataFrame(results)
