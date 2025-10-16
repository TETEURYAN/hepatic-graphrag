import spacy
import pandas as pd
from typing import List


class EntityCleaner:
    """
    Serviço de NLP para remover nomes próprios de um dataset
    usando spaCy NER (Named Entity Recognition).
    """

    def __init__(self, model: str = "pt_core_news_sm") -> None:
        self._nlp = spacy.load(model)
        """
        Inicializa o pipeline spaCy.
        
        :param model: Nome do modelo spaCy em português.
        """
        self._nlp = spacy.load(model)

    def _remove_person_names(self, text: str) -> str:
        """
        Remove entidades do tipo PERSON de um texto.
        
        :param text: Texto de entrada.
        :return: Texto sem nomes próprios.
        """
        if not isinstance(text, str):
            return text

        doc = self._nlp(text)
        clean_tokens = []
        skip_indexes = set()

        for ent in doc.ents:
            if ent.label_ == "PER":
                skip_indexes.update(range(ent.start, ent.end))

        for i, token in enumerate(doc):
            if i not in skip_indexes:
                clean_tokens.append(token.text)

        return " ".join(clean_tokens)

    def clean_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove nomes próprios de uma coluna de texto em um DataFrame.
        
        :param df: DataFrame de entrada.
        :param column: Nome da coluna a ser processada.
        :return: DataFrame com coluna limpa.
        """
        if column not in df.columns:
            raise ValueError(f"Coluna '{column}' não encontrada no DataFrame.")

        df[column] = df[column].apply(self._remove_person_names)
        return df

    def clean_multiple(self, datasets: dict, column: str) -> dict:
        """
        Aplica a limpeza em múltiplos DataFrames.
        
        :param datasets: Dicionário {nome: DataFrame}.
        :param column: Nome da coluna alvo.
        :return: Novo dicionário com DataFrames limpos.
        """
        cleaned = {}
        for name, df in datasets.items():
            cleaned[name] = self.clean_column(df.copy(), column)
        return cleaned
