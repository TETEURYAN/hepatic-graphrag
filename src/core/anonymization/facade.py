# src/core/anonymization/facade.py
import pandas as pd
from src.core.anonymization.loader.data_loader import load_all_csvs
from src.core.anonymization.validator.description_matcher import DescriptionMatcher
from src.core.anonymization.validator.anonymization_validator import AnonymizationValidator
from src.core.anonymization.nlp.entity_cleaner import EntityCleaner
from src.core.anonymization.explorer.explorer import DatasetExplorer
from src.core.anonymization.report.report_generator import ReportGenerator
from src.core.anonymization.exceptions import AnonymizationError, EmptyDatasetError

class AnonymizationFeature:
    """
    Fachada que encapsula o fluxo completo de anonimização e validação.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.datasets = {}
        self.cleaner = EntityCleaner()
        self.explorer = DatasetExplorer()
        self.reporter = ReportGenerator()

    def load_datasets(self):
        """Carrega e normaliza datasets."""
        self.datasets = load_all_csvs(self.data_dir, concat=False)
        return self.datasets

    def validate_descriptions(self, dataset_name: str) -> pd.DataFrame:
        """Executa a verificação de anonimização."""
        df = self.datasets.get(dataset_name)
        if df is None:
            raise AnonymizationError(f"Dataset '{dataset_name}' não encontrado.")
        if df.empty:
            raise EmptyDatasetError(dataset_name)

        matcher = DescriptionMatcher()
        mismatches = matcher.find_mismatches(df)
        examples = matcher.sample_mismatches(df, n=5)
        return mismatches, examples

    def validate_anonymization(self, dataset_name: str) -> pd.DataFrame:
        """Executa verificação mais profunda (fuzzy matching e entidades)."""
        df = self.datasets.get(dataset_name)
        if df is None or df.empty:
            raise EmptyDatasetError(dataset_name)

        validator = AnonymizationValidator(self.cleaner)
        result = validator.validate(df)
        return result

    def generate_report(self, df: pd.DataFrame, filename: str):
        """Salva relatório em arquivo TXT."""
        return self.reporter.save_dataframe(df, filename)
