
class AnonymizationError(Exception):
    """Erro genérico do módulo de anonimização."""
    pass


class MissingColumnError(AnonymizationError):
    """Erro lançado quando colunas obrigatórias estão ausentes."""

    def __init__(self, columns):
        self.columns = columns
        super().__init__(f"Colunas obrigatórias ausentes: {', '.join(columns)}")


class EmptyDatasetError(AnonymizationError):
    """Erro lançado quando o dataset está vazio."""
    def __init__(self, name):
        super().__init__(f"O dataset '{name}' está vazio.")
