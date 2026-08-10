"""
Configuration dataclasses and project-wide defaults for the GraphRAG experiment runner.
"""

from dataclasses import dataclass, field
from typing import Optional

# ─── Available query methods ─────────────────────────────────────────────────
METHODS: list[str] = ["local", "global"]

# ─── Sample queries (override via CLI) ───────────────────────────────────────
DEFAULT_QUERIES: list[str] = [
    "Quais são as principais opções de tratamento para pacientes com cirrose?",
    "Quais são os fatores de risco e indicadores prognósticos da doença hepática?",
    "Quais complicações são mais frequentes em pacientes com hepatopatia avançada?",
]

# ─── Default response type passed to graphrag query ──────────────────────────
DEFAULT_RESPONSE_TYPE: str = "Multiple Paragraphs"


@dataclass
class QueryItem:
    """Represents a single query with its associated metadata."""

    query: str
    method: str
    category: str = "general"
    ground_truth: Optional[str] = None
    reference_keywords: Optional[list[str]] = None


@dataclass
class ExperimentConfig:
    """All parameters needed to run a full experiment batch."""

    # GraphRAG index roots (baseline vs prompt-tuned)
    root_base: str
    root_tuned: str

    # Parquet/output directories (auto-derived from roots if not provided)
    data_base: str
    data_tuned: str

    # Queries to run (can be strings or QueryItem objects)
    queries: list[str | QueryItem] = field(default_factory=lambda: list(DEFAULT_QUERIES))

    # Methods to evaluate (used only if queries are strings)
    methods: list[str] = field(default_factory=lambda: list(METHODS))

    # graphrag --response-type value
    response_type: str = DEFAULT_RESPONSE_TYPE

    # Where to persist results, logs and CSV
    results_dir: str = "./results"

    # Whether to pass --verbose to graphrag
    verbose: bool = False

    # Per-query timeout in seconds
    timeout: int = 300

    # k for Recall@k and nDCG@k
    k: int = 5

    # Enable BERTScore computation (requires bert_score package)
    use_bertscore: bool = False

    # Optional ground-truth answers aligned with `queries` (same order)
    # Ignored if QueryItem.ground_truth is set.
    ground_truth: Optional[list[str]] = None

    # Optional per-query keyword lists for Recall@k / nDCG@k
    # Ignored if QueryItem.reference_keywords is set.
    reference_keywords: Optional[list[list[str]]] = None
