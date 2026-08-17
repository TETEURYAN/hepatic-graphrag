"""Caminhos usados pelos experimentos, configuráveis por variável de ambiente.

Cada constante tem um default relativo à raiz do repositório (funciona
sem configuração nenhuma, assumindo a convenção de pastas deste projeto),
mas pode ser sobrescrita via `.env` — nenhum script de experimento deve
hardcodar caminho de índice/dataset/saída diretamente.

Uso:
    from src.core.paths import bootstrap_sys_path, REPO_ROOT, GRAPHRAG_ROOT
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def bootstrap_sys_path(*dirs: Path) -> None:
    """Garante que os diretórios dados estejam em sys.path (sem duplicar).

    Chamado no topo de cada script de experimento antes de importar
    `src.core.*` ou módulos irmãos (ex.: `parser_doencas`), já que esses
    scripts são executados de formas diferentes (CLI, Jupyter/VS Code
    interactive window) e não há garantia de que a raiz do repo já esteja
    no path.
    """
    for d in dirs:
        d = str(d)
        if d not in sys.path:
            sys.path.insert(0, d)


def _env_path(var: str, default: Path) -> Path:
    valor = os.environ.get(var)
    return Path(valor) if valor else default


def _env_path_optional(var: str) -> Path | None:
    valor = os.environ.get(var)
    return Path(valor) if valor else None


# ── Índices GraphRAG (pasta `experimentos/`, local e nunca versionada) ────
GRAPHRAG_ROOT = _env_path("GRAPHRAG_ROOT", REPO_ROOT / "experimentos" / "local copy")
GRAPHRAG_DATA = _env_path("GRAPHRAG_DATA", GRAPHRAG_ROOT / "output")
GRAPHRAG_ROOT_BASE = _env_path("GRAPHRAG_ROOT_BASE", REPO_ROOT / "experimentos" / "local")
GRAPHRAG_ROOT_TUNED = _env_path("GRAPHRAG_ROOT_TUNED", REPO_ROOT / "experimentos" / "local copy")

# ── Dataset bruto (pasta `data/`, sensível e nunca versionada) ────────────
CSV_ORIGINAL = _env_path(
    "CSV_ORIGINAL", REPO_ROOT / "experimentos" / "local" / "input" / "data-hepato-sem-anonimizacao.csv"
)
DATASET_ANONIMIZADO = _env_path("DATASET_ANONIMIZADO", REPO_ROOT / "data" / "coluna-evolucoes_anonimizadas.csv")

# ── Ground truth ───────────────────────────────────────────────────────────
GROUND_TRUTH_CSV = _env_path("GROUND_TRUTH_CSV", REPO_ROOT / "progresso" / "ground_truth_anotado.csv")
ANOTACAO_PROGRESSO = _env_path("ANOTACAO_PROGRESSO", REPO_ROOT / "progresso" / "anotacao_progresso.json")

# Candidatos à expansão do ground truth (filtro regex + crivo LLM sobre os
# prontuários fora do ground truth original — ver experiments/ground_truth_expansion/)
CANDIDATOS_EXPANSAO_GT = _env_path(
    "CANDIDATOS_EXPANSAO_GT", REPO_ROOT / "progresso" / "candidatos_expansao_amostra_estratificada.json"
)

# ── Datasets construídos por build_datasets.py — sem default fixo, porque o
# nome inclui timestamp; defina no .env depois de cada rodada ─────────────
DATASET_TUNED_CSV = _env_path(
    "DATASET_TUNED_CSV", REPO_ROOT / "results" / "dataset" / "dataset_record_prompt-tuning_SUBSTITUA.csv"
)
DATASET_BASE_CSV = _env_path_optional("DATASET_BASE_CSV")

# ── Saída dos experimentos ─────────────────────────────────────────────────
RESULTADOS_GRID_DIR = _env_path("RESULTADOS_GRID_DIR", REPO_ROOT / "resultados_grid")
PROGRESSO_GRID_DIR = _env_path("PROGRESSO_GRID_DIR", REPO_ROOT / "progresso_grid")
RESULTADOS_AVALIACAO_DIR = _env_path("RESULTADOS_AVALIACAO_DIR", REPO_ROOT / "resultados_avaliacao")
