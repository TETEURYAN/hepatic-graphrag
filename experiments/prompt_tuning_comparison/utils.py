"""
Utility helpers for the GraphRAG experiment runner:
logging setup, file I/O, terminal table printing, CSV export, and ranking.
"""

import csv
import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, verbose: bool = False) -> None:
    """
    Configure root logger to write to both stdout and a timestamped log file
    inside `log_dir`.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"experiment_{stamp}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Log file: {log_file}")


# ─── File I/O ─────────────────────────────────────────────────────────────────

def save_result(
    content: str,
    output_path: Path,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Persist the raw GraphRAG response as a `.txt` file and, when `metadata`
    is provided, a companion `.json` file with the full result record.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    if metadata:
        json_path = output_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False, default=str)


# ─── Terminal table ───────────────────────────────────────────────────────────

# Column specs: (header label, result-dict key, width, fmt)
_COLUMNS: list[tuple[str, str, int, str]] = [
    ("Query",             "query",                28, "s"),
    ("Método",            "method",                8, "s"),
    ("Prompt Tuning",     "prompt_tuning",         13, "s"),
    ("FactScore",         "fact_score",            10, ".3f"),
    ("Faithfulness",      "faithfulness",          13, ".3f"),
    ("Recall@k",          "recall_at_k",            9, ".3f"),
    ("RadGraph-F1",       "radgraph_f1",           12, ".3f"),
    ("nDCG@k",            "ndcg_at_k",              8, ".3f"),
    ("Temp.Cons.",        "temporal_consistency",  11, ".3f"),
    ("BERTScore",         "bert_score_f1",         10, "s"),
]


def _fmt_cell(value: Any, width: int, fmt: str) -> str:
    """Format a single table cell, right-padding to `width`."""
    if value is None:
        return "N/A".ljust(width)
    if fmt == "s":
        if isinstance(value, bool):
            text = "sim" if value else "não"
        else:
            text = textwrap.shorten(str(value), width=width - 1, placeholder="…")
        return text.ljust(width)
    try:
        return format(float(value), fmt).ljust(width)
    except (ValueError, TypeError):
        return str(value).ljust(width)


def print_results_table(results: list[dict[str, Any]]) -> None:
    """
    Print a formatted comparative table of all experiment results to stdout.
    """
    if not results:
        print("Sem resultados para exibir.")
        return

    header = " | ".join(label.ljust(w) for label, _, w, _ in _COLUMNS)
    separator = "-" * len(header)

    print()
    print("=" * len(header))
    print("RESULTADOS EXPERIMENTO GRAPHRAG")
    print("=" * len(header))
    print(header)
    print(separator)

    for r in results:
        cells = [_fmt_cell(r.get(key), w, fmt) for _, key, w, fmt in _COLUMNS]
        print(" | ".join(cells))

    print(separator)


# ─── CSV export ───────────────────────────────────────────────────────────────

def save_results_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write all result records to a CSV file. Creates parent directories as needed.
    """
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"Resultados salvos em CSV: {output_path}")


# ─── Ranking ──────────────────────────────────────────────────────────────────

_SCORE_KEYS = ("fact_score", "faithfulness", "recall_at_k", "radgraph_f1")


def _average_score(record: dict[str, Any]) -> float:
    """Compute the mean of the four primary metrics for a single result record."""
    values = [record.get(k) for k in _SCORE_KEYS]
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else 0.0


def find_best_method(
    results: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Return a summary of the highest-scoring configuration.

    The winner is determined by the mean of {FactScore, Faithfulness,
    Recall@k, RadGraph-F1}.  Returns None when `results` is empty.
    """
    if not results:
        return None

    scored = [(r, _average_score(r)) for r in results]
    best_record, best_avg = max(scored, key=lambda x: x[1])

    return {
        "result": best_record,
        "avg_score": best_avg,
        "ranking": sorted(scored, key=lambda x: x[1], reverse=True),
    }


def print_ranking_summary(results: list[dict[str, Any]]) -> None:
    """
    Print the top-3 configurations and highlight the absolute winner.
    """
    best = find_best_method(results)
    if not best:
        return

    ranking = best["ranking"]

    print()
    print("=" * 60)
    print("RANKING — TOP CONFIGURAÇÕES")
    print("=" * 60)

    for rank, (r, avg) in enumerate(ranking[:5], start=1):
        tuning = "sim" if r.get("prompt_tuning") else "não"
        marker = " ◀ MELHOR" if rank == 1 else ""
        print(
            f"  #{rank}  método={r.get('method'):<8} "
            f"tuning={tuning}  avg={avg:.3f}{marker}"
        )

    best_r = best["result"]
    tuning_label = "sim" if best_r.get("prompt_tuning") else "não"
    print()
    print(
        f"  MELHOR CONFIGURAÇÃO GERAL: método={best_r.get('method')} | "
        f"prompt tuning={tuning_label} | score médio={best['avg_score']:.3f}"
    )
    print("=" * 60)
