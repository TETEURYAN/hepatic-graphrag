#!/usr/bin/env python3
"""
GraphRAG Experiment Runner — Baseline vs Prompt-Tuning
========================================================
Executa queries com GraphRAG via CLI comparando resultados com e sem
prompt tuning, para um ou mais métodos de busca.

A chamada ao `graphrag query` propriamente dita (subprocess, timeout,
extração da resposta) vive em src/core/query/runner.py — a mesma usada por
experiments/grid_search — para que os dois experimentos nunca tenham lógica
de execução de query duplicada entre si. O progresso é salvo
incrementalmente (src/core/query/progress.py): se a bateria for
interrompida, rodar de novo só refaz o que falta.

Uso básico:
    uv run python experiments/prompt_tuning_comparison/run.py \\
        --root-base  ./experimentos/local \\
        --root-tuned "./experimentos/local copy" \\
        --queries "Quais os tratamentos para cirrose?" "Quais os fatores de risco?"

Uso completo:
    uv run python experiments/prompt_tuning_comparison/run.py \\
        --root-base   ./experimentos/local \\
        --root-tuned  "./experimentos/local copy" \\
        --data-base   ./experimentos/local/output \\
        --data-tuned  "./experimentos/local copy/output" \\
        --methods     local global drift basic \\
        --queries     "Query 1" "Query 2" \\
        --ground-truth "Resposta 1" "Resposta 2" \\
        --k           5 \\
        --timeout     1800 \\
        --results-dir ./results \\
        --csv-output  ./results/experiment_results.csv \\
        --bertscore \\
        --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.paths import (  # noqa: E402
    GRAPHRAG_ROOT_BASE,
    GRAPHRAG_ROOT_TUNED,
    bootstrap_sys_path,
)

bootstrap_sys_path(Path(__file__).resolve().parent)

from config import DEFAULT_QUERIES, METHODS, ExperimentConfig, QueryItem  # noqa: E402
from metrics import compute_all  # noqa: E402
from src.core.query.progress import ProgressTracker  # noqa: E402
from src.core.query.runner import run_graphrag_query as _core_run_graphrag_query  # noqa: E402
from utils import (  # noqa: E402
    find_best_method,
    print_ranking_summary,
    print_results_table,
    save_result,
    save_results_csv,
    setup_logging,
)


def load_queries_from_dir(queries_dir: str | Path) -> list[QueryItem]:
    """
    Recursively scan directory for query_*.txt files and extract metadata.
    Expected structure: <dir>/<category>/<method>/query_<n>.txt
    """
    base_path = Path(queries_dir)
    if not base_path.is_dir():
        logging.error("Diretório de queries não encontrado: %s", base_path)
        return []

    query_items: list[QueryItem] = []
    known_methods = set(METHODS)

    for file_path in base_path.rglob("query_*.txt"):
        try:
            parts = file_path.relative_to(base_path).parts
            method = "local"
            category = "general"

            for part in reversed(parts[:-1]):
                if part.lower() in known_methods:
                    method = part.lower()
                    idx = parts.index(part)
                    if idx > 0:
                        category = parts[idx - 1]
                    break

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    query_items.append(QueryItem(query=text, method=method, category=category))
        except Exception as e:
            logging.warning("Erro ao carregar query de %s: %s", file_path, e)

    logging.info("Carregadas %d queries de %s", len(query_items), base_path)
    return query_items


# ─── GraphRAG CLI execution ───────────────────────────────────────────────────

def run_graphrag_query(query: str, root: str, data: str, method: str, response_type: str, verbose: bool, timeout: int) -> tuple[str, bool]:
    """Fino wrapper sobre src.core.query.runner — mantém a assinatura usada
    pelo resto deste script (sem community_level, que aqui não se aplica)."""
    resposta, sucesso, _duracao = _core_run_graphrag_query(
        query=query, root=root, data=data, method=method,
        response_type=response_type, timeout=timeout,
    )
    return resposta, sucesso


# ─── Single-run execution ─────────────────────────────────────────────────────

def _safe_filename(text: str, max_len: int = 30) -> str:
    slug = text[:max_len].replace(" ", "_").replace("/", "-")
    return "".join(c for c in slug if c.isalnum() or c in "_-")


def _progress_key(category: str, method: str, prompt_tuning: bool, q_idx: int) -> str:
    tuning_label = "tuned" if prompt_tuning else "base"
    return f"{category}::{method}::{tuning_label}::{q_idx}"


def execute_single(
    query: str,
    method: str,
    prompt_tuning: bool,
    cfg: ExperimentConfig,
    q_idx: int,
    tracker: ProgressTracker,
    category: Optional[str] = None,
) -> dict:
    """
    Run one (query × method × condition) experiment and return its result dict.
    Saves the raw response and a JSON metadata sidecar to cfg.results_dir.
    Resumível: se essa combinação já está em `tracker`, reaproveita o
    resultado salvo em vez de rodar a query de novo.
    """
    category = category or "general"
    key = _progress_key(category, method, prompt_tuning, q_idx)
    if key in tracker:
        logging.info("  [cache] %s já processado, pulando.", key)
        return tracker[key]

    root = cfg.root_tuned if prompt_tuning else cfg.root_base
    data = cfg.data_tuned if prompt_tuning else cfg.data_base
    tuning_label = "tuned" if prompt_tuning else "base"

    response, success = run_graphrag_query(
        query=query, root=root, data=data, method=method,
        response_type=cfg.response_type, verbose=cfg.verbose, timeout=cfg.timeout,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_filename(category if category else query)
    filename = f"{method.capitalize()}_{tuning_label}_{slug}_{timestamp}"
    raw_path = Path(cfg.results_dir) / filename

    reference = cfg.ground_truth[q_idx] if cfg.ground_truth else None
    ref_keywords = cfg.reference_keywords[q_idx] if cfg.reference_keywords else None

    metrics = compute_all(
        response=response if success else "",
        query=query, reference=reference, reference_keywords=ref_keywords,
        k=cfg.k, use_bertscore=cfg.use_bertscore,
    )

    record: dict = {
        "query": query, "method": method, "category": category,
        "prompt_tuning": prompt_tuning, "success": success, "timestamp": timestamp,
        "output_file": str(raw_path.with_suffix(".txt")), **metrics,
    }

    save_result(content=response, output_path=raw_path.with_suffix(".txt"), metadata=record)

    status = "OK" if success else "FALHOU"
    logging.info(
        "  [%s] método=%-6s tuning=%-5s | FactScore=%.3f  Faithfulness=%.3f  Recall@%d=%.3f  RadGraph-F1=%.3f",
        status, method, tuning_label, metrics["fact_score"], metrics["faithfulness"], cfg.k, metrics["recall_at_k"], metrics["radgraph_f1"],
    )

    tracker.registra(key, record)
    return record


# ─── Experiment orchestration ─────────────────────────────────────────────────

def run_experiment(cfg: ExperimentConfig) -> list[dict]:
    """
    Itera sobre todas as combinações (query × método × condição). Cada query
    roda com cada método, e cada método roda 2x: uma contra o índice base,
    outra contra o prompt-tuned.
    """
    tracker = ProgressTracker(Path(cfg.results_dir) / "progress.json")
    all_results: list[dict] = []
    counter = 0

    if cfg.queries and isinstance(cfg.queries[0], QueryItem):
        total = len(cfg.queries) * 2
        logging.info("Executando queries categorizadas. Total: %d", total)

        for q_idx, item in enumerate(cfg.queries):
            item = cast(QueryItem, item)
            logging.info("")
            logging.info("─" * 60)
            logging.info("[%s] Query %d/%d: %s", item.category, q_idx + 1, len(cfg.queries), item.query[:80])
            logging.info("─" * 60)

            for prompt_tuning in (False, True):
                counter += 1
                tuning_str = "tuned" if prompt_tuning else "base"
                logging.info("[%d/%d] método=%s  condição=%s", counter, total, item.method, tuning_str)

                record = execute_single(
                    query=item.query, method=item.method, prompt_tuning=prompt_tuning,
                    cfg=cfg, q_idx=q_idx, tracker=tracker, category=item.category,
                )
                all_results.append(record)
    else:
        total = len(cfg.queries) * len(cfg.methods) * 2
        logging.info("Total de execuções planejadas (cross-product): %d", total)

        for q_idx, query in enumerate(cfg.queries):
            query = cast(str, query)
            logging.info("")
            logging.info("─" * 60)
            logging.info("Query %d/%d: %s", q_idx + 1, len(cfg.queries), query[:80])
            logging.info("─" * 60)

            for method in cfg.methods:
                for prompt_tuning in (False, True):
                    counter += 1
                    tuning_str = "tuned" if prompt_tuning else "base"
                    logging.info("[%d/%d] método=%s  condição=%s", counter, total, method, tuning_str)

                    record = execute_single(
                        query=query, method=method, prompt_tuning=prompt_tuning,
                        cfg=cfg, q_idx=q_idx, tracker=tracker,
                    )
                    all_results.append(record)

    return all_results


# ─── CLI argument parsing ─────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    idx = parser.add_argument_group("Caminhos do índice GraphRAG")
    idx.add_argument("--root-base", default=str(GRAPHRAG_ROOT_BASE), metavar="DIR", help="Diretório raiz do índice GraphRAG SEM prompt tuning (padrão: GRAPHRAG_ROOT_BASE do .env)")
    idx.add_argument("--root-tuned", default=str(GRAPHRAG_ROOT_TUNED), metavar="DIR", help="Diretório raiz do índice GraphRAG COM prompt tuning (padrão: GRAPHRAG_ROOT_TUNED do .env)")
    idx.add_argument("--data-base", default=None, metavar="DIR", help="Diretório de dados (parquet) para o índice base. Padrão: <root-base>/output")
    idx.add_argument("--data-tuned", default=None, metavar="DIR", help="Diretório de dados (parquet) para o índice tuned. Padrão: <root-tuned>/output")

    qry = parser.add_argument_group("Configuração das queries")
    qry.add_argument("--queries", nargs="+", default=None, metavar="QUERY", help="Uma ou mais queries a executar (entre aspas)")
    qry.add_argument("--queries-dir", default=None, metavar="DIR", help="Diretório contendo subpastas com arquivos de query (*.txt)")
    qry.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS, metavar="METHOD", help=f"Métodos de busca a usar. Padrão: {METHODS}")
    qry.add_argument("--response-type", default="Multiple Paragraphs", metavar="TYPE", help="Valor de --response-type passado ao graphrag query")
    qry.add_argument("--ground-truth", nargs="+", default=None, metavar="ANS", help="Respostas de referência (ground truth), na mesma ordem das queries")

    met = parser.add_argument_group("Métricas")
    met.add_argument("--k", type=int, default=5, metavar="K", help="k para Recall@k e nDCG@k (padrão: 5)")
    met.add_argument("--bertscore", action="store_true", help="Calcular BERTScore F1 (requer `pip install bert-score` e ground truth fornecido)")

    out = parser.add_argument_group("Saída")
    out.add_argument("--results-dir", default="./results", metavar="DIR", help="Diretório para salvar respostas brutas, progresso e logs (padrão: ./results)")
    out.add_argument("--csv-output", default="./results/experiment_results.csv", metavar="FILE")

    exe = parser.add_argument_group("Execução")
    exe.add_argument("--timeout", type=int, default=1800, metavar="SEC", help="Timeout em segundos por query (padrão: 1800)")
    exe.add_argument("--verbose", action="store_true")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    data_base = args.data_base or str(Path(args.root_base) / "output")
    data_tuned = args.data_tuned or str(Path(args.root_tuned) / "output")

    if args.ground_truth and len(args.ground_truth) != len(args.queries):
        parser.error(f"--ground-truth deve ter o mesmo número de itens que --queries ({len(args.queries)} queries, {len(args.ground_truth)} ground truths)")

    setup_logging(args.results_dir, args.verbose)

    queries_metadata: list[QueryItem] = []
    if args.queries_dir:
        queries_metadata = load_queries_from_dir(args.queries_dir)
        if not queries_metadata:
            logging.error("Nenhuma query encontrada em %s", args.queries_dir)
            sys.exit(1)
        queries = queries_metadata
    elif args.queries:
        queries = args.queries
    else:
        queries = DEFAULT_QUERIES

    cfg = ExperimentConfig(
        root_base=args.root_base, root_tuned=args.root_tuned,
        data_base=data_base, data_tuned=data_tuned,
        queries=queries, methods=args.methods, response_type=args.response_type,
        results_dir=args.results_dir, verbose=args.verbose, timeout=args.timeout,
        k=args.k, use_bertscore=args.bertscore, ground_truth=args.ground_truth,
    )

    logging.info("=" * 60)
    logging.info("GraphRAG Experiment Runner — prompt_tuning_comparison")
    logging.info("  root-base   : %s", cfg.root_base)
    logging.info("  root-tuned  : %s", cfg.root_tuned)
    logging.info("  queries     : %d", len(cfg.queries))
    logging.info("  métodos     : %s", cfg.methods)
    logging.info("  total runs  : %d", len(cfg.queries) * len(cfg.methods) * 2)
    logging.info("=" * 60)

    results = run_experiment(cfg)

    print_results_table(results)
    print_ranking_summary(results)

    csv_path = Path(args.csv_output)
    save_results_csv(results, csv_path)

    logging.info("")
    logging.info("Experimento concluído. %d execuções realizadas.", len(results))
    logging.info("Resultados: %s", Path(args.results_dir).resolve())
    logging.info("CSV       : %s", csv_path.resolve())


if __name__ == "__main__":
    main()
