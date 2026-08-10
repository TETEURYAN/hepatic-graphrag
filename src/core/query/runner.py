"""Execução de queries no GraphRAG via CLI, compartilhada entre os experimentos.

Unifica duas implementações que existiam duplicadas — `roda_query()` em
experimentos_graphrag.py e `run_graphrag_query()` em run_experiments.py —
numa só: paths sempre recebidos por parâmetro (nunca hardcoded), suporte
opcional a --community-level (grid_search) e timeout configurável por
método (drift precisa de bem mais tempo que local/basic).
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

DEFAULT_TIMEOUTS = {
    "drift": 2400,
    "global": 1800,
    "local": 600,
    "basic": 600,
}

_SUCCESS_RE = re.compile(r"SUCCESS:\s*(.+)", re.IGNORECASE | re.DOTALL)


def default_timeout(method: str) -> int:
    return DEFAULT_TIMEOUTS.get(method, 600)


def run_graphrag_query(
    query: str,
    root: str | Path,
    data: str | Path,
    method: str,
    community_level: int | None = None,
    response_type: str = "Multiple Paragraphs",
    timeout: int | None = None,
    use_patched_runner: bool = False,
    uv_no_sync: bool = True,
) -> tuple[str, bool, float]:
    """Executa `graphrag query` via subprocess.

    Retorna (resposta, sucesso, duração_s). Nunca lança exceção — timeout,
    binário ausente e falhas inesperadas viram uma resposta `"ERROR: ..."`
    com sucesso=False, para que o loop do experimento continue.

    `use_patched_runner=True` roda via `src/core/patches/run_graphrag.py`
    em vez de `graphrag` direto — necessário para os métodos global/drift
    contornarem o `response_format=json_object` forçado contra modelos que
    não o suportam (ver `src/core/patches/graphrag_json_patch.py`).
    """
    if timeout is None:
        timeout = default_timeout(method)

    cmd = ["uv", "run"]
    if uv_no_sync:
        cmd.append("--no-sync")

    if use_patched_runner:
        patched_runner = Path(__file__).resolve().parents[1] / "patches" / "run_graphrag.py"
        cmd += ["python", str(patched_runner), "query"]
    else:
        cmd += ["graphrag", "query"]

    cmd += ["--root", str(root), "--data", str(data), "--method", method]
    if community_level is not None:
        cmd += ["--community-level", str(community_level)]
    cmd += ["--response-type", response_type, "--query", query]

    inicio = time.time()
    try:
        resultado = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: query timed out after {timeout}s", False, float(timeout)
    except FileNotFoundError:
        return "ERROR: 'uv'/'graphrag' não encontrado no PATH", False, round(time.time() - inicio, 1)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: unexpected exception — {exc}", False, round(time.time() - inicio, 1)

    duracao = round(time.time() - inicio, 1)

    if resultado.returncode != 0:
        stderr = resultado.stderr.strip()[-800:]
        return f"ERROR: exit {resultado.returncode} — {stderr}", False, duracao

    stdout = resultado.stdout or ""
    match = _SUCCESS_RE.search(stdout)
    resposta = match.group(1).strip() if match else stdout.strip()
    return resposta, True, duracao
