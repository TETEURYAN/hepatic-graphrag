"""
Salva o texto completo de cada prontuário mais confundido pelo grid (mesma
lista de experiments/grid_search/inspect_confusions.py) em um .txt por
prontuário numa pasta — pra ler num editor de texto em vez do terminal.

Ex.: o prontuário 10081883 vira <output_dir>/10081883.txt.

A pasta de saída nunca deve ser versionada (é texto de prontuário real) —
o padrão fica dentro de resultados_grid/, já coberto pelo .gitignore.

Uso:
    uv run python experiments/grid_search/export_confusion_texts.py [--top N] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.paths import CSV_ORIGINAL, GROUND_TRUTH_CSV, PROGRESSO_GRID_DIR, RESULTADOS_GRID_DIR  # noqa: E402

SISTEMAS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


def carrega_textos() -> dict[str, str]:
    df = pd.read_csv(CSV_ORIGINAL, dtype=str).fillna("")
    df["prontuario"] = df["prontuario"].str.strip()
    return (
        df.groupby("prontuario")["descricao"]
        .apply(lambda x: "\n--- Atendimento ---\n".join(x.tolist()))
        .to_dict()
    )


def prontuarios_mais_confundidos(top: int) -> list[tuple[str, str, int, int]]:
    """Retorna [(prontuario_id, doenca_real, n_errado, n_ok), ...] ordenado por taxa de erro."""
    gt = pd.read_csv(GROUND_TRUTH_CSV, dtype=str).fillna("")
    gt["prontuario_id"] = gt["prontuario_id"].str.strip()
    gt_idx = gt.set_index("prontuario_id")["doenca_principal"]

    progresso = {}
    for sid in SISTEMAS:
        with open(PROGRESSO_GRID_DIR / f"progresso_{sid}.json", encoding="utf-8") as f:
            progresso[sid] = json.load(f)

    linhas = []
    for pid in (pid for pid in gt_idx.index if gt_idx[pid] != "nenhuma"):
        real = gt_idx[pid]
        n_ok = n_errado = 0
        for sid in SISTEMAS:
            r = progresso[sid].get(pid, {})
            if r.get("status") != "ok":
                continue
            n_ok += 1
            if r.get("doenca_prevista") != real:
                n_errado += 1
        if n_ok == 0:
            continue
        linhas.append((pid, real, n_errado, n_ok))

    linhas.sort(key=lambda x: (-(x[2] / x[3]), -x[3]))
    return linhas[:top]


def main(top: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    textos = carrega_textos()

    salvos = 0
    for pid, real, n_errado, n_ok in prontuarios_mais_confundidos(top):
        texto = textos.get(pid)
        if texto is None:
            print(f"  aviso: {pid} sem texto-fonte, pulando.")
            continue
        caminho = output_dir / f"{pid}.txt"
        cabecalho = f"Prontuário {pid} — ground truth: {real} (errado em {n_errado}/{n_ok} sistemas)\n{'=' * 80}\n\n"
        caminho.write_text(cabecalho + texto, encoding="utf-8")
        print(f"  {caminho}")
        salvos += 1

    print(f"\n✔ {salvos} arquivos salvos em {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=20, help="Quantos prontuários exportar (padrão: 20)")
    parser.add_argument("--output-dir", type=Path, default=RESULTADOS_GRID_DIR / "prontuarios_confusos")
    args = parser.parse_args()
    main(args.top, args.output_dir)
