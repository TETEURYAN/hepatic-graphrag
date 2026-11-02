"""
Inspeciona os prontuários onde mais sistemas do grid erram a mesma
classificação — mostra o texto original do paciente e a resposta de cada
sistema, pra ler manualmente por que o modelo confunde aquele caso.

Ignora status != 'ok' (sem_texto/timeout/erro) na contagem de erro — esses
já são tratados à parte (ver docs/analise_grid_sem_prontuarios_vazios.md),
aqui o interesse é só confusão real de classificação.

Uso:
    uv run python experiments/grid_search/inspect_confusions.py [--top N]
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

from src.core.paths import CSV_ORIGINAL, GROUND_TRUTH_CSV, PROGRESSO_GRID_DIR  # noqa: E402

SISTEMAS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


def carrega_textos() -> dict[str, str]:
    df = pd.read_csv(CSV_ORIGINAL, dtype=str).fillna("")
    df["prontuario"] = df["prontuario"].str.strip()
    return (
        df.groupby("prontuario")["descricao"]
        .apply(lambda x: "\n--- Atendimento ---\n".join(x.tolist()))
        .to_dict()
    )


def main(top: int, full_text: bool) -> None:
    gt = pd.read_csv(GROUND_TRUTH_CSV, dtype=str).fillna("")
    gt["prontuario_id"] = gt["prontuario_id"].str.strip()
    gt = gt.set_index("prontuario_id")
    gt_idx = gt["doenca_principal"]
    CLASSES = [
        "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
        "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
    ]

    progresso = {}
    for sid in SISTEMAS:
        with open(PROGRESSO_GRID_DIR / f"progresso_{sid}.json", encoding="utf-8") as f:
            progresso[sid] = json.load(f)

    textos = carrega_textos()

    avaliados = [pid for pid in gt_idx.index if gt_idx[pid] != "nenhuma"]

    linhas = []
    for pid in avaliados:
        real = gt_idx[pid]
        respostas = {}
        n_ok = 0
        n_errado = 0
        for sid in SISTEMAS:
            r = progresso[sid].get(pid, {})
            if r.get("status") != "ok":
                continue
            n_ok += 1
            pred = r.get("doenca_prevista")
            respostas[sid] = {"pred": pred, "raw": r.get("resposta_raw", "")}
            if pred != real:
                n_errado += 1
        if n_ok == 0:
            continue
        linhas.append({
            "prontuario_id": pid, "real": real,
            "n_ok": n_ok, "n_errado": n_errado,
            "taxa_erro": n_errado / n_ok,
            "respostas": respostas,
        })

    linhas.sort(key=lambda x: (-x["taxa_erro"], -x["n_ok"]))

    print(f"Top {top} prontuários com mais sistemas errando a classificação:\n")
    for item in linhas[:top]:
        pid = item["prontuario_id"]
        row = gt.loc[pid]
        multirrotulo = ", ".join(c for c in CLASSES if row.get(c) == "1") or "(nenhuma marcada)"

        print("=" * 100)
        print(f"Prontuário {pid} — doenca_principal (ground truth): {item['real']}  "
              f"(errado em {item['n_errado']}/{item['n_ok']} sistemas)")
        print(f"  Multirrótulo bruto marcado pelo anotador: {multirrotulo}")
        print(f"  Confiança da anotação: {row.get('confianca', '?')}")
        if row.get("comentario"):
            print(f"  Comentário do anotador: {row['comentario']}")
        print(f"  O que o sistema tinha extraído na hora da anotação: {row.get('sistema_extraiu', '?')}")
        print("-" * 100)
        for sid, r in item["respostas"].items():
            marca = "✗" if r["pred"] != item["real"] else "✔"
            print(f"  {marca} {sid}: previu={r['pred']!r}  resposta_raw={r['raw'][:200]!r}")
        print("-" * 100)
        texto = textos.get(pid, "(texto não encontrado)")
        if not full_text:
            texto = texto[:2000]
        print(f"Texto do prontuário:\n{texto}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=10, help="Quantos prontuários mostrar (padrão: 10)")
    parser.add_argument("--truncar", action="store_true", help="Corta o texto em 2000 caracteres (padrão: mostra completo)")
    args = parser.parse_args()
    main(args.top, full_text=not args.truncar)
