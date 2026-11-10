"""Exporta prontuários de exemplo em dois grupos, pra inspeção qualitativa:

1. `casos_falso_positivo_binarias/`: prontuários em que o drift (S7/S8)
   "chutou" doenças de marcador objetivo (hepatite B, hepatite C) que NÃO
   estão no rótulo de referência -- não é o caso de confundir uma doença
   por outra, é prever doença a mais do que o necessário.
2. `casos_acerto_inferencia/`: prontuários em que o drift acertou
   corretamente uma doença que exige inferência indireta (cirrose
   hepática, hipertensão portal, esquistossomose) -- presente no rótulo de
   referência E corretamente prevista.

Cada arquivo .txt contém o texto do prontuário seguido da resposta bruta
do LLM pra aquele sistema.

Uso:
    uv run --no-sync python3 experiments/grid_search/exporta_casos_exemplo.py
"""

import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.dataset.clinical_text import carrega_texto_por_prontuario  # noqa: E402
from src.core.paths import CSV_ORIGINAL  # noqa: E402

RESULTADOS_DIR = _REPO_ROOT / "tcc-results-01"
SAIDA_FALSO_POSITIVO = RESULTADOS_DIR / "casos_falso_positivo_binarias"
SAIDA_ACERTO_INFERENCIA = RESULTADOS_DIR / "casos_acerto_inferencia"

CLASSES = [
    "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
    "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
]
SISTEMAS_DRIFT = ["S7", "S8"]

DOENCAS_BINARIAS = ["hepatite_b", "hepatite_c"]  # marcador textual objetivo
DOENCAS_INFERENCIA = ["cirrose_hepatica", "hipertensao_portal", "esquistossomose"]  # inferência indireta

MAX_CASOS_POR_GRUPO = 8


def monta_conteudo(pid: str, texto: str, doenca: str, sid: str, r: dict, rotulo: str) -> str:
    return (
        f"PRONTUÁRIO: {pid}\n"
        f"SISTEMA: {sid} (drift)\n"
        f"DOENÇA EM FOCO: {doenca}\n"
        f"CASO: {rotulo}\n"
        f"DOENÇAS PREVISTAS PELO DRIFT: {', '.join(r['doencas_previstas']) or '(nenhuma)'}\n"
        f"{'=' * 70}\n"
        f"TEXTO DO PRONTUÁRIO\n"
        f"{'=' * 70}\n"
        f"{texto}\n\n"
        f"{'=' * 70}\n"
        f"RESPOSTA BRUTA DO LLM ({sid})\n"
        f"{'=' * 70}\n"
        f"{r['resposta_raw']}\n"
    )


def main():
    with open(_REPO_ROOT / "progresso" / "ground_truth_segunda_opiniao.csv", encoding="utf-8-sig") as f:
        gt_por_id = {
            row["prontuario_id"].strip(): {c for c in CLASSES if row[c] == "1"}
            for row in csv.DictReader(f)
        }

    with open(RESULTADOS_DIR / "progresso_cache.json", encoding="utf-8") as f:
        resultados = json.load(f)

    textos = carrega_texto_por_prontuario(CSV_ORIGINAL, "prontuario", "descricao")

    SAIDA_FALSO_POSITIVO.mkdir(parents=True, exist_ok=True)
    SAIDA_ACERTO_INFERENCIA.mkdir(parents=True, exist_ok=True)

    # ===== grupo 1: falso positivo em doença binária (hepatite B/C) =====
    casos_fp = []
    for pid, gt_set in gt_por_id.items():
        texto = textos.get(pid)
        if not texto:
            continue
        for sid in SISTEMAS_DRIFT:
            r = resultados.get(f"{sid}__{pid}")
            if not r or r.get("status") != "ok":
                continue
            for doenca in DOENCAS_BINARIAS:
                if doenca not in gt_set and doenca in set(r["doencas_previstas"]):
                    casos_fp.append((pid, doenca, sid, r))

    for pid, doenca, sid, r in casos_fp[:MAX_CASOS_POR_GRUPO]:
        conteudo = monta_conteudo(
            pid, textos[pid], doenca, sid, r,
            rotulo=f"FALSO POSITIVO -- drift previu '{doenca}' mas ela NÃO está no rótulo de referência",
        )
        (SAIDA_FALSO_POSITIVO / f"{pid}_{doenca}_{sid}.txt").write_text(conteudo, encoding="utf-8")
    print(f"Grupo 1 (falso positivo binária): {min(len(casos_fp), MAX_CASOS_POR_GRUPO)} "
          f"casos exportados (de {len(casos_fp)} encontrados) em {SAIDA_FALSO_POSITIVO}")

    # ===== grupo 2: acerto em doença de inferência indireta =====
    casos_acerto = []
    for pid, gt_set in gt_por_id.items():
        texto = textos.get(pid)
        if not texto:
            continue
        for sid in SISTEMAS_DRIFT:
            r = resultados.get(f"{sid}__{pid}")
            if not r or r.get("status") != "ok":
                continue
            for doenca in DOENCAS_INFERENCIA:
                if doenca in gt_set and doenca in set(r["doencas_previstas"]):
                    casos_acerto.append((pid, doenca, sid, r))

    for pid, doenca, sid, r in casos_acerto[:MAX_CASOS_POR_GRUPO]:
        conteudo = monta_conteudo(
            pid, textos[pid], doenca, sid, r,
            rotulo=f"ACERTO -- drift previu corretamente '{doenca}', presente no rótulo de referência",
        )
        (SAIDA_ACERTO_INFERENCIA / f"{pid}_{doenca}_{sid}.txt").write_text(conteudo, encoding="utf-8")
    print(f"Grupo 2 (acerto inferência indireta): {min(len(casos_acerto), MAX_CASOS_POR_GRUPO)} "
          f"casos exportados (de {len(casos_acerto)} encontrados) em {SAIDA_ACERTO_INFERENCIA}")


if __name__ == "__main__":
    main()
