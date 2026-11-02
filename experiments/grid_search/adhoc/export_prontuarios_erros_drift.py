"""Salva o texto completo + contexto (ground truth, previsão, resposta bruta)
de cada prontuário em que S7 e/ou S8 (drift) erraram (previsto != ground
truth) -- um .txt por prontuário, pra leitura fora do terminal.

Pasta de saída NUNCA deve ser versionada (texto de prontuário real) --
fica dentro de tcc-results-01/, já coberto pelo .gitignore.

Uso:
    uv run --no-sync python3 experiments/grid_search/export_prontuarios_erros_drift.py
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
SAIDA_DIR = RESULTADOS_DIR / "prontuarios_erros_drift"

CLASSES = [
    "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
    "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
]
SISTEMAS = [("S7", "drift L1"), ("S8", "drift L2")]


def carrega_ground_truth() -> dict[str, set[str]]:
    caminho = _REPO_ROOT / "progresso" / "ground_truth_segunda_opiniao.csv"
    with open(caminho, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {row["prontuario_id"].strip(): {c for c in CLASSES if row[c] == "1"} for row in rows}


def main() -> None:
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    gt_por_id = carrega_ground_truth()
    with open(RESULTADOS_DIR / "progresso_cache.json", encoding="utf-8") as f:
        progresso = json.load(f)
    texto_por_id = carrega_texto_por_prontuario(CSV_ORIGINAL, "prontuario", "descricao")

    # pacientes onde pelo menos um dos dois sistemas errou (ok mas previsto != gt)
    errados: dict[str, dict] = {}
    for pid, gt_set in gt_por_id.items():
        info_sistemas = {}
        algum_erro = False
        for sid, nome in SISTEMAS:
            chave = f"{sid}__{pid}"
            r = progresso.get(chave)
            if r is None or r.get("status") != "ok":
                continue
            pred_set = set(r.get("doencas_previstas", []))
            errou = pred_set != gt_set
            algum_erro = algum_erro or errou
            info_sistemas[sid] = {
                "nome": nome, "previsto": pred_set, "errou": errou,
                "resposta_raw": r.get("resposta_raw", ""),
            }
        if algum_erro and info_sistemas:
            errados[pid] = {"gt": gt_set, "sistemas": info_sistemas}

    print(f"Prontuários com erro em S7 e/ou S8: {len(errados)} de {len(gt_por_id)}\n")

    salvos = 0
    for pid, info in errados.items():
        texto = texto_por_id.get(pid)
        if texto is None:
            continue

        gt_str = ", ".join(sorted(info["gt"])) or "nenhuma"
        blocos = [
            f"PRONTUÁRIO: {pid}",
            f"GROUND TRUTH (real): {gt_str}",
            "",
        ]
        for sid, nome in SISTEMAS:
            s = info["sistemas"].get(sid)
            if s is None:
                blocos.append(f"--- {sid} ({nome}): sem dado ---\n")
                continue
            pred_str = ", ".join(sorted(s["previsto"])) or "nenhuma"
            fp = sorted(s["previsto"] - info["gt"])
            fn = sorted(info["gt"] - s["previsto"])
            marca = "ERRO" if s["errou"] else "ok"
            blocos.append(f"--- {sid} ({nome}): {marca} ---")
            blocos.append(f"Previsto: {pred_str}")
            if fp:
                blocos.append(f"Falso positivo (previu e não tinha): {', '.join(fp)}")
            if fn:
                blocos.append(f"Falso negativo (tinha e não previu): {', '.join(fn)}")
            blocos.append("Resposta bruta do modelo:")
            blocos.append(s["resposta_raw"])
            blocos.append("")

        blocos.append("=" * 80)
        blocos.append("TEXTO COMPLETO DO PRONTUÁRIO")
        blocos.append("=" * 80)
        blocos.append(texto)

        (SAIDA_DIR / f"{pid}.txt").write_text("\n".join(blocos), encoding="utf-8")
        salvos += 1

    print(f"✔ {salvos} arquivos salvos em: {SAIDA_DIR}")


if __name__ == "__main__":
    main()
