"""Gera um novo ground truth a partir das respostas da segunda opinião do LLM
(`progresso/segunda_opiniao_llm.json`) — 141 prontuários originais + 396
candidatos da expansão estratificada = 537. Ainda SEM validação manual
linha a linha (isso é o próximo passo, feito pelo usuário sobre o CSV
gerado aqui).

NÃO sobrescreve `progresso/ground_truth_anotado.csv` (o original, anotado à
mão do zero) — grava um arquivo novo, no mesmo schema, pra poder ser usado
nos próximos experimentos sem perder o histórico da anotação original.

Uso:
    uv run python3 experiments/ground_truth_second_opinion/build_ground_truth.py
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json  # noqa: E402

from src.core.paths import GROUND_TRUTH_CSV  # noqa: E402

PROGRESSO_LLM_PATH = GROUND_TRUTH_CSV.parent / "segunda_opiniao_llm.json"
SAIDA_CSV = GROUND_TRUTH_CSV.parent / "ground_truth_segunda_opiniao.csv"

# Mesma ordem/hierarquia usada em src/core/ground_truth/annotate.py, pra
# manter `doenca_principal` consistente com o critério já usado no ground
# truth original.
CLASSES = [
    "cirrose_hepatica",
    "esquistossomose",
    "hepatite_c",
    "hepatite_b",
    "esteatose_hepatica",
    "hipertensao_portal",
    "hemorragia_digestiva_alta",
    "pbe",
]

HIERARQUIA = [
    "esquistossomose",
    "cirrose_hepatica",
    "hepatite_c",
    "hepatite_b",
    "esteatose_hepatica",
    "hipertensao_portal",
    "hemorragia_digestiva_alta",
    "pbe",
]


def aplica_hierarquia(doencas: set[str]) -> str:
    for doenca in HIERARQUIA:
        if doenca in doencas:
            return doenca
    return "nenhuma"


def main():
    with open(PROGRESSO_LLM_PATH, "r", encoding="utf-8") as f:
        llm = json.load(f)

    pendentes = [pid for pid, r in llm.items() if r["status"] != "ok"]
    if pendentes:
        raise SystemExit(
            f"Ainda há {len(pendentes)} prontuário(s) sem resposta 'ok' em "
            f"{PROGRESSO_LLM_PATH}: {pendentes}. Rode "
            f"experiments/ground_truth_second_opinion/run.py de novo antes."
        )

    timestamp = datetime.now().isoformat()
    linhas = []
    for pid in sorted(llm, key=lambda x: (len(x), x)):
        doencas = set(llm[pid]["doencas_previstas"])
        linha = {"prontuario_id": pid}
        for c in CLASSES:
            linha[c] = 1 if c in doencas else 0
        linha["doenca_principal"] = aplica_hierarquia(doencas)
        linha["confianca"] = ""
        linha["comentario"] = "segunda opiniao LLM -- pendente validacao manual"
        linha["sistema_extraiu"] = ""
        linha["timestamp"] = timestamp
        linhas.append(linha)

    campos = ["prontuario_id"] + CLASSES + [
        "doenca_principal", "confianca", "comentario", "sistema_extraiu", "timestamp",
    ]
    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)

    print(f"✔ {len(linhas)} prontuários gravados em {SAIDA_CSV}")
    contagem = {}
    for linha in linhas:
        contagem[linha["doenca_principal"]] = contagem.get(linha["doenca_principal"], 0) + 1
    print("Distribuição de doenca_principal:")
    for doenca, n in sorted(contagem.items(), key=lambda kv: -kv[1]):
        print(f"  {doenca:<28} {n}")


if __name__ == "__main__":
    main()
