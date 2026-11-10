"""Gráfico simples: para cada par (prontuário, doença) avaliado pelo `drift`
(S7/S8), marca se ele acertou ou errou, em função do tamanho do texto do
prontuário -- a mesma variável que os testes estatísticos (H1/H2 em
testa_hipoteses_drift.py e testa_hipoteses_drift_erro.py) confirmaram como
fator associado tanto ao acerto quanto ao erro do drift.

Uso:
    uv run --no-sync python3 experiments/grid_search/plota_acertos_erros_drift.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.dataset.clinical_text import carrega_texto_por_prontuario  # noqa: E402
from src.core.paths import CSV_ORIGINAL  # noqa: E402

RESULTADOS_DIR = _REPO_ROOT / "tcc-results-01"

CLASSES = [
    "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
    "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
]
SISTEMAS_DRIFT = ["S7", "S8"]

COR_ACERTO = "#4C72B0"   # azul
COR_ERRO = "#DD8452"     # laranja -- par seguro pra daltonismo (evita vermelho/verde)


def main():
    with open(_REPO_ROOT / "progresso" / "ground_truth_segunda_opiniao.csv", encoding="utf-8-sig") as f:
        gt_por_id = {
            row["prontuario_id"].strip(): {c for c in CLASSES if row[c] == "1"}
            for row in csv.DictReader(f)
        }

    with open(RESULTADOS_DIR / "progresso_cache.json", encoding="utf-8") as f:
        resultados = json.load(f)

    textos = carrega_texto_por_prontuario(CSV_ORIGINAL, "prontuario", "descricao")

    tamanhos_acerto, tamanhos_erro = [], []
    for pid, gt_set in gt_por_id.items():
        texto = textos.get(pid)
        if not texto:
            continue
        tamanho = len(texto)
        for sid in SISTEMAS_DRIFT:
            r = resultados.get(f"{sid}__{pid}")
            if not r or r.get("status") != "ok":
                continue
            previstas = set(r["doencas_previstas"])
            for classe in CLASSES:
                presente = classe in gt_set
                previu = classe in previstas
                acertou = presente == previu
                (tamanhos_acerto if acertou else tamanhos_erro).append(tamanho)

    fig, ax = plt.subplots(figsize=(9, 5))

    rng = np.random.default_rng(seed=7)
    y_acerto = 1 + rng.uniform(-0.15, 0.15, size=len(tamanhos_acerto))
    y_erro = 0 + rng.uniform(-0.15, 0.15, size=len(tamanhos_erro))

    ax.scatter(tamanhos_erro, y_erro, s=18, alpha=0.5, color=COR_ERRO,
               edgecolors="none", label=f"Drift erra (n={len(tamanhos_erro)})")
    ax.scatter(tamanhos_acerto, y_acerto, s=18, alpha=0.5, color=COR_ACERTO,
               edgecolors="none", label=f"Drift acerta (n={len(tamanhos_acerto)})")

    for y, dados, cor in [(0, tamanhos_erro, COR_ERRO), (1, tamanhos_acerto, COR_ACERTO)]:
        mediana = np.median(dados)
        ax.plot([mediana, mediana], [y - 0.22, y + 0.22], color=cor, linewidth=2.5)
        ax.annotate(f"mediana: {mediana:.0f} chars", (mediana, y + 0.26),
                    ha="center", fontsize=9, color=cor, fontweight="bold")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Drift erra", "Drift acerta"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Tamanho do texto do prontuário (caracteres)")
    ax.set_title("Acerto e erro do drift (S7/S8) por tamanho do prontuário\n"
                 "cada ponto é um par (prontuário, doença) avaliado")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    saida = RESULTADOS_DIR / "grafico_acertos_erros_drift.png"
    fig.savefig(saida, dpi=150)
    print(f"Gráfico salvo em: {saida}")
    print(f"Total de pares avaliados: {len(tamanhos_acerto) + len(tamanhos_erro)}")
    print(f"Acertos: {len(tamanhos_acerto)} | Erros: {len(tamanhos_erro)}")


if __name__ == "__main__":
    main()
