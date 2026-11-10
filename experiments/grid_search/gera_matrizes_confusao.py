"""Gera matrizes de confusão (por doença, multilabel) para S1, S2, S3, S4, S7, S8
-- exclui S5 e S6 (global, ainda sem dados confiáveis/completos).

Cada sistema vira uma imagem com 8 subplots (uma matriz 2x2 por doença:
tem/não tem x previu/não previu), mais um resumo de métricas no título.

Fontes:
- S1-S4: tcc-results-01/progresso_cache_ling30_s1s6.json (nivelado, ling-3.0-flash,
  prompt reforçado)
- S7/S8: tcc-results-01/progresso_cache.json (run original, já em ling-3.0-flash)
- ground truth: progresso/ground_truth_segunda_opiniao.csv

Uso:
    uv run --no-sync python3 experiments/grid_search/gera_matrizes_confusao.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTADOS_DIR = _REPO_ROOT / "tcc-results-01"
SAIDA_DIR = RESULTADOS_DIR / "matrizes_confusao"
SAIDA_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
    "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
]
NOMES_EXIBICAO = {
    "cirrose_hepatica": "Cirrose hepática",
    "esquistossomose": "Esquistossomose",
    "hepatite_c": "Hepatite C",
    "hepatite_b": "Hepatite B",
    "esteatose_hepatica": "Esteatose hepática",
    "hipertensao_portal": "Hipertensão portal",
    "hemorragia_digestiva_alta": "Hemorragia digestiva alta",
    "pbe": "PBE",
}

# Sistema -> (id, método, nível, arquivo de progresso)
SISTEMAS = [
    ("S1", "basic", 1, "progresso_cache_ling30_s1s6.json"),
    ("S2", "basic", 2, "progresso_cache_ling30_s1s6.json"),
    ("S3", "local", 1, "progresso_cache_ling30_s1s6.json"),
    ("S4", "local", 2, "progresso_cache_ling30_s1s6.json"),
    ("S7", "drift", 1, "progresso_cache.json"),
    ("S8", "drift", 2, "progresso_cache.json"),
]


def carrega_ground_truth() -> dict[str, set[str]]:
    caminho = _REPO_ROOT / "progresso" / "ground_truth_segunda_opiniao.csv"
    with open(caminho, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {row["prontuario_id"].strip(): {c for c in CLASSES if row[c] == "1"} for row in rows}


def carrega_progresso(nome_arquivo: str) -> dict:
    caminho = RESULTADOS_DIR / nome_arquivo
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def gera_matriz_sistema(sid: str, method: str, level: int, progresso: dict, gt_por_id: dict) -> None:
    y_true_por_classe = {c: [] for c in CLASSES}
    y_pred_por_classe = {c: [] for c in CLASSES}
    n_avaliado = exact_matches = 0

    for pid, gt_set in gt_por_id.items():
        chave = f"{sid}__{pid}"
        if chave not in progresso or progresso[chave].get("status") != "ok":
            continue
        pred_set = set(progresso[chave].get("doencas_previstas", []))
        n_avaliado += 1
        if pred_set == gt_set:
            exact_matches += 1
        for c in CLASSES:
            y_true_por_classe[c].append(1 if c in gt_set else 0)
            y_pred_por_classe[c].append(1 if c in pred_set else 0)

    if n_avaliado == 0:
        print(f"⚠ {sid}: nenhum dado avaliável, pulando.")
        return

    fig, eixos = plt.subplots(2, 4, figsize=(18, 9))
    exact_pct = 100 * exact_matches / n_avaliado
    fig.suptitle(
        f"{sid} ({method} L{level}) — N={n_avaliado} — Exact-match: {exact_pct:.1f}%",
        fontsize=15, fontweight="bold",
    )

    for ax, c in zip(eixos.flat, CLASSES):
        y_true = y_true_por_classe[c]
        y_pred = y_pred_por_classe[c]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        precisao = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{NOMES_EXIBICAO[c]}\nP={precisao:.2f}  R={recall:.2f}", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Não previu", "Previu"], fontsize=8)
        ax.set_yticklabels(["Não tem", "Tem"], fontsize=8)
        ax.set_xlabel("Previsto", fontsize=8)
        ax.set_ylabel("Real", fontsize=8)

        cm_max = cm.max() if cm.max() > 0 else 1
        for i in range(2):
            for j in range(2):
                cor_texto = "white" if cm[i, j] > cm_max * 0.6 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color=cor_texto, fontsize=13, fontweight="bold")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    saida = SAIDA_DIR / f"matriz_confusao_{sid}.png"
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ {sid}: salvo em {saida}")


def main():
    gt_por_id = carrega_ground_truth()
    print(f"Ground truth: {len(gt_por_id)} prontuários\n")

    cache_por_arquivo: dict[str, dict] = {}
    for sid, method, level, arquivo in SISTEMAS:
        if arquivo not in cache_por_arquivo:
            cache_por_arquivo[arquivo] = carrega_progresso(arquivo)
        gera_matriz_sistema(sid, method, level, cache_por_arquivo[arquivo], gt_por_id)

    print(f"\nTodas as matrizes em: {SAIDA_DIR}")


if __name__ == "__main__":
    main()
