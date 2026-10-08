"""Gera UMA ÚNICA imagem com tabela de métricas (precisão/recall/F1 por doença
+ resumo geral) pra todos os sistemas, exceto S5/S6 (global, sem dados
confiáveis/completos ainda).

Fontes: mesmas de gera_matrizes_confusao.py.

Uso:
    uv run --no-sync python3 experiments/grid_search/gera_tabela_metricas.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTADOS_DIR = _REPO_ROOT / "tcc-results-01"
SAIDA_PNG = RESULTADOS_DIR / "tabela_metricas_sem_global.png"

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

SISTEMAS = [
    ("S1", "basic L1", "progresso_cache_ling30_s1s6.json"),
    ("S2", "basic L2", "progresso_cache_ling30_s1s6.json"),
    ("S3", "local L1", "progresso_cache_ling30_s1s6.json"),
    ("S4", "local L2", "progresso_cache_ling30_s1s6.json"),
    ("S7", "drift L1", "progresso_cache.json"),
    ("S8", "drift L2", "progresso_cache.json"),
]


def carrega_ground_truth() -> dict[str, set[str]]:
    caminho = _REPO_ROOT / "progresso" / "ground_truth_segunda_opiniao.csv"
    with open(caminho, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {row["prontuario_id"].strip(): {c for c in CLASSES if row[c] == "1"} for row in rows}


def carrega_progresso(nome_arquivo: str) -> dict:
    with open(RESULTADOS_DIR / nome_arquivo, encoding="utf-8") as f:
        return json.load(f)


def calcula_f1_por_doenca(sid: str, progresso: dict, gt_por_id: dict) -> tuple[dict, int, float]:
    por_classe = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
    n_avaliado = exact = 0
    for pid, gt_set in gt_por_id.items():
        chave = f"{sid}__{pid}"
        if chave not in progresso or progresso[chave].get("status") != "ok":
            continue
        pred_set = set(progresso[chave].get("doencas_previstas", []))
        n_avaliado += 1
        if pred_set == gt_set:
            exact += 1
        for c in CLASSES:
            in_gt, in_pred = c in gt_set, c in pred_set
            if in_gt and in_pred:
                por_classe[c]["tp"] += 1
            elif not in_gt and in_pred:
                por_classe[c]["fp"] += 1
            elif in_gt and not in_pred:
                por_classe[c]["fn"] += 1

    f1_por_classe = {}
    for c in CLASSES:
        tp, fp, fn = por_classe[c]["tp"], por_classe[c]["fp"], por_classe[c]["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1_por_classe[c] = (p, r, f1)

    exact_pct = exact / n_avaliado if n_avaliado else 0.0
    return f1_por_classe, n_avaliado, exact_pct


def main():
    gt_por_id = carrega_ground_truth()
    cache_por_arquivo: dict[str, dict] = {}

    dados = {}  # sid -> (f1_por_classe, n, exact_pct)
    for sid, _, arquivo in SISTEMAS:
        if arquivo not in cache_por_arquivo:
            cache_por_arquivo[arquivo] = carrega_progresso(arquivo)
        dados[sid] = calcula_f1_por_doenca(sid, cache_por_arquivo[arquivo], gt_por_id)

    linhas_doenca = CLASSES
    n_sistemas = len(SISTEMAS)
    n_linhas = len(linhas_doenca) + 2  # +1 separador, +1 exact-match

    fig, ax = plt.subplots(figsize=(3 + 2.6 * n_sistemas, 1.2 + 0.7 * n_linhas))
    ax.axis("off")

    cabecalho = [""] + [f"{sid}\n({metodo})" for sid, metodo, _ in SISTEMAS]
    linhas_tabela = []
    cores_tabela = []

    cmap = plt.cm.RdYlGn  # vermelho (ruim) -> verde (bom)

    for c in linhas_doenca:
        linha = [NOMES_EXIBICAO[c]]
        cor_linha = ["#f0f0f0"]
        for sid, _, _ in SISTEMAS:
            p, r, f1 = dados[sid][0][c]
            linha.append(f"F1={f1:.2f}\nP={p:.2f} R={r:.2f}")
            cor_linha.append(cmap(f1))
        linhas_tabela.append(linha)
        cores_tabela.append(cor_linha)

    # linha separadora em branco
    linhas_tabela.append([""] * (n_sistemas + 1))
    cores_tabela.append(["white"] * (n_sistemas + 1))

    # linha de resumo: exact-match
    linha_exact = ["Exact-match\n(todas as 8 corretas)"]
    cor_exact = ["#d9d9d9"]
    for sid, _, _ in SISTEMAS:
        _, n, exact_pct = dados[sid]
        linha_exact.append(f"{exact_pct*100:.1f}%\n(N={n})")
        cor_exact.append(cmap(exact_pct))
    linhas_tabela.append(linha_exact)
    cores_tabela.append(cor_exact)

    tabela = ax.table(
        cellText=linhas_tabela, colLabels=cabecalho, cellColours=cores_tabela,
        colColours=["#333333"] + ["#333333"] * n_sistemas,
        cellLoc="center", loc="center",
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1, 2.6)

    # cabeçalho em branco/negrito, primeira coluna alinhada à esquerda e em negrito
    for (row, col), cell in tabela.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
        if col == 0:
            cell.set_text_props(ha="left", fontweight="bold")
            cell.PAD = 0.02

    fig.suptitle(
        "Métricas por doença e por sistema (F1 / Precisão / Recall)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(SAIDA_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Tabela salva em: {SAIDA_PNG}")


if __name__ == "__main__":
    main()
