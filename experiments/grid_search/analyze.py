# %% [markdown]
# # Análise dos Resultados do Grid GraphRAG
# **Objetivo:** ler os `.json`/`.csv` já gerados por `run.py`
# (em `progresso_grid/` e `resultados_grid/`), reaplicar o parser de doença
# corrigido (`parser_doencas.extrai_doenca`) e recalcular consolidação,
# métricas, testes estatísticos e visualizações — **sem rodar nenhuma query**.
#
# Rode este script depois de qualquer atualização em `parser_doencas.py` para
# reprocessar os resultados existentes sem custo de API.

# %% [markdown]
# ## Célula 1 — Configuração

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.paths import (  # noqa: E402
    GROUND_TRUTH_CSV,
    PROGRESSO_GRID_DIR,
    RESULTADOS_GRID_DIR,
    bootstrap_sys_path,
)

bootstrap_sys_path(Path(__file__).resolve().parent)

from parser_doencas import extrai_doenca  # noqa: E402

# Caminhos vêm de src/core/paths.py (sobrescrevíveis via .env)
GROUND_TRUTH = GROUND_TRUTH_CSV
OUTPUT_DIR = RESULTADOS_GRID_DIR
PROGRESSO_DIR = PROGRESSO_GRID_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SISTEMAS = [
    {"id": "S1", "method": "basic", "community_level": 1},
    {"id": "S2", "method": "basic", "community_level": 2},
    {"id": "S3", "method": "local", "community_level": 1},
    {"id": "S4", "method": "local", "community_level": 2},
    {"id": "S5", "method": "global", "community_level": 1},
    {"id": "S6", "method": "global", "community_level": 2},
    {"id": "S7", "method": "drift", "community_level": 1},
    {"id": "S8", "method": "drift", "community_level": 2},
]

print("✔ Configuração carregada.")

# %% [markdown]
# ## Célula 2 — Reprocessa `doenca_prevista` a partir de `resposta_raw`
#
# Fonte de verdade: `progresso_grid/progresso_S{n}.json` (tem `resposta_raw`
# completo). Reescreve tanto o `.json` de progresso quanto o
# `resultado_S{n}.csv`, sem tocar nos campos que não dependem do parser
# (`status`, `duracao_s`, `returncode`, `timestamp`).

# %%
def arquivo_progresso(sistema_id: str) -> Path:
    return PROGRESSO_DIR / f"progresso_{sistema_id}.json"


def reprocessa_sistema(sistema_cfg: dict) -> pd.DataFrame:
    sid = sistema_cfg["id"]
    path = arquivo_progresso(sid)
    if not path.exists():
        print(f"  {sid}: progresso não encontrado, pulando.")
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        progresso = json.load(f)

    n_alterados = 0
    for pront_id, registro in progresso.items():
        if registro.get("status") != "ok":
            continue  # TIMEOUT / ERRO / SEM_TEXTO já têm doenca_prevista correta
        nova = extrai_doenca(registro.get("resposta_raw", ""))
        if nova != registro.get("doenca_prevista"):
            n_alterados += 1
        registro["doenca_prevista"] = nova

    with open(path, "w", encoding="utf-8") as f:
        json.dump(progresso, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(list(progresso.values()))
    df["method"] = sistema_cfg["method"]
    df["community_level"] = sistema_cfg["community_level"]
    df["sistema_id"] = sid
    df.to_csv(OUTPUT_DIR / f"resultado_{sid}.csv", index=False, encoding="utf-8-sig")

    print(f"  {sid}: {len(df)} registros | {n_alterados} doenca_prevista corrigidas")
    return df


print("Reprocessando sistemas com o parser corrigido:\n")
for s in SISTEMAS:
    reprocessa_sistema(s)

# %% [markdown]
# ## Célula 3 — Consolida todos os resultados

# %%
dfs = []
for sistema in SISTEMAS:
    path = OUTPUT_DIR / f'resultado_{sistema["id"]}.csv'
    if path.exists():
        df_s = pd.read_csv(path, dtype=str).fillna("")
        dfs.append(df_s)
        print(f'  {sistema["id"]} ({sistema["method"]} / level {sistema["community_level"]}): {len(df_s)} linhas')
    else:
        print(f'  {sistema["id"]}: ainda não executado')

if dfs:
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(OUTPUT_DIR / "resultados_consolidados.csv", index=False, encoding="utf-8-sig")
    print(f"\n✔ Consolidado salvo: {len(df_all)} linhas totais")

    gt = pd.read_csv(GROUND_TRUTH, dtype=str).fillna("")
    gt["prontuario_id"] = gt["prontuario_id"].str.strip()

    # Pivot: prontuário × sistema → doença prevista
    df_pivot = df_all.pivot_table(
        index="prontuario_id",
        columns="sistema_id",
        values="doenca_prevista",
        aggfunc="first",
    ).reset_index()

    gt_merge = gt[["prontuario_id", "doenca_principal"]]
    df_pivot = df_pivot.merge(gt_merge, on="prontuario_id", how="left")
    df_pivot.to_csv(OUTPUT_DIR / "pivot_sistemas.csv", index=False, encoding="utf-8-sig")
    print(f"Pivot salvo: {df_pivot.shape}")
    display(df_pivot.head(5))
else:
    print("Nenhum resultado disponível ainda.")

# %% [markdown]
# ## Célula 4 — Métricas por sistema

# %%
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef


def bootstrap_f1(y_true, y_pred, n=5000):
    scores = []
    yt, yp = np.array(y_true), np.array(y_pred)
    for _ in range(n):
        idx = np.random.choice(len(yt), len(yt), replace=True)
        try:
            scores.append(f1_score(yt[idx], yp[idx], average="macro", zero_division=0))
        except Exception:
            pass
    return np.percentile(scores, [2.5, 97.5])


resultados_metricas = []
colunas_sistemas = []

if "df_pivot" in dir() and len(df_pivot) > 0:
    df_eval = df_pivot[df_pivot["doenca_principal"] != "nenhuma"].copy()
    y_true = df_eval["doenca_principal"].tolist()

    colunas_sistemas = [s["id"] for s in SISTEMAS if s["id"] in df_pivot.columns]

    for sid in colunas_sistemas:
        cfg = next(s for s in SISTEMAS if s["id"] == sid)
        y_pred = df_eval[sid].fillna("nao_identificado").tolist()

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        ic = bootstrap_f1(y_true, y_pred)

        resultados_metricas.append({
            "Sistema": sid,
            "Modo": cfg["method"],
            "Community Level": cfg["community_level"],
            "Accuracy": round(acc, 4),
            "F1 macro": round(f1m, 4),
            "F1 weighted": round(f1w, 4),
            "Cohen's Kappa": round(kappa, 4),
            "MCC": round(mcc, 4),
            "F1 IC 2.5%": round(ic[0], 4),
            "F1 IC 97.5%": round(ic[1], 4),
        })

    df_metricas = pd.DataFrame(resultados_metricas).sort_values("F1 macro", ascending=False)
    df_metricas.to_csv(OUTPUT_DIR / "metricas_grid.csv", index=False, encoding="utf-8-sig")

    print("Métricas por sistema (ordenado por F1 macro):\n")
    display(df_metricas)
else:
    print("Execute a célula 3 primeiro.")

# %% [markdown]
# ## Célula 5 — Testes estatísticos (Friedman + Wilcoxon)

# %%
from itertools import combinations

from scipy.stats import friedmanchisquare, wilcoxon

if "df_pivot" in dir() and len(colunas_sistemas) >= 2:
    df_eval = df_pivot[df_pivot["doenca_principal"] != "nenhuma"].copy()
    y_true = df_eval["doenca_principal"].tolist()

    acertos = {}
    for sid in colunas_sistemas:
        y_pred = df_eval[sid].fillna("nao_identificado").tolist()
        acertos[sid] = [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]

    grupos = [acertos[sid] for sid in colunas_sistemas]
    if len(grupos) >= 3:
        stat_f, p_f = friedmanchisquare(*grupos)
        print("Friedman test:")
        print(f"  Estatística : {stat_f:.4f}")
        print(f"  p-valor     : {p_f:.4f}")
        print(f'  Conclusão   : {"Diferença significativa entre sistemas (p<0.05)" if p_f < 0.05 else "Sem diferença significativa (p>=0.05)"}')
    else:
        print("Friedman requer >= 3 sistemas.")

    print("\nWilcoxon post-hoc (com correção Bonferroni):")
    pares = list(combinations(colunas_sistemas, 2))
    alpha_corrigido = 0.05 / len(pares)
    print(f"  Alpha corrigido (Bonferroni): {alpha_corrigido:.4f}\n")

    resultados_wilcoxon = []
    for s1, s2 in pares:
        a1 = acertos[s1]
        a2 = acertos[s2]
        try:
            if a1 == a2:
                p_w = 1.0
            else:
                _, p_w = wilcoxon(a1, a2, zero_method="wilcox")
        except Exception:
            p_w = float("nan")

        sig = "✔ sig." if p_w < alpha_corrigido else "—"
        media1 = round(sum(a1) / len(a1), 4)
        media2 = round(sum(a2) / len(a2), 4)
        melhor = s1 if media1 > media2 else s2

        resultados_wilcoxon.append({
            "Par": f"{s1} vs {s2}",
            "Acc S1": media1,
            "Acc S2": media2,
            "Melhor": melhor,
            "p-valor": round(p_w, 4),
            "Significativo": sig,
        })

    df_wilcoxon = pd.DataFrame(resultados_wilcoxon)
    df_wilcoxon.to_csv(OUTPUT_DIR / "wilcoxon_posthoc.csv", index=False, encoding="utf-8-sig")
    display(df_wilcoxon)
else:
    print("Execute as células 3 e 4 primeiro.")

# %% [markdown]
# ## Célula 6 — Visualizações

# %%
try:
    _EM_NOTEBOOK = get_ipython() is not None  # noqa: F821
except NameError:
    _EM_NOTEBOOK = False

if not _EM_NOTEBOOK:
    import matplotlib
    matplotlib.use("Agg")  # roda como script puro: só salva PNG, não abre janela

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns


def _mostra(fig):
    """plt.show() bloqueia esperando a janela ser fechada quando rodado fora
    de um notebook/kernel interativo — aqui só exibe quando faz sentido."""
    if _EM_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
CORES = {"basic": "#2E86AB", "local": "#3BB273", "global": "#E84855", "drift": "#F4A261"}

if "df_metricas" in dir() and len(df_metricas) > 0:

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparação dos 8 Sistemas — GraphRAG Grid", fontsize=14, fontweight="bold")

    sistemas_labels = df_metricas["Sistema"].tolist()
    cores_barras = [CORES[m] for m in df_metricas["Modo"].tolist()]

    ax = axes[0, 0]
    ax.bar(sistemas_labels, df_metricas["F1 macro"], color=cores_barras, edgecolor="white")
    yerr_low = (df_metricas["F1 macro"] - df_metricas["F1 IC 2.5%"]).tolist()
    yerr_high = (df_metricas["F1 IC 97.5%"] - df_metricas["F1 macro"]).tolist()
    ax.errorbar(sistemas_labels, df_metricas["F1 macro"],
                yerr=[yerr_low, yerr_high], fmt="none", color="black", capsize=4)
    ax.set_title("F1 Macro com IC 95% (bootstrap)")
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(sistemas_labels, df_metricas["Accuracy"], color=cores_barras, edgecolor="white")
    ax.set_title("Accuracy por Sistema")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.bar(sistemas_labels, df_metricas["Cohen's Kappa"], color=cores_barras, edgecolor="white")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Cohen's Kappa por Sistema")
    ax.set_ylabel("Kappa")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(sistemas_labels, df_metricas["MCC"], color=cores_barras, edgecolor="white")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Matthews Correlation Coefficient (MCC)")
    ax.set_ylabel("MCC")
    ax.grid(axis="y", alpha=0.3)

    patches = [mpatches.Patch(color=c, label=m) for m, c in CORES.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, title="Modo de busca",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_grid_metricas.png", bbox_inches="tight")
    _mostra(fig)

    if "df_pivot" in dir():
        df_eval = df_pivot[df_pivot["doenca_principal"] != "nenhuma"].copy()
        y_true = df_eval["doenca_principal"].tolist()
        heatmap_data = pd.DataFrame(index=df_eval["prontuario_id"])
        for sid in colunas_sistemas:
            y_pred = df_eval[sid].fillna("nao_identificado").tolist()
            heatmap_data[sid] = [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]

        fig, ax = plt.subplots(figsize=(12, max(6, len(heatmap_data) // 8)))
        sns.heatmap(heatmap_data.T, cmap="RdYlGn", vmin=0, vmax=1,
                    cbar_kws={"label": "1=acerto / 0=erro"},
                    xticklabels=False, ax=ax)
        ax.set_title("Acertos por Prontuário × Sistema", fontsize=13, fontweight="bold")
        ax.set_ylabel("Sistema")
        ax.set_xlabel("Prontuários (141)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig_heatmap_acertos.png", bbox_inches="tight")
        _mostra(fig)
else:
    print("Execute as células 3 e 4 primeiro.")

# %% [markdown]
# ## Célula 7 — Concordância entre modos (Inter-Mode Agreement)

# %%
if "df_pivot" in dir() and len(colunas_sistemas) > 1:
    df_eval = df_pivot[df_pivot["doenca_principal"] != "nenhuma"].copy()

    def ima(row, cols):
        """Inter-Mode Agreement: proporção de modos que concordam na doença mais votada."""
        votos = [row[c] for c in cols if pd.notna(row[c]) and row[c] != ""]
        if not votos:
            return 0
        moda = max(set(votos), key=votos.count)
        return votos.count(moda) / len(votos)

    df_eval["ima"] = df_eval.apply(lambda r: ima(r, colunas_sistemas), axis=1)
    df_eval["acerto_algum"] = df_eval.apply(
        lambda r: any(r[s] == r["doenca_principal"] for s in colunas_sistemas), axis=1
    )
    df_eval["acerto_todos"] = df_eval.apply(
        lambda r: all(r[s] == r["doenca_principal"] for s in colunas_sistemas), axis=1
    )

    print("Inter-Mode Agreement (IMA):")
    print(f'  IMA médio          : {df_eval["ima"].mean():.4f}')
    print(f'  IMA mediano        : {df_eval["ima"].median():.4f}')
    print(f'  Todos concordam    : {df_eval["ima"].eq(1.0).sum()} prontuários ({df_eval["ima"].eq(1.0).mean()*100:.1f}%)')
    print(f'  Ao menos 1 acerta  : {df_eval["acerto_algum"].sum()} prontuários ({df_eval["acerto_algum"].mean()*100:.1f}%)')
    print(f'  Todos acertam      : {df_eval["acerto_todos"].sum()} prontuários ({df_eval["acerto_todos"].mean()*100:.1f}%)')
    print(f'  Todos erram        : {(~df_eval["acerto_algum"]).sum()} prontuários ({(~df_eval["acerto_algum"]).mean()*100:.1f}%)')

    from scipy.stats import pointbiserialr
    corr, p_corr = pointbiserialr(df_eval["ima"], df_eval["acerto_algum"].astype(int))
    print(f"\n  Correlação IMA × acerto: r={corr:.4f}, p={p_corr:.4f}")
    print("  (alta concordância entre modos prediz acerto?)")

    df_eval[["prontuario_id", "doenca_principal", "ima", "acerto_algum", "acerto_todos"] + colunas_sistemas].to_csv(
        OUTPUT_DIR / "analise_concordancia.csv", index=False, encoding="utf-8-sig"
    )
    print("\n✔ Análise de concordância salva.")
else:
    print("Execute as células anteriores primeiro.")
