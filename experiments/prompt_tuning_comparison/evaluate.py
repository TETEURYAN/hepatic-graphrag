"""
Avaliação GraphRAG — Baseline vs Prompt-Tuned
==============================================
Entrada:
  - progresso/ground_truth_anotado.csv
  - dataset "tuned" gerado por build_datasets.py (--type prompt-tuning)
  - dataset "baseline" gerado por build_datasets.py (--type no-prompt-tuning), opcional

Saída (em resultados_avaliacao/):
  - relatorio_avaliacao.txt
  - metricas_por_classe_tuned.csv / metricas_por_classe_baseline.csv
  - matriz_confusao_tuned.csv / matriz_confusao_baseline.csv
  - comparacao_prontuario_a_prontuario.csv (se houver baseline)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.paths import (  # noqa: E402
    DATASET_BASE_CSV,
    DATASET_TUNED_CSV,
    GROUND_TRUTH_CSV,
    RESULTADOS_AVALIACAO_DIR,
)

# =============================================================================
# CONFIGURAÇÃO — DATASET_TUNED_CSV/DATASET_BASE_CSV vêm de src/core/paths.py,
# sobrescrevíveis via .env (o nome do arquivo inclui timestamp, então mude a
# variável de ambiente a cada rodada de build_datasets.py em vez de editar aqui)
# =============================================================================

OUTPUT_DIR = RESULTADOS_AVALIACAO_DIR

COL_PRONT          = "prontuario_id"
COL_CONDICAO       = "condicao_normalizada"

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

LABELS_PT = {
    "cirrose_hepatica":          "Cirrose hepática",
    "esquistossomose":           "Esquistossomose",
    "hepatite_c":                "Hepatite C",
    "hepatite_b":                "Hepatite B",
    "esteatose_hepatica":        "Esteatose hepática",
    "hipertensao_portal":        "Hipertensão portal",
    "hemorragia_digestiva_alta": "Hemorragia digestiva alta",
    "pbe":                       "PBE",
}

MAPA_NORMALIZACAO = {
    "CIRROSE": "cirrose_hepatica",
    "CIRROSE HEPATICA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA DESCOMPENSADA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA ALCOÓLICA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA VIRAL C": "cirrose_hepatica",
    "CIRROSE MICRONODULAR": "cirrose_hepatica",
    "CIRROSE DESCOMPENSADA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA CRÔNICA": "cirrose_hepatica",
    "HIPERTENSAO PORTAL": "hipertensao_portal",
    "HIPERTENSÃO PORTAL": "hipertensao_portal",
    "HIPERTENSÃO PORTA": "hipertensao_portal",
    "SINDROME DE HIPERTENSAO PORTAL": "hipertensao_portal",
    "SÍNDROME DE HIPERTENSÃO PORTAL": "hipertensao_portal",
    "SÍNDROME DA HIPERTENSÃO PORTAL": "hipertensao_portal",
    "HIPERTENSÃO PORTAL NÃO CIRRÓTICA": "hipertensao_portal",
    "HIPERTENSÃO PORTAL NÃO-CIRRÓTICA": "hipertensao_portal",
    "EHE/HP": "hipertensao_portal",
    "ESQUISTOSSOMOSE": "esquistossomose",
    "EQUISTOSSOMOSE": "esquistossomose",
    "ESQUISTOSSOMOSE HEPATOESPLENICA": "esquistossomose",
    "ESQUISTOSSOMOSE HEPATOESPLÊNICA": "esquistossomose",
    "ESQUISTOSSOMOSE HEPATOESPLÊNICA (EHE)": "esquistossomose",
    "ESQUISTOSSOMOSE MANSONI": "esquistossomose",
    "ESQUISTOSSOMOSE MANSÔNICA": "esquistossomose",
    "EHE": "esquistossomose",
    "HEPATITE C": "hepatite_c",
    "HEPATITE C (HCV)": "hepatite_c",
    "HEPATITE C CRÔNICA": "hepatite_c",
    "HEPATITE CRÔNICA VIRAL": "hepatite_c",
    "HEPATITE B": "hepatite_b",
    "HEPATITE B (HBV)": "hepatite_b",
    "HEPATITE B CRONICA": "hepatite_b",
    "HEPATITE B INATIVA": "hepatite_b",
    "HEPATITE B RESOLVIDA": "hepatite_b",
    "HEPATITE CRÔNICA B": "hepatite_b",
    "HEPATITE CRÔNICA VIRAL B": "hepatite_b",
    "VÍRUS B (HEPATITE B)": "hepatite_b",
    "ESTEATOSE": "esteatose_hepatica",
    "ESTEATOSE HEPATICA": "esteatose_hepatica",
    "ESTEATOSE HEPÁTICA": "esteatose_hepatica",
    "ESTEATOSE HEPATICA DIFUSA": "esteatose_hepatica",
    "DHGNA": "esteatose_hepatica",
    "NAFLD": "esteatose_hepatica",
    "MASLD": "esteatose_hepatica",
    "MASH": "esteatose_hepatica",
    "NASH": "esteatose_hepatica",
    "DOENÇA HEPÁTICA GORDUROSA NÃO ALCOÓLICA": "esteatose_hepatica",
    "DOENÇA HEPÁTICA GORDUROSA NÃO ALCOÓLICA (DHGNA)": "esteatose_hepatica",
    "DOENÇA HEPÁTICA ESTEATÓTICA METABÓLICA": "esteatose_hepatica",
    "DOENÇA HEPÁTICA ESTEATÓTICA METABÓLICA (MASH)": "esteatose_hepatica",
    "ESTEATO-HEPATITE NÃO ALCOÓLICA": "esteatose_hepatica",
    "ESTEATO-HEPATITE NÃO ALCOÓLICA (NASH)": "esteatose_hepatica",
    "ESTEATOHEPATITE": "esteatose_hepatica",
    "ESTEATOHEPATITE NÃO ALCOÓLICA": "esteatose_hepatica",
    "HDA": "hemorragia_digestiva_alta",
    "HEMORRAGIA DIGESTIVA ALTA": "hemorragia_digestiva_alta",
    "HDA (HEMORRAGIA DIGESTIVA ALTA)": "hemorragia_digestiva_alta",
    "HDA VARICOSA": "hemorragia_digestiva_alta",
    "HEMORRAGIA DIGESTIVA ALTA VARICOSA": "hemorragia_digestiva_alta",
    "HEMORRAGIA VARICOSA": "hemorragia_digestiva_alta",
    "PBE": "pbe",
    "PERITONITE BACTERIANA ESPONTÂNEA": "pbe",
    "PBE (PERITONITE BACTERIANA ESPONTÂNEA)": "pbe",
    "PERITONITE BACTERIANA ESPONTANEA": "pbe",
}

# =============================================================================
# FUNÇÕES
# =============================================================================

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normaliza_dataset(csv_path: str) -> pd.DataFrame:
    """Carrega dataset GraphRAG e retorna doença principal por prontuário."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df[COL_PRONT] = df[COL_PRONT].str.strip()
    df["classe"] = df[COL_CONDICAO].str.upper().str.strip().map(MAPA_NORMALIZACAO)

    # Doença principal = classe mais frequente por prontuário
    principal = (
        df[df["classe"].notna()]
        .groupby(COL_PRONT)["classe"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"classe": "doenca_prevista"})
    )
    return principal


def bootstrap_ic(y_true, y_pred, metric_fn, n=5000, ci=0.95):
    """Bootstrap para intervalo de confiança de uma métrica."""
    scores = []
    n_samples = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(n):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    alpha = (1 - ci) / 2
    return np.percentile(scores, [alpha * 100, (1 - alpha) * 100])


def avalia_sistema(gt: pd.DataFrame, previsto: pd.DataFrame, nome: str):
    """
    Avalia um sistema contra o ground truth.
    gt: DataFrame com colunas [prontuario_id, doenca_principal]
    previsto: DataFrame com colunas [prontuario_id, doenca_prevista]
    """
    # Merge — apenas prontuários presentes em ambos
    merged = gt.merge(previsto, on=COL_PRONT, how="left")
    merged["doenca_prevista"] = merged["doenca_prevista"].fillna("nenhuma")

    # Exclui casos onde ground truth é "nenhuma"
    # (o sistema identificou algo mas não era doença prioritária — FP puro)
    # Mantém para análise de FP mas separa para métricas de classificação
    df_fp_puro = merged[merged["doenca_principal"] == "nenhuma"].copy()
    df_eval    = merged[merged["doenca_principal"] != "nenhuma"].copy()

    y_true = df_eval["doenca_principal"].tolist()
    y_pred = df_eval["doenca_prevista"].tolist()

    # Classes presentes no ground truth
    classes_presentes = sorted(set(y_true))

    # Métricas globais
    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_wtd, r_wtd, f1_wtd, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Bootstrap IC para F1 macro
    def f1_macro_fn(yt, yp):
        _, _, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        return f1

    ic_f1 = bootstrap_ic(y_true, y_pred, f1_macro_fn)

    # Métricas por classe
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred,
        labels=classes_presentes,
        average=None,
        zero_division=0
    )

    df_por_classe = pd.DataFrame({
        "classe":    [LABELS_PT.get(c, c) for c in classes_presentes],
        "precision": p_cls,
        "recall":    r_cls,
        "f1":        f1_cls,
        "suporte":   sup_cls.astype(int),
    })

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=classes_presentes)
    df_cm = pd.DataFrame(
        cm,
        index=[LABELS_PT.get(c, c) for c in classes_presentes],
        columns=[LABELS_PT.get(c, c) for c in classes_presentes],
    )

    # Análise de falsos positivos puros
    n_fp_puro = len(df_fp_puro)
    fp_puro_previstos = df_fp_puro["doenca_prevista"].value_counts()

    # Análise de erros (FP e FN por classe)
    df_erros = df_eval[df_eval["doenca_principal"] != df_eval["doenca_prevista"]].copy()

    return {
        "nome":             nome,
        "n_avaliados":      len(df_eval),
        "n_fp_puro":        n_fp_puro,
        "accuracy":         acc,
        "kappa":            kappa,
        "f1_macro":         f1_macro,
        "f1_macro_ic":      ic_f1,
        "f1_weighted":      f1_wtd,
        "precision_macro":  p_macro,
        "recall_macro":     r_macro,
        "precision_wtd":    p_wtd,
        "recall_wtd":       r_wtd,
        "df_por_classe":    df_por_classe,
        "df_cm":            df_cm,
        "df_erros":         df_erros,
        "fp_puro_previstos": fp_puro_previstos,
        "y_true":           y_true,
        "y_pred":           y_pred,
    }


def imprime_resultado(r: dict, saida: list):
    bloco = f"""
{'='*60}
SISTEMA: {r['nome']}
{'='*60}
Prontuários avaliados      : {r['n_avaliados']}
Falsos positivos puros     : {r['n_fp_puro']} (sistema identificou doença inexistente)

--- Métricas globais ---
Accuracy                   : {r['accuracy']:.4f}  ({r['accuracy']*100:.1f}%)
Cohen's Kappa              : {r['kappa']:.4f}
F1 macro                   : {r['f1_macro']:.4f}  IC 95%: [{r['f1_macro_ic'][0]:.4f}, {r['f1_macro_ic'][1]:.4f}]
F1 weighted                : {r['f1_weighted']:.4f}
Precision macro            : {r['precision_macro']:.4f}
Recall macro               : {r['recall_macro']:.4f}

--- Por classe ---
{r['df_por_classe'].to_string(index=False)}

--- Falsos positivos puros (sistema extraiu, não havia doença) ---
{r['fp_puro_previstos'].to_string() if len(r['fp_puro_previstos']) > 0 else '  nenhum'}

--- Erros de classificação (confundiu uma classe com outra) ---
Total de erros: {len(r['df_erros'])}
"""
    if len(r["df_erros"]) > 0:
        bloco += r["df_erros"][
            [COL_PRONT, "doenca_principal", "doenca_prevista"]
        ].head(20).to_string(index=False)

    saida.append(bloco)
    print(bloco)


# =============================================================================
# 1. CARREGA GROUND TRUTH
# =============================================================================

print("Carregando ground truth...")
gt = pd.read_csv(GROUND_TRUTH_CSV, dtype=str).fillna("")
gt[COL_PRONT] = gt[COL_PRONT].str.strip()

print(f"  Total anotado        : {len(gt)}")
print(f"  Com doença válida    : {(gt['doenca_principal'] != 'nenhuma').sum()}")
print(f"  Sem doença (nenhuma) : {(gt['doenca_principal'] == 'nenhuma').sum()}")

# =============================================================================
# 2. CARREGA E NORMALIZA DATASETS
# =============================================================================

print("\nNormalizando datasets...")
prev_tuned = normaliza_dataset(DATASET_TUNED_CSV)
print(f"  Prompt-tuned: {len(prev_tuned)} prontuários com previsão")

prev_base = None
if DATASET_BASE_CSV:
    prev_base = normaliza_dataset(DATASET_BASE_CSV)
    print(f"  Baseline    : {len(prev_base)} prontuários com previsão")

# =============================================================================
# 3. AVALIAÇÃO
# =============================================================================

saida = ["RELATÓRIO DE AVALIAÇÃO — GraphRAG Hepatologia\n"]

print("\nAvaliando sistema prompt-tuned...")
res_tuned = avalia_sistema(gt, prev_tuned, "Prompt-Tuned")
imprime_resultado(res_tuned, saida)

res_base = None
if prev_base is not None:
    print("\nAvaliando sistema baseline...")
    res_base = avalia_sistema(gt, prev_base, "Baseline")
    imprime_resultado(res_base, saida)

# =============================================================================
# 4. COMPARAÇÃO BASELINE vs TUNED (se disponível)
# =============================================================================

if res_base is not None:
    print("\nComparando sistemas...")

    # Alinha prontuários
    merged_comp = gt[[COL_PRONT, "doenca_principal"]].merge(
        prev_tuned.rename(columns={"doenca_prevista": "prev_tuned"}),
        on=COL_PRONT, how="left"
    ).merge(
        prev_base.rename(columns={"doenca_prevista": "prev_base"}),
        on=COL_PRONT, how="left"
    )
    merged_comp = merged_comp[merged_comp["doenca_principal"] != "nenhuma"]
    merged_comp["prev_tuned"] = merged_comp["prev_tuned"].fillna("nenhuma")
    merged_comp["prev_base"]  = merged_comp["prev_base"].fillna("nenhuma")

    # Acertos por prontuário
    merged_comp["acerto_tuned"] = (merged_comp["doenca_principal"] == merged_comp["prev_tuned"]).astype(int)
    merged_comp["acerto_base"]  = (merged_comp["doenca_principal"] == merged_comp["prev_base"]).astype(int)

    # McNemar
    from statsmodels.stats.contingency_tables import mcnemar
    b = ((merged_comp["acerto_tuned"] == 1) & (merged_comp["acerto_base"] == 0)).sum()
    c = ((merged_comp["acerto_tuned"] == 0) & (merged_comp["acerto_base"] == 1)).sum()
    tab = [[0, b], [c, 0]]
    try:
        mc = mcnemar([[merged_comp["acerto_tuned"].sum(), b], [c, merged_comp["acerto_base"].sum()]])
        p_mcnemar = mc.pvalue
    except Exception:
        p_mcnemar = float("nan")

    # Wilcoxon sobre F1 por classe
    f1_tuned_cls = res_tuned["df_por_classe"]["f1"].values
    f1_base_cls  = res_base["df_por_classe"]["f1"].values
    try:
        if len(f1_tuned_cls) == len(f1_base_cls) and not np.all(f1_tuned_cls == f1_base_cls):
            stat_w, p_wilcoxon = wilcoxon(f1_tuned_cls, f1_base_cls)
        else:
            p_wilcoxon = float("nan")
    except Exception:
        p_wilcoxon = float("nan")

    bloco_comp = f"""
{'='*60}
COMPARAÇÃO DIRETA — Baseline vs Prompt-Tuned
{'='*60}
                        Baseline     Tuned       Delta
Accuracy              : {res_base['accuracy']:.4f}       {res_tuned['accuracy']:.4f}      {res_tuned['accuracy']-res_base['accuracy']:+.4f}
F1 macro              : {res_base['f1_macro']:.4f}       {res_tuned['f1_macro']:.4f}      {res_tuned['f1_macro']-res_base['f1_macro']:+.4f}
F1 weighted           : {res_base['f1_weighted']:.4f}       {res_tuned['f1_weighted']:.4f}      {res_tuned['f1_weighted']-res_base['f1_weighted']:+.4f}
Cohen's Kappa         : {res_base['kappa']:.4f}       {res_tuned['kappa']:.4f}      {res_tuned['kappa']-res_base['kappa']:+.4f}
FP puros              : {res_base['n_fp_puro']}           {res_tuned['n_fp_puro']}

Testes estatísticos:
  McNemar p-valor     : {p_mcnemar:.4f}  {'(significativo p<0.05)' if p_mcnemar < 0.05 else '(não significativo)'}
  Wilcoxon p-valor    : {p_wilcoxon:.4f}  {'(significativo p<0.05)' if p_wilcoxon < 0.05 else '(não significativo)'}

F1 macro IC 95%:
  Baseline            : [{res_base['f1_macro_ic'][0]:.4f}, {res_base['f1_macro_ic'][1]:.4f}]
  Prompt-Tuned        : [{res_tuned['f1_macro_ic'][0]:.4f}, {res_tuned['f1_macro_ic'][1]:.4f}]
"""
    saida.append(bloco_comp)
    print(bloco_comp)

    merged_comp.to_csv(OUTPUT_DIR / "comparacao_prontuario_a_prontuario.csv",
                       index=False, encoding="utf-8-sig")

# =============================================================================
# 5. EXPORTA
# =============================================================================

# Relatório texto
with open(OUTPUT_DIR / "relatorio_avaliacao.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(saida))

# Métricas por classe
res_tuned["df_por_classe"].to_csv(
    OUTPUT_DIR / "metricas_por_classe_tuned.csv", index=False, encoding="utf-8-sig"
)
res_tuned["df_cm"].to_csv(
    OUTPUT_DIR / "matriz_confusao_tuned.csv", encoding="utf-8-sig"
)

if res_base:
    res_base["df_por_classe"].to_csv(
        OUTPUT_DIR / "metricas_por_classe_baseline.csv", index=False, encoding="utf-8-sig"
    )
    res_base["df_cm"].to_csv(
        OUTPUT_DIR / "matriz_confusao_baseline.csv", encoding="utf-8-sig"
    )

print(f"\n✔ Resultados salvos em: {OUTPUT_DIR}")
print("  relatorio_avaliacao.txt")
print("  metricas_por_classe_tuned.csv")
print("  matriz_confusao_tuned.csv")
if res_base:
    print("  metricas_por_classe_baseline.csv")
    print("  matriz_confusao_baseline.csv")
    print("  comparacao_prontuario_a_prontuario.csv")