"""Testa hipóteses concretas sobre por que `drift` (S7/S8) ERRA (sobretudo
falso positivo) em casos que a maioria de `basic`/`local` (S1-S4) acerta --
o espelho de `testa_hipoteses_drift.py` (que testava o caso oposto: por que
`drift` ACERTA e os outros erram).

Grupo A: pares (prontuário, doença) onde a doença NÃO é real, E drift prevê
         presença (falso positivo), E a maioria de S1-S4 acerta (diz
         ausente corretamente).
Grupo B (controle): pares (prontuário, doença) onde a doença NÃO é real E
         `drift` TAMBÉM acerta (diz ausente corretamente) -- mesmo universo
         de "doença ausente", only variando se o drift alucinou ou não.

H1 (linguagem de risco/hipótese): nas linhas de falso positivo do grupo A,
   o drift usa linguagem de risco/hipótese ("risco", "possível",
   "sugestivo", "epidemiologia") mais que seria esperado -- medido como
   proporção DENTRO do grupo A (não há "linha" pra comparar no grupo B,
   já que lá o drift corretamente não menciona a doença).
H2 (prontuário longo também é fator de risco pra falso positivo): grupo A
   tem prontuários mais longos que grupo B (Mann-Whitney U).
H3 (termos-gatilho de confusão anatômica, só HDA): nos casos de FP de
   hemorragia_digestiva_alta, termos como "hemorroida"/"fissura anal"/
   "hematoquezia" aparecem mais que na população geral (qui-quadrado).
H4 (evidência não verificável no próprio texto do paciente): o trecho que
   o drift cita como prova da doença (entre aspas, na resposta) não
   aparece verbatim no texto do próprio paciente -- proporção DENTRO do
   grupo A, indício de alucinação/contaminação cruzada.

Uso:
    uv run --no-sync python3 experiments/grid_search/testa_hipoteses_drift_erro.py
"""

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.dataset.clinical_text import carrega_texto_por_prontuario  # noqa: E402
from src.core.paths import CSV_ORIGINAL  # noqa: E402

RESULTADOS_DIR = _REPO_ROOT / "tcc-results-01"

CLASSES = [
    "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
    "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
]
OUTROS = ["S1", "S2", "S3", "S4"]
DRIFT = ["S7", "S8"]

_TERMOS_RISCO = [
    "risco", "possivel", "provavel", "sugestivo", "sugere", "epidemiologia",
    "a esclarecer", "a investigar", "hipotese de", "suspeita",
]
_TERMOS_HDA_BAIXO = ["hemorroida", "fissura anal", "hematoquezia", "sangramento retal", "sangue nas fezes"]


def _normalize(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.lower()


def main():
    with open(_REPO_ROOT / "progresso" / "ground_truth_segunda_opiniao.csv", encoding="utf-8-sig") as f:
        gt_rows = list(csv.DictReader(f))
    gt_por_id = {row["prontuario_id"].strip(): {c for c in CLASSES if row[c] == "1"} for row in gt_rows}

    with open(RESULTADOS_DIR / "progresso_cache.json", encoding="utf-8") as f:
        original = json.load(f)
    with open(RESULTADOS_DIR / "progresso_cache_ling30_s1s6.json", encoding="utf-8") as f:
        ling30 = json.load(f)
    combinado = {}
    for k, r in ling30.items():
        if k.split("__")[0] in ("S1", "S2", "S3", "S4"):
            combinado[k] = r
    for k, r in original.items():
        if k.split("__")[0] in ("S7", "S8"):
            combinado[k] = r

    textos = carrega_texto_por_prontuario(CSV_ORIGINAL, "prontuario", "descricao")

    # monta grupo A (FP do drift, maioria dos outros acerta) e grupo B
    # (controle: doenca tambem ausente, mas drift TAMBEM acerta)
    grupo_a = []   # (pid, classe, linha_fp_do_drift)
    grupo_b = []   # (pid, classe)

    for pid, gt_set in gt_por_id.items():
        for c in CLASSES:
            if c in gt_set:
                continue  # so interessa pares onde a doenca NAO e real (universo do FP)

            linhas_fp_drift = []
            algum_fp_no_drift = False
            algum_tn_no_drift = False
            for sid in DRIFT:
                chave = f"{sid}__{pid}"
                r = combinado.get(chave)
                if not r or r.get("status") != "ok":
                    continue
                previu = c in set(r["doencas_previstas"])
                if previu:
                    algum_fp_no_drift = True
                    for linha in r["resposta_raw"].split("\n"):
                        if _normalize(c.replace("_", " ")) in _normalize(linha) or any(
                            _normalize(t) in _normalize(linha) for t in [c]
                        ):
                            linhas_fp_drift.append(linha.strip())
                else:
                    algum_tn_no_drift = True

            outros_acertaram = outros_avaliados = 0
            for sid in OUTROS:
                chave = f"{sid}__{pid}"
                r = combinado.get(chave)
                if not r or r.get("status") != "ok":
                    continue
                previu = c in set(r["doencas_previstas"])
                outros_avaliados += 1
                if not previu:
                    outros_acertaram += 1
            if outros_avaliados == 0:
                continue

            if algum_fp_no_drift and outros_acertaram > outros_avaliados / 2:
                grupo_a.append((pid, c, linhas_fp_drift))
            elif algum_tn_no_drift and not algum_fp_no_drift:
                grupo_b.append((pid, c))

    print(f"Grupo A (drift dá falso positivo, maioria dos outros acerta): {len(grupo_a)} pares")
    print(f"Grupo B (controle -- doença ausente, drift também acerta): {len(grupo_b)} pares\n")

    # ===== H1: linguagem de risco/hipótese nas linhas de FP (descritivo, dentro do grupo A) =====
    com_risco = 0
    for pid, c, linhas in grupo_a:
        texto_linhas = _normalize(" ".join(linhas))
        if any(t in texto_linhas for t in _TERMOS_RISCO):
            com_risco += 1
    print("=== H1: linhas de falso positivo do drift usam linguagem de risco/hipótese ===")
    print(f"  {com_risco}/{len(grupo_a)} casos ({com_risco/len(grupo_a)*100:.1f}%) têm pelo menos um termo de risco/hipótese na justificativa")
    print(f"  {'CONFIRMADA (proporção relevante)' if com_risco/len(grupo_a) > 0.15 else 'proporção baixa, hipótese fraca'}\n")

    # ===== H2: prontuário longo também é fator de risco pra FP =====
    tam_a = [len(textos.get(pid, "")) for pid, _, _ in grupo_a]
    tam_b = [len(textos.get(pid, "")) for pid, _ in grupo_b]
    u2, p2 = stats.mannwhitneyu(tam_a, tam_b, alternative="greater")
    print("=== H2: grupo A (FP) tem prontuários mais LONGOS que grupo B (TN correto) ===")
    print(f"  mediana A: {pd.Series(tam_a).median():.0f} chars | mediana B: {pd.Series(tam_b).median():.0f} chars")
    print(f"  Mann-Whitney U, p-valor (A > B): {p2:.6f}  {'CONFIRMADA (p<0.05)' if p2 < 0.05 else 'não confirmada'}\n")

    # ===== H3: termos-gatilho de confusão anatomica, so HDA =====
    hda_a = [(pid, c, l) for pid, c, l in grupo_a if c == "hemorragia_digestiva_alta"]
    hda_b = [(pid, c) for pid, c in grupo_b if c == "hemorragia_digestiva_alta"]

    def tem_termo_baixo(pid):
        texto_norm = _normalize(textos.get(pid, ""))
        return any(t in texto_norm for t in _TERMOS_HDA_BAIXO)

    com_termo_a = sum(1 for pid, _, _ in hda_a if tem_termo_baixo(pid))
    com_termo_b = sum(1 for pid, _ in hda_b if tem_termo_baixo(pid))
    tabela = [[com_termo_a, len(hda_a) - com_termo_a], [com_termo_b, len(hda_b) - com_termo_b]]
    print("=== H3: termos de sangramento BAIXO aparecem mais nos FP de HDA que no controle ===")
    print(f"  N grupo A (FP de HDA): {len(hda_a)} | N grupo B (TN correto de HDA): {len(hda_b)}")
    print(f"  % com termo de sangramento baixo -- A: {com_termo_a/len(hda_a)*100 if hda_a else 0:.1f}% | B: {com_termo_b/len(hda_b)*100 if hda_b else 0:.1f}%")
    if min(len(hda_a), len(hda_b)) >= 5 and all(sum(row) > 0 for row in tabela):
        chi2, p3, _, _ = stats.chi2_contingency(tabela)
        print(f"  qui-quadrado, p-valor: {p3:.6f}  {'CONFIRMADA (p<0.05)' if p3 < 0.05 else 'não confirmada'}\n")
    else:
        print("  amostra pequena demais pra qui-quadrado confiável -- reportado só descritivamente\n")

    # ===== H4: evidencia citada nao existe verbatim no texto do paciente =====
    nao_verificavel = 0
    total_com_citacao = 0
    exemplos = []
    for pid, c, linhas in grupo_a:
        texto_paciente_norm = _normalize(textos.get(pid, ""))
        for linha in linhas:
            citacoes = re.findall(r'"([^"]{15,})"', linha)
            for cit in citacoes:
                total_com_citacao += 1
                cit_norm = _normalize(cit)
                # normaliza espacos/pontuacao pra comparacao mais tolerante
                cit_norm_compacta = re.sub(r'\s+', ' ', cit_norm).strip()
                if cit_norm_compacta not in re.sub(r'\s+', ' ', texto_paciente_norm):
                    nao_verificavel += 1
                    if len(exemplos) < 5:
                        exemplos.append((pid, c, cit[:100]))
    print("=== H4: evidência citada entre aspas não existe verbatim no texto do próprio paciente ===")
    if total_com_citacao:
        print(f"  {nao_verificavel}/{total_com_citacao} citações ({nao_verificavel/total_com_citacao*100:.1f}%) não encontradas verbatim no texto do paciente")
        print(f"  {'CONFIRMADA (proporção relevante)' if nao_verificavel/total_com_citacao > 0.15 else 'proporção baixa'}")
    else:
        print("  nenhuma citação entre aspas encontrada nas linhas de FP")
    for pid, c, cit in exemplos:
        print(f"    exemplo: {pid} / {c} / \"{cit}\"")

    resultados = {
        "n_grupo_a": len(grupo_a), "n_grupo_b": len(grupo_b),
        "H1_linguagem_risco": {"com_risco": com_risco, "total": len(grupo_a)},
        "H2_tamanho_texto": {"mediana_A": float(pd.Series(tam_a).median()), "mediana_B": float(pd.Series(tam_b).median()), "p_valor": float(p2)},
        "H3_termos_hda_baixo": {"n_A": len(hda_a), "n_B": len(hda_b), "com_termo_A": com_termo_a, "com_termo_B": com_termo_b},
        "H4_evidencia_nao_verificavel": {"nao_verificavel": nao_verificavel, "total_citacoes": total_com_citacao},
    }
    with open(RESULTADOS_DIR / "hipoteses_drift_erro_resultado.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nResultado salvo em: {RESULTADOS_DIR / 'hipoteses_drift_erro_resultado.json'}")


if __name__ == "__main__":
    main()
