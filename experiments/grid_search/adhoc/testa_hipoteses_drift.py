"""Testa hipóteses concretas sobre por que `drift` (S7/S8) acerta doenças
que a maioria de `basic`/`local` (S1-S4) erra.

Grupo A: pares (prontuário, doença) onde a doença é real E drift acerta E
         a maioria de S1-S4 erra.
Grupo B: todos os outros pares (prontuário, doença) onde a doença é real
         (controle -- "casos normais", sem o padrão do grupo A).

4 hipóteses, cada uma com teste estatístico formal (não só comparação de
médias):

H1 (tamanho do texto): grupo A tem prontuários mais longos que o grupo B.
    Teste: Mann-Whitney U (não paramétrico, não assume distribuição normal).
H2 (termo literal ausente): no texto do grupo A, o termo literal da doença
    (ex.: "cirrose") aparece com menos frequência que no grupo B -- a
    evidência é mais indireta/sinônima.
    Teste: qui-quadrado de proporções.
H3 (nº de atendimentos): grupo A tem mais atendimentos (registros
    concatenados) que o grupo B -- a informação relevante está mais
    "diluída"/distante no documento.
    Teste: Mann-Whitney U.
H4 (entidade nunca extraída): a entidade canônica da doença nunca foi
    criada na indexação pro paciente do grupo A, mais frequentemente que
    no grupo B -- `local`/`basic` dependem de entidade extraída, `drift`
    não tanto.
    Teste: qui-quadrado de proporções.

Uso:
    uv run --no-sync python3 experiments/grid_search/testa_hipoteses_drift.py
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

sys.path.insert(0, str(_REPO_ROOT / "experiments" / "grid_search"))
from parser_doencas import MAPA_DOENCAS  # noqa: E402

RESULTADOS_DIR = _REPO_ROOT / "tcc-results-01"
INDICE_DIR = _REPO_ROOT / "experimentos" / "local copy" / "output"

CLASSES = [
    "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
    "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
]
OUTROS = ["S1", "S2", "S3", "S4"]
DRIFT = ["S7", "S8"]

# termos literais "canônicos" de cada doença, pra checagem de H2 (regex, sem acento)
TERMOS_LITERAIS = {
    "cirrose_hepatica": ["cirrose"],
    "esquistossomose": ["esquistossomose", "esquistossomótica"],
    "hepatite_c": ["hepatite c"],
    "hepatite_b": ["hepatite b"],
    "esteatose_hepatica": ["esteatose"],
    "hipertensao_portal": ["hipertensao portal"],
    "hemorragia_digestiva_alta": ["hemorragia digestiva alta", r"\bhda\b"],
    "pbe": ["peritonite bacteriana", r"\bpbe\b"],
}

# titulo de entidade "canonica" esperado por doenca, pra checagem H4
ENTIDADE_CANONICA = {
    "cirrose_hepatica": ["CIRROSE", "CIRROSE HEPÁTICA", "CIRROSE HEPATICA"],
    "hipertensao_portal": ["HIPERTENSÃO PORTAL", "SÍNDROME DE HIPERTENSÃO PORTAL",
                            "HIPERTENSAO PORTAL", "SINDROME DE HIPERTENSAO PORTAL"],
    "esquistossomose": ["ESQUISTOSSOMOSE"],
    "hepatite_b": ["HEPATITE B"],
    "hepatite_c": ["HEPATITE C"],
    "esteatose_hepatica": ["ESTEATOSE HEPÁTICA", "ESTEATOSE HEPATICA"],
    "hemorragia_digestiva_alta": ["HEMORRAGIA DIGESTIVA ALTA", "HDA"],
    "pbe": ["PERITONITE BACTERIANA ESPONTÂNEA", "PBE"],
}


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

    # separa grupo A (vitorias do drift) vs grupo B (demais positivos reais)
    grupo_a, grupo_b = [], []
    for pid, gt_set in gt_por_id.items():
        for c in CLASSES:
            if c not in gt_set:
                continue  # so interessa pares onde a doenca E REAL
            drift_ok = []
            for sid in DRIFT:
                chave = f"{sid}__{pid}"
                if chave in combinado and combinado[chave].get("status") == "ok":
                    previu = c in set(combinado[chave]["doencas_previstas"])
                    drift_ok.append(previu)
            if not drift_ok or not any(drift_ok):
                grupo_b.append((pid, c))
                continue
            outros_erraram = outros_avaliados = 0
            for sid in OUTROS:
                chave = f"{sid}__{pid}"
                if chave in combinado and combinado[chave].get("status") == "ok":
                    previu = c in set(combinado[chave]["doencas_previstas"])
                    outros_avaliados += 1
                    if not previu:
                        outros_erraram += 1
            if outros_avaliados == 0:
                continue
            if outros_erraram > outros_avaliados / 2:
                grupo_a.append((pid, c))
            else:
                grupo_b.append((pid, c))

    print(f"Grupo A (drift vence): {len(grupo_a)} pares (prontuário, doença)")
    print(f"Grupo B (controle/normal): {len(grupo_b)} pares\n")

    # ===== H1: tamanho do texto =====
    tam_a = [len(textos.get(pid, "")) for pid, _ in grupo_a]
    tam_b = [len(textos.get(pid, "")) for pid, _ in grupo_b]
    u1, p1 = stats.mannwhitneyu(tam_a, tam_b, alternative="greater")
    print("=== H1: grupo A tem prontuários mais LONGOS que grupo B ===")
    print(f"  mediana A: {pd.Series(tam_a).median():.0f} chars | mediana B: {pd.Series(tam_b).median():.0f} chars")
    print(f"  Mann-Whitney U, p-valor (A > B): {p1:.6f}  {'CONFIRMADA (p<0.05)' if p1 < 0.05 else 'não confirmada'}\n")

    # ===== H2: termo literal ausente =====
    def tem_termo_literal(pid: str, classe: str) -> bool:
        texto_norm = _normalize(textos.get(pid, ""))
        return any(re.search(t, texto_norm) for t in TERMOS_LITERAIS[classe])

    literal_a = [tem_termo_literal(pid, c) for pid, c in grupo_a]
    literal_b = [tem_termo_literal(pid, c) for pid, c in grupo_b]
    prop_a = sum(literal_a) / len(literal_a)
    prop_b = sum(literal_b) / len(literal_b)
    tabela = [[sum(literal_a), len(literal_a) - sum(literal_a)],
              [sum(literal_b), len(literal_b) - sum(literal_b)]]
    chi2, p2, _, _ = stats.chi2_contingency(tabela)
    print("=== H2: termo literal da doença aparece MENOS no texto do grupo A ===")
    print(f"  % com termo literal -- A: {prop_a*100:.1f}% | B: {prop_b*100:.1f}%")
    print(f"  qui-quadrado, p-valor: {p2:.6f}  "
          f"{'CONFIRMADA (p<0.05 e A < B)' if p2 < 0.05 and prop_a < prop_b else 'não confirmada'}\n")

    # ===== H3: numero de atendimentos =====
    def n_atendimentos(pid: str) -> int:
        return textos.get(pid, "").count("--- Atendimento ---") + 1

    atend_a = [n_atendimentos(pid) for pid, _ in grupo_a]
    atend_b = [n_atendimentos(pid) for pid, _ in grupo_b]
    u3, p3 = stats.mannwhitneyu(atend_a, atend_b, alternative="greater")
    print("=== H3: grupo A tem MAIS atendimentos (registros concatenados) que grupo B ===")
    print(f"  mediana A: {pd.Series(atend_a).median():.0f} | mediana B: {pd.Series(atend_b).median():.0f}")
    print(f"  Mann-Whitney U, p-valor (A > B): {p3:.6f}  {'CONFIRMADA (p<0.05)' if p3 < 0.05 else 'não confirmada'}\n")

    # ===== H4: entidade canonica nunca extraida =====
    print("Carregando índice pra H4 (pode levar alguns segundos)...")
    tu = pd.read_parquet(INDICE_DIR / "text_units.parquet")
    entities = pd.read_parquet(INDICE_DIR / "entities.parquet")

    cache_entidades_paciente: dict[str, set[str]] = {}

    def entidades_do_paciente(pid: str) -> set[str]:
        if pid in cache_entidades_paciente:
            return cache_entidades_paciente[pid]
        tu_p = tu[tu["text"].str.contains(pid, na=False, regex=False)]
        entity_ids = set()
        for ids in tu_p["entity_ids"]:
            if ids is not None:
                entity_ids.update(ids)
        titulos = set(entities[entities["id"].isin(entity_ids)]["title"]) if entity_ids else set()
        cache_entidades_paciente[pid] = titulos
        return titulos

    def entidade_canonica_existe(pid: str, classe: str) -> bool:
        if classe not in ENTIDADE_CANONICA:
            return True  # sem checagem definida, nao conta contra a hipotese
        titulos = entidades_do_paciente(pid)
        alvos = {_normalize(t) for t in ENTIDADE_CANONICA[classe]}
        return any(_normalize(t) in alvos for t in titulos)

    print("  (checando entidades grupo A...)")
    ent_a = [entidade_canonica_existe(pid, c) for pid, c in grupo_a]
    print("  (checando entidades grupo B...)")
    ent_b = [entidade_canonica_existe(pid, c) for pid, c in grupo_b]
    prop_ent_a = sum(ent_a) / len(ent_a)
    prop_ent_b = sum(ent_b) / len(ent_b)
    tabela2 = [[sum(ent_a), len(ent_a) - sum(ent_a)],
               [sum(ent_b), len(ent_b) - sum(ent_b)]]
    chi2b, p4, _, _ = stats.chi2_contingency(tabela2)
    print("\n=== H4: entidade CANÔNICA da doença NÃO foi extraída na indexação, mais no grupo A ===")
    print(f"  % com entidade canônica existente -- A: {prop_ent_a*100:.1f}% | B: {prop_ent_b*100:.1f}%")
    print(f"  qui-quadrado, p-valor: {p4:.6f}  "
          f"{'CONFIRMADA (p<0.05 e A < B)' if p4 < 0.05 and prop_ent_a < prop_ent_b else 'não confirmada'}\n")

    resultados = {
        "H1_tamanho_texto": {"mediana_A": float(pd.Series(tam_a).median()), "mediana_B": float(pd.Series(tam_b).median()), "p_valor": float(p1)},
        "H2_termo_literal": {"prop_A": prop_a, "prop_B": prop_b, "p_valor": float(p2)},
        "H3_n_atendimentos": {"mediana_A": float(pd.Series(atend_a).median()), "mediana_B": float(pd.Series(atend_b).median()), "p_valor": float(p3)},
        "H4_entidade_canonica": {"prop_A": prop_ent_a, "prop_B": prop_ent_b, "p_valor": float(p4)},
        "n_grupo_a": len(grupo_a), "n_grupo_b": len(grupo_b),
    }
    with open(RESULTADOS_DIR / "hipoteses_drift_resultado.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"Resultado salvo em: {RESULTADOS_DIR / 'hipoteses_drift_resultado.json'}")


if __name__ == "__main__":
    main()
