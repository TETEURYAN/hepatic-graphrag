"""
Script de Anotação Interativa — Ground Truth Multirrótulo → Single-label
=========================================================================
Fluxo:
  1. Carrega os 145 prontuários identificados pelo GraphRAG
  2. Para cada prontuário, exibe o texto original e o que o sistema extraiu
  3. Você anota quais das 8 doenças estão presentes (0/1)
  4. Aplica hierarquia causal para definir a doença principal (single-label)
  5. Salva progresso a cada anotação (pode interromper e continuar)

Uso:
  python anotar_prontuarios.py
"""

import os
import csv
import json
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.paths import (  # noqa: E402
    ANOTACAO_PROGRESSO,
    CSV_ORIGINAL,
    DATASET_TUNED_CSV,
    GROUND_TRUTH_CSV,
)

# =============================================================================
# CONFIGURAÇÃO — caminhos vêm de src/core/paths.py (sobrescrevíveis via .env)
# =============================================================================

CSV_PRONTUARIOS   = CSV_ORIGINAL         # CSV original (colunas: prontuario, descricao, ...)
COL_ID            = "prontuario"        # coluna identificadora
COL_TEXTO         = "descricao"         # coluna com o texto clínico

CSV_GRAPHRAG      = DATASET_TUNED_CSV   # dataset gerado por build_datasets.py (--type prompt-tuning)

ARQUIVO_PROGRESSO = ANOTACAO_PROGRESSO  # permite pausar e continuar
ARQUIVO_SAIDA     = GROUND_TRUTH_CSV    # este script é quem gera esse arquivo

# =============================================================================
# CLASSES E HIERARQUIA
# =============================================================================

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

LABELS = {
    "cirrose_hepatica":          "Cirrose hepática",
    "esquistossomose":           "Esquistossomose",
    "hepatite_c":                "Hepatite C",
    "hepatite_b":                "Hepatite B",
    "esteatose_hepatica":        "Esteatose hepática",
    "hipertensao_portal":        "Hipertensão portal",
    "hemorragia_digestiva_alta": "Hemorragia digestiva alta (HDA) — motivo principal?",
    "pbe":                       "PBE — motivo principal?",
}

# Hierarquia causal: posição 0 = maior prioridade
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

# Normalização para identificar prontuários com doença prioritária
MAPA_NORMALIZACAO = {
    # Cirrose
    "CIRROSE": "cirrose_hepatica",
    "CIRROSE HEPATICA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA DESCOMPENSADA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA ALCOÓLICA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA VIRAL C": "cirrose_hepatica",
    "CIRROSE MICRONODULAR": "cirrose_hepatica",
    "CIRROSE DESCOMPENSADA": "cirrose_hepatica",
    "CIRROSE HEPÁTICA CRÔNICA": "cirrose_hepatica",
    # Hipertensão portal
    "HIPERTENSAO PORTAL": "hipertensao_portal",
    "HIPERTENSÃO PORTAL": "hipertensao_portal",
    "HIPERTENSÃO PORTA": "hipertensao_portal",
    "SINDROME DE HIPERTENSAO PORTAL": "hipertensao_portal",
    "SÍNDROME DE HIPERTENSÃO PORTAL": "hipertensao_portal",
    "SÍNDROME DA HIPERTENSÃO PORTAL": "hipertensao_portal",
    "HIPERTENSÃO PORTAL NÃO CIRRÓTICA": "hipertensao_portal",
    "HIPERTENSÃO PORTAL NÃO-CIRRÓTICA": "hipertensao_portal",
    "EHE/HP": "hipertensao_portal",
    # Esquistossomose
    "ESQUISTOSSOMOSE": "esquistossomose",
    "EQUISTOSSOMOSE": "esquistossomose",
    "ESQUISTOSSOMOSE HEPATOESPLENICA": "esquistossomose",
    "ESQUISTOSSOMOSE HEPATOESPLÊNICA": "esquistossomose",
    "ESQUISTOSSOMOSE HEPATOESPLÊNICA (EHE)": "esquistossomose",
    "ESQUISTOSSOMOSE MANSONI": "esquistossomose",
    "ESQUISTOSSOMOSE MANSÔNICA": "esquistossomose",
    "EHE": "esquistossomose",
    # Hepatite C
    "HEPATITE C": "hepatite_c",
    "HEPATITE C (HCV)": "hepatite_c",
    "HEPATITE C CRÔNICA": "hepatite_c",
    "HEPATITE CRÔNICA VIRAL": "hepatite_c",
    # Hepatite B
    "HEPATITE B": "hepatite_b",
    "HEPATITE B (HBV)": "hepatite_b",
    "HEPATITE B CRONICA": "hepatite_b",
    "HEPATITE B INATIVA": "hepatite_b",
    "HEPATITE B RESOLVIDA": "hepatite_b",
    "HEPATITE CRÔNICA B": "hepatite_b",
    "HEPATITE CRÔNICA VIRAL B": "hepatite_b",
    "VÍRUS B (HEPATITE B)": "hepatite_b",
    # Esteatose
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
    # HDA
    "HDA": "hemorragia_digestiva_alta",
    "HEMORRAGIA DIGESTIVA ALTA": "hemorragia_digestiva_alta",
    "HDA (HEMORRAGIA DIGESTIVA ALTA)": "hemorragia_digestiva_alta",
    "HDA VARICOSA": "hemorragia_digestiva_alta",
    "HEMORRAGIA DIGESTIVA ALTA VARICOSA": "hemorragia_digestiva_alta",
    "HEMORRAGIA VARICOSA": "hemorragia_digestiva_alta",
    # PBE
    "PBE": "pbe",
    "PERITONITE BACTERIANA ESPONTÂNEA": "pbe",
    "PBE (PERITONITE BACTERIANA ESPONTÂNEA)": "pbe",
    "PERITONITE BACTERIANA ESPONTANEA": "pbe",
}

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def limpa_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def aplica_hierarquia(anotacao: dict) -> str:
    """Retorna a doença principal baseada na hierarquia causal."""
    for doenca in HIERARQUIA:
        if anotacao.get(doenca) == 1:
            return doenca
    return "nenhuma"

def carrega_progresso() -> dict:
    if os.path.exists(ARQUIVO_PROGRESSO):
        with open(ARQUIVO_PROGRESSO, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salva_progresso(progresso: dict):
    with open(ARQUIVO_PROGRESSO, "w", encoding="utf-8") as f:
        json.dump(progresso, f, ensure_ascii=False, indent=2)

def salva_csv(progresso: dict):
    """Salva o estado atual como CSV final."""
    linhas = []
    for pront_id, dados in progresso.items():
        linha = {"prontuario_id": pront_id}
        for c in CLASSES:
            linha[c] = dados["anotacao"].get(c, 0)
        linha["doenca_principal"]   = dados["doenca_principal"]
        linha["confianca"]          = dados.get("confianca", "")
        linha["comentario"]         = dados.get("comentario", "")
        linha["sistema_extraiu"]    = dados.get("sistema_extraiu", "")
        linha["timestamp"]          = dados.get("timestamp", "")
        linhas.append(linha)

    df = pd.DataFrame(linhas)
    colunas = (
        ["prontuario_id"] + CLASSES +
        ["doenca_principal", "confianca", "comentario",
         "sistema_extraiu", "timestamp"]
    )
    df[colunas].to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")

def formata_texto(texto: str, largura: int = 100) -> str:
    """Quebra o texto em linhas para exibição no terminal."""
    if not texto:
        return "(texto não disponível)"
    linhas = []
    for paragrafo in texto.split("\n"):
        while len(paragrafo) > largura:
            linhas.append(paragrafo[:largura])
            paragrafo = paragrafo[largura:]
        linhas.append(paragrafo)
    return "\n".join(linhas[:80])  # limita a 80 linhas

def get_input_binario(label: str) -> int:
    """Pede 0 ou 1 para o anotador."""
    while True:
        resp = input(f"  {'[0/1]':<6} {label}: ").strip()
        if resp in ("0", "1"):
            return int(resp)
        if resp == "":
            return 0
        print("         Digite 0 (ausente) ou 1 (presente). Enter = 0.")

def get_confianca() -> str:
    while True:
        resp = input("\n  Confiança na anotação [A]lta / [M]édia / [B]aixa: ").strip().upper()
        if resp in ("A", "M", "B", "ALTA", "MÉDIA", "MEDIA", "BAIXA"):
            mapa = {"A": "ALTA", "M": "MÉDIA", "B": "BAIXA",
                    "ALTA": "ALTA", "MÉDIA": "MÉDIA", "MEDIA": "MÉDIA", "BAIXA": "BAIXA"}
            return mapa[resp]
        print("         Digite A, M ou B.")

# =============================================================================
# CARGA DOS DADOS
# =============================================================================

print("Carregando dados...")

# Prontuários originais
df_pront = pd.read_csv(CSV_PRONTUARIOS, dtype=str).fillna("")
df_pront[COL_ID] = df_pront[COL_ID].str.strip()
texto_por_id = dict(zip(df_pront[COL_ID], df_pront[COL_TEXTO]))

# Dataset GraphRAG
df_graph = pd.read_csv(CSV_GRAPHRAG, dtype=str).fillna("")
df_graph["prontuario_id"] = df_graph["prontuario_id"].str.strip()

# Identifica os 145 prontuários com doença prioritária
df_graph["classe"] = df_graph["condicao_normalizada"].str.upper().str.strip().map(MAPA_NORMALIZACAO)
pront_com_doenca = (
    df_graph[df_graph["classe"].notna()]
    ["prontuario_id"]
    .unique()
    .tolist()
)
pront_com_doenca = sorted(pront_com_doenca)

# O que o sistema extraiu por prontuário
sistema_por_pront = (
    df_graph[df_graph["classe"].notna()]
    .groupby("prontuario_id")["condicao_normalizada"]
    .apply(lambda x: " | ".join(sorted(set(x.str.upper()))))
    .to_dict()
)

# =============================================================================
# LOOP DE ANOTAÇÃO
# =============================================================================

progresso = carrega_progresso()
ja_anotados = set(progresso.keys())
pendentes = [p for p in pront_com_doenca if p not in ja_anotados]

print(f"\n  Total de prontuários: {len(pront_com_doenca)}")
print(f"  Já anotados        : {len(ja_anotados)}")
print(f"  Pendentes          : {len(pendentes)}")

if not pendentes:
    print("\n✔ Todos os prontuários já foram anotados!")
    salva_csv(progresso)
    print(f"  Arquivo salvo: {ARQUIVO_SAIDA}")
    exit()

input("\n  Pressione Enter para começar a anotação...")

for i, pront_id in enumerate(pendentes):
    limpa_tela()

    # Cabeçalho de progresso
    total_pendentes = len(pendentes)
    pct = (i / total_pendentes) * 100
    print("=" * 70)
    print(f"  ANOTAÇÃO  {i+1}/{total_pendentes}  ({pct:.0f}% concluído)  |  ID: {pront_id}")
    print("=" * 70)

    # Texto original do prontuário
    texto = texto_por_id.get(pront_id, "")
    print("\n── ID DO PRONTUÁRIO " + "─")
    print(pront_id)
    print("\n── TEXTO DO PRONTUÁRIO " + "─" * 47)
    print(formata_texto(texto))

    # O que o sistema extraiu
    extraiu = sistema_por_pront.get(pront_id, "")
    print("\n── SISTEMA EXTRAIU " + "─" * 51)
    print(f"  {extraiu}")

    # Anotação
    print("\n── ANOTAÇÃO " + "─" * 58)
    print("  Digite 1 se a doença está presente, 0 se ausente.")
    print("  Para HDA e PBE: marque 1 SOMENTE se for o motivo principal.\n")

    anotacao = {}
    for classe in CLASSES:
        anotacao[classe] = get_input_binario(LABELS[classe])

    # Aplica hierarquia
    doenca_principal = aplica_hierarquia(anotacao)

    # Resumo
    print("\n── RESUMO " + "─" * 61)
    presentes = [LABELS[c] for c, v in anotacao.items() if v == 1]
    print(f"  Doenças marcadas   : {', '.join(presentes) if presentes else 'nenhuma'}")
    print(f"  Doença principal   : {doenca_principal.upper().replace('_', ' ')}")
    print(f"  (pela hierarquia causal)")

    # Confiança e comentário
    confianca = get_confianca()
    comentario = input("  Comentário (opcional, Enter para pular): ").strip()

    # Confirmação
    print()
    confirma = input("  Confirmar anotação? [S/n]: ").strip().upper()
    if confirma == "N":
        print("  Anotação descartada. Repetindo este prontuário...")
        pendentes.insert(i + 1, pront_id)
        continue

    # Salva progresso
    progresso[pront_id] = {
        "anotacao":         anotacao,
        "doenca_principal": doenca_principal,
        "confianca":        confianca,
        "comentario":       comentario,
        "sistema_extraiu":  extraiu,
        "timestamp":        datetime.now().isoformat(),
    }
    salva_progresso(progresso)
    salva_csv(progresso)

    print(f"\n  ✔ Salvo. ({len(progresso)}/{len(pront_com_doenca)} concluídos)\n")
    input("  Enter para o próximo...")

# =============================================================================
# FINALIZAÇÃO
# =============================================================================

limpa_tela()
print("=" * 70)
print("  ANOTAÇÃO CONCLUÍDA")
print("=" * 70)

df_final = pd.read_csv(ARQUIVO_SAIDA)
print(f"\n  Total anotado: {len(df_final)} prontuários")
print(f"\n  Distribuição da doença principal:")
print(df_final["doenca_principal"].value_counts().to_string())
print(f"\n  Distribuição de confiança:")
print(df_final["confianca"].value_counts().to_string())
print(f"\n✔ Ground truth salvo em: {ARQUIVO_SAIDA}")