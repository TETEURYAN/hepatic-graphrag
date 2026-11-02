"""Validação do prompt multilabel num subconjunto pequeno, antes de rodar o grid inteiro.

Roda os 8 sistemas (mesmo grid de `run.py`) só nos prontuários abaixo — um
subconjunto dos 23 "completamente errados" no single-label atual (todo
sistema com predição válida errou). Objetivo: checar se o modelo responde
bem ao formato multilabel (lista de doenças, uma por linha) e se isso
resolve os casos de causa-raiz vs. complicação (ex.: hipertensão portal
como consequência documentada de esquistossomose/cirrose).

Não toca em `progresso_grid/progresso_S*.json` (dados de produção do grid
single-label) — salva num arquivo de progresso separado, resumível.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.paths import (  # noqa: E402
    CSV_ORIGINAL,
    GRAPHRAG_DATA,
    GRAPHRAG_ROOT,
    GROUND_TRUTH_CSV,
    PROGRESSO_GRID_DIR,
    bootstrap_sys_path,
)

bootstrap_sys_path(Path(__file__).resolve().parent)

from parser_doencas import DOENCAS_VALIDAS, extrai_doencas  # noqa: E402
from src.core.dataset.clinical_text import carrega_texto_por_prontuario  # noqa: E402
from src.core.query.progress import ProgressTracker  # noqa: E402
from src.core.query.runner import default_timeout, run_graphrag_query  # noqa: E402

COL_ID = "prontuario"
COL_TEXTO = "descricao"
PAUSA_ENTRE_QUERIES = 6

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

# Subconjunto dos 23 "completamente errados" (single-label): 3 casos de
# hipertensao_portal (a hipótese principal — causa vs. consequência) + 2
# outras classes, pra checar formato multilabel em geral.
PRONTUARIOS_TESTE = [
    "2406189",   # verdade: hipertensao_portal
    "3130762",   # verdade: hipertensao_portal
    "8774960",   # verdade: hipertensao_portal
    "2318525",   # verdade: esquistossomose
    "10212009",  # verdade: hemorragia_digestiva_alta
]

PROGRESSO_PATH = PROGRESSO_GRID_DIR / "progresso_multilabel_validacao_v3.json"

# %% Carrega dados
gt = pd.read_csv(GROUND_TRUTH_CSV, dtype=str).fillna("")
gt["prontuario_id"] = gt["prontuario_id"].str.strip()

CLASSES = [
    "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
    "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
]

texto_por_id = carrega_texto_por_prontuario(CSV_ORIGINAL, COL_ID, COL_TEXTO)


def monta_prompt_multilabel(pront_id: str, texto: str) -> str:
    doencas = ", ".join(DOENCAS_VALIDAS[:-1])  # exclui 'Nenhuma' da lista de opções
    return (
        f"Abaixo estão as evoluções/anamneses de um único paciente, "
        f"extraídas do prontuário eletrônico (nº {pront_id}). "
        f"Com base EXCLUSIVAMENTE nesse conteúdo (usando seu conhecimento médico geral apenas para interpretá-lo), "
        f"quais das doenças hepáticas abaixo estão CLARAMENTE documentadas como PRESENTES nesse paciente "
        f"(diagnóstico explícito, exame de imagem/laboratorial confirmatório sem ressalva, ou estadiamento formal)? "
        f"Considere tanto a doença de base quanto suas complicações, se ambas estiverem documentadas "
        f"(ex.: esquistossomose E a hipertensão portal decorrente dela, se as duas estiverem descritas).\n\n"
        f"IMPORTANTE sobre resultados de exame: resultados como 'NR', 'não reagente', 'negativo', 'NDN' ou "
        f"'normal' indicam AUSÊNCIA da condição testada, não presença — NÃO liste a doença nesse caso "
        f"(ex.: 'Anti-HCV: NR' significa que o paciente NÃO tem hepatite C). "
        f"NÃO liste doenças que o paciente negou, que foram apenas mencionadas em discussão/diferencial sem "
        f"confirmação, que foram investigadas com resultado negativo, ou que foram descartadas/excluídas no "
        f"texto — liste SOMENTE as que estão confirmadas como presentes. Para hemorragia digestiva alta e "
        f"PBE especificamente: só conta se for sangramento/peritonite ATIVO ou muito recente, não episódio "
        f"antigo já resolvido.\n\n"
        f"Lista de doenças possíveis: {doencas}.\n\n"
        f"Responda EXATAMENTE neste formato, sem nenhum texto antes ou depois das tags (sem preâmbulo, sem "
        f"análise, sem conclusão):\n"
        f"<RESPOSTA>\n"
        f"[uma doença por linha, no formato 'NOME DA DOENÇA: trecho curto do texto que confirma isso' — "
        f"ou apenas 'Nenhuma' se nenhuma doença da lista estiver claramente documentada como presente]\n"
        f"</RESPOSTA>\n\n"
        f"--- PRONTUÁRIO {pront_id} ---\n"
        f"{texto[:8000]}\n"
        f"--- FIM DO PRONTUÁRIO {pront_id} ---"
    )


def roda_query(pront_id: str, texto: str, method: str, community_level: int) -> dict:
    prompt = monta_prompt_multilabel(pront_id, texto)
    resposta_raw, sucesso, duracao = run_graphrag_query(
        query=prompt,
        root=GRAPHRAG_ROOT,
        data=GRAPHRAG_DATA,
        method=method,
        community_level=community_level,
        response_type="Only the <RESPOSTA>...</RESPOSTA> block requested, nothing else",
        timeout=default_timeout(method),
    )

    if not sucesso and resposta_raw.startswith("ERROR: query timed out"):
        return {
            "prontuario_id": pront_id,
            "doencas_previstas": [],
            "resposta_raw": "",
            "duracao_s": duracao,
            "status": "timeout",
            "returncode": -1,
            "timestamp": datetime.now().isoformat(),
        }

    doencas = sorted(extrai_doencas(resposta_raw)) if sucesso else []
    status = "ok" if sucesso else "erro_returncode"

    return {
        "prontuario_id": pront_id,
        "doencas_previstas": doencas,
        "resposta_raw": resposta_raw[:800],
        "duracao_s": duracao,
        "status": status,
        "returncode": 0 if sucesso else -1,
        "timestamp": datetime.now().isoformat(),
    }


# %% Execução
print(f"Prontuários de teste : {PRONTUARIOS_TESTE}")
print(f"Sistemas             : {[s['id'] for s in SISTEMAS]}")
print(f"Total de execuções   : {len(PRONTUARIOS_TESTE) * len(SISTEMAS)}")
print(f"Progresso salvo em   : {PROGRESSO_PATH}\n")

tracker = ProgressTracker(PROGRESSO_PATH)
inicio_total = time.time()

for sistema in SISTEMAS:
    sid = sistema["id"]
    method = sistema["method"]
    level = sistema["community_level"]

    print(f"\n{'=' * 60}")
    print(f"Sistema {sid} — {method.upper()} / community-level {level}")
    print(f"{'=' * 60}")

    for pront_id in PRONTUARIOS_TESTE:
        chave = f"{sid}__{pront_id}"
        if chave in tracker:
            print(f"  {pront_id}: já feito, pulando.")
            continue

        texto = texto_por_id.get(pront_id, "")
        if not texto:
            tracker.registra(chave, {
                "prontuario_id": pront_id, "sistema_id": sid,
                "doencas_previstas": [], "resposta_raw": "", "duracao_s": 0,
                "status": "sem_texto", "returncode": 0,
                "timestamp": datetime.now().isoformat(),
            })
            print(f"  {pront_id}: SEM_TEXTO")
            continue

        resultado = roda_query(pront_id, texto, method, level)
        resultado["sistema_id"] = sid
        tracker.registra(chave, resultado)
        print(f"  {pront_id}: {resultado['doencas_previstas']} ({resultado['duracao_s']}s, {resultado['status']})")
        time.sleep(PAUSA_ENTRE_QUERIES)

tempo_total = time.time() - inicio_total
print(f"\n✔ Concluído em {tempo_total/60:.1f} min")

# %% Comparação: verdade x single-label antigo x multilabel novo
print("\n" + "=" * 100)
print("COMPARAÇÃO: ground truth (multilabel) vs. single-label antigo vs. multilabel novo")
print("=" * 100)

progresso_antigo = {}
for s in SISTEMAS:
    path_antigo = PROGRESSO_GRID_DIR / f"progresso_{s['id']}.json"
    if path_antigo.exists():
        import json
        with open(path_antigo, "r", encoding="utf-8") as f:
            progresso_antigo[s["id"]] = json.load(f)

for pront_id in PRONTUARIOS_TESTE:
    row = gt[gt["prontuario_id"] == pront_id]
    verdade_multilabel = sorted(c for c in CLASSES if not row.empty and row.iloc[0].get(c) == "1")
    verdade_principal = row.iloc[0]["doenca_principal"] if not row.empty else "?"

    print(f"\nProntuário {pront_id}")
    print(f"  Verdade (doenca_principal): {verdade_principal}")
    print(f"  Verdade (multilabel)      : {verdade_multilabel}")
    for s in SISTEMAS:
        sid = s["id"]
        antigo = progresso_antigo.get(sid, {}).get(pront_id, {}).get("doenca_prevista", "?")
        novo = tracker[f"{sid}__{pront_id}"]["doencas_previstas"] if f"{sid}__{pront_id}" in tracker else "?"
        print(f"    {sid} ({s['method']:<6} L{s['community_level']}): antigo={antigo!r:<28} novo={novo}")
