"""Mede o custo REAL de `global`/`drift` com os parâmetros de redução de
custo (`experimentos/local copy/settings_custo_tuned.yaml`) vs. o baseline
(settings.yaml de produção, sem alteração nenhuma) — mesmo prontuário, os
dois configs, pra isolar exatamente o efeito da mudança.

Só testa S5-S8 (global/drift) — S1-S4 (basic/local) já são 1 chamada só,
os parâmetros ajustados aqui não afetam eles.

Prontuários usados são DIFERENTES do que já foi testado em
`teste_custo_multilabel.py` (que usou 2406189).

Config testado (ver comentários em settings_custo_tuned.yaml):
- global: --dynamic-community-selection ligado (rating 0-5 antes do map-call
  completo), threshold subido de 1 pra 3, usa resumo em vez de full_context
  na fase de rating, poda a hierarquia dinâmica em 1 nível em vez de 2,
  data_max_tokens/max_context_tokens cortados de 12000 pra 6000.
- drift: n_depth 3->1, drift_k_followups 20->6, primer_folds 5->3,
  data_max_tokens/local_search_max_data_tokens 12000->6000,
  top_k_entities/relationships 10->5.

Não toca no settings.yaml de produção — usa --config apontando pro arquivo
separado. Mede custo real via /credits antes/depois de cada query (mesma
técnica de `teste_custo_multilabel.py`).

Uso:
    uv run python3 experiments/grid_search/teste_custo_global_drift_tuned.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.paths import (  # noqa: E402
    CSV_ORIGINAL,
    GRAPHRAG_DATA,
    GRAPHRAG_ROOT,
    PROGRESSO_GRID_DIR,
    bootstrap_sys_path,
)

bootstrap_sys_path(Path(__file__).resolve().parent)
from parser_doencas import DOENCAS_VALIDAS, extrai_doencas  # noqa: E402
from src.core.dataset.clinical_text import carrega_texto_por_prontuario  # noqa: E402
from src.core.query.runner import default_timeout, run_graphrag_query  # noqa: E402

API_KEY = os.environ["GRAPHRAG_API_KEY"]
SALDO_MINIMO = 0.20  # USD

PRONTUARIOS_TESTE = ["3130762", "2318525"]  # diferentes do 2406189 já usado antes

CONFIG_TUNED = GRAPHRAG_ROOT / "settings_custo_tuned.yaml"

SISTEMAS = [
    {"id": "S5", "method": "global", "community_level": 1},
    {"id": "S6", "method": "global", "community_level": 2},
    {"id": "S7", "method": "drift", "community_level": 1},
    {"id": "S8", "method": "drift", "community_level": 2},
]

RELATORIO_PATH = PROGRESSO_GRID_DIR / "teste_custo_global_drift_tuned.json"


def monta_prompt_multilabel(pront_id: str, texto: str) -> str:
    doencas = ", ".join(DOENCAS_VALIDAS[:-1])
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


def consulta_saldo() -> float:
    r = requests.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()["data"]
    return d["total_credits"] - d["total_usage"]


def espera_saldo_mudar(saldo_antes: float, tentativas: int = 12, intervalo: int = 5) -> float:
    for _ in range(tentativas):
        time.sleep(intervalo)
        saldo = consulta_saldo()
        if abs(saldo - saldo_antes) > 1e-9:
            return saldo
    return consulta_saldo()


def roda_uma(pront_id: str, texto: str, sistema: dict, tuned: bool, relatorio: dict) -> None:
    chave = f"{sistema['id']}__{pront_id}__{'tuned' if tuned else 'baseline'}"
    if chave in relatorio and relatorio[chave].get("status") == "ok":
        print(f"{chave}: já medido, pulando (custo US$ {relatorio[chave]['custo_usd']:.5f}).")
        return

    saldo_antes = consulta_saldo()
    if saldo_antes < SALDO_MINIMO:
        print(f"\n⚠ ABORTADO em {chave}: saldo US$ {saldo_antes:.4f} abaixo do mínimo.")
        raise SystemExit(1)

    prompt = monta_prompt_multilabel(pront_id, texto)
    print(f"{chave} — saldo antes: US$ {saldo_antes:.4f}")

    kwargs = {}
    if tuned:
        kwargs["config"] = CONFIG_TUNED
        if sistema["method"] == "global":
            kwargs["dynamic_community_selection"] = True

    resposta_raw, sucesso, duracao = run_graphrag_query(
        query=prompt,
        root=GRAPHRAG_ROOT,
        data=GRAPHRAG_DATA,
        method=sistema["method"],
        community_level=sistema["community_level"],
        response_type="Only the <RESPOSTA>...</RESPOSTA> block requested, nothing else",
        timeout=default_timeout(sistema["method"]),
        **kwargs,
    )

    saldo_depois = espera_saldo_mudar(saldo_antes)
    custo = round(saldo_antes - saldo_depois, 6)
    doencas = sorted(extrai_doencas(resposta_raw)) if sucesso else []

    resultado = {
        "prontuario_id": pront_id,
        "sistema_id": sistema["id"],
        "method": sistema["method"],
        "community_level": sistema["community_level"],
        "tuned": tuned,
        "doencas_previstas": doencas,
        "resposta_raw": resposta_raw[:1500],
        "duracao_s": duracao,
        "status": "ok" if sucesso else "erro",
        "saldo_antes_usd": round(saldo_antes, 6),
        "saldo_depois_usd": round(saldo_depois, 6),
        "custo_usd": custo,
        "medicao_incerta": custo <= 0,
        "timestamp": datetime.now().isoformat(),
    }
    relatorio[chave] = resultado
    with open(RELATORIO_PATH, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    print(f"  doenças: {doencas}")
    print(f"  duração: {duracao:.1f}s | custo: US$ {custo:.5f}"
          f"{' (incerto)' if resultado['medicao_incerta'] else ''}\n")


def main():
    texto_por_id = carrega_texto_por_prontuario(CSV_ORIGINAL)

    relatorio = {}
    if RELATORIO_PATH.exists():
        with open(RELATORIO_PATH, "r", encoding="utf-8") as f:
            relatorio = json.load(f)

    saldo_inicial = consulta_saldo()
    print(f"Saldo inicial: US$ {saldo_inicial:.4f}")
    print(f"Prontuários: {PRONTUARIOS_TESTE}")
    print(f"Config tuned: {CONFIG_TUNED}\n")

    for pront_id in PRONTUARIOS_TESTE:
        texto = texto_por_id.get(pront_id, "")
        if not texto:
            print(f"⚠ {pront_id} sem texto, pulando.")
            continue
        for sistema in SISTEMAS:
            # baseline primeiro, depois tuned — assim dá pra comparar par a par
            roda_uma(pront_id, texto, sistema, tuned=False, relatorio=relatorio)
            roda_uma(pront_id, texto, sistema, tuned=True, relatorio=relatorio)

    saldo_final = consulta_saldo()
    print(f"\n{'=' * 70}\nRESUMO — baseline vs. tuned (dynamic selection + limites de contexto)\n{'=' * 70}")
    print(f"Saldo inicial: US$ {saldo_inicial:.4f} | Saldo final: US$ {saldo_final:.4f} | "
          f"Gasto total: US$ {saldo_inicial - saldo_final:.5f}\n")

    for pront_id in PRONTUARIOS_TESTE:
        print(f"--- Prontuário {pront_id} ---")
        for sistema in SISTEMAS:
            sid = sistema["id"]
            base = relatorio.get(f"{sid}__{pront_id}__baseline")
            tun = relatorio.get(f"{sid}__{pront_id}__tuned")
            if not base or not tun:
                continue
            reducao = (1 - tun["custo_usd"] / base["custo_usd"]) * 100 if base["custo_usd"] > 0 else float("nan")
            print(f"  {sid} ({sistema['method']:<6} L{sistema['community_level']}): "
                  f"baseline US$ {base['custo_usd']:.5f} ({base['duracao_s']:.0f}s)  ->  "
                  f"tuned US$ {tun['custo_usd']:.5f} ({tun['duracao_s']:.0f}s)  "
                  f"[{reducao:+.1f}%]")
        print()

    print(f"Relatório completo em: {RELATORIO_PATH}")


if __name__ == "__main__":
    main()
