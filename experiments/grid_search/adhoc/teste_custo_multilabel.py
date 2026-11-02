"""Mede o custo REAL (em USD) de rodar cada um dos 8 sistemas do grid com o
prompt multilabel — 1 query por sistema, num único prontuário já conhecido
(2406189, caso de hipertensão portal usado antes em `validar_prompt_multilabel.py`).

Não dá pra pegar histórico de token/custo via API do OpenRouter com esta
chave (`/activity` exige management key, `/generation` exige um ID que
nunca guardamos, e o `query.log` do GraphRAG não loga uso de token). Então
a medição é direta: consulta `/credits` antes e depois de cada query e usa
a diferença de saldo como custo real daquele sistema.

Os resultados são salvos como dados de PRODUÇÃO de verdade (mesmo formato
que um futuro `run_multilabel.py` usaria) — se decidirmos rodar o grid
completo depois, esse prontuário já sai pronto, a query não é desperdiçada.

Trava de segurança: aborta se o saldo cair abaixo de SALDO_MINIMO antes de
disparar a próxima query (evita zerar o saldo por engano).

Uso:
    uv run python3 experiments/grid_search/teste_custo_multilabel.py
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
SALDO_MINIMO = 0.20  # USD — aborta antes de disparar query nova se saldo cair abaixo disso

PRONTUARIO_TESTE = "2406189"  # verdade: hipertensao_portal (ver validar_prompt_multilabel.py)

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

RELATORIO_PATH = PROGRESSO_GRID_DIR / "teste_custo_grid_multilabel.json"


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
    """A cobrança no OpenRouter não é instantânea — espera até o saldo
    refletir a query recém-feita (ou desiste após ~1 min e usa o último
    valor lido, marcando a medição como incerta)."""
    for _ in range(tentativas):
        time.sleep(intervalo)
        saldo = consulta_saldo()
        if abs(saldo - saldo_antes) > 1e-9:
            return saldo
    return consulta_saldo()


def main():
    texto_por_id = carrega_texto_por_prontuario(CSV_ORIGINAL)
    texto = texto_por_id.get(PRONTUARIO_TESTE, "")
    if not texto:
        raise SystemExit(f"Prontuário {PRONTUARIO_TESTE} sem texto — ajuste PRONTUARIO_TESTE.")

    prompt = monta_prompt_multilabel(PRONTUARIO_TESTE, texto)
    print(f"Prontuário de teste: {PRONTUARIO_TESTE} ({len(texto)} chars de texto bruto)")
    print(f"Sistemas: {[s['id'] for s in SISTEMAS]}\n")

    relatorio = {}
    if RELATORIO_PATH.exists():
        with open(RELATORIO_PATH, "r", encoding="utf-8") as f:
            relatorio = json.load(f)

    saldo_inicial = consulta_saldo()
    print(f"Saldo inicial: US$ {saldo_inicial:.4f}\n")

    for sistema in SISTEMAS:
        sid = sistema["id"]
        if sid in relatorio and relatorio[sid].get("status") == "ok":
            print(f"{sid}: já medido, pulando (custo US$ {relatorio[sid]['custo_usd']:.5f}).")
            continue

        saldo_antes = consulta_saldo()
        if saldo_antes < SALDO_MINIMO:
            print(f"\n⚠ ABORTADO: saldo US$ {saldo_antes:.4f} abaixo do mínimo de segurança "
                  f"(US$ {SALDO_MINIMO:.2f}). Sistemas restantes não foram testados.")
            break

        print(f"{'=' * 60}\n{sid} — {sistema['method'].upper()} / community-level {sistema['community_level']}\n{'=' * 60}")
        print(f"  saldo antes: US$ {saldo_antes:.4f}")

        inicio = time.time()
        resposta_raw, sucesso, duracao = run_graphrag_query(
            query=prompt,
            root=GRAPHRAG_ROOT,
            data=GRAPHRAG_DATA,
            method=sistema["method"],
            community_level=sistema["community_level"],
            response_type="Only the <RESPOSTA>...</RESPOSTA> block requested, nothing else",
            timeout=default_timeout(sistema["method"]),
        )

        saldo_depois = espera_saldo_mudar(saldo_antes)
        custo = round(saldo_antes - saldo_depois, 6)
        doencas = sorted(extrai_doencas(resposta_raw)) if sucesso else []

        resultado = {
            "prontuario_id": PRONTUARIO_TESTE,
            "sistema_id": sid,
            "method": sistema["method"],
            "community_level": sistema["community_level"],
            "doencas_previstas": doencas,
            "resposta_raw": resposta_raw[:1500],
            "duracao_s": duracao,
            "status": "ok" if sucesso else "erro",
            "saldo_antes_usd": round(saldo_antes, 6),
            "saldo_depois_usd": round(saldo_depois, 6),
            "custo_usd": custo,
            "medicao_incerta": custo <= 0,  # saldo não mudou dentro do tempo de espera
            "timestamp": datetime.now().isoformat(),
        }
        relatorio[sid] = resultado

        with open(RELATORIO_PATH, "w", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)

        # Também grava no formato de produção (1 arquivo por sistema, chave
        # = prontuario_id) — reaproveitável se o grid multilabel completo
        # for rodado depois, sem precisar repetir esta query.
        path_producao = PROGRESSO_GRID_DIR / f"progresso_{sid}_multilabel.json"
        prod = {}
        if path_producao.exists():
            with open(path_producao, "r", encoding="utf-8") as f:
                prod = json.load(f)
        prod[PRONTUARIO_TESTE] = {
            "prontuario_id": PRONTUARIO_TESTE,
            "doencas_previstas": doencas,
            "resposta_raw": resposta_raw[:1500],
            "duracao_s": duracao,
            "status": "ok" if sucesso else "erro",
            "timestamp": resultado["timestamp"],
        }
        with open(path_producao, "w", encoding="utf-8") as f:
            json.dump(prod, f, ensure_ascii=False, indent=2)

        print(f"  doenças previstas: {doencas}")
        print(f"  duração: {duracao:.1f}s | saldo depois: US$ {saldo_depois:.4f} | "
              f"custo: US$ {custo:.5f}{' (incerto, saldo nao mudou a tempo)' if resultado['medicao_incerta'] else ''}\n")

    saldo_final = consulta_saldo()
    custo_total = round(saldo_inicial - saldo_final, 6)

    print(f"\n{'=' * 60}\nRESUMO\n{'=' * 60}")
    print(f"Saldo inicial : US$ {saldo_inicial:.4f}")
    print(f"Saldo final   : US$ {saldo_final:.4f}")
    print(f"Custo total (8 sistemas, 1 prontuário): US$ {custo_total:.5f}\n")
    for sid, r in relatorio.items():
        print(f"  {sid} ({r['method']:<6} L{r['community_level']}): US$ {r['custo_usd']:.5f}  "
              f"({r['duracao_s']:.0f}s)  previsto={r['doencas_previstas']}")

    print(f"\nRelatório completo salvo em: {RELATORIO_PATH}")
    print(f"Resultados de produção salvos em: {PROGRESSO_GRID_DIR}/progresso_S*_multilabel.json")

    n_queries_total = 537 * 8
    if custo_total > 0:
        custo_medio_por_query = custo_total / len([r for r in relatorio.values() if r["status"] == "ok"])
        print(f"\nExtrapolação p/ grid completo (537 prontuários x 8 sistemas = {n_queries_total} queries):")
        print(f"  custo médio medido por query: US$ {custo_medio_por_query:.5f}")
        print(f"  estimativa total: US$ {custo_medio_por_query * n_queries_total:.2f}")


if __name__ == "__main__":
    main()
