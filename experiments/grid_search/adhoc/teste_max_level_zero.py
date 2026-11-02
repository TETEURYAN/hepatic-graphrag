"""Testa `dynamic_search_max_level: 0` (sem fallback pro nível 1, 501
comunidades) contra a config atual (`max_level: 1`, causa dos timeouts de
90min em ~83% das tentativas de S6/S5) -- em 2 pacientes NOVOS (não usados
em nenhum teste anterior), pra ver se resolve o tempo/custo sem gerar
resultado degenerado.

Script de teste pontual -- não deve ser commitado.
"""

import json
import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.dataset.clinical_text import carrega_texto_por_prontuario  # noqa: E402
from src.core.paths import CSV_ORIGINAL, GRAPHRAG_DATA, GRAPHRAG_ROOT, bootstrap_sys_path  # noqa: E402
from src.core.query.runner import run_graphrag_query  # noqa: E402

bootstrap_sys_path(Path(__file__).resolve().parent)
from parser_doencas import DOENCAS_VALIDAS, extrai_doencas  # noqa: E402

# Pacientes novos, nunca testados em nenhum experimento anterior de custo/tuning
PACIENTES_TESTE = ["10041689", "10174423"]  # dois que estavam entre os timeouts originais de S6

SISTEMAS = [
    {"id": "S5", "method": "global", "community_level": 1},
    {"id": "S6", "method": "global", "community_level": 2},
]

MODELO = "inclusionai/ling-3.0-flash"


def monta_prompt_multilabel(pront_id: str, texto: str) -> str:
    doencas = ", ".join(DOENCAS_VALIDAS[:-1])
    return (
        f"Abaixo estão as evoluções/anamneses de um único paciente, "
        f"extraídas do prontuário eletrônico (nº {pront_id}). "
        f"Com base EXCLUSIVAMENTE nesse conteúdo (usando seu conhecimento médico geral apenas para interpretá-lo), "
        f"quais das doenças hepáticas abaixo estão CLARAMENTE documentadas como PRESENTES nesse paciente "
        f"(diagnóstico explícito, exame de imagem/laboratorial confirmatório sem ressalva, ou estadiamento formal "
        f"LITERALMENTE citado no texto)? "
        f"Considere tanto a doença de base quanto suas complicações, se ambas estiverem documentadas "
        f"(ex.: esquistossomose E a hipertensão portal decorrente dela, se as duas estiverem descritas).\n\n"
        f"IMPORTANTE sobre resultados de exame: resultados como 'NR', 'não reagente', 'negativo', 'NDN' ou "
        f"'normal' indicam AUSÊNCIA da condição testada, não presença — NÃO liste a doença nesse caso "
        f"(ex.: 'Anti-HCV: NR' significa que o paciente NÃO tem hepatite C). "
        f"NÃO liste doenças que o paciente negou, que foram apenas mencionadas em discussão/diferencial sem "
        f"confirmação, que foram investigadas com resultado negativo, ou que foram descartadas/excluídas no "
        f"texto — liste SOMENTE as que estão confirmadas como presentes. NÃO liste uma doença só porque o "
        f"texto menciona risco, probabilidade ou fator de risco para ela. Para hemorragia digestiva alta e "
        f"PBE especificamente: só conta se for sangramento/peritonite ATIVO ou muito recente.\n\n"
        f"Lista de doenças possíveis: {doencas}.\n\n"
        f"Responda EXATAMENTE neste formato, sem nenhum texto antes ou depois das tags:\n"
        f"<RESPOSTA>\n"
        f"[uma doença por linha, no formato 'NOME DA DOENÇA: trecho curto do texto que confirma isso' — "
        f"NOME DA DOENÇA deve ser EXATAMENTE um dos nomes da lista acima — ou apenas 'Nenhuma']\n"
        f"</RESPOSTA>\n\n"
        f"--- PRONTUÁRIO {pront_id} ---\n"
        f"{texto[:8000]}\n"
        f"--- FIM DO PRONTUÁRIO {pront_id} ---"
    )


def gera_config(root: Path, max_level: int, threshold: int, sufixo: str) -> Path:
    base_path = root / "settings.yaml"
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["models"]["default_chat_model"]["model"] = MODELO
    config["models"]["default_chat_model"]["reasoning_effort"] = "low"
    config["models"]["default_chat_model"]["max_tokens"] = 2000
    config.setdefault("global_search", {}).update({
        "dynamic_search_threshold": threshold,
        "dynamic_search_use_summary": True,
        "dynamic_search_max_level": max_level,
        "dynamic_search_keep_parent": False,
        "dynamic_search_num_repeats": 1,
        "data_max_tokens": 6000,
        "max_context_tokens": 6000,
        "map_max_length": 500,
        "reduce_max_length": 1000,
    })

    saida = root / f"settings_teste_{sufixo}.yaml"
    with open(saida, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return saida


def main():
    texto_por_id = carrega_texto_por_prontuario(CSV_ORIGINAL, "prontuario", "descricao")

    # config nova: max_level=1 (mantém fallback possível), threshold=1 (bem mais
    # permissivo que o 3 original -- objetivo: nivel 0 passar com mais frequencia,
    # evitando cair no fallback caro pras 501 comunidades do nivel 1, sem cortar
    # a granularidade como o max_level=0 fez)
    config_novo = gera_config(GRAPHRAG_ROOT, max_level=1, threshold=1, sufixo="threshold_baixo")
    print(f"Config testada: {config_novo}\n")

    resultados = {}
    for pid in PACIENTES_TESTE:
        texto = texto_por_id.get(pid, "")
        if not texto:
            print(f"⚠ {pid} sem texto, pulando.")
            continue
        prompt = monta_prompt_multilabel(pid, texto)

        for sistema in SISTEMAS:
            sid, method, level = sistema["id"], sistema["method"], sistema["community_level"]
            chave = f"{sid}__{pid}"
            print(f"--- {chave} (max_level=1, threshold=1) ---", flush=True)
            inicio = time.time()
            resposta_raw, sucesso, duracao = run_graphrag_query(
                query=prompt, root=GRAPHRAG_ROOT, data=GRAPHRAG_DATA, method=method,
                community_level=level, timeout=2700,  # 45min
                config=config_novo,
                dynamic_community_selection=True,
                use_patched_runner=True,
            )
            doencas = sorted(extrai_doencas(resposta_raw)) if sucesso else []
            resultados[chave] = {
                "status": "ok" if sucesso else "erro", "duracao_s": duracao,
                "doencas": doencas, "resposta_raw": resposta_raw[:500],
            }
            print(f"  status={'ok' if sucesso else 'erro'}  duracao={duracao:.1f}s  doencas={doencas}", flush=True)
            print(f"  resposta: {resposta_raw[:300]}\n", flush=True)

            # salva incrementalmente -- permite acompanhar progresso sem esperar o fim
            with open(_REPO_ROOT / "tcc-results-01" / "teste_max_level_zero.json", "w", encoding="utf-8") as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\n=== RESUMO ===")
    for k, r in resultados.items():
        print(f"{k}: {r['status']} em {r['duracao_s']:.1f}s -> {r['doencas']}")


if __name__ == "__main__":
    main()
