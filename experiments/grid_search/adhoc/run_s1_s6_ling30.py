"""Re-roda S1-S6 (multilabel, 537 prontuários) com `inclusionai/ling-3.0-flash`
em vez do `ling-2.6-flash` usado em `tcc-results-01/progresso_cache.json` —
pra nivelar com S7/S8 (que já rodaram nesse modelo) e permitir comparação
limpa, mesmo modelo em todos os 8 sistemas.

NÃO toca em `tcc-results-01/progresso_cache.json` (resultado original) — grava
num JSON novo e separado, pra comparação lado a lado depois. Mantém o mesmo
tuning de custo (dynamic-community-selection + limites de contexto) que já
foi usado em S5/S6 originalmente, trocando só o modelo — isola a variável.

Script de exploração/comparação (como teste_custo_*.py) — não é infraestrutura
do pipeline, não deve ser commitado.

Uso:
    uv run --no-sync python3 experiments/grid_search/run_s1_s6_ling30.py
"""

import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.dataset.clinical_text import carrega_texto_por_prontuario  # noqa: E402
from src.core.paths import CSV_ORIGINAL, GRAPHRAG_DATA, GRAPHRAG_ROOT, GROUND_TRUTH_CSV, bootstrap_sys_path  # noqa: E402
from src.core.query.runner import run_graphrag_query  # noqa: E402

bootstrap_sys_path(Path(__file__).resolve().parent)
from parser_doencas import DOENCAS_VALIDAS, extrai_doencas  # noqa: E402

MODELO_NOVO = "inclusionai/ling-3.0-flash"

# Timeout por método -- override LOCAL desse script (não mexe em
# `default_timeout()` de src/core/query/runner.py, que é compartilhado por
# outros scripts). ling-3.0-flash é bem mais lento que ling-2.6-flash
# (raciocínio invisível); usuário já observou queries de global/drift
# passando de 1h no passado, contra os 1800s/2400s do timeout padrão --
# aumentado pra >1h em cada método, pra não matar query que ainda terminaria.
TIMEOUTS_LOCAIS_S = {"basic": 3900, "local": 3900, "global": 5400, "drift": 5400}


def timeout_local(method: str) -> int:
    return TIMEOUTS_LOCAIS_S.get(method, 3900)

# Ordem do mais barato pro mais caro (custo estimado por sistema, 537
# prontuários: S1 US$0,27 (já feito) < S2 US$0,18 < S3 US$0,21 < S4 US$0,71
# < S6 US$6,96 < S5 US$7,10). S1 fica na lista pra consistência do código,
# mas o cache resumível pula ele automaticamente (já 537/537 ok).
SISTEMAS = [
    {"id": "S1", "method": "basic", "community_level": 1},
    {"id": "S2", "method": "basic", "community_level": 2},
    {"id": "S3", "method": "local", "community_level": 1},
    {"id": "S4", "method": "local", "community_level": 2},
    {"id": "S6", "method": "global", "community_level": 2},
    {"id": "S5", "method": "global", "community_level": 1},
]

# Concorrência -- o teto real NÃO é o OpenRouter (que não documenta limite
# pra modelos pagos), é a RAM desta máquina: 25 workers derrubou o processo
# via OOM killer do kernel, 12 chegou a deixar só 116MB livres, e mesmo 8
# ainda apresentou queda progressiva de memória ao longo de ~8h de execução
# contínua (sugere acúmulo lento, não só pico instantâneo). Além disso,
# `basic` e `local` NÃO têm o mesmo custo de RAM por processo -- medido
# direto via `ps` (2026-09-26): `basic` ~617MB/processo, `local` ~1043MB/
# processo (quase o dobro -- carrega embeddings de entidades/relacionamentos
# que `basic` não usa). Com 6 workers de `local`, isso sozinho já ocupava
# 6,2GB dos 7,6GB totais, deixando só 742MB livres -- quase OOM de novo.
# Separado por método a partir daqui, com margem individual real.
MAX_WORKERS_POR_METODO = {"basic": 6, "local": 3, "global": 2, "drift": 2}

# Disjuntor adaptado pra concorrência: com várias queries em voo ao mesmo
# tempo, "erros consecutivos" não faz mais sentido estritamente sequencial
# -- olha a taxa de erro numa janela deslizante das últimas N conclusões.
JANELA_DISJUNTOR = 15
TAXA_ERRO_MAXIMA_DISJUNTOR = 0.6  # aborta o sistema se >60% das últimas N falharem

_lock_progresso = threading.Lock()

# Mesmos overrides validados em teste_custo_global_drift_tuned.py / já usados
# em produção pra gerar tcc-results-01 — mantidos aqui pra isolar só a
# variável "modelo" na comparação (não misturar com variável "tuning").
GLOBAL_SEARCH_TUNED = {
    "dynamic_search_threshold": 3,
    "dynamic_search_use_summary": True,
    "dynamic_search_max_level": 1,
    "dynamic_search_keep_parent": False,
    "dynamic_search_num_repeats": 1,
    "data_max_tokens": 6000,
    "max_context_tokens": 6000,
    "map_max_length": 500,
    "reduce_max_length": 1000,
}

SALDO_MINIMO = 0.30  # USD -- aborta antes de disparar query nova se saldo cair abaixo disso

# Tempo esperado por método -- referência pra sinalizar quando uma query sair
# muito fora do padrão (pode indicar timeout, retry silencioso, ou regressão
# do fix). Calibrado pro ling-3.0-flash (não pro ling-2.6-flash original):
# esse modelo é de raciocínio, mediu-se ~145-210s médios até pra "basic"
# (1 chamada só) contra ~22s do ling-2.6-flash -- usar a referência antiga
# geraria falso alarme de "LENTO" na maioria das queries normais.
DURACAO_ESPERADA_S = {"basic": 180, "local": 200, "global": 300, "drift": 400}
FATOR_ALERTA_LENTIDAO = 3.0  # avisa se duracao > mediana esperada * este fator

SAIDA_DIR = _REPO_ROOT / "tcc-results-01"
PROGRESSO_PATH = SAIDA_DIR / "progresso_cache_ling30_s1s6.json"
PREDICOES_CSV = SAIDA_DIR / "predicoes_ling30_s1s6.csv"
METRICAS_CSV = SAIDA_DIR / "metricas_ling30_s1s6.csv"

CLASSES = [
    "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
    "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
]


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
        f"texto menciona risco, probabilidade ou fator de risco para ela (ex.: 'risco elevado de PBE' ou "
        f"'epidemiologia positiva para esquistossomose' NÃO são confirmação — são apenas risco/exposição, "
        f"sem diagnóstico). Para hemorragia digestiva alta e PBE especificamente: só conta se for "
        f"sangramento/peritonite ATIVO ou muito recente, não episódio antigo já resolvido (ex.: 'HDA há "
        f"alguns anos' ou 'episódio prévio' NÃO contam).\n\n"
        f"ATENÇÃO a três confusões comuns — evite-as:\n"
        f"1) Hemorragia digestiva ALTA é sangramento de esôfago/estômago/duodeno (ex.: varizes esofágicas "
        f"rotas, úlcera péptica sangrante) — NÃO é sangramento baixo/retal (ex.: hemorroida, fissura anal, "
        f"sangramento nas fezes de origem colorretal). Não confunda os dois.\n"
        f"2) 'Hipertensão portal' é pressão elevada especificamente no sistema venoso porta (sinais: varizes "
        f"esofágicas, esplenomegalia, ascite, circulação colateral, dilatação/trombose da veia porta) — é "
        f"TOTALMENTE DIFERENTE de 'hipertensão arterial'/pressão arterial (PA, ex.: 'PA 140x90mmHg'). Não "
        f"liste hipertensão portal só porque o texto menciona pressão arterial elevada ou hipertensão arterial "
        f"sistêmica (HAS) sem nenhum sinal específico de hipertensão portal.\n"
        f"3) NÃO cite nem infira escores de estadiamento (Child-Pugh, MELD, FIB-4 etc.) que não estejam "
        f"literalmente escritos no texto com o valor exato — nunca estime ou calcule um escore que o texto "
        f"não apresenta explicitamente. NÃO use achados de outros sistemas/órgãos sem relação hepática direta "
        f"(ex.: megaesôfago chagásico, doenças biliares isoladas, cistos, achados ósseos ou pulmonares) como "
        f"evidência de doença hepática, a menos que o texto estabeleça explicitamente essa relação causal.\n\n"
        f"Lista de doenças possíveis: {doencas}.\n\n"
        f"Responda EXATAMENTE neste formato, sem nenhum texto antes ou depois das tags (sem preâmbulo, sem "
        f"análise, sem conclusão):\n"
        f"<RESPOSTA>\n"
        f"[uma doença por linha, no formato 'NOME DA DOENÇA: trecho curto do texto que confirma isso' — "
        f"NOME DA DOENÇA deve ser EXATAMENTE um dos nomes da lista acima, escrito literalmente como está "
        f"nela (ex.: 'Esquistossomose', não 'esquistossomose hepatoesplênica' ou qualquer variação/sinônimo) "
        f"— ou apenas 'Nenhuma' se nenhuma doença da lista estiver claramente documentada como presente]\n"
        f"</RESPOSTA>\n\n"
        f"--- PRONTUÁRIO {pront_id} ---\n"
        f"{texto[:8000]}\n"
        f"--- FIM DO PRONTUÁRIO {pront_id} ---"
    )


def consulta_saldo() -> float:
    r = requests.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {os.environ['GRAPHRAG_API_KEY']}"},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()["data"]
    return d["total_credits"] - d["total_usage"]


def gera_config_ling30(root: Path) -> Path:
    base_path = root / "settings.yaml"
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["models"]["default_chat_model"]["model"] = MODELO_NOVO
    # ling-3.0-flash é um modelo de raciocínio (hybrid-reasoning) -- sem
    # limitar o esforço, ele gera tokens de "pensamento" invisíveis sem
    # restrição, o que mediu ~6-8x mais lento que ling-2.6-flash num método
    # de 1 chamada só (basic), sem nenhum outro motivo estrutural pra isso.
    # `reasoning_effort` é suportado nativamente pelo GraphRAG
    # (graphrag/config/models/language_model_config.py) e default é None
    # (sem limite) -- seta pra "low" pra conter isso.
    config["models"]["default_chat_model"]["reasoning_effort"] = "low"
    # Diagnosticado (2026-09-25): mesmo com effort=low, o raciocínio invisível
    # consome uma fração fixa dos tokens de saída (ex.: 447 de 532 num teste
    # isolado) -- sem `max_tokens` generoso, chamadas estruturadas curtas
    # (ex.: rate_relevancy do dynamic-community-selection, que sem essa
    # config usa o default None do GraphRAG) ficam sem espaço pra resposta
    # de verdade e voltam vazias/malformadas. Custo extra é desprezível
    # (2000 tokens de completion ~= US$0,000126 no preço do ling-3.0-flash).
    config["models"]["default_chat_model"]["max_tokens"] = 2000
    config.setdefault("global_search", {}).update(GLOBAL_SEARCH_TUNED)

    saida = root / "settings_ling30_s1s6_auto.yaml"
    with open(saida, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return saida


def carrega_progresso() -> dict:
    if PROGRESSO_PATH.exists():
        with open(PROGRESSO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def salva_progresso(progresso: dict) -> None:
    SAIDA_DIR.mkdir(exist_ok=True)
    with open(PROGRESSO_PATH, "w", encoding="utf-8") as f:
        json.dump(progresso, f, ensure_ascii=False, indent=2)


def calcula_metricas(progresso: dict, gt_por_id: dict[str, set[str]]) -> list[dict]:
    linhas = []
    for sistema in SISTEMAS:
        sid = sistema["id"]
        por_classe = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in CLASSES}
        exact_matches = n_avaliado = 0

        for pid, gt_set in gt_por_id.items():
            chave = f"{sid}__{pid}"
            if chave not in progresso or progresso[chave].get("status") != "ok":
                continue
            pred_set = set(progresso[chave]["doencas_previstas"])
            n_avaliado += 1
            if pred_set == gt_set:
                exact_matches += 1
            for c in CLASSES:
                in_gt, in_pred = c in gt_set, c in pred_set
                if in_gt and in_pred:
                    por_classe[c]["tp"] += 1
                elif in_gt and not in_pred:
                    por_classe[c]["fn"] += 1
                elif not in_gt and in_pred:
                    por_classe[c]["fp"] += 1
                else:
                    por_classe[c]["tn"] += 1

        soma_tp = sum(v["tp"] for v in por_classe.values())
        soma_fp = sum(v["fp"] for v in por_classe.values())
        soma_fn = sum(v["fn"] for v in por_classe.values())

        precisoes, recalls, f1s = [], [], []
        linha = {
            "sistema": sid, "metodo": sistema["method"], "community_level": sistema["community_level"],
            "n_avaliado": n_avaliado,
            "exact_match_accuracy": round(exact_matches / n_avaliado, 4) if n_avaliado else None,
        }
        for c in CLASSES:
            tp, fp, fn = por_classe[c]["tp"], por_classe[c]["fp"], por_classe[c]["fn"]
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            precisoes.append(p); recalls.append(r); f1s.append(f1)
            linha[f"f1_{c}"] = round(f1, 4)

        linha["precision_macro"] = round(sum(precisoes) / len(precisoes), 4)
        linha["recall_macro"] = round(sum(recalls) / len(recalls), 4)
        linha["f1_macro"] = round(sum(f1s) / len(f1s), 4)
        p_micro = soma_tp / (soma_tp + soma_fp) if (soma_tp + soma_fp) else 0.0
        r_micro = soma_tp / (soma_tp + soma_fn) if (soma_tp + soma_fn) else 0.0
        linha["precision_micro"] = round(p_micro, 4)
        linha["recall_micro"] = round(r_micro, 4)
        linha["f1_micro"] = round(2 * p_micro * r_micro / (p_micro + r_micro), 4) if (p_micro + r_micro) else 0.0
        linhas.append(linha)
    return linhas


def salva_predicoes_csv(progresso: dict, gt_por_id: dict[str, set[str]]) -> None:
    linhas = []
    for sistema in SISTEMAS:
        sid = sistema["id"]
        for pid, gt_set in gt_por_id.items():
            chave = f"{sid}__{pid}"
            if chave not in progresso:
                continue
            r = progresso[chave]
            pred_set = set(r.get("doencas_previstas", []))
            linhas.append({
                "sistema": sid, "prontuario_id": pid, "status": r.get("status"),
                "ground_truth": ";".join(sorted(gt_set)) or "nenhuma",
                "previsto": ";".join(sorted(pred_set)) or "nenhuma",
                "acerto_exato": pred_set == gt_set,
                "duracao_s": r.get("duracao_s"),
            })
    pd.DataFrame(linhas).to_csv(PREDICOES_CSV, index=False, encoding="utf-8")


def processa_um(sistema: dict, pid: str, texto_por_id: dict, config_path: Path) -> dict:
    sid, method, level = sistema["id"], sistema["method"], sistema["community_level"]
    texto = texto_por_id[pid]
    prompt = monta_prompt_multilabel(pid, texto)
    timeout = timeout_local(method)

    resposta_raw, sucesso, duracao = run_graphrag_query(
        query=prompt, root=GRAPHRAG_ROOT, data=GRAPHRAG_DATA, method=method,
        community_level=level, timeout=timeout,
        config=config_path,
        dynamic_community_selection=(method == "global"),
        use_patched_runner=True,  # aplica reasoning_effort_patch.py (extra_body reasoning:low)
    )
    doencas = sorted(extrai_doencas(resposta_raw)) if sucesso else []
    status = "ok" if sucesso else "erro"

    return {
        "sistema_id": sid, "prontuario_id": pid, "method": method, "community_level": level,
        "doencas_previstas": doencas, "resposta_raw": resposta_raw[:1500],
        "duracao_s": duracao, "status": status, "timestamp": datetime.now().isoformat(),
    }


def roda_sistema_paralelo(sistema: dict, pids: list[str], texto_por_id: dict,
                           config_path: Path, progresso: dict, contador_global: dict) -> bool:
    """Processa todos os pids pendentes de UM sistema em paralelo. Devolve
    False se abortou (saldo baixo ou disjuntor), True se terminou tudo."""
    sid, method, level = sistema["id"], sistema["method"], sistema["community_level"]
    n_workers = MAX_WORKERS_POR_METODO.get(method, 4)
    janela_status = deque(maxlen=JANELA_DISJUNTOR)
    abortado = False

    print(f"\n--- Iniciando {sid} ({method} L{level}) — {len(pids)} pendentes, "
          f"{n_workers} em paralelo ---")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(processa_um, sistema, pid, texto_por_id, config_path): pid
            for pid in pids
        }
        for future in as_completed(futures):
            pid = futures[future]
            chave = f"{sid}__{pid}"
            try:
                resultado = future.result()
            except Exception as exc:  # noqa: BLE001
                resultado = {
                    "sistema_id": sid, "prontuario_id": pid, "method": method, "community_level": level,
                    "doencas_previstas": [], "resposta_raw": f"ERROR: exceção não tratada — {exc}",
                    "duracao_s": 0, "status": "erro", "timestamp": datetime.now().isoformat(),
                }

            with _lock_progresso:
                progresso[chave] = resultado
                salva_progresso(progresso)
                contador_global["i"] += 1
                i = contador_global["i"]

            status, duracao, doencas = resultado["status"], resultado["duracao_s"], resultado["doencas_previstas"]
            janela_status.append(status)

            avisos = []
            if status != "ok":
                avisos.append("ERRO")
            esperado = DURACAO_ESPERADA_S.get(method, 60)
            if duracao > esperado * FATOR_ALERTA_LENTIDAO:
                avisos.append(f"LENTO (esperado ~{esperado}s)")
            if status == "ok" and not doencas and "nenhuma" not in resultado["resposta_raw"].lower():
                avisos.append("RESPOSTA VAZIA SEM 'NENHUMA' -- checar resposta_raw")

            marca = f"  <<< {' | '.join(avisos)}" if avisos else ""
            print(f"  [{i}] {sid} {pid}: {doencas} ({duracao:.0f}s){marca}")

            if len(janela_status) == JANELA_DISJUNTOR:
                taxa_erro = sum(1 for s in janela_status if s != "ok") / JANELA_DISJUNTOR
                if taxa_erro >= TAXA_ERRO_MAXIMA_DISJUNTOR:
                    print(f"\n⚠ ABORTADO: {taxa_erro:.0%} de erro nas últimas {JANELA_DISJUNTOR} "
                          f"conclusões de {sid} — possível regressão do bug de JSON/timeout. "
                          f"Progresso salvo, resumível após investigar.")
                    abortado = True

            if contador_global["i"] % 10 == 0:
                saldo = consulta_saldo()
                if saldo < SALDO_MINIMO:
                    print(f"\n⚠ ABORTADO: saldo US$ {saldo:.4f} abaixo do mínimo de segurança "
                          f"(US$ {SALDO_MINIMO:.2f}). Progresso salvo até aqui, resumível.")
                    abortado = True

            if abortado:
                executor.shutdown(wait=False, cancel_futures=True)
                break

    return not abortado


def main():
    print(f"Modelo: {MODELO_NOVO}")
    config_path = gera_config_ling30(GRAPHRAG_ROOT)
    print(f"Config gerada: {config_path}\n")

    gt_df = pd.read_csv(GROUND_TRUTH_CSV, dtype=str).fillna("")
    gt_df["prontuario_id"] = gt_df["prontuario_id"].str.strip()
    gt_por_id = {row["prontuario_id"]: {c for c in CLASSES if row[c] == "1"} for _, row in gt_df.iterrows()}
    print(f"Ground truth: {len(gt_por_id)} prontuários")

    texto_por_id = carrega_texto_por_prontuario(CSV_ORIGINAL, "prontuario", "descricao")
    print(f"Textos carregados: {len(texto_por_id)}\n")

    progresso = carrega_progresso()
    pendentes_por_sistema = {
        s["id"]: [pid for pid in gt_por_id
                  if pid in texto_por_id and progresso.get(f"{s['id']}__{pid}", {}).get("status") != "ok"]
        for s in SISTEMAS
    }
    total = len(SISTEMAS) * len(gt_por_id)
    total_pendentes = sum(len(v) for v in pendentes_por_sistema.values())
    saldo = consulta_saldo()
    print(f"Total de queries ({len(SISTEMAS)} sistemas x {len(gt_por_id)}): {total}")
    print(f"Já concluídas: {total - total_pendentes} | Pendentes: {total_pendentes}")
    print(f"Saldo atual: US$ {saldo:.4f} (trava de segurança em US$ {SALDO_MINIMO:.2f})")
    print(f"Cache: {PROGRESSO_PATH}\n")

    inicio_total = time.time()
    contador_global = {"i": 0}
    for sistema in SISTEMAS:
        pids = pendentes_por_sistema[sistema["id"]]
        if not pids:
            continue
        continua = roda_sistema_paralelo(sistema, pids, texto_por_id, config_path, progresso, contador_global)
        if not continua:
            break

    if total_pendentes:
        print(f"\n✔ Rodada concluída em {(time.time() - inicio_total) / 60:.1f} min.\n")
    else:
        print("✔ Nada pendente.\n")

    salva_predicoes_csv(progresso, gt_por_id)
    metricas = calcula_metricas(progresso, gt_por_id)
    pd.DataFrame(metricas).to_csv(METRICAS_CSV, index=False, encoding="utf-8")

    print(f"\n{'Sistema':<8}{'Método':<10}{'N':>5}{'Exact-match':>13}{'F1 macro':>10}{'F1 micro':>10}")
    for m in metricas:
        print(f"{m['sistema']:<8}{m['metodo'] + ' L' + str(m['community_level']):<10}{m['n_avaliado']:>5}"
              f"{(m['exact_match_accuracy'] or 0):>13.3f}{m['f1_macro']:>10.3f}{m['f1_micro']:>10.3f}")

    print(f"\nPredições: {PREDICOES_CSV}")
    print(f"Métricas: {METRICAS_CSV}")


if __name__ == "__main__":
    main()
