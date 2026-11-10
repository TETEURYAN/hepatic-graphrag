"""Testa se o prompt reforçado (regra explícita "HDA != sangramento baixo")
corrige os falsos positivos de hemorragia_digestiva_alta que o `drift`
original (S7, sem essa regra) cometeu.

Reusa a MESMA config tuned (DRIFT_SEARCH_TUNED) e o MESMO modelo
(ling-3.0-flash) do run original de S7 -- a ÚNICA variável trocada é o
prompt (o antigo, sem a regra de HDA, vs o novo, com ela).

Uso:
    uv run --no-sync python3 experiments/grid_search/testa_fix_hda_drift.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.dataset.clinical_text import carrega_texto_por_prontuario  # noqa: E402
from src.core.paths import CSV_ORIGINAL, GRAPHRAG_DATA, GRAPHRAG_ROOT  # noqa: E402
from src.core.query.runner import run_graphrag_query  # noqa: E402

sys.path.insert(0, str(_REPO_ROOT / "experiments" / "grid_search"))
from parser_doencas import extrai_doencas  # noqa: E402
from run_s1_s6_ling30 import monta_prompt_multilabel  # noqa: E402 (prompt REFORÇADO, com a regra de HDA)

RESULTADOS_DIR = _REPO_ROOT / "tcc-results-01"
MODELO = "inclusionai/ling-3.0-flash"  # mesmo modelo que o S7 original usou

DRIFT_SEARCH_TUNED = {
    "n_depth": 1,
    "drift_k_followups": 6,
    "primer_folds": 3,
    "primer_llm_max_tokens": 6000,
    "data_max_tokens": 6000,
    "local_search_max_data_tokens": 6000,
    "local_search_top_k_mapped_entities": 5,
    "local_search_top_k_relationships": 5,
}

# amostra de pacientes onde S7 (prompt ANTIGO) deu falso positivo de HDA
PACIENTES_TESTE = [
    "566844", "592097", "958660", "1028182", "1047554", "1505692",
    "1874197", "1944347", "2318525", "2566438", "2651230", "2755270",
]


def gera_config(root: Path) -> Path:
    base_path = root / "settings.yaml"
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["models"]["default_chat_model"]["model"] = MODELO
    config["models"]["default_chat_model"]["reasoning_effort"] = "low"
    config["models"]["default_chat_model"]["max_tokens"] = 2000
    config.setdefault("drift_search", {}).update(DRIFT_SEARCH_TUNED)
    saida = root / "settings_teste_fix_hda.yaml"
    with open(saida, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return saida


def processa_um(pid: str, texto: str, config_path: Path) -> dict:
    prompt = monta_prompt_multilabel(pid, texto)
    inicio = time.time()
    # SEM use_patched_runner -- o run original de S7/S8 (standalone_grid_multilabel.py)
    # chamava `graphrag query` direto, sem nenhum patch (nem reasoning_effort_patch,
    # nem graphrag_json_patch). Pra isolar SÓ a variável do prompt, replico exatamente
    # essa condição -- usar os patches aqui seria introduzir uma segunda variável
    # (e o reasoning_effort_patch parece interagir mal com as chamadas encadeadas
    # do drift, achado em teste: 4/4 timeouts de 45min com use_patched_runner=True,
    # mesmo sequencial).
    resposta_raw, sucesso, duracao = run_graphrag_query(
        query=prompt, root=GRAPHRAG_ROOT, data=GRAPHRAG_DATA, method="drift",
        community_level=1, timeout=14400,  # 4h -- margem bem maior, ja que ate 90min nao foi suficiente
        config=config_path,
    )
    doencas = sorted(extrai_doencas(resposta_raw)) if sucesso else []
    return {
        "prontuario_id": pid, "status": "ok" if sucesso else "erro",
        "duracao_s": duracao, "doencas_previstas": doencas,
        "resposta_raw": resposta_raw[:800],
    }


def main():
    texto_por_id = carrega_texto_por_prontuario(CSV_ORIGINAL, "prontuario", "descricao")
    config_path = gera_config(GRAPHRAG_ROOT)
    print(f"Config: {config_path}")
    print(f"Testando {len(PACIENTES_TESTE)} pacientes (FP de HDA no S7 original)\n")

    resultados = {}
    with ThreadPoolExecutor(max_workers=1) as executor:  # sequencial -- evita contencao de recursos
        futures = {
            executor.submit(processa_um, pid, texto_por_id[pid], config_path): pid
            for pid in PACIENTES_TESTE if pid in texto_por_id
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                r = future.result()
            except Exception as exc:  # noqa: BLE001
                r = {"prontuario_id": pid, "status": "erro", "doencas_previstas": [],
                     "resposta_raw": f"EXCEPTION: {exc}", "duracao_s": 0}
            resultados[pid] = r
            corrigido = "hemorragia_digestiva_alta" not in r["doencas_previstas"]
            marca = "✔ CORRIGIDO" if r["status"] == "ok" and corrigido else (
                "✘ AINDA ERRA" if r["status"] == "ok" else "ERRO")
            print(f"  {pid}: {r['status']} ({r['duracao_s']:.0f}s) -> {r['doencas_previstas']}  {marca}", flush=True)

            with open(RESULTADOS_DIR / "teste_fix_hda_drift.json", "w", encoding="utf-8") as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\n=== RESUMO ===")
    ok = [r for r in resultados.values() if r["status"] == "ok"]
    corrigidos = [r for r in ok if "hemorragia_digestiva_alta" not in r["doencas_previstas"]]
    print(f"Processados com sucesso: {len(ok)}/{len(PACIENTES_TESTE)}")
    print(f"Corrigidos (HDA sumiu da previsão): {len(corrigidos)}/{len(ok)}")
    print(f"Ainda com falso positivo de HDA: {len(ok) - len(corrigidos)}/{len(ok)}")
    print(f"\nResultado salvo em: {RESULTADOS_DIR / 'teste_fix_hda_drift.json'}")


if __name__ == "__main__":
    main()
