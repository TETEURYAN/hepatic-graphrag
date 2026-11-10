"""Testa se um prompt mais reforçado (v4, com três regras novas direcionadas)
reduz a alucinação/erro do `drift` em hepatite B e hepatite C -- as duas
doenças de marcador textual objetivo onde a investigação qualitativa
encontrou: (1) ignorar status de cura/resolução em hepatite C tratada com
DAA, (2) confundir marcador sorológico de imunidade vacinal (Anti-HBs
reagente) com infecção ativa de hepatite B, e (3) pelo menos um caso de
citação de resultado de exame inexistente no texto (alucinação pura,
paciente 592097: "HBsAg positivo (sorologia 05/06/23)" nao consta em
nenhum lugar do prontuário).

Roda CADA paciente com DOIS prompts, alternando, pra isolar o efeito do
prompt (mesma config, mesmo modelo, mesmo nível de comunidade, mesma
ausência de patches):
  - v3: o prompt atual (monta_prompt_multilabel, já usado em S1-S4 e no
    teste de HDA) -- serve de controle/replicação.
  - v4: v3 + três regras novas (ver PROMPT_V4_EXTRA) direcionadas aos três
    mecanismos acima.

IMPORTANTE (achado 2026-10-16): este teste usa `ling-3.0-flash`, mas a
rodada FINAL que gerou os falsos positivos originais (S7/S8, em
`progresso_cache.json`) na verdade rodou em `ling-2.6-flash` -- confirmado
pelo `model:` em `settings.yaml`/`settings_custo_tuned.yaml` e pelos tempos
reais do cache (S7 média 214s ali vs. horas aqui). `ling-2.6-flash` foi
descontinuado no OpenRouter (endpoints=[]), então não é mais possível
replicar a condição original exatamente -- este teste não "isola só o
prompt", é um experimento novo num modelo mais lento e com raciocínio, cujo
resultado deve ser interpretado como tal, não como réplica do S7.

ATUALIZAÇÃO (2026-10-19): a demora extrema (horas por consulta) não era
raciocínio nem modelo -- eram dois bugs reais, corrigidos hoje:
(1) `reasoning_effort: low` do GraphRAG nunca chegava ao OpenRouter (descartado
pelo litellm); `reasoning_effort_patch.py` agora injeta
`extra_body={"reasoning": {"enabled": False}}`, que de fato zera os tokens
de raciocínio invisíveis (confirmado: chamada de LLM caiu de
minutos/horas para 2-8s). (2) `graphrag_json_patch.py` chamava
`inspect.stack()` a cada chamada de LLM, resolvendo o caminho real de cada
frame no disco -- catastroficamente lento no disco WSL/Windows montado
deste projeto (/mnt/c/...). Reescrito pra ler `frame.f_globals["__name__"]`
direto, sem tocar o disco (1000 chamadas: de dezenas de segundos para
0,2ms). Este script, porém, NUNCA usava `use_patched_runner=True` antes de
hoje -- ou seja, nenhum dos dois patches chegava a ser carregado nas
tentativas anteriores. Corrigido junto com o timeout (4h -> 45min, já que
uma chamada individual agora é rápida; o que pode continuar lento é uma
busca vetorial real no LanceDB, não mais raciocínio ou nosso código).

Uso:
    uv run --no-sync python3 experiments/grid_search/testa_fix_alucinacao_hepb_hepc.py
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
from parser_doencas import DOENCAS_VALIDAS, extrai_doencas  # noqa: E402
from run_s1_s6_ling30 import monta_prompt_multilabel as monta_prompt_v3  # noqa: E402

RESULTADOS_DIR = _REPO_ROOT / "tcc-results-01"
MODELO = "inclusionai/ling-3.0-flash"

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

# pacientes com falso positivo de hepatite B ou C no drift (S7/S8) original,
# com a doença alvo (a que o drift alucinou/errou) e o mecanismo observado
PACIENTES_TESTE = [
    ("592097",  "hepatite_b", "alucinacao_pura"),
    ("1505692", "hepatite_b", "confusao_vacinal"),
    ("1828003", "hepatite_b", "promocao_contextual"),
    ("3678224", "hepatite_c", "ignora_cura"),
    ("7506371", "hepatite_c", "ignora_cura"),
    ("8486722", "hepatite_c", "ignora_cura"),
    ("8751638", "hepatite_c", "ignora_cura"),
]

PROMPT_V4_EXTRA = (
    "\n\nATENÇÃO a mais três regras, específicas para hepatite B e C:\n"
    "4) Se o texto indica que a hepatite C foi TRATADA e CURADA (ex.: 'PCR-HCV não detectável' ou "
    "'indetectável' após uso de antivirais de ação direta -- sofosbuvir, daclatasvir, velpatasvir etc. --, "
    "ou menção explícita a 'RVS'/'resposta virológica sustentada'/'curado'), NÃO liste hepatite C como "
    "presente: trate-a como doença RESOLVIDA, equivalente a ausente, mesmo que o histórico da doença "
    "apareça mencionado no texto.\n"
    "5) Para hepatite B, distinga marcadores de IMUNIDADE VACINAL de marcadores de INFECÇÃO ATIVA. "
    "'Anti-HBs reagente' ISOLADO (sem HBsAg positivo, sem Anti-HBc IgM ou IgG reagente, sem HBV-DNA "
    "detectável) indica apenas resposta à vacina ou imunidade adquirida -- NÃO é hepatite B. Só confirme "
    "hepatite B por HBsAg positivo, Anti-HBc reagente, HBV-DNA/PCR-HBV detectável, ou diagnóstico clínico "
    "explícito de hepatite B (não apenas histórico familiar ou menção lateral dentro de outro diagnóstico).\n"
    "6) Toda citação entre aspas na sua resposta deve ser uma cópia VERBATIM de um trecho que realmente "
    "existe no texto do prontuário abaixo -- nunca invente, combine ou parafraseie um resultado de exame, "
    "data ou achado que não esteja literalmente escrito no texto fornecido.\n"
)


def monta_prompt_v4(pront_id: str, texto: str) -> str:
    base = monta_prompt_v3(pront_id, texto)
    marcador = "Lista de doenças possíveis:"
    idx = base.find(marcador)
    return base[:idx] + PROMPT_V4_EXTRA.strip() + "\n\n" + base[idx:]


def gera_config(root: Path) -> Path:
    base_path = root / "settings.yaml"
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["models"]["default_chat_model"]["model"] = MODELO
    config["models"]["default_chat_model"]["reasoning_effort"] = "low"
    config["models"]["default_chat_model"]["max_tokens"] = 2000
    config.setdefault("drift_search", {}).update(DRIFT_SEARCH_TUNED)
    saida = root / "settings_teste_alucinacao_hepbc.yaml"
    with open(saida, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return saida


def processa_um(pid: str, doenca: str, mecanismo: str, versao_prompt: str, texto: str, config_path: Path) -> dict:
    monta = monta_prompt_v3 if versao_prompt == "v3" else monta_prompt_v4
    prompt = monta(pid, texto)
    resposta_raw, sucesso, duracao = run_graphrag_query(
        query=prompt, root=GRAPHRAG_ROOT, data=GRAPHRAG_DATA, method="drift",
        community_level=1, timeout=86400,  # 24h, a pedido explícito (2026-10-19) --
        # subprocess.communicate(timeout=...) usa time.monotonic() internamente, que
        # no Linux NÃO avança durante suspensão do sistema (CLOCK_MONOTONIC exclui
        # suspend; CLOCK_BOOTTIME inclui) -- tempo suspenso não conta pro timeout.
        use_patched_runner=True,  # CRÍTICO: sem isso, nenhum dos três patches de
        # 2026-10-19 (reasoning_effort_patch, graphrag_json_patch,
        # batch_report_embeddings_patch) é carregado -- bug encontrado nesta mesma
        # data: este script nunca usava os patches antes de hoje.
        config=config_path,
    )
    doencas = sorted(extrai_doencas(resposta_raw)) if sucesso else []
    return {
        "prontuario_id": pid, "doenca_alvo": doenca, "mecanismo": mecanismo,
        "versao_prompt": versao_prompt, "status": "ok" if sucesso else "erro",
        "duracao_s": duracao, "doencas_previstas": doencas,
        "resposta_raw": resposta_raw[:1000],
    }


def main():
    texto_por_id = carrega_texto_por_prontuario(CSV_ORIGINAL, "prontuario", "descricao")
    config_path = gera_config(GRAPHRAG_ROOT)
    print(f"Config: {config_path}")
    print(f"Testando {len(PACIENTES_TESTE)} pacientes x 2 versões de prompt (v3, v4) = "
          f"{len(PACIENTES_TESTE) * 2} consultas\n", flush=True)

    tarefas = []
    for pid, doenca, mecanismo in PACIENTES_TESTE:
        for versao in ("v3", "v4"):
            tarefas.append((pid, doenca, mecanismo, versao))

    resultados = {}
    saida_path = RESULTADOS_DIR / "teste_alucinacao_hepb_hepc.json"
    with ThreadPoolExecutor(max_workers=1) as executor:  # sequencial, mesma cautela do teste de HDA
        futures = {
            executor.submit(processa_um, pid, doenca, mecanismo, versao, texto_por_id[pid], config_path): (pid, versao)
            for pid, doenca, mecanismo, versao in tarefas if pid in texto_por_id
        }
        for future in as_completed(futures):
            chave = futures[future]
            pid, versao = chave
            try:
                r = future.result()
            except Exception as exc:  # noqa: BLE001
                r = {"prontuario_id": pid, "versao_prompt": versao, "status": "erro",
                     "doencas_previstas": [], "resposta_raw": f"EXCEPTION: {exc}", "duracao_s": 0}
            resultados[f"{pid}__{versao}"] = r
            doenca_alvo = r.get("doenca_alvo", "")
            ainda_errado = doenca_alvo in r.get("doencas_previstas", [])
            marca = ("✘ AINDA ALUCINA" if ainda_errado else "✔ CORRIGIDO") if r["status"] == "ok" else "ERRO"
            print(f"  {pid} [{versao}]: {r['status']} ({r.get('duracao_s', 0):.0f}s) "
                  f"-> {r.get('doencas_previstas', [])}  {marca}", flush=True)
            with open(saida_path, "w", encoding="utf-8") as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\n=== RESUMO ===")
    for versao in ("v3", "v4"):
        subset = [r for k, r in resultados.items() if k.endswith(f"__{versao}") and r["status"] == "ok"]
        corrigidos = [r for r in subset if r.get("doenca_alvo") not in r.get("doencas_previstas", [])]
        print(f"Prompt {versao}: {len(subset)} ok, {len(corrigidos)} corrigidos "
              f"(doença alvo NÃO listada), {len(subset) - len(corrigidos)} ainda alucinam")
    print(f"\nResultado salvo em: {saida_path}")


if __name__ == "__main__":
    main()
