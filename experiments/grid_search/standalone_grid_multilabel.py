# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "pandas",
#     "graphrag==2.7.0",
#     "pyyaml",
# ]
# ///
"""Bateria S1-S8 (multilabel) contra os 537 prontuários — script ÚNICO e
AUTOCONTIDO, pensado pra rodar num computador que NÃO tem este repositório
clonado (só precisa do índice GraphRAG já construído + 2 arquivos CSV).

Não depende de `src/core/*` nem de nenhum outro arquivo do projeto — toda a
lógica (normalização de ID, parser de doenças multilabel, runner de query
GraphRAG com timeout/kill de processo órfão, prompt, cache resumível e
cálculo de métricas) está inline neste único arquivo.

============================== PRÉ-REQUISITOS ==============================
1. O índice GraphRAG já construído nesta máquina (pasta com `output/` e
   `settings.yaml` — a mesma que você já usa pra rodar `graphrag query`).
2. O CSV bruto (não processado) com as colunas de prontuário/texto — mesmo
   arquivo que alimentou o índice.
3. O CSV do ground truth de 537 prontuários (`ground_truth_segunda_opiniao.csv`
   gerado no outro computador) — precisa copiar esse arquivo pra cá também,
   é só ~150KB.
4. `uv` instalado (https://docs.astral.sh/uv/). NÃO precisa `pip install`
   nada manualmente — o cabeçalho `# /// script ... ///` no topo deste
   arquivo diz ao `uv` pra criar um ambiente isolado com `pandas`, `pyyaml` e
   `graphrag==2.7.0` (mesma versão usada pra gerar o índice) automaticamente.

`USAR_CONFIG_TUNED = True` (mais abaixo) já vem ligado por padrão: gera
automaticamente, na primeira execução, um `settings_custo_tuned_auto.yaml`
do lado do `settings.yaml` do índice (a partir dele, só sobrescrevendo
`global_search`/`drift_search`) e usa isso em S5-S8 — medido empiricamente
em -68% de custo agregado (ver `teste_custo_global_drift_tuned.py`).
S1-S4 (basic/local) sempre rodam na config baseline do índice.

============================== COMO RODAR ===================================
1. Edite as constantes na seção "EDITE AQUI" logo abaixo com os caminhos
   reais desta máquina.
2. Rode:
       uv run standalone_grid_multilabel.py
3. Pode interromper (Ctrl+C, fechar o terminal, desligar o PC) a qualquer
   momento — o progresso é salvo em disco a cada query (`progresso_cache.json`
   por padrão). Rodar de novo retoma exatamente de onde parou, sem repetir
   nada que já tenha dado "ok".
4. Quando os 8 sistemas × 537 prontuários estiverem completos, o script
   detecta automaticamente e calcula as métricas (precision/recall/F1 por
   classe e por sistema, macro/micro, exact-match) — salva num CSV e imprime
   um resumo. Rodar de novo depois de completo só recalcula as métricas na
   hora, não refaz nenhuma query.
==============================================================================
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# =============================================================================
# EDITE AQUI — caminhos desta máquina
# =============================================================================

# Pasta raiz do índice GraphRAG (contém `output/`, `settings.yaml`, `prompts/`)
GRAPHRAG_ROOT = Path("./local copy")
GRAPHRAG_DATA = GRAPHRAG_ROOT / "output"

# CSV bruto (não processado) com o texto clínico — mesmo que alimentou o índice
CSV_BRUTO = Path("./data-hepato-sem-anonimizacao.csv")
COL_ID = "prontuario"
COL_TEXTO = "descricao"  # troque para "descricao_anonimizada" se for o dataset anonimizado

# CSV do ground truth de 537 prontuários (copiado do outro computador)
GROUND_TRUTH_CSV = Path("./ground_truth_segunda_opiniao.csv")

# Onde salvar o progresso resumível e os resultados
PROGRESSO_PATH = Path("./progresso_cache.json")
PREDICOES_CSV = Path("./predicoes_grid_multilabel.csv")
METRICAS_CSV = Path("./metricas_grid_multilabel.csv")

# Aplica a config de redução de custo (dynamic-community-selection + limites
# de contexto) em global/drift (S5-S8) — medida empiricamente: -68% de custo
# agregado em 2 prontuários de teste (S6 -71%/-52%, S7 -82%/-66%, S8 -59%/-74%,
# S5 misto: -51% num paciente, +245% no outro em termos relativos, mas
# pequeno em valor absoluto ~US$0,003). S1-S4 (basic/local) não usam nenhum
# desses parâmetros — sempre rodam com a config baseline do índice.
USAR_CONFIG_TUNED = True

# =============================================================================
# Fim da seção editável — o resto roda sem precisar mexer
# =============================================================================

CLASSES = [
    "cirrose_hepatica", "esquistossomose", "hepatite_c", "hepatite_b",
    "esteatose_hepatica", "hipertensao_portal", "hemorragia_digestiva_alta", "pbe",
]

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

DEFAULT_TIMEOUTS = {"drift": 2400, "global": 1800, "local": 600, "basic": 600}

# Overrides de custo — mesmos valores validados em
# experiments/grid_search/teste_custo_global_drift_tuned.py (ver defaults de
# fábrica em graphrag/config/defaults.py: GlobalSearchDefaults/DriftSearchDefaults).
GLOBAL_SEARCH_TUNED = {
    "dynamic_search_threshold": 3,       # default 1 -- só segue quem for rating >= 3/5
    "dynamic_search_use_summary": True,  # default False -- usa resumo curto na fase de rating
    "dynamic_search_max_level": 1,       # default 2 -- poda a hierarquia dinâmica mais cedo
    "dynamic_search_keep_parent": False,
    "dynamic_search_num_repeats": 1,
    "data_max_tokens": 6000,             # default 12000
    "max_context_tokens": 6000,          # default 12000
    "map_max_length": 500,               # default 1000 (palavras)
    "reduce_max_length": 1000,           # default 2000
}
DRIFT_SEARCH_TUNED = {
    "n_depth": 1,                        # default 3
    "drift_k_followups": 6,              # default 20
    "primer_folds": 3,                   # default 5
    "primer_llm_max_tokens": 6000,       # default 12000
    "data_max_tokens": 6000,             # default 12000
    "local_search_max_data_tokens": 6000,  # default 12000
    "local_search_top_k_mapped_entities": 5,   # default 10
    "local_search_top_k_relationships": 5,     # default 10
}


# =============================================================================
# Normalização de ID / carregamento de texto (equivalente a
# src/core/dataset/clinical_text.py)
# =============================================================================

_PREFIXO_RE = re.compile(r'^PACIENTE\s*-\s*', re.IGNORECASE)


def normaliza_prontuario_id(pid: str) -> str:
    return _PREFIXO_RE.sub('', str(pid).strip()).strip()


def carrega_texto_por_prontuario(csv_path: Path, col_id: str, col_texto: str) -> dict[str, str]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df[col_id] = df[col_id].apply(normaliza_prontuario_id)
    df = df.drop_duplicates(subset=[col_id, col_texto], keep="first")
    return (
        df.groupby(col_id)[col_texto]
        .apply(lambda x: "\n--- Atendimento ---\n".join(x.tolist()))
        .to_dict()
    )


# =============================================================================
# Parser multilabel de doenças (equivalente a experiments/grid_search/parser_doencas.py)
# =============================================================================

MAPA_DOENCAS = {
    'cirrose': 'cirrose_hepatica', 'cirrose hepatica': 'cirrose_hepatica',
    'cirrose hepatica descompensada': 'cirrose_hepatica', 'cirrose hepatica alcoolica': 'cirrose_hepatica',
    'cirrose hepatica viral c': 'cirrose_hepatica', 'cirrose micronodular': 'cirrose_hepatica',
    'cirrose descompensada': 'cirrose_hepatica', 'cirrose hepatica cronica': 'cirrose_hepatica',
    'hipertensao portal': 'hipertensao_portal', 'hipertensao porta': 'hipertensao_portal',
    'sindrome de hipertensao portal': 'hipertensao_portal', 'sindrome da hipertensao portal': 'hipertensao_portal',
    'hipertensao portal nao cirrotica': 'hipertensao_portal', 'ehe/hp': 'hipertensao_portal',
    'esquistossomose': 'esquistossomose', 'equistossomose': 'esquistossomose',
    'esquistossomose hepatoesplenica': 'esquistossomose', 'esquistossomose hepatoesplenica (ehe)': 'esquistossomose',
    'esquistossomose mansoni': 'esquistossomose', 'esquistossomose mansonica': 'esquistossomose',
    'ehe': 'esquistossomose',
    'hepatite c': 'hepatite_c', 'hepatite c (hcv)': 'hepatite_c', 'hepatite c cronica': 'hepatite_c',
    'hepatite cronica viral c': 'hepatite_c',
    'hepatite b': 'hepatite_b', 'hepatite b (hbv)': 'hepatite_b', 'hepatite b cronica': 'hepatite_b',
    'hepatite b inativa': 'hepatite_b', 'hepatite b resolvida': 'hepatite_b',
    'hepatite cronica b': 'hepatite_b', 'hepatite cronica viral b': 'hepatite_b',
    'virus b (hepatite b)': 'hepatite_b',
    'esteatose': 'esteatose_hepatica', 'esteatose hepatica': 'esteatose_hepatica',
    'esteatose hepatica difusa': 'esteatose_hepatica', 'dhgna': 'esteatose_hepatica',
    'nafld': 'esteatose_hepatica', 'masld': 'esteatose_hepatica', 'mash': 'esteatose_hepatica',
    'nash': 'esteatose_hepatica', 'doenca hepatica gordurosa nao alcoolica': 'esteatose_hepatica',
    'doenca hepatica gordurosa nao alcoolica (dhgna)': 'esteatose_hepatica',
    'doenca hepatica esteatotica metabolica': 'esteatose_hepatica',
    'doenca hepatica esteatotica metabolica (mash)': 'esteatose_hepatica',
    'esteato-hepatite nao alcoolica': 'esteatose_hepatica', 'esteato-hepatite nao alcoolica (nash)': 'esteatose_hepatica',
    'esteatohepatite': 'esteatose_hepatica', 'esteatohepatite nao alcoolica': 'esteatose_hepatica',
    'hda': 'hemorragia_digestiva_alta', 'hemorragia digestiva alta': 'hemorragia_digestiva_alta',
    'hemorragia digestiva alta varicosa': 'hemorragia_digestiva_alta', 'hemorragia varicosa': 'hemorragia_digestiva_alta',
    'pbe': 'pbe', 'peritonite bacteriana espontanea': 'pbe', 'peritonite bacteriana espontanea (pbe)': 'pbe',
    'nenhuma': 'nenhuma',
}
_TERMOS_ORDENADOS = sorted(MAPA_DOENCAS.items(), key=lambda kv: -len(kv[0]))

_MARCADORES_NEGACAO = [
    'ausente', 'ausencia', 'descartad', 'exclu', 'negativ',
    'nao confirmad', 'nao evidenciad', 'sem evidencia', 'nao ha evidencia',
    'negado', 'nao documentad', 'nao presente', 'nega ', 'nego ', 'negou ',
    r'\bnr\b',
]
_RESPOSTA_ANCORADA_RE = re.compile(r'<RESPOSTA>(.*?)</RESPOSTA>', re.IGNORECASE | re.DOTALL)


def _normalize(texto: str) -> str:
    texto = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in texto if not unicodedata.combining(c)).lower()


def _linha_negada(linha_norm: str) -> bool:
    return any(re.search(m, linha_norm) for m in _MARCADORES_NEGACAO)


def extrai_bloco_ancorado(resposta: str) -> str:
    m = _RESPOSTA_ANCORADA_RE.search(resposta)
    return m.group(1) if m else resposta


def extrai_doencas(resposta: str) -> set[str]:
    if not resposta:
        return set()
    resposta = extrai_bloco_ancorado(resposta)
    encontradas = set()
    for linha in resposta.split('\n'):
        linha_norm = _normalize(linha)
        if _linha_negada(linha_norm):
            continue
        for termo, classe in _TERMOS_ORDENADOS:
            if classe == 'nenhuma':
                continue
            termo_norm = _normalize(termo)
            if re.search(r'\b' + re.escape(termo_norm) + r'\b', linha_norm):
                encontradas.add(classe)
    return encontradas


# =============================================================================
# Prompt multilabel (mesmo usado na segunda opinião / validação do grid)
# =============================================================================

DOENCAS_TEXTO = (
    "Cirrose hepática, Esquistossomose, Hepatite B, Hepatite C, Esteatose hepática, "
    "Hipertensão portal, Hemorragia digestiva alta, Peritonite bacteriana espontânea"
)


def monta_prompt_multilabel(pront_id: str, texto: str) -> str:
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
        f"Lista de doenças possíveis: {DOENCAS_TEXTO}.\n\n"
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


# =============================================================================
# Config tuned — gerada a partir do settings.yaml que já existe no índice,
# só sobrescrevendo as seções global_search/drift_search. Não mexe no
# settings.yaml original, escreve um arquivo novo do lado.
# =============================================================================

def gera_config_tuned(root: Path) -> Path:
    base_path = root / "settings.yaml"
    if not base_path.exists():
        raise SystemExit(f"✗ {base_path} não encontrado — não dá pra gerar a config tuned.")

    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("global_search", {}).update(GLOBAL_SEARCH_TUNED)
    config.setdefault("drift_search", {}).update(DRIFT_SEARCH_TUNED)

    saida = root / "settings_custo_tuned_auto.yaml"
    with open(saida, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return saida


# =============================================================================
# Runner de query GraphRAG (equivalente a src/core/query/runner.py)
# =============================================================================

_SUCCESS_RE = re.compile(r"SUCCESS:\s*(.+)", re.IGNORECASE | re.DOTALL)


def run_graphrag_query(query: str, root: Path, data: Path, method: str,
                        community_level: int, timeout: int,
                        config: Path | None = None,
                        dynamic_community_selection: bool = False) -> tuple[str, bool, float]:
    cmd = [
        "graphrag", "query",
        "--root", str(root), "--data", str(data), "--method", method,
        "--community-level", str(community_level),
    ]
    if config is not None:
        cmd += ["--config", str(config)]
    if dynamic_community_selection:
        cmd += ["--dynamic-community-selection"]
    cmd += [
        "--response-type", "Only the <RESPOSTA>...</RESPOSTA> block requested, nothing else",
        "--query", query,
    ]
    inicio = time.time()
    kwargs = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # permite matar o grupo inteiro no timeout

    try:
        processo = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", **kwargs,
        )
        try:
            stdout, stderr = processo.communicate(timeout=timeout)
            returncode = processo.returncode
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(processo.pid), signal.SIGKILL)
                else:
                    processo.kill()
            except (ProcessLookupError, OSError):
                pass
            processo.communicate()
            return f"ERROR: query timed out after {timeout}s", False, float(timeout)
    except FileNotFoundError:
        return "ERROR: 'graphrag' não encontrado no PATH", False, round(time.time() - inicio, 1)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: unexpected exception — {exc}", False, round(time.time() - inicio, 1)

    duracao = round(time.time() - inicio, 1)
    if returncode != 0:
        stderr_tail = (stderr or "").strip()[-800:]
        return f"ERROR: exit {returncode} — {stderr_tail}", False, duracao

    stdout = stdout or ""
    match = _SUCCESS_RE.search(stdout)
    resposta = match.group(1).strip() if match else stdout.strip()
    return resposta, True, duracao


# =============================================================================
# Cache resumível
# =============================================================================

def carrega_progresso() -> dict:
    if PROGRESSO_PATH.exists():
        with open(PROGRESSO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def salva_progresso(progresso: dict) -> None:
    with open(PROGRESSO_PATH, "w", encoding="utf-8") as f:
        json.dump(progresso, f, ensure_ascii=False, indent=2)


# =============================================================================
# Métricas multilabel
# =============================================================================

def calcula_metricas(progresso: dict, gt_por_id: dict[str, set[str]]) -> list[dict]:
    linhas = []
    for sistema in SISTEMAS:
        sid = sistema["id"]
        por_classe = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in CLASSES}
        exact_matches = 0
        n_avaliado = 0

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


# =============================================================================
# Main
# =============================================================================

def main():
    for caminho, nome in [(GRAPHRAG_ROOT, "GRAPHRAG_ROOT"), (CSV_BRUTO, "CSV_BRUTO"), (GROUND_TRUTH_CSV, "GROUND_TRUTH_CSV")]:
        if not caminho.exists():
            raise SystemExit(f"✗ {nome} não encontrado: {caminho} — edite a constante no topo do arquivo.")

    print("Carregando ground truth...")
    gt_df = pd.read_csv(GROUND_TRUTH_CSV, dtype=str).fillna("")
    gt_df["prontuario_id"] = gt_df["prontuario_id"].apply(normaliza_prontuario_id)
    gt_por_id = {
        row["prontuario_id"]: {c for c in CLASSES if row[c] == "1"}
        for _, row in gt_df.iterrows()
    }
    print(f"  {len(gt_por_id)} prontuários no ground truth.\n")

    print("Carregando texto bruto (agrupando por prontuário normalizado)...")
    texto_por_id = carrega_texto_por_prontuario(CSV_BRUTO, COL_ID, COL_TEXTO)
    faltando = set(gt_por_id) - set(texto_por_id)
    if faltando:
        print(f"  ⚠ {len(faltando)} prontuários do ground truth sem texto no CSV bruto: {sorted(faltando)[:10]}...")
    print(f"  {len(texto_por_id)} prontuários com texto disponível.\n")

    config_tuned = None
    if USAR_CONFIG_TUNED:
        print("Gerando config de custo reduzido (dynamic-community-selection + limites de contexto)...")
        config_tuned = gera_config_tuned(GRAPHRAG_ROOT)
        print(f"  {config_tuned}\n")

    progresso = carrega_progresso()

    pendentes = [
        (s, pid) for s in SISTEMAS for pid in gt_por_id
        if pid in texto_por_id and progresso.get(f"{s['id']}__{pid}", {}).get("status") != "ok"
    ]
    total_queries = len(SISTEMAS) * len(gt_por_id)
    print(f"Total de queries (8 sistemas × {len(gt_por_id)} prontuários): {total_queries}")
    print(f"Já concluídas: {total_queries - len(pendentes)} | Pendentes: {len(pendentes)}")
    print(f"Cache: {PROGRESSO_PATH}\n")

    inicio_total = time.time()
    for i, (sistema, pid) in enumerate(pendentes, 1):
        sid, method, level = sistema["id"], sistema["method"], sistema["community_level"]
        chave = f"{sid}__{pid}"
        texto = texto_por_id[pid]
        prompt = monta_prompt_multilabel(pid, texto)
        timeout = DEFAULT_TIMEOUTS[method]

        # S1-S4 (basic/local) sempre na config baseline -- os parâmetros
        # tuned só existem em global_search/drift_search, não afetam esses.
        usa_tuned = USAR_CONFIG_TUNED and method in ("global", "drift")
        resposta_raw, sucesso, duracao = run_graphrag_query(
            prompt, GRAPHRAG_ROOT, GRAPHRAG_DATA, method, level, timeout,
            config=config_tuned if usa_tuned else None,
            dynamic_community_selection=usa_tuned and method == "global",
        )
        doencas = sorted(extrai_doencas(resposta_raw)) if sucesso else []
        status = "ok" if sucesso else "erro"

        progresso[chave] = {
            "sistema_id": sid, "prontuario_id": pid, "method": method, "community_level": level,
            "doencas_previstas": doencas, "resposta_raw": resposta_raw[:1500],
            "duracao_s": duracao, "status": status, "timestamp": datetime.now().isoformat(),
        }
        salva_progresso(progresso)  # salva a CADA query -- resumível se desligar/interromper

        marca = "" if status == "ok" else "  <<< ERRO"
        print(f"  [{i}/{len(pendentes)}] {sid} {pid}: {doencas} ({duracao:.0f}s){marca}")

    if pendentes:
        print(f"\n✔ Bateria concluída em {(time.time() - inicio_total) / 60:.1f} min.\n")
    else:
        print("✔ Nada pendente — todas as queries já estavam em cache.\n")

    print("Calculando métricas...")
    salva_predicoes_csv(progresso, gt_por_id)
    metricas = calcula_metricas(progresso, gt_por_id)
    pd.DataFrame(metricas).to_csv(METRICAS_CSV, index=False, encoding="utf-8")

    print(f"\n{'Sistema':<8}{'Método':<10}{'N':>5}{'Exact-match':>13}{'F1 macro':>10}{'F1 micro':>10}")
    for m in metricas:
        print(f"{m['sistema']:<8}{m['metodo'] + ' L' + str(m['community_level']):<10}{m['n_avaliado']:>5}"
              f"{(m['exact_match_accuracy'] or 0):>13.3f}{m['f1_macro']:>10.3f}{m['f1_micro']:>10.3f}")

    print(f"\nPredições detalhadas: {PREDICOES_CSV}")
    print(f"Métricas por sistema: {METRICAS_CSV}")


if __name__ == "__main__":
    main()
