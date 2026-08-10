# %% [markdown]
# # Experimentos GraphRAG — Grid de Configurações
# **Objetivo:** Avaliar 8 sistemas (4 modos × 2 community-levels) nos 141 prontuários do ground truth.
#
# | Sistema | Modo | Community Level |
# |---|---|---|
# | S1 | basic | 1 |
# | S2 | basic | 2 |
# | S3 | local | 1 |
# | S4 | local | 2 |
# | S5 | global | 1 |
# | S6 | global | 2 |
# | S7 | drift | 1 (default) |
# | S8 | drift | 2 |
#
# **Progresso salvo automaticamente** — pode interromper e continuar a qualquer momento.
#
# Reprodução: rode este script (célula a célula, num Jupyter/VS Code interactive
# window) com o repositório clonado e `experimentos/<seu índice>` já indexado
# (`graphrag index`). Não roda sozinho sem um índice GraphRAG local — ver
# README.md desta pasta.

# %% [markdown]
# ## Célula 1 — Configuração

# %%
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display

# Bootstrap mínimo — só o suficiente pra importar src.core.paths, que resolve
# o resto (inclusive o path deste diretório, pra `import parser_doencas`)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.paths import (  # noqa: E402
    CSV_ORIGINAL,
    GRAPHRAG_DATA,
    GRAPHRAG_ROOT,
    GROUND_TRUTH_CSV,
    PROGRESSO_GRID_DIR,
    RESULTADOS_GRID_DIR,
    bootstrap_sys_path,
)

bootstrap_sys_path(Path(__file__).resolve().parent)

from parser_doencas import DOENCAS_VALIDAS, extrai_doenca  # noqa: E402
from src.core.query.progress import ProgressTracker  # noqa: E402
from src.core.query.runner import default_timeout, run_graphrag_query  # noqa: E402

# Caminhos vêm de src/core/paths.py (sobrescrevíveis via .env) — ajuste lá,
# não aqui.
GROUND_TRUTH = GROUND_TRUTH_CSV
COL_ID = "prontuario"
COL_TEXTO = "descricao"
OUTPUT_DIR = RESULTADOS_GRID_DIR
PROGRESSO_DIR = PROGRESSO_GRID_DIR

# Pausa entre queries (segundos) — evita rate limiting
PAUSA_ENTRE_QUERIES = 6
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESSO_DIR.mkdir(parents=True, exist_ok=True)

# Grid de experimentos
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

print("✔ Configuração carregada.")
print(f"  GraphRAG root : {GRAPHRAG_ROOT}")
print(f"  Output dir    : {OUTPUT_DIR}")
print(f"  Progresso dir : {PROGRESSO_DIR}")
print(f"  Sistemas      : {len(SISTEMAS)}")

# %% [markdown]
# ## Célula 2 — Carrega dados

# %%
# Ground truth
gt = pd.read_csv(GROUND_TRUTH, dtype=str).fillna("")
gt["prontuario_id"] = gt["prontuario_id"].str.strip()
lista_prontuarios = gt["prontuario_id"].tolist()

# Textos originais
df_orig = pd.read_csv(CSV_ORIGINAL, dtype=str).fillna("")
df_orig[COL_ID] = df_orig[COL_ID].str.strip()

# Agrega textos por prontuário (pode ter múltiplas consultas)
texto_por_id = (
    df_orig.groupby(COL_ID)[COL_TEXTO]
    .apply(lambda x: "\n--- Atendimento ---\n".join(x.tolist()))
    .to_dict()
)

print(f"Ground truth   : {len(gt)} prontuários")
print(f"Textos carregados : {len(texto_por_id)}")

# Verifica prontuários sem texto
sem_texto = [p for p in lista_prontuarios if p not in texto_por_id]
if sem_texto:
    print(f"⚠  {len(sem_texto)} prontuários sem texto: {sem_texto[:5]}")
else:
    print("✔ Todos os prontuários têm texto disponível.")

# %% [markdown]
# ## Célula 3 — Funções principais

# %%
def monta_prompt(pront_id: str, texto: str) -> str:
    """Monta o prompt clínico para classificação."""
    doencas = ", ".join(DOENCAS_VALIDAS[:-1]) + " ou Nenhuma"
    return (
        f"Abaixo estão as evoluções/anamneses de um único paciente, "
        f"extraídas do prontuário eletrônico (nº {pront_id}). "
        f"Com base EXCLUSIVAMENTE nesse conteúdo (usando seu conhecimento médico geral apenas para interpretá-lo), "
        f"qual é a doença hepática principal desse paciente? "
        f"Escolha apenas uma entre: {doencas}. "
        f"Se nenhuma dessas condições estiver claramente documentada, responda 'Nenhuma'. "
        f"Responda em uma frase curta, citando apenas o nome da doença.\n"
        f"--- PRONTUÁRIO {pront_id} ---\n"
        f"{texto[:8000]}\n"  # limita para não estourar contexto
        f"--- FIM DO PRONTUÁRIO {pront_id} ---"
    )


def roda_query(pront_id: str, texto: str, method: str, community_level: int) -> dict:
    """Executa uma query no GraphRAG (via src.core.query.runner) e devolve o resultado."""
    prompt = monta_prompt(pront_id, texto)
    resposta_raw, sucesso, duracao = run_graphrag_query(
        query=prompt,
        root=GRAPHRAG_ROOT,
        data=GRAPHRAG_DATA,
        method=method,
        community_level=community_level,
        response_type="Single sentence",
        timeout=default_timeout(method),
    )

    if not sucesso and resposta_raw.startswith("ERROR: query timed out"):
        return {
            "prontuario_id": pront_id,
            "doenca_prevista": "TIMEOUT",
            "resposta_raw": "",
            "duracao_s": duracao,
            "status": "timeout",
            "returncode": -1,
            "timestamp": datetime.now().isoformat(),
        }

    doenca = extrai_doenca(resposta_raw) if sucesso else "ERRO"
    status = "ok" if sucesso else "erro_returncode"

    return {
        "prontuario_id": pront_id,
        "doenca_prevista": doenca,
        "resposta_raw": resposta_raw[:500],
        "duracao_s": duracao,
        "status": status,
        "returncode": 0 if sucesso else -1,
        "timestamp": datetime.now().isoformat(),
    }


def salva_csv_sistema(sistema_id: str, progresso: dict, sistema_cfg: dict):
    """Salva CSV parcial/final de um sistema."""
    linhas = list(progresso.values())
    if not linhas:
        return
    df = pd.DataFrame(linhas)
    df["method"] = sistema_cfg["method"]
    df["community_level"] = sistema_cfg["community_level"]
    df["sistema_id"] = sistema_id
    path = OUTPUT_DIR / f"resultado_{sistema_id}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")


print("✔ Funções carregadas.")

# %% [markdown]
# ## Célula 4 — Status atual do progresso

# %%
print("Status atual dos experimentos:\n")
print(f'{"Sistema":<8} {"Modo":<10} {"Level":<8} {"Concluídos":<12} {"Pendentes":<10} {"Timeouts":<10} {"Erros"}')
print("-" * 65)

for s in SISTEMAS:
    tracker = ProgressTracker(PROGRESSO_DIR / f'progresso_{s["id"]}.json')
    concluidos = len(tracker)
    pendentes = len(lista_prontuarios) - concluidos
    timeouts = sum(1 for v in tracker.valores() if v.get("status") == "timeout")
    erros = sum(1 for v in tracker.valores() if "erro" in v.get("status", "") or "excecao" in v.get("status", ""))
    print(f'{s["id"]:<8} {s["method"]:<10} {s["community_level"]:<8} {concluidos:<12} {pendentes:<10} {timeouts:<10} {erros}')

print(f"\nTotal de prontuários por sistema: {len(lista_prontuarios)}")
print(f"Total de execuções no grid      : {len(lista_prontuarios) * len(SISTEMAS)}")

# %% [markdown]
# ## Célula 5 — Execução do grid
#
# **Configure `SISTEMAS_RODAR` para escolher quais sistemas executar.**
# Use `'todos'` para rodar o grid completo, ou liste os IDs: `['S1', 'S2']`

# %%
# ── CONFIGURE AQUI ───────────────────────────────────────────────
SISTEMAS_RODAR = "todos"  # 'todos' ou lista: ['S1', 'S3', 'S7']
MODO_TESTE = False  # True = roda só 3 prontuários por sistema (piloto)
# ────────────────────────────────────────────────────────────────

sistemas_selecionados = SISTEMAS if SISTEMAS_RODAR == "todos" else [
    s for s in SISTEMAS if s["id"] in SISTEMAS_RODAR
]

prontuarios_rodar = lista_prontuarios[:3] if MODO_TESTE else lista_prontuarios

print(f'Sistemas selecionados : {[s["id"] for s in sistemas_selecionados]}')
print(f"Prontuários por sistema: {len(prontuarios_rodar)}")
print(f"Total de execuções    : {len(sistemas_selecionados) * len(prontuarios_rodar)}")

# Estimativa de tempo
tempos_estimados = {"basic": 200, "local": 300, "global": 300, "drift": 2400}
tempo_total = sum(
    tempos_estimados[s["method"]] * len(prontuarios_rodar)
    for s in sistemas_selecionados
)
print(f"Tempo estimado        : {tempo_total // 3600}h {(tempo_total % 3600) // 60}min")

confirma = input("\nConfirmar execução? [s/N]: ").strip().lower()
if confirma != "s":
    print("Execução cancelada.")
else:
    inicio_total = time.time()

    for idx_s, sistema in enumerate(sistemas_selecionados):
        sid = sistema["id"]
        method = sistema["method"]
        level = sistema["community_level"]

        print(f'\n{"=" * 60}')
        print(f"Sistema {sid} — {method.upper()} / community-level {level}")
        print(f'{"=" * 60}')

        tracker = ProgressTracker(PROGRESSO_DIR / f"progresso_{sid}.json")
        pendentes = tracker.pendentes(prontuarios_rodar)

        if not pendentes:
            print(f"  ✔ Já concluído ({len(tracker)} prontuários). Pulando.")
            continue

        print(f"  Já concluídos : {len(tracker)}")
        print(f"  Pendentes     : {len(pendentes)}")

        for idx_p, pront_id in enumerate(pendentes):
            texto = texto_por_id.get(pront_id, "")
            if not texto:
                tracker.registra(pront_id, {
                    "prontuario_id": pront_id,
                    "doenca_prevista": "SEM_TEXTO",
                    "resposta_raw": "",
                    "duracao_s": 0,
                    "status": "sem_texto",
                    "returncode": -1,
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            pct = (idx_p + 1) / len(pendentes) * 100
            elapsed = time.time() - inicio_total
            print(
                f"  [{idx_p + 1:>3}/{len(pendentes)}] {pct:>5.1f}% | "
                f"ID: {pront_id} | "
                f"Elapsed: {int(elapsed // 60)}m{int(elapsed % 60):02d}s",
                end=" ... ",
                flush=True,
            )

            resultado = roda_query(pront_id, texto, method, level)

            status_icon = "✔" if resultado["status"] == "ok" else "⚠" if resultado["status"] == "timeout" else "✗"
            print(f"{status_icon} {resultado['doenca_prevista']:<30} [{resultado['duracao_s']}s]")

            tracker.registra(pront_id, resultado)
            salva_csv_sistema(sid, tracker._dados, sistema)

            if idx_p < len(pendentes) - 1:
                time.sleep(PAUSA_ENTRE_QUERIES)

        total = len(tracker)
        ok = sum(1 for v in tracker.valores() if v["status"] == "ok")
        timeouts = sum(1 for v in tracker.valores() if v["status"] == "timeout")
        erros = total - ok - timeouts
        print(f"\n  Sistema {sid} concluído: {ok} OK | {timeouts} timeouts | {erros} erros")
        salva_csv_sistema(sid, tracker._dados, sistema)

    tempo_total_real = time.time() - inicio_total
    print(f'\n{"=" * 60}')
    print(f"Grid concluído em {int(tempo_total_real // 3600)}h {int((tempo_total_real % 3600) // 60)}min")
    print(f'{"=" * 60}')

# %% [markdown]
# ## Análise dos resultados
#
# A consolidação, o parsing da doença prevista, as métricas, os testes
# estatísticos e as visualizações ficam em `analyze.py` — esse script só lê
# os `.json`/`.csv` gerados aqui (não roda nenhuma query) e pode ser
# reexecutado a qualquer momento para reprocessar os resultados existentes,
# inclusive depois de ajustes no parser (`parser_doencas.py`), sem custo de API.
