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

# %% [markdown]
# ## Célula 1 — Configuração

# %% [markdown]
# 

# %%
import subprocess
import json
import time
import os
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from IPython.display import display, clear_output
import ipywidgets as widgets

from parser_doencas import extrai_doenca, DOENCAS_VALIDAS

# ── AJUSTE ESTES CAMINHOS ────────────────────────────────────────────────────
GRAPHRAG_ROOT = r'/mnt/c/Users/Matheus/Documents/trabalho de conclusão de curso/anamnese/experimentos/local copy'
GRAPHRAG_DATA = r'/mnt/c/Users/Matheus/Documents/trabalho de conclusão de curso/anamnese/experimentos/local copy/output'
GROUND_TRUTH  = './progresso/ground_truth_anotado.csv'
CSV_ORIGINAL  = './experimentos/local/input/data-hepato-sem-anonimizacao.csv'
COL_ID        = 'prontuario'
COL_TEXTO     = 'descricao'

# Pasta onde os resultados serão salvos
OUTPUT_DIR    = './resultados_grid'
PROGRESSO_DIR = './progresso_grid'

# Timeout por query (segundos) — DRIFT pode precisar de mais
TIMEOUT_BASIC_LOCAL = 10000
TIMEOUT_GLOBAL      = 10000
TIMEOUT_DRIFT       = 9999

# Pausa entre queries (segundos) — evita rate limiting
PAUSA_ENTRE_QUERIES = 6
# ────────────────────────────────────────────────────────────────────────────

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PROGRESSO_DIR).mkdir(parents=True, exist_ok=True)

# Grid de experimentos
SISTEMAS = [
    {'id': 'S1', 'method': 'basic',  'community_level': 1},
    {'id': 'S2', 'method': 'basic',  'community_level': 2},
    {'id': 'S3', 'method': 'local',  'community_level': 1},
    {'id': 'S4', 'method': 'local',  'community_level': 2},
    {'id': 'S5', 'method': 'global', 'community_level': 1},
    {'id': 'S6', 'method': 'global', 'community_level': 2},
    {'id': 'S7', 'method': 'drift',  'community_level': 1},
    {'id': 'S8', 'method': 'drift',  'community_level': 2},
]

print('✔ Configuração carregada.')
print(f'  Output dir    : {OUTPUT_DIR}')
print(f'  Progresso dir : {PROGRESSO_DIR}')
print(f'  Sistemas      : {len(SISTEMAS)}')

# %% [markdown]
# ## Célula 2 — Carrega dados

# %%
# Ground truth
gt = pd.read_csv(GROUND_TRUTH, dtype=str).fillna('')
gt['prontuario_id'] = gt['prontuario_id'].str.strip()
lista_prontuarios = gt['prontuario_id'].tolist()

# Textos originais
df_orig = pd.read_csv(CSV_ORIGINAL, dtype=str).fillna('')
df_orig[COL_ID] = df_orig[COL_ID].str.strip()

# Agrega textos por prontuário (pode ter múltiplas consultas)
texto_por_id = (
    df_orig.groupby(COL_ID)[COL_TEXTO]
    .apply(lambda x: '\n--- Atendimento ---\n'.join(x.tolist()))
    .to_dict()
)

print(f'Ground truth   : {len(gt)} prontuários')
print(f'Textos carregados : {len(texto_por_id)}')

# Verifica prontuários sem texto
sem_texto = [p for p in lista_prontuarios if p not in texto_por_id]
if sem_texto:
    print(f'⚠  {len(sem_texto)} prontuários sem texto: {sem_texto[:5]}')
else:
    print('✔ Todos os prontuários têm texto disponível.')

# %% [markdown]
# ## Célula 3 — Funções principais

# %%
def monta_prompt(pront_id: str, texto: str) -> str:
    """Monta o prompt clínico para classificação."""
    doencas = ', '.join(DOENCAS_VALIDAS[:-1]) + ' ou Nenhuma'
    return (
        f'Abaixo estão as evoluções/anamneses de um único paciente, '
        f'extraídas do prontuário eletrônico (nº {pront_id}). '
        f'Com base EXCLUSIVAMENTE nesse conteúdo (usando seu conhecimento médico geral apenas para interpretá-lo), '
        f'qual é a doença hepática principal desse paciente? '
        f'Escolha apenas uma entre: {doencas}. '
        f'Se nenhuma dessas condições estiver claramente documentada, responda \'Nenhuma\'. '
        f'Responda em uma frase curta, citando apenas o nome da doença.\n'
        f'--- PRONTUÁRIO {pront_id} ---\n'
        f'{texto[:8000]}\n'   # limita para não estourar contexto
        f'--- FIM DO PRONTUÁRIO {pront_id} ---'
    )


def get_timeout(method: str) -> int:
    if method == 'drift':
        return TIMEOUT_DRIFT
    elif method == 'global':
        return TIMEOUT_GLOBAL
    return TIMEOUT_BASIC_LOCAL


def roda_query(pront_id: str, texto: str, method: str, community_level: int) -> dict:
    """Executa uma query no GraphRAG e retorna o resultado."""
    prompt = monta_prompt(pront_id, texto)
    cmd = [
        'uv', 'run', '--no-sync', 'graphrag', 'query',
        '--root', GRAPHRAG_ROOT,
        '--data', GRAPHRAG_DATA,
        '--method', method,
        '--community-level', str(community_level),
        '--response-type', 'Single sentence',
        '--query', prompt,
    ]

    inicio = time.time()
    try:
        resultado = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=get_timeout(method),
            encoding='utf-8',
            errors='replace',
        )
        duracao = round(time.time() - inicio, 1)

        # Extrai a resposta do stdout
        stdout = resultado.stdout or ''
        # GraphRAG imprime a resposta após "SUCCESS:"
        match = re.search(r'SUCCESS:\s*(.+)', stdout, re.IGNORECASE | re.DOTALL)
        resposta_raw = match.group(1).strip() if match else stdout.strip()[-500:]

        doenca = extrai_doenca(resposta_raw)
        status = 'ok' if resultado.returncode == 0 else 'erro_returncode'

        return {
            'prontuario_id': pront_id,
            'doenca_prevista': doenca,
            'resposta_raw': resposta_raw[:500],
            'duracao_s': duracao,
            'status': status,
            'returncode': resultado.returncode,
            'timestamp': datetime.now().isoformat(),
        }

    except subprocess.TimeoutExpired:
        return {
            'prontuario_id': pront_id,
            'doenca_prevista': 'TIMEOUT',
            'resposta_raw': '',
            'duracao_s': get_timeout(method),
            'status': 'timeout',
            'returncode': -1,
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            'prontuario_id': pront_id,
            'doenca_prevista': 'ERRO',
            'resposta_raw': str(e)[:200],
            'duracao_s': round(time.time() - inicio, 1),
            'status': f'excecao: {type(e).__name__}',
            'returncode': -1,
            'timestamp': datetime.now().isoformat(),
        }


def arquivo_progresso(sistema_id: str) -> str:
    return os.path.join(PROGRESSO_DIR, f'progresso_{sistema_id}.json')


def carrega_progresso(sistema_id: str) -> dict:
    path = arquivo_progresso(sistema_id)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def salva_progresso(sistema_id: str, progresso: dict):
    with open(arquivo_progresso(sistema_id), 'w', encoding='utf-8') as f:
        json.dump(progresso, f, ensure_ascii=False, indent=2)


def salva_csv_sistema(sistema_id: str, progresso: dict, sistema_cfg: dict):
    """Salva CSV parcial/final de um sistema."""
    linhas = list(progresso.values())
    if not linhas:
        return
    df = pd.DataFrame(linhas)
    df['method']          = sistema_cfg['method']
    df['community_level'] = sistema_cfg['community_level']
    df['sistema_id']      = sistema_id
    path = os.path.join(OUTPUT_DIR, f'resultado_{sistema_id}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')


print('✔ Funções carregadas.')

# %% [markdown]
# ## Célula 4 — Status atual do progresso

# %%
print('Status atual dos experimentos:\n')
print(f'{"Sistema":<8} {"Modo":<10} {"Level":<8} {"Concluídos":<12} {"Pendentes":<10} {"Timeouts":<10} {"Erros"}')
print('-' * 65)

for s in SISTEMAS:
    prog = carrega_progresso(s['id'])
    concluidos = len(prog)
    pendentes  = len(lista_prontuarios) - concluidos
    timeouts   = sum(1 for v in prog.values() if v.get('status') == 'timeout')
    erros      = sum(1 for v in prog.values() if 'erro' in v.get('status', '') or 'excecao' in v.get('status', ''))
    print(f"{s['id']:<8} {s['method']:<10} {s['community_level']:<8} {concluidos:<12} {pendentes:<10} {timeouts:<10} {erros}")

print(f'\nTotal de prontuários por sistema: {len(lista_prontuarios)}')
print(f'Total de execuções no grid      : {len(lista_prontuarios) * len(SISTEMAS)}')

# %% [markdown]
# ## Célula 5 — Execução do grid
# 
# **Configure `SISTEMAS_RODAR` para escolher quais sistemas executar.**
# Use `'todos'` para rodar o grid completo, ou liste os IDs: `['S1', 'S2']`

# %%
# ── CONFIGURE AQUI ───────────────────────────────────────────────
SISTEMAS_RODAR = 'todos'   # 'todos' ou lista: ['S1', 'S3', 'S7']
MODO_TESTE     = False     # True = roda só 3 prontuários por sistema (piloto)
# ────────────────────────────────────────────────────────────────

sistemas_selecionados = SISTEMAS if SISTEMAS_RODAR == 'todos' else [
    s for s in SISTEMAS if s['id'] in SISTEMAS_RODAR
]

prontuarios_rodar = lista_prontuarios[:3] if MODO_TESTE else lista_prontuarios

print(f'Sistemas selecionados : {[s["id"] for s in sistemas_selecionados]}')
print(f'Prontuários por sistema: {len(prontuarios_rodar)}')
print(f'Total de execuções    : {len(sistemas_selecionados) * len(prontuarios_rodar)}')

# Estimativa de tempo
tempos_estimados = {'basic': 200, 'local': 300, 'global': 300, 'drift': 2400}
tempo_total = sum(
    tempos_estimados[s['method']] * len(prontuarios_rodar)
    for s in sistemas_selecionados
)
print(f'Tempo estimado        : {tempo_total//3600}h {(tempo_total%3600)//60}min')

confirma = input('\nConfirmar execução? [s/N]: ').strip().lower()
if confirma != 's':
    print('Execução cancelada.')
else:
    inicio_total = time.time()

    for idx_s, sistema in enumerate(sistemas_selecionados):
        sid     = sistema['id']
        method  = sistema['method']
        level   = sistema['community_level']

        print(f'\n{"="*60}')
        print(f'Sistema {sid} — {method.upper()} / community-level {level}')
        print(f'{"="*60}')

        progresso = carrega_progresso(sid)
        pendentes = [p for p in prontuarios_rodar if p not in progresso]

        if not pendentes:
            print(f'  ✔ Já concluído ({len(progresso)} prontuários). Pulando.')
            continue

        print(f'  Já concluídos : {len(progresso)}')
        print(f'  Pendentes     : {len(pendentes)}')

        for idx_p, pront_id in enumerate(pendentes):
            texto = texto_por_id.get(pront_id, '')
            if not texto:
                progresso[pront_id] = {
                    'prontuario_id': pront_id,
                    'doenca_prevista': 'SEM_TEXTO',
                    'resposta_raw': '',
                    'duracao_s': 0,
                    'status': 'sem_texto',
                    'returncode': -1,
                    'timestamp': datetime.now().isoformat(),
                }
                salva_progresso(sid, progresso)
                continue

            # Log de progresso
            pct = (idx_p + 1) / len(pendentes) * 100
            elapsed = time.time() - inicio_total
            print(f'  [{idx_p+1:>3}/{len(pendentes)}] {pct:>5.1f}% | '
                  f'ID: {pront_id} | '
                  f'Elapsed: {int(elapsed//60)}m{int(elapsed%60):02d}s',
                  end=' ... ', flush=True)

            # Executa a query
            resultado = roda_query(pront_id, texto, method, level)

            # Log do resultado
            status_icon = '✔' if resultado['status'] == 'ok' else '⚠' if resultado['status'] == 'timeout' else '✗'
            print(f"{status_icon} {resultado['doenca_prevista']:<30} [{resultado['duracao_s']}s]")

            # Salva progresso
            progresso[pront_id] = resultado
            salva_progresso(sid, progresso)
            salva_csv_sistema(sid, progresso, sistema)

            # Pausa entre queries
            if idx_p < len(pendentes) - 1:
                time.sleep(PAUSA_ENTRE_QUERIES)

        # Resumo do sistema
        total    = len(progresso)
        ok       = sum(1 for v in progresso.values() if v['status'] == 'ok')
        timeouts = sum(1 for v in progresso.values() if v['status'] == 'timeout')
        erros    = total - ok - timeouts
        print(f'\n  Sistema {sid} concluído: {ok} OK | {timeouts} timeouts | {erros} erros')
        salva_csv_sistema(sid, progresso, sistema)

    tempo_total_real = time.time() - inicio_total
    print(f'\n{"="*60}')
    print(f'Grid concluído em {int(tempo_total_real//3600)}h {int((tempo_total_real%3600)//60)}min')
    print(f'{"="*60}')


# %% [markdown]
# ## Análise dos resultados
#
# A consolidação, o parsing da doença prevista, as métricas, os testes
# estatísticos e as visualizações foram movidos para
# `analise_resultados_grid.py` — esse script só lê os `.json`/`.csv` gerados
# aqui (não roda nenhuma query) e pode ser reexecutado a qualquer momento
# para reprocessar os resultados existentes, inclusive depois de ajustes no
# parser (`parser_doencas.py`), sem custo de API.
