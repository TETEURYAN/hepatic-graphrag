#!/usr/bin/env bash
# =============================================================================
# run.sh — Wrapper para o GraphRAG Experiment Runner
#
# Edite as variáveis na seção CONFIGURAÇÃO e execute:
#   bash run.sh
#
# Requer: Python 3.10+ com as dependências do projeto instaladas (uv/pip).
# =============================================================================

set -euo pipefail

# ─── CONFIGURAÇÃO ─────────────────────────────────────────────────────────────
# Diretório raiz do índice GraphRAG SEM prompt tuning
ROOT_BASE="${ROOT_BASE:-./experimentos/local}"

# Diretório raiz do índice GraphRAG COM prompt tuning
ROOT_TUNED="${ROOT_TUNED:-./experimentos/local copy}"

# Diretório de saída dos parquets (deixe vazio para usar <root>/output)
DATA_BASE="${DATA_BASE:-}"
DATA_TUNED="${DATA_TUNED:-}"

# Métodos a testar (separados por espaço)
METHODS="${METHODS:-local global drift basic}"

# Número de sentenças para Recall@k e nDCG@k
K="${K:-5}"

# Timeout por query em segundos
TIMEOUT="${TIMEOUT:-300}"

# Onde salvar resultados e logs
RESULTS_DIR="${RESULTS_DIR:-./results}"

# Caminho do CSV de saída
CSV_OUTPUT="${CSV_OUTPUT:-${RESULTS_DIR}/experiment_results.csv}"

# Ativar modo verbose: "true" / "false"
VERBOSE="${VERBOSE:-false}"

# Ativar BERTScore (requer `pip install bert-score`): "true" / "false"
BERTSCORE="${BERTSCORE:-false}"

# ─── QUERIES ──────────────────────────────────────────────────────────────────
# Uma query por linha entre aspas.  Adicione quantas precisar.
QUERIES=(
  "Quais são as principais opções de tratamento para pacientes com cirrose?"
  "Quais são os fatores de risco e indicadores prognósticos da doença hepática?"
  "Quais complicações são mais frequentes em pacientes com hepatopatia avançada?"
)

# ─── GROUND TRUTH (opcional) ──────────────────────────────────────────────────
# Respostas de referência na mesma ordem das QUERIES.
# Comente ou deixe o array vazio se não houver.
# GROUND_TRUTH=(
#   "Resposta de referência para query 1"
#   "Resposta de referência para query 2"
#   "Resposta de referência para query 3"
# )

# =============================================================================
# NÃO EDITE ABAIXO DESTA LINHA (a menos que saiba o que está fazendo)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ─── Detectar runner (uv run ou python direto) ────────────────────────────────
if command -v uv &>/dev/null; then
  PYTHON="uv run python"
else
  PYTHON="python"
fi

echo "========================================"
echo "  GraphRAG Experiment Runner"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "  root-base  : ${ROOT_BASE}"
echo "  root-tuned : ${ROOT_TUNED}"
echo "  métodos    : ${METHODS}"
echo "  runner     : ${PYTHON}"
echo "========================================"
echo ""

# ─── Montar argumentos Python ────────────────────────────────────────────────
ARGS=(
  "${REPO_ROOT}/experiments/prompt_tuning_comparison/run.py"
  --root-base  "${ROOT_BASE}"
  --root-tuned "${ROOT_TUNED}"
  --methods    ${METHODS}
  --k          "${K}"
  --timeout    "${TIMEOUT}"
  --results-dir "${RESULTS_DIR}"
  --csv-output  "${CSV_OUTPUT}"
)

# Data dirs opcionais
[[ -n "${DATA_BASE:-}"  ]] && ARGS+=(--data-base  "${DATA_BASE}")
[[ -n "${DATA_TUNED:-}" ]] && ARGS+=(--data-tuned "${DATA_TUNED}")

# Queries
ARGS+=(--queries)
for q in "${QUERIES[@]}"; do
  ARGS+=("${q}")
done

# Ground truth opcional
if [[ -n "${GROUND_TRUTH+x}" ]] && [[ ${#GROUND_TRUTH[@]} -gt 0 ]]; then
  ARGS+=(--ground-truth)
  for gt in "${GROUND_TRUTH[@]}"; do
    ARGS+=("${gt}")
  done
fi

# Flags booleanas
[[ "${VERBOSE}"   == "true" ]] && ARGS+=(--verbose)
[[ "${BERTSCORE}" == "true" ]] && ARGS+=(--bertscore)

# ─── Criar diretório de resultados ───────────────────────────────────────────
mkdir -p "${RESULTS_DIR}"

# ─── Executar ────────────────────────────────────────────────────────────────
${PYTHON} "${ARGS[@]}"

echo ""
echo "Concluído. Resultados em: ${RESULTS_DIR}"
