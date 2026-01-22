#!/bin/bash

# ============================================================================
# GraphRAG Comprehensive Clinical Evaluation Runner
# Version: 3.0 - Prompt Template System
# ============================================================================

# ===========================
# CONFIGURATION
# ===========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"  # Vai um nível acima

# Paths relativos ao projeto root
ROOT_DIR="${PROJECT_ROOT}/datatest"
CONFIG_FILE="${PROJECT_ROOT}/datatest/settings.yaml"
DATA_DIR="${PROJECT_ROOT}/datatest/output"
PATIENTS_FILE="${PROJECT_ROOT}/patients.txt"
OUTPUT_DIR="${PROJECT_ROOT}/graphrag_results"
PROMPTS_DIR="${PROJECT_ROOT}/prompts/meta-prompts"


# Subdirectories for organized output
TREATMENT_DIR="${OUTPUT_DIR}/01_treatment"
RISK_PROGNOSIS_DIR="${OUTPUT_DIR}/02_risk_prognosis"
COMPLICATIONS_DIR="${OUTPUT_DIR}/03_complications"
ETIOLOGY_DIR="${OUTPUT_DIR}/04_etiology"
THERAPIES_DIR="${OUTPUT_DIR}/05_specific_therapies"
COMPARATIVE_DIR="${OUTPUT_DIR}/06_comparative_analysis"
QUALITY_DIR="${OUTPUT_DIR}/07_quality_audit"
INTEGRATED_DIR="${OUTPUT_DIR}/08_integrated_synthesis"
COHORT_DIR="${OUTPUT_DIR}/09_cohort_analysis"

# Community levels
COMMUNITY_LEVEL_MACRO=1
COMMUNITY_LEVEL_MEDIUM=2
COMMUNITY_LEVEL_MICRO=3
COMMUNITY_LEVEL_FINE=4

# Logging
LOG_FILE="${OUTPUT_DIR}/execution_log.txt"
ERROR_LOG="${OUTPUT_DIR}/error_log.txt"

# ===========================
# SETUP
# ===========================

# Create all output directories
mkdir -p "${TREATMENT_DIR}"
mkdir -p "${RISK_PROGNOSIS_DIR}"
mkdir -p "${COMPLICATIONS_DIR}"
mkdir -p "${ETIOLOGY_DIR}"
mkdir -p "${THERAPIES_DIR}"
mkdir -p "${COMPARATIVE_DIR}"
mkdir -p "${QUALITY_DIR}"
mkdir -p "${INTEGRATED_DIR}"
mkdir -p "${COHORT_DIR}"

# Initialize logs
echo "GraphRAG Batch Execution Started: $(date)" > "${LOG_FILE}"
echo "GraphRAG Error Log: $(date)" > "${ERROR_LOG}"

# Validate dependencies
if [[ ! -f "${PATIENTS_FILE}" ]]; then
  echo "ERROR: ${PATIENTS_FILE} not found." | tee -a "${ERROR_LOG}"
  exit 1
fi

if [[ ! -d "${PROMPTS_DIR}" ]]; then
  echo "ERROR: Prompts directory ${PROMPTS_DIR} not found." | tee -a "${ERROR_LOG}"
  exit 1
fi

if ! command -v graphrag &> /dev/null; then
  echo "ERROR: graphrag command not found." | tee -a "${ERROR_LOG}"
  exit 1
fi

# ===========================
# UTILITY FUNCTIONS
# ===========================

log_info() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

log_error() {
  echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${ERROR_LOG}"
}

load_prompt() {
  local prompt_file="$1"
  local full_path="${PROMPTS_DIR}/${prompt_file}"
  
  if [[ ! -f "${full_path}" ]]; then
    log_error "Prompt file not found: ${full_path}"
    echo "ERROR: Prompt file not found"
    return 1
  fi
  
  cat "${full_path}"
}

substitute_patient_id() {
  local content="$1"
  local patient_id="$2"
  echo "${content//\{\{PATIENT_ID\}\}/${patient_id}}"
}

run_query_from_template() {
  local method="$1"
  local query_template_file="$2"
  local response_template_file="$3"
  local output_file="$4"
  local extra_params="${5:-}"
  
  # Load templates
  local query_template=$(load_prompt "${query_template_file}")
  local response_template=$(load_prompt "${response_template_file}")
  
  if [[ "${query_template}" == "ERROR:"* ]] || [[ "${response_template}" == "ERROR:"* ]]; then
    log_error "Failed to load templates for ${method} query"
    echo "QUERY FAILED - Template loading error" > "${output_file}"
    return 1
  fi
  
  # Substitute patient ID
  local query=$(substitute_patient_id "${query_template}" "${PATIENT_ID}")
  local response_type=$(substitute_patient_id "${response_template}" "${PATIENT_ID}")
  
  log_info "Running ${method} query: $(basename ${query_template_file})"
  
  if graphrag query \
    --method "${method}" \
    --query "${query}" \
    --root "${ROOT_DIR}" \
    --config "${CONFIG_FILE}" \
    --data "${DATA_DIR}" \
    --response-type "${response_type}" \
    ${extra_params} \
    > "${output_file}" 2>> "${ERROR_LOG}"; then
    log_info "✓ Successfully completed ${method} query → ${output_file}"
    return 0
  else
    log_error "✗ Failed ${method} query for patient ${PATIENT_ID}"
    echo "QUERY FAILED - See error log" > "${output_file}"
    return 1
  fi
}

# ===========================
# MAIN PROCESSING LOOP
# ===========================

PATIENT_COUNT=0
TOTAL_QUERIES_PER_PATIENT=16
SUCCESS_COUNT=0
FAILURE_COUNT=0

while IFS= read -r PATIENT_ID || [[ -n "$PATIENT_ID" ]]; do

  # Skip empty lines and comments
  if [[ -z "$PATIENT_ID" ]] || [[ "$PATIENT_ID" =~ ^#.*$ ]]; then
    continue
  fi

  PATIENT_COUNT=$((PATIENT_COUNT + 1))
  
  # Sanitize patient ID for filenames
  SAFE_ID=$(echo "$PATIENT_ID" | tr ' /' '__' | tr -cd '[:alnum:]_-')

  echo ""
  echo "========================================================================"
  echo "PATIENT ${PATIENT_COUNT}: ${PATIENT_ID}"
  echo "========================================================================"
  log_info "Processing patient ${PATIENT_COUNT}: ${PATIENT_ID}"

  # ============================================================================
  # MODULE 1: TREATMENT EVALUATION
  # ============================================================================
  
  echo ""
  echo ">>> Module 1: Treatment Evaluation"
  
  run_query_from_template "basic" \
    "01_treatment/query_basic.txt" \
    "01_treatment/response_basic.txt" \
    "${TREATMENT_DIR}/${SAFE_ID}_01_basic.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "local" \
    "01_treatment/query_local.txt" \
    "01_treatment/response_local.txt" \
    "${TREATMENT_DIR}/${SAFE_ID}_02_local.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "global" \
    "01_treatment/query_global.txt" \
    "01_treatment/response_global.txt" \
    "${TREATMENT_DIR}/${SAFE_ID}_03_global.txt" \
    "--community-level ${COMMUNITY_LEVEL_MEDIUM}" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "drift" \
    "01_treatment/query_drift.txt" \
    "01_treatment/response_drift.txt" \
    "${TREATMENT_DIR}/${SAFE_ID}_04_drift.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  # ============================================================================
  # MODULE 2: RISK STRATIFICATION & PROGNOSIS
  # ============================================================================
  
  echo ""
  echo ">>> Module 2: Risk Stratification & Prognosis"
  
  run_query_from_template "global" \
    "02_risk_prognosis/query_risk_stratification.txt" \
    "02_risk_prognosis/response_risk_stratification.txt" \
    "${RISK_PROGNOSIS_DIR}/${SAFE_ID}_01_risk_stratification.txt" \
    "--community-level ${COMMUNITY_LEVEL_MEDIUM}" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "global" \
    "02_risk_prognosis/query_outcome_prediction.txt" \
    "02_risk_prognosis/response_outcome_prediction.txt" \
    "${RISK_PROGNOSIS_DIR}/${SAFE_ID}_02_outcome_prediction.txt" \
    "--community-level ${COMMUNITY_LEVEL_MICRO} --dynamic-community-selection" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  # ============================================================================
  # MODULE 3: COMPLICATIONS ANALYSIS
  # ============================================================================
  
  echo ""
  echo ">>> Module 3: Complications Analysis"
  
  run_query_from_template "drift" \
    "03_complications/query_decompensation_warnings.txt" \
    "03_complications/response_decompensation_warnings.txt" \
    "${COMPLICATIONS_DIR}/${SAFE_ID}_01_decompensation_warnings.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "local" \
    "03_complications/query_encephalopathy.txt" \
    "03_complications/response_encephalopathy.txt" \
    "${COMPLICATIONS_DIR}/${SAFE_ID}_02_encephalopathy_analysis.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "drift" \
    "03_complications/query_ascites_management.txt" \
    "03_complications/response_ascites_management.txt" \
    "${COMPLICATIONS_DIR}/${SAFE_ID}_03_ascites_management.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  # ============================================================================
  # MODULE 4: ETIOLOGICAL & SURVEILLANCE
  # ============================================================================
  
  echo ""
  echo ">>> Module 4: Etiology & Surveillance"
  
  run_query_from_template "local" \
    "04_etiology/query_etiological_analysis.txt" \
    "04_etiology/response_etiological_analysis.txt" \
    "${ETIOLOGY_DIR}/${SAFE_ID}_01_etiological_analysis.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "drift" \
    "04_etiology/query_hcc_surveillance.txt" \
    "04_etiology/response_hcc_surveillance.txt" \
    "${ETIOLOGY_DIR}/${SAFE_ID}_02_hcc_surveillance.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  # ============================================================================
  # MODULE 5: SPECIFIC THERAPIES
  # ============================================================================
  
  echo ""
  echo ">>> Module 5: Specific Therapies"
  
  run_query_from_template "drift" \
    "05_specific_therapies/query_antiviral_response.txt" \
    "05_specific_therapies/response_antiviral_response.txt" \
    "${THERAPIES_DIR}/${SAFE_ID}_01_antiviral_response.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "local" \
    "05_specific_therapies/query_transplant_candidacy.txt" \
    "05_specific_therapies/response_transplant_candidacy.txt" \
    "${THERAPIES_DIR}/${SAFE_ID}_02_transplant_candidacy.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  # ============================================================================
  # MODULE 6: COMPARATIVE & PHENOTYPING
  # ============================================================================
  
  echo ""
  echo ">>> Module 6: Comparative Analysis"
  
  run_query_from_template "global" \
    "06_comparative/query_management_benchmarking.txt" \
    "06_comparative/response_management_benchmarking.txt" \
    "${COMPARATIVE_DIR}/${SAFE_ID}_01_management_benchmarking.txt" \
    "--community-level ${COMMUNITY_LEVEL_MEDIUM}" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "global" \
    "06_comparative/query_phenotype_classification.txt" \
    "06_comparative/response_phenotype_classification.txt" \
    "${COMPARATIVE_DIR}/${SAFE_ID}_02_phenotype_classification.txt" \
    "--community-level ${COMMUNITY_LEVEL_MICRO} --dynamic-community-selection" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  # ============================================================================
  # MODULE 7: QUALITY & COMPLETENESS AUDIT
  # ============================================================================
  
  echo ""
  echo ">>> Module 7: Quality Audit"
  
  run_query_from_template "local" \
    "07_quality/query_documentation_audit.txt" \
    "07_quality/response_documentation_audit.txt" \
    "${QUALITY_DIR}/${SAFE_ID}_01_documentation_audit.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "local" \
    "07_quality/query_inconsistencies.txt" \
    "07_quality/response_inconsistencies.txt" \
    "${QUALITY_DIR}/${SAFE_ID}_02_inconsistencies.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  # ============================================================================
  # MODULE 8: INTEGRATED SYNTHESIS
  # ============================================================================
  
  echo ""
  echo ">>> Module 8: Integrated Clinical Synthesis"
  
  run_query_from_template "local" \
    "08_integrated/query_context.txt" \
    "08_integrated/response_context.txt" \
    "${INTEGRATED_DIR}/${SAFE_ID}_synthesis_01_context.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "drift" \
    "08_integrated/query_trajectory.txt" \
    "08_integrated/response_trajectory.txt" \
    "${INTEGRATED_DIR}/${SAFE_ID}_synthesis_02_trajectory.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "global" \
    "08_integrated/query_positioning.txt" \
    "08_integrated/response_positioning.txt" \
    "${INTEGRATED_DIR}/${SAFE_ID}_synthesis_03_positioning.txt" \
    "--community-level ${COMMUNITY_LEVEL_MEDIUM}" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  run_query_from_template "global" \
    "08_integrated/query_integrated.txt" \
    "08_integrated/response_integrated.txt" \
    "${INTEGRATED_DIR}/${SAFE_ID}_synthesis_04_integrated.txt" \
    "--community-level ${COMMUNITY_LEVEL_MEDIUM} --dynamic-community-selection" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

  echo ""
  echo "✓ Completed all ${TOTAL_QUERIES_PER_PATIENT} queries for patient ${PATIENT_ID}"
  echo "------------------------------------------------------------------------"

done < "${PATIENTS_FILE}"

# ============================================================================
# COHORT-LEVEL ANALYSES
# ============================================================================

echo ""
echo "========================================================================"
echo "COHORT-LEVEL ANALYSES"
echo "========================================================================"

run_query_from_template "global" \
  "09_cohort/query_high_risk_subgroups.txt" \
  "09_cohort/response_high_risk_subgroups.txt" \
  "${COHORT_DIR}/high_risk_subgroups.txt" \
  "--community-level ${COMMUNITY_LEVEL_MEDIUM} --dynamic-community-selection" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

run_query_from_template "global" \
  "09_cohort/query_therapeutic_gaps.txt" \
  "09_cohort/response_therapeutic_gaps.txt" \
  "${COHORT_DIR}/therapeutic_gaps.txt" \
  "--community-level ${COMMUNITY_LEVEL_MACRO}" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

run_query_from_template "drift" \
  "09_cohort/query_progression_patterns.txt" \
  "09_cohort/response_progression_patterns.txt" \
  "${COHORT_DIR}/progression_patterns.txt" && ((SUCCESS_COUNT++)) || ((FAILURE_COUNT++))

# ============================================================================
# EXECUTION SUMMARY
# ============================================================================

TOTAL_QUERIES=$((PATIENT_COUNT * TOTAL_QUERIES_PER_PATIENT + 3))

echo ""
echo "========================================================================"
echo "EXECUTION SUMMARY"
echo "========================================================================"
echo "Total patients processed: ${PATIENT_COUNT}"
echo "Queries per patient: ${TOTAL_QUERIES_PER_PATIENT}"
echo "Cohort-level queries: 3"
echo "Total queries executed: ${TOTAL_QUERIES}"
echo "Successful queries: ${SUCCESS_COUNT}"
echo "Failed queries: ${FAILURE_COUNT}"
echo "Success rate: $(awk "BEGIN {printf \"%.2f\", (${SUCCESS_COUNT}/${TOTAL_QUERIES})*100}")%"
echo ""
echo "Results saved in: ${OUTPUT_DIR}"
echo "Execution log: ${LOG_FILE}"
echo "Error log: ${ERROR_LOG}"
echo "========================================================================"
echo "Batch execution completed: $(date)"
echo "========================================================================"

log_info "Batch execution completed. Success: ${SUCCESS_COUNT}, Failures: ${FAILURE_COUNT}"
