# -Prompts for Hepatology Clinical Analysis

This directory contains structured prompt templates for comprehensive hepatology patient analysis using GraphRAG.

## Structure
```
meta-prompts/
├── 01_treatment/          # Treatment evaluation (4 query types)
├── 02_risk_prognosis/     # Risk stratification & outcomes (2 query types)
├── 03_complications/      # Complications analysis (3 query types)
├── 04_etiology/           # Etiology & surveillance (2 query types)
├── 05_specific_therapies/ # Antiviral & transplant (2 query types)
├── 06_comparative/        # Benchmarking & phenotyping (2 query types)
├── 07_quality/            # Documentation audit (2 query types)
├── 08_integrated/         # Multi-method synthesis (4 query types)
└── 09_cohort/             # Population-level insights (3 query types)
```

## Template Format

Each module contains pairs of files:
- `query_*.txt` - The question/instruction sent to GraphRAG
- `response_*.txt` - The structured response format specification

## Patient ID Placeholder

All templates use `{{PATIENT_ID}}` as a placeholder that is automatically replaced during execution.

Example:
```
Query: Evaluate the treatment for patient {{PATIENT_ID}}...
```

Becomes:
```
Query: Evaluate the treatment for patient PATIENT_001...
```

## GraphRAG Methods Used

- **BASIC**: Standard RAG retrieval, baseline treatment assessment
- **LOCAL**: Context-specific analysis using graph neighborhood
- **GLOBAL**: Population-level patterns using community detection
- **DRIFT**: Temporal evolution analysis

## Customization

To customize prompts:
1. Edit the appropriate `.txt` file
2. Maintain the `{{PATIENT_ID}}` placeholder
3. Keep structured response format for consistency
4. Test with a single patient before batch execution

## Usage

These templates are automatically loaded by `run_graphrag_analysis.sh`:
```bash
./run_graphrag_analysis.sh
```

The script will:
1. Load each template
2. Substitute patient IDs from `patients.txt`
3. Execute GraphRAG queries
4. Save results to organized output directories

## Quality Assurance

Each template has been designed to:
- Minimize hallucination through explicit constraints
- Ensure clinical accuracy through structured outputs
- Enable quantitative evaluation through standardized formats
- Support inter-rater reliability assessment

## Version Control

When modifying templates:
- Document changes in git commit messages
- Test modified templates on validation set
- Update this README if structure changes
- Maintain backward compatibility when possible

---
