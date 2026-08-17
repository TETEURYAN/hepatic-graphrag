"""Normalização de ID de prontuário e carregamento do texto clínico completo.

O CSV bruto (`CSV_ORIGINAL`) tem dois formatos de ID pro mesmo prontuário —
puro (`"10001436"`) e prefixado (`"PACIENTE - 10001436"`) — por
inconsistência na exportação da fonte. Sem normalizar, qualquer
`groupby`/lookup por string exata trata isso como dois prontuários
diferentes, dividindo o histórico clínico real do paciente em dois grupos.

Achado (2026-08-14): 530 prontuários (16,4% do total pós-normalização) têm
essa divisão, afetando 2.644 atendimentos (41,3% do total) — incluindo os 7
prontuários que o pipeline tratava como "sem texto-fonte" (na verdade têm
texto, só que sob a variante prefixada, nunca lida).

Achado separado (mesma data): 117 atendimentos (1,8% do total, 95
prontuários) são duplicata EXATA — mesmo texto salvo duas vezes sob o mesmo
ID, provável artefato de reexportação da fonte. Sem relação com o prefixo;
`carrega_texto_por_prontuario` remove essas duplicatas antes de concatenar.
"""

import re

import pandas as pd

_PREFIXO_RE = re.compile(r'^PACIENTE\s*-\s*', re.IGNORECASE)


def normaliza_prontuario_id(pid: str) -> str:
    """Remove o prefixo "PACIENTE - " (se houver) e espaços nas pontas."""
    return _PREFIXO_RE.sub('', str(pid).strip()).strip()


def carrega_texto_por_prontuario(
    csv_path,
    col_id: str = "prontuario",
    col_texto: str = "descricao",
) -> dict[str, str]:
    """Lê o CSV bruto e devolve {prontuario_id normalizado: texto concatenado
    de TODOS os atendimentos daquele paciente, juntando as duas variantes de
    ID quando existirem}."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df[col_id] = df[col_id].apply(normaliza_prontuario_id)
    df = df.drop_duplicates(subset=[col_id, col_texto], keep="first")
    return (
        df.groupby(col_id)[col_texto]
        .apply(lambda x: "\n--- Atendimento ---\n".join(x.tolist()))
        .to_dict()
    )
