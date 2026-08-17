"""Filtro de triagem (regex) pra achar candidatos a expansão do ground truth.

Varre o texto bruto dos prontuários (não a resposta do GraphRAG — por isso é
separado de `parser_doencas.extrai_doencas`, que é pro outro formato) atrás
de menções às 8 doenças-alvo, com três correções sobre um regex ingênuo:

1. Ignora a linha se for o cabeçalho de seção "H.D.A:" (História da Doença
   Atual) — colide com a sigla da doença "hemorragia digestiva alta" e por
   si só gerava ~73% dos falsos positivos de HDA num teste manual.
2. Marcadores de negação mais completos que `parser_doencas` (inclui
   "nega"/"nego"/"negou", que é a forma mais comum de negação nesses
   prontuários e não estava coberta antes).
3. Marcadores de hedge (`a esclarecer`, `suspeita de`, `?`, etc.) — hipótese
   não confirmada não conta como achado positivo.
4. HDA e PBE só contam se não estiverem em contexto histórico/resolvido
   ("passado de", "história de" etc.) — mesma regra usada na anotação
   original (só marca 1 se for o motivo principal do atendimento atual).

Isto é um filtro de TRIAGEM — reduz o volume antes da checagem por LLM e da
anotação manual, não decide rótulo final sozinho.
"""

import re
import sys
from pathlib import Path

_GRID_SEARCH_DIR = Path(__file__).resolve().parents[1] / "grid_search"
if str(_GRID_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_GRID_SEARCH_DIR))

from parser_doencas import _TERMOS_ORDENADOS, _normalize  # noqa: E402

_NEGACAO = [
    'ausente', 'ausencia', 'descartad', 'exclu', 'negativ',
    'nao confirmad', 'nao evidenciad', 'sem evidencia', 'nao ha evidencia',
    'negado', 'nao documentad', 'nao presente', 'nega ', 'nego ', 'negou ',
]
_HEDGE = [
    'a esclarecer', 'a definir', 'a investigar', 'epidemiologia positiva',
    'epidemiologia para', 'suspeita de', 'possivel ', 'provavel ',
    'risco para', r'\?',
]
_HISTORICO = [
    'passado de', 'historia de', 'historico de', 'previo de', 'previa de',
    'antecedente de',
]

_SECAO_HDA_RE = re.compile(r'^#?\s*h\.?\s*d\.?\s*a\.?\s*:')


def _linha_tem_padrao(linha_norm: str, padroes: list[str]) -> bool:
    return any(re.search(p, linha_norm) for p in padroes)


def extrai_candidatos(texto: str) -> set[str]:
    """Devolve o conjunto de doenças candidatas encontradas no texto bruto."""
    if not texto:
        return set()

    encontradas = set()
    for linha in texto.split('\n'):
        linha_norm = _normalize(linha)

        if _SECAO_HDA_RE.match(linha_norm.strip()):
            continue
        if _linha_tem_padrao(linha_norm, _NEGACAO) or _linha_tem_padrao(linha_norm, _HEDGE):
            continue

        for termo, classe in _TERMOS_ORDENADOS:
            if classe == 'nenhuma':
                continue
            termo_norm = _normalize(termo)
            padrao = r'\b' + re.escape(termo_norm) + r'\b'
            if not re.search(padrao, linha_norm):
                continue
            if classe in ('hemorragia_digestiva_alta', 'pbe') and _linha_tem_padrao(linha_norm, _HISTORICO):
                continue
            encontradas.add(classe)

    return encontradas
