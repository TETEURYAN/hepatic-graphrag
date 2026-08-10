"""Parser de doença hepática a partir da resposta em texto livre do GraphRAG.

Usado tanto pela execução do grid (`experimentos_graphrag.py`) quanto pela
análise offline (`analise_resultados_grid.py`), para garantir que a mesma
lógica de extração seja aplicada em ambos os lugares.
"""

import re
import unicodedata

DOENCAS_VALIDAS = [
    'Cirrose hepática',
    'Esquistossomose',
    'Hepatite B',
    'Hepatite C',
    'Esteatose hepática',
    'Hipertensão portal',
    'Hemorragia digestiva alta',
    'Peritonite bacteriana espontânea',
    'Nenhuma',
]

# Termo (como pode aparecer na resposta) -> classe do ground truth.
# As chaves não precisam se preocupar com maiúsculas/acentos: `_normalize`
# trata isso antes da comparação.
MAPA_DOENCAS = {
    'cirrose': 'cirrose_hepatica',
    'cirrose hepatica': 'cirrose_hepatica',
    'cirrose hepatica descompensada': 'cirrose_hepatica',
    'cirrose hepatica alcoolica': 'cirrose_hepatica',
    'cirrose hepatica viral c': 'cirrose_hepatica',
    'cirrose micronodular': 'cirrose_hepatica',
    'cirrose descompensada': 'cirrose_hepatica',
    'cirrose hepatica cronica': 'cirrose_hepatica',

    'hipertensao portal': 'hipertensao_portal',
    'hipertensao porta': 'hipertensao_portal',
    'sindrome de hipertensao portal': 'hipertensao_portal',
    'sindrome da hipertensao portal': 'hipertensao_portal',
    'hipertensao portal nao cirrotica': 'hipertensao_portal',
    'ehe/hp': 'hipertensao_portal',

    'esquistossomose': 'esquistossomose',
    'equistossomose': 'esquistossomose',
    'esquistossomose hepatoesplenica': 'esquistossomose',
    'esquistossomose hepatoesplenica (ehe)': 'esquistossomose',
    'esquistossomose mansoni': 'esquistossomose',
    'esquistossomose mansonica': 'esquistossomose',
    'ehe': 'esquistossomose',

    'hepatite c': 'hepatite_c',
    'hepatite c (hcv)': 'hepatite_c',
    'hepatite c cronica': 'hepatite_c',
    'hepatite cronica viral c': 'hepatite_c',

    'hepatite b': 'hepatite_b',
    'hepatite b (hbv)': 'hepatite_b',
    'hepatite b cronica': 'hepatite_b',
    'hepatite b inativa': 'hepatite_b',
    'hepatite b resolvida': 'hepatite_b',
    'hepatite cronica b': 'hepatite_b',
    'hepatite cronica viral b': 'hepatite_b',
    'virus b (hepatite b)': 'hepatite_b',

    'esteatose': 'esteatose_hepatica',
    'esteatose hepatica': 'esteatose_hepatica',
    'esteatose hepatica difusa': 'esteatose_hepatica',
    'dhgna': 'esteatose_hepatica',
    'nafld': 'esteatose_hepatica',
    'masld': 'esteatose_hepatica',
    'mash': 'esteatose_hepatica',
    'nash': 'esteatose_hepatica',
    'doenca hepatica gordurosa nao alcoolica': 'esteatose_hepatica',
    'doenca hepatica gordurosa nao alcoolica (dhgna)': 'esteatose_hepatica',
    'doenca hepatica esteatotica metabolica': 'esteatose_hepatica',
    'doenca hepatica esteatotica metabolica (mash)': 'esteatose_hepatica',
    'esteato-hepatite nao alcoolica': 'esteatose_hepatica',
    'esteato-hepatite nao alcoolica (nash)': 'esteatose_hepatica',
    'esteatohepatite': 'esteatose_hepatica',
    'esteatohepatite nao alcoolica': 'esteatose_hepatica',

    'hda': 'hemorragia_digestiva_alta',
    'hemorragia digestiva alta': 'hemorragia_digestiva_alta',
    'hemorragia digestiva alta varicosa': 'hemorragia_digestiva_alta',
    'hemorragia varicosa': 'hemorragia_digestiva_alta',

    'pbe': 'pbe',
    'peritonite bacteriana espontanea': 'pbe',
    'peritonite bacteriana espontanea (pbe)': 'pbe',

    'nenhuma': 'nenhuma',
}

# Ordem de checagem: termos mais longos primeiro, para que "esteatose hepatica"
# seja preferido a "esteatose" quando ambos aparecem na mesma posição.
_TERMOS_ORDENADOS = sorted(MAPA_DOENCAS.items(), key=lambda kv: -len(kv[0]))


def _normalize(texto: str) -> str:
    """minúsculas + remove acentos, para comparação robusta a maiúsculas/acentuação."""
    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    return texto.lower()


def extrai_doenca(resposta: str) -> str:
    """Extrai a doença da resposta do GraphRAG.

    Estratégia: normaliza a resposta (minúsculas, sem acento) e procura, com
    limites de palavra, todos os termos conhecidos. Entre as ocorrências,
    escolhe a que aparece mais cedo no texto (a resposta é uma frase curta que
    começa com o diagnóstico, conforme instruído no prompt); em caso de
    empate de posição, prefere o termo mais específico (mais longo).
    """
    if not resposta:
        return 'ERRO'

    resp_norm = _normalize(resposta)

    melhor = None  # (posicao, -tamanho_termo, classe)
    for termo, classe in _TERMOS_ORDENADOS:
        termo_norm = _normalize(termo)
        padrao = r'\b' + re.escape(termo_norm) + r'\b'
        m = re.search(padrao, resp_norm)
        if m is None:
            continue
        candidato = (m.start(), -len(termo_norm), classe)
        if melhor is None or candidato < melhor:
            melhor = candidato

    if melhor is None:
        return 'nao_identificado'
    return melhor[2]
