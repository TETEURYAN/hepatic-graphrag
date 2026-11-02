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



# Marcadores de negação/ausência — se aparecerem na mesma linha do termo,
# a doença NÃO entra no conjunto (evita contar "Hepatite B (ausente)" ou
# "Esquistossomose — descartada" como predição positiva). Inclui "nr"/"nega"
# (achado em auditoria: resultado de exame "NR" = não reagente = negativo, e
# "nega" é a forma mais comum de negação nesses prontuários — v3).
_MARCADORES_NEGACAO = [
    'ausente', 'ausencia', 'descartad', 'exclu', 'negativ',
    'nao confirmad', 'nao evidenciad', 'sem evidencia', 'nao ha evidencia',
    'negado', 'nao documentad', 'nao presente', 'nega ', 'nego ', 'negou ',
    r'\bnr\b',
    # frase de negação que não batia com nenhum marcador acima (achado em
    # auditoria -- ex.: prontuário 1505692, "Não há exames ou achados que a
    # confirmem (não está documentada como presente)" virava falso positivo
    # de esquistossomose porque nenhum marcador cobria essa frase)
    'nao ha .{0,40}que .{0,15}confirm', 'nao esta documentad',
    # mais duas frases de negação achadas na mesma auditoria (prontuário
    # 1047554): "Não consta episódio ativo..." e "Não há relato de..." --
    # viravam falso positivo de hemorragia_digestiva_alta e pbe porque
    # nenhum marcador cobria essas frases tambem
    'nao consta', 'nao ha relato',
]

# "Carga viral negativa/indetectável" de HEPATITE B (PCR-HBV, HBV-DNA) é
# marcador de RESPOSTA AO TRATAMENTO num paciente já diagnosticado -- a
# terapia de hepatite B crônica (entecavir/tenofovir) é SUPRESSORA, não
# curativa: o paciente continua tendo a doença mesmo com carga viral
# indetectável. Sem essa exceção, o marcador genérico 'negativ' descartava
# linhas com diagnóstico real e evidência sólida só por citarem carga viral
# controlada como dado adicional (achado em auditoria dos erros do drift em
# S7/S8 -- ex.: prontuário 10133791, "DX HEPATITE B 2007, HBSAG reagente...
# PCR-HBV ... negativo/não detectável" virava falso negativo de hepatite_b).
#
# IMPORTANTE: isso NÃO se aplica a hepatite C -- tratamento com antivirais de
# ação direta (ex.: Viekira Pak) É curativo (RVS = cura), então "PCR-HCV
# negativo APÓS tratamento" costuma indicar doença resolvida de verdade, não
# só controlada -- testado contra caso real onde aplicar a mesma exceção pra
# HCV criava falso positivo (prontuário 8838732, ground truth sem a doença).
_RESULTADO_CONTROLADO = r'(negativ\w*|indetect\w*|nao detect\w*)'
_PADRAO_CARGA_VIRAL_CONTROLADA = re.compile(
    r'(carga viral.{0,15}hbv|pcr[\s-]?hbv|hbv[\s-]?dna|dna[\s-]?hbv)[^.;]{0,40}?'
    rf'{_RESULTADO_CONTROLADO}(\s*[/,]\s*{_RESULTADO_CONTROLADO})*'  # ex.: "não detectável/negativo"
    r'|negativac\w*\s+viral'  # ex.: "em tratamento ... com negativação viral"
)


def _mascara_carga_viral(linha_norm: str) -> str:
    """Remove menções a carga viral negativa/indetectável antes de checar
    negação -- isso é marcador de tratamento bem-sucedido num paciente já
    diagnosticado, não ausência da doença."""
    return _PADRAO_CARGA_VIRAL_CONTROLADA.sub(' ', linha_norm)


def _linha_negada(linha_norm: str) -> bool:
    linha_sem_carga_viral = _mascara_carga_viral(linha_norm)
    return any(re.search(marcador, linha_sem_carga_viral) for marcador in _MARCADORES_NEGACAO)


_RESPOSTA_ANCORADA_RE = re.compile(r'<RESPOSTA>(.*?)</RESPOSTA>', re.IGNORECASE | re.DOTALL)


def extrai_bloco_ancorado(resposta: str) -> str:
    """Se a resposta tiver o formato ancorado `<RESPOSTA>...</RESPOSTA>`
    pedido no prompt v3, devolve só o conteúdo entre as tags — descarta
    preâmbulo/meta-comentário/discussão que alguns modos (ex.: drift, que
    faz auto-crítica iterativa) tendem a vazar antes da resposta final.
    Sem tags (modelo não obedeceu o formato), devolve a resposta inteira."""
    m = _RESPOSTA_ANCORADA_RE.search(resposta)
    return m.group(1) if m else resposta


def extrai_doencas(resposta: str) -> set[str]:
    """Extrai TODAS as doenças mencionadas na resposta (versão multilabel).

    Varre linha a linha (a resposta é uma lista, um item por linha, conforme
    instruído no prompt). Devolve o conjunto de todas as classes encontradas
    em linhas que NÃO contêm marcador de negação/ausência — assim
    "Hepatite B (ausente)" não vira uma predição positiva de hepatite B.
    'nenhuma' nunca entra no conjunto — resposta sem nenhum termo conhecido
    vira conjunto vazio, com o mesmo significado.
    """
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
            padrao = r'\b' + re.escape(termo_norm) + r'\b'
            if re.search(padrao, linha_norm):
                encontradas.add(classe)
    return encontradas
