"""Carregamento do prompt de extração a partir de um arquivo `.txt`.

Antes desta refatoração, o mesmo prompt (com pequenas variações, já
divergentes entre si) existia copiado manualmente em até 6 arquivos `.py`
diferentes — mudar o prompt e saber que a mudança valia pra qualquer
execução exigia editar todos eles à mão, e nada garantia que continuassem
em sincronia (o `run.py`, por exemplo, ficou com uma versão do prompt
anterior à migração para multirrótulo, sem que ninguém percebesse).

Agora o texto do prompt mora em `prompts/query/multilabel/extraction_v3.txt`
— pra testar uma variação, basta editar esse arquivo (ou apontar
`PROMPT_PATH` para outro) e rodar de novo; nenhum código precisa mudar.
"""

from __future__ import annotations

from pathlib import Path

from .. import paths

PROMPT_PATH = paths.REPO_ROOT / "prompts" / "query" / "multilabel" / "extraction_v3.txt"

_TAMANHO_MAX_PRONTUARIO = 8000  # caracteres -- ver Seção "Prompt de extração" da metodologia


def _carrega_template(caminho: Path) -> str:
    return caminho.read_text(encoding="utf-8")


def monta_prompt_multilabel(
    pront_id: str,
    texto: str,
    doencas_validas: list[str],
    prompt_path: Path = PROMPT_PATH,
) -> str:
    """Monta o prompt de extração multirrótulo a partir do template em `.txt`.

    `doencas_validas` deve incluir a doença "Nenhuma" como último item (ver
    `parser_doencas.DOENCAS_VALIDAS`); ela é excluída da lista enumerada no
    prompt (o template já instrui o modelo a responder "Nenhuma" quando
    cabível), mas o chamador decide a lista completa de classes.
    """
    template = _carrega_template(prompt_path)
    doencas = ", ".join(doencas_validas[:-1])
    return template.format(
        pront_id=pront_id,
        doencas=doencas,
        texto=texto[:_TAMANHO_MAX_PRONTUARIO],
    )
