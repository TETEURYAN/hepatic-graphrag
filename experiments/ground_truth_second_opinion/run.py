"""Segunda opinião médica via LLM (chamada DIRETA à API OpenRouter) sobre os
prontuários do ground truth — NÃO é o ground truth em si, é um segundo
anotador independente e cego (não vê nenhuma anotação prévia), pra depois
comparar manualmente com a sua própria anotação. Ver discussão metodológica:
usar um LLM pra *gerar* o ground truth seria circular (mesma família de
tecnologia sendo avaliada); usar como segundo anotador pra triangulação é
válido.

Duas fontes de prontuário, unidas num único progresso resumível:
1. Os 141 originais (reanotados do zero) — texto já unificado/deduplicado em
   `progresso/prontuarios_txt_ground_truth/<id>.txt`.
2. Os 396 candidatos de expansão estratificada
   (`progresso/candidatos_expansao_amostra_estratificada.json`) — texto
   carregado sob demanda do CSV bruto via `carrega_texto_por_prontuario`
   (mesma normalização/deduplicação usada em `screen_candidatos.py`).

Total: 537 prontuários. Uma chamada de chat por prontuário (sem
retrieval/GraphRAG — direto ao modelo). Resumível: salva progresso a cada
prontuário, pode interromper e continuar.

Modelo gratuito (`dots-studio/dots-3-note-preview:free`) pode ter limite de
requisição — toda vez que a resposta indicar indisponibilidade temporária
(erro 429, "rate limit", modelo sobrecarregado, resposta vazia etc.), o
prontuário fica com status != "ok" e uma flag de horário é registrada em
`flags_limite_taxa`; ele é automaticamente reprocessado na próxima execução
do script (o progresso anterior não é sobrescrito, só a lista de flags
cresce a cada nova tentativa que falhar).

Uso:
    uv run python3 experiments/ground_truth_second_opinion/run.py
"""

import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI  # noqa: E402

from src.core.dataset.clinical_text import (  # noqa: E402
    carrega_texto_por_prontuario,
    normaliza_prontuario_id,
)
from src.core.paths import CANDIDATOS_EXPANSAO_GT, CSV_ORIGINAL, GROUND_TRUTH_CSV, bootstrap_sys_path  # noqa: E402

bootstrap_sys_path(_REPO_ROOT / "experiments" / "grid_search")
from parser_doencas import extrai_doencas  # noqa: E402

TXT_DIR = _REPO_ROOT / "progresso" / "prontuarios_txt_ground_truth"
PROGRESSO_PATH = GROUND_TRUTH_CSV.parent / "segunda_opiniao_llm.json"
RESUMO_CSV_PATH = GROUND_TRUTH_CSV.parent / "segunda_opiniao_llm_resumo.csv"

MODEL = "dots-studio/dots-3-note-preview:free"
API_BASE = "https://openrouter.ai/api/v1"
PAUSA_ENTRE_QUERIES = 5  # s — modelo gratuito, evita estourar limite de requisição

client = OpenAI(base_url=API_BASE, api_key=os.environ["GRAPHRAG_API_KEY"])

DOENCAS_TEXTO = (
    "Cirrose hepática, Esquistossomose, Hepatite B, Hepatite C, Esteatose hepática, "
    "Hipertensão portal, Hemorragia digestiva alta, Peritonite bacteriana espontânea"
)

# Frases que indicam indisponibilidade temporária do modelo gratuito — tanto
# em exceções da chamada (429 etc.) quanto no próprio texto de resposta
# (alguns modelos "free" devolvem HTTP 200 com uma mensagem de recusa em vez
# de erro real).
_MARCADORES_LIMITE_TAXA = [
    "rate limit", "rate-limit", "too many requests", "429",
    "no instances available", "no endpoints found", "temporarily unavailable",
    "try again later", "quota exceeded", "model is overloaded", "overloaded",
    "please try again", "service unavailable", "nao podemos responder",
    "nao posso responder", "nao consigo responder", "no momento nao",
    "servico indisponivel", "tente novamente",
]


def _normalize(texto: str) -> str:
    texto = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in texto if not unicodedata.combining(c)).lower()


def _parece_limite_taxa(texto: str) -> bool:
    texto_norm = _normalize(texto)
    return any(marcador in texto_norm for marcador in _MARCADORES_LIMITE_TAXA)


def monta_prompt(pront_id: str, texto: str) -> str:
    return (
        f"Você é um médico revisor fazendo uma segunda avaliação independente de um "
        f"prontuário eletrônico, para fins de pesquisa (validação de um sistema de extração "
        f"automática de diagnósticos). Você NÃO viu nenhuma anotação prévia deste caso.\n\n"
        f"Abaixo estão as evoluções/anamneses de um único paciente (prontuário nº {pront_id}). "
        f"Com base EXCLUSIVAMENTE nesse conteúdo (use seu conhecimento médico geral apenas "
        f"para interpretá-lo, não para presumir informação que não está no texto), quais das "
        f"doenças hepáticas abaixo estão CLARAMENTE documentadas como PRESENTES NO PRÓPRIO "
        f"PACIENTE (diagnóstico explícito, exame de imagem/laboratorial confirmatório sem "
        f"ressalva, ou estadiamento formal)? Considere tanto a doença de base quanto suas "
        f"complicações, se ambas estiverem documentadas (ex.: esquistossomose E a hipertensão "
        f"portal decorrente dela, se as duas estiverem descritas).\n\n"
        f"Regras importantes:\n"
        f"- Resultado de exame como 'NR', 'não reagente', 'negativo', 'NDN' ou 'normal' indica "
        f"AUSÊNCIA da condição testada, não presença (ex.: 'Anti-HCV: NR' significa que o "
        f"paciente NÃO tem hepatite C).\n"
        f"- NÃO conte doenças mencionadas apenas como antecedente FAMILIAR (pai/mãe/irmão/avô "
        f"com a doença) — só conta se for do próprio paciente.\n"
        f"- NÃO conte doenças negadas pelo paciente, mencionadas apenas em hipótese/diferencial "
        f"sem confirmação, investigadas com resultado negativo, ou descartadas/excluídas no "
        f"texto.\n"
        f"- Para Hemorragia digestiva alta e PBE especificamente: só conta se for "
        f"sangramento/peritonite ATIVO ou muito recente — episódio antigo já resolvido não "
        f"conta.\n\n"
        f"Lista de doenças possíveis: {DOENCAS_TEXTO}.\n\n"
        f"Responda EXATAMENTE neste formato, sem nenhum texto antes ou depois das tags (sem "
        f"preâmbulo, sem análise, sem conclusão):\n"
        f"<RESPOSTA>\n"
        f"[uma doença por linha, no formato 'NOME DA DOENÇA: trecho curto do texto que confirma "
        f"isso' — ou apenas 'Nenhuma' se nenhuma doença da lista estiver claramente documentada "
        f"como presente]\n"
        f"</RESPOSTA>\n\n"
        f"--- PRONTUÁRIO {pront_id} ---\n"
        f"{texto}\n"
        f"--- FIM DO PRONTUÁRIO {pront_id} ---"
    )


# Tamanhos de texto tentados em ordem — o provedor gratuito único desse
# modelo (AtlasCloud, via OpenRouter) rejeita ("400 bad request", sem
# detalhe) uma fração de requisições grandes de forma inconsistente (mesmo
# tamanho de prompt funciona pra um paciente e falha pra outro — não é um
# limite fixo de caracteres, parece capacidade/carga do provedor no
# momento). Em vez de desistir, encolhe o texto e tenta de novo antes de
# marcar como erro de verdade.
_TAMANHOS_TEXTO = [8000, 5000, 3000, 1500, 600]


def _e_erro_provedor(msg: str) -> bool:
    msg_norm = _normalize(msg)
    return "bad request" in msg_norm or "provider returned error" in msg_norm


def processa_paciente(pront_id: str, texto: str) -> dict:
    inicio = time.time()
    conteudo = ""
    status = "erro"
    tamanho_usado = _TAMANHOS_TEXTO[0]

    for tamanho in _TAMANHOS_TEXTO:
        tamanho_usado = tamanho
        prompt = monta_prompt(pront_id, texto[:tamanho])
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=6000,
                extra_body={"reasoning": {"effort": "low"}},
            )
            conteudo = resp.choices[0].message.content or ""
            if not conteudo.strip():
                status = "vazio"
            elif _parece_limite_taxa(conteudo):
                status = "limite_taxa"
            else:
                status = "ok"
            break
        except Exception as exc:  # noqa: BLE001
            conteudo = f"ERROR: {exc}"
            if _parece_limite_taxa(str(exc)):
                status = "limite_taxa"
                break
            if _e_erro_provedor(str(exc)) and tamanho != _TAMANHOS_TEXTO[-1]:
                status = "erro"
                continue  # tenta de novo com texto mais curto
            status = "erro"
            break

    duracao = round(time.time() - inicio, 1)
    doencas = sorted(extrai_doencas(conteudo)) if status == "ok" else []

    return {
        "prontuario_id": pront_id,
        "status": status,
        "doencas_previstas": doencas,
        "resposta_raw": conteudo[:2000],
        "duracao_s": duracao,
        "timestamp": datetime.now().isoformat(),
        "tamanho_texto_usado": tamanho_usado if status == "ok" else None,
    }


def carrega_progresso() -> dict:
    if PROGRESSO_PATH.exists():
        with open(PROGRESSO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def salva_progresso(progresso: dict) -> None:
    with open(PROGRESSO_PATH, "w", encoding="utf-8") as f:
        json.dump(progresso, f, ensure_ascii=False, indent=2)


def salva_resumo_csv(progresso: dict) -> None:
    import csv

    with open(RESUMO_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prontuario_id", "status", "doencas_previstas", "duracao_s", "resposta_raw"])
        for pid in sorted(progresso, key=lambda x: (len(x), x)):
            r = progresso[pid]
            writer.writerow([
                pid, r["status"], "; ".join(r["doencas_previstas"]),
                r["duracao_s"], r["resposta_raw"].replace("\n", " ⏎ "),
            ])


def carrega_textos_todos_prontuarios() -> dict[str, str]:
    """{prontuario_id: texto} pra os 141 originais (via .txt já exportado) +
    os 396 candidatos de expansão (via CSV bruto, carregado sob demanda)."""
    textos = {f.stem: f.read_text(encoding="utf-8") for f in TXT_DIR.glob("*.txt")}

    with open(CANDIDATOS_EXPANSAO_GT, "r", encoding="utf-8") as f:
        expansao = json.load(f)
    ids_expansao = {normaliza_prontuario_id(pid) for pid in expansao}
    faltando = ids_expansao - set(textos)
    if faltando:
        textos_csv = carrega_texto_por_prontuario(CSV_ORIGINAL)
        for pid in faltando:
            if pid in textos_csv:
                textos[pid] = textos_csv[pid]
    return textos


def main():
    textos = carrega_textos_todos_prontuarios()
    print(f"Prontuários encontrados (141 originais + expansão estratificada): {len(textos)}")

    progresso = carrega_progresso()
    pendentes = [
        pid for pid in textos
        if pid not in progresso or progresso[pid]["status"] != "ok"
    ]
    pendentes.sort(key=lambda x: (len(x), x))
    print(f"Já com resposta ok: {len(textos) - len(pendentes)} | Pendentes: {len(pendentes)}")
    print(f"Modelo: {MODEL}")
    print(f"Progresso salvo em: {PROGRESSO_PATH}\n")

    inicio_total = time.time()
    for i, pront_id in enumerate(pendentes, 1):
        texto = textos[pront_id]
        resultado = processa_paciente(pront_id, texto)

        flags_antigas = progresso.get(pront_id, {}).get("flags_limite_taxa", [])
        if resultado["status"] in ("limite_taxa", "vazio", "erro"):
            resultado["flags_limite_taxa"] = flags_antigas + [resultado["timestamp"]]
        else:
            resultado["flags_limite_taxa"] = flags_antigas

        progresso[pront_id] = resultado
        salva_progresso(progresso)

        marca = "" if resultado["status"] == "ok" else f"  <<< {resultado['status'].upper()}"
        print(f"  [{i}/{len(pendentes)}] {pront_id}: {resultado['doencas_previstas']} "
              f"({resultado['duracao_s']}s, {resultado['status']}){marca}")

        time.sleep(PAUSA_ENTRE_QUERIES)

    salva_resumo_csv(progresso)

    tempo_total = time.time() - inicio_total
    contagem = {}
    for r in progresso.values():
        contagem[r["status"]] = contagem.get(r["status"], 0) + 1
    print(f"\n✔ Concluído em {tempo_total/60:.1f} min")
    print(f"  Status: {contagem}")
    print(f"  Resumo legível em: {RESUMO_CSV_PATH}")
    print(f"  Progresso completo (com resposta_raw) em: {PROGRESSO_PATH}")


if __name__ == "__main__":
    main()
