"""Segunda passada sobre os candidatos do filtro de regex (`filtro_triagem.py`):
um crivo por LLM, via chamada DIRETA à API (não passa pelo `graphrag query` —
sem indexação/retrieval, só uma chamada de chat curta), pra descartar falsos
positivos antes da anotação manual.

Não decide rótulo final sozinho — só filtra candidatos pra reduzir o volume
que precisa de revisão humana. A anotação final continua manual
(`src/core/ground_truth/annotate.py`).

Resumível: salva progresso a cada prontuário, pode interromper e continuar.
"""

import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from openai import OpenAI

from src.core.dataset.clinical_text import carrega_texto_por_prontuario  # noqa: E402
from src.core.paths import ANOTACAO_PROGRESSO, CSV_ORIGINAL, GROUND_TRUTH_CSV  # noqa: E402
from experiments.ground_truth_expansion.filtro_triagem import extrai_candidatos  # noqa: E402

PROGRESSO_DIR = ANOTACAO_PROGRESSO.parent

LABELS = {
    "cirrose_hepatica": "Cirrose hepática",
    "esquistossomose": "Esquistossomose",
    "hepatite_c": "Hepatite C",
    "hepatite_b": "Hepatite B",
    "esteatose_hepatica": "Esteatose hepática",
    "hipertensao_portal": "Hipertensão portal",
    "hemorragia_digestiva_alta": "Hemorragia digestiva alta",
    "pbe": "Peritonite bacteriana espontânea",
}

MODEL = "inclusionai/ling-2.6-flash"
API_BASE = "https://openrouter.ai/api/v1"
PROGRESSO_PATH = PROGRESSO_DIR / "progresso_screening_candidatos_v3.json"
MAX_WORKERS = 10  # chamadas diretas de chat, sem retrieval -- pode paralelizar

client = OpenAI(base_url=API_BASE, api_key=os.environ["GRAPHRAG_API_KEY"])


def _normalize(texto: str) -> str:
    texto = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in texto if not unicodedata.combining(c)).lower()


def monta_prompt(pront_id: str, texto: str, doencas_candidatas: set[str]) -> str:
    lista = "\n".join(f"- {LABELS[d]}" for d in sorted(doencas_candidatas))
    return (
        f"Abaixo está o texto de um prontuário eletrônico (paciente {pront_id}). "
        f"Um filtro automático encontrou menções às seguintes condições nesse texto:\n"
        f"{lista}\n\n"
        f"Para CADA uma dessas condições, avalie com base EXCLUSIVAMENTE no texto: há evidência "
        f"CLARA e INEQUÍVOCA de que o PRÓPRIO PACIENTE (não familiares) TEM essa condição "
        f"(diagnóstico explícito, exame de imagem/laboratorial confirmatório sem ressalva, ou "
        f"estadiamento formal)? Hipóteses com '?', suspeitas não confirmadas, fatores de "
        f"risco/epidemiologia sem confirmação, episódios históricos/resolvidos, ou doença "
        f"mencionada apenas como antecedente FAMILIAR (pai/mãe/irmão/avô com a doença, não o "
        f"paciente) NÃO contam como evidência clara — responda NÃO nesses casos. Para hemorragia "
        f"digestiva alta e PBE especificamente: só responda SIM se houver sangramento/peritonite "
        f"ATIVO ou muito recente documentado — episódio antigo já resolvido/tratado NÃO conta.\n\n"
        f"Responda em uma linha por condição, EXATAMENTE neste formato:\n"
        f"NOME DA CONDIÇÃO: SIM ou NÃO - justificativa em até 15 palavras\n\n"
        f"--- PRONTUÁRIO {pront_id} ---\n"
        f"{texto[:8000]}\n"
        f"--- FIM DO PRONTUÁRIO {pront_id} ---"
    )


def parse_resposta(resposta: str, doencas_candidatas: set[str]) -> dict[str, dict]:
    """Devolve {classe: {'confirmado': bool, 'justificativa': str}} por doença candidata."""
    resultado = {}
    resp_norm_linhas = resposta.split("\n")
    for classe in doencas_candidatas:
        label_norm = _normalize(LABELS[classe])
        confirmado = False
        justificativa = ""
        for linha in resp_norm_linhas:
            linha_norm = _normalize(linha)
            if label_norm in linha_norm:
                confirmado = bool(re.search(r'\bsim\b', linha_norm)) and not re.search(r'\bnao\b', linha_norm.split(':', 1)[-1][:6])
                justificativa = linha.split("-", 1)[-1].strip() if "-" in linha else linha.strip()
                break
        resultado[classe] = {"confirmado": confirmado, "justificativa": justificativa}
    return resultado


def processa_paciente(pront_id: str, texto: str, doencas_candidatas: set[str]) -> dict:
    prompt = monta_prompt(pront_id, texto, doencas_candidatas)
    inicio = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        texto_resposta = resp.choices[0].message.content or ""
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        texto_resposta = f"ERROR: {exc}"
        status = "erro"

    duracao = round(time.time() - inicio, 1)
    veredito = parse_resposta(texto_resposta, doencas_candidatas) if status == "ok" else {}

    return {
        "prontuario_id": pront_id,
        "doencas_candidatas": sorted(doencas_candidatas),
        "veredito": veredito,
        "resposta_raw": texto_resposta[:1500],
        "duracao_s": duracao,
        "status": status,
    }


def carrega_progresso() -> dict:
    if PROGRESSO_PATH.exists():
        with open(PROGRESSO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def salva_progresso(progresso: dict) -> None:
    with open(PROGRESSO_PATH, "w", encoding="utf-8") as f:
        json.dump(progresso, f, ensure_ascii=False, indent=2)


def main():
    textos = carrega_texto_por_prontuario(CSV_ORIGINAL)

    gt = pd.read_csv(GROUND_TRUTH_CSV, dtype=str).fillna("")
    gt_ids = set(gt["prontuario_id"].str.strip())

    print("Gerando candidatos via filtro de triagem (regex)...")
    candidatos = {}
    for pid, texto in textos.items():
        if pid in gt_ids:
            continue
        doencas = extrai_candidatos(texto)
        if doencas:
            candidatos[pid] = doencas
    print(f"  {len(candidatos)} candidatos encontrados.")

    progresso = carrega_progresso()
    pendentes = [pid for pid in candidatos if pid not in progresso]
    print(f"  Já processados: {len(progresso)} | Pendentes: {len(pendentes)}")

    if not pendentes:
        print("✔ Nada pendente.")
    else:
        inicio_total = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(processa_paciente, pid, textos[pid], candidatos[pid]): pid
                for pid in pendentes
            }
            concluidos = 0
            for future in as_completed(futures):
                pid = futures[future]
                resultado = future.result()
                progresso[pid] = resultado
                concluidos += 1
                if concluidos % 25 == 0:
                    salva_progresso(progresso)
                    decorrido = time.time() - inicio_total
                    taxa = concluidos / decorrido
                    restante = (len(pendentes) - concluidos) / taxa if taxa > 0 else 0
                    print(f"  {concluidos}/{len(pendentes)} | {decorrido/60:.1f}min decorridos | ~{restante/60:.1f}min restantes")

        salva_progresso(progresso)
        print(f"✔ Concluído em {(time.time()-inicio_total)/60:.1f} min")

    # Resumo final
    confirmados = {}
    for pid, r in progresso.items():
        if r["status"] != "ok":
            continue
        ok = {c for c, v in r["veredito"].items() if v["confirmado"]}
        if ok:
            confirmados[pid] = ok

    print(f"\nCandidatos originais (regex): {len(candidatos)}")
    print(f"Confirmados pelo crivo LLM  : {len(confirmados)}")

    with open(_REPO_ROOT / "progresso" / "candidatos_expansao_confirmados_v3.json", "w", encoding="utf-8") as f:
        json.dump({pid: sorted(doencas) for pid, doencas in confirmados.items()}, f, ensure_ascii=False, indent=2)
    print(f"✔ Lista final salva em progresso/candidatos_expansao_confirmados_v3.json")


if __name__ == "__main__":
    main()
