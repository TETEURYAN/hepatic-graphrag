"""
Divide o CSV único de evoluções/anamneses (uma linha por atendimento, todos
os prontuários no mesmo arquivo) em um `.txt` por prontuário, no formato que
`graphrag index` espera em `input/` (um documento por arquivo).

O dataset bruto (`data/coluna-evolucoes_anonimizadas.csv`, ~518 mil linhas)
nunca foi dividido por documento — cada prontuário tem várias linhas
(atendimentos), então indexar o CSV direto trata cada linha como
inteiramente desconectada das outras do mesmo paciente. Este script agrupa
por prontuário, concatena as evoluções em ordem cronológica e escreve um
arquivo de texto por paciente.

Uso:
    uv run python src/core/dataset/split_dataset.py \\
        --csv data/coluna-evolucoes_anonimizadas.csv \\
        --col-text descricao_anonimizada \\
        --output-dir input/texts

A saída (`input/texts/`) nunca é versionada — é derivada do dataset, que
também não é versionado (ver .gitignore).
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.paths import DATASET_ANONIMIZADO  # noqa: E402


@dataclass
class Config:
    csv_path: Path
    output_dir: Path
    col_id: str = "prontuario"
    col_text: str = "descricao_anonimizada"
    col_data: str = "data_atendimento"
    verbose: bool = False


def _slug(prontuario_id: str) -> str:
    """Nome de arquivo seguro a partir do id do prontuário."""
    s = unicodedata.normalize("NFKD", str(prontuario_id))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z0-9_-]", "_", s.strip())


def split(config: Config) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path, dtype=str).fillna("")
    df[config.col_id] = df[config.col_id].str.strip()
    df = df[df[config.col_id] != ""]

    if config.col_data in df.columns:
        df = df.sort_values([config.col_id, config.col_data])

    config.output_dir.mkdir(parents=True, exist_ok=True)

    tem_data = config.col_data in df.columns

    registros = []
    for pront_id, grupo in df.groupby(config.col_id):
        partes = []
        for _, row in grupo.iterrows():
            texto = str(row.get(config.col_text, "")).strip()
            if not texto:
                continue
            data = str(row.get(config.col_data, "")).strip() if tem_data else ""
            cabecalho = f"--- Atendimento em {data} ---\n" if data else "--- Atendimento ---\n"
            partes.append(cabecalho + texto)

        if not partes:
            continue

        nome_arquivo = f"{_slug(pront_id)}.txt"
        (config.output_dir / nome_arquivo).write_text("\n\n".join(partes), encoding="utf-8")
        registros.append({
            "prontuario_id": pront_id,
            "arquivo": nome_arquivo,
            "n_atendimentos": len(partes),
        })

        if config.verbose:
            print(f"  {pront_id} -> {nome_arquivo} ({len(partes)} atendimentos)")

    metadata = pd.DataFrame(registros)
    metadata.to_csv(config.output_dir.parent / "metadata.csv", index=False, encoding="utf-8-sig")

    print(f"✔ {len(metadata)} prontuários divididos em {config.output_dir}")
    print(f"  metadata: {config.output_dir.parent / 'metadata.csv'}")
    return metadata


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", dest="csv_path", default=DATASET_ANONIMIZADO, type=Path, help="Padrão: DATASET_ANONIMIZADO do .env")
    parser.add_argument("--output-dir", dest="output_dir", default=_REPO_ROOT / "input" / "texts", type=Path)
    parser.add_argument("--col-id", default="prontuario")
    parser.add_argument("--col-text", default="descricao_anonimizada")
    parser.add_argument("--col-data", default="data_atendimento")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    return Config(
        csv_path=args.csv_path, output_dir=args.output_dir,
        col_id=args.col_id, col_text=args.col_text, col_data=args.col_data,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    split(parse_args())
