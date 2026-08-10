"""
GraphRAG — Dataset Prontuário/Documento x Doença (CLI)

Unifica três scripts que existiam quase duplicados (create_dataset_doc_sick.py,
create_dataset_record_sick.py, create_dataset_record_sick_nopt.py): mesma
estrutura (Config/Logger/Utils/DatasetBuilder/DataExporter/CLI), variando só
a granularidade da saída e a estratégia de casar um chunk do GraphRAG com o
prontuário/documento de origem.

Granularidades:
  --granularity document  Agrupa por `document_id` (metadado nativo do
                           GraphRAG em text_units) — não precisa casar com o
                           CSV original.
  --granularity record    Agrupa por prontuário, casando cada chunk com uma
                           linha do CSV original via --matcher:
                             fingerprint  Índice de prontuário por não usar
                                          transformação nenhuma no texto —
                                          rápido, usado quando o índice foi
                                          construído sobre texto anonimizado
                                          já limpo (prompt-tuning).
                             regex        Extrai o nº do prontuário direto do
                                          texto do chunk (7-8 dígitos, ou
                                          "PACIENTE - NNNNNNN") — usado quando
                                          o GraphRAG ingeriu o CSV bruto e o
                                          nº aparece no próprio texto
                                          (no-prompt-tuning).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

TIPOS_DOENCA = [
    "CONDICAO_MEDICA", "DIAGNÓSTICO", "DIAGNOSTICO",
    "HIPÓTESE_DIAGNÓSTICO", "HIPOTESE_DIAGNOSTICA",
    "CONDICAO_CLINICA", "CONDIÇÃO_CLÍNICA",
    "SINTOMA", "SINAL_CLINICO", "DOENCA", "DOENÇA",
]


@dataclass
class Config:
    parquet_dir: str
    output_dir: str
    granularity: str = "record"  # "record" | "document"
    matcher: str = "fingerprint"  # "fingerprint" | "regex" — só usado se granularity="record"
    csv_path: str | None = None  # obrigatório se granularity="record"
    filter_disease_types: bool = True
    fingerprint_len: int = 120
    file_type: str = ""
    verbose: bool = False


class Logger:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def info(self, msg: str):
        if self.verbose:
            print(msg)

    def warn(self, msg: str):
        print(f"⚠  {msg}")


class Utils:
    @staticmethod
    def para_lista(val):
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return list(val)
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return [v.strip() for v in val.split(",") if v.strip()]
        if hasattr(val, "__iter__"):
            return list(val)
        return []


# =============================================================================
# REPOSITORY
# =============================================================================

class DataRepository:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def load_all(self) -> dict:
        self.logger.info("Carregando dados...")

        base = self.config.parquet_dir
        data = {
            "entities": pd.read_parquet(os.path.join(base, "entities.parquet")),
            "text_units": pd.read_parquet(os.path.join(base, "text_units.parquet")),
            "communities": pd.read_parquet(os.path.join(base, "communities.parquet")),
            "community_reports": pd.read_parquet(os.path.join(base, "community_reports.parquet")),
        }

        if self.config.granularity == "record":
            data["df_orig"] = pd.read_csv(self.config.csv_path, dtype=str)

        return data


# =============================================================================
# MATCHERS (chunk -> prontuario_id), usados só quando granularity="record"
# =============================================================================

class FingerprintMatcher:
    """Casa chunk <-> prontuário pelos primeiros N caracteres do texto original.

    Rápido e preciso quando o índice foi construído sobre o texto já
    anonimizado/limpo (prompt-tuning) — o chunk reproduz esse texto quase
    literalmente.
    """

    def __init__(self, config: Config, logger: Logger):
        self.fp_len = config.fingerprint_len
        self.logger = logger

    def match_chunks(self, df_orig: pd.DataFrame, text_units: pd.DataFrame, col_text: str, col_id: str) -> dict:
        df_orig = df_orig.copy()
        df_orig[col_text] = df_orig[col_text].fillna("").str.strip()
        df_orig = df_orig[df_orig[col_text] != ""].reset_index(drop=True)

        df_orig["_fp"] = df_orig[col_text].str[: self.fp_len].str.strip()
        if df_orig["_fp"].duplicated().sum() > 0:
            self.logger.info("Colisão detectada — aumentando fingerprint")
            df_orig["_fp"] = df_orig[col_text].str[: self.fp_len * 2].str.strip()
        fp_map = dict(zip(df_orig["_fp"], df_orig[col_id]))

        chunk_map = {}
        nao_encontrados = 0
        for _, chunk in text_units.iterrows():
            texto = str(chunk["text"]).strip()
            fp = texto[: self.fp_len].strip()
            pront_id = fp_map.get(fp)

            if not pront_id:
                trecho = texto[:200]
                for _, row in df_orig.iterrows():
                    if trecho in row[col_text]:
                        pront_id = row[col_id]
                        break

            if pront_id:
                chunk_map[chunk["id"]] = pront_id
            else:
                nao_encontrados += 1

        self.logger.info(f"Chunks sem match: {nao_encontrados}")
        return chunk_map


class RegexMatcher:
    """Casa chunk <-> prontuário extraindo o nº direto do texto do chunk.

    Usado quando o GraphRAG ingeriu o CSV bruto (sem anonimização/limpeza
    prévia), então o número do prontuário ainda aparece no texto indexado.
    """

    _RE_PRONT = re.compile(r"\b(\d{7,8})\b")
    _RE_PACIENTE = re.compile(r"PACIENTE\s*-\s*(\d{7,8})", re.IGNORECASE)

    def __init__(self, config: Config, logger: Logger):
        self.logger = logger

    def _build_prontuario_set(self, df_orig: pd.DataFrame, col_id: str) -> set:
        pront_set = set()
        for val in df_orig[col_id].dropna().unique():
            pront_set.update(self._RE_PRONT.findall(str(val).strip()))
        self.logger.info(f"Prontuários válidos encontrados: {len(pront_set)}")
        return pront_set

    def _build_descricao_index(self, df_orig: pd.DataFrame, col_text: str, col_id: str) -> list:
        idx = []
        for _, row in df_orig.iterrows():
            desc = str(row.get(col_text, "")).strip()
            pront = str(row.get(col_id, "")).strip()
            if desc and pront:
                idx.append((pront, desc))
        self.logger.info(f"Índice descricao construído: {len(idx)} registros")
        return idx

    def _extract_regex(self, texto: str, pront_set: set) -> str | None:
        m = self._RE_PACIENTE.search(texto)
        if m and m.group(1) in pront_set:
            return m.group(1)
        for n in self._RE_PRONT.findall(texto):
            if n in pront_set:
                return n
        return None

    def _fallback_substring(self, texto: str, desc_index: list) -> str | None:
        trecho = texto[50:250].strip() if len(texto) > 250 else texto.strip()
        if len(trecho) < 30:
            return None
        for pront, desc in desc_index:
            if trecho in desc:
                nums = self._RE_PRONT.findall(pront)
                return nums[0] if nums else pront
        return None

    def match_chunks(self, df_orig: pd.DataFrame, text_units: pd.DataFrame, col_text: str, col_id: str) -> dict:
        pront_set = self._build_prontuario_set(df_orig, col_id)
        desc_index = self._build_descricao_index(df_orig, col_text, col_id)

        chunk_map = {}
        stats = {"regex": 0, "fallback": 0, "sem_match": 0}
        for _, chunk in text_units.iterrows():
            texto = str(chunk["text"]).strip()
            chunk_id = chunk["id"]

            pront = self._extract_regex(texto, pront_set)
            if pront:
                chunk_map[chunk_id] = pront
                stats["regex"] += 1
                continue

            pront = self._fallback_substring(texto, desc_index)
            if pront:
                chunk_map[chunk_id] = pront
                stats["fallback"] += 1
                continue

            stats["sem_match"] += 1

        self.logger.info(f"Match — regex: {stats['regex']}, fallback: {stats['fallback']}, sem match: {stats['sem_match']}")
        self.logger.warn(f"Chunks mapeados: {len(chunk_map)}/{len(text_units)} ({len(chunk_map) / len(text_units) * 100:.1f}%)")
        return chunk_map


MATCHERS = {"fingerprint": FingerprintMatcher, "regex": RegexMatcher}

# =============================================================================
# DATASET BUILDER
# =============================================================================

class DatasetBuilder:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def _community_lookup(self, communities: pd.DataFrame, reports: pd.DataFrame):
        comunidade_map = {}
        for _, row in communities.iterrows():
            for eid in Utils.para_lista(row.get("entity_ids", [])):
                comunidade_map[eid] = row["community"]
        report_idx = reports.set_index("community") if "community" in reports.columns else None
        return comunidade_map, report_idx

    def _entidades(self, entities: pd.DataFrame) -> pd.DataFrame:
        if self.config.filter_disease_types:
            entidades = entities[entities["type"].isin(TIPOS_DOENCA)].copy()
        else:
            entidades = entities.copy()
        self.logger.info(f"Entidades: {len(entidades)} (tipos: {entidades['type'].nunique()})")
        entidades["text_unit_ids"] = entidades["text_unit_ids"].apply(Utils.para_lista)
        return entidades

    def build_record(self, data: dict, chunk_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        entidades = self._entidades(data["entities"])
        text_idx = data["text_units"].set_index("id")
        comunidade_map, report_idx = self._community_lookup(data["communities"], data["community_reports"])

        registros = []
        for _, ent in entidades.iterrows():
            eid, condicao, tipo = ent["id"], ent["title"], ent["type"]
            comm_id = comunidade_map.get(eid)
            comm_title = comm_summary = ""
            if comm_id and report_idx is not None and comm_id in report_idx.index:
                rep = report_idx.loc[comm_id]
                if isinstance(rep, pd.DataFrame):
                    rep = rep.iloc[0]
                comm_title, comm_summary = rep.get("title", ""), rep.get("summary", "")

            for uid in ent["text_unit_ids"]:
                if uid not in text_idx.index or uid not in chunk_map:
                    continue
                chunk = text_idx.loc[uid]
                registros.append({
                    "prontuario_id": chunk_map[uid],
                    "condicao": condicao,
                    "tipo_entidade": tipo,
                    "texto_evidencia": chunk["text"],
                    "chunk_id": uid,
                    "entity_id": eid,
                    "community_id": comm_id,
                    "community_title": comm_title,
                    "community_summary": comm_summary,
                })

        df = pd.DataFrame(registros)
        if df.empty:
            raise ValueError(
                "Dataset vazio — nenhuma entidade pôde ser associada a um prontuário. "
                f"chunk_map tem {len(chunk_map)} entradas, entidades tem {len(entidades)} linhas."
            )
        df = df.drop_duplicates(["prontuario_id", "condicao", "chunk_id"]).sort_values(["prontuario_id", "condicao"]).reset_index(drop=True)

        agg = df.groupby("prontuario_id")["condicao"].apply(lambda x: sorted(set(x))).reset_index(name="condicoes_identificadas")
        agg["n_condicoes"] = agg["condicoes_identificadas"].apply(len)

        self.logger.info(f"Dataset: {len(df)} linhas, {agg['prontuario_id'].nunique()} prontuários")
        return df, agg

    def build_document(self, data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        entidades = self._entidades(data["entities"])
        text_idx = data["text_units"].set_index("id")
        comunidade_map, report_idx = self._community_lookup(data["communities"], data["community_reports"])

        registros = []
        for _, ent in entidades.iterrows():
            eid, nome, tipo = ent["id"], ent["title"], ent["type"]
            comm_id = comunidade_map.get(eid)
            comm_title = comm_summary = ""
            if comm_id is not None and report_idx is not None and comm_id in report_idx.index:
                rep = report_idx.loc[comm_id]
                comm_title, comm_summary = rep.get("title", ""), rep.get("summary", "")

            for uid in ent["text_unit_ids"]:
                if uid not in text_idx.index:
                    continue
                chunk = text_idx.loc[uid]
                for doc_id in Utils.para_lista(chunk.get("document_ids", [])):
                    registros.append({
                        "document_id": doc_id,
                        "entity_id": eid,
                        "condicao": nome,
                        "tipo_entidade": tipo,
                        "descricao_entidade": ent.get("description", ""),
                        "chunk_id": uid,
                        "texto_evidencia": chunk["text"],
                        "n_tokens": chunk.get("n_tokens"),
                        "community_id": comm_id,
                        "community_title": comm_title,
                        "community_summary": comm_summary,
                    })

        df = pd.DataFrame(registros)
        if df.empty:
            raise ValueError("Dataset vazio")
        df = df.drop_duplicates(["document_id", "condicao", "chunk_id"]).sort_values(["document_id", "condicao"]).reset_index(drop=True)

        agg = df.groupby("document_id")["condicao"].apply(lambda x: sorted(set(x))).reset_index().rename(columns={"condicao": "condicoes_identificadas"})
        agg["n_condicoes"] = agg["condicoes_identificadas"].apply(len)

        self.logger.info(f"Dataset: {len(df)} linhas, {agg['document_id'].nunique()} documentos")
        return df, agg


# =============================================================================
# EXPORT
# =============================================================================

class DataExporter:
    def __init__(self, config: Config):
        self.output = config.output_dir
        self.file_type = config.file_type
        self.granularity = config.granularity
        os.makedirs(self.output, exist_ok=True)

    def save(self, dataset: pd.DataFrame, agg: pd.DataFrame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        type_part = f"_{self.file_type}" if self.file_type else ""
        base = f"dataset_{self.granularity}{type_part}_{ts}"
        agg_name = f"agg_{self.granularity}{type_part}_{ts}"

        dataset.to_csv(os.path.join(self.output, f"{base}.csv"), index=False)
        dataset.to_parquet(os.path.join(self.output, f"{base}.parquet"), index=False)
        agg.to_csv(os.path.join(self.output, f"{agg_name}.csv"), index=False)

        print(f"✔ Salvo em {self.output} como: {base}.{{csv,parquet}}, {agg_name}.csv")


# =============================================================================
# CLI / APP
# =============================================================================

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--parquet-dir", required=True, help="Diretório com os parquets do índice GraphRAG (<root>/output)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--granularity", choices=["record", "document"], default="record")
    parser.add_argument("--matcher", choices=list(MATCHERS), default="fingerprint", help="Só usado com --granularity record")
    parser.add_argument("--csv", dest="csv_path", default=None, help="CSV original — obrigatório com --granularity record")
    parser.add_argument("--no-filter-disease-types", dest="filter_disease_types", action="store_false", help="Usa todas as entidades, não só os tipos de doença (necessário sem prompt-tuning, onde os tipos são genéricos)")
    parser.add_argument("--type", dest="file_type", default="", help='Rótulo adicionado ao nome do arquivo (ex.: "prompt-tuning", "no-prompt-tuning")')
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.granularity == "record" and not args.csv_path:
        parser.error("--csv é obrigatório com --granularity record")

    return Config(
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        granularity=args.granularity,
        matcher=args.matcher,
        csv_path=args.csv_path,
        filter_disease_types=args.filter_disease_types,
        file_type=args.file_type,
        verbose=args.verbose,
    )


def main() -> None:
    config = parse_args()
    logger = Logger(config.verbose)
    data = DataRepository(config, logger).load_all()
    builder = DatasetBuilder(config, logger)

    if config.granularity == "document":
        dataset, agg = builder.build_document(data)
    else:
        matcher = MATCHERS[config.matcher](config, logger)
        chunk_map = matcher.match_chunks(data["df_orig"], data["text_units"], col_text="descricao", col_id="prontuario")
        dataset, agg = builder.build_record(data, chunk_map)

    DataExporter(config).save(dataset, agg)


if __name__ == "__main__":
    main()
