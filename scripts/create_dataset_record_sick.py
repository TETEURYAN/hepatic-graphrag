"""
GraphRAG — Dataset Prontuário x Doença
"""

import ast
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    csv_path: str
    parquet_dir: str
    output_dir: str
    verbose: bool = False
    fingerprint_len: int = 120


TIPOS_DOENCA = [
    "CONDICAO_MEDICA", "DIAGNÓSTICO", "DIAGNOSTICO",
    "HIPÓTESE_DIAGNÓSTICO", "HIPOTESE_DIAGNOSTICA",
    "CONDICAO_CLINICA", "CONDIÇÃO_CLÍNICA",
    "SINTOMA", "SINAL_CLINICO", "DOENCA", "DOENÇA",
]


# =============================================================================
# LOGGER
# =============================================================================

class Logger:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def info(self, msg: str):
        if self.verbose:
            print(msg)


# =============================================================================
# UTILS
# =============================================================================

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
# REPOSITORY (DATA ACCESS)
# =============================================================================

class DataRepository:

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def load_all(self):
        self.logger.info("Carregando dados...")

        df_orig = pd.read_csv(self.config.csv_path, dtype=str)

        base = self.config.parquet_dir

        data = {
            "df_orig": df_orig,
            "entities": pd.read_parquet(os.path.join(base, "entities.parquet")),
            "text_units": pd.read_parquet(os.path.join(base, "text_units.parquet")),
            "communities": pd.read_parquet(os.path.join(base, "communities.parquet")),
            "community_reports": pd.read_parquet(os.path.join(base, "community_reports.parquet")),
        }

        return data


# =============================================================================
# DOMAIN SERVICE
# =============================================================================

class ProntuarioMatcher:

    def __init__(self, config: Config, logger: Logger):
        self.fp_len = config.fingerprint_len
        self.logger = logger

    def build_index(self, df_orig, col_text, col_id):

        df_orig[col_text] = df_orig[col_text].fillna("").str.strip()
        df_orig = df_orig[df_orig[col_text] != ""].reset_index(drop=True)

        df_orig["_fp"] = df_orig[col_text].str[:self.fp_len].str.strip()

        if df_orig["_fp"].duplicated().sum() > 0:
            self.logger.info("Colisão detectada — aumentando fingerprint")
            df_orig["_fp"] = df_orig[col_text].str[:self.fp_len * 2].str.strip()

        fp_map = dict(zip(df_orig["_fp"], df_orig[col_id]))

        return df_orig, fp_map

    def match_chunks(self, df_orig, text_units, fp_map, col_text, col_id):

        chunk_map = {}
        nao_encontrados = 0

        for _, chunk in text_units.iterrows():
            texto = str(chunk["text"]).strip()
            fp = texto[:self.fp_len].strip()

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


# =============================================================================
# USE CASE
# =============================================================================

class DatasetBuilder:

    def __init__(self, logger: Logger):
        self.logger = logger

    def build(self, data, chunk_map):

        entities = data["entities"]
        text_units = data["text_units"]
        communities = data["communities"]
        reports = data["community_reports"]

        entidades = entities[entities["type"].isin(TIPOS_DOENCA)].copy()
        entidades["text_unit_ids"] = entidades["text_unit_ids"].apply(Utils.para_lista)

        text_idx = text_units.set_index("id")

        comunidade_map = {}
        for _, row in communities.iterrows():
            for eid in Utils.para_lista(row.get("entity_ids", [])):
                comunidade_map[eid] = row["community"]

        report_idx = reports.set_index("community") if "community" in reports.columns else None

        registros = []

        for _, ent in entidades.iterrows():

            eid = ent["id"]
            condicao = ent["title"]

            comm_id = comunidade_map.get(eid)

            comm_title = ""
            comm_summary = ""

            if comm_id and report_idx is not None and comm_id in report_idx.index:
                rep = report_idx.loc[comm_id]
                comm_title = rep.get("title", "")
                comm_summary = rep.get("summary", "")

            for uid in ent["text_unit_ids"]:

                if uid not in text_idx.index:
                    continue
                if uid not in chunk_map:
                    continue

                chunk = text_idx.loc[uid]

                registros.append({
                    "prontuario_id": chunk_map[uid],
                    "condicao": condicao,
                    "tipo_entidade": ent["type"],
                    "texto_evidencia": chunk["text"],
                    "chunk_id": uid,
                    "entity_id": eid,
                    "community_id": comm_id,
                    "community_title": comm_title,
                    "community_summary": comm_summary,
                })

        df = pd.DataFrame(registros)

        if df.empty:
            raise ValueError("Dataset vazio")

        df = df.drop_duplicates(["prontuario_id", "condicao", "chunk_id"])
        df = df.sort_values(["prontuario_id", "condicao"]).reset_index(drop=True)

        agg = (
            df.groupby("prontuario_id")["condicao"]
            .apply(lambda x: sorted(set(x)))
            .reset_index(name="condicoes_identificadas")
        )
        agg["n_condicoes"] = agg["condicoes_identificadas"].apply(len)

        return df, agg


# =============================================================================
# EXPORT
# =============================================================================

class DataExporter:

    def __init__(self, config: Config):
        self.output = config.output_dir
        os.makedirs(self.output, exist_ok=True)

    def save(self, dataset, agg):

        dataset.to_csv(os.path.join(self.output, "dataset.csv"), index=False)
        dataset.to_parquet(os.path.join(self.output, "dataset.parquet"), index=False)
        agg.to_csv(os.path.join(self.output, "agg.csv"), index=False)

        print(f"✔ Salvo em {self.output}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(description="GraphRAG dataset builder")

    parser.add_argument("--csv", required=True, help="Caminho do CSV")
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    return Config(
        csv_path=args.csv,
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        verbose=args.verbose
    )


# =============================================================================
# APP
# =============================================================================

class Application:

    def __init__(self, config: Config):

        self.logger = Logger(config.verbose)

        self.repo = DataRepository(config, self.logger)
        self.matcher = ProntuarioMatcher(config, self.logger)
        self.builder = DatasetBuilder(self.logger)
        self.exporter = DataExporter(config)

        self.config = config

    def run(self):

        data = self.repo.load_all()

        df_orig, fp_map = self.matcher.build_index(
            data["df_orig"],
            col_text="descricao",
            col_id="prontuario"
        )

        chunk_map = self.matcher.match_chunks(
            df_orig,
            data["text_units"],
            fp_map,
            col_text="descricao",
            col_id="prontuario"
        )

        dataset, agg = self.builder.build(data, chunk_map)

        self.exporter.save(dataset, agg)


# =============================================================================
# ENTRYPOINT
# =============================================================================

def main():
    config = parse_args()
    app = Application(config)
    app.run()


if __name__ == "__main__":
    main()