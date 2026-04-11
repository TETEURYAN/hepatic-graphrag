"""
GraphRAG — Dataset Documento x Doença (CLI)
"""

import ast
import argparse
import os
from dataclasses import dataclass
import pandas as pd


# =============================================================================
# CONFIG 
# =============================================================================

@dataclass
class Config:
    input_dir: str
    output_dir: str
    verbose: bool = False


TIPOS_DOENCA = [
    "CONDICAO_MEDICA", "DIAGNÓSTICO", "DIAGNOSTICO",
    "HIPÓTESE_DIAGNÓSTICO", "HIPOTESE_DIAGNOSTICA",
    "CONDICAO_CLINICA", "CONDIÇÃO_CLÍNICA",
    "SINTOMA", "SINAL_CLINICO", "DOENCA", "DOENÇA",
]


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


class Logger:

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def info(self, msg: str):
        if self.verbose:
            print(msg)


# =============================================================================
# REPOSITORY
# =============================================================================

class DataLoader:

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def load(self):
        self.logger.info("Carregando parquets...")

        base = self.config.input_dir

        data = {
            "entities": pd.read_parquet(os.path.join(base, "entities.parquet")),
            "text_units": pd.read_parquet(os.path.join(base, "text_units.parquet")),
            "communities": pd.read_parquet(os.path.join(base, "communities.parquet")),
            "community_reports": pd.read_parquet(os.path.join(base, "community_reports.parquet")),
        }

        self.logger.info(f"entities: {len(data['entities'])}")
        self.logger.info(f"text_units: {len(data['text_units'])}")

        return data


# =============================================================================
# SERVICE 
# =============================================================================

class DatasetBuilder:

    def __init__(self, logger: Logger):
        self.logger = logger

    def build(self, data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:

        entities = data["entities"]
        text_units = data["text_units"]
        communities = data["communities"]
        community_reports = data["community_reports"]

        # --- filtro ---
        entidades_doenca = entities[
            entities["type"].isin(TIPOS_DOENCA)
        ].copy()

        self.logger.info(f"Entidades de doença: {len(entidades_doenca)}")

        entidades_doenca["text_unit_ids"] = entidades_doenca["text_unit_ids"].apply(Utils.para_lista)

        text_units_idx = text_units.set_index("id")

        # --- mapa comunidade ---
        comunidade_por_entidade = {}

        for _, row in communities.iterrows():
            ids = Utils.para_lista(row.get("entity_ids", []))
            for eid in ids:
                comunidade_por_entidade[eid] = row["community"]

        report_idx = (
            community_reports.set_index("community")
            if "community" in community_reports.columns else None
        )

        # --- construção ---
        registros = []

        for _, entidade in entidades_doenca.iterrows():

            eid = entidade["id"]
            nome = entidade["title"]

            community_id = comunidade_por_entidade.get(eid)

            community_title = ""
            community_summary = ""

            if community_id is not None and report_idx is not None:
                if community_id in report_idx.index:
                    rep = report_idx.loc[community_id]
                    community_title = rep.get("title", "")
                    community_summary = rep.get("summary", "")

            for uid in entidade["text_unit_ids"]:
                if uid not in text_units_idx.index:
                    continue

                chunk = text_units_idx.loc[uid]
                doc_ids = Utils.para_lista(chunk.get("document_ids", []))

                for doc_id in doc_ids:
                    registros.append({
                        "document_id": doc_id,
                        "entity_id": eid,
                        "condicao": nome,
                        "tipo_entidade": entidade["type"],
                        "descricao_entidade": entidade.get("description", ""),
                        "chunk_id": uid,
                        "texto_evidencia": chunk["text"],
                        "n_tokens": chunk.get("n_tokens"),
                        "community_id": community_id,
                        "community_title": community_title,
                        "community_summary": community_summary,
                    })

        dataset = pd.DataFrame(registros)

        if dataset.empty:
            raise ValueError("Dataset vazio")

        # --- limpeza ---
        subset = ["document_id", "condicao", "chunk_id"]
        dataset = dataset.drop_duplicates(subset=[c for c in subset if c in dataset.columns])
        dataset = dataset.sort_values(subset).reset_index(drop=True)

        # --- agregação ---
        agg = (
            dataset.groupby("document_id")["condicao"]
            .apply(lambda x: sorted(set(x)))
            .reset_index()
            .rename(columns={"condicao": "condicoes_identificadas"})
        )
        agg["n_condicoes"] = agg["condicoes_identificadas"].apply(len)

        return dataset, agg


# =============================================================================
# EXPORTER
# =============================================================================

class DataExporter:

    def __init__(self, config: Config):
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, dataset: pd.DataFrame, agg: pd.DataFrame):

        dataset.to_csv(os.path.join(self.output_dir, "dataset_documento_doenca.csv"), index=False)
        dataset.to_parquet(os.path.join(self.output_dir, "dataset_documento_doenca.parquet"), index=False)

        agg.to_csv(os.path.join(self.output_dir, "dataset_documento_doenca_agregado.csv"), index=False)

        print(f"\n✔ Arquivos salvos em: {self.output_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    return Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        verbose=args.verbose
    )


# =============================================================================
# APP
# =============================================================================

class Application:

    def __init__(self, config: Config):
        self.logger = Logger(config.verbose)
        self.loader = DataLoader(config, self.logger)
        self.builder = DatasetBuilder(self.logger)
        self.exporter = DataExporter(config)

    def run(self):
        data = self.loader.load()
        dataset, agg = self.builder.build(data)
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