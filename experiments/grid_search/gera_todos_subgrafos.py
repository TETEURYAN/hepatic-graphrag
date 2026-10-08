"""Gera o subgrafo de entidades/relações pra CADA um dos 537 prontuários
usados no experimento -- uma imagem por paciente -- mais:
  1) uma imagem comparando os N maiores grafos (mais entidades/relações);
  2) .txt com o texto de algumas comunidades (as mais citadas na amostra);
  3) análises visuais agregadas (distribuição de tamanho, densidade etc).

Carrega os parquets do índice UMA VEZ (3-4s) e reusa pra todos os 537
pacientes, em vez de recarregar a cada chamada (como o script anterior,
plota_subgrafo_paciente.py, fazia por paciente individual).

Uso:
    uv run --no-sync python3 experiments/grid_search/gera_todos_subgrafos.py
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
INDICE_DIR = _REPO_ROOT / "experimentos" / "local copy" / "output"
RESULTADOS_DIR = _REPO_ROOT / "tcc-results-01"
SAIDA_GRAFOS = RESULTADOS_DIR / "grafos_por_prontuario"
SAIDA_COMUNIDADES_TXT = RESULTADOS_DIR / "comunidades_texto"
NIVEL_COMUNIDADE = 1
N_MAIORES_PARA_COMPARAR = 6


def carrega_ids_prontuarios() -> list[str]:
    with open(_REPO_ROOT / "progresso" / "ground_truth_segunda_opiniao.csv", encoding="utf-8-sig") as f:
        return [row["prontuario_id"].strip() for row in csv.DictReader(f)]


def monta_grafo_paciente(pid: str, tu: pd.DataFrame, entities: pd.DataFrame,
                          ent_para_comunidade: dict, rel: pd.DataFrame) -> nx.Graph | None:
    tu_paciente = tu[tu["text"].str.contains(pid, na=False, regex=False)]
    if tu_paciente.empty:
        return None

    entity_ids = set()
    for ids in tu_paciente["entity_ids"]:
        if ids is not None:
            entity_ids.update(ids)
    if not entity_ids:
        return None

    ents = entities[entities["id"].isin(entity_ids)]
    ents_titulos = set(ents["title"])
    rel_sub = rel[rel["source"].isin(ents_titulos) & rel["target"].isin(ents_titulos)]

    G = nx.Graph()
    for _, e in ents.iterrows():
        G.add_node(e["title"], comunidade=ent_para_comunidade.get(e["id"]), tipo=e["type"])
    for _, r in rel_sub.iterrows():
        G.add_edge(r["source"], r["target"], weight=r.get("weight", 1))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def desenha_grafo(G: nx.Graph, titulo: str, ax=None, com_legenda=True):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))

    contagem = pd.Series([d["comunidade"] for _, d in G.nodes(data=True)]).value_counts()
    destacadas = contagem[contagem >= 2].index.tolist()
    cmap = plt.cm.tab10
    cor_por_comunidade = {c: cmap(i % 10) for i, c in enumerate(destacadas)}
    cores = [cor_por_comunidade.get(G.nodes[n]["comunidade"], "#d9d9d9") for n in G.nodes()]
    tamanhos = [500 if G.nodes[n]["comunidade"] in destacadas else 220 for n in G.nodes()]

    pos = nx.spring_layout(G, k=1.6, seed=7, iterations=60)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, edge_color="gray", width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cores, node_size=tamanhos,
                            edgecolors="black", linewidths=0.4)

    labels = {n: n for n in G.nodes() if G.nodes[n]["comunidade"] in destacadas}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6, font_weight="bold")

    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.axis("off")
    if standalone:
        plt.tight_layout()
        return fig
    return None


def gera_imagens_individuais(ids: list[str], tu, entities, ent_para_comunidade, rel) -> pd.DataFrame:
    SAIDA_GRAFOS.mkdir(parents=True, exist_ok=True)
    stats = []
    for i, pid in enumerate(ids, 1):
        G = monta_grafo_paciente(pid, tu, entities, ent_para_comunidade, rel)
        if G is None or G.number_of_nodes() == 0:
            stats.append({"prontuario_id": pid, "n_nos": 0, "n_arestas": 0, "n_comunidades": 0})
            continue

        n_comunidades = len({d["comunidade"] for _, d in G.nodes(data=True) if d["comunidade"] is not None})
        stats.append({
            "prontuario_id": pid, "n_nos": G.number_of_nodes(),
            "n_arestas": G.number_of_edges(), "n_comunidades": n_comunidades,
        })

        fig, ax = plt.subplots(figsize=(10, 8))
        desenha_grafo(G, f"Prontuário {pid} — {G.number_of_nodes()} entidades, {G.number_of_edges()} relações", ax=ax)
        fig.savefig(SAIDA_GRAFOS / f"{pid}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        if i % 25 == 0:
            print(f"  [{i}/{len(ids)}] processados...", flush=True)

    return pd.DataFrame(stats)


def gera_comparacao_maiores(ids_maiores: list[str], tu, entities, ent_para_comunidade, rel):
    n = len(ids_maiores)
    cols = 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = axes.flat if n > 1 else [axes]

    for ax, pid in zip(axes, ids_maiores):
        G = monta_grafo_paciente(pid, tu, entities, ent_para_comunidade, rel)
        desenha_grafo(G, f"{pid} ({G.number_of_nodes()} entidades, {G.number_of_edges()} relações)", ax=ax)

    for ax in list(axes)[n:]:
        ax.axis("off")

    fig.suptitle(f"Comparação dos {n} maiores grafos (mais entidades) entre os 537 prontuários",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    saida = RESULTADOS_DIR / "comparacao_maiores_grafos.png"
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Comparação dos maiores grafos: {saida}")


def exporta_comunidades_txt(comunidades_alvo: list[int], reports: pd.DataFrame):
    SAIDA_COMUNIDADES_TXT.mkdir(parents=True, exist_ok=True)
    for comm_id in comunidades_alvo:
        r = reports[(reports["community"] == comm_id) & (reports["level"] == NIVEL_COMUNIDADE)]
        if r.empty:
            continue
        row = r.iloc[0]
        texto = (
            f"COMUNIDADE {comm_id} (nível {NIVEL_COMUNIDADE})\n"
            f"Título: {row['title']}\n\n"
            f"RESUMO:\n{row['summary']}\n\n"
            f"CONTEÚDO COMPLETO:\n{row['full_content']}\n"
        )
        (SAIDA_COMUNIDADES_TXT / f"comunidade_{comm_id}.txt").write_text(texto, encoding="utf-8")
    print(f"✔ {len(comunidades_alvo)} comunidades exportadas em: {SAIDA_COMUNIDADES_TXT}")


def gera_analises_visuais(stats: pd.DataFrame):
    validos = stats[stats["n_nos"] > 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(validos["n_nos"], bins=30, color="#4C72B0", edgecolor="black")
    axes[0, 0].set_title("Distribuição: nº de entidades por prontuário")
    axes[0, 0].set_xlabel("Entidades")
    axes[0, 0].set_ylabel("Nº de prontuários")

    axes[0, 1].hist(validos["n_arestas"], bins=30, color="#DD8452", edgecolor="black")
    axes[0, 1].set_title("Distribuição: nº de relações por prontuário")
    axes[0, 1].set_xlabel("Relações")
    axes[0, 1].set_ylabel("Nº de prontuários")

    axes[1, 0].scatter(validos["n_nos"], validos["n_arestas"], alpha=0.4, color="#55A868")
    axes[1, 0].set_title("Entidades vs. Relações (densidade do subgrafo)")
    axes[1, 0].set_xlabel("Entidades")
    axes[1, 0].set_ylabel("Relações")

    axes[1, 1].hist(validos["n_comunidades"], bins=range(0, int(validos["n_comunidades"].max()) + 2),
                     color="#C44E52", edgecolor="black")
    axes[1, 1].set_title("Distribuição: nº de comunidades distintas tocadas por prontuário")
    axes[1, 1].set_xlabel("Comunidades distintas (nível 1)")
    axes[1, 1].set_ylabel("Nº de prontuários")

    fig.suptitle(f"Análise estrutural dos subgrafos — {len(validos)} prontuários com dado no índice "
                 f"(de {len(stats)} usados no experimento)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    saida = RESULTADOS_DIR / "analise_estrutural_subgrafos.png"
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Análise estrutural: {saida}")

    print("\n--- resumo estatístico ---")
    print(validos[["n_nos", "n_arestas", "n_comunidades"]].describe())
    print(f"\nPacientes sem nenhum dado no índice: {(stats['n_nos'] == 0).sum()} de {len(stats)}")


def main():
    ids = carrega_ids_prontuarios()
    print(f"Total de prontuários: {len(ids)}")

    print("Carregando índice...")
    tu = pd.read_parquet(INDICE_DIR / "text_units.parquet")
    entities = pd.read_parquet(INDICE_DIR / "entities.parquet")
    communities = pd.read_parquet(INDICE_DIR / "communities.parquet")
    rel = pd.read_parquet(INDICE_DIR / "relationships.parquet")
    reports = pd.read_parquet(INDICE_DIR / "community_reports.parquet")

    comms_nivel = communities[communities["level"] == NIVEL_COMUNIDADE]
    ent_para_comunidade = {}
    for _, c in comms_nivel.iterrows():
        ids_comunidade = c["entity_ids"]
        if ids_comunidade is None:
            continue
        for eid in ids_comunidade:
            ent_para_comunidade[eid] = c["community"]
    print(f"{len(ent_para_comunidade)} entidades mapeadas pra comunidade (nível {NIVEL_COMUNIDADE})\n")

    print("=== Gerando imagem individual por prontuário ===")
    stats = gera_imagens_individuais(ids, tu, entities, ent_para_comunidade, rel)
    stats.to_csv(RESULTADOS_DIR / "stats_subgrafos_por_prontuario.csv", index=False)
    print(f"✔ {len(ids)} imagens em: {SAIDA_GRAFOS}\n")

    print("=== Comparando os maiores grafos ===")
    maiores = stats.sort_values("n_nos", ascending=False).head(N_MAIORES_PARA_COMPARAR)
    print(maiores)
    gera_comparacao_maiores(maiores["prontuario_id"].tolist(), tu, entities, ent_para_comunidade, rel)

    print("\n=== Exportando texto de comunidades ===")
    comunidades_mais_citadas = pd.Series(list(ent_para_comunidade.values())).value_counts().head(10).index.tolist()
    exporta_comunidades_txt(comunidades_mais_citadas, reports)

    print("\n=== Análises visuais agregadas ===")
    gera_analises_visuais(stats)


if __name__ == "__main__":
    main()
