"""Plota o subgrafo de entidades/relações do índice GraphRAG ligadas a UM
paciente específico -- o grafo completo (26.966 entidades) é ilegível, mas
o subgrafo de um único paciente (dezenas de entidades) é visível e mostra
diretamente o mecanismo investigado: entidades do mesmo paciente caindo em
comunidades diferentes no grafo (ex.: "Doença Hepática Crônica Avançada" vs
"Síndrome de Hipertensão Portal" do prontuário 1800283).

Uso:
    uv run --no-sync python3 experiments/grid_search/plota_subgrafo_paciente.py [PRONTUARIO_ID]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
INDICE_DIR = _REPO_ROOT / "experimentos" / "local copy" / "output"
SAIDA_DIR = _REPO_ROOT / "tcc-results-01"

NIVEL_COMUNIDADE = 1  # mesmo nivel usado na investigacao (local/basic L1-L2)


def acha_text_units_do_paciente(pid: str) -> pd.DataFrame:
    tu = pd.read_parquet(INDICE_DIR / "text_units.parquet")
    return tu[tu["text"].str.contains(pid, na=False)]


def main(pid: str, comunidades_foco: list[int] | None = None) -> None:
    print(f"Paciente: {pid}")
    tu_paciente = acha_text_units_do_paciente(pid)
    if tu_paciente.empty:
        print("⚠ Nenhum text_unit encontrado com esse ID no índice.")
        return
    print(f"  {len(tu_paciente)} text_units encontrados")

    entity_ids = set()
    for ids in tu_paciente["entity_ids"]:
        if ids is not None:
            entity_ids.update(ids)
    print(f"  {len(entity_ids)} entidades ligadas")

    entities = pd.read_parquet(INDICE_DIR / "entities.parquet")
    ents = entities[entities["id"].isin(entity_ids)].copy()

    communities = pd.read_parquet(INDICE_DIR / "communities.parquet")
    comms_nivel = communities[communities["level"] == NIVEL_COMUNIDADE]

    # mapeia entidade -> comunidade (nivel escolhido)
    ent_para_comunidade = {}
    for _, c in comms_nivel.iterrows():
        ids_comunidade = c["entity_ids"]
        if ids_comunidade is None:
            continue
        for eid in ids_comunidade:
            if eid in entity_ids:
                ent_para_comunidade[eid] = c["community"]
    ents["comunidade"] = ents["id"].map(ent_para_comunidade)

    # relacoes entre essas entidades (so edges onde AMBAS as pontas estao no conjunto)
    titulo_por_id = dict(zip(ents["id"], ents["title"]))
    rel = pd.read_parquet(INDICE_DIR / "relationships.parquet")
    rel_sub = rel[rel["source"].isin(ents["title"]) & rel["target"].isin(ents["title"])]
    print(f"  {len(rel_sub)} relações entre essas entidades")

    G = nx.Graph()
    for _, e in ents.iterrows():
        G.add_node(e["title"], comunidade=e["comunidade"], tipo=e["type"])
    for _, r in rel_sub.iterrows():
        G.add_edge(r["source"], r["target"], weight=r.get("weight", 1))

    if comunidades_foco:
        # mantem so os nos das comunidades de foco + vizinhos diretos (1 salto)
        nucleo = {n for n, d in G.nodes(data=True) if d["comunidade"] in comunidades_foco}
        vizinhos = set()
        for n in nucleo:
            vizinhos.update(G.neighbors(n))
        manter = nucleo | vizinhos
        G = G.subgraph(manter).copy()
        print(f"  Filtrado pras comunidades {comunidades_foco} + vizinhos diretos: {G.number_of_nodes()} nós")

    # remove nos isolados (sem nenhuma relacao dentro do subgrafo) p/ visual mais limpo
    isolados = list(nx.isolates(G))
    G.remove_nodes_from(isolados)
    print(f"  {len(isolados)} nós isolados removidos (sem relação dentro do subgrafo)")
    print(f"  Grafo final: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")

    # destaca só as comunidades mais frequentes nesse subgrafo (as que tem mais
    # de 1 entidade aqui) -- o resto (comunidades de 1 entidade so, que so
    # diluem a leitura) fica cinza. Com 30+ comunidades de tamanho 1 num
    # grafo de 63 nos, colorir tudo deixa a legenda inutil (cores repetidas).
    contagem_comunidade = pd.Series([d["comunidade"] for _, d in G.nodes(data=True)]).value_counts()
    if comunidades_foco:
        comunidades_destacadas = list(comunidades_foco)
    else:
        comunidades_destacadas = contagem_comunidade[contagem_comunidade >= 2].index.tolist()
    cmap = plt.cm.tab10
    cor_por_comunidade = {c: cmap(i % 10) for i, c in enumerate(comunidades_destacadas)}
    cores_nos = [cor_por_comunidade.get(G.nodes[n]["comunidade"], "#d9d9d9") for n in G.nodes()]
    tamanhos_nos = [1600 if G.nodes[n]["comunidade"] in comunidades_destacadas else 700 for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(18, 13))
    pos = nx.spring_layout(G, k=2.2, seed=7, iterations=150)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cores_nos, node_size=tamanhos_nos,
                            edgecolors="black", linewidths=0.8)

    # rotula só os nós das comunidades destacadas (os demais poluiriam o centro)
    labels_destacados = {n: n for n in G.nodes() if G.nodes[n]["comunidade"] in comunidades_destacadas}
    nx.draw_networkx_labels(G, pos, labels=labels_destacados, ax=ax, font_size=9, font_weight="bold")
    labels_resto = {n: n for n in G.nodes() if G.nodes[n]["comunidade"] not in comunidades_destacadas}
    nx.draw_networkx_labels(G, pos, labels=labels_resto, ax=ax, font_size=6, font_color="#555555")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cor_por_comunidade[c],
                   markersize=14, label=f"Comunidade {c} (nível {NIVEL_COMUNIDADE}, {contagem_comunidade[c]} entidades aqui)")
        for c in comunidades_destacadas
    ]
    rotulo_resto = "Vizinhos diretos (outra comunidade)" if comunidades_foco else "Outras comunidades (1 entidade cada)"
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d9d9d9",
                               markersize=10, label=rotulo_resto))
    ax.legend(handles=handles, loc="upper left", fontsize=11)

    if comunidades_foco:
        titulo = (f"Prontuário {pid} — comunidades {comunidades_foco} (nível {NIVEL_COMUNIDADE}) "
                  f"e seus vizinhos diretos")
    else:
        titulo = f"Subgrafo de entidades do prontuário {pid} — comunidades com 2+ entidades em destaque (nível {NIVEL_COMUNIDADE})"
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.axis("off")

    sufixo = f"_comunidades_{'_'.join(map(str, comunidades_foco))}" if comunidades_foco else ""
    saida = SAIDA_DIR / f"subgrafo_paciente_{pid}{sufixo}.png"
    plt.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✔ Salvo em: {saida}")


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "1800283"
    foco = [int(c) for c in sys.argv[2:]] if len(sys.argv) > 2 else None
    main(pid, comunidades_foco=foco)
