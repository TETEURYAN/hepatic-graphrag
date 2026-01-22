"""
Script para visualizar os resultados do GraphRAG a partir dos arquivos .parquet
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path

# Configuração do caminho dos dados
OUTPUT_DIR = Path("datatest/output")

def load_data():
    """Carrega todos os arquivos .parquet"""
    data = {}
    
    files = {
        'entities': 'entities.parquet',
        'relationships': 'relationships.parquet',
        'communities': 'communities.parquet',
        'documents': 'documents.parquet',
        'text_units': 'text_units.parquet',
        'community_reports': 'community_reports.parquet'
    }
    
    for key, filename in files.items():
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            try:
                data[key] = pd.read_parquet(filepath)
                print(f"✓ Carregado {filename}: {data[key].shape[0]} registros")
            except Exception as e:
                print(f"✗ Erro ao carregar {filename}: {e}")
        else:
            print(f"✗ Arquivo não encontrado: {filename}")
    
    return data

def visualize_entities(data):
    """Visualiza estatísticas das entidades"""
    if 'entities' not in data:
        return
    
    df = data['entities']
    print("\n=== ANÁLISE DE ENTIDADES ===")
    print(f"Total de entidades: {len(df)}")
    
    # Mostrar colunas disponíveis
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    
    # Criar figura
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top 20 Entidades', 'Distribuição de Tipos', 
                       'Estatísticas Gerais', 'Entidades por Comunidade'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "table"}, {"type": "bar"}]]
    )
    
    # Top 20 entidades (usar title/nome ao invés de ID)
    if 'title' in df.columns:
        # Mostrar top 20 entidades por título
        top_entities = df['title'].value_counts().head(20)
        fig.add_trace(
            go.Bar(x=top_entities.index, y=top_entities.values, name="Frequência"),
            row=1, col=1
        )
    elif 'name' in df.columns:
        top_entities = df['name'].value_counts().head(20)
        fig.add_trace(
            go.Bar(x=top_entities.index, y=top_entities.values, name="Frequência"),
            row=1, col=1
        )
    elif 'id' in df.columns:
        # Se não houver título, mostrar IDs (mas tentar mapear para nomes se possível)
        top_entities = df['id'].value_counts().head(20)
        fig.add_trace(
            go.Bar(x=top_entities.index.astype(str), y=top_entities.values, name="Frequência"),
            row=1, col=1
        )
    
    # Distribuição de tipos (se houver)
    if 'type' in df.columns:
        type_counts = df['type'].value_counts()
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values, name="Tipos"),
            row=1, col=2
        )
    
    # Estatísticas gerais
    stats_data = {
        'Métrica': ['Total de Entidades', 'Tipos Únicos', 'Comunidades Únicas'],
        'Valor': [
            len(df),
            df['type'].nunique() if 'type' in df.columns else 'N/A',
            df['community_id'].nunique() if 'community_id' in df.columns else 'N/A'
        ]
    }
    fig.add_trace(
        go.Table(
            header=dict(values=list(stats_data.keys())),
            cells=dict(values=[stats_data['Métrica'], stats_data['Valor']])
        ),
        row=2, col=1
    )
    
    # Entidades por comunidade (se houver)
    if 'community_id' in df.columns:
        comm_counts = df['community_id'].value_counts().head(20)
        fig.add_trace(
            go.Bar(x=comm_counts.index.astype(str), y=comm_counts.values, name="Por Comunidade"),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Análise de Entidades", showlegend=False)
    fig.write_html("visualizacoes/entidades.html")
    print("✓ Visualização de entidades salva em: visualizacoes/entidades.html")

def create_id_to_name_mapping(data):
    """Cria um mapeamento de ID para nome usando as entidades"""
    id_to_name = {}
    
    if 'entities' in data:
        df_entities = data['entities']
        # Procurar coluna de ID e nome
        id_col = None
        name_col = None
        
        for col in ['id', 'entity_id']:
            if col in df_entities.columns:
                id_col = col
                break
        
        for col in ['title', 'name', 'label']:
            if col in df_entities.columns:
                name_col = col
                break
        
        if id_col and name_col:
            for _, row in df_entities.iterrows():
                entity_id = str(row[id_col])
                entity_name = str(row[name_col]) if pd.notna(row[name_col]) else entity_id
                id_to_name[entity_id] = entity_name
            print(f"✓ Mapeamento criado: {len(id_to_name)} entidades mapeadas")
        else:
            print(f"⚠ Não foi possível criar mapeamento. ID col: {id_col}, Name col: {name_col}")
    
    return id_to_name

def get_name_for_id(entity_id, id_to_name):
    """Retorna o nome da entidade ou o ID se não encontrar"""
    entity_id_str = str(entity_id)
    return id_to_name.get(entity_id_str, entity_id_str)

def visualize_relationships(data):
    """Visualiza o grafo de relacionamentos"""
    if 'relationships' not in data:
        return
    
    df = data['relationships']
    print("\n=== ANÁLISE DE RELACIONAMENTOS ===")
    print(f"Total de relacionamentos: {len(df)}")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    
    # Criar mapeamento de ID para nome
    id_to_name = create_id_to_name_mapping(data)
    
    # Criar grafo NetworkX
    G = nx.Graph()
    
    # Adicionar arestas (ajustar conforme colunas disponíveis)
    source_col = None
    target_col = None
    weight_col = None
    
    for col in ['source', 'source_id', 'from', 'from_id']:
        if col in df.columns:
            source_col = col
            break
    
    for col in ['target', 'target_id', 'to', 'to_id']:
        if col in df.columns:
            target_col = col
            break
    
    for col in ['weight', 'strength', 'score']:
        if col in df.columns:
            weight_col = col
            break
    
    if source_col and target_col:
        for _, row in df.iterrows():
            source_id = str(row[source_col])
            target_id = str(row[target_col])
            # Usar nomes ao invés de IDs
            source = get_name_for_id(source_id, id_to_name)
            target = get_name_for_id(target_id, id_to_name)
            weight = row[weight_col] if weight_col else 1.0
            
            if 'type' in df.columns:
                rel_type = row['type']
                G.add_edge(source, target, weight=weight, type=rel_type)
            else:
                G.add_edge(source, target, weight=weight)
    
    print(f"Nós no grafo: {G.number_of_nodes()}")
    print(f"Arestas no grafo: {G.number_of_edges()}")
    
    # Para grafos muito grandes, criar uma amostra com os nós mais importantes
    MAX_NODES_FOR_VISUALIZATION = 2000
    use_sample = G.number_of_nodes() > MAX_NODES_FOR_VISUALIZATION
    
    if use_sample:
        print(f"⚠ Grafo muito grande. Criando visualização com amostra dos {MAX_NODES_FOR_VISUALIZATION} nós mais importantes...")
        # Calcular centralidade primeiro
        node_centrality = nx.degree_centrality(G)
        # Selecionar top nós por centralidade
        top_nodes = sorted(node_centrality.items(), key=lambda x: x[1], reverse=True)[:MAX_NODES_FOR_VISUALIZATION]
        top_node_set = set([node[0] for node in top_nodes])
        # Criar subgrafo com top nós e suas conexões
        G_vis = G.subgraph(top_node_set).copy()
        print(f"Subgrafo criado: {G_vis.number_of_nodes()} nós, {G_vis.number_of_edges()} arestas")
    else:
        G_vis = G
    
    # Visualização interativa com Plotly
    # Usar layout mais rápido para grafos grandes
    if G_vis.number_of_nodes() > 500:
        pos = nx.spring_layout(G_vis, k=0.5, iterations=30, seed=42)
    else:
        pos = nx.spring_layout(G_vis, k=1, iterations=50, seed=42)
    
    # Criar dicionário de informações das arestas para hover
    edge_info = {}
    if source_col and target_col:
        for _, row in df.iterrows():
            source_id = str(row[source_col])
            target_id = str(row[target_col])
            source_name = get_name_for_id(source_id, id_to_name)
            target_name = get_name_for_id(target_id, id_to_name)
            edge_key = (source_name, target_name)
            
            info_parts = []
            if 'description' in df.columns and pd.notna(row['description']):
                info_parts.append(f"Descrição: {row['description']}")
            if 'type' in df.columns and pd.notna(row['type']):
                info_parts.append(f"Tipo: {row['type']}")
            if weight_col and pd.notna(row[weight_col]):
                info_parts.append(f"Peso: {row[weight_col]:.3f}")
            
            edge_info[edge_key] = "<br>".join(info_parts) if info_parts else f"{source_name} → {target_name}"
    
    edge_x = []
    edge_y = []
    edge_hover = []
    for edge in G_vis.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        # Adicionar informação de hover
        edge_key = (edge[0], edge[1])
        hover_text = edge_info.get(edge_key, f"{edge[0]} → {edge[1]}")
        edge_hover.extend([hover_text, hover_text, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        hovertext=edge_hover,
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    for node in G_vis.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # node já é o nome, não precisa mapear novamente
        node_text.append(str(node)[:80])  # Limitar tamanho do texto
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',  # Remover texto para melhor performance
        hoverinfo='text',
        text=node_text,
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Centralidade",
                xanchor="left"
            ),
            line=dict(width=2)
        )
    )
    
    # Calcular centralidade (reutilizar se já calculado, mas filtrar para subgrafo)
    if not use_sample:
        node_centrality = nx.degree_centrality(G_vis)
    else:
        # Filtrar centralidade apenas para nós do subgrafo
        node_centrality = {node: node_centrality[node] for node in G_vis.nodes()}
    node_color = [node_centrality[node] for node in G_vis.nodes()]
    node_trace.marker.color = node_color
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Grafo de Relacionamentos',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text=f"Interaja com o grafo para explorar os relacionamentos{' (amostra dos nós mais importantes)' if use_sample else ''}",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color="#888", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    fig.write_html("visualizacoes/relacionamentos.html")
    print("✓ Visualização de relacionamentos salva em: visualizacoes/relacionamentos.html")
    
    # Estatísticas do grafo
    fig_stats = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Distribuição de Graus', 'Top 20 Nós por Centralidade', 
                       'Estatísticas do Grafo')
    )
    
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    fig_stats.add_trace(
        go.Histogram(x=degree_values, nbinsx=20, name="Distribuição"),
        row=1, col=1
    )
    
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    fig_stats.add_trace(
        go.Bar(x=[str(n[0])[:60] for n in top_nodes], y=[n[1] for n in top_nodes], name="Grau"),
        row=1, col=2
    )
    
    # Calcular diâmetro apenas se o grafo for pequeno o suficiente
    try:
        if G.number_of_nodes() < 1000 and nx.is_connected(G):
            diameter = nx.diameter(G)
        else:
            diameter = "N/A (grafo muito grande ou desconectado)"
    except:
        diameter = "N/A (cálculo muito custoso)"
    
    stats_data = {
        'Métrica': ['Nós', 'Arestas', 'Componentes Conectados', 'Densidade', 'Diâmetro'],
        'Valor': [
            G.number_of_nodes(),
            G.number_of_edges(),
            nx.number_connected_components(G),
            f"{nx.density(G):.4f}",
            diameter
        ]
    }
    fig_stats.add_trace(
        go.Table(
            header=dict(values=list(stats_data.keys())),
            cells=dict(values=[stats_data['Métrica'], stats_data['Valor']])
        ),
        row=1, col=3
    )
    
    fig_stats.update_layout(height=400, title_text="Estatísticas do Grafo", showlegend=False)
    fig_stats.write_html("visualizacoes/estatisticas_grafo.html")
    print("✓ Estatísticas do grafo salvas em: visualizacoes/estatisticas_grafo.html")

def visualize_communities(data):
    """Visualiza as comunidades"""
    if 'communities' not in data:
        return
    
    df = data['communities']
    print("\n=== ANÁLISE DE COMUNIDADES ===")
    print(f"Total de comunidades: {len(df)}")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tamanho das Comunidades', 'Distribuição de Tamanhos',
                       'Top 20 Comunidades', 'Estatísticas'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "table"}]]
    )
    
    # Tamanho das comunidades (se houver coluna de tamanho ou contagem)
    if 'entity_count' in df.columns:
        sizes = df['entity_count'].sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=sizes.index.astype(str), y=sizes.values, name="Tamanho"),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=sizes.values, nbinsx=20, name="Distribuição"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=sizes.head(20).index.astype(str), y=sizes.head(20).values, name="Top 20"),
            row=2, col=1
        )
    
    stats_data = {
        'Métrica': ['Total de Comunidades', 'Tamanho Médio', 'Tamanho Máximo', 'Tamanho Mínimo'],
        'Valor': [
            len(df),
            df['entity_count'].mean() if 'entity_count' in df.columns else 'N/A',
            df['entity_count'].max() if 'entity_count' in df.columns else 'N/A',
            df['entity_count'].min() if 'entity_count' in df.columns else 'N/A'
        ]
    }
    fig.add_trace(
        go.Table(
            header=dict(values=list(stats_data.keys())),
            cells=dict(values=[stats_data['Métrica'], stats_data['Valor']])
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Análise de Comunidades", showlegend=False)
    fig.write_html("visualizacoes/comunidades.html")
    print("✓ Visualização de comunidades salva em: visualizacoes/comunidades.html")

def visualize_documents(data):
    """Visualiza informações sobre documentos"""
    if 'documents' not in data:
        return
    
    df = data['documents']
    print("\n=== ANÁLISE DE DOCUMENTOS ===")
    print(f"Total de documentos: {len(df)}")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribuição de Documentos', 'Estatísticas'),
        specs=[[{"type": "bar"}, {"type": "table"}]]
    )
    
    # Visualização básica
    if 'id' in df.columns:
        doc_counts = df['id'].value_counts().head(20)
        fig.add_trace(
            go.Bar(x=doc_counts.index.astype(str), y=doc_counts.values, name="Documentos"),
            row=1, col=1
        )
    
    stats_data = {
        'Métrica': ['Total de Documentos'],
        'Valor': [len(df)]
    }
    fig.add_trace(
        go.Table(
            header=dict(values=list(stats_data.keys())),
            cells=dict(values=[stats_data['Métrica'], stats_data['Valor']])
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Análise de Documentos", showlegend=False)
    fig.write_html("visualizacoes/documentos.html")
    print("✓ Visualização de documentos salva em: visualizacoes/documentos.html")

def visualize_text_units(data):
    """Visualiza unidades de texto"""
    if 'text_units' not in data:
        return
    
    df = data['text_units']
    print("\n=== ANÁLISE DE UNIDADES DE TEXTO ===")
    print(f"Total de unidades de texto: {len(df)}")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribuição', 'Estatísticas'),
        specs=[[{"type": "histogram"}, {"type": "table"}]]
    )
    
    # Tamanho do texto (se houver coluna de texto)
    if 'text' in df.columns:
        text_lengths = df['text'].str.len()
        fig.add_trace(
            go.Histogram(x=text_lengths, nbinsx=30, name="Tamanho do Texto"),
            row=1, col=1
        )
    
    stats_data = {
        'Métrica': ['Total de Unidades', 'Tamanho Médio', 'Tamanho Máximo', 'Tamanho Mínimo'],
        'Valor': [
            len(df),
            df['text'].str.len().mean() if 'text' in df.columns else 'N/A',
            df['text'].str.len().max() if 'text' in df.columns else 'N/A',
            df['text'].str.len().min() if 'text' in df.columns else 'N/A'
        ]
    }
    fig.add_trace(
        go.Table(
            header=dict(values=list(stats_data.keys())),
            cells=dict(values=[stats_data['Métrica'], stats_data['Valor']])
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Análise de Unidades de Texto", showlegend=False)
    fig.write_html("visualizacoes/text_units.html")
    print("✓ Visualização de unidades de texto salva em: visualizacoes/text_units.html")

def create_summary_report(data):
    """Cria um relatório resumo geral"""
    print("\n=== RELATÓRIO RESUMO ===")
    
    summary = {
        'Componente': [],
        'Quantidade': []
    }
    
    for key in ['entities', 'relationships', 'communities', 'documents', 'text_units', 'community_reports']:
        if key in data:
            summary['Componente'].append(key.replace('_', ' ').title())
            summary['Quantidade'].append(len(data[key]))
    
    df_summary = pd.DataFrame(summary)
    
    fig = go.Figure(data=[
        go.Bar(x=df_summary['Componente'], y=df_summary['Quantidade'])
    ])
    
    fig.update_layout(
        title="Resumo Geral do GraphRAG",
        xaxis_title="Componente",
        yaxis_title="Quantidade",
        height=500
    )
    
    fig.write_html("visualizacoes/resumo_geral.html")
    print("✓ Relatório resumo salvo em: visualizacoes/resumo_geral.html")
    
    # Salvar resumo em texto
    with open("visualizacoes/resumo.txt", "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO RESUMO DO GRAPHRAG ===\n\n")
        for _, row in df_summary.iterrows():
            f.write(f"{row['Componente']}: {row['Quantidade']}\n")
    
    print("✓ Resumo em texto salvo em: visualizacoes/resumo.txt")

def main():
    """Função principal"""
    print("=" * 50)
    print("VISUALIZADOR DE RESULTADOS GRAPHRAG")
    print("=" * 50)
    
    # Criar diretório de visualizações
    os.makedirs("visualizacoes", exist_ok=True)
    
    # Carregar dados
    print("\nCarregando dados...")
    data = load_data()
    
    if not data:
        print("❌ Nenhum dado foi carregado. Verifique os caminhos dos arquivos.")
        return
    
    # Gerar visualizações
    print("\n" + "=" * 50)
    print("GERANDO VISUALIZAÇÕES...")
    print("=" * 50)
    
    visualize_entities(data)
    visualize_relationships(data)
    visualize_communities(data)
    visualize_documents(data)
    visualize_text_units(data)
    create_summary_report(data)
    
    print("\n" + "=" * 50)
    print("✓ VISUALIZAÇÕES CONCLUÍDAS!")
    print("=" * 50)
    print("\nArquivos HTML gerados na pasta 'visualizacoes/':")
    print("  - entidades.html")
    print("  - relacionamentos.html")
    print("  - estatisticas_grafo.html")
    print("  - comunidades.html")
    print("  - documentos.html")
    print("  - text_units.html")
    print("  - resumo_geral.html")
    print("\nAbra os arquivos HTML no seu navegador para visualizar os gráficos interativos.")

if __name__ == "__main__":
    main()

