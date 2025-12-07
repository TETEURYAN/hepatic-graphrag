"""
Exemplo de uso do analisador de grafos.

Este script demonstra como usar a feature de análise de grafos
para extrair insights dos nós com mais arestas.
"""

from pathlib import Path
from facade import GraphAnalysisFeature


def main():
    """Exemplo de uso da feature de análise de grafos."""
    
    # Define os caminhos dos arquivos
    base_path = Path(__file__).parent.parent.parent.parent
    relationships_path = base_path / "datatest" / "output" / "relationships.parquet"
    entities_path = base_path / "datatest" / "output" / "entities.parquet"
    
    # Verifica se os arquivos existem
    if not relationships_path.exists():
        print(f"❌ Arquivo não encontrado: {relationships_path}")
        return
    
    print("=" * 60)
    print("Análise de Grafos - Nós com Mais Arestas")
    print("=" * 60)
    
    # Inicializa a feature
    analyzer = GraphAnalysisFeature(
        relationships_path=str(relationships_path),
        entities_path=str(entities_path) if entities_path.exists() else None
    )
    
    # Carrega os dados
    analyzer.analyzer.load_data()
    
    # Analisa os top 20 nós
    print("\nAnalisando nós com mais conexões...")
    top_nodes = analyzer.analyze(top_n=20)
    
    # Exibe resultados
    print("\n" + "=" * 60)
    print("10 NÓS COM MAIS ARESTAS")
    print("=" * 60)
    print(top_nodes[['node_id', 'title', 'type', 'total_degree', 'out_degree', 'in_degree']].head(10).to_string(index=False))
    
    # Gera relatório completo
    output_dir = base_path / "datatest" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = analyzer.generate_report(
        output_path=str(output_dir / "graph_analysis_report.json"),
        top_n=20
    )
    
    print("\n" + "=" * 60)
    print("INSIGHTS DETALHADOS DO NÓ #1")
    print("=" * 60)
    
    if len(top_nodes) > 0:
        top_node_id = top_nodes.iloc[0]['node_id']
        insights = analyzer.get_insights(top_node_id)
        
        print(f"\nNó: {insights['node_id']}")
        print(f"Título: {insights.get('title', 'N/A')}")
        print(f"Tipo: {insights.get('type', 'N/A')}")
        print(f"\nGraus:")
        print(f"  - Total: {insights['degrees']['total']}")
        print(f"  - Saída: {insights['degrees']['out']}")
        print(f"  - Entrada: {insights['degrees']['in']}")
        
        print(f"\nRelacionamentos de Saída (Top 3):")
        for desc, count in list(insights['outgoing_relationships']['top_descriptions'].items())[:3]:
            print(f"  - {desc}: {count}")
        
        print(f"\nRelacionamentos de Entrada (Top 3):")
        for desc, count in list(insights['incoming_relationships']['top_descriptions'].items())[:3]:
            print(f"  - {desc}: {count}")
    
    print("\n" + "=" * 60)
    print("Análise concluída!")
    print(f"Relatório completo salvo em: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

