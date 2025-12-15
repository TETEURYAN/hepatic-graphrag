"""
Exemplo de uso do analisador de relatórios de comunidades.

Este script demonstra como usar a feature de análise de comunidades
para extrair insights dos relatórios gerados por LLM.
"""

from pathlib import Path
from core.analysis.models.facade import GraphAnalysisFeature
from core.analysis.models.community_analyzer import CommunityReportsAnalyzer


def main():
    """Exemplo de uso da feature de análise de comunidades."""
    
    # Define os caminhos dos arquivos
    base_path = Path(__file__).parent.parent.parent.parent
    community_reports_path = base_path / "datatest" / "output" / "community_reports.parquet"
    
    # Verifica se o arquivo existe
    if not community_reports_path.exists():
        print(f"❌ Arquivo não encontrado: {community_reports_path}")
        return
    
    print("=" * 60)
    print("Análise de Relatórios de Comunidades")
    print("=" * 60)
    
    # Opção 1: Usando a facade (recomendado)
    print("\nUsando GraphAnalysisFeature...")
    analyzer = GraphAnalysisFeature(
        relationships_path=str(base_path / "datatest" / "output" / "relationships.parquet"),
        community_reports_path=str(community_reports_path)
    )
    
    # Carrega os dados
    analyzer.community_analyzer.load_data()
    
    # Analisa rankings
    print("\nAnalisando rankings das comunidades...")
    rankings = analyzer.analyze_communities(top_n=20)
    
    if len(rankings) > 0:
        print("\n" + "=" * 60)
        print("TOP 10 COMUNIDADES POR RANKING")
        print("=" * 60)
        print(rankings[['community', 'title', 'rank', 'size', 'findings_count']].head(10).to_string(index=False))
    
    # Obtém top comunidades
    print("\nObtendo top comunidades...")
    top_communities = analyzer.get_top_communities(top_n=10, by='rank')
    
    if len(top_communities) > 0:
        print("\n" + "=" * 60)
        print("DETALHES DAS TOP 5 COMUNIDADES")
        print("=" * 60)
        for idx, row in top_communities.head(5).iterrows():
            print(f"\n{idx + 1}. Comunidade {row['community']}: {row['title']}")
            print(f"   Rank: {row['rank']}")
            print(f"   Tamanho: {row['size']} entidades")
            print(f"   Findings: {row['findings_count']}")
            print(f"   Resumo: {row['summary'][:200]}...")
    
    # Analisa findings
    print("\nAnalisando findings...")
    findings = analyzer.analyze_community_findings(top_n=10)
    
    if len(findings) > 0:
        print(f"\n✓ {len(findings)} findings analisados")
        print("\nPrimeiros 5 findings:")
        print(findings[['community', 'title', 'finding_summary']].head(5).to_string(index=False))
    
    # Gera relatório completo
    output_dir = base_path / "datatest" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGerando relatório completo...")
    report_path = analyzer.generate_community_report(
        output_path=str(output_dir / "community_analysis_report.json"),
        top_n=20
    )
    
    # Opção 2: Usando o analisador diretamente (para análises mais específicas)
    print("\n" + "=" * 60)
    print("ANÁLISES ESPECÍFICAS")
    print("=" * 60)
    
    community_analyzer = CommunityReportsAnalyzer(str(community_reports_path))
    community_analyzer.load_data()
    
    # Análise hierárquica
    print("\nEstrutura hierárquica:")
    hierarchy = community_analyzer.analyze_hierarchy()
    print(f"  Total de comunidades: {hierarchy['total_communities']}")
    print(f"  Comunidades com parent: {hierarchy['hierarchy']['with_parent']}")
    print(f"  Comunidades raiz: {hierarchy['hierarchy']['without_parent']}")
    print(f"  Distribuição por nível: {hierarchy['levels']['count_by_level']}")
    
    # Distribuição de tamanhos
    print("\nDistribuição de tamanhos:")
    size_dist = community_analyzer.analyze_size_distribution()
    print(f"  Média: {size_dist['size_stats']['mean']}")
    print(f"  Mediana: {size_dist['size_stats']['median']}")
    print(f"  Min: {size_dist['size_stats']['min']}, Max: {size_dist['size_stats']['max']}")
    print(f"  Pequenas: {size_dist['size_categories']['small']}")
    print(f"  Médias: {size_dist['size_categories']['medium']}")
    print(f"  Grandes: {size_dist['size_categories']['large']}")
    
    # Padrões textuais
    print("\nPadrões textuais:")
    text_patterns = community_analyzer.analyze_text_patterns()
    print(f"  Palavras mais comuns nos títulos:")
    for item in text_patterns['titles']['top_words'][:5]:
        print(f"    - {item['word']}: {item['count']}")
    print(f"  Palavras-chave nos summaries:")
    for item in text_patterns['summaries']['top_keywords'][:5]:
        print(f"    - {item['word']}: {item['count']}")
    
    print("\n" + "=" * 60)
    print("Análise concluída!")
    print(f"Relatório completo salvo em: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

