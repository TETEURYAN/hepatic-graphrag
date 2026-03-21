"""
Exemplo de uso do analisador de pessoas para análise de evolução clínica.

Este script demonstra como usar a feature de análise de pessoas
para extrair insights sobre evolução clínica baseado nos relacionamentos.
"""

from pathlib import Path
from core.analysis.models.facade import GraphAnalysisFeature
from core.analysis.models.person_analyzer import PersonAnalyzer


def main():
    """Exemplo de uso da feature de análise de pessoas."""
    
    # Define os caminhos dos arquivos
    base_path = Path(__file__).parent.parent.parent.parent
    relationships_path = base_path / "datatest" / "output" / "relationships.parquet"
    entities_path = base_path / "datatest" / "output" / "entities.parquet"
    
    # Verifica se os arquivos existem
    if not relationships_path.exists():
        print(f"❌ Arquivo não encontrado: {relationships_path}")
        return
    
    if not entities_path.exists():
        print(f"❌ Arquivo não encontrado: {entities_path}")
        return
    
    print("=" * 60)
    print("Análise de Evolução Clínica - Entidades do Tipo Person")
    print("=" * 60)
    
    # Opção 1: Usando a facade (recomendado)
    print("\n📊 Usando GraphAnalysisFeature...")
    analyzer = GraphAnalysisFeature(
        relationships_path=str(relationships_path),
        entities_path=str(entities_path)
    )
    
    # Carrega os dados
    analyzer.person_analyzer.load_data()
    
    # Obtém top pessoas
    print("\n👤 Identificando pessoas com mais relacionamentos...")
    top_persons = analyzer.get_top_persons(top_n=10)
    
    if len(top_persons) > 0:
        print("\n" + "=" * 60)
        print("TOP 10 PESSOAS COM MAIS RELACIONAMENTOS")
        print("=" * 60)
        print(top_persons[['person_id', 'title', 'total_degree', 'out_degree', 'in_degree']].to_string(index=False))
        
        # Analisa a pessoa com mais relacionamentos
        # Usa person_name se disponível, senão usa person_id
        if 'person_name' in top_persons.columns:
            top_person_identifier = top_persons.iloc[0]['person_name']
        else:
            top_person_identifier = top_persons.iloc[0]['person_id']
        top_person_title = top_persons.iloc[0]['title']
        
        print("\n" + "=" * 60)
        print(f"ANÁLISE DE EVOLUÇÃO CLÍNICA: {top_person_title}")
        print("=" * 60)
        
        clinical_analysis = analyzer.analyze_person_clinical_evolution(person_id=top_person_identifier)
        
        print(f"\nInformações da Pessoa:")
        print(f"  ID: {clinical_analysis['person_info']['title']}")
        print(f"  Total de relacionamentos: {clinical_analysis['person_info']['total_relationships']}")
        
        print(f"\nCategorias de Relacionamentos:")
        for category, data in clinical_analysis['relationship_categories'].items():
            if data['count'] > 0:
                print(f"  - {category.capitalize()}: {data['count']} relacionamentos")
        
        print(f"\nResumo Clínico:")
        print(f"  {clinical_analysis['clinical_summary']}")
        
        print(f"\nEntidades Conectadas:")
        print(f"  Total: {clinical_analysis['connected_entities']['total']}")
        print(f"  Amostra das principais:")
        for entity in clinical_analysis['connected_entities']['sample'][:5]:
            print(f"    - {entity['title']} ({entity['type']})")
        
        print(f"\nPadrões Temporais:")
        if clinical_analysis['temporal_patterns']['dates_found'] > 0:
            print(f"  Datas encontradas: {clinical_analysis['temporal_patterns']['dates_found']}")
            print(f"  Amostra: {', '.join(clinical_analysis['temporal_patterns']['sample_dates'][:5])}")
        else:
            print(f"  Nenhuma data explícita encontrada nas descrições")
        
        print(f"\nTop Tipos de Relacionamento:")
        for rel_type in clinical_analysis['top_relationship_types'][:5]:
            print(f"  - {rel_type['description'][:80]}... ({rel_type['count']}x)")
        
        # Gera relatório completo
        output_dir = base_path / "datatest" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGerando relatório completo...")
        report_path = analyzer.generate_person_report(
            person_id=top_person_identifier,
            output_path=str(output_dir / "person_analysis_report.json")
        )
        
    else:
        print("⚠ Nenhuma pessoa encontrada")
    
    # Opção 2: Usando o analisador diretamente
    print("\n" + "=" * 60)
    print("ANÁLISE DETALHADA")
    print("=" * 60)
    
    person_analyzer = PersonAnalyzer(
        str(relationships_path),
        str(entities_path)
    )
    person_analyzer.load_data()
    
    top_persons_direct = person_analyzer.get_top_persons(top_n=5)
    
    if len(top_persons_direct) > 0:
        print("\n📊 Top 5 Pessoas:")
        for idx, row in top_persons_direct.iterrows():
            print(f"\n{idx + 1}. {row['title']} (ID: {row['person_id']})")
            print(f"   Relacionamentos: {row['total_degree']} (Saída: {row['out_degree']}, Entrada: {row['in_degree']})")
    
    print("\n" + "=" * 60)
    print("✅ Análise concluída!")
    if len(top_persons) > 0:
        print(f"📄 Relatório completo salvo em: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

