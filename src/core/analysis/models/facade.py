"""
Fachada para facilitar o uso do analisador de grafos.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from graph_analyzer import GraphAnalyzer
from community_analyzer import CommunityReportsAnalyzer
from person_analyzer import PersonAnalyzer


class GraphAnalysisFeature:
    """
    Fachada que encapsula o fluxo completo de análise de grafos.
    """

    def __init__(
        self,
        relationships_path: str,
        entities_path: Optional[str] = None,
        community_reports_path: Optional[str] = None
    ):
        """
        Inicializa a feature de análise de grafos.

        Args:
            relationships_path: Caminho para o arquivo relationships.parquet
            entities_path: Caminho opcional para o arquivo entities.parquet
            community_reports_path: Caminho opcional para o arquivo community_reports.parquet
        """
        self.analyzer = GraphAnalyzer(relationships_path, entities_path)
        self.community_analyzer = None
        self.person_analyzer = None
        
        if community_reports_path:
            self.community_analyzer = CommunityReportsAnalyzer(community_reports_path)
        
        if entities_path:
            self.person_analyzer = PersonAnalyzer(relationships_path, entities_path)

    def analyze(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analisa os nós com mais arestas e retorna insights.

        Args:
            top_n: Número de nós principais para analisar

        Returns:
            DataFrame com os nós mais conectados
        """
        return self.analyzer.analyze_top_nodes(top_n=top_n)

    def get_insights(self, node_id: str) -> dict:
        """
        Retorna insights detalhados sobre um nó específico.

        Args:
            node_id: ID do nó

        Returns:
            Dicionário com insights detalhados
        """
        return self.analyzer.get_node_insights(node_id)

    def generate_report(
        self,
        output_path: str = "output/graph_analysis_report.json",
        top_n: int = 20
    ) -> str:
        """
        Gera relatório completo com insights dos nós principais.

        Args:
            output_path: Caminho para salvar o relatório
            top_n: Número de nós principais para incluir

        Returns:
            Caminho do arquivo gerado
        """
        return self.analyzer.generate_report(output_path, top_n=top_n)

    # Métodos para análise de comunidades
    def analyze_communities(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analisa os rankings das comunidades.

        Args:
            top_n: Número de comunidades principais

        Returns:
            DataFrame com análise de rankings
        """
        if not self.community_analyzer:
            raise ValueError("CommunityReportsAnalyzer não inicializado. Forneça community_reports_path.")
        return self.community_analyzer.analyze_rankings()

    def get_top_communities(self, top_n: int = 10, by: str = 'rank') -> pd.DataFrame:
        """
        Retorna as comunidades mais importantes.

        Args:
            top_n: Número de comunidades
            by: Critério de ordenação ('rank', 'size', ou 'findings_count')

        Returns:
            DataFrame com as top comunidades
        """
        if not self.community_analyzer:
            raise ValueError("CommunityReportsAnalyzer não inicializado. Forneça community_reports_path.")
        return self.community_analyzer.get_top_communities(top_n=top_n, by=by)

    def analyze_community_findings(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analisa os findings das comunidades mais importantes.

        Args:
            top_n: Número de comunidades top para analisar

        Returns:
            DataFrame com análise de findings
        """
        if not self.community_analyzer:
            raise ValueError("CommunityReportsAnalyzer não inicializado. Forneça community_reports_path.")
        return self.community_analyzer.analyze_findings(top_n=top_n)

    def generate_community_report(
        self,
        output_path: str = "output/community_analysis_report.json",
        top_n: int = 20
    ) -> str:
        """
        Gera relatório completo de análise de comunidades.

        Args:
            output_path: Caminho para salvar o relatório
            top_n: Número de comunidades principais para incluir

        Returns:
            Caminho do arquivo gerado
        """
        if not self.community_analyzer:
            raise ValueError("CommunityReportsAnalyzer não inicializado. Forneça community_reports_path.")
        return self.community_analyzer.generate_comprehensive_report(output_path, top_n=top_n)

    # Métodos para análise de pessoas
    def get_top_persons(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna as pessoas com mais relacionamentos.

        Args:
            top_n: Número de pessoas principais

        Returns:
            DataFrame com as top pessoas
        """
        if not self.person_analyzer:
            raise ValueError("PersonAnalyzer não inicializado. Forneça entities_path.")
        return self.person_analyzer.get_top_persons(top_n=top_n)

    def analyze_person_clinical_evolution(self, person_id: Optional[str] = None) -> dict:
        """
        Analisa a evolução clínica de uma pessoa.

        Args:
            person_id: ID da pessoa (se None, usa a pessoa com mais relacionamentos)

        Returns:
            Dicionário com análise da evolução clínica
        """
        if not self.person_analyzer:
            raise ValueError("PersonAnalyzer não inicializado. Forneça entities_path.")
        
        if person_id is None:
            top_persons = self.get_top_persons(top_n=1)
            if len(top_persons) == 0:
                raise ValueError("Nenhuma pessoa encontrada para análise")
            person_id = top_persons.iloc[0]['person_id']
        
        return self.person_analyzer.analyze_clinical_evolution(person_id)

    def generate_person_report(
        self,
        person_id: Optional[str] = None,
        output_path: str = "output/person_analysis_report.json"
    ) -> str:
        """
        Gera relatório completo de análise de pessoa.

        Args:
            person_id: ID da pessoa (se None, usa a pessoa com mais relacionamentos)
            output_path: Caminho para salvar o relatório

        Returns:
            Caminho do arquivo gerado
        """
        if not self.person_analyzer:
            raise ValueError("PersonAnalyzer não inicializado. Forneça entities_path.")
        return self.person_analyzer.generate_person_report(person_id, output_path)

