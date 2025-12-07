"""
Analisador de relatórios de comunidades para extrair insights dos relatórios gerados por LLM.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict
import json
import re


class CommunityReportsAnalyzer:
    """
    Analisa relatórios de comunidades para extrair insights sobre estrutura,
    importância, findings e padrões textuais.
    """

    def __init__(self, community_reports_path: str):
        """
        Inicializa o analisador com o caminho do arquivo.

        Args:
            community_reports_path: Caminho para o arquivo community_reports.parquet
        """
        self.community_reports_path = Path(community_reports_path)
        self.reports_df: Optional[pd.DataFrame] = None
        self.analysis_results: Dict[str, Any] = {}

    def load_data(self) -> None:
        """Carrega os dados do arquivo parquet."""
        if not self.community_reports_path.exists():
            raise FileNotFoundError(
                f"Arquivo de relatórios de comunidades não encontrado: {self.community_reports_path}"
            )

        print(f"Carregando relatórios de comunidades de {self.community_reports_path}...")
        self.reports_df = pd.read_parquet(self.community_reports_path)
        print(f"✓ {len(self.reports_df)} relatórios carregados")

    def _parse_findings(self, findings) -> List[Dict]:
        """
        Parse dos findings que podem estar em diferentes formatos.

        Args:
            findings: Findings em formato string, lista ou dict

        Returns:
            Lista de findings parseados
        """
        # Trata valores None ou NaN
        if findings is None:
            return []
        
        try:
            if pd.isna(findings):
                return []
        except (TypeError, ValueError):
            pass

        # Se já é uma lista
        if isinstance(findings, list):
            # Verifica se a lista está vazia
            if len(findings) == 0:
                return []
            # Se é lista de dicionários, retorna como está
            if all(isinstance(item, dict) for item in findings):
                return findings
            # Se é lista de strings, tenta fazer parse de cada uma
            parsed_list = []
            for item in findings:
                if isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            parsed_list.append(parsed)
                        elif isinstance(parsed, list):
                            parsed_list.extend(parsed)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(item, dict):
                    parsed_list.append(item)
            return parsed_list

        # Se é dicionário, retorna como lista
        if isinstance(findings, dict):
            return [findings]

        # Se é string, tenta fazer parse JSON
        if isinstance(findings, str):
            # Se string vazia, retorna lista vazia
            if not findings.strip():
                return []
            try:
                parsed = json.loads(findings)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        return []

    def analyze_rankings(self) -> pd.DataFrame:
        """
        Analisa os rankings das comunidades e retorna estatísticas.

        Returns:
            DataFrame com análise de rankings
        """
        if self.reports_df is None:
            self.load_data()

        print("\nAnalisando rankings das comunidades...")

        # Filtra apenas comunidades com rank válido
        valid_ranks = self.reports_df[self.reports_df['rank'].notna()].copy()

        if len(valid_ranks) == 0:
            print("⚠ Nenhum rank válido encontrado")
            return pd.DataFrame()

        # Ordena por rank (maior = mais importante)
        sorted_reports = valid_ranks.sort_values('rank', ascending=False)

        results = []
        for idx, row in sorted_reports.iterrows():
            # Usa acesso direto à coluna com tratamento de erros
            try:
                findings_value = row['findings']
            except (KeyError, AttributeError):
                findings_value = None
            
            findings = self._parse_findings(findings_value)
            findings_count = len(findings) if findings else 0

            # Acessa valores de forma segura usando try/except
            try:
                title_val = row['title'] if pd.notna(row['title']) else 'N/A'
                title = str(title_val)[:100] if title_val != 'N/A' else 'N/A'
            except (KeyError, AttributeError):
                title = 'N/A'
            
            try:
                rank_explanation_val = row['rank_explanation'] if pd.notna(row['rank_explanation']) else 'N/A'
                rank_explanation = str(rank_explanation_val)[:200] if rank_explanation_val != 'N/A' else 'N/A'
            except (KeyError, AttributeError):
                rank_explanation = 'N/A'
            
            try:
                summary_val = row['summary'] if pd.notna(row['summary']) else ''
                summary_length = len(str(summary_val)) if summary_val else 0
            except (KeyError, AttributeError):
                summary_length = 0
            
            try:
                period_val = row['period'] if pd.notna(row['period']) else 'N/A'
                period = str(period_val) if period_val != 'N/A' else 'N/A'
            except (KeyError, AttributeError):
                period = 'N/A'

            try:
                community = row['community']
            except (KeyError, AttributeError):
                community = 'N/A'
            
            try:
                rank = row['rank']
            except (KeyError, AttributeError):
                rank = 0
            
            try:
                size = row['size']
            except (KeyError, AttributeError):
                size = 0
            
            try:
                level = row['level']
            except (KeyError, AttributeError):
                level = 0
            
            try:
                has_parent = pd.notna(row['parent'])
            except (KeyError, AttributeError):
                has_parent = False

            results.append({
                'community': community,
                'title': title,
                'rank': rank,
                'rank_explanation': rank_explanation,
                'size': size,
                'level': level,
                'findings_count': findings_count,
                'summary_length': summary_length,
                'has_parent': has_parent,
                'period': period
            })

        return pd.DataFrame(results)

    def analyze_hierarchy(self) -> Dict:
        """
        Analisa a estrutura hierárquica das comunidades.

        Returns:
            Dicionário com estatísticas hierárquicas
        """
        if self.reports_df is None:
            self.load_data()

        print("\nAnalisando estrutura hierárquica...")

        # Conta por nível
        level_counts = self.reports_df['level'].value_counts().sort_index().to_dict()

        # Comunidades com e sem parent
        has_parent = self.reports_df['parent'].notna().sum()
        no_parent = self.reports_df['parent'].isna().sum()

        # Tamanho médio por nível
        avg_size_by_level = self.reports_df.groupby('level')['size'].mean().to_dict()

        # Rank médio por nível
        valid_ranks = self.reports_df[self.reports_df['rank'].notna()]
        avg_rank_by_level = valid_ranks.groupby('level')['rank'].mean().to_dict()

        return {
            'total_communities': len(self.reports_df),
            'levels': {
                'count_by_level': level_counts,
                'avg_size_by_level': {k: round(v, 2) for k, v in avg_size_by_level.items()},
                'avg_rank_by_level': {k: round(v, 2) for k, v in avg_rank_by_level.items()}
            },
            'hierarchy': {
                'with_parent': int(has_parent),
                'without_parent': int(no_parent),
                'root_communities_ratio': round(no_parent / len(self.reports_df), 3)
            }
        }

    def analyze_findings(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analisa os findings das comunidades mais importantes.

        Args:
            top_n: Número de comunidades top para analisar

        Returns:
            DataFrame com análise de findings
        """
        if self.reports_df is None:
            self.load_data()

        print(f"\nAnalisando findings das top {top_n} comunidades...")

        # Ordena por rank ou size
        valid_ranks = self.reports_df[self.reports_df['rank'].notna()].copy()
        if len(valid_ranks) > 0:
            sorted_reports = valid_ranks.sort_values('rank', ascending=False).head(top_n)
        else:
            sorted_reports = self.reports_df.sort_values('size', ascending=False).head(top_n)

        results = []
        for idx, row in sorted_reports.iterrows():
            # Acessa findings de forma segura
            try:
                findings_value = row['findings']
            except (KeyError, AttributeError):
                findings_value = None
            
            findings = self._parse_findings(findings_value)

            if not findings:
                continue

            # Acessa outros valores de forma segura
            try:
                title_val = row['title'] if pd.notna(row['title']) else 'N/A'
                title = str(title_val)[:80] if title_val != 'N/A' else 'N/A'
            except (KeyError, AttributeError):
                title = 'N/A'
            
            try:
                community_val = row['community']
            except (KeyError, AttributeError):
                community_val = 'N/A'
            
            try:
                rank_val = row['rank']
            except (KeyError, AttributeError):
                rank_val = 0
            
            try:
                size_val = row['size']
            except (KeyError, AttributeError):
                size_val = 0

            for finding_idx, finding in enumerate(findings):
                if isinstance(finding, dict):
                    summary = finding.get('summary', 'N/A')
                    explanation = finding.get('explanation', 'N/A')
                else:
                    summary = str(finding)
                    explanation = 'N/A'

                results.append({
                    'community': community_val,
                    'title': title,
                    'rank': rank_val,
                    'finding_index': finding_idx + 1,
                    'finding_summary': str(summary)[:200] if summary != 'N/A' else 'N/A',
                    'finding_explanation_length': len(str(explanation)),
                    'size': size_val
                })

        return pd.DataFrame(results)

    def analyze_text_patterns(self) -> Dict:
        """
        Analisa padrões textuais nos summaries e títulos.

        Returns:
            Dicionário com padrões identificados
        """
        if self.reports_df is None:
            self.load_data()

        print("\nAnalisando padrões textuais...")

        # Palavras mais comuns nos títulos
        titles = self.reports_df['title'].dropna().astype(str)
        title_words = []
        for title in titles:
            # Remove caracteres especiais e divide em palavras
            words = re.findall(r'\b\w+\b', title.lower())
            title_words.extend(words)

        # Remove palavras muito comuns (stopwords básicas)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'de', 'da', 'do', 'dos', 'das'}
        title_words_filtered = [w for w in title_words if w not in stopwords and len(w) > 2]
        top_title_words = Counter(title_words_filtered).most_common(20)

        # Análise de comprimento de summaries
        summaries = self.reports_df['summary'].dropna().astype(str)
        summary_lengths = summaries.str.len()
        summary_word_counts = summaries.str.split().str.len()

        # Palavras-chave nos summaries (palavras que aparecem frequentemente)
        all_summary_words = []
        for summary in summaries:
            words = re.findall(r'\b\w{4,}\b', summary.lower())
            all_summary_words.extend(words)

        top_summary_words = Counter(all_summary_words).most_common(30)

        return {
            'titles': {
                'total_titles': len(titles),
                'avg_title_length': round(titles.str.len().mean(), 2),
                'top_words': [{'word': word, 'count': count} for word, count in top_title_words]
            },
            'summaries': {
                'total_summaries': len(summaries),
                'avg_length_chars': round(summary_lengths.mean(), 2),
                'avg_length_words': round(summary_word_counts.mean(), 2),
                'min_length': int(summary_lengths.min()),
                'max_length': int(summary_lengths.max()),
                'top_keywords': [{'word': word, 'count': count} for word, count in top_summary_words]
            }
        }

    def analyze_size_distribution(self) -> Dict:
        """
        Analisa a distribuição de tamanhos das comunidades.

        Returns:
            Dicionário com estatísticas de tamanho
        """
        if self.reports_df is None:
            self.load_data()

        print("\nAnalisando distribuição de tamanhos...")

        sizes = self.reports_df['size'].dropna()

        return {
            'total_communities': len(sizes),
            'size_stats': {
                'mean': round(sizes.mean(), 2),
                'median': round(sizes.median(), 2),
                'std': round(sizes.std(), 2),
                'min': int(sizes.min()),
                'max': int(sizes.max()),
                'q25': round(sizes.quantile(0.25), 2),
                'q75': round(sizes.quantile(0.75), 2)
            },
            'size_categories': {
                'small': int((sizes < sizes.quantile(0.33)).sum()),
                'medium': int(((sizes >= sizes.quantile(0.33)) & (sizes < sizes.quantile(0.67))).sum()),
                'large': int((sizes >= sizes.quantile(0.67)).sum())
            }
        }

    def get_top_communities(self, top_n: int = 10, by: str = 'rank') -> pd.DataFrame:
        """
        Retorna as comunidades mais importantes.

        Args:
            top_n: Número de comunidades
            by: Critério de ordenação ('rank', 'size', ou 'findings_count')

        Returns:
            DataFrame com as top comunidades
        """
        if self.reports_df is None:
            self.load_data()

        df = self.reports_df.copy()

        if by == 'rank':
            df = df[df['rank'].notna()].sort_values('rank', ascending=False)
        elif by == 'size':
            df = df.sort_values('size', ascending=False)
        elif by == 'findings_count':
            df['findings_count'] = df['findings'].apply(
                lambda x: len(self._parse_findings(x))
            )
            df = df.sort_values('findings_count', ascending=False)
        else:
            raise ValueError(f"Critério inválido: {by}. Use 'rank', 'size' ou 'findings_count'")

        results = []
        for idx, row in df.head(top_n).iterrows():
            # Acessa findings de forma segura
            try:
                findings_value = row['findings']
            except (KeyError, AttributeError):
                findings_value = None
            
            findings = self._parse_findings(findings_value)

            # Acessa outros valores de forma segura
            try:
                title_val = row['title'] if pd.notna(row['title']) else 'N/A'
                title = str(title_val) if title_val != 'N/A' else 'N/A'
            except (KeyError, AttributeError):
                title = 'N/A'
            
            try:
                rank_explanation_val = row['rank_explanation'] if pd.notna(row['rank_explanation']) else 'N/A'
                rank_explanation = str(rank_explanation_val)[:300] if rank_explanation_val != 'N/A' else 'N/A'
            except (KeyError, AttributeError):
                rank_explanation = 'N/A'
            
            try:
                summary_val = row['summary'] if pd.notna(row['summary']) else 'N/A'
                summary = str(summary_val)[:500] if summary_val != 'N/A' else 'N/A'
            except (KeyError, AttributeError):
                summary = 'N/A'
            
            try:
                period_val = row['period'] if pd.notna(row['period']) else 'N/A'
                period = str(period_val) if period_val != 'N/A' else 'N/A'
            except (KeyError, AttributeError):
                period = 'N/A'

            try:
                community = row['community']
            except (KeyError, AttributeError):
                community = 'N/A'
            
            try:
                rank = row['rank']
            except (KeyError, AttributeError):
                rank = 'N/A'
            
            try:
                size = row['size']
            except (KeyError, AttributeError):
                size = 0
            
            try:
                level = row['level']
            except (KeyError, AttributeError):
                level = 0
            
            try:
                has_parent = pd.notna(row['parent'])
            except (KeyError, AttributeError):
                has_parent = False

            results.append({
                'community': community,
                'title': title,
                'rank': rank,
                'rank_explanation': rank_explanation,
                'size': size,
                'level': level,
                'findings_count': len(findings),
                'summary': summary,
                'period': period,
                'has_parent': has_parent
            })

        return pd.DataFrame(results)

    def generate_comprehensive_report(self, output_path: str, top_n: int = 20) -> str:
        """
        Gera relatório completo com todas as análises.

        Args:
            output_path: Caminho para salvar o relatório
            top_n: Número de comunidades top para análise detalhada

        Returns:
            Caminho do arquivo gerado
        """
        print("\n" + "=" * 60)
        print("Gerando relatório completo de análise de comunidades...")
        print("=" * 60)

        if self.reports_df is None:
            self.load_data()

        # Executa todas as análises
        rankings_analysis = self.analyze_rankings()
        hierarchy_analysis = self.analyze_hierarchy()
        findings_analysis = self.analyze_findings(top_n=top_n)
        text_patterns = self.analyze_text_patterns()
        size_distribution = self.analyze_size_distribution()
        top_communities = self.get_top_communities(top_n=top_n, by='rank')

        # Compila relatório
        report = {
            'summary': {
                'total_communities': len(self.reports_df),
                'total_reports_analyzed': top_n,
                'analysis_date': pd.Timestamp.now().isoformat()
            },
            'rankings': {
                'top_communities': top_communities.to_dict('records'),
                'rankings_analysis': rankings_analysis.to_dict('records')[:top_n]
            },
            'hierarchy': hierarchy_analysis,
            'size_distribution': size_distribution,
            'text_patterns': text_patterns,
            'findings_analysis': findings_analysis.to_dict('records') if len(findings_analysis) > 0 else []
        }

        # Salva relatório JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ Relatório JSON salvo em: {output_file.absolute()}")

        # Salva DataFrames como CSV
        csv_dir = output_file.parent / output_file.stem
        csv_dir.mkdir(exist_ok=True)

        if len(rankings_analysis) > 0:
            rankings_analysis.to_csv(csv_dir / 'rankings_analysis.csv', index=False, encoding='utf-8')
            print(f"✓ Rankings CSV salvo")

        if len(top_communities) > 0:
            top_communities.to_csv(csv_dir / 'top_communities.csv', index=False, encoding='utf-8')
            print(f"✓ Top comunidades CSV salvo")

        if len(findings_analysis) > 0:
            findings_analysis.to_csv(csv_dir / 'findings_analysis.csv', index=False, encoding='utf-8')
            print(f"✓ Findings CSV salvo")

        return str(output_file.absolute())

