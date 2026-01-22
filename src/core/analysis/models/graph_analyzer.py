"""
Analisador de grafos para extrair insights dos relacionamentos.
Foca em identificar nós com mais arestas e extrair padrões importantes.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import json


class GraphAnalyzer:
    """
    Analisa grafos de relacionamentos para identificar nós centrais
    e extrair insights sobre padrões de conexão.
    """

    def __init__(self, relationships_path: str, entities_path: Optional[str] = None):
        """
        Inicializa o analisador com os caminhos dos arquivos.

        Args:
            relationships_path: Caminho para o arquivo relationships.parquet
            entities_path: Caminho opcional para o arquivo entities.parquet
        """
        self.relationships_path = Path(relationships_path)
        self.entities_path = Path(entities_path) if entities_path else None
        self.relationships_df: Optional[pd.DataFrame] = None
        self.entities_df: Optional[pd.DataFrame] = None
        self.node_stats: Dict[str, Dict] = {}

    def load_data(self) -> None:
        """Carrega os dados dos arquivos parquet."""
        if not self.relationships_path.exists():
            raise FileNotFoundError(
                f"Arquivo de relacionamentos não encontrado: {self.relationships_path}"
            )

        print(f"Carregando relacionamentos de {self.relationships_path}...")
        self.relationships_df = pd.read_parquet(self.relationships_path)
        print(f"✓ {len(self.relationships_df)} relacionamentos carregados")

        if self.entities_path and self.entities_path.exists():
            print(f"Carregando entidades de {self.entities_path}...")
            self.entities_df = pd.read_parquet(self.entities_path)
            print(f"✓ {len(self.entities_df)} entidades carregadas")
        else:
            print("⚠ Arquivo de entidades não fornecido ou não encontrado")

    def _calculate_node_degrees(self) -> Dict[str, Dict]:
        """
        Calcula o grau (número de arestas) de cada nó.
        
        Returns:
            Dicionário com estatísticas por nó: {
                node_id: {
                    'out_degree': número de arestas saindo,
                    'in_degree': número de arestas entrando,
                    'total_degree': total de arestas,
                    'out_relationships': lista de relacionamentos saindo,
                    'in_relationships': lista de relacionamentos entrando
                }
            }
        """
        if self.relationships_df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")

        node_stats = defaultdict(lambda: {
            'out_degree': 0,
            'in_degree': 0,
            'total_degree': 0,
            'out_relationships': [],
            'in_relationships': []
        })

        # Verifica quais colunas existem (pode ser 'source'/'target' ou 'subject_id'/'object_id')
        has_source_target = 'source' in self.relationships_df.columns and 'target' in self.relationships_df.columns
        has_subject_object = 'subject_id' in self.relationships_df.columns and 'object_id' in self.relationships_df.columns

        if has_source_target:
            source_col = 'source'
            target_col = 'target'
        elif has_subject_object:
            source_col = 'subject_id'
            target_col = 'object_id'
        else:
            raise ValueError(
                "Arquivo de relacionamentos deve ter colunas 'source'/'target' ou 'subject_id'/'object_id'"
            )

        # Calcula graus
        for _, row in self.relationships_df.iterrows():
            source = str(row[source_col])
            target = str(row[target_col])

            # Grau de saída (out-degree)
            node_stats[source]['out_degree'] += 1
            node_stats[source]['out_relationships'].append({
                'target': target,
                'description': row.get('description', ''),
                'weight': row.get('weight', 1.0),
                'id': row.get('id', '')
            })

            # Grau de entrada (in-degree)
            node_stats[target]['in_degree'] += 1
            node_stats[target]['in_relationships'].append({
                'source': source,
                'description': row.get('description', ''),
                'weight': row.get('weight', 1.0),
                'id': row.get('id', '')
            })

        # Calcula grau total
        for node_id in node_stats:
            node_stats[node_id]['total_degree'] = (
                node_stats[node_id]['out_degree'] + node_stats[node_id]['in_degree']
            )

        return dict(node_stats)

    def _enrich_with_entity_info(self, node_stats: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Enriquece as estatísticas dos nós com informações das entidades.

        Args:
            node_stats: Dicionário com estatísticas dos nós

        Returns:
            Dicionário enriquecido com informações das entidades
        """
        if self.entities_df is None:
            return node_stats

        # Cria índice de entidades por ID
        entity_map = {}
        id_col = 'id' if 'id' in self.entities_df.columns else None

        if id_col:
            for _, entity in self.entities_df.iterrows():
                entity_id = str(entity[id_col])
                entity_map[entity_id] = {
                    'title': entity.get('title', ''),
                    'type': entity.get('type', ''),
                    'description': entity.get('description', ''),
                    'human_readable_id': entity.get('human_readable_id', '')
                }

        # Enriquece estatísticas
        for node_id in node_stats:
            if node_id in entity_map:
                node_stats[node_id].update(entity_map[node_id])

        return node_stats

    def analyze_top_nodes(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analisa os nós com mais arestas e retorna um DataFrame com insights.

        Args:
            top_n: Número de nós principais para analisar

        Returns:
            DataFrame com os nós mais conectados e seus insights
        """
        if self.relationships_df is None:
            self.load_data()

        print("\nCalculando graus dos nós...")
        node_stats = self._calculate_node_degrees()

        print("Enriquecendo com informações das entidades...")
        node_stats = self._enrich_with_entity_info(node_stats)

        # Ordena por grau total
        sorted_nodes = sorted(
            node_stats.items(),
            key=lambda x: x[1]['total_degree'],
            reverse=True
        )

        # Prepara dados para DataFrame
        results = []
        for node_id, stats in sorted_nodes[:top_n]:
            # Extrai descrições mais comuns dos relacionamentos
            all_descriptions = [
                rel['description'] for rel in stats['out_relationships']
            ] + [rel['description'] for rel in stats['in_relationships']]
            description_counts = Counter([d for d in all_descriptions if d])

            # Extrai tipos de relacionamento mais comuns
            top_descriptions = description_counts.most_common(3)

            # Calcula peso médio
            all_weights = [
                rel['weight'] for rel in stats['out_relationships']
            ] + [rel['weight'] for rel in stats['in_relationships']]
            avg_weight = sum(all_weights) / len(all_weights) if all_weights else 0.0

            results.append({
                'node_id': node_id,
                'title': stats.get('title', 'N/A'),
                'type': stats.get('type', 'N/A'),
                'description': stats.get('description', 'N/A')[:200] if stats.get('description') else 'N/A',
                'total_degree': stats['total_degree'],
                'out_degree': stats['out_degree'],
                'in_degree': stats['in_degree'],
                'avg_weight': round(avg_weight, 3),
                'top_relationship_types': ', '.join([f"{desc}({count})" for desc, count in top_descriptions]),
                'human_readable_id': stats.get('human_readable_id', 'N/A')
            })

        self.node_stats = dict(sorted_nodes)
        return pd.DataFrame(results)

    def get_node_insights(self, node_id: str) -> Dict:
        """
        Retorna insights detalhados sobre um nó específico.

        Args:
            node_id: ID do nó

        Returns:
            Dicionário com insights detalhados
        """
        if not self.node_stats:
            self.analyze_top_nodes()

        if node_id not in self.node_stats:
            raise ValueError(f"Nó {node_id} não encontrado nas estatísticas")

        stats = self.node_stats[node_id]

        # Análise de relacionamentos de saída
        out_descriptions = Counter([
            rel['description'] for rel in stats['out_relationships']
            if rel['description']
        ])
        out_targets = Counter([
            rel['target'] for rel in stats['out_relationships']
        ])

        # Análise de relacionamentos de entrada
        in_descriptions = Counter([
            rel['description'] for rel in stats['in_relationships']
            if rel['description']
        ])
        in_sources = Counter([
            rel['source'] for rel in stats['in_relationships']
        ])

        return {
            'node_id': node_id,
            'title': stats.get('title', 'N/A'),
            'type': stats.get('type', 'N/A'),
            'description': stats.get('description', 'N/A'),
            'degrees': {
                'total': stats['total_degree'],
                'out': stats['out_degree'],
                'in': stats['in_degree']
            },
            'outgoing_relationships': {
                'total': stats['out_degree'],
                'top_descriptions': dict(out_descriptions.most_common(5)),
                'top_targets': dict(out_targets.most_common(5))
            },
            'incoming_relationships': {
                'total': stats['in_degree'],
                'top_descriptions': dict(in_descriptions.most_common(5)),
                'top_sources': dict(in_sources.most_common(5))
            },
            'avg_weight': sum([
                rel['weight'] for rel in stats['out_relationships'] + stats['in_relationships']
            ]) / stats['total_degree'] if stats['total_degree'] > 0 else 0.0
        }

    def generate_report(self, output_path: str, top_n: int = 20) -> str:
        """
        Gera um relatório completo em formato JSON com insights dos nós principais.

        Args:
            output_path: Caminho para salvar o relatório
            top_n: Número de nós principais para incluir no relatório

        Returns:
            Caminho do arquivo gerado
        """
        print(f"\nGerando relatório com top {top_n} nós...")
        top_nodes_df = self.analyze_top_nodes(top_n=top_n)

        # Prepara relatório completo
        report = {
            'summary': {
                'total_relationships': len(self.relationships_df),
                'total_nodes_analyzed': len(self.node_stats),
                'top_n_nodes': top_n
            },
            'top_nodes': top_nodes_df.to_dict('records'),
            'detailed_insights': {}
        }

        # Adiciona insights detalhados dos top 10
        for i, (node_id, _) in enumerate(sorted(
            self.node_stats.items(),
            key=lambda x: x[1]['total_degree'],
            reverse=True
        )[:10]):
            report['detailed_insights'][node_id] = self.get_node_insights(node_id)

        # Salva relatório
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ Relatório salvo em: {output_file.absolute()}")

        # Também salva DataFrame como CSV
        csv_path = output_file.with_suffix('.csv')
        top_nodes_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ CSV salvo em: {csv_path.absolute()}")

        return str(output_file.absolute())

