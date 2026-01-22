"""
Analisador de entidades do tipo "person" para extrair insights sobre evolução clínica.
Foca em identificar pessoas com mais relacionamentos e analisar sua trajetória clínica.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict
import json
import re
from datetime import datetime


class PersonAnalyzer:
    """
    Analisa entidades do tipo "person" para extrair insights sobre evolução clínica
    baseado nos relacionamentos no grafo.
    """

    def __init__(self, relationships_path: str, entities_path: str):
        """
        Inicializa o analisador com os caminhos dos arquivos.

        Args:
            relationships_path: Caminho para o arquivo relationships.parquet
            entities_path: Caminho para o arquivo entities.parquet
        """
        self.relationships_path = Path(relationships_path)
        self.entities_path = Path(entities_path)
        self.relationships_df: Optional[pd.DataFrame] = None
        self.entities_df: Optional[pd.DataFrame] = None
        self.person_entities: Optional[pd.DataFrame] = None
        self.person_stats: Dict[str, Dict] = {}

    def load_data(self) -> None:
        """Carrega os dados dos arquivos parquet."""
        if not self.relationships_path.exists():
            raise FileNotFoundError(
                f"Arquivo de relacionamentos não encontrado: {self.relationships_path}"
            )
        if not self.entities_path.exists():
            raise FileNotFoundError(
                f"Arquivo de entidades não encontrado: {self.entities_path}"
            )

        print(f"Carregando relacionamentos de {self.relationships_path}...")
        self.relationships_df = pd.read_parquet(self.relationships_path)
        print(f"✓ {len(self.relationships_df)} relacionamentos carregados")

        print(f"Carregando entidades de {self.entities_path}...")
        self.entities_df = pd.read_parquet(self.entities_path)
        print(f"✓ {len(self.entities_df)} entidades carregadas")
        print(self.entities_df.columns)
        # Filtra entidades do tipo "person" (case-insensitive)
        if 'type' in self.entities_df.columns:
            # Converte para string, remove espaços e compara em maiúsculas
            # Isso captura "PERSON", "person", "Person", etc.
            type_series = self.entities_df['type'].astype(str).str.strip().str.upper()
            self.person_entities = self.entities_df[type_series == 'PERSON'].copy()
            
            print(f"✓ {len(self.person_entities)} entidades do tipo 'PERSON' encontradas")
            
            # Debug: mostra alguns exemplos de tipos encontrados se não encontrou nenhuma
            if len(self.person_entities) == 0:
                print("\n⚠ Nenhuma entidade 'PERSON' encontrada. Verificando tipos disponíveis...")
                unique_types = type_series.unique()
                type_counts = type_series.value_counts()
                print(f"  Total de tipos únicos: {len(unique_types)}")
                print(f"  Top 10 tipos mais comuns:")
                for type_name, count in type_counts.head(10).items():
                    print(f"    - '{type_name}': {count} entidades")
                
                # Tenta variações comuns
                variations = ['PESSOA', 'PEOPLE', 'PACIENTE', 'PATIENT', 'HUMAN']
                print(f"\n  Verificando variações...")
                for variation in variations:
                    count = len(self.entities_df[type_series == variation])
                    if count > 0:
                        print(f"    - Encontradas {count} entidades do tipo '{variation}'")
        else:
            print("⚠ Coluna 'type' não encontrada no arquivo de entidades")
            print(f"  Colunas disponíveis: {list(self.entities_df.columns)}")
            self.person_entities = pd.DataFrame()

    def _get_person_ids(self) -> List[str]:
        """Retorna lista de IDs das entidades do tipo person."""
        if self.person_entities is None or len(self.person_entities) == 0:
            return []
        
        if 'id' in self.person_entities.columns:
            return self.person_entities['id'].astype(str).tolist()
        return []

    def _get_person_names(self) -> set:
        """Retorna conjunto de nomes/títulos das entidades do tipo person."""
        if self.person_entities is None or len(self.person_entities) == 0:
            return set()
        
        person_names = set()
        
        # Pega títulos se disponível
        if 'title' in self.person_entities.columns:
            titles = self.person_entities['title'].dropna().astype(str).str.strip()
            person_names.update(titles.tolist())
        
        # Pega IDs também (pode ser que source use ID)
        if 'id' in self.person_entities.columns:
            ids = self.person_entities['id'].astype(str).str.strip()
            person_names.update(ids.tolist())
        
        return person_names

    def _calculate_person_degrees(self) -> Dict[str, Dict]:
        """
        Calcula o grau (número de arestas) de cada pessoa.
        
        Returns:
            Dicionário com estatísticas por pessoa
        """
        if self.relationships_df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")

        # Obtém nomes/títulos das pessoas (não apenas IDs)
        person_names = self._get_person_names()
        if not person_names:
            print("INFO: Nenhuma entidade do tipo 'person' encontrada")
            return {}

        print(f"  Total de nomes de pessoas encontrados: {len(person_names)}")
        print(f"  Exemplos de nomes de pessoas: {list(person_names)[:5]}")

        # Verifica quais colunas existem
        has_source_target = 'source' in self.relationships_df.columns and 'target' in self.relationships_df.columns
        has_subject_object = 'subject_id' in self.relationships_df.columns and 'object_id' in self.relationships_df.columns

        if has_source_target:
            source_col = 'source'
            target_col = 'target'
            print(f"  Usando colunas 'source' e 'target'")
        elif has_subject_object:
            source_col = 'subject_id'
            target_col = 'object_id'
            print(f"  Usando colunas 'subject_id' e 'object_id'")
        else:
            raise ValueError(
                "Arquivo de relacionamentos deve ter colunas 'source'/'target' ou 'subject_id'/'object_id'"
            )

        # Converte source para string e remove espaços para comparação
        self.relationships_df[source_col] = self.relationships_df[source_col].astype(str).str.strip()

        # Verifica alguns exemplos de source nos relacionamentos
        unique_sources = set(self.relationships_df[source_col].unique()[:20])
        print(f"  Exemplos de source nos relacionamentos: {list(unique_sources)[:10]}")

        # Filtra relacionamentos onde source corresponde a um nome de pessoa
        # Usa comparação case-insensitive para maior flexibilidade
        person_names_upper = {name.upper() for name in person_names}
        
        # Cria máscara para relacionamentos onde source é uma pessoa
        source_upper = self.relationships_df[source_col].str.upper()
        person_relationships_mask = source_upper.isin(person_names_upper)
        
        person_relationships = self.relationships_df[person_relationships_mask].copy()
        
        print(f"  Total de relacionamentos onde source é uma pessoa: {len(person_relationships)}")
        
        if len(person_relationships) == 0:
            print("  ⚠ NENHUM RELACIONAMENTO ENCONTRADO!")
            print("  Verificando correspondências parciais...")
            # Tenta encontrar correspondências parciais
            for person_name in list(person_names)[:5]:
                matches = self.relationships_df[
                    self.relationships_df[source_col].str.contains(person_name, case=False, na=False)
                ]
                if len(matches) > 0:
                    print(f"    Encontrados {len(matches)} relacionamentos contendo '{person_name}'")
            return {}

        # Agrupa relacionamentos por pessoa (source)
        person_stats = defaultdict(lambda: {
            'out_degree': 0,
            'in_degree': 0,
            'total_degree': 0,
            'out_relationships': [],
            'in_relationships': [],
            'all_relationships': []
        })

        # Processa relacionamentos onde source é uma pessoa
        for _, row in person_relationships.iterrows():
            source = str(row[source_col]).strip()
            target = str(row[target_col]).strip() if target_col in row else ''
            
            # Normaliza o nome para encontrar a pessoa correspondente (case-insensitive)
            source_upper = source.upper()
            matching_person = None
            for person_name in person_names:
                if person_name.upper() == source_upper:
                    matching_person = person_name
                    break
            
            if matching_person:
                person_stats[matching_person]['out_degree'] += 1
                rel_info = {
                    'target': target,
                    'description': str(row.get('description', '')),
                    'weight': row.get('weight', 1.0),
                    'id': str(row.get('id', '')),
                    'type': 'outgoing'
                }
                person_stats[matching_person]['out_relationships'].append(rel_info)
                person_stats[matching_person]['all_relationships'].append(rel_info)

        # Calcula grau total (por enquanto só out_degree, já que estamos focando em source)
        for person_name in person_stats:
            person_stats[person_name]['total_degree'] = person_stats[person_name]['out_degree']

        print(f"  Pessoas com relacionamentos encontradas: {len(person_stats)}")
        if len(person_stats) > 0:
            top_person = max(person_stats.items(), key=lambda x: x[1]['total_degree'])
            print(f"  Pessoa com mais relacionamentos: '{top_person[0]}' ({top_person[1]['total_degree']} relacionamentos)")

        # Calcula grau total
        for person_id in person_stats:
            person_stats[person_id]['total_degree'] = (
                person_stats[person_id]['out_degree'] + person_stats[person_id]['in_degree']
            )

        return dict(person_stats)

    def _enrich_person_info(self, person_stats: Dict[str, Dict]) -> Dict[str, Dict]:
        """Enriquece as estatísticas com informações das entidades."""
        if self.person_entities is None or len(self.person_entities) == 0:
            return person_stats

        # Cria índice de entidades por título/nome (case-insensitive)
        entity_map = {}
        if 'title' in self.person_entities.columns:
            for _, entity in self.person_entities.iterrows():
                entity_title = str(entity.get('title', '')).strip()
                if entity_title:
                    entity_map[entity_title.upper()] = {
                        'title': entity_title,
                        'type': entity.get('type', ''),
                        'description': entity.get('description', ''),
                        'human_readable_id': entity.get('human_readable_id', ''),
                        'id': str(entity.get('id', ''))
                    }

        # Enriquece estatísticas usando nomes
        for person_name in person_stats:
            person_name_upper = str(person_name).upper()
            if person_name_upper in entity_map:
                person_stats[person_name].update(entity_map[person_name_upper])
            else:
                # Se não encontrou, adiciona o nome como título
                if 'title' not in person_stats[person_name]:
                    person_stats[person_name]['title'] = person_name

        return person_stats

    def get_top_persons(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna as pessoas com mais relacionamentos.

        Args:
            top_n: Número de pessoas principais

        Returns:
            DataFrame com as top pessoas
        """
        if self.relationships_df is None:
            self.load_data()

        print(f"\nIdentificando pessoas com mais relacionamentos...")
        person_stats = self._calculate_person_degrees()
        
        if not person_stats:
            print("Nenhuma pessoa com relacionamentos encontrada")
            return pd.DataFrame()

        person_stats = self._enrich_person_info(person_stats)

        # Ordena por grau total
        sorted_persons = sorted(
            person_stats.items(),
            key=lambda x: x[1]['total_degree'],
            reverse=True
        )

        results = []
        for person_name, stats in sorted_persons[:top_n]:
            # Tenta encontrar o ID correspondente ao nome
            person_id = person_name
            if self.person_entities is not None and 'title' in self.person_entities.columns:
                matching_entity = self.person_entities[
                    self.person_entities['title'].astype(str).str.strip().str.upper() == person_name.upper()
                ]
                if not matching_entity.empty:
                    person_id = str(matching_entity.iloc[0].get('id', person_name))
            
            results.append({
                'person_id': person_id,
                'person_name': person_name,
                'title': stats.get('title', person_name),
                'description': stats.get('description', 'N/A')[:200] if stats.get('description') else 'N/A',
                'total_degree': stats['total_degree'],
                'out_degree': stats['out_degree'],
                'in_degree': stats['in_degree'],
                'human_readable_id': stats.get('human_readable_id', 'N/A')
            })

        self.person_stats = dict(sorted_persons)
        return pd.DataFrame(results)

    def analyze_clinical_evolution(self, person_id: str) -> Dict:
        """
        Analisa a evolução clínica de uma pessoa específica.

        Args:
            person_id: ID da pessoa

        Returns:
            Dicionário com análise da evolução clínica
        """
        if not self.person_stats:
            self.get_top_persons()

        # Tenta encontrar por ID ou nome (case-insensitive)
        person_key = None
        person_id_upper = str(person_id).upper().strip()
        
        # Primeiro tenta encontrar diretamente nas chaves (que são nomes)
        for key in self.person_stats.keys():
            if str(key).upper().strip() == person_id_upper:
                person_key = key
                break
        
        if person_key is None:
            # Se não encontrou, tenta encontrar pelo ID nas entidades
            if self.person_entities is not None:
                # Busca por ID
                if 'id' in self.person_entities.columns:
                    matching_by_id = self.person_entities[
                        self.person_entities['id'].astype(str).str.strip().str.upper() == person_id_upper
                    ]
                    if not matching_by_id.empty:
                        person_title = str(matching_by_id.iloc[0].get('title', '')).strip()
                        if person_title and person_title.upper() in [k.upper() for k in self.person_stats.keys()]:
                            # Encontra a chave exata (preservando case)
                            for key in self.person_stats.keys():
                                if key.upper() == person_title.upper():
                                    person_key = key
                                    break
                
                # Se ainda não encontrou, tenta por título
                if person_key is None and 'title' in self.person_entities.columns:
                    matching_by_title = self.person_entities[
                        self.person_entities['title'].astype(str).str.strip().str.upper() == person_id_upper
                    ]
                    if not matching_by_title.empty:
                        person_title = str(matching_by_title.iloc[0]['title']).strip()
                        if person_title and person_title.upper() in [k.upper() for k in self.person_stats.keys()]:
                            # Encontra a chave exata (preservando case)
                            for key in self.person_stats.keys():
                                if key.upper() == person_title.upper():
                                    person_key = key
                                    break
            
            if person_key is None:
                available = list(self.person_stats.keys())[:10]
                raise ValueError(
                    f"Pessoa '{person_id}' não encontrada nas estatísticas.\n"
                    f"Pessoas disponíveis: {available}\n"
                    f"Total de pessoas: {len(self.person_stats)}"
                )

        stats = self.person_stats[person_key]
        all_relationships = stats.get('all_relationships', [])

        # Categoriza relacionamentos por tipo
        relationship_categories = {
            'consultas': [],
            'exames': [],
            'procedimentos': [],
            'diagnosticos': [],
            'tratamentos': [],
            'medicamentos': [],
            'hospitalizacoes': [],
            'localizacoes': [],
            'outros': []
        }

        # Palavras-chave para categorização
        keywords = {
            'consultas': ['consultation', 'consulta', 'atendimento', 'appointment', 'visit'],
            'exames': ['exam', 'exame', 'test', 'teste', 'ultrasound', 'ultrassom', 'endoscopy', 'endoscopia', 'lab', 'laboratório'],
            'procedimentos': ['procedure', 'procedimento', 'surgery', 'cirurgia', 'cholecystectomy', 'colecistectomia', 'splenectomy', 'esplenectomia'],
            'diagnosticos': ['diagnosis', 'diagnóstico', 'diagnosed', 'diagnosticado', 'disease', 'doença'],
            'tratamentos': ['treatment', 'tratamento', 'therapy', 'terapia'],
            'medicamentos': ['medication', 'medicamento', 'drug', 'medicina'],
            'hospitalizacoes': ['hospitalization', 'hospitalização', 'hospital', 'hospitalizado'],
            'localizacoes': ['resides', 'reside', 'native', 'nativo', 'location', 'localização', 'at', 'em']
        }

        for rel in all_relationships:
            description = rel.get('description', '').lower()
            categorized = False

            for category, category_keywords in keywords.items():
                if any(keyword in description for keyword in category_keywords):
                    relationship_categories[category].append(rel)
                    categorized = True
                    break

            if not categorized:
                relationship_categories['outros'].append(rel)

        # Analisa entidades conectadas
        connected_entities = set()
        for rel in all_relationships:
            if 'target' in rel:
                connected_entities.add(rel['target'])
            if 'source' in rel:
                connected_entities.add(rel['source'])

        # Tenta identificar entidades relacionadas no entities_df
        related_entities_info = []
        if self.entities_df is not None and 'id' in self.entities_df.columns:
            for entity_id in list(connected_entities)[:20]:  # Limita a 20 para performance
                entity_info = self.entities_df[self.entities_df['id'].astype(str) == str(entity_id)]
                if not entity_info.empty:
                    entity_row = entity_info.iloc[0]
                    related_entities_info.append({
                        'id': entity_id,
                        'title': entity_row.get('title', 'N/A'),
                        'type': entity_row.get('type', 'N/A'),
                        'description': entity_row.get('description', 'N/A')[:100] if pd.notna(entity_row.get('description')) else 'N/A'
                    })

        # Extrai padrões temporais (se houver datas nas descrições)
        date_patterns = []
        for rel in all_relationships:
            description = rel.get('description', '')
            # Procura por padrões de data (YYYY-MM-DD, DD/MM/YYYY, etc.)
            dates = re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}\.\d{2}\.\d{4}', description)
            if dates:
                date_patterns.extend(dates)

        # Conta tipos de relacionamento mais comuns
        description_counts = Counter([rel.get('description', '') for rel in all_relationships if rel.get('description')])
        top_descriptions = description_counts.most_common(10)

        return {
            'person_id': person_key,
            'person_info': {
                'title': stats.get('title', 'N/A'),
                'description': stats.get('description', 'N/A'),
                'total_relationships': stats['total_degree']
            },
            'relationship_categories': {
                category: {
                    'count': len(rels),
                    'relationships': rels[:5]  # Top 5 de cada categoria
                }
                for category, rels in relationship_categories.items()
                if len(rels) > 0
            },
            'top_relationship_types': [
                {'description': desc, 'count': count}
                for desc, count in top_descriptions
            ],
            'connected_entities': {
                'total': len(connected_entities),
                'sample': related_entities_info[:10]  # Top 10 entidades relacionadas
            },
            'temporal_patterns': {
                'dates_found': len(date_patterns),
                'sample_dates': list(set(date_patterns))[:10] if date_patterns else []
            },
            'clinical_summary': self._generate_clinical_summary(relationship_categories, stats)
        }

    def _generate_clinical_summary(self, categories: Dict, stats: Dict) -> str:
        """Gera um resumo clínico baseado nas categorias de relacionamento."""
        summary_parts = []

        if categories.get('consultas'):
            summary_parts.append(f"Realizou {len(categories['consultas'])} consultas médicas")
        
        if categories.get('exames'):
            summary_parts.append(f"Submeteu-se a {len(categories['exames'])} exames")
        
        if categories.get('procedimentos'):
            summary_parts.append(f"Passou por {len(categories['procedimentos'])} procedimentos")
        
        if categories.get('diagnosticos'):
            summary_parts.append(f"Recebeu {len(categories['diagnosticos'])} diagnósticos")
        
        if categories.get('hospitalizacoes'):
            summary_parts.append(f"Teve {len(categories['hospitalizacoes'])} hospitalizações")

        if summary_parts:
            return ". ".join(summary_parts) + "."
        else:
            return "Informações clínicas limitadas disponíveis."

    def generate_person_report(self, person_id: Optional[str] = None, output_path: str = "output/person_analysis_report.json") -> str:
        """
        Gera relatório completo de análise de pessoa.

        Args:
            person_id: ID da pessoa (se None, usa a pessoa com mais relacionamentos)
            output_path: Caminho para salvar o relatório

        Returns:
            Caminho do arquivo gerado
        """
        if self.relationships_df is None:
            self.load_data()

        # Se não especificou pessoa, pega a com mais relacionamentos
        if person_id is None:
            top_persons = self.get_top_persons(top_n=1)
            if len(top_persons) == 0:
                raise ValueError("Nenhuma pessoa encontrada para análise")
            # Usa person_name se disponível, senão usa person_id
            if 'person_name' in top_persons.columns:
                person_id = top_persons.iloc[0]['person_name']
            else:
                person_id = top_persons.iloc[0]['person_id']
            print(f"\nAnalisando pessoa com mais relacionamentos: {person_id}")

        print(f"\nAnalisando evolução clínica da pessoa {person_id}...")
        clinical_analysis = self.analyze_clinical_evolution(person_id)

        # Compila relatório
        report = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'person_analysis': clinical_analysis,
            'summary': {
                'total_relationships': clinical_analysis['person_info']['total_relationships'],
                'categories_identified': len([c for c in clinical_analysis['relationship_categories'].values() if c['count'] > 0]),
                'connected_entities': clinical_analysis['connected_entities']['total']
            }
        }

        # Salva relatório JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ Relatório salvo em: {output_file.absolute()}")

        return str(output_file.absolute())

