"""
Script para consultar LLM via OpenRouter usando prompts e dados JSON.
Processa prompts da pasta prompts/insight para as 6 primeiras PERSONS e salva respostas em prompts/insight/output.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv não está instalado. Instale com: pip install python-dotenv"
    )

try:
    import requests
except ImportError:
    raise ImportError(
        "requests não está instalado. Instale com: pip install requests"
    )

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas não está instalado. Instale com: pip install pandas"
    )

# Adiciona o caminho do src ao sys.path para importar PersonAnalyzer
sys.path.insert(0, str(Path(__file__).parent / "src" / "core" / "analysis"))

try:
    from models.person_analyzer import PersonAnalyzer
except ImportError:
    raise ImportError(
        "PersonAnalyzer não encontrado. Verifique se o caminho está correto."
    )


def load_environment() -> str:
    """Carrega variáveis de ambiente do arquivo .env e retorna a API key."""
    load_dotenv()
    api_key = os.getenv("GRAPHRAG_API_KEY")
    if not api_key:
        raise ValueError(
            "GRAPHRAG_API_KEY não encontrada no arquivo .env. "
            "Por favor, adicione GRAPHRAG_API_KEY=sua_chave_aqui no arquivo .env"
        )
    return api_key


def get_top_persons_data(
    relationships_path: str,
    entities_path: str,
    top_n: int = 6
) -> List[Dict[str, Any]]:
    """
    Obtém os dados das top N persons usando PersonAnalyzer.
    
    Args:
        relationships_path: Caminho para o arquivo relationships.parquet
        entities_path: Caminho para o arquivo entities.parquet
        top_n: Número de persons a retornar (padrão: 6)
    
    Returns:
        Lista de dicionários com dados de cada person
    """
    analyzer = PersonAnalyzer(relationships_path, entities_path)
    analyzer.load_data()
    
    top_persons_df = analyzer.get_top_persons(top_n=top_n)
    
    if len(top_persons_df) == 0:
        raise ValueError("Nenhuma pessoa encontrada para análise")
    
    persons_data = []
    for idx, row in top_persons_df.iterrows():
        person_id = row.get('person_name', row.get('person_id', ''))
        if not person_id:
            continue
        
        # Gera análise clínica para cada person
        try:
            clinical_analysis = analyzer.analyze_clinical_evolution(person_id)
            
            # Compila relatório no mesmo formato do JSON original
            person_report = {
                "analysis_date": pd.Timestamp.now().isoformat(),
                "person_analysis": clinical_analysis,
                "summary": {
                    "total_relationships": clinical_analysis['person_info']['total_relationships'],
                    "categories_identified": len([c for c in clinical_analysis['relationship_categories'].values() if c['count'] > 0]),
                    "connected_entities": clinical_analysis['connected_entities']['total']
                }
            }
            
            persons_data.append({
                "person_id": person_id,
                "person_name": row.get('title', person_id),
                "data": person_report
            })
        except Exception as e:
            print(f"  ⚠ Erro ao analisar {person_id}: {e}")
            continue
    
    return persons_data


def load_prompts(prompts_dir: str) -> Dict[str, str]:
    """Carrega todos os prompts da pasta especificada."""
    prompts_path = Path(prompts_dir)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Diretório de prompts não encontrado: {prompts_dir}")
    
    prompts = {}
    for prompt_file in prompts_path.glob("prompt_*.txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts[prompt_file.stem] = f.read()
    
    if not prompts:
        raise FileNotFoundError(f"Nenhum prompt encontrado em {prompts_dir}")
    
    return prompts


def call_openrouter_api(
    api_key: str,
    prompt: str,
    json_data: Dict[str, Any],
    model: str = "mistralai/devstral-2512:free"
) -> str:
    """
    Faz chamada à API do OpenRouter.
    
    Args:
        api_key: Chave de API do OpenRouter
        prompt: Texto do prompt
        json_data: Dados JSON para incluir no contexto
        model: Modelo a ser usado 
    
    Returns:
        Resposta da LLM como string
    """
    url = ""
    
    # Prepara o conteúdo da mensagem combinando prompt e dados JSON
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
    user_message = f"{prompt}\n\n## Dados JSON:\n\n```json\n{json_str}\n```"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com",
        "X-Title": "Anamnese Analysis"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        # Verifica o status code antes de tentar parsear JSON
        if response.status_code != 200:
            error_msg = f"Status code {response.status_code}: {response.text}"
            raise RuntimeError(f"Erro na API do OpenRouter - {error_msg}")
        
        # Tenta parsear o JSON
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Resposta da API não é um JSON válido. "
                f"Status: {response.status_code}, "
                f"Conteúdo: {response.text[:500]}"
            )
        
        # Extrai a resposta do conteúdo
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Resposta inesperada da API: {result}")
    
    except requests.exceptions.Timeout:
        raise RuntimeError("Timeout ao chamar API do OpenRouter (120s)")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro ao chamar API do OpenRouter: {e}")


def save_response(output_dir: str, prompt_name: str, response: str, person_id: str = None):
    """
    Salva a resposta em um arquivo .txt na pasta de output.
    
    Args:
        output_dir: Diretório de saída
        prompt_name: Nome do prompt
        response: Resposta da LLM
        person_id: ID da person (opcional, para incluir no nome do arquivo)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if person_id:
        # Sanitiza o person_id para usar como nome de arquivo
        safe_person_id = person_id.replace(" ", "_").replace("/", "_")
        output_file = output_path / f"{prompt_name}_{safe_person_id}_response.txt"
    else:
        output_file = output_path / f"{prompt_name}_response.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)
    
    print(f"✓ Resposta salva em: {output_file}")


def main():
    """Função principal que orquestra o processo."""
    # Configurações
    RELATIONSHIPS_PATH = "datatest/output/relationships.parquet"
    ENTITIES_PATH = "datatest/output/entities.parquet"
    PROMPTS_DIR = "prompts/insight"
    OUTPUT_DIR = "prompts/insight/output"
    MODEL = "mistralai/devstral-2512:free"  # Pode ser alterado conforme necessário
    TOP_N_PERSONS = 6
    
    print("=" * 60)
    print("Script de Análise com LLM via ")
    print(f"Processando as {TOP_N_PERSONS} primeiras PERSONS")
    print("=" * 60)
    
    try:
        # 1. Carregar variáveis de ambiente
        print("\n[1/6] Carregando variáveis de ambiente...")
        api_key = load_environment()
        print("✓ Variáveis de ambiente carregadas")
        
        # 2. Obter dados das top N persons
        print(f"\n[2/6] Obtendo dados das {TOP_N_PERSONS} primeiras PERSONS...")
        print(f"  Relationships: {RELATIONSHIPS_PATH}")
        print(f"  Entities: {ENTITIES_PATH}")
        
        persons_data = get_top_persons_data(
            RELATIONSHIPS_PATH,
            ENTITIES_PATH,
            top_n=TOP_N_PERSONS
        )
        
        if not persons_data:
            raise ValueError("Nenhuma person encontrada para processar")
        
        print(f"✓ {len(persons_data)} person(s) encontrada(s):")
        for i, person in enumerate(persons_data, 1):
            print(f"  {i}. {person['person_name']} (ID: {person['person_id']})")
        
        # 3. Carregar prompts
        print(f"\n[3/6] Carregando prompts de {PROMPTS_DIR}...")
        prompts = load_prompts(PROMPTS_DIR)
        print(f"✓ {len(prompts)} prompt(s) carregado(s): {', '.join(prompts.keys())}")
        
        # 4. Processar cada person com cada prompt
        print(f"\n[4/6] Processando {len(persons_data)} person(s) com {len(prompts)} prompt(s)...")
        total_processed = 0
        total_errors = 0
        
        for person_idx, person_info in enumerate(persons_data, 1):
            person_id = person_info['person_id']
            person_name = person_info['person_name']
            person_data = person_info['data']
            
            print(f"\n  [{person_idx}/{len(persons_data)}] Processando {person_name} ({person_id})...")
            
            for prompt_name, prompt_text in prompts.items():
                try:
                    print(f"    → Processando {prompt_name}...")
                    response = call_openrouter_api(api_key, prompt_text, person_data, MODEL)
                    save_response(OUTPUT_DIR, prompt_name, response, person_id)
                    total_processed += 1
                    print(f"    ✓ {prompt_name} processado com sucesso")
                except Exception as e:
                    total_errors += 1
                    print(f"    ✗ Erro ao processar {prompt_name}: {e}")
                    continue
        
        # 5. Resumo final
        print("\n" + "=" * 60)
        print("[5/6] Processamento concluído!")
        print("=" * 60)
        print(f"✓ Total processado: {total_processed} combinações (person × prompt)")
        if total_errors > 0:
            print(f" Erros encontrados: {total_errors}")
        print(f"Respostas salvas em: {OUTPUT_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

