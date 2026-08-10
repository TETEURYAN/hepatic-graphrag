import litelm_patch  # aplica o hotfix antes de carregar o GraphRAG
import graphrag_json_patch  # remove response_format=json_object forçado (global/drift search)
import sys
from graphrag.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
