.PHONY: install setup clean

# Cria e ativa o ambiente virtual com uv
setup:
	uv venv
	uv pip install -r pyproject.toml

# Instala dependências (se já existir o ambiente)
install:
	uv pip install -r pyproject.toml

# Remove o ambiente virtual e arquivos temporários
clean:
	rm -rf .venv __pycache__ */__pycache__ .pytest_cache .mypy_cache
