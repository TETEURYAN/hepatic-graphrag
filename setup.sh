#!/bin/bash

# Setup simples para inicializar uv e criar ambiente Python virtual (.venv)
set -e

#=============================================================================
# FUNÇÕES DE LOG
#=============================================================================
print_status() { echo "[INFO] $*"; }
print_success() { echo "[SUCCESS] $*"; }
print_error() { echo "[ERROR] $*" >&2; }

#=============================================================================
# VERIFICA UV
#=============================================================================
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

install_uv() {
    if command_exists uv; then
        print_status "uv já está instalado"
    else
        print_status "Instalando uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        print_success "uv instalado"
    fi
}

#=============================================================================
# CRIA AMBIENTE VIRTUAL
#=============================================================================
setup_virtualenv() {
    print_status "Criando ambiente virtual Python (.venv)..."
    uv venv
    # Ativa ambiente virtual
    # shellcheck source=/dev/null
    source .venv/bin/activate
    print_success "Ambiente virtual ativado"
}

#=============================================================================
# INSTALA DEPENDÊNCIAS
#=============================================================================
install_dependencies() {
    print_status "Instalando dependências do projeto..."
    uv pip install -e . --link-mode=copy
    print_success "Dependências instaladas"
}

#=============================================================================
# EXECUÇÃO
#=============================================================================
main() {
    install_uv
    setup_virtualenv
    install_dependencies
    print_success "Setup concluído! Use 'source .venv/bin/activate' para ativar"
}

main "$@"
