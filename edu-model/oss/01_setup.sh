#!/bin/bash
# ==============================================================================
# GPT-OSS 20B Environment Setup Script
# Sets up uv virtual environment with all required packages
# ==============================================================================

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VENV_DIR=".venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       GPT-OSS 20B Korean Education Environment Setup        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    echo -e "${GREEN}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${GREEN}Options:${NC}"
    echo "  -p, --python VER    Python version (default: $PYTHON_VERSION)"
    echo "  -c, --clean         Remove existing venv and reinstall"
    echo "  -u, --upgrade       Upgrade all packages"
    echo "  -h, --help          Show this help message"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  $0                  # Setup with default Python version"
    echo "  $0 -p 3.12          # Setup with Python 3.12"
    echo "  $0 -c               # Clean install"
    echo "  $0 -u               # Upgrade packages"
    echo ""
}

check_uv() {
    echo -e "${BLUE}🔍 Checking uv installation...${NC}"
    
    if command -v uv &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} uv: $(uv --version)"
        return 0
    else
        echo -e "  ${YELLOW}!${NC} uv not found, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        if command -v uv &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} uv installed: $(uv --version)"
            return 0
        else
            echo -e "  ${RED}✗${NC} uv installation failed"
            echo "  Please install uv manually: https://docs.astral.sh/uv/"
            exit 1
        fi
    fi
}

create_venv() {
    echo -e "${BLUE}📦 Creating virtual environment...${NC}"
    
    if [ -d "$VENV_DIR" ] && [ "$CLEAN_INSTALL" != "1" ]; then
        echo -e "  ${GREEN}✓${NC} Virtual environment already exists"
        return 0
    fi
    
    if [ "$CLEAN_INSTALL" = "1" ] && [ -d "$VENV_DIR" ]; then
        echo -e "  ${YELLOW}!${NC} Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    echo -e "  Creating venv with Python $PYTHON_VERSION..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    echo -e "  ${GREEN}✓${NC} Virtual environment created"
}

install_packages() {
    echo -e "${BLUE}📥 Installing packages...${NC}"
    
    # Activate venv for uv
    export VIRTUAL_ENV="$SCRIPT_DIR/$VENV_DIR"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    
    if [ "$UPGRADE_PACKAGES" = "1" ]; then
        echo -e "  Upgrading all packages..."
        uv pip install --upgrade -r requirements.txt
    else
        echo -e "  Installing from requirements.txt..."
        uv pip install -r requirements.txt
    fi
    
    # Install unsloth from git for latest GPT-OSS support
    echo -e "  Installing unsloth (latest version for GPT-OSS)..."
    uv pip install --upgrade --no-deps "unsloth[base] @ git+https://github.com/unslothai/unsloth"
    uv pip install --upgrade --no-deps "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
    
    echo -e "  ${GREEN}✓${NC} Packages installed"
}

verify_installation() {
    echo -e "${BLUE}✅ Verifying installation...${NC}"
    
    # Use the venv python
    PYTHON="$SCRIPT_DIR/$VENV_DIR/bin/python"
    
    $PYTHON -c "import torch; print(f'  ✓ torch: {torch.__version__}')" 2>/dev/null || echo -e "  ${RED}✗${NC} torch"
    $PYTHON -c "import transformers; print(f'  ✓ transformers: {transformers.__version__}')" 2>/dev/null || echo -e "  ${RED}✗${NC} transformers"
    $PYTHON -c "import unsloth; print('  ✓ unsloth')" 2>/dev/null || echo -e "  ${RED}✗${NC} unsloth"
    $PYTHON -c "import wandb; print(f'  ✓ wandb: {wandb.__version__}')" 2>/dev/null || echo -e "  ${RED}✗${NC} wandb"
    $PYTHON -c "import trl; print(f'  ✓ trl: {trl.__version__}')" 2>/dev/null || echo -e "  ${RED}✗${NC} trl"
    $PYTHON -c "import datasets; print(f'  ✓ datasets: {datasets.__version__}')" 2>/dev/null || echo -e "  ${RED}✗${NC} datasets"
    
    # Check CUDA
    $PYTHON -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
    
    echo ""
}

print_activation_instructions() {
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Setup complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "To activate the environment, run:"
    echo -e "  ${YELLOW}source $VENV_DIR/bin/activate${NC}"
    echo ""
    echo -e "Or use the scripts directly (they auto-activate):"
    echo -e "  ${YELLOW}./02_train.sh${NC}    # Start training"
    echo -e "  ${YELLOW}./03_chat.sh${NC}     # Start chat console"
    echo -e "  ${YELLOW}./04_compare.sh${NC}  # Run model comparison"
    echo ""
    echo -e "${BLUE}Dataset:${NC} neuralfoundry-coder/aihub-korean-education-instruct"
    echo -e "${BLUE}GPU:${NC} CUDA_VISIBLE_DEVICES=1 (single GPU)"
    echo ""
}

# ==============================================================================
# Main
# ==============================================================================

CLEAN_INSTALL=""
UPGRADE_PACKAGES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_INSTALL="1"
            shift
            ;;
        -u|--upgrade)
            UPGRADE_PACKAGES="1"
            shift
            ;;
        -h|--help)
            print_banner
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

print_banner

# Run setup steps
check_uv
create_venv
install_packages
verify_installation
print_activation_instructions

