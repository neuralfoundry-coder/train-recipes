#!/bin/bash
# ==============================================================================
# Qwen3-14B Training Script
# Fine-tuning with Reasoning + Conversational datasets
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

# ==============================================================================
# Load environment variables from env_local
# ==============================================================================
load_env() {
    if [ -f "env_local" ]; then
        # Export all variables from env_local
        set -a
        source env_local
        set +a
        return 0
    else
        echo -e "${YELLOW}Warning: env_local not found${NC}"
        return 1
    fi
}

# Load environment
load_env

# ==============================================================================
# Default values (from env_local or fallback)
# ==============================================================================
GPU_IDS="${TRAIN_GPU_IDS:-0,1}"
CONDA_ENV=""
BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
LEARNING_RATE="${TRAIN_LEARNING_RATE:-2e-4}"
MAX_STEPS="${TRAIN_MAX_STEPS:-30}"
LORA_RANK="${LORA_R:-32}"

# ==============================================================================
# Functions
# ==============================================================================

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          Qwen3-14B Reasoning + Conversational Training       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    echo -e "${GREEN}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${GREEN}Options:${NC}"
    echo "  -g, --gpu IDS       GPU device IDs (default: $GPU_IDS from env_local)"
    echo "  -b, --batch N       Batch size (default: $BATCH_SIZE)"
    echo "  -r, --lr RATE       Learning rate (default: $LEARNING_RATE)"
    echo "  -s, --steps N       Max training steps (default: $MAX_STEPS)"
    echo "  --lora-r N          LoRA rank (default: $LORA_RANK)"
    echo "  -e, --env NAME      Conda environment name to activate"
    echo "  -l, --logs          List recent training logs"
    echo "  -c, --clean         Clean old log directories (keep last 5)"
    echo "  -v, --vars          Show current configuration from env_local"
    echo "  -h, --help          Show this help message"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  $0                  # Train with env_local settings"
    echo "  $0 -g 0             # Train on GPU 0 only"
    echo "  $0 -g 2,3 -b 4      # Train on GPU 2,3 with batch size 4"
    echo "  $0 -s 100 -r 1e-4   # Train 100 steps with lr=1e-4"
    echo "  $0 -v               # Show current configuration"
    echo ""
}

show_vars() {
    echo -e "${BLUE}📋 Current Configuration (from env_local):${NC}"
    echo "============================================"
    echo ""
    echo -e "${GREEN}[Model]${NC}"
    echo "  MODEL_NAME:       ${MODEL_NAME:-not set}"
    echo "  MODEL_SHORT_NAME: ${MODEL_SHORT_NAME:-not set}"
    echo "  MAX_SEQ_LENGTH:   ${MAX_SEQ_LENGTH:-not set}"
    echo "  LOAD_IN_4BIT:     ${LOAD_IN_4BIT:-not set}"
    echo ""
    echo -e "${GREEN}[LoRA]${NC}"
    echo "  LORA_R:           ${LORA_R:-not set}"
    echo "  LORA_ALPHA:       ${LORA_ALPHA:-not set}"
    echo "  LORA_DROPOUT:     ${LORA_DROPOUT:-not set}"
    echo ""
    echo -e "${GREEN}[Training]${NC}"
    echo "  TRAIN_GPU_IDS:    ${TRAIN_GPU_IDS:-not set}"
    echo "  TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE:-not set}"
    echo "  TRAIN_LEARNING_RATE: ${TRAIN_LEARNING_RATE:-not set}"
    echo "  TRAIN_MAX_STEPS:  ${TRAIN_MAX_STEPS:-not set}"
    echo "  TRAIN_CHAT_PERCENTAGE: ${TRAIN_CHAT_PERCENTAGE:-not set}"
    echo "  TRAIN_EVAL_RATIO: ${TRAIN_EVAL_RATIO:-not set}"
    echo ""
    echo -e "${GREEN}[Inference - Normal]${NC}"
    echo "  INFER_TEMPERATURE: ${INFER_TEMPERATURE:-not set}"
    echo "  INFER_TOP_P:      ${INFER_TOP_P:-not set}"
    echo "  INFER_TOP_K:      ${INFER_TOP_K:-not set}"
    echo "  INFER_MAX_TOKENS: ${INFER_MAX_TOKENS:-not set}"
    echo ""
    echo -e "${GREEN}[Inference - Thinking]${NC}"
    echo "  INFER_THINKING_TEMPERATURE: ${INFER_THINKING_TEMPERATURE:-not set}"
    echo "  INFER_THINKING_TOP_P: ${INFER_THINKING_TOP_P:-not set}"
    echo "  INFER_THINKING_MAX_TOKENS: ${INFER_THINKING_MAX_TOKENS:-not set}"
    echo ""
    echo -e "${GREEN}[API Keys]${NC}"
    echo "  HF_TOKEN:         $([ -n "$HF_TOKEN" ] && echo "✓ set" || echo "✗ not set")"
    echo "  WNB_API_KEY:      $([ -n "$WNB_API_KEY" ] && echo "✓ set" || echo "✗ not set")"
    echo ""
    echo "============================================"
    echo ""
}

list_logs() {
    echo -e "${BLUE}📁 Recent Training Logs:${NC}"
    echo "----------------------------------------"
    
    if [ -d "logs" ]; then
        ls -lt logs/ 2>/dev/null | head -10 | while read line; do
            echo "  $line"
        done
        
        echo ""
        echo -e "${YELLOW}Total log directories:${NC} $(ls -d logs/*/ 2>/dev/null | wc -l)"
    else
        echo -e "${YELLOW}  No logs directory found.${NC}"
    fi
    echo ""
}

clean_logs() {
    echo -e "${BLUE}🧹 Cleaning old log directories...${NC}"
    
    if [ -d "logs" ]; then
        total=$(ls -d logs/*/ 2>/dev/null | wc -l)
        
        if [ "$total" -le 5 ]; then
            echo -e "${GREEN}  Only $total directories found. Nothing to clean.${NC}"
        else
            to_remove=$((total - 5))
            echo -e "${YELLOW}  Removing $to_remove old directories...${NC}"
            
            ls -dt logs/*/ 2>/dev/null | tail -n $to_remove | while read dir; do
                echo "    Removing: $dir"
                rm -rf "$dir"
            done
            
            echo -e "${GREEN}  Done! Kept last 5 directories.${NC}"
        fi
    else
        echo -e "${YELLOW}  No logs directory found.${NC}"
    fi
    echo ""
}

check_env() {
    echo -e "${BLUE}🔍 Checking environment...${NC}"
    
    # Check Python
    if command -v python &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} Python: $(python --version 2>&1)"
    else
        echo -e "  ${RED}✗${NC} Python not found"
        exit 1
    fi
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo -e "  ${GREEN}✓${NC} CUDA available: $gpu_count GPU(s)"
    else
        echo -e "  ${RED}✗${NC} nvidia-smi not found"
        exit 1
    fi
    
    # Check required packages
    python -c "import unsloth" 2>/dev/null && echo -e "  ${GREEN}✓${NC} unsloth" || echo -e "  ${RED}✗${NC} unsloth not installed"
    python -c "import torch" 2>/dev/null && echo -e "  ${GREEN}✓${NC} torch" || echo -e "  ${RED}✗${NC} torch not installed"
    python -c "import wandb" 2>/dev/null && echo -e "  ${GREEN}✓${NC} wandb" || echo -e "  ${YELLOW}!${NC} wandb not installed (optional)"
    
    # Check env_local
    if [ -f "env_local" ]; then
        echo -e "  ${GREEN}✓${NC} env_local found"
    else
        echo -e "  ${YELLOW}!${NC} env_local not found"
    fi
    
    echo ""
}

run_training() {
    echo -e "${BLUE}🚀 Starting training...${NC}"
    echo ""
    echo -e "${GREEN}[Training Parameters]${NC}"
    echo "  GPU IDs:       $GPU_IDS"
    echo "  Batch size:    $BATCH_SIZE"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Max steps:     $MAX_STEPS"
    echo "  LoRA rank:     $LORA_RANK"
    echo ""
    
    # Export overrides as environment variables
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    export TRAIN_BATCH_SIZE="$BATCH_SIZE"
    export TRAIN_LEARNING_RATE="$LEARNING_RATE"
    export TRAIN_MAX_STEPS="$MAX_STEPS"
    export LORA_R="$LORA_RANK"
    
    # Run training
    python qwen3_\(14b\)_reasoning_conversational.py
    
    echo ""
    echo -e "${GREEN}✅ Training completed!${NC}"
    echo ""
    
    # Show latest log
    if [ -d "logs" ]; then
        latest=$(ls -dt logs/*/ 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo -e "${BLUE}📁 Latest log directory:${NC}"
            echo "  $latest"
            echo ""
            echo -e "${BLUE}📄 Log files:${NC}"
            ls -la "$latest" 2>/dev/null | grep -v "^total" | while read line; do
                echo "  $line"
            done
        fi
    fi
}

# ==============================================================================
# Main
# ==============================================================================

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu)
            GPU_IDS="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -r|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -s|--steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --lora-r)
            LORA_RANK="$2"
            shift 2
            ;;
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -l|--logs)
            print_banner
            list_logs
            exit 0
            ;;
        -c|--clean)
            print_banner
            clean_logs
            exit 0
            ;;
        -v|--vars)
            print_banner
            show_vars
            exit 0
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

# Print banner
print_banner

# Activate conda environment if specified
if [ -n "$CONDA_ENV" ]; then
    echo -e "${BLUE}🐍 Activating conda environment: $CONDA_ENV${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo ""
fi

# Check environment
check_env

# Confirm before training
echo -e "${YELLOW}Press Enter to start training, or Ctrl+C to cancel...${NC}"
read

# Run training
run_training
