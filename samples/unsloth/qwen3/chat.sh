#!/bin/bash
# ==============================================================================
# Qwen3-14B Chat Console Script
# Interactive testing with fine-tuned model
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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ==============================================================================
# Load environment variables from env_local
# ==============================================================================
load_env() {
    if [ -f "env_local" ]; then
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
GPU_ID="${INFER_GPU_ID:-0}"
MODEL_PATH=""
MAX_TOKENS="${INFER_MAX_TOKENS:-1024}"
THINKING_MAX_TOKENS="${INFER_THINKING_MAX_TOKENS:-2048}"
TEMPERATURE="${INFER_TEMPERATURE:-0.7}"
TOP_P="${INFER_TOP_P:-0.8}"
TOP_K="${INFER_TOP_K:-20}"
THINKING_TEMPERATURE="${INFER_THINKING_TEMPERATURE:-0.6}"
THINKING_TOP_P="${INFER_THINKING_TOP_P:-0.95}"
THINKING_TOP_K="${INFER_THINKING_TOP_K:-20}"
THINKING_MODE=""
CONDA_ENV=""

# ==============================================================================
# Functions
# ==============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║            Qwen3-14B Interactive Chat Console                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    echo -e "${GREEN}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${GREEN}Options:${NC}"
    echo "  -m, --model PATH    Path to fine-tuned LoRA model"
    echo "  -g, --gpu ID        GPU device ID (default: $GPU_ID from env_local)"
    echo "  -t, --tokens N      Max new tokens (default: $MAX_TOKENS)"
    echo "  -T, --thinking      Start with thinking mode enabled"
    echo "  -e, --env NAME      Conda environment name to activate"
    echo "  -l, --list          List available trained models"
    echo "  -v, --vars          Show inference configuration from env_local"
    echo "  -h, --help          Show this help message"
    echo ""
    echo -e "${GREEN}Inference Settings (from env_local):${NC}"
    echo "  Normal mode:   temp=$TEMPERATURE, top_p=$TOP_P, top_k=$TOP_K"
    echo "  Thinking mode: temp=$THINKING_TEMPERATURE, top_p=$THINKING_TOP_P, top_k=$THINKING_TOP_K"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  $0                  # Auto-detect latest model"
    echo "  $0 -l               # List available models"
    echo "  $0 -m ./logs/qwen3-14b_.../lora_model"
    echo "  $0 -g 1 -T          # Use GPU 1 with thinking mode"
    echo "  $0 -t 2048          # Set max tokens to 2048"
    echo ""
    echo -e "${GREEN}In-console Commands:${NC}"
    echo "  /help      - Show available commands"
    echo "  /thinking  - Toggle thinking mode"
    echo "  /clear     - Clear conversation history"
    echo "  /exit      - Exit the console"
    echo ""
}

show_vars() {
    echo -e "${BLUE}📋 Inference Configuration (from env_local):${NC}"
    echo "============================================"
    echo ""
    echo -e "${GREEN}[Model]${NC}"
    echo "  MODEL_NAME:       ${MODEL_NAME:-not set}"
    echo "  MODEL_SHORT_NAME: ${MODEL_SHORT_NAME:-not set}"
    echo "  MAX_SEQ_LENGTH:   ${MAX_SEQ_LENGTH:-not set}"
    echo ""
    echo -e "${GREEN}[Inference - Normal Mode]${NC}"
    echo "  INFER_GPU_ID:     ${INFER_GPU_ID:-not set}"
    echo "  INFER_TEMPERATURE: ${INFER_TEMPERATURE:-not set}"
    echo "  INFER_TOP_P:      ${INFER_TOP_P:-not set}"
    echo "  INFER_TOP_K:      ${INFER_TOP_K:-not set}"
    echo "  INFER_MAX_TOKENS: ${INFER_MAX_TOKENS:-not set}"
    echo ""
    echo -e "${GREEN}[Inference - Thinking Mode]${NC}"
    echo "  INFER_THINKING_TEMPERATURE: ${INFER_THINKING_TEMPERATURE:-not set}"
    echo "  INFER_THINKING_TOP_P: ${INFER_THINKING_TOP_P:-not set}"
    echo "  INFER_THINKING_TOP_K: ${INFER_THINKING_TOP_K:-not set}"
    echo "  INFER_THINKING_MAX_TOKENS: ${INFER_THINKING_MAX_TOKENS:-not set}"
    echo ""
    echo -e "${GREEN}[API Keys]${NC}"
    echo "  HF_TOKEN:         $([ -n "$HF_TOKEN" ] && echo "✓ set" || echo "✗ not set")"
    echo ""
    echo "============================================"
    echo ""
}

list_models() {
    echo -e "${BLUE}📁 Available Trained Models:${NC}"
    echo "============================================"
    
    if [ -d "logs" ]; then
        count=0
        for dir in $(ls -dt logs/*/ 2>/dev/null); do
            model_path="$dir/lora_model"
            if [ -d "$model_path" ]; then
                count=$((count + 1))
                dir_name=$(basename "$dir")
                mod_time=$(stat -c %y "$model_path" 2>/dev/null | cut -d'.' -f1)
                
                if [ -f "$model_path/adapter_config.json" ]; then
                    status="${GREEN}✓${NC}"
                else
                    status="${YELLOW}?${NC}"
                fi
                
                echo -e "  [$count] $status $dir_name"
                echo -e "      Path: $model_path"
                echo -e "      Modified: $mod_time"
                echo ""
            fi
        done
        
        if [ $count -eq 0 ]; then
            echo -e "${YELLOW}  No trained models found in logs/.${NC}"
            echo -e "  Run ${GREEN}./train.sh${NC} to train a model first."
        else
            echo "============================================"
            echo -e "${GREEN}Total:${NC} $count model(s) found"
            echo ""
            echo -e "Use ${GREEN}$0 -m <path>${NC} to load a specific model"
        fi
    else
        echo -e "${YELLOW}  No logs directory found.${NC}"
        echo -e "  Run ${GREEN}./train.sh${NC} to train a model first."
    fi
    echo ""
}

find_latest_model() {
    if [ -d "logs" ]; then
        for dir in $(ls -dt logs/*/ 2>/dev/null); do
            model_path="$dir/lora_model"
            if [ -d "$model_path" ] && [ -f "$model_path/adapter_config.json" ]; then
                echo "$model_path"
                return 0
            fi
        done
    fi
    
    if [ -d "lora_model" ]; then
        echo "lora_model"
        return 0
    fi
    
    return 1
}

check_model() {
    local path="$1"
    
    if [ -d "$path" ]; then
        if [ -f "$path/adapter_config.json" ]; then
            return 0
        fi
    fi
    
    return 1
}

run_chat() {
    echo -e "${BLUE}🚀 Starting chat console...${NC}"
    echo ""
    
    # Build command with env_local values
    CMD="python chat_console.py --gpu $GPU_ID --max-tokens $MAX_TOKENS"
    
    if [ -n "$MODEL_PATH" ]; then
        CMD="$CMD --model-path $MODEL_PATH"
    fi
    
    if [ -n "$THINKING_MODE" ]; then
        CMD="$CMD --thinking"
    fi
    
    # Export inference config for Python script
    export INFER_TEMPERATURE="$TEMPERATURE"
    export INFER_TOP_P="$TOP_P"
    export INFER_TOP_K="$TOP_K"
    export INFER_THINKING_TEMPERATURE="$THINKING_TEMPERATURE"
    export INFER_THINKING_TOP_P="$THINKING_TOP_P"
    export INFER_THINKING_TOP_K="$THINKING_TOP_K"
    export INFER_MAX_TOKENS="$MAX_TOKENS"
    export INFER_THINKING_MAX_TOKENS="$THINKING_MAX_TOKENS"
    
    echo -e "${YELLOW}Command:${NC} $CMD"
    echo ""
    
    eval $CMD
}

# ==============================================================================
# Main
# ==============================================================================

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -t|--tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -T|--thinking)
            THINKING_MODE="1"
            shift
            ;;
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -l|--list)
            print_banner
            list_models
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

# Auto-detect model if not specified
if [ -z "$MODEL_PATH" ]; then
    echo -e "${BLUE}🔍 Auto-detecting model...${NC}"
    MODEL_PATH=$(find_latest_model)
    
    if [ -n "$MODEL_PATH" ]; then
        echo -e "  ${GREEN}✓${NC} Found: $MODEL_PATH"
    else
        echo -e "  ${RED}✗${NC} No trained model found"
        echo ""
        echo -e "Please train a model first: ${GREEN}./train.sh${NC}"
        echo -e "Or specify a model path: ${GREEN}$0 -m <path>${NC}"
        exit 1
    fi
    echo ""
else
    if ! check_model "$MODEL_PATH"; then
        echo -e "${RED}✗ Invalid model path: $MODEL_PATH${NC}"
        echo "  Make sure the path contains adapter_config.json"
        exit 1
    fi
fi

# Show settings
echo -e "${BLUE}⚙️  Settings:${NC}"
echo "  Model: $MODEL_PATH"
echo "  GPU: $GPU_ID"
echo "  Max tokens: $MAX_TOKENS (thinking: $THINKING_MAX_TOKENS)"
echo "  Thinking mode: $([ -n "$THINKING_MODE" ] && echo "ON" || echo "OFF")"
echo ""
echo -e "${BLUE}📊 Generation Config:${NC}"
echo "  Normal:   temp=$TEMPERATURE, top_p=$TOP_P, top_k=$TOP_K"
echo "  Thinking: temp=$THINKING_TEMPERATURE, top_p=$THINKING_TOP_P, top_k=$THINKING_TOP_K"
echo ""

# Run chat
run_chat
