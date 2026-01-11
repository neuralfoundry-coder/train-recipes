#!/bin/bash
# ==============================================================================
# AI Trustworthiness Evaluation Script
# Evaluate fine-tuned model on trustworthiness benchmark
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
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

VENV_DIR=".venv"

# ==============================================================================
# Load environment variables from env_local
# ==============================================================================
if [ -f "env_local" ]; then
    echo -e "${BLUE}[CONFIG]${NC} Loading configuration from env_local..."
    set -a
    source env_local
    set +a
fi

# ==============================================================================
# Default Values (from env_local or hardcoded)
# ==============================================================================
GPU_ID="${EVAL_GPU_ID:-${INFER_GPU_ID:-0}}"
LORA_PATH="${EVAL_LORA_PATH:-}"
SAMPLE_COUNT="${EVAL_SAMPLE_COUNT:-100}"
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
TOP_P="${EVAL_TOP_P:-0.9}"
TOP_K="${EVAL_TOP_K:-50}"
SEED="${EVAL_SEED:-42}"
OUTPUT_DIR=""
SKIP_VENV="false"
OPEN_REPORT="false"

# ==============================================================================
# Usage Information
# ==============================================================================
show_usage() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║      🔬 AI TRUSTWORTHINESS EVALUATION SCRIPT                         ║"
    echo "║      Korean LLM Trustworthiness Benchmark                            ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "${GREEN}Usage:${NC}"
    echo "  $0 [OPTIONS]"
    echo ""
    echo -e "${GREEN}Required:${NC}"
    echo "  -l, --lora-path PATH     Path to the fine-tuned LoRA model directory"
    echo ""
    echo -e "${GREEN}Options:${NC}"
    echo "  -g, --gpu ID             GPU ID to use (default: $GPU_ID)"
    echo "  -n, --samples N          Number of samples to evaluate (default: $SAMPLE_COUNT)"
    echo "  -t, --tokens N           Max new tokens to generate (default: $MAX_NEW_TOKENS)"
    echo "  --temperature FLOAT      Sampling temperature (default: $TEMPERATURE)"
    echo "  --top-p FLOAT            Top-p sampling (default: $TOP_P)"
    echo "  --top-k INT              Top-k sampling (default: $TOP_K)"
    echo "  --seed INT               Random seed (default: $SEED)"
    echo "  -o, --output-dir DIR     Output directory for reports"
    echo "  --open                   Open HTML report after generation"
    echo "  --no-venv                Skip virtual environment activation"
    echo "  -h, --help               Show this help message"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  # Evaluate with specific LoRA model"
    echo "  $0 -l ./logs/gemma3-1b_sft_instruction_.../lora_model"
    echo ""
    echo "  # Evaluate with 200 samples"
    echo "  $0 -l ./lora_model -n 200"
    echo ""
    echo "  # Use specific GPU and open report"
    echo "  $0 -l ./lora_model -g 1 --open"
    echo ""
    echo "  # Find latest LoRA model automatically"
    echo "  $0 -l \$(ls -td ./logs/*/lora_model 2>/dev/null | head -1)"
    echo ""
}

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--lora-path)
            LORA_PATH="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -n|--samples)
            SAMPLE_COUNT="$2"
            shift 2
            ;;
        -t|--tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --open)
            OPEN_REPORT="true"
            shift
            ;;
        --no-venv)
            SKIP_VENV="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# ==============================================================================
# Validate Required Arguments
# ==============================================================================
if [ -z "$LORA_PATH" ]; then
    echo -e "${RED}[ERROR]${NC} --lora-path is required"
    echo ""
    echo -e "${YELLOW}Hint:${NC} Find your LoRA model path:"
    echo "  ls -la $SCRIPT_DIR/logs/*/lora_model"
    echo ""
    echo "Or use the latest one:"
    echo "  $0 -l \$(ls -td $SCRIPT_DIR/logs/*/lora_model 2>/dev/null | head -1)"
    echo ""
    show_usage
    exit 1
fi

# Check if LoRA path exists
if [ ! -d "$LORA_PATH" ]; then
    echo -e "${RED}[ERROR]${NC} LoRA model path does not exist: $LORA_PATH"
    echo ""
    echo -e "${YELLOW}Available LoRA models:${NC}"
    ls -la "$SCRIPT_DIR/logs/*/lora_model" 2>/dev/null || echo "  No LoRA models found in logs/"
    exit 1
fi

# ==============================================================================
# Virtual Environment Activation
# ==============================================================================
if [ "$SKIP_VENV" = "false" ] && [ -d "$VENV_DIR" ]; then
    echo -e "${BLUE}[VENV]${NC} Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}[VENV]${NC} Virtual environment activated"
fi

# ==============================================================================
# Environment Checks
# ==============================================================================
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              🔬 AI TRUSTWORTHINESS EVALUATION                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}[CHECK]${NC} Verifying environment..."

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python not found"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Python: $(python --version 2>&1)"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo -e "  ${GREEN}✓${NC} CUDA: $GPU_COUNT GPU(s) available"
else
    echo -e "  ${YELLOW}⚠${NC} nvidia-smi not found"
fi

# Check required packages
echo -e "${BLUE}[CHECK]${NC} Verifying required packages..."

python -c "import torch" 2>/dev/null && echo -e "  ${GREEN}✓${NC} torch" || {
    echo -e "  ${RED}✗${NC} torch not installed"
    exit 1
}

python -c "import unsloth" 2>/dev/null && echo -e "  ${GREEN}✓${NC} unsloth" || {
    echo -e "  ${RED}✗${NC} unsloth not installed"
    exit 1
}

python -c "import tqdm" 2>/dev/null && echo -e "  ${GREEN}✓${NC} tqdm" || {
    echo -e "  ${YELLOW}⚠${NC} tqdm not installed, installing..."
    pip install tqdm -q
}

# ==============================================================================
# Display Configuration
# ==============================================================================
echo ""
echo -e "${BLUE}[CONFIG]${NC} Evaluation Configuration:"
echo -e "  ${PURPLE}GPU:${NC}             $GPU_ID"
echo -e "  ${PURPLE}LoRA Path:${NC}       $LORA_PATH"
echo -e "  ${PURPLE}Samples:${NC}         $SAMPLE_COUNT"
echo -e "  ${PURPLE}Max Tokens:${NC}      $MAX_NEW_TOKENS"
echo -e "  ${PURPLE}Temperature:${NC}     $TEMPERATURE"
echo -e "  ${PURPLE}Top-P:${NC}           $TOP_P"
echo -e "  ${PURPLE}Top-K:${NC}           $TOP_K"
echo -e "  ${PURPLE}Seed:${NC}            $SEED"
[ -n "$OUTPUT_DIR" ] && echo -e "  ${PURPLE}Output Dir:${NC}      $OUTPUT_DIR"
echo ""

# ==============================================================================
# Build Python Arguments
# ==============================================================================
PYTHON_ARGS=""
PYTHON_ARGS="$PYTHON_ARGS --lora-path \"$LORA_PATH\""
PYTHON_ARGS="$PYTHON_ARGS --gpu $GPU_ID"
PYTHON_ARGS="$PYTHON_ARGS --samples $SAMPLE_COUNT"
PYTHON_ARGS="$PYTHON_ARGS --max-tokens $MAX_NEW_TOKENS"
PYTHON_ARGS="$PYTHON_ARGS --temperature $TEMPERATURE"
PYTHON_ARGS="$PYTHON_ARGS --top-p $TOP_P"
PYTHON_ARGS="$PYTHON_ARGS --top-k $TOP_K"
PYTHON_ARGS="$PYTHON_ARGS --seed $SEED"
[ -n "$OUTPUT_DIR" ] && PYTHON_ARGS="$PYTHON_ARGS --output-dir \"$OUTPUT_DIR\""

# ==============================================================================
# Export Environment Variables
# ==============================================================================
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# ==============================================================================
# Run Evaluation
# ==============================================================================
echo -e "${GREEN}[START]${NC} Starting evaluation..."
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

cd "$SCRIPT_DIR"

# Run Python script
REPORT_PATH=$(eval python evaluate.py $PYTHON_ARGS 2>&1 | tee /dev/stderr | grep "HTML report saved:" | sed 's/.*HTML report saved: //')

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"

# ==============================================================================
# Post-Processing
# ==============================================================================
if [ -n "$REPORT_PATH" ] && [ -f "$REPORT_PATH" ]; then
    echo -e "${GREEN}[SUCCESS]${NC} Evaluation complete!"
    echo ""
    echo -e "${CYAN}Report Location:${NC}"
    echo -e "  📊 HTML: $REPORT_PATH"
    
    # JSON file
    JSON_PATH="${REPORT_PATH%.html}.json"
    JSON_PATH=$(dirname "$REPORT_PATH")/results.json
    [ -f "$JSON_PATH" ] && echo -e "  📝 JSON: $JSON_PATH"
    
    echo ""
    
    # Open report if requested
    if [ "$OPEN_REPORT" = "true" ]; then
        echo -e "${BLUE}[OPEN]${NC} Opening report in browser..."
        if command -v xdg-open &> /dev/null; then
            xdg-open "$REPORT_PATH" &
        elif command -v open &> /dev/null; then
            open "$REPORT_PATH" &
        else
            echo -e "${YELLOW}⚠${NC} Could not open browser automatically"
            echo -e "  Open manually: file://$REPORT_PATH"
        fi
    else
        echo -e "${YELLOW}Tip:${NC} Open the HTML report in your browser to view results"
        echo -e "  Or run with ${GREEN}--open${NC} flag to auto-open"
    fi
else
    echo -e "${YELLOW}[INFO]${NC} Check the output above for results"
fi

echo ""
echo -e "${GREEN}[DONE]${NC} Evaluation finished!"

