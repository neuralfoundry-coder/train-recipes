#!/bin/bash
# ==============================================================================
# GPT-OSS 20B Model Comparison Script
# Compares base model vs fine-tuned model outputs with HTML report generation
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_LOCAL_FILE="$SCRIPT_DIR/env_local"
VENV_DIR="$SCRIPT_DIR/.venv"

# ==============================================================================
# Load Default Configuration from env_local
# ==============================================================================
if [ -f "$ENV_LOCAL_FILE" ]; then
    echo -e "${BLUE}[CONFIG]${NC} Loading configuration from env_local..."
    
    # Parse env_local and set variables (handle quoted values)
    while IFS='=' read -r key value; do
        # Skip empty lines and comments
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        # Remove leading/trailing whitespace from key
        key=$(echo "$key" | xargs)
        # Remove quotes from value
        value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
        # Export the variable
        export "$key"="$value"
    done < "$ENV_LOCAL_FILE"
fi

# ==============================================================================
# Default Values (from env_local or hardcoded)
# ==============================================================================
GPU_ID="${COMPARE_GPU_ID:-${INFER_GPU_ID:-1}}"
LORA_PATH="${COMPARE_LORA_PATH:-}"
SAMPLE_COUNT="${COMPARE_SAMPLE_COUNT:-50}"
MAX_NEW_TOKENS="${COMPARE_MAX_NEW_TOKENS:-512}"
TEMPERATURE="${COMPARE_TEMPERATURE:-0.7}"
TOP_P="${COMPARE_TOP_P:-0.9}"
TOP_K="${COMPARE_TOP_K:-50}"
REASONING_EFFORT="${COMPARE_REASONING_EFFORT:-medium}"
SEED="${COMPARE_SEED:-42}"
OUTPUT_DIR=""
SKIP_VENV="false"
OPEN_REPORT="false"

# ==============================================================================
# Usage Information
# ==============================================================================
show_usage() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║    🎓 GPT-OSS 20B Korean Education Model Comparison Tool            ║"
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
    echo "  -n, --samples N          Number of samples to compare (default: $SAMPLE_COUNT)"
    echo "  -t, --tokens N           Max new tokens to generate (default: $MAX_NEW_TOKENS)"
    echo "  --temperature FLOAT      Sampling temperature (default: $TEMPERATURE)"
    echo "  --top-p FLOAT            Top-p sampling (default: $TOP_P)"
    echo "  --top-k INT              Top-k sampling (default: $TOP_K)"
    echo "  -r, --reasoning LEVEL    Reasoning effort: low, medium, high (default: $REASONING_EFFORT)"
    echo "  --seed INT               Random seed (default: $SEED)"
    echo "  -o, --output-dir DIR     Output directory for reports"
    echo "  --open                   Open HTML report after generation"
    echo "  --no-venv                Skip virtual environment activation"
    echo "  -h, --help               Show this help message"
    echo ""
    echo -e "${GREEN}GPT-OSS Reasoning Effort Levels:${NC}"
    echo "  low    - Fast responses, minimal reasoning"
    echo "  medium - Balanced performance and speed"
    echo "  high   - Best reasoning, slower responses"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  # Compare with specific LoRA model"
    echo "  $0 -l ./logs/gpt-oss-20b_20250111_120000/lora_model"
    echo ""
    echo "  # Compare 30 samples with high reasoning"
    echo "  $0 -l ./lora_model -n 30 -r high"
    echo ""
    echo "  # Use specific GPU and longer output"
    echo "  $0 -l ./lora_model -g 1 -t 1024"
    echo ""
    echo "  # Find latest LoRA model automatically"
    echo "  $0 -l \$(ls -td ./logs/*/lora_model 2>/dev/null | head -1)"
    echo ""
    echo -e "${GREEN}Environment Variables (via env_local):${NC}"
    echo "  COMPARE_GPU_ID           GPU to use for comparison"
    echo "  COMPARE_SAMPLE_COUNT     Default sample count"
    echo "  COMPARE_MAX_NEW_TOKENS   Default max new tokens"
    echo "  COMPARE_TEMPERATURE      Default temperature"
    echo "  COMPARE_TOP_P            Default top-p"
    echo "  COMPARE_TOP_K            Default top-k"
    echo "  COMPARE_REASONING_EFFORT Default reasoning effort"
    echo "  COMPARE_SEED             Random seed"
    echo "  COMPARE_LORA_PATH        Default LoRA model path"
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
        -r|--reasoning)
            REASONING_EFFORT="$2"
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
echo -e "${CYAN}║          🎓 GPT-OSS 20B Korean Education Model Comparison           ║${NC}"
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
    echo -e "  ${YELLOW}⚠${NC} nvidia-smi not found (CPU mode)"
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

python -c "import datasets" 2>/dev/null && echo -e "  ${GREEN}✓${NC} datasets" || {
    echo -e "  ${RED}✗${NC} datasets not installed"
    exit 1
}

# ==============================================================================
# Display Configuration
# ==============================================================================
echo ""
echo -e "${BLUE}[CONFIG]${NC} Comparison Configuration:"
echo -e "  ${PURPLE}GPU:${NC}             $GPU_ID"
echo -e "  ${PURPLE}LoRA Path:${NC}       $LORA_PATH"
echo -e "  ${PURPLE}Samples:${NC}         $SAMPLE_COUNT"
echo -e "  ${PURPLE}Max Tokens:${NC}      $MAX_NEW_TOKENS"
echo -e "  ${PURPLE}Temperature:${NC}     $TEMPERATURE"
echo -e "  ${PURPLE}Top-P:${NC}           $TOP_P"
echo -e "  ${PURPLE}Top-K:${NC}           $TOP_K"
echo -e "  ${PURPLE}Reasoning:${NC}       $REASONING_EFFORT"
echo -e "  ${PURPLE}Seed:${NC}            $SEED"
[ -n "$OUTPUT_DIR" ] && echo -e "  ${PURPLE}Output Dir:${NC}      $OUTPUT_DIR"
echo ""

# ==============================================================================
# Build Python Arguments
# ==============================================================================
PYTHON_ARGS=""
PYTHON_ARGS="$PYTHON_ARGS --lora-path \"$LORA_PATH\""
PYTHON_ARGS="$PYTHON_ARGS --samples $SAMPLE_COUNT"
PYTHON_ARGS="$PYTHON_ARGS --max-tokens $MAX_NEW_TOKENS"
PYTHON_ARGS="$PYTHON_ARGS --temperature $TEMPERATURE"
PYTHON_ARGS="$PYTHON_ARGS --top-p $TOP_P"
PYTHON_ARGS="$PYTHON_ARGS --top-k $TOP_K"
PYTHON_ARGS="$PYTHON_ARGS --reasoning $REASONING_EFFORT"
PYTHON_ARGS="$PYTHON_ARGS --seed $SEED"
[ -n "$OUTPUT_DIR" ] && PYTHON_ARGS="$PYTHON_ARGS --output-dir \"$OUTPUT_DIR\""

# ==============================================================================
# Export Environment Variables
# ==============================================================================
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export COMPARE_GPU_ID="$GPU_ID"
export COMPARE_SAMPLE_COUNT="$SAMPLE_COUNT"
export COMPARE_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
export COMPARE_TEMPERATURE="$TEMPERATURE"
export COMPARE_TOP_P="$TOP_P"
export COMPARE_TOP_K="$TOP_K"
export COMPARE_REASONING_EFFORT="$REASONING_EFFORT"
export COMPARE_SEED="$SEED"

# ==============================================================================
# Run Comparison
# ==============================================================================
echo -e "${GREEN}[START]${NC} Starting model comparison..."
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

cd "$SCRIPT_DIR"

# Run Python script
REPORT_PATH=$(eval python compare_models.py $PYTHON_ARGS 2>&1 | tee /dev/stderr | grep "Report generated:" | sed 's/.*Report generated: //')

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"

# ==============================================================================
# Post-Processing
# ==============================================================================
if [ -n "$REPORT_PATH" ] && [ -f "$REPORT_PATH" ]; then
    echo -e "${GREEN}[SUCCESS]${NC} Comparison complete!"
    echo ""
    echo -e "${CYAN}Report Location:${NC}"
    echo -e "  📊 HTML: $REPORT_PATH"
    
    # JSON file (same name with .json extension)
    JSON_PATH="${REPORT_PATH%.html}.json"
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
        echo -e "${YELLOW}Tip:${NC} Open the HTML report in your browser to view the comparison"
        echo -e "  Or run with ${GREEN}--open${NC} flag to auto-open"
    fi
else
    echo -e "${YELLOW}[INFO]${NC} Check the output above for results"
fi

echo ""
echo -e "${GREEN}[DONE]${NC} Model comparison finished!"

