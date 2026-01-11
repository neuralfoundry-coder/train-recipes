# Train Recipes 🚀

A collection of fine-tuning recipes for Large Language Models (LLMs) using various frameworks and techniques.

## Overview

This repository provides ready-to-use training scripts and configurations for fine-tuning LLMs with:
- **LoRA/QLoRA** for efficient fine-tuning
- **Reasoning + Conversational** mixed dataset training
- **Unsloth** for optimized training speed
- **Wandb** integration for experiment tracking

## Directory Structure

```
train-recipes/
├── samples/
│   └── unsloth/
│       └── qwen3/                    # Qwen3-14B fine-tuning
│           ├── qwen3_(14b)_reasoning_conversational.py  # Training module
│           ├── chat_console.py       # Interactive chat console
│           ├── train.sh              # Training launcher script
│           ├── chat.sh               # Chat launcher script
│           └── env_local             # Configuration file (git ignored)
├── reasoning/                        # Reasoning-focused recipes
└── README.md
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA 11.8+ with compatible GPU (24GB+ VRAM recommended)
- [Unsloth](https://github.com/unslothai/unsloth) installed

### 2. Setup Configuration

Create `env_local` file in the recipe directory:

```bash
cd samples/unsloth/qwen3/
cp env_local.example env_local  # Or create manually
```

Edit `env_local` with your settings:

```bash
# API Tokens
HF_TOKEN="your_huggingface_token"
WNB_API_KEY="your_wandb_api_key"

# Model Configuration
MODEL_NAME="unsloth/Qwen3-14B"
MAX_SEQ_LENGTH="32768"
LOAD_IN_4BIT="false"

# LoRA Configuration
LORA_R="32"
LORA_ALPHA="32"

# Training Configuration
TRAIN_GPU_IDS="0,1"
TRAIN_BATCH_SIZE="2"
TRAIN_LEARNING_RATE="2e-4"
TRAIN_MAX_STEPS="30"

# Inference Configuration
INFER_TEMPERATURE="0.7"
INFER_TOP_P="0.8"
INFER_MAX_TOKENS="16384"
```

### 3. Run Training

```bash
# View current configuration
./train.sh -v

# Start training with env_local settings
./train.sh

# Override settings via command line
./train.sh -g 0 -b 4 -s 100 -r 1e-4
```

### 4. Test the Model

```bash
# List available trained models
./chat.sh -l

# Start interactive chat (auto-detects latest model)
./chat.sh

# Start with thinking mode enabled
./chat.sh -T

# Specify a model path
./chat.sh -m ./logs/qwen3-14b_reasoning-conversational_20260111_143052/lora_model
```

## Training Scripts

### `train.sh` Options

| Option | Description |
|--------|-------------|
| `-g, --gpu IDS` | GPU device IDs (e.g., `0,1`) |
| `-b, --batch N` | Batch size |
| `-r, --lr RATE` | Learning rate |
| `-s, --steps N` | Max training steps |
| `--lora-r N` | LoRA rank |
| `-e, --env NAME` | Conda environment to activate |
| `-l, --logs` | List recent training logs |
| `-c, --clean` | Clean old log directories |
| `-v, --vars` | Show current configuration |

### `chat.sh` Options

| Option | Description |
|--------|-------------|
| `-m, --model PATH` | Path to LoRA model |
| `-g, --gpu ID` | GPU device ID |
| `-t, --tokens N` | Max new tokens |
| `-T, --thinking` | Enable thinking mode |
| `-l, --list` | List available models |
| `-v, --vars` | Show inference configuration |

## Chat Console Commands

Once in the interactive console:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/thinking` | Toggle thinking mode |
| `/clear` | Clear conversation history |
| `/mode` | Show current settings |
| `/tokens N` | Set max tokens |
| `/single` | Single-turn mode |
| `/multi` | Multi-turn mode |
| `/exit` | Exit console |

## Output Structure

Training outputs are saved with unique timestamps:

```
logs/
└── qwen3-14b_reasoning-conversational_20260111_143052/
    ├── train/
    │   ├── train.log           # Training log
    │   ├── config.txt          # Configuration snapshot
    │   ├── training_stats.txt  # Final statistics
    │   └── tensorboard/        # TensorBoard logs
    ├── eval/
    │   └── dataset_info.txt    # Dataset statistics
    ├── checkpoints/            # Training checkpoints
    └── lora_model/             # Final LoRA adapters
```

## Available Recipes

### Qwen3-14B Reasoning + Conversational

Fine-tunes Qwen3-14B with a mix of:
- **Reasoning data (75%)**: [OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)
- **Conversational data (25%)**: [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)

Features:
- Supports both thinking and non-thinking inference modes
- Optimized generation parameters for each mode
- Full logging and experiment tracking

## Configuration Priority

Settings are loaded in this order (higher priority first):

1. **Command-line arguments**
2. **Environment variables** (`export TRAIN_BATCH_SIZE=4`)
3. **env_local file**
4. **Default values** in code

## License

This project is licensed under the LGPL-3.0 License.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for optimized training
- [Hugging Face](https://huggingface.co/) for model hosting
- [Weights & Biases](https://wandb.ai/) for experiment tracking

