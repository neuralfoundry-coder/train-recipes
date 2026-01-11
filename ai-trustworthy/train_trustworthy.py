# -*- coding: utf-8 -*-
"""
AI Trustworthiness Fine-tuning with Unsloth
Korean LLM Trustworthiness Benchmark Dataset
- Categories: Helpfulness, Harmlessness, Honesty
- Training modes: SFT, DPO
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path


# ==============================================================================
# Load Environment Configuration
# ==============================================================================
def load_env_local():
    """Load environment variables from env_local file in the same directory."""
    env_file = Path(__file__).parent / "env_local"
    env_vars = {}
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    env_vars[key.strip()] = value
    return env_vars


def get_env(key, default=None, env_local=None, cast_type=None):
    """Get environment variable with priority: os.environ > env_local > default."""
    value = os.environ.get(key)
    if value is None and env_local:
        value = env_local.get(key)
    if value is None:
        value = default
    
    if cast_type and value is not None:
        if cast_type == bool:
            return str(value).lower() in ('true', '1', 'yes')
        return cast_type(value)
    return value


# Load env_local
ENV_LOCAL = load_env_local()

# GPU Configuration (must be set before torch import)
CUDA_VISIBLE_DEVICES = get_env("CUDA_VISIBLE_DEVICES", 
                                get_env("TRAIN_GPU_IDS", "0", ENV_LOCAL), 
                                ENV_LOCAL)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Now import torch and other heavy modules
import torch
import wandb
import pandas as pd
from datasets import load_dataset, Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer


# ==============================================================================
# Configuration (from env_local with defaults)
# ==============================================================================

# API Tokens
HF_TOKEN = get_env("HF_TOKEN", None, ENV_LOCAL)
WNB_API_KEY = get_env("WNB_API_KEY", None, ENV_LOCAL)

# Model Configuration
MODEL_NAME = get_env("MODEL_NAME", "unsloth/gemma-3-1b-it", ENV_LOCAL)
MODEL_SHORT_NAME = get_env("MODEL_SHORT_NAME", "gemma3-1b", ENV_LOCAL)
CHAT_TEMPLATE = get_env("CHAT_TEMPLATE", "gemma3", ENV_LOCAL)
MAX_SEQ_LENGTH = get_env("MAX_SEQ_LENGTH", 4096, ENV_LOCAL, int)
LOAD_IN_4BIT = get_env("LOAD_IN_4BIT", False, ENV_LOCAL, bool)
LOAD_IN_8BIT = get_env("LOAD_IN_8BIT", False, ENV_LOCAL, bool)

# Dataset Configuration
DATASET_NAME = get_env("DATASET_NAME", "neuralfoundry-coder/korean-llm-trustworthiness-benchmark-full", ENV_LOCAL)
DATASET_SUBSET = get_env("DATASET_SUBSET", "sft_instruction", ENV_LOCAL)

# LoRA Configuration
LORA_R = get_env("LORA_R", 64, ENV_LOCAL, int)
LORA_ALPHA = get_env("LORA_ALPHA", 64, ENV_LOCAL, int)
LORA_DROPOUT = get_env("LORA_DROPOUT", 0, ENV_LOCAL, int)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration
EVAL_RATIO = get_env("TRAIN_EVAL_RATIO", 0.05, ENV_LOCAL, float)
BATCH_SIZE = get_env("TRAIN_BATCH_SIZE", 4, ENV_LOCAL, int)
GRADIENT_ACCUMULATION_STEPS = get_env("TRAIN_GRADIENT_ACCUMULATION_STEPS", 4, ENV_LOCAL, int)
WARMUP_STEPS = get_env("TRAIN_WARMUP_STEPS", 10, ENV_LOCAL, int)
LEARNING_RATE = get_env("TRAIN_LEARNING_RATE", 2e-4, ENV_LOCAL, float)
SEED = get_env("TRAIN_SEED", 3407, ENV_LOCAL, int)

# Epochs vs Max Steps (epochs takes priority if set)
_epochs_str = get_env("TRAIN_EPOCHS", "", ENV_LOCAL)
_max_steps_str = get_env("TRAIN_MAX_STEPS", "", ENV_LOCAL)

if _epochs_str and _epochs_str.strip():
    NUM_TRAIN_EPOCHS = int(_epochs_str)
    MAX_STEPS = None
else:
    NUM_TRAIN_EPOCHS = 1
    MAX_STEPS = int(_max_steps_str) if _max_steps_str else None

# Inference Configuration
INFER_TEMPERATURE = get_env("INFER_TEMPERATURE", 0.7, ENV_LOCAL, float)
INFER_TOP_P = get_env("INFER_TOP_P", 0.9, ENV_LOCAL, float)
INFER_TOP_K = get_env("INFER_TOP_K", 50, ENV_LOCAL, int)
INFER_MAX_TOKENS = get_env("INFER_MAX_TOKENS", 1024, ENV_LOCAL, int)

# Timestamp for unique run identification
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output paths
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs" / f"{MODEL_SHORT_NAME}_{DATASET_SUBSET}_{RUN_TIMESTAMP}"
TRAIN_LOG_DIR = LOG_DIR / "train"
EVAL_LOG_DIR = LOG_DIR / "eval"
OUTPUT_DIR = LOG_DIR / "checkpoints"
LORA_OUTPUT_PATH = LOG_DIR / "lora_model"

# Wandb Configuration
WANDB_PROJECT = get_env("WANDB_PROJECT", "ai-trustworthiness-benchmark", ENV_LOCAL)
WANDB_RUN_NAME = f"{MODEL_SHORT_NAME}-{DATASET_SUBSET}-r{LORA_R}-{RUN_TIMESTAMP}"

# System prompts by category
SYSTEM_PROMPTS = {
    "helpfulness": "당신은 사용자의 질문에 도움이 되는 답변을 제공하는 AI 어시스턴트입니다. 질문이 불명확하거나 정보가 부족할 경우 적절한 추가 질문을 통해 사용자의 의도를 파악해야 합니다.",
    "harmlessness": "당신은 편견 없이 공정하고 해롭지 않은 답변을 제공하는 AI 어시스턴트입니다. 혐오, 편견, 불법적인 내용을 피하고 윤리적으로 올바른 응답을 해야 합니다.",
    "honesty": "당신은 정확하고 신뢰할 수 있는 정보를 제공하는 AI 어시스턴트입니다. 확실하지 않은 정보는 그렇다고 밝히고, 잘못된 정보를 제공하지 않도록 주의해야 합니다.",
}


# ==============================================================================
# Logging Setup
# ==============================================================================
def setup_logging():
    """Setup logging to file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler = logging.FileHandler(TRAIN_LOG_DIR / "train.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Initialize logging
logger = setup_logging()


# ==============================================================================
# Data Preparation
# ==============================================================================
def prepare_sft_dataset(tokenizer):
    """Prepare SFT instruction dataset."""
    logger.info(f"Loading dataset: {DATASET_NAME} (type={DATASET_SUBSET})")
    
    # Load train split and filter by type
    full_dataset = load_dataset(DATASET_NAME, split="train", token=HF_TOKEN)
    dataset = full_dataset.filter(lambda x: x["type"] == DATASET_SUBSET)
    logger.info(f"Loaded {len(dataset)} samples (filtered from {len(full_dataset)} total)")
    
    def convert_to_conversation(example):
        """Convert SFT format to conversation format."""
        category = example.get("category", "")
        system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["honesty"])
        
        conversations = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        
        # Add system prompt if supported
        if example.get("system"):
            conversations.insert(0, {"role": "system", "content": example["system"]})
        
        return {"conversations": conversations}
    
    dataset = dataset.map(convert_to_conversation)
    
    # Apply chat template
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            )
            # Remove BOS if present (Gemma3 specific)
            if text.startswith('<bos>'):
                text = text.removeprefix('<bos>')
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return dataset


def prepare_fact_checking_dataset(tokenizer):
    """Prepare fact checking dataset."""
    logger.info(f"Loading dataset: {DATASET_NAME} (type=fact_checking)")
    
    # Load train split and filter by type
    full_dataset = load_dataset(DATASET_NAME, split="train", token=HF_TOKEN)
    dataset = full_dataset.filter(lambda x: x["type"] == "fact_checking")
    logger.info(f"Loaded {len(dataset)} samples (filtered from {len(full_dataset)} total)")
    
    def convert_to_conversation(example):
        """Convert fact checking format to conversation format."""
        conversations = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        return {"conversations": conversations}
    
    dataset = dataset.map(convert_to_conversation)
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            )
            if text.startswith('<bos>'):
                text = text.removeprefix('<bos>')
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return dataset


def prepare_dpo_dataset(tokenizer):
    """Prepare DPO preference dataset for SFT training (using chosen responses)."""
    logger.info(f"Loading dataset: {DATASET_NAME} (type=dpo_preference)")
    
    # Load train split and filter by type
    full_dataset = load_dataset(DATASET_NAME, split="train", token=HF_TOKEN)
    dataset = full_dataset.filter(lambda x: x["type"] == "dpo_preference")
    logger.info(f"Loaded {len(dataset)} samples (filtered from {len(full_dataset)} total)")
    
    def convert_to_conversation(example):
        """Convert DPO format to conversation format (using chosen response)."""
        category = example.get("category", "")
        system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["honesty"])
        
        conversations = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]}
        ]
        
        # Add system prompt if provided
        if example.get("system"):
            conversations.insert(0, {"role": "system", "content": example["system"]})
        
        return {"conversations": conversations}
    
    dataset = dataset.map(convert_to_conversation)
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            )
            if text.startswith('<bos>'):
                text = text.removeprefix('<bos>')
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return dataset


# ==============================================================================
# Training Pipeline
# ==============================================================================
def main():
    logger.info("=" * 70)
    logger.info("AI TRUSTWORTHINESS FINE-TUNING")
    logger.info("Korean LLM Trustworthiness Benchmark")
    logger.info("=" * 70)
    
    # Log configuration
    logger.info("[STEP 1/8] Logging Configuration")
    logger.info(f"  Log directory: {LOG_DIR}")
    logger.info(f"  Train logs: {TRAIN_LOG_DIR}")
    logger.info(f"  Eval logs: {EVAL_LOG_DIR}")
    logger.info(f"  Checkpoints: {OUTPUT_DIR}")
    logger.info(f"  LoRA output: {LORA_OUTPUT_PATH}")
    
    # Log training configuration
    logger.info("-" * 70)
    logger.info("Configuration (from env_local):")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Dataset: {DATASET_NAME}/{DATASET_SUBSET}")
    logger.info(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    logger.info(f"  Load in 4bit: {LOAD_IN_4BIT}")
    logger.info(f"  Load in 8bit: {LOAD_IN_8BIT}")
    logger.info(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    logger.info(f"  Batch size: {BATCH_SIZE}, GA steps: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    if MAX_STEPS:
        logger.info(f"  Max steps: {MAX_STEPS}")
    else:
        logger.info(f"  Epochs: {NUM_TRAIN_EPOCHS}")
    logger.info(f"  Eval ratio: {EVAL_RATIO:.0%}")
    logger.info(f"  GPU: {CUDA_VISIBLE_DEVICES}")
    logger.info("-" * 70)
    
    # Save config to file
    with open(TRAIN_LOG_DIR / "config.txt", "w") as f:
        f.write(f"MODEL_NAME={MODEL_NAME}\n")
        f.write(f"DATASET_NAME={DATASET_NAME}\n")
        f.write(f"DATASET_SUBSET={DATASET_SUBSET}\n")
        f.write(f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}\n")
        f.write(f"LOAD_IN_4BIT={LOAD_IN_4BIT}\n")
        f.write(f"LOAD_IN_8BIT={LOAD_IN_8BIT}\n")
        f.write(f"LORA_R={LORA_R}\n")
        f.write(f"LORA_ALPHA={LORA_ALPHA}\n")
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"GRADIENT_ACCUMULATION_STEPS={GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"LEARNING_RATE={LEARNING_RATE}\n")
        f.write(f"MAX_STEPS={MAX_STEPS}\n")
        f.write(f"NUM_TRAIN_EPOCHS={NUM_TRAIN_EPOCHS}\n")
        f.write(f"EVAL_RATIO={EVAL_RATIO}\n")
        f.write(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n")
    
    # ========================================================================
    # Initialize Wandb
    # ========================================================================
    logger.info("[STEP 2/8] Initializing Wandb")
    if WNB_API_KEY:
        os.environ["WANDB_API_KEY"] = WNB_API_KEY
        wandb.login(key=WNB_API_KEY)
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model_name": MODEL_NAME,
                "dataset": f"{DATASET_NAME}/{DATASET_SUBSET}",
                "max_seq_length": MAX_SEQ_LENGTH,
                "load_in_4bit": LOAD_IN_4BIT,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "eval_ratio": EVAL_RATIO,
            },
        )
        logger.info(f"  Wandb initialized: {WANDB_PROJECT}/{WANDB_RUN_NAME}")
    else:
        logger.warning("  WNB_API_KEY not found, wandb logging disabled")
    
    # ========================================================================
    # Load Model
    # ========================================================================
    logger.info("[STEP 3/8] Loading Model")
    logger.info(f"  Loading {MODEL_NAME}...")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        full_finetuning=False,
        token=HF_TOKEN,
    )
    logger.info("  Model loaded successfully")
    
    # Add LoRA adapters
    logger.info("  Adding LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    logger.info("  LoRA adapters added successfully")
    
    # Apply chat template
    logger.info(f"  Applying chat template: {CHAT_TEMPLATE}")
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    
    # ========================================================================
    # Prepare Dataset
    # ========================================================================
    logger.info("[STEP 4/8] Preparing Dataset")
    
    if DATASET_SUBSET == "fact_checking":
        dataset = prepare_fact_checking_dataset(tokenizer)
    elif DATASET_SUBSET == "dpo_preference":
        dataset = prepare_dpo_dataset(tokenizer)
    else:  # sft_instruction or default
        dataset = prepare_sft_dataset(tokenizer)
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=SEED)
    
    eval_size = max(1, int(len(dataset) * EVAL_RATIO))
    eval_dataset = dataset.select(range(eval_size))
    train_dataset = dataset.select(range(eval_size, len(dataset)))
    
    logger.info(f"  Train dataset: {len(train_dataset)} samples")
    logger.info(f"  Eval dataset: {len(eval_dataset)} samples ({EVAL_RATIO:.0%})")
    
    # Show sample
    logger.info("  Sample data:")
    sample = train_dataset[0]["text"][:500]
    logger.info(f"    {sample}...")
    
    # Save dataset info
    with open(EVAL_LOG_DIR / "dataset_info.txt", "w") as f:
        f.write(f"Dataset: {DATASET_NAME}/{DATASET_SUBSET}\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Eval samples: {len(eval_dataset)}\n")
        f.write(f"Eval ratio: {EVAL_RATIO:.0%}\n")
    
    # ========================================================================
    # Setup Trainer
    # ========================================================================
    logger.info("[STEP 5/8] Setting up Trainer")
    
    sft_config_args = {
        "output_dir": str(OUTPUT_DIR),
        "dataset_text_field": "text",
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": 1,
        "logging_dir": str(TRAIN_LOG_DIR / "tensorboard"),
        "optim": "adamw_8bit",
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
        "seed": SEED,
        "report_to": "wandb" if WNB_API_KEY else "none",
        "run_name": WANDB_RUN_NAME if WNB_API_KEY else None,
    }
    
    if MAX_STEPS:
        sft_config_args["max_steps"] = MAX_STEPS
        sft_config_args["eval_strategy"] = "steps"
        sft_config_args["eval_steps"] = max(10, MAX_STEPS // 10)
        sft_config_args["save_strategy"] = "steps"
        sft_config_args["save_steps"] = max(10, MAX_STEPS // 5)
    else:
        sft_config_args["num_train_epochs"] = NUM_TRAIN_EPOCHS
        sft_config_args["eval_strategy"] = "epoch"
        sft_config_args["save_strategy"] = "epoch"
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**sft_config_args),
    )
    
    # Train on responses only
    logger.info("  Setting up train_on_responses_only...")
    if CHAT_TEMPLATE == "gemma3":
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
    
    logger.info("  Trainer configured successfully")
    
    # ========================================================================
    # GPU Memory Stats
    # ========================================================================
    logger.info("[STEP 6/8] GPU Memory Check")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"  GPU: {gpu_stats.name}")
    logger.info(f"  Max memory: {max_memory} GB")
    logger.info(f"  Reserved memory before training: {start_gpu_memory} GB")
    
    # ========================================================================
    # Training
    # ========================================================================
    logger.info("[STEP 7/8] Starting Training")
    logger.info("-" * 70)
    
    trainer_stats = trainer.train()

    logger.info("-" * 70)
    logger.info("  Training completed!")
    
    # Final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    
    logger.info(f"  Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds ({trainer_stats.metrics['train_runtime']/60:.2f} minutes)")
    logger.info(f"  Peak memory: {used_memory} GB")
    logger.info(f"  Peak memory for training: {used_memory_for_lora} GB")
    
    # Save training stats
    with open(TRAIN_LOG_DIR / "training_stats.txt", "w") as f:
        f.write(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds\n")
        f.write(f"Peak memory: {used_memory} GB\n")
        f.write(f"Peak memory for training: {used_memory_for_lora} GB\n")
        for key, value in trainer_stats.metrics.items():
            f.write(f"{key}: {value}\n")
    
    # ========================================================================
    # Save Model
    # ========================================================================
    logger.info("[STEP 8/8] Saving Model")
    LORA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_OUTPUT_PATH))
    tokenizer.save_pretrained(str(LORA_OUTPUT_PATH))
    logger.info(f"  LoRA adapters saved to: {LORA_OUTPUT_PATH}")
    
    # ========================================================================
    # Inference Example
    # ========================================================================
    logger.info("=" * 70)
    logger.info("INFERENCE EXAMPLE")
    logger.info("=" * 70)
    
    logger.info(f"Config: temp={INFER_TEMPERATURE}, top_p={INFER_TOP_P}, top_k={INFER_TOP_K}")
    
    # Test with a trustworthiness question
    test_prompts = [
        "난방비가 적게 나오는 지역의 주민들은 모두 소득 수준이 낮은 것일까요?",
        "문학에 대해 설명해주세요.",
    ]
    
    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.startswith('<bos>'):
            text = text.removeprefix('<bos>')
        
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=INFER_MAX_TOKENS,
            temperature=INFER_TEMPERATURE,
            top_p=INFER_TOP_P,
            top_k=INFER_TOP_K,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )
        logger.info("-" * 40)
    
    # ========================================================================
    # Finish
    # ========================================================================
    if WNB_API_KEY:
        wandb.finish()
    
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"All logs saved to: {LOG_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

