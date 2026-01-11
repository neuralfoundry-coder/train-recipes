# -*- coding: utf-8 -*-
"""
GPT-OSS 20B Korean Education Fine-tuning with Unsloth
- Dataset: neuralfoundry-coder/aihub-korean-education-instruct
- Based on OpenAI's GPT-OSS model with Harmony format
- Supports reasoning effort levels (low, medium, high)
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
                                get_env("TRAIN_GPU_IDS", "1", ENV_LOCAL), 
                                ENV_LOCAL)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Now import torch and other heavy modules
import torch
import wandb
import pandas as pd
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer


# ==============================================================================
# Configuration (from env_local with defaults)
# ==============================================================================

# API Tokens
HF_TOKEN = get_env("HF_TOKEN", None, ENV_LOCAL)
WNB_API_KEY = get_env("WNB_API_KEY", None, ENV_LOCAL)

# Model Configuration
MODEL_NAME = get_env("MODEL_NAME", "unsloth/gpt-oss-20b-unsloth-bnb-4bit", ENV_LOCAL)
MODEL_SHORT_NAME = get_env("MODEL_SHORT_NAME", "gpt-oss-20b", ENV_LOCAL)
DATASET_NAME = get_env("DATASET_NAME", "neuralfoundry-coder/aihub-korean-education-instruct", ENV_LOCAL)
MAX_SEQ_LENGTH = get_env("MAX_SEQ_LENGTH", 2048, ENV_LOCAL, int)
LOAD_IN_4BIT = get_env("LOAD_IN_4BIT", True, ENV_LOCAL, bool)
LOAD_IN_8BIT = get_env("LOAD_IN_8BIT", False, ENV_LOCAL, bool)

# LoRA Configuration
LORA_R = get_env("LORA_R", 16, ENV_LOCAL, int)
LORA_ALPHA = get_env("LORA_ALPHA", 32, ENV_LOCAL, int)
LORA_DROPOUT = get_env("LORA_DROPOUT", 0, ENV_LOCAL, int)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration
EVAL_RATIO = get_env("TRAIN_EVAL_RATIO", 0.05, ENV_LOCAL, float)
BATCH_SIZE = get_env("TRAIN_BATCH_SIZE", 1, ENV_LOCAL, int)
GRADIENT_ACCUMULATION_STEPS = get_env("TRAIN_GRADIENT_ACCUMULATION_STEPS", 4, ENV_LOCAL, int)
WARMUP_STEPS = get_env("TRAIN_WARMUP_STEPS", 10, ENV_LOCAL, int)
LEARNING_RATE = get_env("TRAIN_LEARNING_RATE", 2e-4, ENV_LOCAL, float)
SEED = get_env("TRAIN_SEED", 3407, ENV_LOCAL, int)

# Epochs vs Max Steps (epochs takes priority if set)
_epochs_str = get_env("TRAIN_EPOCHS", "", ENV_LOCAL)
_max_steps_str = get_env("TRAIN_MAX_STEPS", "100", ENV_LOCAL)

if _epochs_str and _epochs_str.strip():
    NUM_TRAIN_EPOCHS = int(_epochs_str)
    MAX_STEPS = None  # Disable max_steps when epochs is set
else:
    NUM_TRAIN_EPOCHS = 1
    MAX_STEPS = int(_max_steps_str) if _max_steps_str else 100

# Inference Configuration (for test at end of training)
INFER_REASONING_EFFORT = get_env("INFER_REASONING_EFFORT", "medium", ENV_LOCAL)
INFER_TEMPERATURE = get_env("INFER_TEMPERATURE", 0.7, ENV_LOCAL, float)
INFER_TOP_P = get_env("INFER_TOP_P", 0.9, ENV_LOCAL, float)
INFER_TOP_K = get_env("INFER_TOP_K", 50, ENV_LOCAL, int)
INFER_MAX_TOKENS = get_env("INFER_MAX_TOKENS", 1024, ENV_LOCAL, int)

# Timestamp for unique run identification
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output paths
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs" / f"{MODEL_SHORT_NAME}_{RUN_TIMESTAMP}"
TRAIN_LOG_DIR = LOG_DIR / "train"
EVAL_LOG_DIR = LOG_DIR / "eval"
OUTPUT_DIR = LOG_DIR / "checkpoints"
LORA_OUTPUT_PATH = LOG_DIR / "lora_model"

# Wandb Configuration
WANDB_PROJECT = get_env("WANDB_PROJECT", "gpt-oss-20b-korean-education", ENV_LOCAL)
WANDB_RUN_NAME = f"{MODEL_SHORT_NAME}-r{LORA_R}-lr{LEARNING_RATE}-{RUN_TIMESTAMP}"


# ==============================================================================
# GPT-OSS Specific Configuration
# ==============================================================================
# GPT-OSS uses OpenAI Harmony format with special tokens
# Note: These markers must match EXACTLY what the tokenizer outputs
# For GPT-OSS, the response starts after the assistant role token
GPT_OSS_KWARGS = dict(
    instruction_part="<|end|><|start|>assistant",  # End of user message, start of assistant
    response_part="<|message|>",  # Content follows <|message|> token
)


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
# Dataset Processing for Korean Education
# ==============================================================================
def load_korean_education_dataset(tokenizer):
    """
    Load and process the Korean education instruction dataset.
    The dataset is in ShareGPT format with 'conversations' field.
    """
    logger.info(f"Loading dataset: {DATASET_NAME}")
    
    # Load dataset from HuggingFace
    dataset = load_dataset(DATASET_NAME, split="train", token=HF_TOKEN)
    logger.info(f"  Total samples: {len(dataset)}")
    
    def format_conversations(examples):
        """Format conversations for GPT-OSS training."""
        texts = []
        
        for conversations in examples["conversations"]:
            # Apply GPT-OSS chat template
            # The dataset has ShareGPT format: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
            try:
                formatted = tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=False,
                    # GPT-OSS supports reasoning_effort parameter
                    reasoning_effort=INFER_REASONING_EFFORT,
                )
                texts.append(formatted)
            except Exception as e:
                # Fallback: convert to simple format if template fails
                text_parts = []
                for msg in conversations:
                    role = msg.get("role", msg.get("from", "user"))
                    content = msg.get("content", msg.get("value", ""))
                    if role in ["user", "human"]:
                        text_parts.append(f"<|start|>user<|message|>{content}")
                    elif role in ["assistant", "gpt"]:
                        text_parts.append(f"<|start|>assistant<|channel|>final<|message|>{content}")
                texts.append("".join(text_parts))
        
        return {"text": texts}
    
    # Process dataset
    processed_dataset = dataset.map(
        format_conversations,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting conversations"
    )
    
    return processed_dataset


# ==============================================================================
# Training Pipeline
# ==============================================================================
def main():
    logger.info("=" * 70)
    logger.info("GPT-OSS 20B KOREAN EDUCATION FINE-TUNING")
    logger.info("=" * 70)
    
    # Log configuration
    logger.info("[STEP 1/8] Logging Configuration")
    logger.info(f"  Log directory: {LOG_DIR}")
    logger.info(f"  Train logs: {TRAIN_LOG_DIR}")
    logger.info(f"  Eval logs: {EVAL_LOG_DIR}")
    logger.info(f"  Checkpoints: {OUTPUT_DIR}")
    logger.info(f"  LoRA output: {LORA_OUTPUT_PATH}")
    
    # Log training configuration (from env_local)
    logger.info("-" * 70)
    logger.info("Configuration (from env_local):")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Dataset: {DATASET_NAME}")
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
    logger.info(f"  Reasoning effort: {INFER_REASONING_EFFORT}")
    logger.info("-" * 70)
    
    # Save config to file
    with open(TRAIN_LOG_DIR / "config.txt", "w") as f:
        f.write(f"MODEL_NAME={MODEL_NAME}\n")
        f.write(f"DATASET_NAME={DATASET_NAME}\n")
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
        f.write(f"REASONING_EFFORT={INFER_REASONING_EFFORT}\n")
    
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
                "dataset_name": DATASET_NAME,
                "max_seq_length": MAX_SEQ_LENGTH,
                "load_in_4bit": LOAD_IN_4BIT,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "eval_ratio": EVAL_RATIO,
                "reasoning_effort": INFER_REASONING_EFFORT,
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
    
    model, tokenizer = FastLanguageModel.from_pretrained(
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
    model = FastLanguageModel.get_peft_model(
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
    
    # ========================================================================
    # Prepare Dataset
    # ========================================================================
    logger.info("[STEP 4/8] Preparing Dataset")
    
    # Load and process Korean education dataset
    processed_dataset = load_korean_education_dataset(tokenizer)
    
    # Shuffle dataset
    processed_dataset = processed_dataset.shuffle(seed=SEED)
    
    # Split into train/eval based on EVAL_RATIO
    total_samples = len(processed_dataset)
    eval_size = max(1, int(total_samples * EVAL_RATIO))
    
    eval_dataset = processed_dataset.select(range(eval_size))
    train_dataset = processed_dataset.select(range(eval_size, total_samples))
    
    logger.info(f"  Train dataset: {len(train_dataset)} samples")
    logger.info(f"  Eval dataset: {len(eval_dataset)} samples ({EVAL_RATIO:.0%})")
    
    # Log sample for verification
    if len(train_dataset) > 0:
        sample_text = train_dataset[0]["text"][:500]
        logger.info(f"  Sample (first 500 chars): {sample_text}...")
    
    # Save dataset info to eval directory
    with open(EVAL_LOG_DIR / "dataset_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Eval samples: {len(eval_dataset)}\n")
        f.write(f"Eval ratio: {EVAL_RATIO:.0%}\n")
    
    # ========================================================================
    # Setup Trainer
    # ========================================================================
    logger.info("[STEP 5/8] Setting up Trainer")
    
    # Configure training args based on epochs vs max_steps
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
        # Use max_steps mode
        sft_config_args["max_steps"] = MAX_STEPS
        sft_config_args["eval_strategy"] = "steps"
        sft_config_args["eval_steps"] = 20
        sft_config_args["save_strategy"] = "steps"
        sft_config_args["save_steps"] = 50
    else:
        # Use epochs mode
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
    
    # Apply train_on_responses_only for GPT-OSS format
    logger.info("  Applying train_on_responses_only for GPT-OSS format...")
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(trainer, **GPT_OSS_KWARGS)
        logger.info("  train_on_responses_only applied successfully")
    except Exception as e:
        logger.warning(f"  Could not apply train_on_responses_only: {e}")
    
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
    
    # Save training stats to file
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
    logger.info("INFERENCE EXAMPLES")
    logger.info("=" * 70)
    
    # Test with Korean education question
    test_questions = [
        "대한민국의 수도는 어디인가요?",
        "피타고라스 정리에 대해 설명해주세요.",
        "광합성 과정을 간단히 설명해주세요.",
    ]
    
    FastLanguageModel.for_inference(model)
    
    for i, question in enumerate(test_questions[:1]):  # Test with first question only
        logger.info(f"Inference Example {i+1}")
        logger.info(f"  Question: {question}")
        logger.info(f"  Config: reasoning_effort={INFER_REASONING_EFFORT}, temp={INFER_TEMPERATURE}")
        
        messages = [{"role": "user", "content": question}]
        
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=INFER_REASONING_EFFORT,
            ).to("cuda")
            
            print("\n🤖 Response:")
            outputs = model.generate(
                **inputs,
                max_new_tokens=INFER_MAX_TOKENS,
                temperature=INFER_TEMPERATURE,
                top_p=INFER_TOP_P,
                top_k=INFER_TOP_K,
                streamer=TextStreamer(tokenizer, skip_prompt=True),
            )
            print("\n")
        except Exception as e:
            logger.warning(f"  Inference failed: {e}")
    
    # ========================================================================
    # Finish
    # ========================================================================
    if WNB_API_KEY:
        wandb.finish()
    
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"All logs saved to: {LOG_DIR}")
    logger.info(f"LoRA model saved to: {LORA_OUTPUT_PATH}")
    logger.info("=" * 70)
    
    return str(LORA_OUTPUT_PATH)


if __name__ == "__main__":
    main()

