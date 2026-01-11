# -*- coding: utf-8 -*-
"""
Model Comparison Tool - Compare Base Model vs Fine-tuned Model
Generates HTML report with side-by-side comparison of model outputs.
"""

import os
import sys
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import html


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
                                get_env("COMPARE_GPU_ID", 
                                        get_env("INFER_GPU_ID", "0", ENV_LOCAL), 
                                        ENV_LOCAL),
                                ENV_LOCAL)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Now import torch and other heavy modules
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from tqdm import tqdm


# ==============================================================================
# Configuration
# ==============================================================================
HF_TOKEN = get_env("HF_TOKEN", None, ENV_LOCAL)
MODEL_NAME = get_env("MODEL_NAME", "unsloth/Qwen3-14B", ENV_LOCAL)
MODEL_SHORT_NAME = get_env("MODEL_SHORT_NAME", "qwen3-14b", ENV_LOCAL)
MAX_SEQ_LENGTH = get_env("MAX_SEQ_LENGTH", 32768, ENV_LOCAL, int)
LOAD_IN_4BIT = get_env("LOAD_IN_4BIT", False, ENV_LOCAL, bool)
LOAD_IN_8BIT = get_env("LOAD_IN_8BIT", False, ENV_LOCAL, bool)

# Comparison Configuration
SAMPLE_COUNT = get_env("COMPARE_SAMPLE_COUNT", 100, ENV_LOCAL, int)
MAX_NEW_TOKENS = get_env("COMPARE_MAX_NEW_TOKENS", 512, ENV_LOCAL, int)
TEMPERATURE = get_env("COMPARE_TEMPERATURE", 0.7, ENV_LOCAL, float)
TOP_P = get_env("COMPARE_TOP_P", 0.8, ENV_LOCAL, float)
TOP_K = get_env("COMPARE_TOP_K", 20, ENV_LOCAL, int)
ENABLE_THINKING = get_env("COMPARE_ENABLE_THINKING", False, ENV_LOCAL, bool)
SEED = get_env("COMPARE_SEED", 42, ENV_LOCAL, int)

# LoRA Model Path (required)
LORA_MODEL_PATH = get_env("COMPARE_LORA_PATH", None, ENV_LOCAL)

# Dataset Configuration
DATASET_SOURCE = get_env("COMPARE_DATASET_SOURCE", "reasoning", ENV_LOCAL)  # "reasoning", "chat", or "both"
CHAT_PERCENTAGE = get_env("TRAIN_CHAT_PERCENTAGE", 0.25, ENV_LOCAL, float)

# Timestamp for unique run identification
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "comparison_reports"


# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class ComparisonResult:
    """Single comparison result between base and fine-tuned model."""
    index: int
    input_text: str
    source: str  # "reasoning" or "chat"
    base_output: str
    finetuned_output: str
    base_time_ms: float
    finetuned_time_ms: float


@dataclass
class ComparisonReport:
    """Complete comparison report metadata."""
    timestamp: str
    base_model: str
    lora_model_path: str
    sample_count: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    enable_thinking: bool
    seed: int
    gpu_id: str
    total_time_seconds: float
    avg_base_time_ms: float
    avg_finetuned_time_ms: float


# ==============================================================================
# Logging Setup
# ==============================================================================
def setup_logging():
    """Setup logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("comparison")


logger = setup_logging()


# ==============================================================================
# Dataset Loading
# ==============================================================================
def load_comparison_dataset(source: str, sample_count: int, seed: int) -> List[Dict[str, Any]]:
    """Load and sample dataset for comparison."""
    logger.info(f"Loading dataset (source: {source}, samples: {sample_count})...")
    
    samples = []
    random.seed(seed)
    
    if source in ["reasoning", "both"]:
        logger.info("  Loading reasoning dataset (OpenMathReasoning-mini)...")
        reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
        reasoning_samples = [
            {"input": item["problem"], "source": "reasoning", "reference": item.get("generated_solution", "")}
            for item in reasoning_dataset
        ]
        if source == "reasoning":
            samples = random.sample(reasoning_samples, min(sample_count, len(reasoning_samples)))
        else:
            # For "both", take proportional samples
            reasoning_count = int(sample_count * (1 - CHAT_PERCENTAGE))
            samples.extend(random.sample(reasoning_samples, min(reasoning_count, len(reasoning_samples))))
    
    if source in ["chat", "both"]:
        logger.info("  Loading chat dataset (FineTome-100k)...")
        chat_dataset = load_dataset("mlabonne/FineTome-100k", split="train")
        chat_dataset = standardize_sharegpt(chat_dataset)
        
        chat_samples = []
        for item in chat_dataset:
            conversations = item.get("conversations", [])
            if conversations and len(conversations) >= 1:
                # Get first user message
                user_msg = None
                for conv in conversations:
                    if conv.get("role") == "user" or conv.get("from") == "human":
                        user_msg = conv.get("content") or conv.get("value", "")
                        break
                if user_msg:
                    chat_samples.append({
                        "input": user_msg,
                        "source": "chat",
                        "reference": ""
                    })
        
        if source == "chat":
            samples = random.sample(chat_samples, min(sample_count, len(chat_samples)))
        else:
            # For "both", take proportional samples
            chat_count = sample_count - len(samples)
            samples.extend(random.sample(chat_samples, min(chat_count, len(chat_samples))))
    
    # Shuffle final samples
    random.shuffle(samples)
    
    logger.info(f"  Total samples: {len(samples)}")
    return samples[:sample_count]


# ==============================================================================
# Model Loading
# ==============================================================================
def load_base_model(model_name: str, max_seq_length: int, load_in_4bit: bool, load_in_8bit: bool):
    """Load the base model without LoRA adapters."""
    logger.info(f"Loading base model: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        token=HF_TOKEN,
    )
    
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model(lora_path: str, max_seq_length: int, load_in_4bit: bool, load_in_8bit: bool):
    """Load the fine-tuned model with LoRA adapters."""
    logger.info(f"Loading fine-tuned model from: {lora_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        token=HF_TOKEN,
    )
    
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ==============================================================================
# Inference
# ==============================================================================
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool
) -> tuple[str, float]:
    """Generate response from model and return output with timing."""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Time the generation
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    end_time.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_time.elapsed_time(end_time)
    
    # Decode output (skip input tokens)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response, elapsed_ms


# ==============================================================================
# HTML Report Generation
# ==============================================================================
def generate_html_report(
    results: List[ComparisonResult],
    report_meta: ComparisonReport,
    output_path: Path
) -> None:
    """Generate HTML comparison report."""
    
    # Calculate statistics
    reasoning_count = sum(1 for r in results if r.source == "reasoning")
    chat_count = sum(1 for r in results if r.source == "chat")
    
    # CSS Styles
    css = """
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-orange: #d29922;
            --accent-purple: #a371f7;
            --accent-red: #f85149;
            --border-color: #30363d;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 0;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 40px 60px;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 40px 60px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }
        
        .stat-card .label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        .stat-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent-blue);
        }
        
        .stat-card.green .value { color: var(--accent-green); }
        .stat-card.orange .value { color: var(--accent-orange); }
        .stat-card.purple .value { color: var(--accent-purple); }
        
        .config-section {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 40px;
        }
        
        .config-section h3 {
            color: var(--accent-purple);
            margin-bottom: 16px;
            font-size: 1.1rem;
        }
        
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 12px;
        }
        
        .config-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 0.9rem;
        }
        
        .config-item .key {
            color: var(--text-secondary);
        }
        
        .config-item .val {
            color: var(--accent-green);
            font-weight: 500;
        }
        
        .comparison-item {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            margin-bottom: 24px;
            overflow: hidden;
        }
        
        .comparison-header {
            background: var(--bg-tertiary);
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
        }
        
        .comparison-header .index {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--accent-blue);
        }
        
        .comparison-header .source-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
        }
        
        .source-badge.reasoning {
            background: rgba(163, 113, 247, 0.2);
            color: var(--accent-purple);
        }
        
        .source-badge.chat {
            background: rgba(63, 185, 80, 0.2);
            color: var(--accent-green);
        }
        
        .input-section {
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .section-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .section-label .icon {
            font-size: 1rem;
        }
        
        .input-text {
            background: var(--bg-primary);
            padding: 16px;
            border-radius: 8px;
            border-left: 3px solid var(--accent-orange);
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.95rem;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .outputs-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
        }
        
        .output-section {
            padding: 24px;
        }
        
        .output-section.base {
            border-right: 1px solid var(--border-color);
        }
        
        .output-section.base .section-label {
            color: var(--accent-orange);
        }
        
        .output-section.finetuned .section-label {
            color: var(--accent-green);
        }
        
        .output-text {
            background: var(--bg-primary);
            padding: 16px;
            border-radius: 8px;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.9rem;
            max-height: 400px;
            overflow-y: auto;
            line-height: 1.7;
        }
        
        .output-section.base .output-text {
            border-left: 3px solid var(--accent-orange);
        }
        
        .output-section.finetuned .output-text {
            border-left: 3px solid var(--accent-green);
        }
        
        .timing-badge {
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 10px;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            margin-left: auto;
        }
        
        .footer {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            margin-top: 40px;
        }
        
        .nav-buttons {
            position: fixed;
            bottom: 30px;
            right: 30px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .nav-btn {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            transition: all 0.2s;
        }
        
        .nav-btn:hover {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }
        
        @media (max-width: 1200px) {
            .outputs-container {
                grid-template-columns: 1fr;
            }
            
            .output-section.base {
                border-right: none;
                border-bottom: 1px solid var(--border-color);
            }
        }
    </style>
    """
    
    # Generate comparison items HTML
    comparison_items_html = ""
    for result in results:
        comparison_items_html += f"""
        <div class="comparison-item" id="item-{result.index}">
            <div class="comparison-header">
                <span class="index">#{result.index + 1}</span>
                <span class="source-badge {result.source}">{result.source}</span>
            </div>
            <div class="input-section">
                <div class="section-label">
                    <span class="icon">📝</span> Input Prompt
                </div>
                <div class="input-text">{html.escape(result.input_text)}</div>
            </div>
            <div class="outputs-container">
                <div class="output-section base">
                    <div class="section-label">
                        <span class="icon">🔶</span> Base Model Output
                        <span class="timing-badge">{result.base_time_ms:.0f}ms</span>
                    </div>
                    <div class="output-text">{html.escape(result.base_output)}</div>
                </div>
                <div class="output-section finetuned">
                    <div class="section-label">
                        <span class="icon">🟢</span> Fine-tuned Model Output
                        <span class="timing-badge">{result.finetuned_time_ms:.0f}ms</span>
                    </div>
                    <div class="output-text">{html.escape(result.finetuned_output)}</div>
                </div>
            </div>
        </div>
        """
    
    # Generate full HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report - {report_meta.timestamp}</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    {css}
</head>
<body>
    <div class="header">
        <h1>🔬 Model Comparison Report</h1>
        <div class="subtitle">Base Model vs Fine-tuned Model • {report_meta.timestamp}</div>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Total Samples</div>
                <div class="value">{report_meta.sample_count}</div>
            </div>
            <div class="stat-card green">
                <div class="label">Reasoning Samples</div>
                <div class="value">{reasoning_count}</div>
            </div>
            <div class="stat-card orange">
                <div class="label">Chat Samples</div>
                <div class="value">{chat_count}</div>
            </div>
            <div class="stat-card purple">
                <div class="label">Total Time</div>
                <div class="value">{report_meta.total_time_seconds:.1f}s</div>
            </div>
            <div class="stat-card">
                <div class="label">Avg Base Time</div>
                <div class="value">{report_meta.avg_base_time_ms:.0f}ms</div>
            </div>
            <div class="stat-card green">
                <div class="label">Avg Fine-tuned Time</div>
                <div class="value">{report_meta.avg_finetuned_time_ms:.0f}ms</div>
            </div>
        </div>
        
        <div class="config-section">
            <h3>⚙️ Configuration</h3>
            <div class="config-grid">
                <div class="config-item">
                    <span class="key">Base Model</span>
                    <span class="val">{html.escape(report_meta.base_model)}</span>
                </div>
                <div class="config-item">
                    <span class="key">LoRA Path</span>
                    <span class="val">{html.escape(str(report_meta.lora_model_path))}</span>
                </div>
                <div class="config-item">
                    <span class="key">Max New Tokens</span>
                    <span class="val">{report_meta.max_new_tokens}</span>
                </div>
                <div class="config-item">
                    <span class="key">Temperature</span>
                    <span class="val">{report_meta.temperature}</span>
                </div>
                <div class="config-item">
                    <span class="key">Top-P</span>
                    <span class="val">{report_meta.top_p}</span>
                </div>
                <div class="config-item">
                    <span class="key">Top-K</span>
                    <span class="val">{report_meta.top_k}</span>
                </div>
                <div class="config-item">
                    <span class="key">Thinking Mode</span>
                    <span class="val">{report_meta.enable_thinking}</span>
                </div>
                <div class="config-item">
                    <span class="key">Seed</span>
                    <span class="val">{report_meta.seed}</span>
                </div>
                <div class="config-item">
                    <span class="key">GPU</span>
                    <span class="val">{report_meta.gpu_id}</span>
                </div>
            </div>
        </div>
        
        <h2 style="margin-bottom: 24px; color: var(--text-primary);">📊 Comparison Results</h2>
        
        {comparison_items_html}
    </div>
    
    <div class="footer">
        <p>Generated by Model Comparison Tool • {report_meta.timestamp}</p>
    </div>
    
    <div class="nav-buttons">
        <button class="nav-btn" onclick="window.scrollTo({{top: 0, behavior: 'smooth'}})">⬆</button>
        <button class="nav-btn" onclick="window.scrollTo({{top: document.body.scrollHeight, behavior: 'smooth'}})">⬇</button>
    </div>
    
    <script>
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowUp' && e.ctrlKey) {{
                window.scrollTo({{top: 0, behavior: 'smooth'}});
            }} else if (e.key === 'ArrowDown' && e.ctrlKey) {{
                window.scrollTo({{top: document.body.scrollHeight, behavior: 'smooth'}});
            }}
        }});
    </script>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to: {output_path}")


# ==============================================================================
# Main Comparison Function
# ==============================================================================
def run_comparison(
    lora_path: str,
    sample_count: int = 100,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    enable_thinking: bool = False,
    dataset_source: str = "reasoning",
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Path:
    """Run full model comparison and generate HTML report."""
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON TOOL")
    logger.info("=" * 70)
    logger.info(f"  Base Model: {MODEL_NAME}")
    logger.info(f"  LoRA Path: {lora_path}")
    logger.info(f"  Samples: {sample_count}")
    logger.info(f"  Dataset Source: {dataset_source}")
    logger.info(f"  Max New Tokens: {max_new_tokens}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Thinking Mode: {enable_thinking}")
    logger.info("=" * 70)
    
    # Verify LoRA path exists
    lora_path_obj = Path(lora_path)
    if not lora_path_obj.exists():
        raise FileNotFoundError(f"LoRA model path does not exist: {lora_path}")
    
    # Load datasets
    samples = load_comparison_dataset(dataset_source, sample_count, seed)
    
    # Load base model
    logger.info("-" * 70)
    base_model, base_tokenizer = load_base_model(
        MODEL_NAME, MAX_SEQ_LENGTH, LOAD_IN_4BIT, LOAD_IN_8BIT
    )
    logger.info("Base model loaded successfully")
    
    # Generate base model outputs
    logger.info("-" * 70)
    logger.info("Generating base model outputs...")
    base_outputs = []
    for sample in tqdm(samples, desc="Base Model"):
        output, time_ms = generate_response(
            base_model, base_tokenizer, sample["input"],
            max_new_tokens, temperature, top_p, top_k, enable_thinking
        )
        base_outputs.append({"output": output, "time_ms": time_ms})
    
    # Cleanup base model
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()
    logger.info("Base model unloaded")
    
    # Load fine-tuned model
    logger.info("-" * 70)
    finetuned_model, finetuned_tokenizer = load_finetuned_model(
        str(lora_path_obj), MAX_SEQ_LENGTH, LOAD_IN_4BIT, LOAD_IN_8BIT
    )
    logger.info("Fine-tuned model loaded successfully")
    
    # Generate fine-tuned model outputs
    logger.info("-" * 70)
    logger.info("Generating fine-tuned model outputs...")
    finetuned_outputs = []
    for sample in tqdm(samples, desc="Fine-tuned Model"):
        output, time_ms = generate_response(
            finetuned_model, finetuned_tokenizer, sample["input"],
            max_new_tokens, temperature, top_p, top_k, enable_thinking
        )
        finetuned_outputs.append({"output": output, "time_ms": time_ms})
    
    # Cleanup fine-tuned model
    del finetuned_model
    del finetuned_tokenizer
    torch.cuda.empty_cache()
    logger.info("Fine-tuned model unloaded")
    
    # Compile results
    logger.info("-" * 70)
    logger.info("Compiling comparison results...")
    
    results = []
    for i, (sample, base_out, ft_out) in enumerate(zip(samples, base_outputs, finetuned_outputs)):
        results.append(ComparisonResult(
            index=i,
            input_text=sample["input"],
            source=sample["source"],
            base_output=base_out["output"],
            finetuned_output=ft_out["output"],
            base_time_ms=base_out["time_ms"],
            finetuned_time_ms=ft_out["time_ms"],
        ))
    
    end_time = datetime.now()
    total_seconds = (end_time - start_time).total_seconds()
    
    # Calculate averages
    avg_base_time = sum(r.base_time_ms for r in results) / len(results)
    avg_ft_time = sum(r.finetuned_time_ms for r in results) / len(results)
    
    # Create report metadata
    report_meta = ComparisonReport(
        timestamp=RUN_TIMESTAMP,
        base_model=MODEL_NAME,
        lora_model_path=str(lora_path),
        sample_count=sample_count,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
        seed=seed,
        gpu_id=CUDA_VISIBLE_DEVICES,
        total_time_seconds=total_seconds,
        avg_base_time_ms=avg_base_time,
        avg_finetuned_time_ms=avg_ft_time,
    )
    
    # Generate output path
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    output_file = output_dir / f"comparison_{MODEL_SHORT_NAME}_{RUN_TIMESTAMP}.html"
    
    # Generate HTML report
    generate_html_report(results, report_meta, output_file)
    
    # Save JSON data for future analysis
    json_file = output_dir / f"comparison_{MODEL_SHORT_NAME}_{RUN_TIMESTAMP}.json"
    json_data = {
        "metadata": asdict(report_meta),
        "results": [asdict(r) for r in results]
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON data saved to: {json_file}")
    
    logger.info("=" * 70)
    logger.info("COMPARISON COMPLETE")
    logger.info(f"  Total Time: {total_seconds:.1f}s")
    logger.info(f"  Avg Base Model Time: {avg_base_time:.0f}ms")
    logger.info(f"  Avg Fine-tuned Model Time: {avg_ft_time:.0f}ms")
    logger.info(f"  Report: {output_file}")
    logger.info("=" * 70)
    
    return output_file


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare base model vs fine-tuned model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_models.py --lora-path ./logs/qwen3-14b_*/lora_model
  python compare_models.py --lora-path ./lora_model --samples 50 --thinking
  python compare_models.py --lora-path ./lora_model --source both --tokens 1024
        """
    )
    
    parser.add_argument(
        "--lora-path", "-l",
        type=str,
        default=LORA_MODEL_PATH,
        help="Path to the fine-tuned LoRA model directory"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=SAMPLE_COUNT,
        help=f"Number of samples to compare (default: {SAMPLE_COUNT})"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Maximum new tokens to generate (default: {MAX_NEW_TOKENS})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=TOP_P,
        help=f"Top-p sampling (default: {TOP_P})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Top-k sampling (default: {TOP_K})"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        choices=["reasoning", "chat", "both"],
        default=DATASET_SOURCE,
        help=f"Dataset source (default: {DATASET_SOURCE})"
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=ENABLE_THINKING,
        help="Enable thinking mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # Validate LoRA path
    if not args.lora_path:
        parser.error("--lora-path is required. Specify the path to your fine-tuned model.")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        report_path = run_comparison(
            lora_path=args.lora_path,
            sample_count=args.samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_thinking=args.thinking,
            dataset_source=args.source,
            seed=args.seed,
            output_dir=output_dir,
        )
        print(f"\n✅ Report generated: {report_path}")
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()

