# -*- coding: utf-8 -*-
"""
Model Comparison Tool - Compare Base Model vs Fine-tuned Model
Generates HTML report with side-by-side comparison of model outputs.
Designed for GPT-OSS 20B with Korean Education dataset.
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
                                        get_env("INFER_GPU_ID", "1", ENV_LOCAL), 
                                        ENV_LOCAL),
                                ENV_LOCAL)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Now import torch and other heavy modules
import torch
from datasets import load_dataset
from tqdm import tqdm


# ==============================================================================
# Configuration
# ==============================================================================
HF_TOKEN = get_env("HF_TOKEN", None, ENV_LOCAL)
MODEL_NAME = get_env("MODEL_NAME", "unsloth/gpt-oss-20b-unsloth-bnb-4bit", ENV_LOCAL)
MODEL_SHORT_NAME = get_env("MODEL_SHORT_NAME", "gpt-oss-20b", ENV_LOCAL)
DATASET_NAME = get_env("DATASET_NAME", "neuralfoundry-coder/aihub-korean-education-instruct", ENV_LOCAL)
MAX_SEQ_LENGTH = get_env("MAX_SEQ_LENGTH", 2048, ENV_LOCAL, int)
LOAD_IN_4BIT = get_env("LOAD_IN_4BIT", True, ENV_LOCAL, bool)
LOAD_IN_8BIT = get_env("LOAD_IN_8BIT", False, ENV_LOCAL, bool)

# Comparison Configuration
SAMPLE_COUNT = get_env("COMPARE_SAMPLE_COUNT", 50, ENV_LOCAL, int)
MAX_NEW_TOKENS = get_env("COMPARE_MAX_NEW_TOKENS", 512, ENV_LOCAL, int)
TEMPERATURE = get_env("COMPARE_TEMPERATURE", 0.7, ENV_LOCAL, float)
TOP_P = get_env("COMPARE_TOP_P", 0.9, ENV_LOCAL, float)
TOP_K = get_env("COMPARE_TOP_K", 50, ENV_LOCAL, int)
REASONING_EFFORT = get_env("COMPARE_REASONING_EFFORT", "medium", ENV_LOCAL)
SEED = get_env("COMPARE_SEED", 42, ENV_LOCAL, int)

# LoRA Model Path (required)
LORA_MODEL_PATH = get_env("COMPARE_LORA_PATH", None, ENV_LOCAL)

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
    category: str
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
    dataset_name: str
    sample_count: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    reasoning_effort: str
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
def load_comparison_dataset(sample_count: int, seed: int) -> List[Dict[str, Any]]:
    """Load and sample Korean education dataset for comparison."""
    logger.info(f"Loading dataset: {DATASET_NAME}")
    
    # Load dataset from HuggingFace
    dataset = load_dataset(DATASET_NAME, split="train", token=HF_TOKEN)
    logger.info(f"  Total samples: {len(dataset)}")
    
    random.seed(seed)
    
    samples = []
    indices = random.sample(range(len(dataset)), min(sample_count, len(dataset)))
    
    for idx in indices:
        item = dataset[idx]
        conversations = item.get("conversations", [])
        
        # Get first user message
        user_msg = ""
        for conv in conversations:
            role = conv.get("role", conv.get("from", "user"))
            if role in ["user", "human"]:
                user_msg = conv.get("content", conv.get("value", ""))
                break
        
        if user_msg:
            # Try to get category from metadata or use default
            category = item.get("category", item.get("source", "general"))
            samples.append({
                "input": user_msg,
                "category": str(category)[:50] if category else "general",
            })
    
    logger.info(f"  Sampled: {len(samples)} samples")
    return samples


# ==============================================================================
# Model Loading
# ==============================================================================
def load_base_model(model_name: str, max_seq_length: int, load_in_4bit: bool, load_in_8bit: bool):
    """Load the base model without LoRA adapters."""
    from unsloth import FastLanguageModel
    
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
    from unsloth import FastLanguageModel
    
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
    reasoning_effort: str
) -> tuple:
    """Generate response from model and return output with timing."""
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Apply chat template with GPT-OSS reasoning effort
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=reasoning_effort,
        ).to("cuda")
    except Exception:
        # Fallback without reasoning_effort
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
    
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
    
    # Calculate statistics by category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = 0
        categories[r.category] += 1
    
    # CSS Styles - Korean Education themed
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
        
        :root {
            --bg-primary: #0f1419;
            --bg-secondary: #1a2028;
            --bg-tertiary: #242d38;
            --text-primary: #e8eaed;
            --text-secondary: #9aa0a6;
            --accent-blue: #4285f4;
            --accent-green: #34a853;
            --accent-orange: #fbbc04;
            --accent-purple: #a142f4;
            --accent-red: #ea4335;
            --accent-teal: #00bcd4;
            --border-color: #3c4043;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.7;
            padding: 0;
        }
        
        .header {
            background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 40px 60px;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent-teal), var(--accent-purple));
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
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }
        
        .stat-card .label {
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        
        .stat-card .value {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--accent-teal);
        }
        
        .stat-card.green .value { color: var(--accent-green); }
        .stat-card.orange .value { color: var(--accent-orange); }
        .stat-card.purple .value { color: var(--accent-purple); }
        .stat-card.blue .value { color: var(--accent-blue); }
        
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
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 10px;
        }
        
        .config-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 0.85rem;
        }
        
        .config-item .key {
            color: var(--text-secondary);
        }
        
        .config-item .val {
            color: var(--accent-teal);
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
            color: var(--accent-teal);
        }
        
        .comparison-header .category-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            background: rgba(0, 188, 212, 0.2);
            color: var(--accent-teal);
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
            line-height: 1.8;
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
            background: var(--accent-teal);
            border-color: var(--accent-teal);
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
                <span class="category-badge">{html.escape(result.category)}</span>
            </div>
            <div class="input-section">
                <div class="section-label">
                    <span class="icon">📝</span> 입력 프롬프트 (Input)
                </div>
                <div class="input-text">{html.escape(result.input_text)}</div>
            </div>
            <div class="outputs-container">
                <div class="output-section base">
                    <div class="section-label">
                        <span class="icon">🔶</span> 베이스 모델 출력
                        <span class="timing-badge">{result.base_time_ms:.0f}ms</span>
                    </div>
                    <div class="output-text">{html.escape(result.base_output)}</div>
                </div>
                <div class="output-section finetuned">
                    <div class="section-label">
                        <span class="icon">🟢</span> 파인튜닝 모델 출력
                        <span class="timing-badge">{result.finetuned_time_ms:.0f}ms</span>
                    </div>
                    <div class="output-text">{html.escape(result.finetuned_output)}</div>
                </div>
            </div>
        </div>
        """
    
    # Category stats
    category_stats_html = ""
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:6]:
        category_stats_html += f"""
        <div class="stat-card">
            <div class="label">{html.escape(cat[:15])}</div>
            <div class="value">{count}</div>
        </div>
        """
    
    # Generate full HTML
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-OSS 모델 비교 리포트 - {report_meta.timestamp}</title>
    {css}
</head>
<body>
    <div class="header">
        <h1>🎓 GPT-OSS 20B 한국어 교육 모델 비교</h1>
        <div class="subtitle">Base Model vs Fine-tuned Model • {report_meta.timestamp}</div>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">총 샘플 수</div>
                <div class="value">{report_meta.sample_count}</div>
            </div>
            <div class="stat-card green">
                <div class="label">총 소요 시간</div>
                <div class="value">{report_meta.total_time_seconds:.1f}s</div>
            </div>
            <div class="stat-card orange">
                <div class="label">Base 평균</div>
                <div class="value">{report_meta.avg_base_time_ms:.0f}ms</div>
            </div>
            <div class="stat-card blue">
                <div class="label">Fine-tuned 평균</div>
                <div class="value">{report_meta.avg_finetuned_time_ms:.0f}ms</div>
            </div>
            <div class="stat-card purple">
                <div class="label">Reasoning</div>
                <div class="value">{report_meta.reasoning_effort}</div>
            </div>
        </div>
        
        <div class="config-section">
            <h3>⚙️ 설정 정보</h3>
            <div class="config-grid">
                <div class="config-item">
                    <span class="key">Base Model</span>
                    <span class="val">{html.escape(report_meta.base_model.split('/')[-1])}</span>
                </div>
                <div class="config-item">
                    <span class="key">LoRA Path</span>
                    <span class="val">{html.escape(str(Path(report_meta.lora_model_path).name))}</span>
                </div>
                <div class="config-item">
                    <span class="key">Dataset</span>
                    <span class="val">{html.escape(report_meta.dataset_name.split('/')[-1])}</span>
                </div>
                <div class="config-item">
                    <span class="key">Max Tokens</span>
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
                    <span class="key">Reasoning Effort</span>
                    <span class="val">{report_meta.reasoning_effort}</span>
                </div>
                <div class="config-item">
                    <span class="key">GPU</span>
                    <span class="val">{report_meta.gpu_id}</span>
                </div>
            </div>
        </div>
        
        <h2 style="margin-bottom: 24px; color: var(--text-primary);">📊 비교 결과</h2>
        
        {comparison_items_html}
    </div>
    
    <div class="footer">
        <p>Generated by GPT-OSS Model Comparison Tool • {report_meta.timestamp}</p>
        <p>Dataset: {html.escape(report_meta.dataset_name)}</p>
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
    sample_count: int = 50,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    reasoning_effort: str = "medium",
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Path:
    """Run full model comparison and generate HTML report."""
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("GPT-OSS 20B MODEL COMPARISON TOOL")
    logger.info("=" * 70)
    logger.info(f"  Base Model: {MODEL_NAME}")
    logger.info(f"  LoRA Path: {lora_path}")
    logger.info(f"  Dataset: {DATASET_NAME}")
    logger.info(f"  Samples: {sample_count}")
    logger.info(f"  Max New Tokens: {max_new_tokens}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Reasoning Effort: {reasoning_effort}")
    logger.info("=" * 70)
    
    # Verify LoRA path exists
    lora_path_obj = Path(lora_path)
    if not lora_path_obj.exists():
        raise FileNotFoundError(f"LoRA model path does not exist: {lora_path}")
    
    # Load datasets
    samples = load_comparison_dataset(sample_count, seed)
    
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
            max_new_tokens, temperature, top_p, top_k, reasoning_effort
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
            max_new_tokens, temperature, top_p, top_k, reasoning_effort
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
            category=sample["category"],
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
        dataset_name=DATASET_NAME,
        sample_count=sample_count,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        reasoning_effort=reasoning_effort,
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
        description="Compare GPT-OSS base model vs fine-tuned model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_models.py --lora-path ./logs/gpt-oss-20b_*/lora_model
  python compare_models.py --lora-path ./lora_model --samples 50 --reasoning high
  python compare_models.py --lora-path ./lora_model --tokens 1024
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
        "--reasoning", "-r",
        type=str,
        choices=["low", "medium", "high"],
        default=REASONING_EFFORT,
        help=f"Reasoning effort level (default: {REASONING_EFFORT})"
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
            reasoning_effort=args.reasoning,
            seed=args.seed,
            output_dir=output_dir,
        )
        print(f"\n✅ Report generated: {report_path}")
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()

