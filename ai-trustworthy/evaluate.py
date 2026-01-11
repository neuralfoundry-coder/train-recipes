# -*- coding: utf-8 -*-
"""
AI Trustworthiness Evaluation Script
- Evaluate fine-tuned model on trustworthiness benchmark
- Generate comparison reports (base vs fine-tuned)
- Categories: Helpfulness, Harmlessness, Honesty
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


# ==============================================================================
# Load Environment Configuration
# ==============================================================================
def load_env_local():
    """Load environment variables from env_local file."""
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


ENV_LOCAL = load_env_local()


class TrustworthinessEvaluator:
    """Evaluate AI Trustworthiness benchmark."""
    
    def __init__(
        self,
        lora_path: str,
        base_model: str = None,
        gpu_id: str = "0",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        sample_count: int = 100,
        seed: int = 42,
    ):
        self.lora_path = lora_path
        self.base_model = base_model or get_env("MODEL_NAME", "unsloth/gemma-3-1b-it", ENV_LOCAL)
        self.gpu_id = gpu_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.sample_count = sample_count
        self.seed = seed
        
        self.hf_token = get_env("HF_TOKEN", None, ENV_LOCAL)
        self.dataset_name = get_env("DATASET_NAME", "neuralfoundry-coder/korean-llm-trustworthiness-benchmark-full", ENV_LOCAL)
        self.chat_template = get_env("CHAT_TEMPLATE", "gemma3", ENV_LOCAL)
        self.max_seq_length = get_env("MAX_SEQ_LENGTH", 4096, ENV_LOCAL, int)
        
        self.model = None
        self.tokenizer = None
        self.results = []
        
    def load_model(self):
        """Load the fine-tuned model."""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template
        
        print("\n" + "=" * 60)
        print("🔬 Loading Model for Evaluation")
        print("=" * 60)
        print(f"📁 LoRA path: {self.lora_path}")
        print(f"📁 Base model: {self.base_model}")
        print(f"🎮 GPU: {self.gpu_id}")
        print()
        
        # Load fine-tuned model
        print("⏳ Loading fine-tuned model...")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.lora_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            token=self.hf_token,
        )
        
        # Apply chat template
        self.tokenizer = get_chat_template(self.tokenizer, chat_template=self.chat_template)
        
        # Enable inference mode
        FastModel.for_inference(self.model)
        
        print("✅ Model loaded successfully!")
        print()
        
    def load_evaluation_data(self, subset: str = "sft_instruction") -> List[Dict]:
        """Load evaluation data from dataset."""
        from datasets import load_dataset
        
        print(f"📊 Loading evaluation data: {self.dataset_name}/{subset}")
        
        dataset = load_dataset(self.dataset_name, split=subset, token=self.hf_token)
        
        # Sample data
        random.seed(self.seed)
        if len(dataset) > self.sample_count:
            indices = random.sample(range(len(dataset)), self.sample_count)
            data = [dataset[i] for i in indices]
        else:
            data = list(dataset)
        
        print(f"   Loaded {len(data)} samples")
        return data
        
    def generate_response(self, prompt: str) -> str:
        """Generate response for a prompt."""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        if text.startswith('<bos>'):
            text = text.removeprefix('<bos>')
        
        outputs = self.model.generate(
            **self.tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
        )
        
        generated_ids = outputs[0][len(self.tokenizer(text, return_tensors="pt")["input_ids"][0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
        
    def evaluate_sft(self) -> Dict:
        """Evaluate on SFT instruction dataset."""
        from tqdm import tqdm
        
        print("\n" + "=" * 60)
        print("📝 Evaluating SFT Instruction Dataset")
        print("=" * 60)
        
        data = self.load_evaluation_data("sft_instruction")
        
        results = {
            "total": len(data),
            "by_category": {},
            "samples": [],
        }
        
        for item in tqdm(data, desc="Evaluating"):
            instruction = item.get("instruction", "")
            expected = item.get("output", "")
            category = item.get("category", "unknown")
            
            # Generate response
            generated = self.generate_response(instruction)
            
            # Store result
            sample = {
                "instruction": instruction,
                "expected": expected,
                "generated": generated,
                "category": category,
            }
            results["samples"].append(sample)
            
            # Aggregate by category
            if category not in results["by_category"]:
                results["by_category"][category] = {"count": 0, "samples": []}
            results["by_category"][category]["count"] += 1
            results["by_category"][category]["samples"].append(sample)
        
        return results
        
    def evaluate_fact_checking(self) -> Dict:
        """Evaluate on fact checking dataset."""
        from tqdm import tqdm
        
        print("\n" + "=" * 60)
        print("🔍 Evaluating Fact Checking Dataset")
        print("=" * 60)
        
        data = self.load_evaluation_data("fact_checking")
        
        results = {
            "total": len(data),
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "samples": [],
        }
        
        for item in tqdm(data, desc="Evaluating"):
            instruction = item.get("instruction", "")
            expected = item.get("output", "")
            is_correct = item.get("is_correct", False)
            
            # Generate response
            generated = self.generate_response(instruction)
            
            # Check if model's judgment matches expected
            model_says_correct = "정답" in generated.lower() or "올바" in generated.lower()
            model_says_incorrect = "오답" in generated.lower() or "틀" in generated.lower()
            
            # Simple accuracy check
            if is_correct and model_says_correct:
                results["correct"] += 1
                match = True
            elif not is_correct and model_says_incorrect:
                results["correct"] += 1
                match = True
            else:
                results["incorrect"] += 1
                match = False
            
            sample = {
                "instruction": instruction,
                "expected": expected,
                "generated": generated,
                "is_correct": is_correct,
                "model_match": match,
            }
            results["samples"].append(sample)
        
        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
        
        return results
        
    def generate_html_report(self, sft_results: Dict, fact_results: Dict, output_path: Path):
        """Generate HTML evaluation report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>AI Trustworthiness Evaluation Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Noto Sans KR', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 3px solid #4a90d9; padding-bottom: 10px; }}
        h2 {{ color: #4a90d9; margin-top: 30px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; padding: 10px 20px; margin: 5px; background: #4a90d9; color: white; border-radius: 4px; }}
        .metric-label {{ font-size: 12px; opacity: 0.8; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: bold; }}
        tr:hover {{ background: #f5f5f5; }}
        .sample {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #4a90d9; }}
        .sample-instruction {{ font-weight: bold; color: #333; }}
        .sample-expected {{ color: #28a745; margin: 10px 0; }}
        .sample-generated {{ color: #007bff; margin: 10px 0; }}
        .category-badge {{ display: inline-block; padding: 4px 12px; background: #e9ecef; border-radius: 12px; font-size: 12px; margin: 2px; }}
        .success {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .info {{ background: #e7f3ff; padding: 15px; border-radius: 4px; margin: 10px 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 AI Trustworthiness Evaluation Report</h1>
    <div class="info">
        <strong>Evaluation Time:</strong> {timestamp}<br>
        <strong>Model:</strong> {self.lora_path}<br>
        <strong>Base Model:</strong> {self.base_model}<br>
        <strong>Sample Count:</strong> {self.sample_count}
    </div>
    
    <h2>📊 Overall Summary</h2>
    <div class="card">
        <div class="metric">
            <div class="metric-label">SFT Samples</div>
            <div class="metric-value">{sft_results['total']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Fact Check Accuracy</div>
            <div class="metric-value">{fact_results['accuracy']*100:.1f}%</div>
        </div>
    </div>
    
    <h2>📝 SFT Instruction Results</h2>
    <div class="card">
        <h3>Category Distribution</h3>
        <table>
            <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>"""
        
        for cat, info in sft_results.get("by_category", {}).items():
            pct = info["count"] / sft_results["total"] * 100 if sft_results["total"] > 0 else 0
            html += f"""
            <tr>
                <td><span class="category-badge">{cat}</span></td>
                <td>{info['count']}</td>
                <td>{pct:.1f}%</td>
            </tr>"""
        
        html += """
        </table>
        
        <h3>Sample Outputs (First 10)</h3>"""
        
        for i, sample in enumerate(sft_results.get("samples", [])[:10]):
            html += f"""
        <div class="sample">
            <div class="sample-instruction">📋 {sample['instruction'][:200]}...</div>
            <div class="sample-expected"><strong>Expected:</strong> {sample['expected'][:200]}...</div>
            <div class="sample-generated"><strong>Generated:</strong> {sample['generated'][:200]}...</div>
            <span class="category-badge">{sample['category']}</span>
        </div>"""
        
        html += f"""
    </div>
    
    <h2>🔍 Fact Checking Results</h2>
    <div class="card">
        <div class="metric">
            <div class="metric-label">Total</div>
            <div class="metric-value">{fact_results['total']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Correct</div>
            <div class="metric-value success">{fact_results['correct']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Incorrect</div>
            <div class="metric-value fail">{fact_results['incorrect']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{fact_results['accuracy']*100:.1f}%</div>
        </div>
        
        <h3>Sample Outputs (First 10)</h3>"""
        
        for i, sample in enumerate(fact_results.get("samples", [])[:10]):
            match_class = "success" if sample.get('model_match') else "fail"
            match_text = "✓ Match" if sample.get('model_match') else "✗ Mismatch"
            html += f"""
        <div class="sample">
            <div class="sample-instruction">📋 {sample['instruction'][:200]}...</div>
            <div class="sample-expected"><strong>Expected:</strong> {sample['expected'][:100]}...</div>
            <div class="sample-generated"><strong>Generated:</strong> {sample['generated'][:200]}...</div>
            <span class="{match_class}">{match_text}</span>
        </div>"""
        
        html += """
    </div>
    
    <div class="info" style="margin-top: 30px;">
        <p>Generated by AI Trustworthiness Evaluation Script</p>
        <p>Dataset: Korean LLM Trustworthiness Benchmark</p>
    </div>
</div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"📄 HTML report saved: {output_path}")
        
    def run_evaluation(self, output_dir: Path = None):
        """Run full evaluation pipeline."""
        # Setup output directory
        if output_dir is None:
            output_dir = Path(__file__).parent / "eval_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.load_model()
        
        # Run evaluations
        sft_results = self.evaluate_sft()
        fact_results = self.evaluate_fact_checking()
        
        # Save raw results
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "lora_path": self.lora_path,
                "base_model": self.base_model,
                "sample_count": self.sample_count,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
            },
            "sft": sft_results,
            "fact_checking": fact_results,
        }
        
        json_path = output_dir / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"📄 JSON results saved: {json_path}")
        
        # Generate HTML report
        html_path = output_dir / "report.html"
        self.generate_html_report(sft_results, fact_results, html_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        print(f"SFT Samples: {sft_results['total']}")
        print(f"Categories: {list(sft_results.get('by_category', {}).keys())}")
        print(f"Fact Check Accuracy: {fact_results['accuracy']*100:.1f}%")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="AI Trustworthiness Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--lora-path", "-l",
        type=str,
        required=True,
        help="Path to fine-tuned LoRA model"
    )
    parser.add_argument(
        "--gpu", "-g",
        type=str,
        default=get_env("EVAL_GPU_ID", "0", ENV_LOCAL),
        help="GPU device ID"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=get_env("EVAL_SAMPLE_COUNT", 100, ENV_LOCAL, int),
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=get_env("EVAL_MAX_NEW_TOKENS", 512, ENV_LOCAL, int),
        help="Max new tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=get_env("EVAL_TEMPERATURE", 0.7, ENV_LOCAL, float),
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=get_env("EVAL_TOP_P", 0.9, ENV_LOCAL, float),
        help="Top-p sampling"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=get_env("EVAL_TOP_K", 50, ENV_LOCAL, int),
        help="Top-k sampling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=get_env("EVAL_SEED", 42, ENV_LOCAL, int),
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    evaluator = TrustworthinessEvaluator(
        lora_path=args.lora_path,
        gpu_id=args.gpu,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sample_count=args.samples,
        seed=args.seed,
    )
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    evaluator.run_evaluation(output_dir)


if __name__ == "__main__":
    main()

