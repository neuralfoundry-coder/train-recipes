# -*- coding: utf-8 -*-
"""
GPT-OSS 20B Interactive Chat Console
- Load fine-tuned LoRA model and chat interactively
- Supports GPT-OSS reasoning effort levels (low, medium, high)
- Configuration loaded from env_local
"""

import os
import sys
import argparse
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


# Load env_local first
ENV_LOCAL = load_env_local()

# GPU Configuration (must be set before torch import)
DEFAULT_GPU_ID = get_env("INFER_GPU_ID", "1", ENV_LOCAL)


class ChatConsole:
    """Interactive chat console for GPT-OSS model."""
    
    def __init__(self, model_path: str = None, base_model: str = None, 
                 max_seq_length: int = None, gpu_id: str = None):
        # Load configuration from env_local
        self.env_local = ENV_LOCAL
        
        # Model configuration
        self.model_path = model_path or "lora_model"
        self.base_model = base_model or get_env("MODEL_NAME", "unsloth/gpt-oss-20b-unsloth-bnb-4bit", self.env_local)
        self.max_seq_length = max_seq_length or get_env("MAX_SEQ_LENGTH", 2048, self.env_local, int)
        
        # GPU configuration
        self.gpu_id = gpu_id or DEFAULT_GPU_ID
        
        # HF Token
        self.hf_token = get_env("HF_TOKEN", None, self.env_local)
        
        # GPT-OSS specific: Reasoning effort levels
        self.reasoning_effort = get_env("INFER_REASONING_EFFORT", "medium", self.env_local)
        self.REASONING_LEVELS = ["low", "medium", "high"]
        
        # Generation settings from env_local
        self.generation_config = {
            "temperature": get_env("INFER_TEMPERATURE", 0.7, self.env_local, float),
            "top_p": get_env("INFER_TOP_P", 0.9, self.env_local, float),
            "top_k": get_env("INFER_TOP_K", 50, self.env_local, int),
        }
        
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.max_new_tokens = get_env("INFER_MAX_TOKENS", 1024, self.env_local, int)
        
    def load_model(self):
        """Load the fine-tuned model."""
        # Set GPU before import
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        
        # Now import heavy modules
        from unsloth import FastLanguageModel
        
        print("\n" + "=" * 60)
        print("🚀 Loading GPT-OSS 20B Fine-tuned Model")
        print("=" * 60)
        
        # Resolve model path
        model_path = Path(self.model_path)
        if not model_path.is_absolute():
            script_dir = Path(__file__).parent
            model_path = script_dir / self.model_path
        
        if not model_path.exists():
            # Try to find latest log directory
            logs_dir = Path(__file__).parent / "logs"
            if logs_dir.exists():
                log_dirs = sorted(logs_dir.iterdir(), reverse=True)
                for log_dir in log_dirs:
                    potential_path = log_dir / "lora_model"
                    if potential_path.exists():
                        model_path = potential_path
                        print(f"📁 Found model in: {model_path}")
                        break
        
        if not model_path.exists():
            print(f"❌ Model not found at: {model_path}")
            print("   Please train a model first or specify correct path with --model-path")
            sys.exit(1)
        
        print(f"📁 Model path: {model_path}")
        print(f"📁 Base model: {self.base_model}")
        print(f"📁 Max sequence length: {self.max_seq_length}")
        print(f"🎮 GPU: {self.gpu_id}")
        print(f"🧠 Reasoning effort: {self.reasoning_effort}")
        print()
        
        # Load model
        print("⏳ Loading model... (this may take a few minutes)")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            token=self.hf_token,
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        
        print("✅ Model loaded successfully!")
        print()
        
    def print_help(self):
        """Print help message."""
        print("\n" + "=" * 60)
        print("📖 Available Commands")
        print("=" * 60)
        print("  /help, /h      - Show this help message")
        print("  /exit, /quit   - Exit the chat console")
        print("  /clear         - Clear conversation history")
        print(f"  /reasoning     - Change reasoning effort (current: {self.reasoning_effort})")
        print("                   Options: low, medium, high")
        print("  /mode          - Show current mode settings")
        print(f"  /tokens N      - Set max new tokens (current: {self.max_new_tokens})")
        print("  /history       - Show conversation history")
        print("  /single        - Single turn mode (clear history after each response)")
        print("  /multi         - Multi turn mode (keep history)")
        print("=" * 60 + "\n")
        
    def print_mode(self):
        """Print current mode settings."""
        print("\n📊 Current Settings (from env_local):")
        print(f"   Reasoning effort: {self.reasoning_effort} 🧠")
        print(f"   Temperature: {self.generation_config['temperature']}")
        print(f"   Top-p: {self.generation_config['top_p']}")
        print(f"   Top-k: {self.generation_config['top_k']}")
        print(f"   Max new tokens: {self.max_new_tokens}")
        print(f"   History length: {len(self.conversation_history)} turns\n")
        
    def change_reasoning(self, level: str = None):
        """Change reasoning effort level."""
        if level is None:
            # Cycle through levels
            current_idx = self.REASONING_LEVELS.index(self.reasoning_effort)
            next_idx = (current_idx + 1) % len(self.REASONING_LEVELS)
            self.reasoning_effort = self.REASONING_LEVELS[next_idx]
        elif level.lower() in self.REASONING_LEVELS:
            self.reasoning_effort = level.lower()
        else:
            print(f"\n❌ Invalid reasoning level: {level}")
            print(f"   Options: {', '.join(self.REASONING_LEVELS)}\n")
            return
            
        print(f"\n🧠 Reasoning effort changed to: {self.reasoning_effort}")
        print("   low    = Fast responses, minimal reasoning")
        print("   medium = Balanced performance and speed")
        print("   high   = Best reasoning, slower responses\n")
        
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("\n🗑️  Conversation history cleared.\n")
        
    def show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("\n📜 No conversation history.\n")
            return
            
        print("\n📜 Conversation History:")
        print("-" * 40)
        for i, msg in enumerate(self.conversation_history):
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"  [{i+1}] {role}: {content}")
        print("-" * 40 + "\n")
        
    def generate_response(self, user_input: str):
        """Generate response for user input."""
        from transformers import TextStreamer
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Prepare messages
        messages = self.conversation_history.copy()
        
        # Apply chat template with GPT-OSS reasoning effort
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=self.reasoning_effort,
            ).to("cuda")
        except Exception as e:
            # Fallback without reasoning_effort if not supported
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")
        
        # Generate
        print(f"\n🤖 Assistant (reasoning: {self.reasoning_effort}):")
        print()
        
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.generation_config["temperature"],
            top_p=self.generation_config["top_p"],
            top_k=self.generation_config["top_k"],
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Extract assistant response
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        print()
        
    def run(self):
        """Run the interactive chat console."""
        # Load model
        self.load_model()
        
        # Print welcome message
        print("=" * 60)
        print("💬 GPT-OSS 20B Korean Education Chat Console")
        print("=" * 60)
        print("Type your message and press Enter to chat.")
        print("Type /help for available commands.")
        print()
        print("📊 Generation Config (from env_local):")
        print(f"   Reasoning effort: {self.reasoning_effort}")
        print(f"   Temperature: {self.generation_config['temperature']}")
        print(f"   Top-p: {self.generation_config['top_p']}")
        print(f"   Top-k: {self.generation_config['top_k']}")
        print(f"   Max tokens: {self.max_new_tokens}")
        print("=" * 60 + "\n")
        
        single_turn = False
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.lower().split()
                    cmd = parts[0]
                    
                    if cmd in ["/exit", "/quit"]:
                        print("\n👋 Goodbye!\n")
                        break
                    elif cmd in ["/help", "/h"]:
                        self.print_help()
                    elif cmd == "/clear":
                        self.clear_history()
                    elif cmd == "/reasoning":
                        level = parts[1] if len(parts) > 1 else None
                        self.change_reasoning(level)
                    elif cmd == "/mode":
                        self.print_mode()
                    elif cmd == "/history":
                        self.show_history()
                    elif cmd == "/single":
                        single_turn = True
                        print("\n🔄 Single turn mode enabled (history cleared after each response).\n")
                    elif cmd == "/multi":
                        single_turn = False
                        print("\n🔄 Multi turn mode enabled (history preserved).\n")
                    elif cmd == "/tokens":
                        try:
                            tokens = int(parts[1])
                            self.max_new_tokens = tokens
                            print(f"\n🔢 Max new tokens set to: {tokens}\n")
                        except (IndexError, ValueError):
                            print("\n❌ Usage: /tokens N (e.g., /tokens 2048)\n")
                    else:
                        print(f"\n❌ Unknown command: {cmd}. Type /help for available commands.\n")
                    continue
                
                # Generate response
                self.generate_response(user_input)
                
                # Clear history if single turn mode
                if single_turn:
                    self.conversation_history = []
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                continue


def main():
    # Load env_local for defaults in help text
    env_local = ENV_LOCAL
    default_gpu = get_env("INFER_GPU_ID", "1", env_local)
    default_tokens = get_env("INFER_MAX_TOKENS", 1024, env_local, int)
    default_model = get_env("MODEL_NAME", "unsloth/gpt-oss-20b-unsloth-bnb-4bit", env_local)
    default_reasoning = get_env("INFER_REASONING_EFFORT", "medium", env_local)
    
    parser = argparse.ArgumentParser(
        description="GPT-OSS 20B Interactive Chat Console",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Configuration is loaded from env_local file.

Current defaults (from env_local):
  GPU ID:           {default_gpu}
  Max tokens:       {default_tokens}
  Base model:       {default_model}
  Reasoning effort: {default_reasoning}

Reasoning Effort Levels:
  low    - Fast responses, minimal reasoning
  medium - Balanced performance and speed  
  high   - Best reasoning, slower responses

Examples:
  # Use default settings from env_local
  python chat_console.py
  
  # Specify model path
  python chat_console.py --model-path ./logs/gpt-oss-20b_.../lora_model
  
  # Start with high reasoning effort
  python chat_console.py --reasoning high
  
  # Override env_local settings
  python chat_console.py --gpu 1 --max-tokens 2048
        """
    )
    
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA model directory"
    )
    parser.add_argument(
        "--base-model", "-b",
        type=str,
        default=None,
        help=f"Base model name (default from env_local: {default_model})"
    )
    parser.add_argument(
        "--max-seq-length", "-s",
        type=int,
        default=None,
        help="Maximum sequence length (default from env_local)"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=None,
        help=f"Maximum new tokens to generate (default from env_local: {default_tokens})"
    )
    parser.add_argument(
        "--reasoning", "-r",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help=f"Reasoning effort level (default from env_local: {default_reasoning})"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help=f"GPU device ID to use (default from env_local: {default_gpu})"
    )
    
    args = parser.parse_args()
    
    # Create console with args (None values will use env_local defaults)
    console = ChatConsole(
        model_path=args.model_path,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        gpu_id=args.gpu,
    )
    
    # Override settings if specified
    if args.max_tokens:
        console.max_new_tokens = args.max_tokens
    if args.reasoning:
        console.reasoning_effort = args.reasoning
        
    console.run()


if __name__ == "__main__":
    main()

