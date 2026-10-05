"""Inference CLI demonstrating adapter swapping and text generation."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import clear_gpu_memory, get_gpu_memory_stats, setup_environment


def parse_args():
    parser = argparse.ArgumentParser(description="Inference CLI for LoRA and LoRA+ models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model name or path")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA or LoRA+ adapter directory")
    parser.add_argument("--prompt", type=str, default="Explain the difference between standard LoRA and LoRA+.", help="Instruction prompt")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt loop")
    return parser.parse_args()


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    formatted_prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n### Response:\n"
    )
    device = next(model.parameters()).device
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    setup_environment()
    clear_gpu_memory()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  LoRA / LoRA+ Inference Engine")
    print("=" * 60)
    print(f"  Base Model: {args.model}")
    print(f"  Adapter:    {args.adapter or 'None (Base Model Only)'}")
    print(f"  Device:     {device}")
    print("=" * 60)

    # 1. Load Base Model
    cache_dir = r"D:\Lora++\.cache\huggingface\hub"
    print(f"Loading base model '{args.model}'...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    tokenizer_path = args.adapter if (args.adapter and os.path.exists(os.path.join(args.adapter, "tokenizer_config.json"))) else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    # 2. Attach adapter if specified
    if args.adapter and os.path.exists(args.adapter):
        print(f"Attaching fine-tuned adapter from '{args.adapter}'...")
        model = PeftModel.from_pretrained(base_model, args.adapter)
        print("Adapter attached successfully.")
    else:
        model = base_model
        print("Running in zero-shot base model mode.")

    mem = get_gpu_memory_stats()
    print(f"Inference VRAM Allocated: {mem['allocated_mb']} MB")

    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to end.")
        while True:
            try:
                user_prompt = input("\nPrompt: ").strip()
                if user_prompt.lower() in ["exit", "quit"]:
                    break
                if not user_prompt:
                    continue
                response = generate(model, tokenizer, user_prompt, args.max_new_tokens, args.temperature)
                print(f"\nResponse:\n{response}")
            except KeyboardInterrupt:
                break
    else:
        print(f"\nInstruction:\n{args.prompt}\n")
        response = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
        print(f"Model Response:\n{response}\n")

    print("\nNote: LoRA+ modifies the optimization dynamics during training (different learning rates for A & B);")
    print("at inference time, the adapter architecture remains standard low-rank matrices W' = W + BA.")


if __name__ == "__main__":
    main()
