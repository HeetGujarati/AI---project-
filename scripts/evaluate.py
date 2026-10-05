"""Standalone evaluation CLI to evaluate checkpoints and calculate perplexity."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import get_tokenized_dataset
from src.evaluation import compute_eval_loss_and_perplexity, generate_sample_responses
from src.utils import clear_gpu_memory, print_environment_banner, setup_environment


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LoRA/LoRA+ models")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model name")
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter checkpoint directory")
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned", help="Dataset name")
    parser.add_argument("--max_eval_samples", type=int, default=100, help="Max eval samples")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment()
    print_environment_banner()

    print(f"Loading base model '{args.base_model}'...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)

    print(f"Loading adapter from '{args.adapter}'...")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print(f"Loading evaluation dataset...")
    _, eval_dataset = get_tokenized_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        max_seq_length=args.max_seq_length,
        max_eval_samples=args.max_eval_samples,
    )

    print("\nComputing Validation Loss & Perplexity...")
    eval_loss, ppl = compute_eval_loss_and_perplexity(
        model=model,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
    )

    print("=" * 50)
    print(f"  Checkpoint:      {args.adapter}")
    print(f"  Validation Loss: {eval_loss:.4f}")
    print(f"  Perplexity:      {ppl:.2f}")
    print("=" * 50)

    print("\nSample Generations:")
    samples = generate_sample_responses(model, tokenizer)
    for s in samples:
        print(f"Prompt: {s['instruction']}")
        print(f"Response: {s['response']}\n")

    clear_gpu_memory()


if __name__ == "__main__":
    main()
