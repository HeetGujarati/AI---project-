"""Single-run training CLI supporting Standard LoRA and LoRA+ on RTX 4060."""

import argparse
import os
import sys
import time

# Set safety and D: drive caching environment variables before importing torch/numpy
os.environ["HF_HOME"] = r"D:\Lora++\.cache\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"D:\Lora++\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\Lora++\.cache\huggingface\datasets"
os.environ["TORCH_HOME"] = r"D:\Lora++\.cache\torch"
os.environ["PIP_CACHE_DIR"] = r"D:\Lora++\.cache\pip"
os.environ["TEMP"] = r"D:\Lora++\.cache\temp"
os.environ["TMP"] = r"D:\Lora++\.cache\temp"
os.environ["TMPDIR"] = r"D:\Lora++\.cache\temp"
os.environ["PYTHONPYCACHEPREFIX"] = r"D:\Lora++\.cache\pycache"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import DataCollatorForSeq2Seq

from src.dataset import get_tokenized_dataset
from src.evaluation import compute_eval_loss_and_perplexity, generate_sample_responses
from src.lora_config import apply_lora_to_model, create_lora_config
from src.metrics import ExperimentMetrics, MetricsTracker
from src.model import load_model_and_tokenizer
from src.trainer import LoraPlusTrainer, LoraPlusTrainingArguments, MetricsCallback
from src.utils import (
    clear_gpu_memory,
    get_gpu_info,
    get_gpu_memory_stats,
    print_environment_banner,
    save_json,
    set_seed,
    setup_environment,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM with LoRA, LoRA+, or Dyn-LoRA+ on RTX 4060")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model name or path")
    parser.add_argument("--fallback_model", type=str, default="distilgpt2", help="Fallback model name")
    parser.add_argument("--method", type=str, choices=["lora", "loraplus", "dyn_loraplus"], default="loraplus", help="Fine-tuning method")
    parser.add_argument("--ratio", type=float, default=16.0, help="LoRA+ learning rate ratio (eta_B / eta_A)")
    parser.add_argument("--lambda_max", type=float, default=16.0, help="Dyn-LoRA+ maximum early ratio")
    parser.add_argument("--lambda_min", type=float, default=1.0, help="Dyn-LoRA+ minimum terminal ratio")
    parser.add_argument("--schedule", type=str, choices=["cosine", "linear", "exponential"], default="cosine", help="Dyn-LoRA+ decay schedule")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="Base learning rate (eta_A)")
    parser.add_argument("--embedding_lr", type=float, default=1e-6, help="Embedding learning rate")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank r")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_train_samples", type=int, default=500, help="Maximum training samples")
    parser.add_argument("--max_eval_samples", type=int, default=100, help="Maximum evaluation samples")
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints", help="Output checkpoint directory")
    parser.add_argument("--results_dir", type=str, default="experiments/results", help="Results directory")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment()
    set_seed(args.seed)

    is_dyn = (args.method == "dyn_loraplus")
    effective_ratio = 1.0 if args.method == "lora" else (args.lambda_max if is_dyn else args.ratio)
    lr_b = args.lr_a * effective_ratio
    exp_name = args.name or (f"lora" if args.method == "lora" else ("dyn_loraplus" if is_dyn else f"loraplus_{int(args.ratio)}"))

    print_environment_banner()
    print(f"Starting Experiment: {exp_name.upper()}")
    print(f"  Method:     {args.method.upper()}")
    print(f"  Base LR:    {args.lr_a}")
    if is_dyn:
        print(f"  Schedule:   {args.schedule.upper()} (Ratio: {args.lambda_max:.1f}x -> {args.lambda_min:.1f}x)")
    else:
        print(f"  Ratio:      {effective_ratio:.1f}x")
    print(f"  Adapter LR: {lr_b}")
    print(f"  Model:      {args.model}")

    # 1. Load model and tokenizer
    model, tokenizer, loaded_model_name = load_model_and_tokenizer(
        model_name=args.model,
        fallback_name=args.fallback_model,
        gradient_checkpointing=True,
    )

    # 2. Apply LoRA adapter
    lora_cfg = create_lora_config(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
    )
    model, param_stats = apply_lora_to_model(model, lora_cfg)

    # 3. Load dataset
    train_dataset, eval_dataset = get_tokenized_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        max_seq_length=args.max_seq_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )

    # 4. Metrics setup
    metrics = ExperimentMetrics(
        experiment_name=exp_name,
        method=args.method,
        lr_a=args.lr_a,
        lr_b=lr_b,
        lr_ratio=effective_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        total_params=param_stats["total_params"],
        trainable_params=param_stats["trainable_params"],
        trainable_pct=param_stats["trainable_percentage"],
        lambda_max=args.lambda_max if is_dyn else None,
        lambda_min=args.lambda_min if is_dyn else None,
        schedule=args.schedule if is_dyn else None,
    )
    tracker = MetricsTracker(metrics)
    metrics_cb = MetricsCallback(tracker, lr_a=args.lr_a, lr_b=lr_b)

    # 5. Training Arguments
    exp_output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    training_args = LoraPlusTrainingArguments(
        output_dir=exp_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr_a,
        weight_decay=0.01,
        warmup_steps=2,
        logging_steps=5,
        eval_strategy="steps" if len(eval_dataset) > 0 else "no",
        eval_steps=max(10, len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps * 2)),
        save_strategy="no",
        fp16=True,
        report_to="none",
        loraplus_lr_ratio=effective_ratio if args.method in ["loraplus", "dyn_loraplus"] else None,
        loraplus_lr_embedding=args.embedding_lr,
        loraplus_dynamic=is_dyn,
        loraplus_lambda_max=args.lambda_max,
        loraplus_lambda_min=args.lambda_min,
        loraplus_schedule=args.schedule,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = LoraPlusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[metrics_cb],
        tracker=tracker,
    )

    # 6. Train model
    print(f"\n--- Launching Training: {exp_name} ---")
    start_wall_time = time.time()
    train_result = trainer.train()
    total_train_time = time.time() - start_wall_time

    # Record training metrics
    metrics.train_loss = round(train_result.training_loss, 4)
    metrics.train_steps = train_result.global_step
    metrics.optimizer_steps = train_result.global_step
    metrics.finalize(total_train_time)

    print(f"\nTraining Complete in {total_train_time:.1f}s. Final Train Loss: {metrics.train_loss}")

    # 7. Evaluate on validation set
    print("\nRunning Evaluation...")
    eval_loss, eval_ppl = compute_eval_loss_and_perplexity(
        model=model,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
    )
    metrics.eval_loss = eval_loss
    metrics.eval_perplexity = eval_ppl
    print(f"Validation Loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")

    # Post-training Adapter SVD Analysis
    from src.lora_plus_optimizer import compute_adapter_svd
    svd_stats = compute_adapter_svd(model, scaling=args.alpha / args.rank, target_rank=args.rank)
    metrics.effective_rank = svd_stats["effective_rank"]
    metrics.spectral_entropy = svd_stats["spectral_entropy"]
    metrics.condition_number = svd_stats["condition_number"]
    metrics.singular_values = svd_stats["singular_values"]
    print(f"Adapter SVD Analysis: Effective Rank = {metrics.effective_rank:.2f}/{args.rank} | "
          f"Spectral Entropy = {metrics.spectral_entropy:.2f} | Cond = {metrics.condition_number:.1f}")

    # 8. Save adapter checkpoint and tokenizer
    print(f"Saving checkpoint to {exp_output_dir}...")
    model.save_pretrained(exp_output_dir)
    tokenizer.save_pretrained(exp_output_dir)

    # 9. Save metrics
    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, f"{exp_name}_metrics.json")
    metrics.save(metrics_path)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved metrics to {metrics_path}")

    # 10. Sample inference test
    print("\n--- Testing Sample Generation ---")
    sample_outputs = generate_sample_responses(model, tokenizer)
    for sample in sample_outputs:
        print(f"Q: {sample['instruction']}")
        print(f"A: {sample['response'][:150]}...\n")

    clear_gpu_memory()
    mem = get_gpu_memory_stats()
    print(f"Peak VRAM Allocated: {metrics.peak_memory_allocated_mb} MB | Reserved: {metrics.peak_memory_reserved_mb} MB")
    print(f"=== Completed {exp_name} Successfully ===")


if __name__ == "__main__":
    main()
