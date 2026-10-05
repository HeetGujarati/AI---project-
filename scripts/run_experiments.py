"""Automated Master Experiment Suite for LoRA vs LoRA+ (Ratios 4, 8, 16, 32) on RTX 4060."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import (
    clear_gpu_memory,
    get_gpu_info,
    get_gpu_memory_stats,
    load_json,
    load_yaml_config,
    print_environment_banner,
    save_json,
    setup_environment,
)
from src.visualization import generate_all_plots


def format_markdown_table(experiments: List[Dict[str, Any]]) -> str:
    """Creates a formatted markdown comparison table from empirical results."""
    header = (
        "| Experiment | Method | Ratio (lambda) | Train Loss | Val Loss | Perplexity | Total Time (s) | Step Time (s) | Peak VRAM (MB) | Trainable Params |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for e in experiments:
        name = e["experiment_name"]
        method = e["method"].upper()
        if e.get("method") == "dyn_loraplus":
            ratio = f"Dyn({e.get('lambda_max', 16.0):.0f}→{e.get('lambda_min', 1.0):.0f})"
        else:
            ratio = f"{e.get('lr_ratio', 1.0):.0f}x"
        tr_loss = f"{e.get('train_loss', 0.0):.4f}"
        val_loss = f"{e.get('eval_loss', 0.0):.4f}"
        ppl = f"{e.get('eval_perplexity', 0.0):.2f}"
        t_time = f"{e.get('total_train_time_sec', 0.0):.1f}"
        s_time = f"{e.get('time_per_step_sec', 0.0):.3f}"
        vram = f"{e.get('peak_memory_allocated_mb', 0.0):.1f}"
        params = f"{e.get('trainable_params', 0):,} ({e.get('trainable_pct', 0.0):.2f}%)"
        rows.append(f"| {name} | {method} | {ratio} | {tr_loss} | {val_loss} | {ppl} | {t_time} | {s_time} | {vram} | {params} |")

    return header + "\n".join(rows) + "\n"


def generate_final_report(
    experiments: List[Dict[str, Any]],
    gpu_info: Dict[str, Any],
    output_path: str = "experiments/results/final_report.md",
) -> None:
    """Generates the comprehensive 5-page IEEE research report suite (PDF, TeX, HTML, MD)."""
    try:
        from scripts.generate_ieee_report import generate_all_ieee_reports

        summary_path = os.path.join("experiments", "results", "comparison_metrics.json")
        res = generate_all_ieee_reports(
            metrics_path=summary_path,
            plots_dir="experiments/plots",
            output_dir=os.path.dirname(output_path),
            gpu_info=gpu_info,
        )
        print(f"Generated comprehensive IEEE research report suite successfully:")
        for k, v in res.items():
            print(f"  [{k.upper()}] -> {v}")
    except Exception as e:
        print(f"Warning: Could not invoke generate_all_ieee_reports: {e}")
        # Fallback to saving markdown report directly
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        table_md = format_markdown_table(experiments)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Empirical Evaluation of LoRA+ on NVIDIA RTX 4060\n\n{table_md}\n")
        print(f"Generated fallback research report: {output_path}")


def run_single_experiment(exp_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single experiment via scripts/train.py subprocess."""
    name = exp_cfg["name"]
    method = exp_cfg["method"]
    lr_a = exp_cfg.get("lr_a", 0.0001)
    ratio = exp_cfg.get("ratio", 1.0 if method == "lora" else 16.0)

    model_name = global_cfg.get("model", {}).get("name", "Qwen/Qwen2.5-1.5B")
    training_cfg = global_cfg.get("training", {})

    cmd = [
        sys.executable,
        os.path.join("scripts", "train.py"),
        "--model", model_name,
        "--method", method,
        "--name", name,
        "--ratio", str(ratio),
        "--lr_a", str(lr_a),
        "--epochs", str(training_cfg.get("epochs", 1)),
        "--batch_size", str(training_cfg.get("batch_size", 1)),
        "--gradient_accumulation_steps", str(training_cfg.get("gradient_accumulation_steps", 8)),
        "--max_seq_length", str(training_cfg.get("max_seq_length", 512)),
        "--max_train_samples", str(training_cfg.get("max_train_samples", 500)),
        "--max_eval_samples", str(training_cfg.get("max_eval_samples", 100)),
        "--seed", str(training_cfg.get("seed", 42)),
    ]

    if method == "dyn_loraplus":
        lambda_max = exp_cfg.get("lambda_max", 16.0)
        lambda_min = exp_cfg.get("lambda_min", 1.0)
        schedule = exp_cfg.get("schedule", "cosine")
        cmd.extend([
            "--lambda_max", str(lambda_max),
            "--lambda_min", str(lambda_min),
            "--schedule", str(schedule),
        ])

    print(f"\n=======================================================")
    print(f"  EXECUTING EXPERIMENT: {name} (Method: {method}, Ratio: {ratio}x)")
    print(f"=======================================================")

    start_time = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - start_time

    # Load recorded metrics
    metrics_file = os.path.join("experiments", "results", f"{name}_metrics.json")
    if os.path.exists(metrics_file):
        metrics_data = load_json(metrics_file)
    else:
        metrics_data = {
            "experiment_name": name,
            "method": method,
            "lr_ratio": ratio,
            "total_train_time_sec": elapsed,
        }

    return metrics_data


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Master Experiment Suite Runner")
    parser.add_argument("--reports-only", action="store_true", help="Generate plots and reports from existing comparison_metrics.json")
    args = parser.parse_args()

    setup_environment()
    print_environment_banner()
    gpu_info = get_gpu_info()

    summary_path = os.path.join("experiments", "results", "comparison_metrics.json")

    if args.reports_only and os.path.exists(summary_path):
        print(f"Loading existing metrics from '{summary_path}'...")
        all_metrics = load_json(summary_path)
    else:
        config_path = os.path.join("config", "experiments.yaml")
        config = load_yaml_config(config_path)

        experiments_to_run = config.get("experiments", [])
        print(f"Loaded {len(experiments_to_run)} experiments from '{config_path}'.")

        all_metrics = []

        for exp in experiments_to_run:
            clear_gpu_memory()
            metrics = run_single_experiment(exp, config)
            all_metrics.append(metrics)
            clear_gpu_memory()

        # Save aggregated metrics
        save_json(all_metrics, summary_path)
        print(f"\nSaved aggregated metrics to {summary_path}")

    # Generate comparison table
    table_md = format_markdown_table(all_metrics)
    table_file = os.path.join("experiments", "results", "summary_table.md")
    with open(table_file, "w", encoding="utf-8") as f:
        f.write(table_md)
    print("\nEmpirical Results Summary Table:")
    print(table_md)

    # Generate all plots (including interactive Apache ECharts dashboard)
    print("Generating visualization plots...")
    plot_files = generate_all_plots(all_metrics, output_dir="experiments/plots")
    for pf in plot_files:
        print(f"  Plot created: {pf}")

    # Generate final research report
    generate_final_report(all_metrics, gpu_info, output_path="experiments/results/final_report.md")
    print("\n=== All Experiments Completed Successfully ===")


if __name__ == "__main__":
    main()
