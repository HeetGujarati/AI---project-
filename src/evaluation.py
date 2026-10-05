"""Evaluation routines, perplexity calculation, and sample text generation benchmarks."""

import math
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.dataset import FALLBACK_SAMPLES, format_instruction
from src.utils import clear_gpu_memory


def compute_eval_loss_and_perplexity(
    model: PreTrainedModel,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    max_eval_samples: Optional[int] = 100,
) -> Tuple[float, float]:
    """Computes exact cross-entropy loss and perplexity on the evaluation split."""
    clear_gpu_memory()
    model.eval()

    eval_sub = eval_dataset
    if max_eval_samples and len(eval_dataset) > max_eval_samples:
        eval_sub = eval_dataset.select(range(max_eval_samples))

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    dataloader = DataLoader(eval_sub, batch_size=batch_size, collate_fn=collator)

    total_loss = 0.0
    total_steps = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    try:
        perplexity = round(math.exp(min(avg_loss, 50.0)), 4)
    except OverflowError:
        perplexity = float("inf")

    clear_gpu_memory()
    return round(avg_loss, 4), perplexity


def generate_sample_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, str]]:
    """Generates qualitative text completions for benchmark comparison."""
    if prompts is None:
        prompts = [
            "Explain what Low-Rank Adaptation (LoRA) is in machine learning.",
            "Why does LoRA+ use a higher learning rate for matrix B than matrix A?",
            "What are the primary hardware specs of the NVIDIA RTX 4060?",
        ]

    model.eval()
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        formatted_prompt = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the generated response
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        results.append({
            "instruction": prompt,
            "response": response,
        })

    clear_gpu_memory()
    return results
