"""Dataset preparation and instruction-following tokenization pipeline."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

PROMPT_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

PROMPT_TEMPLATE_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

FALLBACK_SAMPLES = [
    {
        "instruction": "Explain what Low-Rank Adaptation (LoRA) is in machine learning.",
        "input": "",
        "output": (
            "Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique for large pre-trained "
            "models. Instead of updating all original weights, LoRA freezes the pre-trained weights and injects "
            "trainable rank decomposition matrices (A and B) into the model's layers, dramatically reducing "
            "trainable parameters and GPU memory."
        ),
    },
    {
        "instruction": "What is the key theoretical contribution of LoRA+?",
        "input": "",
        "output": (
            "LoRA+ observes that in standard LoRA, using identical learning rates for matrices A and B leads to "
            "suboptimal feature learning as network width scales. By setting the learning rate of matrix B higher "
            "than matrix A by a constant factor lambda (eta_B = lambda * eta_A), both matrices learn at balanced "
            "rates, accelerating convergence and improving downstream task performance."
        ),
    },
    {
        "instruction": "Write a Python function to compute the learning rate ratio for LoRA+.",
        "input": "lr_A = 1e-4, ratio = 16",
        "output": (
            "def compute_loraplus_lr(lr_a: float, ratio: float) -> tuple[float, float]:\n"
            "    lr_b = lr_a * ratio\n"
            "    return lr_a, lr_b\n\n"
            "# Example:\n"
            "# lr_a, lr_b = compute_loraplus_lr(1e-4, 16) -> (0.0001, 0.0016)"
        ),
    },
    {
        "instruction": "Summarize the hardware characteristics of the NVIDIA RTX 4060.",
        "input": "",
        "output": (
            "The NVIDIA GeForce RTX 4060 features 8 GB of GDDR6 VRAM, 3072 CUDA cores, 4th-generation Tensor "
            "Cores, and Ada Lovelace architecture. In FP16 mixed precision, it provides excellent efficiency "
            "for parameter-efficient fine-tuning of 1B-2B class language models."
        ),
    },
]


def format_instruction(sample: Dict[str, str]) -> Tuple[str, str]:
    """Formats sample into prompt and full text (prompt + response)."""
    instruction = sample.get("instruction", "").strip()
    input_text = sample.get("input", "").strip()
    output_text = sample.get("output", "").strip()

    if input_text:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            f"Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

    full_text = prompt + output_text
    return prompt, full_text


def get_tokenized_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "yahma/alpaca-cleaned",
    max_seq_length: int = 512,
    max_train_samples: Optional[int] = 500,
    max_eval_samples: Optional[int] = 100,
    seed: int = 42,
    cache_dir: str = r"D:\Lora++\.cache\huggingface\datasets",
) -> Tuple[Dataset, Dataset]:
    """Loads, filters, and tokenizes instruction dataset with prompt masking for causal LM."""
    os.makedirs(cache_dir, exist_ok=True)
    raw_dataset = None

    try:
        print(f"Loading dataset '{dataset_name}' (cache: {cache_dir})...")
        raw_dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
        print(f"Loaded {len(raw_dataset)} raw samples from '{dataset_name}'.")
    except Exception as e:
        print(f"Warning: Could not load remote dataset '{dataset_name}' ({e}). Using embedded fallback dataset.")
        # Replicate fallback samples to satisfy sample counts
        multiplier = max(1, ((max_train_samples or 50) + (max_eval_samples or 20)) // len(FALLBACK_SAMPLES) + 1)
        synthetic = (FALLBACK_SAMPLES * multiplier)[: (max_train_samples or 50) + (max_eval_samples or 20)]
        raw_dataset = Dataset.from_list(synthetic)

    # Shuffle dataset
    raw_dataset = raw_dataset.shuffle(seed=seed)

    total_samples = len(raw_dataset)
    train_size = max_train_samples if max_train_samples and max_train_samples < total_samples else int(total_samples * 0.9)
    eval_size = max_eval_samples if max_eval_samples and max_eval_samples < (total_samples - train_size) else min(100, total_samples - train_size)

    train_raw = raw_dataset.select(range(train_size))
    eval_raw = raw_dataset.select(range(train_size, train_size + eval_size))

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for i in range(len(examples["instruction"])):
            sample = {
                "instruction": examples["instruction"][i],
                "input": examples.get("input", [""] * len(examples["instruction"]))[i],
                "output": examples.get("output", [""] * len(examples["instruction"]))[i],
            }
            prompt, full_text = format_instruction(sample)

            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            full_tokens = tokenizer.encode(full_text, add_special_tokens=True)

            if tokenizer.eos_token_id is not None and (len(full_tokens) == 0 or full_tokens[-1] != tokenizer.eos_token_id):
                full_tokens.append(tokenizer.eos_token_id)

            # Truncate
            if len(full_tokens) > max_seq_length:
                full_tokens = full_tokens[:max_seq_length]

            # Mask prompt tokens in labels with -100
            prompt_len = min(len(prompt_tokens), len(full_tokens))
            labels = [-100] * prompt_len + full_tokens[prompt_len:]

            attention_mask = [1] * len(full_tokens)

            batch_input_ids.append(full_tokens)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

    tokenized_train = train_raw.map(
        tokenize_fn,
        batched=True,
        batch_size=100,
        remove_columns=train_raw.column_names,
        desc="Tokenizing training dataset",
    )
    tokenized_eval = eval_raw.map(
        tokenize_fn,
        batched=True,
        batch_size=100,
        remove_columns=eval_raw.column_names,
        desc="Tokenizing evaluation dataset",
    )

    print(f"Dataset prepared: {len(tokenized_train)} train samples, {len(tokenized_eval)} eval samples.")
    return tokenized_train, tokenized_eval
