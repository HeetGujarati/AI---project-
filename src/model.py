"""Model loading, tokenizer initialization, and RTX 4060 memory optimization."""

import os
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.utils import clear_gpu_memory, get_gpu_memory_stats


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    fallback_name: str = "distilgpt2",
    torch_dtype: torch.dtype = torch.float16,
    gradient_checkpointing: bool = True,
    cache_dir: str = r"D:\Lora++\.cache\huggingface\hub",
    trust_remote_code: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, str]:
    """Loads target causal language model and tokenizer with FP16 and gradient checkpointing.
    
    Falls back gracefully to fallback_name if the primary model cannot be loaded.
    """
    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_model_name = model_name

    clear_gpu_memory()

    print(f"Loading tokenizer for '{selected_model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            selected_model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        print(f"Warning: Failed to load tokenizer for '{selected_model_name}' ({e}). Falling back to '{fallback_name}'.")
        selected_model_name = fallback_name
        tokenizer = AutoTokenizer.from_pretrained(
            selected_model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

    # Configure pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"Loading model '{selected_model_name}' on {device} (FP16, cache: {cache_dir})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            selected_model_name,
            torch_dtype=torch_dtype if device == "cuda" else torch.float32,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"Warning: Failed to load '{selected_model_name}' ({e}). Attempting fallback to '{fallback_name}'.")
        selected_model_name = fallback_name
        tokenizer = AutoTokenizer.from_pretrained(
            selected_model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

        model = AutoModelForCausalLM.from_pretrained(
            selected_model_name,
            torch_dtype=torch_dtype if device == "cuda" else torch.float32,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

    # Resize embeddings if necessary
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing if requested
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    mem = get_gpu_memory_stats()
    print(f"Model '{selected_model_name}' loaded successfully. VRAM Allocated: {mem['allocated_mb']} MB, Reserved: {mem['reserved_mb']} MB.")

    return model, tokenizer, selected_model_name
