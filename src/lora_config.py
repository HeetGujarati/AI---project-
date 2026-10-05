"""LoRA configuration and adapter integration utilities using Hugging Face PEFT."""

from typing import Dict, List, Optional, Tuple
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model


def get_default_target_modules(model_type: str = "qwen2") -> List[str]:
    """Returns appropriate linear projection names targeted for low-rank adaptation."""
    model_type = model_type.lower()
    if "qwen" in model_type or "llama" in model_type:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "gpt2" in model_type:
        return ["c_attn", "c_proj", "c_fc"]
    else:
        # Generic fallback
        return ["q_proj", "v_proj"]


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    task_type: TaskType = TaskType.CAUSAL_LM,
    bias: str = "none",
) -> LoraConfig:
    """Instantiates a validated PEFT LoraConfig."""
    if target_modules is None:
        target_modules = get_default_target_modules("qwen2")

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        bias=bias,
    )


def apply_lora_to_model(
    model: nn.Module,
    lora_config: LoraConfig,
) -> Tuple[nn.Module, Dict[str, int]]:
    """Applies LoRA adapter layers to base model, freezes base parameters, and logs efficiency."""
    peft_model = get_peft_model(model, lora_config)

    trainable_params = 0
    total_params = 0
    for _, param in peft_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_pct = (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0

    print("=" * 60)
    print("  LoRA Adapter Applied Successfully")
    print("=" * 60)
    print(f"  LoRA Rank (r):          {lora_config.r}")
    print(f"  LoRA Alpha:             {lora_config.lora_alpha}")
    print(f"  LoRA Dropout:           {lora_config.lora_dropout}")
    print(f"  Target Modules:         {lora_config.target_modules}")
    print(f"  Total Parameters:       {total_params:,}")
    print(f"  Trainable Parameters:   {trainable_params:,}")
    print(f"  Trainable Percentage:   {trainable_pct:.4f}%")
    print("=" * 60)

    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": round(trainable_pct, 4),
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
    }

    return peft_model, stats
