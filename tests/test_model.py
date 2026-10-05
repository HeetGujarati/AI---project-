"""Unit tests for model adapter creation and parameter efficiency."""

import pytest
import torch
import torch.nn as nn

from src.lora_config import apply_lora_to_model, create_lora_config


class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.v_proj(x) + self.k_proj(x))


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_size)
        self.block = SimpleTransformerBlock(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 100, bias=False)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        h = self.block(h)
        return self.lm_head(h)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}


def test_lora_adapter_application():
    """Verify that LoRA adapter only marks low-rank matrices as trainable."""
    model = SimpleModel(hidden_size=64)
    lora_cfg = create_lora_config(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )

    peft_model, stats = apply_lora_to_model(model, lora_cfg)

    assert stats["trainable_params"] > 0
    assert stats["trainable_params"] < stats["total_params"]
    assert stats["trainable_percentage"] < 50.0

    # Ensure base linear layer weights are frozen
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            assert param.requires_grad, f"LoRA param {name} should be trainable"
        else:
            assert not param.requires_grad, f"Base param {name} should be frozen"
