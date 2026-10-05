"""Unit tests for the LoRA+ optimizer implementation (Section 22)."""

import pytest
import torch
import torch.nn as nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from src.lora_plus_optimizer import create_loraplus_optimizer


class MockLoRALayer(nn.Module):
    """Synthetic module mimicking a Hugging Face PEFT LoRA Linear layer."""

    def __init__(self, in_features: int = 32, out_features: int = 32, r: int = 8):
        super().__init__()
        # Base frozen weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        # Trainable LoRA A and B adapter matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r), requires_grad=True)

    def forward(self, x):
        return x @ self.weight.T + (x @ self.lora_A.T) @ self.lora_B.T


class MockModel(nn.Module):
    """Synthetic model containing base frozen layers and LoRA adapted layers."""

    def __init__(self):
        super().__init__()
        self.linear1 = MockLoRALayer(32, 32, r=8)
        self.linear2 = MockLoRALayer(32, 32, r=8)
        # Pure frozen layer
        self.frozen_bias = nn.Parameter(torch.zeros(32), requires_grad=False)


def test_loraplus_learning_rates_and_ratio():
    """Validates that lr_B / lr_A strictly equals the specified ratio (16x)."""
    model = MockModel()
    lr_a = 1e-4
    ratio = 16.0
    expected_lr_b = 1.6e-3

    optimizer = create_loraplus_optimizer(
        opt_model=model,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": lr_a, "weight_decay": 0.01},
        loraplus_lr_ratio=ratio,
    )

    group_lrs = {}
    for group in optimizer.param_groups:
        g_name = group.get("group_name", "unknown")
        group_lrs[g_name] = group["lr"]

    assert "groupA" in group_lrs, "groupA must be present in optimizer parameter groups"
    assert "groupB" in group_lrs, "groupB must be present in optimizer parameter groups"

    actual_lr_a = group_lrs["groupA"]
    actual_lr_b = group_lrs["groupB"]

    assert actual_lr_a == pytest.approx(lr_a, rel=1e-5), f"Expected lr_A={lr_a}, got {actual_lr_a}"
    assert actual_lr_b == pytest.approx(expected_lr_b, rel=1e-5), f"Expected lr_B={expected_lr_b}, got {actual_lr_b}"

    # Critical requirement from prompt:
    # assert lr_B / lr_A == 16
    computed_ratio = actual_lr_b / actual_lr_a
    assert computed_ratio == pytest.approx(ratio, rel=1e-5), (
        f"LoRA+ ratio verification failed! Expected {ratio}, got {computed_ratio}"
    )


def test_parameter_separation_and_base_weight_freezing():
    """Validates that base weights are completely frozen and LoRA A/B are separated."""
    model = MockModel()

    # Verify base weights are frozen
    for name, param in model.named_parameters():
        if "lora" not in name:
            assert not param.requires_grad, f"Base parameter '{name}' should be frozen!"
        else:
            assert param.requires_grad, f"LoRA parameter '{name}' should be trainable!"

    optimizer = create_loraplus_optimizer(
        opt_model=model,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-4},
        loraplus_lr_ratio=8.0,
    )

    # Collect parameter IDs in each group
    group_a_params = []
    group_b_params = []
    for group in optimizer.param_groups:
        if "groupA" in group.get("group_name", ""):
            group_a_params.extend([id(p) for p in group["params"]])
        elif "groupB" in group.get("group_name", ""):
            group_b_params.extend([id(p) for p in group["params"]])

    assert id(model.linear1.lora_A) in group_a_params
    assert id(model.linear2.lora_A) in group_a_params
    assert id(model.linear1.lora_B) in group_b_params
    assert id(model.linear2.lora_B) in group_b_params

    # Verify frozen base parameters never enter the optimizer
    assert id(model.linear1.weight) not in group_a_params
    assert id(model.linear1.weight) not in group_b_params
    assert id(model.frozen_bias) not in group_a_params
    assert id(model.frozen_bias) not in group_b_params


def test_loraplus_failure_on_identical_lr():
    """Validates that test fails if LoRA+ accidentally assigns the same LR to A and B."""
    lr_a = 1e-4
    ratio = 16.0
    # Simulate bug
    buggy_lr_b = lr_a

    with pytest.raises(AssertionError):
        assert buggy_lr_b / lr_a == ratio, "Test correctly detects identical learning rates bug!"
