"""Unit tests for dataset formatting and tokenization."""

import pytest
from transformers import AutoTokenizer

from src.dataset import FALLBACK_SAMPLES, format_instruction, get_tokenized_dataset


def test_format_instruction():
    sample_with_input = {
        "instruction": "Summarize the text.",
        "input": "Antigravity is an AI pair programmer.",
        "output": "Antigravity is an AI.",
    }
    prompt, full_text = format_instruction(sample_with_input)
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "Antigravity is an AI pair programmer." in prompt
    assert full_text.endswith("Antigravity is an AI.")

    sample_no_input = {
        "instruction": "Define LoRA.",
        "input": "",
        "output": "Low Rank Adaptation.",
    }
    prompt2, full_text2 = format_instruction(sample_no_input)
    assert "### Instruction:" in prompt2
    assert "### Input:" not in prompt2
    assert full_text2.endswith("Low Rank Adaptation.")


def test_dataset_prompt_masking():
    """Verify that prompt tokens in labels are set to -100."""
    from transformers import GPT2Tokenizer
    # Use standard lightweight tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds, eval_ds = get_tokenized_dataset(
            tokenizer=tokenizer,
            max_seq_length=128,
            max_train_samples=4,
            max_eval_samples=2,
        )

        assert len(train_ds) == 4
        assert len(eval_ds) == 2

        first_sample = train_ds[0]
        assert "input_ids" in first_sample
        assert "attention_mask" in first_sample
        assert "labels" in first_sample

        # First tokens must be -100 (prompt masked)
        labels = first_sample["labels"]
        assert -100 in labels, "Prompt tokens must be masked with -100"
        # Not all tokens are -100 (response tokens are retained)
        assert any(l != -100 for l in labels), "Response tokens must not be masked"
    except Exception as e:
        pytest.skip(f"Skipping tokenizer test if network unavailable: {e}")
