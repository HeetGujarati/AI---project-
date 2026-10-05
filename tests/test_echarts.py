"""Unit tests for the Apache ECharts dashboard generator."""

import os
import tempfile
import pytest

from src.echarts_dashboard import generate_echarts_dashboard


def test_generate_echarts_dashboard():
    mock_experiments = [
        {
            "experiment_name": "lora",
            "method": "lora",
            "lr_ratio": 1.0,
            "train_loss": 1.8542,
            "eval_loss": 1.7820,
            "eval_perplexity": 5.94,
            "total_train_time_sec": 120.5,
            "time_per_step_sec": 0.385,
            "peak_memory_allocated_mb": 3450.0,
            "peak_memory_reserved_mb": 4100.0,
            "trainable_params": 2359296,
            "total_params": 1543714816,
            "trainable_pct": 0.1528,
            "loss_history": [{"step": 5, "loss": 2.1}, {"step": 10, "loss": 1.85}],
            "eval_loss_history": [{"step": 10, "eval_loss": 1.78}],
        },
        {
            "experiment_name": "loraplus_16",
            "method": "loraplus",
            "ratio": 16.0,
            "lr_ratio": 16.0,
            "train_loss": 1.6210,
            "eval_loss": 1.5430,
            "eval_perplexity": 4.68,
            "total_train_time_sec": 122.1,
            "time_per_step_sec": 0.388,
            "peak_memory_allocated_mb": 3455.0,
            "peak_memory_reserved_mb": 4120.0,
            "trainable_params": 2359296,
            "total_params": 1543714816,
            "trainable_pct": 0.1528,
            "loss_history": [{"step": 5, "loss": 1.95}, {"step": 10, "loss": 1.62}],
            "eval_loss_history": [{"step": 10, "eval_loss": 1.54}],
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = generate_echarts_dashboard(mock_experiments, output_dir=tmpdir)
        assert os.path.exists(out_path), "Dashboard HTML file should exist"
        
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check critical ECharts requirements
        assert "echarts.min.js" in content, "Should include ECharts library"
        assert "chart-train-loss" in content, "Should include training loss chart container"
        assert "chart-eval-loss" in content, "Should include eval loss chart container"
        assert "chart-perplexity" in content, "Should include perplexity chart container"
        assert "chart-vram" in content, "Should include VRAM chart container"
        assert "chart-params" in content, "Should include parameter breakdown container"
        assert "loraplus_16" in content, "Should embed experiment data"
        assert "echarts.init" in content, "Should initialize ECharts instances"
