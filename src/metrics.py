"""Experiment metrics tracking, efficiency measurement, and serialization."""

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_gpu_memory_stats, save_json


@dataclass
class ExperimentMetrics:
    experiment_name: str
    method: str  # "lora" or "loraplus"
    lr_a: float
    lr_b: float
    lr_ratio: float
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 512

    # Parameter stats
    total_params: int = 0
    trainable_params: int = 0
    trainable_pct: float = 0.0

    # Training stats
    train_loss: float = 0.0
    train_steps: int = 0
    total_train_time_sec: float = 0.0
    time_per_step_sec: float = 0.0
    optimizer_steps: int = 0

    # Evaluation stats
    eval_loss: float = 0.0
    eval_perplexity: float = 0.0

    # Memory stats (MB)
    peak_memory_allocated_mb: float = 0.0
    peak_memory_reserved_mb: float = 0.0

    # Dyn-LoRA+ dynamic scheduling & spectral metrics
    lambda_max: Optional[float] = None
    lambda_min: Optional[float] = None
    schedule: Optional[str] = None
    effective_rank: Optional[float] = None
    spectral_entropy: Optional[float] = None
    condition_number: Optional[float] = None
    singular_values: List[float] = field(default_factory=list)
    current_ratio: float = 1.0

    # Loss trajectories
    loss_history: List[Dict[str, Any]] = field(default_factory=list)
    eval_loss_history: List[Dict[str, Any]] = field(default_factory=list)

    def finalize(self, total_time: float) -> None:
        self.total_train_time_sec = round(total_time, 2)
        if self.train_steps > 0:
            self.time_per_step_sec = round(total_time / self.train_steps, 4)
        if self.eval_loss > 0.0:
            try:
                self.eval_perplexity = round(math.exp(min(self.eval_loss, 50.0)), 4)
            except OverflowError:
                self.eval_perplexity = float("inf")

        mem = get_gpu_memory_stats()
        self.peak_memory_allocated_mb = mem["max_allocated_mb"]
        self.peak_memory_reserved_mb = mem["max_reserved_mb"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str) -> None:
        save_json(self.to_dict(), filepath)


class MetricsTracker:
    """Monitors live training metrics, loss steps, and wall-clock time."""

    def __init__(self, metrics: ExperimentMetrics):
        self.metrics = metrics
        self.start_time: float = 0.0
        self.step_start_time: float = 0.0

    def start_training(self) -> None:
        self.start_time = time.time()
        self.step_start_time = self.start_time

    def log_step(self, step: int, loss: float, lr_a: float, lr_b: float, ratio: Optional[float] = None) -> None:
        elapsed = time.time() - self.start_time
        entry = {
            "step": step,
            "loss": round(loss, 6),
            "lr_a": lr_a,
            "lr_b": round(lr_b, 6),
            "elapsed_time": round(elapsed, 2),
        }
        if ratio is not None:
            entry["ratio"] = round(ratio, 4)
        self.metrics.loss_history.append(entry)

    def log_eval(self, step: int, eval_loss: float) -> None:
        try:
            ppl = round(math.exp(min(eval_loss, 50.0)), 4)
        except OverflowError:
            ppl = float("inf")
        self.metrics.eval_loss_history.append({
            "step": step,
            "eval_loss": round(eval_loss, 6),
            "perplexity": ppl,
        })
        self.metrics.eval_loss = round(eval_loss, 6)
        self.metrics.eval_perplexity = ppl

    def finish(self) -> ExperimentMetrics:
        total_time = time.time() - self.start_time
        self.metrics.finalize(total_time)
        return self.metrics
