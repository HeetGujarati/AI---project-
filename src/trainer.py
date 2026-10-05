"""Custom Trainer subclass integrating LoRA+ optimizer creation and step metrics monitoring."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.lora_plus_optimizer import create_loraplus_optimizer
from src.metrics import MetricsTracker


@dataclass
class LoraPlusTrainingArguments(TrainingArguments):
    """Extends Hugging Face TrainingArguments with LoRA+ and Dyn-LoRA+ specific hyperparameters."""

    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA+ learning rate multiplier ratio: eta_B = lambda * eta_A."},
    )
    loraplus_lr_embedding: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Learning rate for LoRA adapter embedding layers."},
    )
    loraplus_dynamic: bool = field(
        default=False,
        metadata={"help": "Enable Dyn-LoRA+ dynamic ratio annealing schedule."},
    )
    loraplus_lambda_max: float = field(
        default=16.0,
        metadata={"help": "Maximum ratio at start of training (lambda_max)."},
    )
    loraplus_lambda_min: float = field(
        default=1.0,
        metadata={"help": "Minimum ratio at terminal training steps (lambda_min)."},
    )
    loraplus_schedule: str = field(
        default="cosine",
        metadata={"help": "Annealing decay schedule: 'cosine', 'linear', or 'exponential'."},
    )


class MetricsCallback(TrainerCallback):
    """Callback to capture live step losses, learning rates, dynamic ratios, and eval performance."""

    def __init__(self, tracker: MetricsTracker, lr_a: float, lr_b: float, trainer: Optional[Trainer] = None):
        super().__init__()
        self.tracker = tracker
        self.lr_a = lr_a
        self.lr_b = lr_b
        self.trainer = trainer

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tracker.start_training()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict] = None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            cur_ratio = getattr(self.tracker.metrics, "current_ratio", self.lr_b / max(self.lr_a, 1e-9))
            cur_lr_b = self.lr_a * cur_ratio
            self.tracker.log_step(
                step=state.global_step,
                loss=logs["loss"],
                lr_a=self.lr_a,
                lr_b=cur_lr_b,
                ratio=cur_ratio,
            )
        if "eval_loss" in logs:
            self.tracker.log_eval(
                step=state.global_step,
                eval_loss=logs["eval_loss"],
            )


class LoraPlusTrainer(Trainer):
    """Hugging Face Trainer subclass that instantiates the LoRA+ grouped optimizer and dynamic scheduler."""

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[LoraPlusTrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        tracker: Optional[MetricsTracker] = None,
        **kwargs,
    ):
        import inspect
        proc = processing_class or tokenizer
        init_kwargs = {
            "model": model,
            "args": args,
            "data_collator": data_collator,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "model_init": model_init,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "optimizers": optimizers,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        }
        sig = inspect.signature(Trainer.__init__)
        if "processing_class" in sig.parameters:
            init_kwargs["processing_class"] = proc
        else:
            init_kwargs["tokenizer"] = proc

        init_kwargs.update(kwargs)
        super().__init__(**init_kwargs)
        self.tracker = tracker
        self.dynamic_scheduler = None

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Overrides default optimizer creation to apply LoRA+ separate parameter groups and Dyn-LoRA+ scheduler."""
        ratio = getattr(self.args, "loraplus_lr_ratio", None)
        is_dynamic = getattr(self.args, "loraplus_dynamic", False)

        if (ratio is None or ratio <= 1.0) and not is_dynamic:
            return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            embedding_lr = getattr(self.args, "loraplus_lr_embedding", 1e-6)
            initial_ratio = getattr(self.args, "loraplus_lambda_max", 16.0) if is_dynamic else ratio

            self.optimizer = create_loraplus_optimizer(
                opt_model=self.model,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                loraplus_lr_ratio=initial_ratio,
                loraplus_lr_embedding=embedding_lr,
            )

            if is_dynamic:
                from src.lora_plus_optimizer import DynamicLoraPlusScheduler
                total_steps = self.args.max_steps
                if total_steps <= 0:
                    train_dataloader = self.get_train_dataloader()
                    num_update_steps_per_epoch = max(len(train_dataloader) // self.args.gradient_accumulation_steps, 1)
                    total_steps = num_update_steps_per_epoch * int(self.args.num_train_epochs)

                self.dynamic_scheduler = DynamicLoraPlusScheduler(
                    optimizer=self.optimizer,
                    total_steps=max(total_steps, 1),
                    lambda_max=getattr(self.args, "loraplus_lambda_max", 16.0),
                    lambda_min=getattr(self.args, "loraplus_lambda_min", 1.0),
                    schedule=getattr(self.args, "loraplus_schedule", "cosine"),
                )
                print(f"[Dyn-LoRA+] Initialized Dynamic Ratio Scheduler: {self.dynamic_scheduler.schedule.upper()} "
                      f"({self.dynamic_scheduler.lambda_max:.1f}x -> {self.dynamic_scheduler.lambda_min:.1f}x across {total_steps} steps)")

        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """Executes forward-backward step and updates dynamic asymmetry ratio."""
        import inspect
        sig = inspect.signature(super().training_step)
        if "num_items_in_batch" in sig.parameters:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        if self.dynamic_scheduler is not None:
            cur_ratio = self.dynamic_scheduler.step()
            if self.tracker is not None:
                self.tracker.metrics.current_ratio = cur_ratio

        return loss
