import abc
import math

import torch


class LrScheduler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_lr(self, step: int) -> float:
        """Returns the learning rate for the given step."""
        raise NotImplementedError("|get_lr| is an abstract method.")


class ConstantLrScheduler(LrScheduler):
    """Constant learning rate scheduler."""

    def __init__(self, learning_rate: float):
        self._learning_rate = learning_rate

    def get_lr(self, step: int) -> float:
        del step  # Unused.
        return self._learning_rate


class CosineLrScheduler(LrScheduler):
    """Cosine learning rate scheduler."""

    def __init__(self, learning_rate: float, warmup_steps: int, lr_decay_steps: int):
        self._peak_lr = learning_rate
        self._start_lr = learning_rate / 10.0
        self._warmup_steps = warmup_steps
        self._lr_decay_steps = lr_decay_steps

        if self._lr_decay_steps <= self._warmup_steps:
            raise ValueError("|lr_decay_steps| must be greater than |warmup_steps|.")

    def get_lr(self, step: int) -> float:
        if step < self._warmup_steps:
            return self._peak_lr * step / self._warmup_steps

        if step > self._lr_decay_steps:
            return self._start_lr

        decay_ratio = (step - self._warmup_steps) / (
            self._lr_decay_steps - self._warmup_steps
        )
        if decay_ratio < 0.0 or decay_ratio > 1.0:
            raise RuntimeError(
                "Decay ratio must be in [0.0, 1.0]. Fix LR scheduler settings."
            )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self._start_lr + coeff * (self._peak_lr - self._start_lr)


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    betas: tuple[float, float],
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        fused=True,
    )
