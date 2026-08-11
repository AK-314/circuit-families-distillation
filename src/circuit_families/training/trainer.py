"""Minimal deterministic full-batch training primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from circuit_families.config import validate_training_config
from circuit_families.training.metrics import (
    classification_accuracy,
    cross_entropy_loss,
    gradient_norm,
)


def build_optimizer(
    model: torch.nn.Module,
    training_config: Mapping[str, Any],
) -> torch.optim.AdamW:
    """Construct the frozen AdamW optimiser."""

    validate_training_config(training_config)
    optimizer_config = training_config["optimizer"]

    return torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=(
            optimizer_config["beta1"],
            optimizer_config["beta2"],
        ),
        eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )


def train_full_batch_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Perform one full-batch optimiser update."""

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)

    if not torch.isfinite(loss):
        raise FloatingPointError("Training loss is not finite.")

    accuracy = classification_accuracy(logits, targets)
    loss.backward()

    current_gradient_norm = gradient_norm(model)

    if current_gradient_norm is None:
        raise RuntimeError("No parameter gradients were produced.")

    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(
            parameter.grad
        ).all():
            raise FloatingPointError(
                "A parameter gradient contains non-finite values."
            )

    optimizer.step()

    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(
                "A model parameter contains non-finite values."
            )

    metrics = {
        "cross_entropy": float(loss.detach().item()),
        "accuracy": float(accuracy.detach().item()),
        "gradient_norm": current_gradient_norm,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }

    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Training-step metrics are not finite.")

    return metrics
