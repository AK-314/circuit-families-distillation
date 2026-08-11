"""Loss, accuracy, evaluation, and norm calculations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

OUTPUT_CLASS_COUNT = 113


def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a PyTorch tensor.")

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape "
            "(batch_size, sequence_length, output_classes)."
        )

    if logits.shape[0] == 0:
        raise ValueError("logits batch dimension must not be empty.")

    if logits.shape[1] == 0:
        raise ValueError("logits sequence dimension must not be empty.")

    if logits.shape[2] != OUTPUT_CLASS_COUNT:
        raise ValueError(
            f"logits must contain exactly {OUTPUT_CLASS_COUNT} "
            "output classes."
        )


def _validate_targets(
    targets: torch.Tensor,
    *,
    expected_batch_size: int,
) -> None:
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a PyTorch tensor.")

    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional.")

    if targets.shape[0] != expected_batch_size:
        raise ValueError(
            "targets length must equal the logits batch size."
        )

    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long.")

    if targets.numel() == 0:
        raise ValueError("targets must not be empty.")

    minimum = int(targets.min().item())
    maximum = int(targets.max().item())

    if minimum < 0 or maximum >= OUTPUT_CLASS_COUNT:
        raise ValueError(
            "targets must contain only output classes 0 through 112."
        )


def final_position_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return answer logits from the final sequence position only."""

    _validate_logits(logits)
    return logits[:, -1, :]


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Calculate categorical cross-entropy at the final position only."""

    answer_logits = final_position_logits(logits)
    _validate_targets(
        targets,
        expected_batch_size=answer_logits.shape[0],
    )

    return functional.cross_entropy(answer_logits, targets)


def classification_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Calculate top-one accuracy at the final position only."""

    answer_logits = final_position_logits(logits)
    _validate_targets(
        targets,
        expected_batch_size=answer_logits.shape[0],
    )

    predictions = answer_logits.argmax(dim=-1)
    return (predictions == targets).to(torch.float32).mean()


def parameter_norm(model: torch.nn.Module) -> float:
    """Return the global L2 norm of all model parameters."""

    total_squared = None

    with torch.no_grad():
        for parameter in model.parameters():
            squared = parameter.detach().pow(2).sum()
            if total_squared is None:
                total_squared = squared
            else:
                total_squared = total_squared + squared

    if total_squared is None:
        return 0.0

    norm = float(total_squared.sqrt().item())

    if not math.isfinite(norm):
        raise FloatingPointError("Parameter norm is not finite.")

    return norm


def gradient_norm(model: torch.nn.Module) -> float | None:
    """Return the global L2 norm of available parameter gradients."""

    total_squared = None

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            squared = parameter.grad.detach().pow(2).sum()
            if total_squared is None:
                total_squared = squared
            else:
                total_squared = total_squared + squared

    if total_squared is None:
        return None

    norm = float(total_squared.sqrt().item())

    if not math.isfinite(norm):
        raise FloatingPointError("Gradient norm is not finite.")

    return norm


def evaluate_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Evaluate loss and accuracy without constructing gradients."""

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            accuracy = classification_accuracy(logits, targets)
    finally:
        model.train(was_training)

    metrics = {
        "cross_entropy": float(loss.item()),
        "accuracy": float(accuracy.item()),
    }

    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Evaluation metrics are not finite.")

    return metrics
