"""Tests for final-position metrics and full-batch optimisation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from circuit_families.config import (
    load_model_config,
    load_training_config,
)
from circuit_families.models.transformer import build_transformer
from circuit_families.training.metrics import (
    classification_accuracy,
    cross_entropy_loss,
    evaluate_model,
    gradient_norm,
    parameter_norm,
)
from circuit_families.training.trainer import (
    build_optimizer,
    train_full_batch_step,
)


def _cpu_model(seed: int = 0):
    return build_transformer(
        load_model_config("configs/model.yaml"),
        seed=seed,
        device="cpu",
    )


def _small_batch() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(
        [
            [0, 0, 113],
            [1, 2, 113],
            [56, 57, 113],
            [112, 112, 113],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor([0, 3, 0, 111], dtype=torch.long)
    return inputs, targets


def test_loss_uses_only_final_sequence_position() -> None:
    generator = torch.Generator().manual_seed(0)
    first = torch.randn(
        2,
        3,
        113,
        generator=generator,
    )
    second = first.clone()

    second[:, 0, :] = 1_000.0
    second[:, 1, :] = -1_000.0

    targets = torch.tensor([7, 19], dtype=torch.long)

    assert torch.equal(
        cross_entropy_loss(first, targets),
        cross_entropy_loss(second, targets),
    )
    assert torch.equal(
        classification_accuracy(first, targets),
        classification_accuracy(second, targets),
    )


def test_loss_and_accuracy_match_manual_final_position_values() -> None:
    logits = torch.zeros(2, 3, 113)
    logits[0, -1, 4] = 3.0
    logits[1, -1, 8] = 2.0
    targets = torch.tensor([4, 7], dtype=torch.long)

    expected_loss = functional.cross_entropy(
        logits[:, -1, :],
        targets,
    )

    assert torch.equal(
        cross_entropy_loss(logits, targets),
        expected_loss,
    )
    assert classification_accuracy(logits, targets).item() == 0.5


def test_equals_token_cannot_be_used_as_a_target() -> None:
    logits = torch.zeros(1, 3, 113)
    targets = torch.tensor([113], dtype=torch.long)

    try:
        cross_entropy_loss(logits, targets)
    except ValueError as exc:
        assert "0 through 112" in str(exc)
    else:
        raise AssertionError("Target class 113 was not rejected.")


def test_evaluation_is_finite_and_does_not_create_gradients() -> None:
    model = _cpu_model()
    inputs, targets = _small_batch()

    model.train()
    metrics = evaluate_model(model, inputs, targets)

    assert model.training
    assert math.isfinite(metrics["cross_entropy"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_parameter_norm_is_positive_and_gradient_norm_starts_empty() -> None:
    model = _cpu_model()

    assert parameter_norm(model) > 0.0
    assert gradient_norm(model) is None


def test_one_full_batch_step_produces_finite_gradients_and_updates_parameters() -> None:
    model = _cpu_model()
    optimizer = build_optimizer(
        model,
        load_training_config("configs/training.yaml"),
    )
    inputs, targets = _small_batch()

    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    metrics = train_full_batch_step(
        model,
        optimizer,
        inputs,
        targets,
    )

    assert math.isfinite(metrics["cross_entropy"])
    assert math.isfinite(metrics["gradient_norm"])
    assert metrics["gradient_norm"] > 0.0
    assert metrics["learning_rate"] == 0.001

    after = model.state_dict()

    assert any(
        not torch.equal(before[name], after[name])
        for name in before
    )


def test_optimizer_uses_frozen_adamw_settings() -> None:
    model = _cpu_model()
    optimizer = build_optimizer(
        model,
        load_training_config("configs/training.yaml"),
    )
    group = optimizer.param_groups[0]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert group["lr"] == 0.001
    assert group["betas"] == (0.9, 0.98)
    assert group["eps"] == 1.0e-8
    assert group["weight_decay"] == 1.0
