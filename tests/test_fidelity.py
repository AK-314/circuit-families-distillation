"""Tests for exact Stage 8 behavioural-fidelity metrics."""

from __future__ import annotations

import math

import pytest
import torch

from circuit_families.config import load_model_config
from circuit_families.interpretability.fidelity import (
    evaluate_component_mask,
)
from circuit_families.interpretability.masks import ComponentMask
from circuit_families.models import build_transformer


def _model():
    return build_transformer(
        load_model_config("configs/model.yaml"),
        seed=0,
        device="cpu",
    )


def _examples(count: int = 17) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [
        (left, right)
        for left in range(113)
        for right in range(113)
    ][:count]

    inputs = torch.tensor(
        [
            [left, right, 113]
            for left, right in pairs
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            (left + right) % 113
            for left, right in pairs
        ],
        dtype=torch.long,
    )
    return inputs, targets


def _assert_metric_values_are_finite(metrics) -> None:
    for name, value in metrics.to_record().items():
        if isinstance(value, float):
            assert math.isfinite(value), name


def test_all_retained_has_exact_full_model_fidelity() -> None:
    model = _model()
    inputs, targets = _examples()

    model.train()
    metrics = evaluate_component_mask(
        model,
        inputs,
        targets,
        ComponentMask.all_retained(),
        batch_size=5,
    )

    assert model.training
    assert metrics.primary_fidelity == 1.0
    assert metrics.prediction_agreement_count == 17
    assert metrics.full_accuracy == metrics.masked_accuracy
    assert metrics.full_cross_entropy == metrics.masked_cross_entropy
    assert metrics.accuracy_change == 0.0
    assert metrics.cross_entropy_change == 0.0
    assert metrics.mean_kl_divergence == 0.0
    assert metrics.mean_jensen_shannon_divergence == 0.0
    assert metrics.maximum_absolute_logit_difference == 0.0
    assert metrics.retained_attention_head_count == 4
    assert metrics.retained_mlp_neuron_count == 512
    assert metrics.retained_component_count == 516
    assert metrics.retained_component_proportion == 1.0
    assert metrics.evaluated_example_count == 17
    assert metrics.evaluation_batch_size == 5


def test_all_ablated_metrics_are_finite_and_deterministic() -> None:
    model = _model()
    inputs, targets = _examples()
    mask = ComponentMask.all_ablated()

    first = evaluate_component_mask(
        model,
        inputs,
        targets,
        mask,
        batch_size=4,
    )
    second = evaluate_component_mask(
        model,
        inputs,
        targets,
        mask,
        batch_size=4,
    )

    assert first == second
    _assert_metric_values_are_finite(first)

    assert first.retained_attention_head_count == 0
    assert first.retained_mlp_neuron_count == 0
    assert first.retained_component_count == 0
    assert first.retained_component_proportion == 0.0
    assert first.maximum_absolute_logit_difference > 0.0


def test_batched_and_unbatched_evaluation_agree() -> None:
    model = _model()
    inputs, targets = _examples()
    mask = ComponentMask.from_ablated_identifiers(
        ["H0", "N0", "N17", "N511"]
    )

    unbatched = evaluate_component_mask(
        model,
        inputs,
        targets,
        mask,
        batch_size=17,
    )
    singleton_batches = evaluate_component_mask(
        model,
        inputs,
        targets,
        mask,
        batch_size=1,
    )

    assert (
        unbatched.prediction_agreement_count
        == singleton_batches.prediction_agreement_count
    )
    assert unbatched.primary_fidelity == (
        singleton_batches.primary_fidelity
    )
    assert unbatched.full_accuracy == singleton_batches.full_accuracy
    assert (
        unbatched.masked_accuracy
        == singleton_batches.masked_accuracy
    )

    for name in (
        "full_cross_entropy",
        "masked_cross_entropy",
        "cross_entropy_change",
        "mean_kl_divergence",
        "mean_jensen_shannon_divergence",
        "maximum_absolute_logit_difference",
    ):
        assert getattr(unbatched, name) == pytest.approx(
            getattr(singleton_batches, name),
            abs=1.0e-6,
            rel=1.0e-6,
        )


def test_metric_reductions_use_every_example() -> None:
    model = _model()
    inputs, targets = _examples(count=7)
    mask = ComponentMask.one_head_ablated("H0")

    metrics = evaluate_component_mask(
        model,
        inputs,
        targets,
        mask,
        batch_size=3,
    )

    first_six = evaluate_component_mask(
        model,
        inputs[:6],
        targets[:6],
        mask,
        batch_size=3,
    )
    final_one = evaluate_component_mask(
        model,
        inputs[6:],
        targets[6:],
        mask,
        batch_size=1,
    )

    weighted_cross_entropy = (
        first_six.masked_cross_entropy * 6
        + final_one.masked_cross_entropy
    ) / 7
    weighted_kl = (
        first_six.mean_kl_divergence * 6
        + final_one.mean_kl_divergence
    ) / 7

    assert metrics.masked_cross_entropy == pytest.approx(
        weighted_cross_entropy,
        abs=1.0e-7,
        rel=1.0e-7,
    )
    assert metrics.mean_kl_divergence == pytest.approx(
        weighted_kl,
        abs=1.0e-7,
        rel=1.0e-7,
    )


def test_single_head_and_neuron_metrics_record_correct_counts() -> None:
    model = _model()
    inputs, targets = _examples()

    head = evaluate_component_mask(
        model,
        inputs,
        targets,
        ComponentMask.one_head_ablated("H0"),
        batch_size=8,
    )
    neuron = evaluate_component_mask(
        model,
        inputs,
        targets,
        ComponentMask.one_neuron_ablated("N0"),
        batch_size=8,
    )

    assert head.retained_attention_head_count == 3
    assert head.retained_mlp_neuron_count == 512
    assert head.retained_component_count == 515

    assert neuron.retained_attention_head_count == 4
    assert neuron.retained_mlp_neuron_count == 511
    assert neuron.retained_component_count == 515


def test_targets_outside_valid_output_classes_are_rejected() -> None:
    model = _model()
    inputs, targets = _examples(count=2)
    targets[1] = 113

    with pytest.raises(ValueError, match="0 through 112"):
        evaluate_component_mask(
            model,
            inputs,
            targets,
            ComponentMask.all_retained(),
            batch_size=2,
        )


def test_invalid_batch_sizes_are_rejected() -> None:
    model = _model()
    inputs, targets = _examples(count=2)

    with pytest.raises(ValueError, match="positive"):
        evaluate_component_mask(
            model,
            inputs,
            targets,
            ComponentMask.all_retained(),
            batch_size=0,
        )

    with pytest.raises(TypeError, match="integer"):
        evaluate_component_mask(
            model,
            inputs,
            targets,
            ComponentMask.all_retained(),
            batch_size=True,
        )
