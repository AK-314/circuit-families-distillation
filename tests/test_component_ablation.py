"""Tests for exact attention-head and MLP-neuron zero ablation."""

from __future__ import annotations

import math

import pytest
import torch

from circuit_families.config import load_model_config
from circuit_families.interpretability.component_ablation import (
    apply_attention_head_mask,
    apply_mlp_neuron_mask,
    masked_model_logits,
)
from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_HOOK_NAME,
    MLP_NEURON_HOOK_NAME,
    ComponentMask,
)
from circuit_families.models import build_transformer
from circuit_families.training import canonical_state_hash


def _model():
    return build_transformer(
        load_model_config("configs/model.yaml"),
        seed=0,
        device="cpu",
    )


def _inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 113],
            [1, 2, 113],
            [56, 57, 113],
            [112, 112, 113],
        ],
        dtype=torch.long,
    )


def _hook_counts(model) -> dict[str, int]:
    return {
        name: len(model.hook_dict[name]._forward_hooks)
        for name in (
            ATTENTION_HEAD_HOOK_NAME,
            MLP_NEURON_HOOK_NAME,
        )
    }


def test_head_mask_zeros_only_the_selected_complete_head() -> None:
    activation = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(1, 3, 4, 2)
    mask = ComponentMask.one_head_ablated("H1")

    result = apply_attention_head_mask(activation, mask)

    assert torch.equal(result[:, :, 0, :], activation[:, :, 0, :])
    assert torch.count_nonzero(result[:, :, 1, :]).item() == 0
    assert torch.equal(result[:, :, 2, :], activation[:, :, 2, :])
    assert torch.equal(result[:, :, 3, :], activation[:, :, 3, :])


def test_neuron_mask_zeros_only_the_selected_post_activation() -> None:
    activation = torch.arange(
        2 * 3 * 512,
        dtype=torch.float32,
    ).reshape(2, 3, 512)
    mask = ComponentMask.one_neuron_ablated("N17")

    result = apply_mlp_neuron_mask(activation, mask)

    assert torch.equal(result[:, :, :17], activation[:, :, :17])
    assert torch.count_nonzero(result[:, :, 17]).item() == 0
    assert torch.equal(result[:, :, 18:], activation[:, :, 18:])


def test_all_retained_reproduces_full_model_exactly() -> None:
    model = _model()
    inputs = _inputs()

    model.train()

    with torch.inference_mode():
        full_logits = model.eval()(inputs)

    model.train()
    masked_logits = masked_model_logits(
        model,
        inputs,
        ComponentMask.all_retained(),
    )

    assert torch.equal(masked_logits, full_logits)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_all_ablated_is_finite_deterministic_and_non_identity() -> None:
    model = _model()
    inputs = _inputs()
    mask = ComponentMask.all_ablated()

    with torch.inference_mode():
        full_logits = model.eval()(inputs)

    first = masked_model_logits(model, inputs, mask)
    second = masked_model_logits(model, inputs, mask)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert not torch.equal(first, full_logits)
    assert math.isfinite(float(first.sum().item()))


def test_single_head_and_neuron_ablation_are_deterministic() -> None:
    model = _model()
    inputs = _inputs()
    initial_hash = canonical_state_hash(model.state_dict())

    head_mask = ComponentMask.one_head_ablated("H0")
    neuron_mask = ComponentMask.one_neuron_ablated("N0")

    head_first = masked_model_logits(model, inputs, head_mask)
    head_second = masked_model_logits(model, inputs, head_mask)
    neuron_first = masked_model_logits(model, inputs, neuron_mask)
    neuron_second = masked_model_logits(model, inputs, neuron_mask)

    assert torch.equal(head_first, head_second)
    assert torch.equal(neuron_first, neuron_second)
    assert canonical_state_hash(model.state_dict()) == initial_hash

    with torch.inference_mode():
        full_logits = model.eval()(inputs)

    assert not torch.equal(head_first, full_logits)
    assert not torch.equal(neuron_first, full_logits)


def test_repeated_masked_evaluations_do_not_accumulate_hooks() -> None:
    model = _model()
    inputs = _inputs()
    mask = ComponentMask.one_head_ablated("H0")
    initial_counts = _hook_counts(model)

    for _ in range(3):
        masked_model_logits(model, inputs, mask)
        assert _hook_counts(model) == initial_counts


def test_exception_during_forward_cleans_up_hooks(
    monkeypatch,
) -> None:
    model = _model()
    inputs = _inputs()
    initial_counts = _hook_counts(model)

    def fail_forward(*args, **kwargs):
        raise RuntimeError("deliberate forward failure")

    monkeypatch.setattr(model, "forward", fail_forward)

    with pytest.raises(RuntimeError, match="deliberate"):
        masked_model_logits(
            model,
            inputs,
            ComponentMask.all_retained(),
        )

    assert _hook_counts(model) == initial_counts


def test_masked_logits_have_only_113_output_classes() -> None:
    logits = masked_model_logits(
        _model(),
        _inputs(),
        ComponentMask.all_retained(),
    )

    assert logits.shape == (4, 3, 113)
    assert logits[:, -1, :].argmax(dim=-1).max().item() < 113
