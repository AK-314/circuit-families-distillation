"""Tests for the frozen TransformerLens model and device selection."""

from __future__ import annotations

import pytest
import torch

from circuit_families.config import load_model_config
from circuit_families.models.transformer import (
    EXPECTED_PARAMETER_COUNT,
    build_transformer,
    parameter_count,
)
from circuit_families.training.device import (
    device_record,
    resolve_device,
)


def _cpu_model(seed: int):
    return build_transformer(
        load_model_config("configs/model.yaml"),
        seed=seed,
        device="cpu",
    )


def test_model_has_frozen_shapes_and_parameter_count() -> None:
    model = _cpu_model(seed=0)

    assert model.cfg.n_layers == 1
    assert model.cfg.n_ctx == 3
    assert model.cfg.d_vocab == 114
    assert model.cfg.d_vocab_out == 113
    assert model.cfg.normalization_type is None
    assert model.cfg.attention_dir == "causal"
    assert tuple(model.W_E.shape) == (114, 128)
    assert tuple(model.W_U.shape) == (128, 113)
    assert parameter_count(model) == EXPECTED_PARAMETER_COUNT


def test_model_output_has_exactly_113_answer_classes() -> None:
    model = _cpu_model(seed=0)
    tokens = torch.tensor(
        [
            [0, 0, 113],
            [56, 57, 113],
            [112, 112, 113],
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        logits = model(tokens)

    assert logits.shape == (3, 3, 113)
    assert logits[:, -1, :].shape == (3, 113)


def test_equals_token_is_embeddable_but_not_predictable() -> None:
    model = _cpu_model(seed=0)

    assert model.W_E.shape[0] == 114
    assert model.W_U.shape[-1] == 113

    tokens = torch.tensor([[1, 2, 113]], dtype=torch.long)

    with torch.inference_mode():
        logits = model(tokens)

    prediction = logits[:, -1, :].argmax(dim=-1)

    assert prediction.item() < 113


def test_model_uses_float32_without_norm_or_dropout() -> None:
    model = _cpu_model(seed=0)

    assert {
        parameter.dtype
        for parameter in model.parameters()
    } == {torch.float32}

    module_names = {
        type(module).__name__
        for module in model.modules()
    }

    assert not any("LayerNorm" in name for name in module_names)
    assert not any("Dropout" in name for name in module_names)


def test_same_seed_reproduces_initial_model_exactly() -> None:
    first = _cpu_model(seed=7)
    second = _cpu_model(seed=7)

    first_state = first.state_dict()
    second_state = second.state_dict()

    assert first_state.keys() == second_state.keys()

    for name in first_state:
        assert torch.equal(first_state[name], second_state[name]), name


def test_different_seed_changes_initial_model() -> None:
    first = _cpu_model(seed=0)
    second = _cpu_model(seed=1)

    assert any(
        not torch.equal(
            first.state_dict()[name],
            second.state_dict()[name],
        )
        for name in first.state_dict()
    )


def test_default_device_priority_prefers_cuda_then_cpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "circuit_families.training.device.mps_is_available",
        lambda: True,
    )

    assert resolve_device().type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device().type == "cpu"


def test_explicit_mps_override_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "circuit_families.training.device.mps_is_available",
        lambda: True,
    )

    with pytest.raises(
        ValueError,
        match="cuda, cpu",
    ):
        resolve_device("mps")


def test_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "circuit_families.training.device.mps_is_available",
        lambda: False,
    )

    assert resolve_device().type == "cpu"


def test_explicit_cpu_override_and_device_record() -> None:
    device = resolve_device("cpu")
    record = device_record(device)

    assert device == torch.device("cpu")
    assert record["selected_device"] == "cpu"
    assert record["device_type"] == "cpu"
    assert record["selected_device_name"] == "CPU"
