from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from circuit_families.stage12r1 import (
    GateConfig,
    GateRunIdentity,
    deterministic_binary_mask,
    deterministic_gate_values,
    expected_l0_probability,
    gate_state_record,
    seeded_stochastic_gate_sample,
    validate_log_alpha,
)


def config() -> GateConfig:
    return GateConfig(
        temperature=0.7,
        stretch_lower=-0.1,
        stretch_upper=1.1,
    )


def identity(**updates) -> GateRunIdentity:
    base = GateRunIdentity(
        method_name="stage12r1_hard_concrete",
        method_version="technical-v1",
        configuration_reference="fixture://gate-config",
        run_id="technical-run",
        condition_identity="synthetic-condition-a",
        restart_index=0,
        seed_value=17,
    )
    return replace(base, **updates)


def test_same_complete_identity_reproduces_cpu_sample() -> None:
    x = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)
    a = seeded_stochastic_gate_sample(x, config(), identity())
    b = seeded_stochastic_gate_sample(x, config(), identity())
    assert torch.equal(a, b)


def test_seed_and_condition_identity_change_rng_stream() -> None:
    x = torch.zeros(16, dtype=torch.float64)
    base = seeded_stochastic_gate_sample(x, config(), identity())
    changed_seed = seeded_stochastic_gate_sample(
        x, config(), identity(seed_value=18)
    )
    changed_condition = seeded_stochastic_gate_sample(
        x,
        config(),
        identity(condition_identity="synthetic-condition-b"),
    )
    assert not torch.equal(base, changed_seed)
    assert not torch.equal(base, changed_condition)


def test_deterministic_extraction_is_order_independent() -> None:
    x = torch.tensor([-10.0, -1.0, 1.0, 10.0])
    mask = deterministic_binary_mask(
        x, config(), threshold=0.5, component_count=4
    )
    assert mask == deterministic_binary_mask(
        x.clone(), config(), threshold=0.5, component_count=4
    )


def test_expected_l0_is_finite_bounded_and_differentiable() -> None:
    x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    l0 = expected_l0_probability(x, config())
    assert torch.isfinite(l0).all()
    assert ((l0 >= 0) & (l0 <= 1)).all()
    l0.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_saturated_gates_remain_representable() -> None:
    x = torch.tensor([-1000.0, 1000.0])
    values = deterministic_gate_values(x, config())
    assert tuple(values.tolist()) == (0.0, 1.0)


def test_all_on_and_all_off_masks_are_representable() -> None:
    off = deterministic_binary_mask(
        torch.full((4,), -1000.0),
        config(),
        threshold=0.5,
        component_count=4,
    )
    on = deterministic_binary_mask(
        torch.full((4,), 1000.0),
        config(),
        threshold=0.5,
        component_count=4,
    )
    assert off == (0, 0, 0, 0)
    assert on == (1, 1, 1, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"stretch_lower": 0.0},
        {"stretch_upper": 1.0},
        {"sample_epsilon": 0.5},
    ],
)
def test_invalid_gate_configuration_rejects(kwargs) -> None:
    params = dict(
        temperature=0.7,
        stretch_lower=-0.1,
        stretch_upper=1.1,
    )
    params.update(kwargs)
    with pytest.raises(ValueError):
        GateConfig(**params)


def test_invalid_shape_nonfinite_and_basis_mismatch_reject() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_log_alpha(torch.zeros((2, 2)), component_count=4)
    with pytest.raises(ValueError, match="finite"):
        validate_log_alpha(torch.tensor([0.0, float("nan")]), component_count=2)
    with pytest.raises(ValueError, match="supplied component basis"):
        validate_log_alpha(torch.zeros(3), component_count=4)


def test_gate_state_record_and_hash_are_stable() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    a = gate_state_record(
        x,
        config(),
        component_basis_identity="toy-basis-v1",
        component_count=3,
    )
    b = gate_state_record(
        x.clone(),
        config(),
        component_basis_identity="toy-basis-v1",
        component_count=3,
    )
    assert a == b
    assert len(a.state_sha256) == 64


def test_basis_identity_changes_state_hash() -> None:
    x = torch.zeros(3)
    a = gate_state_record(
        x,
        config(),
        component_basis_identity="basis-a",
        component_count=3,
    )
    b = gate_state_record(
        x,
        config(),
        component_basis_identity="basis-b",
        component_count=3,
    )
    assert a.state_sha256 != b.state_sha256


def test_config_forbids_scientific_or_production_classification() -> None:
    with pytest.raises(ValueError, match="scientific_data=false"):
        replace(config(), scientific_data=True)
    with pytest.raises(ValueError, match="production_eligible=false"):
        replace(config(), production_eligible=True)
