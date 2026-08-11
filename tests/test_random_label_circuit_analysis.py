"""Tests for the frozen Stage 14 analysis primitives."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from circuit_families.analysis.random_label_circuit_analysis import (
    ANALYSIS_CONFIGURATION_SHA256,
    ANALYSIS_RUN_ID,
    DISTINCTNESS_SENSITIVITY_GRID,
    EXPECTED_SUBSET_COUNTS,
    FIDELITY_SENSITIVITY_GRID,
    MATCHED_CHECKPOINT_STEPS,
    TRANSFER_GROUPING_SENSITIVITY_GRID,
    agreement_passes_threshold,
    component_mask_record,
    load_frozen_analysis_configuration,
    load_stage14_masking_module,
    subset_context,
    subset_contexts,
)
from circuit_families.interpretability.masks import ComponentMask


@dataclass(frozen=True)
class SyntheticContext:
    """Small context matching the production context surface."""

    checkpoint_phase: str
    inputs: torch.Tensor
    targets: torch.Tensor


def complete_modular_inputs() -> torch.Tensor:
    """Return all 113² inputs in frozen lexicographic order."""

    pairs = [
        (left, right, 113)
        for left in range(113)
        for right in range(113)
    ]
    return torch.tensor(pairs, dtype=torch.long)


def test_load_frozen_analysis_configuration() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=Path.cwd()
    )

    assert configuration.sha256 == ANALYSIS_CONFIGURATION_SHA256
    assert configuration.analysis_run_id == ANALYSIS_RUN_ID
    assert configuration.checkpoint_steps == MATCHED_CHECKPOINT_STEPS
    assert configuration.fidelity_grid == FIDELITY_SENSITIVITY_GRID
    assert (
        configuration.distinctness_grid
        == DISTINCTNESS_SENSITIVITY_GRID
    )
    assert (
        configuration.transfer_grouping_grid
        == TRANSFER_GROUPING_SENSITIVITY_GRID
    )


def test_stage14_masking_adapter_loads_safely() -> None:
    module = load_stage14_masking_module(Path.cwd())

    assert hasattr(module, "Stage14MaskingContext")
    assert callable(module.load_context)


def test_exact_threshold_arithmetic_at_boundary() -> None:
    assert agreement_passes_threshold(
        agreement_count=99,
        evaluated_example_count=100,
        threshold=Fraction(99, 100),
    )
    assert not agreement_passes_threshold(
        agreement_count=98,
        evaluated_example_count=100,
        threshold=Fraction(99, 100),
    )

    assert agreement_passes_threshold(
        agreement_count=12_642,
        evaluated_example_count=12_769,
        threshold={
            "numerator": 99,
            "denominator": 100,
        },
    )
    assert not agreement_passes_threshold(
        agreement_count=12_641,
        evaluated_example_count=12_769,
        threshold={
            "numerator": 99,
            "denominator": 100,
        },
    )


@pytest.mark.parametrize(
    ("agreement_count", "example_count"),
    (
        (-1, 100),
        (101, 100),
        (0, 0),
    ),
)
def test_exact_threshold_rejects_invalid_counts(
    agreement_count: int,
    example_count: int,
) -> None:
    with pytest.raises(ValueError):
        agreement_passes_threshold(
            agreement_count=agreement_count,
            evaluated_example_count=example_count,
            threshold=Fraction(99, 100),
        )


def test_subset_contexts_follow_frozen_partition() -> None:
    inputs = complete_modular_inputs()
    targets = torch.arange(
        inputs.shape[0],
        dtype=torch.long,
    ) % 113
    context = SyntheticContext(
        checkpoint_phase="synthetic",
        inputs=inputs,
        targets=targets,
    )

    subsets = subset_contexts(context)

    assert tuple(subsets) == ("Q1", "Q2", "Q3", "Q4")

    for subset_name, subset in subsets.items():
        assert (
            subset.inputs.shape[0]
            == EXPECTED_SUBSET_COUNTS[subset_name]
        )
        assert subset.targets.shape[0] == subset.inputs.shape[0]
        assert subset.checkpoint_phase == (
            f"synthetic|subset={subset_name}"
        )

    assert sum(
        subset.inputs.shape[0]
        for subset in subsets.values()
    ) == 12_769


def test_subset_context_rejects_unknown_subset() -> None:
    context = SyntheticContext(
        checkpoint_phase="synthetic",
        inputs=complete_modular_inputs(),
        targets=torch.zeros(12_769, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="Unknown transfer subset"):
        subset_context(context, "Q5")


def test_component_mask_record_is_deterministic() -> None:
    mask = ComponentMask.from_retained_identifiers(
        ("H0", "H2", "N0", "N511")
    )

    first = component_mask_record(mask)
    second = component_mask_record(mask)

    assert first == second
    assert first["retained_attention_head_count"] == 2
    assert first["retained_mlp_neuron_count"] == 2
    assert first["retained_component_count"] == 4
    assert first["retained_component_ids"] == [
        "H0",
        "H2",
        "N0",
        "N511",
    ]
    assert len(first["mask_identity_sha256"]) == 64
