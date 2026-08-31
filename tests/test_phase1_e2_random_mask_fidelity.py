"""Focused tests for Phase I E2 random-mask fidelity nulls."""

from __future__ import annotations

from fractions import Fraction

import pytest

from circuit_families.analysis.phase1_e2_random_mask_fidelity import (
    E2ValidationError,
    NullProfile,
    clopper_pearson_interval,
    derive_seed,
    minimum_agreement_count,
    sample_masks,
)


def profile() -> NullProfile:
    return NullProfile(
        model_seed=2,
        checkpoint_step=9050,
        retained_heads=4,
        retained_neurons=8,
        retained_components=12,
        observed_circuit_ids=("C1",),
    )


def test_minimum_agreement_count_is_exact() -> None:
    assert minimum_agreement_count(Fraction(99, 100), example_count=12769) == 12642
    assert minimum_agreement_count(Fraction(1, 2), example_count=3) == 2


def test_size_matched_sampling_preserves_only_total_size() -> None:
    _, masks = sample_masks(
        profile(),
        "size_matched",
        analysis_id="fixture",
        replicates=50,
    )
    assert len(masks) == 50
    assert all(item.mask.retained_component_count == 12 for item in masks)
    assert any(item.mask.retained_attention_head_count != 4 for item in masks)


def test_basis_stratified_sampling_preserves_composition() -> None:
    first_seed, first = sample_masks(
        profile(),
        "basis_stratified",
        analysis_id="fixture",
        replicates=50,
    )
    second_seed, second = sample_masks(
        profile(),
        "basis_stratified",
        analysis_id="fixture",
        replicates=50,
    )
    assert first_seed == second_seed
    assert first == second
    assert all(item.mask.retained_attention_head_count == 4 for item in first)
    assert all(item.mask.retained_mlp_neuron_count == 8 for item in first)


def test_seed_is_profile_and_null_specific() -> None:
    base = derive_seed(
        analysis_id="fixture",
        profile=profile(),
        null_model="size_matched",
        replicates=100,
    )
    stratified = derive_seed(
        analysis_id="fixture",
        profile=profile(),
        null_model="basis_stratified",
        replicates=100,
    )
    changed_profile = NullProfile(
        **{**profile().__dict__, "retained_neurons": 9, "retained_components": 13}
    )
    changed = derive_seed(
        analysis_id="fixture",
        profile=changed_profile,
        null_model="size_matched",
        replicates=100,
    )
    assert base != stratified
    assert base != changed


def test_clopper_pearson_boundaries_and_known_zero_success_bound() -> None:
    zero = clopper_pearson_interval(0, 100, confidence_level=0.95)
    full = clopper_pearson_interval(100, 100, confidence_level=0.95)
    assert zero == pytest.approx((0.0, 1.0 - 0.025 ** (1.0 / 100.0)))
    assert full == pytest.approx((0.025 ** (1.0 / 100.0), 1.0))
    with pytest.raises(E2ValidationError, match="cannot exceed"):
        clopper_pearson_interval(2, 1, confidence_level=0.95)
