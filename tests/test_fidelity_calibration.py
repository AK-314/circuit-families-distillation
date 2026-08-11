from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
from hashlib import sha256

import numpy as np
import pytest

from circuit_families.analysis.fidelity_calibration import (
    BIT_GENERATOR_NAME,
    CANDIDATE_THRESHOLDS,
    COMPONENT_UNIVERSE_SIZE,
    FULL_DATASET_EXAMPLE_COUNT,
    CalibrationCandidate,
    agreement_passes_threshold,
    component_identifiers,
    derive_random_seed,
    duplicate_mask_count,
    minimum_agreement_count,
    sample_matched_size_masks,
    select_primary_threshold,
    validate_selection_record_fields,
)


def make_candidates(
    *,
    first_passes: int = 5,
    second_passes: int = 5,
) -> list[CalibrationCandidate]:
    pass_counts = [first_passes, second_passes, 100, 100, 100, 100]
    return [
        CalibrationCandidate(
            threshold=threshold,
            retained_components=100,
            random_mask_pass_count=pass_count,
            fourier_compatible_or_explained=True,
            exact_evaluations=10_000,
        )
        for threshold, pass_count in zip(
            CANDIDATE_THRESHOLDS,
            pass_counts,
            strict=True,
        )
    ]


def test_exactly_six_frozen_thresholds_in_descending_order() -> None:
    assert CANDIDATE_THRESHOLDS == (
        Fraction(99, 100),
        Fraction(39, 40),
        Fraction(19, 20),
        Fraction(9, 10),
        Fraction(17, 20),
        Fraction(4, 5),
    )


def test_component_order_is_frozen() -> None:
    identifiers = component_identifiers()
    assert len(identifiers) == COMPONENT_UNIVERSE_SIZE
    assert identifiers[:4] == ("H0", "H1", "H2", "H3")
    assert identifiers[4] == "N0"
    assert identifiers[-1] == "N511"


@pytest.mark.parametrize("threshold", CANDIDATE_THRESHOLDS)
def test_seed_derivation_matches_frozen_formula(threshold: Fraction) -> None:
    result = derive_random_seed(threshold)
    expected_material = (
        "circuit-families|stage11-random-mask-calibration|"
        "training_run=modular-addition-training-s1-5f1bc9dee7ab|"
        "checkpoint_step=9050|"
        f"threshold={float(threshold):.6f}|"
        "component_universe=516|"
        "replicates=100"
    )
    expected_digest = sha256(expected_material.encode("utf-8")).hexdigest()

    assert result.seed_material == expected_material
    assert result.seed_digest == expected_digest
    assert result.seed_uint64 == int(expected_digest[:16], 16)
    assert result.bit_generator == BIT_GENERATOR_NAME
    assert result.numpy_version == np.__version__


def test_threshold_specific_seeds_are_distinct() -> None:
    seeds = [derive_random_seed(threshold).seed_uint64 for threshold in CANDIDATE_THRESHOLDS]
    assert len(set(seeds)) == len(CANDIDATE_THRESHOLDS)


def test_sampling_is_deterministic_and_uses_pcg64() -> None:
    first = sample_matched_size_masks(Fraction(99, 100), retained_count=146)
    second = sample_matched_size_masks(Fraction(99, 100), retained_count=146)

    assert first == second
    seed = derive_random_seed(Fraction(99, 100))
    generator = np.random.Generator(np.random.PCG64(seed.seed_uint64))
    expected_first = tuple(
        sorted(
            int(index)
            for index in generator.choice(
                COMPONENT_UNIVERSE_SIZE,
                size=146,
                replace=False,
            )
        )
    )
    assert first[0].retained_indices == expected_first


@pytest.mark.parametrize("retained_count", [0, 64, 146, 258, 516])
def test_every_mask_has_exact_target_count(retained_count: int) -> None:
    masks = sample_matched_size_masks(
        Fraction(4, 5),
        retained_count=retained_count,
        replicates=10,
    )
    assert len(masks) == 10
    for mask in masks:
        assert mask.retained_component_count == retained_count
        assert len(set(mask.retained_indices)) == retained_count
        assert tuple(sorted(mask.retained_indices)) == mask.retained_indices
        assert mask.retained_head_count + mask.retained_neuron_count == retained_count


def test_sampling_can_include_all_component_types_without_stratification() -> None:
    masks = sample_matched_size_masks(
        Fraction(39, 40),
        retained_count=119,
        replicates=100,
    )
    head_counts = {mask.retained_head_count for mask in masks}

    assert len(head_counts) > 1
    assert any(index < 4 for mask in masks for index in mask.retained_indices)
    assert any(index >= 4 for mask in masks for index in mask.retained_indices)


def test_duplicate_masks_are_counted_without_rejection() -> None:
    masks = sample_matched_size_masks(
        Fraction(9, 10),
        retained_count=0,
        replicates=4,
    )
    assert duplicate_mask_count(masks) == 3


@pytest.mark.parametrize("threshold", CANDIDATE_THRESHOLDS)
def test_exact_integer_threshold_boundary(threshold: Fraction) -> None:
    minimum = minimum_agreement_count(threshold)

    assert agreement_passes_threshold(minimum, threshold)
    assert not agreement_passes_threshold(minimum - 1, threshold)
    assert (
        minimum * threshold.denominator
        >= FULL_DATASET_EXAMPLE_COUNT * threshold.numerator
    )


def test_exactly_five_random_passes_qualifies() -> None:
    selected, results = select_primary_threshold(make_candidates(first_passes=5))
    assert selected == Fraction(99, 100)
    assert results[0].random_mask_pass_count_at_most_5 is True
    assert results[0].qualifies is True


def test_six_random_passes_fails_and_next_threshold_is_selected() -> None:
    selected, results = select_primary_threshold(
        make_candidates(first_passes=6, second_passes=5)
    )
    assert selected == Fraction(39, 40)
    assert results[0].qualifies is False
    assert results[1].qualifies is True


@pytest.mark.parametrize(
    ("retained_components", "exact_evaluations", "fourier_ok", "qualifies"),
    [
        (258, 10_000, True, True),
        (259, 10_000, True, False),
        (258, 10_001, True, False),
        (258, 10_000, False, False),
    ],
)
def test_frozen_qualification_boundaries(
    retained_components: int,
    exact_evaluations: int,
    fourier_ok: bool,
    qualifies: bool,
) -> None:
    candidates = make_candidates()
    candidates[0] = CalibrationCandidate(
        threshold=Fraction(99, 100),
        retained_components=retained_components,
        random_mask_pass_count=5,
        fourier_compatible_or_explained=fourier_ok,
        exact_evaluations=exact_evaluations,
    )

    selected, results = select_primary_threshold(candidates)
    assert results[0].qualifies is qualifies
    assert (selected == Fraction(99, 100)) is qualifies


def test_highest_qualifying_threshold_cannot_be_replaced() -> None:
    candidates = [
        CalibrationCandidate(
            threshold=threshold,
            retained_components=100,
            random_mask_pass_count=0,
            fourier_compatible_or_explained=True,
            exact_evaluations=1,
        )
        for threshold in CANDIDATE_THRESHOLDS
    ]
    selected, _ = select_primary_threshold(reversed(candidates))
    assert selected == Fraction(99, 100)


def test_no_threshold_is_explicit() -> None:
    candidates = [
        CalibrationCandidate(
            threshold=threshold,
            retained_components=259,
            random_mask_pass_count=100,
            fourier_compatible_or_explained=False,
            exact_evaluations=10_001,
        )
        for threshold in CANDIDATE_THRESHOLDS
    ]
    selected, results = select_primary_threshold(candidates)
    assert selected is None
    assert not any(result.qualifies for result in results)


def test_candidate_grid_must_be_exact() -> None:
    with pytest.raises(ValueError, match="exactly the six"):
        select_primary_threshold(make_candidates()[:-1])


@pytest.mark.parametrize(
    "prohibited_field",
    [
        "pre_grokking",
        "transition",
        "family_count",
        "random_label_results",
        "no_generalisation_results",
        "across_seed_outcomes",
        "anticipated_stage12_behaviour",
        "hypothesis_effect",
    ],
)
def test_prohibited_evidence_fields_are_rejected(prohibited_field: str) -> None:
    with pytest.raises(ValueError, match="prohibited evidence"):
        validate_selection_record_fields(
            {
                "threshold": 0.99,
                "retained_components": 146,
                prohibited_field: "must not enter selection",
            }
        )


def test_candidate_schema_contains_only_permitted_fields() -> None:
    assert {field.name for field in fields(CalibrationCandidate)} == {
        "threshold",
        "retained_components",
        "random_mask_pass_count",
        "fourier_compatible_or_explained",
        "exact_evaluations",
    }
