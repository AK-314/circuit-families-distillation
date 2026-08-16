from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_families.stage2_scientific_skeleton import (
    ScientificSkeletonError,
    load_scientific_skeleton,
    validate_scientific_skeleton,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "followup/manifests/stage2_scientific_skeleton_freeze_v1.json"


def canonical() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def mutate(item_id: str, field: str, value: str) -> dict:
    record = canonical()
    target = next(x for x in record["frozen_items"] if x["item_id"] == item_id)
    target[field] = value
    return record


def test_canonical_manifest_passes() -> None:
    record = load_scientific_skeleton(MANIFEST)
    assert len(record["frozen_items"]) == 18


def test_hierarchy_mutation_rejected() -> None:
    record = mutate("FS-002", "normative_statement", "student -> teacher")
    with pytest.raises(ScientificSkeletonError, match="FS-002 lost required frozen meaning"):
        validate_scientific_skeleton(record)


def test_population_unit_mutation_rejected() -> None:
    record = mutate(
        "FS-003",
        "normative_statement",
        "Student initialization is the population-level unit.",
    )
    with pytest.raises(ScientificSkeletonError, match="FS-003 lost required frozen meaning"):
        validate_scientific_skeleton(record)


def test_hard_soft_pooling_mutation_rejected() -> None:
    record = mutate(
        "FS-006",
        "normative_statement",
        "Hard-target and soft-target students may be pooled.",
    )
    with pytest.raises(ScientificSkeletonError, match="FS-006 lost required frozen meaning"):
        validate_scientific_skeleton(record)


def test_intact_mask_fallback_mutation_rejected() -> None:
    record = mutate(
        "FS-011",
        "normative_statement",
        "Endpoint 1 is the smallest recovered qualifying component proportion.",
    )
    with pytest.raises(ScientificSkeletonError, match="FS-011 lost required frozen meaning"):
        validate_scientific_skeleton(record)


def test_endpoint1_upper_bound_mutation_rejected() -> None:
    record = mutate(
        "FS-012",
        "normative_statement",
        "Endpoint 1 is the globally minimal sufficient proportion.",
    )
    with pytest.raises(ScientificSkeletonError, match="FS-012 lost required frozen meaning"):
        validate_scientific_skeleton(record)


def test_endpoint2_exact_count_mutation_rejected() -> None:
    record = mutate(
        "FS-013",
        "normative_statement",
        "Endpoint 2 is the exact number of distinct circuits.",
    )
    with pytest.raises(ScientificSkeletonError, match="FS-013 lost required frozen meaning"):
        validate_scientific_skeleton(record)


def test_outcome_categories_prediction_mutation_rejected() -> None:
    record = mutate(
        "FS-015",
        "normative_statement",
        "The six outcome categories are predictions.",
    )
    with pytest.raises(ScientificSkeletonError, match="FS-015 lost required frozen meaning"):
        validate_scientific_skeleton(record)


def test_fourier_primary_mutation_rejected() -> None:
    record = mutate(
        "FS-016",
        "normative_statement",
        "Fourier interchange is a primary experiment with three controls.",
    )
    with pytest.raises(ScientificSkeletonError, match="FS-016 lost required frozen meaning"):
        validate_scientific_skeleton(record)


def test_gated_extension_promotion_rejected() -> None:
    record = mutate(
        "FS-017",
        "normative_statement",
        "Entropy estimation is a required primary analysis.",
    )
    with pytest.raises(ScientificSkeletonError, match="FS-017 lost required frozen meaning"):
        validate_scientific_skeleton(record)


def test_partial_freeze_cannot_be_promoted() -> None:
    record = canonical()
    record["freeze_scope"]["numeric_protocol_fully_frozen"] = True
    with pytest.raises(ScientificSkeletonError, match="partial-freeze boundary"):
        validate_scientific_skeleton(record)

ALL_FROZEN_ITEM_IDS = [
    f"FS-{index:03d}"
    for index in range(1, 19)
]


@pytest.mark.parametrize("item_id", ALL_FROZEN_ITEM_IDS)
def test_every_frozen_normative_statement_is_immutable(
    item_id: str,
) -> None:
    record = mutate(
        item_id,
        "normative_statement",
        "CORRUPTED SCIENTIFIC STATEMENT",
    )

    with pytest.raises(
        ScientificSkeletonError,
        match=rf"{item_id} lost required frozen meaning",
    ):
        validate_scientific_skeleton(record)


@pytest.mark.parametrize("item_id", ALL_FROZEN_ITEM_IDS)
def test_every_frozen_claim_limit_is_immutable(
    item_id: str,
) -> None:
    record = mutate(
        item_id,
        "claim_limit",
        "CORRUPTED CLAIM BOUNDARY",
    )

    with pytest.raises(
        ScientificSkeletonError,
        match=rf"{item_id} lost required claim boundary",
    ):
        validate_scientific_skeleton(record)


def test_prohibited_claim_roster_is_immutable() -> None:
    record = canonical()
    record["claims_boundary"]["prohibited_claims"] = []

    with pytest.raises(
        ScientificSkeletonError,
        match="claims_boundary violates frozen Stage 2 claim contract",
    ):
        validate_scientific_skeleton(record)


def test_production_ready_boundary_is_immutable() -> None:
    record = canonical()
    record["claims_boundary"]["production_ready"] = True

    with pytest.raises(
        ScientificSkeletonError,
        match="claims_boundary violates frozen Stage 2 claim contract",
    ):
        validate_scientific_skeleton(record)
