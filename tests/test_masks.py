"""Tests for frozen searchable components and mask serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_HOOK_NAME,
    ATTENTION_HEAD_IDS,
    COMPONENT_LOCATION_BY_ID,
    COMPONENT_LOCATIONS,
    MLP_NEURON_HOOK_NAME,
    MLP_NEURON_IDS,
    SEARCHABLE_COMPONENT_COUNT,
    SEARCHABLE_COMPONENT_IDS,
    ComponentMask,
    component_location,
    load_component_mask,
    save_component_mask,
)
from circuit_families.training.checkpoints import file_sha256


def test_searchable_component_universe_is_exact_and_unique() -> None:
    assert SEARCHABLE_COMPONENT_COUNT == 516
    assert len(SEARCHABLE_COMPONENT_IDS) == 516
    assert len(set(SEARCHABLE_COMPONENT_IDS)) == 516

    assert ATTENTION_HEAD_IDS == ("H0", "H1", "H2", "H3")
    assert MLP_NEURON_IDS[0] == "N0"
    assert MLP_NEURON_IDS[-1] == "N511"
    assert len(MLP_NEURON_IDS) == 512


def test_every_identifier_has_exactly_one_frozen_location() -> None:
    assert len(COMPONENT_LOCATIONS) == 516
    assert len(COMPONENT_LOCATION_BY_ID) == 516
    assert set(COMPONENT_LOCATION_BY_ID) == set(
        SEARCHABLE_COMPONENT_IDS
    )

    for index, identifier in enumerate(ATTENTION_HEAD_IDS):
        location = component_location(identifier)
        assert location.component_class == "attention_head"
        assert location.hook_name == ATTENTION_HEAD_HOOK_NAME
        assert location.activation_axis == 2
        assert location.index == index

    for index, identifier in enumerate(MLP_NEURON_IDS):
        location = component_location(identifier)
        assert location.component_class == "mlp_neuron"
        assert location.hook_name == MLP_NEURON_HOOK_NAME
        assert location.activation_axis == 2
        assert location.index == index


def test_all_retained_and_all_ablated_counts() -> None:
    retained = ComponentMask.all_retained()
    ablated = ComponentMask.all_ablated()

    assert retained.retained_attention_head_count == 4
    assert retained.retained_mlp_neuron_count == 512
    assert retained.retained_component_count == 516
    assert retained.retained_component_proportion == 1.0
    assert retained.retained_component_ids == SEARCHABLE_COMPONENT_IDS
    assert retained.ablated_component_ids == ()

    assert ablated.retained_attention_head_count == 0
    assert ablated.retained_mlp_neuron_count == 0
    assert ablated.retained_component_count == 0
    assert ablated.retained_component_proportion == 0.0
    assert ablated.retained_component_ids == ()
    assert ablated.ablated_component_ids == SEARCHABLE_COMPONENT_IDS


def test_single_component_and_identifier_constructors() -> None:
    head = ComponentMask.one_head_ablated("H0")
    neuron = ComponentMask.one_neuron_ablated("N511")

    assert head.ablated_component_ids == ("H0",)
    assert head.retained_attention_head_count == 3
    assert head.retained_mlp_neuron_count == 512

    assert neuron.ablated_component_ids == ("N511",)
    assert neuron.retained_attention_head_count == 4
    assert neuron.retained_mlp_neuron_count == 511

    retained = ComponentMask.from_retained_identifiers(
        ["H1", "N0", "N511"]
    )
    assert retained.retained_component_ids == ("H1", "N0", "N511")

    ablated = ComponentMask.from_ablated_identifiers(
        ["H2", "N3"]
    )
    assert ablated.ablated_component_ids == ("H2", "N3")


def test_invalid_lengths_binary_values_and_swapped_masks_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        ComponentMask(
            attention_head_mask=(1,) * 3,
            mlp_neuron_mask=(1,) * 512,
        )

    with pytest.raises(ValueError, match="exactly 512"):
        ComponentMask(
            attention_head_mask=(1,) * 4,
            mlp_neuron_mask=(1,) * 511,
        )

    with pytest.raises(ValueError, match="binary integer"):
        ComponentMask(
            attention_head_mask=(1, 1, 2, 1),
            mlp_neuron_mask=(1,) * 512,
        )

    with pytest.raises(ValueError, match="exactly 4"):
        ComponentMask(
            attention_head_mask=(1,) * 512,
            mlp_neuron_mask=(1,) * 4,
        )


def test_unknown_duplicate_and_wrong_class_identifiers_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        ComponentMask.from_retained_identifiers(["H4"])

    with pytest.raises(ValueError, match="duplicates"):
        ComponentMask.from_ablated_identifiers(["N0", "N0"])

    with pytest.raises(ValueError, match="not an attention-head"):
        ComponentMask.one_head_ablated("N0")

    with pytest.raises(ValueError, match="not an MLP-neuron"):
        ComponentMask.one_neuron_ablated("H0")


def test_mask_serialization_is_deterministic_and_exact(
    tmp_path: Path,
) -> None:
    mask = ComponentMask.from_ablated_identifiers(
        ["H0", "H3", "N0", "N17", "N511"]
    )
    first_path = save_component_mask(tmp_path / "first.json", mask)
    second_path = save_component_mask(tmp_path / "second.json", mask)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert file_sha256(first_path) == file_sha256(second_path)

    reloaded = load_component_mask(first_path)

    assert reloaded == mask
    assert reloaded.mask_id == mask.mask_id
    assert reloaded.attention_head_mask == mask.attention_head_mask
    assert reloaded.mlp_neuron_mask == mask.mlp_neuron_mask
    assert (
        reloaded.retained_component_count
        == mask.retained_component_count
    )
    assert reloaded.retained_component_ids == mask.retained_component_ids


def test_mask_reload_rejects_missing_or_inconsistent_metadata(
    tmp_path: Path,
) -> None:
    path = save_component_mask(
        tmp_path / "mask.json",
        ComponentMask.one_head_ablated("H0"),
    )
    record = json.loads(path.read_text(encoding="utf-8"))

    del record["mlp_neuron_mask"]
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_component_mask(path)

    path = save_component_mask(
        tmp_path / "mask.json",
        ComponentMask.one_head_ablated("H0"),
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    record["architecture"]["n_heads"] = 8
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible"):
        load_component_mask(path)

    path = save_component_mask(
        tmp_path / "mask.json",
        ComponentMask.one_head_ablated("H0"),
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    record["retained_component_count"] = 516
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent"):
        load_component_mask(path)
