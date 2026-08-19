from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_families.stage5bc.target_cache import (
    ATOMIC_COMPLETION_PROTOCOL,
    CACHE_MANIFEST_SCHEMA_VERSION,
    CENTRING_SEMANTICS,
    FULL_DOMAIN_EXAMPLE_COUNT,
    STAGE4_CACHE_KINDS,
    TargetCacheContractError,
    TargetCacheManifest,
)

FIXTURE = Path(
    "tests/fixtures/stage5bc/technical_cache_manifest_v1.json"
)
SCHEMA = Path(
    "followup/schemas/stage5bc/"
    "teacher_target_cache_manifest_v1.schema.json"
)


def _raw() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _full_domain_mapping() -> dict:
    raw = _raw()
    count = FULL_DOMAIN_EXAMPLE_COUNT

    raw["manifest_id"] = "cache-full-domain-shape-contract/v1"
    raw["technical_fixture"] = False
    raw["stage4_record_serializable"] = True
    raw["example_count"] = count
    raw["input_order"]["example_count"] = count

    raw["representations"]["raw_logits"]["shape"][0] = count
    raw["representations"]["centred_logits"]["shape"][0] = count
    raw["representations"]["argmax"]["shape"][0] = count

    raw["payload"]["storage_class"] = "external_large_object"
    raw["payload"]["path"] = "external-cache/teacher-target-cache.bin"

    return raw


def test_technical_fixture_round_trips_with_exact_contract() -> None:
    manifest = TargetCacheManifest.from_json_file(FIXTURE)

    assert manifest.example_count == 4
    assert manifest.class_count == 3
    assert manifest.technical_fixture is True
    assert manifest.stage4_record_serializable is False

    rendered = manifest.to_mapping()

    assert rendered["schema_version"] == CACHE_MANIFEST_SCHEMA_VERSION
    assert tuple(rendered["stage4_cache_kinds"]) == STAGE4_CACHE_KINDS
    assert rendered["input_order"]["exact_order_required"] is True
    assert (
        rendered["representations"]["centred_logits"]["representation"]
        == CENTRING_SEMANTICS
    )
    assert (
        rendered["completion"]["atomic_write_protocol"]
        == ATOMIC_COMPLETION_PROTOCOL
    )


def test_full_domain_12769_shape_contract_is_supported_without_payload_generation() -> None:
    manifest = TargetCacheManifest.from_mapping(_full_domain_mapping())

    assert manifest.example_count == FULL_DOMAIN_EXAMPLE_COUNT
    assert manifest.technical_fixture is False
    assert manifest.stage4_record_serializable is True
    assert (
        manifest.to_mapping()["payload"]["storage_class"]
        == "external_large_object"
    )


def test_sub_full_domain_nontechnical_cache_is_rejected() -> None:
    raw = _raw()
    raw["technical_fixture"] = False
    raw["stage4_record_serializable"] = True
    raw["payload"]["storage_class"] = "external_large_object"

    with pytest.raises(
        TargetCacheContractError,
        match="sub-full-domain caches must be marked technical_fixture",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_technical_fixture_can_never_be_stage4_record_serializable() -> None:
    raw = _raw()
    raw["stage4_record_serializable"] = True

    with pytest.raises(
        TargetCacheContractError,
        match="sub-full-domain technical fixtures cannot be serialized",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_technical_fixture_requires_small_file_storage_class() -> None:
    raw = _raw()
    raw["payload"]["storage_class"] = "external_large_object"

    with pytest.raises(
        TargetCacheContractError,
        match="technical fixtures must use technical small-file storage",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_full_domain_stage4_contract_requires_external_storage() -> None:
    raw = _full_domain_mapping()
    raw["payload"]["storage_class"] = "technical_fixture_small_file"

    with pytest.raises(
        TargetCacheContractError,
        match="external_large_object",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_input_order_count_must_match_manifest_count() -> None:
    raw = _raw()
    raw["input_order"]["example_count"] = 3

    with pytest.raises(
        TargetCacheContractError,
        match="input_order.example_count",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_exact_order_requirement_cannot_be_disabled() -> None:
    raw = _raw()
    raw["input_order"]["exact_order_required"] = False

    with pytest.raises(
        TargetCacheContractError,
        match="exact_order_required must be true",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_reordered_input_hash_changes_canonical_manifest_hash() -> None:
    first = TargetCacheManifest.from_mapping(_raw())

    changed = _raw()
    changed["input_order"]["ordered_input_ids_sha256"] = "e" * 64
    second = TargetCacheManifest.from_mapping(changed)

    assert first.manifest_sha256() != second.manifest_sha256()


@pytest.mark.parametrize(
    ("path", "shape"),
    [
        (("raw_logits",), [3, 3]),
        (("centred_logits",), [4, 2]),
        (("argmax",), [3]),
    ],
)
def test_representation_shapes_must_match_example_and_class_counts(
    path: tuple[str, ...],
    shape: list[int],
) -> None:
    raw = _raw()
    raw["representations"][path[0]]["shape"] = shape

    with pytest.raises(TargetCacheContractError, match="shape mismatch"):
        TargetCacheManifest.from_mapping(raw)


def test_raw_logits_dtype_is_explicit() -> None:
    raw = _raw()
    raw["representations"]["raw_logits"]["dtype"] = "unknown"

    with pytest.raises(
        TargetCacheContractError,
        match="explicit floating dtype",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_centred_logit_semantics_are_exact() -> None:
    raw = _raw()
    raw["representations"]["centred_logits"]["representation"] = (
        "subtract_global_mean"
    )

    with pytest.raises(
        TargetCacheContractError,
        match="subtract_per_input_class_mean",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_argmax_hash_is_mandatory_sha256() -> None:
    raw = _raw()
    raw["representations"]["argmax"]["sha256"] = "not-a-hash"

    with pytest.raises(
        TargetCacheContractError,
        match="lowercase SHA-256 hex",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_probability_absence_cannot_hide_extra_metadata() -> None:
    raw = _raw()
    raw["representations"]["probabilities"]["dtype"] = "float32"

    with pytest.raises(
        TargetCacheContractError,
        match="must contain only 'present'",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_optional_probabilities_can_be_explicitly_present() -> None:
    raw = _raw()
    raw["representations"]["probabilities"] = {
        "present": True,
        "representation": "teacher_probabilities",
        "shape": [4, 3],
        "dtype": "float32",
        "sha256": "f" * 64,
    }

    manifest = TargetCacheManifest.from_mapping(raw)

    assert manifest.to_mapping()["representations"]["probabilities"][
        "present"
    ] is True


@pytest.mark.parametrize(
    "field",
    [
        "dataset_sha256",
        "split_sha256",
        "task_config_sha256",
        "model_config_sha256",
        "training_config_sha256",
        "component_basis_sha256",
    ],
)
def test_all_provenance_hashes_are_mandatory_sha256(field: str) -> None:
    raw = _raw()
    raw["provenance_hashes"][field] = "bad"

    with pytest.raises(
        TargetCacheContractError,
        match="lowercase SHA-256 hex",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_teacher_reference_is_exactly_stage4_teacher_reference() -> None:
    raw = _raw()
    raw["teacher_reference"]["record_type"] = "student_attempt"

    with pytest.raises(
        TargetCacheContractError,
        match="teacher_reference.record_type",
    ):
        TargetCacheManifest.from_mapping(raw)


@pytest.mark.parametrize(
    "bad_path",
    [
        "/absolute/cache.bin",
        "../escape/cache.bin",
        "dir\\cache.bin",
    ],
)
def test_payload_path_must_be_portable_relative(bad_path: str) -> None:
    raw = _raw()
    raw["payload"]["path"] = bad_path

    with pytest.raises(TargetCacheContractError):
        TargetCacheManifest.from_mapping(raw)


def test_payload_sha256_is_mandatory() -> None:
    raw = _raw()
    raw["payload"]["sha256"] = "0"

    with pytest.raises(
        TargetCacheContractError,
        match="payload.sha256",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_completion_must_declare_atomic_protocol() -> None:
    raw = _raw()
    raw["completion"]["atomic_write_protocol"] = "plain-write/v1"

    with pytest.raises(
        TargetCacheContractError,
        match="atomic_write_protocol",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_incomplete_cache_manifest_is_rejected() -> None:
    raw = _raw()
    raw["completion"]["completion_state"] = "partial"

    with pytest.raises(
        TargetCacheContractError,
        match="completion_state must be 'complete'",
    ):
        TargetCacheManifest.from_mapping(raw)


def test_canonical_manifest_bytes_are_key_order_independent() -> None:
    raw = _raw()
    reordered = dict(reversed(list(raw.items())))

    first = TargetCacheManifest.from_mapping(raw)
    second = TargetCacheManifest.from_mapping(reordered)

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.manifest_sha256() == second.manifest_sha256()


def test_schema_encodes_core_part_f_boundary() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    properties = schema["properties"]

    assert schema["additionalProperties"] is False
    assert properties["schema_version"]["const"] == (
        CACHE_MANIFEST_SCHEMA_VERSION
    )
    assert properties["stage4_cache_kinds"]["const"] == list(
        STAGE4_CACHE_KINDS
    )
    assert (
        properties["input_order"]["properties"]["exact_order_required"][
            "const"
        ]
        is True
    )
    assert (
        schema["$defs"]["centred_representation"]["properties"][
            "representation"
        ]["const"]
        == CENTRING_SEMANTICS
    )
