from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from circuit_families.stage5bc.target_cache import (
    TargetCacheBatch,
    TargetCacheContractError,
    build_target_cache,
    centre_logits,
    load_target_cache,
)

CANONICAL_IDS = (
    "input-000",
    "input-001",
    "input-002",
    "input-003",
)

BASE_TEACHER_REFERENCE = {
    "record_type": "teacher_reference",
    "schema_version": "teacher_reference/v1",
    "condition_id": (
        "cfdid:v1:d3|teacher_seed=1|"
        "phase=stable%20post-grokking|"
        "distillation_condition=hard_target"
    ),
    "record_sha256": "5" * 64,
}

BASE_PROVENANCE = {
    "dataset_sha256": "6" * 64,
    "split_sha256": "7" * 64,
    "task_config_sha256": "8" * 64,
    "model_config_sha256": "9" * 64,
    "training_config_sha256": "a" * 64,
    "component_basis_sha256": "b" * 64,
}


def _logits() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 3.0, 2.0],
            [-1.0, -2.0, 4.0],
            [9.0, 2.0, 1.0],
            [0.25, 0.5, 0.75],
        ],
        dtype=torch.float32,
    )


def _probabilities() -> torch.Tensor:
    return torch.tensor(
        [
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.2, 0.3, 0.5],
        ],
        dtype=torch.float32,
    )


def _batches(
    *,
    logits: torch.Tensor | None = None,
    input_ids: tuple[str, ...] = CANONICAL_IDS,
    probabilities: torch.Tensor | None = None,
):
    values = _logits() if logits is None else logits

    yield TargetCacheBatch(
        input_ids=input_ids[:2],
        raw_logits=values[:2],
        probabilities=(
            probabilities[:2]
            if probabilities is not None
            else None
        ),
    )
    yield TargetCacheBatch(
        input_ids=input_ids[2:],
        raw_logits=values[2:],
        probabilities=(
            probabilities[2:]
            if probabilities is not None
            else None
        ),
    )


def _build(
    root: Path,
    *,
    logits: torch.Tensor | None = None,
    input_ids: tuple[str, ...] = CANONICAL_IDS,
    teacher_reference: dict | None = None,
    provenance: dict | None = None,
    ordering_ref: str = "technical-order/v1",
    probabilities: torch.Tensor | None = None,
    expected_input_ids: tuple[str, ...] | None = CANONICAL_IDS,
):
    return build_target_cache(
        output_root=root,
        manifest_relative_path="cache/manifest.json",
        payload_relative_path="cache/payload.bin",
        completion_relative_path="cache/complete.json",
        manifest_id="technical-integrity-cache/v1",
        ordering_ref=ordering_ref,
        expected_example_count=4,
        expected_class_count=3,
        teacher_reference=(
            BASE_TEACHER_REFERENCE
            if teacher_reference is None
            else teacher_reference
        ),
        provenance_hashes=(
            BASE_PROVENANCE
            if provenance is None
            else provenance
        ),
        batches=_batches(
            logits=logits,
            input_ids=input_ids,
            probabilities=probabilities,
        ),
        technical_fixture=True,
        stage4_record_serializable=False,
        expected_input_ids=expected_input_ids,
    )


def test_repeated_builds_are_byte_identical(tmp_path: Path) -> None:
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")

    assert first.payload_path.read_bytes() == second.payload_path.read_bytes()
    assert (
        first.manifest_path.read_bytes()
        == second.manifest_path.read_bytes()
    )
    assert (
        first.completion_path.read_bytes()
        == second.completion_path.read_bytes()
    )
    assert (
        first.manifest.manifest_sha256()
        == second.manifest.manifest_sha256()
    )


def test_builder_rejects_reordered_canonical_inputs(
    tmp_path: Path,
) -> None:
    reordered = (
        "input-001",
        "input-000",
        "input-002",
        "input-003",
    )

    with pytest.raises(
        TargetCacheContractError,
        match="canonical input order mismatch",
    ):
        _build(
            tmp_path,
            input_ids=reordered,
            expected_input_ids=CANONICAL_IDS,
        )


def test_loader_rejects_reordered_cache_against_expected_sequence(
    tmp_path: Path,
) -> None:
    reordered = (
        "input-001",
        "input-000",
        "input-002",
        "input-003",
    )
    _build(
        tmp_path,
        input_ids=reordered,
        expected_input_ids=None,
    )

    with pytest.raises(
        TargetCacheContractError,
        match="canonical input sequence",
    ):
        load_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
            expected_input_ids=CANONICAL_IDS,
        )


def test_loader_rejects_wrong_teacher_reference_context(
    tmp_path: Path,
) -> None:
    _build(tmp_path)

    wrong = copy.deepcopy(BASE_TEACHER_REFERENCE)
    wrong["record_sha256"] = "e" * 64

    with pytest.raises(
        TargetCacheContractError,
        match="teacher-reference context mismatch",
    ):
        load_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
            expected_teacher_reference=wrong,
        )


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
def test_loader_rejects_wrong_provenance_context(
    tmp_path: Path,
    field: str,
) -> None:
    _build(tmp_path)

    wrong = copy.deepcopy(BASE_PROVENANCE)
    wrong[field] = "e" * 64

    with pytest.raises(
        TargetCacheContractError,
        match="cache provenance context mismatch",
    ):
        load_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
            expected_provenance_hashes=wrong,
        )


@pytest.mark.parametrize(
    ("condition", "wrong_kind"),
    [
        ("hard_target", "teacher_logits"),
        ("soft_target", "teacher_argmax"),
    ],
)
def test_loader_rejects_hard_soft_cache_kind_confusion(
    tmp_path: Path,
    condition: str,
    wrong_kind: str,
) -> None:
    teacher = copy.deepcopy(BASE_TEACHER_REFERENCE)
    teacher["condition_id"] = (
        "cfdid:v1:d3|teacher_seed=1|"
        "phase=stable%20post-grokking|"
        f"distillation_condition={condition}"
    )

    _build(tmp_path, teacher_reference=teacher)

    with pytest.raises(
        TargetCacheContractError,
        match="hard/soft cache-kind context mismatch",
    ):
        load_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
            expected_stage4_cache_kind=wrong_kind,
        )


def test_loader_accepts_matching_hard_cache_kind_context(
    tmp_path: Path,
) -> None:
    _build(tmp_path)

    loaded = load_target_cache(
        output_root=tmp_path,
        manifest_relative_path="cache/manifest.json",
        expected_stage4_cache_kind="teacher_argmax",
    )

    assert torch.equal(
        loaded.stage4_view("teacher_argmax"),
        loaded.argmax,
    )


def test_builder_rejects_nonfinite_logits(tmp_path: Path) -> None:
    logits = _logits()
    logits[2, 1] = float("nan")

    with pytest.raises(
        TargetCacheContractError,
        match="non-finite",
    ):
        _build(tmp_path, logits=logits)


def test_builder_rejects_class_shape_corruption(
    tmp_path: Path,
) -> None:
    values = torch.ones((4, 2), dtype=torch.float32)

    with pytest.raises(
        TargetCacheContractError,
        match="class count",
    ):
        _build(tmp_path, logits=values)


def test_builder_rejects_streaming_dtype_change(
    tmp_path: Path,
) -> None:
    first = TargetCacheBatch(
        input_ids=CANONICAL_IDS[:2],
        raw_logits=_logits()[:2],
    )
    second = TargetCacheBatch(
        input_ids=CANONICAL_IDS[2:],
        raw_logits=_logits()[2:].to(torch.float64),
    )

    with pytest.raises(
        TargetCacheContractError,
        match="dtype changed across streaming batches",
    ):
        build_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
            payload_relative_path="cache/payload.bin",
            completion_relative_path="cache/complete.json",
            manifest_id="technical-dtype-corruption/v1",
            ordering_ref="technical-order/v1",
            expected_example_count=4,
            expected_class_count=3,
            teacher_reference=BASE_TEACHER_REFERENCE,
            provenance_hashes=BASE_PROVENANCE,
            batches=(first, second),
            technical_fixture=True,
            stage4_record_serializable=False,
            expected_input_ids=CANONICAL_IDS,
        )


def test_additive_gauge_preserves_centred_logits_and_argmax(
    tmp_path: Path,
) -> None:
    base_logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=torch.float32,
    )
    shifts = torch.tensor(
        [[10.0], [-3.0], [100.0], [7.0]],
        dtype=torch.float32,
    )
    shifted_logits = base_logits + shifts

    first = _build(tmp_path / "base", logits=base_logits)
    second = _build(tmp_path / "shifted", logits=shifted_logits)

    first_loaded = load_target_cache(
        output_root=tmp_path / "base",
        manifest_relative_path="cache/manifest.json",
    )
    second_loaded = load_target_cache(
        output_root=tmp_path / "shifted",
        manifest_relative_path="cache/manifest.json",
    )

    assert torch.equal(
        first_loaded.centred_logits,
        centre_logits(base_logits),
    )
    assert torch.equal(
        first_loaded.centred_logits,
        second_loaded.centred_logits,
    )
    assert torch.equal(first_loaded.argmax, second_loaded.argmax)

    first_manifest = first.manifest.to_mapping()
    second_manifest = second.manifest.to_mapping()

    assert (
        first_manifest["representations"]["raw_logits"]["sha256"]
        != second_manifest["representations"]["raw_logits"]["sha256"]
    )
    assert (
        first_manifest["representations"]["centred_logits"]["sha256"]
        == second_manifest["representations"]["centred_logits"]["sha256"]
    )
    assert (
        first_manifest["representations"]["argmax"]["sha256"]
        == second_manifest["representations"]["argmax"]["sha256"]
    )


def test_missing_completion_record_is_rejected(
    tmp_path: Path,
) -> None:
    built = _build(tmp_path)
    built.completion_path.unlink()

    with pytest.raises(
        TargetCacheContractError,
        match="completion record is missing",
    ):
        load_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
        )


def test_payload_tampering_is_rejected_before_decode(
    tmp_path: Path,
) -> None:
    built = _build(tmp_path)

    payload = bytearray(built.payload_path.read_bytes())
    payload[-1] ^= 0x01
    built.payload_path.write_bytes(payload)

    with pytest.raises(
        TargetCacheContractError,
        match="payload SHA-256 mismatch",
    ):
        load_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
        )


def test_truncated_payload_is_rejected(tmp_path: Path) -> None:
    built = _build(tmp_path)

    payload = built.payload_path.read_bytes()
    built.payload_path.write_bytes(payload[: len(payload) // 2])

    with pytest.raises(
        TargetCacheContractError,
        match="payload SHA-256 mismatch",
    ):
        load_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
        )


def test_manifest_tampering_is_rejected_by_expected_teacher_context(
    tmp_path: Path,
) -> None:
    built = _build(tmp_path)

    raw = built.manifest.to_mapping()
    raw["teacher_reference"]["record_sha256"] = "e" * 64

    from circuit_families.stage5bc.target_cache import TargetCacheManifest

    tampered = TargetCacheManifest.from_mapping(raw)
    built.manifest_path.write_bytes(tampered.canonical_bytes())

    with pytest.raises(
        TargetCacheContractError,
        match="teacher-reference context mismatch",
    ):
        load_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
            expected_teacher_reference=BASE_TEACHER_REFERENCE,
        )


@pytest.mark.parametrize(
    "case",
    [
        "input_order",
        "ordering_ref",
        "raw_logits",
        "teacher_reference",
        "dataset",
        "split",
        "task_config",
        "model_config",
        "training_config",
        "component_basis",
        "probabilities",
    ],
)
def test_cache_reference_changes_when_any_governed_input_changes(
    tmp_path: Path,
    case: str,
) -> None:
    base = _build(tmp_path / "base")
    base_hash = base.manifest.manifest_sha256()

    input_ids = CANONICAL_IDS
    ordering_ref = "technical-order/v1"
    logits = _logits()
    teacher = copy.deepcopy(BASE_TEACHER_REFERENCE)
    provenance = copy.deepcopy(BASE_PROVENANCE)
    probabilities = None
    expected_input_ids = CANONICAL_IDS

    if case == "input_order":
        input_ids = (
            "input-001",
            "input-000",
            "input-002",
            "input-003",
        )
        expected_input_ids = None
    elif case == "ordering_ref":
        ordering_ref = "technical-order-alternative/v1"
    elif case == "raw_logits":
        logits = logits.clone()
        logits[0, 0] += 1.0
    elif case == "teacher_reference":
        teacher["record_sha256"] = "e" * 64
    elif case == "dataset":
        provenance["dataset_sha256"] = "e" * 64
    elif case == "split":
        provenance["split_sha256"] = "e" * 64
    elif case == "task_config":
        provenance["task_config_sha256"] = "e" * 64
    elif case == "model_config":
        provenance["model_config_sha256"] = "e" * 64
    elif case == "training_config":
        provenance["training_config_sha256"] = "e" * 64
    elif case == "component_basis":
        provenance["component_basis_sha256"] = "e" * 64
    elif case == "probabilities":
        probabilities = _probabilities()
    else:
        raise AssertionError(case)

    changed = _build(
        tmp_path / "changed",
        logits=logits,
        input_ids=input_ids,
        teacher_reference=teacher,
        provenance=provenance,
        ordering_ref=ordering_ref,
        probabilities=probabilities,
        expected_input_ids=expected_input_ids,
    )

    assert changed.manifest.manifest_sha256() != base_hash
