from __future__ import annotations

import json
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

TEACHER_REFERENCE = {
    "record_type": "teacher_reference",
    "schema_version": "teacher_reference/v1",
    "condition_id": (
        "cfdid:v1:d3|teacher_seed=1|"
        "phase=stable%20post-grokking|"
        "distillation_condition=hard_target"
    ),
    "record_sha256": "5" * 64,
}

PROVENANCE = {
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


def _stream(
    *,
    with_probabilities: bool = False,
):
    logits = _logits()
    probabilities = _probabilities() if with_probabilities else None

    yield TargetCacheBatch(
        input_ids=("input-000", "input-001"),
        raw_logits=logits[:2],
        probabilities=(
            probabilities[:2]
            if probabilities is not None
            else None
        ),
    )
    yield TargetCacheBatch(
        input_ids=("input-002", "input-003"),
        raw_logits=logits[2:],
        probabilities=(
            probabilities[2:]
            if probabilities is not None
            else None
        ),
    )


def _build(
    root: Path,
    *,
    with_probabilities: bool = False,
):
    return build_target_cache(
        output_root=root,
        manifest_relative_path="cache/manifest.json",
        payload_relative_path="cache/payload.bin",
        completion_relative_path="cache/complete.json",
        manifest_id="technical-streaming-cache/v1",
        ordering_ref="technical-input-order/v1",
        expected_example_count=4,
        expected_class_count=3,
        teacher_reference=TEACHER_REFERENCE,
        provenance_hashes=PROVENANCE,
        batches=_stream(with_probabilities=with_probabilities),
        technical_fixture=True,
        stage4_record_serializable=False,
    )


def test_streaming_build_and_strict_load_round_trip(tmp_path: Path) -> None:
    built = _build(tmp_path)
    loaded = load_target_cache(
        output_root=tmp_path,
        manifest_relative_path="cache/manifest.json",
    )

    assert built.payload_path.is_file()
    assert built.completion_path.is_file()
    assert built.manifest_path.is_file()

    assert loaded.input_ids == (
        "input-000",
        "input-001",
        "input-002",
        "input-003",
    )

    assert torch.equal(loaded.raw_logits, _logits())
    assert torch.equal(
        loaded.centred_logits,
        centre_logits(_logits()),
    )
    assert torch.equal(
        loaded.argmax,
        torch.tensor([1, 2, 0, 2], dtype=torch.int64),
    )
    assert loaded.probabilities is None


def test_stage4_hard_and_logit_views_share_one_loaded_cache(
    tmp_path: Path,
) -> None:
    _build(tmp_path)
    loaded = load_target_cache(
        output_root=tmp_path,
        manifest_relative_path="cache/manifest.json",
    )

    assert torch.equal(
        loaded.stage4_view("teacher_argmax"),
        loaded.argmax,
    )
    assert torch.equal(
        loaded.stage4_view("teacher_logits"),
        loaded.centred_logits,
    )

    with pytest.raises(
        TargetCacheContractError,
        match="unsupported Stage 4 cache kind",
    ):
        loaded.stage4_view("unknown")


def test_centring_is_per_input_class_mean() -> None:
    raw = _logits()
    centred = centre_logits(raw)

    expected = raw - raw.mean(dim=-1, keepdim=True)

    assert torch.equal(centred, expected)
    assert torch.allclose(
        centred.mean(dim=-1),
        torch.zeros(raw.shape[0]),
        atol=1e-7,
        rtol=0.0,
    )


def test_optional_probabilities_stream_and_verify(tmp_path: Path) -> None:
    _build(tmp_path, with_probabilities=True)

    loaded = load_target_cache(
        output_root=tmp_path,
        manifest_relative_path="cache/manifest.json",
    )

    assert loaded.probabilities is not None
    assert torch.equal(loaded.probabilities, _probabilities())

    manifest = loaded.manifest.to_mapping()
    assert manifest["representations"]["probabilities"]["present"] is True


def test_builder_rejects_duplicate_canonical_input_ids(
    tmp_path: Path,
) -> None:
    logits = _logits()

    batches = [
        TargetCacheBatch(
            input_ids=("same", "input-001"),
            raw_logits=logits[:2],
        ),
        TargetCacheBatch(
            input_ids=("same", "input-003"),
            raw_logits=logits[2:],
        ),
    ]

    with pytest.raises(
        TargetCacheContractError,
        match="duplicate canonical input ID",
    ):
        build_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
            payload_relative_path="cache/payload.bin",
            completion_relative_path="cache/complete.json",
            manifest_id="technical-duplicate-test/v1",
            ordering_ref="technical-input-order/v1",
            expected_example_count=4,
            expected_class_count=3,
            teacher_reference=TEACHER_REFERENCE,
            provenance_hashes=PROVENANCE,
            batches=batches,
            technical_fixture=True,
            stage4_record_serializable=False,
        )


def test_builder_rejects_missing_streamed_example(
    tmp_path: Path,
) -> None:
    logits = _logits()

    batches = [
        TargetCacheBatch(
            input_ids=("input-000", "input-001"),
            raw_logits=logits[:2],
        ),
        TargetCacheBatch(
            input_ids=("input-002",),
            raw_logits=logits[2:3],
        ),
    ]

    with pytest.raises(
        TargetCacheContractError,
        match="streamed example count mismatch",
    ):
        build_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
            payload_relative_path="cache/payload.bin",
            completion_relative_path="cache/complete.json",
            manifest_id="technical-missing-test/v1",
            ordering_ref="technical-input-order/v1",
            expected_example_count=4,
            expected_class_count=3,
            teacher_reference=TEACHER_REFERENCE,
            provenance_hashes=PROVENANCE,
            batches=batches,
            technical_fixture=True,
            stage4_record_serializable=False,
        )


def test_loader_verifies_payload_hash_before_decoding(
    tmp_path: Path,
) -> None:
    built = _build(tmp_path)

    payload = bytearray(built.payload_path.read_bytes())
    payload[-1] ^= 1
    built.payload_path.write_bytes(payload)

    with pytest.raises(
        TargetCacheContractError,
        match="payload SHA-256 mismatch",
    ):
        load_target_cache(
            output_root=tmp_path,
            manifest_relative_path="cache/manifest.json",
        )


def test_builder_outputs_are_tiny_technical_files(
    tmp_path: Path,
) -> None:
    built = _build(tmp_path)

    sizes = {
        "manifest": built.manifest_path.stat().st_size,
        "payload": built.payload_path.stat().st_size,
        "completion": built.completion_path.stat().st_size,
    }

    assert max(sizes.values()) < 64 * 1024
    assert built.manifest.technical_fixture is True
    assert built.manifest.stage4_record_serializable is False

    completion = json.loads(
        built.completion_path.read_text(encoding="utf-8")
    )
    assert completion["example_count"] == 4
