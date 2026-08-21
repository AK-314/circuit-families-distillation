from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_families.stage5d import (
    SyntheticIngestionError,
    canonical_json_bytes,
    load_and_normalize_ingestion,
    normalize_ingestion_mapping,
    normalized_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = (
    ROOT
    / "tests/fixtures/stage5d/synthetic_ingestion_envelope_v1.json"
)


def _raw() -> dict[str, object]:
    return json.loads(ENVELOPE.read_text(encoding="utf-8"))


def _collection(
    raw: dict[str, object],
    name: str,
) -> list[dict[str, object]]:
    value = raw[name]
    assert isinstance(value, list)
    return value


def _record(
    wrapper: dict[str, object],
) -> dict[str, object]:
    value = wrapper["record"]
    assert isinstance(value, dict)
    return value


def test_normalized_output_is_canonical_and_preserves_states() -> None:
    normalized = load_and_normalize_ingestion(ENVELOPE)

    assert normalized["schema_version"] == (
        "stage5d_normalized_universe_v1"
    )
    assert normalized["scientific_data"] is False
    assert normalized["production_eligible"] is False

    attempts = normalized["student_attempts"]
    eligibilities = normalized["eligibility_records"]
    cells = normalized["cell_expectations"]

    assert isinstance(attempts, list)
    assert isinstance(eligibilities, list)
    assert isinstance(cells, list)

    assert any(record["outcome"] == "failed" for record in attempts)
    assert any(
        record["status"] == "ineligible"
        for record in eligibilities
    )
    assert any(
        record["status"] == "inapplicable"
        for record in eligibilities
    )
    assert any(record["state"] == "unavailable" for record in cells)
    assert any(record["state"] == "unresolved" for record in cells)

    encoded = canonical_json_bytes(normalized)
    assert encoded.endswith(b"\n")
    assert normalized_sha256(normalized) == normalized_sha256(
        json.loads(encoded)
    )


def test_input_order_does_not_change_normalized_output() -> None:
    raw = _raw()
    baseline = normalize_ingestion_mapping(raw)

    shuffled = copy.deepcopy(raw)
    rng = random.Random(20260820)

    for name in (
        "method_budgets",
        "teacher_inventories",
        "student_attempts",
        "eligibility_records",
        "direct_teacher_endpoints",
        "student_endpoints",
        "cell_expectations",
    ):
        collection = _collection(shuffled, name)
        rng.shuffle(collection)

    observed = normalize_ingestion_mapping(shuffled)

    assert canonical_json_bytes(observed) == canonical_json_bytes(
        baseline
    )
    assert normalized_sha256(observed) == normalized_sha256(baseline)
    assert observed["source_provenance"]["source_sha256"] == (
        baseline["source_provenance"]["source_sha256"]
    )


def test_duplicate_attempt_id_is_rejected() -> None:
    raw = _raw()
    attempts = _collection(raw, "student_attempts")
    attempts.append(copy.deepcopy(attempts[0]))

    with pytest.raises(
        SyntheticIngestionError,
        match="duplicate attempt_id",
    ):
        normalize_ingestion_mapping(raw)


def test_conflicting_duplicate_attempt_id_is_rejected() -> None:
    raw = _raw()
    attempts = _collection(raw, "student_attempts")

    duplicate = copy.deepcopy(attempts[0])
    record = _record(duplicate)
    record["outcome"] = "failed"
    record["failure_reason"] = "conflicting_duplicate_fixture"
    attempts.append(duplicate)

    with pytest.raises(
        SyntheticIngestionError,
        match="duplicate attempt_id",
    ):
        normalize_ingestion_mapping(raw)


def test_orphan_eligibility_record_is_rejected() -> None:
    raw = _raw()
    eligibility = _collection(raw, "eligibility_records")
    record = _record(eligibility[0])
    record["attempt_id"] = "missing_attempt"

    with pytest.raises(
        SyntheticIngestionError,
        match="unknown attempt",
    ):
        normalize_ingestion_mapping(raw)


def test_mixed_provenance_is_rejected() -> None:
    raw = _raw()
    endpoints = _collection(raw, "student_endpoints")
    endpoints[0]["provenance_id"] = "other_provenance"

    with pytest.raises(
        SyntheticIngestionError,
        match="mixed provenance",
    ):
        normalize_ingestion_mapping(raw)


def test_defined_endpoint_from_ineligible_attempt_is_rejected() -> None:
    raw = _raw()

    endpoints = _collection(raw, "student_endpoints")
    endpoint = _record(endpoints[0])
    attempt_id = endpoint["attempt_id"]

    for wrapper in _collection(raw, "eligibility_records"):
        eligibility = _record(wrapper)
        if eligibility["attempt_id"] == attempt_id:
            eligibility["status"] = "ineligible"
            eligibility["reason"] = "synthetic_mutation"
            break
    else:
        raise AssertionError("matching eligibility record not found")

    with pytest.raises(
        SyntheticIngestionError,
        match="cannot come from ineligible",
    ):
        normalize_ingestion_mapping(raw)


def test_hard_soft_identity_collision_is_rejected() -> None:
    raw = _raw()

    endpoints = _collection(raw, "student_endpoints")
    endpoint = _record(endpoints[0])
    identity = endpoint["identity"]
    assert isinstance(identity, dict)

    current = identity["distillation_condition"]
    identity["distillation_condition"] = (
        "soft" if current == "hard" else "hard"
    )

    with pytest.raises(
        SyntheticIngestionError,
        match="identity does not match its attempt",
    ):
        normalize_ingestion_mapping(raw)


def test_record_unreachable_from_teacher_inventory_is_rejected() -> None:
    raw = _raw()

    attempts = _collection(raw, "student_attempts")
    attempt = _record(attempts[0])
    attempt["teacher_seed"] = 999

    with pytest.raises(
        SyntheticIngestionError,
        match="unknown teacher seed",
    ):
        normalize_ingestion_mapping(raw)


def test_record_unreachable_from_phase_inventory_is_rejected() -> None:
    raw = _raw()

    attempts = _collection(raw, "student_attempts")
    attempt = _record(attempts[0])
    attempt["phase"] = "phase_not_in_inventory"

    with pytest.raises(
        SyntheticIngestionError,
        match="absent from teacher inventory",
    ):
        normalize_ingestion_mapping(raw)


def test_unavailable_phase_cannot_emit_defined_teacher_endpoint() -> None:
    raw = _raw()

    direct = _collection(raw, "direct_teacher_endpoints")
    unavailable = next(
        wrapper
        for wrapper in direct
        if _record(wrapper)["state"] == "unavailable"
    )
    record = _record(unavailable)
    record["state"] = "defined"
    record["value"] = 0.5

    with pytest.raises(
        SyntheticIngestionError,
        match=r"(references unavailable phase|must preserve unavailable phase state)",
    ):
        normalize_ingestion_mapping(raw)


def test_endpoint_boundary_values_survive_normalization() -> None:
    normalized = load_and_normalize_ingestion(ENVELOPE)

    all_endpoints = [
        *normalized["direct_teacher_endpoints"],
        *normalized["student_endpoints"],
    ]

    assert any(
        record["identity"]["endpoint_id"] == "endpoint_1"
        and record["state"] == "defined"
        and record["value"] == 1.0
        for record in all_endpoints
    )
    assert any(
        record["identity"]["endpoint_id"] == "endpoint_2"
        and record["state"] == "defined"
        and record["value"] == 0
        for record in all_endpoints
    )


def test_pythonhashseed_does_not_change_normalized_hash() -> None:
    code = f"""
from pathlib import Path
from circuit_families.stage5d import (
    load_and_normalize_ingestion,
    normalized_sha256,
)
path = Path({str(ENVELOPE)!r})
print(normalized_sha256(load_and_normalize_ingestion(path)))
"""

    hashes = []

    for seed in ("1", "987654"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = str(ROOT / "src")

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        hashes.append(result.stdout.strip())

    assert hashes[0] == hashes[1]
