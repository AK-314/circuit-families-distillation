from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    Stage3AvailabilityIndex,
)
from circuit_families.stage4_schema_common import (
    CommonSchemaContract,
    Stage4SchemaError,
)
from circuit_families.stage4_schema_graph import (
    validate_stage4_record_graph,
)

ROOT = Path(__file__).resolve().parents[1]

VALID_PATH = (
    ROOT / "tests/fixtures/stage4/synthetic_valid_bundle_v1.json"
)
INVALID_PATH = (
    ROOT / "tests/fixtures/stage4/synthetic_invalid_bundle_v1.json"
)

VOCAB = json.loads(
    (
        ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
    ).read_text()
)

IDENTITY = json.loads(
    (
        ROOT
        / "followup/configs/stage4_condition_identity_spec_v1.json"
    ).read_text()
)

REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)

REGISTRY = json.loads(REGISTRY_PATH.read_text())
REGISTRY_SHA = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()

DECISIONS = json.loads(
    (
        ROOT / "followup/configs/stage2_unresolved_decisions_v1.json"
    ).read_text()
)

STAGE3 = Stage3AvailabilityIndex.from_registry(REGISTRY)
CONTRACT = CommonSchemaContract.from_specs(VOCAB, IDENTITY)


def unresolved_ids():
    wanted = {f"UD-{n:03d}" for n in range(3, 15)}
    found = set()

    def walk(value):
        if isinstance(value, dict):
            ident = value.get("decision_id")

            if (
                ident in wanted
                and value.get("status") == "unresolved"
            ):
                found.add(ident)

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(DECISIONS)

    assert found == wanted
    return found


UNRESOLVED = unresolved_ids()


def load(path):
    return json.loads(path.read_text())


def validate(bundle):
    validate_stage4_record_graph(
        bundle["records_by_sha256"],
        contract=CONTRACT,
        stage3=STAGE3,
        stage3_registry=REGISTRY,
        stage3_registry_sha256=REGISTRY_SHA,
        current_unresolved_decision_ids=UNRESOLVED,
    )


def test_valid_bundle_contract_metadata():
    bundle = load(VALID_PATH)

    assert bundle["bundle_schema_version"] == "stage4-synthetic-bundle/v1"
    assert bundle["fixture_kind"] == "valid"
    assert bundle["scientific_data"] is False
    assert bundle["production_eligible"] is False
    assert bundle["expected_validation"] == "pass"


def test_valid_bundle_has_all_14_record_types():
    bundle = load(VALID_PATH)

    record_types = {
        record["record_type"]
        for record in bundle["records_by_sha256"].values()
    }

    assert len(record_types) == 14
    assert record_types == set(VOCAB["record_types"])


def test_valid_bundle_graph_passes():
    validate(load(VALID_PATH))


def test_invalid_bundle_contract_metadata():
    bundle = load(INVALID_PATH)

    assert bundle["bundle_schema_version"] == "stage4-synthetic-bundle/v1"
    assert bundle["fixture_kind"] == "invalid"
    assert bundle["scientific_data"] is False
    assert bundle["production_eligible"] is False
    assert bundle["expected_validation"] == "fail"
    assert bundle["mutation"] == (
        "endpoint1_not_smallest_qualifying_exact_evaluation"
    )


def test_invalid_bundle_fails_for_declared_reason():
    bundle = load(INVALID_PATH)

    with pytest.raises(
        Stage4SchemaError,
        match=bundle["expected_error_contains"],
    ):
        validate(bundle)


@pytest.mark.parametrize("path", [VALID_PATH, INVALID_PATH])
def test_bundle_has_no_private_absolute_paths(path):
    assert "/Users/" not in path.read_text()


@pytest.mark.parametrize("path", [VALID_PATH, INVALID_PATH])
def test_bundle_has_no_predecessor_absolute_paths(path):
    assert "/Projects/circuit-families/" not in path.read_text()


@pytest.mark.parametrize("path", [VALID_PATH, INVALID_PATH])
def test_bundle_declares_nonproduction_fixture(path):
    bundle = load(path)

    assert bundle["production_eligible"] is False
    assert bundle["scientific_data"] is False
