"""Cross-deliverable contract tests for Stage 1 of the follow-up."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from circuit_families.followup_namespace import (
    APPROVED_LOGICAL_ROOTS,
    NAMESPACE_VERSION,
)
from circuit_families.predecessor_link import load_predecessor_link

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "followup/manifests/predecessor_link_v1.json"
VISIBILITY = ROOT / "docs/distillation_followup/prior_results_visibility.md"

EXPECTED_PREDECESSOR_COMMIT = (
    "a55509537a70a225fedc5ce3a1c8236110974a6e"
)
EXPECTED_PROTOCOL_SHA256 = (
    "39a2852052a2e6c0e28f722d8c644be2a6897b2444f969342789385d048b47a7"
)
EXPECTED_IMPLEMENTATION_ORDER_SHA256 = (
    "416e40146eee8aaf8a42a8e6c28b1e9219c33cdc90f46b01789ce5903fed1e34"
)
EXPECTED_FREEZE_SHA256 = (
    "248a111f7c328fac81ebe5be5cbd45582487635d43e31848c6a0cd6f06a29f30"
)
EXPECTED_DATASET_SHA256 = (
    "af13d2181f5f1122bc528c6dfadbdc67b0a38ea02c10b4fd504a492aca8afafa"
)
EXPECTED_SPLIT_SHA256 = (
    "c83ac398724817fae6a0d137d0f1c6d0b8786eee43efaff5c3d34de0a891b7f2"
)


def test_namespace_contract_is_exact() -> None:
    assert NAMESPACE_VERSION == "circuit-families-distillation/v1"
    assert len(APPROVED_LOGICAL_ROOTS) == 13
    assert set(APPROVED_LOGICAL_ROOTS) == {
        "archives",
        "configs",
        "discovery_raw",
        "excluded_development",
        "figures",
        "local_scratch",
        "manifests",
        "notes",
        "reproduction_bundles",
        "reviewed_tables",
        "student_checkpoints",
        "student_outputs",
        "teacher_cache",
    }


def test_canonical_predecessor_identity_is_exact() -> None:
    record = load_predecessor_link(MANIFEST)
    predecessor = record["predecessor"]

    assert predecessor["analysis_freeze_commit"] == EXPECTED_PREDECESSOR_COMMIT
    assert predecessor["protocol"]["sha256"] == EXPECTED_PROTOCOL_SHA256
    assert (
        predecessor["implementation_order"]["sha256"]
        == EXPECTED_IMPLEMENTATION_ORDER_SHA256
    )
    assert (
        predecessor["analysis_freeze_manifest"]["sha256"]
        == EXPECTED_FREEZE_SHA256
    )


def test_canonical_dataset_identity_is_exact() -> None:
    record = load_predecessor_link(MANIFEST)
    dataset = record["dataset"]

    assert dataset["run_id"] == "modular-addition-dataset-s0-7ef9c73ff18f"
    assert dataset["dataset_sha256"] == EXPECTED_DATASET_SHA256
    assert dataset["split_sha256"] == EXPECTED_SPLIT_SHA256


def test_teacher_run_roster_is_exact_and_stage3_is_unresolved() -> None:
    record = load_predecessor_link(MANIFEST)

    assert len(record["teacher_runs"]) == 5
    assert [row["teacher_seed"] for row in record["teacher_runs"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [row["run_id"] for row in record["teacher_runs"]] == [
        "stage18-main-training-s0-58b8c1235464",
        "modular-addition-training-s1-5f1bc9dee7ab",
        "stage18-main-training-s2-c70f62c0fa7c",
        "stage18-main-training-s3-4c0c7c63ce2f",
        "stage18-main-training-s4-c2881c226349",
    ]

    stage3 = record["stage3_checkpoint_registry"]
    assert stage3["status"] == "deferred_to_stage_3"
    assert stage3["resolved"] is False
    assert stage3["selection_records"] == []


def test_stage1_visibility_state_is_exact() -> None:
    record = load_predecessor_link(MANIFEST)
    visibility = record["prior_results_visibility"]

    assert visibility == {
        "followup_distillation_endpoints_produced": False,
        "followup_predictive_fidelity_endpoints_produced": False,
        "predecessor_analysis_freeze_complete": True,
        "predecessor_primary_analysis_visible": True,
        "predecessor_results_are_blinded_pilot_evidence": False,
    }
    assert record["metadata"] == {
        "record_type": "predecessor_link",
        "scientific_execution": False,
        "stage": 1,
    }


def test_canonical_path_fields_are_portable() -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def visit(value, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key == "path":
            assert isinstance(value, str)
            assert not Path(value).is_absolute()
            assert not value.startswith("/Users/")
            assert ".." not in Path(value).parts

    visit(record)


def test_visibility_declaration_matches_machine_state() -> None:
    record = load_predecessor_link(MANIFEST)
    text = VISIBILITY.read_text(encoding="utf-8")

    assert record["prior_results_visibility"][
        "predecessor_primary_analysis_visible"
    ] is True
    assert "already visible before this" in text

    assert record["prior_results_visibility"][
        "predecessor_results_are_blinded_pilot_evidence"
    ] is False
    assert "not classified as blinded pilot evidence" in text

    assert record["prior_results_visibility"][
        "followup_distillation_endpoints_produced"
    ] is False
    assert "no follow-up distillation endpoint has been produced" in text

    assert record["prior_results_visibility"][
        "followup_predictive_fidelity_endpoints_produced"
    ] is False
    assert "no follow-up predictive-fidelity endpoint has been produced" in text

    assert "no student training has begun" in text
    assert "no follow-up circuit discovery has begun" in text
    assert "no follow-up comparative scientific analysis has been run" in text


def test_git_policy_classifications_are_enforced() -> None:
    """Exercise the repository's real Git ignore policy, not documentation text."""

    def is_ignored(relative_path: str) -> bool:
        completed = subprocess.run(
            [
                "git",
                "check-ignore",
                "-q",
                "--no-index",
                relative_path,
            ],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode not in {0, 1}:
            raise AssertionError(
                "git check-ignore failed for "
                f"{relative_path!r} with return code "
                f"{completed.returncode}"
            )
        return completed.returncode == 0

    ignored = [
        "followup/local/scratch/audit_probe.bin",
        "followup/artifacts/teacher_cache/audit_probe.bin",
        "followup/artifacts/student_checkpoints/audit_probe.bin",
        "followup/artifacts/student_outputs/audit_probe.bin",
        "followup/artifacts/discovery_raw/audit_probe.bin",
        "followup/artifacts/archives/audit_probe.bin",
        "followup/artifacts/reproduction_bundles/audit_probe.bin",
        "followup/excluded_development/audit_probe.bin",
    ]
    trackable = [
        "followup/excluded_development/audit_probe.json",
        "followup/manifests/audit_probe.json",
        "followup/configs/audit_probe.yaml",
        "followup/reviewed/tables/audit_probe.csv",
        "followup/reviewed/notes/audit_probe.md",
    ]

    for relative_path in ignored:
        assert is_ignored(relative_path), (
            f"Git policy unexpectedly permits {relative_path}"
        )

    for relative_path in trackable:
        assert not is_ignored(relative_path), (
            f"Git policy unexpectedly ignores {relative_path}"
        )


def test_teacher_roster_schema_is_exactly_five_seeds_zero_through_four() -> None:
    schema = json.loads(
        (ROOT / "schemas/predecessor_link_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    teacher_runs = schema["properties"]["teacher_runs"]

    assert teacher_runs["type"] == "array"
    assert teacher_runs["minItems"] == 5
    assert teacher_runs["maxItems"] == 5

    item = teacher_runs["items"]
    assert item["type"] == "object"
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {
        "teacher_seed",
        "run_id",
        "manifest",
    }

    teacher_seed = item["properties"]["teacher_seed"]
    assert teacher_seed["type"] == "integer"
    assert teacher_seed["minimum"] == 0
    assert teacher_seed["maximum"] == 4
