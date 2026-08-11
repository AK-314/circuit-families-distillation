from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "manifests/post_stage17_additional_control_seed_count_freeze.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_additional_control_count_is_frozen_prospectively_at_zero() -> None:
    manifest = _manifest()
    lifecycle = manifest["lifecycle"]
    decision = manifest["decision"]

    assert lifecycle["stage18_scientific_outputs_visible"] is False
    assert lifecycle["stage18_training_started"] is False
    assert lifecycle["stage18_analysis_started"] is False
    assert decision["second_no_generalisation_seed"] == "not_executable"
    assert decision["additional_random_label_seed_count"] == 0
    assert decision["additional_random_label_seeds"] == []
    assert decision["main_seed_completion_priority"] is True
    assert decision["stage15_status"] == "unavailable"
    assert decision["replacement_control"] is None


def test_decision_uses_only_frozen_resource_evidence() -> None:
    manifest = _manifest()
    basis = manifest["decision_basis"]
    resources = manifest["resource_basis"]

    assert basis["committed_compute_projection_used"] is True
    assert basis["available_disk_capacity_used"] is True
    assert basis["production_concurrency_used"] is True
    assert basis["expected_main_seed_workload_used"] is True
    assert basis["project_schedule_used"] is True
    assert basis["main_seed_circuit_outcomes_used"] is False
    assert basis["anticipated_control_direction_used"] is False
    assert basis["publication_attractiveness_used"] is False
    assert resources["production_worker_count"] == 12
    assert resources["threads_per_worker"] == 1
    assert resources["projected_definitive_plus_reproduction_bytes"] == 459775077926


def test_required_main_seed_scope_is_unchanged() -> None:
    manifest = _manifest()

    assert manifest["required_main_seeds"] == [0, 1, 2, 3, 4]
    assert manifest["control_priority"] == [
        "second_no_generalisation_seed",
        "second_random_label_seed",
        "further_control_seeds",
    ]


def test_decision_sources_and_note_hash_exactly() -> None:
    manifest = _manifest()

    for record in manifest["source_document_hashes"].values():
        assert _sha256(ROOT / record["path"]) == record["post_decision_sha256"]
    note = manifest["outputs"]["note"]
    assert _sha256(ROOT / note["path"]) == note["sha256"]
    freeze = manifest["resource_basis"]["freeze_manifest"]
    projection = manifest["resource_basis"]["compute_projection"]
    assert _sha256(ROOT / freeze["path"]) == freeze["sha256"]
    assert _sha256(ROOT / projection["path"]) == projection["sha256"]


def test_pending_protocol_entry_is_resolved() -> None:
    protocol = (ROOT / "experimental_protocol.md").read_text(encoding="utf-8")
    order = (ROOT / "implementation_order.md").read_text(encoding="utf-8")
    pending = protocol.split("# **Pending protocol entries**", maxsplit=1)[1]

    assert "Additional control-seed count" not in pending
    assert "Post-Stage-17 additional-control seed-count freeze record" in protocol
    assert "zero additional random-label seeds" in order
