from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "manifests/post_stage17_checkpoint_grid_and_concurrency_freeze.json"
SUMMARY_PATH = ROOT / "results/tables/post_stage17_concurrency_benchmark_summary.csv"
NOTE_PATH = ROOT / "results/notes/post_stage17_checkpoint_grid_and_concurrency_freeze.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_full_grid_is_frozen_prospectively_before_stage18() -> None:
    manifest = _manifest()
    lifecycle = manifest["lifecycle"]
    decision_basis = manifest["decision_basis"]
    grid = manifest["checkpoint_grid"]

    assert lifecycle["checkpoint_grid_decision_made"] is True
    assert lifecycle["stage18_started"] is False
    assert decision_basis["scientific_outcomes_used"] is False
    assert decision_basis["stage17_family_sizes_used"] is False
    assert decision_basis["stage17_transfer_results_used"] is False
    assert grid["selection"] == "full_seven_checkpoint_grid"
    assert grid["checkpoint_steps"] == [200, 3400, 7450, 8150, 8500, 8650, 9050]
    assert grid["selected_prospectively"] is True
    assert grid["applies_uniformly_to_scaled_main_seeds"] is True
    assert grid["sensitivity_cells_per_checkpoint"] == 18
    assert grid["total_projected_main_sensitivity_cells"] == 5 * 7 * 18
    assert grid["remaining_main_sensitivity_cells"] == 5 * 7 * 18 - 18


def test_production_concurrency_preserves_headroom_and_isolation() -> None:
    concurrency = _manifest()["concurrency"]

    assert concurrency["production_worker_count"] == 12
    assert concurrency["production_intra_op_threads_per_worker"] == 1
    assert concurrency["production_inter_op_threads_per_worker"] == 1
    assert concurrency["environment"] == {
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    assert concurrency["compute_only_ceiling_worker_count"] == 14
    assert concurrency["compute_only_ceiling_is_production_default"] is False
    assert concurrency["nominal_unallocated_physical_cores_at_production"] == 6
    assert concurrency["minimum_reserved_headroom_cores"] == 4
    assert concurrency["isolated_output_root_per_worker"] is True
    assert concurrency["shared_writable_raw_directories_allowed"] is False
    assert concurrency["shared_writable_tables_allowed"] is False
    assert concurrency["shared_writable_manifests_allowed"] is False
    assert concurrency["deterministic_merge_serial"] is True
    assert concurrency["archive_creation_serial"] is True
    assert concurrency["final_reporting_serial"] is True


def test_committed_compute_sources_retain_frozen_hashes() -> None:
    calibration = _manifest()["calibration"]

    for key in ("stage12_compute_projection", "stage17_manifest", "stage17_runtime"):
        source = calibration[key]
        assert _sha256(ROOT / source["path"]) == source["sha256"]


def test_authoritative_document_hash_chain_is_exact() -> None:
    manifest = _manifest()
    control_freeze = json.loads(
        (ROOT / "manifests/post_stage17_additional_control_seed_count_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    stage15 = json.loads(
        (ROOT / "manifests/stage15_no_generalisation_unavailable.json").read_text(encoding="utf-8")
    )

    for name, document in manifest["source_document_hashes"].items():
        stage15_document = stage15["source_document_hashes"][name]
        assert document["pre_freeze_sha256"] == stage15_document["post_resolution_sha256"]
        control_document = control_freeze["source_document_hashes"][name]
        assert document["post_freeze_sha256"] == control_document["pre_decision_sha256"]
        assert control_document["post_decision_sha256"] == _sha256(ROOT / document["path"])


def test_benchmark_summary_is_complete_and_supports_the_frozen_layout() -> None:
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 15
    assert all(row["integrity_passed"] == "True" for row in rows)
    extended = [row for row in rows if row["report_id"] == "extended-180s"]
    assert len(extended) == 9

    by_layout = {(int(row["workers"]), int(row["intra_op_threads"])): row for row in extended[:-1]}
    assert float(by_layout[(12, 1)]["aggregate_cycles_per_second"]) == (3.4644848207614247)
    assert float(by_layout[(14, 1)]["aggregate_cycles_per_second"]) == (3.621917887743947)
    assert float(by_layout[(14, 1)]["aggregate_cycles_per_second"]) > float(
        by_layout[(16, 1)]["aggregate_cycles_per_second"]
    )
    assert float(by_layout[(14, 1)]["aggregate_cycles_per_second"]) > float(
        by_layout[(18, 1)]["aggregate_cycles_per_second"]
    )


def test_authoritative_documents_record_the_freeze_and_remove_the_pending_entry() -> None:
    protocol = (ROOT / "experimental_protocol.md").read_text(encoding="utf-8")
    order = (ROOT / "implementation_order.md").read_text(encoding="utf-8")
    note = NOTE_PATH.read_text(encoding="utf-8")

    assert "Post-Stage-17 checkpoint-grid and concurrency freeze record" in protocol
    pending = protocol.split("# **Pending protocol entries**", maxsplit=1)[1]
    assert "Full or reduced scaled checkpoint grid" not in pending
    assert "Post-Stage-17 checkpoint-grid and concurrency freeze" in order
    assert "Stage 18 has not begun" in note


def test_stage18_outputs_require_the_training_lifecycle_transition() -> None:
    training_manifest = ROOT / "manifests/stage18_training.json"
    if training_manifest.is_file():
        payload = json.loads(training_manifest.read_text(encoding="utf-8"))
        assert payload["experiment_stage"] == 18
        assert payload["checkpoint_count"] == 35
        assert payload["stage19_started"] is False
        return
    permitted = {
        ROOT / "results/tables/stage18_main_seed_registry_pre_execution.csv",
        ROOT / "results/tables/stage18_cell_registry_pre_execution.csv",
        ROOT / "results/tables/stage18_worker_shards_pre_execution.csv",
        *{ROOT / f"manifests/stage18_worker_shards/worker_{index:02d}.json" for index in range(12)},
    }
    for directory_name in ("manifests", "results", "figures"):
        for path in (ROOT / directory_name).rglob("*"):
            if path.is_file():
                if "stage18" in path.as_posix().lower():
                    assert path in permitted
