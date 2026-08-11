from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from circuit_families.analysis import stage18_execution
from circuit_families.analysis.stage18_execution import (
    compare_reproduction,
    refresh_archive_index,
    serial_merge_fresh,
    validate_existing_stage18_outputs,
    write_merged_tables,
)
from circuit_families.analysis.stage18_scaling import build_stage18_registry
from circuit_families.training import file_sha256


def _write(root: Path, relative: str, value: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_reproduction_comparison_detects_deterministic_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reproduction = tmp_path / "reproduction"
    relative = "results/tables/stage18_family_summary.csv"
    _write(reference, relative, "cell_id,family_size\na,1\n")
    _write(reproduction, relative, "cell_id,family_size\na,1\n")

    matching = compare_reproduction(reference, reproduction, run_id="stage18-scaling-test")
    assert matching["passed"] is True
    assert matching["deterministic_mismatch_count"] == 0

    _write(reproduction, relative, "cell_id,family_size\na,0\n")
    mismatching = compare_reproduction(reference, reproduction, run_id="stage18-scaling-test")
    assert mismatching["passed"] is False
    assert mismatching["deterministic_mismatch_count"] == 1


def test_reproduction_comparison_normalizes_archive_packaging_only(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reproduction = tmp_path / "reproduction"
    run_id = "stage18-scaling-test"
    scientific = f"results/raw/{run_id}/workers/worker_00/cell/scientific.json"
    _write(reference, scientific, '{"value":1}\n')
    _write(reproduction, scientific, '{"value":1}\n')
    _write(reference, f"results/raw/{run_id}/workers/worker_00/cell/.DS_Store", "primary")
    _write(reproduction, f"results/raw/{run_id}/workers/worker_01/.DS_Store", "reproduction")
    _write(reference, f"results/raw/{run_id}/worker_runtime.json", "1\n")
    _write(reproduction, f"results/raw/{run_id}/worker_runtime.json", "2\n")
    archive = f"results/archives/{run_id}"
    primary_inventory = {
        "model_seed": 0,
        "checkpoint_step": 200,
        "cell_ids": ["cell"],
        "reference_stage17": False,
        "members": [scientific, f"results/raw/{run_id}/workers/worker_00/cell/.DS_Store"],
    }
    reproduction_inventory = {
        **primary_inventory,
        "members": [scientific, f"results/raw/{run_id}/workers/worker_01/.DS_Store"],
    }
    _write(reference, f"{archive}/seed_0_step_200_inventory.json", json.dumps(primary_inventory))
    _write(
        reproduction,
        f"{archive}/seed_0_step_200_inventory.json",
        json.dumps(reproduction_inventory),
    )
    _write(reference, f"{archive}/seed_0_step_200.tar.gz", "primary archive")
    _write(reproduction, f"{archive}/seed_0_step_200.tar.gz", "reproduction archive")
    _write(reference, f"{archive}/index.json", "primary index")
    _write(reproduction, f"{archive}/index.json", "reproduction index")

    comparison = compare_reproduction(reference, reproduction, run_id=run_id)

    assert comparison["passed"] is True
    assert comparison["compared_file_count"] == 1
    assert comparison["archive_inventory_count"] == 1
    assert comparison["archive_inventory_mismatch_count"] == 0


def test_reproduction_comparison_normalizes_ds_store_inventory_chain(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reproduction = tmp_path / "reproduction"
    run_id = "stage18-scaling-test"
    base = f"results/raw/{run_id}/workers/worker_00/cells/cell"
    primary_inventory = {
        "file_count": 2,
        "files": [
            {"path": ".DS_Store", "sha256": "primary", "size_bytes": 10},
            {"path": "scientific.json", "sha256": "same", "size_bytes": 20},
        ],
    }
    reproduction_inventory = {
        "file_count": 1,
        "files": [{"path": "scientific.json", "sha256": "same", "size_bytes": 20}],
    }
    _write(reference, f"{base}/search/hash_inventory.json", json.dumps(primary_inventory))
    _write(
        reproduction,
        f"{base}/search/hash_inventory.json",
        json.dumps(reproduction_inventory),
    )
    _write(
        reference,
        f"{base}/search_result.json",
        json.dumps({"result": 1, "search_integrity": {"hash_inventory_sha256": "primary"}}),
    )
    _write(
        reproduction,
        f"{base}/search_result.json",
        json.dumps({"result": 1, "search_integrity": {"hash_inventory_sha256": "reproduction"}}),
    )

    comparison = compare_reproduction(reference, reproduction, run_id=run_id)

    assert comparison["passed"] is True
    assert comparison["normalized_metadata_file_count"] == 2


def test_reproduction_comparison_uses_declared_stage18_outputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reproduction = tmp_path / "reproduction"
    run_id = "stage18-scaling-test"
    relative = "results/tables/stage18_family_summary.csv"
    manifest = {"outputs": {relative: "manifest digest is not trusted for comparison"}}
    for root in (reference, reproduction):
        _write(root, relative, "value\n1\n")
        _write(root, f"manifests/stage18_scaling_{run_id}.json", json.dumps(manifest))
    _write(
        reference,
        "results/notes/stage18_reproduction_downstream_concurrency_amendment.md",
        "administrative note\n",
    )

    comparison = compare_reproduction(reference, reproduction, run_id=run_id)

    assert comparison["passed"] is True
    assert comparison["compared_file_count"] == 1


def test_reproduction_comparison_detects_extra_file_and_inventory_change(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reproduction = tmp_path / "reproduction"
    run_id = "stage18-scaling-test"
    _write(reference, "results/tables/stage18_family_summary.csv", "value\n1\n")
    _write(reproduction, "results/tables/stage18_family_summary.csv", "value\n1\n")
    _write(reproduction, "results/tables/stage18_unexpected.csv", "value\n2\n")
    archive = f"results/archives/{run_id}/seed_0_step_200_inventory.json"
    base = {
        "model_seed": 0,
        "checkpoint_step": 200,
        "cell_ids": ["cell"],
        "reference_stage17": False,
        "members": ["results/raw/scientific.json"],
    }
    _write(reference, archive, json.dumps(base))
    _write(reproduction, archive, json.dumps({**base, "members": []}))

    comparison = compare_reproduction(reference, reproduction, run_id=run_id)

    assert comparison["passed"] is False
    assert comparison["deterministic_mismatch_count"] == 2
    assert comparison["archive_inventory_mismatch_count"] == 1


def test_archive_member_filter_excludes_runtime_and_ds_store(tmp_path: Path) -> None:
    scientific = tmp_path / "scientific.json"
    runtime = tmp_path / "transfer_runtime.json"
    metadata = tmp_path / ".DS_Store"
    for path in (scientific, runtime, metadata):
        path.write_text("value\n", encoding="utf-8")

    assert stage18_execution._archive_member_allowed(scientific) is True
    assert stage18_execution._archive_member_allowed(runtime) is False
    assert stage18_execution._archive_member_allowed(metadata) is False


def test_serial_merge_accepts_restart_rows_without_legacy_run_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cell = build_stage18_registry()[0]
    transfer = SimpleNamespace(
        profile_rows=(),
        distance_rows=(),
        group_rows=(),
        evaluation_rows=(),
    )
    monkeypatch.setattr(stage18_execution, "_load_search", lambda *_: object())
    monkeypatch.setattr(stage18_execution, "_minimal_transfer", lambda *_: transfer)
    monkeypatch.setattr(
        stage18_execution,
        "_scientific_rows",
        lambda *_: {
            "restarts_table": [{"cell_id": cell.cell_id}],
            "family_summary_table": [{"cell_id": cell.cell_id, "stage17_run_id": "legacy"}],
        },
    )

    merged = serial_merge_fresh(tmp_path, "stage18-scaling-test", (cell,))

    for rows in (rows for rows in merged.values() if rows):
        assert rows[0]["stage18_run_id"] == "stage18-scaling-test"
        assert "stage17_run_id" not in rows[0]


def test_resume_only_validation_requires_complete_search_transfer_and_masking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cell = build_stage18_registry()[0]
    run_id = "stage18-scaling-test"
    search_path = stage18_execution._search_path(tmp_path, run_id, cell)
    cell_root = search_path.parent
    _write(tmp_path, str(search_path.relative_to(tmp_path)), "{}\n")
    _write(tmp_path, str(cell_root.relative_to(tmp_path) / "transfer/summary.json"), "{}\n")
    _write(tmp_path, str(cell_root.relative_to(tmp_path) / "transfer_runtime.json"), "[]\n")
    _write(
        tmp_path,
        "results/tables/stage18_masking_validation.csv",
        "model_seed,checkpoint_step,status\n0,200,passed\n",
    )
    monkeypatch.setattr(stage18_execution, "FRESH_CELL_COUNT", 1)
    monkeypatch.setattr(stage18_execution, "CHECKPOINT_STEPS", (200,))

    masking_path = validate_existing_stage18_outputs(tmp_path, run_id, (cell,), (cell.model_seed,))

    assert masking_path == tmp_path / "results/tables/stage18_masking_validation.csv"


def test_merged_tables_include_deterministic_figure_sources(tmp_path: Path) -> None:
    keys = (
        "frontier_table",
        "family_size_heatmap_source_table",
        "family_size_curves_source_table",
        "family_size_distinctness_source_table",
    )

    outputs = write_merged_tables(tmp_path, {key: ({"value": key},) for key in keys})

    assert {path.name for path in outputs} == {
        "stage18_frontier.csv",
        "stage18_family_size_heatmap_source.csv",
        "stage18_family_size_curves_source.csv",
        "stage18_family_size_distinctness_source.csv",
    }


def test_archive_index_refresh_records_required_sizes(tmp_path: Path) -> None:
    run_id = "stage18-scaling-test"
    archive_root = tmp_path / "results/archives" / run_id
    archive_root.mkdir(parents=True)
    rows = []
    for index in range(35):
        seed, step = divmod(index, 7)
        member = tmp_path / f"results/raw/member_{index}.json"
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text(f"member {index}\n", encoding="utf-8")
        inventory = archive_root / f"seed_{seed}_step_{step}_inventory.json"
        inventory.write_text(
            '{"members":["' + str(member.relative_to(tmp_path)) + '"]}\n',
            encoding="utf-8",
        )
        archive = archive_root / f"seed_{seed}_step_{step}.tar.gz"
        archive.write_bytes(f"archive {index}\n".encode())
        rows.append(
            {
                "model_seed": seed,
                "checkpoint_step": step,
                "path": str(archive.relative_to(tmp_path)),
                "sha256": file_sha256(archive),
            }
        )
    (archive_root / "index.json").write_text(
        json.dumps({"schema_version": 1, "stage18_run_id": run_id, "shards": rows}),
        encoding="utf-8",
    )

    path = refresh_archive_index(tmp_path, run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["shards"]) == 35
    assert all(row["raw_file_count"] == 1 for row in payload["shards"])
    assert all(row["member_count"] == 2 for row in payload["shards"])
    assert all(row["uncompressed_size"] > 0 for row in payload["shards"])
