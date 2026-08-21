from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_families.stage5d import (
    OUTPUT_OBJECT_IDS,
    RECONSTRUCTION_MANIFEST_SCHEMA_VERSION,
    Stage5DOutputError,
    build_stage5d_output_bundle,
    canonical_json_bytes,
    load_and_normalize_ingestion,
    load_technical_analysis_profile_set,
    normalized_sha256,
    reconstruct_stage5d_output_bundle,
    technical_profile_sha256,
    validate_stage5d_output_bundle,
    write_stage5d_output_bundle,
)
from circuit_families.stage5d import outputs as outputs_module

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = (
    ROOT
    / "tests/fixtures/stage5d/"
    "synthetic_ingestion_envelope_v1.json"
)
PROFILES = (
    ROOT
    / "followup/configs/stage5d/"
    "technical_analysis_profiles_v1.json"
)
CLI = ROOT / "scripts/validate_stage5d_outputs.py"


@pytest.fixture
def normalized():
    return load_and_normalize_ingestion(ENVELOPE)


@pytest.fixture
def profile():
    return load_technical_analysis_profile_set(PROFILES).require(
        "fixture_median_min2"
    )


@pytest.fixture
def bundle(normalized, profile):
    return build_stage5d_output_bundle(normalized, profile)


def test_output_tables_and_rows_have_deterministic_order(bundle) -> None:
    objects = bundle["output_objects"]

    assert set(objects) == set(OUTPUT_OBJECT_IDS)
    for object_id in OUTPUT_OBJECT_IDS:
        rows = objects[object_id]["rows"]
        assert rows == sorted(rows, key=canonical_json_bytes)


def test_output_and_object_hashes_are_deterministic(
    normalized,
    profile,
    bundle,
) -> None:
    rebuilt = build_stage5d_output_bundle(normalized, profile)

    assert rebuilt == bundle
    assert rebuilt["sha256"] == bundle["sha256"]
    assert {
        key: value["sha256"]
        for key, value in rebuilt["output_objects"].items()
    } == {
        key: value["sha256"]
        for key, value in bundle["output_objects"].items()
    }


def test_reconstruction_manifest_is_complete_and_linked(
    normalized,
    profile,
    bundle,
) -> None:
    manifest = bundle["reconstruction_manifest"]

    assert (
        manifest["schema_version"]
        == RECONSTRUCTION_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["classification"] == "synthetic_technical_only"
    assert manifest["synthetic_only"] is True
    assert manifest["scientific_data"] is False
    assert manifest["production_eligible"] is False
    assert manifest["normalized_input"]["sha256"] == normalized_sha256(
        normalized
    )
    assert manifest["technical_profile"]["sha256"] == (
        technical_profile_sha256(profile)
    )
    assert manifest["technical_profile"]["profile_id"] == profile.profile_id
    assert manifest["reducer_configuration"] == {
        "cell_reducer": profile.settings.cell_reducer,
        "minimum_eligible_students": (
            profile.settings.minimum_eligible_students
        ),
        "phase_pairs": [
            list(pair)
            for pair in profile.settings.phase_pairs
        ],
        "population_reducer": profile.settings.population_reducer,
        "student_member_unit": "student_initialization",
        "population_unit": "teacher_seed",
    }
    assert manifest["unresolved_decision_dependencies"] == [
        "UD-004",
        "UD-011",
        "UD-012",
    ]
    assert manifest["resolved_decisions"] == []

    manifest_objects = {
        row["object_id"]: row
        for row in manifest["output_objects"]
    }
    assert set(manifest_objects) == set(OUTPUT_OBJECT_IDS)
    for object_id, output in bundle["output_objects"].items():
        link = manifest_objects[object_id]
        assert link["object_sha256"] == output["sha256"]
        assert link["source_record_ids"] == output["source_record_ids"]


def test_reconstruction_requires_exact_source_ids_and_hashes(
    normalized,
    profile,
    bundle,
) -> None:
    manifest = bundle["reconstruction_manifest"]
    assert reconstruct_stage5d_output_bundle(
        normalized,
        profile,
        manifest,
    ) == bundle
    validate_stage5d_output_bundle(bundle, normalized, profile)

    altered = copy.deepcopy(manifest)
    altered["output_objects"][0]["source_record_ids"].append(
        "not_a_synthetic_source"
    )
    with pytest.raises(Stage5DOutputError, match="source IDs and hashes"):
        reconstruct_stage5d_output_bundle(normalized, profile, altered)


def test_hard_and_soft_student_tables_are_separate(bundle) -> None:
    hard_rows = bundle["output_objects"]["hard_student_summaries"]["rows"]
    soft_rows = bundle["output_objects"]["soft_student_summaries"]["rows"]

    assert hard_rows and soft_rows
    assert {
        row["key"]["distillation_condition"]
        for row in hard_rows
    } == {"hard"}
    assert {
        row["key"]["distillation_condition"]
        for row in soft_rows
    } == {"soft"}
    assert {
        canonical_json_bytes(row["key"])
        for row in hard_rows
    }.isdisjoint(
        canonical_json_bytes(row["key"])
        for row in soft_rows
    )


def test_population_outputs_preserve_teacher_seed_boundary(bundle) -> None:
    objects = bundle["output_objects"]
    cell_rows = (
        objects["hard_student_summaries"]["rows"]
        + objects["soft_student_summaries"]["rows"]
    )
    realization_count = sum(
        len(row["eligible_initializations"])
        for row in cell_rows
    )

    population_rows = (
        objects["phase_population_summaries"]["rows"]
        + objects["teacher_student_population_summaries"]["rows"]
    )
    assert population_rows
    assert realization_count > max(
        row["number_defined_teacher_seeds"]
        for row in population_rows
    )
    for row in population_rows:
        assert row["population_unit"] == "teacher_seed"
        assert row["number_defined_teacher_seeds"] == len(
            row["contributing_teacher_seeds"]
        )
        assert "number_student_realizations" not in row


def test_unresolved_cells_do_not_carry_numeric_result_values(bundle) -> None:
    rows = bundle["output_objects"]["unresolved_cell_accounting"]["rows"]

    assert rows
    for row in rows:
        assert row["missingness_kind"] in {"unresolved", "insufficient"}
        assert "summary_value" not in row
        assert "range_value" not in row
        assert "mad_value" not in row

    result_fields = {
        "phase_contrasts": ("left_value", "right_value", "delta"),
        "teacher_student_contrasts": (
            "student_summary_value",
            "teacher_value",
            "delta",
        ),
    }
    for object_id, fields in result_fields.items():
        blocked = [
            row
            for row in bundle["output_objects"][object_id]["rows"]
            if row["state"] != "defined"
        ]
        assert blocked
        assert all(
            all(row[field] is None for field in fields)
            for row in blocked
        )


def test_validate_only_cli_writes_nothing_and_is_cwd_independent(
    tmp_path,
) -> None:
    working_directory = tmp_path / "unrelated_cwd"
    working_directory.mkdir()
    before = tuple(working_directory.iterdir())

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--ingestion",
            str(ENVELOPE),
            "--profile-set",
            str(PROFILES),
            "--profile-id",
            "fixture_median_min2",
        ],
        cwd=working_directory,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "STAGE5D_OUTPUT_VALIDATION=PASS" in result.stdout
    assert "OUTPUT_WRITTEN=NO" in result.stdout
    assert tuple(working_directory.iterdir()) == before


def test_validate_only_cli_accepts_external_runtime_without_local_venv(
    tmp_path,
) -> None:
    checkout = tmp_path / "checkout_without_local_venv"
    checkout_src = checkout / "src"
    checkout_scripts = checkout / "scripts"
    checkout_scripts.mkdir(parents=True)
    shutil.copytree(ROOT / "src/circuit_families", checkout_src / "circuit_families")
    checkout_cli = checkout_scripts / CLI.name
    shutil.copy2(CLI, checkout_cli)

    assert not (checkout / ".venv").exists()
    assert Path(sys.prefix).resolve() != (checkout / ".venv").resolve()

    working_directory = tmp_path / "external_runtime_cwd"
    working_directory.mkdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("STAGE5D_REEXECUTED", None)

    result = subprocess.run(
        [
            sys.executable,
            str(checkout_cli),
            "--ingestion",
            str(ENVELOPE),
            "--profile-set",
            str(PROFILES),
            "--profile-id",
            "fixture_median_min2",
        ],
        cwd=working_directory,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "STAGE5D_OUTPUT_VALIDATION=PASS" in result.stdout
    assert "OUTPUT_WRITTEN=NO" in result.stdout
    assert tuple(working_directory.iterdir()) == ()


def test_explicit_temporary_output_writes_atomically(
    tmp_path,
    bundle,
    monkeypatch,
) -> None:
    output_root = tmp_path / "explicit_stage5d_output"
    replacements: list[tuple[Path, Path]] = []
    real_replace = outputs_module.os.replace

    def recording_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.is_file()
        assert source_path.parent == destination_path.parent
        replacements.append((source_path, destination_path))
        real_replace(source, destination)

    monkeypatch.setattr(outputs_module.os, "replace", recording_replace)
    bundle_path, manifest_path = write_stage5d_output_bundle(
        bundle,
        output_root,
    )

    assert len(replacements) == 2
    assert bundle_path.is_file()
    assert manifest_path.is_file()
    assert json.loads(bundle_path.read_text(encoding="utf-8")) == bundle
    assert not list(output_root.glob("*.tmp"))
    assert not list(output_root.glob(".*.tmp"))


def test_output_root_outside_system_temporary_directory_is_rejected(
    bundle,
) -> None:
    with pytest.raises(Stage5DOutputError, match="system temporary"):
        write_stage5d_output_bundle(
            bundle,
            ROOT / "forbidden_stage5d_output",
        )
