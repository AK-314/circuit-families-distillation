"""Tests for Stage 9 single-cell and pilot orchestration helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from circuit_families.config import load_model_config
from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
)
from circuit_families.interpretability.masks import ComponentMask
from circuit_families.interpretability.sparse_search import (
    CheckpointSearchExecution,
    SparseSearchArtifacts,
    greedy_sparse_search,
    run_checkpoint_sparse_search,
)
from circuit_families.models import build_transformer
from circuit_families.training import canonical_state_hash, file_sha256

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_sparse_search.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_sparse_search_script",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None

MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
sys.modules["run_sparse_search"] = MODULE

PILOT_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_stage9_pilot.py"
)
PILOT_SPEC = importlib.util.spec_from_file_location(
    "run_stage9_pilot_script",
    PILOT_SCRIPT_PATH,
)
assert PILOT_SPEC is not None
assert PILOT_SPEC.loader is not None

PILOT_MODULE = importlib.util.module_from_spec(PILOT_SPEC)
sys.modules[PILOT_SPEC.name] = PILOT_MODULE
PILOT_SPEC.loader.exec_module(PILOT_MODULE)


def _context(
    *,
    phase: str,
    step: int,
) -> CheckpointEvaluationContext:
    model = build_transformer(
        load_model_config("configs/model.yaml"),
        seed=0,
        device="cpu",
    )
    inputs = torch.tensor([[0, 0, 113]], dtype=torch.long)
    targets = torch.tensor([0], dtype=torch.long)

    return CheckpointEvaluationContext(
        run_id="fixture-run",
        checkpoint_phase=phase,
        checkpoint_step=step,
        checkpoint_path=Path(f"step_{step:08d}.pt"),
        checkpoint_sha256="a" * 64,
        checkpoint_manifest_path=Path("checkpoints.json"),
        checkpoint_manifest_sha256="b" * 64,
        training_manifest_path=Path("training.json"),
        training_manifest_sha256="c" * 64,
        model_state_sha256=canonical_state_hash(
            model.state_dict()
        ),
        task_config_sha256="d" * 64,
        model_config_sha256="e" * 64,
        training_config_sha256="f" * 64,
        combined_config_sha256="1" * 64,
        dataset_sha256="2" * 64,
        split_sha256="3" * 64,
        dataset_archive_sha256="4" * 64,
        dataset_metadata_sha256="5" * 64,
        example_ordering="fixture",
        model=model,
        inputs=inputs,
        targets=targets,
        device=torch.device("cpu"),
    )


def test_threshold_slugs_are_exact_and_distinct() -> None:
    assert MODULE.threshold_slug(0.80) == "threshold_08000"
    assert MODULE.threshold_slug(0.85) == "threshold_08500"
    assert MODULE.threshold_slug(0.90) == "threshold_09000"
    assert MODULE.threshold_slug(0.95) == "threshold_09500"
    assert MODULE.threshold_slug(0.975) == "threshold_09750"
    assert MODULE.threshold_slug(0.99) == "threshold_09900"

    assert len(
        {
            MODULE.threshold_slug(value)
            for value in (0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
        }
    ) == 6

    with pytest.raises(ValueError, match="four decimal"):
        MODULE.threshold_slug(0.90001)


def test_default_cell_directory_is_deterministic(
    tmp_path: Path,
) -> None:
    path = MODULE.default_cell_directory(
        tmp_path,
        stage9_run_id="stage9-fixture",
        checkpoint_step=9050,
        fidelity_threshold=0.975,
    )

    assert path == (
        tmp_path
        / "results"
        / "raw"
        / "stage9-fixture"
        / "step_00009050"
        / "threshold_09750"
    )


def test_cell_metadata_marks_calibration_eligibility(
    tmp_path: Path,
) -> None:
    post_context = _context(
        phase="stable post-grokking",
        step=9050,
    )
    pre_context = _context(
        phase="pre-grokking",
        step=200,
    )

    post_execution = run_checkpoint_sparse_search(
        post_context,
        fidelity_threshold=0.99,
        ranking_batch_size=1,
        evaluation_batch_size=1,
        exact_evaluation_budget=0,
    )
    pre_execution = run_checkpoint_sparse_search(
        pre_context,
        fidelity_threshold=0.99,
        ranking_batch_size=1,
        evaluation_batch_size=1,
        exact_evaluation_budget=0,
    )

    post = MODULE.build_cell_metadata(
        repository=tmp_path,
        context=post_context,
        stage8_manifest_path=tmp_path / "stage8.json",
        stage8_manifest_sha256="6" * 64,
        implementation_git_commit="7" * 40,
        training_git_commit="8" * 40,
        git_status_at_start="",
        execution=post_execution,
        fidelity_threshold=0.99,
        ranking_batch_size=1,
        evaluation_batch_size=1,
        exact_evaluation_budget=0,
        method_development=False,
    )
    pre = MODULE.build_cell_metadata(
        repository=tmp_path,
        context=pre_context,
        stage8_manifest_path=tmp_path / "stage8.json",
        stage8_manifest_sha256="6" * 64,
        implementation_git_commit="7" * 40,
        training_git_commit="8" * 40,
        git_status_at_start="dirty",
        execution=pre_execution,
        fidelity_threshold=0.99,
        ranking_batch_size=1,
        evaluation_batch_size=1,
        exact_evaluation_budget=0,
        method_development=True,
    )

    assert post["threshold_calibration_eligibility"] == (
        MODULE.CALIBRATION_ELIGIBLE
    )
    assert post["scientific_output_eligible"]
    assert not post["primary_fidelity_threshold_selected"]
    assert post["training_provenance"][
        "training_git_commit"
    ] == "8" * 40
    assert post["config_provenance"][
        "combined_config_sha256"
    ] == post_context.combined_config_sha256
    assert post["dataset_provenance"][
        "dataset_sha256"
    ] == post_context.dataset_sha256
    assert len(post["pseudo_targets"]["sha256"]) == 64
    assert len(post["full_model_reference"]["sha256"]) == 64
    assert post["search_integrity"]["model_state_unchanged"]
    assert post["search_integrity"]["hook_counts_unchanged"]
    assert not post[
        "git_status_recorded_in_deterministic_metadata"
    ]

    assert pre["threshold_calibration_eligibility"] == (
        "excluded_from_primary_threshold_calibration"
    )
    assert not pre["scientific_output_eligible"]
    assert pre["development_label"] == (
        MODULE.METHOD_DEVELOPMENT_LABEL
    )



def test_stage8_manifest_validation_checks_frozen_sources(
    tmp_path: Path,
) -> None:
    checkpoint_manifest = tmp_path / "checkpoints.json"
    checkpoint_manifest.write_text("{}\n", encoding="utf-8")

    stage8_manifest = tmp_path / "stage8.json"
    stage8_record = {
        "validation_status": "passed",
        "source_training_run_id": "fixture-run",
        "component_definitions": {
            "searchable_component_count": 516,
            "attention_heads": {
                "hook_name": "blocks.0.attn.hook_z",
            },
            "mlp_neurons": {
                "hook_name": "blocks.0.mlp.hook_post",
            },
        },
        "source_manifests": {
            "checkpoint_selection": {
                "path": "checkpoints.json",
                "sha256": file_sha256(checkpoint_manifest),
            },
        },
        "selected_checkpoints": [
            {
                "training_step": 9050,
                "checkpoint_sha256": "a" * 64,
                "model_state_sha256": "b" * 64,
            },
        ],
    }
    stage8_manifest.write_text(
        json.dumps(stage8_record),
        encoding="utf-8",
    )

    record, digest = MODULE.validate_stage8_manifest(
        repository=tmp_path,
        stage8_manifest_path=stage8_manifest,
        checkpoint_manifest_path=checkpoint_manifest,
        run_id="fixture-run",
        checkpoint_step=9050,
    )

    assert record == stage8_record
    assert digest == file_sha256(stage8_manifest)

    stage8_record["source_training_run_id"] = "wrong-run"
    stage8_manifest.write_text(
        json.dumps(stage8_record),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source training run"):
        MODULE.validate_stage8_manifest(
            repository=tmp_path,
            stage8_manifest_path=stage8_manifest,
            checkpoint_manifest_path=checkpoint_manifest,
            run_id="fixture-run",
            checkpoint_step=9050,
        )


def test_runtime_telemetry_is_separate_from_scientific_hashes(
    tmp_path: Path,
) -> None:
    result = greedy_sparse_search(
        ranking_function=lambda mask: pytest.fail(
            "ranking should not run with zero budget"
        ),
        exact_evaluation_function=lambda mask: pytest.fail(
            "evaluation should not run with zero budget"
        ),
        initial_metrics=MODULE.run_checkpoint_sparse_search.__globals__[
            "evaluate_component_mask"
        ](
            _context(
                phase="stable post-grokking",
                step=9050,
            ).model,
            torch.tensor([[0, 0, 113]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            ComponentMask.all_retained(),
            batch_size=1,
        ),
        fidelity_threshold=0.99,
        exact_evaluation_budget=0,
    )

    execution = CheckpointSearchExecution(
        result=result,
        pseudo_target_sha256="a" * 64,
        pseudo_target_count=1,
        ranking_batch_size=1,
        evaluation_batch_size=1,
        model_state_sha256_before="b" * 64,
        model_state_sha256_after="b" * 64,
        hook_counts_before=(),
        hook_counts_after=(),
    )
    artifacts = SparseSearchArtifacts(
        output_directory=tmp_path,
        final_mask_path=tmp_path / "final_mask.json",
        final_mask_sha256="1" * 64,
        accepted_mask_paths=(),
        accepted_mask_sha256s=(),
        accepted_removal_trajectory_path=(
            tmp_path / "accepted_removals.jsonl"
        ),
        accepted_removal_trajectory_sha256="2" * 64,
        candidate_evaluation_log_path=(
            tmp_path / "candidate_evaluations.jsonl"
        ),
        candidate_evaluation_log_sha256="3" * 64,
        cell_summary_path=tmp_path / "cell_summary.json",
        cell_summary_sha256="4" * 64,
        hashes_path=tmp_path / "hashes.json",
        hashes_sha256="5" * 64,
    )

    path = MODULE.write_runtime_telemetry(
        tmp_path / "runtime_telemetry.json",
        elapsed_runtime_seconds=1.25,
        execution=execution,
        artifacts=artifacts,
        method_development=True,
        implementation_git_commit="6" * 40,
        git_status_at_start="dirty",
    )

    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["nondeterministic_runtime_telemetry"]
    assert record[
        "excluded_from_deterministic_scientific_hashes"
    ]
    assert record["elapsed_runtime_seconds"] == 1.25
    assert record["scientific_artifact_hashes"][
        "final_mask_sha256"
    ] == "1" * 64
    assert file_sha256(path)



def test_frozen_stage9_execution_plan_is_exact() -> None:
    plan = PILOT_MODULE.frozen_execution_plan()

    assert len(plan) == 18
    assert [
        (
            cell.checkpoint_step,
            cell.fidelity_threshold,
        )
        for cell in plan
    ] == [
        (step, threshold)
        for step, _ in (
            (9050, "stable post-grokking"),
            (200, "pre-grokking"),
            (8150, "50%"),
        )
        for threshold in (
            0.99,
            0.975,
            0.95,
            0.90,
            0.85,
            0.80,
        )
    ]

    assert [
        cell.execution_index
        for cell in plan
    ] == list(range(1, 19))


def test_stage9_run_id_is_deterministic_and_sensitive() -> None:
    configuration = PILOT_MODULE.stage9_configuration_record(
        source_training_run_id="fixture-run-s1-abcdef",
        checkpoint_manifest_sha256="a" * 64,
        stage8_manifest_sha256="b" * 64,
        implementation_git_commit="c" * 40,
        ranking_batch_size=256,
        evaluation_batch_size=256,
        exact_evaluation_budget=10_000,
        device="cpu",
        method_development=False,
    )

    first = PILOT_MODULE.deterministic_stage9_run_id(
        configuration
    )
    second = PILOT_MODULE.deterministic_stage9_run_id(
        configuration
    )

    assert first == second
    assert first.startswith("stage9-sparse-s1-")

    changed = {
        **configuration,
        "exact_evaluation_budget_per_cell": 9_999,
    }

    assert (
        PILOT_MODULE.deterministic_stage9_run_id(changed)
        != first
    )


def test_pilot_output_paths_are_deterministic(
    tmp_path: Path,
) -> None:
    paths = PILOT_MODULE.pilot_output_paths(
        tmp_path,
        stage9_run_id="stage9-sparse-s1-test",
        source_training_run_id=(
            "modular-addition-training-s1-test"
        ),
        method_development=False,
    )

    assert paths.raw_directory == (
        tmp_path
        / "results"
        / "raw"
        / "stage9-sparse-s1-test"
    )
    assert paths.result_table == (
        tmp_path
        / "results"
        / "tables"
        / "seed_1_stage9_sparse_search.csv"
    )
    assert paths.runtime_table == (
        tmp_path
        / "results"
        / "tables"
        / "seed_1_stage9_sparse_search_runtime.csv"
    )


def test_single_cell_command_uses_uv_and_frozen_commit(
    tmp_path: Path,
) -> None:
    cell = PILOT_MODULE.frozen_execution_plan()[0]

    command = PILOT_MODULE.build_single_cell_command(
        repository=tmp_path,
        run_id="fixture-run",
        checkpoint_manifest=tmp_path / "checkpoints.json",
        stage8_manifest=tmp_path / "stage8.json",
        cell=cell,
        output_directory=tmp_path / "cell",
        ranking_batch_size=256,
        evaluation_batch_size=256,
        exact_evaluation_budget=10_000,
        device="cpu",
        implementation_git_commit="d" * 40,
        method_development=False,
    )

    assert command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_sparse_search.py",
    ]
    assert "--expected-implementation-commit" in command
    assert "d" * 40 in command
    assert "--method-development" not in command


def test_result_table_serialization_is_deterministic(
    tmp_path: Path,
) -> None:
    rows = [
        {
            field: ""
            for field in PILOT_MODULE.RESULT_TABLE_FIELDS
        },
        {
            field: ""
            for field in PILOT_MODULE.RESULT_TABLE_FIELDS
        },
    ]
    rows[0]["checkpoint_step"] = 9050
    rows[0]["fidelity_threshold"] = 0.99
    rows[1]["checkpoint_step"] = 200
    rows[1]["fidelity_threshold"] = 0.80

    first = PILOT_MODULE.write_csv(
        tmp_path / "first.csv",
        rows,
        fields=PILOT_MODULE.RESULT_TABLE_FIELDS,
    )
    second = PILOT_MODULE.write_csv(
        tmp_path / "second.csv",
        rows,
        fields=PILOT_MODULE.RESULT_TABLE_FIELDS,
    )

    assert first.read_bytes() == second.read_bytes()
    assert file_sha256(first) == file_sha256(second)
    assert first.read_text(
        encoding="utf-8"
    ).splitlines()[0].split(",") == list(
        PILOT_MODULE.RESULT_TABLE_FIELDS
    )
