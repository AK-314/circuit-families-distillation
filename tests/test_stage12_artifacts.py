"""Tests for deterministic Stage 12 raw cell artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_families.analysis.stage12_artifacts import (
    write_stage12_cell_artifacts,
)
from circuit_families.interpretability.diversity_forced_search import (
    CheckpointFamilySearchExecution,
    run_sequential_family_search,
)
from circuit_families.interpretability.fidelity import (
    MaskEvaluationMetrics,
)
from circuit_families.interpretability.masks import (
    SEARCHABLE_COMPONENT_COUNT,
    ComponentMask,
    component_location,
)
from circuit_families.interpretability.sparse_search import (
    MEANINGFULLY_SPARSE_MAX_COMPONENTS,
    ComponentRanking,
    RankingResult,
)

COMPONENT_INDEX = {
    identifier: index
    for index, identifier in enumerate(
        ComponentMask.all_retained().retained_component_ids
    )
}


def synthetic_metrics(
    mask: ComponentMask,
    *,
    fidelity: float,
) -> MaskEvaluationMetrics:
    agreement_count = (
        100
        if fidelity == 1.0
        else 0
    )

    return MaskEvaluationMetrics(
        primary_fidelity=fidelity,
        prediction_agreement_count=agreement_count,
        full_accuracy=1.0,
        masked_accuracy=fidelity,
        accuracy_change=fidelity - 1.0,
        full_cross_entropy=0.0,
        masked_cross_entropy=1.0 - fidelity,
        cross_entropy_change=1.0 - fidelity,
        mean_kl_divergence=1.0 - fidelity,
        mean_jensen_shannon_divergence=(
            1.0 - fidelity
        ),
        maximum_absolute_logit_difference=(
            1.0 - fidelity
        ),
        retained_attention_head_count=(
            mask.retained_attention_head_count
        ),
        retained_mlp_neuron_count=(
            mask.retained_mlp_neuron_count
        ),
        retained_component_count=(
            mask.retained_component_count
        ),
        retained_component_proportion=(
            mask.retained_component_proportion
        ),
        evaluated_example_count=100,
        evaluation_batch_size=100,
    )


def synthetic_ranking(
    mask: ComponentMask,
) -> RankingResult:
    rankings = tuple(
        ComponentRanking(
            component_identifier=identifier,
            component_index=COMPONENT_INDEX[
                identifier
            ],
            component_class=(
                component_location(
                    identifier
                ).component_class
            ),
            gate_gradient=0.0,
            estimated_removal_damage=float(
                COMPONENT_INDEX[identifier]
            ),
            ranking_position=position,
        )
        for position, identifier in enumerate(
            mask.retained_component_ids,
            start=1,
        )
    )

    return RankingResult(
        mean_pseudo_target_loss=0.0,
        mean_gate_gradients=(
            (0.0,) * SEARCHABLE_COMPONENT_COUNT
        ),
        ranked_components=rankings,
        evaluated_example_count=100,
        ranking_batch_size=100,
        retained_component_count=(
            mask.retained_component_count
        ),
        model_state_sha256_before="state",
        model_state_sha256_after="state",
        hook_counts_before=(),
        hook_counts_after=(),
    )


def boundary_evaluator(
    mask: ComponentMask,
) -> MaskEvaluationMetrics:
    fidelity = (
        1.0
        if mask.retained_component_count
        >= MEANINGFULLY_SPARSE_MAX_COMPONENTS
        else 0.0
    )

    return synthetic_metrics(
        mask,
        fidelity=fidelity,
    )


def synthetic_checkpoint_execution(
) -> CheckpointFamilySearchExecution:
    initial_mask = ComponentMask.all_retained()
    family = run_sequential_family_search(
        base_ranking_function=synthetic_ranking,
        exact_evaluation_function=boundary_evaluator,
        initial_metrics=synthetic_metrics(
            initial_mask,
            fidelity=1.0,
        ),
        fidelity_threshold=0.99,
        distinctness_cutoff=0.5,
        model_seed=1,
        checkpoint_index=7,
        family_target=1,
        max_restarts_per_alternative=1,
        per_requested_circuit_budget=5_000,
        per_cell_budget=5_000,
        reuse_coefficient=0.5,
    )

    assert family.family_size == 1
    assert len(family.restart_outcomes) == 1

    return CheckpointFamilySearchExecution(
        result=family,
        pseudo_target_sha256="a" * 64,
        pseudo_target_count=100,
        ranking_batch_size=100,
        evaluation_batch_size=100,
        full_model_reference_sha256="b" * 64,
        full_model_reference_example_count=100,
        full_model_reference_batch_size=100,
        model_state_sha256_before="state",
        model_state_sha256_after="state",
        hook_counts_before=(),
        hook_counts_after=(),
    )


def directory_snapshot(
    root: Path,
) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes()
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_stage12_cell_artifacts_are_byte_deterministic(
    tmp_path: Path,
) -> None:
    execution = synthetic_checkpoint_execution()
    metadata = {
        "stage12_run_id": "fixture-run",
        "checkpoint_step": 9050,
        "distinctness_cutoff": 0.5,
    }

    first = write_stage12_cell_artifacts(
        tmp_path / "first",
        execution,
        cell_metadata=metadata,
    )
    second = write_stage12_cell_artifacts(
        tmp_path / "second",
        execution,
        cell_metadata=metadata,
    )

    assert directory_snapshot(
        first.output_directory
    ) == directory_snapshot(
        second.output_directory
    )

    summary = json.loads(
        first.cell_summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        summary["family_search"]["family_size"]
        == 1
    )
    assert (
        summary["family_search"][
            "restart_outcome_count"
        ]
        == 1
    )
    assert (
        summary["checkpoint_integrity"][
            "model_state_unchanged"
        ]
        is True
    )
    assert (
        summary["runtime_telemetry"][
            "included_in_deterministic_artifacts"
        ]
        is False
    )

    family_lines = (
        first.family_members_path.read_text(
            encoding="utf-8"
        )
        .splitlines()
    )
    assert len(family_lines) == 1

    member = json.loads(family_lines[0])
    assert member["member_label"] == "C1"
    assert member["retained_component_count"] == 258

    assert (
        first.pairwise_overlaps_path.read_text(
            encoding="utf-8"
        )
        == ""
    )
    assert len(first.restart_summary_paths) == 1
    assert len(first.ranking_log_paths) >= 1
    assert len(first.sparse_search_artifacts) == 1

    inventory = json.loads(
        first.hash_inventory_path.read_text(
            encoding="utf-8"
        )
    )
    assert inventory["file_count"] == len(
        inventory["files"]
    )
    assert all(
        len(record["sha256"]) == 64
        for record in inventory["files"]
    )


def test_stage12_cell_writer_rejects_nonempty_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "stale.txt").write_text(
        "stale",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="must be empty",
    ):
        write_stage12_cell_artifacts(
            output,
            synthetic_checkpoint_execution(),
            cell_metadata={"fixture": True},
        )

    assert (
        output / "stale.txt"
    ).read_text(encoding="utf-8") == "stale"
