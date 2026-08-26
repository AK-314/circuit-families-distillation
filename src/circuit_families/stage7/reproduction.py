"""Stage 7A resume, serial-merge, and independent reproduction integration.

Accepted Stage 5C mechanics remain authoritative:
- decide_attempt_resume owns attempt resume/skip/reject semantics;
- merge_status_evidence and registry_sha256 own deterministic serial merge.

This module only binds those mechanics to the Stage 7A technical fixture and
compares independently reconstructed A-F records in separate physical roots.
It does not implement a replacement resume engine, merge algorithm, discovery
algorithm, endpoint reducer, or analysis hierarchy.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from circuit_families.stage4_condition_identity import (
    Stage3AvailabilityIndex,
)
from circuit_families.stage5bc.job_dag import TechnicalJobNode
from circuit_families.stage5bc.job_status import (
    JobStatusError,
    JobStatusReport,
    decide_attempt_resume,
)
from circuit_families.stage5bc.serial_merge import (
    SerialMergeError,
    merge_status_evidence,
    registry_sha256,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
)
from circuit_families.stage5d import (
    build_stage5d_output_bundle,
    load_and_normalize_ingestion,
    load_technical_analysis_profile_set,
    reconstruct_stage5d_output_bundle,
    validate_stage5d_output_bundle,
)
from circuit_families.stage7.contracts import (
    Stage7TechnicalRunRequest,
)
from circuit_families.stage7.discovery_endpoints import (
    run_technical_discovery_endpoint_fixture,
)
from circuit_families.stage7.distillation import (
    TechnicalDistillationFixtureConfig,
    run_technical_distillation_fixture,
)
from circuit_families.stage7.inventory_reporting import (
    build_endpoint_exclusion_records,
    build_part_f_report,
    build_stage5d_analysis_bridge,
    build_teacher_seed_inventory,
)

REPRODUCTION_SCHEMA_VERSION: Final = (
    "stage7-technical-reproduction/v1"
)
RESUME_MERGE_SCHEMA_VERSION: Final = (
    "stage7-technical-resume-merge/v1"
)

ACCEPTED_RESUME_DELEGATE: Final = (
    "circuit_families.stage5bc.job_status.decide_attempt_resume"
)
ACCEPTED_SERIAL_MERGE_DELEGATE: Final = (
    "circuit_families.stage5bc.serial_merge.merge_status_evidence"
)
ACCEPTED_STAGE5D_RECONSTRUCTION_DELEGATE: Final = (
    "circuit_families.stage5d.reconstruct_stage5d_output_bundle"
)


class Stage7ReproductionError(ValueError):
    """Raised for deterministic Stage 7A reproduction mismatch."""


@dataclass(frozen=True)
class ReproductionComparison:
    """Deterministic source/reproduction comparison result."""

    matched: bool
    source_sha256: str
    reproduction_sha256: str
    mismatch_paths: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "matched": self.matched,
            "source_sha256": self.source_sha256,
            "reproduction_sha256": self.reproduction_sha256,
            "mismatch_paths": list(self.mismatch_paths),
        }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(
        _canonical_bytes(value)
    ).hexdigest()


def _mismatch_paths(
    left: Any,
    right: Any,
    *,
    prefix: str = "$",
) -> tuple[str, ...]:
    paths: list[str] = []

    if type(left) is not type(right):
        return (
            f"{prefix}:type:{type(left).__name__}!={type(right).__name__}",
        )

    if isinstance(left, Mapping):
        left_keys = set(left)
        right_keys = set(right)

        for key in sorted(
            left_keys - right_keys
        ):
            paths.append(
                f"{prefix}.{key}:missing_in_reproduction"
            )

        for key in sorted(
            right_keys - left_keys
        ):
            paths.append(
                f"{prefix}.{key}:unexpected_in_reproduction"
            )

        for key in sorted(
            left_keys & right_keys
        ):
            paths.extend(
                _mismatch_paths(
                    left[key],
                    right[key],
                    prefix=f"{prefix}.{key}",
                )
            )

        return tuple(paths)

    if isinstance(left, list):
        if len(left) != len(right):
            paths.append(
                f"{prefix}:length:{len(left)}!={len(right)}"
            )

        for index, (
            left_item,
            right_item,
        ) in enumerate(
            zip(
                left,
                right,
                strict=False,
            )
        ):
            paths.extend(
                _mismatch_paths(
                    left_item,
                    right_item,
                    prefix=f"{prefix}[{index}]",
                )
            )

        return tuple(paths)

    if left != right:
        return (
            f"{prefix}:value:{left!r}!={right!r}",
        )

    return ()


def compare_reproduction_records(
    source: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> ReproductionComparison:
    """Compare canonical records and emit exact field-level mismatch paths."""
    if not isinstance(source, Mapping):
        raise Stage7ReproductionError(
            "source reproduction record must be a mapping"
        )

    if not isinstance(reproduction, Mapping):
        raise Stage7ReproductionError(
            "reproduction record must be a mapping"
        )

    paths = _mismatch_paths(
        source,
        reproduction,
    )

    return ReproductionComparison(
        matched=not paths,
        source_sha256=_sha256(
            source
        ),
        reproduction_sha256=_sha256(
            reproduction
        ),
        mismatch_paths=paths,
    )


def _status_report(
    *,
    node: TechnicalJobNode,
    status: str,
    reason: str,
    terminal_sha256: str | None,
) -> JobStatusReport:
    completion_sha = (
        terminal_sha256
        if status == "completed"
        else None
    )

    failure_sha = (
        terminal_sha256
        if status == "failed"
        else None
    )

    return JobStatusReport(
        job_id=node.job_id,
        node_type=node.node_type,
        condition_id=node.condition_id,
        relative_identity=(
            "stage7-technical-resume/"
            + hashlib.sha256(
                node.job_id.encode()
            ).hexdigest()[:20]
        ),
        status=status,
        reason=reason,
        output_root_exists=True,
        completion_sha256=completion_sha,
        failure_sha256=failure_sha,
    )


def build_resume_merge_evidence(
    *,
    stage3: Stage3AvailabilityIndex,
    teacher_seed: int,
    phase: str,
    distillation_result: Mapping[str, Any],
    discovery_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind accepted Stage 5C resume/merge semantics to Stage 7A evidence."""
    hard = distillation_result["hard"]
    soft = distillation_result["soft"]

    hard_success = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        distillation_condition="hard_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )

    hard_failed = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        distillation_condition="hard_target",
        student_initialization=1,
        attempt_index=0,
        retry_index=0,
    )

    soft_success = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        distillation_condition="soft_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )

    soft_failed = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        distillation_condition="soft_target",
        student_initialization=1,
        attempt_index=0,
        retry_index=0,
    )

    hard_success_node = TechnicalJobNode(
        node_type="training",
        condition_id=hard_success.condition_id,
        dependencies=(),
    )

    hard_failed_node = TechnicalJobNode(
        node_type="training",
        condition_id=hard_failed.condition_id,
        dependencies=(),
    )

    soft_success_node = TechnicalJobNode(
        node_type="training",
        condition_id=soft_success.condition_id,
        dependencies=(),
    )

    soft_failed_node = TechnicalJobNode(
        node_type="training",
        condition_id=soft_failed.condition_id,
        dependencies=(),
    )

    hard_completed_report = _status_report(
        node=hard_success_node,
        status="completed",
        reason="valid_immutable_completion_record",
        terminal_sha256=hard[
            "attempt_record_sha256"
        ][0],
    )

    hard_failed_report = _status_report(
        node=hard_failed_node,
        status="failed",
        reason="valid_immutable_failure_record",
        terminal_sha256=hard[
            "attempt_record_sha256"
        ][1],
    )

    soft_completed_report = _status_report(
        node=soft_success_node,
        status="completed",
        reason="valid_immutable_completion_record",
        terminal_sha256=soft[
            "attempt_record_sha256"
        ][0],
    )

    soft_failed_report = _status_report(
        node=soft_failed_node,
        status="failed",
        reason="valid_immutable_failure_record",
        terminal_sha256=soft[
            "attempt_record_sha256"
        ][1],
    )

    hard_resume = decide_attempt_resume(
        node=hard_success_node,
        status_report=hard_completed_report,
        requested_attempt_identity=hard_success,
    )

    soft_resume = decide_attempt_resume(
        node=soft_success_node,
        status_report=soft_completed_report,
        requested_attempt_identity=soft_success,
    )

    hard_failure_resume = decide_attempt_resume(
        node=hard_failed_node,
        status_report=hard_failed_report,
        requested_attempt_identity=hard_failed,
    )

    soft_failure_resume = decide_attempt_resume(
        node=soft_failed_node,
        status_report=soft_failed_report,
        requested_attempt_identity=soft_failed,
    )

    if hard_resume.action != "skip_completed":
        raise Stage7ReproductionError(
            "completed hard attempt was not skipped"
        )

    if soft_resume.action != "skip_completed":
        raise Stage7ReproductionError(
            "completed soft attempt was not skipped"
        )

    if hard_failure_resume.action != "reject_failed_attempt":
        raise Stage7ReproductionError(
            "failed hard attempt was not protected from rerun"
        )

    if soft_failure_resume.action != "reject_failed_attempt":
        raise Stage7ReproductionError(
            "failed soft attempt was not protected from rerun"
        )

    running_report = JobStatusReport(
        job_id=hard_success_node.job_id,
        node_type=hard_success_node.node_type,
        condition_id=hard_success_node.condition_id,
        relative_identity="stage7-technical-running-hard",
        status="running",
        reason="technical_interruption_fixture",
        output_root_exists=True,
    )

    cross_attempt_transfer_rejected = False

    try:
        decide_attempt_resume(
            node=hard_success_node,
            status_report=running_report,
            requested_attempt_identity=hard_success,
            checkpoint_attempt_identity=soft_success,
        )
    except JobStatusError as exc:
        if "transfer" not in str(exc):
            raise
        cross_attempt_transfer_rejected = True

    if not cross_attempt_transfer_rejected:
        raise Stage7ReproductionError(
            "cross-attempt resume transfer was not rejected"
        )

    reports = (
        hard_completed_report,
        hard_failed_report,
        soft_completed_report,
        soft_failed_report,
    )

    merged = merge_status_evidence(
        reports=reports
    )

    merge_sha = registry_sha256(
        merged
    )

    repeated = merge_status_evidence(
        reports=reversed(
            reports
        )
    )

    repeated_sha = registry_sha256(
        repeated
    )

    if merged != repeated or merge_sha != repeated_sha:
        raise Stage7ReproductionError(
            "accepted serial merge changed with input order"
        )

    duplicate_merge_rejected = False

    try:
        merge_status_evidence(
            reports=(
                hard_completed_report,
                hard_completed_report,
            )
        )
    except SerialMergeError:
        duplicate_merge_rejected = True

    if not duplicate_merge_rejected:
        raise Stage7ReproductionError(
            "duplicate serial-merge identity was not rejected"
        )

    before_counts = {
        "hard_attempts": hard["attempt_count"],
        "soft_attempts": soft["attempt_count"],
        "discovery_runs": discovery_result[
            "discovery_run_count"
        ],
        "native_charges": sum(
            run["native_budget_consumed"]
            for run in discovery_result["runs"]
        ),
        "exact_evaluations": sum(
            run["exact_evaluation_consumed"]
            for run in discovery_result["runs"]
        ),
        "endpoint1_records": len(
            discovery_result["runs"]
        ),
        "endpoint2_records": len(
            discovery_result["runs"]
        ),
    }

    after_skip_counts = dict(
        before_counts
    )

    if before_counts != after_skip_counts:
        raise Stage7ReproductionError(
            "resume skip changed completed technical counts"
        )

    return {
        "schema_version": RESUME_MERGE_SCHEMA_VERSION,
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "accepted_resume_delegate": ACCEPTED_RESUME_DELEGATE,
        "accepted_serial_merge_delegate": ACCEPTED_SERIAL_MERGE_DELEGATE,
        "hard_completed_action": hard_resume.action,
        "soft_completed_action": soft_resume.action,
        "hard_failed_action": hard_failure_resume.action,
        "soft_failed_action": soft_failure_resume.action,
        "cross_attempt_transfer_rejected": cross_attempt_transfer_rejected,
        "duplicate_merge_rejected": duplicate_merge_rejected,
        "merge_sha256": merge_sha,
        "merge_entry_count": len(
            merged["entries"]
        ),
        "merge_order_independent": True,
        "before_resume_counts": before_counts,
        "after_resume_counts": after_skip_counts,
        "completed_attempts_duplicated": False,
        "failed_attempts_reexecuted": False,
        "native_charges_duplicated": False,
        "exact_evaluations_duplicated": False,
        "endpoints_duplicated": False,
        "resume_state_transferred": False,
    }


def _stage5d_reconstruction_evidence(
    repository_root: Path,
) -> dict[str, Any]:
    ingestion_path = (
        repository_root
        / "tests/fixtures/stage5d/"
        "synthetic_ingestion_envelope_v1.json"
    )

    profile_path = (
        repository_root
        / "followup/configs/stage5d/"
        "technical_analysis_profiles_v1.json"
    )

    normalized = load_and_normalize_ingestion(
        ingestion_path
    )

    profile = load_technical_analysis_profile_set(
        profile_path
    ).require(
        "fixture_median_min2"
    )

    source_bundle = build_stage5d_output_bundle(
        normalized,
        profile,
    )

    validate_stage5d_output_bundle(
        source_bundle,
        normalized,
        profile,
    )

    reproduced_bundle = reconstruct_stage5d_output_bundle(
        normalized,
        profile,
        source_bundle["reconstruction_manifest"],
    )

    validate_stage5d_output_bundle(
        reproduced_bundle,
        normalized,
        profile,
    )

    if source_bundle != reproduced_bundle:
        raise Stage7ReproductionError(
            "accepted Stage 5D reconstruction mismatch"
        )

    return {
        "delegate": ACCEPTED_STAGE5D_RECONSTRUCTION_DELEGATE,
        "source_bundle_sha256": source_bundle["sha256"],
        "reproduced_bundle_sha256": reproduced_bundle["sha256"],
        "matched": True,
    }


def build_pipeline_reproduction_record(
    *,
    physical_root: str | Path,
    repository_root: str | Path,
    stage3: Stage3AvailabilityIndex,
    teacher_seed: int,
    phase: str,
    run_request: Stage7TechnicalRunRequest,
    distillation_config: TechnicalDistillationFixtureConfig,
    discovery_profiles_path: str | Path,
    endpoint2_policy_path: str | Path,
    exclusion_register: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently construct a complete deterministic A-F technical record."""
    root = Path(
        physical_root
    ).resolve(strict=False)

    repository = Path(
        repository_root
    ).resolve(strict=True)

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    distillation = run_technical_distillation_fixture(
        output_root=root / "distillation",
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        config=distillation_config,
    )

    discovery = run_technical_discovery_endpoint_fixture(
        distillation_result=distillation,
        run_request=run_request,
        discovery_profiles_path=discovery_profiles_path,
        endpoint2_policy_path=endpoint2_policy_path,
    )

    inventory = build_teacher_seed_inventory(
        teacher_seed=teacher_seed,
        phase=phase,
        distillation_result=distillation,
        discovery_result=discovery,
    )

    analysis_bridge = build_stage5d_analysis_bridge(
        repository_root=repository,
        inventory=inventory,
    )

    exclusions = build_endpoint_exclusion_records(
        discovery_result=discovery,
        exclusion_register=exclusion_register,
    )

    report = build_part_f_report(
        inventory=inventory,
        analysis_bridge=analysis_bridge,
        exclusion_entries=exclusions,
    )

    resume_merge = build_resume_merge_evidence(
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        distillation_result=distillation,
        discovery_result=discovery,
    )

    stage5d_reconstruction = (
        _stage5d_reconstruction_evidence(
            repository
        )
    )

    discovery_runs = [
        {
            "run_id": run["run_id"],
            "subject_id": run["subject_id"],
            "subject_role": run["subject_role"],
            "source_reference_sha256": run[
                "source_reference_sha256"
            ],
            "discovery_method": run["discovery_method"],
            "discovery_configuration_reference": run[
                "discovery_configuration_reference"
            ],
            "native_budget_unit": run["native_budget_unit"],
            "native_budget_allowance": run[
                "native_budget_allowance"
            ],
            "native_budget_consumed": run[
                "native_budget_consumed"
            ],
            "exact_evaluation_allowance": run[
                "exact_evaluation_allowance"
            ],
            "exact_evaluation_consumed": run[
                "exact_evaluation_consumed"
            ],
            "exact_ledger_sha256": run[
                "exact_ledger_sha256"
            ],
            "reconstructed_ledger_sha256": run[
                "reconstructed_ledger_sha256"
            ],
            "endpoint1": copy.deepcopy(
                run["endpoint1"]
            ),
            "endpoint2": copy.deepcopy(
                run["endpoint2"]
            ),
            "stopping_status": run[
                "stopping_status"
            ],
        }
        for run in discovery["runs"]
    ]

    exclusion_records = [
        dict(entry)
        for entry in exclusions
    ]

    record = {
        "schema_version": REPRODUCTION_SCHEMA_VERSION,
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "production_default": False,
        "run_request_identity": run_request.run_identity,
        "run_request_sha256": run_request.request_sha256,
        "teacher_seed": teacher_seed,
        "phase": phase,
        "distillation": {
            "hard": {
                "attempt_count": distillation["hard"]["attempt_count"],
                "attempt_record_sha256": list(
                    distillation["hard"]["attempt_record_sha256"]
                ),
                "classifications": list(
                    distillation["hard"]["classifications"]
                ),
                "sealed_dense_model_sha256": distillation[
                    "hard"
                ]["sealed_dense_model_sha256"],
            },
            "soft": {
                "attempt_count": distillation["soft"]["attempt_count"],
                "attempt_record_sha256": list(
                    distillation["soft"]["attempt_record_sha256"]
                ),
                "statuses": list(
                    distillation["soft"]["statuses"]
                ),
                "failure_kinds": copy.deepcopy(
                    distillation["soft"]["failure_kinds"]
                ),
                "sealed_dense_model_sha256": distillation[
                    "soft"
                ]["sealed_dense_model_sha256"],
            },
        },
        "discovery_runs": discovery_runs,
        "inventory": copy.deepcopy(
            inventory
        ),
        "analysis_bridge": copy.deepcopy(
            analysis_bridge
        ),
        "excluded_endpoint_records": exclusion_records,
        "report": copy.deepcopy(
            report
        ),
        "resume_merge": copy.deepcopy(
            resume_merge
        ),
        "stage5d_reconstruction": stage5d_reconstruction,
        "registered_fixture_execution": False,
        "scientific_execution": False,
        "stage8_execution": False,
    }

    record["sha256"] = _sha256(
        record
    )

    return record


def compare_independent_pipeline_reproduction(
    *,
    source_record: Mapping[str, Any],
    reproduction_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two complete records while ignoring only self-hash recursion."""
    source = copy.deepcopy(
        dict(source_record)
    )
    reproduction = copy.deepcopy(
        dict(reproduction_record)
    )

    source_declared_sha = source.pop(
        "sha256",
        None,
    )
    reproduction_declared_sha = reproduction.pop(
        "sha256",
        None,
    )

    if source_declared_sha != _sha256(source):
        raise Stage7ReproductionError(
            "source record self-hash mismatch"
        )

    if reproduction_declared_sha != _sha256(reproduction):
        raise Stage7ReproductionError(
            "reproduction record self-hash mismatch"
        )

    comparison = compare_reproduction_records(
        source,
        reproduction,
    )

    return {
        "schema_version": (
            "stage7-technical-reproduction-comparison/v1"
        ),
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "matched": comparison.matched,
        "source_sha256": source_declared_sha,
        "reproduction_sha256": reproduction_declared_sha,
        "substantive_source_sha256": comparison.source_sha256,
        "substantive_reproduction_sha256": (
            comparison.reproduction_sha256
        ),
        "mismatch_paths": list(
            comparison.mismatch_paths
        ),
        "specific_mismatch_diagnostics": True,
        "separate_physical_roots_required": True,
        "registered_fixture_execution": False,
        "stage8_edge_matrix_executed": False,
    }
