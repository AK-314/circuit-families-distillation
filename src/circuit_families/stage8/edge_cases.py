"""Executable Stage 8 matrix for prescribed scientific failure states.

The matrix uses synthetic technical inputs and accepted Stage 5--7 public
interfaces.  It produces no scientific endpoint data and freezes no decision.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.job_dag import TechnicalJobNode
from circuit_families.stage5bc.job_status import (
    JobStatusReport,
    decide_attempt_resume,
)
from circuit_families.stage5bc.serial_merge import (
    SerialMergeError,
    entry_from_job_status,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
    build_student_condition_id,
)
from circuit_families.stage5d.cells import build_student_cell_summaries
from circuit_families.stage5d.profiles import TechnicalAnalysisProfile
from circuit_families.stage6a import (
    COMPONENT_COUNT,
    ExactLedgerBuilder,
    TerminationStatus,
    reduce_endpoint1,
)
from circuit_families.stage6b import (
    CanonicalDecisionVector,
    evaluate_hard_target_eligibility,
)
from circuit_families.stage6c import (
    CENTRING_REF,
    TECHNICAL_POLICY_STATUS,
    TECHNICAL_SOFT_DISCREPANCY_METRIC,
    TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
    CanonicalSoftOutput,
    SoftRepresentationMetadata,
    TechnicalArgmaxRequirementMetadata,
    TechnicalSoftPolicy,
    TechnicalToleranceMetadata,
    evaluate_soft_target_eligibility,
)
from circuit_families.stage6d import (
    DiscoveryRequest,
    GreedyDeletionAdapter,
    deterministic_seed_evidence,
)
from circuit_families.stage6e import (
    ExactCandidateEvidence,
    load_technical_policy,
    qualify_and_deduplicate,
    recompute_endpoint2,
)

EDGE_CASE_SCHEMA_VERSION = "stage8-technical-edge-case-matrix/v1"
RESULT_SCHEMA_VERSION = "stage8-technical-edge-case-results/v1"
FULL_DOMAIN_COUNT = 12_769
ORDERING_REF = "lexicographic-modular-addition-inputs/v1"
ORDERING_SHA256 = "a" * 64


class EdgeCaseMatrixError(ValueError):
    """Raised when the Stage 8 matrix or an observed outcome is invalid."""


@dataclass(frozen=True)
class _ObservedCase:
    case_id: str
    observed: str
    diagnostics: Mapping[str, Any]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _load_matrix(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "classification",
        "scientific_data",
        "production_eligible",
        "resolves_decisions",
        "cases",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise EdgeCaseMatrixError("Stage 8 matrix keys mismatch")
    if raw["schema_version"] != EDGE_CASE_SCHEMA_VERSION:
        raise EdgeCaseMatrixError("unsupported Stage 8 matrix schema")
    if raw["classification"] != "synthetic_technical_only":
        raise EdgeCaseMatrixError("Stage 8 matrix must be technical-only")
    if raw["scientific_data"] is not False:
        raise EdgeCaseMatrixError("Stage 8 matrix cannot contain scientific data")
    if raw["production_eligible"] is not False:
        raise EdgeCaseMatrixError("Stage 8 matrix cannot be production eligible")
    if raw["resolves_decisions"] != []:
        raise EdgeCaseMatrixError("Stage 8 matrix cannot resolve decisions")
    cases = raw["cases"]
    if not isinstance(cases, list) or not cases:
        raise EdgeCaseMatrixError("Stage 8 cases must be a non-empty list")
    identities: list[str] = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "expected"}:
            raise EdgeCaseMatrixError("Stage 8 case keys mismatch")
        if not all(isinstance(case[key], str) and case[key] for key in case):
            raise EdgeCaseMatrixError("Stage 8 case values must be non-empty strings")
        identities.append(case["case_id"])
    if len(set(identities)) != len(identities):
        raise EdgeCaseMatrixError("duplicate Stage 8 case identity")
    return raw


def _stage3(repository_root: Path) -> Stage3AvailabilityIndex:
    record = json.loads(
        (
            repository_root
            / "followup/manifests/stage3_teacher_registry_v1.json"
        ).read_text(encoding="utf-8")
    )
    return Stage3AvailabilityIndex.from_registry(record)


def _teacher_condition(stage3: Stage3AvailabilityIndex, condition: str) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=0,
            phase="stable post-grokking",
            distillation_condition=condition,
        ),
        stage3,
    )


def _soft_policy(stage3: Stage3AvailabilityIndex) -> TechnicalSoftPolicy:
    teacher_condition = _teacher_condition(stage3, "soft_target")
    return TechnicalSoftPolicy(
        schema_version=TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
        policy_ref="technical-stage8-soft-boundary/v1",
        status=TECHNICAL_POLICY_STATUS,
        scientific_data=False,
        production_eligible=False,
        resolves_ud006=False,
        representation=SoftRepresentationMetadata(
            representation_ref="centred-logits-stage8/v1",
            cache_kind="teacher_logits",
            centering_ref=CENTRING_REF,
            teacher_condition_id=teacher_condition,
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            temperature_candidate=None,
            normalization_candidate_ref=None,
        ),
        tolerance=TechnicalToleranceMetadata(
            metric_ref=TECHNICAL_SOFT_DISCREPANCY_METRIC,
            comparison="less_than_or_equal",
            candidate_value=0.25,
            status=TECHNICAL_POLICY_STATUS,
        ),
        argmax_requirement=TechnicalArgmaxRequirementMetadata(
            requirement_ref="technical-exact-argmax/v1",
            candidate_required=True,
            status=TECHNICAL_POLICY_STATUS,
        ),
    )


def _soft_evidence(
    stage3: Stage3AvailabilityIndex,
    *,
    displacement: float,
    gauge_shift: float = 0.0,
):
    teacher_condition = _teacher_condition(stage3, "soft_target")
    student_condition = build_student_condition_id(
        stage3=stage3,
        teacher_seed=0,
        phase="stable post-grokking",
        distillation_condition="soft_target",
        student_initialization=0,
    )
    teacher_logits = torch.zeros((FULL_DOMAIN_COUNT, 2), dtype=torch.float64)
    student_logits = torch.empty_like(teacher_logits)
    student_logits[:, 0] = displacement + gauge_shift
    student_logits[:, 1] = -displacement + gauge_shift
    return evaluate_soft_target_eligibility(
        teacher=CanonicalSoftOutput(
            role="soft_target_teacher",
            condition_id=teacher_condition,
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            logits=teacher_logits,
            record_status="sealed",
        ),
        student=CanonicalSoftOutput(
            role="soft_target_student",
            condition_id=student_condition,
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            logits=student_logits,
            record_status="sealed",
        ),
        policy=_soft_policy(stage3),
        stage3=stage3,
    )


def _discovery_request(*, exact_allowance: int) -> DiscoveryRequest:
    method = "greedy_deletion"
    version = "inherited-technical-adapter/v1"
    reference = "technical-stage8-greedy/v1"
    return DiscoveryRequest(
        run_id="stage8-technical-edge-case",
        method_name=method,
        method_version=version,
        configuration_reference=reference,
        seed_evidence=deterministic_seed_evidence(
            method_name=method,
            method_version=version,
            configuration_reference=reference,
            seed_value=8,
        ),
        native_budget_unit="ranked_component_proposals",
        native_budget_allowance=4,
        exact_evaluation_allowance=exact_allowance,
        maximum_restarts=0,
        synthetic_fixture=True,
        production_eligible=False,
    )


def _analysis_profile(repository_root: Path) -> TechnicalAnalysisProfile:
    record = json.loads(
        (
            repository_root
            / "followup/configs/stage5d/technical_analysis_profiles_v1.json"
        ).read_text(encoding="utf-8")
    )
    return TechnicalAnalysisProfile.from_mapping(record["profiles"][0])


def _analysis_identity(*, initialization: int | None = None) -> dict[str, Any]:
    return {
        "teacher_seed": 0,
        "phase": "phase_early",
        "distillation_condition": "hard_target",
        "student_initialization": initialization,
        "subject_kind": "student",
        "method_id": "stage8-method",
        "endpoint_id": "endpoint_1",
        "protocol_id": "stage8-technical",
        "fidelity_id": "stage8-fidelity",
        "budget_id": "stage8-budget",
    }


def _analysis_fixture() -> dict[str, Any]:
    cell_identity = _analysis_identity()
    member_identity = _analysis_identity(initialization=0)
    return {
        "cell_expectations": [
            {
                "identity": cell_identity,
                "state": "missing",
                "reason": "synthetic_missing_student_cell",
            },
            {
                "identity": {**cell_identity, "phase": "phase_late"},
                "state": "expected",
                "reason": None,
            },
        ],
        "eligibility_records": [
            {"attempt_id": "attempt-one", "status": "eligible"}
        ],
        "student_endpoints": [
            {
                "identity": {**member_identity, "phase": "phase_late"},
                "state": "defined",
                "value": 0.5,
                "attempt_id": "attempt-one",
            }
        ],
    }


def _observe_cases(repository_root: Path) -> tuple[_ObservedCase, ...]:
    stage3 = _stage3(repository_root)
    cases: list[_ObservedCase] = []

    teacher_condition = _teacher_condition(stage3, "direct_teacher")
    hard_student_condition = build_student_condition_id(
        stage3=stage3,
        teacher_seed=0,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
    )
    teacher_decisions = (0,) * FULL_DOMAIN_COUNT
    student_decisions = (*teacher_decisions[:-1], 1)
    hard = evaluate_hard_target_eligibility(
        teacher=CanonicalDecisionVector(
            role="direct_teacher",
            condition_id=teacher_condition,
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            decisions=teacher_decisions,
        ),
        student=CanonicalDecisionVector(
            role="hard_target_student",
            condition_id=hard_student_condition,
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            decisions=student_decisions,
        ),
        stage3=stage3,
    )
    cases.append(
        _ObservedCase(
            "hard_exactly_one_mismatch",
            "ineligible_12768_of_12769",
            {
                "agreement_count": hard.agreement_count,
                "eligible": hard.eligible,
                "total_count": hard.total_count,
            },
        )
    )

    below = _soft_evidence(stage3, displacement=0.49)
    above = _soft_evidence(stage3, displacement=0.51)
    cases.extend(
        (
            _ObservedCase(
                "soft_immediately_below_tolerance",
                "eligible" if below.eligible else "ineligible",
                {
                    "discrepancy": below.discrepancy,
                    "tolerance": below.tolerance,
                },
            ),
            _ObservedCase(
                "soft_immediately_above_tolerance",
                "eligible" if above.eligible else "ineligible",
                {
                    "discrepancy": above.discrepancy,
                    "tolerance": above.tolerance,
                },
            ),
        )
    )

    intact_only = ExactLedgerBuilder(
        evaluator=lambda mask: 1.0 if sum(mask) == COMPONENT_COUNT else 0.0,
        fidelity_threshold=0.9,
    )
    intact_only.add_mask((1,) + (0,) * (COMPONENT_COUNT - 1), proposal_index=0)
    endpoint1 = reduce_endpoint1(
        intact_only.seal(),
        termination=TerminationStatus(
            status="completed",
            procedure_censored=False,
        ),
    )
    cases.append(
        _ObservedCase(
            "no_sparse_qualifying_circuit",
            "endpoint1_equals_1" if endpoint1.retained_proportion == 1.0 else "bad",
            {"retained_proportion": endpoint1.retained_proportion},
        )
    )

    policy = load_technical_policy(
        repository_root
        / "followup/configs/stage6e/technical_endpoint2_policy_v1.json"
    )
    full_mask = (1,) * COMPONENT_COUNT
    from circuit_families.stage6a import canonical_mask_identity

    full_identity = canonical_mask_identity(range(COMPONENT_COUNT))
    evidence = ExactCandidateEvidence(
        model_id="stage8-model",
        discovery_method_id="stage8-method",
        discovery_config_id="stage8-config",
        source_budget_reference=policy.source_budget_reference,
        fidelity_metric_reference=policy.fidelity_metric_reference,
        component_basis_reference=policy.component_basis_reference,
        component_basis_size=policy.component_basis_size,
        mask=full_mask,
        mask_identity=full_identity,
        exact_fidelity=1.0,
        proposal_reference="stage8-proposal",
        exact_evaluation_reference="stage8-exact",
        source_ledger_reference="stage8-ledger",
        source_ledger_hash="b" * 64,
        recomputed_ledger_hash="b" * 64,
    )
    qualification = qualify_and_deduplicate(
        (evidence,),
        policy,
        model_id="stage8-model",
        discovery_method_id="stage8-method",
        discovery_config_id="stage8-config",
    )
    endpoint2 = recompute_endpoint2(qualification, policy)
    cases.append(
        _ObservedCase(
            "no_packing_eligible_circuit",
            "packing_lower_bound_zero" if endpoint2.packing_lower_bound == 0 else "bad",
            {
                "packing_lower_bound": endpoint2.packing_lower_bound,
                "qualified_candidate_count": endpoint2.qualified_candidate_count,
            },
        )
    )

    def fail_source(_request: DiscoveryRequest, _inherited: object):
        raise RuntimeError("forced Stage 8 optimization failure")

    failed = GreedyDeletionAdapter(
        proposal_source=fail_source,
        evaluator=lambda _mask: 1.0,
        fidelity_threshold=0.9,
    ).run(_discovery_request(exact_allowance=3))
    cases.append(
        _ObservedCase(
            "method_optimization_failure",
            "failed_reported" if failed.stopping_status == "failed" else "bad",
            {"stopping_status": failed.stopping_status},
        )
    )

    proposal = (0,) + (1,) * (COMPONENT_COUNT - 1)
    exhausted = GreedyDeletionAdapter(
        proposal_source=lambda _request, _inherited: (proposal,),
        evaluator=lambda _mask: 1.0,
        fidelity_threshold=0.9,
    ).run(_discovery_request(exact_allowance=1))
    cases.append(
        _ObservedCase(
            "exact_evaluation_budget_exhaustion",
            (
                "exact_budget_exhausted"
                if exhausted.stopping_status == "exact_budget_exhausted"
                else "bad"
            ),
            {
                "exact_budget_exhausted": exhausted.exact_budget_exhausted,
                "stopping_status": exhausted.stopping_status,
            },
        )
    )

    duplicate = ExactLedgerBuilder(evaluator=lambda _mask: 1.0, fidelity_threshold=0.9)
    duplicate.add_mask(proposal, proposal_index=0)
    duplicate.add_mask(proposal, proposal_index=1)
    cases.append(
        _ObservedCase(
            "duplicate_proposals",
            (
                "one_unique_exact_evaluation"
                if len(duplicate.proposals) == 2 and len(duplicate.evaluations) == 2
                else "bad"
            ),
            {
                "proposal_count": len(duplicate.proposals),
                "total_exact_evaluations_including_intact": len(duplicate.evaluations),
                "unique_proposal_exact_evaluations": len(duplicate.evaluations) - 1,
            },
        )
    )

    analysis = _analysis_fixture()
    summaries = build_student_cell_summaries(
        analysis,
        _analysis_profile(repository_root),
    )
    by_phase = {summary.key.phase: summary for summary in summaries}
    cases.extend(
        (
            _ObservedCase(
                "missing_student_cell",
                "missing_reported" if by_phase["phase_early"].state == "missing" else "bad",
                {"state": by_phase["phase_early"].state},
            ),
            _ObservedCase(
                "cell_below_minimum_eligible_count",
                (
                    "unresolved_reported"
                    if by_phase["phase_late"].state == "unresolved"
                    else "bad"
                ),
                {
                    "state": by_phase["phase_late"].state,
                    "reason": by_phase["phase_late"].reason,
                },
            ),
        )
    )

    attempt = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=0,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )
    node = TechnicalJobNode(
        node_type="training",
        condition_id=attempt.condition_id,
        dependencies=(),
    )
    running = JobStatusReport(
        job_id=node.job_id,
        node_type=node.node_type,
        condition_id=node.condition_id,
        relative_identity="stage8/running",
        status="running",
        reason="forced_interruption",
        output_root_exists=True,
    )
    resume = decide_attempt_resume(
        node=node,
        status_report=running,
        requested_attempt_identity=attempt,
        checkpoint_attempt_identity=attempt,
    )
    cases.append(
        _ObservedCase(
            "interrupted_and_resumed_worker",
            "resume_same_attempt" if resume.action == "resume_existing" else "bad",
            {"resume_action": resume.action},
        )
    )

    conflicting = JobStatusReport(
        job_id=node.job_id,
        node_type=node.node_type,
        condition_id=node.condition_id,
        relative_identity="stage8/conflicting",
        status="conflicting",
        reason="forced_conflicting_worker_inventory",
        output_root_exists=True,
    )
    conflict_rejected = False
    try:
        entry_from_job_status(conflicting)
    except SerialMergeError:
        conflict_rejected = True
    cases.append(
        _ObservedCase(
            "conflicting_worker_inventory",
            "conflict_rejected" if conflict_rejected else "bad",
            {"rejected": conflict_rejected},
        )
    )

    shifted = _soft_evidence(stage3, displacement=0.49, gauge_shift=17.0)
    gauge_invariant = (
        shifted.eligible == below.eligible
        and abs(shifted.discrepancy - below.discrepancy) <= 1e-12
    )
    cases.append(
        _ObservedCase(
            "additive_logit_gauge_shift",
            "eligibility_invariant" if gauge_invariant else "bad",
            {
                "eligible_equal": shifted.eligible == below.eligible,
                "discrepancy_absolute_difference": abs(
                    shifted.discrepancy - below.discrepancy
                ),
                "hash_equal": (
                    shifted.student_soft_output_sha256
                    == below.student_soft_output_sha256
                ),
            },
        )
    )

    permuted = {
        key: list(reversed(value)) if isinstance(value, list) else value
        for key, value in analysis.items()
    }
    permuted_summaries = build_student_cell_summaries(
        permuted,
        _analysis_profile(repository_root),
    )
    cases.append(
        _ObservedCase(
            "result_order_permutation",
            "summary_invariant" if summaries == permuted_summaries else "bad",
            {"summary_count": len(summaries)},
        )
    )

    return tuple(cases)


def run_edge_case_matrix(
    *,
    repository_root: str | Path,
    matrix_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute all prescribed Stage 8 cases and return canonical evidence."""
    root = Path(repository_root).resolve()
    source = (
        root / "followup/configs/stage8/technical_edge_case_matrix_v1.json"
        if matrix_path is None
        else Path(matrix_path).resolve()
    )
    matrix = _load_matrix(source)
    expected = {case["case_id"]: case["expected"] for case in matrix["cases"]}
    observed_cases = _observe_cases(root)
    observed = {case.case_id: case for case in observed_cases}
    if set(observed) != set(expected):
        raise EdgeCaseMatrixError(
            "Stage 8 observed case set differs from the prospectively declared matrix"
        )

    results: list[dict[str, Any]] = []
    for case in matrix["cases"]:
        observation = observed[case["case_id"]]
        passed = observation.observed == case["expected"]
        results.append(
            {
                "case_id": observation.case_id,
                "expected": case["expected"],
                "observed": observation.observed,
                "status": "pass" if passed else "fail",
                "diagnostics": dict(observation.diagnostics),
            }
        )
    failed = [result["case_id"] for result in results if result["status"] != "pass"]
    record: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "matrix_schema_version": EDGE_CASE_SCHEMA_VERSION,
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "resolves_decisions": [],
        "case_count": len(results),
        "passed_count": len(results) - len(failed),
        "failed_count": len(failed),
        "cases": results,
        "stage8_complete": not failed,
        "stage9_started": False,
    }
    record["record_sha256"] = hashlib.sha256(_canonical_bytes(record)).hexdigest()
    if failed:
        raise EdgeCaseMatrixError(f"Stage 8 cases failed: {failed}")
    return record
