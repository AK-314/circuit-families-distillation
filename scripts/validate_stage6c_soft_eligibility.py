#!/usr/bin/env python3
"""Validate Stage 6C soft eligibility using in-memory technical fixtures only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_ROOT = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_ROOT / "bin/python"
SRC_ROOT = REPO_ROOT / "src"


def _ensure_repository_runtime() -> None:
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_parts = (
        [] if not existing_pythonpath else existing_pythonpath.split(os.pathsep)
    )
    if str(SRC_ROOT) not in pythonpath_parts:
        pythonpath_parts.insert(0, str(SRC_ROOT))
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    if Path(sys.prefix).resolve() == VENV_ROOT.resolve():
        return
    if not VENV_PYTHON.is_file():
        return
    if os.environ.get("STAGE6C_REEXECUTED") == "1":
        raise RuntimeError("Stage 6C CLI could not enter repository virtualenv")

    environment = dict(os.environ)
    environment["STAGE6C_REEXECUTED"] = "1"
    os.execve(
        VENV_PYTHON,
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


_ensure_repository_runtime()
sys.dont_write_bytecode = True
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from circuit_families.stage4_condition_identity import (  # noqa: E402
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.attempt_records import (  # noqa: E402
    emit_technical_attempt_record,
)
from circuit_families.stage5bc.student_identity import (  # noqa: E402
    build_student_attempt_identity,
)
from circuit_families.stage5bc.target_cache import (  # noqa: E402
    FULL_DOMAIN_EXAMPLE_COUNT,
)
from circuit_families.stage6c import (  # noqa: E402
    CENTRING_REF,
    TECHNICAL_POLICY_STATUS,
    TECHNICAL_SOFT_DISCREPANCY_METRIC,
    TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
    CanonicalSoftOutput,
    SoftAttemptLedger,
    SoftRepresentationMetadata,
    TechnicalArgmaxRequirementMetadata,
    TechnicalSoftPolicy,
    TechnicalToleranceMetadata,
    assess_soft_attempt,
    evaluate_soft_target_eligibility,
    generate_soft_sealing_evidence,
    soft_circuit_release_gate,
)

REGISTRY_PATH = REPO_ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
ORDERING_REF = "technical-stage6c-full-domain-order/v1"
ORDERING_SHA256 = "a" * 64


class Stage6CValidationError(RuntimeError):
    """Raised when a technical validation invariant does not hold."""


def _artifact(*, path: str, digest: str, storage_class: str) -> dict[str, str]:
    return {
        "path": path,
        "sha256": digest,
        "storage_class": storage_class,
    }


def _condition_id(
    stage3: Stage3AvailabilityIndex,
    *,
    initialization: int | None = None,
) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="soft_target",
            student_initialization=initialization,
        ),
        stage3,
    )


def _technical_policy(stage3: Stage3AvailabilityIndex) -> TechnicalSoftPolicy:
    return TechnicalSoftPolicy(
        schema_version=TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
        policy_ref="technical-stage6c-cli-soft-candidate/v1",
        status=TECHNICAL_POLICY_STATUS,
        scientific_data=False,
        production_eligible=False,
        resolves_ud006=False,
        representation=SoftRepresentationMetadata(
            representation_ref="technical-stage6c-cli-centred-logits/v1",
            cache_kind="teacher_logits",
            centering_ref=CENTRING_REF,
            teacher_condition_id=_condition_id(stage3),
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            temperature_candidate=None,
            normalization_candidate_ref=None,
        ),
        tolerance=TechnicalToleranceMetadata(
            metric_ref=TECHNICAL_SOFT_DISCREPANCY_METRIC,
            comparison="less_than_or_equal",
            candidate_value=0.0,
            status=TECHNICAL_POLICY_STATUS,
        ),
        argmax_requirement=TechnicalArgmaxRequirementMetadata(
            requirement_ref="technical-stage6c-cli-argmax-candidate/v1",
            candidate_required=True,
            status=TECHNICAL_POLICY_STATUS,
        ),
    )


def _attempt(
    stage3: Stage3AvailabilityIndex,
    *,
    retry_index: int,
) -> dict[str, object]:
    identity = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="soft_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=retry_index,
    )
    record = emit_technical_attempt_record(
        stage3=stage3,
        attempt_identity=identity,
        target_cache_reference={
            "record_type": "teacher_output_cache",
            "schema_version": "teacher_output_cache/v1",
            "condition_id": _condition_id(stage3),
            "record_sha256": "c" * 64,
        },
        outcome_kind="succeeded",
        student_architecture_ref="synthetic-student-architecture/v1",
        replication_policy_ref="synthetic-replication-policy/v1",
        training_config_ref="synthetic-training-config/v1",
        training_log=_artifact(
            path=f"synthetic/stage6c/cli-attempt-r{retry_index}.log",
            digest="d" * 64,
            storage_class="external_log",
        ),
        model_checkpoint=_artifact(
            path=f"synthetic/stage6c/cli-student-r{retry_index}.pt",
            digest="b" * 64,
            storage_class="external_checkpoint",
        ),
    )
    record["record_status"] = "sealed"
    return record


def _evaluate(
    *,
    stage3: Stage3AvailabilityIndex,
    policy: TechnicalSoftPolicy,
    attempt: dict[str, object],
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
):
    return evaluate_soft_target_eligibility(
        teacher=CanonicalSoftOutput(
            role="soft_target_teacher",
            condition_id=_condition_id(stage3),
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            logits=teacher_logits,
            record_status="sealed",
        ),
        student=CanonicalSoftOutput(
            role="soft_target_student",
            condition_id=attempt["condition_id"],
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            logits=student_logits,
            record_status="sealed",
        ),
        policy=policy,
        stage3=stage3,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage6CValidationError(message)


def _run_validation() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    stage3 = Stage3AvailabilityIndex.from_registry(registry)
    policy = _technical_policy(stage3)
    teacher_logits = torch.tensor([1.0, -1.0], dtype=torch.float64).repeat(
        FULL_DOMAIN_EXAMPLE_COUNT,
        1,
    )
    negative_logits = teacher_logits.clone()
    negative_logits[-1] = torch.tensor([-1.0, 1.0], dtype=torch.float64)

    positive_attempt = _attempt(stage3, retry_index=0)
    negative_attempt = _attempt(stage3, retry_index=1)
    positive_evaluation = _evaluate(
        stage3=stage3,
        policy=policy,
        attempt=positive_attempt,
        teacher_logits=teacher_logits,
        student_logits=teacher_logits,
    )
    negative_evaluation = _evaluate(
        stage3=stage3,
        policy=policy,
        attempt=negative_attempt,
        teacher_logits=teacher_logits,
        student_logits=negative_logits,
    )
    positive = assess_soft_attempt(
        attempt_record=positive_attempt,
        stage3=stage3,
        evaluation=positive_evaluation,
    )
    negative = assess_soft_attempt(
        attempt_record=negative_attempt,
        stage3=stage3,
        evaluation=negative_evaluation,
    )

    ledger = SoftAttemptLedger()
    ledger.add(positive)
    ledger.add(negative)
    sealing = generate_soft_sealing_evidence(
        assessment=positive,
        stage3=stage3,
        checkpoint=positive.attempt_record["payload"]["model_checkpoint"],
        dense_output=_artifact(
            path="synthetic/stage6c/cli-centred-soft-output.bin",
            digest=positive_evaluation.student_soft_output_sha256,
            storage_class="external_large_object",
        ),
        architecture_ref="synthetic-student-architecture/v1",
    )
    positive_gate = soft_circuit_release_gate(
        assessment=positive,
        sealing=sealing,
    )
    negative_gate = soft_circuit_release_gate(
        assessment=negative,
        sealing=None,
    )

    _require(positive_evaluation.total_count == 12_769, "positive domain count")
    _require(positive_evaluation.argmax_agreement_count == 12_769, "positive argmax")
    _require(positive_evaluation.eligible, "positive eligibility")
    _require(positive_gate.allowed, "positive release gate")
    _require(negative_evaluation.total_count == 12_769, "negative domain count")
    _require(negative_evaluation.argmax_agreement_count == 12_768, "negative argmax")
    _require(not negative_evaluation.tolerance_passed, "negative tolerance")
    _require(not negative_evaluation.argmax_rule_passed, "negative argmax rule")
    _require(
        negative.failure_kinds
        == ("tolerance_failure", "argmax_rule_failure"),
        "negative failure taxonomy",
    )
    _require(not negative_gate.allowed, "negative release gate")
    _require(ledger.attempt_count == 2, "attempt accounting")
    _require(ledger.failure_count("tolerance_failure") == 1, "tolerance retention")
    _require(ledger.failure_count("argmax_rule_failure") == 1, "argmax retention")

    print(
        "POSITIVE "
        "domain_count=12769/12769 "
        f"argmax_agreement={positive_evaluation.argmax_agreement_count}/12769 "
        f"discrepancy={positive_evaluation.discrepancy:.17g} "
        f"tolerance={positive_evaluation.tolerance:.17g} "
        f"teacher_soft_hash={positive_evaluation.teacher_soft_output_sha256} "
        f"student_soft_hash={positive_evaluation.student_soft_output_sha256} "
        "eligibility_status=ELIGIBLE failure_kinds=NONE "
        "sealing_status=SEALED_HASH_CONSISTENT downstream_gate=ALLOWED"
    )
    print(
        "NEGATIVE "
        "domain_count=12769/12769 "
        f"argmax_agreement={negative_evaluation.argmax_agreement_count}/12769 "
        f"discrepancy={negative_evaluation.discrepancy:.17g} "
        f"tolerance={negative_evaluation.tolerance:.17g} "
        f"teacher_soft_hash={negative_evaluation.teacher_soft_output_sha256} "
        f"student_soft_hash={negative_evaluation.student_soft_output_sha256} "
        "eligibility_status=INELIGIBLE "
        "failure_kinds=tolerance_failure,argmax_rule_failure "
        "sealing_status=UNSEALED downstream_gate=BLOCKED"
    )
    print(
        "ACCOUNTING attempts=2 tolerance_failure=1 argmax_rule_failure=1 "
        "negative_attempt_retained=YES"
    )
    print(
        "FIXTURE_SCOPE=SYNTHETIC_TECHNICAL_IN_MEMORY VALIDATION_WRITES=NO "
        "REAL_TRAINING=NO REAL_TEACHER_CACHE=NO SCIENTIFIC_DATA=NO"
    )


def main() -> int:
    try:
        _run_validation()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"STAGE6C_VALIDATION=FAIL reason={exc}", file=sys.stderr)
        return 2
    print("STAGE6C_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
