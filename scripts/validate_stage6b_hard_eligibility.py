#!/usr/bin/env python3
"""Validate the Stage 6B hard-eligibility boundary using technical fixtures."""

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
    if os.environ.get("STAGE6B_REEXECUTED") == "1":
        raise RuntimeError("Stage 6B CLI could not enter repository virtualenv")

    environment = dict(os.environ)
    environment["STAGE6B_REEXECUTED"] = "1"
    os.execve(
        VENV_PYTHON,
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


_ensure_repository_runtime()
sys.dont_write_bytecode = True
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
from circuit_families.stage6b import (  # noqa: E402
    CanonicalDecisionVector,
    HardAttemptLedger,
    assess_hard_attempt,
    circuit_release_gate,
    evaluate_hard_target_eligibility,
    generate_hard_sealing_evidence,
)

REGISTRY_PATH = REPO_ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
ORDERING_REF = "modular-addition-full-domain-order/v1"
ORDERING_SHA256 = "a" * 64


class Stage6BValidationError(RuntimeError):
    """Raised when a technical validation invariant does not hold."""


def _artifact(*, path: str, digest: str, storage_class: str) -> dict[str, str]:
    return {
        "path": path,
        "sha256": digest * 64,
        "storage_class": storage_class,
    }


def _condition_id(
    stage3: Stage3AvailabilityIndex,
    *,
    condition: str,
    initialization: int | None = None,
) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition=condition,
            student_initialization=initialization,
        ),
        stage3,
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
        distillation_condition="hard_target",
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
            "condition_id": _condition_id(stage3, condition="hard_target"),
            "record_sha256": "c" * 64,
        },
        outcome_kind="succeeded",
        student_architecture_ref="synthetic-student-architecture/v1",
        replication_policy_ref="synthetic-replication-policy/v1",
        training_config_ref="synthetic-training-config/v1",
        training_log=_artifact(
            path=f"synthetic/stage6b/attempt-r{retry_index}.log",
            digest="d",
            storage_class="external_log",
        ),
        model_checkpoint=_artifact(
            path=f"synthetic/stage6b/student-r{retry_index}.pt",
            digest="b",
            storage_class="external_checkpoint",
        ),
    )
    record["record_status"] = "sealed"
    return record


def _evaluate(
    *,
    stage3: Stage3AvailabilityIndex,
    attempt: dict[str, object],
    teacher_decisions: tuple[int, ...],
    student_decisions: tuple[int, ...],
):
    return evaluate_hard_target_eligibility(
        teacher=CanonicalDecisionVector(
            role="direct_teacher",
            condition_id=_condition_id(stage3, condition="direct_teacher"),
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            decisions=teacher_decisions,
        ),
        student=CanonicalDecisionVector(
            role="hard_target_student",
            condition_id=attempt["condition_id"],
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            decisions=student_decisions,
        ),
        stage3=stage3,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage6BValidationError(message)


def _run_validation() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    stage3 = Stage3AvailabilityIndex.from_registry(registry)
    teacher_decisions = tuple(
        index % 113 for index in range(FULL_DOMAIN_EXAMPLE_COUNT)
    )
    mutated = list(teacher_decisions)
    mutated[-1] = (mutated[-1] + 1) % 113

    positive_attempt = _attempt(stage3, retry_index=0)
    negative_attempt = _attempt(stage3, retry_index=1)
    positive_evaluation = _evaluate(
        stage3=stage3,
        attempt=positive_attempt,
        teacher_decisions=teacher_decisions,
        student_decisions=teacher_decisions,
    )
    negative_evaluation = _evaluate(
        stage3=stage3,
        attempt=negative_attempt,
        teacher_decisions=teacher_decisions,
        student_decisions=tuple(mutated),
    )
    positive = assess_hard_attempt(
        attempt_record=positive_attempt,
        stage3=stage3,
        evaluation=positive_evaluation,
    )
    negative = assess_hard_attempt(
        attempt_record=negative_attempt,
        stage3=stage3,
        evaluation=negative_evaluation,
    )

    ledger = HardAttemptLedger()
    ledger.add(positive)
    ledger.add(negative)
    checkpoint = positive.attempt_record["payload"]["model_checkpoint"]
    sealing = generate_hard_sealing_evidence(
        assessment=positive,
        stage3=stage3,
        checkpoint=checkpoint,
        dense_output=_artifact(
            path="synthetic/stage6b/dense-output.bin",
            digest="e",
            storage_class="external_large_object",
        ),
        architecture_ref="synthetic-student-architecture/v1",
    )
    positive_gate = circuit_release_gate(assessment=positive, sealing=sealing)
    negative_gate = circuit_release_gate(assessment=negative, sealing=None)

    _require(positive_evaluation.agreement_count == 12_769, "positive count")
    _require(positive_evaluation.eligible, "positive eligibility")
    _require(positive_gate.allowed, "positive release gate")
    _require(negative_evaluation.agreement_count == 12_768, "negative count")
    _require(not negative_evaluation.eligible, "negative eligibility")
    _require(negative.classification == "subperfect_agreement", "failure kind")
    _require(not negative_gate.allowed, "negative release gate")
    _require(ledger.attempt_count == 2, "attempt accounting")
    _require(ledger.classification_count("subperfect_agreement") == 1, "retention")

    print(
        "POSITIVE "
        f"agreement_count={positive_evaluation.agreement_count} "
        f"total_count={positive_evaluation.total_count} "
        f"teacher_decision_hash={positive_evaluation.teacher_decisions_sha256} "
        f"student_decision_hash={positive_evaluation.student_decisions_sha256} "
        "eligibility_status=ELIGIBLE failure_kind=NONE "
        "sealing_status=SEALED_HASH_CONSISTENT downstream_gate=ALLOWED"
    )
    print(
        "NEGATIVE "
        f"agreement_count={negative_evaluation.agreement_count} "
        f"total_count={negative_evaluation.total_count} "
        f"teacher_decision_hash={negative_evaluation.teacher_decisions_sha256} "
        f"student_decision_hash={negative_evaluation.student_decisions_sha256} "
        "eligibility_status=INELIGIBLE failure_kind=subperfect_agreement "
        "sealing_status=UNSEALED downstream_gate=BLOCKED"
    )
    print(
        "ACCOUNTING attempts=2 subperfect_agreement=1 "
        "negative_attempt_retained=YES"
    )
    print(
        "VALIDATION_WRITES=NO REAL_TRAINING=NO REAL_TEACHER_CACHE=NO "
        "SCIENTIFIC_DATA=NO"
    )


def main() -> int:
    try:
        _run_validation()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"STAGE6B_VALIDATION=FAIL reason={exc}", file=sys.stderr)
        return 2
    print("STAGE6B_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
