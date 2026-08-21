from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SYNTHETIC_UNIVERSE_SCHEMA_VERSION = "stage5d_synthetic_universe_v1"
SYNTHETIC_CLASSIFICATION = "synthetic_technical_only"

SUBJECT_KINDS = frozenset({"teacher", "student"})
DISTILLATION_CONDITIONS = frozenset({"hard", "soft"})
ENDPOINT_IDS = frozenset({"endpoint_1", "endpoint_2"})
ENDPOINT_STATES = frozenset(
    {"defined", "failed", "unavailable", "unresolved", "inapplicable"}
)
ATTEMPT_OUTCOMES = frozenset(
    {"completed", "failed", "unavailable", "inapplicable"}
)
ELIGIBILITY_STATES = frozenset(
    {"eligible", "ineligible", "unresolved", "inapplicable"}
)
PHASE_STATES = frozenset({"available", "unavailable"})
CELL_STATES = frozenset(
    {"expected", "unavailable", "unresolved", "inapplicable"}
)


class SyntheticRecordError(ValueError):
    pass


def expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntheticRecordError(f"{label} must be an object")
    return value


def expect_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise SyntheticRecordError(
            f"{label} keys mismatch: "
            f"missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def expect_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SyntheticRecordError(f"{label} must be a non-empty string")
    return value


def expect_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise SyntheticRecordError(
            f"{label} must be a non-negative integer"
        )
    return value


def expect_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise SyntheticRecordError(f"{label} must be a positive integer")
    return value


def optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return expect_str(value, label)


def optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return expect_nonnegative_int(value, label)


@dataclass(frozen=True, slots=True)
class AnalysisIdentity:
    subject_kind: str
    teacher_seed: int
    phase: str
    distillation_condition: str | None
    student_initialization: int | None
    method_id: str
    endpoint_id: str
    protocol_id: str
    fidelity_id: str
    budget_id: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AnalysisIdentity:
        expect_exact_keys(
            raw,
            {
                "subject_kind",
                "teacher_seed",
                "phase",
                "distillation_condition",
                "student_initialization",
                "method_id",
                "endpoint_id",
                "protocol_id",
                "fidelity_id",
                "budget_id",
            },
            "analysis identity",
        )

        subject_kind = expect_str(raw["subject_kind"], "subject_kind")
        if subject_kind not in SUBJECT_KINDS:
            raise SyntheticRecordError(
                f"unsupported subject_kind: {subject_kind}"
            )

        teacher_seed = expect_nonnegative_int(
            raw["teacher_seed"],
            "teacher_seed",
        )
        phase = expect_str(raw["phase"], "phase")
        condition = optional_str(
            raw["distillation_condition"],
            "distillation_condition",
        )
        initialization = optional_nonnegative_int(
            raw["student_initialization"],
            "student_initialization",
        )

        if subject_kind == "teacher":
            if condition is not None or initialization is not None:
                raise SyntheticRecordError(
                    "teacher identities cannot contain student condition "
                    "or initialization"
                )
        else:
            if condition not in DISTILLATION_CONDITIONS:
                raise SyntheticRecordError(
                    "student identities require hard or soft condition"
                )
            if initialization is None:
                raise SyntheticRecordError(
                    "student identities require student_initialization"
                )

        method_id = expect_str(raw["method_id"], "method_id")
        endpoint_id = expect_str(raw["endpoint_id"], "endpoint_id")
        if endpoint_id not in ENDPOINT_IDS:
            raise SyntheticRecordError(
                f"unsupported endpoint_id: {endpoint_id}"
            )

        return cls(
            subject_kind=subject_kind,
            teacher_seed=teacher_seed,
            phase=phase,
            distillation_condition=condition,
            student_initialization=initialization,
            method_id=method_id,
            endpoint_id=endpoint_id,
            protocol_id=expect_str(raw["protocol_id"], "protocol_id"),
            fidelity_id=expect_str(raw["fidelity_id"], "fidelity_id"),
            budget_id=expect_str(raw["budget_id"], "budget_id"),
        )


def validate_endpoint_value(
    endpoint_id: str,
    state: str,
    value: Any,
) -> int | float | None:
    if state not in ENDPOINT_STATES:
        raise SyntheticRecordError(f"unsupported endpoint state: {state}")

    if state != "defined":
        if value is not None:
            raise SyntheticRecordError(
                f"endpoint state {state} requires value=null"
            )
        return None

    if endpoint_id == "endpoint_1":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SyntheticRecordError(
                "endpoint_1 defined value must be numeric"
            )
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise SyntheticRecordError(
                "endpoint_1 must lie in the closed interval [0, 1]"
            )
        return numeric

    if endpoint_id == "endpoint_2":
        if type(value) is not int or value < 0:
            raise SyntheticRecordError(
                "endpoint_2 defined value must be a non-negative integer"
            )
        return value

    raise SyntheticRecordError(f"unsupported endpoint_id: {endpoint_id}")


@dataclass(frozen=True, slots=True)
class DirectTeacherEndpointRecord:
    record_id: str
    identity: AnalysisIdentity
    state: str
    value: int | float | None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> DirectTeacherEndpointRecord:
        expect_exact_keys(
            raw,
            {"record_id", "identity", "state", "value"},
            "direct teacher endpoint",
        )

        identity = AnalysisIdentity.from_mapping(
            expect_mapping(raw["identity"], "identity")
        )
        if identity.subject_kind != "teacher":
            raise SyntheticRecordError(
                "direct teacher endpoint requires teacher identity"
            )

        state = expect_str(raw["state"], "state")
        value = validate_endpoint_value(
            identity.endpoint_id,
            state,
            raw["value"],
        )

        return cls(
            record_id=expect_str(raw["record_id"], "record_id"),
            identity=identity,
            state=state,
            value=value,
        )


@dataclass(frozen=True, slots=True)
class StudentAttemptRecord:
    attempt_id: str
    teacher_seed: int
    phase: str
    distillation_condition: str
    student_initialization: int
    attempt_index: int
    outcome: str
    failure_reason: str | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StudentAttemptRecord:
        expect_exact_keys(
            raw,
            {
                "attempt_id",
                "teacher_seed",
                "phase",
                "distillation_condition",
                "student_initialization",
                "attempt_index",
                "outcome",
                "failure_reason",
            },
            "student attempt",
        )

        condition = expect_str(
            raw["distillation_condition"],
            "distillation_condition",
        )
        if condition not in DISTILLATION_CONDITIONS:
            raise SyntheticRecordError(
                f"unsupported distillation condition: {condition}"
            )

        outcome = expect_str(raw["outcome"], "outcome")
        if outcome not in ATTEMPT_OUTCOMES:
            raise SyntheticRecordError(
                f"unsupported attempt outcome: {outcome}"
            )

        failure_reason = optional_str(
            raw["failure_reason"],
            "failure_reason",
        )

        if outcome == "failed" and failure_reason is None:
            raise SyntheticRecordError(
                "failed attempts require failure_reason"
            )
        if outcome != "failed" and failure_reason is not None:
            raise SyntheticRecordError(
                "non-failed attempts cannot carry failure_reason"
            )

        return cls(
            attempt_id=expect_str(raw["attempt_id"], "attempt_id"),
            teacher_seed=expect_nonnegative_int(
                raw["teacher_seed"],
                "teacher_seed",
            ),
            phase=expect_str(raw["phase"], "phase"),
            distillation_condition=condition,
            student_initialization=expect_nonnegative_int(
                raw["student_initialization"],
                "student_initialization",
            ),
            attempt_index=expect_nonnegative_int(
                raw["attempt_index"],
                "attempt_index",
            ),
            outcome=outcome,
            failure_reason=failure_reason,
        )


@dataclass(frozen=True, slots=True)
class StudentEligibilityRecord:
    eligibility_id: str
    attempt_id: str
    status: str
    reason: str | None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> StudentEligibilityRecord:
        expect_exact_keys(
            raw,
            {"eligibility_id", "attempt_id", "status", "reason"},
            "student eligibility",
        )

        status = expect_str(raw["status"], "status")
        if status not in ELIGIBILITY_STATES:
            raise SyntheticRecordError(
                f"unsupported eligibility status: {status}"
            )

        reason = optional_str(raw["reason"], "reason")
        if status == "eligible" and reason is not None:
            raise SyntheticRecordError(
                "eligible records cannot carry a failure reason"
            )
        if status != "eligible" and reason is None:
            raise SyntheticRecordError(
                f"{status} eligibility records require a reason"
            )

        return cls(
            eligibility_id=expect_str(
                raw["eligibility_id"],
                "eligibility_id",
            ),
            attempt_id=expect_str(raw["attempt_id"], "attempt_id"),
            status=status,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class StudentEndpointRecord:
    record_id: str
    attempt_id: str
    identity: AnalysisIdentity
    state: str
    value: int | float | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StudentEndpointRecord:
        expect_exact_keys(
            raw,
            {"record_id", "attempt_id", "identity", "state", "value"},
            "student endpoint",
        )

        identity = AnalysisIdentity.from_mapping(
            expect_mapping(raw["identity"], "identity")
        )
        if identity.subject_kind != "student":
            raise SyntheticRecordError(
                "student endpoint requires student identity"
            )

        state = expect_str(raw["state"], "state")
        value = validate_endpoint_value(
            identity.endpoint_id,
            state,
            raw["value"],
        )

        return cls(
            record_id=expect_str(raw["record_id"], "record_id"),
            attempt_id=expect_str(raw["attempt_id"], "attempt_id"),
            identity=identity,
            state=state,
            value=value,
        )


@dataclass(frozen=True, slots=True)
class MethodBudgetRecord:
    method_id: str
    budget_id: str
    native_budget: int
    exact_eval_allowance: int
    resource_unit: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MethodBudgetRecord:
        expect_exact_keys(
            raw,
            {
                "method_id",
                "budget_id",
                "native_budget",
                "exact_eval_allowance",
                "resource_unit",
            },
            "method budget",
        )

        return cls(
            method_id=expect_str(raw["method_id"], "method_id"),
            budget_id=expect_str(raw["budget_id"], "budget_id"),
            native_budget=expect_positive_int(
                raw["native_budget"],
                "native_budget",
            ),
            exact_eval_allowance=expect_nonnegative_int(
                raw["exact_eval_allowance"],
                "exact_eval_allowance",
            ),
            resource_unit=expect_str(
                raw["resource_unit"],
                "resource_unit",
            ),
        )


@dataclass(frozen=True, slots=True)
class TeacherSeedInventory:
    teacher_seed: int
    phase_status: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TeacherSeedInventory:
        expect_exact_keys(
            raw,
            {"teacher_seed", "phase_status"},
            "teacher seed inventory",
        )

        status_raw = expect_mapping(raw["phase_status"], "phase_status")
        if not status_raw:
            raise SyntheticRecordError(
                "teacher seed inventory requires at least one phase"
            )

        phase_status: list[tuple[str, str]] = []
        for phase, state_raw in status_raw.items():
            phase_name = expect_str(phase, "phase name")
            state = expect_str(state_raw, f"phase_status[{phase_name}]")
            if state not in PHASE_STATES:
                raise SyntheticRecordError(
                    f"unsupported phase state: {state}"
                )
            phase_status.append((phase_name, state))

        return cls(
            teacher_seed=expect_nonnegative_int(
                raw["teacher_seed"],
                "teacher_seed",
            ),
            phase_status=tuple(sorted(phase_status)),
        )

    def phase_state(self, phase: str) -> str | None:
        for phase_name, state in self.phase_status:
            if phase_name == phase:
                return state
        return None


@dataclass(frozen=True, slots=True)
class StudentCellIdentity:
    teacher_seed: int
    phase: str
    distillation_condition: str
    method_id: str
    endpoint_id: str
    protocol_id: str
    fidelity_id: str
    budget_id: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StudentCellIdentity:
        expect_exact_keys(
            raw,
            {
                "teacher_seed",
                "phase",
                "distillation_condition",
                "method_id",
                "endpoint_id",
                "protocol_id",
                "fidelity_id",
                "budget_id",
            },
            "student cell identity",
        )

        condition = expect_str(
            raw["distillation_condition"],
            "distillation_condition",
        )
        if condition not in DISTILLATION_CONDITIONS:
            raise SyntheticRecordError(
                "student cell condition must be hard or soft"
            )

        endpoint_id = expect_str(raw["endpoint_id"], "endpoint_id")
        if endpoint_id not in ENDPOINT_IDS:
            raise SyntheticRecordError(
                f"unsupported endpoint_id: {endpoint_id}"
            )

        return cls(
            teacher_seed=expect_nonnegative_int(
                raw["teacher_seed"],
                "teacher_seed",
            ),
            phase=expect_str(raw["phase"], "phase"),
            distillation_condition=condition,
            method_id=expect_str(raw["method_id"], "method_id"),
            endpoint_id=endpoint_id,
            protocol_id=expect_str(raw["protocol_id"], "protocol_id"),
            fidelity_id=expect_str(raw["fidelity_id"], "fidelity_id"),
            budget_id=expect_str(raw["budget_id"], "budget_id"),
        )


@dataclass(frozen=True, slots=True)
class SyntheticCellExpectation:
    cell_id: str
    identity: StudentCellIdentity
    state: str
    reason: str | None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> SyntheticCellExpectation:
        expect_exact_keys(
            raw,
            {"cell_id", "identity", "state", "reason"},
            "synthetic cell expectation",
        )

        state = expect_str(raw["state"], "state")
        if state not in CELL_STATES:
            raise SyntheticRecordError(
                f"unsupported cell state: {state}"
            )

        reason = optional_str(raw["reason"], "reason")
        if state == "expected" and reason is not None:
            raise SyntheticRecordError(
                "expected cells cannot carry a reason"
            )
        if state != "expected" and reason is None:
            raise SyntheticRecordError(
                f"{state} cells require an explicit reason"
            )

        return cls(
            cell_id=expect_str(raw["cell_id"], "cell_id"),
            identity=StudentCellIdentity.from_mapping(
                expect_mapping(raw["identity"], "identity")
            ),
            state=state,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class SyntheticUniverse:
    schema_version: str
    classification: str
    method_budgets: tuple[MethodBudgetRecord, ...]
    teacher_inventories: tuple[TeacherSeedInventory, ...]
    student_attempts: tuple[StudentAttemptRecord, ...]
    eligibility_records: tuple[StudentEligibilityRecord, ...]
    direct_teacher_endpoints: tuple[DirectTeacherEndpointRecord, ...]
    student_endpoints: tuple[StudentEndpointRecord, ...]
    cell_expectations: tuple[SyntheticCellExpectation, ...]
