"""Generic training trajectory and Stage 3 phase-selection adapter.

The historical Stage 3 selection implementation remains the authority for the
historical rule. This module only translates a generic, sealed teacher training
trajectory into the exact metric records consumed by those frozen functions.

No circuit, distillation, endpoint, discovery, causal, or scientific-result
fields are admitted to the trajectory schema.

Future/expanded phase rules must be injected explicitly with their own rule ID
and version. They cannot silently masquerade as the historical Stage 3 rule.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol

from circuit_families.analysis.phase_detection import (
    TRANSITION_TARGETS,
    find_pre_grokking_checkpoint,
    find_stable_post_sequence,
    select_transition_landmarks,
)
from circuit_families.stage12p1.tasks import (
    canonical_json_bytes,
    canonical_sha256,
)

TRAJECTORY_SCHEMA_VERSION = "stage12p1-teacher-trajectory/v1"
PHASE_SELECTION_SCHEMA_VERSION = "stage12p1-phase-selection/v1"

HISTORICAL_RULE_ID = "historical-stage3-phase-selection"
HISTORICAL_RULE_VERSION = "stage3-frozen/v1"

PHASE_ROLES = (
    "pre",
    "transition",
    "stable_post",
)

_PHASE_LABELS = {
    "pre": "pre-grokking",
    "transition": "50%",
    "stable_post": "stable post-grokking",
}

_POINT_KEYS = frozenset(
    {
        "training_step",
        "train_accuracy",
        "test_accuracy",
        "train_loss",
        "test_loss",
        "checkpoint_path",
        "checkpoint_sha256",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class PhaseAdapterError(ValueError):
    """Raised when trajectory or phase evidence violates the adapter contract."""


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PhaseAdapterError(f"{name} must be lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PhaseAdapterError(f"{name} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise PhaseAdapterError(f"{name} may not contain control characters")
    return value


def _portable_relative_path(value: Any, *, name: str) -> str:
    text = _require_nonempty_string(value, name=name)
    if text.startswith("~") or "\\" in text:
        raise PhaseAdapterError(f"{name} must be a portable relative POSIX path")

    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PhaseAdapterError(f"{name} must be a portable relative POSIX path")

    return path.as_posix()


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PhaseAdapterError(f"{name} must be finite")
    return result


def _accuracy(value: Any, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise PhaseAdapterError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class TrajectoryPoint:
    """One saved-checkpoint training/test metric record."""

    training_step: int
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float
    checkpoint_path: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.training_step, bool)
            or not isinstance(self.training_step, int)
            or self.training_step < 0
        ):
            raise PhaseAdapterError(
                "trajectory training_step must be a non-negative integer"
            )

        object.__setattr__(
            self,
            "train_accuracy",
            _accuracy(self.train_accuracy, name="train_accuracy"),
        )
        object.__setattr__(
            self,
            "test_accuracy",
            _accuracy(self.test_accuracy, name="test_accuracy"),
        )
        object.__setattr__(
            self,
            "train_loss",
            _finite_number(self.train_loss, name="train_loss"),
        )
        object.__setattr__(
            self,
            "test_loss",
            _finite_number(self.test_loss, name="test_loss"),
        )
        object.__setattr__(
            self,
            "checkpoint_path",
            _portable_relative_path(
                self.checkpoint_path,
                name="checkpoint_path",
            ),
        )
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _require_sha256(
                self.checkpoint_sha256,
                name="checkpoint_sha256",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "training_step": self.training_step,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    def source_sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True)
class TeacherTrajectory:
    """Immutable generic metric trajectory for one explicit teacher seed."""

    teacher_seed_id: str
    teacher_seed: int
    task_identity_sha256: str
    teacher_artifact_sha256: str
    points: tuple[TrajectoryPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "teacher_seed_id",
            _require_nonempty_string(
                self.teacher_seed_id,
                name="teacher_seed_id",
            ),
        )

        if (
            isinstance(self.teacher_seed, bool)
            or not isinstance(self.teacher_seed, int)
            or self.teacher_seed < 0
        ):
            raise PhaseAdapterError(
                "teacher_seed must be a non-negative integer"
            )

        object.__setattr__(
            self,
            "task_identity_sha256",
            _require_sha256(
                self.task_identity_sha256,
                name="task_identity_sha256",
            ),
        )
        object.__setattr__(
            self,
            "teacher_artifact_sha256",
            _require_sha256(
                self.teacher_artifact_sha256,
                name="teacher_artifact_sha256",
            ),
        )

        if not isinstance(self.points, tuple) or not self.points:
            raise PhaseAdapterError(
                "teacher trajectory must contain at least one point"
            )

        previous_step: int | None = None
        for point in self.points:
            if not isinstance(point, TrajectoryPoint):
                raise PhaseAdapterError(
                    "trajectory points must be TrajectoryPoint values"
                )
            if previous_step is not None and point.training_step <= previous_step:
                raise PhaseAdapterError(
                    "trajectory steps must be strictly increasing and unique"
                )
            previous_step = point.training_step

    def material_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "classification": "technical_fixture",
            "scientific_data": False,
            "production_eligible": False,
            "teacher_seed_id": self.teacher_seed_id,
            "teacher_seed": self.teacher_seed,
            "task_identity_sha256": self.task_identity_sha256,
            "teacher_artifact_sha256": self.teacher_artifact_sha256,
            "points": [point.to_mapping() for point in self.points],
        }

    def content_sha256(self) -> str:
        return canonical_sha256(self.material_mapping())

    def sealed_mapping(self) -> dict[str, Any]:
        record = self.material_mapping()
        record["content_sha256"] = self.content_sha256()
        return record


def build_teacher_trajectory(
    *,
    teacher_seed_id: str,
    teacher_seed: int,
    task_identity_sha256: str,
    teacher_artifact_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> TeacherTrajectory:
    """Build a sealed trajectory from exact training/test metric records."""
    if not isinstance(records, Sequence) or isinstance(
        records,
        (str, bytes, bytearray),
    ):
        raise PhaseAdapterError("trajectory records must be a sequence")

    points: list[TrajectoryPoint] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PhaseAdapterError(
                f"trajectory record {index} must be a mapping"
            )

        if set(record) != _POINT_KEYS:
            missing = sorted(_POINT_KEYS - set(record))
            extra = sorted(set(record) - _POINT_KEYS)
            raise PhaseAdapterError(
                f"trajectory record {index} keys mismatch: "
                f"missing={missing!r}, extra={extra!r}"
            )

        points.append(
            TrajectoryPoint(
                training_step=record["training_step"],
                train_accuracy=record["train_accuracy"],
                test_accuracy=record["test_accuracy"],
                train_loss=record["train_loss"],
                test_loss=record["test_loss"],
                checkpoint_path=record["checkpoint_path"],
                checkpoint_sha256=record["checkpoint_sha256"],
            )
        )

    return TeacherTrajectory(
        teacher_seed_id=teacher_seed_id,
        teacher_seed=teacher_seed,
        task_identity_sha256=task_identity_sha256,
        teacher_artifact_sha256=teacher_artifact_sha256,
        points=tuple(points),
    )


def validate_teacher_trajectory(
    record: Mapping[str, Any],
) -> TeacherTrajectory:
    """Validate content hash and reconstruct one immutable trajectory."""
    if not isinstance(record, Mapping):
        raise PhaseAdapterError("sealed trajectory must be a mapping")

    required = {
        "schema_version",
        "classification",
        "scientific_data",
        "production_eligible",
        "teacher_seed_id",
        "teacher_seed",
        "task_identity_sha256",
        "teacher_artifact_sha256",
        "points",
        "content_sha256",
    }
    if set(record) != required:
        raise PhaseAdapterError("sealed trajectory keys mismatch")

    if record["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise PhaseAdapterError("trajectory schema_version mismatch")
    if record["classification"] != "technical_fixture":
        raise PhaseAdapterError("trajectory classification mismatch")
    if record["scientific_data"] is not False:
        raise PhaseAdapterError("trajectory scientific_data must be false")
    if record["production_eligible"] is not False:
        raise PhaseAdapterError("trajectory production_eligible must be false")

    trajectory = build_teacher_trajectory(
        teacher_seed_id=record["teacher_seed_id"],
        teacher_seed=record["teacher_seed"],
        task_identity_sha256=record["task_identity_sha256"],
        teacher_artifact_sha256=record["teacher_artifact_sha256"],
        records=record["points"],
    )

    stored = _require_sha256(
        record["content_sha256"],
        name="content_sha256",
    )
    if trajectory.content_sha256() != stored:
        raise PhaseAdapterError("trajectory content hash mismatch")

    return trajectory


@dataclass(frozen=True)
class PhaseDecision:
    """Restricted output of an injected phase rule.

    An injected rule may identify selected training steps or explicit
    unavailability, but it cannot attach arbitrary scientific payloads.
    """

    role: str
    phase_label: str
    availability_status: str
    selected_step: int | None = None
    unavailable_reason: str | None = None
    transition_target: float | None = None
    transition_absolute_distance: float | None = None
    stable_supporting_steps: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.role not in PHASE_ROLES:
            raise PhaseAdapterError(f"invalid phase role: {self.role!r}")
        _require_nonempty_string(self.phase_label, name="phase_label")

        if self.availability_status not in {"selected", "unavailable"}:
            raise PhaseAdapterError(
                "availability_status must be selected or unavailable"
            )

        if self.availability_status == "selected":
            if (
                isinstance(self.selected_step, bool)
                or not isinstance(self.selected_step, int)
                or self.selected_step < 0
            ):
                raise PhaseAdapterError(
                    "selected phase requires non-negative selected_step"
                )
            if self.unavailable_reason is not None:
                raise PhaseAdapterError(
                    "selected phase cannot carry unavailable_reason"
                )
        else:
            if self.selected_step is not None:
                raise PhaseAdapterError(
                    "unavailable phase cannot carry selected_step"
                )
            if (
                not isinstance(self.unavailable_reason, str)
                or not self.unavailable_reason
                or "\n" in self.unavailable_reason
                or "\r" in self.unavailable_reason
            ):
                raise PhaseAdapterError(
                    "unavailable phase requires a single-line reason"
                )

        if self.transition_target is not None:
            object.__setattr__(
                self,
                "transition_target",
                _accuracy(
                    self.transition_target,
                    name="transition_target",
                ),
            )
        if self.transition_absolute_distance is not None:
            distance = _finite_number(
                self.transition_absolute_distance,
                name="transition_absolute_distance",
            )
            if distance < 0:
                raise PhaseAdapterError(
                    "transition_absolute_distance must be non-negative"
                )
            object.__setattr__(
                self,
                "transition_absolute_distance",
                distance,
            )

        if self.stable_supporting_steps is not None:
            if (
                not isinstance(self.stable_supporting_steps, tuple)
                or not self.stable_supporting_steps
                or any(
                    isinstance(step, bool)
                    or not isinstance(step, int)
                    or step < 0
                    for step in self.stable_supporting_steps
                )
            ):
                raise PhaseAdapterError(
                    "stable_supporting_steps must be a non-empty tuple "
                    "of non-negative integers"
                )
            if tuple(sorted(set(self.stable_supporting_steps))) != (
                self.stable_supporting_steps
            ):
                raise PhaseAdapterError(
                    "stable_supporting_steps must be strictly increasing"
                )


class PhaseSelectionRule(Protocol):
    """Explicit versioned injected phase-selection rule."""

    rule_id: str
    rule_version: str

    def select(
        self,
        trajectory: TeacherTrajectory,
    ) -> tuple[PhaseDecision, ...]:
        ...


@dataclass(frozen=True)
class HistoricalStage3PhaseRule:
    """Adapter around the repository's untouched frozen Stage 3 selectors."""

    rule_id: str = HISTORICAL_RULE_ID
    rule_version: str = HISTORICAL_RULE_VERSION

    def select(
        self,
        trajectory: TeacherTrajectory,
    ) -> tuple[PhaseDecision, ...]:
        rows = [point.to_mapping() for point in trajectory.points]

        pre_record = find_pre_grokking_checkpoint(rows)
        if pre_record is None:
            pre = PhaseDecision(
                role="pre",
                phase_label=_PHASE_LABELS["pre"],
                availability_status="unavailable",
                unavailable_reason=(
                    "no saved checkpoint satisfies the frozen "
                    "pre-grokking rule"
                ),
            )
        else:
            pre = PhaseDecision(
                role="pre",
                phase_label=_PHASE_LABELS["pre"],
                availability_status="selected",
                selected_step=int(pre_record["training_step"]),
            )

        try:
            stable_sequence, stable_record = find_stable_post_sequence(rows)
        except ValueError:
            stable_sequence = None
            stable_record = None
            stable = PhaseDecision(
                role="stable_post",
                phase_label=_PHASE_LABELS["stable_post"],
                availability_status="unavailable",
                unavailable_reason=(
                    "no earliest sequence of five consecutive saved "
                    "checkpoints satisfies the frozen stable-post rule"
                ),
            )
        else:
            stable = PhaseDecision(
                role="stable_post",
                phase_label=_PHASE_LABELS["stable_post"],
                availability_status="selected",
                selected_step=int(stable_record["training_step"]),
                stable_supporting_steps=tuple(
                    int(record["training_step"])
                    for record in stable_sequence
                ),
            )

        if pre_record is None:
            transition = PhaseDecision(
                role="transition",
                phase_label=_PHASE_LABELS["transition"],
                availability_status="unavailable",
                unavailable_reason="selected pre-grokking bound is unavailable",
            )
        elif stable_record is None:
            transition = PhaseDecision(
                role="transition",
                phase_label=_PHASE_LABELS["transition"],
                availability_status="unavailable",
                unavailable_reason="selected stable-post bound is unavailable",
            )
        else:
            try:
                landmarks = select_transition_landmarks(
                    rows,
                    pre_step=int(pre_record["training_step"]),
                    stable_post_step=int(stable_record["training_step"]),
                )
            except ValueError:
                transition = PhaseDecision(
                    role="transition",
                    phase_label=_PHASE_LABELS["transition"],
                    availability_status="unavailable",
                    unavailable_reason=(
                        "no saved checkpoint lies strictly between selected "
                        "pre-grokking and stable-post bounds"
                    ),
                )
            else:
                target = float(TRANSITION_TARGETS["50%"])
                record = landmarks.get("50%")
                if record is None:
                    transition = PhaseDecision(
                        role="transition",
                        phase_label=_PHASE_LABELS["transition"],
                        availability_status="unavailable",
                        unavailable_reason=(
                            "frozen selector returned no 50% transition landmark"
                        ),
                    )
                else:
                    observed = float(record["test_accuracy"])
                    transition = PhaseDecision(
                        role="transition",
                        phase_label=_PHASE_LABELS["transition"],
                        availability_status="selected",
                        selected_step=int(record["training_step"]),
                        transition_target=target,
                        transition_absolute_distance=abs(observed - target),
                    )

        return (pre, transition, stable)


def _validate_rule_identity(
    rule: PhaseSelectionRule,
) -> tuple[str, str]:
    rule_id = _require_nonempty_string(
        getattr(rule, "rule_id", None),
        name="phase rule_id",
    )
    rule_version = _require_nonempty_string(
        getattr(rule, "rule_version", None),
        name="phase rule_version",
    )

    if (
        rule_id == HISTORICAL_RULE_ID
        and not isinstance(rule, HistoricalStage3PhaseRule)
    ):
        raise PhaseAdapterError(
            "injected phase rule may not masquerade as historical Stage 3"
        )

    return rule_id, rule_version


def _validated_decisions(
    *,
    trajectory: TeacherTrajectory,
    rule: PhaseSelectionRule,
) -> tuple[PhaseDecision, ...]:
    result = rule.select(trajectory)

    if not isinstance(result, tuple):
        raise PhaseAdapterError(
            "phase rule must return a tuple of PhaseDecision values"
        )
    if len(result) != len(PHASE_ROLES):
        raise PhaseAdapterError(
            "phase rule must return exactly pre/transition/stable_post"
        )
    if any(not isinstance(item, PhaseDecision) for item in result):
        raise PhaseAdapterError(
            "phase rule outputs must be PhaseDecision values"
        )

    roles = tuple(item.role for item in result)
    if roles != PHASE_ROLES:
        raise PhaseAdapterError(
            f"phase-rule role order must be {PHASE_ROLES!r}"
        )

    point_steps = {point.training_step for point in trajectory.points}

    for decision in result:
        if (
            decision.availability_status == "selected"
            and decision.selected_step not in point_steps
        ):
            raise PhaseAdapterError(
                f"phase rule selected unknown trajectory step "
                f"{decision.selected_step}"
            )

        if decision.role != "transition" and (
            decision.transition_target is not None
            or decision.transition_absolute_distance is not None
        ):
            raise PhaseAdapterError(
                "transition metadata may appear only on transition role"
            )

        if (
            decision.role != "stable_post"
            and decision.stable_supporting_steps is not None
        ):
            raise PhaseAdapterError(
                "stable supporting steps may appear only on stable_post role"
            )

        if decision.stable_supporting_steps is not None:
            if any(
                step not in point_steps
                for step in decision.stable_supporting_steps
            ):
                raise PhaseAdapterError(
                    "stable supporting step is absent from trajectory"
                )
            if decision.selected_step != decision.stable_supporting_steps[-1]:
                raise PhaseAdapterError(
                    "stable supporting sequence must end at selected step"
                )

    return result


def _point_by_step(
    trajectory: TeacherTrajectory,
    step: int,
) -> TrajectoryPoint:
    for point in trajectory.points:
        if point.training_step == step:
            return point
    raise PhaseAdapterError(f"trajectory step {step} does not exist")


def _phase_record(
    *,
    trajectory: TeacherTrajectory,
    decision: PhaseDecision,
    rule_id: str,
    rule_version: str,
) -> dict[str, Any]:
    common = {
        "role": decision.role,
        "phase_label": decision.phase_label,
        "availability_status": decision.availability_status,
        "selection_rule_id": rule_id,
        "selection_rule_version": rule_version,
    }

    if decision.availability_status == "unavailable":
        return {
            **common,
            "unavailable_reason": decision.unavailable_reason,
        }

    assert decision.selected_step is not None
    point = _point_by_step(trajectory, decision.selected_step)

    record: dict[str, Any] = {
        **common,
        "training_step": point.training_step,
        "train_accuracy": point.train_accuracy,
        "test_accuracy": point.test_accuracy,
        "train_loss": point.train_loss,
        "test_loss": point.test_loss,
        "checkpoint_path": point.checkpoint_path,
        "checkpoint_sha256": point.checkpoint_sha256,
        "source_point_sha256": point.source_sha256(),
    }

    if decision.role == "transition":
        record["transition_target"] = decision.transition_target
        record["transition_absolute_distance"] = (
            decision.transition_absolute_distance
        )

    if decision.role == "stable_post":
        record["stable_supporting_steps"] = list(
            decision.stable_supporting_steps or ()
        )

    return record


def select_teacher_phases(
    trajectory: TeacherTrajectory | Mapping[str, Any],
    *,
    rule: PhaseSelectionRule | None = None,
) -> dict[str, Any]:
    """Select phases from training/test metrics only and seal the result."""
    value = (
        trajectory
        if isinstance(trajectory, TeacherTrajectory)
        else validate_teacher_trajectory(trajectory)
    )

    selected_rule: PhaseSelectionRule = (
        HistoricalStage3PhaseRule()
        if rule is None
        else rule
    )

    rule_id, rule_version = _validate_rule_identity(selected_rule)
    decisions = _validated_decisions(
        trajectory=value,
        rule=selected_rule,
    )

    material = {
        "schema_version": PHASE_SELECTION_SCHEMA_VERSION,
        "classification": "technical_fixture",
        "scientific_data": False,
        "production_eligible": False,
        "teacher_seed_id": value.teacher_seed_id,
        "teacher_seed": value.teacher_seed,
        "task_identity_sha256": value.task_identity_sha256,
        "teacher_artifact_sha256": value.teacher_artifact_sha256,
        "trajectory_sha256": value.content_sha256(),
        "selection_rule": {
            "rule_id": rule_id,
            "rule_version": rule_version,
        },
        "phase_records": [
            _phase_record(
                trajectory=value,
                decision=decision,
                rule_id=rule_id,
                rule_version=rule_version,
            )
            for decision in decisions
        ],
    }

    record = copy.deepcopy(material)
    record["content_sha256"] = canonical_sha256(material)
    return record


def validate_phase_selection_artifact(
    record: Mapping[str, Any],
    *,
    trajectory: TeacherTrajectory | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate immutable phase evidence and optionally its trajectory binding."""
    if not isinstance(record, Mapping):
        raise PhaseAdapterError("phase selection artifact must be a mapping")

    required = {
        "schema_version",
        "classification",
        "scientific_data",
        "production_eligible",
        "teacher_seed_id",
        "teacher_seed",
        "task_identity_sha256",
        "teacher_artifact_sha256",
        "trajectory_sha256",
        "selection_rule",
        "phase_records",
        "content_sha256",
    }
    if set(record) != required:
        raise PhaseAdapterError("phase selection artifact keys mismatch")

    if record["schema_version"] != PHASE_SELECTION_SCHEMA_VERSION:
        raise PhaseAdapterError("phase artifact schema_version mismatch")
    if record["classification"] != "technical_fixture":
        raise PhaseAdapterError("phase artifact classification mismatch")
    if record["scientific_data"] is not False:
        raise PhaseAdapterError("phase artifact scientific_data must be false")
    if record["production_eligible"] is not False:
        raise PhaseAdapterError(
            "phase artifact production_eligible must be false"
        )

    _require_nonempty_string(
        record["teacher_seed_id"],
        name="teacher_seed_id",
    )
    if (
        isinstance(record["teacher_seed"], bool)
        or not isinstance(record["teacher_seed"], int)
        or record["teacher_seed"] < 0
    ):
        raise PhaseAdapterError("teacher_seed is invalid")

    for name in (
        "task_identity_sha256",
        "teacher_artifact_sha256",
        "trajectory_sha256",
        "content_sha256",
    ):
        _require_sha256(record[name], name=name)

    rule = record["selection_rule"]
    if not isinstance(rule, Mapping) or set(rule) != {
        "rule_id",
        "rule_version",
    }:
        raise PhaseAdapterError("selection_rule keys mismatch")
    _require_nonempty_string(rule["rule_id"], name="selection_rule.rule_id")
    _require_nonempty_string(
        rule["rule_version"],
        name="selection_rule.rule_version",
    )

    phase_records = record["phase_records"]
    if not isinstance(phase_records, list) or len(phase_records) != 3:
        raise PhaseAdapterError(
            "phase artifact requires exactly three phase records"
        )
    if [item.get("role") for item in phase_records] != list(PHASE_ROLES):
        raise PhaseAdapterError("phase artifact role order is invalid")

    material = copy.deepcopy(dict(record))
    stored_hash = material.pop("content_sha256")
    if canonical_sha256(material) != stored_hash:
        raise PhaseAdapterError("phase artifact content hash mismatch")

    if trajectory is not None:
        value = (
            trajectory
            if isinstance(trajectory, TeacherTrajectory)
            else validate_teacher_trajectory(trajectory)
        )
        expected = {
            "teacher_seed_id": value.teacher_seed_id,
            "teacher_seed": value.teacher_seed,
            "task_identity_sha256": value.task_identity_sha256,
            "teacher_artifact_sha256": value.teacher_artifact_sha256,
            "trajectory_sha256": value.content_sha256(),
        }
        for field, expected_value in expected.items():
            if record[field] != expected_value:
                raise PhaseAdapterError(
                    f"phase artifact {field} does not match trajectory"
                )

        by_step = {
            point.training_step: point
            for point in value.points
        }
        for item in phase_records:
            if item["availability_status"] == "selected":
                step = item.get("training_step")
                if step not in by_step:
                    raise PhaseAdapterError(
                        "phase artifact selected unknown trajectory step"
                    )
                point = by_step[step]
                if item.get("source_point_sha256") != point.source_sha256():
                    raise PhaseAdapterError(
                        "phase artifact source-point hash mismatch"
                    )

    return copy.deepcopy(dict(record))


def phase_artifact_sha256(record: Mapping[str, Any]) -> str:
    """Return exact canonical serialized artifact SHA-256."""
    validated = validate_phase_selection_artifact(record)
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()
