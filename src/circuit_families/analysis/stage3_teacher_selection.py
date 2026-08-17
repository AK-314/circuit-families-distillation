"""In-memory Stage 3 teacher phase candidate extraction.

Selection delegates to the inherited frozen phase_detection implementation.
Only canonical predecessor training metrics are read. No registry or scientific
output is written here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from circuit_families.analysis.phase_detection import (
    find_pre_grokking_checkpoint,
    find_stable_post_sequence,
    select_transition_landmarks,
)
from circuit_families.analysis.stage3_teacher_inputs import (
    LinkedTeacher,
    Stage3InputError,
    ValidatedTeacherInput,
    load_stage1_canonical_roster,
    validate_teacher_input,
)

PHASE_PRE = "pre-grokking"
PHASE_TRANSITION = "50%"
PHASE_STABLE = "stable post-grokking"


@dataclass(frozen=True)
class PhaseCandidate:
    phase_label: str
    availability_status: str
    record: Mapping[str, Any] | None
    unavailable_reason: str | None
    transition_target: float | None = None
    transition_absolute_distance: float | None = None
    stable_supporting_sequence_steps: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TeacherCandidates:
    seed: int
    run_id: str
    validated_input: ValidatedTeacherInput
    pre: PhaseCandidate
    transition_landmarks: Mapping[str, Mapping[str, Any]]
    transition_50: PhaseCandidate
    stable: PhaseCandidate


def _load_validated_metrics(
    linked: LinkedTeacher,
    predecessor_root: str | Path,
) -> tuple[ValidatedTeacherInput, list[Mapping[str, Any]]]:
    validated = validate_teacher_input(
        linked,
        predecessor_root,
        verify_checkpoint_hashes=True,
    )

    metrics_path = Path(predecessor_root).resolve() / validated.metrics_path
    rows: list[Mapping[str, Any]] = []

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise Stage3InputError("validated metrics row is not an object")
                rows.append(value)

    if len(rows) != validated.metrics_row_count:
        raise Stage3InputError(
            f"seed {validated.seed}: metrics changed after validation"
        )

    return validated, rows


def _selected(
    phase_label: str,
    record: Mapping[str, Any],
    *,
    transition_target: float | None = None,
    transition_absolute_distance: float | None = None,
    stable_supporting_sequence_steps: tuple[int, ...] | None = None,
) -> PhaseCandidate:
    return PhaseCandidate(
        phase_label=phase_label,
        availability_status="selected",
        record=record,
        unavailable_reason=None,
        transition_target=transition_target,
        transition_absolute_distance=transition_absolute_distance,
        stable_supporting_sequence_steps=stable_supporting_sequence_steps,
    )


def _unavailable(phase_label: str, reason: str) -> PhaseCandidate:
    return PhaseCandidate(
        phase_label=phase_label,
        availability_status="unavailable",
        record=None,
        unavailable_reason=reason,
    )


def extract_teacher_candidates(
    linked: LinkedTeacher,
    predecessor_root: str | Path,
) -> TeacherCandidates:
    """Compute one teacher's frozen-rule candidates in memory only."""

    validated, rows = _load_validated_metrics(linked, predecessor_root)

    pre_record = find_pre_grokking_checkpoint(rows)
    if pre_record is None:
        pre = _unavailable(
            PHASE_PRE,
            "no saved checkpoint satisfies the frozen pre-grokking rule",
        )
    else:
        pre = _selected(PHASE_PRE, pre_record)

    try:
        stable_sequence, stable_record = find_stable_post_sequence(rows)
    except ValueError:
        stable_sequence = None
        stable_record = None
        stable = _unavailable(
            PHASE_STABLE,
            "no earliest sequence of five consecutive saved checkpoints "
            "with test_accuracy >= 0.99",
        )
    else:
        stable_steps = tuple(
            int(record["training_step"]) for record in stable_sequence
        )
        stable = _selected(
            PHASE_STABLE,
            stable_record,
            stable_supporting_sequence_steps=stable_steps,
        )

    transition_landmarks: Mapping[str, Mapping[str, Any]] = {}

    if pre_record is None:
        transition = _unavailable(
            PHASE_TRANSITION,
            "selected pre-grokking bound is unavailable",
        )
    elif stable_record is None:
        transition = _unavailable(
            PHASE_TRANSITION,
            "selected stable-post bound is unavailable",
        )
    else:
        pre_step = int(pre_record["training_step"])
        stable_step = int(stable_record["training_step"])

        try:
            transition_landmarks = select_transition_landmarks(
                rows,
                pre_step=pre_step,
                stable_post_step=stable_step,
            )
        except ValueError:
            transition = _unavailable(
                PHASE_TRANSITION,
                "no saved checkpoint lies strictly between selected "
                "pre-grokking and stable-post bounds",
            )
        else:
            record_50 = transition_landmarks.get("50%")
            if record_50 is None:
                transition = _unavailable(
                    PHASE_TRANSITION,
                    "frozen selector returned no 50% transition landmark",
                )
            else:
                test_accuracy = float(record_50["test_accuracy"])
                transition = _selected(
                    PHASE_TRANSITION,
                    record_50,
                    transition_target=0.50,
                    transition_absolute_distance=abs(test_accuracy - 0.50),
                )

    return TeacherCandidates(
        seed=validated.seed,
        run_id=validated.run_id,
        validated_input=validated,
        pre=pre,
        transition_landmarks=transition_landmarks,
        transition_50=transition,
        stable=stable,
    )


def extract_all_teacher_candidates(
    successor_root: str | Path,
    predecessor_root: str | Path,
) -> tuple[TeacherCandidates, ...]:
    """Compute canonical seed-major candidates without writing outputs."""

    linked = load_stage1_canonical_roster(successor_root)

    candidates = tuple(
        extract_teacher_candidates(item, predecessor_root)
        for item in linked
    )

    if [item.seed for item in candidates] != [0, 1, 2, 3, 4]:
        raise Stage3InputError("candidate extraction lost canonical seed order")

    return candidates
