"""Deterministic Stage 3 teacher-registry construction and serialization.

This module constructs small canonical metadata records from already-selected
in-memory candidates. It performs no phase selection, model inference, or
scientific endpoint computation.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from circuit_families.analysis.stage3_teacher_selection import (
    PhaseCandidate,
    TeacherCandidates,
)

REGISTRY_SCHEMA_VERSION = "1"
RECORD_SCHEMA_VERSION = "1"
REGISTRY_NAMESPACE = "circuit-families-distillation/stage3-teacher-registry"

CANONICAL_SEEDS = (0, 1, 2, 3, 4)
CANONICAL_PHASES = (
    "pre-grokking",
    "50%",
    "stable post-grokking",
)

PRE_RULE = (
    "latest saved checkpoint with train_accuracy >= 0.999 and "
    "test_accuracy <= 0.05 before the first saved checkpoint with "
    "test_accuracy >= 0.10"
)
TRANSITION_RULE = (
    "saved checkpoint strictly after the selected pre-grokking step and "
    "strictly before the selected stable-post step minimizing "
    "abs(test_accuracy - 0.50); exact ties choose the earlier training step"
)
STABLE_RULE = (
    "fifth checkpoint in the earliest sequence of five consecutive saved "
    "checkpoints with test_accuracy >= 0.99"
)

RULE_BY_PHASE = {
    "pre-grokking": PRE_RULE,
    "50%": TRANSITION_RULE,
    "stable post-grokking": STABLE_RULE,
}

_ALLOWED_DENSE_SELECTED = {"not-generated", "pre-existing-sealed"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Stage3RegistryError(ValueError):
    """Raised when a Stage 3 canonical registry record is invalid."""


@dataclass(frozen=True)
class RegistryTeacherProvenance:
    """Portable provenance needed for every phase record of one teacher."""

    teacher_seed: int
    canonical_run_id: str
    training_manifest_path: str
    metrics_path: str
    training_manifest_sha256: str
    metrics_sha256: str
    training_interval: int
    evaluation_interval: int
    checkpoint_interval: int
    run_max_step: int
    predecessor_analysis_freeze_commit: str
    training_code_commit: str
    model_identity: Any
    training_config_identity: Any
    dataset_identity: Any
    split_identity: Any
    selected_dense_output_status: str = "not-generated"


TABLE_COLUMNS = (
    "teacher_seed",
    "canonical_run_id",
    "phase_label",
    "availability_status",
    "training_step",
    "train_accuracy",
    "test_accuracy",
    "train_loss",
    "test_loss",
    "transition_target",
    "transition_absolute_distance",
    "stable_supporting_sequence_steps",
    "checkpoint_path",
    "checkpoint_sha256",
    "unavailable_reason",
)


def _portable_relative_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Stage3RegistryError(f"{label} must be a non-empty string")
    if value.startswith("/") or value.startswith("~") or "\\" in value:
        raise Stage3RegistryError(f"{label} is not a portable relative path")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise Stage3RegistryError(f"{label} contains path escape")
    return path.as_posix()


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Stage3RegistryError(f"{label} must be lowercase SHA-256 hex")
    return value


def _finite_float(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Stage3RegistryError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Stage3RegistryError(f"{label} must be finite")
    return result


def _canonical_json_value(value: Any) -> Any:
    """Normalize nested identity metadata deterministically."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Stage3RegistryError("identity metadata contains non-finite float")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_canonical_json_value(item) for item in value]
    raise Stage3RegistryError(
        f"identity metadata contains unsupported type {type(value).__name__}"
    )


def _validate_provenance(
    provenance: RegistryTeacherProvenance,
    candidate: TeacherCandidates,
) -> None:
    if provenance.teacher_seed != candidate.seed:
        raise Stage3RegistryError("teacher seed/provenance mismatch")
    if provenance.canonical_run_id != candidate.run_id:
        raise Stage3RegistryError("teacher run/provenance mismatch")

    validated = candidate.validated_input
    if provenance.training_manifest_path != validated.manifest_path:
        raise Stage3RegistryError("training manifest path disagrees with validation")
    if provenance.metrics_path != validated.metrics_path:
        raise Stage3RegistryError("metrics path disagrees with validation")
    if provenance.training_manifest_sha256 != validated.manifest_sha256:
        raise Stage3RegistryError("training manifest hash disagrees with validation")
    if provenance.metrics_sha256 != validated.metrics_sha256:
        raise Stage3RegistryError("metrics hash disagrees with validation")
    if provenance.evaluation_interval != validated.evaluation_interval:
        raise Stage3RegistryError("evaluation interval disagrees with validation")
    if provenance.checkpoint_interval != validated.checkpoint_interval:
        raise Stage3RegistryError("checkpoint interval disagrees with validation")

    _portable_relative_path(
        provenance.training_manifest_path,
        "training_manifest_path",
    )
    _portable_relative_path(provenance.metrics_path, "metrics_path")
    _sha256(provenance.training_manifest_sha256, "training_manifest_sha256")
    _sha256(provenance.metrics_sha256, "metrics_sha256")

    for label, value in (
        ("training_interval", provenance.training_interval),
        ("evaluation_interval", provenance.evaluation_interval),
        ("checkpoint_interval", provenance.checkpoint_interval),
        ("run_max_step", provenance.run_max_step),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise Stage3RegistryError(f"{label} must be a positive integer")

    if provenance.selected_dense_output_status not in _ALLOWED_DENSE_SELECTED:
        raise Stage3RegistryError("invalid selected dense-output status")


def _phase_candidate(
    teacher: TeacherCandidates,
    phase_label: str,
) -> PhaseCandidate:
    if phase_label == "pre-grokking":
        return teacher.pre
    if phase_label == "50%":
        return teacher.transition_50
    if phase_label == "stable post-grokking":
        return teacher.stable
    raise Stage3RegistryError(f"unknown phase label {phase_label!r}")


def _common_record(
    provenance: RegistryTeacherProvenance,
    phase_label: str,
    availability_status: str,
) -> dict[str, Any]:
    return {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "registry_namespace": REGISTRY_NAMESPACE,
        "teacher_seed": provenance.teacher_seed,
        "canonical_run_id": provenance.canonical_run_id,
        "phase_label": phase_label,
        "selection_rule": RULE_BY_PHASE[phase_label],
        "availability_status": availability_status,
        "training_manifest_path": _portable_relative_path(
            provenance.training_manifest_path,
            "training_manifest_path",
        ),
        "metrics_path": _portable_relative_path(
            provenance.metrics_path,
            "metrics_path",
        ),
        "training_manifest_sha256": _sha256(
            provenance.training_manifest_sha256,
            "training_manifest_sha256",
        ),
        "metrics_sha256": _sha256(
            provenance.metrics_sha256,
            "metrics_sha256",
        ),
        "training_interval": provenance.training_interval,
        "evaluation_interval": provenance.evaluation_interval,
        "checkpoint_interval": provenance.checkpoint_interval,
        "run_max_step": provenance.run_max_step,
        "predecessor_analysis_freeze_commit": (
            provenance.predecessor_analysis_freeze_commit
        ),
        "training_code_commit": provenance.training_code_commit,
        "model_identity": _canonical_json_value(provenance.model_identity),
        "training_config_identity": _canonical_json_value(
            provenance.training_config_identity
        ),
        "dataset_identity": _canonical_json_value(provenance.dataset_identity),
        "split_identity": _canonical_json_value(provenance.split_identity),
    }


def build_phase_record(
    teacher: TeacherCandidates,
    provenance: RegistryTeacherProvenance,
    phase_label: str,
) -> dict[str, Any]:
    """Build one canonical selected or unavailable phase record."""

    if phase_label not in CANONICAL_PHASES:
        raise Stage3RegistryError(f"invalid phase label {phase_label!r}")

    _validate_provenance(provenance, teacher)
    candidate = _phase_candidate(teacher, phase_label)

    if candidate.phase_label != phase_label:
        raise Stage3RegistryError("phase candidate label mismatch")

    if candidate.availability_status == "unavailable":
        if candidate.record is not None:
            raise Stage3RegistryError("unavailable candidate contains selected record")
        if not candidate.unavailable_reason:
            raise Stage3RegistryError("unavailable candidate lacks reason")

        record = _common_record(provenance, phase_label, "unavailable")
        record["dense_output_status"] = "unavailable"
        record["unavailable_reason"] = candidate.unavailable_reason
        return record

    if candidate.availability_status != "selected":
        raise Stage3RegistryError("invalid availability status")
    if candidate.record is None:
        raise Stage3RegistryError("selected candidate lacks metrics record")

    source = candidate.record
    record = _common_record(provenance, phase_label, "selected")

    training_step = source.get("training_step")
    if not isinstance(training_step, int) or isinstance(training_step, bool):
        raise Stage3RegistryError("training_step must be integer")

    record["training_step"] = training_step
    record["train_accuracy"] = _finite_float(
        source.get("train_accuracy"), "train_accuracy"
    )
    record["test_accuracy"] = _finite_float(
        source.get("test_accuracy"), "test_accuracy"
    )
    record["train_loss"] = _finite_float(source.get("train_loss"), "train_loss")
    record["test_loss"] = _finite_float(source.get("test_loss"), "test_loss")

    checkpoint_path = source.get("checkpoint_path")
    checkpoint_sha256 = source.get("checkpoint_sha256")
    record["checkpoint_path"] = _portable_relative_path(
        checkpoint_path,
        "checkpoint_path",
    )
    record["checkpoint_sha256"] = _sha256(
        checkpoint_sha256,
        "checkpoint_sha256",
    )

    if phase_label == "50%":
        if candidate.transition_target != 0.50:
            raise Stage3RegistryError("50% candidate target must equal 0.50")
        record["transition_target"] = 0.50
        record["transition_absolute_distance"] = _finite_float(
            candidate.transition_absolute_distance,
            "transition_absolute_distance",
        )

    if phase_label == "stable post-grokking":
        sequence = candidate.stable_supporting_sequence_steps
        if sequence is None or len(sequence) != 5:
            raise Stage3RegistryError(
                "stable selected record requires five-step supporting sequence"
            )
        steps = [int(step) for step in sequence]
        if steps[-1] != training_step:
            raise Stage3RegistryError(
                "stable supporting sequence does not end at selected step"
            )
        record["stable_supporting_sequence_steps"] = steps

    record["dense_output_status"] = provenance.selected_dense_output_status
    return record


def build_registry(
    candidates: Sequence[TeacherCandidates],
    provenance_by_seed: Mapping[int, RegistryTeacherProvenance],
) -> dict[str, Any]:
    """Build the complete deterministic 15-cell canonical registry."""

    by_seed: dict[int, TeacherCandidates] = {}
    for candidate in candidates:
        if candidate.seed in by_seed:
            raise Stage3RegistryError(f"duplicate teacher seed {candidate.seed}")
        by_seed[candidate.seed] = candidate

    if tuple(sorted(by_seed)) != CANONICAL_SEEDS:
        raise Stage3RegistryError(
            f"teacher seeds {tuple(sorted(by_seed))} != {CANONICAL_SEEDS}"
        )
    if tuple(sorted(provenance_by_seed)) != CANONICAL_SEEDS:
        raise Stage3RegistryError("provenance seed set is not canonical 0..4")

    records: list[dict[str, Any]] = []
    for seed in CANONICAL_SEEDS:
        teacher = by_seed[seed]
        provenance = provenance_by_seed[seed]
        for phase in CANONICAL_PHASES:
            records.append(build_phase_record(teacher, provenance, phase))

    selected_count = sum(
        record["availability_status"] == "selected" for record in records
    )
    unavailable_count = len(records) - selected_count

    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "namespace": REGISTRY_NAMESPACE,
        "canonical_seed_order": list(CANONICAL_SEEDS),
        "canonical_phase_order": list(CANONICAL_PHASES),
        "expected_cell_count": 15,
        "selected_cell_count": selected_count,
        "unavailable_cell_count": unavailable_count,
        "records": records,
    }


def build_phase_selection_table(
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build deterministic compact seed/phase rows from a canonical registry."""

    records = registry.get("records")
    if not isinstance(records, list) or len(records) != 15:
        raise Stage3RegistryError("registry must contain exactly 15 records")

    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise Stage3RegistryError("registry record must be an object")

        sequence = record.get("stable_supporting_sequence_steps")
        if sequence is None:
            sequence_text = ""
        else:
            sequence_text = "|".join(str(int(step)) for step in sequence)

        rows.append(
            {
                "teacher_seed": record["teacher_seed"],
                "canonical_run_id": record["canonical_run_id"],
                "phase_label": record["phase_label"],
                "availability_status": record["availability_status"],
                "training_step": record.get("training_step", ""),
                "train_accuracy": record.get("train_accuracy", ""),
                "test_accuracy": record.get("test_accuracy", ""),
                "train_loss": record.get("train_loss", ""),
                "test_loss": record.get("test_loss", ""),
                "transition_target": record.get("transition_target", ""),
                "transition_absolute_distance": record.get(
                    "transition_absolute_distance", ""
                ),
                "stable_supporting_sequence_steps": sequence_text,
                "checkpoint_path": record.get("checkpoint_path", ""),
                "checkpoint_sha256": record.get("checkpoint_sha256", ""),
                "unavailable_reason": record.get("unavailable_reason", ""),
            }
        )

    return rows


def serialize_registry_json(registry: Mapping[str, Any]) -> bytes:
    """Serialize canonical registry deterministically as UTF-8 JSON."""

    normalized = _canonical_json_value(registry)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=False,
    )
    return (text + "\n").encode("utf-8")


def _csv_scalar(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Stage3RegistryError("table contains non-finite float")
        return repr(value)
    return str(value)


def serialize_phase_selection_table_csv(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize canonical phase table deterministically as UTF-8 CSV."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(TABLE_COLUMNS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()

    for row in rows:
        if tuple(row.keys()) != TABLE_COLUMNS:
            raise Stage3RegistryError("phase table row field order is noncanonical")
        writer.writerow({key: _csv_scalar(row[key]) for key in TABLE_COLUMNS})

    return buffer.getvalue().encode("utf-8")
