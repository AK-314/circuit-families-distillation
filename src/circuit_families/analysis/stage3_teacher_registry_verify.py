"""Verification for the deterministic Stage 3 teacher registry.

Portable mode verifies canonical structure without claiming private physical
checks passed. Physical mode additionally revalidates predecessor inputs,
recomputes frozen-rule selections, and compares every canonical phase record.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from circuit_families.analysis.stage3_teacher_registry import (
    CANONICAL_PHASES,
    CANONICAL_SEEDS,
    RECORD_SCHEMA_VERSION,
    REGISTRY_NAMESPACE,
    REGISTRY_SCHEMA_VERSION,
    RegistryTeacherProvenance,
    build_phase_record,
)
from circuit_families.analysis.stage3_teacher_selection import (
    TeacherCandidates,
    extract_all_teacher_candidates,
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

COMMON_RECORD_FIELDS = (
    "record_schema_version",
    "registry_namespace",
    "teacher_seed",
    "canonical_run_id",
    "phase_label",
    "selection_rule",
    "availability_status",
    "training_manifest_path",
    "metrics_path",
    "training_manifest_sha256",
    "metrics_sha256",
    "training_interval",
    "evaluation_interval",
    "checkpoint_interval",
    "run_max_step",
    "predecessor_analysis_freeze_commit",
    "training_code_commit",
    "model_identity",
    "training_config_identity",
    "dataset_identity",
    "split_identity",
    "dense_output_status",
)

SELECTED_FIELDS = (
    "training_step",
    "train_accuracy",
    "test_accuracy",
    "train_loss",
    "test_loss",
    "checkpoint_path",
    "checkpoint_sha256",
)

PHASE_SPECIFIC_FIELDS = {
    "pre-grokking": (),
    "50%": (
        "transition_target",
        "transition_absolute_distance",
    ),
    "stable post-grokking": (
        "stable_supporting_sequence_steps",
    ),
}

IDENTITY_FIELDS = (
    "canonical_run_id",
    "training_manifest_path",
    "metrics_path",
    "training_manifest_sha256",
    "metrics_sha256",
    "training_interval",
    "evaluation_interval",
    "checkpoint_interval",
    "run_max_step",
    "predecessor_analysis_freeze_commit",
    "training_code_commit",
    "model_identity",
    "training_config_identity",
    "dataset_identity",
    "split_identity",
)


class Stage3RegistryVerificationError(ValueError):
    """Raised when canonical Stage 3 verification fails."""


def _portable_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Stage3RegistryVerificationError(f"{label} must be non-empty string")
    if value.startswith("/") or value.startswith("~") or "\\" in value:
        raise Stage3RegistryVerificationError(f"{label} is not portable")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise Stage3RegistryVerificationError(f"{label} escapes its root")
    return path.as_posix()


def _sha64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Stage3RegistryVerificationError(f"{label} is not SHA-256")
    return value


def _commit40(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise Stage3RegistryVerificationError(f"{label} is not a 40-hex commit")
    return value


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Stage3RegistryVerificationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise Stage3RegistryVerificationError(f"{label} must be finite")
    return number


def _require_fields(record: Mapping[str, Any], fields: Sequence[str]) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise Stage3RegistryVerificationError(
            "record missing fields: " + ",".join(missing)
        )


def verify_registry_structure(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Verify schema, cell accounting, ordering, and portable identities."""

    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise Stage3RegistryVerificationError("wrong registry schema_version")
    if registry.get("namespace") != REGISTRY_NAMESPACE:
        raise Stage3RegistryVerificationError("wrong registry namespace")
    if registry.get("canonical_seed_order") != list(CANONICAL_SEEDS):
        raise Stage3RegistryVerificationError("wrong canonical seed order")
    if registry.get("canonical_phase_order") != list(CANONICAL_PHASES):
        raise Stage3RegistryVerificationError("wrong canonical phase order")
    if registry.get("expected_cell_count") != 15:
        raise Stage3RegistryVerificationError("expected_cell_count must be 15")

    records = registry.get("records")
    if not isinstance(records, list) or len(records) != 15:
        raise Stage3RegistryVerificationError(
            "registry must contain exactly 15 explicit records"
        )

    expected_order = [
        (seed, phase)
        for seed in CANONICAL_SEEDS
        for phase in CANONICAL_PHASES
    ]
    observed_order: list[tuple[Any, Any]] = []

    selected_count = 0
    unavailable_count = 0

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise Stage3RegistryVerificationError(
                f"record {index} must be an object"
            )

        _require_fields(record, COMMON_RECORD_FIELDS)

        if record["record_schema_version"] != RECORD_SCHEMA_VERSION:
            raise Stage3RegistryVerificationError(
                f"record {index} has wrong schema version"
            )
        if record["registry_namespace"] != REGISTRY_NAMESPACE:
            raise Stage3RegistryVerificationError(
                f"record {index} has wrong namespace"
            )

        seed = record["teacher_seed"]
        phase = record["phase_label"]
        observed_order.append((seed, phase))

        if seed not in CANONICAL_SEEDS:
            raise Stage3RegistryVerificationError(
                f"record {index} invalid teacher seed"
            )
        if phase not in CANONICAL_PHASES:
            raise Stage3RegistryVerificationError(
                f"record {index} invalid phase label"
            )

        _portable_path(
            record["training_manifest_path"],
            f"record {index} training_manifest_path",
        )
        _portable_path(record["metrics_path"], f"record {index} metrics_path")
        _sha64(
            record["training_manifest_sha256"],
            f"record {index} training_manifest_sha256",
        )
        _sha64(record["metrics_sha256"], f"record {index} metrics_sha256")
        _commit40(
            record["predecessor_analysis_freeze_commit"],
            f"record {index} predecessor_analysis_freeze_commit",
        )
        _commit40(
            record["training_code_commit"],
            f"record {index} training_code_commit",
        )

        for field in (
            "training_interval",
            "evaluation_interval",
            "checkpoint_interval",
            "run_max_step",
        ):
            value = record[field]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise Stage3RegistryVerificationError(
                    f"record {index} invalid {field}"
                )

        availability = record["availability_status"]

        if availability == "selected":
            selected_count += 1
            _require_fields(record, SELECTED_FIELDS)

            for field in (
                "train_accuracy",
                "test_accuracy",
                "train_loss",
                "test_loss",
            ):
                _finite_number(record[field], f"record {index} {field}")

            step = record["training_step"]
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise Stage3RegistryVerificationError(
                    f"record {index} invalid training_step"
                )

            _portable_path(
                record["checkpoint_path"],
                f"record {index} checkpoint_path",
            )
            _sha64(
                record["checkpoint_sha256"],
                f"record {index} checkpoint_sha256",
            )

            if "unavailable_reason" in record:
                raise Stage3RegistryVerificationError(
                    f"record {index} selected but has unavailable_reason"
                )

            dense = record["dense_output_status"]
            if dense not in {"not-generated", "pre-existing-sealed"}:
                raise Stage3RegistryVerificationError(
                    f"record {index} invalid selected dense_output_status"
                )

            _require_fields(record, PHASE_SPECIFIC_FIELDS[phase])

            if phase == "50%":
                if float(record["transition_target"]) != 0.50:
                    raise Stage3RegistryVerificationError(
                        f"record {index} transition target is not 0.50"
                    )
                distance = _finite_number(
                    record["transition_absolute_distance"],
                    f"record {index} transition_absolute_distance",
                )
                if distance < 0:
                    raise Stage3RegistryVerificationError(
                        f"record {index} negative transition distance"
                    )
                actual_distance = abs(float(record["test_accuracy"]) - 0.50)
                if not math.isclose(
                    distance,
                    actual_distance,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise Stage3RegistryVerificationError(
                        f"record {index} transition distance mismatch"
                    )

            if phase == "stable post-grokking":
                sequence = record["stable_supporting_sequence_steps"]
                if not isinstance(sequence, list) or len(sequence) != 5:
                    raise Stage3RegistryVerificationError(
                        f"record {index} stable support must have five steps"
                    )
                if sequence[-1] != step:
                    raise Stage3RegistryVerificationError(
                        f"record {index} stable sequence does not end at selected step"
                    )

        elif availability == "unavailable":
            unavailable_count += 1
            reason = record.get("unavailable_reason")
            if not isinstance(reason, str) or not reason:
                raise Stage3RegistryVerificationError(
                    f"record {index} unavailable without reason"
                )

            forbidden = set(SELECTED_FIELDS)
            forbidden.update(PHASE_SPECIFIC_FIELDS[phase])
            present = sorted(field for field in forbidden if field in record)
            if present:
                raise Stage3RegistryVerificationError(
                    f"record {index} unavailable contains selected fields: "
                    + ",".join(present)
                )

            if record["dense_output_status"] != "unavailable":
                raise Stage3RegistryVerificationError(
                    f"record {index} unavailable dense-output status mismatch"
                )

        else:
            raise Stage3RegistryVerificationError(
                f"record {index} invalid availability status"
            )

    if observed_order != expected_order:
        raise Stage3RegistryVerificationError("registry record order is noncanonical")

    if registry.get("selected_cell_count") != selected_count:
        raise Stage3RegistryVerificationError("selected-cell accounting mismatch")
    if registry.get("unavailable_cell_count") != unavailable_count:
        raise Stage3RegistryVerificationError("unavailable-cell accounting mismatch")
    if selected_count + unavailable_count != 15:
        raise Stage3RegistryVerificationError("cell accounting does not total 15")

    return {
        "structural_status": "PASS",
        "record_count": 15,
        "selected_cell_count": selected_count,
        "unavailable_cell_count": unavailable_count,
        "ordering_status": "PASS",
        "portable_path_status": "PASS",
        "physical_status": "SKIPPED",
    }


def _records_by_seed(
    registry: Mapping[str, Any],
) -> dict[int, list[Mapping[str, Any]]]:
    records = registry["records"]
    return {
        seed: [
            record
            for record in records
            if record["teacher_seed"] == seed
        ]
        for seed in CANONICAL_SEEDS
    }


def _provenance_from_records(
    seed_records: Sequence[Mapping[str, Any]],
) -> RegistryTeacherProvenance:
    if len(seed_records) != 3:
        raise Stage3RegistryVerificationError(
            "each seed must have exactly three phase records"
        )

    first = seed_records[0]

    for field in IDENTITY_FIELDS:
        values = [record[field] for record in seed_records]
        if values[1:] != values[:-1]:
            raise Stage3RegistryVerificationError(
                f"per-seed provenance differs across phases for {field}"
            )

    selected_dense = {
        record["dense_output_status"]
        for record in seed_records
        if record["availability_status"] == "selected"
    }
    if len(selected_dense) != 1:
        raise Stage3RegistryVerificationError(
            "selected dense-output status differs within seed"
        )

    return RegistryTeacherProvenance(
        teacher_seed=int(first["teacher_seed"]),
        canonical_run_id=str(first["canonical_run_id"]),
        training_manifest_path=str(first["training_manifest_path"]),
        metrics_path=str(first["metrics_path"]),
        training_manifest_sha256=str(first["training_manifest_sha256"]),
        metrics_sha256=str(first["metrics_sha256"]),
        training_interval=int(first["training_interval"]),
        evaluation_interval=int(first["evaluation_interval"]),
        checkpoint_interval=int(first["checkpoint_interval"]),
        run_max_step=int(first["run_max_step"]),
        predecessor_analysis_freeze_commit=str(
            first["predecessor_analysis_freeze_commit"]
        ),
        training_code_commit=str(first["training_code_commit"]),
        model_identity=first["model_identity"],
        training_config_identity=first["training_config_identity"],
        dataset_identity=first["dataset_identity"],
        split_identity=first["split_identity"],
        selected_dense_output_status=next(iter(selected_dense)),
    )


def _rule_diagnostics(
    teacher: TeacherCandidates,
    predecessor_root: str | Path,
) -> dict[str, Any]:
    metrics_path = (
        Path(predecessor_root).resolve()
        / teacher.validated_input.metrics_path
    )
    with metrics_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    result: dict[str, Any] = {"seed": teacher.seed}

    if teacher.pre.availability_status == "selected":
        record = teacher.pre.record
        assert record is not None
        first_10 = next(
            int(row["training_step"])
            for row in rows
            if float(row["test_accuracy"]) >= 0.10
        )
        result["pre_train_margin"] = float(record["train_accuracy"]) - 0.999
        result["pre_test_margin"] = 0.05 - float(record["test_accuracy"])
        result["pre_gap_to_first_10"] = (
            first_10 - int(record["training_step"])
        )
    else:
        result["pre_status"] = "unavailable"

    if teacher.transition_50.availability_status == "selected":
        record = teacher.transition_50.record
        assert record is not None
        result["transition_distance"] = abs(
            float(record["test_accuracy"]) - 0.50
        )
    else:
        result["transition_status"] = "unavailable"

    if teacher.stable.availability_status == "selected":
        sequence = teacher.stable.stable_supporting_sequence_steps
        assert sequence is not None
        by_step = {
            int(row["training_step"]): row
            for row in rows
        }
        result["stable_margin"] = min(
            float(by_step[step]["test_accuracy"])
            for step in sequence
        ) - 0.99
    else:
        result["stable_status"] = "unavailable"

    return result


def verify_registry_physical(
    registry: Mapping[str, Any],
    successor_root: str | Path,
    predecessor_root: str | Path,
) -> dict[str, Any]:
    """Recompute physical provenance and frozen selections from predecessor."""

    structural = verify_registry_structure(registry)

    candidates = extract_all_teacher_candidates(
        successor_root,
        predecessor_root,
    )
    if [candidate.seed for candidate in candidates] != list(CANONICAL_SEEDS):
        raise Stage3RegistryVerificationError(
            "physical recomputation lost canonical seed order"
        )

    candidate_by_seed = {
        candidate.seed: candidate
        for candidate in candidates
    }
    records_by_seed = _records_by_seed(registry)

    diagnostics: list[dict[str, Any]] = []

    for seed in CANONICAL_SEEDS:
        teacher = candidate_by_seed[seed]
        seed_records = records_by_seed[seed]
        provenance = _provenance_from_records(seed_records)

        validated = teacher.validated_input

        if provenance.canonical_run_id != validated.run_id:
            raise Stage3RegistryVerificationError(
                f"seed {seed} canonical run mismatch"
            )
        if provenance.training_manifest_path != validated.manifest_path:
            raise Stage3RegistryVerificationError(
                f"seed {seed} manifest path mismatch"
            )
        if provenance.metrics_path != validated.metrics_path:
            raise Stage3RegistryVerificationError(
                f"seed {seed} metrics path mismatch"
            )
        if provenance.training_manifest_sha256 != validated.manifest_sha256:
            raise Stage3RegistryVerificationError(
                f"seed {seed} manifest hash mismatch"
            )
        if provenance.metrics_sha256 != validated.metrics_sha256:
            raise Stage3RegistryVerificationError(
                f"seed {seed} metrics hash mismatch"
            )
        if provenance.evaluation_interval != validated.evaluation_interval:
            raise Stage3RegistryVerificationError(
                f"seed {seed} evaluation interval mismatch"
            )
        if provenance.checkpoint_interval != validated.checkpoint_interval:
            raise Stage3RegistryVerificationError(
                f"seed {seed} checkpoint interval mismatch"
            )
        if provenance.run_max_step != validated.last_step:
            raise Stage3RegistryVerificationError(
                f"seed {seed} run max-step mismatch"
            )

        for phase, actual in zip(
            CANONICAL_PHASES,
            seed_records,
            strict=True,
        ):
            expected = build_phase_record(
                teacher,
                provenance,
                phase,
            )
            if dict(actual) != expected:
                raise Stage3RegistryVerificationError(
                    f"seed {seed} phase {phase!r} differs from "
                    "physical frozen-rule recomputation"
                )

        diagnostics.append(
            _rule_diagnostics(teacher, predecessor_root)
        )

    result = dict(structural)
    result["physical_status"] = "PASS"
    result["source_hash_status"] = "PASS"
    result["checkpoint_hash_status"] = "PASS"
    result["selection_recomputation_status"] = "PASS"
    result["rule_margin_recomputation_status"] = "PASS"
    result["rule_diagnostics"] = diagnostics
    return result


def verify_resolution_linkage(
    resolution: Mapping[str, Any],
    *,
    successor_root: str | Path | None = None,
    registry_sha256: str | None = None,
    table_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify separate Stage 3 resolution/sealing linkage for UD-001/UD-002."""

    required = {
        "resolution_schema_version",
        "registry_namespace",
        "resolution_status",
        "resolves_decision_ids",
        "historical_stage2_register_mutated",
        "resolution_source",
    }
    missing = sorted(required - set(resolution))
    if missing:
        raise Stage3RegistryVerificationError(
            "resolution record missing: " + ",".join(missing)
        )

    if resolution["resolution_schema_version"] != "1":
        raise Stage3RegistryVerificationError(
            "wrong resolution_schema_version"
        )
    if resolution["registry_namespace"] != REGISTRY_NAMESPACE:
        raise Stage3RegistryVerificationError(
            "resolution registry namespace mismatch"
        )
    if resolution["resolution_status"] != "resolved":
        raise Stage3RegistryVerificationError(
            "resolution_status must be resolved"
        )
    if resolution["resolves_decision_ids"] != ["UD-001", "UD-002"]:
        raise Stage3RegistryVerificationError(
            "resolution must target exactly UD-001 then UD-002"
        )
    if resolution["historical_stage2_register_mutated"] is not False:
        raise Stage3RegistryVerificationError(
            "historical Stage 2 register must remain unmodified"
        )
    if resolution["resolution_source"] != "Stage 3 sealed teacher registry":
        raise Stage3RegistryVerificationError(
            "wrong Stage 3 resolution source"
        )

    if registry_sha256 is not None:
        if resolution.get("registry_sha256") != registry_sha256:
            raise Stage3RegistryVerificationError(
                "resolution registry SHA-256 mismatch"
            )
    if table_sha256 is not None:
        if resolution.get("phase_selection_table_sha256") != table_sha256:
            raise Stage3RegistryVerificationError(
                "resolution phase-table SHA-256 mismatch"
            )

    stage2_status = "SKIPPED"
    if successor_root is not None:
        path = (
            Path(successor_root).resolve()
            / "followup/configs/stage2_unresolved_decisions_v1.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        decisions = data.get("decisions")
        if not isinstance(decisions, list):
            raise Stage3RegistryVerificationError(
                "Stage 2 decisions list missing"
            )

        found = {
            item.get("decision_id"): item
            for item in decisions
            if isinstance(item, Mapping)
            and item.get("decision_id") in {"UD-001", "UD-002"}
        }
        if set(found) != {"UD-001", "UD-002"}:
            raise Stage3RegistryVerificationError(
                "historical UD-001/UD-002 records missing"
            )

        for decision_id in ("UD-001", "UD-002"):
            record = found[decision_id]
            if record.get("status") != "unresolved":
                raise Stage3RegistryVerificationError(
                    f"historical {decision_id} status was rewritten"
                )

        stage2_status = "PASS"

    return {
        "resolution_linkage_status": "PASS",
        "historical_stage2_status": stage2_status,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
