from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cells import (
    StudentCellKey,
    StudentCellSummary,
    build_student_cell_summaries,
    extract_direct_teacher_values,
)
from .contrasts import (
    MissingnessRecord,
    PhaseContrast,
    TeacherStudentContrast,
    build_cross_method_budget_warnings,
    build_method_accounting,
    build_missingness_records,
    build_phase_contrasts,
    build_teacher_student_contrasts,
)
from .normalization import canonical_json_bytes, normalized_sha256
from .population import (
    PopulationSummary,
    build_phase_population_summaries,
    build_teacher_student_population_summaries,
)
from .profiles import TechnicalAnalysisProfile

OUTPUT_BUNDLE_SCHEMA_VERSION = "stage5d_output_bundle_v1"
OUTPUT_OBJECT_SCHEMA_VERSION = "stage5d_output_object_v1"
RECONSTRUCTION_MANIFEST_SCHEMA_VERSION = (
    "stage5d_reconstruction_manifest_v1"
)
OUTPUT_CLASSIFICATION = "synthetic_technical_only"
OUTPUT_OBJECT_IDS = (
    "direct_teacher_summaries",
    "hard_student_summaries",
    "soft_student_summaries",
    "dispersion_summaries",
    "phase_contrasts",
    "teacher_student_contrasts",
    "phase_population_summaries",
    "teacher_student_population_summaries",
    "method_budget_summaries",
    "failure_accounting",
    "unresolved_cell_accounting",
    "missingness_summaries",
)


class Stage5DOutputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _SourceIndex:
    cell_sources: Mapping[StudentCellKey, tuple[str, ...]]
    phase_sources: Mapping[str, tuple[str, ...]]
    teacher_student_sources: Mapping[str, tuple[str, ...]]


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _profile_mapping(profile: TechnicalAnalysisProfile) -> dict[str, Any]:
    return asdict(profile)


def technical_profile_sha256(profile: TechnicalAnalysisProfile) -> str:
    return _sha256(_profile_mapping(profile))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    return value


def _sorted_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    copied = [_json_value(row) for row in rows]
    return sorted(
        copied,
        key=lambda row: canonical_json_bytes(row),
    )


def _student_key(identity: Mapping[str, Any]) -> StudentCellKey:
    return StudentCellKey(
        teacher_seed=int(identity["teacher_seed"]),
        phase=str(identity["phase"]),
        distillation_condition=str(identity["distillation_condition"]),
        method_id=str(identity["method_id"]),
        endpoint_id=str(identity["endpoint_id"]),
        protocol_id=str(identity["protocol_id"]),
        fidelity_id=str(identity["fidelity_id"]),
        budget_id=str(identity["budget_id"]),
    )


def _source_index(
    normalized: Mapping[str, Any],
    cells: Sequence[StudentCellSummary],
    phase_contrasts: Sequence[PhaseContrast],
    teacher_student_contrasts: Sequence[TeacherStudentContrast],
) -> _SourceIndex:
    cell_sources: dict[StudentCellKey, set[str]] = defaultdict(set)
    eligibility_by_attempt: dict[str, str] = {}

    for raw in normalized["eligibility_records"]:
        eligibility_by_attempt[str(raw["attempt_id"])] = str(
            raw["eligibility_id"]
        )

    for raw in normalized["cell_expectations"]:
        cell_sources[_student_key(raw["identity"])].add(
            str(raw["cell_id"])
        )

    for raw in normalized["student_endpoints"]:
        key = _student_key(raw["identity"])
        attempt_id = str(raw["attempt_id"])
        cell_sources[key].update(
            {
                str(raw["record_id"]),
                attempt_id,
            }
        )
        eligibility_id = eligibility_by_attempt.get(attempt_id)
        if eligibility_id is not None:
            cell_sources[key].add(eligibility_id)

    frozen_cells = {
        key: tuple(sorted(source_ids))
        for key, source_ids in cell_sources.items()
    }

    phase_sources: dict[str, tuple[str, ...]] = {}
    for row in phase_contrasts:
        source_ids: set[str] = set()
        if row.left_key is not None:
            source_ids.update(frozen_cells.get(row.left_key, ()))
        if row.right_key is not None:
            source_ids.update(frozen_cells.get(row.right_key, ()))
        phase_sources[repr(row.key)] = tuple(sorted(source_ids))

    teacher_student_sources: dict[str, tuple[str, ...]] = {}
    for row in teacher_student_contrasts:
        key = StudentCellKey(
            teacher_seed=row.key.teacher_seed,
            phase=row.key.phase,
            distillation_condition=row.key.distillation_condition,
            method_id=row.key.method_id,
            endpoint_id=row.key.endpoint_id,
            protocol_id=row.key.protocol_id,
            fidelity_id=row.key.fidelity_id,
            budget_id=row.key.budget_id,
        )
        source_ids = set(frozen_cells.get(key, ()))
        if row.teacher_record_id is not None:
            source_ids.add(row.teacher_record_id)
        teacher_student_sources[repr(row.key)] = tuple(sorted(source_ids))

    return _SourceIndex(
        cell_sources=frozen_cells,
        phase_sources=phase_sources,
        teacher_student_sources=teacher_student_sources,
    )


def _row_with_sources(
    value: Any,
    source_ids: Sequence[str],
) -> dict[str, Any]:
    row = asdict(value)
    row["source_record_ids"] = sorted(set(source_ids))
    return row


def _output_object(
    object_id: str,
    *,
    row_unit: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if object_id not in OUTPUT_OBJECT_IDS:
        raise Stage5DOutputError(f"unsupported output object: {object_id}")

    ordered_rows = _sorted_rows(rows)
    source_ids = sorted(
        {
            str(source_id)
            for row in ordered_rows
            for source_id in row.get("source_record_ids", [])
        }
    )
    payload: dict[str, Any] = {
        "schema_version": OUTPUT_OBJECT_SCHEMA_VERSION,
        "object_id": object_id,
        "classification": OUTPUT_CLASSIFICATION,
        "row_unit": row_unit,
        "rows": ordered_rows,
        "source_record_ids": source_ids,
    }
    payload["sha256"] = _sha256(payload)
    return payload


def _validate_synthetic_boundary(
    normalized: Mapping[str, Any],
    profile: TechnicalAnalysisProfile,
) -> None:
    if normalized.get("classification") != OUTPUT_CLASSIFICATION:
        raise Stage5DOutputError("normalized input is not synthetic-only")
    if normalized.get("scientific_data") is not False:
        raise Stage5DOutputError("scientific input is forbidden")
    if normalized.get("production_eligible") is not False:
        raise Stage5DOutputError("production input is forbidden")
    if (
        not profile.synthetic_only
        or profile.scientific_data
        or profile.production_eligible
    ):
        raise Stage5DOutputError("technical profile violates synthetic firewall")
    if profile.resolves_decisions:
        raise Stage5DOutputError("technical profile may not resolve UD items")


def _cell_rows(
    cells: Sequence[StudentCellSummary],
    source_index: _SourceIndex,
    condition: str,
) -> list[dict[str, Any]]:
    return [
        _row_with_sources(
            cell,
            source_index.cell_sources.get(cell.key, ()),
        )
        for cell in cells
        if cell.key.distillation_condition == condition
    ]


def _dispersion_rows(
    cells: Sequence[StudentCellSummary],
    source_index: _SourceIndex,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell.state != "defined":
            continue
        if cell.range_value is None or cell.mad_value is None:
            raise Stage5DOutputError(
                "defined cell requires explicit dispersion values"
            )
        rows.append(
            {
                "key": asdict(cell.key),
                "state": "defined",
                "member_unit": "student_initialization",
                "population_unit": "teacher_seed",
                "number_student_realizations": len(cell.member_values),
                "range_value": float(cell.range_value),
                "mad_value": float(cell.mad_value),
                "source_record_ids": list(
                    source_index.cell_sources.get(cell.key, ())
                ),
            }
        )
    return rows


def _population_rows(
    summaries: Sequence[PopulationSummary],
    source_map: Mapping[str, tuple[str, ...]],
    contrasts: Sequence[PhaseContrast | TeacherStudentContrast],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        identity = asdict(summary.metric_identity)
        matching_sources: set[str] = set()
        for contrast in contrasts:
            contrast_identity = asdict(contrast.key)
            if all(
                contrast_identity.get(field) == value
                for field, value in identity.items()
            ):
                matching_sources.update(
                    source_map.get(repr(contrast.key), ())
                )
        rows.append(_row_with_sources(summary, matching_sources))
    return rows


def _phase_contrast_row(
    row: PhaseContrast,
    source_ids: Sequence[str],
) -> dict[str, Any]:
    output = _row_with_sources(row, source_ids)
    if row.state != "defined":
        output["left_value"] = None
        output["right_value"] = None
        output["delta"] = None
    return output


def _teacher_student_contrast_row(
    row: TeacherStudentContrast,
    source_ids: Sequence[str],
) -> dict[str, Any]:
    output = _row_with_sources(row, source_ids)
    if row.state != "defined":
        output["student_summary_value"] = None
        output["teacher_value"] = None
        output["delta"] = None
    return output


def _failure_rows(normalized: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in normalized["student_attempts"]:
        if attempt["outcome"] == "failed":
            rows.append(
                {
                    "failure_scope": "student_attempt",
                    "source_record_id": str(attempt["attempt_id"]),
                    "reason": str(attempt["failure_reason"]),
                    "source_record_ids": [str(attempt["attempt_id"])],
                }
            )
    for collection, scope in (
        ("direct_teacher_endpoints", "direct_teacher_endpoint"),
        ("student_endpoints", "student_endpoint"),
    ):
        for endpoint in normalized[collection]:
            if endpoint["state"] == "failed":
                rows.append(
                    {
                        "failure_scope": scope,
                        "source_record_id": str(endpoint["record_id"]),
                        "reason": "endpoint_state_failed",
                        "source_record_ids": [str(endpoint["record_id"])],
                    }
                )
    return rows


def _unresolved_cell_rows(
    cells: Sequence[StudentCellSummary],
    source_index: _SourceIndex,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell.state != "unresolved":
            continue
        kind = (
            "insufficient"
            if cell.reason is not None
            and cell.reason.startswith("insufficient_eligible_students:")
            else "unresolved"
        )
        rows.append(
            {
                "key": asdict(cell.key),
                "missingness_kind": kind,
                "reason": cell.reason,
                "member_unit": "student_initialization",
                "population_unit": "teacher_seed",
                "source_record_ids": list(
                    source_index.cell_sources.get(cell.key, ())
                ),
            }
        )
    return rows


def _missingness_sources(
    record: MissingnessRecord,
    source_index: _SourceIndex,
    cells: Sequence[StudentCellSummary],
) -> tuple[str, ...]:
    if record.scope in {"student_attempt", "student_eligibility"}:
        return (record.identity,)
    if record.scope == "student_cell":
        cell = next(
            (row for row in cells if repr(row.key) == record.identity),
            None,
        )
        return (
            ()
            if cell is None
            else source_index.cell_sources.get(cell.key, ())
        )
    if record.scope == "phase_contrast":
        return source_index.phase_sources.get(record.identity, ())
    if record.scope == "teacher_student_contrast":
        return source_index.teacher_student_sources.get(record.identity, ())
    raise Stage5DOutputError(
        f"unsupported missingness scope: {record.scope}"
    )


def _missingness_summary_rows(
    records: Sequence[MissingnessRecord],
    source_index: _SourceIndex,
    cells: Sequence[StudentCellSummary],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[MissingnessRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.kind, record.scope)].append(record)

    rows: list[dict[str, Any]] = []
    for (kind, scope), group in sorted(grouped.items()):
        source_ids = {
            source_id
            for record in group
            for source_id in _missingness_sources(
                record,
                source_index,
                cells,
            )
        }
        rows.append(
            {
                "missingness_kind": kind,
                "scope": scope,
                "count": len(group),
                "identities": sorted(record.identity for record in group),
                "source_record_ids": sorted(source_ids),
            }
        )
    return rows


def build_stage5d_output_bundle(
    normalized: Mapping[str, Any],
    profile: TechnicalAnalysisProfile,
) -> dict[str, Any]:
    _validate_synthetic_boundary(normalized, profile)

    cells = build_student_cell_summaries(normalized, profile)
    teachers = extract_direct_teacher_values(normalized)
    phase_contrasts = build_phase_contrasts(cells, profile)
    teacher_student_contrasts = build_teacher_student_contrasts(
        cells,
        teachers,
    )
    phase_population = build_phase_population_summaries(
        phase_contrasts,
        profile,
    )
    teacher_student_population = (
        build_teacher_student_population_summaries(
            teacher_student_contrasts,
            profile,
        )
    )
    source_index = _source_index(
        normalized,
        cells,
        phase_contrasts,
        teacher_student_contrasts,
    )
    missingness = build_missingness_records(
        normalized,
        cells,
        phase_contrasts,
        teacher_student_contrasts,
    )

    objects = {
        "direct_teacher_summaries": _output_object(
            "direct_teacher_summaries",
            row_unit="teacher_seed_endpoint",
            rows=[
                _row_with_sources(teacher, (teacher.record_id,))
                for teacher in teachers
            ],
        ),
        "hard_student_summaries": _output_object(
            "hard_student_summaries",
            row_unit="teacher_seed_student_cell",
            rows=_cell_rows(cells, source_index, "hard"),
        ),
        "soft_student_summaries": _output_object(
            "soft_student_summaries",
            row_unit="teacher_seed_student_cell",
            rows=_cell_rows(cells, source_index, "soft"),
        ),
        "dispersion_summaries": _output_object(
            "dispersion_summaries",
            row_unit="teacher_seed_student_cell",
            rows=_dispersion_rows(cells, source_index),
        ),
        "phase_contrasts": _output_object(
            "phase_contrasts",
            row_unit="teacher_seed_contrast",
            rows=[
                _phase_contrast_row(
                    row,
                    source_index.phase_sources.get(repr(row.key), ()),
                )
                for row in phase_contrasts
            ],
        ),
        "teacher_student_contrasts": _output_object(
            "teacher_student_contrasts",
            row_unit="teacher_seed_contrast",
            rows=[
                _teacher_student_contrast_row(
                    row,
                    source_index.teacher_student_sources.get(
                        repr(row.key),
                        (),
                    ),
                )
                for row in teacher_student_contrasts
            ],
        ),
        "phase_population_summaries": _output_object(
            "phase_population_summaries",
            row_unit="teacher_seed_population",
            rows=_population_rows(
                phase_population,
                source_index.phase_sources,
                phase_contrasts,
            ),
        ),
        "teacher_student_population_summaries": _output_object(
            "teacher_student_population_summaries",
            row_unit="teacher_seed_population",
            rows=_population_rows(
                teacher_student_population,
                source_index.teacher_student_sources,
                teacher_student_contrasts,
            ),
        ),
        "method_budget_summaries": _output_object(
            "method_budget_summaries",
            row_unit="method_condition",
            rows=[
                {
                    "row_kind": "method_accounting",
                    **asdict(row),
                    "source_record_ids": [row.budget_id],
                }
                for row in build_method_accounting(normalized, cells)
            ]
            + [
                {
                    "row_kind": "cross_method_budget_warning",
                    **asdict(row),
                    "source_record_ids": sorted(
                        {row.left_budget_id, row.right_budget_id}
                    ),
                }
                for row in build_cross_method_budget_warnings(normalized)
            ],
        ),
        "failure_accounting": _output_object(
            "failure_accounting",
            row_unit="failed_synthetic_record",
            rows=_failure_rows(normalized),
        ),
        "unresolved_cell_accounting": _output_object(
            "unresolved_cell_accounting",
            row_unit="unresolved_teacher_seed_student_cell",
            rows=_unresolved_cell_rows(cells, source_index),
        ),
        "missingness_summaries": _output_object(
            "missingness_summaries",
            row_unit="missingness_kind_scope",
            rows=_missingness_summary_rows(
                missingness,
                source_index,
                cells,
            ),
        ),
    }

    object_manifest = [
        {
            "object_id": object_id,
            "schema_version": output["schema_version"],
            "object_sha256": output["sha256"],
            "source_record_ids": output["source_record_ids"],
        }
        for object_id, output in sorted(objects.items())
    ]
    profile_hash = technical_profile_sha256(profile)
    manifest: dict[str, Any] = {
        "schema_version": RECONSTRUCTION_MANIFEST_SCHEMA_VERSION,
        "classification": OUTPUT_CLASSIFICATION,
        "synthetic_only": True,
        "scientific_data": False,
        "production_eligible": False,
        "normalized_input": {
            "schema_version": normalized["schema_version"],
            "sha256": normalized_sha256(normalized),
            "source_provenance": normalized["source_provenance"],
        },
        "technical_profile": {
            "schema_version": profile.schema_version,
            "profile_id": profile.profile_id,
            "sha256": profile_hash,
        },
        "reducer_configuration": {
            "cell_reducer": profile.settings.cell_reducer,
            "minimum_eligible_students": (
                profile.settings.minimum_eligible_students
            ),
            "phase_pairs": [list(pair) for pair in profile.settings.phase_pairs],
            "population_reducer": profile.settings.population_reducer,
            "student_member_unit": "student_initialization",
            "population_unit": "teacher_seed",
        },
        "unresolved_decision_dependencies": sorted(
            profile.decision_dependencies
        ),
        "resolved_decisions": [],
        "output_objects": object_manifest,
    }
    manifest["sha256"] = _sha256(manifest)

    bundle: dict[str, Any] = {
        "schema_version": OUTPUT_BUNDLE_SCHEMA_VERSION,
        "classification": OUTPUT_CLASSIFICATION,
        "scientific_data": False,
        "production_eligible": False,
        "output_objects": objects,
        "reconstruction_manifest": manifest,
    }
    bundle["sha256"] = _sha256(bundle)
    return bundle


def reconstruct_stage5d_output_bundle(
    normalized: Mapping[str, Any],
    profile: TechnicalAnalysisProfile,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rebuilt = build_stage5d_output_bundle(normalized, profile)
    expected = rebuilt["reconstruction_manifest"]
    if manifest != expected:
        raise Stage5DOutputError(
            "reconstruction manifest does not match source IDs and hashes"
        )
    return rebuilt


def validate_stage5d_output_bundle(
    bundle: Mapping[str, Any],
    normalized: Mapping[str, Any],
    profile: TechnicalAnalysisProfile,
) -> None:
    manifest = bundle.get("reconstruction_manifest")
    if not isinstance(manifest, Mapping):
        raise Stage5DOutputError("output bundle lacks reconstruction manifest")
    rebuilt = reconstruct_stage5d_output_bundle(
        normalized,
        profile,
        manifest,
    )
    if bundle != rebuilt:
        raise Stage5DOutputError("output bundle content or hash mismatch")


def _temporary_output_root(output_root: str | Path) -> Path:
    root = Path(output_root).resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    try:
        root.relative_to(temporary_root)
    except ValueError as exc:
        raise Stage5DOutputError(
            "explicit output root must be inside the system temporary directory"
        ) from exc
    if root == temporary_root:
        raise Stage5DOutputError(
            "explicit output root must be a dedicated temporary subdirectory"
        )
    return root


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_stage5d_output_bundle(
    bundle: Mapping[str, Any],
    output_root: str | Path,
) -> tuple[Path, Path]:
    root = _temporary_output_root(output_root)
    bundle_path = root / "stage5d_output_bundle.json"
    manifest_path = root / "stage5d_reconstruction_manifest.json"
    manifest = bundle.get("reconstruction_manifest")
    if not isinstance(manifest, Mapping):
        raise Stage5DOutputError("cannot write bundle without manifest")

    _atomic_json_write(bundle_path, bundle)
    _atomic_json_write(manifest_path, manifest)
    return bundle_path, manifest_path


def load_stage5d_output_bundle(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise Stage5DOutputError("output bundle root must be an object")
    return raw


def assert_deterministic_numeric_values(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise Stage5DOutputError("non-finite numeric output is forbidden")
    if isinstance(value, Mapping):
        for child in value.values():
            assert_deterministic_numeric_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_deterministic_numeric_values(child)
