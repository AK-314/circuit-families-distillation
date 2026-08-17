"""Read-only validation of Stage 3 canonical teacher physical inputs.

This module performs provenance/input validation only. It does not select phases,
run models, inspect circuit results, or write scientific outputs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from circuit_families.training.checkpoints import file_sha256

STAGE1_LINK = Path("followup/manifests/predecessor_link_v1.json")
EXPECTED_EVAL_INTERVAL = 50
EXPECTED_CHECKPOINT_INTERVAL = 50
EXPECTED_MAX_STEP = 40_000
EXPECTED_STEPS = tuple(range(0, EXPECTED_MAX_STEP + 1, EXPECTED_CHECKPOINT_INTERVAL))

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_STEP_FILE = re.compile(r"^step_(\d+)\.pt$")


class Stage3InputError(ValueError):
    """Raised when canonical Stage 3 teacher input provenance is invalid."""


@dataclass(frozen=True)
class CanonicalTeacher:
    seed: int
    run_id: str
    manifest_path: str


CANONICAL_TEACHERS = (
    CanonicalTeacher(
        0,
        "stage18-main-training-s0-58b8c1235464",
        "manifests/training_stage18-main-training-s0-58b8c1235464.json",
    ),
    CanonicalTeacher(
        1,
        "modular-addition-training-s1-5f1bc9dee7ab",
        "manifests/training_modular-addition-training-s1-5f1bc9dee7ab.json",
    ),
    CanonicalTeacher(
        2,
        "stage18-main-training-s2-c70f62c0fa7c",
        "manifests/training_stage18-main-training-s2-c70f62c0fa7c.json",
    ),
    CanonicalTeacher(
        3,
        "stage18-main-training-s3-4c0c7c63ce2f",
        "manifests/training_stage18-main-training-s3-4c0c7c63ce2f.json",
    ),
    CanonicalTeacher(
        4,
        "stage18-main-training-s4-c2881c226349",
        "manifests/training_stage18-main-training-s4-c2881c226349.json",
    ),
)


@dataclass(frozen=True)
class LinkedTeacher:
    teacher: CanonicalTeacher
    manifest_sha256: str


@dataclass(frozen=True)
class ValidatedTeacherInput:
    seed: int
    run_id: str
    manifest_path: str
    manifest_sha256: str
    metrics_path: str
    metrics_sha256: str
    metrics_row_count: int
    first_step: int
    last_step: int
    checkpoint_directory: str
    checkpoint_count: int
    evaluation_interval: int
    checkpoint_interval: int


def _walk(node: Any) -> Iterable[Any]:
    yield node
    if isinstance(node, Mapping):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for value in node:
            yield from _walk(value)


def _values_for_key(node: Any, key: str) -> list[Any]:
    values: list[Any] = []
    for value in _walk(node):
        if isinstance(value, Mapping) and key in value:
            values.append(value[key])
    return values


def _strings(node: Any) -> list[str]:
    return [value for value in _walk(node) if isinstance(value, str)]


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Stage3InputError(f"{label} is not a lowercase SHA-256 hex digest")
    return value


def _portable_relative_path(value: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise Stage3InputError(f"{label} must be a non-empty relative POSIX path")
    if value.startswith("/") or value.startswith("~") or "\\" in value:
        raise Stage3InputError(f"{label} is not portable: {value!r}")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise Stage3InputError(f"{label} escapes its declared root: {value!r}")
    return path


def _resolve_under(root: Path, relative: str, label: str) -> Path:
    portable = _portable_relative_path(relative, label)
    root_resolved = root.resolve()
    path = (root_resolved / Path(*portable.parts)).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise Stage3InputError(f"{label} escapes predecessor root: {relative!r}") from exc
    return path


def _path_hash_pairs(node: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    if isinstance(node, Mapping):
        path_value = node.get("path")
        sha_value = node.get("sha256")
        if isinstance(path_value, str) and isinstance(sha_value, str):
            if _HEX64.fullmatch(sha_value):
                pairs.append((path_value, sha_value))

        for key, value in node.items():
            if not isinstance(value, str):
                continue
            if key.endswith("_path"):
                prefix = key[:-5]
                sibling = node.get(f"{prefix}_sha256")
                if isinstance(sibling, str) and _HEX64.fullmatch(sibling):
                    pairs.append((value, sibling))
                generic = node.get("sha256")
                if isinstance(generic, str) and _HEX64.fullmatch(generic):
                    pairs.append((value, generic))

        for value in node.values():
            pairs.extend(_path_hash_pairs(value))

    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for value in node:
            pairs.extend(_path_hash_pairs(value))

    return pairs


def _find_teacher_runs(node: Any) -> list[Mapping[str, Any]]:
    candidates: list[list[Mapping[str, Any]]] = []
    for value in _walk(node):
        if isinstance(value, Mapping) and "teacher_runs" in value:
            raw = value["teacher_runs"]
            if isinstance(raw, list) and all(isinstance(x, Mapping) for x in raw):
                candidates.append(raw)

    if len(candidates) != 1:
        raise Stage3InputError(
            f"expected exactly one Stage 1 teacher_runs roster, found {len(candidates)}"
        )
    return candidates[0]


def _entry_seed(entry: Mapping[str, Any]) -> int:
    for key in ("teacher_seed", "model_seed", "seed"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    raise Stage3InputError("teacher roster entry has no integer seed identity")


def _entry_run_id(entry: Mapping[str, Any]) -> str:
    value = entry.get("run_id")
    if isinstance(value, str):
        return value

    values = [x for x in _values_for_key(entry, "run_id") if isinstance(x, str)]
    unique = sorted(set(values))
    if len(unique) != 1:
        raise Stage3InputError(f"teacher roster run_id is ambiguous: {unique}")
    return unique[0]


def _entry_manifest_path(entry: Mapping[str, Any]) -> str:
    candidates = sorted(
        {
            value
            for value in _strings(entry)
            if value.startswith("manifests/training_") and value.endswith(".json")
        }
    )
    if len(candidates) != 1:
        raise Stage3InputError(f"teacher manifest path is ambiguous: {candidates}")
    return candidates[0]


def _declared_hash_for_path(node: Any, path: str, label: str) -> str:
    hashes = sorted(
        {
            sha
            for declared_path, sha in _path_hash_pairs(node)
            if declared_path == path
        }
    )
    if len(hashes) != 1:
        raise Stage3InputError(
            f"{label}: expected exactly one declared SHA-256 for {path!r}, found {hashes}"
        )
    return _validate_sha256(hashes[0], label)


def load_stage1_canonical_roster(successor_root: str | Path) -> tuple[LinkedTeacher, ...]:
    successor = Path(successor_root).resolve()
    link_path = successor / STAGE1_LINK
    if not link_path.is_file():
        raise Stage3InputError(f"Stage 1 predecessor link missing: {link_path}")

    try:
        link = json.loads(link_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage3InputError(f"cannot parse Stage 1 predecessor link: {exc}") from exc

    entries = _find_teacher_runs(link)
    if len(entries) != len(CANONICAL_TEACHERS):
        raise Stage3InputError(
            f"Stage 1 teacher roster count {len(entries)} != expected 5"
        )

    linked: list[LinkedTeacher] = []

    for index, (entry, expected) in enumerate(zip(entries, CANONICAL_TEACHERS, strict=True)):
        seed = _entry_seed(entry)
        run_id = _entry_run_id(entry)
        manifest_path = _entry_manifest_path(entry)

        if seed != expected.seed:
            raise Stage3InputError(
                f"teacher roster index {index}: seed {seed} != expected {expected.seed}"
            )
        if run_id != expected.run_id:
            raise Stage3InputError(
                f"seed {expected.seed}: run_id {run_id!r} != canonical {expected.run_id!r}"
            )
        if manifest_path != expected.manifest_path:
            raise Stage3InputError(
                f"seed {expected.seed}: manifest {manifest_path!r} "
                f"!= canonical {expected.manifest_path!r}"
            )

        manifest_sha256 = _declared_hash_for_path(
            entry,
            manifest_path,
            f"seed {expected.seed} training manifest",
        )
        linked.append(LinkedTeacher(expected, manifest_sha256))

    return tuple(linked)


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage3InputError(f"{label} malformed: {exc}") from exc
    if not isinstance(value, Mapping):
        raise Stage3InputError(f"{label} must contain a JSON object")
    return value


def _load_metrics(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Stage3InputError(
                        f"metrics JSONL has blank row at line {line_number}"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise Stage3InputError(
                        f"metrics JSONL malformed at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise Stage3InputError(
                        f"metrics JSONL line {line_number} is not an object"
                    )
                rows.append(row)
    except OSError as exc:
        raise Stage3InputError(f"cannot read metrics JSONL: {exc}") from exc
    return rows


def _manifest_metrics_identity(
    manifest: Mapping[str, Any],
    expected_run_id: str,
) -> tuple[str, str]:
    expected_path = f"results/raw/{expected_run_id}/metrics.jsonl"

    output_paths = manifest.get("output_paths")
    if not isinstance(output_paths, Mapping):
        raise Stage3InputError("manifest output_paths must be an object")

    hashes = manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        raise Stage3InputError("manifest hashes must be an object")

    path = output_paths.get("metrics_jsonl")
    if not isinstance(path, str):
        raise Stage3InputError(
            "manifest output_paths.metrics_jsonl must be a string"
        )

    if path != expected_path:
        raise Stage3InputError(
            f"wrong metrics output path {path!r}; expected {expected_path!r}"
        )

    sha = _validate_sha256(
        hashes.get("metrics_jsonl_sha256"),
        "metrics SHA-256",
    )

    return path, sha


def _validate_manifest_identity(
    manifest: Mapping[str, Any],
    expected: CanonicalTeacher,
) -> None:
    run_ids = {
        value
        for value in _values_for_key(manifest, "run_id")
        if isinstance(value, str)
    }
    if not run_ids or expected.run_id not in run_ids:
        raise Stage3InputError(
            f"manifest does not identify canonical run {expected.run_id!r}"
        )

    model_seeds = {
        value
        for value in _values_for_key(manifest, "model_seed")
        if isinstance(value, int) and not isinstance(value, bool)
    }
    if model_seeds and model_seeds != {expected.seed}:
        raise Stage3InputError(
            f"manifest model_seed identities {sorted(model_seeds)} "
            f"!= canonical seed {expected.seed}"
        )

    eval_intervals = {
        value
        for value in _values_for_key(manifest, "evaluation_interval")
        if isinstance(value, int) and not isinstance(value, bool)
    }
    if eval_intervals and eval_intervals != {EXPECTED_EVAL_INTERVAL}:
        raise Stage3InputError(
            f"evaluation intervals {sorted(eval_intervals)} "
            f"!= expected {EXPECTED_EVAL_INTERVAL}"
        )

    checkpoint_intervals = {
        value
        for value in _values_for_key(manifest, "checkpoint_interval")
        if isinstance(value, int) and not isinstance(value, bool)
    }
    if checkpoint_intervals and checkpoint_intervals != {EXPECTED_CHECKPOINT_INTERVAL}:
        raise Stage3InputError(
            f"checkpoint intervals {sorted(checkpoint_intervals)} "
            f"!= expected {EXPECTED_CHECKPOINT_INTERVAL}"
        )

    output_paths = manifest.get("output_paths")
    if not isinstance(output_paths, Mapping):
        raise Stage3InputError("manifest output_paths must be an object")

    expected_checkpoint_directory = f"checkpoints/{expected.run_id}"
    observed_checkpoint_directory = output_paths.get("checkpoint_directory")
    if observed_checkpoint_directory != expected_checkpoint_directory:
        raise Stage3InputError(
            f"wrong checkpoint directory {observed_checkpoint_directory!r}; "
            f"expected {expected_checkpoint_directory!r}"
        )


def _numeric_field(row: Mapping[str, Any], field: str, line_number: int) -> float:
    value = row.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Stage3InputError(
            f"metrics line {line_number}: {field} must be numeric"
        )
    return float(value)


def validate_teacher_input(
    linked: LinkedTeacher,
    predecessor_root: str | Path,
    *,
    verify_checkpoint_hashes: bool = True,
) -> ValidatedTeacherInput:
    predecessor = Path(predecessor_root).resolve()
    expected = linked.teacher

    manifest_physical = _resolve_under(
        predecessor,
        expected.manifest_path,
        f"seed {expected.seed} manifest path",
    )
    if not manifest_physical.is_file():
        raise Stage3InputError(
            f"seed {expected.seed}: canonical training manifest missing"
        )

    manifest_actual_sha = file_sha256(manifest_physical)
    if manifest_actual_sha != linked.manifest_sha256:
        raise Stage3InputError(
            f"seed {expected.seed}: training manifest SHA-256 mismatch"
        )

    manifest = _load_json(
        manifest_physical,
        f"seed {expected.seed} training manifest",
    )
    _validate_manifest_identity(manifest, expected)

    metrics_rel, metrics_expected_sha = _manifest_metrics_identity(
        manifest,
        expected.run_id,
    )
    metrics_physical = _resolve_under(
        predecessor,
        metrics_rel,
        f"seed {expected.seed} metrics path",
    )
    if not metrics_physical.is_file():
        raise Stage3InputError(f"seed {expected.seed}: canonical metrics file missing")

    metrics_actual_sha = file_sha256(metrics_physical)
    if metrics_actual_sha != metrics_expected_sha:
        raise Stage3InputError(
            f"seed {expected.seed}: metrics SHA-256 mismatch"
        )

    rows = _load_metrics(metrics_physical)
    if len(rows) != len(EXPECTED_STEPS):
        raise Stage3InputError(
            f"seed {expected.seed}: metrics row count {len(rows)} "
            f"!= expected {len(EXPECTED_STEPS)}"
        )

    observed_steps: list[int] = []
    checkpoint_paths: list[Path] = []
    checkpoint_relpaths: list[str] = []

    for line_number, row in enumerate(rows, start=1):
        run_id = row.get("run_id")
        if run_id != expected.run_id:
            raise Stage3InputError(
                f"seed {expected.seed} metrics line {line_number}: "
                f"run_id {run_id!r} != {expected.run_id!r}"
            )

        step = row.get("training_step")
        if not isinstance(step, int) or isinstance(step, bool):
            raise Stage3InputError(
                f"seed {expected.seed} metrics line {line_number}: invalid training_step"
            )
        observed_steps.append(step)

        for field in (
            "train_accuracy",
            "test_accuracy",
            "train_loss",
            "test_loss",
        ):
            _numeric_field(row, field, line_number)

        checkpoint_rel = row.get("checkpoint_path")
        if not isinstance(checkpoint_rel, str):
            raise Stage3InputError(
                f"seed {expected.seed} metrics line {line_number}: "
                "checkpoint_path missing"
            )

        portable = _portable_relative_path(
            checkpoint_rel,
            f"seed {expected.seed} checkpoint path",
        )
        expected_checkpoint_parent = (
            PurePosixPath("checkpoints") / expected.run_id
        )
        if portable.parent != expected_checkpoint_parent:
            raise Stage3InputError(
                f"seed {expected.seed}: checkpoint path parent {portable.parent!s} "
                f"!= canonical {expected_checkpoint_parent!s}"
            )

        match = _STEP_FILE.fullmatch(portable.name)
        if match is None or int(match.group(1)) != step:
            raise Stage3InputError(
                f"seed {expected.seed}: checkpoint filename/step mismatch at step {step}"
            )

        checkpoint_path = _resolve_under(
            predecessor,
            checkpoint_rel,
            f"seed {expected.seed} checkpoint path",
        )
        if not checkpoint_path.is_file():
            raise Stage3InputError(
                f"seed {expected.seed}: missing checkpoint for step {step}"
            )

        checkpoint_sha = _validate_sha256(
            row.get("checkpoint_sha256"),
            f"seed {expected.seed} step {step} checkpoint SHA-256",
        )

        if verify_checkpoint_hashes and file_sha256(checkpoint_path) != checkpoint_sha:
            raise Stage3InputError(
                f"seed {expected.seed}: checkpoint SHA-256 mismatch at step {step}"
            )

        checkpoint_paths.append(checkpoint_path)
        checkpoint_relpaths.append(checkpoint_rel)

    if len(observed_steps) != len(set(observed_steps)):
        raise Stage3InputError(f"seed {expected.seed}: duplicate metrics steps")

    if tuple(observed_steps) != tuple(sorted(observed_steps)):
        raise Stage3InputError(f"seed {expected.seed}: metrics steps are unsorted")

    if tuple(observed_steps) != EXPECTED_STEPS:
        raise Stage3InputError(
            f"seed {expected.seed}: saved-step grid differs from expected "
            f"0..{EXPECTED_MAX_STEP} by {EXPECTED_CHECKPOINT_INTERVAL}"
        )

    if len(checkpoint_relpaths) != len(set(checkpoint_relpaths)):
        raise Stage3InputError(f"seed {expected.seed}: duplicate checkpoint paths")

    parents = {path.parent for path in checkpoint_paths}
    if len(parents) != 1:
        raise Stage3InputError(
            f"seed {expected.seed}: checkpoints span multiple physical directories"
        )
    checkpoint_dir = next(iter(parents))

    physical_files = sorted(checkpoint_dir.glob("step_*.pt"))
    physical_steps: list[int] = []
    for path in physical_files:
        match = _STEP_FILE.fullmatch(path.name)
        if match is None:
            continue
        physical_steps.append(int(match.group(1)))

    if len(physical_steps) != len(set(physical_steps)):
        raise Stage3InputError(
            f"seed {expected.seed}: duplicate physical checkpoint steps"
        )
    if tuple(sorted(physical_steps)) != EXPECTED_STEPS:
        raise Stage3InputError(
            f"seed {expected.seed}: physical checkpoint grid differs from metrics grid"
        )

    checkpoint_dir_rel = checkpoint_dir.resolve().relative_to(predecessor).as_posix()

    return ValidatedTeacherInput(
        seed=expected.seed,
        run_id=expected.run_id,
        manifest_path=expected.manifest_path,
        manifest_sha256=manifest_actual_sha,
        metrics_path=metrics_rel,
        metrics_sha256=metrics_actual_sha,
        metrics_row_count=len(rows),
        first_step=observed_steps[0],
        last_step=observed_steps[-1],
        checkpoint_directory=checkpoint_dir_rel,
        checkpoint_count=len(physical_steps),
        evaluation_interval=EXPECTED_EVAL_INTERVAL,
        checkpoint_interval=EXPECTED_CHECKPOINT_INTERVAL,
    )


def validate_all_teacher_inputs(
    successor_root: str | Path,
    predecessor_root: str | Path,
    *,
    verify_checkpoint_hashes: bool = True,
) -> tuple[ValidatedTeacherInput, ...]:
    linked = load_stage1_canonical_roster(successor_root)
    return tuple(
        validate_teacher_input(
            item,
            predecessor_root,
            verify_checkpoint_hashes=verify_checkpoint_hashes,
        )
        for item in linked
    )
