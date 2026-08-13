"""Strict validation for the Stage 1 predecessor-link manifest."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from circuit_families.config import mapping_hash
from circuit_families.followup_namespace import NAMESPACE_VERSION

SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_TEACHER_SEEDS = (0, 1, 2, 3, 4)


class PredecessorLinkError(ValueError):
    """Raised when a predecessor-link record violates the Stage 1 contract."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PredecessorLinkError(f"{field} must be an object.")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PredecessorLinkError(f"{field} must be an array.")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    field: str,
    required: set[str],
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)

    if missing:
        raise PredecessorLinkError(
            f"{field} is missing required fields: {', '.join(missing)}."
        )
    if unknown:
        raise PredecessorLinkError(
            f"{field} contains unknown fields: {', '.join(unknown)}."
        )


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredecessorLinkError(f"{field} must be a non-empty string.")
    return value


def _require_sha256(value: Any, field: str) -> str:
    text = _require_string(value, field)
    if not _SHA256.fullmatch(text):
        raise PredecessorLinkError(
            f"{field} must be a lowercase SHA-256 hex digest."
        )
    return text


def _require_commit(value: Any, field: str) -> str:
    text = _require_string(value, field)
    if not _GIT_COMMIT.fullmatch(text):
        raise PredecessorLinkError(
            f"{field} must be a lowercase 40-character Git commit."
        )
    return text


def _require_portable_path(value: Any, field: str) -> str:
    text = _require_string(value, field)
    path = PurePosixPath(text)

    if path.is_absolute():
        raise PredecessorLinkError(
            f"{field} must be a portable repository-relative path, "
            f"not an absolute private path: {text}"
        )

    if re.match(r"^[A-Za-z]:[\\/]", text):
        raise PredecessorLinkError(
            f"{field} must be portable and must not use an absolute "
            f"machine-specific path: {text}"
        )

    if ".." in path.parts:
        raise PredecessorLinkError(
            f"{field} must not contain parent traversal: {text}"
        )

    return path.as_posix()


def _validate_path_hash(value: Any, field: str) -> None:
    record = _mapping(value, field)
    _strict_keys(
        record,
        field=field,
        required={"path", "sha256"},
    )
    _require_portable_path(record["path"], f"{field}.path")
    _require_sha256(record["sha256"], f"{field}.sha256")


def _validate_config_identity(value: Any, field: str) -> None:
    record = _mapping(value, field)
    _strict_keys(
        record,
        field=field,
        required={"path", "file_sha256", "mapping_sha256"},
    )
    _require_portable_path(record["path"], f"{field}.path")
    _require_sha256(record["file_sha256"], f"{field}.file_sha256")
    _require_sha256(record["mapping_sha256"], f"{field}.mapping_sha256")


def validate_predecessor_link(record: Mapping[str, Any]) -> None:
    """Validate one complete Stage 1 predecessor-link mapping."""

    root = _mapping(record, "predecessor_link")
    _strict_keys(
        root,
        field="predecessor_link",
        required={
            "schema_version",
            "namespace_version",
            "predecessor",
            "successor_snapshot",
            "dataset",
            "architecture",
            "component_basis",
            "teacher_runs",
            "stage3_checkpoint_registry",
            "prior_results_visibility",
            "metadata",
        },
    )

    if root["schema_version"] != SCHEMA_VERSION:
        raise PredecessorLinkError(
            f"Unsupported predecessor-link schema version: "
            f"{root['schema_version']!r}."
        )

    if root["namespace_version"] != NAMESPACE_VERSION:
        raise PredecessorLinkError(
            "predecessor_link.namespace_version must equal "
            f"{NAMESPACE_VERSION!r}."
        )

    predecessor = _mapping(root["predecessor"], "predecessor")
    _strict_keys(
        predecessor,
        field="predecessor",
        required={
            "repository",
            "analysis_freeze_commit",
            "protocol",
            "implementation_order",
            "analysis_freeze_manifest",
        },
    )
    _require_string(predecessor["repository"], "predecessor.repository")
    _require_commit(
        predecessor["analysis_freeze_commit"],
        "predecessor.analysis_freeze_commit",
    )
    _validate_path_hash(predecessor["protocol"], "predecessor.protocol")
    _validate_path_hash(
        predecessor["implementation_order"],
        "predecessor.implementation_order",
    )
    _validate_path_hash(
        predecessor["analysis_freeze_manifest"],
        "predecessor.analysis_freeze_manifest",
    )

    successor = _mapping(root["successor_snapshot"], "successor_snapshot")
    _strict_keys(
        successor,
        field="successor_snapshot",
        required={
            "repository",
            "initial_commit",
            "relationship",
            "overlapping_file_count",
            "byte_identical_overlapping_file_count",
            "changed_overlapping_file_count",
        },
    )
    _require_string(successor["repository"], "successor_snapshot.repository")
    _require_commit(
        successor["initial_commit"],
        "successor_snapshot.initial_commit",
    )
    if (
        successor["relationship"]
        != "clean_source_snapshot_without_predecessor_scientific_outputs"
    ):
        raise PredecessorLinkError(
            "successor_snapshot.relationship is inconsistent with the "
            "Stage 1 preservation model."
        )

    for key in (
        "overlapping_file_count",
        "byte_identical_overlapping_file_count",
        "changed_overlapping_file_count",
    ):
        value = successor[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PredecessorLinkError(
                f"successor_snapshot.{key} must be a non-negative integer."
            )

    if (
        successor["byte_identical_overlapping_file_count"]
        + successor["changed_overlapping_file_count"]
        != successor["overlapping_file_count"]
    ):
        raise PredecessorLinkError(
            "successor_snapshot overlapping-file counts are contradictory."
        )

    dataset = _mapping(root["dataset"], "dataset")
    _strict_keys(
        dataset,
        field="dataset",
        required={
            "manifest",
            "run_id",
            "dataset_sha256",
            "split_sha256",
            "archive_sha256",
            "metadata_sha256",
            "task_config",
        },
    )
    _validate_path_hash(dataset["manifest"], "dataset.manifest")
    _require_string(dataset["run_id"], "dataset.run_id")
    for key in (
        "dataset_sha256",
        "split_sha256",
        "archive_sha256",
        "metadata_sha256",
    ):
        _require_sha256(dataset[key], f"dataset.{key}")
    _validate_config_identity(dataset["task_config"], "dataset.task_config")

    architecture = _mapping(root["architecture"], "architecture")
    _strict_keys(
        architecture,
        field="architecture",
        required={"model_config", "training_config", "transformer_source"},
    )
    _validate_config_identity(
        architecture["model_config"],
        "architecture.model_config",
    )
    _validate_config_identity(
        architecture["training_config"],
        "architecture.training_config",
    )
    _validate_path_hash(
        architecture["transformer_source"],
        "architecture.transformer_source",
    )

    basis = _mapping(root["component_basis"], "component_basis")
    _strict_keys(
        basis,
        field="component_basis",
        required={
            "status",
            "stage8_masking_manifest",
            "masks_source",
            "component_ablation_source",
            "dedicated_component_basis_sha256",
        },
    )
    if basis["status"] != "reused_predecessor_definition":
        raise PredecessorLinkError(
            "component_basis.status must be "
            "'reused_predecessor_definition'."
        )
    _validate_path_hash(
        basis["stage8_masking_manifest"],
        "component_basis.stage8_masking_manifest",
    )
    _validate_path_hash(
        basis["masks_source"],
        "component_basis.masks_source",
    )
    _validate_path_hash(
        basis["component_ablation_source"],
        "component_basis.component_ablation_source",
    )
    if basis["dedicated_component_basis_sha256"] is not None:
        raise PredecessorLinkError(
            "component_basis.dedicated_component_basis_sha256 must remain "
            "null because no dedicated predecessor hash was recorded."
        )

    teacher_runs = _sequence(root["teacher_runs"], "teacher_runs")
    expected_seeds = list(EXPECTED_TEACHER_SEEDS)

    if len(teacher_runs) != len(expected_seeds):
        raise PredecessorLinkError(
            "teacher_runs must contain exactly five entries for seeds 0-4: "
            f"expected_count={len(expected_seeds)}, "
            f"actual_count={len(teacher_runs)}."
        )

    seen_seeds: set[int] = set()
    actual_seeds: list[int] = []

    for index, item in enumerate(teacher_runs):
        field = f"teacher_runs[{index}]"
        run = _mapping(item, field)
        _strict_keys(
            run,
            field=field,
            required={"teacher_seed", "run_id", "manifest"},
        )

        seed = run["teacher_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PredecessorLinkError(
                f"{field}.teacher_seed must be an integer in the range 0-4: "
                f"actual={seed!r}."
            )
        if seed not in EXPECTED_TEACHER_SEEDS:
            raise PredecessorLinkError(
                f"{field}.teacher_seed is outside the frozen Stage 1 roster: "
                f"expected_one_of={expected_seeds}, actual={seed}."
            )
        if seed in seen_seeds:
            raise PredecessorLinkError(
                f"Duplicate teacher seed in predecessor link: {seed}."
            )

        seen_seeds.add(seed)
        actual_seeds.append(seed)

        _require_string(run["run_id"], f"{field}.run_id")
        _validate_path_hash(run["manifest"], f"{field}.manifest")

    if set(actual_seeds) != set(expected_seeds):
        raise PredecessorLinkError(
            "teacher_runs does not contain the exact frozen Stage 1 seed set: "
            f"expected={expected_seeds}, actual={actual_seeds}."
        )

    if actual_seeds != expected_seeds:
        raise PredecessorLinkError(
            "teacher_runs must use deterministic canonical seed ordering: "
            f"expected={expected_seeds}, actual={actual_seeds}."
        )

    stage3 = _mapping(
        root["stage3_checkpoint_registry"],
        "stage3_checkpoint_registry",
    )
    _strict_keys(
        stage3,
        field="stage3_checkpoint_registry",
        required={
            "status",
            "resolved",
            "selection_records",
            "deferred_fields",
        },
    )
    if stage3["status"] != "deferred_to_stage_3":
        raise PredecessorLinkError(
            "stage3_checkpoint_registry.status must remain "
            "'deferred_to_stage_3'."
        )
    if stage3["resolved"] is not False:
        raise PredecessorLinkError(
            "Stage 3 checkpoint registry cannot be marked resolved in Stage 1."
        )
    if list(_sequence(stage3["selection_records"], "stage3_checkpoint_registry.selection_records")):
        raise PredecessorLinkError(
            "Stage 3 selection_records must be empty in the Stage 1 "
            "predecessor link."
        )
    deferred = _sequence(
        stage3["deferred_fields"],
        "stage3_checkpoint_registry.deferred_fields",
    )
    if not deferred:
        raise PredecessorLinkError(
            "stage3_checkpoint_registry.deferred_fields must not be empty."
        )
    for index, value in enumerate(deferred):
        _require_string(
            value,
            f"stage3_checkpoint_registry.deferred_fields[{index}]",
        )

    visibility = _mapping(
        root["prior_results_visibility"],
        "prior_results_visibility",
    )
    expected_visibility = {
        "predecessor_primary_analysis_visible": True,
        "predecessor_analysis_freeze_complete": True,
        "followup_distillation_endpoints_produced": False,
        "followup_predictive_fidelity_endpoints_produced": False,
        "predecessor_results_are_blinded_pilot_evidence": False,
    }
    _strict_keys(
        visibility,
        field="prior_results_visibility",
        required=set(expected_visibility),
    )
    for key, expected in expected_visibility.items():
        if visibility[key] is not expected:
            raise PredecessorLinkError(
                f"prior_results_visibility.{key} must be {expected!r}."
            )

    metadata = _mapping(root["metadata"], "metadata")
    _strict_keys(
        metadata,
        field="metadata",
        required={"record_type", "stage", "scientific_execution"},
    )
    if metadata["record_type"] != "predecessor_link":
        raise PredecessorLinkError(
            "metadata.record_type must be 'predecessor_link'."
        )
    if metadata["stage"] != 1:
        raise PredecessorLinkError("metadata.stage must be 1.")
    if metadata["scientific_execution"] is not False:
        raise PredecessorLinkError(
            "metadata.scientific_execution must be false for Stage 1."
        )


def load_predecessor_link(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate one predecessor-link JSON file."""

    file_path = Path(path)
    try:
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PredecessorLinkError(
            f"Could not read predecessor-link file {file_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PredecessorLinkError(
            f"Invalid JSON in predecessor-link file {file_path}: {exc}"
        ) from exc

    record = dict(_mapping(loaded, "predecessor_link"))
    validate_predecessor_link(record)
    return record


def verify_predecessor_link_physical(
    record: Mapping[str, Any],
    *,
    predecessor_root: str | Path,
) -> None:
    """Verify canonical predecessor identities against a physical checkout.

    This function is provenance-only. It reads Git identity and the exact files
    named in the predecessor-link record; it performs no scientific analysis.
    """

    import hashlib
    import subprocess

    validate_predecessor_link(record)

    root = Path(predecessor_root).expanduser().resolve(strict=True)
    predecessor = _mapping(record["predecessor"], "predecessor")

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PredecessorLinkError(
            f"Could not verify predecessor Git commit at {root}: {exc}"
        ) from exc

    actual_commit = completed.stdout.strip()
    expected_commit = predecessor["analysis_freeze_commit"]
    if actual_commit != expected_commit:
        raise PredecessorLinkError(
            "Physical predecessor commit mismatch: "
            f"expected={expected_commit}, actual={actual_commit}."
        )

    def verify_config_identity(value: Any, field: str) -> None:
        item = _mapping(value, field)
        relative = _require_portable_path(item["path"], f"{field}.path")
        expected_file = _require_sha256(
            item["file_sha256"],
            f"{field}.file_sha256",
        )
        expected_mapping = _require_sha256(
            item["mapping_sha256"],
            f"{field}.mapping_sha256",
        )

        path = root / relative
        if not path.is_file():
            raise PredecessorLinkError(
                f"Physical predecessor config is missing for {field}: {relative}"
            )

        actual_file = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_file != expected_file:
            raise PredecessorLinkError(
                f"Physical predecessor config file hash mismatch for {field}: "
                f"path={relative}, expected={expected_file}, actual={actual_file}."
            )

        import yaml

        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, Mapping):
            raise PredecessorLinkError(
                f"Physical config mapping missing for {field}."
            )

        actual_mapping = mapping_hash(parsed)
        if actual_mapping != expected_mapping:
            raise PredecessorLinkError(
                f"Physical predecessor config mapping hash mismatch for {field}: "
                f"expected={expected_mapping}, actual={actual_mapping}."
            )

    def verify_path_hash(value: Any, field: str) -> None:
        item = _mapping(value, field)
        relative = _require_portable_path(item["path"], f"{field}.path")
        expected = _require_sha256(item["sha256"], f"{field}.sha256")
        path = root / relative

        if not path.is_file():
            raise PredecessorLinkError(
                f"Physical predecessor file is missing for {field}: {relative}"
            )

        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise PredecessorLinkError(
                f"Physical predecessor hash mismatch for {field}: "
                f"path={relative}, expected={expected}, actual={actual}."
            )

    verify_path_hash(predecessor["protocol"], "predecessor.protocol")
    verify_path_hash(
        predecessor["implementation_order"],
        "predecessor.implementation_order",
    )
    verify_path_hash(
        predecessor["analysis_freeze_manifest"],
        "predecessor.analysis_freeze_manifest",
    )

    dataset = _mapping(record["dataset"], "dataset")
    verify_path_hash(dataset["manifest"], "dataset.manifest")
    verify_config_identity(dataset["task_config"], "dataset.task_config")

    dataset_manifest_item = _mapping(dataset["manifest"], "dataset.manifest")
    dataset_manifest_relative = _require_portable_path(
        dataset_manifest_item["path"],
        "dataset.manifest.path",
    )
    dataset_manifest_path = root / dataset_manifest_relative

    try:
        physical_dataset_manifest = json.loads(
            dataset_manifest_path.read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise PredecessorLinkError(
            "Could not read physical predecessor dataset manifest "
            f"{dataset_manifest_relative}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PredecessorLinkError(
            "Physical predecessor dataset manifest is invalid JSON: "
            f"path={dataset_manifest_relative}, error={exc}"
        ) from exc

    physical_dataset_manifest = _mapping(
        physical_dataset_manifest,
        "physical_dataset_manifest",
    )

    expected_run_id = _require_string(dataset["run_id"], "dataset.run_id")
    actual_run_id = physical_dataset_manifest.get("run_id")
    if actual_run_id != expected_run_id:
        raise PredecessorLinkError(
            "Physical predecessor dataset run ID mismatch: "
            f"expected={expected_run_id!r}, actual={actual_run_id!r}."
        )

    physical_hashes = _mapping(
        physical_dataset_manifest.get("hashes"),
        "physical_dataset_manifest.hashes",
    )
    for key in (
        "dataset_sha256",
        "split_sha256",
        "archive_sha256",
        "metadata_sha256",
    ):
        expected_hash = _require_sha256(dataset[key], f"dataset.{key}")
        actual_hash = physical_hashes.get(key)
        if actual_hash != expected_hash:
            raise PredecessorLinkError(
                "Physical predecessor dataset hash mismatch for "
                f"dataset.{key}: expected={expected_hash!r}, "
                f"actual={actual_hash!r}."
            )

    task_config = _mapping(dataset["task_config"], "dataset.task_config")
    expected_config_path = _require_portable_path(
        task_config["path"],
        "dataset.task_config.path",
    )
    expected_config_mapping = _require_sha256(
        task_config["mapping_sha256"],
        "dataset.task_config.mapping_sha256",
    )

    physical_config = _mapping(
        physical_dataset_manifest.get("config"),
        "physical_dataset_manifest.config",
    )
    actual_config_path = physical_config.get("path")
    if actual_config_path != expected_config_path:
        raise PredecessorLinkError(
            "Physical predecessor dataset task-config path mismatch: "
            f"expected={expected_config_path!r}, actual={actual_config_path!r}."
        )

    actual_config_mapping = physical_config.get("sha256")
    if actual_config_mapping != expected_config_mapping:
        raise PredecessorLinkError(
            "Physical predecessor dataset task-config mapping hash mismatch: "
            f"expected={expected_config_mapping!r}, "
            f"actual={actual_config_mapping!r}."
        )

    manifest_config_hash = physical_hashes.get("config_sha256")
    if manifest_config_hash != expected_config_mapping:
        raise PredecessorLinkError(
            "Physical predecessor dataset hashes.config_sha256 mismatch: "
            f"expected={expected_config_mapping!r}, "
            f"actual={manifest_config_hash!r}."
        )

    architecture = _mapping(record["architecture"], "architecture")
    verify_config_identity(
        architecture["model_config"],
        "architecture.model_config",
    )
    verify_config_identity(
        architecture["training_config"],
        "architecture.training_config",
    )
    verify_path_hash(
        architecture["transformer_source"],
        "architecture.transformer_source",
    )

    basis = _mapping(record["component_basis"], "component_basis")
    verify_path_hash(
        basis["stage8_masking_manifest"],
        "component_basis.stage8_masking_manifest",
    )
    verify_path_hash(
        basis["masks_source"],
        "component_basis.masks_source",
    )
    verify_path_hash(
        basis["component_ablation_source"],
        "component_basis.component_ablation_source",
    )

    for index, run_value in enumerate(record["teacher_runs"]):
        field = f"teacher_runs[{index}]"
        run = _mapping(run_value, field)
        verify_path_hash(run["manifest"], f"{field}.manifest")

        manifest_item = _mapping(run["manifest"], f"{field}.manifest")
        manifest_relative = _require_portable_path(
            manifest_item["path"],
            f"{field}.manifest.path",
        )
        manifest_path = root / manifest_relative

        try:
            physical_teacher_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except OSError as exc:
            raise PredecessorLinkError(
                "Could not read physical predecessor teacher manifest "
                f"{manifest_relative}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PredecessorLinkError(
                "Physical predecessor teacher manifest is invalid JSON: "
                f"path={manifest_relative}, error={exc}"
            ) from exc

        physical_teacher_manifest = _mapping(
            physical_teacher_manifest,
            f"physical_{field}_manifest",
        )

        expected_run_id = _require_string(
            run["run_id"],
            f"{field}.run_id",
        )
        actual_run_id = physical_teacher_manifest.get("run_id")
        if actual_run_id != expected_run_id:
            raise PredecessorLinkError(
                "Physical predecessor teacher run ID mismatch for "
                f"{field}: expected={expected_run_id!r}, "
                f"actual={actual_run_id!r}."
            )

        physical_seed = _mapping(
            physical_teacher_manifest.get("seed"),
            f"physical_{field}_manifest.seed",
        )

        actual_seed_name = physical_seed.get("name")
        if actual_seed_name != "model_seed":
            raise PredecessorLinkError(
                "Physical predecessor teacher seed name mismatch for "
                f"{field}: expected='model_seed', "
                f"actual={actual_seed_name!r}."
            )

        expected_seed = run["teacher_seed"]
        actual_seed = physical_seed.get("value")
        if actual_seed != expected_seed:
            raise PredecessorLinkError(
                "Physical predecessor teacher seed mismatch for "
                f"{field}: expected={expected_seed!r}, "
                f"actual={actual_seed!r}."
            )
