"""Run the frozen Stage 12 diversity-search pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from circuit_families.analysis.fidelity_calibration import (
    file_sha256,
    write_csv_records,
    write_deterministic_tar_gz,
)
from circuit_families.analysis.stage12_artifacts import (
    Stage12CellArtifacts,
    write_stage12_cell_artifacts,
)
from circuit_families.analysis.stage12_compute_projection import (
    PilotComputeProfile,
    write_compute_projection_table,
)
from circuit_families.analysis.stage12_control_execution import (
    execute_stage12_control_suite,
)
from circuit_families.analysis.stage12_frontier import (
    Stage12FrontierRuntime,
    write_frontier_table,
)
from circuit_families.analysis.stage12_negative_controls import (
    load_stage11_random_mask_controls,
    negative_control_rows,
    stage11_random_mask_control_result,
    write_negative_control_table,
)
from circuit_families.analysis.stage12_reporting import (
    Stage12ReportCell,
    write_stage12_report_artifacts,
)
from circuit_families.config import mapping_hash
from circuit_families.interpretability.diversity_forced_search import (
    CheckpointFamilySearchExecution,
    derive_search_seed,
    run_checkpoint_family_search,
)
from circuit_families.interpretability.fidelity import (
    compute_full_model_reference,
    evaluate_component_mask,
    load_checkpoint_evaluation_context,
)
from circuit_families.interpretability.masks import (
    ComponentMask,
)
from circuit_families.interpretability.sparse_search import (
    rank_retained_components,
)
from circuit_families.manifests import package_versions
from circuit_families.training import canonical_state_hash

EXPECTED_SEARCH_CONFIG_HASH = (
    "34e930a262d0b0f84ebb24eb6be111e6"
    "b7c243c00b2a716df5104934f1625ea4"
)

RUNTIME_COLUMNS = (
    "stage12_run_id",
    "cell_id",
    "checkpoint_step",
    "distinctness_cutoff",
    "record_type",
    "requested_member_index",
    "accepted_circuit",
    "restart_count",
    "exact_evaluations_used",
    "elapsed_seconds",
    "included_in_deterministic_scientific_hashes",
)


@dataclass(frozen=True)
class Stage9ReferenceCell:
    """Committed Stage 9 cell used for C1 reproduction."""

    archive_prefix: str
    final_mask_record: Mapping[str, Any]
    final_mask_sha256: str
    accepted_removals_sha256: str
    candidate_evaluations_sha256: str
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class Stage12OutputPaths:
    """All final Stage 12 output locations."""

    raw_directory: Path
    family_summary: Path
    circuits: Path
    pairwise_overlap: Path
    restarts: Path
    negative_controls: Path
    frontier: Path
    compute_projection: Path
    runtime: Path
    validation_note: Path
    archive: Path
    manifest: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Stage 12 diversity-search pilot."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage8-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage9-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage9-table",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage9-archive",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage11-archive",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--primary-threshold-manifest",
        type=Path,
        default=Path(
            "manifests/primary_fidelity_threshold.json"
        ),
    )
    parser.add_argument(
        "--search-config",
        type=Path,
        default=Path("configs/search.yaml"),
    )
    parser.add_argument(
        "--ranking-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--parallel-worker-count",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--parallel-efficiency",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--resource-ceiling-seconds",
        type=float,
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--expected-implementation-commit",
    )
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
    )
    return parser.parse_args()


def stable_json_write(
    path: Path,
    value: Mapping[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def git_output(
    repository: Path,
    *arguments: str,
) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_clean_repository(
    repository: Path,
) -> str:
    status = git_output(
        repository,
        "status",
        "--short",
    )

    if status:
        raise RuntimeError(
            "Stage 12 scientific outputs require a clean "
            "implementation commit. Current status:\n"
            + status
        )

    return git_output(
        repository,
        "rev-parse",
        "HEAD",
    )


def resolve_path(
    repository: Path,
    path: str | Path,
) -> Path:
    value = Path(path)

    if value.is_absolute():
        return value

    return repository / value


def relative_path(
    repository: Path,
    path: Path,
) -> str:
    return path.resolve().relative_to(
        repository.resolve()
    ).as_posix()


def load_json_object(
    path: Path,
    label: str,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"{label} must contain a JSON object."
        )

    return value


def load_search_config(
    path: Path,
) -> dict[str, Any]:
    value = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            "Search configuration must be a mapping."
        )

    actual_hash = mapping_hash(value)

    if actual_hash != EXPECTED_SEARCH_CONFIG_HASH:
        raise ValueError(
            "Search configuration hash mismatch: "
            f"{actual_hash}"
        )

    return value


def fraction_value(
    value: Any,
) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(
            "Boolean values are not valid fractions."
        )

    return Fraction(str(value))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def deterministic_stage12_run_id(
    configuration: Mapping[str, Any],
) -> str:
    material = json.dumps(
        dict(configuration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256(material).hexdigest()
    model_seed = int(
        configuration["model_seed"]
    )
    return (
        f"stage12-diversity-s{model_seed}-"
        f"{digest[:12]}"
    )


def build_output_paths(
    repository: Path,
    *,
    stage12_run_id: str,
    search_config: Mapping[str, Any],
) -> Stage12OutputPaths:
    outputs = search_config.get("outputs")

    if not isinstance(outputs, Mapping):
        raise ValueError(
            "Search configuration outputs are missing."
        )

    def configured_path(name: str) -> Path:
        raw = outputs.get(name)

        if not isinstance(raw, str):
            raise ValueError(
                f"Output path {name!r} is missing."
            )

        return repository / raw.format(
            stage12_run_id=stage12_run_id
        )

    return Stage12OutputPaths(
        raw_directory=(
            repository
            / "results"
            / "raw"
            / stage12_run_id
        ),
        family_summary=configured_path(
            "family_summary_table"
        ),
        circuits=configured_path("circuit_table"),
        pairwise_overlap=configured_path(
            "pairwise_overlap_table"
        ),
        restarts=configured_path(
            "restart_table"
        ),
        negative_controls=configured_path(
            "negative_control_table"
        ),
        frontier=configured_path(
            "frontier_table"
        ),
        compute_projection=configured_path(
            "compute_projection_table"
        ),
        runtime=configured_path("runtime_table"),
        validation_note=configured_path(
            "validation_note"
        ),
        archive=configured_path(
            "archive_template"
        ),
        manifest=configured_path(
            "manifest_template"
        ),
    )


def validate_absent_outputs(
    outputs: Stage12OutputPaths,
) -> None:
    existing = [
        path
        for path in asdict(outputs).values()
        if Path(path).exists()
    ]

    if existing:
        rendered = "\n".join(
            str(path)
            for path in existing
        )
        raise FileExistsError(
            "Stage 12 refuses to overwrite existing "
            "outputs:\n"
            + rendered
        )


def cleanup_outputs(
    outputs: Stage12OutputPaths,
) -> None:
    for value in asdict(outputs).values():
        path = Path(value)

        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _archive_files(
    archive_path: Path,
) -> dict[str, bytes]:
    files: dict[str, bytes] = {}

    with tarfile.open(
        archive_path,
        mode="r:gz",
    ) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            handle = archive.extractfile(member)

            if handle is None:
                raise RuntimeError(
                    "Could not read archive member "
                    f"{member.name!r}."
                )

            files[member.name] = handle.read()

    return files


def _summary_checkpoint_step(
    summary: Mapping[str, Any],
) -> int | None:
    metadata = summary.get("cell_metadata")

    candidates: list[Any] = [
        summary.get("checkpoint_step"),
    ]

    if isinstance(metadata, Mapping):
        candidates.extend(
            (
                metadata.get("checkpoint_step"),
                metadata.get("training_step"),
            )
        )

    for value in candidates:
        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def load_stage9_reference_cell(
    archive_path: str | Path,
    *,
    checkpoint_step: int = 9050,
    fidelity_threshold: Fraction = Fraction(
        99,
        100,
    ),
) -> Stage9ReferenceCell:
    """Load the exact committed Stage 9 C1 source cell."""

    files = _archive_files(Path(archive_path))
    matches: list[
        tuple[str, Mapping[str, Any]]
    ] = []

    for name, content in files.items():
        if not name.endswith(
            "/cell_summary.json"
        ):
            continue

        value = json.loads(
            content.decode("utf-8")
        )

        if not isinstance(value, Mapping):
            continue

        search = value.get("search")

        if not isinstance(search, Mapping):
            continue

        try:
            threshold = fraction_value(
                search["fidelity_threshold"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            continue

        if (
            threshold == fidelity_threshold
            and _summary_checkpoint_step(value)
            == checkpoint_step
        ):
            matches.append((name, value))

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one Stage 9 reference "
            "cell; found "
            f"{len(matches)}."
        )

    summary_name, summary = matches[0]
    prefix = summary_name.rsplit("/", 1)[0]
    required = {
        "final_mask": (
            f"{prefix}/final_mask.json"
        ),
        "accepted_removals": (
            f"{prefix}/accepted_removals.jsonl"
        ),
        "candidate_evaluations": (
            f"{prefix}/candidate_evaluations.jsonl"
        ),
    }

    missing = [
        name
        for name in required.values()
        if name not in files
    ]

    if missing:
        raise RuntimeError(
            "Stage 9 reference cell is missing: "
            + ", ".join(missing)
        )

    final_mask_record = json.loads(
        files[required["final_mask"]].decode(
            "utf-8"
        )
    )

    if not isinstance(
        final_mask_record,
        Mapping,
    ):
        raise ValueError(
            "Stage 9 final-mask record must be "
            "a mapping."
        )

    ComponentMask.from_record(
        final_mask_record
    )

    return Stage9ReferenceCell(
        archive_prefix=prefix,
        final_mask_record=final_mask_record,
        final_mask_sha256=sha256_bytes(
            files[required["final_mask"]]
        ),
        accepted_removals_sha256=sha256_bytes(
            files[required["accepted_removals"]]
        ),
        candidate_evaluations_sha256=(
            sha256_bytes(
                files[
                    required[
                        "candidate_evaluations"
                    ]
                ]
            )
        ),
        summary=summary,
    )


def validate_stage9_reference(
    reference: Stage9ReferenceCell,
) -> None:
    search = reference.summary["search"]
    final_mask = ComponentMask.from_record(
        reference.final_mask_record
    )

    if final_mask.retained_component_count != 146:
        raise ValueError(
            "Stage 9 primary C1 retained count is "
            "not 146."
        )

    if int(
        search["exact_evaluations_used"]
    ) != 6098:
        raise ValueError(
            "Stage 9 primary C1 exact-evaluation "
            "count is not 6098."
        )

    if float(
        search["fidelity_threshold"]
    ) != 0.99:
        raise ValueError(
            "Stage 9 primary threshold is not 0.99."
        )


def validate_frozen_inputs(
    *,
    repository: Path,
    run_id: str,
    checkpoint_manifest: Path,
    stage8_manifest: Path,
    stage9_manifest: Path,
    stage9_table: Path,
    stage9_archive: Path,
    stage11_archive: Path,
    primary_threshold_manifest: Path,
    search_config: Mapping[str, Any],
) -> Stage9ReferenceCell:
    files = (
        checkpoint_manifest,
        stage8_manifest,
        stage9_manifest,
        stage9_table,
        stage9_archive,
        stage11_archive,
        primary_threshold_manifest,
    )

    for path in files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Stage 12 input is absent: "
                f"{path}"
            )

    source = search_config["source"]

    if source["training_run_id"] != run_id:
        raise ValueError(
            "Source training run does not match "
            "configs/search.yaml."
        )

    if int(source["checkpoint_step"]) != 9050:
        raise ValueError(
            "Stage 12 checkpoint is not step 9050."
        )

    fidelity = search_config["fidelity"]

    if (
        int(fidelity["threshold_numerator"]) != 99
        or int(
            fidelity["threshold_denominator"]
        )
        != 100
    ):
        raise ValueError(
            "Stage 12 primary fidelity is not 99/100."
        )

    order = [
        fraction_value(value)
        for value in search_config[
            "distinctness"
        ]["definitive_execution_order"]
    ]

    if order != [
        Fraction(1, 2),
        Fraction(1, 4),
        Fraction(3, 4),
    ]:
        raise ValueError(
            "Stage 12 cutoff order is not frozen."
        )

    primary = load_json_object(
        primary_threshold_manifest,
        "primary-threshold manifest",
    )

    if primary.get("freeze_status") != "frozen":
        raise ValueError(
            "Primary threshold is not frozen."
        )

    primary_threshold = primary.get(
        "primary_fidelity_threshold"
    )

    if not isinstance(
        primary_threshold,
        Mapping,
    ):
        raise ValueError(
            "Primary threshold record is missing."
        )

    if (
        int(primary_threshold["numerator"]) != 99
        or int(primary_threshold["denominator"])
        != 100
    ):
        raise ValueError(
            "Primary-threshold manifest mismatch."
        )

    checkpoint = primary.get("checkpoint")

    if (
        not isinstance(checkpoint, Mapping)
        or int(checkpoint["training_step"])
        != 9050
    ):
        raise ValueError(
            "Primary-threshold checkpoint mismatch."
        )

    integrity = primary.get("integrity")

    if (
        not isinstance(integrity, Mapping)
        or integrity.get(
            "stage12_started_at_freeze"
        )
        is not False
    ):
        raise ValueError(
            "Threshold-freeze Stage 12 boundary "
            "is invalid."
        )

    random_records = (
        load_stage11_random_mask_controls(
            stage11_archive
        )
    )
    random_result = (
        stage11_random_mask_control_result(
            random_records
        )
    )

    if not random_result.validation_passed:
        raise ValueError(
            "Stage 11 random-mask control failed."
        )

    reference = load_stage9_reference_cell(
        stage9_archive
    )
    validate_stage9_reference(reference)

    return reference


def stage12_configuration_record(
    *,
    search_config: Mapping[str, Any],
    implementation_commit: str,
    input_hashes: Mapping[str, str],
    ranking_batch_size: int,
    evaluation_batch_size: int,
    device: str,
) -> dict[str, Any]:
    return {
        "experiment_type": (
            "stage12_diversity_forced_search"
        ),
        "model_seed": int(
            search_config["source"]["model_seed"]
        ),
        "checkpoint_step": int(
            search_config["source"][
                "checkpoint_step"
            ]
        ),
        "checkpoint_index": int(
            search_config["source"][
                "checkpoint_index"
            ]
        ),
        "fidelity_threshold": (
            int(
                search_config["fidelity"][
                    "threshold_numerator"
                ]
            )
            / int(
                search_config["fidelity"][
                    "threshold_denominator"
                ]
            )
        ),
        "distinctness_cutoff_order": [
            float(value)
            for value in search_config[
                "distinctness"
            ]["definitive_execution_order"]
        ],
        "family_target": int(
            search_config["budgets"][
                "family_target"
            ]
        ),
        "max_restarts_per_alternative": int(
            search_config["restarts"][
                "maximum_per_requested_alternative"
            ]
        ),
        "per_requested_circuit_budget": int(
            search_config["budgets"][
                "per_requested_circuit_exact_evaluations"
            ]
        ),
        "per_cell_budget": int(
            search_config["budgets"][
                "per_cell_exact_evaluations"
            ]
        ),
        "reuse_coefficient": float(
            search_config["ranking"][
                "reuse_coefficient"
            ]
        ),
        "tie_tolerance": float(
            search_config["ranking"][
                "numerically_indistinguishable_tolerance"
            ]
        ),
        "ranking_batch_size": (
            ranking_batch_size
        ),
        "evaluation_batch_size": (
            evaluation_batch_size
        ),
        "device": device,
        "search_config_sha256": (
            mapping_hash(search_config)
        ),
        "implementation_git_commit": (
            implementation_commit
        ),
        "input_hashes": dict(input_hashes),
    }


def c1_reproduction_record(
    *,
    reference: Stage9ReferenceCell,
    execution: CheckpointFamilySearchExecution,
    artifacts: Stage12CellArtifacts,
    cell_id: str,
) -> dict[str, Any]:
    family = execution.result

    if not family.members:
        raise RuntimeError(
            "Stage 12 did not recover C1."
        )

    member = family.members[0]

    if member.member_index != 1:
        raise RuntimeError(
            "First Stage 12 member is not C1."
        )

    c1_outcomes = [
        outcome
        for outcome in family.restart_outcomes
        if outcome.requested_member_index == 1
    ]

    if len(c1_outcomes) != 1:
        raise RuntimeError(
            "C1 must have exactly one restart outcome."
        )

    if not artifacts.sparse_search_artifacts:
        raise RuntimeError(
            "C1 sparse-search artifacts are absent."
        )

    search_artifacts = (
        artifacts.sparse_search_artifacts[0]
    )
    source_search = reference.summary["search"]
    source_metrics = reference.summary[
        "final_metrics"
    ]
    source_mask = ComponentMask.from_record(
        reference.final_mask_record
    )

    checks = {
        "mask_id_identical": (
            member.mask.mask_id
            == source_mask.mask_id
        ),
        "mask_bytes_identical": (
            search_artifacts.final_mask_sha256
            == reference.final_mask_sha256
        ),
        "accepted_removal_trajectory_identical": (
            search_artifacts
            .accepted_removal_trajectory_sha256
            == reference
            .accepted_removals_sha256
        ),
        "candidate_evaluation_log_identical": (
            search_artifacts
            .candidate_evaluation_log_sha256
            == reference
            .candidate_evaluations_sha256
        ),
        "retained_component_count_identical": (
            member.mask.retained_component_count
            == source_mask.retained_component_count
        ),
        "prediction_agreement_count_identical": (
            member.metrics
            .prediction_agreement_count
            == int(
                source_metrics[
                    "prediction_agreement_count"
                ]
            )
        ),
        "primary_fidelity_identical": (
            member.metrics.primary_fidelity
            == float(
                source_metrics[
                    "primary_fidelity"
                ]
            )
        ),
        "exact_evaluations_identical": (
            member.search_result
            .exact_evaluations_used
            == int(
                source_search[
                    "exact_evaluations_used"
                ]
            )
        ),
        "ranking_passes_identical": (
            member.search_result
            .ranking_passes_used
            == int(
                source_search[
                    "ranking_passes_used"
                ]
            )
        ),
        "search_status_identical": (
            member.search_result.status
            == source_search["status"]
        ),
    }
    passed = all(checks.values())

    record = {
        "cell_id": cell_id,
        "stage9_archive_prefix": (
            reference.archive_prefix
        ),
        "stage12_member_index": 1,
        "stage12_restart_index": (
            member.selected_restart_index
        ),
        "stage9_retained_components": (
            source_mask.retained_component_count
        ),
        "stage12_retained_components": (
            member.mask.retained_component_count
        ),
        "stage9_exact_evaluations": int(
            source_search[
                "exact_evaluations_used"
            ]
        ),
        "stage12_exact_evaluations": (
            member.search_result
            .exact_evaluations_used
        ),
        "stage9_primary_fidelity": float(
            source_metrics["primary_fidelity"]
        ),
        "stage12_primary_fidelity": (
            member.metrics.primary_fidelity
        ),
        "checks": checks,
        "passed": passed,
    }

    if not passed:
        failed = [
            name
            for name, value in checks.items()
            if not value
        ]
        raise RuntimeError(
            "Stage 12 C1 reproduction failed: "
            + ", ".join(failed)
        )

    return record


def member_runtime_callbacks(
) -> tuple[
    Callable[[int], None],
    Callable[[int], None],
    dict[int, float],
]:
    """Return isolated callbacks and elapsed times for one cell."""

    started: dict[int, float] = {}
    elapsed: dict[int, float] = {}

    def member_started_callback(
        member_index: int,
    ) -> None:
        if member_index in started:
            raise RuntimeError(
                "Member timer started twice."
            )

        started[member_index] = time.perf_counter()

    def member_finished_callback(
        member_index: int,
    ) -> None:
        if member_index not in started:
            raise RuntimeError(
                "Member timer finished before starting."
            )

        elapsed[member_index] = (
            time.perf_counter()
            - started[member_index]
        )

    return (
        member_started_callback,
        member_finished_callback,
        elapsed,
    )


def runtime_rows_for_cell(
    *,
    stage12_run_id: str,
    cell_id: str,
    checkpoint_step: int,
    cutoff: Fraction,
    execution: CheckpointFamilySearchExecution,
    member_elapsed: Mapping[int, float],
    cell_elapsed_seconds: float,
) -> list[dict[str, Any]]:
    family = execution.result
    grouped: dict[int, list[Any]] = {}

    for outcome in family.restart_outcomes:
        grouped.setdefault(
            outcome.requested_member_index,
            [],
        ).append(outcome)

    accepted = {
        member.member_index
        for member in family.members
    }

    rows: list[dict[str, Any]] = [
        {
            "stage12_run_id": stage12_run_id,
            "cell_id": cell_id,
            "checkpoint_step": checkpoint_step,
            "distinctness_cutoff": float(cutoff),
            "record_type": "cell",
            "requested_member_index": "",
            "accepted_circuit": "",
            "restart_count": len(
                family.restart_outcomes
            ),
            "exact_evaluations_used": (
                family.exact_evaluations_used
            ),
            "elapsed_seconds": (
                cell_elapsed_seconds
            ),
            "included_in_deterministic_"
            "scientific_hashes": False,
        }
    ]

    if set(grouped) != set(member_elapsed):
        raise RuntimeError(
            "Member runtime coverage does not match "
            "restart outcomes."
        )

    for member_index in sorted(grouped):
        outcomes = grouped[member_index]
        rows.append(
            {
                "stage12_run_id": (
                    stage12_run_id
                ),
                "cell_id": cell_id,
                "checkpoint_step": (
                    checkpoint_step
                ),
                "distinctness_cutoff": (
                    float(cutoff)
                ),
                "record_type": (
                    "requested_member"
                ),
                "requested_member_index": (
                    member_index
                ),
                "accepted_circuit": (
                    member_index in accepted
                ),
                "restart_count": len(outcomes),
                "exact_evaluations_used": sum(
                    outcome.execution.result
                    .exact_evaluations_used
                    for outcome in outcomes
                ),
                "elapsed_seconds": (
                    member_elapsed[
                        member_index
                    ]
                ),
                "included_in_deterministic_"
                "scientific_hashes": False,
            }
        )

    return rows


def pilot_compute_profile(
    *,
    stage12_run_id: str,
    report_cells: Sequence[
        Stage12ReportCell
    ],
    runtime_rows: Sequence[
        Mapping[str, Any]
    ],
    device: str,
    ranking_batch_size: int,
    evaluation_batch_size: int,
) -> PilotComputeProfile:
    cell_rows = [
        row
        for row in runtime_rows
        if row["record_type"] == "cell"
    ]
    member_rows = [
        row
        for row in runtime_rows
        if row["record_type"]
        == "requested_member"
    ]

    accepted_rows = [
        row
        for row in member_rows
        if row["accepted_circuit"] is True
    ]
    failed_rows = [
        row
        for row in member_rows
        if row["accepted_circuit"] is False
    ]

    return PilotComputeProfile(
        stage12_run_id=stage12_run_id,
        pilot_cell_count=len(report_cells),
        pilot_exact_evaluations=sum(
            cell.execution.result
            .exact_evaluations_used
            for cell in report_cells
        ),
        pilot_runtime_seconds=sum(
            float(row["elapsed_seconds"])
            for row in cell_rows
        ),
        recovered_circuit_count=len(
            accepted_rows
        ),
        recovered_circuit_exact_evaluations=sum(
            int(row["exact_evaluations_used"])
            for row in accepted_rows
        ),
        recovered_circuit_runtime_seconds=sum(
            float(row["elapsed_seconds"])
            for row in accepted_rows
        ),
        failed_requested_alternative_count=len(
            failed_rows
        ),
        failed_requested_alternative_exact_evaluations=sum(
            int(row["exact_evaluations_used"])
            for row in failed_rows
        ),
        failed_requested_alternative_runtime_seconds=sum(
            float(row["elapsed_seconds"])
            for row in failed_rows
        ),
        restart_count=sum(
            int(row["restart_count"])
            for row in member_rows
        ),
        restart_runtime_seconds=sum(
            float(row["elapsed_seconds"])
            for row in member_rows
        ),
        device=device,
        ranking_batch_size=ranking_batch_size,
        evaluation_batch_size=(
            evaluation_batch_size
        ),
    )


def copy_report_outputs(
    *,
    report: Any,
    outputs: Stage12OutputPaths,
) -> None:
    copies = (
        (
            report.family_summary_path,
            outputs.family_summary,
        ),
        (
            report.circuit_path,
            outputs.circuits,
        ),
        (
            report.pairwise_overlap_path,
            outputs.pairwise_overlap,
        ),
        (
            report.restart_path,
            outputs.restarts,
        ),
        (
            report.validation_note_path,
            outputs.validation_note,
        ),
    )

    for source, destination in copies:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copyfile(
            source,
            destination,
        )


def main() -> None:
    args = parse_args()
    repository = (
        args.repository_root.resolve()
    )
    search_config_path = resolve_path(
        repository,
        args.search_config,
    )
    search_config = load_search_config(
        search_config_path
    )

    resolved_inputs = {
        "checkpoint_manifest": resolve_path(
            repository,
            args.checkpoint_manifest,
        ),
        "stage8_manifest": resolve_path(
            repository,
            args.stage8_manifest,
        ),
        "stage9_manifest": resolve_path(
            repository,
            args.stage9_manifest,
        ),
        "stage9_table": resolve_path(
            repository,
            args.stage9_table,
        ),
        "stage9_archive": resolve_path(
            repository,
            args.stage9_archive,
        ),
        "stage11_archive": resolve_path(
            repository,
            args.stage11_archive,
        ),
        "primary_threshold_manifest": (
            resolve_path(
                repository,
                args.primary_threshold_manifest,
            )
        ),
        "search_config": search_config_path,
    }

    reference = validate_frozen_inputs(
        repository=repository,
        run_id=args.run_id,
        checkpoint_manifest=resolved_inputs[
            "checkpoint_manifest"
        ],
        stage8_manifest=resolved_inputs[
            "stage8_manifest"
        ],
        stage9_manifest=resolved_inputs[
            "stage9_manifest"
        ],
        stage9_table=resolved_inputs[
            "stage9_table"
        ],
        stage9_archive=resolved_inputs[
            "stage9_archive"
        ],
        stage11_archive=resolved_inputs[
            "stage11_archive"
        ],
        primary_threshold_manifest=(
            resolved_inputs[
                "primary_threshold_manifest"
            ]
        ),
        search_config=search_config,
    )

    current_commit = git_output(
        repository,
        "rev-parse",
        "HEAD",
    )
    input_hashes = {
        name: file_sha256(path)
        for name, path in resolved_inputs.items()
    }
    configuration = stage12_configuration_record(
        search_config=search_config,
        implementation_commit=current_commit,
        input_hashes=input_hashes,
        ranking_batch_size=(
            args.ranking_batch_size
        ),
        evaluation_batch_size=(
            args.evaluation_batch_size
        ),
        device=args.device,
    )
    stage12_run_id = (
        deterministic_stage12_run_id(
            configuration
        )
    )

    if args.validate_inputs_only:
        print(
            f"stage12_run_id: {stage12_run_id}"
        )
        print(
            "implementation_git_commit: "
            f"{current_commit}"
        )
        print(
            "checkpoint_step: "
            f"{configuration['checkpoint_step']}"
        )
        print(
            "fidelity_threshold: "
            f"{configuration['fidelity_threshold']:.6f}"
        )
        print(
            "distinctness_execution_order: "
            + ", ".join(
                f"{value:.2f}"
                for value in configuration[
                    "distinctness_cutoff_order"
                ]
            )
        )
        print(
            "stage9_c1_retained_components: "
            f"{ComponentMask.from_record(reference.final_mask_record).retained_component_count}"
        )
        print(
            "stage9_c1_exact_evaluations: "
            f"{reference.summary['search']['exact_evaluations_used']}"
        )
        print(
            "stage11_primary_random_masks: 100"
        )
        print("input_validation: passed")
        return

    implementation_commit = (
        require_clean_repository(repository)
    )

    if (
        args.expected_implementation_commit
        and implementation_commit
        != args.expected_implementation_commit
    ):
        raise RuntimeError(
            "Implementation commit mismatch: "
            f"expected "
            f"{args.expected_implementation_commit}, "
            f"found {implementation_commit}."
        )

    if implementation_commit != current_commit:
        raise RuntimeError(
            "Repository HEAD changed during input "
            "validation."
        )

    outputs = build_output_paths(
        repository,
        stage12_run_id=stage12_run_id,
        search_config=search_config,
    )
    validate_absent_outputs(outputs)

    source = search_config["source"]
    fidelity = search_config["fidelity"]
    budgets = search_config["budgets"]
    restarts = search_config["restarts"]
    ranking = search_config["ranking"]
    cutoffs = [
        fraction_value(value)
        for value in search_config[
            "distinctness"
        ]["definitive_execution_order"]
    ]
    threshold = (
        int(fidelity["threshold_numerator"])
        / int(fidelity["threshold_denominator"])
    )

    context = load_checkpoint_evaluation_context(
        repository_root=repository,
        run_id=args.run_id,
        checkpoint_manifest_path=(
            args.checkpoint_manifest
        ),
        checkpoint_step=int(
            source["checkpoint_step"]
        ),
        device_override=args.device,
    )
    initial_model_state = (
        canonical_state_hash(
            context.model.state_dict()
        )
    )

    report_cells: list[
        Stage12ReportCell
    ] = []
    cell_artifacts: list[
        Stage12CellArtifacts
    ] = []
    executions: list[
        CheckpointFamilySearchExecution
    ] = []
    c1_records: list[
        dict[str, Any]
    ] = []
    runtime_rows: list[
        dict[str, Any]
    ] = []
    frontier_runtimes: list[
        Stage12FrontierRuntime
    ] = []

    try:
        outputs.raw_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        for execution_index, cutoff in enumerate(
            cutoffs,
            start=1,
        ):
            cell_id = (
                f"cutoff-{float(cutoff):.2f}"
            )
            raw_cell_directory = (
                outputs.raw_directory
                / cell_id
            )
            (
                member_started_callback,
                member_finished_callback,
                member_elapsed,
            ) = member_runtime_callbacks()

            print(
                f"[{execution_index:02d}/"
                f"{len(cutoffs):02d}] "
                f"checkpoint=9050 "
                f"threshold={threshold:.6f} "
                f"cutoff={float(cutoff):.2f}"
            )

            cell_started = time.perf_counter()
            execution = (
                run_checkpoint_family_search(
                    context,
                    fidelity_threshold=threshold,
                    distinctness_cutoff=cutoff,
                    model_seed=int(
                        source["model_seed"]
                    ),
                    checkpoint_index=int(
                        source[
                            "checkpoint_index"
                        ]
                    ),
                    ranking_batch_size=(
                        args.ranking_batch_size
                    ),
                    evaluation_batch_size=(
                        args.evaluation_batch_size
                    ),
                    family_target=int(
                        budgets["family_target"]
                    ),
                    max_restarts_per_alternative=int(
                        restarts[
                            "maximum_per_requested_alternative"
                        ]
                    ),
                    per_requested_circuit_budget=int(
                        budgets[
                            "per_requested_circuit_exact_evaluations"
                        ]
                    ),
                    per_cell_budget=int(
                        budgets[
                            "per_cell_exact_evaluations"
                        ]
                    ),
                    reuse_coefficient=float(
                        ranking[
                            "reuse_coefficient"
                        ]
                    ),
                    tie_tolerance=float(
                        ranking[
                            "numerically_indistinguishable_tolerance"
                        ]
                    ),
                    member_started_callback=(
                        member_started_callback
                    ),
                    member_finished_callback=(
                        member_finished_callback
                    ),
                )
            )
            cell_elapsed = (
                time.perf_counter()
                - cell_started
            )

            artifacts = (
                write_stage12_cell_artifacts(
                    raw_cell_directory,
                    execution,
                    cell_metadata={
                        "stage12_run_id": (
                            stage12_run_id
                        ),
                        "execution_index": (
                            execution_index
                        ),
                        "source_training_run_id": (
                            args.run_id
                        ),
                        "checkpoint_step": int(
                            source[
                                "checkpoint_step"
                            ]
                        ),
                        "fidelity_threshold": (
                            threshold
                        ),
                        "distinctness_cutoff": (
                            float(cutoff)
                        ),
                        "search_config_sha256": (
                            mapping_hash(
                                search_config
                            )
                        ),
                        "implementation_git_commit": (
                            implementation_commit
                        ),
                    },
                )
            )
            c1_record = c1_reproduction_record(
                reference=reference,
                execution=execution,
                artifacts=artifacts,
                cell_id=cell_id,
            )

            cell_runtime_rows = (
                runtime_rows_for_cell(
                    stage12_run_id=(
                        stage12_run_id
                    ),
                    cell_id=cell_id,
                    checkpoint_step=int(
                        source["checkpoint_step"]
                    ),
                    cutoff=cutoff,
                    execution=execution,
                    member_elapsed=member_elapsed,
                    cell_elapsed_seconds=(
                        cell_elapsed
                    ),
                )
            )
            runtime_rows.extend(
                cell_runtime_rows
            )

            for row in cell_runtime_rows:
                if (
                    row["record_type"]
                    == "requested_member"
                ):
                    frontier_runtimes.append(
                        Stage12FrontierRuntime(
                            cell_id=cell_id,
                            requested_member_index=int(
                                row[
                                    "requested_member_index"
                                ]
                            ),
                            runtime_seconds=float(
                                row[
                                    "elapsed_seconds"
                                ]
                            ),
                        )
                    )

            report_cells.append(
                Stage12ReportCell(
                    cell_id=cell_id,
                    checkpoint_step=int(
                        source["checkpoint_step"]
                    ),
                    distinctness_cutoff=cutoff,
                    execution=execution,
                    raw_cell_directory=(
                        f"{stage12_run_id}/"
                        f"{cell_id}"
                    ),
                )
            )
            cell_artifacts.append(artifacts)
            executions.append(execution)
            c1_records.append(c1_record)

            print(
                "completed: "
                f"status={execution.result.status} "
                f"family_size="
                f"{execution.result.family_size} "
                f"exact_evaluations="
                f"{execution.result.exact_evaluations_used}"
            )

        primary_execution = executions[0]
        primary_c1 = (
            primary_execution.result.members[0]
        )
        primary_c1_outcome = (
            primary_execution
            .result
            .restart_outcomes[0]
        )

        if not (
            primary_c1_outcome
            .execution
            .ranking_results
        ):
            raise RuntimeError(
                "Primary C1 has no ranking record."
            )

        original_ranking = (
            primary_c1_outcome
            .execution
            .ranking_results[0]
            .ranking_result
        )
        shuffle_seed = derive_search_seed(
            model_seed=int(
                source["model_seed"]
            ),
            checkpoint_index=int(
                source["checkpoint_index"]
            ),
            family_member_index=2,
            restart_index=0,
        )

        control_started = time.perf_counter()
        full_reference = (
            compute_full_model_reference(
                context.model,
                context.inputs,
                context.targets,
                batch_size=(
                    args.evaluation_batch_size
                ),
            )
        )
        pseudo_targets = (
            full_reference.predictions
            .detach()
            .clone()
        )
        all_retained = (
            ComponentMask.all_retained()
        )
        control_initial_metrics = (
            evaluate_component_mask(
                context.model,
                context.inputs,
                context.targets,
                all_retained,
                batch_size=(
                    args.evaluation_batch_size
                ),
                full_model_reference=(
                    full_reference
                ),
            )
        )

        def control_base_ranking(
            mask: ComponentMask,
        ):
            if mask == all_retained:
                return original_ranking

            return rank_retained_components(
                context.model,
                context.inputs,
                pseudo_targets,
                mask,
                batch_size=(
                    args.ranking_batch_size
                ),
            )

        def control_exact_evaluation(
            mask: ComponentMask,
        ):
            return evaluate_component_mask(
                context.model,
                context.inputs,
                context.targets,
                mask,
                batch_size=(
                    args.evaluation_batch_size
                ),
                full_model_reference=(
                    full_reference
                ),
            )

        control_execution = (
            execute_stage12_control_suite(
                stage11_archive_path=(
                    resolved_inputs[
                        "stage11_archive"
                    ]
                ),
                primary_c1_search=(
                    primary_c1.search_result
                ),
                primary_c1_mask=(
                    primary_c1.mask
                ),
                original_ranking=(
                    original_ranking
                ),
                base_ranking_function=(
                    control_base_ranking
                ),
                exact_evaluation_function=(
                    control_exact_evaluation
                ),
                initial_metrics=(
                    control_initial_metrics
                ),
                shuffle_seed_integer=(
                    shuffle_seed.integer_seed
                ),
                fidelity_threshold=threshold,
                distinctness_cutoff=(
                    cutoffs[0]
                ),
                reuse_coefficient=float(
                    ranking[
                        "reuse_coefficient"
                    ]
                ),
            )
        )
        control_elapsed = (
            time.perf_counter()
            - control_started
        )
        negative_control_results = (
            control_execution
            .negative_controls
        )
        stress_test_results = (
            control_execution
            .stress_tests
        )
        control_results = (
            *negative_control_results,
            *stress_test_results,
        )

        runtime_rows.append(
            {
                "stage12_run_id": (
                    stage12_run_id
                ),
                "cell_id": (
                    "controls-and-stress"
                ),
                "checkpoint_step": int(
                    source["checkpoint_step"]
                ),
                "distinctness_cutoff": "",
                "record_type": (
                    "control_suite"
                ),
                "requested_member_index": "",
                "accepted_circuit": False,
                "restart_count": 0,
                "exact_evaluations_used": (
                    control_execution
                    .model_exact_evaluations_used
                ),
                "elapsed_seconds": (
                    control_elapsed
                ),
                "included_in_deterministic_"
                "scientific_hashes": False,
            }
        )

        with tempfile.TemporaryDirectory() as raw:
            temporary = Path(raw)
            report = (
                write_stage12_report_artifacts(
                    temporary / "reports",
                    stage12_run_id=(
                        stage12_run_id
                    ),
                    seed=int(
                        source["model_seed"]
                    ),
                    cells=report_cells,
                    execution_order=cutoffs,
                )
            )
            copy_report_outputs(
                report=report,
                outputs=outputs,
            )

        write_negative_control_table(
            outputs.negative_controls,
            stage12_run_id=stage12_run_id,
            results=control_results,
        )
        write_csv_records(
            outputs.runtime,
            fieldnames=RUNTIME_COLUMNS,
            rows=runtime_rows,
        )
        write_frontier_table(
            outputs.frontier,
            stage12_run_id=stage12_run_id,
            cells=report_cells,
            execution_order=cutoffs,
            runtimes=frontier_runtimes,
        )
        profile = pilot_compute_profile(
            stage12_run_id=stage12_run_id,
            report_cells=report_cells,
            runtime_rows=runtime_rows,
            device=str(context.device),
            ranking_batch_size=(
                args.ranking_batch_size
            ),
            evaluation_batch_size=(
                args.evaluation_batch_size
            ),
        )
        write_compute_projection_table(
            outputs.compute_projection,
            profile=profile,
            parallel_worker_count=(
                args.parallel_worker_count
            ),
            parallel_efficiency_assumption=(
                args.parallel_efficiency
            ),
            resource_ceiling_seconds=(
                args.resource_ceiling_seconds
            ),
        )

        note = outputs.validation_note.read_text(
            encoding="utf-8"
        )
        note += (
            "\n## C1 exact reproduction\n\n"
            + "\n".join(
                "- "
                f"`{record['cell_id']}`: passed "
                f"({record['stage12_retained_components']} "
                "components; "
                f"{record['stage12_exact_evaluations']} "
                "exact evaluations)."
                for record in c1_records
            )
            + "\n\n"
            + "## Negative controls\n\n"
            + "\n".join(
                "- "
                f"`{result.control_name}`: "
                f"`{result.observed_outcome}`."
                for result in (
                    negative_control_results
                )
            )
            + "\n\n"
            + "## Method stress tests\n\n"
            + "\n".join(
                "- "
                f"`{result.control_name}`: "
                f"`{result.observed_outcome}`."
                for result in (
                    stress_test_results
                )
            )
            + "\n"
        )
        outputs.validation_note.write_text(
            note,
            encoding="utf-8",
        )

        run_record = {
            "schema_version": 1,
            "stage12_run_id": stage12_run_id,
            "configuration": configuration,
            "c1_reproduction": c1_records,
            "negative_controls": (
                negative_control_rows(
                    stage12_run_id=(
                        stage12_run_id
                    ),
                    results=(
                        negative_control_results
                    ),
                )
            ),
            "stress_tests": (
                negative_control_rows(
                    stage12_run_id=(
                        stage12_run_id
                    ),
                    results=(
                        stress_test_results
                    ),
                )
            ),
            "control_execution": (
                control_execution.to_record()
            ),
            "cells": [
                {
                    "cell_id": (
                        report_cell.cell_id
                    ),
                    "distinctness_cutoff": (
                        float(
                            report_cell
                            .distinctness_cutoff
                        )
                    ),
                    "status": (
                        report_cell.execution
                        .result.status
                    ),
                    "family_size": (
                        report_cell.execution
                        .result.family_size
                    ),
                    "exact_evaluations_used": (
                        report_cell.execution
                        .result
                        .exact_evaluations_used
                    ),
                    "cell_summary_sha256": (
                        artifacts
                        .cell_summary_sha256
                    ),
                    "hash_inventory_sha256": (
                        artifacts
                        .hash_inventory_sha256
                    ),
                }
                for report_cell, artifacts in zip(
                    report_cells,
                    cell_artifacts,
                    strict=True,
                )
            ],
            "runtime_included_in_deterministic_"
            "scientific_hashes": False,
            "stage13_started": False,
        }
        stable_json_write(
            outputs.raw_directory
            / "stage12_run_record.json",
            run_record,
        )
        write_deterministic_tar_gz(
            source_directory=(
                outputs.raw_directory
            ),
            archive_path=outputs.archive,
        )

        final_model_state = (
            canonical_state_hash(
                context.model.state_dict()
            )
        )

        if final_model_state != initial_model_state:
            raise RuntimeError(
                "Stage 12 changed the model state."
            )

        if any(
            parameter.grad is not None
            for parameter in context.model.parameters()
        ):
            raise RuntimeError(
                "Stage 12 left model gradients "
                "populated."
            )

        deterministic_outputs = {
            "family_summary_table": (
                outputs.family_summary
            ),
            "circuit_table": outputs.circuits,
            "pairwise_overlap_table": (
                outputs.pairwise_overlap
            ),
            "restart_table": outputs.restarts,
            "negative_control_table": (
                outputs.negative_controls
            ),
            "validation_note": (
                outputs.validation_note
            ),
            "archive": outputs.archive,
        }
        runtime_bearing_outputs = {
            "frontier_table": outputs.frontier,
            "compute_projection_table": (
                outputs.compute_projection
            ),
            "runtime_table": outputs.runtime,
        }
        manifest = {
            "schema_version": 1,
            "experiment_type": (
                "stage12_diversity_forced_search"
            ),
            "stage12_run_id": stage12_run_id,
            "creation_timestamp_utc": (
                datetime.now(UTC).isoformat()
            ),
            "stage12_implementation_git_commit": (
                implementation_commit
            ),
            "source_training_run_id": (
                args.run_id
            ),
            "checkpoint": {
                "training_step": (
                    context.checkpoint_step
                ),
                "phase": context.checkpoint_phase,
                "path": relative_path(
                    repository,
                    context.checkpoint_path,
                ),
                "checkpoint_sha256": (
                    context.checkpoint_sha256
                ),
                "model_state_sha256": (
                    context.model_state_sha256
                ),
            },
            "source_artifacts": {
                name: {
                    "path": relative_path(
                        repository,
                        path,
                    ),
                    "sha256": (
                        input_hashes[name]
                    ),
                }
                for name, path in (
                    (
                        "checkpoint_manifest",
                        resolved_inputs[
                            "checkpoint_manifest"
                        ],
                    ),
                    (
                        "stage8_manifest",
                        resolved_inputs[
                            "stage8_manifest"
                        ],
                    ),
                    (
                        "stage9_manifest",
                        resolved_inputs[
                            "stage9_manifest"
                        ],
                    ),
                    (
                        "stage9_table",
                        resolved_inputs[
                            "stage9_table"
                        ],
                    ),
                    (
                        "stage9_archive",
                        resolved_inputs[
                            "stage9_archive"
                        ],
                    ),
                    (
                        "stage11_archive",
                        resolved_inputs[
                            "stage11_archive"
                        ],
                    ),
                    (
                        "primary_threshold_manifest",
                        resolved_inputs[
                            "primary_threshold_manifest"
                        ],
                    ),
                    (
                        "search_config",
                        resolved_inputs[
                            "search_config"
                        ],
                    ),
                )
            },
            "configuration": configuration,
            "execution": {
                "cutoff_order": [
                    float(value)
                    for value in cutoffs
                ],
                "completed_cell_count": len(
                    report_cells
                ),
                "command": sys.argv,
            },
            "c1_reproduction": c1_records,
            "negative_controls": {
                "control_count": len(
                    negative_control_results
                ),
                "all_passed": all(
                    result.validation_passed
                    for result in (
                        negative_control_results
                    )
                ),
                "scientific_family_results": 0,
            },
            "stress_tests": {
                "test_count": len(
                    stress_test_results
                ),
                "all_passed": all(
                    result.validation_passed
                    for result in (
                        stress_test_results
                    )
                ),
                "primary_scientific_family": False,
            },
            "control_execution": (
                control_execution.to_record()
            ),
            "integrity": {
                "clean_implementation_commit_verified": (
                    True
                ),
                "scientific_outputs_generated_from_exact_commit": (
                    implementation_commit
                ),
                "model_state_sha256_before": (
                    initial_model_state
                ),
                "model_state_sha256_after": (
                    final_model_state
                ),
                "model_state_unchanged": (
                    initial_model_state
                    == final_model_state
                ),
                "parameter_gradients_absent": True,
                "pre_grokking_family_search_run": (
                    False
                ),
                "transition_family_search_run": (
                    False
                ),
                "stage13_started": False,
                "runtime_excluded_from_deterministic_"
                "scientific_hashes": True,
            },
            "outputs": {
                **{
                    name: {
                        "path": relative_path(
                            repository,
                            path,
                        ),
                        "sha256": file_sha256(
                            path
                        ),
                        "included_in_deterministic_"
                        "scientific_hashes": True,
                    }
                    for name, path in (
                        deterministic_outputs.items()
                    )
                },
                **{
                    name: {
                        "path": relative_path(
                            repository,
                            path,
                        ),
                        "sha256": file_sha256(
                            path
                        ),
                        "included_in_deterministic_"
                        "scientific_hashes": False,
                    }
                    for name, path in (
                        runtime_bearing_outputs.items()
                    )
                },
            },
            "software": {
                "python": sys.version,
                "packages": package_versions(),
            },
        }
        stable_json_write(
            outputs.manifest,
            manifest,
        )

        shutil.rmtree(
            outputs.raw_directory
        )

    except Exception:
        cleanup_outputs(outputs)
        raise

    print(f"stage12_run_id: {stage12_run_id}")
    print(
        "implementation_git_commit: "
        f"{implementation_commit}"
    )
    print(
        "completed_cell_count: "
        f"{len(report_cells)}"
    )
    print(
        "family_sizes: "
        + ", ".join(
            str(
                cell.execution.result.family_size
            )
            for cell in report_cells
        )
    )
    print(
        "manifest: "
        f"{relative_path(repository, outputs.manifest)}"
    )


if __name__ == "__main__":
    main()
