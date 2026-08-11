"""Validate and plan the frozen Stage 14 random-label analysis."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from circuit_families.analysis.fidelity_calibration import (
    write_csv_records,
)
from circuit_families.analysis.random_label_circuit_analysis import (
    ANALYSIS_CONFIGURATION_PATH,
    ANALYSIS_RUN_ID,
    FrozenStage14AnalysisConfiguration,
    file_sha256,
    fraction_from_record,
    load_frozen_analysis_configuration,
    load_random_label_checkpoint_context,
)
from circuit_families.data.input_subsets import SUBSET_NAMES
from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
)
from circuit_families.interpretability.sparse_search import (
    CheckpointSearchExecution,
    SparseSearchArtifacts,
    run_checkpoint_sparse_search,
    write_sparse_search_artifacts,
)

ExecutionMode = Literal["execute", "reference_primary"]
Workload = Literal[
    "primary_sparse",
    "primary_diversity",
    "fidelity_sensitivity",
    "distinctness_sensitivity",
    "global_family_transfer",
    "subset_discovery",
    "transfer_grouping",
]

SOURCE_RECORD_NAMES = (
    "training_manifest",
    "foundation_manifest",
    "checkpoint_selection_manifest",
    "checkpoint_table",
    "masking_validation_table",
    "primary_threshold_manifest",
    "stage12_search_configuration",
    "stage12_compute_projection",
    "experimental_protocol",
    "implementation_order",
    "dataset_archive",
)

STAGE15_ADMINISTRATIVE_PATHS = frozenset(
    {
        "manifests/stage15_no_generalisation_unavailable.json",
        "results/notes/stage15_no_generalisation_unavailable.md",
    }
)

OUTPUT_RECORD_NAMES = (
    "sparse_search_table",
    "family_summary_table",
    "circuits_table",
    "pairwise_overlap_table",
    "restart_table",
    "frontier_table",
    "fidelity_sensitivity_table",
    "distinctness_sensitivity_table",
    "transfer_table",
    "runtime_table",
    "analysis_note",
    "archive",
    "manifest",
    "raw_output_directory",
)


@dataclass(frozen=True)
class Stage14AnalysisCell:
    """One immutable workload cell in the frozen execution plan."""

    sequence_index: int
    cell_id: str
    workload: Workload
    execution_mode: ExecutionMode
    checkpoint_index: int
    checkpoint_step: int
    fidelity_threshold: Fraction | None = None
    distinctness_cutoff: Fraction | None = None
    discovery_subset: str | None = None
    grouping_tolerance: Fraction | None = None
    dependency_cell_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return a deterministic serialisable representation."""

        return {
            "sequence_index": self.sequence_index,
            "cell_id": self.cell_id,
            "workload": self.workload,
            "execution_mode": self.execution_mode,
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_step": self.checkpoint_step,
            "fidelity_threshold": (
                None
                if self.fidelity_threshold is None
                else {
                    "numerator": self.fidelity_threshold.numerator,
                    "denominator": self.fidelity_threshold.denominator,
                    "float": float(self.fidelity_threshold),
                }
            ),
            "distinctness_cutoff": (
                None
                if self.distinctness_cutoff is None
                else {
                    "numerator": self.distinctness_cutoff.numerator,
                    "denominator": self.distinctness_cutoff.denominator,
                    "float": float(self.distinctness_cutoff),
                }
            ),
            "discovery_subset": self.discovery_subset,
            "grouping_tolerance": (
                None
                if self.grouping_tolerance is None
                else {
                    "numerator": self.grouping_tolerance.numerator,
                    "denominator": self.grouping_tolerance.denominator,
                    "float": float(self.grouping_tolerance),
                }
            ),
            "dependency_cell_id": self.dependency_cell_id,
        }


@dataclass(frozen=True)
class Stage14ExecutionPlan:
    """Complete frozen Stage 14 execution plan."""

    analysis_run_id: str
    cells: tuple[Stage14AnalysisCell, ...]

    @property
    def execute_cell_count(self) -> int:
        """Return the number of cells that require fresh execution."""

        return sum(
            cell.execution_mode == "execute"
            for cell in self.cells
        )

    @property
    def reference_cell_count(self) -> int:
        """Return the number of duplicate primary references."""

        return sum(
            cell.execution_mode == "reference_primary"
            for cell in self.cells
        )

    def workload_count(self, workload: Workload) -> int:
        """Count plan cells belonging to one workload."""

        return sum(
            cell.workload == workload
            for cell in self.cells
        )

    def records(self) -> list[dict[str, Any]]:
        """Return all cells in frozen sequence order."""

        return [
            cell.to_record()
            for cell in self.cells
        ]


@dataclass(frozen=True)
class Stage14OutputContract:
    """Frozen relative output paths for the integrated analysis."""

    records: tuple[tuple[str, Path], ...]

    def resolve(
        self,
        output_root: str | Path,
    ) -> tuple[tuple[str, Path], ...]:
        """Resolve every output beneath one selected artifact root."""

        root = Path(output_root).resolve()

        return tuple(
            (
                name,
                (
                    file_name
                    if file_name.is_absolute()
                    else root / file_name
                ).resolve(),
            )
            for name, file_name in self.records
        )

    def as_mapping(self) -> dict[str, str]:
        """Return the frozen relative output mapping."""

        return {
            name: str(file_name)
            for name, file_name in self.records
        }


@dataclass(frozen=True)
class Stage14ValidationReport:
    """Read-only validation result for one intended execution root."""

    analysis_run_id: str
    current_commit: str
    repository_clean: bool
    configuration_sha256: str
    input_root: Path
    output_root: Path
    verified_sources: tuple[tuple[str, Path, str], ...]
    verified_checkpoints: tuple[tuple[int, Path, str], ...]
    existing_outputs: tuple[tuple[str, Path], ...]
    stage15_artifacts: tuple[Path, ...]
    execution_plan: Stage14ExecutionPlan

    def to_record(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serialisable validation record."""

        return {
            "analysis_run_id": self.analysis_run_id,
            "current_commit": self.current_commit,
            "repository_clean": self.repository_clean,
            "configuration_sha256": self.configuration_sha256,
            "input_root": str(self.input_root),
            "output_root": str(self.output_root),
            "verified_sources": [
                {
                    "source_name": name,
                    "path": str(file_name),
                    "sha256": digest,
                }
                for name, file_name, digest in self.verified_sources
            ],
            "verified_checkpoints": [
                {
                    "checkpoint_step": checkpoint_step,
                    "path": str(file_name),
                    "sha256": digest,
                }
                for (
                    checkpoint_step,
                    file_name,
                    digest,
                ) in self.verified_checkpoints
            ],
            "existing_outputs": [
                {
                    "output_name": name,
                    "path": str(file_name),
                }
                for name, file_name in self.existing_outputs
            ],
            "stage15_artifacts": [
                str(file_name)
                for file_name in self.stage15_artifacts
            ],
            "execution_plan": {
                "cell_count": len(self.execution_plan.cells),
                "execute_cell_count": (
                    self.execution_plan.execute_cell_count
                ),
                "reference_cell_count": (
                    self.execution_plan.reference_cell_count
                ),
                "cells": self.execution_plan.records(),
            },
        }


def _git_output(
    repository_root: Path,
    *arguments: str,
) -> str:
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository_root),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def current_git_commit(
    repository_root: str | Path,
) -> str:
    """Return the current exact Git commit."""

    return _git_output(
        Path(repository_root).resolve(),
        "rev-parse",
        "HEAD",
    )


def repository_is_clean(
    repository_root: str | Path,
) -> bool:
    """Return whether tracked, staged and untracked status is empty."""

    return (
        _git_output(
            Path(repository_root).resolve(),
            "status",
            "--short",
        )
        == ""
    )


def _resolve_relative(
    root: Path,
    value: str | Path,
) -> Path:
    file_name = Path(value)

    if file_name.is_absolute():
        return file_name.resolve()

    return (root / file_name).resolve()


def _threshold_tag(value: Fraction) -> str:
    scaled = value * 10_000

    if scaled.denominator != 1:
        raise ValueError(
            f"Threshold cannot be represented in basis points: {value}"
        )

    return f"{scaled.numerator:05d}"


def _append_cell(
    cells: list[Stage14AnalysisCell],
    *,
    cell_id: str,
    workload: Workload,
    execution_mode: ExecutionMode,
    checkpoint_index: int,
    checkpoint_step: int,
    fidelity_threshold: Fraction | None = None,
    distinctness_cutoff: Fraction | None = None,
    discovery_subset: str | None = None,
    grouping_tolerance: Fraction | None = None,
    dependency_cell_id: str | None = None,
) -> None:
    cells.append(
        Stage14AnalysisCell(
            sequence_index=len(cells) + 1,
            cell_id=cell_id,
            workload=workload,
            execution_mode=execution_mode,
            checkpoint_index=checkpoint_index,
            checkpoint_step=checkpoint_step,
            fidelity_threshold=fidelity_threshold,
            distinctness_cutoff=distinctness_cutoff,
            discovery_subset=discovery_subset,
            grouping_tolerance=grouping_tolerance,
            dependency_cell_id=dependency_cell_id,
        )
    )


def build_execution_plan(
    configuration: FrozenStage14AnalysisConfiguration,
) -> Stage14ExecutionPlan:
    """Build the exact pre-results Stage 14 workload matrix."""

    payload = configuration.payload
    primary_cells = payload[
        "execution_matrix"
    ]["primary_dynamics"]["cells"]

    primary_threshold = fraction_from_record(
        payload["fidelity"]["primary_threshold"]
    )
    primary_cutoff = fraction_from_record(
        payload["distinctness"]["primary_cutoff"]
    )
    transfer_tolerances = tuple(
        fraction_from_record(record)
        for record in payload[
            "transfer"
        ]["grouping_sensitivity_grid"]
    )

    cells: list[Stage14AnalysisCell] = []

    for primary_cell in primary_cells:
        checkpoint_index = int(
            primary_cell["checkpoint_index"]
        )
        checkpoint_step = int(
            primary_cell["checkpoint_step"]
        )

        _append_cell(
            cells,
            cell_id=(
                "primary-sparse-"
                f"step-{checkpoint_step:08d}"
            ),
            workload="primary_sparse",
            execution_mode="execute",
            checkpoint_index=checkpoint_index,
            checkpoint_step=checkpoint_step,
            fidelity_threshold=primary_threshold,
        )

    for primary_cell in primary_cells:
        checkpoint_index = int(
            primary_cell["checkpoint_index"]
        )
        checkpoint_step = int(
            primary_cell["checkpoint_step"]
        )
        sparse_cell_id = (
            "primary-sparse-"
            f"step-{checkpoint_step:08d}"
        )

        _append_cell(
            cells,
            cell_id=(
                "primary-family-"
                f"step-{checkpoint_step:08d}"
            ),
            workload="primary_diversity",
            execution_mode="execute",
            checkpoint_index=checkpoint_index,
            checkpoint_step=checkpoint_step,
            fidelity_threshold=primary_threshold,
            distinctness_cutoff=primary_cutoff,
            dependency_cell_id=sparse_cell_id,
        )

    fidelity_cells = payload[
        "execution_matrix"
    ]["fidelity_sensitivity"]["cells"]

    for fidelity_cell in fidelity_cells:
        threshold = fraction_from_record(
            fidelity_cell["fidelity_threshold"]
        )
        checkpoint_step = int(
            fidelity_cell["checkpoint_step"]
        )
        primary_reference = bool(
            fidelity_cell["primary_cell_reference"]
        )

        _append_cell(
            cells,
            cell_id=(
                "fidelity-sensitivity-"
                f"step-{checkpoint_step:08d}-"
                f"threshold-{_threshold_tag(threshold)}"
            ),
            workload="fidelity_sensitivity",
            execution_mode=(
                "reference_primary"
                if primary_reference
                else "execute"
            ),
            checkpoint_index=6,
            checkpoint_step=checkpoint_step,
            fidelity_threshold=threshold,
            distinctness_cutoff=primary_cutoff,
            dependency_cell_id=(
                "primary-family-"
                f"step-{checkpoint_step:08d}"
                if primary_reference
                else None
            ),
        )

    distinctness_cells = payload[
        "execution_matrix"
    ]["distinctness_sensitivity"]["cells"]

    for distinctness_cell in distinctness_cells:
        cutoff = fraction_from_record(
            distinctness_cell["jaccard_cutoff"]
        )
        checkpoint_step = int(
            distinctness_cell["checkpoint_step"]
        )
        primary_reference = bool(
            distinctness_cell["primary_cell_reference"]
        )

        _append_cell(
            cells,
            cell_id=(
                "distinctness-sensitivity-"
                f"step-{checkpoint_step:08d}-"
                f"cutoff-{_threshold_tag(cutoff)}"
            ),
            workload="distinctness_sensitivity",
            execution_mode=(
                "reference_primary"
                if primary_reference
                else "execute"
            ),
            checkpoint_index=6,
            checkpoint_step=checkpoint_step,
            fidelity_threshold=primary_threshold,
            distinctness_cutoff=cutoff,
            dependency_cell_id=(
                "primary-family-"
                f"step-{checkpoint_step:08d}"
                if primary_reference
                else None
            ),
        )

    for primary_cell in primary_cells:
        checkpoint_index = int(
            primary_cell["checkpoint_index"]
        )
        checkpoint_step = int(
            primary_cell["checkpoint_step"]
        )
        family_cell_id = (
            "primary-family-"
            f"step-{checkpoint_step:08d}"
        )

        _append_cell(
            cells,
            cell_id=(
                "global-family-transfer-"
                f"step-{checkpoint_step:08d}"
            ),
            workload="global_family_transfer",
            execution_mode="execute",
            checkpoint_index=checkpoint_index,
            checkpoint_step=checkpoint_step,
            fidelity_threshold=primary_threshold,
            distinctness_cutoff=primary_cutoff,
            dependency_cell_id=family_cell_id,
        )

    for primary_cell in primary_cells:
        checkpoint_index = int(
            primary_cell["checkpoint_index"]
        )
        checkpoint_step = int(
            primary_cell["checkpoint_step"]
        )

        for subset_name in SUBSET_NAMES:
            _append_cell(
                cells,
                cell_id=(
                    "subset-discovery-"
                    f"step-{checkpoint_step:08d}-"
                    f"{subset_name.lower()}"
                ),
                workload="subset_discovery",
                execution_mode="execute",
                checkpoint_index=checkpoint_index,
                checkpoint_step=checkpoint_step,
                fidelity_threshold=primary_threshold,
                discovery_subset=subset_name,
            )

    for primary_cell in primary_cells:
        checkpoint_index = int(
            primary_cell["checkpoint_index"]
        )
        checkpoint_step = int(
            primary_cell["checkpoint_step"]
        )
        transfer_cell_id = (
            "global-family-transfer-"
            f"step-{checkpoint_step:08d}"
        )

        for tolerance in transfer_tolerances:
            _append_cell(
                cells,
                cell_id=(
                    "transfer-grouping-"
                    f"step-{checkpoint_step:08d}-"
                    f"tolerance-{_threshold_tag(tolerance)}"
                ),
                workload="transfer_grouping",
                execution_mode="execute",
                checkpoint_index=checkpoint_index,
                checkpoint_step=checkpoint_step,
                grouping_tolerance=tolerance,
                dependency_cell_id=transfer_cell_id,
            )

    plan = Stage14ExecutionPlan(
        analysis_run_id=configuration.analysis_run_id,
        cells=tuple(cells),
    )

    expected_counts = {
        "primary_sparse": 7,
        "primary_diversity": 7,
        "fidelity_sensitivity": 6,
        "distinctness_sensitivity": 3,
        "global_family_transfer": 7,
        "subset_discovery": 28,
        "transfer_grouping": 21,
    }

    for workload, expected_count in expected_counts.items():
        observed_count = plan.workload_count(workload)

        if observed_count != expected_count:
            raise ValueError(
                f"{workload} cell count mismatch: "
                f"expected {expected_count}, found {observed_count}."
            )

    if len(plan.cells) != 79:
        raise ValueError(
            f"Execution-plan size mismatch: {len(plan.cells)}."
        )

    identifiers = [
        cell.cell_id
        for cell in plan.cells
    ]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "Execution-plan cell identifiers are not unique."
        )

    if tuple(
        cell.sequence_index
        for cell in plan.cells
    ) != tuple(range(1, len(plan.cells) + 1)):
        raise ValueError(
            "Execution-plan sequence indices are not contiguous."
        )

    if plan.reference_cell_count != 2:
        raise ValueError(
            "Expected exactly two duplicate primary reference cells."
        )

    return plan


def output_contract(
    configuration: FrozenStage14AnalysisConfiguration,
) -> Stage14OutputContract:
    """Load the exact output contract from the frozen configuration."""

    output_payload = configuration.payload.get("outputs")

    if not isinstance(output_payload, Mapping):
        raise TypeError("Frozen output contract is missing.")

    observed_names = tuple(output_payload)

    if observed_names != OUTPUT_RECORD_NAMES:
        raise ValueError(
            "Frozen output record names or ordering changed: "
            f"{observed_names!r}."
        )

    records = []

    for name in OUTPUT_RECORD_NAMES:
        value = output_payload[name]

        if not isinstance(value, str) or not value:
            raise TypeError(
                f"Frozen output path is invalid: {name}"
            )

        records.append((name, Path(value)))

    return Stage14OutputContract(
        records=tuple(records)
    )


def existing_output_paths(
    contract: Stage14OutputContract,
    *,
    output_root: str | Path,
) -> tuple[tuple[str, Path], ...]:
    """Return frozen output locations that already physically exist."""

    return tuple(
        (name, file_name)
        for name, file_name in contract.resolve(output_root)
        if file_name.exists()
    )


def find_stage15_artifacts(
    artifact_root: str | Path,
) -> tuple[Path, ...]:
    """Return any physical Stage 15 artifact beneath standard roots."""

    root = Path(artifact_root).resolve()
    matches = []

    for relative_root in (
        Path("results"),
        Path("manifests"),
        Path("figures"),
    ):
        search_root = root / relative_root

        if not search_root.exists():
            continue

        for file_name in search_root.rglob("*"):
            if (
                file_name.is_file()
                and "stage15"
                in file_name.name.lower().replace("-", "")
                and file_name.relative_to(root).as_posix()
                not in STAGE15_ADMINISTRATIVE_PATHS
            ):
                matches.append(file_name.resolve())

    return tuple(sorted(set(matches)))


def _verified_source_records(
    configuration: FrozenStage14AnalysisConfiguration,
    *,
    input_root: Path,
) -> tuple[tuple[str, Path, str], ...]:
    source = configuration.payload["source"]
    records = []

    for source_name in SOURCE_RECORD_NAMES:
        source_record = source.get(source_name)

        if not isinstance(source_record, Mapping):
            raise TypeError(
                f"Frozen source record is missing: {source_name}"
            )

        relative_name = source_record.get("path")
        expected_sha256 = source_record.get("sha256")

        if not isinstance(relative_name, str):
            raise TypeError(
                f"Frozen source path is invalid: {source_name}"
            )

        if not isinstance(expected_sha256, str):
            raise TypeError(
                f"Frozen source hash is invalid: {source_name}"
            )

        file_name = _resolve_relative(
            input_root,
            relative_name,
        )

        if not file_name.is_file():
            raise FileNotFoundError(file_name)

        observed_sha256 = file_sha256(file_name)

        if observed_sha256 != expected_sha256 and not _is_attested_administrative_document_update(
            source_name=source_name,
            relative_name=relative_name,
            expected_sha256=expected_sha256,
            observed_sha256=observed_sha256,
            input_root=input_root,
        ):
            raise ValueError(
                f"Source hash mismatch for {source_name}: "
                f"expected {expected_sha256}, "
                f"found {observed_sha256}."
            )

        records.append(
            (
                source_name,
                file_name,
                observed_sha256,
            )
        )

    return tuple(records)


def _is_attested_administrative_document_update(
    *,
    source_name: str,
    relative_name: str,
    expected_sha256: str,
    observed_sha256: str,
    input_root: Path,
) -> bool:
    """Accept the exact administrative document chain through Stage 18 pre-results."""

    if source_name not in {"experimental_protocol", "implementation_order"}:
        return False

    resolution_path = input_root / "manifests/stage15_no_generalisation_unavailable.json"
    if not resolution_path.is_file():
        return False

    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    documents = resolution.get("source_document_hashes")
    if resolution.get("status") != "unavailable" or not isinstance(documents, Mapping):
        return False

    record = documents.get(source_name)
    if not (
        isinstance(record, Mapping)
        and record.get("path") == relative_name
        and record.get("pre_resolution_sha256") == expected_sha256
    ):
        return False

    stage15_sha256 = record.get("post_resolution_sha256")
    if stage15_sha256 == observed_sha256:
        return True
    if not isinstance(stage15_sha256, str):
        return False

    freeze_path = (
        input_root
        / "manifests/post_stage17_checkpoint_grid_and_concurrency_freeze.json"
    )
    if not freeze_path.is_file():
        return False
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    lifecycle = freeze.get("lifecycle")
    freeze_documents = freeze.get("source_document_hashes")
    if not (
        freeze.get("record_type")
        == "administrative_checkpoint_grid_and_concurrency_freeze"
        and isinstance(lifecycle, Mapping)
        and lifecycle.get("checkpoint_grid_decision_made") is True
        and lifecycle.get("stage18_started") is False
        and isinstance(freeze_documents, Mapping)
    ):
        return False

    freeze_record = freeze_documents.get(source_name)
    if not (
        isinstance(freeze_record, Mapping)
        and freeze_record.get("path") == relative_name
        and freeze_record.get("pre_freeze_sha256") == stage15_sha256
    ):
        return False
    freeze_sha256 = freeze_record.get("post_freeze_sha256")
    if freeze_sha256 == observed_sha256:
        return True
    if not isinstance(freeze_sha256, str):
        return False

    control_path = (
        input_root
        / "manifests/post_stage17_additional_control_seed_count_freeze.json"
    )
    if not control_path.is_file():
        return False
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control_lifecycle = control.get("lifecycle")
    control_decision = control.get("decision")
    control_documents = control.get("source_document_hashes")
    if not (
        control.get("record_type")
        == "administrative_additional_control_seed_count_freeze"
        and isinstance(control_lifecycle, Mapping)
        and control_lifecycle.get("stage18_scientific_outputs_visible") is False
        and isinstance(control_decision, Mapping)
        and control_decision.get("additional_random_label_seed_count") == 0
        and isinstance(control_documents, Mapping)
    ):
        return False
    control_record = control_documents.get(source_name)
    return bool(
        isinstance(control_record, Mapping)
        and control_record.get("path") == relative_name
        and control_record.get("pre_decision_sha256") == freeze_sha256
        and control_record.get("post_decision_sha256") == observed_sha256
    )


def _verified_checkpoint_records(
    configuration: FrozenStage14AnalysisConfiguration,
    *,
    input_root: Path,
    verify_checkpoint_hashes: bool,
) -> tuple[tuple[int, Path, str], ...]:
    primary_cells = configuration.payload[
        "execution_matrix"
    ]["primary_dynamics"]["cells"]
    records = []

    for cell in primary_cells:
        checkpoint_step = int(cell["checkpoint_step"])
        relative_name = cell["checkpoint_path"]
        expected_sha256 = cell["checkpoint_sha256"]

        if not isinstance(relative_name, str):
            raise TypeError(
                f"Checkpoint path is invalid at step {checkpoint_step}."
            )

        if not isinstance(expected_sha256, str):
            raise TypeError(
                f"Checkpoint hash is invalid at step {checkpoint_step}."
            )

        file_name = _resolve_relative(
            input_root,
            relative_name,
        )

        if not file_name.is_file():
            raise FileNotFoundError(file_name)

        if verify_checkpoint_hashes:
            observed_sha256 = file_sha256(file_name)

            if observed_sha256 != expected_sha256:
                raise ValueError(
                    f"Checkpoint hash mismatch at step "
                    f"{checkpoint_step}: expected {expected_sha256}, "
                    f"found {observed_sha256}."
                )
        else:
            observed_sha256 = expected_sha256

        records.append(
            (
                checkpoint_step,
                file_name,
                observed_sha256,
            )
        )

    return tuple(records)


def validate_analysis_inputs(
    *,
    repository_root: str | Path,
    expected_implementation_commit: str,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
    require_clean_repository: bool = True,
    require_outputs_absent: bool = True,
    verify_checkpoint_hashes: bool = True,
) -> Stage14ValidationReport:
    """Validate all frozen inputs without creating or changing outputs."""

    repository = Path(repository_root).resolve()
    selected_input_root = (
        repository
        if input_root is None
        else Path(input_root).resolve()
    )
    selected_output_root = (
        repository
        if output_root is None
        else Path(output_root).resolve()
    )

    if not repository.is_dir():
        raise NotADirectoryError(repository)

    if not selected_input_root.is_dir():
        raise NotADirectoryError(selected_input_root)

    if (
        not isinstance(expected_implementation_commit, str)
        or not expected_implementation_commit
    ):
        raise ValueError(
            "expected_implementation_commit must not be empty."
        )

    current_commit = current_git_commit(repository)

    if current_commit != expected_implementation_commit:
        raise ValueError(
            "Implementation commit mismatch: "
            f"expected {expected_implementation_commit}, "
            f"found {current_commit}."
        )

    clean = repository_is_clean(repository)

    if require_clean_repository and not clean:
        raise ValueError(
            "Repository must be clean before Stage 14 execution."
        )

    configuration = load_frozen_analysis_configuration(
        repository_root=repository,
        configuration_path=ANALYSIS_CONFIGURATION_PATH,
    )
    plan = build_execution_plan(configuration)
    contract = output_contract(configuration)

    verified_sources = _verified_source_records(
        configuration,
        input_root=selected_input_root,
    )
    verified_checkpoints = _verified_checkpoint_records(
        configuration,
        input_root=selected_input_root,
        verify_checkpoint_hashes=verify_checkpoint_hashes,
    )
    existing_outputs = existing_output_paths(
        contract,
        output_root=selected_output_root,
    )

    if require_outputs_absent and existing_outputs:
        rendered = ", ".join(
            f"{name}={file_name}"
            for name, file_name in existing_outputs
        )
        raise FileExistsError(
            "Frozen Stage 14 outputs already exist: "
            f"{rendered}"
        )

    stage15_artifacts = find_stage15_artifacts(
        selected_output_root
    )

    if stage15_artifacts:
        rendered = ", ".join(
            str(file_name)
            for file_name in stage15_artifacts
        )
        raise FileExistsError(
            f"Stage 15 artifacts already exist: {rendered}"
        )

    if configuration.analysis_run_id != ANALYSIS_RUN_ID:
        raise ValueError(
            "Frozen Stage 14 analysis run identity changed."
        )

    return Stage14ValidationReport(
        analysis_run_id=configuration.analysis_run_id,
        current_commit=current_commit,
        repository_clean=clean,
        configuration_sha256=configuration.sha256,
        input_root=selected_input_root,
        output_root=selected_output_root,
        verified_sources=verified_sources,
        verified_checkpoints=verified_checkpoints,
        existing_outputs=existing_outputs,
        stage15_artifacts=stage15_artifacts,
        execution_plan=plan,
    )

PRIMARY_SPARSE_COLUMNS = (
    "analysis_run_id",
    "source_training_run_id",
    "workload",
    "cell_id",
    "checkpoint_index",
    "checkpoint_phase",
    "checkpoint_step",
    "checkpoint_sha256",
    "fidelity_threshold_numerator",
    "fidelity_threshold_denominator",
    "fidelity_threshold",
    "search_status",
    "retained_attention_head_count",
    "retained_mlp_neuron_count",
    "retained_component_count",
    "retained_component_proportion",
    "primary_fidelity",
    "prediction_agreement_count",
    "evaluated_example_count",
    "masked_accuracy",
    "masked_cross_entropy",
    "mean_kl_divergence",
    "mean_jensen_shannon_divergence",
    "maximum_absolute_logit_difference",
    "accepted_removal_count",
    "exact_evaluation_budget",
    "exact_evaluations_used",
    "ranking_passes_used",
    "candidate_batches_tested",
    "rejected_candidate_count",
    "budget_remaining",
    "budget_exhausted",
    "locally_single_deletion_minimal",
    "meaningfully_sparse",
    "stopping_reason",
    "failure_detail",
    "pseudo_target_sha256",
    "full_model_reference_sha256",
    "model_state_sha256_before",
    "model_state_sha256_after",
    "hooks_unchanged",
    "raw_cell_directory",
    "final_mask_path",
    "final_mask_sha256",
    "accepted_removal_trajectory_path",
    "accepted_removal_trajectory_sha256",
    "candidate_evaluation_log_path",
    "candidate_evaluation_log_sha256",
    "cell_summary_path",
    "cell_summary_sha256",
    "hashes_path",
    "hashes_sha256",
)

RUNTIME_COLUMNS = (
    "analysis_run_id",
    "workload",
    "cell_id",
    "checkpoint_index",
    "checkpoint_step",
    "fidelity_threshold",
    "distinctness_cutoff",
    "discovery_subset",
    "grouping_tolerance",
    "record_type",
    "exact_evaluations_used",
    "elapsed_seconds",
    "included_in_deterministic_scientific_hashes",
)


@dataclass(frozen=True)
class PrimarySparseCellExecution:
    """Completed primary sparse search and its written artifacts."""

    cell: Stage14AnalysisCell
    source_context: Any
    execution: CheckpointSearchExecution
    artifacts: SparseSearchArtifacts
    elapsed_seconds: float


@dataclass(frozen=True)
class PrimarySparseWorkloadResult:
    """Outputs created by the seven primary sparse searches."""

    analysis_run_id: str
    implementation_commit: str
    raw_output_directory: Path
    sparse_search_table: Path
    runtime_table: Path
    cells: tuple[PrimarySparseCellExecution, ...]


def adapt_random_label_search_context(
    source_context: Any,
) -> CheckpointEvaluationContext:
    """Project the validated random-label context into the search type."""

    if isinstance(source_context, CheckpointEvaluationContext):
        return source_context

    target_fields = tuple(
        field.name
        for field in fields(CheckpointEvaluationContext)
    )
    missing = tuple(
        field_name
        for field_name in target_fields
        if not hasattr(source_context, field_name)
    )

    if missing:
        raise TypeError(
            "Random-label context cannot be adapted; "
            f"missing fields: {missing!r}."
        )

    adapted = CheckpointEvaluationContext(
        **{
            field_name: getattr(
                source_context,
                field_name,
            )
            for field_name in target_fields
        }
    )

    if adapted.model is not source_context.model:
        raise RuntimeError(
            "Context adaptation changed model object identity."
        )

    if adapted.inputs is not source_context.inputs:
        raise RuntimeError(
            "Context adaptation changed input tensor identity."
        )

    if adapted.targets is not source_context.targets:
        raise RuntimeError(
            "Context adaptation changed target tensor identity."
        )

    if (
        adapted.model_state_sha256
        != source_context.model_state_sha256
    ):
        raise RuntimeError(
            "Context adaptation changed the model-state hash."
        )

    return adapted


def primary_sparse_cells(
    plan: Stage14ExecutionPlan,
) -> tuple[Stage14AnalysisCell, ...]:
    """Return the seven primary sparse cells in frozen order."""

    cells = tuple(
        cell
        for cell in plan.cells
        if cell.workload == "primary_sparse"
    )

    if len(cells) != 7:
        raise ValueError(
            f"Expected seven primary sparse cells, found {len(cells)}."
        )

    if any(
        cell.execution_mode != "execute"
        for cell in cells
    ):
        raise ValueError(
            "Every primary sparse cell must require fresh execution."
        )

    if tuple(
        cell.checkpoint_step
        for cell in cells
    ) != (
        200,
        3_400,
        7_450,
        8_150,
        8_500,
        8_650,
        9_050,
    ):
        raise ValueError(
            "Primary sparse checkpoint order differs from the freeze."
        )

    return cells


def _relative_output_path(
    *,
    output_root: Path,
    file_name: Path,
) -> str:
    try:
        return file_name.resolve().relative_to(
            output_root.resolve()
        ).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Output path escapes output root: {file_name}"
        ) from error


def primary_sparse_table_row(
    *,
    analysis_run_id: str,
    output_root: Path,
    result: PrimarySparseCellExecution,
) -> dict[str, Any]:
    """Return one deterministic primary sparse report row."""

    cell = result.cell
    source_context = result.source_context
    execution = result.execution
    search = execution.result
    metrics = search.final_metrics
    mask = search.final_mask
    artifacts = result.artifacts
    threshold = cell.fidelity_threshold

    if threshold is None:
        raise ValueError(
            "Primary sparse cell has no fidelity threshold."
        )

    return {
        "analysis_run_id": analysis_run_id,
        "source_training_run_id": source_context.run_id,
        "workload": cell.workload,
        "cell_id": cell.cell_id,
        "checkpoint_index": cell.checkpoint_index,
        "checkpoint_phase": source_context.checkpoint_phase,
        "checkpoint_step": cell.checkpoint_step,
        "checkpoint_sha256": source_context.checkpoint_sha256,
        "fidelity_threshold_numerator": threshold.numerator,
        "fidelity_threshold_denominator": threshold.denominator,
        "fidelity_threshold": float(threshold),
        "search_status": search.status,
        "retained_attention_head_count": (
            mask.retained_attention_head_count
        ),
        "retained_mlp_neuron_count": (
            mask.retained_mlp_neuron_count
        ),
        "retained_component_count": (
            mask.retained_component_count
        ),
        "retained_component_proportion": (
            mask.retained_component_proportion
        ),
        "primary_fidelity": metrics.primary_fidelity,
        "prediction_agreement_count": (
            metrics.prediction_agreement_count
        ),
        "evaluated_example_count": (
            metrics.evaluated_example_count
        ),
        "masked_accuracy": metrics.masked_accuracy,
        "masked_cross_entropy": metrics.masked_cross_entropy,
        "mean_kl_divergence": (
            metrics.mean_kl_divergence
        ),
        "mean_jensen_shannon_divergence": (
            metrics.mean_jensen_shannon_divergence
        ),
        "maximum_absolute_logit_difference": (
            metrics.maximum_absolute_logit_difference
        ),
        "accepted_removal_count": len(
            search.accepted_removals
        ),
        "exact_evaluation_budget": (
            search.exact_evaluation_budget
        ),
        "exact_evaluations_used": (
            search.exact_evaluations_used
        ),
        "ranking_passes_used": (
            search.ranking_passes_used
        ),
        "candidate_batches_tested": (
            search.candidate_batches_tested
        ),
        "rejected_candidate_count": (
            search.rejected_candidate_count
        ),
        "budget_remaining": search.budget_remaining,
        "budget_exhausted": search.budget_exhausted,
        "locally_single_deletion_minimal": (
            search.locally_single_deletion_minimal
        ),
        "meaningfully_sparse": search.meaningfully_sparse,
        "stopping_reason": search.stopping_reason,
        "failure_detail": (
            ""
            if search.failure_detail is None
            else search.failure_detail
        ),
        "pseudo_target_sha256": (
            execution.pseudo_target_sha256
        ),
        "full_model_reference_sha256": (
            execution.full_model_reference_sha256
        ),
        "model_state_sha256_before": (
            execution.model_state_sha256_before
        ),
        "model_state_sha256_after": (
            execution.model_state_sha256_after
        ),
        "hooks_unchanged": (
            execution.hook_counts_before
            == execution.hook_counts_after
        ),
        "raw_cell_directory": _relative_output_path(
            output_root=output_root,
            file_name=artifacts.output_directory,
        ),
        "final_mask_path": _relative_output_path(
            output_root=output_root,
            file_name=artifacts.final_mask_path,
        ),
        "final_mask_sha256": (
            artifacts.final_mask_sha256
        ),
        "accepted_removal_trajectory_path": (
            _relative_output_path(
                output_root=output_root,
                file_name=(
                    artifacts
                    .accepted_removal_trajectory_path
                ),
            )
        ),
        "accepted_removal_trajectory_sha256": (
            artifacts
            .accepted_removal_trajectory_sha256
        ),
        "candidate_evaluation_log_path": (
            _relative_output_path(
                output_root=output_root,
                file_name=(
                    artifacts
                    .candidate_evaluation_log_path
                ),
            )
        ),
        "candidate_evaluation_log_sha256": (
            artifacts
            .candidate_evaluation_log_sha256
        ),
        "cell_summary_path": _relative_output_path(
            output_root=output_root,
            file_name=artifacts.cell_summary_path,
        ),
        "cell_summary_sha256": (
            artifacts.cell_summary_sha256
        ),
        "hashes_path": _relative_output_path(
            output_root=output_root,
            file_name=artifacts.hashes_path,
        ),
        "hashes_sha256": artifacts.hashes_sha256,
    }


def primary_sparse_runtime_row(
    *,
    analysis_run_id: str,
    result: PrimarySparseCellExecution,
) -> dict[str, Any]:
    """Return nondeterministic runtime telemetry for one sparse cell."""

    cell = result.cell

    if cell.fidelity_threshold is None:
        raise ValueError(
            "Primary sparse cell has no fidelity threshold."
        )

    return {
        "analysis_run_id": analysis_run_id,
        "workload": cell.workload,
        "cell_id": cell.cell_id,
        "checkpoint_index": cell.checkpoint_index,
        "checkpoint_step": cell.checkpoint_step,
        "fidelity_threshold": float(
            cell.fidelity_threshold
        ),
        "distinctness_cutoff": "",
        "discovery_subset": "",
        "grouping_tolerance": "",
        "record_type": "cell",
        "exact_evaluations_used": (
            result.execution.result.exact_evaluations_used
        ),
        "elapsed_seconds": result.elapsed_seconds,
        "included_in_deterministic_scientific_hashes": False,
    }


def execute_primary_sparse_workload(
    *,
    repository_root: str | Path,
    expected_implementation_commit: str,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
    device: str = "cpu",
    progress_callback: Callable[[str], None] | None = None,
) -> PrimarySparseWorkloadResult:
    """Run and serialize the seven frozen primary sparse searches."""

    if device not in {"cpu", "cuda"}:
        raise ValueError(
            "Stage 14 scientific execution supports only CPU or CUDA."
        )

    repository = Path(repository_root).resolve()
    selected_input_root = (
        repository
        if input_root is None
        else Path(input_root).resolve()
    )
    selected_output_root = (
        repository
        if output_root is None
        else Path(output_root).resolve()
    )

    validation = validate_analysis_inputs(
        repository_root=repository,
        expected_implementation_commit=(
            expected_implementation_commit
        ),
        input_root=selected_input_root,
        output_root=selected_output_root,
        require_clean_repository=True,
        require_outputs_absent=True,
        verify_checkpoint_hashes=True,
    )
    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    plan = validation.execution_plan
    cells = primary_sparse_cells(plan)
    search_configuration = configuration.payload["search"]
    output_mapping = dict(
        output_contract(configuration).resolve(
            selected_output_root
        )
    )

    raw_output_directory = output_mapping[
        "raw_output_directory"
    ]
    sparse_search_table = output_mapping[
        "sparse_search_table"
    ]
    runtime_table = output_mapping["runtime_table"]

    created_files = (
        sparse_search_table,
        runtime_table,
    )
    completed: list[PrimarySparseCellExecution] = []

    try:
        raw_output_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        for execution_index, cell in enumerate(
            cells,
            start=1,
        ):
            message = (
                f"[{execution_index:02d}/{len(cells):02d}] "
                f"primary_sparse checkpoint={cell.checkpoint_step}"
            )

            if progress_callback is not None:
                progress_callback(message)

            source_context = (
                load_random_label_checkpoint_context(
                    repository_root=repository,
                    configuration=configuration,
                    checkpoint_step=cell.checkpoint_step,
                    device=device,
                    output_root=selected_input_root,
                )
            )
            search_context = (
                adapt_random_label_search_context(
                    source_context
                )
            )

            if cell.fidelity_threshold is None:
                raise ValueError(
                    "Primary sparse cell has no threshold."
                )

            started = time.perf_counter()
            execution = run_checkpoint_sparse_search(
                search_context,
                fidelity_threshold=float(
                    cell.fidelity_threshold
                ),
                ranking_batch_size=int(
                    search_configuration[
                        "ranking_batch_size"
                    ]
                ),
                evaluation_batch_size=int(
                    search_configuration[
                        "evaluation_batch_size"
                    ]
                ),
                exact_evaluation_budget=int(
                    search_configuration[
                        "per_requested_circuit_exact_evaluations"
                    ]
                ),
            )
            elapsed_seconds = (
                time.perf_counter() - started
            )

            cell_directory = (
                raw_output_directory
                / "primary_sparse"
                / f"step_{cell.checkpoint_step:08d}"
            )
            artifacts = write_sparse_search_artifacts(
                cell_directory,
                execution.result,
                cell_metadata={
                    "analysis_run_id": (
                        configuration.analysis_run_id
                    ),
                    "analysis_identity_sha256": (
                        configuration
                        .analysis_identity_sha256
                    ),
                    "implementation_git_commit": (
                        expected_implementation_commit
                    ),
                    "workload": cell.workload,
                    "cell_id": cell.cell_id,
                    "sequence_index": cell.sequence_index,
                    "checkpoint_index": (
                        cell.checkpoint_index
                    ),
                    "checkpoint_step": (
                        cell.checkpoint_step
                    ),
                    "checkpoint_sha256": (
                        source_context.checkpoint_sha256
                    ),
                    "fidelity_threshold": float(
                        cell.fidelity_threshold
                    ),
                    "analysis_configuration_sha256": (
                        configuration.sha256
                    ),
                },
            )

            completed.append(
                PrimarySparseCellExecution(
                    cell=cell,
                    source_context=source_context,
                    execution=execution,
                    artifacts=artifacts,
                    elapsed_seconds=elapsed_seconds,
                )
            )

            if progress_callback is not None:
                progress_callback(
                    "completed "
                    f"checkpoint={cell.checkpoint_step} "
                    f"status={execution.result.status} "
                    "retained="
                    f"{execution.result.final_mask.retained_component_count} "
                    "exact_evaluations="
                    f"{execution.result.exact_evaluations_used}"
                )

        write_csv_records(
            sparse_search_table,
            fieldnames=PRIMARY_SPARSE_COLUMNS,
            rows=[
                primary_sparse_table_row(
                    analysis_run_id=(
                        configuration.analysis_run_id
                    ),
                    output_root=selected_output_root,
                    result=result,
                )
                for result in completed
            ],
        )
        write_csv_records(
            runtime_table,
            fieldnames=RUNTIME_COLUMNS,
            rows=[
                primary_sparse_runtime_row(
                    analysis_run_id=(
                        configuration.analysis_run_id
                    ),
                    result=result,
                )
                for result in completed
            ],
        )

        return PrimarySparseWorkloadResult(
            analysis_run_id=configuration.analysis_run_id,
            implementation_commit=(
                expected_implementation_commit
            ),
            raw_output_directory=raw_output_directory,
            sparse_search_table=sparse_search_table,
            runtime_table=runtime_table,
            cells=tuple(completed),
        )

    except Exception:
        shutil.rmtree(
            raw_output_directory,
            ignore_errors=True,
        )

        for file_name in created_files:
            file_name.unlink(missing_ok=True)

        raise
