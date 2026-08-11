"""Frozen registry and orchestration primitives for Stage 17."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tarfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Any, Literal

import yaml

from circuit_families.analysis.distinctness_sweep import (
    DISTINCTNESS_GRID,
    distinctness_display,
    validate_distinctness_cutoff,
)
from circuit_families.analysis.threshold_sweep import (
    FIDELITY_GRID,
    fidelity_display,
    validate_fidelity_threshold,
)
from circuit_families.analysis.transfer import TransferProfile
from circuit_families.config import mapping_hash
from circuit_families.interpretability.masks import ComponentMask
from circuit_families.interpretability.overlap_constraints import jaccard_counts
from circuit_families.training import file_sha256

SearchExecutionMode = Literal["fresh_execution", "reference_existing_result"]
TransferExecutionMode = Literal["fresh_execution", "reference_existing_result"]

MODEL_SEED = 1
CHECKPOINT_STEP = 9_050
CHECKPOINT_INDEX = 7
EXACT_EVALUATION_EXAMPLES = 12_769
MAXIMUM_RETAINED_COMPONENTS = 258
TOTAL_SEARCHABLE_COMPONENTS = 516
PER_REQUESTED_CIRCUIT_BUDGET = 10_000
PER_CELL_BUDGET = 50_000
FAMILY_TARGET = 10
PRIMARY_CELL_KEY = (Fraction(99, 100), Fraction(1, 2))
PRIMARY_TRANSFER_TOLERANCE = Fraction(1, 20)

STAGE12_RUN_ID = "stage12-diversity-s1-020ebf1b5814"
STAGE12_MANIFEST = "manifests/stage12_diversity_stage12-diversity-s1-020ebf1b5814.json"
STAGE12_FAMILY_SUMMARY = "results/tables/seed_1_stage12_family_summary.csv"
STAGE16_RUN_ID = "stage16-transfer-s1-cc55bd4162c8"
STAGE16_MANIFEST = "manifests/stage16_transfer_stage16-transfer-s1-cc55bd4162c8.json"
STAGE16_TRANSFER_PROFILES = "results/tables/seed_1_stage16_transfer_profiles.csv"
CONFIGURATION_PATH = Path("configs/stage17_sensitivity.json")
POST_STAGE17_FREEZE_OUTPUTS = frozenset(
    {
        "manifests/post_stage17_checkpoint_grid_and_concurrency_freeze.json",
        "results/notes/post_stage17_checkpoint_grid_and_concurrency_freeze.md",
        "results/tables/post_stage17_concurrency_benchmark_summary.csv",
        "results/tables/stage18_main_seed_registry_pre_execution.csv",
        "results/tables/stage18_cell_registry_pre_execution.csv",
        "results/tables/stage18_worker_shards_pre_execution.csv",
        *{
            f"manifests/stage18_worker_shards/worker_{index:02d}.json"
            for index in range(12)
        },
    }
)

EXPECTED_CHECKPOINT_SHA256 = "5b449db5ff9a62d5b621450c013bc25949499ed767e6db1723561a4e87ab8d70"
EXPECTED_MODEL_STATE_SHA256 = "18c66dc1802016a1bdf888070e87ebc615a98d7d1fb310613be0a7abdeccc72b"
EXPECTED_STAGE12_HASHES = {
    "stage12_manifest": "883638524b7a1da72b6063b920dad69e63ffb1315b81e3b20cdfe06499122046",
    "stage12_archive": "7cb6fe6f47778df2b5ccc78d84249631ce4546f775d111b3da04dde9a08ff8da",
    "stage12_circuits": "4d1a2e14e5f1f51e775a3105c86c9ac17cd3aa6fefa181ebf5410a268389907f",
    "stage12_pairwise_overlap": (
        "dd3fca529e77a234c30338e3d6da121cc568b8f37c637163b4552bd6b841ea42"
    ),
}
EXPECTED_STAGE16_HASHES = {
    "stage16_manifest": "29d5aa32708276c4737f2d78903a4ad98ace889cc48855385691284b06369187",
    "stage16_archive": "62569d0a6eff197ed6839cf551deacb4061c9ee390faa42aa882c3232a2ca207",
    "stage16_transfer_profiles": (
        "c832a4995a208df2d413afea7ccbc12d62d2bc27c10ab90b1ceb5a6a5e9e65c8"
    ),
    "stage16_transfer_groups": ("215b3602092f76775e31f1f43929b878ae11403108c58d4e8b141eb15700e7b9"),
}


@dataclass(frozen=True)
class Stage17Cell:
    """One prespecified fidelity-by-distinctness cell."""

    cell_index: int
    model_seed: int
    checkpoint_step: int
    fidelity_threshold: Fraction
    fidelity_display: str
    distinctness_cutoff: Fraction
    distinctness_display: str
    search_execution_mode: SearchExecutionMode
    search_source_stage: int | None
    search_source_run_id: str | None
    search_source_manifest: str | None
    search_source_table: str | None
    transfer_execution_mode: TransferExecutionMode
    transfer_source_stage: int | None
    transfer_source_run_id: str | None
    transfer_source_manifest: str | None
    transfer_source_table: str | None
    expected_search_budget: int
    output_status: str
    transfer_grouping_status: str
    cell_id: str

    @property
    def key(self) -> tuple[Fraction, Fraction]:
        return self.fidelity_threshold, self.distinctness_cutoff

    @property
    def is_primary(self) -> bool:
        return self.key == PRIMARY_CELL_KEY

    def to_record(self) -> dict[str, object]:
        """Return the deterministic pre-execution registry row."""

        return {
            "cell_index": self.cell_index,
            "model_seed": self.model_seed,
            "checkpoint_step": self.checkpoint_step,
            "fidelity_numerator": self.fidelity_threshold.numerator,
            "fidelity_denominator": self.fidelity_threshold.denominator,
            "displayed_fidelity": self.fidelity_display,
            "distinctness_numerator": self.distinctness_cutoff.numerator,
            "distinctness_denominator": self.distinctness_cutoff.denominator,
            "displayed_jaccard_cutoff": self.distinctness_display,
            "search_execution_mode": self.search_execution_mode,
            "search_source_stage": self.search_source_stage,
            "search_source_run_id": self.search_source_run_id,
            "search_source_manifest": self.search_source_manifest,
            "search_source_table": self.search_source_table,
            "transfer_execution_mode": self.transfer_execution_mode,
            "transfer_source_stage": self.transfer_source_stage,
            "transfer_source_run_id": self.transfer_source_run_id,
            "transfer_source_manifest": self.transfer_source_manifest,
            "transfer_source_table": self.transfer_source_table,
            "expected_search_budget": self.expected_search_budget,
            "output_status": self.output_status,
            "transfer_grouping_status": self.transfer_grouping_status,
            "cell_id": self.cell_id,
        }


@dataclass(frozen=True)
class FrozenStage17Configuration:
    """Validated pre-results Stage 17 configuration."""

    path: Path
    sha256: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ReferenceCircuit:
    """One exact accepted Stage 12 circuit reference."""

    source_cell_id: str
    circuit_id: str
    member_index: int
    mask: ComponentMask
    mask_member_name: str
    mask_sha256: str
    member_record: dict[str, Any]
    circuit_row: dict[str, str]


@dataclass(frozen=True)
class ReferenceFamily:
    """One exact Stage 12 family-search reference cell."""

    stage17_cell: Stage17Cell
    source_cell_id: str
    summary_record: dict[str, Any]
    summary_row: dict[str, str]
    circuits: tuple[ReferenceCircuit, ...]
    pairwise_records: tuple[dict[str, Any], ...]
    pairwise_rows: tuple[dict[str, str], ...]
    restart_rows: tuple[dict[str, str], ...]
    frontier_rows: tuple[dict[str, str], ...]
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class PrimaryTransferReference:
    """Exact Stage 16 primary-cell transfer reference."""

    profiles: tuple[TransferProfile, ...]
    profile_rows: tuple[dict[str, str], ...]
    evaluation_rows: tuple[dict[str, str], ...]
    distance_rows: tuple[dict[str, str], ...]
    group_rows: tuple[dict[str, str], ...]
    group_count: int
    groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Stage17InputValidation:
    """Complete read-only validation used by validate-only and execution."""

    repository: Path
    configuration: FrozenStage17Configuration
    registry: tuple[Stage17Cell, ...]
    reference_families: tuple[ReferenceFamily, ...]
    primary_transfer_reference: PrimaryTransferReference
    implementation_commit: str
    repository_clean: bool
    source_hashes: dict[str, str]


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_archive_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError(f"Unsafe Stage 12 archive member: {name!r}")
    return name


def _archive_files(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = _safe_archive_name(member.name)
            if name in files:
                raise ValueError(f"Duplicate Stage 12 archive member: {name}")
            if not member.isfile():
                raise ValueError(f"Non-regular Stage 12 archive member: {name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Unreadable Stage 12 archive member: {name}")
            files[name] = handle.read()
    return files


def _json_lines(data: bytes, name: str) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line in data.decode("utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain only JSON objects.")
        records.append(value)
    return tuple(records)


def _git_state(repository: Path) -> tuple[str, bool]:
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean = not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, clean


def load_stage17_configuration(
    repository_root: str | Path,
    path: str | Path = CONFIGURATION_PATH,
) -> FrozenStage17Configuration:
    """Load and cross-check the frozen Stage 17 configuration."""

    repository = Path(repository_root).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repository / config_path
    payload = _load_json_object(config_path, "Stage 17 configuration")

    expected_fidelity = [
        {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "display": fidelity_display(value),
        }
        for value in FIDELITY_GRID
    ]
    expected_distinctness = [
        {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "display": distinctness_display(value),
        }
        for value in DISTINCTNESS_GRID
    ]
    expected_primary = {
        "fidelity_numerator": 99,
        "fidelity_denominator": 100,
        "distinctness_numerator": 1,
        "distinctness_denominator": 2,
    }
    expected_search = {
        "exact_evaluation_examples": EXACT_EVALUATION_EXAMPLES,
        "candidate_removal_batch_size_maximum": 16,
        "ranking_batch_size": 256,
        "evaluation_batch_size": 256,
        "reuse_coefficient": 0.5,
        "maximum_restarts_per_alternative": 5,
        "per_requested_circuit_budget": PER_REQUESTED_CIRCUIT_BUDGET,
        "per_cell_budget": PER_CELL_BUDGET,
        "family_target": FAMILY_TARGET,
        "cell_order": "fidelity_grid_then_distinctness_grid",
        "tie_tolerance": 1.0e-12,
    }
    if payload.get("fidelity_grid") != expected_fidelity:
        raise ValueError("Stage 17 fidelity grid differs from the frozen grid.")
    if payload.get("distinctness_grid") != expected_distinctness:
        raise ValueError("Stage 17 distinctness grid differs from the frozen grid.")
    if payload.get("primary_cell") != expected_primary:
        raise ValueError("Stage 17 primary cell differs from 0.990 x 0.50.")
    if payload.get("search") != expected_search:
        raise ValueError("Stage 17 search configuration differs from the frozen values.")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("Stage 17 source configuration must be a mapping.")
    expected_source_identity = {
        "training_run_id": "modular-addition-training-s1-5f1bc9dee7ab",
        "model_seed": MODEL_SEED,
        "checkpoint_index": CHECKPOINT_INDEX,
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_phase": "stable_post_grokking",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
        "stage12_run_id": STAGE12_RUN_ID,
        "stage16_run_id": STAGE16_RUN_ID,
    }
    for key, expected in expected_source_identity.items():
        if source.get(key) != expected:
            raise ValueError(f"Stage 17 source {key} mismatch.")

    if payload.get("component_universe") != {
        "attention_heads": 4,
        "mlp_neurons": 512,
        "total_searchable_components": TOTAL_SEARCHABLE_COMPONENTS,
    }:
        raise ValueError("Stage 17 component universe mismatch.")
    if payload.get("sparsity") != {
        "maximum_retained_components": MAXIMUM_RETAINED_COMPONENTS,
        "maximum_retained_proportion": 0.5,
    }:
        raise ValueError("Stage 17 sparsity boundary mismatch.")
    if payload.get("transfer") != {
        "profile_order": ["Q1", "Q2", "Q3", "Q4"],
        "distance": "maximum_absolute_difference",
        "grouping": "deterministic_complete_linkage",
        "primary_tolerance_numerator": 1,
        "primary_tolerance_denominator": 20,
    }:
        raise ValueError("Stage 17 transfer configuration mismatch.")
    if payload.get("interpretation") != {
        "qualitative_endpoint": "recovered_structural_family_size_at_least_2",
        "neighbourhood_fidelity_rationals": ["39/40", "99/100"],
        "neighbourhood_distinctness_rationals": ["1/4", "1/2", "3/4"],
        "immediate_neighbour_rational_pairs": [
            ["39/40", "1/2"],
            ["99/100", "1/4"],
            ["99/100", "3/4"],
        ],
        "classification_order": [
            "robust across the frozen sensitivity grid",
            "robust only within a limited neighbourhood of the primary cell",
            "threshold-sensitive or fragile",
            "mixed",
            "unresolved",
        ],
    }:
        raise ValueError("Stage 17 interpretation convention mismatch.")
    if payload.get("lifecycle") != {
        "stage15_status": "unavailable",
        "checkpoint_grid_decision_made": False,
        "stage18_started": False,
    }:
        raise ValueError("Stage 17 lifecycle boundary mismatch.")

    return FrozenStage17Configuration(
        path=config_path,
        sha256=file_sha256(config_path),
        payload=payload,
    )


def _validate_expected_hashes(
    repository: Path,
    source: dict[str, Any],
    expected: dict[str, str],
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for key, expected_hash in expected.items():
        path = repository / str(source[key])
        actual = file_sha256(path)
        if actual != expected_hash or source.get(f"{key}_sha256", actual) != actual:
            raise ValueError(
                f"Pinned {key} hash mismatch: expected {expected_hash}, found {actual}."
            )
        observed[key] = actual
    return observed


def _source_cell_id(cell: Stage17Cell) -> str:
    return f"cutoff-{cell.distinctness_display}"


def load_and_validate_stage12_references(
    repository_root: str | Path,
    configuration: FrozenStage17Configuration,
) -> tuple[ReferenceFamily, ...]:
    """Load and exactly validate all three Stage 12 reference cells."""

    repository = Path(repository_root).resolve()
    source = configuration.payload["source"]
    _validate_expected_hashes(repository, source, EXPECTED_STAGE12_HASHES)
    manifest = _load_json_object(repository / source["stage12_manifest"], "Stage 12 manifest")
    if manifest.get("stage12_run_id") != STAGE12_RUN_ID:
        raise ValueError("Stage 12 run ID mismatch.")
    if manifest.get("source_training_run_id") != source["training_run_id"]:
        raise ValueError("Stage 12 training-run mismatch.")
    if int(manifest["checkpoint"]["training_step"]) != CHECKPOINT_STEP:
        raise ValueError("Stage 12 checkpoint mismatch.")
    stage12_configuration = manifest["configuration"]
    expected_values = {
        "fidelity_threshold": 0.99,
        "family_target": FAMILY_TARGET,
        "max_restarts_per_alternative": 5,
        "per_requested_circuit_budget": PER_REQUESTED_CIRCUIT_BUDGET,
        "per_cell_budget": PER_CELL_BUDGET,
        "ranking_batch_size": 256,
        "evaluation_batch_size": 256,
        "reuse_coefficient": 0.5,
    }
    for key, expected in expected_values.items():
        if stage12_configuration.get(key) != expected:
            raise ValueError(f"Stage 12 frozen configuration mismatch for {key}.")

    archive_path = repository / source["stage12_archive"]
    files = _archive_files(archive_path)
    summary_rows = _read_csv(repository / source["stage12_family_summary"])
    circuit_rows = _read_csv(repository / source["stage12_circuits"])
    overlap_rows = _read_csv(repository / source["stage12_pairwise_overlap"])
    restart_rows = _read_csv(repository / source["stage12_restarts"])
    frontier_rows = _read_csv(repository / source["stage12_frontier"])

    families: list[ReferenceFamily] = []
    for cell in build_stage17_registry():
        if cell.search_execution_mode != "reference_existing_result":
            continue
        source_cell_id = _source_cell_id(cell)
        prefix = f"{STAGE12_RUN_ID}/{source_cell_id}"
        cell_summary_name = f"{prefix}/cell_summary.json"
        members_name = f"{prefix}/family_members.jsonl"
        overlaps_name = f"{prefix}/pairwise_overlaps.jsonl"
        inventory_name = f"{prefix}/hash_inventory.json"
        required = (cell_summary_name, members_name, overlaps_name, inventory_name)
        missing = [name for name in required if name not in files]
        if missing:
            raise FileNotFoundError("Missing Stage 12 archive members: " + ", ".join(missing))

        summary = json.loads(files[cell_summary_name])
        if summary["cell_metadata"]["checkpoint_step"] != CHECKPOINT_STEP:
            raise ValueError(f"Stage 12 {source_cell_id} checkpoint mismatch.")
        if Fraction(str(summary["cell_metadata"]["fidelity_threshold"])) != cell.fidelity_threshold:
            raise ValueError(f"Stage 12 {source_cell_id} fidelity mismatch.")
        cutoff_record = summary["family_search"]["distinctness_cutoff"]
        if (
            Fraction(cutoff_record["numerator"], cutoff_record["denominator"])
            != cell.distinctness_cutoff
        ):
            raise ValueError(f"Stage 12 {source_cell_id} cutoff mismatch.")
        family_search = summary["family_search"]
        for key, expected in (
            ("family_target", FAMILY_TARGET),
            ("max_restarts_per_alternative", 5),
            ("per_requested_circuit_budget", PER_REQUESTED_CIRCUIT_BUDGET),
            ("per_cell_budget", PER_CELL_BUDGET),
        ):
            if int(family_search[key]) != expected:
                raise ValueError(f"Stage 12 {source_cell_id} {key} mismatch.")

        matching_summaries = [row for row in summary_rows if row["cell_id"] == source_cell_id]
        if len(matching_summaries) != 1:
            raise ValueError(f"Stage 12 {source_cell_id} summary row is not unique.")
        summary_row = matching_summaries[0]
        parity = {
            "family_size": int(summary_row["family_size"]),
            "family_target": int(summary_row["family_target"]),
            "exact_evaluations_used": int(summary_row["exact_evaluations_used"]),
            "per_cell_budget": int(summary_row["per_cell_budget"]),
            "budget_remaining": int(summary_row["budget_remaining"]),
            "restart_outcome_count": int(summary_row["restart_outcome_count"]),
        }
        for key, expected in parity.items():
            if int(family_search[key]) != expected:
                raise ValueError(f"Stage 12 {source_cell_id} {key} source mismatch.")
        if family_search["status"] != summary_row["status"]:
            raise ValueError(f"Stage 12 {source_cell_id} status mismatch.")
        if family_search["stopping_reason"] != summary_row["stopping_reason"]:
            raise ValueError(f"Stage 12 {source_cell_id} stopping-reason mismatch.")

        inventory = json.loads(files[inventory_name])
        inventory_hashes = {
            str(record["path"]): str(record["sha256"]) for record in inventory["files"]
        }
        member_records = _json_lines(files[members_name], members_name)
        source_circuit_rows = sorted(
            (row for row in circuit_rows if row["cell_id"] == source_cell_id),
            key=lambda row: int(row["member_index"]),
        )
        if len(member_records) != int(family_search["family_size"]):
            raise ValueError(f"Stage 12 {source_cell_id} family-size/archive mismatch.")
        if len(source_circuit_rows) != len(member_records):
            raise ValueError(f"Stage 12 {source_cell_id} family-size/table mismatch.")

        circuits: list[ReferenceCircuit] = []
        for member_record, circuit_row in zip(member_records, source_circuit_rows, strict=True):
            member_index = int(circuit_row["member_index"])
            restart_index = int(circuit_row["selected_restart_index"])
            relative_mask = (
                f"restarts/C{member_index:02d}/restart_{restart_index:02d}/search/final_mask.json"
            )
            member_name = f"{prefix}/{relative_mask}"
            data = files.get(member_name)
            if data is None:
                raise FileNotFoundError(member_name)
            digest = hashlib.sha256(data).hexdigest()
            if inventory_hashes.get(relative_mask) != digest:
                raise ValueError(
                    f"Stage 12 mask hash mismatch for {source_cell_id} C{member_index}."
                )
            mask = ComponentMask.from_record(json.loads(data))
            if mask.mask_id != circuit_row["mask_id"] or mask.mask_id != member_record["mask_id"]:
                raise ValueError(
                    f"Stage 12 mask identity mismatch for {source_cell_id} C{member_index}."
                )
            if mask.to_record() != member_record["mask"]:
                raise ValueError(
                    f"Stage 12 mask bytes disagree for {source_cell_id} C{member_index}."
                )
            agreement = int(circuit_row["prediction_agreement_count"])
            evaluated = int(circuit_row["evaluated_example_count"])
            if evaluated != EXACT_EVALUATION_EXAMPLES or not (
                agreement * cell.fidelity_threshold.denominator
                >= evaluated * cell.fidelity_threshold.numerator
            ):
                raise ValueError(
                    f"Stage 12 exact fidelity failure for {source_cell_id} C{member_index}."
                )
            if mask.retained_component_count > MAXIMUM_RETAINED_COMPONENTS:
                raise ValueError(f"Stage 12 sparsity failure for {source_cell_id} C{member_index}.")
            if not math.isclose(
                agreement / evaluated,
                float(circuit_row["primary_fidelity"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    f"Stage 12 fidelity/count mismatch for {source_cell_id} C{member_index}."
                )
            circuits.append(
                ReferenceCircuit(
                    source_cell_id=source_cell_id,
                    circuit_id=str(circuit_row["member_label"]),
                    member_index=member_index,
                    mask=mask,
                    mask_member_name=member_name,
                    mask_sha256=digest,
                    member_record=dict(member_record),
                    circuit_row=dict(circuit_row),
                )
            )

        pairwise_records = _json_lines(files[overlaps_name], overlaps_name)
        source_overlap_rows = tuple(row for row in overlap_rows if row["cell_id"] == source_cell_id)
        expected_pairs = len(circuits) * (len(circuits) - 1) // 2
        if len(pairwise_records) != expected_pairs or len(source_overlap_rows) != expected_pairs:
            raise ValueError(f"Stage 12 pair-count mismatch for {source_cell_id}.")
        by_id = {circuit.circuit_id: circuit for circuit in circuits}
        for row in source_overlap_rows:
            left, right = row["left_member_label"], row["right_member_label"]
            intersection, union = jaccard_counts(by_id[left].mask, by_id[right].mask)
            exact = Fraction(intersection, union)
            recorded = Fraction(int(row["jaccard_numerator"]), int(row["jaccard_denominator"]))
            if exact != recorded or exact > cell.distinctness_cutoff:
                raise ValueError(f"Stage 12 overlap mismatch for {source_cell_id} {left}/{right}.")

        families.append(
            ReferenceFamily(
                stage17_cell=cell,
                source_cell_id=source_cell_id,
                summary_record=summary,
                summary_row=dict(summary_row),
                circuits=tuple(circuits),
                pairwise_records=pairwise_records,
                pairwise_rows=source_overlap_rows,
                restart_rows=tuple(row for row in restart_rows if row["cell_id"] == source_cell_id),
                frontier_rows=tuple(
                    row for row in frontier_rows if row["cell_id"] == source_cell_id
                ),
                source_paths=(cell_summary_name, members_name, overlaps_name, inventory_name),
            )
        )

    if len(families) != 3:
        raise ValueError("Stage 17 requires exactly three Stage 12 reference families.")
    return tuple(families)


def load_and_validate_stage16_primary_transfer(
    repository_root: str | Path,
    configuration: FrozenStage17Configuration,
    primary_family: ReferenceFamily,
) -> PrimaryTransferReference:
    """Load and exactly validate the primary Stage 16 transfer reference."""

    repository = Path(repository_root).resolve()
    source = configuration.payload["source"]
    _validate_expected_hashes(repository, source, EXPECTED_STAGE16_HASHES)
    manifest = _load_json_object(repository / source["stage16_manifest"], "Stage 16 manifest")
    if manifest.get("stage16_run_id") != STAGE16_RUN_ID:
        raise ValueError("Stage 16 run ID mismatch.")
    if int(manifest["checkpoint"]["step"]) != CHECKPOINT_STEP:
        raise ValueError("Stage 16 checkpoint mismatch.")
    frozen_analysis = manifest["frozen_analysis"]
    fidelity = frozen_analysis["fidelity_threshold"]
    if Fraction(fidelity["numerator"], fidelity["denominator"]) != Fraction(99, 100):
        raise ValueError("Stage 16 primary fidelity mismatch.")
    cutoff = frozen_analysis["jaccard_cutoff"]
    if Fraction(cutoff["numerator"], cutoff["denominator"]) != Fraction(1, 2):
        raise ValueError("Stage 16 primary family cell mismatch.")
    tolerance = frozen_analysis["primary_grouping_tolerance"]
    if Fraction(tolerance["numerator"], tolerance["denominator"]) != PRIMARY_TRANSFER_TOLERANCE:
        raise ValueError("Stage 16 primary transfer tolerance mismatch.")

    profile_rows = tuple(_read_csv(repository / source["stage16_transfer_profiles"]))
    evaluation_rows = tuple(
        _read_csv(repository / "results/tables/seed_1_stage16_global_family_transfer.csv")
    )
    distance_rows = tuple(_read_csv(repository / source["stage16_transfer_distances"]))
    all_group_rows = _read_csv(repository / source["stage16_transfer_groups"])
    group_rows = tuple(
        row
        for row in all_group_rows
        if Fraction(int(row["tolerance_numerator"]), int(row["tolerance_denominator"]))
        == PRIMARY_TRANSFER_TOLERANCE
    )
    expected_ids = tuple(circuit.circuit_id for circuit in primary_family.circuits)
    if tuple(row["circuit_id"] for row in profile_rows) != expected_ids:
        raise ValueError("Stage 16 primary transfer circuit identities mismatch.")
    if len(evaluation_rows) != len(expected_ids) * 4:
        raise ValueError("Stage 16 primary transfer evaluation count mismatch.")
    for index, circuit_id in enumerate(expected_ids):
        rows = evaluation_rows[index * 4 : (index + 1) * 4]
        if tuple(row["evaluation_subset"] for row in rows) != ("Q1", "Q2", "Q3", "Q4"):
            raise ValueError(f"Stage 16 transfer subset order mismatch for {circuit_id}.")
        if any(row["circuit_id"] != circuit_id for row in rows):
            raise ValueError(f"Stage 16 transfer evaluation identity mismatch for {circuit_id}.")
        source_mask = primary_family.circuits[index]
        if any(row["mask_sha256"] != source_mask.mask_sha256 for row in rows):
            raise ValueError(f"Stage 16 transfer mask hash mismatch for {circuit_id}.")
        profile_row = profile_rows[index]
        if tuple(float(row["fidelity"]) for row in rows) != tuple(
            float(profile_row[f"q{subset_index}_fidelity"]) for subset_index in range(1, 5)
        ):
            raise ValueError(f"Stage 16 transfer profile/evaluation mismatch for {circuit_id}.")

    profiles = tuple(
        TransferProfile(
            circuit_id=row["circuit_id"],
            q1_fidelity=float(row["q1_fidelity"]),
            q2_fidelity=float(row["q2_fidelity"]),
            q3_fidelity=float(row["q3_fidelity"]),
            q4_fidelity=float(row["q4_fidelity"]),
        )
        for row in profile_rows
    )
    if not group_rows:
        raise ValueError("Stage 16 primary transfer group row is absent.")
    group_count = int(group_rows[0]["group_count"])
    if any(int(row["group_count"]) != group_count for row in group_rows):
        raise ValueError("Stage 16 primary transfer group count is inconsistent.")
    groups = tuple(tuple(json.loads(row["ordered_members_json"])) for row in group_rows)
    if sorted(member for group in groups for member in group) != sorted(expected_ids):
        raise ValueError("Stage 16 primary transfer groups do not partition the family.")
    if any(row["complete_linkage_valid"] != "True" for row in group_rows):
        raise ValueError("Stage 16 primary transfer group violates complete linkage.")
    expected_pairs = len(expected_ids) * (len(expected_ids) - 1) // 2
    if len(distance_rows) != expected_pairs:
        raise ValueError("Stage 16 primary transfer distance count mismatch.")

    return PrimaryTransferReference(
        profiles=profiles,
        profile_rows=profile_rows,
        evaluation_rows=evaluation_rows,
        distance_rows=distance_rows,
        group_rows=group_rows,
        group_count=group_count,
        groups=groups,
    )


def validate_stage17_inputs(
    repository_root: str | Path,
    *,
    configuration_path: str | Path = CONFIGURATION_PATH,
    stage12_manifest: str | Path | None = None,
    stage16_manifest: str | Path | None = None,
    checkpoint_step: int = CHECKPOINT_STEP,
    require_clean: bool = False,
) -> Stage17InputValidation:
    """Validate all frozen Stage 17 inputs without creating files."""

    repository = Path(repository_root).resolve()
    configuration = load_stage17_configuration(repository, configuration_path)
    source = configuration.payload["source"]
    if checkpoint_step != CHECKPOINT_STEP:
        raise ValueError("Only frozen checkpoint step 9050 is permitted.")
    for requested, key, label in (
        (stage12_manifest, "stage12_manifest", "Stage 12"),
        (stage16_manifest, "stage16_manifest", "Stage 16"),
    ):
        if requested is None:
            continue
        requested_path = Path(requested)
        if not requested_path.is_absolute():
            requested_path = repository / requested_path
        if requested_path.resolve() != (repository / source[key]).resolve():
            raise ValueError(f"Only the frozen {label} manifest is permitted.")

    required_paths = (
        "checkpoint_manifest",
        "checkpoint_path",
        "primary_threshold_manifest",
        "stage11_manifest",
        "search_config",
        "stage12_manifest",
        "stage12_archive",
        "stage12_family_summary",
        "stage12_circuits",
        "stage12_pairwise_overlap",
        "stage12_restarts",
        "stage12_frontier",
        "stage15_manifest",
        "stage16_manifest",
        "stage16_archive",
        "stage16_transfer_profiles",
        "stage16_transfer_distances",
        "stage16_transfer_groups",
    )
    for key in required_paths:
        path = repository / source[key]
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen Stage 17 input {key}: {path}")

    for key in ("primary_threshold_manifest", "stage11_manifest"):
        actual = file_sha256(repository / source[key])
        if actual != source[f"{key}_sha256"]:
            raise ValueError(f"Pinned {key} hash mismatch.")

    forbidden = []
    for directory_name in ("manifests", "results", "figures"):
        directory = repository / directory_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(repository).as_posix()
            if relative_path in POST_STAGE17_FREEZE_OUTPUTS:
                continue
            lowered = path.name.lower()
            if (
                "stage18" in lowered
                or "checkpoint_grid" in lowered
                or "checkpoint-grid" in lowered
                or "scaled_checkpoint" in lowered
                or "scaled-checkpoint" in lowered
            ):
                forbidden.append(path)
    if forbidden:
        raise ValueError(
            "Post-Stage-17 lifecycle outputs must remain absent: "
            + ", ".join(str(path) for path in sorted(forbidden))
        )

    if file_sha256(repository / source["checkpoint_path"]) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Stage 17 checkpoint SHA-256 mismatch.")
    checkpoint_manifest = _load_json_object(
        repository / source["checkpoint_manifest"], "checkpoint manifest"
    )
    stable = checkpoint_manifest["selected_stable_post_checkpoint"]
    if int(stable["training_step"]) != CHECKPOINT_STEP:
        raise ValueError("Frozen stable-post checkpoint step mismatch.")
    if stable["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen stable-post checkpoint hash mismatch.")

    primary_threshold = _load_json_object(
        repository / source["primary_threshold_manifest"], "primary-threshold manifest"
    )
    selected = primary_threshold["primary_fidelity_threshold"]
    if Fraction(int(selected["numerator"]), int(selected["denominator"])) != Fraction(99, 100):
        raise ValueError("Stage 11 primary threshold is not exactly 99/100.")

    search_config_path = repository / source["search_config"]
    search_config = yaml.safe_load(search_config_path.read_text(encoding="utf-8"))
    if not isinstance(search_config, dict):
        raise ValueError("Frozen search configuration must be a mapping.")
    if file_sha256(search_config_path) != source["search_config_sha256"]:
        raise ValueError("Frozen search-configuration file hash mismatch.")
    if mapping_hash(search_config) != source["search_config_mapping_sha256"]:
        raise ValueError("Frozen search-configuration mapping hash mismatch.")

    stage15 = _load_json_object(repository / source["stage15_manifest"], "Stage 15 manifest")
    if (
        stage15.get("status") != "unavailable"
        or stage15.get("stage13_selection_outcome") != "no_qualifying_fraction"
        or "produced no qualifying matched no-generalisation fraction"
        not in str(stage15.get("scientific_reason"))
        or stage15.get("family_size") is not None
        or stage15.get("transfer_group_count") is not None
        or stage15.get("replacement_control_introduced") is not False
        or stage15.get("stage15_control_training_executed") is not False
        or stage15.get("stage15_circuit_analysis_executed") is not False
    ):
        raise ValueError("Stage 15 must remain unavailable with null endpoints.")

    implementation_commit, repository_clean = _git_state(repository)
    if require_clean and not repository_clean:
        raise RuntimeError("Definitive Stage 17 execution requires a clean implementation commit.")
    registry = build_stage17_registry()
    reference_families = load_and_validate_stage12_references(repository, configuration)
    primary_family = next(family for family in reference_families if family.stage17_cell.is_primary)
    transfer_reference = load_and_validate_stage16_primary_transfer(
        repository, configuration, primary_family
    )
    source_hashes = {
        "configuration": configuration.sha256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "primary_threshold_manifest": file_sha256(
            repository / source["primary_threshold_manifest"]
        ),
        "stage11_manifest": file_sha256(repository / source["stage11_manifest"]),
        "search_config": file_sha256(search_config_path),
        **_validate_expected_hashes(repository, source, EXPECTED_STAGE12_HASHES),
        **_validate_expected_hashes(repository, source, EXPECTED_STAGE16_HASHES),
    }
    return Stage17InputValidation(
        repository=repository,
        configuration=configuration,
        registry=registry,
        reference_families=reference_families,
        primary_transfer_reference=transfer_reference,
        implementation_commit=implementation_commit,
        repository_clean=repository_clean,
        source_hashes=source_hashes,
    )


def circuit_size_summary(
    retained_component_counts: tuple[int, ...],
) -> dict[str, int | float | None]:
    """Return null-safe per-cell circuit-size summaries."""

    if any(value < 0 or value > MAXIMUM_RETAINED_COMPONENTS for value in retained_component_counts):
        raise ValueError("Circuit-size summary received an invalid retained count.")
    if not retained_component_counts:
        return {
            "circuit_count": 0,
            "minimum_retained_components": None,
            "maximum_retained_components": None,
            "mean_retained_components": None,
            "median_retained_components": None,
            "minimum_retained_proportion": None,
            "maximum_retained_proportion": None,
            "mean_retained_proportion": None,
            "median_retained_proportion": None,
        }
    proportions = tuple(value / TOTAL_SEARCHABLE_COMPONENTS for value in retained_component_counts)
    return {
        "circuit_count": len(retained_component_counts),
        "minimum_retained_components": min(retained_component_counts),
        "maximum_retained_components": max(retained_component_counts),
        "mean_retained_components": mean(retained_component_counts),
        "median_retained_components": median(retained_component_counts),
        "minimum_retained_proportion": min(proportions),
        "maximum_retained_proportion": max(proportions),
        "mean_retained_proportion": mean(proportions),
        "median_retained_proportion": median(proportions),
    }


def structural_overlap_summary(
    overlaps: tuple[Fraction, ...],
    *,
    family_size: int,
    cutoff: Fraction | int | float | str,
) -> dict[str, int | float | bool | None]:
    """Return null-safe exact pairwise structural summaries."""

    exact_cutoff = validate_distinctness_cutoff(cutoff)
    expected_pairs = family_size * (family_size - 1) // 2
    if len(overlaps) != expected_pairs:
        raise ValueError("Pairwise overlap count does not match family size.")
    if not overlaps:
        return {
            "pair_count": 0,
            "minimum_pairwise_jaccard_overlap": None,
            "maximum_pairwise_jaccard_overlap": None,
            "mean_pairwise_overlap": None,
            "median_pairwise_overlap": None,
            "minimum_structural_distance": None,
            "maximum_structural_distance": None,
            "mean_structural_distance": None,
            "cutoff_compliance": None,
        }
    values = tuple(float(value) for value in overlaps)
    distances = tuple(1.0 - value for value in values)
    return {
        "pair_count": len(overlaps),
        "minimum_pairwise_jaccard_overlap": min(values),
        "maximum_pairwise_jaccard_overlap": max(values),
        "mean_pairwise_overlap": mean(values),
        "median_pairwise_overlap": median(values),
        "minimum_structural_distance": min(distances),
        "maximum_structural_distance": max(distances),
        "mean_structural_distance": mean(distances),
        "cutoff_compliance": max(overlaps) <= exact_cutoff,
    }


def deterministic_cell_id(fidelity: Fraction, distinctness: Fraction) -> str:
    """Return the frozen Stage 17 scientific cell identifier."""

    fidelity = validate_fidelity_threshold(fidelity)
    distinctness = validate_distinctness_cutoff(distinctness)
    return (
        f"s{MODEL_SEED}-step{CHECKPOINT_STEP}-"
        f"f{fidelity.numerator}of{fidelity.denominator}-"
        f"d{distinctness.numerator}of{distinctness.denominator}"
    )


def build_stage17_registry() -> tuple[Stage17Cell, ...]:
    """Build and validate the exact row-major 18-cell registry."""

    cells: list[Stage17Cell] = []
    for fidelity in FIDELITY_GRID:
        for distinctness in DISTINCTNESS_GRID:
            reference_search = fidelity == Fraction(99, 100)
            reference_transfer = (fidelity, distinctness) == PRIMARY_CELL_KEY
            cells.append(
                Stage17Cell(
                    cell_index=len(cells) + 1,
                    model_seed=MODEL_SEED,
                    checkpoint_step=CHECKPOINT_STEP,
                    fidelity_threshold=fidelity,
                    fidelity_display=fidelity_display(fidelity),
                    distinctness_cutoff=distinctness,
                    distinctness_display=distinctness_display(distinctness),
                    search_execution_mode=(
                        "reference_existing_result" if reference_search else "fresh_execution"
                    ),
                    search_source_stage=12 if reference_search else None,
                    search_source_run_id=STAGE12_RUN_ID if reference_search else None,
                    search_source_manifest=STAGE12_MANIFEST if reference_search else None,
                    search_source_table=STAGE12_FAMILY_SUMMARY if reference_search else None,
                    transfer_execution_mode=(
                        "reference_existing_result" if reference_transfer else "fresh_execution"
                    ),
                    transfer_source_stage=16 if reference_transfer else None,
                    transfer_source_run_id=STAGE16_RUN_ID if reference_transfer else None,
                    transfer_source_manifest=STAGE16_MANIFEST if reference_transfer else None,
                    transfer_source_table=(
                        STAGE16_TRANSFER_PROFILES if reference_transfer else None
                    ),
                    expected_search_budget=PER_CELL_BUDGET,
                    output_status="not_executed",
                    transfer_grouping_status="not_evaluated",
                    cell_id=deterministic_cell_id(fidelity, distinctness),
                )
            )

    registry = tuple(cells)
    validate_stage17_registry(registry)
    return registry


def validate_stage17_registry(cells: tuple[Stage17Cell, ...]) -> None:
    """Enforce the complete frozen-grid and reference identities."""

    if len(cells) != 18:
        raise ValueError(f"Stage 17 registry must contain 18 cells, found {len(cells)}.")
    if tuple(cell.cell_index for cell in cells) != tuple(range(1, 19)):
        raise ValueError("Stage 17 cell indices must be consecutive and one-based.")

    expected_keys = tuple(
        (fidelity, distinctness) for fidelity in FIDELITY_GRID for distinctness in DISTINCTNESS_GRID
    )
    if tuple(cell.key for cell in cells) != expected_keys:
        raise ValueError("Stage 17 registry does not match the frozen row-major grid.")
    if len({cell.cell_id for cell in cells}) != 18:
        raise ValueError("Stage 17 cell identifiers must be unique.")
    if sum(cell.search_execution_mode == "fresh_execution" for cell in cells) != 15:
        raise ValueError("Stage 17 must contain exactly 15 fresh search cells.")
    if sum(cell.search_execution_mode == "reference_existing_result" for cell in cells) != 3:
        raise ValueError("Stage 17 must contain exactly three search references.")
    if sum(cell.transfer_execution_mode == "reference_existing_result" for cell in cells) != 1:
        raise ValueError("Stage 17 must contain exactly one transfer reference.")
    if sum(cell.is_primary for cell in cells) != 1:
        raise ValueError("Stage 17 must contain exactly one primary cell.")


def cell_for(
    fidelity: Fraction | int | float | str,
    distinctness: Fraction | int | float | str,
) -> Stage17Cell:
    """Return one planned cell and reject every unplanned pair."""

    key = (
        validate_fidelity_threshold(fidelity),
        validate_distinctness_cutoff(distinctness),
    )
    return next(cell for cell in build_stage17_registry() if cell.key == key)
