"""Deterministic Stage 16 genuine-task functional-transfer analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from circuit_families.analysis.fidelity_calibration import write_csv_records
from circuit_families.analysis.random_label_circuit_analysis import (
    canonical_json_sha256,
    subset_context,
)
from circuit_families.analysis.stage14_random_label_reporting import (
    write_deterministic_archive,
)
from circuit_families.analysis.stage14_random_label_runner import (
    current_git_commit,
    repository_is_clean,
)
from circuit_families.analysis.transfer import (
    TransferEvaluation,
    TransferProfile,
    evaluate_transfer_profile,
    grouping_sensitivity,
)
from circuit_families.data.input_subsets import (
    SUBSET_NAMES,
    generate_input_subsets,
)
from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
    MaskEvaluationMetrics,
    compute_full_model_reference,
    evaluate_component_mask,
    load_checkpoint_evaluation_context,
)
from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_HOOK_NAME,
    MLP_NEURON_HOOK_NAME,
    SEARCHABLE_COMPONENT_COUNT,
    ComponentMask,
)
from circuit_families.interpretability.overlap_constraints import (
    jaccard_counts,
)
from circuit_families.interpretability.sparse_search import (
    CANDIDATE_BATCH_SIZE,
    MEANINGFULLY_SPARSE_MAX_COMPONENTS,
    CheckpointSearchExecution,
    run_checkpoint_sparse_search,
    write_sparse_search_artifacts,
)
from circuit_families.training import canonical_state_hash, file_sha256

CONFIGURATION_PATH = Path("configs/stage16_transfer.json")
PRIMARY_CELL_ID = "cutoff-0.50"
FIDELITY_THRESHOLD = Fraction(99, 100)
JACCARD_CUTOFF = Fraction(1, 2)
GROUPING_TOLERANCES = (Fraction(1, 40), Fraction(1, 20), Fraction(1, 10))
PRIMARY_GROUPING_TOLERANCE = Fraction(1, 20)
EXPECTED_SUBSET_COUNTS = {"Q1": 3249, "Q2": 3192, "Q3": 3192, "Q4": 3136}
STAGE12_ARCHIVE_PREFIX = (
    "stage12-diversity-s1-020ebf1b5814/cutoff-0.50"
)

SCIENTIFIC_TABLE_NAMES = (
    "global_family_transfer",
    "transfer_profiles",
    "transfer_distances",
    "transfer_groups",
    "subset_discovery",
    "subset_transfer",
    "structural_functional_comparison",
)


@dataclass(frozen=True)
class FrozenStage16Configuration:
    """Validated pre-results Stage 16 configuration."""

    path: Path
    sha256: str
    payload: dict[str, Any]
    run_id: str


@dataclass(frozen=True)
class Stage12Circuit:
    """One accepted member of the primary Stage 12 family."""

    circuit_id: str
    member_index: int
    mask: ComponentMask
    mask_member_name: str
    mask_sha256: str
    global_metrics: Mapping[str, Any]
    stage12_row: Mapping[str, str]


@dataclass(frozen=True)
class Stage16InputValidation:
    """Read-only validation result used by definitive execution."""

    repository: Path
    output_root: Path
    configuration: FrozenStage16Configuration
    implementation_commit: str
    repository_clean: bool
    context: CheckpointEvaluationContext
    circuits: tuple[Stage12Circuit, ...]
    overlaps: Mapping[tuple[str, str], Fraction]
    subset_indices: Mapping[str, np.ndarray]
    subset_hashes: Mapping[str, str]
    subset_source_sha256: str


@dataclass(frozen=True)
class Stage16ExecutionResult:
    """Locations and identities produced by one complete Stage 16 run."""

    run_id: str
    implementation_commit: str
    manifest: Path
    archive: Path
    note: Path
    runtime_table: Path
    scientific_tables: tuple[Path, ...]


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return value


def _fraction(record: Mapping[str, Any]) -> Fraction:
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def load_stage16_configuration(
    repository_root: str | Path,
    configuration_path: str | Path = CONFIGURATION_PATH,
) -> FrozenStage16Configuration:
    """Load and strictly validate the committed pre-results configuration."""

    repository = Path(repository_root).resolve()
    path = Path(configuration_path)
    if not path.is_absolute():
        path = repository / path
    payload = _load_json_object(path, "Stage 16 configuration")

    expected = {
        "schema_version": 1,
        "experiment_stage": 16,
        "source_training_run_id": "modular-addition-training-s1-5f1bc9dee7ab",
        "model_seed": 1,
        "checkpoint_step": 9050,
        "primary_family_cell": PRIMARY_CELL_ID,
        "source_family_size": 7,
        "meaningfully_sparse_max_components": MEANINGFULLY_SPARSE_MAX_COMPONENTS,
        "searchable_component_count": SEARCHABLE_COMPONENT_COUNT,
        "subset_order": list(SUBSET_NAMES),
        "subset_counts": dict(EXPECTED_SUBSET_COUNTS),
        "transfer_profile_distance": "maximum_absolute_fidelity_difference",
        "maximum_distance_subset_tie_break": "Q1_Q2_Q3_Q4_order",
        "linkage": "complete",
        "group_label_convention": "G01_G02_in_deterministic_cluster_order",
        "candidate_batch_size_maximum": CANDIDATE_BATCH_SIZE,
        "stage17_started": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Frozen Stage 16 configuration changed at {key!r}: "
                f"expected {value!r}, found {payload.get(key)!r}."
            )
    if _fraction(payload["fidelity_threshold"]) != FIDELITY_THRESHOLD:
        raise ValueError("Stage 16 fidelity threshold must be exactly 99/100.")
    if _fraction(payload["jaccard_cutoff"]) != JACCARD_CUTOFF:
        raise ValueError("Stage 16 Jaccard cutoff must be exactly 1/2.")
    tolerances = tuple(_fraction(value) for value in payload["grouping_tolerances"])
    if tolerances != GROUPING_TOLERANCES:
        raise ValueError("Stage 16 grouping tolerance grid changed.")
    if _fraction(payload["primary_grouping_tolerance"]) != PRIMARY_GROUPING_TOLERANCE:
        raise ValueError("Stage 16 primary grouping tolerance changed.")
    digest = file_sha256(path)
    return FrozenStage16Configuration(
        path=path,
        sha256=digest,
        payload=payload,
        run_id=f"stage16-transfer-s1-{digest[:12]}",
    )


def _safe_archive_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe Stage 12 archive member: {name!r}")
    return path.as_posix()


def _archive_files(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = _safe_archive_name(member.name)
            if name in files:
                raise ValueError(f"Duplicate Stage 12 archive member: {name}")
            if not member.isfile():
                if member.isdir():
                    continue
                raise ValueError(f"Non-regular Stage 12 archive member: {name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Unreadable Stage 12 archive member: {name}")
            files[name] = handle.read()
    return files


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_and_validate_stage12_family(
    repository_root: str | Path,
    configuration: FrozenStage16Configuration,
) -> tuple[tuple[Stage12Circuit, ...], dict[tuple[str, str], Fraction]]:
    """Load only the seven accepted primary Stage 12 circuits."""

    repository = Path(repository_root).resolve()
    payload = configuration.payload
    manifest_path = repository / payload["stage12_manifest"]
    archive_path = repository / payload["stage12_archive"]
    circuits_path = repository / payload["stage12_circuits_table"]
    family_path = repository / payload["stage12_family_summary_table"]
    overlap_path = repository / payload["stage12_pairwise_overlap_table"]
    manifest = _load_json_object(manifest_path, "Stage 12 manifest")

    if manifest.get("stage12_run_id") != payload["stage12_run_id"]:
        raise ValueError("Stage 12 run ID mismatch.")
    if manifest.get("source_training_run_id") != payload["source_training_run_id"]:
        raise ValueError("Stage 12 source training run mismatch.")
    if int(manifest["checkpoint"]["training_step"]) != payload["checkpoint_step"]:
        raise ValueError("Stage 12 checkpoint step mismatch.")
    if float(manifest["configuration"]["fidelity_threshold"]) != 0.99:
        raise ValueError("Stage 12 fidelity threshold mismatch.")

    output_records = {
        "archive": archive_path,
        "circuit_table": circuits_path,
        "family_summary_table": family_path,
        "pairwise_overlap_table": overlap_path,
    }
    for key, path in output_records.items():
        expected_hash = manifest["outputs"][key]["sha256"]
        observed_hash = file_sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Stage 12 {key} hash mismatch: expected {expected_hash}, "
                f"found {observed_hash}."
            )

    summary_rows = [
        row for row in _read_csv(family_path) if row["cell_id"] == PRIMARY_CELL_ID
    ]
    if len(summary_rows) != 1 or int(summary_rows[0]["family_size"]) != 7:
        raise ValueError("Stage 12 primary family must contain exactly seven circuits.")

    source_rows = [
        row for row in _read_csv(circuits_path) if row["cell_id"] == PRIMARY_CELL_ID
    ]
    if len(source_rows) != 7:
        raise ValueError("Stage 12 primary circuit table must contain seven rows.")
    source_rows.sort(key=lambda row: int(row["member_index"]))
    if [int(row["member_index"]) for row in source_rows] != list(range(1, 8)):
        raise ValueError("Stage 12 primary member indices must be 1 through 7.")

    files = _archive_files(archive_path)
    inventory_name = f"{STAGE12_ARCHIVE_PREFIX}/hash_inventory.json"
    inventory = json.loads(files[inventory_name])
    inventory_hashes = {
        str(record["path"]): str(record["sha256"])
        for record in inventory["files"]
    }
    members_name = f"{STAGE12_ARCHIVE_PREFIX}/family_members.jsonl"
    member_records = [
        json.loads(line)
        for line in files[members_name].decode("utf-8").splitlines()
        if line
    ]
    accepted = {
        str(record["member_label"]): record
        for record in member_records
        if int(record["member_index"]) <= 7
    }
    if set(accepted) != {f"C{index}" for index in range(1, 8)}:
        raise ValueError("Stage 12 archive accepted-member identities mismatch.")

    circuits: list[Stage12Circuit] = []
    for row in source_rows:
        index = int(row["member_index"])
        circuit_id = row["member_label"]
        restart = int(row["selected_restart_index"])
        relative = (
            f"restarts/C{index:02d}/restart_{restart:02d}/search/final_mask.json"
        )
        member_name = f"{STAGE12_ARCHIVE_PREFIX}/{relative}"
        data = files.get(member_name)
        if data is None:
            raise FileNotFoundError(member_name)
        observed_hash = _sha256_bytes(data)
        if inventory_hashes.get(relative) != observed_hash:
            raise ValueError(f"Stage 12 mask inventory hash mismatch for {circuit_id}.")
        mask = ComponentMask.from_record(json.loads(data))
        record = accepted[circuit_id]
        if mask.mask_id != row["mask_id"] or record["mask"]["mask_id"] != mask.mask_id:
            raise ValueError(f"Stage 12 mask identity mismatch for {circuit_id}.")
        if mask.retained_component_count != int(row["retained_component_count"]):
            raise ValueError(f"Stage 12 retained-count mismatch for {circuit_id}.")
        if mask.retained_attention_head_count != int(
            row["retained_attention_head_count"]
        ) or mask.retained_mlp_neuron_count != int(row["retained_mlp_neuron_count"]):
            raise ValueError(f"Stage 12 retained-class counts mismatch for {circuit_id}.")
        if not math.isclose(
            mask.retained_component_proportion,
            float(row["retained_component_proportion"]),
            abs_tol=1.0e-15,
            rel_tol=0.0,
        ):
            raise ValueError(f"Stage 12 retained proportion mismatch for {circuit_id}.")
        if mask.retained_component_count > MEANINGFULLY_SPARSE_MAX_COMPONENTS:
            raise ValueError(f"Stage 12 circuit {circuit_id} exceeds 258 components.")
        agreement = int(row["prediction_agreement_count"])
        example_count = int(row["evaluated_example_count"])
        if agreement * 100 < 99 * example_count:
            raise ValueError(f"Stage 12 circuit {circuit_id} fails exact 0.99 fidelity.")
        if not math.isclose(
            agreement / example_count,
            float(row["primary_fidelity"]),
            abs_tol=1.0e-15,
            rel_tol=0.0,
        ):
            raise ValueError(f"Stage 12 fidelity/count mismatch for {circuit_id}.")
        if not bool(record["locally_single_deletion_minimal"]):
            raise ValueError(f"Stage 12 circuit {circuit_id} is not locally minimal.")
        if not bool(record["meaningfully_sparse"]):
            raise ValueError(f"Stage 12 circuit {circuit_id} is not meaningfully sparse.")
        circuits.append(
            Stage12Circuit(
                circuit_id=circuit_id,
                member_index=index,
                mask=mask,
                mask_member_name=member_name,
                mask_sha256=observed_hash,
                global_metrics=dict(record["metrics"]),
                stage12_row=dict(row),
            )
        )

    overlaps: dict[tuple[str, str], Fraction] = {}
    primary_overlap_rows = [
        row for row in _read_csv(overlap_path) if row["cell_id"] == PRIMARY_CELL_ID
    ]
    if len(primary_overlap_rows) != 21:
        raise ValueError("Stage 12 primary overlap table must contain 21 pairs.")
    by_id = {circuit.circuit_id: circuit for circuit in circuits}
    for row in primary_overlap_rows:
        left = row["left_member_label"]
        right = row["right_member_label"]
        intersection, union = jaccard_counts(by_id[left].mask, by_id[right].mask)
        exact = Fraction(intersection, union)
        recorded = Fraction(int(row["jaccard_numerator"]), int(row["jaccard_denominator"]))
        if exact != recorded or exact > JACCARD_CUTOFF:
            raise ValueError(f"Stage 12 overlap mismatch for {left}, {right}.")
        overlaps[(left, right)] = exact
    return tuple(circuits), overlaps


def validate_transfer_partition(
    context: CheckpointEvaluationContext,
    *,
    subset_source_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str], str]:
    """Validate Q1-Q4 membership, ordering, labels and canonical hashes."""

    pairs = context.inputs[:, :2].detach().cpu().numpy().astype(np.int64, copy=False)
    subsets = generate_input_subsets(pairs)
    if tuple(subsets) != SUBSET_NAMES:
        raise ValueError("Transfer subset identifiers or ordering changed.")
    observed: set[int] = set()
    hashes: dict[str, str] = {}
    for name in SUBSET_NAMES:
        indices = np.asarray(subsets[name], dtype=np.int64)
        if indices.ndim != 1 or indices.size != EXPECTED_SUBSET_COUNTS[name]:
            raise ValueError(f"Transfer subset {name} count mismatch.")
        if len(set(indices.tolist())) != indices.size:
            raise ValueError(f"Transfer subset {name} contains duplicate indices.")
        if indices.size > 1 and not bool(np.all(indices[1:] > indices[:-1])):
            raise ValueError(f"Transfer subset {name} ordering is not deterministic.")
        values = set(indices.tolist())
        if observed.intersection(values):
            raise ValueError("Transfer subsets are not pairwise disjoint.")
        observed.update(values)
        hashes[name] = canonical_json_sha256(
            {"subset_id": name, "ordered_indices": indices.tolist()}
        )
    if observed != set(range(12_769)):
        raise ValueError("Transfer subset union is not the full 12,769-example universe.")
    targets = context.targets.detach().cpu().numpy().astype(np.int64, copy=False)
    expected_targets = (pairs[:, 0] + pairs[:, 1]) % 113
    if not np.array_equal(targets, expected_targets):
        raise ValueError("Transfer partition does not use genuine modular-addition labels.")
    return dict(subsets), hashes, file_sha256(subset_source_path)


def _stage_output_matches(root: Path, stage: int) -> tuple[Path, ...]:
    tokens = (f"stage{stage}", f"stage_{stage}")
    matches: list[Path] = []
    for directory in (root / "manifests", root / "results", root / "figures"):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            compact = path.name.lower().replace("-", "").replace("_", "")
            if path.is_file() and any(token.replace("_", "") in compact for token in tokens):
                matches.append(path)
    return tuple(sorted(matches))


def validate_stage16_inputs(
    *,
    repository_root: str | Path,
    expected_implementation_commit: str,
    stage12_manifest: str | Path | None = None,
    checkpoint_step: int = 9050,
    device: str = "cpu",
    output_root: str | Path | None = None,
    require_outputs_absent: bool = True,
) -> Stage16InputValidation:
    """Validate every frozen Stage 16 input without creating files."""

    repository = Path(repository_root).resolve()
    selected_output = repository if output_root is None else Path(output_root).resolve()
    configuration = load_stage16_configuration(repository)
    if stage12_manifest is not None:
        requested = Path(stage12_manifest)
        if not requested.is_absolute():
            requested = repository / requested
        configured = repository / configuration.payload["stage12_manifest"]
        if requested.resolve() != configured.resolve():
            raise ValueError("Only the frozen Stage 12 manifest is permitted.")
    if checkpoint_step != configuration.payload["checkpoint_step"]:
        raise ValueError("Only frozen checkpoint step 9050 is permitted.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Stage 16 supports only CPU or CUDA; MPS is prohibited.")
    commit = current_git_commit(repository)
    if commit != expected_implementation_commit:
        raise ValueError(
            f"Implementation commit mismatch: expected {expected_implementation_commit}, "
            f"found {commit}."
        )
    clean = repository_is_clean(repository)
    if not clean:
        raise ValueError("Stage 16 validation requires a clean repository.")
    if require_outputs_absent and _stage_output_matches(selected_output, 16):
        raise FileExistsError("Stage 16 scientific outputs already exist.")
    if _stage_output_matches(selected_output, 17):
        raise FileExistsError("Stage 17 outputs exist; Stage 16 must stop.")

    stage15 = _load_json_object(
        repository / configuration.payload["stage15_manifest"],
        "Stage 15 unavailable manifest",
    )
    if (
        stage15.get("status") != "unavailable"
        or stage15.get("family_size") is not None
        or stage15.get("transfer_group_count") is not None
        or stage15.get("replacement_control_introduced") is not False
    ):
        raise ValueError("Stage 15 unavailable semantics changed.")

    context = load_checkpoint_evaluation_context(
        repository_root=repository,
        run_id=configuration.payload["source_training_run_id"],
        checkpoint_manifest_path=configuration.payload["checkpoint_manifest"],
        checkpoint_step=checkpoint_step,
        device_override=device,
    )
    if context.checkpoint_phase != "stable post-grokking":
        raise ValueError("Stage 16 checkpoint is not the stable post-grokking checkpoint.")
    circuits, overlaps = load_and_validate_stage12_family(repository, configuration)
    subset_source = repository / "src/circuit_families/data/input_subsets.py"
    subset_indices, subset_hashes, subset_source_hash = validate_transfer_partition(
        context, subset_source_path=subset_source
    )
    return Stage16InputValidation(
        repository=repository,
        output_root=selected_output,
        configuration=configuration,
        implementation_commit=commit,
        repository_clean=clean,
        context=context,
        circuits=circuits,
        overlaps=overlaps,
        subset_indices=subset_indices,
        subset_hashes=subset_hashes,
        subset_source_sha256=subset_source_hash,
    )


def _stable_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _correct_count(metrics: MaskEvaluationMetrics) -> int:
    value = metrics.masked_accuracy * metrics.evaluated_example_count
    rounded = int(round(value))
    if not math.isclose(value, rounded, abs_tol=1.0e-8, rel_tol=0.0):
        raise ValueError("Masked accuracy does not reconstruct an integer correct count.")
    return rounded


def _full_correct_count(metrics: MaskEvaluationMetrics) -> int:
    value = metrics.full_accuracy * metrics.evaluated_example_count
    rounded = int(round(value))
    if not math.isclose(value, rounded, abs_tol=1.0e-8, rel_tol=0.0):
        raise ValueError("Full accuracy does not reconstruct an integer correct count.")
    return rounded


def _gradient_and_hook_evidence(context: CheckpointEvaluationContext) -> tuple[bool, int]:
    gradients_absent = all(parameter.grad is None for parameter in context.model.parameters())
    hook_count = sum(
        len(context.model.hook_dict[name]._forward_hooks)
        for name in (ATTENTION_HEAD_HOOK_NAME, MLP_NEURON_HOOK_NAME)
    )
    return gradients_absent, hook_count


def _metrics_fields(metrics: MaskEvaluationMetrics) -> dict[str, object]:
    return {
        "subset_example_count": metrics.evaluated_example_count,
        "prediction_agreement_count": metrics.prediction_agreement_count,
        "fidelity": metrics.primary_fidelity,
        "correct_prediction_count": _correct_count(metrics),
        "ground_truth_accuracy": metrics.masked_accuracy,
        "cross_entropy": metrics.masked_cross_entropy,
        "kl_divergence": metrics.mean_kl_divergence,
        "jensen_shannon_divergence": metrics.mean_jensen_shannon_divergence,
    }


def _validate_metrics_against_record(
    metrics: MaskEvaluationMetrics,
    recorded: Mapping[str, Any],
    *,
    circuit_id: str,
) -> None:
    observed = metrics.to_record()
    required = (
        "primary_fidelity",
        "prediction_agreement_count",
        "full_accuracy",
        "masked_accuracy",
        "accuracy_change",
        "full_cross_entropy",
        "masked_cross_entropy",
        "cross_entropy_change",
        "mean_kl_divergence",
        "mean_jensen_shannon_divergence",
        "maximum_absolute_logit_difference",
        "retained_attention_head_count",
        "retained_mlp_neuron_count",
        "retained_component_count",
        "retained_component_proportion",
        "evaluated_example_count",
        "evaluation_batch_size",
    )
    for field in required:
        if field not in recorded:
            raise ValueError(f"Stage 12 metric {field!r} is absent for {circuit_id}.")
        left = observed[field]
        right = recorded[field]
        if isinstance(left, float):
            if not math.isclose(left, float(right), abs_tol=1.0e-12, rel_tol=0.0):
                raise RuntimeError(
                    f"Stage 12 metric {field!r} mismatch for {circuit_id}: "
                    f"expected {right}, found {left}."
                )
        elif left != right:
            raise RuntimeError(
                f"Stage 12 metric {field!r} mismatch for {circuit_id}: "
                f"expected {right}, found {left}."
            )


GLOBAL_TRANSFER_COLUMNS = (
    "stage16_run_id",
    "circuit_id",
    "source_family",
    "discovery_domain",
    "evaluation_subset",
    "subset_example_count",
    "prediction_agreement_count",
    "fidelity",
    "correct_prediction_count",
    "ground_truth_accuracy",
    "cross_entropy",
    "kl_divergence",
    "jensen_shannon_divergence",
    "retained_component_count",
    "mask_sha256",
    "checkpoint_sha256",
    "model_state_sha256_before",
    "model_state_sha256_after",
    "hook_count_before",
    "hook_count_after",
)

PROFILE_COLUMNS = (
    "stage16_run_id",
    "circuit_id",
    "q1_fidelity",
    "q2_fidelity",
    "q3_fidelity",
    "q4_fidelity",
    "q1_accuracy",
    "q2_accuracy",
    "q3_accuracy",
    "q4_accuracy",
    "q1_cross_entropy",
    "q2_cross_entropy",
    "q3_cross_entropy",
    "q4_cross_entropy",
    "q1_kl_divergence",
    "q2_kl_divergence",
    "q3_kl_divergence",
    "q4_kl_divergence",
    "q1_jensen_shannon_divergence",
    "q2_jensen_shannon_divergence",
    "q3_jensen_shannon_divergence",
    "q4_jensen_shannon_divergence",
)

DISTANCE_COLUMNS = (
    "stage16_run_id",
    "circuit_i",
    "circuit_j",
    "circuit_i_profile_json",
    "circuit_j_profile_json",
    "q1_absolute_difference",
    "q2_absolute_difference",
    "q3_absolute_difference",
    "q4_absolute_difference",
    "maximum_absolute_difference",
    "maximum_distance_subset",
    "structural_jaccard_overlap",
    "structural_jaccard_distance",
    "same_group_at_0_025",
    "same_group_at_0_050",
    "same_group_at_0_100",
)

GROUP_COLUMNS = (
    "stage16_run_id",
    "tolerance_numerator",
    "tolerance_denominator",
    "tolerance",
    "group_count",
    "group_id",
    "ordered_members_json",
    "within_group_maximum_distance",
    "between_group_minimum_distance",
    "complete_linkage_valid",
)

DISCOVERY_COLUMNS = (
    "stage16_run_id",
    "discovery_subset",
    "discovery_status",
    "search_status",
    "circuit_id",
    "mask_sha256",
    "retained_component_count",
    "discovery_subset_fidelity",
    "prediction_agreement_count",
    "evaluated_example_count",
    "exact_evaluations_used",
    "ranking_passes_used",
    "candidate_batches_tested",
    "locally_single_deletion_minimal",
    "meaningfully_sparse",
    "transfer_eligible",
    "stopping_reason",
    "raw_search_directory",
)

SUBSET_TRANSFER_COLUMNS = (
    "stage16_run_id",
    "discovery_subset",
    "evaluation_subset",
    "discovery_status",
    "circuit_id",
    "subset_example_count",
    "prediction_agreement_count",
    "fidelity",
    "correct_prediction_count",
    "ground_truth_accuracy",
    "cross_entropy",
    "kl_divergence",
    "jensen_shannon_divergence",
    "retained_component_count",
    "mask_sha256",
)

RUNTIME_COLUMNS = (
    "stage16_run_id",
    "workload",
    "cell_id",
    "exact_evaluations_used",
    "elapsed_seconds",
)


def _group_maps(
    profiles: Sequence[TransferProfile],
) -> tuple[dict[Fraction, dict[str, str]], list[dict[str, object]]]:
    group_maps: dict[Fraction, dict[str, str]] = {}
    rows: list[dict[str, object]] = []
    pair_distances = {
        tuple(sorted((left.circuit_id, right.circuit_id))): max(
            abs(a - b) for a, b in zip(left.values, right.values, strict=True)
        )
        for left_index, left in enumerate(profiles)
        for right in profiles[left_index + 1 :]
    }
    for grouping in grouping_sensitivity(profiles, tolerances=GROUPING_TOLERANCES):
        mapping: dict[str, str] = {}
        for group_index, members in enumerate(grouping.groups, start=1):
            group_id = f"G{group_index:02d}"
            mapping.update({member: group_id for member in members})
        group_maps[grouping.tolerance] = mapping
        for group_index, members in enumerate(grouping.groups, start=1):
            within_values = [
                pair_distances[tuple(sorted((left, right)))]
                for left_index, left in enumerate(members)
                for right in members[left_index + 1 :]
            ]
            outside = tuple(
                profile.circuit_id
                for profile in profiles
                if profile.circuit_id not in members
            )
            between_values = [
                pair_distances[tuple(sorted((left, right)))]
                for left in members
                for right in outside
            ]
            within = max(within_values, default=0.0)
            rows.append(
                {
                    "stage16_run_id": "",
                    "tolerance_numerator": grouping.tolerance.numerator,
                    "tolerance_denominator": grouping.tolerance.denominator,
                    "tolerance": float(grouping.tolerance),
                    "group_count": grouping.group_count,
                    "group_id": f"G{group_index:02d}",
                    "ordered_members_json": json.dumps(list(members), separators=(",", ":")),
                    "within_group_maximum_distance": within,
                    "between_group_minimum_distance": min(between_values, default=""),
                    "complete_linkage_valid": within <= float(grouping.tolerance),
                }
            )
    return group_maps, rows


def _pairwise_rows(
    *,
    run_id: str,
    profiles: Sequence[TransferProfile],
    overlaps: Mapping[tuple[str, str], Fraction],
    group_maps: Mapping[Fraction, Mapping[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(profiles):
        for right in profiles[left_index + 1 :]:
            differences = tuple(abs(a - b) for a, b in zip(left.values, right.values, strict=True))
            maximum = max(differences)
            maximum_subset = SUBSET_NAMES[differences.index(maximum)]
            key = tuple(sorted((left.circuit_id, right.circuit_id)))
            overlap = overlaps[key]
            rows.append(
                {
                    "stage16_run_id": run_id,
                    "circuit_i": left.circuit_id,
                    "circuit_j": right.circuit_id,
                    "circuit_i_profile_json": json.dumps(list(left.values), separators=(",", ":")),
                    "circuit_j_profile_json": json.dumps(list(right.values), separators=(",", ":")),
                    "q1_absolute_difference": differences[0],
                    "q2_absolute_difference": differences[1],
                    "q3_absolute_difference": differences[2],
                    "q4_absolute_difference": differences[3],
                    "maximum_absolute_difference": maximum,
                    "maximum_distance_subset": maximum_subset,
                    "structural_jaccard_overlap": float(overlap),
                    "structural_jaccard_distance": float(1 - overlap),
                    "same_group_at_0_025": group_maps[Fraction(1, 40)][left.circuit_id]
                    == group_maps[Fraction(1, 40)][right.circuit_id],
                    "same_group_at_0_050": group_maps[Fraction(1, 20)][left.circuit_id]
                    == group_maps[Fraction(1, 20)][right.circuit_id],
                    "same_group_at_0_100": group_maps[Fraction(1, 10)][left.circuit_id]
                    == group_maps[Fraction(1, 10)][right.circuit_id],
                }
            )
    if len(rows) != 21:
        raise RuntimeError("Seven global circuits must produce exactly 21 pairs.")
    return rows


def _discovery_status(execution: CheckpointSearchExecution) -> str:
    result = execution.result
    if result.status == "valid_sparse_circuit":
        return "valid_meaningfully_sparse"
    if result.status == "ranking_failure":
        return "search_failure"
    return result.status


def null_subset_transfer_rows(
    *,
    run_id: str,
    discovery_subset: str,
    discovery_status: str,
) -> list[dict[str, object]]:
    """Return an explicit four-cell null row for failed discovery."""

    if discovery_subset not in SUBSET_NAMES:
        raise ValueError(f"Unknown discovery subset: {discovery_subset}")
    return [
        {
            "stage16_run_id": run_id,
            "discovery_subset": discovery_subset,
            "evaluation_subset": evaluation_subset,
            "discovery_status": discovery_status,
            "circuit_id": "",
            "subset_example_count": EXPECTED_SUBSET_COUNTS[evaluation_subset],
            "prediction_agreement_count": "",
            "fidelity": "",
            "correct_prediction_count": "",
            "ground_truth_accuracy": "",
            "cross_entropy": "",
            "kl_divergence": "",
            "jensen_shannon_divergence": "",
            "retained_component_count": "",
            "mask_sha256": "",
        }
        for evaluation_subset in SUBSET_NAMES
    ]


def _note_text(
    *,
    run_id: str,
    validation: Stage16InputValidation,
    group_counts: Mapping[Fraction, int | None],
    discovery_rows: Sequence[Mapping[str, object]],
) -> str:
    primary = group_counts[PRIMARY_GROUPING_TOLERANCE]
    if primary == 1:
        conclusion = (
            "The seven structurally distinct circuits are largely functionally "
            "interchangeable under the frozen transfer rule."
        )
    elif isinstance(primary, int) and primary > 1:
        conclusion = "Multiple transfer-distinct functional groups persist."
    else:
        conclusion = "The result is mixed or unresolved."
    outcomes = "\n".join(
        f"- {row['discovery_subset']}: `{row['discovery_status']}`"
        for row in discovery_rows
    )
    counts = ", ".join(
        f"{float(tolerance):.3f}: {group_counts[tolerance]}"
        for tolerance in GROUPING_TOLERANCES
    )
    return (
        "# Stage 16 genuine-task functional-transfer analysis\n\n"
        f"- Stage 16 run: `{run_id}`\n"
        "- Source training run: `modular-addition-training-s1-5f1bc9dee7ab`\n"
        "- Model seed: `1`\n"
        "- Stable post-grokking checkpoint: step `9050`\n"
        "- Source structural family: Stage 12 primary `cutoff-0.50` cell\n"
        "- Structural family size: `7`\n"
        "- Fidelity threshold: `0.99`\n"
        "- Sparsity boundary: at most `258 / 516` components\n"
        "- Transfer subsets: Q1 lower/lower, Q2 lower/higher, "
        "Q3 higher/lower, Q4 higher/higher\n"
        f"- Transfer-distinct group counts ({counts})\n\n"
        "## Subset-discovery outcomes\n\n"
        f"{outcomes}\n\n"
        "## Primary conclusion\n\n"
        f"{conclusion}\n\n"
        "The group count is procedure-dependent: it is the transfer-distinct "
        "group count under the frozen fidelity profile, maximum-distance rule, "
        "complete linkage and tolerance. It is not the true number of mechanisms. "
        "Transfer-equivalent circuits are not thereby mechanistically identical.\n\n"
        "Subset discovery is a bounded greedy search. Failure or budget exhaustion "
        "does not establish that no eligible sparse circuit exists. Stage 15 remains "
        "unavailable rather than an executed empty family. Stage 17 was not begun.\n"
    )


def execute_stage16(
    *,
    repository_root: str | Path,
    expected_implementation_commit: str,
    stage12_manifest: str | Path | None = None,
    checkpoint_step: int = 9050,
    device: str = "cpu",
    output_root: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Stage16ExecutionResult:
    """Execute the complete, clean-commit-gated Stage 16 workload."""

    validation = validate_stage16_inputs(
        repository_root=repository_root,
        expected_implementation_commit=expected_implementation_commit,
        stage12_manifest=stage12_manifest,
        checkpoint_step=checkpoint_step,
        device=device,
        output_root=output_root,
        require_outputs_absent=True,
    )
    root = validation.output_root
    run_id = validation.configuration.run_id
    raw_root = root / "results/raw" / run_id
    table_root = root / "results/tables"
    note = root / "results/notes/seed_1_stage16_functional_transfer.md"
    archive = root / "results/archives" / f"{run_id}.tar.gz"
    manifest = root / "manifests" / f"stage16_transfer_{run_id}.json"
    runtime_table = table_root / "seed_1_stage16_runtime.csv"
    table_paths = {
        name: table_root / f"seed_1_stage16_{name}.csv"
        for name in SCIENTIFIC_TABLE_NAMES
    }
    generated_roots = [raw_root, *table_paths.values(), note, archive, manifest, runtime_table]
    context = validation.context
    model_state_before = canonical_state_hash(context.model.state_dict())
    gradients_before, hooks_before = _gradient_and_hook_evidence(context)
    if model_state_before != context.model_state_sha256 or not gradients_before:
        raise RuntimeError("Initial Stage 16 model integrity check failed.")

    global_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    discovery_rows: list[dict[str, object]] = []
    subset_transfer_rows: list[dict[str, object]] = []
    global_transfers: list[TransferEvaluation] = []
    discovery_executions: dict[str, CheckpointSearchExecution] = {}
    all_retained_records: list[dict[str, object]] = []

    try:
        raw_root.mkdir(parents=True, exist_ok=False)
        evaluation_batch_size = int(validation.configuration.payload["evaluation_batch_size"])
        ranking_batch_size = int(validation.configuration.payload["ranking_batch_size"])
        discovery_budget = int(
            validation.configuration.payload[
                "per_subset_discovery_exact_evaluation_budget"
            ]
        )

        for subset_name in SUBSET_NAMES:
            if progress_callback is not None:
                progress_callback(f"all-retained identity: {subset_name}")
            started = time.perf_counter()
            subset = subset_context(context, subset_name)
            reference = compute_full_model_reference(
                subset.model,
                subset.inputs,
                subset.targets,
                batch_size=evaluation_batch_size,
            )
            metrics = evaluate_component_mask(
                subset.model,
                subset.inputs,
                subset.targets,
                ComponentMask.all_retained(),
                batch_size=evaluation_batch_size,
                full_model_reference=reference,
            )
            if metrics.prediction_agreement_count != metrics.evaluated_example_count:
                raise RuntimeError(f"All-retained identity failed on {subset_name}.")
            all_retained_records.append(
                {"subset": subset_name, **metrics.to_record()}
            )
            runtime_rows.append(
                {
                    "stage16_run_id": run_id,
                    "workload": "all_retained_identity",
                    "cell_id": subset_name,
                    "exact_evaluations_used": 1,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        _stable_json(raw_root / "all_retained_subset_identity.json", all_retained_records)

        full_reference = compute_full_model_reference(
            context.model,
            context.inputs,
            context.targets,
            batch_size=evaluation_batch_size,
        )
        revalidation_records: list[dict[str, object]] = []
        for circuit in validation.circuits:
            if progress_callback is not None:
                progress_callback(f"Stage 12 global revalidation: {circuit.circuit_id}")
            started = time.perf_counter()
            metrics = evaluate_component_mask(
                context.model,
                context.inputs,
                context.targets,
                circuit.mask,
                batch_size=evaluation_batch_size,
                full_model_reference=full_reference,
            )
            expected_agreement = int(circuit.stage12_row["prediction_agreement_count"])
            if metrics.prediction_agreement_count != expected_agreement:
                raise RuntimeError(
                    f"Global Stage 12 agreement mismatch for {circuit.circuit_id}."
                )
            recorded = circuit.global_metrics
            _validate_metrics_against_record(
                metrics,
                recorded,
                circuit_id=circuit.circuit_id,
            )
            if _correct_count(metrics) != int(
                round(float(recorded["masked_accuracy"]) * metrics.evaluated_example_count)
            ):
                raise RuntimeError(
                    f"Global Stage 12 correct-count mismatch for {circuit.circuit_id}."
                )
            if metrics.retained_component_count > MEANINGFULLY_SPARSE_MAX_COMPONENTS:
                raise RuntimeError(f"Revalidated circuit {circuit.circuit_id} is nonsparse.")
            revalidation_records.append(
                {
                    "circuit_id": circuit.circuit_id,
                    "mask_sha256": circuit.mask_sha256,
                    **metrics.to_record(),
                }
            )
            runtime_rows.append(
                {
                    "stage16_run_id": run_id,
                    "workload": "stage12_global_revalidation",
                    "cell_id": circuit.circuit_id,
                    "exact_evaluations_used": 1,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        _stable_json(raw_root / "stage12_global_revalidation.json", revalidation_records)

        for circuit in validation.circuits:
            if progress_callback is not None:
                progress_callback(f"global-family transfer: {circuit.circuit_id}")
            started = time.perf_counter()
            before = canonical_state_hash(context.model.state_dict())
            gradients_cell_before, hooks_cell_before = _gradient_and_hook_evidence(context)
            transfer = evaluate_transfer_profile(
                context=context,
                mask=circuit.mask,
                circuit_id=circuit.circuit_id,
                batch_size=evaluation_batch_size,
            )
            after = canonical_state_hash(context.model.state_dict())
            gradients_cell_after, hooks_cell_after = _gradient_and_hook_evidence(context)
            if before != after or not gradients_cell_before or not gradients_cell_after:
                raise RuntimeError(f"Global transfer changed model state for {circuit.circuit_id}.")
            if hooks_cell_before != hooks_cell_after:
                raise RuntimeError(f"Global transfer leaked hooks for {circuit.circuit_id}.")
            global_transfers.append(transfer)
            metrics_by_subset = {
                evaluation.evaluation_subset: evaluation.metrics
                for evaluation in transfer.evaluations
            }
            agreement_sum = sum(
                metrics.prediction_agreement_count for metrics in metrics_by_subset.values()
            )
            correct_sum = sum(_correct_count(metrics) for metrics in metrics_by_subset.values())
            if agreement_sum != int(circuit.stage12_row["prediction_agreement_count"]):
                raise RuntimeError(
                    f"Q1-Q4 agreement reconstruction failed for {circuit.circuit_id}."
                )
            expected_correct = int(
                round(
                    float(circuit.global_metrics["masked_accuracy"])
                    * int(circuit.stage12_row["evaluated_example_count"])
                )
            )
            if correct_sum != expected_correct:
                raise RuntimeError(
                    f"Q1-Q4 accuracy reconstruction failed for {circuit.circuit_id}."
                )
            for evaluation in transfer.evaluations:
                global_rows.append(
                    {
                        "stage16_run_id": run_id,
                        "circuit_id": circuit.circuit_id,
                        "source_family": PRIMARY_CELL_ID,
                        "discovery_domain": "global",
                        "evaluation_subset": evaluation.evaluation_subset,
                        **_metrics_fields(evaluation.metrics),
                        "retained_component_count": circuit.mask.retained_component_count,
                        "mask_sha256": circuit.mask_sha256,
                        "checkpoint_sha256": context.checkpoint_sha256,
                        "model_state_sha256_before": before,
                        "model_state_sha256_after": after,
                        "hook_count_before": hooks_cell_before,
                        "hook_count_after": hooks_cell_after,
                    }
                )
            profile_rows.append(
                {
                    "stage16_run_id": run_id,
                    "circuit_id": circuit.circuit_id,
                    **{
                        f"{name.lower()}_fidelity": metrics_by_subset[name].primary_fidelity
                        for name in SUBSET_NAMES
                    },
                    **{
                        f"{name.lower()}_accuracy": metrics_by_subset[name].masked_accuracy
                        for name in SUBSET_NAMES
                    },
                    **{
                        f"{name.lower()}_cross_entropy": (
                            metrics_by_subset[name].masked_cross_entropy
                        )
                        for name in SUBSET_NAMES
                    },
                    **{
                        f"{name.lower()}_kl_divergence": metrics_by_subset[name].mean_kl_divergence
                        for name in SUBSET_NAMES
                    },
                    **{
                        f"{name.lower()}_jensen_shannon_divergence": (
                            metrics_by_subset[name].mean_jensen_shannon_divergence
                        )
                        for name in SUBSET_NAMES
                    },
                }
            )
            _stable_json(
                raw_root / "global_family_transfer" / f"{circuit.circuit_id}.json",
                {
                    "circuit_id": circuit.circuit_id,
                    "mask_sha256": circuit.mask_sha256,
                    "profile": transfer.profile.as_mapping(),
                    "evaluations": [
                        {
                            "evaluation_subset": evaluation.evaluation_subset,
                            **evaluation.metrics.to_record(),
                        }
                        for evaluation in transfer.evaluations
                    ],
                },
            )
            runtime_rows.append(
                {
                    "stage16_run_id": run_id,
                    "workload": "global_family_transfer",
                    "cell_id": circuit.circuit_id,
                    "exact_evaluations_used": len(SUBSET_NAMES),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        if len(global_rows) != 28:
            raise RuntimeError("Stage 16 must produce exactly 28 global transfer rows.")

        profiles = tuple(transfer.profile for transfer in global_transfers)
        group_maps, group_rows = _group_maps(profiles)
        for row in group_rows:
            row["stage16_run_id"] = run_id
        if not all(bool(row["complete_linkage_valid"]) for row in group_rows):
            raise RuntimeError("Complete-linkage validation failed.")
        pairwise_rows = _pairwise_rows(
            run_id=run_id,
            profiles=profiles,
            overlaps=validation.overlaps,
            group_maps=group_maps,
        )

        for subset_name in SUBSET_NAMES:
            if progress_callback is not None:
                progress_callback(f"subset discovery and transfer: {subset_name}")
            started = time.perf_counter()
            search_context = subset_context(context, subset_name)
            execution = run_checkpoint_sparse_search(
                search_context,
                fidelity_threshold=float(FIDELITY_THRESHOLD),
                ranking_batch_size=ranking_batch_size,
                evaluation_batch_size=evaluation_batch_size,
                exact_evaluation_budget=discovery_budget,
            )
            discovery_executions[subset_name] = execution
            result = execution.result
            search_directory = raw_root / "subset_discovery" / subset_name / "search"
            artifacts = write_sparse_search_artifacts(
                search_directory,
                result,
                cell_metadata={
                    "schema_version": 1,
                    "experiment_stage": 16,
                    "stage16_run_id": run_id,
                    "source_training_run_id": context.run_id,
                    "checkpoint_step": context.checkpoint_step,
                    "discovery_subset": subset_name,
                    "fidelity_threshold_numerator": 99,
                    "fidelity_threshold_denominator": 100,
                    "exact_evaluation_budget": discovery_budget,
                    "implementation_commit": expected_implementation_commit,
                },
            )
            status = _discovery_status(execution)
            eligible = (
                result.status == "valid_sparse_circuit"
                and result.final_metrics.prediction_agreement_count * 100
                >= 99 * result.final_metrics.evaluated_example_count
                and result.final_mask.retained_component_count
                <= MEANINGFULLY_SPARSE_MAX_COMPONENTS
            )
            circuit_id = f"{subset_name}-discovered" if eligible else ""
            discovery_rows.append(
                {
                    "stage16_run_id": run_id,
                    "discovery_subset": subset_name,
                    "discovery_status": status,
                    "search_status": result.status,
                    "circuit_id": circuit_id,
                    "mask_sha256": artifacts.final_mask_sha256,
                    "retained_component_count": result.final_mask.retained_component_count,
                    "discovery_subset_fidelity": result.final_metrics.primary_fidelity,
                    "prediction_agreement_count": result.final_metrics.prediction_agreement_count,
                    "evaluated_example_count": result.final_metrics.evaluated_example_count,
                    "exact_evaluations_used": result.exact_evaluations_used,
                    "ranking_passes_used": result.ranking_passes_used,
                    "candidate_batches_tested": result.candidate_batches_tested,
                    "locally_single_deletion_minimal": result.locally_single_deletion_minimal,
                    "meaningfully_sparse": result.meaningfully_sparse,
                    "transfer_eligible": eligible,
                    "stopping_reason": result.stopping_reason,
                    "raw_search_directory": _relative(root, search_directory),
                }
            )
            if eligible:
                transfer = evaluate_transfer_profile(
                    context=context,
                    mask=result.final_mask,
                    circuit_id=circuit_id,
                    discovery_subset=subset_name,
                    batch_size=evaluation_batch_size,
                )
                diagonal = transfer.evaluations[SUBSET_NAMES.index(subset_name)]
                if diagonal.metrics.primary_fidelity < 0.99:
                    raise RuntimeError(f"Discovery diagonal fidelity failed for {subset_name}.")
                for evaluation in transfer.evaluations:
                    subset_transfer_rows.append(
                        {
                            "stage16_run_id": run_id,
                            "discovery_subset": subset_name,
                            "evaluation_subset": evaluation.evaluation_subset,
                            "discovery_status": status,
                            "circuit_id": circuit_id,
                            **_metrics_fields(evaluation.metrics),
                            "retained_component_count": result.final_mask.retained_component_count,
                            "mask_sha256": artifacts.final_mask_sha256,
                        }
                    )
                _stable_json(
                    raw_root / "subset_transfer" / f"{subset_name}.json",
                    {
                        "discovery_subset": subset_name,
                        "circuit_id": circuit_id,
                        "mask_sha256": artifacts.final_mask_sha256,
                        "evaluations": [
                            {
                                "evaluation_subset": evaluation.evaluation_subset,
                                **evaluation.metrics.to_record(),
                            }
                            for evaluation in transfer.evaluations
                        ],
                    },
                )
            else:
                subset_transfer_rows.extend(
                    null_subset_transfer_rows(
                        run_id=run_id,
                        discovery_subset=subset_name,
                        discovery_status=status,
                    )
                )
            runtime_rows.append(
                {
                    "stage16_run_id": run_id,
                    "workload": "subset_discovery_and_transfer",
                    "cell_id": subset_name,
                    "exact_evaluations_used": result.exact_evaluations_used
                    + (len(SUBSET_NAMES) if eligible else 0),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        if len(discovery_rows) != 4 or len(subset_transfer_rows) != 16:
            raise RuntimeError("Stage 16 subset-discovery matrix shape is invalid.")

        write_csv_records(
            table_paths["global_family_transfer"],
            fieldnames=GLOBAL_TRANSFER_COLUMNS,
            rows=global_rows,
        )
        write_csv_records(
            table_paths["transfer_profiles"],
            fieldnames=PROFILE_COLUMNS,
            rows=profile_rows,
        )
        write_csv_records(
            table_paths["transfer_distances"],
            fieldnames=DISTANCE_COLUMNS,
            rows=pairwise_rows,
        )
        write_csv_records(
            table_paths["transfer_groups"],
            fieldnames=GROUP_COLUMNS,
            rows=group_rows,
        )
        write_csv_records(
            table_paths["subset_discovery"],
            fieldnames=DISCOVERY_COLUMNS,
            rows=discovery_rows,
        )
        write_csv_records(
            table_paths["subset_transfer"],
            fieldnames=SUBSET_TRANSFER_COLUMNS,
            rows=subset_transfer_rows,
        )
        write_csv_records(
            table_paths["structural_functional_comparison"],
            fieldnames=DISTANCE_COLUMNS,
            rows=pairwise_rows,
        )
        write_csv_records(runtime_table, fieldnames=RUNTIME_COLUMNS, rows=runtime_rows)

        group_counts = {
            tolerance: len(set(group_maps[tolerance].values()))
            for tolerance in GROUPING_TOLERANCES
        }
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(
            _note_text(
                run_id=run_id,
                validation=validation,
                group_counts=group_counts,
                discovery_rows=discovery_rows,
            ),
            encoding="utf-8",
        )

        if progress_callback is not None:
            progress_callback("integrity audit and deterministic reporting")

        model_state_after = canonical_state_hash(context.model.state_dict())
        gradients_after, hooks_after = _gradient_and_hook_evidence(context)
        if model_state_after != model_state_before:
            raise RuntimeError("Stage 16 changed model state.")
        if not gradients_after or hooks_after != hooks_before:
            raise RuntimeError("Stage 16 gradient or hook cleanup failed.")

        scientific_members = sorted(
            [
                *table_paths.values(),
                note,
                *(path for path in raw_root.rglob("*") if path.is_file()),
            ],
            key=lambda path: _relative(root, path),
        )
        write_deterministic_archive(archive, root=root, members=scientific_members)
        output_records = {
            name: {"path": _relative(root, path), "sha256": file_sha256(path)}
            for name, path in table_paths.items()
        }
        manifest_payload = {
            "schema_version": 1,
            "experiment_stage": 16,
            "stage16_run_id": run_id,
            "implementation_commit": expected_implementation_commit,
            "source_training_run_id": context.run_id,
            "model_seed": 1,
            "checkpoint": {
                "step": context.checkpoint_step,
                "path": _relative(validation.repository, context.checkpoint_path),
                "sha256": context.checkpoint_sha256,
                "model_state_sha256": context.model_state_sha256,
            },
            "stage12": {
                "run_id": validation.configuration.payload["stage12_run_id"],
                "manifest": validation.configuration.payload["stage12_manifest"],
                "manifest_sha256": file_sha256(
                    validation.repository / validation.configuration.payload["stage12_manifest"]
                ),
                "archive": validation.configuration.payload["stage12_archive"],
                "archive_sha256": file_sha256(
                    validation.repository / validation.configuration.payload["stage12_archive"]
                ),
                "circuit_table": validation.configuration.payload["stage12_circuits_table"],
                "circuit_table_sha256": file_sha256(
                    validation.repository
                    / validation.configuration.payload["stage12_circuits_table"]
                ),
                "overlap_table": validation.configuration.payload[
                    "stage12_pairwise_overlap_table"
                ],
                "overlap_table_sha256": file_sha256(
                    validation.repository
                    / validation.configuration.payload["stage12_pairwise_overlap_table"]
                ),
                "primary_family_cell": PRIMARY_CELL_ID,
                "source_family_size": 7,
                "circuits": [
                    {
                        "circuit_id": circuit.circuit_id,
                        "mask_member_name": circuit.mask_member_name,
                        "mask_sha256": circuit.mask_sha256,
                        "retained_component_count": circuit.mask.retained_component_count,
                        "global_prediction_agreement_count": int(
                            circuit.stage12_row["prediction_agreement_count"]
                        ),
                        "global_fidelity": float(circuit.stage12_row["primary_fidelity"]),
                    }
                    for circuit in validation.circuits
                ],
            },
            "frozen_analysis": {
                "configuration": _relative(validation.repository, validation.configuration.path),
                "configuration_sha256": validation.configuration.sha256,
                "fidelity_threshold": {"numerator": 99, "denominator": 100},
                "sparsity_boundary": 258,
                "searchable_component_count": 516,
                "jaccard_cutoff": {"numerator": 1, "denominator": 2},
                "transfer_fidelity_definition": "masked/full top-one prediction agreement",
                "transfer_profile_field_order": list(SUBSET_NAMES),
                "distance_definition": "maximum absolute fidelity-profile difference",
                "linkage_method": "deterministic complete linkage",
                "grouping_tolerances": [
                    {"numerator": value.numerator, "denominator": value.denominator}
                    for value in GROUPING_TOLERANCES
                ],
                "primary_grouping_tolerance": {"numerator": 1, "denominator": 20},
                "group_label_convention": "G01_G02_in_deterministic_cluster_order",
                "per_subset_discovery_budget": discovery_budget,
            },
            "transfer_subsets": {
                "source_path": "src/circuit_families/data/input_subsets.py",
                "source_sha256": validation.subset_source_sha256,
                "membership_hash_convention": (
                    "sha256(canonical_json({subset_id,ordered_indices}))"
                ),
                "records": [
                    {
                        "subset_id": name,
                        "count": EXPECTED_SUBSET_COUNTS[name],
                        "membership_sha256": validation.subset_hashes[name],
                    }
                    for name in SUBSET_NAMES
                ],
                "pairwise_disjoint": True,
                "union_example_count": 12_769,
                "genuine_labels_validated": True,
            },
            "results": {
                "global_transfer_evaluation_count": len(global_rows),
                "structural_pair_count": len(pairwise_rows),
                "subset_discovery_cell_count": len(discovery_rows),
                "valid_subset_discovered_circuit_count": sum(
                    bool(row["transfer_eligible"]) for row in discovery_rows
                ),
                "subset_transfer_evaluation_count": sum(
                    bool(row["circuit_id"]) for row in subset_transfer_rows
                ),
                "discovery_statuses": {
                    str(row["discovery_subset"]): str(row["discovery_status"])
                    for row in discovery_rows
                },
                "transfer_group_counts": {
                    f"{float(tolerance):.3f}": group_counts[tolerance]
                    for tolerance in GROUPING_TOLERANCES
                },
            },
            "integrity": {
                "repository_clean_at_start": validation.repository_clean,
                "model_state_sha256_before": model_state_before,
                "model_state_sha256_after": model_state_after,
                "model_state_unchanged": model_state_before == model_state_after,
                "parameter_gradients_absent": gradients_before and gradients_after,
                "hook_count_before": hooks_before,
                "hook_count_after": hooks_after,
                "hooks_restored_to_baseline": hooks_before == hooks_after,
                "all_retained_subset_fidelity_identity": True,
                "global_agreement_reconstruction": True,
                "global_accuracy_reconstruction": True,
                "complete_linkage_validation": True,
                "runtime_excluded_from_deterministic_scientific_hashes": True,
            },
            "outputs": {
                **output_records,
                "note": {"path": _relative(root, note), "sha256": file_sha256(note)},
                "runtime_table": {
                    "path": _relative(root, runtime_table),
                    "sha256": file_sha256(runtime_table),
                    "deterministic_scientific_output": False,
                },
                "archive": {"path": _relative(root, archive), "sha256": file_sha256(archive)},
            },
            "software": {
                "python": __import__("sys").version,
                "numpy": np.__version__,
                "device": str(context.device),
            },
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "stage15_status": "unavailable",
            "stage17_started": False,
        }
        _stable_json(manifest, manifest_payload)
        if progress_callback is not None:
            progress_callback("Stage 16 definitive execution complete")
        return Stage16ExecutionResult(
            run_id=run_id,
            implementation_commit=expected_implementation_commit,
            manifest=manifest,
            archive=archive,
            note=note,
            runtime_table=runtime_table,
            scientific_tables=tuple(table_paths.values()),
        )
    except Exception:
        if raw_root.exists():
            shutil.rmtree(raw_root)
        for path in generated_roots[1:]:
            if path.is_file():
                path.unlink()
        raise


def compare_reproduction(
    *,
    reference_root: str | Path,
    reproduction_root: str | Path,
) -> dict[str, Any]:
    """Compare deterministic Stage 16 outputs byte-for-byte."""

    reference = Path(reference_root).resolve()
    reproduction = Path(reproduction_root).resolve()
    relative_paths = [
        *(Path("results/tables") / f"seed_1_stage16_{name}.csv" for name in SCIENTIFIC_TABLE_NAMES),
        Path("results/notes/seed_1_stage16_functional_transfer.md"),
    ]
    reference_archives = sorted(
        (reference / "results/archives").glob("stage16-transfer-s1-*.tar.gz")
    )
    reproduction_archives = sorted(
        (reproduction / "results/archives").glob("stage16-transfer-s1-*.tar.gz")
    )
    if len(reference_archives) != 1 or len(reproduction_archives) != 1:
        raise ValueError("Expected exactly one Stage 16 archive in each comparison root.")
    relative_paths.append(reference_archives[0].relative_to(reference))
    comparisons = []
    for relative in relative_paths:
        left = reference / relative
        right = reproduction / relative
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(relative)
        left_hash = file_sha256(left)
        right_hash = file_sha256(right)
        comparisons.append(
            {
                "path": relative.as_posix(),
                "reference_sha256": left_hash,
                "reproduction_sha256": right_hash,
                "identical": left_hash == right_hash,
            }
        )
    if not all(record["identical"] for record in comparisons):
        raise RuntimeError("Deterministic Stage 16 reproduction comparison failed.")
    return {
        "deterministic_outputs_identical": True,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "runtime_compared_semantically": True,
        "manifest_timestamp_compared_semantically": True,
    }
