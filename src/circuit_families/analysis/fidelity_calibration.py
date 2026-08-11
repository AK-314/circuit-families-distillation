"""Deterministic core logic for Stage 11 fidelity calibration."""

from __future__ import annotations

import csv
import io
import json
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from circuit_families.interpretability.masks import ComponentMask

COMPONENT_UNIVERSE_SIZE: Final = 516
ATTENTION_HEAD_COUNT: Final = 4
MLP_NEURON_COUNT: Final = 512
RANDOM_MASKS_PER_THRESHOLD: Final = 100
MEANINGFULLY_SPARSE_MAX_COMPONENTS: Final = 258
MAX_RANDOM_MASK_PASSES: Final = 5
MAX_EXACT_EVALUATIONS: Final = 10_000
FULL_DATASET_EXAMPLE_COUNT: Final = 12_769
PERCENTILE_METHOD: Final = "linear"
BIT_GENERATOR_NAME: Final = "PCG64"

SOURCE_TRAINING_RUN_ID: Final = "modular-addition-training-s1-5f1bc9dee7ab"
STABLE_POST_CHECKPOINT_STEP: Final = 9050

CANDIDATE_THRESHOLDS: Final[tuple[Fraction, ...]] = (
    Fraction(99, 100),
    Fraction(39, 40),
    Fraction(19, 20),
    Fraction(9, 10),
    Fraction(17, 20),
    Fraction(4, 5),
)

PROHIBITED_SELECTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "pre_grokking",
        "transition",
        "pre_post_delta",
        "circuit_size_difference",
        "family_size_difference",
        "diversity_family_count",
        "family_count",
        "random_label",
        "random_label_results",
        "no_generalisation",
        "no_generalisation_results",
        "other_seed",
        "across_seed",
        "across_seed_outcomes",
        "anticipated_stage12_behaviour",
        "anticipated_stage12_behavior",
        "hypothesis_effect",
    }
)


@dataclass(frozen=True)
class DerivedRandomSeed:
    """Canonical threshold-specific random stream definition."""

    threshold: Fraction
    threshold_decimal: str
    seed_material: str
    seed_digest: str
    seed_uint64: int
    bit_generator: str
    numpy_version: str


@dataclass(frozen=True)
class SampledMask:
    """One uniformly sampled retained subset over the 516 components."""

    mask_index: int
    retained_indices: tuple[int, ...]
    retained_component_identifiers: tuple[str, ...]
    head_mask: tuple[int, ...]
    neuron_mask: tuple[int, ...]
    mask_sha256: str

    @property
    def retained_head_count(self) -> int:
        return sum(self.head_mask)

    @property
    def retained_neuron_count(self) -> int:
        return sum(self.neuron_mask)

    @property
    def retained_component_count(self) -> int:
        return len(self.retained_indices)


@dataclass(frozen=True)
class CalibrationCandidate:
    """Permitted evidence for mechanical Stage 11 qualification."""

    threshold: Fraction
    retained_components: int
    random_mask_pass_count: int
    fourier_compatible_or_explained: bool
    exact_evaluations: int

    def __post_init__(self) -> None:
        if self.threshold not in CANDIDATE_THRESHOLDS:
            raise ValueError(f"unexpected candidate threshold: {self.threshold}")
        if not 0 <= self.retained_components <= COMPONENT_UNIVERSE_SIZE:
            raise ValueError("retained_components is outside the component universe")
        if not 0 <= self.random_mask_pass_count <= RANDOM_MASKS_PER_THRESHOLD:
            raise ValueError("random_mask_pass_count must be between 0 and 100")
        if self.exact_evaluations < 0:
            raise ValueError("exact_evaluations must be non-negative")


@dataclass(frozen=True)
class QualificationResult:
    """Mechanical qualification result for one candidate threshold."""

    threshold: Fraction
    meaningfully_sparse: bool
    random_mask_pass_count_at_most_5: bool
    fourier_compatible_or_explained: bool
    within_10000_evaluation_budget: bool
    qualifies: bool


def component_identifiers() -> tuple[str, ...]:
    """Return the frozen Stage 8/9 component ordering."""

    return tuple(
        [f"H{index}" for index in range(ATTENTION_HEAD_COUNT)]
        + [f"N{index}" for index in range(MLP_NEURON_COUNT)]
    )


def threshold_decimal(threshold: Fraction) -> str:
    """Format a frozen candidate threshold to exactly six decimal places."""

    if threshold not in CANDIDATE_THRESHOLDS:
        raise ValueError(f"unexpected candidate threshold: {threshold}")
    return f"{float(threshold):.6f}"


def derive_random_seed(
    threshold: Fraction,
    *,
    training_run_id: str = SOURCE_TRAINING_RUN_ID,
    checkpoint_step: int = STABLE_POST_CHECKPOINT_STEP,
    component_universe: int = COMPONENT_UNIVERSE_SIZE,
    replicates: int = RANDOM_MASKS_PER_THRESHOLD,
) -> DerivedRandomSeed:
    """Derive the frozen threshold-specific PCG64 seed."""

    formatted_threshold = threshold_decimal(threshold)
    material = (
        "circuit-families|stage11-random-mask-calibration|"
        f"training_run={training_run_id}|"
        f"checkpoint_step={checkpoint_step}|"
        f"threshold={formatted_threshold}|"
        f"component_universe={component_universe}|"
        f"replicates={replicates}"
    )
    digest = sha256(material.encode("utf-8")).hexdigest()
    seed_uint64 = int(digest[:16], 16)
    return DerivedRandomSeed(
        threshold=threshold,
        threshold_decimal=formatted_threshold,
        seed_material=material,
        seed_digest=digest,
        seed_uint64=seed_uint64,
        bit_generator=BIT_GENERATOR_NAME,
        numpy_version=np.__version__,
    )


def mask_sha256(
    head_mask: Sequence[int],
    neuron_mask: Sequence[int],
) -> str:
    """Hash the canonical binary mask serialization."""

    if len(head_mask) != ATTENTION_HEAD_COUNT:
        raise ValueError("head mask must contain exactly four values")
    if len(neuron_mask) != MLP_NEURON_COUNT:
        raise ValueError("neuron mask must contain exactly 512 values")
    if any(value not in (0, 1) for value in (*head_mask, *neuron_mask)):
        raise ValueError("mask values must be binary")

    serialization = (
        "head_mask="
        + ",".join(str(value) for value in head_mask)
        + "\n"
        + "neuron_mask="
        + ",".join(str(value) for value in neuron_mask)
        + "\n"
    )
    return sha256(serialization.encode("utf-8")).hexdigest()


def sample_matched_size_masks(
    threshold: Fraction,
    *,
    retained_count: int,
    replicates: int = RANDOM_MASKS_PER_THRESHOLD,
) -> tuple[SampledMask, ...]:
    """Uniformly sample retained subsets using the frozen PCG64 stream."""

    if threshold not in CANDIDATE_THRESHOLDS:
        raise ValueError(f"unexpected candidate threshold: {threshold}")
    if not 0 <= retained_count <= COMPONENT_UNIVERSE_SIZE:
        raise ValueError("retained_count is outside the component universe")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    seed = derive_random_seed(threshold, replicates=replicates)
    generator = np.random.Generator(np.random.PCG64(seed.seed_uint64))
    identifiers = component_identifiers()
    sampled: list[SampledMask] = []

    for mask_index in range(replicates):
        retained_indices = tuple(
            sorted(
                int(index)
                for index in generator.choice(
                    COMPONENT_UNIVERSE_SIZE,
                    size=retained_count,
                    replace=False,
                )
            )
        )

        head_mask = tuple(
            1 if index in retained_indices else 0
            for index in range(ATTENTION_HEAD_COUNT)
        )
        neuron_mask = tuple(
            1 if index + ATTENTION_HEAD_COUNT in retained_indices else 0
            for index in range(MLP_NEURON_COUNT)
        )
        retained_identifiers = tuple(identifiers[index] for index in retained_indices)

        sampled.append(
            SampledMask(
                mask_index=mask_index,
                retained_indices=retained_indices,
                retained_component_identifiers=retained_identifiers,
                head_mask=head_mask,
                neuron_mask=neuron_mask,
                mask_sha256=mask_sha256(head_mask, neuron_mask),
            )
        )

    return tuple(sampled)


def minimum_agreement_count(
    threshold: Fraction,
    *,
    example_count: int = FULL_DATASET_EXAMPLE_COUNT,
) -> int:
    """Return the minimum integer agreement count that passes a threshold."""

    if threshold not in CANDIDATE_THRESHOLDS:
        raise ValueError(f"unexpected candidate threshold: {threshold}")
    if example_count <= 0:
        raise ValueError("example_count must be positive")

    numerator = threshold.numerator * example_count
    denominator = threshold.denominator
    return (numerator + denominator - 1) // denominator


def agreement_passes_threshold(
    agreement_count: int,
    threshold: Fraction,
    *,
    example_count: int = FULL_DATASET_EXAMPLE_COUNT,
) -> bool:
    """Apply the exact rational Stage 11 pass comparison."""

    if not 0 <= agreement_count <= example_count:
        raise ValueError("agreement_count is outside the evaluated dataset")
    return (
        agreement_count * threshold.denominator
        >= example_count * threshold.numerator
    )


def validate_selection_record_fields(record: Mapping[str, object]) -> None:
    """Reject prohibited evidence before mechanical threshold selection."""

    prohibited = sorted(PROHIBITED_SELECTION_FIELDS.intersection(record))
    if prohibited:
        raise ValueError(
            "selection record contains prohibited evidence fields: "
            + ", ".join(prohibited)
        )


def qualify_candidate(candidate: CalibrationCandidate) -> QualificationResult:
    """Apply exactly the four frozen Stage 11 qualification criteria."""

    meaningfully_sparse = (
        candidate.retained_components <= MEANINGFULLY_SPARSE_MAX_COMPONENTS
    )
    random_mask_pass_count_at_most_5 = (
        candidate.random_mask_pass_count <= MAX_RANDOM_MASK_PASSES
    )
    within_budget = candidate.exact_evaluations <= MAX_EXACT_EVALUATIONS

    qualifies = (
        meaningfully_sparse
        and random_mask_pass_count_at_most_5
        and candidate.fourier_compatible_or_explained
        and within_budget
    )

    return QualificationResult(
        threshold=candidate.threshold,
        meaningfully_sparse=meaningfully_sparse,
        random_mask_pass_count_at_most_5=random_mask_pass_count_at_most_5,
        fourier_compatible_or_explained=(
            candidate.fourier_compatible_or_explained
        ),
        within_10000_evaluation_budget=within_budget,
        qualifies=qualifies,
    )


def select_primary_threshold(
    candidates: Iterable[CalibrationCandidate],
) -> tuple[Fraction | None, tuple[QualificationResult, ...]]:
    """Select the first qualifying threshold in frozen descending order."""

    candidate_by_threshold: dict[Fraction, CalibrationCandidate] = {}
    for candidate in candidates:
        if candidate.threshold in candidate_by_threshold:
            raise ValueError(f"duplicate threshold: {candidate.threshold}")
        candidate_by_threshold[candidate.threshold] = candidate

    missing = [
        threshold
        for threshold in CANDIDATE_THRESHOLDS
        if threshold not in candidate_by_threshold
    ]
    unexpected = [
        threshold
        for threshold in candidate_by_threshold
        if threshold not in CANDIDATE_THRESHOLDS
    ]
    if missing or unexpected or len(candidate_by_threshold) != len(
        CANDIDATE_THRESHOLDS
    ):
        raise ValueError(
            "candidate grid must contain exactly the six frozen thresholds"
        )

    results = tuple(
        qualify_candidate(candidate_by_threshold[threshold])
        for threshold in CANDIDATE_THRESHOLDS
    )
    selected = next(
        (result.threshold for result in results if result.qualifies),
        None,
    )
    return selected, results


def duplicate_mask_count(masks: Sequence[SampledMask]) -> int:
    """Count repeated masks without rejecting or redrawing them."""

    unique_count = len({mask.mask_sha256 for mask in masks})
    return len(masks) - unique_count



@dataclass(frozen=True)
class Stage9CircuitRecord:
    """Verified stable-post Stage 9 circuit eligible for calibration."""

    threshold: Fraction
    checkpoint_step: int
    checkpoint_sha256: str
    search_status: str
    retained_components: int
    retained_proportion: float
    exact_fidelity: float
    exact_agreement_count: int
    exact_evaluations: int
    final_mask_member: str
    final_mask_sha256: str
    mask: ComponentMask


@dataclass(frozen=True)
class Stage10CompatibilityRecord:
    """Committed Stage 10 diagnostic evidence for one Stage 9 circuit."""

    threshold: Fraction
    stage10_run_id: str
    classification: str
    addition_manifold_fraction: float
    correct_shift_rank: int
    mismatch_explanation_status: str | None

    @property
    def compatible_or_explained(self) -> bool:
        return (
            self.classification == "clear_match"
            or self.mismatch_explanation_status == "satisfactorily_explained"
        )


@dataclass(frozen=True)
class CalibrationSourceRecords:
    """Verified permitted Stage 11 calibration inputs."""

    stage9_run_id: str
    stage10_run_id: str
    circuits: tuple[Stage9CircuitRecord, ...]
    fourier_records: tuple[Stage10CompatibilityRecord, ...]


def file_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(file_path: Path, description: str) -> dict[str, Any]:
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {file_path}")
    value = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _fraction_from_candidate_decimal(value: object) -> Fraction:
    try:
        decimal_text = str(value)
        candidate = Fraction(decimal_text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid candidate threshold: {value!r}") from exc

    if candidate not in CANDIDATE_THRESHOLDS:
        raise ValueError(f"threshold is outside the frozen grid: {value!r}")
    return candidate


def _bool_from_csv(value: object, field_name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field_name} must be exactly 'True' or 'False'")


def _safe_archive_member_name(name: str) -> PurePosixPath:
    if (
        not name
        or name.startswith("./")
        or name.endswith("/")
        or "\\" in name
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")

    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or "." in member.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")

    return member


def _stage9_archive_member_for_record(
    stage9_run_id: str,
    recorded_path: str,
) -> str:
    recorded = PurePosixPath(recorded_path)
    expected_prefix = PurePosixPath("results/raw") / stage9_run_id
    try:
        relative = recorded.relative_to(expected_prefix)
    except ValueError as exc:
        raise ValueError(
            "Stage 9 final-mask path is outside the recorded raw run tree"
        ) from exc
    member = PurePosixPath(stage9_run_id) / relative
    return str(_safe_archive_member_name(str(member)))


def _read_verified_archive_member(
    archive: tarfile.TarFile,
    *,
    member_name: str,
    expected_sha256: str,
) -> bytes:
    safe_name = str(_safe_archive_member_name(member_name))
    try:
        member = archive.getmember(safe_name)
    except KeyError as exc:
        raise FileNotFoundError(
            f"archive member does not exist: {safe_name}"
        ) from exc

    if not member.isfile():
        raise ValueError(f"archive member is not a regular file: {safe_name}")

    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"could not read archive member: {safe_name}")

    payload = extracted.read()
    actual_sha256 = sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"archive member SHA-256 mismatch for {safe_name}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return payload


def parse_stable_post_stage9_row(
    row: Mapping[str, str],
) -> dict[str, object]:
    """Validate one permitted stable-post Stage 9 calibration row."""

    if row.get("phase") != "stable post-grokking":
        raise ValueError("Stage 11 calibration accepts only stable-post rows")
    if row.get("checkpoint_step") != str(STABLE_POST_CHECKPOINT_STEP):
        raise ValueError(
            "Stage 11 calibration accepts only checkpoint step 9050"
        )
    if row.get("source_training_run_id") != SOURCE_TRAINING_RUN_ID:
        raise ValueError("Stage 9 source training run mismatch")
    if row.get("search_status") != "valid_sparse_circuit":
        raise ValueError("Stage 9 row is not a valid sparse circuit")
    if (
        row.get("threshold_calibration_eligibility")
        != "eligible_for_later_stage11_primary_threshold_calibration"
    ):
        raise ValueError("Stage 9 row is not eligible for Stage 11 calibration")

    threshold = _fraction_from_candidate_decimal(row.get("fidelity_threshold"))
    retained_components = int(row["total_retained_components"])
    retained_proportion = float(row["retained_proportion"])
    exact_fidelity = float(row["final_exact_fidelity"])
    exact_agreement_count = round(
        exact_fidelity * FULL_DATASET_EXAMPLE_COUNT
    )

    if exact_agreement_count / FULL_DATASET_EXAMPLE_COUNT != exact_fidelity:
        raise ValueError(
            "Stage 9 fidelity does not reproduce from an integer agreement count"
        )

    meaningfully_sparse = _bool_from_csv(
        row["meaningfully_sparse"],
        "meaningfully_sparse",
    )
    if meaningfully_sparse != (
        retained_components <= MEANINGFULLY_SPARSE_MAX_COMPONENTS
    ):
        raise ValueError("Stage 9 sparsity flag is inconsistent")

    return {
        "threshold": threshold,
        "checkpoint_step": int(row["checkpoint_step"]),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "search_status": row["search_status"],
        "retained_components": retained_components,
        "retained_proportion": retained_proportion,
        "exact_fidelity": exact_fidelity,
        "exact_agreement_count": exact_agreement_count,
        "exact_evaluations": int(row["exact_evaluations_used"]),
        "final_mask_path": row["final_mask_path"],
        "final_mask_sha256": row["final_mask_sha256"],
    }


def _load_stage9_circuits(
    *,
    stage9_manifest: Mapping[str, Any],
    stage9_table_path: Path,
    stage9_archive_path: Path,
) -> tuple[Stage9CircuitRecord, ...]:
    stage9_run_id = str(stage9_manifest.get("stage9_run_id"))
    if stage9_run_id != "stage9-sparse-s1-27fffed087e6":
        raise ValueError("unexpected Stage 9 run ID")
    if stage9_manifest.get("source_training_run_id") != SOURCE_TRAINING_RUN_ID:
        raise ValueError("Stage 9 manifest source training run mismatch")

    outputs = stage9_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Stage 9 manifest outputs must be a mapping")

    table_record = outputs.get("deterministic_result_table")
    archive_record = outputs.get("raw_artifact_archive")
    if not isinstance(table_record, Mapping) or not isinstance(
        archive_record,
        Mapping,
    ):
        raise ValueError("Stage 9 manifest output records are incomplete")

    if file_sha256(stage9_table_path) != table_record.get("sha256"):
        raise ValueError("Stage 9 table SHA-256 mismatch")
    if file_sha256(stage9_archive_path) != archive_record.get("sha256"):
        raise ValueError("Stage 9 archive SHA-256 mismatch")

    with stage9_table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    candidate_rows = [
        row
        for row in rows
        if row.get("phase") == "stable post-grokking"
        and row.get("checkpoint_step") == str(STABLE_POST_CHECKPOINT_STEP)
    ]
    if len(candidate_rows) != len(CANDIDATE_THRESHOLDS):
        raise ValueError("expected exactly six stable-post Stage 9 rows")

    parsed_by_threshold: dict[Fraction, dict[str, object]] = {}
    for row in candidate_rows:
        parsed = parse_stable_post_stage9_row(row)
        threshold = parsed["threshold"]
        if not isinstance(threshold, Fraction):
            raise TypeError("parsed threshold must be a Fraction")
        if threshold in parsed_by_threshold:
            raise ValueError(f"duplicate Stage 9 threshold: {threshold}")
        parsed_by_threshold[threshold] = parsed

    if tuple(
        threshold
        for threshold in CANDIDATE_THRESHOLDS
        if threshold in parsed_by_threshold
    ) != CANDIDATE_THRESHOLDS:
        raise ValueError("Stage 9 candidate grid does not match the frozen grid")

    cell_records = outputs.get("cells")
    if not isinstance(cell_records, list):
        raise ValueError("Stage 9 manifest cells must be a list")

    manifest_cells: dict[Fraction, Mapping[str, Any]] = {}
    for cell in cell_records:
        if not isinstance(cell, Mapping):
            continue
        if (
            cell.get("phase") == "stable post-grokking"
            and cell.get("checkpoint_step") == STABLE_POST_CHECKPOINT_STEP
        ):
            threshold = _fraction_from_candidate_decimal(
                cell.get("fidelity_threshold")
            )
            manifest_cells[threshold] = cell

    if set(manifest_cells) != set(CANDIDATE_THRESHOLDS):
        raise ValueError("Stage 9 manifest stable-post cells are incomplete")

    circuits: list[Stage9CircuitRecord] = []
    with tarfile.open(stage9_archive_path, mode="r:gz") as archive:
        for threshold in CANDIDATE_THRESHOLDS:
            parsed = parsed_by_threshold[threshold]
            cell = manifest_cells[threshold]
            final_mask_record = cell.get("final_mask")
            if not isinstance(final_mask_record, Mapping):
                raise ValueError("Stage 9 cell final-mask record is invalid")

            if final_mask_record.get("path") != parsed["final_mask_path"]:
                raise ValueError("Stage 9 mask path mismatch")
            if final_mask_record.get("sha256") != parsed["final_mask_sha256"]:
                raise ValueError("Stage 9 mask hash mismatch")

            member_name = _stage9_archive_member_for_record(
                stage9_run_id,
                str(parsed["final_mask_path"]),
            )
            payload = _read_verified_archive_member(
                archive,
                member_name=member_name,
                expected_sha256=str(parsed["final_mask_sha256"]),
            )
            record = json.load(io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8"))
            mask = ComponentMask.from_record(record)

            retained_components = int(parsed["retained_components"])
            retained_proportion = float(parsed["retained_proportion"])
            if mask.retained_component_count != retained_components:
                raise ValueError("Stage 9 mask retained count mismatch")
            if mask.retained_component_proportion != retained_proportion:
                raise ValueError("Stage 9 mask retained proportion mismatch")

            circuits.append(
                Stage9CircuitRecord(
                    threshold=threshold,
                    checkpoint_step=int(parsed["checkpoint_step"]),
                    checkpoint_sha256=str(parsed["checkpoint_sha256"]),
                    search_status=str(parsed["search_status"]),
                    retained_components=retained_components,
                    retained_proportion=retained_proportion,
                    exact_fidelity=float(parsed["exact_fidelity"]),
                    exact_agreement_count=int(parsed["exact_agreement_count"]),
                    exact_evaluations=int(parsed["exact_evaluations"]),
                    final_mask_member=member_name,
                    final_mask_sha256=str(parsed["final_mask_sha256"]),
                    mask=mask,
                )
            )

    return tuple(circuits)


def _load_stage10_compatibility(
    *,
    stage10_manifest: Mapping[str, Any],
    stage10_table_path: Path,
    circuits: Sequence[Stage9CircuitRecord],
) -> tuple[Stage10CompatibilityRecord, ...]:
    stage10_run_id = str(stage10_manifest.get("stage10_run_id"))
    if stage10_run_id != "stage10-fourier-s1-a6f6a5773057":
        raise ValueError("unexpected Stage 10 run ID")
    if stage10_manifest.get("source_training_run_id") != SOURCE_TRAINING_RUN_ID:
        raise ValueError("Stage 10 source training run mismatch")

    outputs = stage10_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Stage 10 outputs must be a mapping")
    table_record = outputs.get("circuit_fourier_table")
    if not isinstance(table_record, Mapping):
        raise ValueError("Stage 10 circuit table record is missing")
    if file_sha256(stage10_table_path) != table_record.get("sha256"):
        raise ValueError("Stage 10 circuit table SHA-256 mismatch")

    with stage10_table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(CANDIDATE_THRESHOLDS):
        raise ValueError("expected exactly six Stage 10 circuit rows")

    manifest_classification = stage10_manifest.get("classification")
    if not isinstance(manifest_classification, Mapping):
        raise ValueError("Stage 10 classification record is missing")
    circuit_classifications = manifest_classification.get("circuits")
    if not isinstance(circuit_classifications, Mapping):
        raise ValueError("Stage 10 circuit classifications are missing")

    circuit_by_threshold = {circuit.threshold: circuit for circuit in circuits}
    records: dict[Fraction, Stage10CompatibilityRecord] = {}

    for row in rows:
        threshold = _fraction_from_candidate_decimal(row["fidelity_threshold"])
        circuit = circuit_by_threshold.get(threshold)
        if circuit is None:
            raise ValueError("Stage 10 row has no matching Stage 9 circuit")

        classification = row["diagnostic_classification"]
        manifest_value = circuit_classifications.get(str(float(threshold)))
        if classification != manifest_value:
            raise ValueError("Stage 10 classification mismatch")
        if row["mask_id"] != circuit.mask.mask_id:
            raise ValueError("Stage 10 mask ID mismatch")
        if int(row["retained_components"]) != circuit.retained_components:
            raise ValueError("Stage 10 retained count mismatch")
        if float(row["exact_fidelity"]) != circuit.exact_fidelity:
            raise ValueError("Stage 10 exact fidelity mismatch")

        records[threshold] = Stage10CompatibilityRecord(
            threshold=threshold,
            stage10_run_id=stage10_run_id,
            classification=classification,
            addition_manifold_fraction=float(
                row["addition_manifold_fraction"]
            ),
            correct_shift_rank=int(row["correct_shift_rank"]),
            mismatch_explanation_status=None,
        )

    if set(records) != set(CANDIDATE_THRESHOLDS):
        raise ValueError("Stage 10 candidate grid is incomplete")

    return tuple(records[threshold] for threshold in CANDIDATE_THRESHOLDS)


def load_calibration_source_records(
    *,
    stage9_manifest_path: str | Path,
    stage9_table_path: str | Path,
    stage9_archive_path: str | Path,
    stage10_manifest_path: str | Path,
) -> CalibrationSourceRecords:
    """Load and verify only the permitted Stage 11 source records."""

    stage9_manifest_file = Path(stage9_manifest_path)
    stage9_table_file = Path(stage9_table_path)
    stage9_archive_file = Path(stage9_archive_path)
    stage10_manifest_file = Path(stage10_manifest_path)

    stage9_manifest = _load_json_object(
        stage9_manifest_file,
        "Stage 9 manifest",
    )
    stage10_manifest = _load_json_object(
        stage10_manifest_file,
        "Stage 10 manifest",
    )

    circuits = _load_stage9_circuits(
        stage9_manifest=stage9_manifest,
        stage9_table_path=stage9_table_file,
        stage9_archive_path=stage9_archive_file,
    )

    stage10_outputs = stage10_manifest.get("outputs")
    if not isinstance(stage10_outputs, Mapping):
        raise ValueError("Stage 10 manifest outputs must be a mapping")
    circuit_table_record = stage10_outputs.get("circuit_fourier_table")
    if not isinstance(circuit_table_record, Mapping):
        raise ValueError("Stage 10 circuit table record is missing")

    stage10_table_file = Path(str(circuit_table_record["path"]))
    fourier_records = _load_stage10_compatibility(
        stage10_manifest=stage10_manifest,
        stage10_table_path=stage10_table_file,
        circuits=circuits,
    )

    return CalibrationSourceRecords(
        stage9_run_id=str(stage9_manifest["stage9_run_id"]),
        stage10_run_id=str(stage10_manifest["stage10_run_id"]),
        circuits=circuits,
        fourier_records=fourier_records,
    )


RANDOM_MASK_EVALUATION_COLUMNS: Final[tuple[str, ...]] = (
    "stage11_run_id",
    "source_training_run_id",
    "checkpoint_step",
    "checkpoint_sha256",
    "fidelity_threshold",
    "mask_index",
    "seed_material",
    "seed_digest",
    "seed_uint64",
    "bit_generator",
    "numpy_version",
    "sampling_definition",
    "target_retained_components",
    "retained_heads",
    "retained_neurons",
    "retained_components",
    "mask_id",
    "mask_sha256",
    "prediction_agreement_count",
    "evaluated_example_count",
    "primary_fidelity",
    "passes_candidate_threshold",
    "full_accuracy",
    "masked_accuracy",
    "accuracy_change",
    "full_cross_entropy",
    "masked_cross_entropy",
    "cross_entropy_change",
    "mean_kl_divergence",
    "mean_jensen_shannon_divergence",
    "maximum_absolute_logit_difference",
)

THRESHOLD_CALIBRATION_COLUMNS: Final[tuple[str, ...]] = (
    "stage11_run_id",
    "source_training_run_id",
    "checkpoint_step",
    "descending_evaluation_rank",
    "fidelity_threshold",
    "threshold_numerator",
    "threshold_denominator",
    "stage9_retained_components",
    "stage9_retained_proportion",
    "stage9_exact_fidelity",
    "stage9_exact_agreement_count",
    "stage9_exact_evaluations",
    "stage9_within_10000_evaluation_budget",
    "stage10_classification",
    "stage10_compatible_or_explained",
    "random_seed_material",
    "random_seed_digest",
    "random_seed_uint64",
    "random_bit_generator",
    "random_numpy_version",
    "random_mask_count",
    "random_mask_duplicate_count",
    "random_mask_pass_count",
    "random_mask_pass_fraction",
    "random_mask_fidelity_min",
    "random_mask_fidelity_p05",
    "random_mask_fidelity_median",
    "random_mask_fidelity_p95",
    "random_mask_fidelity_max",
    "meaningfully_sparse",
    "random_mask_pass_count_at_most_5",
    "within_10000_evaluation_budget",
    "qualifies",
    "selected_primary_threshold",
)

RUNTIME_COLUMNS: Final[tuple[str, ...]] = (
    "stage11_run_id",
    "fidelity_threshold",
    "mask_count",
    "elapsed_seconds",
    "seconds_per_mask",
    "included_in_deterministic_scientific_hashes",
)


def sampled_mask_to_component_mask(sampled: SampledMask) -> ComponentMask:
    """Convert one sampled Stage 11 mask to the canonical mask type."""

    mask = ComponentMask(
        attention_head_mask=sampled.head_mask,
        mlp_neuron_mask=sampled.neuron_mask,
    )
    if mask.retained_component_ids != sampled.retained_component_identifiers:
        raise ValueError("sampled-mask identifier ordering is inconsistent")
    return mask


def random_mask_evaluation_record(
    *,
    stage11_run_id: str,
    circuit: Stage9CircuitRecord,
    sampled: SampledMask,
    metrics: Any,
) -> dict[str, object]:
    """Build one deterministic scientific evaluation record."""

    seed = derive_random_seed(circuit.threshold)
    canonical_mask = sampled_mask_to_component_mask(sampled)

    if metrics.evaluated_example_count != FULL_DATASET_EXAMPLE_COUNT:
        raise ValueError("random-mask evaluation did not cover all examples")
    if metrics.retained_component_count != circuit.retained_components:
        raise ValueError("random-mask evaluation retained-count mismatch")
    if metrics.retained_attention_head_count != sampled.retained_head_count:
        raise ValueError("random-mask head-count mismatch")
    if metrics.retained_mlp_neuron_count != sampled.retained_neuron_count:
        raise ValueError("random-mask neuron-count mismatch")
    if (
        metrics.prediction_agreement_count / FULL_DATASET_EXAMPLE_COUNT
        != metrics.primary_fidelity
    ):
        raise ValueError("random-mask fidelity is not count-exact")

    passes = agreement_passes_threshold(
        metrics.prediction_agreement_count,
        circuit.threshold,
    )

    return {
        "stage11_run_id": stage11_run_id,
        "source_training_run_id": SOURCE_TRAINING_RUN_ID,
        "checkpoint_step": STABLE_POST_CHECKPOINT_STEP,
        "checkpoint_sha256": circuit.checkpoint_sha256,
        "fidelity_threshold": float(circuit.threshold),
        "mask_index": sampled.mask_index,
        "seed_material": seed.seed_material,
        "seed_digest": seed.seed_digest,
        "seed_uint64": seed.seed_uint64,
        "bit_generator": seed.bit_generator,
        "numpy_version": seed.numpy_version,
        "sampling_definition": (
            "uniform subset without replacement over all 516 searchable "
            "components; no head/neuron stratification"
        ),
        "target_retained_components": circuit.retained_components,
        "retained_heads": metrics.retained_attention_head_count,
        "retained_neurons": metrics.retained_mlp_neuron_count,
        "retained_components": metrics.retained_component_count,
        "mask_id": canonical_mask.mask_id,
        "mask_sha256": sampled.mask_sha256,
        "prediction_agreement_count": metrics.prediction_agreement_count,
        "evaluated_example_count": metrics.evaluated_example_count,
        "primary_fidelity": metrics.primary_fidelity,
        "passes_candidate_threshold": passes,
        "full_accuracy": metrics.full_accuracy,
        "masked_accuracy": metrics.masked_accuracy,
        "accuracy_change": metrics.accuracy_change,
        "full_cross_entropy": metrics.full_cross_entropy,
        "masked_cross_entropy": metrics.masked_cross_entropy,
        "cross_entropy_change": metrics.cross_entropy_change,
        "mean_kl_divergence": metrics.mean_kl_divergence,
        "mean_jensen_shannon_divergence": (
            metrics.mean_jensen_shannon_divergence
        ),
        "maximum_absolute_logit_difference": (
            metrics.maximum_absolute_logit_difference
        ),
    }


def summarise_threshold_evaluations(
    *,
    stage11_run_id: str,
    descending_evaluation_rank: int,
    circuit: Stage9CircuitRecord,
    fourier_record: Stage10CompatibilityRecord,
    evaluation_rows: Sequence[Mapping[str, object]],
    masks: Sequence[SampledMask],
) -> tuple[CalibrationCandidate, dict[str, object]]:
    """Summarise one threshold and apply the frozen qualification rule."""

    if len(evaluation_rows) != RANDOM_MASKS_PER_THRESHOLD:
        raise ValueError("each threshold requires exactly 100 evaluations")
    if len(masks) != RANDOM_MASKS_PER_THRESHOLD:
        raise ValueError("each threshold requires exactly 100 sampled masks")

    fidelities = np.asarray(
        [float(row["primary_fidelity"]) for row in evaluation_rows],
        dtype=np.float64,
    )
    pass_count = sum(
        bool(row["passes_candidate_threshold"]) for row in evaluation_rows
    )

    if not 1 <= descending_evaluation_rank <= len(CANDIDATE_THRESHOLDS):
        raise ValueError("descending_evaluation_rank must be between 1 and 6")

    seed = derive_random_seed(circuit.threshold)
    candidate = CalibrationCandidate(
        threshold=circuit.threshold,
        retained_components=circuit.retained_components,
        random_mask_pass_count=pass_count,
        fourier_compatible_or_explained=(
            fourier_record.compatible_or_explained
        ),
        exact_evaluations=circuit.exact_evaluations,
    )
    qualification = qualify_candidate(candidate)

    percentiles = np.percentile(
        fidelities,
        [5, 50, 95],
        method=PERCENTILE_METHOD,
    )

    row: dict[str, object] = {
        "stage11_run_id": stage11_run_id,
        "source_training_run_id": SOURCE_TRAINING_RUN_ID,
        "checkpoint_step": STABLE_POST_CHECKPOINT_STEP,
        "descending_evaluation_rank": descending_evaluation_rank,
        "fidelity_threshold": float(circuit.threshold),
        "threshold_numerator": circuit.threshold.numerator,
        "threshold_denominator": circuit.threshold.denominator,
        "stage9_retained_components": circuit.retained_components,
        "stage9_retained_proportion": circuit.retained_proportion,
        "stage9_exact_fidelity": circuit.exact_fidelity,
        "stage9_exact_agreement_count": circuit.exact_agreement_count,
        "stage9_exact_evaluations": circuit.exact_evaluations,
        "stage9_within_10000_evaluation_budget": (
            circuit.exact_evaluations <= MAX_EXACT_EVALUATIONS
        ),
        "stage10_classification": fourier_record.classification,
        "stage10_compatible_or_explained": (
            fourier_record.compatible_or_explained
        ),
        "random_seed_material": seed.seed_material,
        "random_seed_digest": seed.seed_digest,
        "random_seed_uint64": seed.seed_uint64,
        "random_bit_generator": seed.bit_generator,
        "random_numpy_version": seed.numpy_version,
        "random_mask_count": len(evaluation_rows),
        "random_mask_duplicate_count": duplicate_mask_count(masks),
        "random_mask_pass_count": pass_count,
        "random_mask_pass_fraction": pass_count / len(evaluation_rows),
        "random_mask_fidelity_min": float(fidelities.min()),
        "random_mask_fidelity_p05": float(percentiles[0]),
        "random_mask_fidelity_median": float(percentiles[1]),
        "random_mask_fidelity_p95": float(percentiles[2]),
        "random_mask_fidelity_max": float(fidelities.max()),
        "meaningfully_sparse": qualification.meaningfully_sparse,
        "random_mask_pass_count_at_most_5": (
            qualification.random_mask_pass_count_at_most_5
        ),
        "within_10000_evaluation_budget": (
            qualification.within_10000_evaluation_budget
        ),
        "qualifies": qualification.qualifies,
        "selected_primary_threshold": False,
    }
    validate_selection_record_fields(row)
    return candidate, row


def write_csv_records(
    file_path: str | Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    """Write deterministic CSV records with a frozen column order."""

    output = Path(file_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output


def write_deterministic_tar_gz(
    *,
    source_directory: str | Path,
    archive_path: str | Path,
) -> Path:
    """Write a byte-deterministic gzip-compressed PAX tar archive."""

    import gzip

    source = Path(source_directory)
    output = Path(archive_path)
    if not source.is_dir():
        raise FileNotFoundError(f"archive source does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        item for item in source.rglob("*") if item.is_file()
    )
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            compresslevel=9,
            mtime=0,
        ) as gzip_handle:
            with tarfile.open(
                fileobj=gzip_handle,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for item in files:
                    relative = item.relative_to(source.parent)
                    member = archive.gettarinfo(
                        str(item),
                        arcname=relative.as_posix(),
                    )
                    member.uid = 0
                    member.gid = 0
                    member.uname = ""
                    member.gname = ""
                    member.mtime = 0
                    member.mode = 0o644
                    with item.open("rb") as handle:
                        archive.addfile(member, handle)

    return output
