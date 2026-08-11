"""Shared primitives for the frozen Stage 14 random-label analysis."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import yaml

from circuit_families.data.input_subsets import (
    SUBSET_NAMES,
    generate_input_subsets,
)
from circuit_families.interpretability.fidelity import (
    MaskEvaluationMetrics,
)
from circuit_families.interpretability.masks import ComponentMask

ANALYSIS_CONFIGURATION_PATH = Path(
    "configs/stage14_random_label_analysis.yaml"
)
ANALYSIS_CONFIGURATION_SHA256 = (
    "9cf400e4ebdcf1cfb93ab654c6253133549475ca3b88487d936718cf594bc899"
)
ANALYSIS_RUN_ID = "stage14-random-label-analysis-s0-7b472aa5163a"
ANALYSIS_IDENTITY_SHA256 = (
    "7b472aa5163a6b771ea15cb4bfb0d5ea11871ec3cbc1591865cb22bc2ecae270"
)
CONFIGURATION_FREEZE_COMMIT = (
    "1800a059e58a720e29e2eff3ac4c339138afee72"
)

MATCHED_CHECKPOINT_STEPS = (
    200,
    3_400,
    7_450,
    8_150,
    8_500,
    8_650,
    9_050,
)
FIDELITY_SENSITIVITY_GRID = (
    Fraction(4, 5),
    Fraction(17, 20),
    Fraction(9, 10),
    Fraction(19, 20),
    Fraction(39, 40),
    Fraction(99, 100),
)
DISTINCTNESS_SENSITIVITY_GRID = (
    Fraction(1, 4),
    Fraction(1, 2),
    Fraction(3, 4),
)
TRANSFER_GROUPING_SENSITIVITY_GRID = (
    Fraction(1, 40),
    Fraction(1, 20),
    Fraction(1, 10),
)

PRIMARY_FIDELITY_THRESHOLD = Fraction(99, 100)
PRIMARY_DISTINCTNESS_CUTOFF = Fraction(1, 2)
PRIMARY_TRANSFER_GROUPING_TOLERANCE = Fraction(1, 20)

EXPECTED_SUBSET_COUNTS = {
    "Q1": 3_249,
    "Q2": 3_192,
    "Q3": 3_192,
    "Q4": 3_136,
}


@dataclass(frozen=True)
class FrozenStage14AnalysisConfiguration:
    """Validated view of the committed Stage 14 analysis matrix."""

    path: Path
    sha256: str
    payload: dict[str, Any]
    analysis_run_id: str
    analysis_identity_sha256: str
    checkpoint_steps: tuple[int, ...]
    fidelity_grid: tuple[Fraction, ...]
    distinctness_grid: tuple[Fraction, ...]
    transfer_grouping_grid: tuple[Fraction, ...]


def file_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 digest of one physical file."""

    digest = hashlib.sha256()

    with Path(file_path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash one value using the repository's canonical JSON convention."""

    serialised = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def fraction_from_record(record: Mapping[str, Any]) -> Fraction:
    """Read an exact rational value from a frozen configuration record."""

    if not isinstance(record, Mapping):
        raise TypeError("Rational record must be a mapping.")

    numerator = record.get("numerator")
    denominator = record.get("denominator")

    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise TypeError("Rational numerator must be an integer.")

    if isinstance(denominator, bool) or not isinstance(denominator, int):
        raise TypeError("Rational denominator must be an integer.")

    if denominator <= 0:
        raise ValueError("Rational denominator must be positive.")

    return Fraction(numerator, denominator)


def agreement_passes_threshold(
    *,
    agreement_count: int,
    evaluated_example_count: int,
    threshold: Fraction | Mapping[str, Any],
) -> bool:
    """Apply an exact rational fidelity threshold without float rounding."""

    if isinstance(agreement_count, bool) or not isinstance(
        agreement_count,
        int,
    ):
        raise TypeError("agreement_count must be an integer.")

    if isinstance(evaluated_example_count, bool) or not isinstance(
        evaluated_example_count,
        int,
    ):
        raise TypeError("evaluated_example_count must be an integer.")

    if evaluated_example_count <= 0:
        raise ValueError("evaluated_example_count must be positive.")

    if not 0 <= agreement_count <= evaluated_example_count:
        raise ValueError(
            "agreement_count must be between zero and the example count."
        )

    exact_threshold = (
        fraction_from_record(threshold)
        if isinstance(threshold, Mapping)
        else threshold
    )

    if not isinstance(exact_threshold, Fraction):
        raise TypeError("threshold must be a Fraction or rational record.")

    if not 0 <= exact_threshold <= 1:
        raise ValueError("threshold must be between zero and one.")

    return (
        agreement_count * exact_threshold.denominator
        >= exact_threshold.numerator * evaluated_example_count
    )


def _require_equal(
    observed: object,
    expected: object,
    description: str,
) -> None:
    if observed != expected:
        raise ValueError(
            f"{description}: expected {expected!r}, found {observed!r}."
        )


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path.resolve()

    return (root / path).resolve()


def load_frozen_analysis_configuration(
    *,
    repository_root: str | Path,
    configuration_path: str | Path = ANALYSIS_CONFIGURATION_PATH,
) -> FrozenStage14AnalysisConfiguration:
    """Load and validate the committed pre-results analysis matrix."""

    repository = Path(repository_root).resolve()
    configuration_file = _resolve_path(
        repository,
        configuration_path,
    )

    if not configuration_file.is_file():
        raise FileNotFoundError(configuration_file)

    observed_sha256 = file_sha256(configuration_file)
    _require_equal(
        observed_sha256,
        ANALYSIS_CONFIGURATION_SHA256,
        "Analysis-configuration SHA-256 mismatch",
    )

    payload = yaml.safe_load(
        configuration_file.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise TypeError("Analysis configuration must contain a mapping.")

    _require_equal(
        payload.get("schema_version"),
        1,
        "Analysis schema version mismatch",
    )
    _require_equal(
        payload.get("experiment_type"),
        "stage14_random_label_circuit_analysis",
        "Analysis experiment type mismatch",
    )
    _require_equal(
        payload.get("freeze_status"),
        "frozen_before_random_label_circuit_results",
        "Analysis freeze status mismatch",
    )
    _require_equal(
        payload.get("analysis_run_id"),
        ANALYSIS_RUN_ID,
        "Analysis run ID mismatch",
    )
    _require_equal(
        payload.get("analysis_identity_sha256"),
        ANALYSIS_IDENTITY_SHA256,
        "Analysis identity mismatch",
    )

    source = payload.get("source")

    if not isinstance(source, dict):
        raise TypeError("Analysis source record is missing.")

    _require_equal(
        source.get("model_seed"),
        0,
        "Model seed mismatch",
    )
    _require_equal(
        source.get("random_label_seed"),
        1,
        "Random-label seed mismatch",
    )
    _require_equal(
        source.get("control_classification"),
        "memorisation_control",
        "Control classification mismatch",
    )

    execution_matrix = payload.get("execution_matrix")

    if not isinstance(execution_matrix, dict):
        raise TypeError("Execution matrix is missing.")

    primary = execution_matrix.get("primary_dynamics")

    if not isinstance(primary, dict):
        raise TypeError("Primary dynamics matrix is missing.")

    primary_cells = primary.get("cells")

    if not isinstance(primary_cells, list):
        raise TypeError("Primary dynamics cells are missing.")

    checkpoint_steps = tuple(
        int(cell["checkpoint_step"])
        for cell in primary_cells
    )
    _require_equal(
        checkpoint_steps,
        MATCHED_CHECKPOINT_STEPS,
        "Matched checkpoint grid mismatch",
    )

    mismatches = tuple(
        int(cell["absolute_step_mismatch"])
        for cell in primary_cells
    )
    _require_equal(
        mismatches,
        (0,) * len(MATCHED_CHECKPOINT_STEPS),
        "Checkpoint mismatch record differs",
    )

    fidelity = payload.get("fidelity")

    if not isinstance(fidelity, dict):
        raise TypeError("Fidelity configuration is missing.")

    fidelity_grid = tuple(
        fraction_from_record(record)
        for record in fidelity["sensitivity_grid"]
    )
    _require_equal(
        fidelity_grid,
        FIDELITY_SENSITIVITY_GRID,
        "Fidelity sensitivity grid mismatch",
    )
    _require_equal(
        fraction_from_record(fidelity["primary_threshold"]),
        PRIMARY_FIDELITY_THRESHOLD,
        "Primary fidelity threshold mismatch",
    )

    distinctness = payload.get("distinctness")

    if not isinstance(distinctness, dict):
        raise TypeError("Distinctness configuration is missing.")

    distinctness_grid = tuple(
        fraction_from_record(record)
        for record in distinctness["sensitivity_grid"]
    )
    _require_equal(
        distinctness_grid,
        DISTINCTNESS_SENSITIVITY_GRID,
        "Distinctness sensitivity grid mismatch",
    )
    _require_equal(
        fraction_from_record(distinctness["primary_cutoff"]),
        PRIMARY_DISTINCTNESS_CUTOFF,
        "Primary distinctness cutoff mismatch",
    )

    transfer = payload.get("transfer")

    if not isinstance(transfer, dict):
        raise TypeError("Transfer configuration is missing.")

    transfer_grid = tuple(
        fraction_from_record(record)
        for record in transfer["grouping_sensitivity_grid"]
    )
    _require_equal(
        transfer_grid,
        TRANSFER_GROUPING_SENSITIVITY_GRID,
        "Transfer-grouping sensitivity grid mismatch",
    )
    _require_equal(
        fraction_from_record(
            transfer["primary_grouping_tolerance"]
        ),
        PRIMARY_TRANSFER_GROUPING_TOLERANCE,
        "Primary transfer-grouping tolerance mismatch",
    )

    subset_records = transfer.get("subsets")

    if not isinstance(subset_records, list):
        raise TypeError("Transfer subset records are missing.")

    subset_counts = {
        str(record["subset_name"]): int(record["example_count"])
        for record in subset_records
    }
    _require_equal(
        subset_counts,
        EXPECTED_SUBSET_COUNTS,
        "Transfer subset counts mismatch",
    )

    integrity = payload.get("integrity")

    if not isinstance(integrity, dict):
        raise TypeError("Integrity configuration is missing.")

    _require_equal(
        integrity.get("stage15_started"),
        False,
        "Stage 15 status mismatch",
    )

    return FrozenStage14AnalysisConfiguration(
        path=configuration_file,
        sha256=observed_sha256,
        payload=payload,
        analysis_run_id=ANALYSIS_RUN_ID,
        analysis_identity_sha256=ANALYSIS_IDENTITY_SHA256,
        checkpoint_steps=checkpoint_steps,
        fidelity_grid=fidelity_grid,
        distinctness_grid=distinctness_grid,
        transfer_grouping_grid=transfer_grid,
    )


def recorded_source_path(
    *,
    repository_root: str | Path,
    configuration: FrozenStage14AnalysisConfiguration,
    source_name: str,
) -> Path:
    """Resolve and hash-check one source recorded in the frozen matrix."""

    source = configuration.payload["source"]
    record = source.get(source_name)

    if not isinstance(record, Mapping):
        raise KeyError(f"Unknown or invalid source record: {source_name}")

    path_value = record.get("path")
    expected_sha256 = record.get("sha256")

    if not isinstance(path_value, str):
        raise TypeError(f"Source path is invalid: {source_name}")

    if not isinstance(expected_sha256, str):
        raise TypeError(f"Source hash is invalid: {source_name}")

    file_path = _resolve_path(
        Path(repository_root).resolve(),
        path_value,
    )

    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    _require_equal(
        file_sha256(file_path),
        expected_sha256,
        f"Source hash mismatch for {source_name}",
    )

    return file_path


def _load_json_mapping(file_path: Path) -> dict[str, Any]:
    value = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {file_path}")

    return value


def load_stage14_masking_module(
    repository_root: str | Path,
) -> ModuleType:
    """Load the already validated Stage 14 random-label context adapter."""

    repository = Path(repository_root).resolve()
    script_path = (
        repository / "scripts/validate_stage14_masking.py"
    )

    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    module_name = "_stage14_masking_context_adapter"
    specification = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            f"Unable to load Stage 14 masking adapter: {script_path}"
        )

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def load_random_label_checkpoint_context(
    *,
    repository_root: str | Path,
    configuration: FrozenStage14AnalysisConfiguration,
    checkpoint_step: int,
    device: str | torch.device = "cpu",
    output_root: str | Path | None = None,
) -> Any:
    """Load one exact Stage 14 checkpoint through the masking adapter."""

    if isinstance(checkpoint_step, bool) or not isinstance(
        checkpoint_step,
        int,
    ):
        raise TypeError("checkpoint_step must be an integer.")

    if checkpoint_step not in configuration.checkpoint_steps:
        raise ValueError(
            f"Checkpoint step is outside the frozen grid: {checkpoint_step}"
        )

    repository = Path(repository_root).resolve()
    selected_output_root = (
        repository
        if output_root is None
        else Path(output_root).resolve()
    )

    checkpoint_manifest_path = recorded_source_path(
        repository_root=repository,
        configuration=configuration,
        source_name="checkpoint_selection_manifest",
    )
    training_manifest_path = recorded_source_path(
        repository_root=repository,
        configuration=configuration,
        source_name="training_manifest",
    )

    checkpoint_manifest = _load_json_mapping(
        checkpoint_manifest_path
    )
    training_manifest = _load_json_mapping(
        training_manifest_path
    )

    matched_records = checkpoint_manifest.get(
        "matched_checkpoints"
    )

    if not isinstance(matched_records, list):
        raise TypeError("Matched checkpoint records are missing.")

    matching_records = [
        record
        for record in matched_records
        if int(record["requested_step"]) == checkpoint_step
    ]

    if len(matching_records) != 1:
        raise ValueError(
            "Expected exactly one matched checkpoint record at "
            f"step {checkpoint_step}; found {len(matching_records)}."
        )

    module = load_stage14_masking_module(repository)

    context = module.load_context(
        repository=repository,
        output_root=selected_output_root,
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_manifest=checkpoint_manifest,
        training_manifest_path=training_manifest_path,
        training_manifest=training_manifest,
        matched_record=matching_records[0],
        device=torch.device(device),
    )

    _require_equal(
        int(context.checkpoint_step),
        checkpoint_step,
        "Loaded context step mismatch",
    )

    return context


def subset_indices_from_inputs(
    inputs: torch.Tensor,
    subset_name: str,
) -> torch.Tensor:
    """Return frozen Q1-Q4 row indices for encoded model inputs."""

    if subset_name not in SUBSET_NAMES:
        raise ValueError(
            f"Unknown transfer subset {subset_name!r}; "
            f"expected one of {SUBSET_NAMES!r}."
        )

    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor.")

    if inputs.ndim != 2 or inputs.shape[1] < 2:
        raise ValueError(
            "inputs must have shape [examples, sequence] with "
            "the two operands in the first two positions."
        )

    operand_pairs = (
        inputs[:, :2]
        .detach()
        .to(device="cpu", dtype=torch.long)
        .numpy()
        .astype(np.int64, copy=False)
    )
    subset_mapping = generate_input_subsets(operand_pairs)
    indices = subset_mapping[subset_name]

    expected_count = EXPECTED_SUBSET_COUNTS[subset_name]

    if int(indices.shape[0]) != expected_count:
        raise ValueError(
            f"Subset {subset_name} count mismatch: "
            f"expected {expected_count}, found {indices.shape[0]}."
        )

    return torch.as_tensor(
        indices,
        dtype=torch.long,
        device=inputs.device,
    )


def subset_context(
    context: Any,
    subset_name: str,
) -> Any:
    """Return a dataclass copy restricted to one frozen transfer subset."""

    if not dataclasses.is_dataclass(context):
        raise TypeError("context must be a dataclass instance.")

    indices = subset_indices_from_inputs(
        context.inputs,
        subset_name,
    )

    subset_inputs = context.inputs.index_select(0, indices)
    subset_targets = context.targets.index_select(0, indices)

    replacements: dict[str, Any] = {
        "inputs": subset_inputs,
        "targets": subset_targets,
    }

    field_names = {
        field.name
        for field in dataclasses.fields(context)
    }

    if "checkpoint_phase" in field_names:
        replacements["checkpoint_phase"] = (
            f"{context.checkpoint_phase}|subset={subset_name}"
        )

    return dataclasses.replace(
        context,
        **replacements,
    )


def subset_contexts(context: Any) -> dict[str, Any]:
    """Return the four frozen transfer-subset contexts."""

    return {
        subset_name: subset_context(context, subset_name)
        for subset_name in SUBSET_NAMES
    }


def component_mask_record(mask: ComponentMask) -> dict[str, Any]:
    """Return a deterministic, human-readable mask identity record."""

    if not isinstance(mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask.")

    identity = {
        "attention_head_mask": list(mask.attention_head_mask),
        "mlp_neuron_mask": list(mask.mlp_neuron_mask),
    }

    return {
        "mask_identity_sha256": canonical_json_sha256(identity),
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
        "retained_component_ids": list(
            mask.retained_component_ids
        ),
        "ablated_component_ids": list(
            mask.ablated_component_ids
        ),
        **identity,
    }


def metrics_record(
    metrics: MaskEvaluationMetrics,
) -> dict[str, Any]:
    """Serialise one immutable fidelity-metric record."""

    if not isinstance(metrics, MaskEvaluationMetrics):
        raise TypeError("metrics must be MaskEvaluationMetrics.")

    return dataclasses.asdict(metrics)
