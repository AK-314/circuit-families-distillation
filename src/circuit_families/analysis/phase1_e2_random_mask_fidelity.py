"""Phase I E2 matched random-mask behavioural-fidelity nulls."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from circuit_families.analysis.phase1_e1_jaccard_null import (
    CircuitRecord,
    ValidatedInputs,
    canonical_json_bytes,
    file_sha256,
)
from circuit_families.analysis.phase1_e1_jaccard_null import (
    validate_inputs as validate_e1_inputs,
)
from circuit_families.interpretability.fidelity import MaskEvaluationMetrics
from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_COUNT,
    ATTENTION_HEAD_IDS,
    MLP_NEURON_COUNT,
    MLP_NEURON_IDS,
    SEARCHABLE_COMPONENT_COUNT,
    ComponentMask,
)

NULL_MODELS = ("size_matched", "basis_stratified")


class E2ValidationError(ValueError):
    """Raised when an E2 input or frozen contract is inconsistent."""


@dataclass(frozen=True)
class ObservedCircuit:
    """One observed qualifying circuit and its compact provenance."""

    circuit: CircuitRecord
    primary_fidelity: float
    prediction_agreement_count: int


@dataclass(frozen=True)
class CheckpointSource:
    """One authenticated model checkpoint used by E2."""

    model_seed: int
    run_id: str
    checkpoint_manifest: Path
    checkpoint_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class NullProfile:
    """One unique model-seed and observed head/MLP composition profile."""

    model_seed: int
    checkpoint_step: int
    retained_heads: int
    retained_neurons: int
    retained_components: int
    observed_circuit_ids: tuple[str, ...]

    @property
    def profile_id(self) -> str:
        return (
            f"s{self.model_seed}-step{self.checkpoint_step}-"
            f"h{self.retained_heads}-n{self.retained_neurons}-"
            f"k{self.retained_components}"
        )


@dataclass(frozen=True)
class SeedEvidence:
    """Stage-11-style SHA-256 to uint64 PCG64 seed evidence."""

    canonical_material: str
    sha256_digest: str
    integer_seed: int


@dataclass(frozen=True)
class SampledMask:
    """One deterministic random-mask draw."""

    mask_index: int
    mask: ComponentMask
    mask_sha256: str


@dataclass(frozen=True)
class MaskEvaluation:
    """One evaluated random mask with profile and seed provenance."""

    profile: NullProfile
    null_model: str
    sampled: SampledMask
    metrics: MaskEvaluationMetrics
    passes_threshold: bool
    seed: SeedEvidence


@dataclass(frozen=True)
class ValidatedE2Inputs:
    """Fully validated E2 configuration and comparison records."""

    configuration: Mapping[str, Any]
    configuration_sha256: str
    e1_inputs: ValidatedInputs
    observed_circuits: tuple[ObservedCircuit, ...]
    profiles: tuple[NullProfile, ...]
    checkpoints: Mapping[int, CheckpointSource]
    source_hashes: Mapping[str, str]


def load_configuration(path: Path) -> tuple[dict[str, Any], str]:
    """Load and canonically hash an E2 configuration."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E2ValidationError("E2 configuration must be a JSON object.")
    return value, hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise E2ValidationError(f"{label} must be an integer >= {minimum}.")
    return value


def _verify_file(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required E2 {label} is absent: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise E2ValidationError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual}."
        )
    return actual


def minimum_agreement_count(
    threshold: Fraction,
    *,
    example_count: int,
) -> int:
    """Return the smallest integer agreement count passing a threshold."""
    if not isinstance(threshold, Fraction):
        raise TypeError("threshold must be a Fraction.")
    example_count = _require_int(example_count, "example_count", minimum=1)
    numerator = threshold.numerator * example_count
    return (numerator + threshold.denominator - 1) // threshold.denominator


def _load_observed_circuits(
    e1_inputs: ValidatedInputs,
    *,
    source_root: Path,
) -> tuple[ObservedCircuit, ...]:
    source = e1_inputs.configuration["source"]["circuits_table"]
    path = source_root / source["path"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed_rows = {
        (
            int(row["model_seed"]),
            int(row["checkpoint_step"]),
            row["cell_id"],
            row["circuit_id"],
        ): row
        for row in rows
    }
    observed: list[ObservedCircuit] = []
    for circuit in e1_inputs.circuits:
        try:
            row = keyed_rows[circuit.key]
        except KeyError as exc:
            raise E2ValidationError(
                "E1 circuit does not join back to its authenticated compact row."
            ) from exc
        fidelity = float(row["fidelity"])
        agreement_count = int(row["agreement_count"])
        evaluated_count = int(row["evaluated_example_count"])
        if agreement_count / evaluated_count != fidelity:
            raise E2ValidationError("Observed circuit fidelity is not count-exact.")
        observed.append(
            ObservedCircuit(
                circuit=circuit,
                primary_fidelity=fidelity,
                prediction_agreement_count=agreement_count,
            )
        )
    return tuple(observed)


def build_profiles(
    observed_circuits: Sequence[ObservedCircuit],
) -> tuple[NullProfile, ...]:
    """Deduplicate random-mask execution by seed and exact composition."""
    grouped: defaultdict[tuple[int, int, int, int], list[str]] = defaultdict(list)
    for observed in observed_circuits:
        circuit = observed.circuit
        grouped[
            (
                circuit.model_seed,
                circuit.checkpoint_step,
                circuit.retained_heads,
                circuit.retained_neurons,
            )
        ].append(circuit.circuit_id)
    profiles = tuple(
        NullProfile(
            model_seed=key[0],
            checkpoint_step=key[1],
            retained_heads=key[2],
            retained_neurons=key[3],
            retained_components=key[2] + key[3],
            observed_circuit_ids=tuple(sorted(circuit_ids)),
        )
        for key, circuit_ids in sorted(grouped.items())
    )
    if len(profiles) != 21:
        raise E2ValidationError(
            f"Frozen E2 comparison set must yield 21 profiles, found {len(profiles)}."
        )
    return profiles


def _load_checkpoints(
    configuration: Mapping[str, Any],
    *,
    source_root: Path,
) -> tuple[dict[int, CheckpointSource], dict[str, str]]:
    raw_sources = configuration.get("checkpoint_sources")
    if not isinstance(raw_sources, list):
        raise E2ValidationError("checkpoint_sources must be a list.")
    checkpoints: dict[int, CheckpointSource] = {}
    hashes: dict[str, str] = {}
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise E2ValidationError("Every checkpoint source must be a mapping.")
        seed = _require_int(raw.get("model_seed"), "model_seed")
        manifest = source_root / str(raw["checkpoint_manifest"])
        checkpoint = source_root / str(raw["checkpoint_path"])
        manifest_hash = _verify_file(
            manifest,
            str(raw["checkpoint_manifest_sha256"]),
            f"seed {seed} checkpoint manifest",
        )
        checkpoint_hash = _verify_file(
            checkpoint,
            str(raw["checkpoint_sha256"]),
            f"seed {seed} checkpoint",
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        stable = payload.get("selected_stable_post_checkpoint")
        if not isinstance(stable, Mapping):
            raise E2ValidationError("Checkpoint manifest lacks selected_stable_post_checkpoint.")
        if (
            int(stable.get("training_step", -1)) != 9050
            or stable.get("checkpoint_sha256") != checkpoint_hash
            or stable.get("checkpoint_path") != raw["checkpoint_path"]
            or payload.get("run_id") != raw["run_id"]
        ):
            raise E2ValidationError(f"Seed {seed} checkpoint source is internally inconsistent.")
        if seed in checkpoints:
            raise E2ValidationError(f"Duplicate checkpoint source for seed {seed}.")
        checkpoints[seed] = CheckpointSource(
            model_seed=seed,
            run_id=str(raw["run_id"]),
            checkpoint_manifest=manifest,
            checkpoint_manifest_sha256=manifest_hash,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_hash,
        )
        hashes[f"seed_{seed}_checkpoint_manifest"] = manifest_hash
        hashes[f"seed_{seed}_checkpoint"] = checkpoint_hash
    return checkpoints, hashes


def validate_inputs(
    configuration_path: Path,
    *,
    repository_root: Path,
    source_root: Path,
) -> ValidatedE2Inputs:
    """Validate E2 configuration, E1 registry, dataset, and checkpoints."""
    configuration, configuration_hash = load_configuration(configuration_path)
    if configuration.get("schema_version") != 1:
        raise E2ValidationError("E2 schema_version must be 1.")
    if configuration.get("experiment_type") != "phase1_e2_random_mask_fidelity_null":
        raise E2ValidationError("Unexpected E2 experiment_type.")
    if tuple(configuration.get("null_models", ())) != NULL_MODELS:
        raise E2ValidationError(f"null_models must be {NULL_MODELS} in order.")
    registry = configuration.get("comparison_registry")
    if not isinstance(registry, Mapping):
        raise E2ValidationError("comparison_registry must be a mapping.")
    e1_config_path = repository_root / str(registry["path"])
    _verify_file(e1_config_path, str(registry["sha256"]), "E1 comparison registry")
    e1_inputs = validate_e1_inputs(e1_config_path, source_root=source_root)
    observed = _load_observed_circuits(e1_inputs, source_root=source_root)
    profiles = build_profiles(observed)
    checkpoints, checkpoint_hashes = _load_checkpoints(
        configuration,
        source_root=source_root,
    )
    selected_seeds = {circuit.circuit.model_seed for circuit in observed}
    if set(checkpoints) != selected_seeds:
        raise E2ValidationError(
            f"Checkpoint seeds {sorted(checkpoints)} do not match selected "
            f"seeds {sorted(selected_seeds)}."
        )
    dataset = configuration.get("dataset")
    if not isinstance(dataset, Mapping):
        raise E2ValidationError("dataset must be a mapping.")
    dataset_path = source_root / str(dataset["path"])
    dataset_hash = _verify_file(dataset_path, str(dataset["sha256"]), "dataset archive")
    fidelity = configuration.get("fidelity")
    if not isinstance(fidelity, Mapping):
        raise E2ValidationError("fidelity must be a mapping.")
    threshold = Fraction(
        _require_int(fidelity.get("threshold_numerator"), "threshold_numerator"),
        _require_int(
            fidelity.get("threshold_denominator"),
            "threshold_denominator",
            minimum=1,
        ),
    )
    example_count = _require_int(dataset.get("example_count"), "example_count", minimum=1)
    minimum_count = minimum_agreement_count(threshold, example_count=example_count)
    if threshold != Fraction(99, 100) or minimum_count != fidelity.get("minimum_agreement_count"):
        raise E2ValidationError("Frozen E2 fidelity threshold/count is inconsistent.")
    sampling = configuration.get("sampling")
    if not isinstance(sampling, Mapping):
        raise E2ValidationError("sampling must be a mapping.")
    if sampling.get("replicates_per_unique_seed_composition_profile") != 100:
        raise E2ValidationError("Frozen E2 requires 100 replicates per profile/null.")
    return ValidatedE2Inputs(
        configuration=configuration,
        configuration_sha256=configuration_hash,
        e1_inputs=e1_inputs,
        observed_circuits=observed,
        profiles=profiles,
        checkpoints=checkpoints,
        source_hashes={
            **e1_inputs.source_hashes,
            **checkpoint_hashes,
            "dataset": dataset_hash,
            "e1_comparison_registry": file_sha256(e1_config_path),
        },
    )


def derive_seed(
    *,
    analysis_id: str,
    profile: NullProfile,
    null_model: str,
    replicates: int,
) -> SeedEvidence:
    """Derive one Stage-11-style uint64 PCG64 stream."""
    if null_model not in NULL_MODELS:
        raise E2ValidationError(f"Unknown null model: {null_model!r}.")
    material = (
        "circuit-families|phase1-e2-random-mask-fidelity|"
        f"analysis_id={analysis_id}|"
        f"model_seed={profile.model_seed}|"
        f"checkpoint_step={profile.checkpoint_step}|"
        f"retained_heads={profile.retained_heads}|"
        f"retained_neurons={profile.retained_neurons}|"
        f"null_model={null_model}|replicates={replicates}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return SeedEvidence(
        canonical_material=material,
        sha256_digest=digest.hex(),
        integer_seed=int.from_bytes(digest[:8], byteorder="big"),
    )


def _mask_sha256(mask: ComponentMask) -> str:
    return hashlib.sha256(canonical_json_bytes(mask.to_record())).hexdigest()


def sample_masks(
    profile: NullProfile,
    null_model: str,
    *,
    analysis_id: str,
    replicates: int,
) -> tuple[SeedEvidence, tuple[SampledMask, ...]]:
    """Sample independent matched masks for one unique profile/null."""
    replicates = _require_int(replicates, "replicates", minimum=1)
    seed = derive_seed(
        analysis_id=analysis_id,
        profile=profile,
        null_model=null_model,
        replicates=replicates,
    )
    generator = np.random.Generator(np.random.PCG64(seed.integer_seed))
    sampled: list[SampledMask] = []
    for mask_index in range(replicates):
        if null_model == "size_matched":
            retained_indices = sorted(
                int(value)
                for value in generator.choice(
                    SEARCHABLE_COMPONENT_COUNT,
                    size=profile.retained_components,
                    replace=False,
                )
            )
            identifiers = tuple(
                (ATTENTION_HEAD_IDS + MLP_NEURON_IDS)[index] for index in retained_indices
            )
        elif null_model == "basis_stratified":
            head_indices = sorted(
                int(value)
                for value in generator.choice(
                    ATTENTION_HEAD_COUNT,
                    size=profile.retained_heads,
                    replace=False,
                )
            )
            neuron_indices = sorted(
                int(value)
                for value in generator.choice(
                    MLP_NEURON_COUNT,
                    size=profile.retained_neurons,
                    replace=False,
                )
            )
            identifiers = tuple(ATTENTION_HEAD_IDS[index] for index in head_indices) + tuple(
                MLP_NEURON_IDS[index] for index in neuron_indices
            )
        else:
            raise E2ValidationError(f"Unknown null model: {null_model!r}.")
        mask = ComponentMask.from_retained_identifiers(identifiers)
        if mask.retained_component_count != profile.retained_components:
            raise RuntimeError("Sampled mask does not preserve total retained size.")
        if null_model == "basis_stratified" and (
            mask.retained_attention_head_count != profile.retained_heads
            or mask.retained_mlp_neuron_count != profile.retained_neurons
        ):
            raise RuntimeError("Basis-stratified mask does not preserve composition.")
        sampled.append(
            SampledMask(
                mask_index=mask_index,
                mask=mask,
                mask_sha256=_mask_sha256(mask),
            )
        )
    return seed, tuple(sampled)


def evaluate_profile(
    profile: NullProfile,
    null_model: str,
    *,
    analysis_id: str,
    replicates: int,
    minimum_count: int,
    evaluator: Callable[[ComponentMask], MaskEvaluationMetrics],
) -> tuple[MaskEvaluation, ...]:
    """Sample and evaluate every random mask for one profile/null."""
    seed, masks = sample_masks(
        profile,
        null_model,
        analysis_id=analysis_id,
        replicates=replicates,
    )
    results: list[MaskEvaluation] = []
    for sampled in masks:
        metrics = evaluator(sampled.mask)
        if metrics.retained_component_count != profile.retained_components:
            raise RuntimeError("Evaluator returned inconsistent retained-component count.")
        if metrics.evaluated_example_count <= 0:
            raise RuntimeError("Evaluator did not cover any examples.")
        if metrics.prediction_agreement_count / metrics.evaluated_example_count != (
            metrics.primary_fidelity
        ):
            raise RuntimeError("Random-mask fidelity is not count-exact.")
        results.append(
            MaskEvaluation(
                profile=profile,
                null_model=null_model,
                sampled=sampled,
                metrics=metrics,
                passes_threshold=metrics.prediction_agreement_count >= minimum_count,
                seed=seed,
            )
        )
    return tuple(results)


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return math.fsum(
        math.comb(n, value) * probability**value * (1.0 - probability) ** (n - value)
        for value in range(k + 1)
    )


def _bisect_monotone(
    function: Callable[[float], float],
    target: float,
    *,
    increasing: bool,
) -> float:
    low = 0.0
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        value = function(midpoint)
        if (value < target) == increasing:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float,
) -> tuple[float, float]:
    """Return the exact equal-tail binomial proportion interval."""
    successes = _require_int(successes, "successes")
    trials = _require_int(trials, "trials", minimum=1)
    if successes > trials:
        raise E2ValidationError("successes cannot exceed trials.")
    if not 0.0 < confidence_level < 1.0:
        raise E2ValidationError("confidence_level must lie between zero and one.")
    alpha = (1.0 - confidence_level) / 2.0
    if successes == 0:
        lower = 0.0
    else:
        lower = _bisect_monotone(
            lambda probability: (
                1.0
                - _binomial_cdf(
                    successes - 1,
                    trials,
                    probability,
                )
            ),
            alpha,
            increasing=True,
        )
    if successes == trials:
        upper = 1.0
    else:
        upper = _bisect_monotone(
            lambda probability: _binomial_cdf(successes, trials, probability),
            alpha,
            increasing=False,
        )
    return lower, upper


def _evaluation_row(evaluation: MaskEvaluation, analysis_id: str) -> dict[str, Any]:
    metrics = evaluation.metrics
    mask = evaluation.sampled.mask
    return {
        "analysis_id": analysis_id,
        "profile_id": evaluation.profile.profile_id,
        "model_seed": evaluation.profile.model_seed,
        "checkpoint_step": evaluation.profile.checkpoint_step,
        "null_model": evaluation.null_model,
        "mask_index": evaluation.sampled.mask_index,
        "seed_canonical_material": evaluation.seed.canonical_material,
        "seed_sha256_digest": evaluation.seed.sha256_digest,
        "seed_integer": evaluation.seed.integer_seed,
        "seed_bit_generator": "numpy.random.PCG64",
        "target_retained_heads": evaluation.profile.retained_heads,
        "target_retained_neurons": evaluation.profile.retained_neurons,
        "target_retained_components": evaluation.profile.retained_components,
        "sampled_retained_heads": mask.retained_attention_head_count,
        "sampled_retained_neurons": mask.retained_mlp_neuron_count,
        "sampled_retained_components": mask.retained_component_count,
        "mask_id": mask.mask_id,
        "mask_sha256": evaluation.sampled.mask_sha256,
        "prediction_agreement_count": metrics.prediction_agreement_count,
        "evaluated_example_count": metrics.evaluated_example_count,
        "primary_fidelity": metrics.primary_fidelity,
        "passes_0_990_threshold": evaluation.passes_threshold,
        "full_accuracy": metrics.full_accuracy,
        "masked_accuracy": metrics.masked_accuracy,
        "accuracy_change": metrics.accuracy_change,
        "full_cross_entropy": metrics.full_cross_entropy,
        "masked_cross_entropy": metrics.masked_cross_entropy,
        "cross_entropy_change": metrics.cross_entropy_change,
        "mean_kl_divergence": metrics.mean_kl_divergence,
        "mean_jensen_shannon_divergence": metrics.mean_jensen_shannon_divergence,
        "maximum_absolute_logit_difference": metrics.maximum_absolute_logit_difference,
    }


def _profile_summary_rows(
    evaluations: Sequence[MaskEvaluation],
    observed: Sequence[ObservedCircuit],
    *,
    confidence_level: float,
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[MaskEvaluation]] = defaultdict(list)
    observed_by_profile: defaultdict[str, list[ObservedCircuit]] = defaultdict(list)
    profile_by_key = {
        (
            item.profile.model_seed,
            item.profile.retained_heads,
            item.profile.retained_neurons,
        ): item.profile.profile_id
        for item in evaluations
    }
    for item in evaluations:
        grouped[(item.profile.profile_id, item.null_model)].append(item)
    for circuit in observed:
        key = (
            circuit.circuit.model_seed,
            circuit.circuit.retained_heads,
            circuit.circuit.retained_neurons,
        )
        observed_by_profile[profile_by_key[key]].append(circuit)
    rows: list[dict[str, Any]] = []
    for (profile_id, null_model), values in sorted(grouped.items()):
        profile = values[0].profile
        fidelities = np.asarray(
            [item.metrics.primary_fidelity for item in values],
            dtype=np.float64,
        )
        pass_count = sum(item.passes_threshold for item in values)
        lower, upper = clopper_pearson_interval(
            pass_count,
            len(values),
            confidence_level=confidence_level,
        )
        observed_values = observed_by_profile[profile_id]
        percentiles = np.quantile(fidelities, [0.05, 0.5, 0.95], method="linear")
        rows.append(
            {
                "profile_id": profile_id,
                "model_seed": profile.model_seed,
                "checkpoint_step": profile.checkpoint_step,
                "null_model": null_model,
                "target_retained_heads": profile.retained_heads,
                "target_retained_neurons": profile.retained_neurons,
                "target_retained_components": profile.retained_components,
                "observed_circuit_ids_json": json.dumps(profile.observed_circuit_ids),
                "observed_circuit_count": len(observed_values),
                "observed_fidelity_min": min(item.primary_fidelity for item in observed_values),
                "observed_fidelity_max": max(item.primary_fidelity for item in observed_values),
                "random_mask_count": len(values),
                "random_mask_duplicate_count": len(values)
                - len({item.sampled.mask_sha256 for item in values}),
                "random_mask_pass_count": pass_count,
                "random_mask_pass_fraction": pass_count / len(values),
                "pass_fraction_ci_low": lower,
                "pass_fraction_ci_high": upper,
                "random_fidelity_min": float(fidelities.min()),
                "random_fidelity_p05": float(percentiles[0]),
                "random_fidelity_median": float(percentiles[1]),
                "random_fidelity_mean": float(fidelities.mean()),
                "random_fidelity_p95": float(percentiles[2]),
                "random_fidelity_max": float(fidelities.max()),
                "observed_min_minus_random_max": min(
                    item.primary_fidelity for item in observed_values
                )
                - float(fidelities.max()),
            }
        )
    return rows


def _circuit_summary_rows(
    profile_rows: Sequence[Mapping[str, Any]],
    evaluations: Sequence[MaskEvaluation],
    observed: Sequence[ObservedCircuit],
) -> list[dict[str, Any]]:
    profile_lookup = {(str(row["profile_id"]), str(row["null_model"])): row for row in profile_rows}
    evaluations_by_profile: defaultdict[tuple[str, str], list[MaskEvaluation]] = defaultdict(list)
    for item in evaluations:
        evaluations_by_profile[(item.profile.profile_id, item.null_model)].append(item)
    profile_id_by_composition = {
        (
            item.profile.model_seed,
            item.profile.retained_heads,
            item.profile.retained_neurons,
        ): item.profile.profile_id
        for item in evaluations
    }
    rows: list[dict[str, Any]] = []
    for circuit in observed:
        key = (
            circuit.circuit.model_seed,
            circuit.circuit.retained_heads,
            circuit.circuit.retained_neurons,
        )
        profile_id = profile_id_by_composition[key]
        for null_model in NULL_MODELS:
            summary = profile_lookup[(profile_id, null_model)]
            random_values = evaluations_by_profile[(profile_id, null_model)]
            at_least_observed = sum(
                item.metrics.primary_fidelity >= circuit.primary_fidelity for item in random_values
            )
            rows.append(
                {
                    "model_seed": circuit.circuit.model_seed,
                    "checkpoint_step": circuit.circuit.checkpoint_step,
                    "cell_id": circuit.circuit.cell_id,
                    "circuit_id": circuit.circuit.circuit_id,
                    "observed_mask_sha256": circuit.circuit.mask_sha256,
                    "observed_retained_heads": circuit.circuit.retained_heads,
                    "observed_retained_neurons": circuit.circuit.retained_neurons,
                    "observed_retained_components": circuit.circuit.retained_components,
                    "observed_prediction_agreement_count": (circuit.prediction_agreement_count),
                    "observed_primary_fidelity": circuit.primary_fidelity,
                    "profile_id": profile_id,
                    "null_model": null_model,
                    "random_mask_count": summary["random_mask_count"],
                    "random_mask_pass_count": summary["random_mask_pass_count"],
                    "random_mask_pass_fraction": summary["random_mask_pass_fraction"],
                    "pass_fraction_ci_low": summary["pass_fraction_ci_low"],
                    "pass_fraction_ci_high": summary["pass_fraction_ci_high"],
                    "random_fidelity_median": summary["random_fidelity_median"],
                    "random_fidelity_max": summary["random_fidelity_max"],
                    "random_ge_observed_count": at_least_observed,
                    "random_ge_observed_plus_one_tail": (at_least_observed + 1)
                    / (len(random_values) + 1),
                }
            )
    return rows


def _seed_summary_rows(
    profile_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        grouped[(int(row["model_seed"]), str(row["null_model"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (model_seed, null_model), values in sorted(grouped.items()):
        rows.append(
            {
                "model_seed": model_seed,
                "null_model": null_model,
                "profile_count": len(values),
                "random_mask_evaluation_count": sum(
                    int(item["random_mask_count"]) for item in values
                ),
                "random_mask_pass_count": sum(
                    int(item["random_mask_pass_count"]) for item in values
                ),
                "profile_pass_fraction_max": max(
                    float(item["random_mask_pass_fraction"]) for item in values
                ),
                "profile_random_fidelity_median_median": median(
                    float(item["random_fidelity_median"]) for item in values
                ),
                "profile_random_fidelity_max_max": max(
                    float(item["random_fidelity_max"]) for item in values
                ),
                "profile_observed_min_minus_random_max_min": min(
                    float(item["observed_min_minus_random_max"]) for item in values
                ),
                "inferential_unit": "none_profiles_and_masks_are_descriptive_only",
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty E2 CSV artifact.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def execute_analysis(
    inputs: ValidatedE2Inputs,
    *,
    evaluator_factory: Callable[
        [CheckpointSource, int],
        tuple[Callable[[ComponentMask], MaskEvaluationMetrics], Callable[[], None]],
    ],
) -> tuple[tuple[MaskEvaluation, ...], list[dict[str, Any]]]:
    """Evaluate every profile/null, grouped by model to reuse full references."""
    configuration = inputs.configuration
    analysis_id = str(configuration["analysis_id"])
    replicates = int(configuration["sampling"]["replicates_per_unique_seed_composition_profile"])
    batch_size = int(configuration["execution"]["evaluation_batch_size"])
    minimum_count = int(configuration["fidelity"]["minimum_agreement_count"])
    evaluations: list[MaskEvaluation] = []
    runtime_rows: list[dict[str, Any]] = []
    profiles_by_seed: defaultdict[int, list[NullProfile]] = defaultdict(list)
    for profile in inputs.profiles:
        profiles_by_seed[profile.model_seed].append(profile)
    for model_seed in sorted(profiles_by_seed):
        source = inputs.checkpoints[model_seed]
        evaluator, finalize = evaluator_factory(source, batch_size)
        try:
            for profile in profiles_by_seed[model_seed]:
                for null_model in NULL_MODELS:
                    started = time.perf_counter()
                    profile_evaluations = evaluate_profile(
                        profile,
                        null_model,
                        analysis_id=analysis_id,
                        replicates=replicates,
                        minimum_count=minimum_count,
                        evaluator=evaluator,
                    )
                    elapsed = time.perf_counter() - started
                    evaluations.extend(profile_evaluations)
                    runtime_rows.append(
                        {
                            "analysis_id": analysis_id,
                            "profile_id": profile.profile_id,
                            "model_seed": model_seed,
                            "null_model": null_model,
                            "mask_count": len(profile_evaluations),
                            "elapsed_seconds": elapsed,
                            "seconds_per_mask": elapsed / len(profile_evaluations),
                            "included_in_deterministic_scientific_hashes": False,
                        }
                    )
        finally:
            finalize()
    return tuple(evaluations), runtime_rows


def write_outputs(
    output_directory: Path,
    inputs: ValidatedE2Inputs,
    evaluations: Sequence[MaskEvaluation],
    runtime_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Write deterministic scientific E2 tables plus excluded runtime telemetry."""
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError("E2 output directory must be absent or empty.")
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = inputs.configuration["outputs"]
    paths = {name: output_directory / filename for name, filename in outputs.items()}
    analysis_id = str(inputs.configuration["analysis_id"])
    evaluation_rows = [_evaluation_row(item, analysis_id) for item in evaluations]
    confidence_level = float(inputs.configuration["inference"]["confidence_level"])
    profile_rows = _profile_summary_rows(
        evaluations,
        inputs.observed_circuits,
        confidence_level=confidence_level,
    )
    circuit_rows = _circuit_summary_rows(
        profile_rows,
        evaluations,
        inputs.observed_circuits,
    )
    seed_rows = _seed_summary_rows(profile_rows)
    _write_csv(paths["evaluations"], evaluation_rows)
    _write_csv(paths["profile_summary"], profile_rows)
    _write_csv(paths["circuit_summary"], circuit_rows)
    _write_csv(paths["seed_summary"], seed_rows)
    _write_csv(paths["runtime"], runtime_rows)
    deterministic_paths = {
        name: path for name, path in paths.items() if name not in {"manifest", "runtime"}
    }
    manifest = {
        "schema_version": 1,
        "experiment_type": inputs.configuration["experiment_type"],
        "analysis_id": analysis_id,
        "configuration_sha256": inputs.configuration_sha256,
        "source_hashes": dict(inputs.source_hashes),
        "comparison_set": dict(inputs.e1_inputs.configuration["comparison_set"]),
        "component_universe": dict(inputs.e1_inputs.configuration["component_universe"]),
        "observed_circuit_count": len(inputs.observed_circuits),
        "unique_profile_count": len(inputs.profiles),
        "null_models": list(NULL_MODELS),
        "random_mask_evaluation_count": len(evaluations),
        "sampling": dict(inputs.configuration["sampling"]),
        "fidelity": dict(inputs.configuration["fidelity"]),
        "inference": dict(inputs.configuration["inference"]),
        "scientific_scope": {
            "full_domain_exact_top1_fidelity": True,
            "cross_model_pooling_for_inference": False,
            "random_masks_conditioned_on_observed_membership_frequencies": False,
            "runtime_included_in_scientific_hashes": False,
        },
        "outputs": {
            path.relative_to(output_directory).as_posix(): file_sha256(path)
            for path in deterministic_paths.values()
        },
        "runtime": {
            "path": paths["runtime"].relative_to(output_directory).as_posix(),
            "included_in_deterministic_scientific_hashes": False,
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return paths
