"""Phase I E1 size-matched nulls for recovered-circuit Jaccard overlap."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_COUNT,
    ATTENTION_HEAD_IDS,
    MLP_NEURON_COUNT,
    SEARCHABLE_COMPONENT_COUNT,
)
from circuit_families.seeds import numpy_generator

NULL_MODELS = ("size_matched", "basis_stratified")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class E1ValidationError(ValueError):
    """Raised when an E1 input or frozen contract is inconsistent."""


@dataclass(frozen=True)
class CircuitRecord:
    """One selected compact canonical circuit record."""

    model_seed: int
    checkpoint_step: int
    cell_id: str
    circuit_id: str
    mask_sha256: str
    retained_heads: int
    retained_neurons: int
    retained_components: int

    @property
    def key(self) -> tuple[int, int, str, str]:
        return (self.model_seed, self.checkpoint_step, self.cell_id, self.circuit_id)


@dataclass(frozen=True)
class ObservedPair:
    """One selected within-family observed pair."""

    pair_id: str
    model_seed: int
    checkpoint_step: int
    cell_id: str
    left: CircuitRecord
    right: CircuitRecord
    intersection: int
    union: int

    @property
    def jaccard(self) -> float:
        return self.intersection / self.union if self.union else 1.0


@dataclass(frozen=True)
class SeedEvidence:
    """Repository-standard SHA-256 to 32-bit PCG64 seed evidence."""

    canonical_material: str
    sha256_digest: str
    integer_seed: int


@dataclass(frozen=True)
class DistributionPoint:
    """One possible total intersection and its exact probability."""

    intersection: int
    probability: float


@dataclass(frozen=True)
class NullResult:
    """Exact and sampled comparison for one pair under one null."""

    null_model: str
    distribution: tuple[DistributionPoint, ...]
    exact_mean: float
    exact_std: float
    ci_low: float
    ci_high: float
    lower_tail: float
    strict_lower_tail: float
    upper_tail: float
    mid_percentile: float
    z_score: float | None
    sample_mean: float
    sample_std: float
    sample_ci_low: float
    sample_ci_high: float
    sample_lower_tail: float
    sampled_intersection_counts: Mapping[int, int]
    seed: SeedEvidence


@dataclass(frozen=True)
class ValidatedInputs:
    """Validated E1 configuration and selected comparison set."""

    configuration: Mapping[str, Any]
    configuration_sha256: str
    source_hashes: Mapping[str, str]
    circuits: tuple[CircuitRecord, ...]
    pairs: tuple[ObservedPair, ...]


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a mapping deterministically for identity hashing."""
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def load_configuration(path: Path) -> tuple[dict[str, Any], str]:
    """Load an E1 JSON configuration and return its canonical hash."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E1ValidationError("E1 configuration must be a JSON object.")
    return value, hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise E1ValidationError(f"{label} must be an integer >= {minimum}.")
    return value


def _parse_int(value: str, label: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise E1ValidationError(f"{label} must be an integer.") from exc
    if parsed < minimum:
        raise E1ValidationError(f"{label} must be >= {minimum}.")
    return parsed


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _verify_source_artifact(source_root: Path, record: Mapping[str, Any], label: str) -> Path:
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise E1ValidationError(f"{label} path and sha256 must be strings.")
    path = source_root / raw_path
    if not path.is_file():
        raise FileNotFoundError(f"Required E1 {label} is absent: {path}")
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise E1ValidationError(
            f"{label} SHA-256 mismatch: expected {expected_hash}, found {actual_hash}."
        )
    return path


def _validate_universe(configuration: Mapping[str, Any]) -> None:
    universe = configuration.get("component_universe")
    if not isinstance(universe, Mapping):
        raise E1ValidationError("component_universe must be a mapping.")
    expected = {
        "attention_head_count": ATTENTION_HEAD_COUNT,
        "mlp_neuron_count": MLP_NEURON_COUNT,
        "total_count": SEARCHABLE_COMPONENT_COUNT,
    }
    for field, value in expected.items():
        if universe.get(field) != value:
            raise E1ValidationError(f"component_universe {field} must be {value}.")
    if tuple(universe.get("attention_head_identifiers", ())) != ATTENTION_HEAD_IDS:
        raise E1ValidationError("Frozen attention-head identifiers do not match masks.py.")
    if universe.get("mlp_neuron_identifier_template") != "N0-N511":
        raise E1ValidationError("Frozen MLP-neuron identifiers must be N0-N511.")


def _selection_predicate(row: Mapping[str, str], selection: Mapping[str, Any]) -> bool:
    return (
        _parse_int(row["model_seed"], "model_seed") in set(selection["model_seeds"])
        and _parse_int(row["checkpoint_step"], "checkpoint_step") == selection["checkpoint_step"]
        and _parse_int(row["fidelity_numerator"], "fidelity_numerator")
        == selection["fidelity_numerator"]
        and _parse_int(row["fidelity_denominator"], "fidelity_denominator", minimum=1)
        == selection["fidelity_denominator"]
        and _parse_int(row["distinctness_numerator"], "distinctness_numerator")
        == selection["distinctness_numerator"]
        and _parse_int(row["distinctness_denominator"], "distinctness_denominator", minimum=1)
        == selection["distinctness_denominator"]
        and row["cell_id"] in set(selection["cell_ids"])
    )


def _load_circuits(path: Path, selection: Mapping[str, Any]) -> tuple[CircuitRecord, ...]:
    selected = [row for row in _read_csv(path) if _selection_predicate(row, selection)]
    circuits: list[CircuitRecord] = []
    for row in selected:
        if row.get("threshold_pass") != "True":
            raise E1ValidationError("Selected circuit does not pass its frozen threshold.")
        mask_sha256 = row.get("mask_sha256", "")
        if not _SHA256_RE.fullmatch(mask_sha256):
            raise E1ValidationError(
                "Selected circuit mask_sha256 is not canonical lowercase SHA-256."
            )
        circuit = CircuitRecord(
            model_seed=_parse_int(row["model_seed"], "model_seed"),
            checkpoint_step=_parse_int(row["checkpoint_step"], "checkpoint_step"),
            cell_id=row["cell_id"],
            circuit_id=row["circuit_id"],
            mask_sha256=mask_sha256,
            retained_heads=_parse_int(row["retained_heads"], "retained_heads"),
            retained_neurons=_parse_int(row["retained_neurons"], "retained_neurons"),
            retained_components=_parse_int(row["retained_components"], "retained_components"),
        )
        if circuit.retained_heads > ATTENTION_HEAD_COUNT:
            raise E1ValidationError("Circuit retains more than four attention heads.")
        if circuit.retained_neurons > MLP_NEURON_COUNT:
            raise E1ValidationError("Circuit retains more than 512 MLP neurons.")
        if circuit.retained_components != circuit.retained_heads + circuit.retained_neurons:
            raise E1ValidationError("Circuit retained-component accounting is inconsistent.")
        circuits.append(circuit)
    circuits.sort(key=lambda item: item.key)
    if len({item.key for item in circuits}) != len(circuits):
        raise E1ValidationError("Selected circuit keys are not unique.")
    return tuple(circuits)


def _pair_identifier(left: CircuitRecord, right: CircuitRecord) -> str:
    return f"s{left.model_seed}:{left.cell_id}:{left.circuit_id}--{right.circuit_id}"


def _load_pairs(
    path: Path,
    selection: Mapping[str, Any],
    circuits: Sequence[CircuitRecord],
) -> tuple[ObservedPair, ...]:
    by_key = {item.key: item for item in circuits}
    selected = [row for row in _read_csv(path) if _selection_predicate(row, selection)]
    pairs: list[ObservedPair] = []
    for row in selected:
        base = (
            _parse_int(row["model_seed"], "model_seed"),
            _parse_int(row["checkpoint_step"], "checkpoint_step"),
            row["cell_id"],
        )
        try:
            left = by_key[(*base, row["circuit_i"])]
            right = by_key[(*base, row["circuit_j"])]
        except KeyError as exc:
            raise E1ValidationError("Pairwise row does not join to selected circuits.") from exc
        if left.circuit_id >= right.circuit_id:
            raise E1ValidationError("Pairwise circuit order must be canonical and increasing.")
        intersection = _parse_int(row["intersection_count"], "intersection_count")
        union = _parse_int(row["union_count"], "union_count")
        expected_union = left.retained_components + right.retained_components - intersection
        if union != expected_union or intersection > min(
            left.retained_components, right.retained_components
        ):
            raise E1ValidationError("Observed pair intersection/union accounting is inconsistent.")
        numerator = _parse_int(row["jaccard_numerator"], "jaccard_numerator")
        denominator = _parse_int(row["jaccard_denominator"], "jaccard_denominator", minimum=1)
        if numerator * union != intersection * denominator:
            raise E1ValidationError("Observed Jaccard rational is inconsistent with counts.")
        pairs.append(
            ObservedPair(
                pair_id=_pair_identifier(left, right),
                model_seed=left.model_seed,
                checkpoint_step=left.checkpoint_step,
                cell_id=left.cell_id,
                left=left,
                right=right,
                intersection=intersection,
                union=union,
            )
        )
    pairs.sort(key=lambda item: (item.model_seed, item.cell_id, item.pair_id))
    if len({item.pair_id for item in pairs}) != len(pairs):
        raise E1ValidationError("Selected pair identifiers are not unique.")
    return tuple(pairs)


def _validate_selected_counts(
    selection: Mapping[str, Any],
    circuits: Sequence[CircuitRecord],
    pairs: Sequence[ObservedPair],
) -> None:
    expected_sizes = {
        int(seed): int(size) for seed, size in selection["expected_family_sizes"].items()
    }
    actual_sizes = Counter(item.model_seed for item in circuits)
    if dict(sorted(actual_sizes.items())) != dict(sorted(expected_sizes.items())):
        raise E1ValidationError(
            "Selected family sizes mismatch: "
            f"expected {expected_sizes}, found {dict(actual_sizes)}."
        )
    if len(circuits) != selection["expected_circuit_count"]:
        raise E1ValidationError("Selected circuit count does not match the frozen comparison set.")
    if len(pairs) != selection["expected_pair_count"]:
        raise E1ValidationError("Selected pair count does not match the frozen comparison set.")
    expected_pairs = {seed: size * (size - 1) // 2 for seed, size in expected_sizes.items()}
    actual_pairs = Counter(item.model_seed for item in pairs)
    if dict(sorted(actual_pairs.items())) != dict(sorted(expected_pairs.items())):
        raise E1ValidationError("Selected rows are not the complete within-family pair set.")


def validate_inputs(
    configuration_path: Path,
    *,
    source_root: Path,
) -> ValidatedInputs:
    """Validate frozen inputs and select the 27-circuit/78-pair E1 set."""
    configuration, configuration_hash = load_configuration(configuration_path)
    if configuration.get("schema_version") != 1:
        raise E1ValidationError("E1 schema_version must be 1.")
    if configuration.get("experiment_type") != "phase1_e1_size_matched_jaccard_null":
        raise E1ValidationError("Unexpected E1 experiment_type.")
    _validate_universe(configuration)
    if tuple(configuration.get("null_models", ())) != NULL_MODELS:
        raise E1ValidationError(f"null_models must be {NULL_MODELS} in order.")
    source = configuration.get("source")
    selection = configuration.get("comparison_set")
    if not isinstance(source, Mapping) or not isinstance(selection, Mapping):
        raise E1ValidationError("source and comparison_set must be mappings.")
    if selection.get("scope") != "within_family_pairs_only":
        raise E1ValidationError("E1 comparison scope must be within_family_pairs_only.")
    manifest_path = _verify_source_artifact(source_root, source["manifest"], "manifest")
    circuits_path = _verify_source_artifact(source_root, source["circuits_table"], "circuits table")
    pairs_path = _verify_source_artifact(
        source_root, source["pairwise_overlap_table"], "pairwise-overlap table"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise E1ValidationError("Source manifest outputs mapping is absent.")
    for key in ("circuits_table", "pairwise_overlap_table"):
        record = source[key]
        if outputs.get(record["path"]) != record["sha256"]:
            raise E1ValidationError(f"Source manifest does not authenticate {key}.")
    circuits = _load_circuits(circuits_path, selection)
    pairs = _load_pairs(pairs_path, selection, circuits)
    _validate_selected_counts(selection, circuits, pairs)
    return ValidatedInputs(
        configuration=configuration,
        configuration_sha256=configuration_hash,
        source_hashes={
            "manifest": file_sha256(manifest_path),
            "circuits_table": file_sha256(circuits_path),
            "pairwise_overlap_table": file_sha256(pairs_path),
        },
        circuits=circuits,
        pairs=pairs,
    )


def hypergeometric_pmf(
    population_size: int,
    left_size: int,
    right_size: int,
) -> tuple[DistributionPoint, ...]:
    """Return exact fixed-size subset-intersection probabilities."""
    population_size = _require_int(population_size, "population_size", minimum=1)
    left_size = _require_int(left_size, "left_size")
    right_size = _require_int(right_size, "right_size")
    if left_size > population_size or right_size > population_size:
        raise E1ValidationError("Subset sizes cannot exceed their population.")
    low = max(0, left_size + right_size - population_size)
    high = min(left_size, right_size)
    denominator = math.comb(population_size, right_size)
    points = tuple(
        DistributionPoint(
            intersection=value,
            probability=(
                math.comb(left_size, value)
                * math.comb(population_size - left_size, right_size - value)
                / denominator
            ),
        )
        for value in range(low, high + 1)
    )
    if not math.isclose(sum(point.probability for point in points), 1.0, abs_tol=1e-12):
        raise RuntimeError("Hypergeometric probabilities do not sum to one.")
    return points


def convolve_intersections(
    left: Sequence[DistributionPoint],
    right: Sequence[DistributionPoint],
) -> tuple[DistributionPoint, ...]:
    """Convolve independent stratum-level intersection distributions."""
    probabilities: defaultdict[int, float] = defaultdict(float)
    for left_point in left:
        for right_point in right:
            probabilities[left_point.intersection + right_point.intersection] += (
                left_point.probability * right_point.probability
            )
    return tuple(
        DistributionPoint(intersection=value, probability=probabilities[value])
        for value in sorted(probabilities)
    )


def exact_distribution(pair: ObservedPair, null_model: str) -> tuple[DistributionPoint, ...]:
    """Build the exact intersection distribution for a requested E1 null."""
    if null_model == "size_matched":
        return hypergeometric_pmf(
            SEARCHABLE_COMPONENT_COUNT,
            pair.left.retained_components,
            pair.right.retained_components,
        )
    if null_model == "basis_stratified":
        heads = hypergeometric_pmf(
            ATTENTION_HEAD_COUNT,
            pair.left.retained_heads,
            pair.right.retained_heads,
        )
        neurons = hypergeometric_pmf(
            MLP_NEURON_COUNT,
            pair.left.retained_neurons,
            pair.right.retained_neurons,
        )
        return convolve_intersections(heads, neurons)
    raise E1ValidationError(f"Unknown null model: {null_model!r}.")


def jaccard_from_intersection(intersection: int, left_size: int, right_size: int) -> float:
    """Compute Jaccard similarity, defining empty/empty as one."""
    union = left_size + right_size - intersection
    return intersection / union if union else 1.0


def derive_null_seed(analysis_id: str, pair_id: str, null_model: str) -> SeedEvidence:
    """Derive a Stage-12-style 32-bit PCG64 seed for one pair/null."""
    material = (
        "circuit-families|phase1-e1-jaccard-null|"
        f"analysis_id={analysis_id}|pair_id={pair_id}|null_model={null_model}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return SeedEvidence(
        canonical_material=material,
        sha256_digest=digest.hex(),
        integer_seed=int.from_bytes(digest[:4], byteorder="big"),
    )


def _weighted_quantile(
    distribution: Sequence[DistributionPoint],
    probability: float,
    left_size: int,
    right_size: int,
) -> float:
    cumulative = 0.0
    for point in distribution:
        cumulative += point.probability
        if cumulative + 1e-15 >= probability:
            return jaccard_from_intersection(point.intersection, left_size, right_size)
    return jaccard_from_intersection(distribution[-1].intersection, left_size, right_size)


def _sample_intersections(
    pair: ObservedPair,
    null_model: str,
    *,
    draw_count: int,
    seed: SeedEvidence,
) -> np.ndarray:
    generator = numpy_generator(seed.integer_seed)
    if null_model == "size_matched":
        return generator.hypergeometric(
            pair.left.retained_components,
            SEARCHABLE_COMPONENT_COUNT - pair.left.retained_components,
            pair.right.retained_components,
            size=draw_count,
        )
    if null_model == "basis_stratified":
        head_intersections = generator.hypergeometric(
            pair.left.retained_heads,
            ATTENTION_HEAD_COUNT - pair.left.retained_heads,
            pair.right.retained_heads,
            size=draw_count,
        )
        neuron_intersections = generator.hypergeometric(
            pair.left.retained_neurons,
            MLP_NEURON_COUNT - pair.left.retained_neurons,
            pair.right.retained_neurons,
            size=draw_count,
        )
        return head_intersections + neuron_intersections
    raise E1ValidationError(f"Unknown null model: {null_model!r}.")


def analyse_pair(
    pair: ObservedPair,
    null_model: str,
    *,
    analysis_id: str,
    draw_count: int,
    confidence_level: float,
) -> NullResult:
    """Compare one observed Jaccard against one exact and sampled null."""
    distribution = exact_distribution(pair, null_model)
    left_size = pair.left.retained_components
    right_size = pair.right.retained_components
    values = np.asarray(
        [
            jaccard_from_intersection(point.intersection, left_size, right_size)
            for point in distribution
        ],
        dtype=np.float64,
    )
    probabilities = np.asarray([point.probability for point in distribution], dtype=np.float64)
    exact_mean = float(np.dot(values, probabilities))
    exact_variance = float(np.dot((values - exact_mean) ** 2, probabilities))
    exact_std = math.sqrt(max(0.0, exact_variance))
    lower_tail = sum(
        point.probability for point in distribution if point.intersection <= pair.intersection
    )
    strict_lower = sum(
        point.probability for point in distribution if point.intersection < pair.intersection
    )
    upper_tail = sum(
        point.probability for point in distribution if point.intersection >= pair.intersection
    )
    mass_at_observed = sum(
        point.probability for point in distribution if point.intersection == pair.intersection
    )
    alpha = (1.0 - confidence_level) / 2.0
    seed = derive_null_seed(analysis_id, pair.pair_id, null_model)
    sampled_intersections = _sample_intersections(
        pair, null_model, draw_count=draw_count, seed=seed
    )
    sampled_values = sampled_intersections / (left_size + right_size - sampled_intersections)
    sample_counts = Counter(int(value) for value in sampled_intersections)
    return NullResult(
        null_model=null_model,
        distribution=distribution,
        exact_mean=exact_mean,
        exact_std=exact_std,
        ci_low=_weighted_quantile(distribution, alpha, left_size, right_size),
        ci_high=_weighted_quantile(distribution, 1.0 - alpha, left_size, right_size),
        lower_tail=lower_tail,
        strict_lower_tail=strict_lower,
        upper_tail=upper_tail,
        mid_percentile=strict_lower + 0.5 * mass_at_observed,
        z_score=((pair.jaccard - exact_mean) / exact_std if exact_std > 0.0 else None),
        sample_mean=float(np.mean(sampled_values)),
        sample_std=float(np.std(sampled_values, ddof=0)),
        sample_ci_low=float(np.quantile(sampled_values, alpha, method="inverted_cdf")),
        sample_ci_high=float(np.quantile(sampled_values, 1.0 - alpha, method="inverted_cdf")),
        sample_lower_tail=(int(np.count_nonzero(sampled_intersections <= pair.intersection)) + 1)
        / (draw_count + 1),
        sampled_intersection_counts=dict(sample_counts),
        seed=seed,
    )


def analyse_inputs(inputs: ValidatedInputs) -> list[tuple[ObservedPair, NullResult]]:
    """Run both frozen E1 nulls for every selected pair."""
    monte_carlo = inputs.configuration["monte_carlo"]
    inference = inputs.configuration["inference"]
    draw_count = _require_int(monte_carlo["draws_per_pair"], "draws_per_pair", minimum=1)
    confidence_level = float(inference["confidence_level"])
    if not 0.0 < confidence_level < 1.0:
        raise E1ValidationError("confidence_level must lie strictly between zero and one.")
    analysis_id = str(inputs.configuration["analysis_id"])
    return [
        (
            pair,
            analyse_pair(
                pair,
                null_model,
                analysis_id=analysis_id,
                draw_count=draw_count,
                confidence_level=confidence_level,
            ),
        )
        for pair in inputs.pairs
        for null_model in NULL_MODELS
    ]


def _pair_result_row(pair: ObservedPair, result: NullResult, draw_count: int) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "model_seed": pair.model_seed,
        "checkpoint_step": pair.checkpoint_step,
        "cell_id": pair.cell_id,
        "left_circuit_id": pair.left.circuit_id,
        "right_circuit_id": pair.right.circuit_id,
        "left_mask_sha256": pair.left.mask_sha256,
        "right_mask_sha256": pair.right.mask_sha256,
        "left_retained_heads": pair.left.retained_heads,
        "left_retained_neurons": pair.left.retained_neurons,
        "left_retained_components": pair.left.retained_components,
        "right_retained_heads": pair.right.retained_heads,
        "right_retained_neurons": pair.right.retained_neurons,
        "right_retained_components": pair.right.retained_components,
        "observed_intersection": pair.intersection,
        "observed_union": pair.union,
        "observed_jaccard": pair.jaccard,
        "null_model": result.null_model,
        "exact_null_mean": result.exact_mean,
        "exact_null_std": result.exact_std,
        "exact_ci_low": result.ci_low,
        "exact_ci_high": result.ci_high,
        "exact_lower_tail_probability": result.lower_tail,
        "exact_strict_lower_tail_probability": result.strict_lower_tail,
        "exact_upper_tail_probability": result.upper_tail,
        "exact_mid_percentile": result.mid_percentile,
        "z_score": "" if result.z_score is None else result.z_score,
        "monte_carlo_draw_count": draw_count,
        "monte_carlo_mean": result.sample_mean,
        "monte_carlo_std": result.sample_std,
        "monte_carlo_ci_low": result.sample_ci_low,
        "monte_carlo_ci_high": result.sample_ci_high,
        "monte_carlo_lower_tail_plus_one": result.sample_lower_tail,
        "seed_canonical_material": result.seed.canonical_material,
        "seed_sha256_digest": result.seed.sha256_digest,
        "seed_integer": result.seed.integer_seed,
        "seed_bit_generator": "numpy.random.PCG64",
    }


def _distribution_rows(
    analyses: Sequence[tuple[ObservedPair, NullResult]], draw_count: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, result in analyses:
        cumulative = 0.0
        for point in result.distribution:
            cumulative += point.probability
            sample_count = result.sampled_intersection_counts.get(point.intersection, 0)
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "model_seed": pair.model_seed,
                    "cell_id": pair.cell_id,
                    "null_model": result.null_model,
                    "intersection": point.intersection,
                    "jaccard": jaccard_from_intersection(
                        point.intersection,
                        pair.left.retained_components,
                        pair.right.retained_components,
                    ),
                    "exact_probability": point.probability,
                    "exact_cumulative_probability": cumulative,
                    "monte_carlo_count": sample_count,
                    "monte_carlo_probability": sample_count / draw_count,
                }
            )
    return rows


def _summary_rows(
    analyses: Sequence[tuple[ObservedPair, NullResult]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[tuple[ObservedPair, NullResult]]] = defaultdict(list)
    for pair, result in analyses:
        grouped[(result.null_model, f"seed_{pair.model_seed}")].append((pair, result))
        grouped[(result.null_model, "pooled_descriptive")].append((pair, result))
    rows: list[dict[str, Any]] = []
    for (null_model, group), values in sorted(grouped.items()):
        observed = [pair.jaccard for pair, _ in values]
        expected = [result.exact_mean for _, result in values]
        lower_tails = [result.lower_tail for _, result in values]
        upper_tails = [result.upper_tail for _, result in values]
        z_scores = [result.z_score for _, result in values if result.z_score is not None]
        below_interval = sum(pair.jaccard < result.ci_low for pair, result in values)
        above_interval = sum(pair.jaccard > result.ci_high for pair, result in values)
        rows.append(
            {
                "null_model": null_model,
                "aggregation": group,
                "pair_count": len(values),
                "observed_jaccard_mean": sum(observed) / len(observed),
                "observed_jaccard_median": median(observed),
                "null_expectation_mean": sum(expected) / len(expected),
                "null_expectation_median": median(expected),
                "observed_minus_null_mean": sum(
                    observed_value - expected_value
                    for observed_value, expected_value in zip(observed, expected, strict=True)
                )
                / len(values),
                "median_exact_lower_tail_probability": median(lower_tails),
                "pair_count_lower_tail_le_0_05": sum(value <= 0.05 for value in lower_tails),
                "median_exact_upper_tail_probability": median(upper_tails),
                "pair_count_upper_tail_le_0_05": sum(value <= 0.05 for value in upper_tails),
                "pair_count_below_exact_central_interval": below_interval,
                "pair_count_inside_exact_central_interval": (
                    len(values) - below_interval - above_interval
                ),
                "pair_count_above_exact_central_interval": above_interval,
                "median_z_score": "" if not z_scores else median(z_scores),
                "inferential_unit": "none_pairs_are_dependent_descriptive_summary_only",
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty E1 CSV artifact.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_directory: Path,
    inputs: ValidatedInputs,
    analyses: Sequence[tuple[ObservedPair, NullResult]],
) -> dict[str, Path]:
    """Write deterministic machine-readable E1 artifacts without overwrite."""
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError("E1 output directory must be absent or empty.")
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = inputs.configuration["outputs"]
    draw_count = int(inputs.configuration["monte_carlo"]["draws_per_pair"])
    paths = {
        "pairwise_results": output_directory / outputs["pairwise_results"],
        "null_distributions": output_directory / outputs["null_distributions"],
        "summary": output_directory / outputs["summary"],
        "manifest": output_directory / outputs["manifest"],
    }
    _write_csv(
        paths["pairwise_results"],
        [_pair_result_row(pair, result, draw_count) for pair, result in analyses],
    )
    _write_csv(paths["null_distributions"], _distribution_rows(analyses, draw_count))
    _write_csv(paths["summary"], _summary_rows(analyses))
    manifest = {
        "schema_version": 1,
        "experiment_type": inputs.configuration["experiment_type"],
        "analysis_id": inputs.configuration["analysis_id"],
        "configuration_sha256": inputs.configuration_sha256,
        "source_hashes": dict(inputs.source_hashes),
        "component_universe": dict(inputs.configuration["component_universe"]),
        "comparison_set": dict(inputs.configuration["comparison_set"]),
        "circuit_count": len(inputs.circuits),
        "pair_count": len(inputs.pairs),
        "pair_null_comparison_count": len(analyses),
        "null_models": list(NULL_MODELS),
        "monte_carlo": dict(inputs.configuration["monte_carlo"]),
        "inference": dict(inputs.configuration["inference"]),
        "scientific_scope": {
            "cross_model_pairs_included": False,
            "pooled_pair_inference_performed": False,
            "behavioural_fidelity_evaluated": False,
            "component_frequency_preserved": False,
        },
        "outputs": {
            path.relative_to(output_directory).as_posix(): file_sha256(path)
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return paths
