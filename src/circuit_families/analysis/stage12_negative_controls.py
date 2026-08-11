"""Deterministic negative controls for Stage 12."""

from __future__ import annotations

import json
import math
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from circuit_families.analysis.fidelity_calibration import (
    file_sha256,
    write_csv_records,
)
from circuit_families.interpretability.fidelity import (
    MaskEvaluationMetrics,
)
from circuit_families.interpretability.masks import (
    ComponentMask,
)
from circuit_families.interpretability.overlap_constraints import (
    jaccard_fraction,
)
from circuit_families.interpretability.sparse_search import (
    CandidateEvaluation,
    ComponentRanking,
    RankingResult,
    SparseSearchResult,
)

STAGE11_RANDOM_MASK_CONTROL = (
    "stage11_matched_size_random_masks"
)
DEGRADED_C1_CONTROL = "degraded_c1_terminal_deletion"
SHUFFLED_RANKING_CONTROL = "shuffled_ranking"
FIDELITY_IMPOSSIBLE_CONTROL = "fidelity_impossible"
DISTINCTNESS_IMPOSSIBLE_CONTROL = (
    "distinctness_impossible"
)

EXPECTED_REJECTION = "rejection"
EXPECTED_DETERMINISTIC_PERTURBATION = (
    "deterministic_ranking_perturbation"
)

NEGATIVE_CONTROL_COLUMNS = (
    "stage12_run_id",
    "control_name",
    "control_scope",
    "expected_outcome",
    "observed_outcome",
    "validation_passed",
    "scientific_family_result",
    "record_count",
    "qualifying_count",
    "selected_component",
    "selected_component_index",
    "mask_id",
    "retained_component_count",
    "primary_fidelity",
    "fidelity_threshold",
    "jaccard_overlap",
    "distinctness_cutoff",
    "seed_integer",
    "bit_generator",
    "details_json",
)


@dataclass(frozen=True)
class Stage11RandomMaskControlRecord:
    """One committed Stage 11 matched-size random mask."""

    archive_member: str
    mask_index: int
    fidelity_threshold: Fraction
    mask: ComponentMask
    metrics: MaskEvaluationMetrics
    passes_candidate_threshold: bool


@dataclass(frozen=True)
class DegradedC1Selection:
    """Mechanically selected terminal deletion from C1."""

    candidate_component: str
    component_index: int
    candidate_mask: ComponentMask
    metrics: MaskEvaluationMetrics
    exact_fidelity: float
    fidelity_threshold: float
    rejection_reason: str | None


@dataclass(frozen=True)
class NegativeControlResult:
    """One deterministic Stage 12 control conclusion."""

    control_name: str
    control_scope: str
    expected_outcome: str
    observed_outcome: str
    validation_passed: bool
    record_count: int = 1
    qualifying_count: int = 0
    selected_component: str = ""
    selected_component_index: int | str = ""
    mask_id: str = ""
    retained_component_count: int | str = ""
    primary_fidelity: float | str = ""
    fidelity_threshold: float | str = ""
    jaccard_overlap: float | str = ""
    distinctness_cutoff: float | str = ""
    seed_integer: int | str = ""
    bit_generator: str = ""
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class NegativeControlArtifacts:
    """Path and hash for the Stage 12 control table."""

    table_path: Path
    table_sha256: str
    row_count: int


def _fraction_from_number(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(
            "Boolean values are not valid fractions."
        )

    if isinstance(value, int):
        return Fraction(value, 1)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Fraction value must be finite."
            )
        return Fraction(str(value))

    if isinstance(value, str):
        return Fraction(value)

    raise TypeError(
        "Fraction value must be int, float, or str."
    )


def _metrics_from_record(
    record: Mapping[str, Any],
) -> MaskEvaluationMetrics:
    required = {
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
    }
    missing = sorted(required.difference(record))

    if missing:
        raise ValueError(
            "Metric record is missing fields: "
            + ", ".join(missing)
        )

    return MaskEvaluationMetrics(
        primary_fidelity=float(
            record["primary_fidelity"]
        ),
        prediction_agreement_count=int(
            record["prediction_agreement_count"]
        ),
        full_accuracy=float(record["full_accuracy"]),
        masked_accuracy=float(
            record["masked_accuracy"]
        ),
        accuracy_change=float(
            record["accuracy_change"]
        ),
        full_cross_entropy=float(
            record["full_cross_entropy"]
        ),
        masked_cross_entropy=float(
            record["masked_cross_entropy"]
        ),
        cross_entropy_change=float(
            record["cross_entropy_change"]
        ),
        mean_kl_divergence=float(
            record["mean_kl_divergence"]
        ),
        mean_jensen_shannon_divergence=float(
            record["mean_jensen_shannon_divergence"]
        ),
        maximum_absolute_logit_difference=float(
            record[
                "maximum_absolute_logit_difference"
            ]
        ),
        retained_attention_head_count=int(
            record["retained_attention_head_count"]
        ),
        retained_mlp_neuron_count=int(
            record["retained_mlp_neuron_count"]
        ),
        retained_component_count=int(
            record["retained_component_count"]
        ),
        retained_component_proportion=float(
            record["retained_component_proportion"]
        ),
        evaluated_example_count=int(
            record["evaluated_example_count"]
        ),
        evaluation_batch_size=int(
            record["evaluation_batch_size"]
        ),
    )


def load_stage11_random_mask_controls(
    archive_path: str | Path,
    *,
    fidelity_threshold: Fraction = Fraction(99, 100),
) -> tuple[Stage11RandomMaskControlRecord, ...]:
    """Load one threshold's committed Stage 11 mask records."""

    if not isinstance(fidelity_threshold, Fraction):
        raise TypeError(
            "fidelity_threshold must be a Fraction."
        )

    threshold_directory = (
        f"threshold_{float(fidelity_threshold):.6f}"
        "/masks/"
    )
    records: list[Stage11RandomMaskControlRecord] = []

    with tarfile.open(
        Path(archive_path),
        mode="r:gz",
    ) as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if (
                    member.isfile()
                    and threshold_directory
                    in member.name
                    and member.name.endswith(".json")
                )
            ),
            key=lambda member: member.name,
        )

        for member in members:
            handle = archive.extractfile(member)

            if handle is None:
                raise RuntimeError(
                    "Could not read archive member "
                    f"{member.name!r}."
                )

            raw = json.load(handle)

            if not isinstance(raw, Mapping):
                raise ValueError(
                    "Stage 11 archive record must be "
                    "a mapping."
                )

            threshold = _fraction_from_number(
                raw["fidelity_threshold"]
            )

            if threshold != fidelity_threshold:
                raise ValueError(
                    "Stage 11 control threshold mismatch."
                )

            mask_record = raw.get("mask")
            metric_record = raw.get("metrics")

            if not isinstance(mask_record, Mapping):
                raise ValueError(
                    "Stage 11 mask record is missing."
                )

            if not isinstance(metric_record, Mapping):
                raise ValueError(
                    "Stage 11 metric record is missing."
                )

            mask = ComponentMask.from_record(
                mask_record
            )
            metrics = _metrics_from_record(
                metric_record
            )

            if (
                metrics.retained_component_count
                != mask.retained_component_count
            ):
                raise ValueError(
                    "Stage 11 mask and metric retained "
                    "counts disagree."
                )

            records.append(
                Stage11RandomMaskControlRecord(
                    archive_member=member.name,
                    mask_index=int(raw["mask_index"]),
                    fidelity_threshold=threshold,
                    mask=mask,
                    metrics=metrics,
                    passes_candidate_threshold=bool(
                        raw[
                            "passes_candidate_threshold"
                        ]
                    ),
                )
            )

    indices = [
        record.mask_index
        for record in records
    ]

    if len(indices) != len(set(indices)):
        raise ValueError(
            "Stage 11 mask indices are not unique."
        )

    return tuple(
        sorted(
            records,
            key=lambda record: record.mask_index,
        )
    )


def stage11_random_mask_control_result(
    records: Sequence[Stage11RandomMaskControlRecord],
    *,
    expected_count: int = 100,
    expected_retained_count: int = 146,
) -> NegativeControlResult:
    """Validate the matched-size Stage 11 random-mask control."""

    values = tuple(records)

    if len(values) != expected_count:
        raise ValueError(
            "Unexpected Stage 11 random-mask count."
        )

    thresholds = {
        record.fidelity_threshold
        for record in values
    }

    if len(thresholds) != 1:
        raise ValueError(
            "Stage 11 records contain mixed thresholds."
        )

    retained_counts = {
        record.mask.retained_component_count
        for record in values
    }

    if retained_counts != {expected_retained_count}:
        raise ValueError(
            "Stage 11 records do not match the C1 size."
        )

    qualifying = sum(
        record.passes_candidate_threshold
        for record in values
    )
    threshold = next(iter(thresholds))

    return NegativeControlResult(
        control_name=STAGE11_RANDOM_MASK_CONTROL,
        control_scope=(
            "committed_stage11_primary_threshold_masks"
        ),
        expected_outcome=EXPECTED_REJECTION,
        observed_outcome=(
            "rejection"
            if qualifying == 0
            else "unexpected_qualification"
        ),
        validation_passed=(qualifying == 0),
        record_count=len(values),
        qualifying_count=qualifying,
        retained_component_count=(
            expected_retained_count
        ),
        fidelity_threshold=float(threshold),
        details={
            "mask_indices": [
                record.mask_index
                for record in values
            ],
            "minimum_primary_fidelity": min(
                record.metrics.primary_fidelity
                for record in values
            ),
            "maximum_primary_fidelity": max(
                record.metrics.primary_fidelity
                for record in values
            ),
        },
    )


def select_degraded_c1_candidate(
    result: SparseSearchResult,
) -> DegradedC1Selection:
    """Select the lowest-index rejected terminal C1 deletion."""

    if not isinstance(result, SparseSearchResult):
        raise TypeError(
            "result must be a SparseSearchResult."
        )

    final_ids = set(
        result.final_mask.retained_component_ids
    )
    expected_count = (
        result.final_mask.retained_component_count
        - 1
    )
    candidates: list[CandidateEvaluation] = []

    for evaluation in result.candidate_evaluations:
        candidate_ids = set(
            evaluation.candidate_mask
            .retained_component_ids
        )

        if (
            evaluation.accepted
            or evaluation.passed_threshold
            or evaluation.candidate_component
            not in final_ids
            or evaluation.candidate_mask
            .retained_component_count
            != expected_count
            or candidate_ids
            != final_ids
            - {evaluation.candidate_component}
        ):
            continue

        candidates.append(evaluation)

    if not candidates:
        raise ValueError(
            "No rejected terminal single-deletion "
            "candidate was found."
        )

    selected = min(
        candidates,
        key=lambda evaluation: (
            evaluation.component_index,
            evaluation.candidate_component,
        ),
    )

    return DegradedC1Selection(
        candidate_component=(
            selected.candidate_component
        ),
        component_index=selected.component_index,
        candidate_mask=selected.candidate_mask,
        metrics=selected.metrics,
        exact_fidelity=selected.exact_fidelity,
        fidelity_threshold=result.fidelity_threshold,
        rejection_reason=selected.rejection_reason,
    )


def degraded_c1_control_result(
    result: SparseSearchResult,
) -> NegativeControlResult:
    """Build the degraded-C1 negative-control conclusion."""

    selected = select_degraded_c1_candidate(
        result
    )
    rejected = (
        selected.exact_fidelity
        < selected.fidelity_threshold
    )

    return NegativeControlResult(
        control_name=DEGRADED_C1_CONTROL,
        control_scope=(
            "lowest_stable_component_index_among_"
            "tested_rejected_terminal_deletions"
        ),
        expected_outcome=EXPECTED_REJECTION,
        observed_outcome=(
            "rejection"
            if rejected
            else "unexpected_qualification"
        ),
        validation_passed=rejected,
        selected_component=(
            selected.candidate_component
        ),
        selected_component_index=(
            selected.component_index
        ),
        mask_id=selected.candidate_mask.mask_id,
        retained_component_count=(
            selected.candidate_mask
            .retained_component_count
        ),
        primary_fidelity=selected.exact_fidelity,
        fidelity_threshold=(
            selected.fidelity_threshold
        ),
        details={
            "rejection_reason": (
                selected.rejection_reason
            ),
        },
    )


def shuffled_ranking(
    ranking: RankingResult,
    *,
    integer_seed: int,
) -> RankingResult:
    """Return a deterministic permutation of one ranking."""

    if not isinstance(ranking, RankingResult):
        raise TypeError(
            "ranking must be a RankingResult."
        )

    if (
        isinstance(integer_seed, bool)
        or not isinstance(integer_seed, int)
        or integer_seed < 0
    ):
        raise ValueError(
            "integer_seed must be a non-negative integer."
        )

    generator = np.random.Generator(
        np.random.PCG64(integer_seed)
    )
    count = len(ranking.ranked_components)
    order = generator.permutation(count)

    permuted: list[ComponentRanking] = []

    for new_position, old_position in enumerate(
        order,
        start=1,
    ):
        original = ranking.ranked_components[
            int(old_position)
        ]
        permuted.append(
            replace(
                original,
                ranking_position=new_position,
            )
        )

    return replace(
        ranking,
        ranked_components=tuple(permuted),
        score_definition=(
            "deterministic shuffled-ranking "
            "negative control"
        ),
    )


def shuffled_ranking_control_result(
    original: RankingResult,
    shuffled: RankingResult,
    *,
    integer_seed: int,
) -> NegativeControlResult:
    """Validate deterministic ranking perturbation."""

    original_ids = [
        value.component_identifier
        for value in original.ranked_components
    ]
    shuffled_ids = [
        value.component_identifier
        for value in shuffled.ranked_components
    ]

    same_population = (
        sorted(original_ids)
        == sorted(shuffled_ids)
    )
    changed_order = (
        len(original_ids) <= 1
        or original_ids != shuffled_ids
    )
    valid_positions = [
        value.ranking_position
        for value in shuffled.ranked_components
    ] == list(
        range(1, len(shuffled_ids) + 1)
    )
    passed = (
        same_population
        and changed_order
        and valid_positions
    )

    return NegativeControlResult(
        control_name=SHUFFLED_RANKING_CONTROL,
        control_scope="method_validation_only",
        expected_outcome=(
            EXPECTED_DETERMINISTIC_PERTURBATION
        ),
        observed_outcome=(
            "deterministic_ranking_perturbation"
            if passed
            else "invalid_ranking_perturbation"
        ),
        validation_passed=passed,
        record_count=len(shuffled_ids),
        seed_integer=integer_seed,
        bit_generator="numpy.random.PCG64",
        details={
            "same_component_population": (
                same_population
            ),
            "ordering_changed": changed_order,
            "ranking_positions_valid": (
                valid_positions
            ),
        },
    )


def fidelity_impossible_control_result(
    *,
    mask: ComponentMask,
    metrics: MaskEvaluationMetrics,
    fidelity_threshold: float,
) -> NegativeControlResult:
    """Validate a construction that cannot pass fidelity."""

    rejected = (
        metrics.primary_fidelity
        < fidelity_threshold
    )

    return NegativeControlResult(
        control_name=FIDELITY_IMPOSSIBLE_CONTROL,
        control_scope=(
            "deterministic_all_components_hard_"
            "excluded_or_equivalent"
        ),
        expected_outcome=EXPECTED_REJECTION,
        observed_outcome=(
            "rejection"
            if rejected
            else "unexpected_qualification"
        ),
        validation_passed=rejected,
        mask_id=mask.mask_id,
        retained_component_count=(
            mask.retained_component_count
        ),
        primary_fidelity=(
            metrics.primary_fidelity
        ),
        fidelity_threshold=fidelity_threshold,
        details={
            "construction_verified": (
                mask.retained_component_count == 0
            ),
        },
    )


def distinctness_impossible_control_result(
    *,
    accepted_mask: ComponentMask,
    candidate_mask: ComponentMask,
    distinctness_cutoff: Fraction,
) -> NegativeControlResult:
    """Validate a fixture that cannot satisfy distinctness."""

    if not isinstance(distinctness_cutoff, Fraction):
        raise TypeError(
            "distinctness_cutoff must be a Fraction."
        )

    overlap = jaccard_fraction(
        accepted_mask,
        candidate_mask,
    )
    rejected = overlap > distinctness_cutoff

    return NegativeControlResult(
        control_name=DISTINCTNESS_IMPOSSIBLE_CONTROL,
        control_scope=(
            "deterministic_fixture_all_fidelity_valid_"
            "candidates_violate_cutoff"
        ),
        expected_outcome=EXPECTED_REJECTION,
        observed_outcome=(
            "rejection"
            if rejected
            else "unexpected_qualification"
        ),
        validation_passed=rejected,
        mask_id=candidate_mask.mask_id,
        retained_component_count=(
            candidate_mask.retained_component_count
        ),
        jaccard_overlap=float(overlap),
        distinctness_cutoff=float(
            distinctness_cutoff
        ),
        details={
            "jaccard_numerator": overlap.numerator,
            "jaccard_denominator": overlap.denominator,
        },
    )


def negative_control_rows(
    *,
    stage12_run_id: str,
    results: Sequence[NegativeControlResult],
) -> list[dict[str, Any]]:
    """Return stable rows for all Stage 12 controls."""

    if not stage12_run_id:
        raise ValueError(
            "stage12_run_id must not be empty."
        )

    values = tuple(results)

    if not values:
        raise ValueError(
            "results must not be empty."
        )

    names = [
        result.control_name
        for result in values
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            "Control names must be unique."
        )

    rows: list[dict[str, Any]] = []

    for result in sorted(
        values,
        key=lambda value: value.control_name,
    ):
        rows.append(
            {
                "stage12_run_id": stage12_run_id,
                "control_name": result.control_name,
                "control_scope": result.control_scope,
                "expected_outcome": (
                    result.expected_outcome
                ),
                "observed_outcome": (
                    result.observed_outcome
                ),
                "validation_passed": (
                    result.validation_passed
                ),
                "scientific_family_result": False,
                "record_count": result.record_count,
                "qualifying_count": (
                    result.qualifying_count
                ),
                "selected_component": (
                    result.selected_component
                ),
                "selected_component_index": (
                    result.selected_component_index
                ),
                "mask_id": result.mask_id,
                "retained_component_count": (
                    result.retained_component_count
                ),
                "primary_fidelity": (
                    result.primary_fidelity
                ),
                "fidelity_threshold": (
                    result.fidelity_threshold
                ),
                "jaccard_overlap": (
                    result.jaccard_overlap
                ),
                "distinctness_cutoff": (
                    result.distinctness_cutoff
                ),
                "seed_integer": (
                    result.seed_integer
                ),
                "bit_generator": (
                    result.bit_generator
                ),
                "details_json": json.dumps(
                    (
                        {}
                        if result.details is None
                        else dict(result.details)
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
            }
        )

    return rows


def write_negative_control_table(
    path: str | Path,
    *,
    stage12_run_id: str,
    results: Sequence[NegativeControlResult],
) -> NegativeControlArtifacts:
    """Write the deterministic negative-control table."""

    rows = negative_control_rows(
        stage12_run_id=stage12_run_id,
        results=results,
    )
    output = write_csv_records(
        path,
        fieldnames=NEGATIVE_CONTROL_COLUMNS,
        rows=rows,
    )

    return NegativeControlArtifacts(
        table_path=output,
        table_sha256=file_sha256(output),
        row_count=len(rows),
    )
