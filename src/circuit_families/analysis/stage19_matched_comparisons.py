"""Deterministic Stage 19 matching, Pareto, and degenerate-cell analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


CHECKPOINT_STEPS = (200, 3400, 7450, 8150, 8500, 8650, 9050)
PRIMARY_FIDELITY = (99, 100)
PRIMARY_DISTINCTNESS = (1, 2)
FIDELITY_TOLERANCE = Decimal("0.01")
SPARSITY_TOLERANCE = Decimal("5")
MatchMethod = Literal["matched_fidelity", "matched_sparsity"]


@dataclass(frozen=True)
class Circuit:
    cell_id: str
    circuit_id: str
    mask_sha256: str
    model_seed: int
    checkpoint_index: int
    checkpoint_step: int
    retained_components: int
    fidelity: Decimal

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.cell_id, self.circuit_id, self.mask_sha256)


@dataclass(frozen=True)
class Match:
    left: Circuit
    right: Circuit
    absolute_difference: Decimal


@dataclass(frozen=True)
class MatchingResult:
    matches: tuple[Match, ...]
    unmatched_left: tuple[Circuit, ...]
    unmatched_right: tuple[Circuit, ...]

    @property
    def total_absolute_difference(self) -> Decimal:
        return sum((match.absolute_difference for match in self.matches), Decimal(0))


@dataclass(frozen=True)
class Stage19Tables:
    input_registry: tuple[dict[str, object], ...]
    matched_fidelity_pairs: tuple[dict[str, object], ...]
    matched_fidelity_summary: tuple[dict[str, object], ...]
    matched_sparsity_pairs: tuple[dict[str, object], ...]
    matched_sparsity_summary: tuple[dict[str, object], ...]
    unmatched_circuits: tuple[dict[str, object], ...]
    pareto_frontiers: tuple[dict[str, object], ...]
    empty_cells: tuple[dict[str, object], ...]
    excluded_comparisons: tuple[dict[str, object], ...]
    comparison_sources: tuple[dict[str, object], ...]


def read_csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _boolean(value: object) -> bool:
    return str(value).strip().lower() == "true"


def circuits_from_rows(rows: Iterable[Mapping[str, object]]) -> tuple[Circuit, ...]:
    circuits = []
    for row in rows:
        circuits.append(
            Circuit(
                cell_id=str(row["cell_id"]),
                circuit_id=str(row["circuit_id"]),
                mask_sha256=str(row["mask_sha256"]),
                model_seed=int(str(row["model_seed"])),
                checkpoint_index=int(str(row["checkpoint_index"])),
                checkpoint_step=int(str(row["checkpoint_step"])),
                retained_components=int(str(row["retained_components"])),
                fidelity=Decimal(str(row["fidelity"])),
            )
        )
    return tuple(sorted(circuits, key=lambda circuit: circuit.identity))


def _candidate_key(
    candidate: tuple[int, Decimal, tuple[tuple[int, int], ...]],
) -> tuple[int, Decimal, tuple[tuple[int, int], ...]]:
    count, cost, pairs = candidate
    return (-count, cost, pairs)


def optimal_matching(
    left: Sequence[Circuit],
    right: Sequence[Circuit],
    *,
    value: Callable[[Circuit], Decimal],
    tolerance: Decimal,
) -> MatchingResult:
    """Maximise match count, then minimise total absolute difference."""
    ordered_left = tuple(sorted(left, key=lambda circuit: (value(circuit), circuit.identity)))
    ordered_right = tuple(sorted(right, key=lambda circuit: (value(circuit), circuit.identity)))

    @cache
    def solve(i: int, j: int) -> tuple[int, Decimal, tuple[tuple[int, int], ...]]:
        if i == len(ordered_left) or j == len(ordered_right):
            return (0, Decimal(0), ())
        candidates = [solve(i + 1, j), solve(i, j + 1)]
        difference = abs(value(ordered_left[i]) - value(ordered_right[j]))
        if difference <= tolerance:
            count, cost, pairs = solve(i + 1, j + 1)
            candidates.append((count + 1, cost + difference, ((i, j), *pairs)))
        return min(candidates, key=_candidate_key)

    _, _, indexes = solve(0, 0)
    matched_left = {i for i, _ in indexes}
    matched_right = {j for _, j in indexes}
    matches = tuple(
        Match(
            left=ordered_left[i],
            right=ordered_right[j],
            absolute_difference=abs(value(ordered_left[i]) - value(ordered_right[j])),
        )
        for i, j in indexes
    )
    return MatchingResult(
        matches=matches,
        unmatched_left=tuple(
            circuit for index, circuit in enumerate(ordered_left) if index not in matched_left
        ),
        unmatched_right=tuple(
            circuit for index, circuit in enumerate(ordered_right) if index not in matched_right
        ),
    )


def _primary_cell(row: Mapping[str, object]) -> bool:
    explicit = row.get("primary_cell")
    if explicit not in (None, ""):
        return _boolean(explicit)
    return (
        int(str(row["fidelity_numerator"])),
        int(str(row["fidelity_denominator"])),
    ) == PRIMARY_FIDELITY and (
        int(str(row["distinctness_numerator"])),
        int(str(row["distinctness_denominator"])),
    ) == PRIMARY_DISTINCTNESS


def _comparison_status(left_size: int, right_size: int, matched_count: int) -> str:
    if left_size == 0 and right_size == 0:
        return "both_families_empty"
    if left_size == 0:
        return "left_family_empty"
    if right_size == 0:
        return "right_family_empty"
    if matched_count == 0:
        return "no_valid_match_within_tolerance"
    return "matched"


def _median(values: Iterable[Decimal]) -> Decimal | None:
    collected = tuple(values)
    return median(collected) if collected else None


def _median_overlap(
    cell_id: str,
    circuits: Sequence[Circuit],
    overlap: Mapping[tuple[str, str, str], Decimal],
) -> Decimal | None:
    values = []
    for first, second in combinations(circuits, 2):
        key = (cell_id, *sorted((first.circuit_id, second.circuit_id)))
        if key not in overlap:
            raise ValueError(f"Missing Stage 18 pairwise overlap for {key!r}.")
        values.append(overlap[key])
    return _median(values)


def _summary_row(
    *,
    stage18_run_id: str,
    method: MatchMethod,
    seed: int,
    left_step: int,
    right_step: int,
    left_family: Sequence[Circuit],
    right_family: Sequence[Circuit],
    left_summary: Mapping[str, object],
    right_summary: Mapping[str, object],
    result: MatchingResult,
    overlap: Mapping[tuple[str, str, str], Decimal],
) -> dict[str, object]:
    matched_left = tuple(match.left for match in result.matches)
    matched_right = tuple(match.right for match in result.matches)
    left_overlap = _median_overlap(str(left_summary["cell_id"]), matched_left, overlap)
    right_overlap = _median_overlap(str(right_summary["cell_id"]), matched_right, overlap)
    left_diversity = None if left_overlap is None else Decimal(1) - left_overlap
    right_diversity = None if right_overlap is None else Decimal(1) - right_overlap
    left_fidelity = _median(circuit.fidelity for circuit in matched_left)
    right_fidelity = _median(circuit.fidelity for circuit in matched_right)
    left_size = _median(Decimal(circuit.retained_components) for circuit in matched_left)
    right_size = _median(Decimal(circuit.retained_components) for circuit in matched_right)
    tolerance = FIDELITY_TOLERANCE if method == "matched_fidelity" else SPARSITY_TOLERANCE
    return {
        "stage18_run_id": stage18_run_id,
        "comparison_id": f"s{seed}-step{left_step}-to-step{right_step}",
        "matching_method": method,
        "model_seed": seed,
        "left_checkpoint_step": left_step,
        "right_checkpoint_step": right_step,
        "left_cell_id": left_summary["cell_id"],
        "right_cell_id": right_summary["cell_id"],
        "left_family_size": len(left_family),
        "right_family_size": len(right_family),
        "left_family_right_censored": _boolean(left_summary["right_censored"]),
        "right_family_right_censored": _boolean(right_summary["right_censored"]),
        "matching_tolerance": tolerance,
        "matched_count": len(result.matches),
        "unmatched_left_count": len(result.unmatched_left),
        "unmatched_right_count": len(result.unmatched_right),
        "total_absolute_match_difference": result.total_absolute_difference,
        "comparison_status": _comparison_status(
            len(left_family), len(right_family), len(result.matches)
        ),
        "left_matched_median_fidelity": left_fidelity,
        "right_matched_median_fidelity": right_fidelity,
        "fidelity_change_right_minus_left": (
            None if left_fidelity is None else right_fidelity - left_fidelity
        ),
        "left_matched_median_retained_components": left_size,
        "right_matched_median_retained_components": right_size,
        "retained_component_change_right_minus_left": (
            None if left_size is None else right_size - left_size
        ),
        "left_matched_median_pairwise_overlap": left_overlap,
        "right_matched_median_pairwise_overlap": right_overlap,
        "left_matched_structural_diversity": left_diversity,
        "right_matched_structural_diversity": right_diversity,
        "structural_diversity_change_right_minus_left": (
            None if left_diversity is None else right_diversity - left_diversity
        ),
    }


def _pair_rows(
    stage18_run_id: str,
    method: MatchMethod,
    seed: int,
    left_step: int,
    right_step: int,
    result: MatchingResult,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "stage18_run_id": stage18_run_id,
            "comparison_id": f"s{seed}-step{left_step}-to-step{right_step}",
            "matching_method": method,
            "model_seed": seed,
            "left_checkpoint_step": left_step,
            "right_checkpoint_step": right_step,
            "match_index": index,
            "left_cell_id": match.left.cell_id,
            "left_circuit_id": match.left.circuit_id,
            "left_mask_sha256": match.left.mask_sha256,
            "left_fidelity": match.left.fidelity,
            "left_retained_components": match.left.retained_components,
            "right_cell_id": match.right.cell_id,
            "right_circuit_id": match.right.circuit_id,
            "right_mask_sha256": match.right.mask_sha256,
            "right_fidelity": match.right.fidelity,
            "right_retained_components": match.right.retained_components,
            "absolute_matching_difference": match.absolute_difference,
            "fidelity_change_right_minus_left": match.right.fidelity - match.left.fidelity,
            "retained_component_change_right_minus_left": (
                match.right.retained_components - match.left.retained_components
            ),
        }
        for index, match in enumerate(result.matches, start=1)
    )


def _unmatched_rows(
    stage18_run_id: str,
    method: MatchMethod,
    seed: int,
    left_step: int,
    right_step: int,
    result: MatchingResult,
) -> tuple[dict[str, object], ...]:
    rows = []
    for side, circuits in (("left", result.unmatched_left), ("right", result.unmatched_right)):
        for circuit in circuits:
            rows.append(
                {
                    "stage18_run_id": stage18_run_id,
                    "comparison_id": f"s{seed}-step{left_step}-to-step{right_step}",
                    "matching_method": method,
                    "model_seed": seed,
                    "left_checkpoint_step": left_step,
                    "right_checkpoint_step": right_step,
                    "side": side,
                    "cell_id": circuit.cell_id,
                    "circuit_id": circuit.circuit_id,
                    "mask_sha256": circuit.mask_sha256,
                    "fidelity": circuit.fidelity,
                    "retained_components": circuit.retained_components,
                    "unmatched_reason": "empty_opposing_family"
                    if not result.matches
                    and (not result.unmatched_left or not result.unmatched_right)
                    else "no_available_partner_within_tolerance",
                }
            )
    return tuple(rows)


def _pareto_rows(
    stage18_run_id: str,
    circuits: Sequence[Circuit],
    family_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    family_by_cell = {str(row["cell_id"]): row for row in family_rows}
    grouped: dict[tuple[int, int], list[Circuit]] = {}
    for circuit in circuits:
        grouped.setdefault((circuit.model_seed, circuit.checkpoint_step), []).append(circuit)
    rows = []
    for (seed, step), candidates in sorted(grouped.items()):
        by_mask: dict[str, list[Circuit]] = {}
        for circuit in candidates:
            by_mask.setdefault(circuit.mask_sha256, []).append(circuit)
        unique = []
        for mask, occurrences in sorted(by_mask.items()):
            fidelities = {circuit.fidelity for circuit in occurrences}
            sizes = {circuit.retained_components for circuit in occurrences}
            if len(fidelities) != 1 or len(sizes) != 1:
                raise ValueError(f"Inconsistent repeated mask metrics for {mask}.")
            canonical = min(occurrences, key=lambda circuit: circuit.identity)
            unique.append((canonical, occurrences))
        frontier = []
        for candidate, occurrences in unique:
            dominated = any(
                other.retained_components <= candidate.retained_components
                and other.fidelity >= candidate.fidelity
                and (
                    other.retained_components < candidate.retained_components
                    or other.fidelity > candidate.fidelity
                )
                for other, _ in unique
            )
            if not dominated:
                frontier.append((candidate, occurrences))
        frontier.sort(
            key=lambda item: (item[0].retained_components, -item[0].fidelity, item[0].mask_sha256)
        )
        for index, (circuit, occurrences) in enumerate(frontier, start=1):
            rows.append(
                {
                    "stage18_run_id": stage18_run_id,
                    "model_seed": seed,
                    "checkpoint_step": step,
                    "frontier_index": index,
                    "frontier_size": len(frontier),
                    "unique_candidate_count": len(unique),
                    "mask_sha256": circuit.mask_sha256,
                    "fidelity": circuit.fidelity,
                    "retained_components": circuit.retained_components,
                    "canonical_cell_id": circuit.cell_id,
                    "canonical_circuit_id": circuit.circuit_id,
                    "source_cell_count": len({item.cell_id for item in occurrences}),
                    "source_cell_ids": json.dumps(sorted({item.cell_id for item in occurrences})),
                    "source_any_right_censored": any(
                        _boolean(family_by_cell[item.cell_id]["right_censored"])
                        for item in occurrences
                    ),
                    "fidelity_preference": "higher_is_better",
                    "retained_components_preference": "fewer_is_better",
                    "dominance_rule": (
                        "weakly_higher_fidelity_and_weakly_fewer_components_"
                        "with_at_least_one_strict"
                    ),
                    "duplicate_rule": "deduplicate_by_mask_sha256_then_canonical_identity",
                    "interpolation": False,
                    "family_size_interpretation": "fixed_budget_recovered_lower_bound",
                }
            )
    return tuple(rows)


def build_stage19_tables(
    *,
    stage18_run_id: str,
    circuit_rows: Sequence[Mapping[str, object]],
    family_rows: Sequence[Mapping[str, object]],
    overlap_rows: Sequence[Mapping[str, object]],
) -> Stage19Tables:
    circuits = circuits_from_rows(circuit_rows)
    all_by_cell: dict[str, tuple[Circuit, ...]] = {}
    for cell_id in sorted({circuit.cell_id for circuit in circuits}):
        all_by_cell[cell_id] = tuple(circuit for circuit in circuits if circuit.cell_id == cell_id)
    primary = {
        (int(str(row["model_seed"])), int(str(row["checkpoint_step"]))): row
        for row in family_rows
        if _primary_cell(row)
    }
    expected = {(seed, step) for seed in range(5) for step in CHECKPOINT_STEPS}
    if set(primary) != expected:
        raise ValueError("Stage 19 requires exactly 35 primary Stage 18 cells.")
    overlap = {
        (
            str(row["cell_id"]),
            *sorted((str(row["circuit_i"]), str(row["circuit_j"]))),
        ): Decimal(str(row["jaccard_overlap"]))
        for row in overlap_rows
    }
    input_registry = tuple(
        {
            "stage18_run_id": stage18_run_id,
            "global_cell_index": row["global_cell_index"],
            "cell_id": row["cell_id"],
            "model_seed": row["model_seed"],
            "checkpoint_index": row["checkpoint_index"],
            "checkpoint_step": row["checkpoint_step"],
            "displayed_fidelity": row["displayed_fidelity"],
            "displayed_jaccard_cutoff": row["displayed_jaccard_cutoff"],
            "primary_cell": _primary_cell(row),
            "family_size": row["family_size"],
            "empty_family": int(str(row["family_size"])) == 0,
            "singleton_family": int(str(row["family_size"])) == 1,
            "right_censored": _boolean(row["right_censored"]),
            "search_status": row["status"],
            "stopping_reason": row["stopping_reason"],
            "search_execution_mode": row.get("search_execution_mode", "test_fixture"),
            "analysis_role": (
                "primary_checkpoint_matching_and_sensitivity"
                if _primary_cell(row)
                else "sensitivity_and_pareto"
            ),
            "unavailable_control": False,
            "included_in_stage19_universe": True,
        }
        for row in sorted(family_rows, key=lambda item: int(str(item["global_cell_index"])))
    )

    fidelity_pairs: list[dict[str, object]] = []
    fidelity_summary: list[dict[str, object]] = []
    sparsity_pairs: list[dict[str, object]] = []
    sparsity_summary: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    for seed in range(5):
        for left_step, right_step in combinations(CHECKPOINT_STEPS, 2):
            left_summary = primary[(seed, left_step)]
            right_summary = primary[(seed, right_step)]
            left_family = all_by_cell.get(str(left_summary["cell_id"]), ())
            right_family = all_by_cell.get(str(right_summary["cell_id"]), ())
            if int(str(left_summary["family_size"])) != len(left_family):
                raise ValueError("Left primary family size does not match Stage 18 circuits.")
            if int(str(right_summary["family_size"])) != len(right_family):
                raise ValueError("Right primary family size does not match Stage 18 circuits.")
            for method, value, tolerance, pair_target, summary_target in (
                (
                    "matched_fidelity",
                    lambda circuit: circuit.fidelity,
                    FIDELITY_TOLERANCE,
                    fidelity_pairs,
                    fidelity_summary,
                ),
                (
                    "matched_sparsity",
                    lambda circuit: Decimal(circuit.retained_components),
                    SPARSITY_TOLERANCE,
                    sparsity_pairs,
                    sparsity_summary,
                ),
            ):
                result = optimal_matching(
                    left_family,
                    right_family,
                    value=value,
                    tolerance=tolerance,
                )
                pair_target.extend(
                    _pair_rows(stage18_run_id, method, seed, left_step, right_step, result)
                )
                summary_target.append(
                    _summary_row(
                        stage18_run_id=stage18_run_id,
                        method=method,
                        seed=seed,
                        left_step=left_step,
                        right_step=right_step,
                        left_family=left_family,
                        right_family=right_family,
                        left_summary=left_summary,
                        right_summary=right_summary,
                        result=result,
                        overlap=overlap,
                    )
                )
                unmatched.extend(
                    _unmatched_rows(stage18_run_id, method, seed, left_step, right_step, result)
                )

    empty = tuple(
        {
            "stage18_run_id": stage18_run_id,
            "global_cell_index": row["global_cell_index"],
            "cell_id": row["cell_id"],
            "model_seed": row["model_seed"],
            "checkpoint_index": row["checkpoint_index"],
            "checkpoint_step": row["checkpoint_step"],
            "displayed_fidelity": row["displayed_fidelity"],
            "displayed_jaccard_cutoff": row["displayed_jaccard_cutoff"],
            "primary_cell": _primary_cell(row),
            "status": row["status"],
            "stopping_reason": row["stopping_reason"],
            "family_size": 0,
            "eligible_for_matched_comparison": False,
            "comparison_type": "not_applicable_cell_outcome",
            "source_condition": row["cell_id"],
            "target_condition": "",
            "missing_side": "executed_cell_family",
            "exact_reason": row["stopping_reason"],
            "source_family_empty": True,
            "target_family_empty": "",
            "no_valid_match_exists": "",
            "metric_undefined": True,
            "excluded_from_matched_effect": True,
            "remains_in_primary_family_size_analysis": True,
            "outcome_type": "executed_empty_family",
            "unavailable_control": False,
            "undefined_metrics": (
                "median_circuit_size;median_pairwise_overlap;transfer_group_count;"
                "matched_fidelity;matched_sparsity"
            ),
            "handling": "reported_as_empty_not_imputed",
        }
        for row in family_rows
        if int(str(row["family_size"])) == 0
    )
    all_summaries = tuple(fidelity_summary + sparsity_summary)
    excluded = tuple(
        {
            "stage18_run_id": stage18_run_id,
            "comparison_id": row["comparison_id"],
            "comparison_type": row["matching_method"],
            "model_seed": row["model_seed"],
            "source_checkpoint_step": row["left_checkpoint_step"],
            "target_checkpoint_step": row["right_checkpoint_step"],
            "source_condition": row["left_cell_id"],
            "target_condition": row["right_cell_id"],
            "missing_side": (
                "both"
                if row["comparison_status"] == "both_families_empty"
                else "source"
                if row["comparison_status"] == "left_family_empty"
                else "target"
                if row["comparison_status"] == "right_family_empty"
                else "neither_no_valid_match"
            ),
            "exact_reason": row["comparison_status"],
            "source_family_empty": int(row["left_family_size"]) == 0,
            "target_family_empty": int(row["right_family_size"]) == 0,
            "no_valid_match_exists": row["comparison_status"] == "no_valid_match_within_tolerance",
            "metric_undefined": True,
            "excluded_from_matched_effect": True,
            "remains_in_primary_family_size_analysis": True,
        }
        for row in all_summaries
        if row["comparison_status"] != "matched"
    )
    comparison_sources = tuple(
        {
            "stage18_run_id": stage18_run_id,
            "comparison_id": row["comparison_id"],
            "matching_method": row["matching_method"],
            "model_seed": row["model_seed"],
            "source_checkpoint_step": row["left_checkpoint_step"],
            "target_checkpoint_step": row["right_checkpoint_step"],
            "source_cell_id": row["left_cell_id"],
            "target_cell_id": row["right_cell_id"],
            "source_family_size": row["left_family_size"],
            "target_family_size": row["right_family_size"],
            "comparison_status": row["comparison_status"],
            "matching_direction": "symmetric_joint_pair_selection",
            "matching_unit": "circuit_within_trained_model_seed",
            "independent_inference_unit": "trained_model_seed",
        }
        for row in all_summaries
    )
    return Stage19Tables(
        input_registry=input_registry,
        matched_fidelity_pairs=tuple(fidelity_pairs),
        matched_fidelity_summary=tuple(fidelity_summary),
        matched_sparsity_pairs=tuple(sparsity_pairs),
        matched_sparsity_summary=tuple(sparsity_summary),
        unmatched_circuits=tuple(unmatched),
        pareto_frontiers=_pareto_rows(stage18_run_id, circuits, family_rows),
        empty_cells=empty,
        excluded_comparisons=excluded,
        comparison_sources=comparison_sources,
    )
