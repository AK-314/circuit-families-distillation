"""Seed-level paired summaries for the frozen Stage 20 analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING, Literal

from circuit_families.analysis.stage19_matched_comparisons import CHECKPOINT_STEPS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


MetricStatus = Literal["defined", "undefined"]
METRICS = (
    "structural_family_size",
    "transfer_distinct_group_count",
    "median_pairwise_overlap",
    "median_circuit_size",
    "mean_transfer_fidelity",
    "matched_fidelity_structural_diversity",
    "matched_sparsity_fidelity",
)
CHECKPOINT_METRICS = METRICS[:5]
MATCHED_METRICS = METRICS[5:]
METRIC_LABELS = {
    "structural_family_size": "Recovered structural family size",
    "transfer_distinct_group_count": "Transfer-distinct group count",
    "median_pairwise_overlap": "Median pairwise overlap",
    "median_circuit_size": "Median circuit size (components)",
    "mean_transfer_fidelity": "Mean transfer fidelity",
    "matched_fidelity_structural_diversity": "Matched-fidelity structural diversity Δ",
    "matched_sparsity_fidelity": "Matched-sparsity fidelity Δ",
}


@dataclass(frozen=True)
class GridComparison:
    comparison_type: str
    comparison_label: str
    left_step: int
    right_step: int


def fixed_comparison_registry() -> tuple[GridComparison, ...]:
    first, *middle, last = CHECKPOINT_STEPS
    rows = [GridComparison("pre_to_post", "pre_to_post", first, last)]
    rows.extend(
        GridComparison("pre_to_transition", f"pre_to_grid_{index}", first, step)
        for index, step in enumerate(middle, start=2)
    )
    rows.extend(
        GridComparison("transition_to_post", f"grid_{index}_to_post", step, last)
        for index, step in enumerate(middle, start=2)
    )
    rows.extend(
        GridComparison(
            "adjacent_landmarks",
            f"adjacent_grid_{index}_to_{index + 1}",
            left,
            right,
        )
        for index, (left, right) in enumerate(
            zip(CHECKPOINT_STEPS[:-1], CHECKPOINT_STEPS[1:], strict=True), start=1
        )
    )
    return tuple(rows)


def phase_label(step: int, *, first_ten_percent_step: int, stable_post_step: int) -> str:
    if step < first_ten_percent_step:
        return "delayed_pre_generalisation"
    if step < stable_post_step:
        return "transition"
    return "stable_post"


def _decimal(value: object) -> Decimal | None:
    if value in (None, ""):
        return None
    return Decimal(str(value))


def _primary(row: Mapping[str, object]) -> bool:
    explicit = row.get("primary_cell")
    if explicit not in (None, ""):
        return str(explicit).lower() == "true"
    return (
        int(str(row["fidelity_numerator"])) == 99
        and int(str(row["fidelity_denominator"])) == 100
        and int(str(row["distinctness_numerator"])) == 1
        and int(str(row["distinctness_denominator"])) == 2
    )


def build_checkpoint_metrics(
    *,
    family_rows: Sequence[Mapping[str, object]],
    circuit_size_rows: Sequence[Mapping[str, object]],
    transfer_profile_rows: Sequence[Mapping[str, object]],
    training_runs: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    families = {
        (int(str(row["model_seed"])), int(str(row["checkpoint_step"]))): row
        for row in family_rows
        if _primary(row)
    }
    sizes = {
        (int(str(row["model_seed"])), int(str(row["checkpoint_step"]))): row
        for row in circuit_size_rows
        if (
            int(str(row["fidelity_numerator"])) == 99
            and int(str(row["fidelity_denominator"])) == 100
            and int(str(row["distinctness_numerator"])) == 1
            and int(str(row["distinctness_denominator"])) == 2
        )
    }
    primary_cells = {key: str(row["cell_id"]) for key, row in families.items()}
    transfer_values: dict[str, list[Decimal]] = {}
    for row in transfer_profile_rows:
        cell_id = str(row["cell_id"])
        if cell_id not in primary_cells.values():
            continue
        for name in ("q1_fidelity", "q2_fidelity", "q3_fidelity", "q4_fidelity"):
            transfer_values.setdefault(cell_id, []).append(Decimal(str(row[name])))
    training = {int(str(row["model_seed"])): row for row in training_runs}
    expected = {(seed, step) for seed in range(5) for step in CHECKPOINT_STEPS}
    if set(families) != expected or set(sizes) != expected or set(training) != set(range(5)):
        raise ValueError("Stage 20 requires five seeds across all seven primary checkpoints.")

    rows = []
    for seed in range(5):
        run = training[seed]
        first_ten = int(str(run["first_ten_percent_test_step"]))
        stable_post = int(str(run["stable_post_step"]))
        for checkpoint_index, step in enumerate(CHECKPOINT_STEPS, start=1):
            family = families[(seed, step)]
            size = sizes[(seed, step)]
            family_size = int(str(family["family_size"]))
            cell_id = str(family["cell_id"])
            transfer = transfer_values.get(cell_id, [])
            rows.append(
                {
                    "model_seed": seed,
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_step": step,
                    "phase_label": phase_label(
                        step,
                        first_ten_percent_step=first_ten,
                        stable_post_step=stable_post,
                    ),
                    "first_ten_percent_test_step": first_ten,
                    "stable_post_step": stable_post,
                    "cell_id": cell_id,
                    "right_censored": str(family["right_censored"]).lower() == "true",
                    "structural_family_size": Decimal(family_size),
                    "transfer_distinct_group_count": _decimal(family["transfer_group_count"]),
                    "median_pairwise_overlap": _decimal(family["median_pairwise_overlap"]),
                    "median_circuit_size": _decimal(size["median_retained_components"]),
                    "mean_transfer_fidelity": mean(transfer) if transfer else None,
                }
            )
    return tuple(rows)


def _alignment(comparison_type: str, left_phase: str, right_phase: str) -> str:
    expected = {
        "pre_to_post": ("delayed_pre_generalisation", "stable_post"),
        "pre_to_transition": ("delayed_pre_generalisation", "transition"),
        "transition_to_post": ("transition", "stable_post"),
    }.get(comparison_type)
    if expected is None:
        return "not_applicable_adjacent"
    return "phase_aligned" if (left_phase, right_phase) == expected else "phase_misaligned"


def _matched_lookup(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int, int], Mapping[str, object]]:
    return {
        (
            int(str(row["model_seed"])),
            int(str(row["left_checkpoint_step"])),
            int(str(row["right_checkpoint_step"])),
        ): row
        for row in rows
    }


def build_paired_deltas(
    *,
    checkpoint_metrics: Sequence[Mapping[str, object]],
    matched_fidelity_rows: Sequence[Mapping[str, object]],
    matched_sparsity_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    metrics = {
        (int(str(row["model_seed"])), int(str(row["checkpoint_step"]))): row
        for row in checkpoint_metrics
    }
    fidelity = _matched_lookup(matched_fidelity_rows)
    sparsity = _matched_lookup(matched_sparsity_rows)
    rows = []
    for seed in range(5):
        for comparison in fixed_comparison_registry():
            left = metrics[(seed, comparison.left_step)]
            right = metrics[(seed, comparison.right_step)]
            alignment = _alignment(
                comparison.comparison_type,
                str(left["phase_label"]),
                str(right["phase_label"]),
            )
            for metric in METRICS:
                if metric == "matched_fidelity_structural_diversity":
                    source = fidelity[(seed, comparison.left_step, comparison.right_step)]
                    value = _decimal(source["structural_diversity_change_right_minus_left"])
                    left_value = _decimal(source["left_matched_structural_diversity"])
                    right_value = _decimal(source["right_matched_structural_diversity"])
                    reason = (
                        "matched_fidelity_requires_two_matched_circuits_per_condition"
                        if value is None
                        else ""
                    )
                elif metric == "matched_sparsity_fidelity":
                    source = sparsity[(seed, comparison.left_step, comparison.right_step)]
                    value = _decimal(source["fidelity_change_right_minus_left"])
                    left_value = _decimal(source["left_matched_median_fidelity"])
                    right_value = _decimal(source["right_matched_median_fidelity"])
                    reason = "matched_sparsity_pair_unavailable" if value is None else ""
                else:
                    left_value = _decimal(left[metric])
                    right_value = _decimal(right[metric])
                    value = (
                        None
                        if left_value is None or right_value is None
                        else right_value - left_value
                    )
                    reason = "metric_undefined_for_one_or_both_cells" if value is None else ""
                rows.append(
                    {
                        "comparison_type": comparison.comparison_type,
                        "comparison_label": comparison.comparison_label,
                        "model_seed": seed,
                        "independent_unit": "trained_model_seed",
                        "left_checkpoint_step": comparison.left_step,
                        "right_checkpoint_step": comparison.right_step,
                        "left_phase_label": left["phase_label"],
                        "right_phase_label": right["phase_label"],
                        "phase_alignment_status": alignment,
                        "metric": metric,
                        "left_value": left_value,
                        "right_value": right_value,
                        "source_defined": left_value is not None,
                        "target_defined": right_value is not None,
                        "paired_change_right_minus_left": value,
                        "metric_status": "defined" if value is not None else "undefined",
                        "undefined_reason": reason,
                        "inclusion_status": (
                            "included_in_metric_summary"
                            if value is not None
                            else "excluded_from_metric_summary_undefined"
                        ),
                        "exclusion_reason": reason,
                        "right_censoring_involved": bool(
                            left["right_censored"] or right["right_censored"]
                        ),
                        "direction": (
                            "undefined"
                            if value is None
                            else "positive"
                            if value > 0
                            else "negative"
                            if value < 0
                            else "zero"
                        ),
                    }
                )
    return tuple(rows)


def exact_two_sided_sign_probability(positive: int, negative: int) -> Decimal | None:
    nonzero = positive + negative
    if nonzero == 0:
        return None
    tail = sum(math.comb(nonzero, index) for index in range(min(positive, negative) + 1))
    return min(Decimal(1), Decimal(2 * tail) / Decimal(2**nonzero))


def build_seed_level_summaries(
    paired_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str, int, int], list[Mapping[str, object]]] = {}
    for row in paired_rows:
        key = (
            str(row["comparison_type"]),
            str(row["comparison_label"]),
            str(row["metric"]),
            int(str(row["left_checkpoint_step"])),
            int(str(row["right_checkpoint_step"])),
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        values = [
            Decimal(str(row["paired_change_right_minus_left"]))
            for row in group
            if row["metric_status"] == "defined"
        ]
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        zero = sum(value == 0 for value in values)
        nonzero = positive + negative
        summaries.append(
            {
                "comparison_type": key[0],
                "comparison_label": key[1],
                "metric": key[2],
                "left_checkpoint_step": key[3],
                "right_checkpoint_step": key[4],
                "independent_unit": "trained_model_seed",
                "total_seed_count": len(group),
                "defined_seed_count": len(values),
                "undefined_seed_count": len(group) - len(values),
                "positive_count": positive,
                "negative_count": negative,
                "zero_count": zero,
                "sign_consistency": (
                    None if nonzero == 0 else Decimal(max(positive, negative)) / Decimal(nonzero)
                ),
                "median_paired_change": median(values) if values else None,
                "mean_paired_change": mean(values) if values else None,
                "minimum_paired_change": min(values) if values else None,
                "maximum_paired_change": max(values) if values else None,
                "exact_two_sided_sign_probability": exact_two_sided_sign_probability(
                    positive, negative
                ),
                "exact_test_name": "two_sided_exact_sign_test_nonzero_differences",
                "null_hypothesis": "positive_and_negative_directions_are_equiprobable",
                "zero_difference_handling": "reported_but_excluded_from_sign_test",
                "multiple_comparison_adjustment": "none_prespecified_descriptive_summaries",
                "inference_scope": "descriptive_small_sample_seed_level",
                "phase_aligned_seed_count": sum(
                    row["phase_alignment_status"] in ("phase_aligned", "not_applicable_adjacent")
                    for row in group
                ),
                "right_censored_seed_count": sum(
                    bool(row["right_censoring_involved"]) for row in group
                ),
                "seed_values_json": json.dumps(
                    {
                        str(row["model_seed"]): row["paired_change_right_minus_left"]
                        for row in group
                    },
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                ),
                "included_seed_ids_json": json.dumps(
                    sorted(
                        int(str(row["model_seed"]))
                        for row in group
                        if row["metric_status"] == "defined"
                    ),
                    separators=(",", ":"),
                ),
                "excluded_seed_ids_json": json.dumps(
                    sorted(
                        int(str(row["model_seed"]))
                        for row in group
                        if row["metric_status"] == "undefined"
                    ),
                    separators=(",", ":"),
                ),
            }
        )
    return tuple(summaries)


def build_trajectory_source(
    checkpoint_metrics: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Return a five-seed, seven-checkpoint source grid for all frozen metrics."""
    checkpoints = {
        (int(str(row["model_seed"])), int(str(row["checkpoint_step"]))): row
        for row in checkpoint_metrics
    }
    adjacent = {
        (
            int(str(row["model_seed"])),
            int(str(row["right_checkpoint_step"])),
            str(row["metric"]),
        ): row
        for row in paired_rows
        if row["comparison_type"] == "adjacent_landmarks"
    }
    rows = []
    for seed in range(5):
        for step in CHECKPOINT_STEPS:
            checkpoint = checkpoints[(seed, step)]
            for metric in CHECKPOINT_METRICS:
                value = checkpoint[metric]
                rows.append(
                    {
                        "model_seed": seed,
                        "checkpoint_step": step,
                        "metric": metric,
                        "trajectory_value": value,
                        "value_status": "defined" if value not in (None, "") else "undefined",
                        "undefined_reason": (
                            "" if value not in (None, "") else "empty_family_dependent_metric"
                        ),
                        "trajectory_definition": "raw_checkpoint_value",
                        "independent_unit": "trained_model_seed",
                    }
                )
            for metric in MATCHED_METRICS:
                comparison = adjacent.get((seed, step, metric))
                value = None if comparison is None else comparison["paired_change_right_minus_left"]
                rows.append(
                    {
                        "model_seed": seed,
                        "checkpoint_step": step,
                        "metric": metric,
                        "trajectory_value": value,
                        "value_status": "defined" if value not in (None, "") else "undefined",
                        "undefined_reason": (
                            "no_preceding_checkpoint_for_adjacent_change"
                            if comparison is None
                            else str(comparison["undefined_reason"])
                        ),
                        "trajectory_definition": "adjacent_change_right_minus_left",
                        "independent_unit": "trained_model_seed",
                    }
                )
    return tuple(rows)


def _save_figure(figure: object, repository: Path, stem: str) -> tuple[Path, Path]:
    png = repository / f"figures/{stem}.png"
    pdf = repository / f"figures/{stem}.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=220, metadata={"Software": "circuit_families"})
    figure.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    return png, pdf


def generate_seed_trajectory_figures(
    repository: Path,
    *,
    trajectory_rows: Sequence[Mapping[str, object]],
) -> tuple[Path, ...]:
    """Plot every prespecified metric with trained model seed as the replication unit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    outputs: list[Path] = []

    figure, axes = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
    flat_axes = axes.ravel()
    for metric, axis in zip(CHECKPOINT_METRICS, flat_axes, strict=False):
        for seed in range(5):
            rows = sorted(
                (
                    row
                    for row in trajectory_rows
                    if int(str(row["model_seed"])) == seed and row["metric"] == metric
                ),
                key=lambda row: int(str(row["checkpoint_step"])),
            )
            values = [
                math.nan
                if row["trajectory_value"] in (None, "")
                else float(str(row["trajectory_value"]))
                for row in rows
            ]
            axis.plot(
                [CHECKPOINT_STEPS.index(int(str(row["checkpoint_step"]))) for row in rows],
                values,
                marker="o",
                linewidth=1.2,
                label=f"seed {seed}",
            )
        axis.set(
            title=METRIC_LABELS[metric],
            xlabel="Checkpoint step",
            ylabel="Observed value",
            xticks=range(len(CHECKPOINT_STEPS)),
            xticklabels=[str(step) for step in CHECKPOINT_STEPS],
        )
        axis.tick_params(axis="x", rotation=45)
    flat_axes[0].legend(frameon=False, ncol=3)
    flat_axes[-1].axis("off")
    figure.suptitle("Stage 20 raw seed trajectories (each line is one independently trained model)")
    outputs.extend(_save_figure(figure, repository, "stage20_seed_checkpoint_trajectories"))
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for metric, axis in zip(MATCHED_METRICS, axes, strict=True):
        any_defined = False
        for seed in range(5):
            rows = sorted(
                (
                    row
                    for row in trajectory_rows
                    if row["metric"] == metric and int(str(row["model_seed"])) == seed
                ),
                key=lambda row: int(str(row["checkpoint_step"])),
            )
            values = []
            for row in rows:
                value = row["trajectory_value"]
                values.append(math.nan if value in (None, "") else float(str(value)))
                any_defined = any_defined or value not in (None, "")
            axis.plot(
                [CHECKPOINT_STEPS.index(int(str(row["checkpoint_step"]))) for row in rows],
                values,
                marker="o",
                linewidth=1.2,
                label=f"seed {seed}",
            )
        axis.axhline(0, color="0.65", linewidth=0.8)
        axis.set(
            title=METRIC_LABELS[metric],
            xlabel="Right checkpoint of adjacent pair",
            ylabel="Paired change (right − left)",
            xticks=range(len(CHECKPOINT_STEPS)),
            xticklabels=[str(step) for step in CHECKPOINT_STEPS],
        )
        axis.tick_params(axis="x", rotation=45)
        if not any_defined:
            axis.text(
                0.5,
                0.5,
                "No defined adjacent matched estimates",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
    axes[0].legend(frameon=False, ncol=3)
    figure.suptitle(
        "Stage 20 adjacent matched-comparison trajectories "
        "(each line is one independently trained model)"
    )
    outputs.extend(_save_figure(figure, repository, "stage20_matched_adjacent_trajectories"))
    plt.close(figure)
    return tuple(outputs)
