from __future__ import annotations

from decimal import Decimal
from itertools import combinations

from circuit_families.analysis.stage19_matched_comparisons import CHECKPOINT_STEPS
from circuit_families.analysis.stage20_paired_inference import (
    METRICS,
    build_paired_deltas,
    build_seed_level_summaries,
    build_trajectory_source,
    exact_two_sided_sign_probability,
    fixed_comparison_registry,
    generate_seed_trajectory_figures,
    phase_label,
)


def test_fixed_registry_has_all_prespecified_comparison_types() -> None:
    registry = fixed_comparison_registry()
    assert len(registry) == 17
    assert sum(row.comparison_type == "pre_to_post" for row in registry) == 1
    assert sum(row.comparison_type == "pre_to_transition" for row in registry) == 5
    assert sum(row.comparison_type == "transition_to_post" for row in registry) == 5
    assert sum(row.comparison_type == "adjacent_landmarks" for row in registry) == 6


def test_phase_labels_use_each_seed_training_landmarks() -> None:
    assert phase_label(200, first_ten_percent_step=6950, stable_post_step=10400) == (
        "delayed_pre_generalisation"
    )
    assert phase_label(7450, first_ten_percent_step=6950, stable_post_step=10400) == "transition"
    assert phase_label(9050, first_ten_percent_step=6950, stable_post_step=10400) == "transition"
    assert phase_label(10400, first_ten_percent_step=6950, stable_post_step=10400) == "stable_post"


def test_exact_sign_probability_is_small_sample_and_two_sided() -> None:
    assert exact_two_sided_sign_probability(5, 0) == Decimal("0.0625")
    assert exact_two_sided_sign_probability(4, 1) == Decimal("0.375")
    assert exact_two_sided_sign_probability(0, 0) is None


def _checkpoint_metrics() -> list[dict[str, object]]:
    rows = []
    for seed in range(5):
        for index, step in enumerate(CHECKPOINT_STEPS, start=1):
            rows.append(
                {
                    "model_seed": seed,
                    "checkpoint_step": step,
                    "phase_label": (
                        "delayed_pre_generalisation"
                        if step == 200
                        else "transition"
                        if seed == 3 or step != 9050
                        else "stable_post"
                    ),
                    "right_censored": False,
                    "structural_family_size": Decimal(index),
                    "transfer_distinct_group_count": None,
                    "median_pairwise_overlap": None,
                    "median_circuit_size": None,
                    "mean_transfer_fidelity": None,
                }
            )
    return rows


def _matched_rows(method: str) -> list[dict[str, object]]:
    rows = []
    for seed in range(5):
        for left, right in combinations(CHECKPOINT_STEPS, 2):
            rows.append(
                {
                    "model_seed": seed,
                    "left_checkpoint_step": left,
                    "right_checkpoint_step": right,
                    "left_matched_structural_diversity": "",
                    "right_matched_structural_diversity": "",
                    "structural_diversity_change_right_minus_left": "",
                    "left_matched_median_fidelity": "" if method == "fidelity" else "0.99",
                    "right_matched_median_fidelity": "" if method == "fidelity" else "0.995",
                    "fidelity_change_right_minus_left": "" if method == "fidelity" else "0.005",
                }
            )
    return rows


def test_paired_rows_keep_undefined_metrics_and_phase_misalignment_explicit() -> None:
    paired = build_paired_deltas(
        checkpoint_metrics=_checkpoint_metrics(),
        matched_fidelity_rows=_matched_rows("fidelity"),
        matched_sparsity_rows=_matched_rows("sparsity"),
    )
    assert len(paired) == 5 * 17 * len(METRICS)
    seed_three = next(
        row
        for row in paired
        if row["model_seed"] == 3
        and row["comparison_type"] == "pre_to_post"
        and row["metric"] == "structural_family_size"
    )
    assert seed_three["phase_alignment_status"] == "phase_misaligned"
    assert seed_three["paired_change_right_minus_left"] == Decimal(6)
    undefined = next(
        row for row in paired if row["metric"] == "matched_fidelity_structural_diversity"
    )
    assert undefined["metric_status"] == "undefined"
    summaries = build_seed_level_summaries(paired)
    family = next(
        row
        for row in summaries
        if row["comparison_type"] == "pre_to_post" and row["metric"] == "structural_family_size"
    )
    assert family["defined_seed_count"] == 5
    assert family["positive_count"] == 5
    assert family["exact_two_sided_sign_probability"] == Decimal("0.0625")


def test_seed_trajectory_figures_cover_checkpoint_and_matched_metrics(tmp_path) -> None:
    checkpoint_metrics = _checkpoint_metrics()
    paired = build_paired_deltas(
        checkpoint_metrics=checkpoint_metrics,
        matched_fidelity_rows=_matched_rows("fidelity"),
        matched_sparsity_rows=_matched_rows("sparsity"),
    )
    outputs = generate_seed_trajectory_figures(
        tmp_path,
        trajectory_rows=build_trajectory_source(checkpoint_metrics, paired),
    )
    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert len(outputs) == 4
    assert all(path.stat().st_size > 0 for path in outputs)


def test_trajectory_source_has_all_metrics_seeds_and_checkpoints() -> None:
    checkpoint_metrics = _checkpoint_metrics()
    paired = build_paired_deltas(
        checkpoint_metrics=checkpoint_metrics,
        matched_fidelity_rows=_matched_rows("fidelity"),
        matched_sparsity_rows=_matched_rows("sparsity"),
    )
    source = build_trajectory_source(checkpoint_metrics, paired)
    assert len(source) == 5 * len(CHECKPOINT_STEPS) * len(METRICS)
    first_matched = next(
        row
        for row in source
        if row["model_seed"] == 0
        and row["checkpoint_step"] == CHECKPOINT_STEPS[0]
        and row["metric"] == "matched_fidelity_structural_diversity"
    )
    assert first_matched["value_status"] == "undefined"
    assert first_matched["undefined_reason"] == "no_preceding_checkpoint_for_adjacent_change"
