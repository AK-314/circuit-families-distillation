"""Saved-table-only generation of the five principal Stage 21 figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from circuit_families.analysis.stage18_scaling import write_csv

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


CHECKPOINT_STEPS = (200, 3400, 7450, 8150, 8500, 8650, 9050)


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def numeric_matrix(
    rows: Sequence[Mapping[str, object]],
    *,
    row_values: Sequence[int],
    column_values: Sequence[int],
    row_key: str,
    column_key: str,
    value_key: str,
) -> np.ndarray:
    lookup = {(int(str(row[row_key])), int(str(row[column_key]))): row[value_key] for row in rows}
    matrix = np.full((len(row_values), len(column_values)), np.nan)
    for row_index, row_value in enumerate(row_values):
        for column_index, column_value in enumerate(column_values):
            value = lookup.get((row_value, column_value))
            if value not in (None, ""):
                matrix[row_index, column_index] = float(value)
    return matrix


def _save(figure: object, repository: Path, stem: str) -> tuple[Path, Path]:
    png = repository / f"figures/{stem}.png"
    pdf = repository / f"figures/{stem}.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=220, metadata={"Software": "circuit_families"})
    figure.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    return png, pdf


def _heatmap(axis: object, matrix: np.ndarray, *, title: str, colorbar_label: str) -> None:
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set(
        title=title,
        xlabel="Checkpoint step",
        ylabel="Model seed",
        xticks=range(len(CHECKPOINT_STEPS)),
        xticklabels=[str(step) for step in CHECKPOINT_STEPS],
        yticks=range(5),
        yticklabels=[str(seed) for seed in range(5)],
    )
    axis.tick_params(axis="x", rotation=45)
    axis.figure.colorbar(image, ax=axis, label=colorbar_label, shrink=0.85)


def _training_source(repository: Path) -> tuple[dict[str, object], ...]:
    manifest = json.loads(
        (repository / "manifests/stage18_training.json").read_text(encoding="utf-8")
    )
    rows = []
    for run in sorted(manifest["runs"], key=lambda item: int(item["model_seed"])):
        metrics = repository / str(run["metrics_path"])
        for line in metrics.read_text(encoding="utf-8").splitlines():
            metric = json.loads(line)
            step = int(metric["training_step"])
            rows.append(
                {
                    "model_seed": int(run["model_seed"]),
                    "training_step": step,
                    "train_accuracy": metric["train_accuracy"],
                    "test_accuracy": metric["test_accuracy"],
                    "analysed_checkpoint": step in CHECKPOINT_STEPS,
                    "aggregation_unit": "trained_model_seed",
                }
            )
    return tuple(rows)


def generate_principal_figures(repository: Path) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    outputs: list[Path] = []

    training = _training_source(repository)
    training_source = write_csv(
        repository / "results/tables/stage21_figure1_training_curves_source.csv", training
    )
    outputs.append(training_source)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharex=True)
    for seed in range(5):
        rows = [row for row in training if int(row["model_seed"]) == seed]
        steps = [int(row["training_step"]) for row in rows]
        axes[0].plot(steps, [float(row["train_accuracy"]) for row in rows], label=f"seed {seed}")
        axes[1].plot(steps, [float(row["test_accuracy"]) for row in rows], label=f"seed {seed}")
    for axis in axes:
        for step in CHECKPOINT_STEPS:
            axis.axvline(step, color="0.75", linewidth=0.6, zorder=0)
        axis.set(xlabel="Training step", ylim=(-0.02, 1.02))
    axes[0].set(title="Training accuracy", ylabel="Accuracy")
    axes[1].set(title="Test accuracy")
    axes[1].legend(ncol=1, frameon=False, loc="lower right")
    figure.suptitle("Figure 1. Genuine-task training trajectories and analysed checkpoints")
    outputs.extend(_save(figure, repository, "stage21_figure1_training_trajectories"))
    plt.close(figure)

    family = _read_csv(repository / "results/tables/stage18_family_summary.csv")
    family_source_rows = tuple(
        {
            "model_seed": row["model_seed"],
            "checkpoint_step": row["checkpoint_step"],
            "displayed_fidelity": row["displayed_fidelity"],
            "displayed_jaccard_cutoff": row["displayed_jaccard_cutoff"],
            "family_size": row["family_size"],
            "right_censored": row["right_censored"],
            "aggregation_unit": "trained_model_seed",
        }
        for row in family
    )
    family_source = write_csv(
        repository / "results/tables/stage21_figure2_family_dynamics_source.csv",
        family_source_rows,
    )
    outputs.append(family_source)
    primary = [
        row
        for row in family_source_rows
        if row["displayed_fidelity"] == "0.990" and row["displayed_jaccard_cutoff"] == "0.50"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for seed in range(5):
        rows = sorted(
            (row for row in primary if int(row["model_seed"]) == seed),
            key=lambda row: int(row["checkpoint_step"]),
        )
        axes[0].plot(
            [CHECKPOINT_STEPS.index(int(row["checkpoint_step"])) for row in rows],
            [int(row["family_size"]) for row in rows],
            marker="o",
            label=f"seed {seed}",
        )
    axes[0].set(
        title="Primary family trajectories",
        xlabel="Checkpoint step",
        ylabel="Recovered family size",
        xticks=range(len(CHECKPOINT_STEPS)),
        xticklabels=[str(step) for step in CHECKPOINT_STEPS],
    )
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(frameon=False)
    fidelity_levels = ("0.800", "0.850", "0.900", "0.950", "0.975", "0.990")
    fidelity_matrix = np.array(
        [
            [
                np.mean(
                    [
                        int(row["family_size"])
                        for row in family_source_rows
                        if row["displayed_fidelity"] == fidelity
                        and row["displayed_jaccard_cutoff"] == "0.50"
                        and int(row["checkpoint_step"]) == step
                    ]
                )
                for step in CHECKPOINT_STEPS
            ]
            for fidelity in fidelity_levels
        ]
    )
    image = axes[1].imshow(fidelity_matrix, aspect="auto", cmap="magma", interpolation="nearest")
    axes[1].set(
        title="Mean family size across fidelity thresholds",
        xlabel="Checkpoint step",
        ylabel="Required fidelity",
        xticks=range(7),
        xticklabels=[str(step) for step in CHECKPOINT_STEPS],
        yticks=range(6),
        yticklabels=fidelity_levels,
    )
    axes[1].tick_params(axis="x", rotation=45)
    figure.colorbar(image, ax=axes[1], label="Mean family size", shrink=0.85)
    figure.suptitle("Figure 2. Circuit-family dynamics and fidelity sensitivity")
    outputs.extend(_save(figure, repository, "stage21_figure2_family_dynamics"))
    plt.close(figure)

    pairwise = _read_csv(repository / "results/tables/stage18_pairwise_overlap.csv")
    final_step = CHECKPOINT_STEPS[-1]
    primary_family = {
        (int(row["model_seed"]), int(row["checkpoint_step"])): row
        for row in family
        if row["displayed_fidelity"] == "0.990" and row["displayed_jaccard_cutoff"] == "0.50"
    }
    structural_source_rows_list = []
    final_pairwise = [
        row
        for row in pairwise
        if row["displayed_fidelity"] == "0.990"
        and row["displayed_jaccard_cutoff"] == "0.50"
        and int(row["checkpoint_step"]) == final_step
    ]
    for seed in range(5):
        family_row = primary_family[(seed, final_step)]
        for member in range(1, int(family_row["family_size"]) + 1):
            circuit = f"C{member}"
            structural_source_rows_list.append(
                {
                    "record_type": "overlap_matrix",
                    "model_seed": seed,
                    "checkpoint_step": final_step,
                    "cell_id": family_row["cell_id"],
                    "circuit_i": circuit,
                    "circuit_j": circuit,
                    "jaccard_overlap": 1.0,
                    "displayed_jaccard_cutoff": "0.50",
                    "family_size": family_row["family_size"],
                    "display_unit": "circuit_pair_within_trained_model",
                    "independent_inference_unit": "trained_model_seed",
                }
            )
    for row in final_pairwise:
        structural_source_rows_list.append(
            {
                "record_type": "overlap_matrix",
                "model_seed": row["model_seed"],
                "checkpoint_step": row["checkpoint_step"],
                "cell_id": row["cell_id"],
                "circuit_i": row["circuit_i"],
                "circuit_j": row["circuit_j"],
                "jaccard_overlap": row["jaccard_overlap"],
                "displayed_jaccard_cutoff": row["displayed_jaccard_cutoff"],
                "family_size": primary_family[
                    (int(row["model_seed"]), int(row["checkpoint_step"]))
                ]["family_size"],
                "display_unit": "circuit_pair_within_trained_model",
                "independent_inference_unit": "trained_model_seed",
            }
        )
    for row in family:
        if row["displayed_fidelity"] != "0.990":
            continue
        structural_source_rows_list.append(
            {
                "record_type": "distinctness_sensitivity",
                "model_seed": row["model_seed"],
                "checkpoint_step": row["checkpoint_step"],
                "cell_id": row["cell_id"],
                "circuit_i": "",
                "circuit_j": "",
                "jaccard_overlap": "",
                "displayed_jaccard_cutoff": row["displayed_jaccard_cutoff"],
                "family_size": row["family_size"],
                "display_unit": "trained_model_seed",
                "independent_inference_unit": "trained_model_seed",
            }
        )
    structural_source_rows = tuple(structural_source_rows_list)
    structural_source = write_csv(
        repository / "results/tables/stage21_figure3_structural_source.csv",
        structural_source_rows,
    )
    outputs.append(structural_source)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    flat_axes = axes.ravel()
    overlap_image = None
    for seed, axis in enumerate(flat_axes[:5]):
        family_row = primary_family[(seed, final_step)]
        family_size = int(family_row["family_size"])
        axis.set_title(f"Seed {seed}, step {final_step} (n={family_size} circuits)")
        if family_size == 0:
            axis.text(0.5, 0.5, "Empty family", transform=axis.transAxes, ha="center", va="center")
            axis.axis("off")
            continue
        matrix = np.eye(family_size)
        rows = [
            row
            for row in structural_source_rows
            if row["record_type"] == "overlap_matrix" and int(row["model_seed"]) == seed
        ]
        for row in rows:
            left = int(str(row["circuit_i"])[1:]) - 1
            right = int(str(row["circuit_j"])[1:]) - 1
            value = float(row["jaccard_overlap"])
            matrix[left, right] = value
            matrix[right, left] = value
        overlap_image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        labels = [f"C{index}" for index in range(1, family_size + 1)]
        axis.set(
            xlabel="Circuit",
            ylabel="Circuit",
            xticks=range(family_size),
            xticklabels=labels,
            yticks=range(family_size),
            yticklabels=labels,
        )
    if overlap_image is not None:
        figure.colorbar(
            overlap_image,
            ax=flat_axes[:5].tolist(),
            label="Pairwise Jaccard overlap",
            shrink=0.65,
        )
    sensitivity_axis = flat_axes[5]
    for cutoff in ("0.25", "0.50", "0.75"):
        values = [
            np.mean(
                [
                    int(row["family_size"])
                    for row in structural_source_rows
                    if row["record_type"] == "distinctness_sensitivity"
                    if row["displayed_jaccard_cutoff"] == cutoff
                    and int(row["checkpoint_step"]) == step
                ]
            )
            for step in CHECKPOINT_STEPS
        ]
        sensitivity_axis.plot(
            range(len(CHECKPOINT_STEPS)), values, marker="o", label=f"cutoff {cutoff}"
        )
    sensitivity_axis.set(
        title="Distinctness sensitivity",
        xlabel="Checkpoint step",
        ylabel="Mean recovered family size",
        xticks=range(len(CHECKPOINT_STEPS)),
        xticklabels=[str(step) for step in CHECKPOINT_STEPS],
    )
    sensitivity_axis.tick_params(axis="x", rotation=45)
    sensitivity_axis.legend(frameon=False)
    figure.suptitle("Figure 3. Circuit-level structural-overlap matrices and sensitivity")
    outputs.extend(_save(figure, repository, "stage21_figure3_structural_sensitivity"))
    plt.close(figure)

    transfer = _read_csv(repository / "results/tables/stage20_seed_checkpoint_metrics.csv")
    transfer_profiles = _read_csv(repository / "results/tables/stage18_transfer_profiles.csv")
    transfer_source_rows_list = []
    for seed in range(5):
        family_row = primary_family[(seed, final_step)]
        for row in transfer_profiles:
            if row["cell_id"] != family_row["cell_id"]:
                continue
            for subset in ("q1", "q2", "q3", "q4"):
                transfer_source_rows_list.append(
                    {
                        "record_type": "functional_transfer_matrix",
                        "model_seed": seed,
                        "checkpoint_step": final_step,
                        "cell_id": family_row["cell_id"],
                        "circuit_id": row["circuit_id"],
                        "test_subset": subset,
                        "transfer_fidelity": row[f"{subset}_fidelity"],
                        "transfer_distinct_group_count": "",
                        "display_unit": "circuit_by_test_subset_within_trained_model",
                        "independent_inference_unit": "trained_model_seed",
                    }
                )
    for row in transfer:
        transfer_source_rows_list.append(
            {
                "record_type": "transfer_group_trajectory",
                "model_seed": row["model_seed"],
                "checkpoint_step": row["checkpoint_step"],
                "cell_id": row["cell_id"],
                "circuit_id": "",
                "test_subset": "",
                "transfer_fidelity": "",
                "transfer_distinct_group_count": row["transfer_distinct_group_count"],
                "display_unit": "trained_model_seed",
                "independent_inference_unit": "trained_model_seed",
            }
        )
    transfer_source_rows = tuple(transfer_source_rows_list)
    transfer_source = write_csv(
        repository / "results/tables/stage21_figure4_transfer_source.csv", transfer_source_rows
    )
    outputs.append(transfer_source)
    matrix_rows = [
        row for row in transfer_source_rows if row["record_type"] == "functional_transfer_matrix"
    ]
    all_transfer_values = [float(row["transfer_fidelity"]) for row in matrix_rows]
    transfer_min = min(all_transfer_values)
    transfer_max = max(all_transfer_values)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    flat_axes = axes.ravel()
    transfer_image = None
    for seed, axis in enumerate(flat_axes[:5]):
        rows = [row for row in matrix_rows if int(row["model_seed"]) == seed]
        group_row = next(
            row
            for row in transfer
            if int(row["model_seed"]) == seed and int(row["checkpoint_step"]) == final_step
        )
        groups = group_row["transfer_distinct_group_count"] or "undefined"
        axis.set_title(f"Seed {seed}, step {final_step} (groups={groups})")
        if not rows:
            axis.text(0.5, 0.5, "Empty family", transform=axis.transAxes, ha="center", va="center")
            axis.axis("off")
            continue
        circuits = sorted({row["circuit_id"] for row in rows}, key=lambda value: int(value[1:]))
        matrix = np.array(
            [
                [
                    float(
                        next(
                            row["transfer_fidelity"]
                            for row in rows
                            if row["circuit_id"] == circuit and row["test_subset"] == subset
                        )
                    )
                    for subset in ("q1", "q2", "q3", "q4")
                ]
                for circuit in circuits
            ]
        )
        transfer_image = axis.imshow(
            matrix,
            vmin=transfer_min,
            vmax=transfer_max,
            cmap="magma",
            aspect="auto",
        )
        axis.set(
            xlabel="Held-out test subset",
            ylabel="Circuit",
            xticks=range(4),
            xticklabels=("Q1", "Q2", "Q3", "Q4"),
            yticks=range(len(circuits)),
            yticklabels=circuits,
        )
    if transfer_image is not None:
        figure.colorbar(
            transfer_image,
            ax=flat_axes[:5].tolist(),
            label="Transfer fidelity",
            shrink=0.65,
        )
    group_axis = flat_axes[5]
    for seed in range(5):
        rows = sorted(
            (
                row
                for row in transfer_source_rows
                if row["record_type"] == "transfer_group_trajectory"
                and int(row["model_seed"]) == seed
            ),
            key=lambda row: int(row["checkpoint_step"]),
        )
        values = [
            np.nan
            if row["transfer_distinct_group_count"] == ""
            else float(row["transfer_distinct_group_count"])
            for row in rows
        ]
        group_axis.plot(range(len(CHECKPOINT_STEPS)), values, marker="o", label=f"seed {seed}")
    group_axis.set(
        title="Transfer-distinct group trajectories",
        xlabel="Checkpoint step",
        ylabel="Group count",
        xticks=range(len(CHECKPOINT_STEPS)),
        xticklabels=[str(step) for step in CHECKPOINT_STEPS],
    )
    group_axis.tick_params(axis="x", rotation=45)
    group_axis.legend(frameon=False, ncol=2)
    figure.suptitle("Figure 4. Circuit-level functional-transfer matrices and groups")
    outputs.extend(_save(figure, repository, "stage21_figure4_transfer_dynamics"))
    plt.close(figure)

    random_rows = _read_csv(
        repository / "results/tables/seed_0_stage14_random_label_family_summary.csv"
    )
    controls = []
    for row in primary:
        controls.append(
            {
                "condition": "genuine_task",
                "model_seed": row["model_seed"],
                "checkpoint_step": row["checkpoint_step"],
                "family_size": row["family_size"],
                "availability": "available",
                "aggregation_unit": "trained_model_seed",
            }
        )
    for row in random_rows:
        controls.append(
            {
                "condition": "random_label",
                "model_seed": 0,
                "checkpoint_step": row["checkpoint_step"],
                "family_size": row["family_size"],
                "availability": "available_descriptive_single_seed",
                "aggregation_unit": "trained_model_seed",
            }
        )
    for step in CHECKPOINT_STEPS:
        controls.append(
            {
                "condition": "matched_no_generalisation",
                "model_seed": "",
                "checkpoint_step": step,
                "family_size": "",
                "availability": "unavailable_stage15",
                "aggregation_unit": "condition_availability_record",
            }
        )
    control_source = write_csv(
        repository / "results/tables/stage21_figure5_controls_source.csv", tuple(controls)
    )
    outputs.append(control_source)
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for seed in range(5):
        rows = [
            row
            for row in controls
            if row["condition"] == "genuine_task" and int(row["model_seed"]) == seed
        ]
        axis.plot(
            [CHECKPOINT_STEPS.index(int(row["checkpoint_step"])) for row in rows],
            [int(row["family_size"]) for row in rows],
            color="tab:blue",
            alpha=0.25,
            linewidth=1,
        )
    genuine_means = [
        np.mean(
            [
                int(row["family_size"])
                for row in controls
                if row["condition"] == "genuine_task" and int(row["checkpoint_step"]) == step
            ]
        )
        for step in CHECKPOINT_STEPS
    ]
    random = [row for row in controls if row["condition"] == "random_label"]
    axis.plot(
        range(len(CHECKPOINT_STEPS)),
        genuine_means,
        marker="o",
        linewidth=2.5,
        label="Genuine task mean",
    )
    axis.plot(
        [CHECKPOINT_STEPS.index(int(row["checkpoint_step"])) for row in random],
        [int(row["family_size"]) for row in random],
        marker="s",
        linestyle="--",
        label="Random-label seed 0 (descriptive)",
    )
    axis.text(
        0.02,
        0.96,
        "Matched no-generalisation control: unavailable (Stage 15)",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    axis.set(
        title="Figure 5. Genuine-task models and available controls",
        xlabel="Checkpoint step",
        ylabel="Primary recovered family size",
        xticks=range(len(CHECKPOINT_STEPS)),
        xticklabels=[str(step) for step in CHECKPOINT_STEPS],
    )
    axis.tick_params(axis="x", rotation=45)
    axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    outputs.extend(_save(figure, repository, "stage21_figure5_controls"))
    plt.close(figure)

    caption = repository / "figures/stage21_principal_figures_caption.txt"
    caption.write_text(principal_figures_caption(), encoding="utf-8")
    outputs.append(caption)
    return tuple(outputs)


def principal_figures_caption() -> str:
    return (
        "\n\n".join(
            (
                "Stage 21 principal figures — provisionally validated. These outputs were "
                "generated from the definitive Stage 18 run. Independent Stage 18 reproduction "
                "comparison was pending at generation, so every result must be revalidated after "
                "that comparison. All panels regenerate from committed tables or committed, "
                "hash-pinned training logs without retraining, circuit search, or interpolation.",
                "Figure 1. Question: where do the seven fixed circuit-analysis checkpoints fall "
                "relative to memorisation and delayed generalisation? Raw, unsmoothed training "
                "and test accuracies logged every 50 training steps are shown for five "
                "independently trained genuine-task seeds (n=5; one line and one independent unit "
                "per seed). Vertical markers give the fixed step grid 200, 3400, 7450, 8150, "
                "8500, 8650, and 9050; circuits are matched by these fixed training steps, not by "
                "shifting seeds onto a common behavioural timeline. A complete grokking seed must "
                "reach at least 99.9% training accuracy, retain a delayed interval below 10% test "
                "accuracy after memorisation, later rise, and achieve stable test accuracy of at "
                "least 99%. No across-seed aggregation is plotted. The curves establish "
                "behavioural timing only and do not identify a mechanism.",
                "Figure 2. Question: how does fixed-budget circuit recoverability change across "
                "training and fidelity requirements? The left panel shows every seed's primary "
                "recovered family size at exact behavioural-fidelity threshold 0.990 and Jaccard "
                "structural-distance cutoff 0.50 across all seven checkpoints. Fidelity is the "
                "proportion of examples on which the masked circuit and dense checkpoint agree in "
                "top-one prediction. Family size is a recovered lower bound under the frozen "
                "search budget, not an exhaustive circuit count; a value reaching the search "
                "ceiling of 10 is right-censored and retained as such (no Stage 18 cell here was "
                "right-censored). Zero denotes an executed empty family, whereas the unavailable "
                "Stage 15 control is never encoded as zero. The right panel is the unweighted "
                "five-seed mean at fidelity thresholds 0.800, 0.850, 0.900, 0.950, 0.975, and "
                "0.990 at cutoff 0.50; the companion structural sensitivity uses cutoffs 0.25, "
                "0.50, and 0.75 in Figure 3. Seeds are the independent units; threshold cells are "
                "not replications. Descriptively, primary sparse recoverability emerges late and "
                "remains heterogeneous, including a final empty seed-3 family, so the plot does "
                "not establish universal contraction or a unique mechanism.",
                "Figure 3. Question: how structurally similar are the recovered alternatives, and "
                "how sensitive is recovery to the frozen distinctness cutoff? At final step 9050, "
                "the first five panels contain each primary family's complete circuit-level "
                "Jaccard-overlap matrix, where overlap is intersection size divided by union size "
                "and structural distance is one minus overlap. A family is the deterministically "
                "ordered set C1…Cn of recovered circuits meeting fidelity, sparsity, "
                "search-budget, and pairwise structural-distance requirements; matrix rows and "
                "columns follow that order, are symmetric, and have unit diagonals. Seed 3 is "
                "explicitly labelled "
                "as an empty family rather than displayed as a fabricated zero matrix. The final "
                "panel shows the unweighted n=5 seed mean of recovered family size at fidelity "
                "0.990 for cutoffs 0.25, 0.50, and 0.75 across the seven fixed checkpoints; zeros "
                "remain executed empty results. The trained seed is the independent unit; circuit "
                "pairs are display units only. Persistent alternatives and varying overlaps are "
                "descriptive and do not prove that circuits implement distinct algorithms.",
                "Figure 4. Question: do structurally recovered circuits also separate by "
                "functional transfer? At final step 9050, rows are primary-family circuits in "
                "C1…Cn order and columns are the fixed held-out operand subsets Q1, Q2, Q3, and "
                "Q4. Each cell is behavioural transfer fidelity—the circuit/dense-model top-one "
                "agreement on that subset—not ground-truth accuracy. Transfer-profile distance "
                "is the maximum absolute fidelity difference across Q1–Q4; deterministic complete "
                "linkage at tolerance 0.050 defines the displayed group count. A group therefore "
                "similarity under this rule, not mechanistic identity. Seed 3 is explicitly empty. "
                "The final panel shows one seven-checkpoint line per independently trained seed; "
                "empty-family group counts are null gaps, never numerical zeros or connected "
                "interpolations. Descriptively, each nonempty final family forms one transfer "
                "group under this tolerance, which does not prove its circuits are mechanistically "
                "equal.",
                "Figure 5. Question: how do genuine-task family trajectories compare with the "
                "available frozen controls? Thin lines preserve all five genuine-task seed values "
                "and the thick line is their unweighted mean. The random-label control was "
                "executed at the same seven fixed checkpoint steps but has one seed (n=1), so its "
                "individual values are descriptive only. The matched no-generalisation control "
                "was unavailable "
                "under the frozen Stage 15 protocol and is labelled unavailable, not drawn as an "
                "empty or zero-height result. Family size uses primary fidelity 0.990 and cutoff "
                "0.50; a zero is an executed empty family and the search ceiling 10 would be "
                "right-censored. The independent unit is the trained model seed. This incomplete "
                "control set cannot support a causal claim that grokking produced the observed "
                "differences.",
                "Aggregation and evidential limits. The independently trained model seed is the "
                "replication unit (n=5 genuine task; n=1 random-label control). "
                "Circuits, pairs, thresholds, checkpoints, search restarts, and transfer subsets "
                "within a seed are not treated as independent replications. Any displayed mean "
                "is an unweighted mean across the five genuine-task seeds. Undefined values "
                "remain null, empty-family size remains zero, and missing values are not imputed. "
                "The figures "
                "summarise recovered families under fixed definitions and budget; they neither "
                "enumerate all valid circuits nor establish mechanistic identity or universality.",
            )
        )
        + "\n"
    )
