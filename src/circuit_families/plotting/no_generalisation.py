"""Deterministic Stage 13 no-generalisation curve artifacts."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from circuit_families.analysis.no_generalisation_selection import (
    CANDIDATE_FRACTIONS,
    MATCHED_HORIZON,
)
from circuit_families.training.checkpoints import file_sha256

STAGE13_METRICS_COLUMNS = (
    "fraction",
    "exact_training_example_count",
    "subset_identifier",
    "subset_sha256",
    "run_id",
    "training_git_commit",
    "device",
    "training_step",
    "learning_rate",
    "weight_norm",
    "gradient_norm",
    "train_loss",
    "test_loss",
    "train_accuracy",
    "test_accuracy",
    "checkpoint_path",
    "checkpoint_sha256",
    "model_state_sha256",
    "optimizer_state_sha256",
)


def _serialise(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Stage 13 metrics must contain only finite values."
            )
        return repr(value)

    return str(value)


def write_stage13_metrics_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write every plotted Stage 13 metric in stable CSV form."""

    if not rows:
        raise ValueError("Stage 13 metric rows must not be empty.")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=STAGE13_METRICS_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            if set(row) != set(STAGE13_METRICS_COLUMNS):
                raise ValueError(
                    "Stage 13 metric row fields do not match "
                    "the frozen schema."
                )

            writer.writerow(
                {
                    column: _serialise(row[column])
                    for column in STAGE13_METRICS_COLUMNS
                }
            )

    return output


def read_stage13_metrics_csv(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Read typed Stage 13 metric rows."""

    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Stage 13 metrics table does not exist: {source}"
        )

    rows: list[dict[str, Any]] = []

    with source.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if tuple(reader.fieldnames or ()) != STAGE13_METRICS_COLUMNS:
            raise ValueError(
                "Stage 13 metrics columns do not match "
                "the frozen order."
            )

        for raw in reader:
            row: dict[str, Any] = {
                "fraction": float(raw["fraction"]),
                "exact_training_example_count": int(
                    raw["exact_training_example_count"]
                ),
                "subset_identifier": raw["subset_identifier"],
                "subset_sha256": raw["subset_sha256"],
                "run_id": raw["run_id"],
                "training_git_commit": raw[
                    "training_git_commit"
                ],
                "device": raw["device"],
                "training_step": int(raw["training_step"]),
                "learning_rate": float(raw["learning_rate"]),
                "weight_norm": float(raw["weight_norm"]),
                "gradient_norm": (
                    None
                    if raw["gradient_norm"] == ""
                    else float(raw["gradient_norm"])
                ),
                "train_loss": float(raw["train_loss"]),
                "test_loss": float(raw["test_loss"]),
                "train_accuracy": float(raw["train_accuracy"]),
                "test_accuracy": float(raw["test_accuracy"]),
                "checkpoint_path": raw["checkpoint_path"],
                "checkpoint_sha256": raw["checkpoint_sha256"],
                "model_state_sha256": raw["model_state_sha256"],
                "optimizer_state_sha256": raw[
                    "optimizer_state_sha256"
                ],
            }

            for name in (
                "learning_rate",
                "weight_norm",
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy",
            ):
                if not math.isfinite(float(row[name])):
                    raise ValueError(
                        f"Stage 13 metric {name} is non-finite."
                    )

            gradient_norm = row["gradient_norm"]

            if (
                gradient_norm is not None
                and not math.isfinite(gradient_norm)
            ):
                raise ValueError(
                    "Stage 13 gradient norm is non-finite."
                )

            rows.append(row)

    return rows


def validate_stage13_metrics_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the complete five-candidate saved-metric grid."""

    if not rows:
        raise ValueError("Stage 13 metric rows must not be empty.")

    by_fraction: dict[float, list[Mapping[str, Any]]] = {}

    for row in rows:
        fraction = float(row["fraction"])

        if fraction not in CANDIDATE_FRACTIONS:
            raise ValueError(
                "Stage 13 metrics contain a non-frozen fraction."
            )

        by_fraction.setdefault(fraction, []).append(row)

    if set(by_fraction) != set(CANDIDATE_FRACTIONS):
        raise ValueError(
            "Stage 13 metrics must contain all five candidates."
        )

    expected_steps = list(range(0, MATCHED_HORIZON + 1, 50))
    expected_count = len(expected_steps)

    for fraction, candidate_rows in by_fraction.items():
        ordered = sorted(
            candidate_rows,
            key=lambda row: int(row["training_step"]),
        )
        steps = [
            int(row["training_step"])
            for row in ordered
        ]

        if steps != expected_steps:
            raise ValueError(
                f"Stage 13 fraction {fraction:.2f} does not "
                "contain the complete saved-evaluation grid."
            )

        if len(candidate_rows) != expected_count:
            raise ValueError(
                f"Stage 13 fraction {fraction:.2f} has the "
                "wrong metric-record count."
            )

        for name in (
            "run_id",
            "training_git_commit",
            "device",
            "subset_identifier",
            "subset_sha256",
            "exact_training_example_count",
        ):
            values = {
                row[name]
                for row in candidate_rows
            }

            if len(values) != 1:
                raise ValueError(
                    f"Stage 13 fraction {fraction:.2f} has "
                    f"inconsistent {name} values."
                )


def plot_stage13_training_curves(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_fraction: float | None,
    png_path: str | Path,
    pdf_path: str | Path,
) -> tuple[Path, Path]:
    """Plot the three frozen Stage 13 curve panels."""

    validate_stage13_metrics_rows(rows)

    if (
        selected_fraction is not None
        and selected_fraction not in CANDIDATE_FRACTIONS
    ):
        raise ValueError(
            "Selected fraction is outside the frozen grid."
        )

    output_png = Path(png_path)
    output_pdf = Path(pdf_path)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        sharex=True,
        constrained_layout=True,
    )
    train_axis, test_axis, loss_axis = axes

    by_fraction = {
        fraction: sorted(
            (
                row
                for row in rows
                if float(row["fraction"]) == fraction
            ),
            key=lambda row: int(row["training_step"]),
        )
        for fraction in CANDIDATE_FRACTIONS
    }

    for fraction in CANDIDATE_FRACTIONS:
        candidate_rows = by_fraction[fraction]
        steps = [
            int(row["training_step"])
            for row in candidate_rows
        ]
        label = f"{fraction:.0%}"
        line_width = (
            3.0
            if selected_fraction == fraction
            else 1.5
        )
        line_label = (
            f"{label} — selected"
            if selected_fraction == fraction
            else label
        )

        train_axis.plot(
            steps,
            [
                float(row["train_accuracy"])
                for row in candidate_rows
            ],
            label=line_label,
            linewidth=line_width,
        )
        test_axis.plot(
            steps,
            [
                float(row["test_accuracy"])
                for row in candidate_rows
            ],
            label=line_label,
            linewidth=line_width,
        )
        loss_axis.plot(
            steps,
            [
                float(row["test_loss"])
                for row in candidate_rows
            ],
            label=line_label,
            linewidth=line_width,
        )

    train_axis.axhline(
        0.999,
        linestyle="--",
        linewidth=1,
        label="99.9% qualification threshold",
    )
    test_axis.axhline(
        0.10,
        linestyle="--",
        linewidth=1,
        label="10% qualification ceiling",
    )

    for axis in axes:
        axis.axvline(
            5_000,
            linestyle=":",
            linewidth=1,
            label="Step 5,000",
        )
        axis.axvline(
            MATCHED_HORIZON,
            linestyle=":",
            linewidth=1,
            label="Matched horizon 9,050",
        )
        axis.grid(True, alpha=0.25)
        axis.set_xlim(0, MATCHED_HORIZON)

    train_axis.set_ylabel("Training accuracy")
    train_axis.set_ylim(-0.02, 1.02)
    test_axis.set_ylabel("Test accuracy")
    test_axis.set_ylim(-0.02, 1.02)
    loss_axis.set_ylabel("Test cross-entropy")
    loss_axis.set_xlabel("Training step")
    loss_axis.set_yscale("log")

    train_axis.legend(
        loc="lower right",
        ncols=2,
    )
    test_axis.legend(
        loc="upper left",
        ncols=2,
    )
    loss_axis.legend(
        loc="best",
        ncols=2,
    )

    figure.suptitle(
        "Stage 13 matched no-generalisation pilot grid"
    )

    figure.savefig(
        output_png,
        dpi=200,
        metadata={"Software": "circuit-families"},
    )
    figure.savefig(
        output_pdf,
        metadata={
            "Creator": "circuit-families",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)

    return output_png, output_pdf


def write_stage13_figure_caption(
    path: str | Path,
    *,
    selected_fraction: float | None,
) -> Path:
    """Write the frozen Stage 13 figure caption."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    selected_text = (
        "No candidate qualified."
        if selected_fraction is None
        else (
            f"The selected matched no-generalisation fraction "
            f"is {selected_fraction:.0%}."
        )
    )

    caption = (
        "Stage 13 matched no-generalisation pilot curves. "
        "Training accuracy, test accuracy and test cross-entropy "
        "are shown for the frozen 5%, 10%, 15%, 20% and 25% "
        "nested training-prefix candidates through the matched "
        "9,050-step horizon. Horizontal markers show the 99.9% "
        "training-accuracy threshold and 10% test-accuracy ceiling; "
        "vertical markers show the step-5,000 memorisation deadline "
        "and matched horizon. Selection used these training and test "
        "curves only; no circuit-family metric was inspected. "
        + selected_text
    )

    output.write_text(caption + "\n", encoding="utf-8")
    return output


def stage13_figure_hashes(
    *,
    png_path: str | Path,
    pdf_path: str | Path,
    caption_path: str | Path,
) -> dict[str, str]:
    """Return physical hashes for all Stage 13 figure artifacts."""

    return {
        "training_curves_png_sha256": file_sha256(png_path),
        "training_curves_pdf_sha256": file_sha256(pdf_path),
        "training_curves_caption_sha256": file_sha256(
            caption_path
        ),
    }
