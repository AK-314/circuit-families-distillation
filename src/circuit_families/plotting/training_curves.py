"""Reproducible Stage 6 training-curve tables and figures."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from circuit_families.config import load_config, load_training_config
from circuit_families.training.checkpoints import file_sha256
from circuit_families.training.logging import read_jsonl
from circuit_families.training.stage6_validation import (
    GrokkingDiagnostics,
    compute_grokking_diagnostics,
    validate_metric_records,
)

FIGURE_SOURCE_COLUMNS = (
    "run_id",
    "mode",
    "training_step",
    "learning_rate",
    "weight_norm",
    "gradient_norm",
    "train_loss",
    "test_loss",
    "train_accuracy",
    "test_accuracy",
)


@dataclass(frozen=True)
class TrainingCurveArtifacts:
    """Paths, hashes, and diagnostics for generated Stage 6 artifacts."""

    csv_path: Path
    png_path: Path
    pdf_path: Path
    caption_path: Path
    csv_sha256: str
    diagnostics: GrokkingDiagnostics


def _serialise_csv_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        return repr(value)

    return str(value)


def write_figure_source_csv(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write the exact plotted values in a stable CSV representation."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIGURE_SOURCE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    column: _serialise_csv_value(record[column])
                    for column in FIGURE_SOURCE_COLUMNS
                }
            )

    return output_path


def read_figure_source_csv(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Read a Stage 6 figure-source CSV with typed values."""

    input_path = Path(path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Figure-source CSV does not exist: {input_path}"
        )

    rows: list[dict[str, Any]] = []

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if tuple(reader.fieldnames or ()) != FIGURE_SOURCE_COLUMNS:
            raise ValueError(
                "Figure-source CSV columns do not match the frozen order."
            )

        for raw in reader:
            rows.append(
                {
                    "run_id": raw["run_id"],
                    "mode": raw["mode"],
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
                }
            )

    return rows


def validate_figure_source_csv(
    records: Sequence[Mapping[str, Any]],
    csv_path: str | Path,
) -> None:
    """Confirm that the figure-source CSV exactly matches metrics JSONL."""

    csv_rows = read_figure_source_csv(csv_path)

    if len(csv_rows) != len(records):
        raise ValueError(
            "Figure-source CSV row count does not match metrics JSONL."
        )

    for record, csv_row in zip(records, csv_rows, strict=True):
        for column in FIGURE_SOURCE_COLUMNS:
            if csv_row[column] != record[column]:
                raise ValueError(
                    "Figure-source CSV mismatch at "
                    f"step {record['training_step']} for {column}."
                )


def plot_training_curves(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    png_path: str | Path,
    pdf_path: str | Path,
) -> tuple[Path, Path]:
    """Plot accuracy and loss over the complete saved run horizon."""

    if not rows:
        raise ValueError("Training-curve rows must not be empty.")

    output_png = Path(png_path)
    output_pdf = Path(pdf_path)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    steps = [int(row["training_step"]) for row in rows]
    train_accuracy = [float(row["train_accuracy"]) for row in rows]
    test_accuracy = [float(row["test_accuracy"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    test_loss = [float(row["test_loss"]) for row in rows]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    accuracy_axis, loss_axis = axes

    accuracy_axis.plot(steps, train_accuracy, label="Training accuracy")
    accuracy_axis.plot(steps, test_accuracy, label="Test accuracy")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(-0.02, 1.02)
    accuracy_axis.grid(True, alpha=0.25)
    accuracy_axis.legend()

    loss_axis.plot(steps, train_loss, label="Training loss")
    loss_axis.plot(steps, test_loss, label="Test loss")
    loss_axis.set_xlabel("Training step")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.set_yscale("log")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend()

    final_step = steps[-1]
    accuracy_axis.set_xlim(0, final_step)
    figure.suptitle(f"Modular-addition training trajectory\n{run_id}")

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


def _resolve_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def create_training_curve_artifacts(
    *,
    repository_root: str | Path,
    metrics_path: str | Path,
    manifest_path: str | Path,
    csv_path: str | Path,
    png_path: str | Path,
    pdf_path: str | Path,
    caption_path: str | Path,
) -> TrainingCurveArtifacts:
    """Create the Stage 6 source table, figure files, and caption."""

    repository = Path(repository_root).resolve()
    metrics_file = Path(metrics_path)
    manifest_file = Path(manifest_path)

    records = read_jsonl(metrics_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    execution = manifest["execution"]

    validate_metric_records(
        records,
        max_steps=execution["max_steps"],
        evaluation_interval=execution["evaluation_interval"],
        include_step_zero=execution["evaluate_step_zero"],
    )

    diagnostics = compute_grokking_diagnostics(records)

    output_csv = write_figure_source_csv(records, csv_path)
    validate_figure_source_csv(records, output_csv)

    source_rows = read_figure_source_csv(output_csv)

    output_png, output_pdf = plot_training_curves(
        source_rows,
        run_id=manifest["run_id"],
        png_path=png_path,
        pdf_path=pdf_path,
    )

    task_config_path = _resolve_path(
        repository,
        manifest["configs"]["task"]["path"],
    )
    training_config_path = _resolve_path(
        repository,
        manifest["configs"]["training"]["path"],
    )

    task_config = load_config(task_config_path)
    training_config = load_training_config(training_config_path)

    train_count = manifest["dataset"]["train_count"]
    total_count = manifest["dataset"]["total_count"]
    training_fraction = task_config["split"]["primary_train_fraction"]

    caption = (
        "Provisional Figure 1. Training and test accuracy and cross-entropy "
        "loss over the complete modular-addition run. "
        f"Run ID: {manifest['run_id']}. "
        f"Model seed: {manifest['seed']['value']}. "
        f"Training fraction: {training_fraction:.2f} "
        f"({train_count}/{total_count}). "
        f"Optimizer: {training_config['optimizer']['name']}. "
        f"Weight decay: {training_config['optimizer']['weight_decay']}. "
        f"Evaluation interval: {execution['evaluation_interval']} steps. "
        f"Final training step: {records[-1]['training_step']}. "
        "Frozen grokking criteria met: "
        f"{'yes' if diagnostics.met_frozen_criteria else 'no'}. "
        "No formal Stage 7 checkpoint landmarks are marked."
    )

    output_caption = Path(caption_path)
    output_caption.parent.mkdir(parents=True, exist_ok=True)
    output_caption.write_text(caption + "\n", encoding="utf-8")

    return TrainingCurveArtifacts(
        csv_path=output_csv,
        png_path=output_png,
        pdf_path=output_pdf,
        caption_path=output_caption,
        csv_sha256=file_sha256(output_csv),
        diagnostics=diagnostics,
    )
