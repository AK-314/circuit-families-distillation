"""Tests for reproducible Stage 6 training-curve artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from circuit_families.plotting.training_curves import (
    create_training_curve_artifacts,
    read_figure_source_csv,
)
from circuit_families.training.logging import append_jsonl


def _record(
    step: int,
    *,
    train_accuracy: float,
    test_accuracy: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "test-run",
        "mode": "full",
        "training_step": step,
        "learning_rate": 0.001,
        "weight_norm": 1.0,
        "gradient_norm": None if step == 0 else 0.1,
        "train_loss": 1.0 / (step + 1),
        "test_loss": 2.0 / (step + 1),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "model_state_sha256": None,
        "optimizer_state_sha256": None,
    }


def test_training_curve_artifacts_are_generated_from_metrics(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"

    records = [
        _record(0, train_accuracy=0.01, test_accuracy=0.01),
        _record(50, train_accuracy=1.0, test_accuracy=0.05),
        _record(100, train_accuracy=1.0, test_accuracy=0.06),
        _record(150, train_accuracy=1.0, test_accuracy=0.20),
        _record(200, train_accuracy=1.0, test_accuracy=0.60),
        _record(250, train_accuracy=1.0, test_accuracy=0.995),
        _record(300, train_accuracy=1.0, test_accuracy=1.0),
    ]

    for record in records:
        append_jsonl(metrics_path, record)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "seed": {"name": "model_seed", "value": 0},
                "execution": {
                    "max_steps": 300,
                    "evaluation_interval": 50,
                    "evaluate_step_zero": True,
                },
                "dataset": {
                    "train_count": 3830,
                    "total_count": 12769,
                },
                "configs": {
                    "task": {"path": "configs/task.yaml"},
                    "training": {"path": "configs/training.yaml"},
                },
            }
        ),
        encoding="utf-8",
    )

    result = create_training_curve_artifacts(
        repository_root=Path.cwd(),
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        csv_path=tmp_path / "source.csv",
        png_path=tmp_path / "figure.png",
        pdf_path=tmp_path / "figure.pdf",
        caption_path=tmp_path / "caption.txt",
    )

    assert result.csv_path.is_file()
    assert result.png_path.is_file()
    assert result.pdf_path.is_file()
    assert result.caption_path.is_file()
    assert len(result.csv_sha256) == 64
    assert result.diagnostics.met_frozen_criteria

    csv_rows = read_figure_source_csv(result.csv_path)

    assert len(csv_rows) == len(records)
    assert [row["training_step"] for row in csv_rows] == [
        record["training_step"] for record in records
    ]
    assert "Frozen grokking criteria met: yes" in (
        result.caption_path.read_text(encoding="utf-8")
    )

    second = create_training_curve_artifacts(
        repository_root=Path.cwd(),
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        csv_path=tmp_path / "source_second.csv",
        png_path=tmp_path / "figure_second.png",
        pdf_path=tmp_path / "figure_second.pdf",
        caption_path=tmp_path / "caption_second.txt",
    )

    assert result.csv_path.read_bytes() == second.csv_path.read_bytes()
    assert result.png_path.read_bytes() == second.png_path.read_bytes()
    assert result.pdf_path.read_bytes() == second.pdf_path.read_bytes()
    assert (
        result.caption_path.read_bytes()
        == second.caption_path.read_bytes()
    )
