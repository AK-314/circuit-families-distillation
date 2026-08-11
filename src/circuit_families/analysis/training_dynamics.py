"""Checkpoint-based loss and parameter-norm diagnostics."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from circuit_families.training.checkpoints import (
    file_sha256,
    load_checkpoint_payload,
)
from circuit_families.training.logging import read_jsonl

PARAMETER_GROUPS = (
    "embedding",
    "positional_embedding",
    "attention",
    "mlp",
    "unembedding",
)


@dataclass(frozen=True)
class CheckpointNorms:
    """Norm diagnostics extracted from one saved checkpoint."""

    training_step: int
    total_parameter_norm: float
    embedding_parameter_norm: float
    positional_embedding_norm: float
    attention_parameter_norm: float
    mlp_parameter_norm: float
    unembedding_parameter_norm: float
    optimizer_first_moment_norm: float
    optimizer_second_moment_norm: float
    checkpoint_path: Path
    checkpoint_sha256: str


def parameter_group(name: str) -> str:
    """Return the unique diagnostic group for one trainable parameter."""

    if name.startswith("embed."):
        return "embedding"

    if name.startswith("pos_embed."):
        return "positional_embedding"

    if name.startswith("blocks.") and ".attn." in name:
        return "attention"

    if name.startswith("blocks.") and ".mlp." in name:
        return "mlp"

    if name.startswith("unembed."):
        return "unembedding"

    raise ValueError(f"Unclassified trainable parameter: {name}")


def _aggregate_tensor_norm(
    tensors: list[torch.Tensor],
    *,
    label: str,
) -> float:
    """Return sqrt(sum(tensor**2)) using float64 accumulation."""

    total_squared = 0.0

    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} contains a non-tensor value.")

        detached = tensor.detach().to(
            device="cpu",
            dtype=torch.float64,
        )

        if not torch.isfinite(detached).all():
            raise FloatingPointError(
                f"{label} contains a non-finite tensor."
            )

        total_squared += float(torch.sum(detached * detached).item())

    norm = math.sqrt(total_squared)

    if not math.isfinite(norm):
        raise FloatingPointError(f"{label} norm is non-finite.")

    return norm


def model_parameter_group_norms(
    model_state: Mapping[str, Any],
) -> dict[str, float]:
    """Calculate total and disjoint trainable-parameter group norms."""

    grouped: dict[str, list[torch.Tensor]] = {
        group: []
        for group in PARAMETER_GROUPS
    }

    ignored_buffers = {
        "blocks.0.attn.mask",
        "blocks.0.attn.IGNORE",
    }

    for name, value in model_state.items():
        if name in ignored_buffers:
            continue

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Model-state value for {name} is not a tensor."
            )

        grouped[parameter_group(name)].append(value)

    if any(not tensors for tensors in grouped.values()):
        missing = [
            group
            for group, tensors in grouped.items()
            if not tensors
        ]
        raise ValueError(
            "Missing parameter groups: " + ", ".join(missing)
        )

    group_norms = {
        group: _aggregate_tensor_norm(
            tensors,
            label=f"{group} parameters",
        )
        for group, tensors in grouped.items()
    }

    total = math.sqrt(
        sum(norm * norm for norm in group_norms.values())
    )

    return {
        "total_parameter_norm": total,
        "embedding_parameter_norm": group_norms["embedding"],
        "positional_embedding_norm": group_norms[
            "positional_embedding"
        ],
        "attention_parameter_norm": group_norms["attention"],
        "mlp_parameter_norm": group_norms["mlp"],
        "unembedding_parameter_norm": group_norms["unembedding"],
    }


def optimizer_moment_norms(
    optimizer_state: Mapping[str, Any],
) -> tuple[float, float]:
    """Calculate aggregate Adam first- and second-moment norms."""

    state = optimizer_state.get("state")

    if not isinstance(state, Mapping):
        raise TypeError("Optimizer state must contain a state mapping.")

    if not state:
        return 0.0, 0.0

    first_moments: list[torch.Tensor] = []
    second_moments: list[torch.Tensor] = []

    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping):
            raise TypeError(
                "Each optimizer parameter state must be a mapping."
            )

        first = parameter_state.get("exp_avg")
        second = parameter_state.get("exp_avg_sq")

        if not isinstance(first, torch.Tensor):
            raise ValueError(
                "Optimizer parameter state is missing exp_avg."
            )

        if not isinstance(second, torch.Tensor):
            raise ValueError(
                "Optimizer parameter state is missing exp_avg_sq."
            )

        first_moments.append(first)
        second_moments.append(second)

    return (
        _aggregate_tensor_norm(
            first_moments,
            label="optimizer first moments",
        ),
        _aggregate_tensor_norm(
            second_moments,
            label="optimizer second moments",
        ),
    )


def extract_checkpoint_norms(
    checkpoint_path: str | Path,
) -> CheckpointNorms:
    """Extract all required norms from one validated checkpoint."""

    path = Path(checkpoint_path)
    payload = load_checkpoint_payload(
        path,
        map_location="cpu",
    )

    model_norms = model_parameter_group_norms(
        payload["model_state"]
    )
    first_moment, second_moment = optimizer_moment_norms(
        payload["optimizer_state"]
    )

    return CheckpointNorms(
        training_step=int(payload["training_step"]),
        total_parameter_norm=model_norms[
            "total_parameter_norm"
        ],
        embedding_parameter_norm=model_norms[
            "embedding_parameter_norm"
        ],
        positional_embedding_norm=model_norms[
            "positional_embedding_norm"
        ],
        attention_parameter_norm=model_norms[
            "attention_parameter_norm"
        ],
        mlp_parameter_norm=model_norms[
            "mlp_parameter_norm"
        ],
        unembedding_parameter_norm=model_norms[
            "unembedding_parameter_norm"
        ],
        optimizer_first_moment_norm=first_moment,
        optimizer_second_moment_norm=second_moment,
        checkpoint_path=path,
        checkpoint_sha256=file_sha256(path),
    )


DIAGNOSTIC_COLUMNS = (
    "training_step",
    "train_loss",
    "test_loss",
    "train_accuracy",
    "test_accuracy",
    "total_parameter_norm",
    "embedding_parameter_norm",
    "positional_embedding_norm",
    "attention_parameter_norm",
    "mlp_parameter_norm",
    "unembedding_parameter_norm",
    "optimizer_first_moment_norm",
    "optimizer_second_moment_norm",
    "checkpoint_path",
    "checkpoint_sha256",
)


def extract_diagnostic_rows(
    *,
    metrics_path: str | Path,
    repository_root: str | Path,
) -> list[dict[str, object]]:
    """Extract deterministic loss-and-norm rows for every checkpoint."""

    repository = Path(repository_root).resolve()
    records = read_jsonl(metrics_path)
    rows: list[dict[str, object]] = []

    for record in records:
        checkpoint_path = Path(str(record["checkpoint_path"]))

        if not checkpoint_path.is_absolute():
            checkpoint_path = repository / checkpoint_path

        norms = extract_checkpoint_norms(checkpoint_path)
        step = int(record["training_step"])

        if norms.training_step != step:
            raise ValueError(
                f"Checkpoint step mismatch at metrics step {step}."
            )

        if norms.checkpoint_sha256 != record["checkpoint_sha256"]:
            raise ValueError(
                f"Checkpoint hash mismatch at step {step}."
            )

        recorded_weight_norm = float(record["weight_norm"])

        if not math.isclose(
            norms.total_parameter_norm,
            recorded_weight_norm,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "Recomputed total parameter norm does not match "
                f"recorded weight_norm at step {step}: "
                f"{norms.total_parameter_norm} versus "
                f"{recorded_weight_norm}."
            )

        rows.append(
            {
                "training_step": step,
                "train_loss": float(record["train_loss"]),
                "test_loss": float(record["test_loss"]),
                "train_accuracy": float(record["train_accuracy"]),
                "test_accuracy": float(record["test_accuracy"]),
                "total_parameter_norm": (
                    norms.total_parameter_norm
                ),
                "embedding_parameter_norm": (
                    norms.embedding_parameter_norm
                ),
                "positional_embedding_norm": (
                    norms.positional_embedding_norm
                ),
                "attention_parameter_norm": (
                    norms.attention_parameter_norm
                ),
                "mlp_parameter_norm": (
                    norms.mlp_parameter_norm
                ),
                "unembedding_parameter_norm": (
                    norms.unembedding_parameter_norm
                ),
                "optimizer_first_moment_norm": (
                    norms.optimizer_first_moment_norm
                ),
                "optimizer_second_moment_norm": (
                    norms.optimizer_second_moment_norm
                ),
                "checkpoint_path": str(
                    Path(str(record["checkpoint_path"]))
                ),
                "checkpoint_sha256": norms.checkpoint_sha256,
            }
        )

    return rows


def write_diagnostic_csv(
    rows: list[dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Write deterministic checkpoint diagnostics as CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=DIAGNOSTIC_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: (
                        repr(row[column])
                        if isinstance(row[column], float)
                        else str(row[column])
                    )
                    for column in DIAGNOSTIC_COLUMNS
                }
            )

    return output


def read_diagnostic_csv(
    path: str | Path,
) -> list[dict[str, object]]:
    """Read the deterministic loss-and-norm diagnostic table."""

    input_path = Path(path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Diagnostic CSV does not exist: {input_path}"
        )

    rows: list[dict[str, object]] = []

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if tuple(reader.fieldnames or ()) != DIAGNOSTIC_COLUMNS:
            raise ValueError(
                "Diagnostic CSV columns do not match the required order."
            )

        for raw in reader:
            rows.append(
                {
                    "training_step": int(raw["training_step"]),
                    "train_loss": float(raw["train_loss"]),
                    "test_loss": float(raw["test_loss"]),
                    "train_accuracy": float(raw["train_accuracy"]),
                    "test_accuracy": float(raw["test_accuracy"]),
                    "total_parameter_norm": float(
                        raw["total_parameter_norm"]
                    ),
                    "embedding_parameter_norm": float(
                        raw["embedding_parameter_norm"]
                    ),
                    "positional_embedding_norm": float(
                        raw["positional_embedding_norm"]
                    ),
                    "attention_parameter_norm": float(
                        raw["attention_parameter_norm"]
                    ),
                    "mlp_parameter_norm": float(
                        raw["mlp_parameter_norm"]
                    ),
                    "unembedding_parameter_norm": float(
                        raw["unembedding_parameter_norm"]
                    ),
                    "optimizer_first_moment_norm": float(
                        raw["optimizer_first_moment_norm"]
                    ),
                    "optimizer_second_moment_norm": float(
                        raw["optimizer_second_moment_norm"]
                    ),
                    "checkpoint_path": raw["checkpoint_path"],
                    "checkpoint_sha256": raw["checkpoint_sha256"],
                }
            )

    return rows


def plot_loss_norm_diagnostics(
    rows: list[dict[str, object]],
    *,
    run_id: str,
    stable_post_step: int | None,
    png_path: str | Path,
    pdf_path: str | Path,
) -> tuple[Path, Path]:
    """Plot loss, parameter norms, and optimiser moments."""

    if not rows:
        raise ValueError("Diagnostic rows must not be empty.")

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_png = Path(png_path)
    output_pdf = Path(pdf_path)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    steps = [int(row["training_step"]) for row in rows]

    figure, axes = plt.subplots(
        5,
        1,
        figsize=(11, 14),
        sharex=True,
        constrained_layout=True,
    )

    loss_axis = axes[0]
    total_axis = axes[1]
    unembedding_axis = axes[2]
    first_moment_axis = axes[3]
    second_moment_axis = axes[4]

    loss_axis.plot(
        steps,
        [float(row["train_loss"]) for row in rows],
        label="Training loss",
    )
    loss_axis.plot(
        steps,
        [float(row["test_loss"]) for row in rows],
        label="Test loss",
    )
    loss_axis.set_yscale("log")
    loss_axis.set_ylabel("Cross-entropy loss\n(log scale)")
    loss_axis.legend()

    total_axis.plot(
        steps,
        [float(row["total_parameter_norm"]) for row in rows],
    )
    total_axis.set_ylabel("Total parameter\nL2 norm")

    unembedding_axis.plot(
        steps,
        [float(row["unembedding_parameter_norm"]) for row in rows],
    )
    unembedding_axis.set_ylabel("Unembedding\nL2 norm")

    first_moment_axis.plot(
        steps,
        [float(row["optimizer_first_moment_norm"]) for row in rows],
    )
    first_moment_axis.set_ylabel("Adam first-moment\nL2 norm")

    second_moment_axis.plot(
        steps,
        [float(row["optimizer_second_moment_norm"]) for row in rows],
    )
    second_moment_axis.set_ylabel("Adam second-moment\nL2 norm")
    second_moment_axis.set_xlabel("Training step")

    for axis in axes:
        axis.grid(True, alpha=0.25)

        if stable_post_step is not None:
            axis.axvline(
                stable_post_step,
                linestyle="--",
                linewidth=1.0,
                label=(
                    "Selected stable-post checkpoint"
                    if axis is loss_axis
                    else None
                ),
            )

    if stable_post_step is not None:
        loss_axis.legend()

    figure.suptitle(
        "Seed-0 loss and norm diagnostics\n"
        f"{run_id}"
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


def write_diagnostic_caption(
    *,
    run_id: str,
    checkpoint_count: int,
    stable_post_step: int | None,
    output_path: str | Path,
) -> Path:
    """Write the bounded diagnostic caption."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    stable_text = (
        "No stable-post checkpoint was selected."
        if stable_post_step is None
        else (
            "The formally selected stable-post checkpoint is step "
            f"{stable_post_step}."
        )
    )

    caption = (
        "Loss-versus-norm diagnostic for the complete saved checkpoint "
        f"trajectory of run {run_id}. "
        f"The figure uses {checkpoint_count} existing checkpoints and does "
        "not involve retraining or checkpoint reselection. It shows training "
        "and test cross-entropy loss, total parameter norm, unembedding norm, "
        "and aggregate Adam first- and second-moment norms. "
        f"{stable_text} "
        "The figure is descriptive only and does not establish a slingshot "
        "mechanism. "
        "The repeated loss cycles broadly align with oscillations in total "
        "parameter norm, while unembedding norm shows related but not "
        "uniformly clearer cyclic structure. Optimizer moments exhibit "
        "several large transient spikes but do not consistently align with "
        "every loss spike. The selected stable-post checkpoint at step "
        f"{stable_post_step} lies on a declining-loss segment rather than "
        "at a major spike. Overall, the behaviour is consistent with some "
        "expected norm dynamics of slingshot-like behaviour, but it is not "
        "sufficient to establish the slingshot mechanism."
    )

    output.write_text(caption + "\n", encoding="utf-8")
    return output
