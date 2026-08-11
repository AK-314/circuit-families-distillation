"""Evaluate one deterministic component mask at one frozen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
    evaluate_component_mask,
    load_checkpoint_evaluation_context,
)
from circuit_families.interpretability.masks import (
    ComponentMask,
    load_component_mask,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate one Stage 8 component mask."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
    )
    return parser.parse_args()


def _display_path(
    repository: Path,
    path: Path,
) -> str:
    resolved = path.resolve()

    try:
        return str(resolved.relative_to(repository))
    except ValueError:
        return str(resolved)


def build_evaluation_record(
    *,
    repository_root: str | Path,
    context: CheckpointEvaluationContext,
    mask: ComponentMask,
    mask_path: str | Path,
    batch_size: int,
) -> dict[str, Any]:
    """Build one deterministic provenance-bearing result record."""

    repository = Path(repository_root).resolve()
    metrics = evaluate_component_mask(
        context.model,
        context.inputs,
        context.targets,
        mask,
        batch_size=batch_size,
    )

    return {
        "schema_version": 1,
        "run_id": context.run_id,
        "checkpoint_phase": context.checkpoint_phase,
        "checkpoint_step": context.checkpoint_step,
        "checkpoint_path": _display_path(
            repository,
            context.checkpoint_path,
        ),
        "checkpoint_sha256": context.checkpoint_sha256,
        "model_state_sha256": context.model_state_sha256,
        "checkpoint_manifest_path": _display_path(
            repository,
            context.checkpoint_manifest_path,
        ),
        "checkpoint_manifest_sha256": (
            context.checkpoint_manifest_sha256
        ),
        "training_manifest_path": _display_path(
            repository,
            context.training_manifest_path,
        ),
        "training_manifest_sha256": (
            context.training_manifest_sha256
        ),
        "task_config_sha256": context.task_config_sha256,
        "model_config_sha256": context.model_config_sha256,
        "training_config_sha256": (
            context.training_config_sha256
        ),
        "combined_config_sha256": (
            context.combined_config_sha256
        ),
        "dataset_sha256": context.dataset_sha256,
        "split_sha256": context.split_sha256,
        "dataset_archive_sha256": (
            context.dataset_archive_sha256
        ),
        "dataset_metadata_sha256": (
            context.dataset_metadata_sha256
        ),
        "example_ordering": context.example_ordering,
        "evaluated_sequence_position": -1,
        "output_classes": {
            "minimum": 0,
            "maximum": 112,
            "count": 113,
            "equals_token_eligible": False,
        },
        "full_model_reference": {
            "method": "computed_live",
            "checkpoint_specific": True,
            "cached": False,
        },
        "mask_path": _display_path(
            repository,
            Path(mask_path),
        ),
        "mask_id": mask.mask_id,
        "mask": mask.to_record(),
        "metrics": metrics.to_record(),
        "device": str(context.device),
    }


def write_evaluation_record(
    path: str | Path,
    record: dict[str, Any],
) -> Path:
    """Write one deterministic JSON evaluation record."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(
        record,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    output_path.write_text(serialised + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    """Load, validate, evaluate, and print one mask."""

    args = parse_args()
    repository = args.repository_root.resolve()
    mask_path = (
        args.mask
        if args.mask.is_absolute()
        else repository / args.mask
    )
    mask = load_component_mask(mask_path)

    context = load_checkpoint_evaluation_context(
        repository_root=repository,
        run_id=args.run_id,
        checkpoint_manifest_path=args.checkpoint_manifest,
        checkpoint_step=args.checkpoint_step,
        device_override=args.device,
    )

    record = build_evaluation_record(
        repository_root=repository,
        context=context,
        mask=mask,
        mask_path=mask_path,
        batch_size=args.batch_size,
    )

    if args.output_json is not None:
        output_path = (
            args.output_json
            if args.output_json.is_absolute()
            else repository / args.output_json
        )
        write_evaluation_record(output_path, record)

    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
