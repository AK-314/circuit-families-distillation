"""Generate provisional Stage 6 training-curve artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from circuit_families.plotting.training_curves import (
    create_training_curve_artifacts,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate Stage 6 training-curve artifacts."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--output-stem",
        default="seed_0_training_curves",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and report the Stage 6 figure artifacts."""

    args = parse_args()
    repository = args.repository_root.resolve()

    metrics_path = (
        repository
        / "results/raw"
        / args.run_id
        / "metrics.jsonl"
    )
    manifest_path = (
        repository
        / "manifests"
        / f"training_{args.run_id}.json"
    )

    result = create_training_curve_artifacts(
        repository_root=repository,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        csv_path=repository
        / "results/tables"
        / f"{args.output_stem.replace('_training_curves', '')}_training_metrics.csv",
        png_path=repository / "figures" / f"{args.output_stem}.png",
        pdf_path=repository / "figures" / f"{args.output_stem}.pdf",
        caption_path=repository
        / "figures"
        / f"{args.output_stem}_caption.txt",
    )

    print(f"source_csv: {result.csv_path}")
    print(f"source_csv_sha256: {result.csv_sha256}")
    print(f"png: {result.png_path}")
    print(f"pdf: {result.pdf_path}")
    print(f"caption: {result.caption_path}")
    print(
        "frozen_grokking_criteria_met: "
        f"{result.diagnostics.met_frozen_criteria}"
    )


if __name__ == "__main__":
    main()
