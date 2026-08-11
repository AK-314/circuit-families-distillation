"""Generate bounded loss-versus-norm diagnostic artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_families.analysis.training_dynamics import (
    extract_diagnostic_rows,
    plot_loss_norm_diagnostics,
    read_diagnostic_csv,
    write_diagnostic_caption,
    write_diagnostic_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate seed-0 loss/norm diagnostics."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    return parser.parse_args()


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()

    manifest_path = (
        repository
        / "manifests"
        / f"training_{args.run_id}.json"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    metrics_path = _resolve(
        repository,
        manifest["output_paths"]["metrics_jsonl"],
    )

    rows = extract_diagnostic_rows(
        metrics_path=metrics_path,
        repository_root=repository,
    )

    csv_path = (
        repository
        / "results"
        / "tables"
        / "seed_0_loss_norm_diagnostics.csv"
    )

    write_diagnostic_csv(rows, csv_path)

    rows = read_diagnostic_csv(csv_path)

    phase_manifest = json.loads(
        (
            repository
            / "manifests"
            / "checkpoints_seed_0.json"
        ).read_text(encoding="utf-8")
    )

    stable_post = (
        phase_manifest["selected_stable_post_checkpoint"]
    )

    stable_post_step = (
        None
        if stable_post is None
        else stable_post["training_step"]
    )

    png_path = (
        repository
        / "figures"
        / "seed_0_loss_norm_diagnostics.png"
    )

    pdf_path = (
        repository
        / "figures"
        / "seed_0_loss_norm_diagnostics.pdf"
    )

    caption_path = (
        repository
        / "figures"
        / "seed_0_loss_norm_diagnostics_caption.txt"
    )

    plot_loss_norm_diagnostics(
        rows,
        run_id=args.run_id,
        stable_post_step=stable_post_step,
        png_path=png_path,
        pdf_path=pdf_path,
    )

    write_diagnostic_caption(
        run_id=args.run_id,
        checkpoint_count=len(rows),
        stable_post_step=stable_post_step,
        output_path=caption_path,
    )

    print(f"rows: {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"png: {png_path}")
    print(f"pdf: {pdf_path}")
    print(f"caption: {caption_path}")


if __name__ == "__main__":
    main()
