"""Run or validate Phase I E1 size-matched Jaccard null analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_families.analysis.phase1_e1_jaccard_null import (
    analyse_inputs,
    validate_inputs,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Root containing the frozen predecessor manifest and compact Stage 18 tables.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1_e1_jaccard_null.json"),
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()
    source_root = (args.source_root or repository).resolve()
    configuration_path = args.config
    if not configuration_path.is_absolute():
        configuration_path = repository / configuration_path
    inputs = validate_inputs(configuration_path, source_root=source_root)
    validation = {
        "status": "validation_passed",
        "analysis_id": inputs.configuration["analysis_id"],
        "circuit_count": len(inputs.circuits),
        "pair_count": len(inputs.pairs),
        "model_seeds": sorted({pair.model_seed for pair in inputs.pairs}),
        "component_count": inputs.configuration["component_universe"]["total_count"],
        "source_hashes": dict(inputs.source_hashes),
    }
    if args.validate_only:
        print(json.dumps(validation, sort_keys=True))
        return
    if args.output_directory is None:
        raise SystemExit("--output-directory is required unless --validate-only is used.")
    output_directory = args.output_directory
    if not output_directory.is_absolute():
        output_directory = repository / output_directory
    analyses = analyse_inputs(inputs)
    paths = write_outputs(output_directory, inputs, analyses)
    print(
        json.dumps(
            {
                **validation,
                "status": "analysis_complete",
                "pair_null_comparison_count": len(analyses),
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
