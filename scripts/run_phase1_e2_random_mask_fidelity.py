"""Run or validate Phase I E2 matched random-mask fidelity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_families.analysis.phase1_e2_random_mask_fidelity import (
    CheckpointSource,
    execute_analysis,
    validate_inputs,
    write_outputs,
)
from circuit_families.interpretability.fidelity import (
    compute_full_model_reference,
    evaluate_component_mask,
    load_checkpoint_evaluation_context,
)
from circuit_families.training import canonical_state_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1_e2_random_mask_fidelity_null.json"),
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()
    source_root = args.source_root.resolve()
    config = args.config if args.config.is_absolute() else repository / args.config
    inputs = validate_inputs(
        config,
        repository_root=repository,
        source_root=source_root,
    )
    replicates = int(
        inputs.configuration["sampling"]["replicates_per_unique_seed_composition_profile"]
    )
    validation = {
        "status": "validation_passed",
        "analysis_id": inputs.configuration["analysis_id"],
        "observed_circuit_count": len(inputs.observed_circuits),
        "unique_profile_count": len(inputs.profiles),
        "null_model_count": 2,
        "planned_random_mask_evaluations": len(inputs.profiles) * 2 * replicates,
        "model_seeds": sorted(inputs.checkpoints),
        "source_hashes": dict(inputs.source_hashes),
    }
    if args.validate_only:
        print(json.dumps(validation, sort_keys=True))
        return
    if args.output_directory is None:
        raise SystemExit("--output-directory is required unless --validate-only is used.")
    checkpoint_step = int(inputs.configuration["execution"]["checkpoint_step"])

    def evaluator_factory(source: CheckpointSource, batch_size: int):
        context = load_checkpoint_evaluation_context(
            repository_root=source_root,
            run_id=source.run_id,
            checkpoint_manifest_path=source.checkpoint_manifest,
            checkpoint_step=checkpoint_step,
            device_override=args.device,
        )
        reference = compute_full_model_reference(
            context.model,
            context.inputs,
            context.targets,
            batch_size=batch_size,
        )
        state_before = canonical_state_hash(context.model.state_dict())
        hooks_before = (
            len(context.model.blocks[0].attn.hook_z.fwd_hooks),
            len(context.model.blocks[0].mlp.hook_post.fwd_hooks),
        )

        def evaluator(mask):
            return evaluate_component_mask(
                context.model,
                context.inputs,
                context.targets,
                mask,
                batch_size=batch_size,
                full_model_reference=reference,
            )

        def finalize() -> None:
            state_after = canonical_state_hash(context.model.state_dict())
            hooks_after = (
                len(context.model.blocks[0].attn.hook_z.fwd_hooks),
                len(context.model.blocks[0].mlp.hook_post.fwd_hooks),
            )
            if state_after != state_before:
                raise RuntimeError("Model state changed during E2 evaluation.")
            if hooks_after != hooks_before:
                raise RuntimeError("Hook counts changed during E2 evaluation.")

        return evaluator, finalize

    evaluations, runtime_rows = execute_analysis(
        inputs,
        evaluator_factory=evaluator_factory,
    )
    output_directory = args.output_directory
    if not output_directory.is_absolute():
        output_directory = repository / output_directory
    paths = write_outputs(
        output_directory,
        inputs,
        evaluations,
        runtime_rows,
    )
    print(
        json.dumps(
            {
                **validation,
                "status": "analysis_complete",
                "random_mask_evaluation_count": len(evaluations),
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
