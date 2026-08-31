#!/usr/bin/env python3
"""Benchmark frozen Stage 13 search mechanics on constructed data only.

This deliberately excludes registered models, checkpoints, activations, and
scientific outcomes.  It measures the native Stage 12-R1 optimizer mechanics
and the Stage 12-R3 ordinary-restart/exact-ledger bridge at Package A sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import tempfile
import time
from pathlib import Path

import torch

from circuit_families.stage6a import COMPONENT_COUNT
from circuit_families.stage6e.records import load_technical_policy
from circuit_families.stage12r1 import (
    GateConfig,
    GateRunIdentity,
    OptimizerConfig,
    optimize_gates,
)
from circuit_families.stage12r3.ordinary_restart import (
    OrdinaryRestartContext,
    OrdinaryRestartProfile,
    RestartDiscoveryOutput,
    run_ordinary_restart_baseline,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    REPOSITORY_ROOT
    / "followup/configs/stage6e/technical_endpoint2_policy_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
    return float(value) / divisor


def _hard_concrete(repetitions: int) -> dict[str, object]:
    durations: list[float] = []
    output_bytes: list[int] = []
    for repetition in range(repetitions):
        identity = GateRunIdentity(
            method_name="stage13-constructed-hard-concrete-benchmark",
            method_version="v1",
            configuration_reference="constructed://stage13/package-a-profile",
            run_id=f"constructed-benchmark-{repetition}",
            condition_identity="constructed-516-component-quadratic",
            restart_index=repetition,
            seed_value=1701,
        )
        config = OptimizerConfig(
            learning_rate=0.01,
            max_steps=5000,
            sparsity_coefficient=0.001,
            checkpoint_every=50,
            checkpoint_retention=2,
        )
        gates = GateConfig(
            temperature=2.0 / 3.0,
            stretch_lower=-0.1,
            stretch_upper=1.1,
        )
        target = torch.linspace(0.0, 1.0, COMPONENT_COUNT, dtype=torch.float32)

        def objective(
            output: torch.Tensor,
            step: int,
            target_values: torch.Tensor = target,
        ) -> torch.Tensor:
            del step
            return ((output - target_values) ** 2).mean()

        with tempfile.TemporaryDirectory(prefix="stage13-hc-") as directory:
            started = time.perf_counter()
            result = optimize_gates(
                initial_log_alpha=torch.zeros(COMPONENT_COUNT),
                component_basis_identity="constructed-stage13-basis-516",
                component_count=COMPONENT_COUNT,
                gate_config=gates,
                run_identity=identity,
                optimizer_config=config,
                native_budget_allowance=5000,
                dense_mask_adapter=lambda sample, step: sample,
                objective_adapter=objective,
                checkpoint_directory=Path(directory),
            )
            durations.append(time.perf_counter() - started)
            if result.native_budget_consumed != 5000:
                raise RuntimeError("hard-concrete benchmark did not consume profile")
            output_bytes.append(
                sum(path.stat().st_size for path in Path(directory).iterdir())
            )
    return {
        "profile": {
            "component_count": COMPONENT_COUNT,
            "steps_per_run": 5000,
            "checkpoint_every": 50,
            "checkpoint_retention": 2,
            "dtype": "float32",
            "repetitions": repetitions,
        },
        "elapsed_seconds": durations,
        "median_seconds": statistics.median(durations),
        "median_steps_per_second": 5000.0 / statistics.median(durations),
        "retained_checkpoint_bytes": output_bytes,
    }


def _ordinary_restart(repetitions: int) -> dict[str, object]:
    policy = load_technical_policy(POLICY_PATH)
    durations: list[float] = []
    for repetition in range(repetitions):
        profile = OrdinaryRestartProfile(
            profile_id="stage13-constructed-ordinary-profile-v1",
            run_id=f"constructed-ordinary-{repetition}",
            method_name="stage12r1_hard_concrete",
            method_version="technical-v1",
            discovery_config_id="constructed-stage13-package-a",
            model_id="constructed-stage13-model",
            component_basis_reference=policy.component_basis_reference,
            fidelity_threshold=policy.fidelity_threshold,
            restart_count=16,
            root_seed=2901,
            native_budget_per_restart=5000,
            exact_evaluation_allowance=256,
        )

        def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
            proposals = []
            for proposal_index in range(16):
                values = [0] * COMPONENT_COUNT
                # Complete restart identity makes all 256 constructed masks
                # different, without consulting earlier restart state.
                offset = context.restart_index * 16 + proposal_index
                values[offset] = 1
                values[(offset + 257) % COMPONENT_COUNT] = 1
                proposals.append(tuple(values))
            return RestartDiscoveryOutput(
                proposals=tuple(proposals),
                native_work_consumed=5000,
            )

        started = time.perf_counter()
        result = run_ordinary_restart_baseline(
            profile=profile,
            policy=policy,
            evaluator=lambda mask: 1.0,
            discovery_procedure=procedure,
        )
        durations.append(time.perf_counter() - started)
        if result.requested_restart_count != 16:
            raise RuntimeError("ordinary-restart benchmark profile changed")
    return {
        "profile": {
            "restarts": 16,
            "declared_native_units_per_restart": 5000,
            "proposals_per_restart": 16,
            "common_exact_allowance_including_intact": 256,
            "repetitions": repetitions,
        },
        "elapsed_seconds": durations,
        "median_seconds": statistics.median(durations),
        "median_restarts_per_second": 16.0 / statistics.median(durations),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    started = time.perf_counter()
    record = {
        "schema_version": "stage13-search-profile-benchmark/v1",
        "purpose": "non-scientific feasibility evidence only",
        "scientific_data": False,
        "production_eligible": False,
        "registered_or_private_artifacts_accessed": False,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "inputs": {
            "policy_path": str(POLICY_PATH.relative_to(REPOSITORY_ROOT)),
            "policy_sha256": _sha256(POLICY_PATH),
            "registered_models": 0,
            "constructed_only": True,
        },
        "hard_concrete_native_mechanics": _hard_concrete(args.repetitions),
        "ordinary_restart_bridge": _ordinary_restart(args.repetitions),
        "peak_process_rss_mib": _peak_rss_mib(),
        "total_elapsed_seconds": time.perf_counter() - started,
        "exclusions": [
            "teacher or student forward/backward passes",
            "production CUDA throughput",
            "production exact fidelity evaluation",
            "filesystem contention under campaign concurrency",
        ],
    }
    payload = json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
