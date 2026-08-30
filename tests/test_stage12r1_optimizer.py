from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from circuit_families.stage12r1 import (
    GateConfig,
    GateRunIdentity,
    OptimizerConfig,
    optimize_gates,
)


def gate_config() -> GateConfig:
    return GateConfig(
        temperature=0.7,
        stretch_lower=-0.1,
        stretch_upper=1.1,
    )


def identity(**updates) -> GateRunIdentity:
    base = GateRunIdentity(
        method_name="stage12r1_hard_concrete",
        method_version="technical-v1",
        configuration_reference="fixture://optimizer",
        run_id="optimizer-fixture",
        condition_identity="constructed-known-relevant-components",
        restart_index=0,
        seed_value=23,
    )
    return replace(base, **updates)


def optimizer_config(**updates) -> OptimizerConfig:
    base = OptimizerConfig(
        learning_rate=0.05,
        max_steps=8,
        sparsity_coefficient=0.01,
        checkpoint_every=1,
        checkpoint_retention=2,
    )
    return replace(base, **updates)


def dense_adapter(
    gates: torch.Tensor,
    step: int,
) -> torch.Tensor:
    del step
    return gates


def known_target_objective(
    output: torch.Tensor,
    step: int,
) -> torch.Tensor:
    del step
    target = torch.tensor(
        [1.0, 1.0, 0.0, 0.0],
        dtype=output.dtype,
        device=output.device,
    )
    return ((output - target) ** 2).mean()


def run(
    tmp_path: Path,
    *,
    config: OptimizerConfig | None = None,
    resume_from: Path | None = None,
    interrupt_predicate=None,
    objective=known_target_objective,
    budget: int = 8,
    run_identity: GateRunIdentity | None = None,
):
    return optimize_gates(
        initial_log_alpha=torch.zeros(4, dtype=torch.float64),
        component_basis_identity="technical-basis-4",
        component_count=4,
        gate_config=gate_config(),
        run_identity=identity() if run_identity is None else run_identity,
        optimizer_config=optimizer_config() if config is None else config,
        native_budget_allowance=budget,
        dense_mask_adapter=dense_adapter,
        objective_adapter=objective,
        checkpoint_directory=tmp_path,
        resume_from=resume_from,
        interrupt_predicate=interrupt_predicate,
    )


def test_constructed_fixture_uses_bounded_optimizer_steps(tmp_path) -> None:
    result = run(tmp_path)

    assert result.terminal_state == "completed"
    assert result.native_budget_unit == "optimizer_step"
    assert result.native_budget_consumed == 8
    assert result.next_step == 8
    assert len(result.trajectory) == 8
    assert result.scientific_data is False
    assert result.production_eligible is False


def test_interrupted_resume_matches_uninterrupted(tmp_path) -> None:
    uninterrupted = run(tmp_path / "full")

    interrupted = run(
        tmp_path / "resume",
        interrupt_predicate=lambda next_step: next_step == 3,
    )
    assert interrupted.terminal_state == "interrupted"
    assert interrupted.latest_checkpoint is not None

    resumed = run(
        tmp_path / "resume",
        resume_from=Path(interrupted.latest_checkpoint),
    )

    assert resumed.terminal_state == "completed"
    assert resumed.gate_state_sha256 == uninterrupted.gate_state_sha256
    assert resumed.trajectory == uninterrupted.trajectory
    assert resumed.native_budget_consumed == 8


def test_checkpoint_retention_is_bounded(tmp_path) -> None:
    result = run(tmp_path)

    assert result.terminal_state == "completed"
    checkpoints = sorted(tmp_path.glob("optimizer-step-*.json"))
    assert len(checkpoints) == 2
    assert checkpoints[-1].name == "optimizer-step-00000008.json"


def test_tampered_checkpoint_rejects(tmp_path) -> None:
    interrupted = run(
        tmp_path,
        interrupt_predicate=lambda next_step: next_step == 2,
    )
    assert interrupted.latest_checkpoint is not None

    path = Path(interrupted.latest_checkpoint)
    record = json.loads(path.read_text())
    record["payload"]["log_alpha"][0] += 1.0
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, resume_from=path)


def test_mismatched_resume_identity_rejects(tmp_path) -> None:
    interrupted = run(
        tmp_path,
        interrupt_predicate=lambda next_step: next_step == 2,
    )
    assert interrupted.latest_checkpoint is not None

    with pytest.raises(ValueError, match="run identity mismatch"):
        run(
            tmp_path,
            resume_from=Path(interrupted.latest_checkpoint),
            run_identity=identity(
                condition_identity="different-condition",
            ),
        )


def test_mismatched_optimizer_configuration_rejects(tmp_path) -> None:
    interrupted = run(
        tmp_path,
        interrupt_predicate=lambda next_step: next_step == 2,
    )
    assert interrupted.latest_checkpoint is not None

    with pytest.raises(
        ValueError,
        match="optimizer configuration mismatch",
    ):
        run(
            tmp_path,
            resume_from=Path(interrupted.latest_checkpoint),
            config=optimizer_config(learning_rate=0.025),
        )


def test_zero_native_budget_performs_no_steps(tmp_path) -> None:
    result = run(tmp_path, budget=0)

    assert result.terminal_state == "exhausted"
    assert result.native_budget_consumed == 0
    assert result.next_step == 0
    assert result.trajectory == ()


def test_smaller_native_budget_exhausts_cleanly(tmp_path) -> None:
    result = run(tmp_path, budget=3)

    assert result.terminal_state == "exhausted"
    assert result.native_budget_consumed == 3
    assert result.next_step == 3


def test_zero_gradient_fixture_is_valid(tmp_path) -> None:
    def zero_gradient(
        output: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        del step
        return (output * 0.0).sum()

    result = run(
        tmp_path,
        config=optimizer_config(
            max_steps=2,
            sparsity_coefficient=0.0,
        ),
        objective=zero_gradient,
        budget=2,
    )

    assert result.terminal_state == "completed"
    assert result.native_budget_consumed == 2


def test_nonfinite_objective_records_numerical_failure(tmp_path) -> None:
    def nonfinite(
        output: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        del output, step
        return torch.tensor(float("inf"), dtype=torch.float64)

    result = run(
        tmp_path,
        config=optimizer_config(max_steps=2),
        objective=nonfinite,
        budget=2,
    )

    assert result.terminal_state == "numerical_failure"
    assert result.failure_reason == "nonfinite_objective"
    assert result.native_budget_consumed == 0


def test_nonfinite_gradient_records_numerical_failure(tmp_path) -> None:
    def bad_gradient(
        output: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        del step
        return (
            output
            * torch.tensor(
                float("nan"),
                dtype=output.dtype,
                device=output.device,
            )
        ).sum()

    result = run(
        tmp_path,
        config=optimizer_config(max_steps=2),
        objective=bad_gradient,
        budget=2,
    )

    assert result.terminal_state == "numerical_failure"
    assert result.native_budget_consumed == 0


def test_native_optimizer_has_no_exact_endpoint_dependency(tmp_path) -> None:
    result = run(
        tmp_path,
        config=optimizer_config(max_steps=1),
        budget=1,
    )

    assert result.terminal_state == "completed"
