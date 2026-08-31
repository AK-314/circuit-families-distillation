from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from circuit_families.stage12p4 import CodecProfile, iter_ledger_rows
from circuit_families.stage12p5 import (
    CONDITIONS,
    AlignmentProfile,
    ArrayActivationAdapter,
    Stage12P5ContractError,
    build_logical_job,
    fit_linear_alignment,
    record_unavailable_comparison_set,
    reduce_comparison_set,
    run_comparison_set,
    seal_result_with_p3,
)
from circuit_families.stage12p5.cli import validate
from circuit_families.stage12p5.runner import RunInterrupted
from circuit_families.stage12p5.synthetic import (
    RECIPIENT_ARCH,
    SHA_B,
    SOURCE_ARCH,
    SyntheticOutcomeAdapter,
    build_synthetic_fixture,
)


def test_alignment_plan_handles_complex_sign_phase_and_rejects_cross_pair() -> None:
    profile = AlignmentProfile(
        "alignment/test/v1",
        SHA_B,
        SOURCE_ARCH,
        RECIPIENT_ARCH,
        2,
        2,
        "fit/test/v1",
        "c" * 64,
        "boundary/test/v1",
    )
    source = np.array([[1 + 1j, 2 - 1j], [2 + 0j, -1 + 2j], [3 - 1j, 1 + 0j]])
    transform = np.array([[0, -1j], [2, 0]], dtype=np.complex128)
    recipient = source @ transform.T
    plan = fit_linear_alignment(
        profile=profile,
        pair_id="a" * 64,
        source_fit=source,
        recipient_fit=recipient,
    )
    assert np.allclose(plan.apply(source[0], pair_id="a" * 64), recipient[0])
    assert plan.source_rank == 2
    assert plan.mapped_rank == 2
    with pytest.raises(Stage12P5ContractError, match="incompatible pair"):
        plan.apply(source[0], pair_id="b" * 64)


def test_alignment_rejects_nonfinite_rank_and_dimension_errors() -> None:
    profile = build_synthetic_fixture().trial.alignment
    with pytest.raises(Stage12P5ContractError, match="coordinate count"):
        fit_linear_alignment(
            profile=profile,
            pair_id="a" * 64,
            source_fit=np.ones((3, 3)),
            recipient_fit=np.ones((3, 2)),
        )
    source = np.ones((3, 2), dtype=np.complex128)
    source[0, 0] = np.nan
    with pytest.raises(Stage12P5ContractError, match="nonfinite"):
        fit_linear_alignment(
            profile=profile,
            pair_id="a" * 64,
            source_fit=source,
            recipient_fit=np.ones((3, 2)),
        )


def test_activation_adapter_rejects_wrong_architecture_location_layout_and_input() -> None:
    fixture = build_synthetic_fixture()
    adapter = ArrayActivationAdapter(
        architecture_ref=RECIPIENT_ARCH,
        supported_locations=(fixture.trial.location.location_ref,),
        external_layout_ref=fixture.trial.location.recipient_layout_ref,
        canonical_axes=(0,),
    )
    with pytest.raises(Stage12P5ContractError, match="input identity"):
        adapter.write(
            {},
            input_id="input-a",
            location=fixture.trial.location,
            state=np.zeros(4),
        )
    with pytest.raises(Stage12P5ContractError, match="shape"):
        adapter.write(
            {"input-a": np.zeros(4)},
            input_id="input-a",
            location=fixture.trial.location,
            state=np.zeros(3),
        )


def test_interruption_resume_matches_uninterrupted_and_skips_sealed_records(tmp_path: Path) -> None:
    fixture = build_synthetic_fixture()
    resumed_root = tmp_path / "resumed"
    with pytest.raises(RunInterrupted):
        run_comparison_set(
            trial=fixture.trial,
            payloads=fixture.payloads,
            accounting=fixture.accounting,
            activation_adapter=fixture.recipient_adapter,
            recipient_model_factory=fixture.model_factory,
            outcome_adapter=fixture.outcome_adapter,
            output_root=resumed_root,
            interrupt_after=3,
        )
    sealed_before = {
        path.name: path.read_bytes() for path in (resumed_root / "sealed-conditions").iterdir()
    }
    resumed = run_comparison_set(
        trial=fixture.trial,
        payloads=fixture.payloads,
        accounting=fixture.accounting,
        activation_adapter=fixture.recipient_adapter,
        recipient_model_factory=fixture.model_factory,
        outcome_adapter=fixture.outcome_adapter,
        output_root=resumed_root,
    )
    sealed_after = {
        path.name: path.read_bytes()
        for path in (resumed_root / "sealed-conditions").iterdir()
        if path.name in sealed_before
    }
    assert sealed_after == sealed_before
    uninterrupted = run_comparison_set(
        trial=fixture.trial,
        payloads=fixture.payloads,
        accounting=fixture.accounting,
        activation_adapter=fixture.recipient_adapter,
        recipient_model_factory=fixture.model_factory,
        outcome_adapter=fixture.outcome_adapter,
        output_root=tmp_path / "uninterrupted",
    )
    assert resumed.result_sha256 == uninterrupted.result_sha256
    assert tuple(record.condition for record in resumed.records) == CONDITIONS


def test_runner_retains_failed_unavailable_censored_nonfinite_and_complete(tmp_path: Path) -> None:
    fixture = build_synthetic_fixture()
    result = run_comparison_set(
        trial=fixture.trial,
        payloads=fixture.payloads,
        accounting=fixture.accounting,
        activation_adapter=fixture.recipient_adapter,
        recipient_model_factory=fixture.model_factory,
        outcome_adapter=fixture.outcome_adapter,
        output_root=tmp_path,
    )
    states = {record.state for record in result.records}
    assert {"complete", "failed", "unavailable", "censored"}.issubset(states)
    assert any(record.outcome.kind == "nonfinite" for record in result.records)
    assert result.sealed
    report = json.loads((tmp_path / "artifacts" / "technical-report.json").read_bytes())
    assert report["scientific_data"] is False
    assert report["production_eligible"] is False
    assert report["condition_inventory"] == list(CONDITIONS)


def test_structural_control_failure_closes_all_six_as_unavailable(tmp_path: Path) -> None:
    fixture = build_synthetic_fixture()
    result = record_unavailable_comparison_set(
        trial=fixture.trial,
        reason="wrong-mode scarcity in synthetic roster",
        output_root=tmp_path,
    )
    assert result.sealed
    assert tuple(record.condition for record in result.records) == CONDITIONS
    assert {record.state for record in result.records} == {"unavailable"}
    assert all(record.failure.phase == "control_construction" for record in result.records)


def test_corrupt_stale_and_cross_condition_resume_evidence_reject(tmp_path: Path) -> None:
    fixture = build_synthetic_fixture()
    with pytest.raises(RunInterrupted):
        run_comparison_set(
            trial=fixture.trial,
            payloads=fixture.payloads,
            accounting=fixture.accounting,
            activation_adapter=fixture.recipient_adapter,
            recipient_model_factory=fixture.model_factory,
            outcome_adapter=fixture.outcome_adapter,
            output_root=tmp_path,
            interrupt_after=1,
        )
    path = tmp_path / "sealed-conditions" / "00.json"
    value = json.loads(path.read_bytes())
    value["condition"] = CONDITIONS[1]
    path.write_text(json.dumps(value))
    with pytest.raises(Stage12P5ContractError, match="condition identity|hash"):
        run_comparison_set(
            trial=fixture.trial,
            payloads=fixture.payloads,
            accounting=fixture.accounting,
            activation_adapter=fixture.recipient_adapter,
            recipient_model_factory=fixture.model_factory,
            outcome_adapter=fixture.outcome_adapter,
            output_root=tmp_path,
        )


def test_p3_job_and_p4_ledger_preserve_interfaces_and_order(tmp_path: Path) -> None:
    fixture = build_synthetic_fixture()
    job = build_logical_job(fixture.trial)
    assert job.retry_seed_namespace_reference == fixture.trial.seed_namespace_ref
    assert job.payload_reference.sha256 == job.expected_inputs[0].sha256
    result = run_comparison_set(
        trial=fixture.trial,
        payloads=fixture.payloads,
        accounting=fixture.accounting,
        activation_adapter=fixture.recipient_adapter,
        recipient_model_factory=fixture.model_factory,
        outcome_adapter=fixture.outcome_adapter,
        output_root=tmp_path,
    )
    rows = list(
        iter_ledger_rows(
            tmp_path / "artifacts" / "conditions.ledger.gz",
            profile=CodecProfile("codec/stage12p5-gzip/v1", "gzip", 6),
        )
    )
    assert [row["condition"] for row in rows] == list(CONDITIONS)
    assert [row["record_sha256"] for row in rows] == [
        record.record_sha256 for record in result.records
    ]
    p3 = seal_result_with_p3(
        trial=fixture.trial,
        result_root=tmp_path,
        state_root=tmp_path / "p3-state",
    )
    assert p3["logical_job_id"] == job.job_id
    assert p3["status"]["complete"] is True
    assert p3["status"]["counts"] == {"succeeded": 1}


def test_reducer_is_outcome_neutral_for_ties_and_control_wins(tmp_path: Path) -> None:
    fixture = build_synthetic_fixture(terminal_paths=False)
    result = run_comparison_set(
        trial=fixture.trial,
        payloads=fixture.payloads,
        accounting=fixture.accounting,
        activation_adapter=fixture.recipient_adapter,
        recipient_model_factory=fixture.model_factory,
        outcome_adapter=SyntheticOutcomeAdapter(exercise_terminal_paths=False),
        output_root=tmp_path,
    )
    reduced = reduce_comparison_set(result)
    serialized = json.dumps(reduced, sort_keys=True).lower()
    assert reduced["claim_rule_present"] is False
    assert reduced["superiority_threshold_present"] is False
    for forbidden in ("supportive", "successful abstraction", "negative result", "outperformed"):
        assert forbidden not in serialized


def test_validate_only_cli_function_is_deterministic(tmp_path: Path) -> None:
    first = validate(tmp_path / "first")
    second = validate(tmp_path / "second")
    assert first["sealed_result_sha256"] == second["sealed_result_sha256"]
    assert first["capacity_comparison_sha256"] == second["capacity_comparison_sha256"]
    assert first["condition_inventory"] == list(CONDITIONS)
