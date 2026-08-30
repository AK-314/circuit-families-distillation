from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from circuit_families.stage6a.endpoint import reduce_endpoint1
from circuit_families.stage6a.models import TerminationStatus
from circuit_families.stage12r1 import (
    GateConfig,
    GateRunIdentity,
    ProposalExtractionConfig,
    evaluate_proposals_exact,
    extract_binary_proposals,
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
        configuration_reference="fixture://proposal",
        run_id="proposal-fixture",
        condition_identity="synthetic-only",
        restart_index=0,
        seed_value=41,
    )
    return replace(base, **updates)


def test_bounded_proposals_keep_duplicate_references() -> None:
    log_alpha = torch.tensor(
        [1000.0, -1000.0, -1000.0],
        dtype=torch.float64,
    )
    config = ProposalExtractionConfig(
        thresholds=(0.5, 0.8),
        top_k_sizes=(1,),
        max_proposals=3,
    )

    batch = extract_binary_proposals(
        log_alpha=log_alpha,
        component_basis_identity="toy-basis-3",
        component_count=3,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=config,
    )

    assert batch.proposal_count == 3
    assert batch.unique_mask_count == 1
    assert batch.proposals[0].duplicate_of_proposal_index is None
    assert batch.proposals[1].duplicate_of_proposal_index == 0
    assert batch.proposals[2].duplicate_of_proposal_index == 0
    assert len({item.proposal_reference for item in batch.proposals}) == 3


def test_proposal_provenance_is_complete_and_deterministic() -> None:
    log_alpha = torch.tensor(
        [-1.0, 0.0, 1.0, 2.0],
        dtype=torch.float64,
    )
    config = ProposalExtractionConfig(
        thresholds=(0.5,),
        top_k_sizes=(2,),
        stochastic_draws=2,
        max_proposals=4,
    )

    first = extract_binary_proposals(
        log_alpha=log_alpha,
        component_basis_identity="toy-basis-4",
        component_count=4,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=config,
    )
    second = extract_binary_proposals(
        log_alpha=log_alpha.clone(),
        component_basis_identity="toy-basis-4",
        component_count=4,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=config,
    )

    assert first == second
    assert tuple(p.proposal_index for p in first.proposals) == (0, 1, 2, 3)
    assert all(p.gate_state_sha256 for p in first.proposals)
    assert all(p.deterministic_identity_sha256 for p in first.proposals)
    assert all(p.restart_index == 0 for p in first.proposals)


def test_changed_identity_changes_seeded_stochastic_proposal() -> None:
    log_alpha = torch.zeros(64, dtype=torch.float64)
    config = ProposalExtractionConfig(
        stochastic_draws=1,
        max_proposals=1,
    )

    first = extract_binary_proposals(
        log_alpha=log_alpha,
        component_basis_identity="toy-basis-64",
        component_count=64,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=config,
    )
    second = extract_binary_proposals(
        log_alpha=log_alpha,
        component_basis_identity="toy-basis-64",
        component_count=64,
        gate_config=gate_config(),
        run_identity=identity(seed_value=42),
        extraction_config=config,
    )

    assert first.proposals[0].mask != second.proposals[0].mask


def test_current_exact_bridge_rejects_non_common_basis() -> None:
    batch = extract_binary_proposals(
        log_alpha=torch.zeros(4),
        component_basis_identity="architecture-specific-basis",
        component_count=4,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=ProposalExtractionConfig(
            top_k_sizes=(2,),
        ),
    )

    with pytest.raises(ValueError, match="516-component basis"):
        evaluate_proposals_exact(
            batch=batch,
            evaluator=lambda mask: 1.0,
            fidelity_threshold=0.9,
            exact_evaluation_allowance=2,
        )


def test_surrogate_proposal_never_self_qualifies() -> None:
    log_alpha = torch.cat(
        (
            torch.tensor([1000.0]),
            torch.full((515,), -1000.0),
        )
    )
    batch = extract_binary_proposals(
        log_alpha=log_alpha,
        component_basis_identity="common-516-technical-basis",
        component_count=516,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=ProposalExtractionConfig(
            top_k_sizes=(1,),
        ),
    )

    sparse_mask = batch.proposals[0].mask

    def exact_evaluator(mask):
        if all(mask):
            return 1.0
        if mask == sparse_mask:
            return 0.1
        return 0.0

    exact = evaluate_proposals_exact(
        batch=batch,
        evaluator=exact_evaluator,
        fidelity_threshold=0.9,
        exact_evaluation_allowance=2,
    )

    sparse_entries = [
        entry
        for entry in exact.evaluations
        if entry.retained_count == 1
    ]

    assert len(sparse_entries) == 1
    assert sparse_entries[0].fidelity == 0.1
    assert sparse_entries[0].qualifies is False
    assert exact.exact_budget_charged == 2


def test_duplicate_and_intact_proposals_do_not_add_unique_exact_charge() -> None:
    log_alpha = torch.full((516,), 1000.0)
    batch = extract_binary_proposals(
        log_alpha=log_alpha,
        component_basis_identity="common-516-technical-basis",
        component_count=516,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=ProposalExtractionConfig(
            thresholds=(0.5, 0.8),
        ),
    )

    exact = evaluate_proposals_exact(
        batch=batch,
        evaluator=lambda mask: 1.0,
        fidelity_threshold=0.9,
        exact_evaluation_allowance=1,
    )

    assert exact.terminal_state == "completed"
    assert exact.exact_budget_charged == 1
    assert exact.exact_ledger_evaluation_count == 1
    assert exact.exact_ledger_proposal_count == 2
    assert exact.exact_event_kinds.count("duplicate") == 2


def test_unique_exact_allowance_exhaustion_is_auditable() -> None:
    log_alpha = torch.linspace(
        -3.0,
        3.0,
        516,
        dtype=torch.float64,
    )
    batch = extract_binary_proposals(
        log_alpha=log_alpha,
        component_basis_identity="common-516-technical-basis",
        component_count=516,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=ProposalExtractionConfig(
            top_k_sizes=(1, 2),
        ),
    )

    exact = evaluate_proposals_exact(
        batch=batch,
        evaluator=lambda mask: 1.0,
        fidelity_threshold=0.9,
        exact_evaluation_allowance=2,
    )

    assert exact.terminal_state == "exhausted"
    assert exact.exact_budget_charged == 2
    assert "exhaustion" in exact.exact_event_kinds


def test_endpoint1_reducer_consumes_shared_exact_entries_unchanged() -> None:
    log_alpha = torch.cat(
        (
            torch.full((10,), 1000.0),
            torch.full((506,), -1000.0),
        )
    )
    batch = extract_binary_proposals(
        log_alpha=log_alpha,
        component_basis_identity="common-516-technical-basis",
        component_count=516,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=ProposalExtractionConfig(
            top_k_sizes=(10,),
        ),
    )

    exact = evaluate_proposals_exact(
        batch=batch,
        evaluator=lambda mask: 1.0,
        fidelity_threshold=0.9,
        exact_evaluation_allowance=2,
    )

    endpoint = reduce_endpoint1(
        exact.evaluations,
        termination=TerminationStatus(
            status="completed",
            procedure_censored=False,
        ),
    )

    assert endpoint.retained_proportion == 10 / 516

    selected = next(
        entry
        for entry in exact.evaluations
        if entry.mask_identity == endpoint.mask_identity
    )
    assert selected.retained_count == 10
    assert selected.fidelity == 1.0


def test_evaluator_failure_is_not_converted_to_surrogate_success() -> None:
    log_alpha = torch.cat(
        (
            torch.tensor([1000.0]),
            torch.full((515,), -1000.0),
        )
    )
    batch = extract_binary_proposals(
        log_alpha=log_alpha,
        component_basis_identity="common-516-technical-basis",
        component_count=516,
        gate_config=gate_config(),
        run_identity=identity(),
        extraction_config=ProposalExtractionConfig(
            top_k_sizes=(1,),
        ),
    )

    calls = 0

    def evaluator(mask):
        nonlocal calls
        calls += 1
        if all(mask):
            return 1.0
        raise RuntimeError("synthetic exact evaluator failure")

    with pytest.raises(
        RuntimeError,
        match="synthetic exact evaluator failure",
    ):
        evaluate_proposals_exact(
            batch=batch,
            evaluator=evaluator,
            fidelity_threshold=0.9,
            exact_evaluation_allowance=2,
        )

    assert calls == 2
