from __future__ import annotations

from dataclasses import asdict

from circuit_families.stage6d import (
    DiscoveryRequest,
    DiversityForcedAdapter,
    GreedyDeletionAdapter,
    deterministic_seed_evidence,
)

MASK_SIZE = 516


def mask_without(*indices):
    mask = [1] * MASK_SIZE
    for index in indices:
        mask[index] = 0
    return mask


def evaluator(mask):
    return 1.0 - sum(value == 0 for value in mask) / 1000.0


def request(
    method,
    unit,
    *,
    native=5,
    exact=4,
    restarts=0,
    seed=1,
):
    version = "inherited-technical-adapter/v1"
    ref = f"technical-{method}"
    return DiscoveryRequest(
        run_id=f"run-{method}",
        method_name=method,
        method_version=version,
        configuration_reference=ref,
        seed_evidence=deterministic_seed_evidence(
            method_name=method,
            method_version=version,
            configuration_reference=ref,
            seed_value=seed,
        ),
        native_budget_unit=unit,
        native_budget_allowance=native,
        exact_evaluation_allowance=exact,
        maximum_restarts=restarts,
        synthetic_fixture=True,
        production_eligible=False,
    )


def test_common_shape_distinct_native_units_restart_and_exact_dedup():
    calls = []

    def counted(mask):
        calls.append(tuple(mask))
        return evaluator(mask)

    greedy = GreedyDeletionAdapter(
        proposal_source=lambda request, inherited: [
            mask_without(0),
            mask_without(1),
            mask_without(1),
        ],
        evaluator=counted,
        fidelity_threshold=0.9,
    )

    diversity = DiversityForcedAdapter(
        restart_proposal_source=lambda request, inherited: [
            (0, [mask_without(2)]),
            (1, [mask_without(3), mask_without(3)]),
        ],
        evaluator=counted,
        fidelity_threshold=0.9,
    )

    g = greedy.run(
        request("greedy_deletion", "ranked_component_proposals")
    )
    d = diversity.run(
        request(
            "diversity_forced",
            "restart_ranked_proposals",
            restarts=2,
            seed=2,
        )
    )

    assert tuple(sorted(asdict(g))) == tuple(sorted(asdict(d)))
    assert g.native_budget_unit != d.native_budget_unit
    assert g.proposal_count == 3
    assert d.proposal_count == 3
    assert d.restart_count == 1

    assert g.exact_request_count == 3
    assert d.exact_request_count == 3

    assert g.exact_ledger_evaluation_count == 3
    assert d.exact_ledger_evaluation_count == 3
    assert g.exact_ledger_proposal_count == 3
    assert d.exact_ledger_proposal_count == 3

    assert len(g.exact_ledger_sha256) == 64
    assert len(d.exact_ledger_sha256) == 64


def test_native_exhaustion_is_distinct_from_exact_exhaustion():
    greedy = GreedyDeletionAdapter(
        proposal_source=lambda request, inherited: [
            mask_without(0),
            mask_without(1),
        ],
        evaluator=evaluator,
        fidelity_threshold=0.9,
    )

    native = greedy.run(
        request(
            "greedy_deletion",
            "ranked_component_proposals",
            native=1,
            exact=4,
        )
    )
    assert native.stopping_status == "native_budget_exhausted"
    assert native.native_budget_exhausted is True

    exact = greedy.run(
        request(
            "greedy_deletion",
            "ranked_component_proposals",
            native=4,
            exact=1,
        )
    )
    assert exact.stopping_status == "exact_budget_exhausted"
    assert exact.exact_budget_exhausted is True


def test_evaluator_failure_becomes_failed_result():
    calls = 0

    def failing(mask):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("synthetic failure")
        return 1.0

    adapter = GreedyDeletionAdapter(
        proposal_source=lambda request, inherited: [mask_without(0)],
        evaluator=failing,
        fidelity_threshold=0.9,
    )

    result = adapter.run(
        request(
            "greedy_deletion",
            "ranked_component_proposals",
        )
    )

    assert result.stopping_status == "failed"
    assert result.trajectory[-1].kind == "failure"


def test_inherited_entry_points_are_passed_not_reimplemented():
    seen = {}

    def greedy_source(request, inherited):
        seen["greedy"] = inherited.__module__ + "." + inherited.__name__
        return []

    def diversity_source(request, inherited):
        seen["diversity"] = inherited.__module__ + "." + inherited.__name__
        return []

    GreedyDeletionAdapter(
        proposal_source=greedy_source,
        evaluator=evaluator,
        fidelity_threshold=0.9,
    ).run(
        request(
            "greedy_deletion",
            "ranked_component_proposals",
        )
    )

    DiversityForcedAdapter(
        restart_proposal_source=diversity_source,
        evaluator=evaluator,
        fidelity_threshold=0.9,
    ).run(
        request(
            "diversity_forced",
            "restart_ranked_proposals",
            restarts=1,
        )
    )

    assert seen["greedy"].endswith(
        "interpretability.sparse_search.greedy_sparse_search"
    )
    assert seen["diversity"].endswith(
        "interpretability.diversity_forced_search."
        "run_sequential_family_search"
    )


def test_stage6d_does_not_relabel_old_top_one_metric():
    from pathlib import Path

    root = Path("src/circuit_families/stage6d")
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.glob("*.py")
    ).lower()

    assert "top_one" not in source
    assert "top-one" not in source
