from dataclasses import replace
from pathlib import Path

import pytest

from circuit_families.stage6a.models import canonical_mask_identity
from circuit_families.stage6e.packing import qualify_and_deduplicate
from circuit_families.stage6e.records import (
    PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS,
    CandidateRecord,
    CompatibilityGraphRecord,
    Endpoint2ResultRecord,
    ExactCandidateEvidence,
    PackingProofRecord,
    QualificationResult,
    SelectedMemberRecord,
    load_technical_policy,
)

POLICY_PATH = Path("followup/configs/stage6e/technical_endpoint2_policy_v1.json")


def test_technical_policy_is_explicitly_nonproduction_and_nonresolving() -> None:
    policy = load_technical_policy(POLICY_PATH)

    assert policy.policy_kind == "technical_fixture"
    assert policy.scientific_data is False
    assert policy.production_default is False
    assert policy.resolves_unresolved_decisions == ()
    assert policy.component_basis_size == 516
    assert len(policy.policy_hash) == 64

    with pytest.raises(ValueError, match="production default"):
        replace(policy, production_default=True)

    with pytest.raises(ValueError, match="resolve any UD"):
        replace(policy, resolves_unresolved_decisions=("UD-007",))


def test_policy_identity_is_deterministic() -> None:
    first = load_technical_policy(POLICY_PATH)
    second = load_technical_policy(POLICY_PATH)

    assert first.policy_hash == second.policy_hash
    assert first.policy_id == second.policy_id


def test_candidate_contract_canonicalizes_order_without_losing_provenance() -> None:
    candidate = CandidateRecord(
        model_id="synthetic-model",
        discovery_method_id="synthetic-method",
        discovery_config_id="synthetic-config",
        source_budget_reference="synthetic-budget",
        component_basis_reference="common-component-basis-516/v1",
        component_basis_size=516,
        mask_identity="mask-a",
        retained_components=(9, 2, 5),
        exact_fidelity=0.8,
        proposal_references=("proposal-b", "proposal-a"),
        exact_evaluation_references=("eval-b", "eval-a"),
        source_ledger_references=("ledger-b", "ledger-a"),
    )

    assert candidate.retained_components == (2, 5, 9)
    assert candidate.proposal_references == ("proposal-a", "proposal-b")
    assert candidate.exact_evaluation_references == ("eval-a", "eval-b")
    assert len(candidate.candidate_hash) == 64


def test_graph_contract_is_order_invariant() -> None:
    graph_a = CompatibilityGraphRecord(
        policy_hash="p",
        input_hash="i",
        node_mask_identities=("c", "a", "b"),
        compatible_edges=(("c", "a"), ("b", "a")),
    )
    graph_b = CompatibilityGraphRecord(
        policy_hash="p",
        input_hash="i",
        node_mask_identities=("b", "c", "a"),
        compatible_edges=(("a", "b"), ("a", "c")),
    )

    assert graph_a.node_mask_identities == ("a", "b", "c")
    assert graph_a.compatible_edges == (("a", "b"), ("a", "c"))
    assert graph_a.graph_hash == graph_b.graph_hash


def test_result_contract_uses_only_procedure_lower_bound_semantics() -> None:
    member = SelectedMemberRecord(
        mask_identity="mask-a",
        candidate_hash="candidate-a",
        proposal_references=("proposal-a",),
        exact_evaluation_references=("eval-a",),
    )
    proof = PackingProofRecord(
        policy_hash="policy",
        input_hash="input",
        graph_hash="graph",
        solver_reference="exact-solver",
        tie_break_reference="lexicographic",
        recomputation_reference="stage6e-recompute/v1",
        selected_mask_identities=("mask-a",),
        packing_lower_bound=1,
    )
    result = Endpoint2ResultRecord(
        policy_hash="policy",
        input_hash="input",
        graph_hash="graph",
        raw_candidate_count=2,
        unique_candidate_count=1,
        qualified_candidate_count=1,
        selected_members=(member,),
        packing_lower_bound=1,
        proof=proof,
    )

    assert result.semantics == PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS
    assert result.to_record()["semantics"] == "procedure_dependent_packing_lower_bound"
    assert len(result.result_hash) == 64

    with pytest.raises(ValueError, match="procedure-relative"):
        replace(result, semantics="global_packing_number")


def test_zero_result_is_representable_by_contract() -> None:
    proof = PackingProofRecord(
        policy_hash="policy",
        input_hash="input",
        graph_hash="graph",
        solver_reference="exact-solver",
        tie_break_reference="lexicographic",
        recomputation_reference="stage6e-recompute/v1",
        selected_mask_identities=(),
        packing_lower_bound=0,
    )
    result = Endpoint2ResultRecord(
        policy_hash="policy",
        input_hash="input",
        graph_hash="graph",
        raw_candidate_count=0,
        unique_candidate_count=0,
        qualified_candidate_count=0,
        selected_members=(),
        packing_lower_bound=0,
        proof=proof,
    )

    assert result.packing_lower_bound == 0
    assert result.selected_members == ()




def _binary_mask(retained_count: int) -> tuple[int, ...]:
    return (1,) * retained_count + (0,) * (516 - retained_count)


def _exact_evidence(
    *,
    retained_count: int = 10,
    exact_fidelity: float = 0.8,
    proposal: str = "proposal-a",
    evaluation: str = "eval-a",
    ledger: str = "ledger-a",
    **changes,
) -> ExactCandidateEvidence:
    policy = load_technical_policy(POLICY_PATH)
    mask = _binary_mask(retained_count)
    values = {
        "model_id": "synthetic-model",
        "discovery_method_id": "synthetic-method",
        "discovery_config_id": "synthetic-config",
        "source_budget_reference": policy.source_budget_reference,
        "fidelity_metric_reference": policy.fidelity_metric_reference,
        "component_basis_reference": policy.component_basis_reference,
        "component_basis_size": 516,
        "mask": mask,
        "mask_identity": canonical_mask_identity(
            tuple(index for index, retained in enumerate(mask) if retained)
        ),
        "exact_fidelity": exact_fidelity,
        "proposal_reference": proposal,
        "exact_evaluation_reference": evaluation,
        "source_ledger_reference": ledger,
        "source_ledger_hash": "a" * 64,
        "recomputed_ledger_hash": "a" * 64,
        "sealed": True,
        "final_exact_evaluation": True,
    }
    values.update(changes)
    return ExactCandidateEvidence(**values)


def _qualify(records, policy=None):
    if policy is None:
        policy = load_technical_policy(POLICY_PATH)
    return qualify_and_deduplicate(
        records,
        policy,
        model_id="synthetic-model",
        discovery_method_id="synthetic-method",
        discovery_config_id="synthetic-config",
    )


def test_fidelity_threshold_boundary_is_inclusive() -> None:
    policy = load_technical_policy(POLICY_PATH)

    below = _qualify(
        [_exact_evidence(exact_fidelity=policy.fidelity_threshold - 1e-9)]
    )
    equal = _qualify(
        [_exact_evidence(exact_fidelity=policy.fidelity_threshold)]
    )
    above = _qualify(
        [_exact_evidence(exact_fidelity=policy.fidelity_threshold + 1e-9)]
    )

    assert below.qualified_candidate_count == 0
    assert equal.qualified_candidate_count == 1
    assert above.qualified_candidate_count == 1


def test_component_cap_boundary_is_inclusive() -> None:
    policy = replace(
        load_technical_policy(POLICY_PATH),
        max_component_proportion=0.25,
    )

    inside = _qualify([_exact_evidence(retained_count=128)], policy)
    equal = _qualify([_exact_evidence(retained_count=129)], policy)
    outside = _qualify([_exact_evidence(retained_count=130)], policy)

    assert inside.qualified_candidate_count == 1
    assert equal.qualified_candidate_count == 1
    assert outside.qualified_candidate_count == 0


def test_duplicate_masks_preserve_all_provenance() -> None:
    first = _exact_evidence(
        proposal="proposal-b",
        evaluation="eval-b",
        ledger="ledger-b",
    )
    second = _exact_evidence(
        proposal="proposal-a",
        evaluation="eval-a",
        ledger="ledger-a",
    )

    result = _qualify([first, second])

    assert result.raw_candidate_count == 2
    assert result.unique_candidate_count == 1
    assert result.qualified_candidate_count == 1

    candidate = result.qualified_candidates[0]
    assert candidate.proposal_references == ("proposal-a", "proposal-b")
    assert candidate.exact_evaluation_references == ("eval-a", "eval-b")
    assert candidate.source_ledger_references == ("ledger-a", "ledger-b")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "wrong-model", "model_id mismatch"),
        ("discovery_method_id", "wrong-method", "discovery_method_id mismatch"),
        ("discovery_config_id", "wrong-config", "discovery_config_id mismatch"),
        ("source_budget_reference", "wrong-budget", "source_budget_reference mismatch"),
        ("fidelity_metric_reference", "wrong-metric", "fidelity_metric_reference mismatch"),
        ("component_basis_reference", "wrong-basis", "component_basis_reference mismatch"),
    ],
)
def test_context_identity_mismatches_are_rejected(field, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        _qualify([_exact_evidence(**{field: value})])


def test_bad_mask_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="mask identity mismatch"):
        _qualify([_exact_evidence(mask_identity="not-the-mask-identity")])


def test_unsealed_nonfinal_and_hash_inconsistent_evidence_are_rejected() -> None:
    with pytest.raises(ValueError, match="sealed"):
        _exact_evidence(sealed=False)

    with pytest.raises(ValueError, match="final exact"):
        _exact_evidence(final_exact_evaluation=False)

    with pytest.raises(ValueError, match="ledger hash mismatch"):
        _exact_evidence(recomputed_ledger_hash="b" * 64)


def test_nonfinite_and_malformed_exact_evidence_are_rejected() -> None:
    with pytest.raises(ValueError, match="exact_fidelity"):
        _exact_evidence(exact_fidelity=float("nan"))

    with pytest.raises(ValueError, match="516"):
        _exact_evidence(mask=(1, 0), mask_identity="irrelevant")


def test_duplicate_mask_with_inconsistent_exact_result_is_rejected() -> None:
    first = _exact_evidence(exact_fidelity=0.8, proposal="proposal-a")
    second = _exact_evidence(
        exact_fidelity=0.81,
        proposal="proposal-b",
        evaluation="eval-b",
    )

    with pytest.raises(ValueError, match="inconsistent final exact fidelity"):
        _qualify([first, second])


def _candidate(
    mask_identity: str,
    retained_components: tuple[int, ...],
) -> CandidateRecord:
    return CandidateRecord(
        model_id="synthetic-model",
        discovery_method_id="synthetic-method",
        discovery_config_id="synthetic-config",
        source_budget_reference=load_technical_policy(
            POLICY_PATH
        ).source_budget_reference,
        component_basis_reference=load_technical_policy(
            POLICY_PATH
        ).component_basis_reference,
        component_basis_size=516,
        mask_identity=mask_identity,
        retained_components=retained_components,
        exact_fidelity=0.8,
        proposal_references=(f"proposal-{mask_identity}",),
        exact_evaluation_references=(f"eval-{mask_identity}",),
        source_ledger_references=(f"ledger-{mask_identity}",),
    )


def test_overlap_identical_disjoint_and_empty_cases() -> None:
    from circuit_families.stage6e.packing import retained_jaccard_overlap

    policy = load_technical_policy(POLICY_PATH)

    a = _candidate("a", (0, 1))
    identical = _candidate("b", (0, 1))
    disjoint = _candidate("c", (2, 3))
    empty_a = _candidate("empty-a", ())
    empty_b = _candidate("empty-b", ())

    assert retained_jaccard_overlap(a, identical, policy) == 1.0
    assert retained_jaccard_overlap(a, disjoint, policy) == 0.0
    assert retained_jaccard_overlap(a, empty_a, policy) == 0.0
    assert retained_jaccard_overlap(empty_a, empty_b, policy) == 1.0


def test_overlap_boundary_equal_is_compatible_and_above_is_not() -> None:
    from circuit_families.stage6e.packing import (
        build_compatibility_graph,
        retained_jaccard_overlap,
    )

    policy = replace(
        load_technical_policy(POLICY_PATH),
        max_pairwise_overlap=0.25,
    )

    left = _candidate("left", (0, 1))
    equal = _candidate("equal", (1, 2, 3))
    above = _candidate("above", (1, 2))

    assert retained_jaccard_overlap(left, equal, policy) == 0.25
    assert retained_jaccard_overlap(left, above, policy) > 0.25

    graph = build_compatibility_graph((left, equal, above), policy)

    assert ("equal", "left") in graph.compatible_edges
    assert ("above", "left") not in graph.compatible_edges


def test_empty_and_singleton_graphs_are_defined() -> None:
    from circuit_families.stage6e.packing import build_compatibility_graph

    policy = load_technical_policy(POLICY_PATH)

    empty = build_compatibility_graph((), policy)
    assert empty.node_mask_identities == ()
    assert empty.compatible_edges == ()
    assert len(empty.graph_hash) == 64

    singleton = build_compatibility_graph((_candidate("only", (7,)),), policy)
    assert singleton.node_mask_identities == ("only",)
    assert singleton.compatible_edges == ()
    assert len(singleton.graph_hash) == 64


def test_graph_order_and_hash_are_proposal_order_invariant() -> None:
    from circuit_families.stage6e.packing import build_compatibility_graph

    policy = load_technical_policy(POLICY_PATH)
    a = _candidate("a", (0, 1))
    b = _candidate("b", (2, 3))
    c = _candidate("c", (1, 2))

    first = build_compatibility_graph((a, b, c), policy)
    second = build_compatibility_graph((c, a, b), policy)

    assert first.node_mask_identities == ("a", "b", "c")
    assert first.compatible_edges == second.compatible_edges
    assert first.input_hash == second.input_hash
    assert first.graph_hash == second.graph_hash


def test_graph_rejects_duplicate_mask_identity_input() -> None:
    from circuit_families.stage6e.packing import build_compatibility_graph

    policy = load_technical_policy(POLICY_PATH)
    first = _candidate("duplicate", (0,))
    second = _candidate("duplicate", (1,))

    with pytest.raises(ValueError, match="duplicate mask identities"):
        build_compatibility_graph((first, second), policy)


def test_overlap_rejects_basis_mismatch() -> None:
    from circuit_families.stage6e.packing import retained_jaccard_overlap

    policy = load_technical_policy(POLICY_PATH)
    valid = _candidate("valid", (0,))
    wrong = replace(valid, mask_identity="wrong", component_basis_reference="wrong-basis")

    with pytest.raises(ValueError, match="basis reference"):
        retained_jaccard_overlap(valid, wrong, policy)


def test_exact_solver_returns_maximum_not_greedy_order_result() -> None:
    from circuit_families.stage6e.packing import exact_maximum_compatible_subset

    policy = load_technical_policy(POLICY_PATH)
    graph = CompatibilityGraphRecord(
        policy_hash=policy.policy_hash,
        input_hash="input",
        node_mask_identities=("a", "b", "c", "d"),
        compatible_edges=(
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        ),
    )

    selected = exact_maximum_compatible_subset(graph, policy)

    assert selected == ("a", "b", "c")
    assert len(selected) == 3


def test_exact_solver_uses_lexicographically_smallest_maximum_tie() -> None:
    from circuit_families.stage6e.packing import exact_maximum_compatible_subset

    policy = load_technical_policy(POLICY_PATH)
    graph = CompatibilityGraphRecord(
        policy_hash=policy.policy_hash,
        input_hash="input",
        node_mask_identities=("d", "c", "b", "a"),
        compatible_edges=(
            ("a", "b"),
            ("c", "d"),
        ),
    )

    assert exact_maximum_compatible_subset(graph, policy) == ("a", "b")


def test_exact_solver_zero_and_singleton_are_defined() -> None:
    from circuit_families.stage6e.packing import exact_maximum_compatible_subset

    policy = load_technical_policy(POLICY_PATH)

    empty = CompatibilityGraphRecord(
        policy_hash=policy.policy_hash,
        input_hash="empty",
        node_mask_identities=(),
        compatible_edges=(),
    )
    singleton = CompatibilityGraphRecord(
        policy_hash=policy.policy_hash,
        input_hash="single",
        node_mask_identities=("only",),
        compatible_edges=(),
    )

    assert exact_maximum_compatible_subset(empty, policy) == ()
    assert exact_maximum_compatible_subset(singleton, policy) == ("only",)


def test_endpoint2_recomputation_matches_direct_result() -> None:
    from circuit_families.stage6e.packing import (
        build_compatibility_graph,
        build_endpoint2_result,
        recompute_endpoint2,
    )

    policy = replace(
        load_technical_policy(POLICY_PATH),
        max_pairwise_overlap=0.25,
    )

    qualification = QualificationResult(
        raw_candidate_count=3,
        unique_candidate_count=3,
        qualified_candidate_count=3,
        qualified_candidates=(
            _candidate("a", (0, 1)),
            _candidate("b", (2, 3)),
            _candidate("c", (1, 2)),
        ),
    )

    graph = build_compatibility_graph(
        qualification.qualified_candidates,
        policy,
    )
    direct = build_endpoint2_result(
        qualification,
        graph,
        policy,
    )
    recomputed = recompute_endpoint2(
        qualification,
        policy,
    )

    assert direct.to_record() == recomputed.to_record()
    assert direct.packing_lower_bound == 2
    assert direct.proof.selected_mask_identities == ("a", "b")
    assert direct.proof.graph_hash == direct.graph_hash
    assert direct.proof.input_hash == direct.input_hash


def test_endpoint2_zero_result_has_complete_proof_metadata() -> None:
    from circuit_families.stage6e.packing import recompute_endpoint2

    policy = load_technical_policy(POLICY_PATH)
    qualification = QualificationResult(
        raw_candidate_count=2,
        unique_candidate_count=2,
        qualified_candidate_count=0,
        qualified_candidates=(),
    )

    result = recompute_endpoint2(qualification, policy)

    assert result.packing_lower_bound == 0
    assert result.selected_members == ()
    assert result.proof.selected_mask_identities == ()
    assert result.proof.packing_lower_bound == 0
    assert len(result.input_hash) == 64
    assert len(result.graph_hash) == 64
    assert len(result.proof.proof_hash) == 64


def test_endpoint2_singleton_result_is_one() -> None:
    from circuit_families.stage6e.packing import recompute_endpoint2

    policy = load_technical_policy(POLICY_PATH)
    candidate = _candidate("only", (0, 1))

    qualification = QualificationResult(
        raw_candidate_count=1,
        unique_candidate_count=1,
        qualified_candidate_count=1,
        qualified_candidates=(candidate,),
    )

    result = recompute_endpoint2(qualification, policy)

    assert result.packing_lower_bound == 1
    assert tuple(
        member.mask_identity for member in result.selected_members
    ) == ("only",)


def test_endpoint2_is_permutation_invariant() -> None:
    from circuit_families.stage6e.packing import recompute_endpoint2

    policy = replace(
        load_technical_policy(POLICY_PATH),
        max_pairwise_overlap=0.25,
    )

    a = _candidate("a", (0, 1))
    b = _candidate("b", (2, 3))
    c = _candidate("c", (1, 2))

    first = recompute_endpoint2(
        QualificationResult(3, 3, 3, (a, b, c)),
        policy,
    )
    second = recompute_endpoint2(
        QualificationResult(3, 3, 3, (c, a, b)),
        policy,
    )

    assert first.to_record() == second.to_record()


def test_result_builder_rejects_graph_from_wrong_input() -> None:
    from circuit_families.stage6e.packing import (
        build_compatibility_graph,
        build_endpoint2_result,
    )

    policy = load_technical_policy(POLICY_PATH)
    a = _candidate("a", (0,))
    b = _candidate("b", (1,))

    qualification = QualificationResult(1, 1, 1, (a,))
    wrong_graph = build_compatibility_graph((b,), policy)

    with pytest.raises(ValueError, match="input hash"):
        build_endpoint2_result(qualification, wrong_graph, policy)


def test_candidate_and_graph_records_round_trip_as_canonical_records() -> None:
    candidate = _candidate("roundtrip", (3, 1, 2))
    candidate_record = candidate.to_record()

    assert candidate_record["retained_components"] == [1, 2, 3]
    assert candidate_record["candidate_hash"] == candidate.candidate_hash

    policy = load_technical_policy(POLICY_PATH)
    from circuit_families.stage6e.packing import build_compatibility_graph

    graph = build_compatibility_graph((candidate,), policy)
    graph_record = graph.to_record()

    assert graph_record["node_mask_identities"] == ["roundtrip"]
    assert graph_record["graph_hash"] == graph.graph_hash


def test_procedure_lower_bound_wording_excludes_global_claim() -> None:
    assert PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS == (
        "procedure_dependent_packing_lower_bound"
    )
    assert "global" not in PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS
    assert "true_packing" not in PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS


def test_result_proof_rejects_wrong_hash_linkage() -> None:
    member = SelectedMemberRecord(
        mask_identity="a",
        candidate_hash="candidate",
        proposal_references=("p",),
        exact_evaluation_references=("e",),
    )
    proof = PackingProofRecord(
        policy_hash="policy-a",
        input_hash="input",
        graph_hash="graph",
        solver_reference="solver",
        tie_break_reference="tie",
        recomputation_reference="recompute",
        selected_mask_identities=("a",),
        packing_lower_bound=1,
    )

    with pytest.raises(ValueError, match="policy hash"):
        Endpoint2ResultRecord(
            policy_hash="policy-b",
            input_hash="input",
            graph_hash="graph",
            raw_candidate_count=1,
            unique_candidate_count=1,
            qualified_candidate_count=1,
            selected_members=(member,),
            packing_lower_bound=1,
            proof=proof,
        )


def test_validate_cli_is_cwd_independent_and_read_only(tmp_path) -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    cli = root / "scripts/validate_stage6e_endpoint2.py"

    before = {
        p: p.stat().st_mtime_ns
        for p in root.glob("followup/configs/stage6e/*")
        if p.is_file()
    }

    proc = subprocess.run(
        [sys.executable, str(cli), "--validate-only"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "policy_hash=" in proc.stdout
    assert "input_hash=" in proc.stdout
    assert "graph_hash=" in proc.stdout
    assert "raw_count=" in proc.stdout
    assert "unique_count=" in proc.stdout
    assert "qualified_count=" in proc.stdout
    assert "edge_count=" in proc.stdout
    assert "selected_members=" in proc.stdout
    assert "packing_lower_bound=" in proc.stdout
    assert "recomputation=PASS" in proc.stdout
    assert "semantics=procedure_dependent_packing_lower_bound" in proc.stdout
    assert "STATUS=PASS" in proc.stdout

    after = {
        p: p.stat().st_mtime_ns
        for p in root.glob("followup/configs/stage6e/*")
        if p.is_file()
    }
    assert before == after


def test_endpoint2_result_serialization_and_reconstruction() -> None:
    from circuit_families.stage6e import endpoint2_result_from_record
    from circuit_families.stage6e.packing import recompute_endpoint2

    policy = load_technical_policy(POLICY_PATH)
    candidate = _candidate("serial", (0, 2))

    result = recompute_endpoint2(
        QualificationResult(1, 1, 1, (candidate,)),
        policy,
    )
    record = result.to_record()
    restored = endpoint2_result_from_record(record)

    assert restored.to_record() == record

    corrupted = dict(record)
    corrupted["result_hash"] = "0" * 64
    with pytest.raises(ValueError, match="result hash mismatch"):
        endpoint2_result_from_record(corrupted)


def test_stage4_endpoint_payload_preserves_endpoint2_claim_boundary() -> None:
    from circuit_families.stage6e import (
        endpoint2_result_to_stage4_endpoint_record,
    )
    from circuit_families.stage6e.packing import recompute_endpoint2

    policy = load_technical_policy(POLICY_PATH)
    result = recompute_endpoint2(
        QualificationResult(
            1,
            1,
            1,
            (_candidate("stage4", (0, 1)),),
        ),
        policy,
    )

    payload = endpoint2_result_to_stage4_endpoint_record(result)

    assert payload["endpoint_name"] == "endpoint_2"
    assert payload["endpoint_semantics"] == (
        "procedure_dependent_packing_lower_bound"
    )
    assert payload["value"] == 1
    assert payload["policy_hash"] == result.policy_hash
    assert payload["input_hash"] == result.input_hash
    assert payload["graph_hash"] == result.graph_hash
    assert payload["proof_hash"] == result.proof.proof_hash
    assert payload["result_hash"] == result.result_hash


def test_technical_e2e_duplicate_permutation_tie_and_zero() -> None:
    from circuit_families.stage6a.models import canonical_mask_identity
    from circuit_families.stage6e import ExactCandidateEvidence
    from circuit_families.stage6e.packing import (
        qualify_and_deduplicate,
        recompute_endpoint2,
    )

    policy = replace(
        load_technical_policy(POLICY_PATH),
        max_pairwise_overlap=0.0,
    )

    def evidence(name, retained, fidelity=0.8):
        mask = tuple(1 if i in retained else 0 for i in range(516))
        return ExactCandidateEvidence(
            model_id="synthetic-model",
            discovery_method_id="synthetic-method",
            discovery_config_id="synthetic-config",
            source_budget_reference=policy.source_budget_reference,
            fidelity_metric_reference=policy.fidelity_metric_reference,
            component_basis_reference=policy.component_basis_reference,
            component_basis_size=516,
            mask=mask,
            mask_identity=canonical_mask_identity(tuple(retained)),
            exact_fidelity=fidelity,
            proposal_reference=f"proposal-{name}",
            exact_evaluation_reference=f"eval-{name}",
            source_ledger_reference=f"ledger-{name}",
            source_ledger_hash="a" * 64,
            recomputed_ledger_hash="a" * 64,
        )

    a = evidence("a", (0,))
    a_dup = evidence("a-dup", (0,))
    b = evidence("b", (1,))
    c = evidence("c", (0, 1))

    first_q = qualify_and_deduplicate(
        (a, a_dup, b, c),
        policy,
        model_id="synthetic-model",
        discovery_method_id="synthetic-method",
        discovery_config_id="synthetic-config",
    )
    second_q = qualify_and_deduplicate(
        (c, b, a_dup, a),
        policy,
        model_id="synthetic-model",
        discovery_method_id="synthetic-method",
        discovery_config_id="synthetic-config",
    )

    first = recompute_endpoint2(first_q, policy)
    second = recompute_endpoint2(second_q, policy)

    assert first.to_record() == second.to_record()
    assert first_q.raw_candidate_count == 4
    assert first_q.unique_candidate_count == 3
    assert first.packing_lower_bound == 2

    zero_q = qualify_and_deduplicate(
        (evidence("low", (3,), fidelity=0.1),),
        policy,
        model_id="synthetic-model",
        discovery_method_id="synthetic-method",
        discovery_config_id="synthetic-config",
    )
    zero = recompute_endpoint2(zero_q, policy)

    assert zero.qualified_candidate_count == 0
    assert zero.packing_lower_bound == 0


def test_hash_seed_determinism_for_technical_fixture(tmp_path) -> None:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    cli = root / "scripts/validate_stage6e_endpoint2.py"

    outputs = []
    for seed in ("1", "777"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, str(cli), "--validate-only"],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        outputs.append(proc.stdout)

    assert outputs[0] == outputs[1]
