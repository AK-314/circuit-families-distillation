"""Qualification and deduplication for Stage 6E Endpoint 2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from circuit_families.stage6a.models import (
    canonical_mask_identity,
    retained_proportion,
)
from circuit_families.stage6e.records import (
    CandidateRecord,
    CompatibilityGraphRecord,
    Endpoint2ResultRecord,
    ExactCandidateEvidence,
    PackingProofRecord,
    QualificationResult,
    SelectedMemberRecord,
    TechnicalEndpoint2Policy,
)


def _validate_context(
    evidence: ExactCandidateEvidence,
    policy: TechnicalEndpoint2Policy,
    *,
    model_id: str,
    discovery_method_id: str,
    discovery_config_id: str,
) -> None:
    expected = {
        "model_id": model_id,
        "discovery_method_id": discovery_method_id,
        "discovery_config_id": discovery_config_id,
        "source_budget_reference": policy.source_budget_reference,
        "fidelity_metric_reference": policy.fidelity_metric_reference,
        "component_basis_reference": policy.component_basis_reference,
        "component_basis_size": policy.component_basis_size,
    }

    for field_name, expected_value in expected.items():
        if getattr(evidence, field_name) != expected_value:
            raise ValueError(f"exact evidence {field_name} mismatch")

    retained_components = tuple(
        index for index, retained in enumerate(evidence.mask) if retained
    )
    recomputed_mask_identity = canonical_mask_identity(retained_components)
    if evidence.mask_identity != recomputed_mask_identity:
        raise ValueError("exact evidence mask identity mismatch")


def qualify_and_deduplicate(
    evidence_records: Iterable[ExactCandidateEvidence],
    policy: TechnicalEndpoint2Policy,
    *,
    model_id: str,
    discovery_method_id: str,
    discovery_config_id: str,
) -> QualificationResult:
    """Validate, identity-deduplicate, and qualify final exact evidence."""

    records = tuple(evidence_records)

    grouped: dict[str, list[ExactCandidateEvidence]] = defaultdict(list)
    for evidence in records:
        _validate_context(
            evidence,
            policy,
            model_id=model_id,
            discovery_method_id=discovery_method_id,
            discovery_config_id=discovery_config_id,
        )
        grouped[evidence.mask_identity].append(evidence)

    unique_candidates: list[CandidateRecord] = []

    for mask_identity in sorted(grouped):
        group = grouped[mask_identity]
        first = group[0]

        for duplicate in group[1:]:
            if duplicate.mask != first.mask:
                raise ValueError("identical mask identity maps to different masks")
            if duplicate.exact_fidelity != first.exact_fidelity:
                raise ValueError(
                    "duplicate mask has inconsistent final exact fidelity"
                )

        retained_components = tuple(
            index for index, retained in enumerate(first.mask) if retained
        )

        candidate = CandidateRecord(
            model_id=first.model_id,
            discovery_method_id=first.discovery_method_id,
            discovery_config_id=first.discovery_config_id,
            source_budget_reference=first.source_budget_reference,
            component_basis_reference=first.component_basis_reference,
            component_basis_size=first.component_basis_size,
            mask_identity=first.mask_identity,
            retained_components=retained_components,
            exact_fidelity=first.exact_fidelity,
            proposal_references=tuple(
                sorted({item.proposal_reference for item in group})
            ),
            exact_evaluation_references=tuple(
                sorted({item.exact_evaluation_reference for item in group})
            ),
            source_ledger_references=tuple(
                sorted({item.source_ledger_reference for item in group})
            ),
        )
        unique_candidates.append(candidate)

    qualified = tuple(
        candidate
        for candidate in unique_candidates
        if candidate.exact_fidelity >= policy.fidelity_threshold
        and retained_proportion(len(candidate.retained_components))
        <= policy.max_component_proportion
    )

    return QualificationResult(
        raw_candidate_count=len(records),
        unique_candidate_count=len(unique_candidates),
        qualified_candidate_count=len(qualified),
        qualified_candidates=qualified,
    )


def retained_jaccard_overlap(
    left: CandidateRecord,
    right: CandidateRecord,
    policy: TechnicalEndpoint2Policy,
) -> float:
    """Return technical retained-component Jaccard overlap in the common basis."""

    if policy.overlap_rule_reference != "jaccard-retained-components/technical-v1":
        raise ValueError("unsupported technical overlap rule")

    for candidate in (left, right):
        if candidate.component_basis_size != policy.component_basis_size:
            raise ValueError("candidate component basis size does not match policy")
        if candidate.component_basis_reference != policy.component_basis_reference:
            raise ValueError("candidate component basis reference does not match policy")

    left_set = set(left.retained_components)
    right_set = set(right.retained_components)
    union = left_set | right_set

    if not union:
        return 1.0

    return len(left_set & right_set) / len(union)


def qualified_input_hash(
    candidates: Iterable[CandidateRecord],
) -> str:
    """Canonical content hash for the finite qualified candidate set."""

    import hashlib
    import json

    ordered = sorted(candidates, key=lambda candidate: candidate.mask_identity)

    identities = [candidate.mask_identity for candidate in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("qualified candidates contain duplicate mask identities")

    payload = [candidate.to_record() for candidate in ordered]
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_compatibility_graph(
    candidates: Iterable[CandidateRecord],
    policy: TechnicalEndpoint2Policy,
) -> CompatibilityGraphRecord:
    """Build the deterministic compatibility graph for qualified candidates."""

    candidate_tuple = tuple(candidates)
    ordered = tuple(
        sorted(candidate_tuple, key=lambda candidate: candidate.mask_identity)
    )

    identities = tuple(candidate.mask_identity for candidate in ordered)
    if len(set(identities)) != len(identities):
        raise ValueError("qualified candidates contain duplicate mask identities")

    edges: list[tuple[str, str]] = []
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            overlap = retained_jaccard_overlap(left, right, policy)
            if overlap <= policy.max_pairwise_overlap:
                edges.append((left.mask_identity, right.mask_identity))

    return CompatibilityGraphRecord(
        policy_hash=policy.policy_hash,
        input_hash=qualified_input_hash(ordered),
        node_mask_identities=identities,
        compatible_edges=tuple(edges),
    )


def exact_maximum_compatible_subset(
    graph: CompatibilityGraphRecord,
    policy: TechnicalEndpoint2Policy,
) -> tuple[str, ...]:
    """Return an exact maximum clique with deterministic lexical tie-breaking."""

    if graph.policy_hash != policy.policy_hash:
        raise ValueError("graph policy hash does not match policy")
    if policy.solver_reference != "exact-maximum-compatible-subset/technical-v1":
        raise ValueError("unsupported technical packing solver")
    if (
        policy.tie_break_reference
        != "lexicographically-smallest-mask-identity-tuple/v1"
    ):
        raise ValueError("unsupported technical packing tie-break")

    nodes = graph.node_mask_identities
    edge_set = {frozenset(edge) for edge in graph.compatible_edges}

    best: tuple[str, ...] = ()

    def search(
        chosen: tuple[str, ...],
        remaining: tuple[str, ...],
    ) -> None:
        nonlocal best

        if len(chosen) + len(remaining) < len(best):
            return

        if not remaining:
            if len(chosen) > len(best) or (
                len(chosen) == len(best) and chosen < best
            ):
                best = chosen
            return

        node = remaining[0]
        tail = remaining[1:]

        compatible_tail = tuple(
            other
            for other in tail
            if frozenset((node, other)) in edge_set
            and all(
                frozenset((selected, other)) in edge_set
                for selected in chosen
            )
        )

        if all(
            frozenset((selected, node)) in edge_set
            for selected in chosen
        ):
            search(chosen + (node,), compatible_tail)

        search(chosen, tail)

    search((), nodes)
    return best


def build_endpoint2_result(
    qualification: QualificationResult,
    graph: CompatibilityGraphRecord,
    policy: TechnicalEndpoint2Policy,
) -> Endpoint2ResultRecord:
    """Build Endpoint 2 and sufficient proof metadata from qualified evidence."""

    qualified = qualification.qualified_candidates

    expected_input_hash = qualified_input_hash(qualified)
    if graph.input_hash != expected_input_hash:
        raise ValueError("graph input hash does not match qualified candidates")
    if graph.policy_hash != policy.policy_hash:
        raise ValueError("graph policy hash does not match policy")
    if graph.node_mask_identities != tuple(
        sorted(candidate.mask_identity for candidate in qualified)
    ):
        raise ValueError("graph nodes do not match qualified candidates")

    selected_ids = exact_maximum_compatible_subset(graph, policy)
    by_identity = {
        candidate.mask_identity: candidate
        for candidate in qualified
    }

    selected_members = tuple(
        SelectedMemberRecord(
            mask_identity=mask_identity,
            candidate_hash=by_identity[mask_identity].candidate_hash,
            proposal_references=by_identity[mask_identity].proposal_references,
            exact_evaluation_references=(
                by_identity[mask_identity].exact_evaluation_references
            ),
        )
        for mask_identity in selected_ids
    )

    proof = PackingProofRecord(
        policy_hash=policy.policy_hash,
        input_hash=graph.input_hash,
        graph_hash=graph.graph_hash,
        solver_reference=policy.solver_reference,
        tie_break_reference=policy.tie_break_reference,
        recomputation_reference="stage6e-endpoint2-recompute/technical-v1",
        selected_mask_identities=selected_ids,
        packing_lower_bound=len(selected_ids),
    )

    return Endpoint2ResultRecord(
        policy_hash=policy.policy_hash,
        input_hash=graph.input_hash,
        graph_hash=graph.graph_hash,
        raw_candidate_count=qualification.raw_candidate_count,
        unique_candidate_count=qualification.unique_candidate_count,
        qualified_candidate_count=qualification.qualified_candidate_count,
        selected_members=selected_members,
        packing_lower_bound=len(selected_members),
        proof=proof,
    )


def recompute_endpoint2(
    qualification: QualificationResult,
    policy: TechnicalEndpoint2Policy,
) -> Endpoint2ResultRecord:
    """Independently reconstruct graph, exact packing, and proof."""

    graph = build_compatibility_graph(
        qualification.qualified_candidates,
        policy,
    )
    return build_endpoint2_result(
        qualification,
        graph,
        policy,
    )
