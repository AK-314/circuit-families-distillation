"""Stage 6E technical Endpoint 2 policy and deterministic record contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLICY_SCHEMA_VERSION = "stage6e-technical-endpoint2-policy/v1"
PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS = "procedure_dependent_packing_lower_bound"
TECHNICAL_POLICY_KIND = "technical_fixture"
COMMON_COMPONENT_BASIS_SIZE = 516


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_unit_interval(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class TechnicalEndpoint2Policy:
    schema_version: str
    policy_name: str
    policy_kind: str
    scientific_data: bool
    production_default: bool
    resolves_unresolved_decisions: tuple[str, ...]
    fidelity_metric_reference: str
    fidelity_threshold: float
    component_basis_reference: str
    component_basis_size: int
    component_cap_reference: str
    max_component_proportion: float
    overlap_rule_reference: str
    max_pairwise_overlap: float
    solver_reference: str
    tie_break_reference: str
    source_budget_reference: str

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 6E technical policy schema")
        _require_string("policy_name", self.policy_name)
        if self.policy_kind != TECHNICAL_POLICY_KIND:
            raise ValueError("Stage 6E policy must be explicitly technical_fixture")
        if self.scientific_data is not False:
            raise ValueError("Stage 6E technical policy must declare scientific_data=false")
        if self.production_default is not False:
            raise ValueError("Stage 6E must not define a production default")
        if self.resolves_unresolved_decisions:
            raise ValueError("Stage 6E technical policy must not resolve any UD")
        if self.component_basis_size != COMMON_COMPONENT_BASIS_SIZE:
            raise ValueError("Stage 6E requires the common 516-component basis")

        for name in (
            "fidelity_metric_reference",
            "component_basis_reference",
            "component_cap_reference",
            "overlap_rule_reference",
            "solver_reference",
            "tie_break_reference",
            "source_budget_reference",
        ):
            _require_string(name, getattr(self, name))

        _require_unit_interval("fidelity_threshold", self.fidelity_threshold)
        _require_unit_interval(
            "max_component_proportion",
            self.max_component_proportion,
        )
        _require_unit_interval("max_pairwise_overlap", self.max_pairwise_overlap)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_name": self.policy_name,
            "policy_kind": self.policy_kind,
            "scientific_data": self.scientific_data,
            "production_default": self.production_default,
            "resolves_unresolved_decisions": list(self.resolves_unresolved_decisions),
            "fidelity_metric_reference": self.fidelity_metric_reference,
            "fidelity_threshold": self.fidelity_threshold,
            "component_basis_reference": self.component_basis_reference,
            "component_basis_size": self.component_basis_size,
            "component_cap_reference": self.component_cap_reference,
            "max_component_proportion": self.max_component_proportion,
            "overlap_rule_reference": self.overlap_rule_reference,
            "max_pairwise_overlap": self.max_pairwise_overlap,
            "solver_reference": self.solver_reference,
            "tie_break_reference": self.tie_break_reference,
            "source_budget_reference": self.source_budget_reference,
        }

    @property
    def policy_hash(self) -> str:
        return _content_hash(self._payload())

    @property
    def policy_id(self) -> str:
        return f"stage6e-policy-{self.policy_hash[:16]}"

    def to_record(self) -> dict[str, object]:
        record = self._payload()
        record["policy_hash"] = self.policy_hash
        record["policy_id"] = self.policy_id
        return record


@dataclass(frozen=True)
class CandidateRecord:
    model_id: str
    discovery_method_id: str
    discovery_config_id: str
    source_budget_reference: str
    component_basis_reference: str
    component_basis_size: int
    mask_identity: str
    retained_components: tuple[int, ...]
    exact_fidelity: float
    proposal_references: tuple[str, ...]
    exact_evaluation_references: tuple[str, ...]
    source_ledger_references: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "discovery_method_id",
            "discovery_config_id",
            "source_budget_reference",
            "component_basis_reference",
            "mask_identity",
        ):
            _require_string(name, getattr(self, name))

        if self.component_basis_size != COMMON_COMPONENT_BASIS_SIZE:
            raise ValueError("candidate must use the common 516-component basis")
        _require_unit_interval("exact_fidelity", self.exact_fidelity)

        components = tuple(sorted(self.retained_components))
        if len(set(components)) != len(components):
            raise ValueError("retained_components must be unique")
        if any(
            not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= 516
            for i in components
        ):
            raise ValueError("retained component index outside common basis")
        object.__setattr__(self, "retained_components", components)

        for field_name in (
            "proposal_references",
            "exact_evaluation_references",
            "source_ledger_references",
        ):
            values = tuple(sorted(getattr(self, field_name)))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{field_name} must be non-empty and unique")
            for value in values:
                _require_string(field_name, value)
            object.__setattr__(self, field_name, values)

    def _payload(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "discovery_method_id": self.discovery_method_id,
            "discovery_config_id": self.discovery_config_id,
            "source_budget_reference": self.source_budget_reference,
            "component_basis_reference": self.component_basis_reference,
            "component_basis_size": self.component_basis_size,
            "mask_identity": self.mask_identity,
            "retained_components": list(self.retained_components),
            "exact_fidelity": self.exact_fidelity,
            "proposal_references": list(self.proposal_references),
            "exact_evaluation_references": list(self.exact_evaluation_references),
            "source_ledger_references": list(self.source_ledger_references),
        }

    @property
    def candidate_hash(self) -> str:
        return _content_hash(self._payload())

    def to_record(self) -> dict[str, object]:
        record = self._payload()
        record["candidate_hash"] = self.candidate_hash
        return record


@dataclass(frozen=True)
class CompatibilityGraphRecord:
    policy_hash: str
    input_hash: str
    node_mask_identities: tuple[str, ...]
    compatible_edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_string("policy_hash", self.policy_hash)
        _require_string("input_hash", self.input_hash)

        nodes = tuple(sorted(self.node_mask_identities))
        if len(set(nodes)) != len(nodes):
            raise ValueError("graph nodes must be unique")
        for node in nodes:
            _require_string("node_mask_identity", node)

        node_set = set(nodes)
        canonical_edges: list[tuple[str, str]] = []
        for edge in self.compatible_edges:
            if len(edge) != 2:
                raise ValueError("graph edge must contain exactly two nodes")
            left, right = sorted(edge)
            if left == right:
                raise ValueError("self edges are not permitted")
            if left not in node_set or right not in node_set:
                raise ValueError("graph edge references an unknown node")
            canonical_edges.append((left, right))

        edges = tuple(sorted(canonical_edges))
        if len(set(edges)) != len(edges):
            raise ValueError("graph edges must be unique")

        object.__setattr__(self, "node_mask_identities", nodes)
        object.__setattr__(self, "compatible_edges", edges)

    def _payload(self) -> dict[str, object]:
        return {
            "policy_hash": self.policy_hash,
            "input_hash": self.input_hash,
            "node_mask_identities": list(self.node_mask_identities),
            "compatible_edges": [list(edge) for edge in self.compatible_edges],
        }

    @property
    def graph_hash(self) -> str:
        return _content_hash(self._payload())

    def to_record(self) -> dict[str, object]:
        record = self._payload()
        record["graph_hash"] = self.graph_hash
        return record


@dataclass(frozen=True)
class SelectedMemberRecord:
    mask_identity: str
    candidate_hash: str
    proposal_references: tuple[str, ...]
    exact_evaluation_references: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_string("mask_identity", self.mask_identity)
        _require_string("candidate_hash", self.candidate_hash)

        for field_name in ("proposal_references", "exact_evaluation_references"):
            values = tuple(sorted(getattr(self, field_name)))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{field_name} must be non-empty and unique")
            for value in values:
                _require_string(field_name, value)
            object.__setattr__(self, field_name, values)

    def _payload(self) -> dict[str, object]:
        return {
            "mask_identity": self.mask_identity,
            "candidate_hash": self.candidate_hash,
            "proposal_references": list(self.proposal_references),
            "exact_evaluation_references": list(self.exact_evaluation_references),
        }

    @property
    def member_hash(self) -> str:
        return _content_hash(self._payload())

    def to_record(self) -> dict[str, object]:
        record = self._payload()
        record["member_hash"] = self.member_hash
        return record


@dataclass(frozen=True)
class PackingProofRecord:
    policy_hash: str
    input_hash: str
    graph_hash: str
    solver_reference: str
    tie_break_reference: str
    recomputation_reference: str
    selected_mask_identities: tuple[str, ...]
    packing_lower_bound: int

    def __post_init__(self) -> None:
        for name in (
            "policy_hash",
            "input_hash",
            "graph_hash",
            "solver_reference",
            "tie_break_reference",
            "recomputation_reference",
        ):
            _require_string(name, getattr(self, name))

        selected = tuple(sorted(self.selected_mask_identities))
        if len(set(selected)) != len(selected):
            raise ValueError("selected mask identities must be unique")
        if self.packing_lower_bound != len(selected):
            raise ValueError("packing_lower_bound must equal selected-member count")
        object.__setattr__(self, "selected_mask_identities", selected)

    def _payload(self) -> dict[str, object]:
        return {
            "policy_hash": self.policy_hash,
            "input_hash": self.input_hash,
            "graph_hash": self.graph_hash,
            "solver_reference": self.solver_reference,
            "tie_break_reference": self.tie_break_reference,
            "recomputation_reference": self.recomputation_reference,
            "selected_mask_identities": list(self.selected_mask_identities),
            "packing_lower_bound": self.packing_lower_bound,
        }

    @property
    def proof_hash(self) -> str:
        return _content_hash(self._payload())

    def to_record(self) -> dict[str, object]:
        record = self._payload()
        record["proof_hash"] = self.proof_hash
        return record


@dataclass(frozen=True)
class Endpoint2ResultRecord:
    policy_hash: str
    input_hash: str
    graph_hash: str
    raw_candidate_count: int
    unique_candidate_count: int
    qualified_candidate_count: int
    selected_members: tuple[SelectedMemberRecord, ...]
    packing_lower_bound: int
    proof: PackingProofRecord
    semantics: str = PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS

    def __post_init__(self) -> None:
        for name in ("policy_hash", "input_hash", "graph_hash"):
            _require_string(name, getattr(self, name))

        if self.semantics != PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS:
            raise ValueError("Endpoint 2 semantics must remain procedure-relative")

        counts = (
            self.raw_candidate_count,
            self.unique_candidate_count,
            self.qualified_candidate_count,
            self.packing_lower_bound,
        )
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts):
            raise ValueError("Endpoint 2 counts must be non-negative integers")
        if not (
            self.raw_candidate_count
            >= self.unique_candidate_count
            >= self.qualified_candidate_count
            >= self.packing_lower_bound
        ):
            raise ValueError("Endpoint 2 candidate counts are inconsistent")

        members = tuple(sorted(self.selected_members, key=lambda member: member.mask_identity))
        if len({member.mask_identity for member in members}) != len(members):
            raise ValueError("selected members must have unique mask identities")
        if len(members) != self.packing_lower_bound:
            raise ValueError("packing_lower_bound must equal selected-member count")
        object.__setattr__(self, "selected_members", members)

        if self.proof.policy_hash != self.policy_hash:
            raise ValueError("proof policy hash does not match result")
        if self.proof.input_hash != self.input_hash:
            raise ValueError("proof input hash does not match result")
        if self.proof.graph_hash != self.graph_hash:
            raise ValueError("proof graph hash does not match result")
        if self.proof.packing_lower_bound != self.packing_lower_bound:
            raise ValueError("proof packing count does not match result")
        if self.proof.selected_mask_identities != tuple(
            member.mask_identity for member in members
        ):
            raise ValueError("proof selected identities do not match result members")

    def _payload(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "policy_hash": self.policy_hash,
            "input_hash": self.input_hash,
            "graph_hash": self.graph_hash,
            "raw_candidate_count": self.raw_candidate_count,
            "unique_candidate_count": self.unique_candidate_count,
            "qualified_candidate_count": self.qualified_candidate_count,
            "selected_members": [member.to_record() for member in self.selected_members],
            "packing_lower_bound": self.packing_lower_bound,
            "proof": self.proof.to_record(),
        }

    @property
    def result_hash(self) -> str:
        return _content_hash(self._payload())

    def to_record(self) -> dict[str, object]:
        record = self._payload()
        record["result_hash"] = self.result_hash
        return record


def technical_policy_from_record(record: Mapping[str, Any]) -> TechnicalEndpoint2Policy:
    return TechnicalEndpoint2Policy(
        schema_version=record["schema_version"],
        policy_name=record["policy_name"],
        policy_kind=record["policy_kind"],
        scientific_data=record["scientific_data"],
        production_default=record["production_default"],
        resolves_unresolved_decisions=tuple(record["resolves_unresolved_decisions"]),
        fidelity_metric_reference=record["fidelity_metric_reference"],
        fidelity_threshold=record["fidelity_threshold"],
        component_basis_reference=record["component_basis_reference"],
        component_basis_size=record["component_basis_size"],
        component_cap_reference=record["component_cap_reference"],
        max_component_proportion=record["max_component_proportion"],
        overlap_rule_reference=record["overlap_rule_reference"],
        max_pairwise_overlap=record["max_pairwise_overlap"],
        solver_reference=record["solver_reference"],
        tie_break_reference=record["tie_break_reference"],
        source_budget_reference=record["source_budget_reference"],
    )


def load_technical_policy(path: str | Path) -> TechnicalEndpoint2Policy:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("technical Endpoint 2 policy must be a JSON object")
    return technical_policy_from_record(record)


@dataclass(frozen=True)
class ExactCandidateEvidence:
    """Normalized sealed final-exact evidence consumed by Endpoint 2."""

    model_id: str
    discovery_method_id: str
    discovery_config_id: str
    source_budget_reference: str
    fidelity_metric_reference: str
    component_basis_reference: str
    component_basis_size: int
    mask: tuple[int, ...]
    mask_identity: str
    exact_fidelity: float
    proposal_reference: str
    exact_evaluation_reference: str
    source_ledger_reference: str
    source_ledger_hash: str
    recomputed_ledger_hash: str
    sealed: bool = True
    final_exact_evaluation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "discovery_method_id",
            "discovery_config_id",
            "source_budget_reference",
            "fidelity_metric_reference",
            "component_basis_reference",
            "mask_identity",
            "proposal_reference",
            "exact_evaluation_reference",
            "source_ledger_reference",
            "source_ledger_hash",
            "recomputed_ledger_hash",
        ):
            _require_string(name, getattr(self, name))

        if self.component_basis_size != COMMON_COMPONENT_BASIS_SIZE:
            raise ValueError("exact evidence must use the common 516-component basis")
        if len(self.mask) != COMMON_COMPONENT_BASIS_SIZE:
            raise ValueError("exact evidence mask must contain exactly 516 entries")
        if any(
            not isinstance(bit, int) or isinstance(bit, bool) or bit not in (0, 1)
            for bit in self.mask
        ):
            raise ValueError("exact evidence mask must be a binary 516-vector")

        _require_unit_interval("exact_fidelity", self.exact_fidelity)

        if self.sealed is not True:
            raise ValueError("Endpoint 2 requires sealed exact-evaluation evidence")
        if self.final_exact_evaluation is not True:
            raise ValueError("Endpoint 2 requires final exact evaluation")
        if self.source_ledger_hash != self.recomputed_ledger_hash:
            raise ValueError("exact-evaluation ledger hash mismatch")


@dataclass(frozen=True)
class QualificationResult:
    raw_candidate_count: int
    unique_candidate_count: int
    qualified_candidate_count: int
    qualified_candidates: tuple[CandidateRecord, ...]

    def __post_init__(self) -> None:
        counts = (
            self.raw_candidate_count,
            self.unique_candidate_count,
            self.qualified_candidate_count,
        )
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts):
            raise ValueError("qualification counts must be non-negative integers")
        if not (
            self.raw_candidate_count
            >= self.unique_candidate_count
            >= self.qualified_candidate_count
        ):
            raise ValueError("qualification counts are inconsistent")
        if len(self.qualified_candidates) != self.qualified_candidate_count:
            raise ValueError("qualified count does not match candidate records")


def selected_member_from_record(
    record: Mapping[str, Any],
) -> SelectedMemberRecord:
    member = SelectedMemberRecord(
        mask_identity=record["mask_identity"],
        candidate_hash=record["candidate_hash"],
        proposal_references=tuple(record["proposal_references"]),
        exact_evaluation_references=tuple(
            record["exact_evaluation_references"]
        ),
    )
    if record.get("member_hash") != member.member_hash:
        raise ValueError("selected-member hash mismatch")
    return member


def packing_proof_from_record(
    record: Mapping[str, Any],
) -> PackingProofRecord:
    proof = PackingProofRecord(
        policy_hash=record["policy_hash"],
        input_hash=record["input_hash"],
        graph_hash=record["graph_hash"],
        solver_reference=record["solver_reference"],
        tie_break_reference=record["tie_break_reference"],
        recomputation_reference=record["recomputation_reference"],
        selected_mask_identities=tuple(record["selected_mask_identities"]),
        packing_lower_bound=record["packing_lower_bound"],
    )
    if record.get("proof_hash") != proof.proof_hash:
        raise ValueError("packing-proof hash mismatch")
    return proof


def endpoint2_result_from_record(
    record: Mapping[str, Any],
) -> Endpoint2ResultRecord:
    result = Endpoint2ResultRecord(
        semantics=record["semantics"],
        policy_hash=record["policy_hash"],
        input_hash=record["input_hash"],
        graph_hash=record["graph_hash"],
        raw_candidate_count=record["raw_candidate_count"],
        unique_candidate_count=record["unique_candidate_count"],
        qualified_candidate_count=record["qualified_candidate_count"],
        selected_members=tuple(
            selected_member_from_record(item)
            for item in record["selected_members"]
        ),
        packing_lower_bound=record["packing_lower_bound"],
        proof=packing_proof_from_record(record["proof"]),
    )
    if record.get("result_hash") != result.result_hash:
        raise ValueError("Endpoint 2 result hash mismatch")
    return result


def endpoint2_result_to_stage4_endpoint_record(
    result: Endpoint2ResultRecord,
) -> dict[str, object]:
    """Return the Stage 4 endpoint-record payload for Endpoint 2 evidence."""

    return {
        "endpoint_name": "endpoint_2",
        "endpoint_semantics": result.semantics,
        "value": result.packing_lower_bound,
        "policy_hash": result.policy_hash,
        "input_hash": result.input_hash,
        "graph_hash": result.graph_hash,
        "proof_hash": result.proof.proof_hash,
        "result_hash": result.result_hash,
        "recomputation_reference": result.proof.recomputation_reference,
    }
