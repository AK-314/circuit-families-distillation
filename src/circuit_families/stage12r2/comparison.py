"""Basis-aware Stage 12-R2 technical comparison guards and reducers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from circuit_families.stage12r2.accounting import BasisAccounting
from circuit_families.stage12r2.contracts import BasisContract

OutcomeState = Literal["evaluated", "failed", "unavailable", "censored"]
AccountingMeasure = Literal[
    "raw_component_proportion",
    "parameter_weighted_proportion",
    "type_stratified",
    "parent_coverage",
]


@dataclass(frozen=True)
class ExactLedgerEvidence:
    ledger_reference: str
    mask_identity: str
    basis_hash: str
    model_identity: str
    dense_reference_identity: str
    fidelity_definition_identity: str
    evaluation_domain_identity: str
    intervention_protocol_identity: str
    state: OutcomeState
    exact_fidelity: float | None
    intact_mask: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "ledger_reference",
            "mask_identity",
            "basis_hash",
            "model_identity",
            "dense_reference_identity",
            "fidelity_definition_identity",
            "evaluation_domain_identity",
            "intervention_protocol_identity",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.state not in {"evaluated", "failed", "unavailable", "censored"}:
            raise ValueError("unsupported evidence state")
        if self.state == "evaluated":
            if self.exact_fidelity is None:
                raise ValueError("evaluated evidence requires exact_fidelity")
            if self.failure_reason is not None:
                raise ValueError("evaluated evidence must not carry failure_reason")
        elif self.exact_fidelity is not None:
            raise ValueError("non-evaluated evidence must not invent exact_fidelity")
        if self.state == "failed" and not self.failure_reason:
            raise ValueError("failed evidence requires failure_reason")


@dataclass(frozen=True)
class BasisSensitivitySummary:
    evidence: ExactLedgerEvidence
    accounting: BasisAccounting

    def __post_init__(self) -> None:
        if self.evidence.basis_hash != self.accounting.basis_hash:
            raise ValueError("evidence basis hash does not match accounting basis hash")

    @property
    def raw_component_proportion(self) -> float:
        return self.accounting.raw_retained_proportion

    @property
    def parameter_weighted_proportion(self) -> float:
        return self.accounting.parameter_weight_retained_proportion

    @property
    def parent_coverage(self) -> float | None:
        return self.accounting.parent_neuron_retained_proportion


@dataclass(frozen=True)
class CrossBasisComparisonRequest:
    left: BasisSensitivitySummary
    right: BasisSensitivitySummary
    left_basis: BasisContract
    right_basis: BasisContract
    measure: AccountingMeasure

    def __post_init__(self) -> None:
        if self.measure not in {
            "raw_component_proportion",
            "parameter_weighted_proportion",
            "type_stratified",
            "parent_coverage",
        }:
            raise ValueError("unsupported accounting measure")


def _validate_common_evaluation_context(
    left: ExactLedgerEvidence,
    right: ExactLedgerEvidence,
) -> None:
    fields = (
        "model_identity",
        "dense_reference_identity",
        "fidelity_definition_identity",
        "evaluation_domain_identity",
        "intervention_protocol_identity",
    )
    for field in fields:
        if getattr(left, field) != getattr(right, field):
            raise ValueError(f"cross-basis comparison mismatches {field}")


def _validate_basis_binding(
    summary: BasisSensitivitySummary,
    basis: BasisContract,
) -> None:
    if summary.evidence.basis_hash != basis.basis_hash:
        raise ValueError("result has been relabeled with a different basis")


def _relationship_exists(left: BasisContract, right: BasisContract) -> bool:
    if left.basis_hash == right.basis_hash:
        return True
    if left.relationship is not None:
        if left.relationship.parent_basis_hash == right.basis_hash:
            return True
    if right.relationship is not None:
        if right.relationship.parent_basis_hash == left.basis_hash:
            return True
    return False


def validate_cross_basis_comparison(
    request: CrossBasisComparisonRequest,
) -> None:
    """Reject comparisons that silently change scientific/evaluation meaning."""
    _validate_basis_binding(request.left, request.left_basis)
    _validate_basis_binding(request.right, request.right_basis)
    _validate_common_evaluation_context(
        request.left.evidence,
        request.right.evidence,
    )

    if not _relationship_exists(request.left_basis, request.right_basis):
        raise ValueError("basis relationship identity is missing or incompatible")

    if request.measure == "raw_component_proportion":
        if request.left_basis.basis_hash != request.right_basis.basis_hash:
            raise ValueError(
                "raw component proportions cannot be compared across basis granularity"
            )

    if request.measure == "parameter_weighted_proportion":
        left_definition = (
            request.left.accounting.parameter_weight_denominator_definition
        )
        right_definition = (
            request.right.accounting.parameter_weight_denominator_definition
        )
        if left_definition != right_definition:
            raise ValueError("parameter-weight denominator definitions differ")

    if request.measure == "type_stratified":
        left_types = {
            row.component_type for row in request.left.accounting.type_accounting
        }
        right_types = {
            row.component_type for row in request.right.accounting.type_accounting
        }
        if left_types != right_types:
            raise ValueError("component-type strata differ or are omitted")

    if request.measure == "parent_coverage":
        for summary in (request.left, request.right):
            if summary.accounting.parent_neuron_total_count is None:
                raise ValueError("parent coverage requested but unavailable")


def comparison_values(
    request: CrossBasisComparisonRequest,
) -> tuple[object, object]:
    """Return validated technical summaries without altering exact fidelity."""
    validate_cross_basis_comparison(request)

    if request.measure == "raw_component_proportion":
        return (
            request.left.raw_component_proportion,
            request.right.raw_component_proportion,
        )
    if request.measure == "parameter_weighted_proportion":
        return (
            request.left.parameter_weighted_proportion,
            request.right.parameter_weighted_proportion,
        )
    if request.measure == "parent_coverage":
        return request.left.parent_coverage, request.right.parent_coverage

    def rows(summary: BasisSensitivitySummary) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (row.component_type, row.retained_count, row.total_count)
            for row in summary.accounting.type_accounting
        )

    return rows(request.left), rows(request.right)
