"""Auditable operational information-capacity accounting for all conditions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from circuit_families.stage12p3.records import canonical_sha256

from .contracts import CONDITIONS, CapacityContract, Stage12P5ContractError
from .fourier import InterventionPayload


@dataclass(frozen=True)
class CapacityAccounting:
    condition: str
    capacity_sha256: str
    coordinate_ids: tuple[str, ...]
    numeric_scalar_count: int
    real_degrees_of_freedom: int
    observed_rank: int
    observed_support: int
    scalar_precision_bits: int
    side_information: tuple[str, ...]
    external_identifiers: tuple[str, ...]
    recipient_shape: tuple[int, ...]
    write_budget_scalars: int
    padding_rule: str
    numeric_byte_length: int
    norm: float
    eligible: bool
    ineligibility_reason: str | None
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise Stage12P5ContractError("capacity accounting has unknown condition")
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P5ContractError("capacity accounting must remain technical-only")
        if self.eligible != (self.ineligibility_reason is None):
            raise Stage12P5ContractError("capacity eligibility/reason mismatch")

    @property
    def accounting_sha256(self) -> str:
        value = asdict(self)
        for key in (
            "coordinate_ids",
            "side_information",
            "external_identifiers",
            "recipient_shape",
        ):
            value[key] = list(getattr(self, key))
        return canonical_sha256(value)

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "coordinate_ids",
            "side_information",
            "external_identifiers",
            "recipient_shape",
        ):
            value[key] = list(getattr(self, key))
        value["accounting_sha256"] = self.accounting_sha256
        return value


def _real_dof(values: np.ndarray) -> int:
    return int(values.size * (2 if np.iscomplexobj(values) else 1))


def _rank(values: np.ndarray) -> int:
    array = np.asarray(values)
    if array.ndim <= 1:
        return int(np.count_nonzero(np.abs(array) > 0))
    return int(np.linalg.matrix_rank(array))


def account_payload(
    payload: InterventionPayload,
    contract: CapacityContract,
    *,
    scalar_precision_bits: int,
    hidden_metadata: Mapping[str, Any] | None = None,
) -> CapacityAccounting:
    """Independently prove one payload obeys the declared operational allowance."""
    values = np.asarray(payload.coordinate_values)
    reasons: list[str] = []
    coordinate_ids = tuple(sorted(payload.coordinate_ids))
    if coordinate_ids != contract.allowed_coordinate_ids:
        reasons.append("coordinate support/identifiers differ from the capacity contract")
    observed_support = int(np.count_nonzero(np.abs(values.reshape(-1)) > 0))
    dof = _real_dof(values)
    observed_rank = _rank(values)
    if dof != contract.real_degrees_of_freedom:
        reasons.append("real degree count mismatch, including complex-number accounting")
    if observed_support > contract.maximum_support:
        reasons.append("support exceeds capacity contract")
    if observed_rank > contract.maximum_rank:
        reasons.append("rank exceeds capacity contract")
    if scalar_precision_bits != contract.scalar_precision_bits:
        reasons.append("scalar precision mismatch")
    if tuple(payload.side_information) != contract.allowed_side_information:
        reasons.append("undeclared or missing side information")
    if tuple(payload.external_identifiers) != contract.external_identifier_budget:
        reasons.append("external identifier budget mismatch")
    if tuple(payload.ordinary_state.shape) != contract.recipient_shape:
        reasons.append("recipient shape mismatch")
    if int(payload.ordinary_state.size) != contract.write_budget_scalars:
        reasons.append("recipient write budget mismatch")
    hidden = dict(hidden_metadata or {})
    prohibited_hidden = {
        "mode_label",
        "input_identity",
        "alignment_matrix",
        "coordinate_indices",
        "random_seed",
        "payload_length",
    }
    leaked = sorted(prohibited_hidden.intersection(hidden))
    if leaked:
        reasons.append("hidden side channel: " + ",".join(leaked))
    reason = "; ".join(reasons) if reasons else None
    return CapacityAccounting(
        condition=payload.condition,
        capacity_sha256=contract.capacity_sha256,
        coordinate_ids=coordinate_ids,
        numeric_scalar_count=int(values.size),
        real_degrees_of_freedom=dof,
        observed_rank=observed_rank,
        observed_support=observed_support,
        scalar_precision_bits=scalar_precision_bits,
        side_information=tuple(payload.side_information),
        external_identifiers=tuple(payload.external_identifiers),
        recipient_shape=tuple(payload.ordinary_state.shape),
        write_budget_scalars=int(payload.ordinary_state.size),
        padding_rule=contract.padding_rule,
        numeric_byte_length=int(values.nbytes),
        norm=float(np.linalg.norm(values)),
        eligible=not reasons,
        ineligibility_reason=reason,
    )


def validate_comparison_capacity(
    accounting: Mapping[str, CapacityAccounting], contract: CapacityContract
) -> str:
    """Require complete six-condition accounting under one exact contract hash."""
    if tuple(accounting) != CONDITIONS:
        raise Stage12P5ContractError("capacity accounting inventory is incomplete or reordered")
    for condition in CONDITIONS:
        record = accounting[condition]
        if record.condition != condition or record.capacity_sha256 != contract.capacity_sha256:
            raise Stage12P5ContractError("condition capacity identity mismatch")
        if not record.eligible:
            raise Stage12P5ContractError(
                f"condition {condition} is capacity-ineligible: {record.ineligibility_reason}"
            )
        if record.real_degrees_of_freedom != contract.real_degrees_of_freedom:
            raise Stage12P5ContractError("conditions differ in operational information allowance")
    return canonical_sha256(
        {
            "schema_version": "stage12p5-capacity-comparison/v1",
            "capacity_sha256": contract.capacity_sha256,
            "condition_accounting": [accounting[name].accounting_sha256 for name in CONDITIONS],
            "scientific_data": False,
            "production_eligible": False,
        }
    )
