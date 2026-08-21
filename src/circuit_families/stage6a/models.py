"""Synthetic-only Stage 6A exact ledger and Endpoint 1 data models.

This module defines contracts only. It does not implement discovery,
production endpoint computation, calibration, or scientific data handling.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

COMPONENT_COUNT = 516


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_mask_identity(retained_components: Sequence[int]) -> str:
    validated = []

    for component in retained_components:
        if isinstance(component, bool):
            raise ValueError("boolean component ids are invalid")
        if not isinstance(component, int):
            raise ValueError("component ids must be integers")
        if component < 0 or component >= COMPONENT_COUNT:
            raise ValueError("component id outside component universe")
        validated.append(component)

    components = tuple(sorted(set(validated)))
    payload = {"components": list(components)}
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def retained_proportion(retained_count: int) -> float:
    if retained_count < 0 or retained_count > COMPONENT_COUNT:
        raise ValueError("invalid retained component count")
    return retained_count / COMPONENT_COUNT


@dataclass(frozen=True)
class TechnicalLedgerProfile:
    profile_version: str
    name: str
    synthetic_only: bool
    scientific_data: bool
    production_eligible: bool
    unresolved_decisions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.profile_version:
            raise ValueError("profile version required")
        if not self.synthetic_only:
            raise ValueError("Stage 6A profiles must be synthetic-only")
        if self.scientific_data:
            raise ValueError("Stage 6A profiles cannot contain scientific data")
        if self.production_eligible:
            raise ValueError("Stage 6A profiles cannot be production eligible")

    def to_record(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "name": self.name,
            "synthetic_only": self.synthetic_only,
            "scientific_data": self.scientific_data,
            "production_eligible": self.production_eligible,
            "unresolved_decisions": list(self.unresolved_decisions),
        }


@dataclass(frozen=True)
class ProposalEvent:
    proposal_index: int
    mask_identity: str

    def __post_init__(self) -> None:
        if isinstance(self.proposal_index, bool) or not isinstance(
            self.proposal_index,
            int,
        ):
            raise ValueError("proposal index must be an integer")
        if self.proposal_index < 0:
            raise ValueError("proposal index must be non-negative")
        if not isinstance(self.mask_identity, str) or not self.mask_identity:
            raise ValueError("mask identity must be nonempty")


@dataclass(frozen=True)
class ExactEvaluationEntry:
    mask_identity: str
    retained_count: int
    retained_proportion: float
    fidelity: float
    qualifies: bool
    evaluation_order: int
    exact_budget_charge: int

    def __post_init__(self) -> None:
        if not isinstance(self.mask_identity, str) or not self.mask_identity:
            raise ValueError("mask identity required")
        if not isinstance(self.retained_count, int):
            raise ValueError("retained count must be integer")
        if self.retained_count < 0 or self.retained_count > COMPONENT_COUNT:
            raise ValueError("retained count out of range")
        if not math.isfinite(float(self.retained_proportion)):
            raise ValueError("retained proportion must be finite")
        if self.retained_proportion != self.retained_count / COMPONENT_COUNT:
            raise ValueError("retained proportion mismatch")
        if not math.isfinite(float(self.fidelity)):
            raise ValueError("fidelity must be finite")
        if self.evaluation_order < 0:
            raise ValueError("evaluation order invalid")
        if self.exact_budget_charge < 0:
            raise ValueError("budget charge invalid")


@dataclass(frozen=True)
class TerminationStatus:
    status: str
    procedure_censored: bool

    def __post_init__(self) -> None:
        allowed = {"completed", "censored", "failed"}
        if self.status not in allowed:
            raise ValueError("invalid termination status")
        if not isinstance(self.procedure_censored, bool):
            raise ValueError("procedure_censored must be bool")
        if self.status == "censored" and not self.procedure_censored:
            raise ValueError("censored status requires censor flag")


@dataclass(frozen=True)
class Endpoint1Result:
    retained_proportion: float
    mask_identity: str
    global_minimum_claim: bool
    termination_status: str
    procedure_censored: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.retained_proportion <= 1.0:
            raise ValueError("retained proportion out of range")
        if not isinstance(self.mask_identity, str) or not self.mask_identity:
            raise ValueError("mask identity required")
        if self.global_minimum_claim:
            raise ValueError("global minimum claim forbidden")


def exact_evaluation_entry_to_record(
    entry: ExactEvaluationEntry,
) -> dict[str, Any]:
    return {
        "mask_identity": entry.mask_identity,
        "retained_count": entry.retained_count,
        "retained_proportion": entry.retained_proportion,
        "fidelity": entry.fidelity,
        "qualifies": entry.qualifies,
        "evaluation_order": entry.evaluation_order,
        "exact_budget_charge": entry.exact_budget_charge,
    }


def exact_evaluation_entry_from_record(
    record: Mapping[str, Any],
) -> ExactEvaluationEntry:
    return ExactEvaluationEntry(
        mask_identity=str(record["mask_identity"]),
        retained_count=int(record["retained_count"]),
        retained_proportion=float(record["retained_proportion"]),
        fidelity=float(record["fidelity"]),
        qualifies=bool(record["qualifies"]),
        evaluation_order=int(record["evaluation_order"]),
        exact_budget_charge=int(record["exact_budget_charge"]),
    )


def endpoint1_result_to_record(
    result: Endpoint1Result,
) -> dict[str, Any]:
    return {
        "retained_proportion": result.retained_proportion,
        "mask_identity": result.mask_identity,
        "global_minimum_claim": result.global_minimum_claim,
        "termination_status": result.termination_status,
        "procedure_censored": result.procedure_censored,
    }


def endpoint1_result_from_record(
    record: Mapping[str, Any],
) -> Endpoint1Result:
    return Endpoint1Result(
        retained_proportion=float(record["retained_proportion"]),
        mask_identity=str(record["mask_identity"]),
        global_minimum_claim=bool(record["global_minimum_claim"]),
        termination_status=str(record["termination_status"]),
        procedure_censored=bool(record["procedure_censored"]),
    )


@dataclass(frozen=True)
class SealedLedger:
    profile: TechnicalLedgerProfile
    evaluations: tuple[ExactEvaluationEntry, ...]
    proposals: tuple[ProposalEvent, ...]
    has_intact_baseline: bool
    sealed: bool


def validate_mask_identity(component_ids: Sequence[int]) -> tuple[int, ...]:
    if any(isinstance(x, bool) for x in component_ids):
        raise ValueError("boolean component ids are invalid")

    values = tuple(component_ids)

    if any(not isinstance(x, int) for x in values):
        raise ValueError("component ids must be integers")

    if any(x < 0 or x >= COMPONENT_COUNT for x in values):
        raise ValueError("component id outside universe")

    return tuple(sorted(set(values)))


def _require_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
