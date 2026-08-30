"""Versioned Stage 12-R1 lifecycle records and technical profiles."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .contracts import (
    ALGORITHM_FAMILY,
    NATIVE_BUDGET_UNIT,
    UNRESOLVED_PRODUCTION_DECISIONS,
)

LIFECYCLE_RECORD_VERSION = "stage12r1-lifecycle/v1"
TECHNICAL_PROFILE_VERSION = "stage12r1-technical-profile/v1"

TerminalState = Literal[
    "completed",
    "exhausted",
    "interrupted",
    "numerical_failure",
    "exact_evaluator_failure",
]

FailureKind = Literal[
    "none",
    "native_budget_exhausted",
    "interrupted",
    "nonfinite_objective",
    "nonfinite_gradient",
    "exact_budget_exhausted",
    "exact_evaluator_failure",
    "invalid_resume",
]


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA256")


def _require_reference(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")
    lowered = value.lower()
    if lowered.startswith("/") or "users/" in lowered or "\\users\\" in lowered:
        raise ValueError(f"{name} must not contain private filesystem paths")


@dataclass(frozen=True)
class Stage12R1TechnicalProfile:
    profile_version: str
    profile_id: str
    method_name: str
    method_version: str
    algorithm_family: str
    configuration_reference: str
    native_budget_unit: str
    native_budget_allowance: int | None
    exact_evaluation_allowance: int | None
    maximum_restarts: int | None
    production_algorithm_selected: bool
    scientific_data: bool
    production_eligible: bool
    unresolved_decisions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.profile_version != TECHNICAL_PROFILE_VERSION:
            raise ValueError("unsupported Stage 12-R1 technical profile version")
        for name in (
            "profile_id",
            "method_name",
            "method_version",
            "configuration_reference",
        ):
            _require_reference(getattr(self, name), name)

        if self.algorithm_family != ALGORITHM_FAMILY:
            raise ValueError("technical profile algorithm-family mismatch")
        if self.native_budget_unit != NATIVE_BUDGET_UNIT:
            raise ValueError(
                "native budget unit must remain optimizer_step and non-equivalent"
            )

        for name in (
            "native_budget_allowance",
            "exact_evaluation_allowance",
            "maximum_restarts",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be unresolved or non-negative")

        if self.production_algorithm_selected:
            raise ValueError("Stage 12-R1 cannot select a production algorithm")
        if self.scientific_data:
            raise ValueError("technical profile requires scientific_data=false")
        if self.production_eligible:
            raise ValueError("technical profile requires production_eligible=false")
        if self.unresolved_decisions != UNRESOLVED_PRODUCTION_DECISIONS:
            raise ValueError(
                "RD-005–RD-009/RD-012/RD-014 must remain unresolved"
            )


@dataclass(frozen=True)
class NativeBudgetRecord:
    unit: str
    allowance: int
    consumed: int
    exhausted: bool

    def __post_init__(self) -> None:
        if self.unit != NATIVE_BUDGET_UNIT:
            raise ValueError("native work must be recorded in optimizer_step units")
        if (
            isinstance(self.allowance, bool)
            or not isinstance(self.allowance, int)
            or self.allowance < 0
        ):
            raise ValueError("native allowance must be non-negative")
        if (
            isinstance(self.consumed, bool)
            or not isinstance(self.consumed, int)
            or self.consumed < 0
        ):
            raise ValueError("native consumed must be non-negative")
        if self.consumed > self.allowance:
            raise ValueError("native consumption exceeds allowance")
        if self.exhausted != (self.consumed >= self.allowance):
            raise ValueError("native exhausted flag disagrees with accounting")


@dataclass(frozen=True)
class ExactBudgetRecord:
    allowance: int
    charged: int
    evaluation_count: int
    proposal_count: int
    exhausted: bool

    def __post_init__(self) -> None:
        for name in (
            "allowance",
            "charged",
            "evaluation_count",
            "proposal_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")

        if self.allowance < 1:
            raise ValueError("exact allowance must reserve intact baseline")
        if self.charged > self.allowance:
            raise ValueError("exact charge exceeds allowance")
        if self.evaluation_count != self.charged:
            raise ValueError(
                "unique exact evaluation count must equal charged count"
            )
        if self.exhausted and self.charged < self.allowance:
            raise ValueError("exact exhausted flag disagrees with accounting")


@dataclass(frozen=True)
class ProposalProvenanceSummary:
    gate_state_sha256: str
    extraction_config_sha256: str
    proposal_count: int
    unique_mask_count: int
    duplicate_proposal_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.gate_state_sha256, "gate_state_sha256")
        _require_sha256(
            self.extraction_config_sha256,
            "extraction_config_sha256",
        )
        for name in (
            "proposal_count",
            "unique_mask_count",
            "duplicate_proposal_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")

        if self.unique_mask_count > self.proposal_count:
            raise ValueError("unique mask count exceeds proposal count")
        if (
            self.duplicate_proposal_count
            != self.proposal_count - self.unique_mask_count
        ):
            raise ValueError("duplicate proposal count is inconsistent")


@dataclass(frozen=True)
class ExactBridgeSummary:
    exact_ledger_sha256: str
    exact_ledger_evaluation_count: int
    exact_ledger_proposal_count: int
    qualifying_count: int
    minimum_exact_fidelity: float
    maximum_exact_fidelity: float

    def __post_init__(self) -> None:
        _require_sha256(
            self.exact_ledger_sha256,
            "exact_ledger_sha256",
        )
        for name in (
            "exact_ledger_evaluation_count",
            "exact_ledger_proposal_count",
            "qualifying_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")

        if self.qualifying_count > self.exact_ledger_evaluation_count:
            raise ValueError("qualifying count exceeds evaluated count")

        for value in (
            self.minimum_exact_fidelity,
            self.maximum_exact_fidelity,
        ):
            if not math.isfinite(value):
                raise ValueError("exact fidelity bounds must be finite")

        if self.minimum_exact_fidelity > self.maximum_exact_fidelity:
            raise ValueError("exact fidelity bounds are reversed")


@dataclass(frozen=True)
class Stage12R1LifecycleRecord:
    record_version: str
    run_id: str
    method_name: str
    method_version: str
    algorithm_family: str
    configuration_reference: str
    run_identity_sha256: str
    gate_config_sha256: str
    optimizer_config_sha256: str
    optimizer_result_sha256: str
    checkpoint_identity_sha256: str | None
    native_budget: NativeBudgetRecord
    proposals: ProposalProvenanceSummary | None
    exact_budget: ExactBudgetRecord | None
    exact_bridge: ExactBridgeSummary | None
    terminal_state: TerminalState
    failure_kind: FailureKind
    intact_endpoint1_available: bool
    production_algorithm_selected: bool
    scientific_data: bool
    production_eligible: bool
    unresolved_decisions: tuple[str, ...]
    record_sha256: str

    def __post_init__(self) -> None:
        if self.record_version != LIFECYCLE_RECORD_VERSION:
            raise ValueError("unsupported Stage 12-R1 lifecycle record version")

        for name in (
            "run_id",
            "method_name",
            "method_version",
            "configuration_reference",
        ):
            _require_reference(getattr(self, name), name)

        if self.algorithm_family != ALGORITHM_FAMILY:
            raise ValueError("lifecycle algorithm-family mismatch")

        for name in (
            "run_identity_sha256",
            "gate_config_sha256",
            "optimizer_config_sha256",
            "optimizer_result_sha256",
        ):
            _require_sha256(getattr(self, name), name)

        if self.checkpoint_identity_sha256 is not None:
            _require_sha256(
                self.checkpoint_identity_sha256,
                "checkpoint_identity_sha256",
            )

        if self.production_algorithm_selected:
            raise ValueError("production method remains unresolved")
        if self.scientific_data:
            raise ValueError("lifecycle record requires scientific_data=false")
        if self.production_eligible:
            raise ValueError("lifecycle record requires production_eligible=false")
        if self.unresolved_decisions != UNRESOLVED_PRODUCTION_DECISIONS:
            raise ValueError(
                "RD-005–RD-009/RD-012/RD-014 must remain unresolved"
            )

        if self.exact_bridge is not None and self.exact_budget is None:
            raise ValueError("exact bridge summary requires exact budget record")

        if self.failure_kind == "none":
            if self.terminal_state not in {"completed", "exhausted"}:
                raise ValueError("non-success state requires explicit failure kind")

        if self.failure_kind == "exact_evaluator_failure":
            if self.terminal_state != "exact_evaluator_failure":
                raise ValueError("exact evaluator failure state mismatch")

        if not self.intact_endpoint1_available and self.exact_budget is not None:
            raise ValueError(
                "shared exact bridge always constructs intact Endpoint 1 baseline"
            )

        payload = lifecycle_payload_without_hash(self)
        if _sha256(payload) != self.record_sha256:
            raise ValueError("lifecycle record hash mismatch")


def lifecycle_payload_without_hash(
    record: Stage12R1LifecycleRecord,
) -> dict[str, Any]:
    payload = asdict(record)
    payload.pop("record_sha256", None)
    return payload


def build_lifecycle_record(
    *,
    run_id: str,
    method_name: str,
    method_version: str,
    configuration_reference: str,
    run_identity_sha256: str,
    gate_config_sha256: str,
    optimizer_config_sha256: str,
    optimizer_result_sha256: str,
    checkpoint_identity_sha256: str | None,
    native_budget: NativeBudgetRecord,
    proposals: ProposalProvenanceSummary | None,
    exact_budget: ExactBudgetRecord | None,
    exact_bridge: ExactBridgeSummary | None,
    terminal_state: TerminalState,
    failure_kind: FailureKind,
    intact_endpoint1_available: bool,
) -> Stage12R1LifecycleRecord:
    provisional = {
        "record_version": LIFECYCLE_RECORD_VERSION,
        "run_id": run_id,
        "method_name": method_name,
        "method_version": method_version,
        "algorithm_family": ALGORITHM_FAMILY,
        "configuration_reference": configuration_reference,
        "run_identity_sha256": run_identity_sha256,
        "gate_config_sha256": gate_config_sha256,
        "optimizer_config_sha256": optimizer_config_sha256,
        "optimizer_result_sha256": optimizer_result_sha256,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "native_budget": asdict(native_budget),
        "proposals": None if proposals is None else asdict(proposals),
        "exact_budget": None if exact_budget is None else asdict(exact_budget),
        "exact_bridge": None if exact_bridge is None else asdict(exact_bridge),
        "terminal_state": terminal_state,
        "failure_kind": failure_kind,
        "intact_endpoint1_available": intact_endpoint1_available,
        "production_algorithm_selected": False,
        "scientific_data": False,
        "production_eligible": False,
        "unresolved_decisions": list(UNRESOLVED_PRODUCTION_DECISIONS),
    }

    digest = _sha256(provisional)

    return Stage12R1LifecycleRecord(
        record_version=LIFECYCLE_RECORD_VERSION,
        run_id=run_id,
        method_name=method_name,
        method_version=method_version,
        algorithm_family=ALGORITHM_FAMILY,
        configuration_reference=configuration_reference,
        run_identity_sha256=run_identity_sha256,
        gate_config_sha256=gate_config_sha256,
        optimizer_config_sha256=optimizer_config_sha256,
        optimizer_result_sha256=optimizer_result_sha256,
        checkpoint_identity_sha256=checkpoint_identity_sha256,
        native_budget=native_budget,
        proposals=proposals,
        exact_budget=exact_budget,
        exact_bridge=exact_bridge,
        terminal_state=terminal_state,
        failure_kind=failure_kind,
        intact_endpoint1_available=intact_endpoint1_available,
        production_algorithm_selected=False,
        scientific_data=False,
        production_eligible=False,
        unresolved_decisions=UNRESOLVED_PRODUCTION_DECISIONS,
        record_sha256=digest,
    )


def lifecycle_record_from_mapping(
    record: Mapping[str, Any],
) -> Stage12R1LifecycleRecord:
    expected = {
        "record_version",
        "run_id",
        "method_name",
        "method_version",
        "algorithm_family",
        "configuration_reference",
        "run_identity_sha256",
        "gate_config_sha256",
        "optimizer_config_sha256",
        "optimizer_result_sha256",
        "checkpoint_identity_sha256",
        "native_budget",
        "proposals",
        "exact_budget",
        "exact_bridge",
        "terminal_state",
        "failure_kind",
        "intact_endpoint1_available",
        "production_algorithm_selected",
        "scientific_data",
        "production_eligible",
        "unresolved_decisions",
        "record_sha256",
    }

    if set(record) != expected:
        raise ValueError("lifecycle record keys are not exact")

    native = NativeBudgetRecord(**record["native_budget"])
    proposals = (
        None
        if record["proposals"] is None
        else ProposalProvenanceSummary(**record["proposals"])
    )
    exact_budget = (
        None
        if record["exact_budget"] is None
        else ExactBudgetRecord(**record["exact_budget"])
    )
    exact_bridge = (
        None
        if record["exact_bridge"] is None
        else ExactBridgeSummary(**record["exact_bridge"])
    )

    return Stage12R1LifecycleRecord(
        **{
            **dict(record),
            "native_budget": native,
            "proposals": proposals,
            "exact_budget": exact_budget,
            "exact_bridge": exact_bridge,
            "unresolved_decisions": tuple(record["unresolved_decisions"]),
        }
    )
