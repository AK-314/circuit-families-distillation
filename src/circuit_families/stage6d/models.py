"""Stage 6D method-agnostic discovery contracts.

These records are technical infrastructure only. They do not freeze any
production discovery roster, method version, native budget, restart policy,
termination policy, fidelity setting, or common exact-evaluation allowance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

TECHNICAL_ONLY = True
PRODUCTION_ELIGIBLE = False
UNRESOLVED_DECISIONS = ("UD-007", "UD-009", "UD-010", "UD-014")
RESOURCE_WARNING = (
    "Method-native budget units are method-specific and are not "
    "resource-equivalent across discovery methods."
)

TrajectoryKind = Literal[
    "proposal",
    "restart",
    "exact_request",
    "exact_result",
    "termination",
    "failure",
]

StoppingStatus = Literal[
    "completed",
    "native_budget_exhausted",
    "exact_budget_exhausted",
    "failed",
]


def _nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class DeterministicSeedEvidence:
    seed_derivation_version: str
    seed_material_sha256: str
    seed_value: int

    def __post_init__(self) -> None:
        _nonempty(self.seed_derivation_version, "seed_derivation_version")
        digest = _nonempty(self.seed_material_sha256, "seed_material_sha256")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("seed_material_sha256 must be lowercase SHA-256 hex")
        _nonnegative_int(self.seed_value, "seed_value")


@dataclass(frozen=True)
class TechnicalDiscoveryProfile:
    profile_id: str
    profile_version: str
    method_name: str
    method_version: str
    configuration_reference: str
    native_budget_unit: str
    native_budget_allowance: int
    exact_evaluation_allowance: int
    maximum_restarts: int
    technical_only: bool
    production_eligible: bool
    unresolved_decisions: tuple[str, ...]
    resource_warning: str

    def __post_init__(self) -> None:
        for field in (
            "profile_id",
            "profile_version",
            "method_name",
            "method_version",
            "configuration_reference",
            "native_budget_unit",
            "resource_warning",
        ):
            _nonempty(getattr(self, field), field)

        _nonnegative_int(self.native_budget_allowance, "native_budget_allowance")
        _nonnegative_int(
            self.exact_evaluation_allowance,
            "exact_evaluation_allowance",
        )
        _nonnegative_int(self.maximum_restarts, "maximum_restarts")

        if self.technical_only is not True:
            raise ValueError("Stage 6D technical profiles must be technical_only")
        if self.production_eligible is not False:
            raise ValueError(
                "Stage 6D technical profiles must not be production eligible"
            )
        if tuple(self.unresolved_decisions) != UNRESOLVED_DECISIONS:
            raise ValueError(
                "technical profile must preserve UD-007/009/010/014 unresolved"
            )
        if self.resource_warning != RESOURCE_WARNING:
            raise ValueError("resource warning must use canonical Stage 6D text")


@dataclass(frozen=True)
class DiscoveryRequest:
    run_id: str
    method_name: str
    method_version: str
    configuration_reference: str
    seed_evidence: DeterministicSeedEvidence
    native_budget_unit: str
    native_budget_allowance: int
    exact_evaluation_allowance: int
    maximum_restarts: int
    synthetic_fixture: bool
    production_eligible: bool

    def __post_init__(self) -> None:
        for field in (
            "run_id",
            "method_name",
            "method_version",
            "configuration_reference",
            "native_budget_unit",
        ):
            _nonempty(getattr(self, field), field)

        _nonnegative_int(self.native_budget_allowance, "native_budget_allowance")
        _nonnegative_int(
            self.exact_evaluation_allowance,
            "exact_evaluation_allowance",
        )
        _nonnegative_int(self.maximum_restarts, "maximum_restarts")

        if self.synthetic_fixture is not True:
            raise ValueError("Stage 6D requests must use synthetic fixtures")
        if self.production_eligible is not False:
            raise ValueError("Stage 6D requests must not be production eligible")


@dataclass(frozen=True)
class TrajectoryEvent:
    sequence_index: int
    kind: TrajectoryKind
    restart_index: int
    native_consumed: int
    exact_requested: int
    detail: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonnegative_int(self.sequence_index, "sequence_index")
        _nonnegative_int(self.restart_index, "restart_index")
        _nonnegative_int(self.native_consumed, "native_consumed")
        _nonnegative_int(self.exact_requested, "exact_requested")
        if self.kind not in {
            "proposal",
            "restart",
            "exact_request",
            "exact_result",
            "termination",
            "failure",
        }:
            raise ValueError(f"unsupported trajectory kind: {self.kind!r}")


@dataclass(frozen=True)
class DiscoveryResult:
    run_id: str
    method_name: str
    method_version: str
    configuration_reference: str
    seed_evidence: DeterministicSeedEvidence
    native_budget_unit: str
    native_budget_allowance: int
    native_budget_consumed: int
    native_budget_exhausted: bool
    exact_evaluation_allowance: int
    exact_evaluation_consumed: int
    exact_budget_exhausted: bool
    exact_ledger_sha256: str
    exact_ledger_evaluation_count: int
    exact_ledger_proposal_count: int
    restart_count: int
    proposal_count: int
    exact_request_count: int
    stopping_status: StoppingStatus
    trajectory: tuple[TrajectoryEvent, ...]
    technical_only: bool
    production_eligible: bool
    unresolved_decisions: tuple[str, ...]
    resource_warning: str

    def __post_init__(self) -> None:
        for field in (
            "run_id",
            "method_name",
            "method_version",
            "configuration_reference",
            "native_budget_unit",
            "resource_warning",
        ):
            _nonempty(getattr(self, field), field)

        for field in (
            "native_budget_allowance",
            "native_budget_consumed",
            "exact_evaluation_allowance",
            "exact_evaluation_consumed",
            "exact_ledger_evaluation_count",
            "exact_ledger_proposal_count",
            "restart_count",
            "proposal_count",
            "exact_request_count",
        ):
            _nonnegative_int(getattr(self, field), field)

        digest = _nonempty(self.exact_ledger_sha256, "exact_ledger_sha256")
        if len(digest) != 64 or any(
            ch not in "0123456789abcdef" for ch in digest
        ):
            raise ValueError("exact_ledger_sha256 must be lowercase SHA-256 hex")

        if self.native_budget_consumed > self.native_budget_allowance:
            raise ValueError("native budget consumption exceeds allowance")
        if self.exact_evaluation_consumed > self.exact_evaluation_allowance:
            raise ValueError("exact evaluation consumption exceeds allowance")

        if self.technical_only is not True:
            raise ValueError("Stage 6D results must be technical_only")
        if self.production_eligible is not False:
            raise ValueError("Stage 6D results must not be production eligible")
        if tuple(self.unresolved_decisions) != UNRESOLVED_DECISIONS:
            raise ValueError(
                "result must preserve UD-007/009/010/014 unresolved"
            )
        if self.resource_warning != RESOURCE_WARNING:
            raise ValueError("result must carry canonical resource warning")

        expected = tuple(range(len(self.trajectory)))
        actual = tuple(event.sequence_index for event in self.trajectory)
        if actual != expected:
            raise ValueError(
                "trajectory sequence_index values must be contiguous from zero"
            )


@runtime_checkable
class DiscoveryAdapter(Protocol):
    """Method-agnostic Stage 6D discovery boundary."""

    method_name: str
    method_version: str

    def run(self, request: DiscoveryRequest) -> DiscoveryResult:
        ...
