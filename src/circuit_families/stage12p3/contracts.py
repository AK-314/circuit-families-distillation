"""Scheduler-neutral adapter and operational-status boundary contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .records import LogicalJobSpec, Stage12P3ContractError, require_reference

SCHEDULER_STATES = frozenset({"submitted", "pending", "running", "finished", "failed", "cancelled"})
_FORBIDDEN_STATUS_TERMS = frozenset(
    {"effect", "effect_size", "direction", "endpoint_value", "rank", "condition_comparison"}
)


@dataclass(frozen=True)
class SchedulerSubmission:
    logical_job_id: str
    backend_name: str
    backend_job_id: str
    array_index: int | None = None

    def validate_for(self, job: LogicalJobSpec) -> None:
        if self.logical_job_id != job.job_id:
            raise Stage12P3ContractError("backend adapter changed logical job identity")
        require_reference(self.backend_name, label="backend_name")
        require_reference(self.backend_job_id, label="backend_job_id")
        if self.array_index is not None and (
            isinstance(self.array_index, bool)
            or not isinstance(self.array_index, int)
            or self.array_index < 0
        ):
            raise Stage12P3ContractError("array_index must be a non-negative integer")


@dataclass(frozen=True)
class SchedulerObservation:
    logical_job_id: str
    backend_job_id: str
    scheduler_state: str
    observed_sequence: int

    def __post_init__(self) -> None:
        if self.scheduler_state not in SCHEDULER_STATES:
            raise Stage12P3ContractError("unsupported scheduler state")
        if (
            isinstance(self.observed_sequence, bool)
            or not isinstance(self.observed_sequence, int)
            or self.observed_sequence < 0
        ):
            raise Stage12P3ContractError("observed_sequence must be non-negative")

    @property
    def sealed_success(self) -> bool:
        """Scheduler completion is deliberately never sealed output evidence."""
        return False


@dataclass(frozen=True)
class SchedulerCancellation:
    logical_job_id: str
    backend_job_id: str
    cancellation_request_id: str

    def __post_init__(self) -> None:
        require_reference(self.logical_job_id, label="cancellation logical_job_id")
        require_reference(self.backend_job_id, label="cancellation backend_job_id")
        require_reference(self.cancellation_request_id, label="cancellation_request_id")


def validate_operational_status(value: Mapping[str, Any]) -> None:
    """Reject scientific result directions or rankings anywhere in status."""

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                lowered = str(key).lower()
                if lowered in _FORBIDDEN_STATUS_TERMS or "effect_direction" in lowered:
                    raise Stage12P3ContractError(
                        f"operational status contains forbidden scientific field: {key!r}"
                    )
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
