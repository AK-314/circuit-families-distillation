"""Injected retry, capability, concurrency, and shedding policy records."""

from __future__ import annotations

from dataclasses import dataclass

from .records import ResourceClass, Stage12P3ContractError, require_reference

FAILURE_CATEGORIES = frozenset(
    {
        "worker_error",
        "numerical_failure",
        "validation_failure",
        "resource_exhaustion",
        "interruption",
        "stale_claim",
        "dependency_failure",
        "unavailable_input",
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    reference: str
    maximum_attempts: int
    retryable_categories: tuple[str, ...]
    lease_seconds: int
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        require_reference(self.reference, label="retry policy reference")
        if (
            isinstance(self.maximum_attempts, bool)
            or not isinstance(self.maximum_attempts, int)
            or self.maximum_attempts <= 0
        ):
            raise Stage12P3ContractError("maximum_attempts must be positive")
        if (
            isinstance(self.lease_seconds, bool)
            or not isinstance(self.lease_seconds, int)
            or self.lease_seconds <= 0
        ):
            raise Stage12P3ContractError("lease_seconds must be positive")
        categories = tuple(sorted(self.retryable_categories))
        if len(set(categories)) != len(categories) or not set(categories).issubset(
            FAILURE_CATEGORIES
        ):
            raise Stage12P3ContractError("retryable failure categories are invalid")
        object.__setattr__(self, "retryable_categories", categories)
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P3ContractError("retry policy must remain technical-only")


@dataclass(frozen=True)
class WorkerCapabilities:
    resource_class_reference: str
    cpu_units: int
    accelerator_capabilities: tuple[str, ...]
    memory_bytes: int
    scratch_bytes: int

    def __post_init__(self) -> None:
        require_reference(self.resource_class_reference, label="worker resource class")
        for name in ("cpu_units", "memory_bytes", "scratch_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Stage12P3ContractError(f"{name} must be a non-negative integer")
        capabilities = tuple(sorted(self.accelerator_capabilities))
        if len(set(capabilities)) != len(capabilities):
            raise Stage12P3ContractError("accelerator capabilities must be unique")
        object.__setattr__(self, "accelerator_capabilities", capabilities)

    def satisfies(self, requested: ResourceClass) -> bool:
        """Require the same native class and sufficient generic quantities."""
        return (
            self.resource_class_reference == requested.reference
            and self.cpu_units >= requested.cpu_units
            and self.memory_bytes >= requested.memory_bytes
            and self.scratch_bytes >= requested.scratch_bytes
            and (
                requested.accelerator_capability is None
                or requested.accelerator_capability in self.accelerator_capabilities
            )
        )


@dataclass(frozen=True)
class ConcurrencyProfile:
    reference: str
    limits_by_resource_class: tuple[tuple[str, int], ...]
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        require_reference(self.reference, label="concurrency profile reference")
        ordered = tuple(sorted(self.limits_by_resource_class))
        if len({reference for reference, _ in ordered}) != len(ordered):
            raise Stage12P3ContractError("duplicate resource concurrency limits")
        for reference, limit in ordered:
            require_reference(reference, label="resource concurrency reference")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise Stage12P3ContractError("concurrency limits must be non-negative")
        object.__setattr__(self, "limits_by_resource_class", ordered)
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P3ContractError("concurrency profile must remain technical-only")

    def limit_for(self, reference: str) -> int:
        try:
            return dict(self.limits_by_resource_class)[reference]
        except KeyError as exc:
            raise Stage12P3ContractError(
                "resource class lacks an injected concurrency limit"
            ) from exc


@dataclass(frozen=True)
class TierRule:
    label: str
    protected: bool
    optional: bool
    shedding_rank: int | None

    def __post_init__(self) -> None:
        require_reference(self.label, label="tier label")
        if self.protected and self.optional:
            raise Stage12P3ContractError("a protected tier cannot be optional")
        if self.protected and self.shedding_rank is not None:
            raise Stage12P3ContractError("protected tiers cannot have a shedding rank")
        if self.optional and (
            isinstance(self.shedding_rank, bool) or not isinstance(self.shedding_rank, int)
        ):
            raise Stage12P3ContractError("optional tiers require an integer shedding rank")


@dataclass(frozen=True)
class SheddingPolicy:
    reference: str
    tier_rules: tuple[TierRule, ...]
    reason: str
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        require_reference(self.reference, label="shedding policy reference")
        require_reference(self.reason, label="incomplete campaign reason")
        ordered = tuple(sorted(self.tier_rules, key=lambda rule: rule.label))
        if len({rule.label for rule in ordered}) != len(ordered):
            raise Stage12P3ContractError("duplicate tier rules")
        object.__setattr__(self, "tier_rules", ordered)
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P3ContractError("shedding policy must remain technical-only")

    def rule_for(self, label: str) -> TierRule:
        try:
            return {rule.label: rule for rule in self.tier_rules}[label]
        except KeyError as exc:
            raise Stage12P3ContractError(f"tier {label!r} lacks a shedding rule") from exc
