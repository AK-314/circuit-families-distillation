"""Synthetic-only Stage 6A exact/native budget separation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TechnicalBudgetPolicy:
    """Injected technical policy; does not resolve UD-010."""

    exact_evaluation_allowance: int

    def __post_init__(self) -> None:
        if self.exact_evaluation_allowance < 0:
            raise ValueError("allowance must be non-negative")


@dataclass(frozen=True)
class NativeBudgetUsage:
    """Native search/optimizer accounting kept separate."""

    consumed: int


@dataclass(frozen=True)
class ExactBudgetUsage:
    """Exact ledger accounting."""

    evaluation_count: int
    charged_count: int

    def charge(self, amount: int) -> ExactBudgetUsage:
        if amount < 0:
            raise ValueError("charge must be non-negative")

        return ExactBudgetUsage(
            evaluation_count=self.evaluation_count,
            charged_count=self.charged_count + amount,
        )


def validate_within_allowance(
    usage: ExactBudgetUsage,
    policy: TechnicalBudgetPolicy,
) -> None:
    if usage.charged_count > policy.exact_evaluation_allowance:
        raise ValueError("exact evaluation allowance exceeded")
