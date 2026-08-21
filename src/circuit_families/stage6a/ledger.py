"""Synthetic-only Stage 6A exact evaluation ledger construction."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .models import (
    COMPONENT_COUNT,
    ExactEvaluationEntry,
    ProposalEvent,
    canonical_mask_identity,
    retained_proportion,
)

Mask = tuple[int, ...]
Evaluator = Callable[[Mask], float]


def validate_mask(mask: Sequence[int]) -> Mask:
    values = tuple(int(x) for x in mask)

    if len(values) != COMPONENT_COUNT:
        raise ValueError("mask must contain exactly 516 components")

    if any(value not in (0, 1) for value in values):
        raise ValueError("mask must be binary")

    return values


def _identity(mask: Mask) -> str:
    return canonical_mask_identity(
        index for index, value in enumerate(mask) if value
    )


def _intact_mask() -> Mask:
    return (1,) * COMPONENT_COUNT


@dataclass
class ExactLedgerBuilder:
    evaluator: Evaluator
    fidelity_threshold: float
    evaluations: list[ExactEvaluationEntry] = field(default_factory=list)
    proposals: list[ProposalEvent] = field(default_factory=list)
    _seen: dict[str, ExactEvaluationEntry] = field(default_factory=dict)
    _sealed: bool = False
    _has_intact: bool = False

    def __post_init__(self) -> None:
        self._evaluate_mask(
            _intact_mask(),
            proposal_index=-1,
            record_proposal=False,
        )

    def _evaluate_mask(
        self,
        mask: Sequence[int],
        *,
        proposal_index: int,
        record_proposal: bool,
        exact_budget_charge: int = 1,
    ) -> ExactEvaluationEntry:
        if self._sealed:
            raise RuntimeError("ledger is sealed")

        validate_budget_charge(exact_budget_charge)

        validated = validate_mask(mask)
        identity = _identity(validated)

        if record_proposal:
            self.proposals.append(
                ProposalEvent(
                    proposal_index=proposal_index,
                    mask_identity=identity,
                )
            )

        if identity in self._seen:
            return self._seen[identity]

        fidelity = float(self.evaluator(validated))

        if not math.isfinite(fidelity):
            raise ValueError("fidelity must be finite")

        retained = sum(validated)

        entry = ExactEvaluationEntry(
            mask_identity=identity,
            retained_count=retained,
            retained_proportion=retained_proportion(retained),
            fidelity=fidelity,
            qualifies=fidelity >= self.fidelity_threshold,
            evaluation_order=len(self.evaluations),
            exact_budget_charge=exact_budget_charge,
        )

        self.evaluations.append(entry)
        self._seen[identity] = entry

        if validated == _intact_mask():
            self._has_intact = True

        return entry

    def add_mask(
        self,
        mask: Sequence[int],
        *,
        proposal_index: int,
        exact_budget_charge: int = 1,
    ) -> ExactEvaluationEntry:
        return self._evaluate_mask(
            mask,
            proposal_index=proposal_index,
            record_proposal=True,
            exact_budget_charge=exact_budget_charge,
        )

    def seal(self) -> tuple[ExactEvaluationEntry, ...]:
        if not self._has_intact:
            raise ValueError("ledger missing intact baseline")

        if self._sealed:
            raise RuntimeError("ledger already sealed")

        self._sealed = True
        return tuple(self.evaluations)

def validate_budget_charge(charge: int) -> int:
    """Validate an injected exact-evaluation charge."""
    if not isinstance(charge, int):
        raise TypeError("budget charge must be an integer")
    if charge < 0:
        raise ValueError("budget charge must be non-negative")
    return charge
