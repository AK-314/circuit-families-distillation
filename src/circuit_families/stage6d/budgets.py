"""Separate Stage 6D native and exact-evaluation budget ledgers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from circuit_families.stage6a.budget import (
    ExactBudgetUsage,
    TechnicalBudgetPolicy,
    validate_within_allowance,
)
from circuit_families.stage6a.ledger import ExactLedgerBuilder, validate_mask
from circuit_families.stage6a.models import ExactEvaluationEntry

LedgerState = Literal["active", "exhausted", "failed", "terminated"]


class NativeBudgetExhausted(RuntimeError):
    """Raised before a native-unit allowance would be exceeded."""


class ExactBudgetExhausted(RuntimeError):
    """Raised before a new unique exact evaluation would exceed allowance."""


@dataclass(frozen=True)
class NativeBudgetEvent:
    sequence_index: int
    kind: Literal["consume", "restart", "exhaustion", "failure", "termination"]
    consumed_before: int
    consumed_after: int
    amount: int
    restart_index: int | None
    detail: Mapping[str, Any]


@dataclass(frozen=True)
class ExactBudgetEvent:
    sequence_index: int
    kind: Literal[
        "baseline",
        "request",
        "duplicate",
        "result",
        "exhaustion",
        "failure",
        "termination",
    ]
    proposal_index: int | None
    mask_identity: str | None
    charged_before: int
    charged_after: int
    evaluation_order: int | None
    detail: Mapping[str, Any]


class NativeBudgetLedger:
    """Method-native accounting with no cross-method equivalence claim."""

    def __init__(self, *, unit: str, allowance: int) -> None:
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("native budget unit must be non-empty")
        if isinstance(allowance, bool) or not isinstance(allowance, int):
            raise TypeError("native allowance must be an integer")
        if allowance < 0:
            raise ValueError("native allowance must be non-negative")

        self.unit = unit
        self.allowance = allowance
        self.consumed = 0
        self.state: LedgerState = "active"
        self.events: list[NativeBudgetEvent] = []

    @property
    def exhausted(self) -> bool:
        return self.consumed >= self.allowance

    def _require_active(self) -> None:
        if self.state != "active":
            raise RuntimeError(f"native budget ledger is {self.state}")

    def _append(
        self,
        *,
        kind: Literal["consume", "restart", "exhaustion", "failure", "termination"],
        before: int,
        after: int,
        amount: int = 0,
        restart_index: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            NativeBudgetEvent(
                sequence_index=len(self.events),
                kind=kind,
                consumed_before=before,
                consumed_after=after,
                amount=amount,
                restart_index=restart_index,
                detail={} if detail is None else dict(detail),
            )
        )

    def consume(
        self,
        amount: int = 1,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self._require_active()

        if isinstance(amount, bool) or not isinstance(amount, int):
            raise TypeError("native budget charge must be an integer")
        if amount < 0:
            raise ValueError("native budget charge must be non-negative")

        before = self.consumed
        after = before + amount

        if after > self.allowance:
            self.state = "exhausted"
            self._append(
                kind="exhaustion",
                before=before,
                after=before,
                amount=amount,
                detail=detail,
            )
            raise NativeBudgetExhausted(
                f"native budget exhausted: {before}+{amount}>{self.allowance}"
            )

        self.consumed = after
        self._append(
            kind="consume",
            before=before,
            after=after,
            amount=amount,
            detail=detail,
        )

    def record_restart(self, restart_index: int) -> None:
        self._require_active()
        if (
            isinstance(restart_index, bool)
            or not isinstance(restart_index, int)
            or restart_index < 0
        ):
            raise ValueError("restart_index must be a non-negative integer")

        self._append(
            kind="restart",
            before=self.consumed,
            after=self.consumed,
            restart_index=restart_index,
        )

    def fail(self, *, detail: Mapping[str, Any] | None = None) -> None:
        self._require_active()
        self.state = "failed"
        self._append(
            kind="failure",
            before=self.consumed,
            after=self.consumed,
            detail=detail,
        )

    def terminate(self, *, detail: Mapping[str, Any] | None = None) -> None:
        self._require_active()
        self.state = "terminated"
        self._append(
            kind="termination",
            before=self.consumed,
            after=self.consumed,
            detail=detail,
        )


class Stage6AExactEvaluationBridge:
    """Shared Stage 6A unique-mask exact accounting for Stage 6D adapters."""

    def __init__(
        self,
        *,
        evaluator,
        fidelity_threshold: float,
        allowance: int,
    ) -> None:
        if isinstance(allowance, bool) or not isinstance(allowance, int):
            raise TypeError("exact allowance must be an integer")
        if allowance < 1:
            raise ValueError(
                "exact allowance must reserve one Stage 6A intact-baseline charge"
            )

        self.policy = TechnicalBudgetPolicy(
            exact_evaluation_allowance=allowance
        )
        self.builder = ExactLedgerBuilder(
            evaluator=evaluator,
            fidelity_threshold=fidelity_threshold,
        )
        self.usage = ExactBudgetUsage(
            evaluation_count=len(self.builder.evaluations),
            charged_count=sum(
                entry.exact_budget_charge for entry in self.builder.evaluations
            ),
        )
        validate_within_allowance(self.usage, self.policy)

        self.state: LedgerState = "active"
        self.events: list[ExactBudgetEvent] = []
        self._seen_masks: dict[tuple[int, ...], ExactEvaluationEntry] = {}

        baseline = self.builder.evaluations[0]
        self.events.append(
            ExactBudgetEvent(
                sequence_index=0,
                kind="baseline",
                proposal_index=None,
                mask_identity=baseline.mask_identity,
                charged_before=0,
                charged_after=self.usage.charged_count,
                evaluation_order=baseline.evaluation_order,
                detail={"source": "stage6a_intact_baseline"},
            )
        )

    def evidence_record(self) -> dict[str, Any]:
        payload = {
            "evaluations": [
                asdict(entry) for entry in self.builder.evaluations
            ],
            "proposals": [
                asdict(proposal) for proposal in self.builder.proposals
            ],
            "charged_count": self.usage.charged_count,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "evaluation_count": len(self.builder.evaluations),
            "proposal_count": len(self.builder.proposals),
            "charged_count": self.usage.charged_count,
        }

    @property
    def allowance(self) -> int:
        return self.policy.exact_evaluation_allowance

    @property
    def exhausted(self) -> bool:
        return self.usage.charged_count >= self.allowance

    def _require_active(self) -> None:
        if self.state != "active":
            raise RuntimeError(f"exact evaluation bridge is {self.state}")

    def _append(
        self,
        *,
        kind: Literal[
            "baseline",
            "request",
            "duplicate",
            "result",
            "exhaustion",
            "failure",
            "termination",
        ],
        proposal_index: int | None,
        mask_identity: str | None,
        before: int,
        after: int,
        evaluation_order: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            ExactBudgetEvent(
                sequence_index=len(self.events),
                kind=kind,
                proposal_index=proposal_index,
                mask_identity=mask_identity,
                charged_before=before,
                charged_after=after,
                evaluation_order=evaluation_order,
                detail={} if detail is None else dict(detail),
            )
        )

    def request(
        self,
        mask,
        *,
        proposal_index: int,
    ) -> ExactEvaluationEntry:
        self._require_active()

        if (
            isinstance(proposal_index, bool)
            or not isinstance(proposal_index, int)
            or proposal_index < 0
        ):
            raise ValueError("proposal_index must be a non-negative integer")

        validated = tuple(validate_mask(mask))
        is_intact = all(value == 1 for value in validated)

        duplicate_entry = self._seen_masks.get(validated)
        if duplicate_entry is None and is_intact:
            duplicate_entry = self.builder.evaluations[0]

        before = self.usage.charged_count

        if duplicate_entry is not None:
            returned = self.builder.add_mask(
                validated,
                proposal_index=proposal_index,
                exact_budget_charge=1,
            )
            if returned != duplicate_entry:
                raise RuntimeError("Stage 6A duplicate identity disagreement")

            self._append(
                kind="duplicate",
                proposal_index=proposal_index,
                mask_identity=returned.mask_identity,
                before=before,
                after=before,
                evaluation_order=returned.evaluation_order,
                detail={"charged": False},
            )
            return returned

        if before >= self.allowance:
            self.state = "exhausted"
            self._append(
                kind="exhaustion",
                proposal_index=proposal_index,
                mask_identity=None,
                before=before,
                after=before,
                detail={"reason": "unique_exact_allowance_exhausted"},
            )
            raise ExactBudgetExhausted(
                "exact evaluation allowance exhausted before unique request"
            )

        self._append(
            kind="request",
            proposal_index=proposal_index,
            mask_identity=None,
            before=before,
            after=before,
            detail={"charged_if_unique": True},
        )

        try:
            entry = self.builder.add_mask(
                validated,
                proposal_index=proposal_index,
                exact_budget_charge=1,
            )
        except Exception as exc:
            self.state = "failed"
            self._append(
                kind="failure",
                proposal_index=proposal_index,
                mask_identity=None,
                before=before,
                after=before,
                detail={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            raise

        self._seen_masks[validated] = entry
        self.usage = ExactBudgetUsage(
            evaluation_count=len(self.builder.evaluations),
            charged_count=sum(
                item.exact_budget_charge for item in self.builder.evaluations
            ),
        )
        validate_within_allowance(self.usage, self.policy)

        self._append(
            kind="result",
            proposal_index=proposal_index,
            mask_identity=entry.mask_identity,
            before=before,
            after=self.usage.charged_count,
            evaluation_order=entry.evaluation_order,
            detail={"charged": True},
        )
        return entry

    def terminate(self) -> tuple[ExactEvaluationEntry, ...]:
        self._require_active()
        sealed = self.builder.seal()
        self.state = "terminated"
        self._append(
            kind="termination",
            proposal_index=None,
            mask_identity=None,
            before=self.usage.charged_count,
            after=self.usage.charged_count,
            detail={"evaluation_count": len(sealed)},
        )
        return sealed
