"""Thin technical adapter for inherited greedy deletion machinery."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from circuit_families.interpretability import sparse_search as inherited_greedy

from .adapters import (
    build_discovery_result,
    event,
    require_request_matches_adapter,
)
from .budgets import (
    ExactBudgetExhausted,
    NativeBudgetExhausted,
    NativeBudgetLedger,
    Stage6AExactEvaluationBridge,
)
from .models import DiscoveryRequest, DiscoveryResult

Mask = Sequence[int]
ProposalSource = Callable[
    [DiscoveryRequest, object],
    Iterable[Mask],
]


@dataclass(frozen=True)
class GreedyDeletionAdapter:
    """Stage 6D boundary around inherited greedy deletion semantics."""

    proposal_source: ProposalSource
    evaluator: Callable[[tuple[int, ...]], float]
    fidelity_threshold: float

    method_name: str = "greedy_deletion"
    method_version: str = "inherited-technical-adapter/v1"
    inherited_entry_point: str = (
        "circuit_families.interpretability.sparse_search.greedy_sparse_search"
    )

    def run(self, request: DiscoveryRequest) -> DiscoveryResult:
        require_request_matches_adapter(
            request,
            method_name=self.method_name,
            method_version=self.method_version,
        )

        native = NativeBudgetLedger(
            unit=request.native_budget_unit,
            allowance=request.native_budget_allowance,
        )
        exact = Stage6AExactEvaluationBridge(
            evaluator=self.evaluator,
            fidelity_threshold=self.fidelity_threshold,
            allowance=request.exact_evaluation_allowance,
        )
        trajectory = []
        proposals = 0
        exact_requests = 0

        try:
            proposal_iterable = self.proposal_source(
                request,
                inherited_greedy.greedy_sparse_search,
            )

            for proposal_index, mask in enumerate(proposal_iterable):
                try:
                    native.consume(
                        detail={
                            "proposal_index": proposal_index,
                            "inherited_entry_point": self.inherited_entry_point,
                        }
                    )
                except NativeBudgetExhausted:
                    trajectory.append(
                        event(
                            sequence_index=len(trajectory),
                            kind="termination",
                            restart_index=0,
                            native_consumed=native.consumed,
                            exact_requested=exact_requests,
                            detail={"reason": "native_budget_exhausted"},
                        )
                    )
                    return build_discovery_result(
                        request=request,
                        native_consumed=native.consumed,
                        native_exhausted=True,
                        exact_consumed=exact.usage.charged_count,
                        exact_exhausted=exact.exhausted,
                        restart_count=0,
                        proposal_count=proposals,
                        exact_request_count=exact_requests,
                        exact_ledger_evidence=exact.evidence_record(),
                        stopping_status="native_budget_exhausted",
                        trajectory=trajectory,
                    )

                proposals += 1
                trajectory.append(
                    event(
                        sequence_index=len(trajectory),
                        kind="proposal",
                        restart_index=0,
                        native_consumed=native.consumed,
                        exact_requested=exact_requests,
                        detail={
                            "proposal_index": proposal_index,
                            "inherited_entry_point": self.inherited_entry_point,
                        },
                    )
                )

                exact_requests += 1
                trajectory.append(
                    event(
                        sequence_index=len(trajectory),
                        kind="exact_request",
                        restart_index=0,
                        native_consumed=native.consumed,
                        exact_requested=exact_requests,
                        detail={"proposal_index": proposal_index},
                    )
                )

                try:
                    entry = exact.request(
                        mask,
                        proposal_index=proposal_index,
                    )
                except ExactBudgetExhausted:
                    trajectory.append(
                        event(
                            sequence_index=len(trajectory),
                            kind="termination",
                            restart_index=0,
                            native_consumed=native.consumed,
                            exact_requested=exact_requests,
                            detail={"reason": "exact_budget_exhausted"},
                        )
                    )
                    return build_discovery_result(
                        request=request,
                        native_consumed=native.consumed,
                        native_exhausted=native.exhausted,
                        exact_consumed=exact.usage.charged_count,
                        exact_exhausted=True,
                        restart_count=0,
                        proposal_count=proposals,
                        exact_request_count=exact_requests,
                        exact_ledger_evidence=exact.evidence_record(),
                        stopping_status="exact_budget_exhausted",
                        trajectory=trajectory,
                    )

                trajectory.append(
                    event(
                        sequence_index=len(trajectory),
                        kind="exact_result",
                        restart_index=0,
                        native_consumed=native.consumed,
                        exact_requested=exact_requests,
                        detail={
                            "proposal_index": proposal_index,
                            "mask_identity": entry.mask_identity,
                            "evaluation_order": entry.evaluation_order,
                        },
                    )
                )

            exact.terminate()
            native.terminate()
            trajectory.append(
                event(
                    sequence_index=len(trajectory),
                    kind="termination",
                    restart_index=0,
                    native_consumed=native.consumed,
                    exact_requested=exact_requests,
                    detail={"reason": "completed"},
                )
            )

            return build_discovery_result(
                request=request,
                native_consumed=native.consumed,
                native_exhausted=native.exhausted,
                exact_consumed=exact.usage.charged_count,
                exact_exhausted=exact.exhausted,
                restart_count=0,
                proposal_count=proposals,
                exact_request_count=exact_requests,
                exact_ledger_evidence=exact.evidence_record(),
                stopping_status="completed",
                trajectory=trajectory,
            )

        except Exception as exc:
            if native.state == "active":
                native.fail(
                    detail={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )

            trajectory.append(
                event(
                    sequence_index=len(trajectory),
                    kind="failure",
                    restart_index=0,
                    native_consumed=native.consumed,
                    exact_requested=exact_requests,
                    detail={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
            )

            return build_discovery_result(
                request=request,
                native_consumed=native.consumed,
                native_exhausted=native.exhausted,
                exact_consumed=exact.usage.charged_count,
                exact_exhausted=exact.exhausted,
                restart_count=0,
                proposal_count=proposals,
                exact_request_count=exact_requests,
                exact_ledger_evidence=exact.evidence_record(),
                stopping_status="failed",
                trajectory=trajectory,
            )
