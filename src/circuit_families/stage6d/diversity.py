"""Thin technical adapter for inherited diversity-forced machinery."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from circuit_families.interpretability import (
    diversity_forced_search as inherited_diversity,
)

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
RestartProposalSource = Callable[
    [DiscoveryRequest, object],
    Iterable[tuple[int, Iterable[Mask]]],
]


@dataclass(frozen=True)
class DiversityForcedAdapter:
    """Stage 6D boundary around inherited diversity-forced semantics."""

    restart_proposal_source: RestartProposalSource
    evaluator: Callable[[tuple[int, ...]], float]
    fidelity_threshold: float

    method_name: str = "diversity_forced"
    method_version: str = "inherited-technical-adapter/v1"
    inherited_entry_point: str = (
        "circuit_families.interpretability.diversity_forced_search."
        "run_sequential_family_search"
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
        restart_count = 0
        global_proposal_index = 0

        try:
            restart_iterable = self.restart_proposal_source(
                request,
                inherited_diversity.run_sequential_family_search,
            )

            for restart_index, masks in restart_iterable:
                if restart_index < 0:
                    raise ValueError("restart_index must be non-negative")
                if restart_index > request.maximum_restarts:
                    raise ValueError(
                        "synthetic restart source exceeded injected maximum_restarts"
                    )

                if restart_index > 0:
                    restart_count += 1
                    native.record_restart(restart_index)
                    trajectory.append(
                        event(
                            sequence_index=len(trajectory),
                            kind="restart",
                            restart_index=restart_index,
                            native_consumed=native.consumed,
                            exact_requested=exact_requests,
                            detail={
                                "inherited_entry_point": self.inherited_entry_point
                            },
                        )
                    )

                for mask in masks:
                    try:
                        native.consume(
                            detail={
                                "proposal_index": global_proposal_index,
                                "restart_index": restart_index,
                                "inherited_entry_point": self.inherited_entry_point,
                            }
                        )
                    except NativeBudgetExhausted:
                        trajectory.append(
                            event(
                                sequence_index=len(trajectory),
                                kind="termination",
                                restart_index=restart_index,
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
                            restart_count=restart_count,
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
                            restart_index=restart_index,
                            native_consumed=native.consumed,
                            exact_requested=exact_requests,
                            detail={
                                "proposal_index": global_proposal_index,
                                "restart_index": restart_index,
                                "inherited_entry_point": self.inherited_entry_point,
                            },
                        )
                    )

                    exact_requests += 1
                    trajectory.append(
                        event(
                            sequence_index=len(trajectory),
                            kind="exact_request",
                            restart_index=restart_index,
                            native_consumed=native.consumed,
                            exact_requested=exact_requests,
                            detail={"proposal_index": global_proposal_index},
                        )
                    )

                    try:
                        entry = exact.request(
                            mask,
                            proposal_index=global_proposal_index,
                        )
                    except ExactBudgetExhausted:
                        trajectory.append(
                            event(
                                sequence_index=len(trajectory),
                                kind="termination",
                                restart_index=restart_index,
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
                            restart_count=restart_count,
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
                            restart_index=restart_index,
                            native_consumed=native.consumed,
                            exact_requested=exact_requests,
                            detail={
                                "proposal_index": global_proposal_index,
                                "mask_identity": entry.mask_identity,
                                "evaluation_order": entry.evaluation_order,
                            },
                        )
                    )

                    global_proposal_index += 1

            exact.terminate()
            native.terminate()
            trajectory.append(
                event(
                    sequence_index=len(trajectory),
                    kind="termination",
                    restart_index=restart_count,
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
                restart_count=restart_count,
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
                    restart_index=restart_count,
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
                restart_count=restart_count,
                proposal_count=proposals,
                exact_request_count=exact_requests,
                exact_ledger_evidence=exact.evidence_record(),
                stopping_status="failed",
                trajectory=trajectory,
            )
