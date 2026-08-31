"""Ordinary independent-restart calibration baseline.

Each restart receives only immutable method/run identity, its derived seed,
the declared native budget, and the common basis identity.  No prior-restart
masks, packing state, diversity pressure, or packing-aware feedback is exposed.

Exact evaluation is performed through Stage 6A.  Final exact evidence is
qualified and packed through Stage 6E.  This remains the same Stage 12-R1
discovery family, not a second independent discovery method.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from circuit_families.stage6a import (
    COMPONENT_COUNT,
    ExactBudgetUsage,
    ExactLedgerBuilder,
    TechnicalBudgetPolicy,
    TerminationStatus,
    reduce_endpoint1,
    validate_within_allowance,
)
from circuit_families.stage6a.models import (
    Endpoint1Result,
    ExactEvaluationEntry,
    canonical_mask_identity,
)
from circuit_families.stage6e.packing import (
    build_compatibility_graph,
    exact_maximum_compatible_subset,
    qualify_and_deduplicate,
    recompute_endpoint2,
)
from circuit_families.stage6e.records import (
    Endpoint2ResultRecord,
    ExactCandidateEvidence,
    QualificationResult,
    TechnicalEndpoint2Policy,
)
from circuit_families.stage12r1 import (
    ALGORITHM_FAMILY,
    NATIVE_BUDGET_UNIT,
    ExactBudgetRecord,
    NativeBudgetRecord,
)
from circuit_families.stage12r2.contracts import (
    canonical_sha256,
    validate_technical_record_payload,
)

RestartTerminalState = Literal["completed", "budget_exhausted", "failed"]
ProposalExactStatus = Literal[
    "evaluated",
    "duplicate_evaluation_reused",
    "evaluation_failed",
    "exact_budget_censored",
]


class OrdinaryRestartError(ValueError):
    """Raised when the ordinary-restart contract is violated."""


@dataclass(frozen=True)
class OrdinaryRestartProfile:
    profile_id: str
    run_id: str
    method_name: str
    method_version: str
    discovery_config_id: str
    model_id: str
    component_basis_reference: str
    fidelity_threshold: float
    restart_count: int
    root_seed: int
    native_budget_per_restart: int
    exact_evaluation_allowance: int
    algorithm_family: str = ALGORITHM_FAMILY
    native_budget_unit: str = NATIVE_BUDGET_UNIT
    uses_diversity_pressure: bool = False
    uses_packing_feedback: bool = False
    uses_prior_restart_mask_exclusion: bool = False
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "run_id",
            "method_name",
            "method_version",
            "discovery_config_id",
            "model_id",
            "component_basis_reference",
        ):
            if not getattr(self, name):
                raise OrdinaryRestartError(f"{name} must be non-empty")

        if self.algorithm_family != ALGORITHM_FAMILY:
            raise OrdinaryRestartError(
                "ordinary restarts must remain the Stage 12-R1 discovery family"
            )
        if self.native_budget_unit != NATIVE_BUDGET_UNIT:
            raise OrdinaryRestartError(
                "native budget unit must remain the Stage 12-R1 method unit"
            )
        if self.restart_count <= 0:
            raise OrdinaryRestartError("restart_count must be positive")
        if self.root_seed < 0:
            raise OrdinaryRestartError("root_seed must be non-negative")
        if self.native_budget_per_restart < 0:
            raise OrdinaryRestartError(
                "native budget per restart must be non-negative"
            )
        if self.exact_evaluation_allowance < 1:
            raise OrdinaryRestartError(
                "exact allowance must include the intact Stage 6A baseline"
            )
        if not 0.0 <= self.fidelity_threshold <= 1.0:
            raise OrdinaryRestartError(
                "fidelity threshold must lie in [0, 1]"
            )
        if (
            self.uses_diversity_pressure
            or self.uses_packing_feedback
            or self.uses_prior_restart_mask_exclusion
        ):
            raise OrdinaryRestartError(
                "ordinary baseline cannot use cross-restart diversity, "
                "packing feedback, or prior-mask exclusion"
            )
        if self.scientific_data or self.production_eligible:
            raise OrdinaryRestartError(
                "ordinary restart baseline must remain technical-only"
            )

        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))

    @property
    def discovery_method_id(self) -> str:
        return f"{self.method_name}:{self.method_version}"


@dataclass(frozen=True)
class OrdinaryRestartContext:
    profile_identity: str
    run_id: str
    restart_index: int
    restart_seed: int
    method_name: str
    method_version: str
    discovery_config_id: str
    algorithm_family: str
    native_budget_unit: str
    native_budget_allowance: int
    component_basis_reference: str
    component_basis_size: int = COMPONENT_COUNT

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class RestartDiscoveryOutput:
    proposals: tuple[tuple[int, ...], ...]
    native_work_consumed: int
    terminal_state: RestartTerminalState = "completed"
    failure_reason: str | None = None
    used_cross_restart_information: bool = False

    def __post_init__(self) -> None:
        if self.native_work_consumed < 0:
            raise OrdinaryRestartError(
                "native work consumed must be non-negative"
            )
        if self.terminal_state not in (
            "completed",
            "budget_exhausted",
            "failed",
        ):
            raise OrdinaryRestartError("invalid restart terminal state")
        if self.terminal_state == "failed" and not self.failure_reason:
            raise OrdinaryRestartError(
                "failed restart requires a failure reason"
            )
        if self.used_cross_restart_information:
            raise OrdinaryRestartError(
                "ordinary restart output declares cross-restart information use"
            )


DiscoveryProcedure = Callable[
    [OrdinaryRestartContext],
    RestartDiscoveryOutput,
]


@dataclass(frozen=True)
class RestartProposalRecord:
    restart_index: int
    proposal_index: int
    proposal_reference: str
    mask_identity: str | None
    exact_status: ProposalExactStatus
    exact_evaluation_order: int | None
    exact_fidelity: float | None
    qualifies: bool | None
    error: str | None


@dataclass(frozen=True)
class OrdinaryRestartRecord:
    restart_index: int
    restart_seed: int
    context_identity: str
    native_budget: NativeBudgetRecord
    terminal_state: RestartTerminalState
    failure_kind: str
    failure_reason: str | None
    proposals: tuple[RestartProposalRecord, ...]
    used_cross_restart_information: bool = False


@dataclass(frozen=True)
class OrdinaryRestartBaselineResult:
    profile_identity: str
    run_identity: str
    discovery_family: str
    discovery_relationship: str
    restart_records: tuple[OrdinaryRestartRecord, ...]
    exact_budget: ExactBudgetRecord
    exact_ledger_hash: str
    exact_ledger_evaluation_count: int
    exact_ledger_proposal_count: int
    exact_request_failure_count: int
    exact_request_censored_count: int
    qualification: QualificationResult
    endpoint1: Endpoint1Result
    endpoint2: Endpoint2ResultRecord
    qualified_mask_identities: tuple[str, ...]
    packing_lower_bound: int
    requested_restart_count: int
    completed_restart_count: int
    failed_restart_count: int
    native_exhausted_restart_count: int
    raw_restart_proposal_count: int
    procedure_censored: bool
    scientific_data: bool = False
    production_eligible: bool = False
    mechanism_count_claim: bool = False

    def __post_init__(self) -> None:
        if self.discovery_family != ALGORITHM_FAMILY:
            raise OrdinaryRestartError(
                "ordinary restart result changed discovery family"
            )
        if (
            self.discovery_relationship
            != "same_discovery_family_ordinary_restart"
        ):
            raise OrdinaryRestartError(
                "ordinary restart cannot be relabelled independent"
            )
        if (
            self.scientific_data
            or self.production_eligible
            or self.mechanism_count_claim
        ):
            raise OrdinaryRestartError(
                "ordinary restart result violates technical claim boundary"
            )
        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class _EvidenceSpec:
    mask: tuple[int, ...]
    entry: ExactEvaluationEntry
    proposal_reference: str


def derive_restart_seed(
    profile: OrdinaryRestartProfile,
    *,
    restart_index: int,
) -> int:
    if restart_index < 0 or restart_index >= profile.restart_count:
        raise OrdinaryRestartError("restart index outside declared range")

    digest = canonical_sha256(
        {
            "seed_contract": "stage12r3-ordinary-restart-seed/v1",
            "profile_identity": profile.identity,
            "run_id": profile.run_id,
            "method_name": profile.method_name,
            "method_version": profile.method_version,
            "discovery_config_id": profile.discovery_config_id,
            "restart_index": restart_index,
            "root_seed": profile.root_seed,
        }
    )
    return int(digest[:16], 16)


def _context(
    profile: OrdinaryRestartProfile,
    *,
    restart_index: int,
) -> OrdinaryRestartContext:
    return OrdinaryRestartContext(
        profile_identity=profile.identity,
        run_id=profile.run_id,
        restart_index=restart_index,
        restart_seed=derive_restart_seed(
            profile,
            restart_index=restart_index,
        ),
        method_name=profile.method_name,
        method_version=profile.method_version,
        discovery_config_id=profile.discovery_config_id,
        algorithm_family=profile.algorithm_family,
        native_budget_unit=profile.native_budget_unit,
        native_budget_allowance=profile.native_budget_per_restart,
        component_basis_reference=profile.component_basis_reference,
    )


def _preflight_mask_identity(mask: Sequence[int]) -> tuple[tuple[int, ...], str]:
    values = tuple(mask)
    if len(values) != COMPONENT_COUNT:
        raise OrdinaryRestartError(
            "ordinary restart proposal must use the common 516-component mask"
        )
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise OrdinaryRestartError(
            "ordinary restart proposal mask must contain only integer 0/1 values"
        )

    retained = tuple(index for index, value in enumerate(values) if value)
    return values, canonical_mask_identity(retained)


def _ledger_hash(builder: ExactLedgerBuilder) -> str:
    return canonical_sha256(
        {
            "ledger_hash_contract": "stage12r3-stage6a-ledger-hash/v1",
            "evaluations": [asdict(entry) for entry in builder.evaluations],
            "proposals": [asdict(event) for event in builder.proposals],
            "sealed": True,
        }
    )


def _validate_profile_policy(
    profile: OrdinaryRestartProfile,
    policy: TechnicalEndpoint2Policy,
) -> None:
    if policy.scientific_data or policy.production_default:
        raise OrdinaryRestartError(
            "Part D requires a technical non-production Stage 6E policy"
        )
    if policy.component_basis_size != COMPONENT_COUNT:
        raise OrdinaryRestartError(
            "Part D requires the common 516-component Stage 6E basis"
        )
    if policy.component_basis_reference != profile.component_basis_reference:
        raise OrdinaryRestartError(
            "Stage 6E policy basis does not match restart profile"
        )
    if policy.fidelity_threshold != profile.fidelity_threshold:
        raise OrdinaryRestartError(
            "Stage 6A and Stage 6E fidelity thresholds must match"
        )


def run_ordinary_restart_baseline(
    *,
    profile: OrdinaryRestartProfile,
    policy: TechnicalEndpoint2Policy,
    evaluator: Callable[[tuple[int, ...]], float],
    discovery_procedure: DiscoveryProcedure,
    execution_order: tuple[int, ...] | None = None,
) -> OrdinaryRestartBaselineResult:
    """Run ordinary same-family independent restarts.

    Native work is bounded per restart.  Exact work is bounded globally and
    deduplicated through one common Stage 6A ledger.
    """

    _validate_profile_policy(profile, policy)

    if execution_order is None:
        execution_order = tuple(range(profile.restart_count))
    if tuple(sorted(execution_order)) != tuple(range(profile.restart_count)):
        raise OrdinaryRestartError(
            "execution_order must be a permutation of declared restarts"
        )

    builder = ExactLedgerBuilder(
        evaluator=evaluator,
        fidelity_threshold=profile.fidelity_threshold,
    )

    intact = (1,) * COMPONENT_COUNT
    try:
        builder.add_mask(
            intact,
            proposal_index=0,
            exact_budget_charge=1,
        )
    except Exception as exc:
        raise OrdinaryRestartError(
            "intact Stage 6A baseline exact evaluation failed"
        ) from exc

    exact_charges = 1
    global_proposal_index = 1
    successful_identities = {
        entry.mask_identity for entry in builder.evaluations
    }

    restart_records: list[OrdinaryRestartRecord] = []
    evidence_specs: list[_EvidenceSpec] = []
    exact_failures = 0
    exact_censored = 0
    raw_proposals = 0

    for restart_index in execution_order:
        context = _context(profile, restart_index=restart_index)

        try:
            output = discovery_procedure(context)
        except Exception as exc:
            restart_records.append(
                OrdinaryRestartRecord(
                    restart_index=restart_index,
                    restart_seed=context.restart_seed,
                    context_identity=context.identity,
                    native_budget=NativeBudgetRecord(
                        unit=profile.native_budget_unit,
                        allowance=profile.native_budget_per_restart,
                        consumed=0,
                        exhausted=False,
                    ),
                    terminal_state="failed",
                    failure_kind="procedure_exception",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    proposals=(),
                )
            )
            continue

        if output.used_cross_restart_information:
            raise OrdinaryRestartError(
                "cross-restart information leakage is prohibited"
            )
        if output.native_work_consumed > profile.native_budget_per_restart:
            raise OrdinaryRestartError(
                "restart exceeded declared method-native budget"
            )

        proposal_records: list[RestartProposalRecord] = []

        for local_index, raw_mask in enumerate(output.proposals):
            raw_proposals += 1
            proposal_reference = (
                f"stage12r3:{profile.run_id}:restart:{restart_index}:"
                f"proposal:{local_index}"
            )

            try:
                mask, mask_identity = _preflight_mask_identity(raw_mask)
            except Exception as exc:
                proposal_records.append(
                    RestartProposalRecord(
                        restart_index=restart_index,
                        proposal_index=local_index,
                        proposal_reference=proposal_reference,
                        mask_identity=None,
                        exact_status="evaluation_failed",
                        exact_evaluation_order=None,
                        exact_fidelity=None,
                        qualifies=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                exact_failures += 1
                global_proposal_index += 1
                continue

            is_duplicate = mask_identity in successful_identities

            if (
                not is_duplicate
                and exact_charges >= profile.exact_evaluation_allowance
            ):
                proposal_records.append(
                    RestartProposalRecord(
                        restart_index=restart_index,
                        proposal_index=local_index,
                        proposal_reference=proposal_reference,
                        mask_identity=mask_identity,
                        exact_status="exact_budget_censored",
                        exact_evaluation_order=None,
                        exact_fidelity=None,
                        qualifies=None,
                        error="global exact-evaluation allowance exhausted",
                    )
                )
                exact_censored += 1
                global_proposal_index += 1
                continue

            try:
                entry = builder.add_mask(
                    mask,
                    proposal_index=global_proposal_index,
                    exact_budget_charge=1,
                )
            except Exception as exc:
                proposal_records.append(
                    RestartProposalRecord(
                        restart_index=restart_index,
                        proposal_index=local_index,
                        proposal_reference=proposal_reference,
                        mask_identity=mask_identity,
                        exact_status="evaluation_failed",
                        exact_evaluation_order=None,
                        exact_fidelity=None,
                        qualifies=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                exact_failures += 1
                global_proposal_index += 1
                continue

            status: ProposalExactStatus
            if is_duplicate:
                status = "duplicate_evaluation_reused"
            else:
                status = "evaluated"
                exact_charges += 1
                successful_identities.add(entry.mask_identity)

            proposal_records.append(
                RestartProposalRecord(
                    restart_index=restart_index,
                    proposal_index=local_index,
                    proposal_reference=proposal_reference,
                    mask_identity=entry.mask_identity,
                    exact_status=status,
                    exact_evaluation_order=entry.evaluation_order,
                    exact_fidelity=entry.fidelity,
                    qualifies=entry.qualifies,
                    error=None,
                )
            )
            evidence_specs.append(
                _EvidenceSpec(
                    mask=mask,
                    entry=entry,
                    proposal_reference=proposal_reference,
                )
            )
            global_proposal_index += 1

        restart_records.append(
            OrdinaryRestartRecord(
                restart_index=restart_index,
                restart_seed=context.restart_seed,
                context_identity=context.identity,
                native_budget=NativeBudgetRecord(
                    unit=profile.native_budget_unit,
                    allowance=profile.native_budget_per_restart,
                    consumed=output.native_work_consumed,
                    exhausted=(
                        output.native_work_consumed
                        >= profile.native_budget_per_restart
                    ),
                ),
                terminal_state=output.terminal_state,
                failure_kind=(
                    "procedure_reported_failure"
                    if output.terminal_state == "failed"
                    else "none"
                ),
                failure_reason=output.failure_reason,
                proposals=tuple(proposal_records),
                used_cross_restart_information=False,
            )
        )

    entries = builder.seal()

    validate_within_allowance(
        ExactBudgetUsage(
            evaluation_count=len(entries),
            charged_count=exact_charges,
        ),
        TechnicalBudgetPolicy(profile.exact_evaluation_allowance),
    )

    procedure_censored = bool(
        exact_censored
        or exact_failures
        or any(record.terminal_state == "failed" for record in restart_records)
    )
    endpoint1 = reduce_endpoint1(
        entries,
        termination=TerminationStatus(
            status="censored" if procedure_censored else "completed",
            procedure_censored=procedure_censored,
        ),
    )

    ledger_hash = _ledger_hash(builder)
    ledger_reference = (
        f"stage12r3-ordinary-restart-ledger:{profile.run_id}"
    )

    evidence = tuple(
        ExactCandidateEvidence(
            model_id=profile.model_id,
            discovery_method_id=profile.discovery_method_id,
            discovery_config_id=profile.discovery_config_id,
            source_budget_reference=policy.source_budget_reference,
            fidelity_metric_reference=policy.fidelity_metric_reference,
            component_basis_reference=policy.component_basis_reference,
            component_basis_size=policy.component_basis_size,
            mask=spec.mask,
            mask_identity=spec.entry.mask_identity,
            exact_fidelity=spec.entry.fidelity,
            proposal_reference=spec.proposal_reference,
            exact_evaluation_reference=(
                "stage12r3-exact:"
                f"{spec.entry.evaluation_order}:"
                f"{spec.entry.mask_identity}"
            ),
            source_ledger_reference=ledger_reference,
            source_ledger_hash=ledger_hash,
            recomputed_ledger_hash=ledger_hash,
        )
        for spec in evidence_specs
    )

    qualification = qualify_and_deduplicate(
        evidence,
        policy,
        model_id=profile.model_id,
        discovery_method_id=profile.discovery_method_id,
        discovery_config_id=profile.discovery_config_id,
    )

    graph = build_compatibility_graph(
        qualification.qualified_candidates,
        policy,
    )
    selected = exact_maximum_compatible_subset(graph, policy)
    endpoint2 = recompute_endpoint2(qualification, policy)

    ordered_records = tuple(
        sorted(restart_records, key=lambda record: record.restart_index)
    )
    failed_count = sum(
        record.terminal_state == "failed"
        for record in ordered_records
    )
    exhausted_count = sum(
        record.native_budget.exhausted
        for record in ordered_records
    )

    exact_budget = ExactBudgetRecord(
        allowance=profile.exact_evaluation_allowance,
        charged=exact_charges,
        evaluation_count=len(entries),
        proposal_count=len(builder.proposals),
        exhausted=exact_charges >= profile.exact_evaluation_allowance,
    )

    result = OrdinaryRestartBaselineResult(
        profile_identity=profile.identity,
        run_identity=canonical_sha256(
            {
                "profile_identity": profile.identity,
                "run_id": profile.run_id,
                "discovery_method_id": profile.discovery_method_id,
                "discovery_config_id": profile.discovery_config_id,
            }
        ),
        discovery_family=profile.algorithm_family,
        discovery_relationship="same_discovery_family_ordinary_restart",
        restart_records=ordered_records,
        exact_budget=exact_budget,
        exact_ledger_hash=ledger_hash,
        exact_ledger_evaluation_count=len(entries),
        exact_ledger_proposal_count=len(builder.proposals),
        exact_request_failure_count=exact_failures,
        exact_request_censored_count=exact_censored,
        qualification=qualification,
        endpoint1=endpoint1,
        endpoint2=endpoint2,
        qualified_mask_identities=tuple(
            candidate.mask_identity
            for candidate in qualification.qualified_candidates
        ),
        packing_lower_bound=len(selected),
        requested_restart_count=profile.restart_count,
        completed_restart_count=(
            profile.restart_count - failed_count
        ),
        failed_restart_count=failed_count,
        native_exhausted_restart_count=exhausted_count,
        raw_restart_proposal_count=raw_proposals,
        procedure_censored=procedure_censored,
    )
    return result
