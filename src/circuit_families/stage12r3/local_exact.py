"""Local exact-fidelity-retaining perturbation calibration.

Parents must already be qualifying exact entries in a sealed synthetic
Stage 6A ledger.  Parent fidelity is never inherited.  Parents and all local
neighbors are evaluated afresh through a new Stage 6A exact ledger before
Stage 6E qualification and packing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
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
    SealedLedger,
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
from circuit_families.stage12r2.contracts import (
    canonical_sha256,
    validate_technical_record_payload,
)

LocalOperation = Literal["self", "add", "drop", "swap", "type_preserving_swap"]
LocalExactStatus = Literal[
    "evaluated",
    "duplicate_evaluation_reused",
    "evaluation_failed",
    "exact_budget_censored",
]


class LocalExactError(ValueError):
    """Raised when local exact calibration violates its contract."""


@dataclass(frozen=True)
class LocalExactProfile:
    profile_id: str
    run_id: str
    model_id: str
    discovery_method_id: str
    discovery_config_id: str
    component_basis_reference: str
    component_types: tuple[str, ...]
    fidelity_threshold: float
    exact_evaluation_allowance: int
    enabled_operations: tuple[str, ...] = (
        "add",
        "drop",
        "swap",
        "type_preserving_swap",
    )
    max_hamming_distance: int = 2
    scientific_data: bool = False
    production_eligible: bool = False
    independent_discovery_claim: bool = False

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "run_id",
            "model_id",
            "discovery_method_id",
            "discovery_config_id",
            "component_basis_reference",
        ):
            if not getattr(self, name):
                raise LocalExactError(f"{name} must be non-empty")

        if len(self.component_types) != COMPONENT_COUNT:
            raise LocalExactError(
                "local exact layer requires one type for each common component"
            )
        if any(not item for item in self.component_types):
            raise LocalExactError("component types must be non-empty")
        if not 0.0 <= self.fidelity_threshold <= 1.0:
            raise LocalExactError("fidelity threshold must lie in [0, 1]")
        if self.exact_evaluation_allowance < 1:
            raise LocalExactError(
                "exact allowance must reserve the intact Stage 6A baseline"
            )
        if self.max_hamming_distance not in (1, 2):
            raise LocalExactError(
                "v1 local neighborhoods support Hamming radius 1 or 2"
            )

        allowed = {
            "add",
            "drop",
            "swap",
            "type_preserving_swap",
        }
        if not self.enabled_operations:
            raise LocalExactError("at least one local operation is required")
        if len(set(self.enabled_operations)) != len(self.enabled_operations):
            raise LocalExactError("enabled operations must be unique")
        if any(item not in allowed for item in self.enabled_operations):
            raise LocalExactError("unsupported local operation")
        if self.scientific_data or self.production_eligible:
            raise LocalExactError("local exact profile must remain technical-only")
        if self.independent_discovery_claim:
            raise LocalExactError(
                "local perturbation is not an independent discovery method"
            )

        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ValidatedSeedRecord:
    seed_index: int
    mask: tuple[int, ...]
    mask_identity: str
    source_evaluation_order: int
    source_exact_fidelity: float
    source_retained_count: int
    source_proposal_indices: tuple[int, ...]


@dataclass(frozen=True)
class LocalProposalRecord:
    proposal_index: int
    proposal_reference: str
    parent_mask_identity: str
    operation: LocalOperation
    changed_indices: tuple[int, ...]
    hamming_distance: int
    mask_identity: str
    exact_status: LocalExactStatus
    exact_evaluation_order: int | None
    exact_fidelity: float | None
    qualifies: bool | None
    error: str | None


@dataclass(frozen=True)
class LocalExactResult:
    profile_identity: str
    source_seed_ledger_reference: str
    source_seed_ledger_hash: str
    validated_seeds: tuple[ValidatedSeedRecord, ...]
    proposals: tuple[LocalProposalRecord, ...]
    exact_ledger_hash: str
    exact_ledger_evaluation_count: int
    exact_ledger_proposal_count: int
    exact_budget_allowance: int
    exact_budget_charged: int
    exact_request_failure_count: int
    exact_request_censored_count: int
    qualification: QualificationResult
    endpoint1: Endpoint1Result
    endpoint2: Endpoint2ResultRecord
    packing_lower_bound: int
    procedure_censored: bool
    discovery_relationship: str = "local_exact_perturbation_not_independent_discovery"
    inherited_fidelity_used: bool = False
    surrogate_fidelity_used: bool = False
    scientific_data: bool = False
    production_eligible: bool = False
    mechanism_count_claim: bool = False

    def __post_init__(self) -> None:
        if (
            self.discovery_relationship
            != "local_exact_perturbation_not_independent_discovery"
        ):
            raise LocalExactError("local layer cannot claim independent discovery")
        if self.inherited_fidelity_used or self.surrogate_fidelity_used:
            raise LocalExactError("local result must use fresh exact fidelity only")
        if (
            self.scientific_data
            or self.production_eligible
            or self.mechanism_count_claim
        ):
            raise LocalExactError("local result exceeds technical claim boundary")
        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class _FreshEvidence:
    mask: tuple[int, ...]
    entry: ExactEvaluationEntry
    proposal_reference: str


def _validated_mask(mask: Sequence[int]) -> tuple[tuple[int, ...], str]:
    values = tuple(mask)
    if len(values) != COMPONENT_COUNT:
        raise LocalExactError("mask must use the common 516-component basis")
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise LocalExactError("mask must contain integer 0/1 values")

    retained = tuple(index for index, value in enumerate(values) if value)
    return values, canonical_mask_identity(retained)


def _ledger_hash(ledger: SealedLedger) -> str:
    return canonical_sha256(
        {
            "ledger_hash_contract": "stage12r3-source-sealed-ledger/v1",
            "profile": asdict(ledger.profile),
            "evaluations": [asdict(item) for item in ledger.evaluations],
            "proposals": [asdict(item) for item in ledger.proposals],
            "has_intact_baseline": ledger.has_intact_baseline,
            "sealed": ledger.sealed,
        }
    )


def _fresh_ledger_hash(builder: ExactLedgerBuilder) -> str:
    return canonical_sha256(
        {
            "ledger_hash_contract": "stage12r3-local-fresh-ledger/v1",
            "evaluations": [asdict(item) for item in builder.evaluations],
            "proposals": [asdict(item) for item in builder.proposals],
            "sealed": True,
        }
    )


def validate_seed_masks(
    *,
    seed_ledger: SealedLedger,
    seed_masks: Iterable[Sequence[int]],
) -> tuple[ValidatedSeedRecord, ...]:
    """Verify supplied seed masks against qualifying exact sealed-ledger entries."""

    if not seed_ledger.sealed:
        raise LocalExactError("seed ledger must be sealed")
    if not seed_ledger.has_intact_baseline:
        raise LocalExactError("seed ledger must contain intact baseline")
    if (
        not seed_ledger.profile.synthetic_only
        or seed_ledger.profile.scientific_data
        or seed_ledger.profile.production_eligible
    ):
        raise LocalExactError("seed ledger must be synthetic technical-only")

    evaluations: dict[str, ExactEvaluationEntry] = {}
    for entry in seed_ledger.evaluations:
        if entry.mask_identity in evaluations:
            raise LocalExactError("seed ledger has duplicate evaluation identity")
        evaluations[entry.mask_identity] = entry

    proposals_by_identity: dict[str, list[int]] = {}
    for event in seed_ledger.proposals:
        proposals_by_identity.setdefault(event.mask_identity, []).append(
            event.proposal_index
        )

    validated: list[ValidatedSeedRecord] = []
    seen: set[str] = set()

    for raw_mask in seed_masks:
        mask, identity = _validated_mask(raw_mask)
        if identity in seen:
            raise LocalExactError("duplicate seed masks are not permitted")
        seen.add(identity)

        entry = evaluations.get(identity)
        if entry is None:
            raise LocalExactError(
                "seed mask is absent from source exact ledger"
            )
        if not entry.qualifies:
            raise LocalExactError(
                "seed mask is not exact-qualified in source ledger"
            )

        retained_count = sum(mask)
        if entry.retained_count != retained_count:
            raise LocalExactError("seed retained count disagrees with ledger")
        if entry.retained_proportion != retained_count / COMPONENT_COUNT:
            raise LocalExactError(
                "seed retained proportion disagrees with ledger"
            )

        proposal_indices = tuple(
            sorted(proposals_by_identity.get(identity, ()))
        )
        if not proposal_indices:
            raise LocalExactError(
                "seed mask lacks source-ledger proposal provenance"
            )

        validated.append(
            ValidatedSeedRecord(
                seed_index=len(validated),
                mask=mask,
                mask_identity=identity,
                source_evaluation_order=entry.evaluation_order,
                source_exact_fidelity=entry.fidelity,
                source_retained_count=entry.retained_count,
                source_proposal_indices=proposal_indices,
            )
        )

    if not validated:
        raise LocalExactError("at least one qualified seed mask is required")

    return tuple(sorted(validated, key=lambda item: item.mask_identity))


def _neighbor_specs(
    profile: LocalExactProfile,
    seed: ValidatedSeedRecord,
) -> tuple[tuple[LocalOperation, tuple[int, ...], tuple[int, ...]], ...]:
    """Return deterministic one-operation neighbors inside declared Hamming radius."""

    retained = tuple(index for index, value in enumerate(seed.mask) if value)
    dropped = tuple(index for index, value in enumerate(seed.mask) if not value)

    specs: list[tuple[LocalOperation, tuple[int, ...], tuple[int, ...]]] = []

    if "add" in profile.enabled_operations and profile.max_hamming_distance >= 1:
        for add_index in dropped:
            values = list(seed.mask)
            values[add_index] = 1
            specs.append(("add", (add_index,), tuple(values)))

    if "drop" in profile.enabled_operations and profile.max_hamming_distance >= 1:
        for drop_index in retained:
            values = list(seed.mask)
            values[drop_index] = 0
            specs.append(("drop", (drop_index,), tuple(values)))

    if "swap" in profile.enabled_operations and profile.max_hamming_distance >= 2:
        for drop_index in retained:
            for add_index in dropped:
                values = list(seed.mask)
                values[drop_index] = 0
                values[add_index] = 1
                specs.append(
                    (
                        "swap",
                        tuple(sorted((drop_index, add_index))),
                        tuple(values),
                    )
                )

    if (
        "type_preserving_swap" in profile.enabled_operations
        and profile.max_hamming_distance >= 2
    ):
        for drop_index in retained:
            for add_index in dropped:
                if (
                    profile.component_types[drop_index]
                    != profile.component_types[add_index]
                ):
                    continue
                values = list(seed.mask)
                values[drop_index] = 0
                values[add_index] = 1
                specs.append(
                    (
                        "type_preserving_swap",
                        tuple(sorted((drop_index, add_index))),
                        tuple(values),
                    )
                )

    return tuple(specs)


def _validate_policy(
    profile: LocalExactProfile,
    policy: TechnicalEndpoint2Policy,
) -> None:
    if policy.scientific_data or policy.production_default:
        raise LocalExactError("Part E requires technical Stage 6E policy")
    if policy.component_basis_size != COMPONENT_COUNT:
        raise LocalExactError("Part E requires common 516-component basis")
    if policy.component_basis_reference != profile.component_basis_reference:
        raise LocalExactError("Stage 6E basis does not match local profile")
    if policy.fidelity_threshold != profile.fidelity_threshold:
        raise LocalExactError(
            "Stage 6A and Stage 6E fidelity thresholds must match"
        )


def run_local_exact_perturbations(
    *,
    profile: LocalExactProfile,
    policy: TechnicalEndpoint2Policy,
    seed_ledger: SealedLedger,
    seed_ledger_reference: str,
    seed_masks: Iterable[Sequence[int]],
    evaluator: Callable[[tuple[int, ...]], float],
) -> LocalExactResult:
    """Re-evaluate qualified parents and bounded local neighbors exactly."""

    _validate_policy(profile, policy)
    if not seed_ledger_reference:
        raise LocalExactError("seed ledger reference must be non-empty")

    seeds = validate_seed_masks(
        seed_ledger=seed_ledger,
        seed_masks=seed_masks,
    )
    source_hash = _ledger_hash(seed_ledger)

    builder = ExactLedgerBuilder(
        evaluator=evaluator,
        fidelity_threshold=profile.fidelity_threshold,
    )

    intact = (1,) * COMPONENT_COUNT
    try:
        builder.add_mask(intact, proposal_index=0, exact_budget_charge=1)
    except Exception as exc:
        raise LocalExactError(
            "fresh local ledger intact baseline evaluation failed"
        ) from exc

    charged = 1
    global_proposal_index = 1
    successful = {entry.mask_identity for entry in builder.evaluations}
    failures = 0
    censored = 0

    records: list[LocalProposalRecord] = []
    evidence_specs: list[_FreshEvidence] = []

    candidate_specs: list[
        tuple[
            ValidatedSeedRecord,
            LocalOperation,
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = []

    for seed in seeds:
        candidate_specs.append((seed, "self", (), seed.mask))
        for operation, changed, mask in _neighbor_specs(profile, seed):
            candidate_specs.append((seed, operation, changed, mask))

    for seed, operation, changed, raw_mask in candidate_specs:
        mask, mask_identity = _validated_mask(raw_mask)
        hamming_distance = sum(
            left != right for left, right in zip(seed.mask, mask, strict=True)
        )
        if hamming_distance > profile.max_hamming_distance:
            raise LocalExactError("generated proposal exceeds local radius")

        proposal_reference = (
            f"stage12r3:{profile.run_id}:parent:{seed.mask_identity}:"
            f"{operation}:{','.join(str(i) for i in changed) or 'self'}"
        )

        duplicate = mask_identity in successful
        if (
            not duplicate
            and charged >= profile.exact_evaluation_allowance
        ):
            records.append(
                LocalProposalRecord(
                    proposal_index=len(records),
                    proposal_reference=proposal_reference,
                    parent_mask_identity=seed.mask_identity,
                    operation=operation,
                    changed_indices=changed,
                    hamming_distance=hamming_distance,
                    mask_identity=mask_identity,
                    exact_status="exact_budget_censored",
                    exact_evaluation_order=None,
                    exact_fidelity=None,
                    qualifies=None,
                    error="global exact-evaluation allowance exhausted",
                )
            )
            censored += 1
            global_proposal_index += 1
            continue

        try:
            entry = builder.add_mask(
                mask,
                proposal_index=global_proposal_index,
                exact_budget_charge=1,
            )
        except Exception as exc:
            records.append(
                LocalProposalRecord(
                    proposal_index=len(records),
                    proposal_reference=proposal_reference,
                    parent_mask_identity=seed.mask_identity,
                    operation=operation,
                    changed_indices=changed,
                    hamming_distance=hamming_distance,
                    mask_identity=mask_identity,
                    exact_status="evaluation_failed",
                    exact_evaluation_order=None,
                    exact_fidelity=None,
                    qualifies=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            failures += 1
            global_proposal_index += 1
            continue

        if duplicate:
            status: LocalExactStatus = "duplicate_evaluation_reused"
        else:
            status = "evaluated"
            charged += 1
            successful.add(entry.mask_identity)

        records.append(
            LocalProposalRecord(
                proposal_index=len(records),
                proposal_reference=proposal_reference,
                parent_mask_identity=seed.mask_identity,
                operation=operation,
                changed_indices=changed,
                hamming_distance=hamming_distance,
                mask_identity=entry.mask_identity,
                exact_status=status,
                exact_evaluation_order=entry.evaluation_order,
                exact_fidelity=entry.fidelity,
                qualifies=entry.qualifies,
                error=None,
            )
        )
        evidence_specs.append(
            _FreshEvidence(
                mask=mask,
                entry=entry,
                proposal_reference=proposal_reference,
            )
        )
        global_proposal_index += 1

    entries = builder.seal()

    validate_within_allowance(
        ExactBudgetUsage(
            evaluation_count=len(entries),
            charged_count=charged,
        ),
        TechnicalBudgetPolicy(profile.exact_evaluation_allowance),
    )

    procedure_censored = bool(failures or censored)
    endpoint1 = reduce_endpoint1(
        entries,
        termination=TerminationStatus(
            status="censored" if procedure_censored else "completed",
            procedure_censored=procedure_censored,
        ),
    )

    exact_ledger_hash = _fresh_ledger_hash(builder)
    exact_ledger_reference = (
        f"stage12r3-local-exact-ledger:{profile.run_id}"
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
                f"stage12r3-local-exact:{spec.entry.evaluation_order}:"
                f"{spec.entry.mask_identity}"
            ),
            source_ledger_reference=exact_ledger_reference,
            source_ledger_hash=exact_ledger_hash,
            recomputed_ledger_hash=exact_ledger_hash,
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

    return LocalExactResult(
        profile_identity=profile.identity,
        source_seed_ledger_reference=seed_ledger_reference,
        source_seed_ledger_hash=source_hash,
        validated_seeds=seeds,
        proposals=tuple(records),
        exact_ledger_hash=exact_ledger_hash,
        exact_ledger_evaluation_count=len(entries),
        exact_ledger_proposal_count=len(builder.proposals),
        exact_budget_allowance=profile.exact_evaluation_allowance,
        exact_budget_charged=charged,
        exact_request_failure_count=failures,
        exact_request_censored_count=censored,
        qualification=qualification,
        endpoint1=endpoint1,
        endpoint2=endpoint2,
        packing_lower_bound=len(selected),
        procedure_censored=procedure_censored,
    )
