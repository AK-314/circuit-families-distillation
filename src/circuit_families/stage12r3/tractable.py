"""Exact feasible-region calibration on a deliberately tractable fixture.

The physical mask remains the common 516-component mask.  Only a small,
declared subset of coordinates is free; every admissible assignment of those
coordinates is enumerated exactly.  The exhaustive arm uses Stage 6A exact
evaluation and Stage 6E qualification/packing unchanged.

An injected search procedure sees only the public tractable search context,
not the qualifying-pattern inventory.  Search output is then evaluated by the
same exact fixture and reducers and compared against the exhaustive reference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

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

from .contracts import CompletenessCertificate


class TractableCalibrationError(ValueError):
    """Raised when the tractable calibration contract is violated."""


@dataclass(frozen=True)
class TractableFixtureProfile:
    profile_id: str
    run_id: str
    model_id: str
    discovery_method_id: str
    discovery_config_id: str
    component_basis_reference: str
    free_component_indices: tuple[int, ...]
    qualifying_free_patterns: tuple[int, ...]
    fidelity_threshold: float
    search_exact_evaluation_allowance: int
    max_free_components: int = 12
    scientific_data: bool = False
    production_eligible: bool = False
    teacher_seed_transfer: bool = False
    main_experiment_transfer: bool = False

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
                raise TractableCalibrationError(f"{name} must be non-empty")

        if not 1 <= self.max_free_components <= 12:
            raise TractableCalibrationError(
                "max_free_components must lie in [1, 12]"
            )
        if not self.free_component_indices:
            raise TractableCalibrationError(
                "tractable fixture requires free components"
            )
        if len(self.free_component_indices) > self.max_free_components:
            raise TractableCalibrationError(
                "free-coordinate universe exceeds tractable bound"
            )
        if tuple(sorted(set(self.free_component_indices))) != (
            self.free_component_indices
        ):
            raise TractableCalibrationError(
                "free component indices must be sorted and unique"
            )
        if any(
            index < 0 or index >= COMPONENT_COUNT
            for index in self.free_component_indices
        ):
            raise TractableCalibrationError(
                "free component index outside common basis"
            )

        universe_size = 1 << len(self.free_component_indices)
        if not self.qualifying_free_patterns:
            raise TractableCalibrationError(
                "fixture requires at least one feasible pattern"
            )
        if tuple(sorted(set(self.qualifying_free_patterns))) != (
            self.qualifying_free_patterns
        ):
            raise TractableCalibrationError(
                "qualifying patterns must be sorted and unique"
            )
        if any(
            pattern < 0 or pattern >= universe_size
            for pattern in self.qualifying_free_patterns
        ):
            raise TractableCalibrationError(
                "qualifying pattern outside declared free-coordinate universe"
            )

        if not 0.0 <= self.fidelity_threshold <= 1.0:
            raise TractableCalibrationError(
                "fidelity threshold must lie in [0, 1]"
            )
        if self.search_exact_evaluation_allowance < 1:
            raise TractableCalibrationError(
                "search exact allowance must reserve intact baseline"
            )
        if self.scientific_data or self.production_eligible:
            raise TractableCalibrationError(
                "tractable fixture must remain technical-only"
            )
        if self.teacher_seed_transfer or self.main_experiment_transfer:
            raise TractableCalibrationError(
                "tractable calibration forbids teacher-seed/main transfer"
            )

        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))

    @property
    def admissible_mask_count(self) -> int:
        return 1 << len(self.free_component_indices)


@dataclass(frozen=True)
class TractableSearchContext:
    fixture_identity: str
    run_id: str
    component_basis_reference: str
    component_basis_size: int
    free_component_indices: tuple[int, ...]
    admissible_mask_count: int
    exact_evaluation_allowance: int
    qualifying_patterns_exposed: bool = False
    teacher_seed_exposed: bool = False
    main_experiment_state_exposed: bool = False

    def __post_init__(self) -> None:
        if (
            self.qualifying_patterns_exposed
            or self.teacher_seed_exposed
            or self.main_experiment_state_exposed
        ):
            raise TractableCalibrationError(
                "search context leaks calibration ground truth or external state"
            )


@dataclass(frozen=True)
class TractableSearchOutput:
    proposals: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SearchProposalRecord:
    proposal_index: int
    proposal_reference: str
    mask_identity: str | None
    status: str
    exact_evaluation_order: int | None
    exact_fidelity: float | None
    qualifies: bool | None
    error: str | None


@dataclass(frozen=True)
class TractableCalibrationResult:
    profile_identity: str
    certificate: CompletenessCertificate
    admissible_mask_count: int
    admissible_mask_identities: tuple[str, ...]
    admissible_universe_hash: str
    feasible_mask_count: int
    feasible_mask_identities: tuple[str, ...]
    feasible_inventory_hash: str
    exhaustive_exact_evaluation_count: int
    exhaustive_endpoint1: Endpoint1Result
    exhaustive_qualification: QualificationResult
    exhaustive_endpoint2: Endpoint2ResultRecord
    certified_packing_optimum: int
    search_proposals: tuple[SearchProposalRecord, ...]
    search_raw_proposal_count: int
    search_valid_proposal_count: int
    search_invalid_proposal_count: int
    search_duplicate_proposal_count: int
    search_exact_censored_count: int
    search_exact_failure_count: int
    search_exact_evaluation_allowance: int
    search_exact_evaluation_count: int
    search_exact_budget_charged: int
    search_exact_evaluation_coverage: float
    search_qualification: QualificationResult
    search_endpoint1: Endpoint1Result
    search_endpoint2: Endpoint2ResultRecord
    recovered_feasible_count: int
    missed_feasible_count: int
    feasible_recall: float
    endpoint1_retained_proportion_gap: float
    packing_gap: int
    search_procedure_failed: bool
    search_procedure_error: str | None
    search_procedure_censored: bool
    scientific_data: bool = False
    production_eligible: bool = False
    teacher_seed_transfer: bool = False
    main_experiment_transfer: bool = False
    mechanism_count_claim: bool = False

    def __post_init__(self) -> None:
        if self.scientific_data or self.production_eligible:
            raise TractableCalibrationError(
                "tractable result must remain technical-only"
            )
        if self.teacher_seed_transfer or self.main_experiment_transfer:
            raise TractableCalibrationError(
                "tractable result must not use teacher/main transfer"
            )
        if self.mechanism_count_claim:
            raise TractableCalibrationError(
                "tractable calibration cannot claim real mechanism count"
            )
        if self.certificate.exactness_claim != "exact":
            raise TractableCalibrationError(
                "v1 tractable runner emits exact exhaustive certificates"
            )
        if not self.certificate.exhaustive:
            raise TractableCalibrationError(
                "exact tractable result must be exhaustive"
            )
        if (
            self.certificate.lower_bound != self.feasible_mask_count
            or self.certificate.upper_bound != self.feasible_mask_count
        ):
            raise TractableCalibrationError(
                "certificate bounds must equal feasible inventory cardinality"
            )
        if len(self.admissible_mask_identities) != self.admissible_mask_count:
            raise TractableCalibrationError(
                "admissible identity count mismatch"
            )
        if len(set(self.admissible_mask_identities)) != (
            len(self.admissible_mask_identities)
        ):
            raise TractableCalibrationError(
                "admissible mask identities must be unique"
            )
        if len(self.feasible_mask_identities) != self.feasible_mask_count:
            raise TractableCalibrationError(
                "feasible identity count mismatch"
            )
        if not set(self.feasible_mask_identities).issubset(
            self.admissible_mask_identities
        ):
            raise TractableCalibrationError(
                "feasible inventory is not a subset of admissible universe"
            )

        expected_universe_hash = canonical_sha256(
            {
                "admissible_mask_identities": list(
                    self.admissible_mask_identities
                )
            }
        )
        if self.admissible_universe_hash != expected_universe_hash:
            raise TractableCalibrationError(
                "admissible universe hash mismatch"
            )

        expected_feasible_hash = canonical_sha256(
            {
                "feasible_mask_identities": list(
                    self.feasible_mask_identities
                )
            }
        )
        if self.feasible_inventory_hash != expected_feasible_hash:
            raise TractableCalibrationError(
                "feasible inventory hash mismatch"
            )

        if self.certified_packing_optimum != (
            self.exhaustive_endpoint2.packing_lower_bound
        ):
            raise TractableCalibrationError(
                "certified packing optimum disagrees with exhaustive Stage 6E"
            )
        if self.recovered_feasible_count + self.missed_feasible_count != (
            self.feasible_mask_count
        ):
            raise TractableCalibrationError(
                "recovered/missed feasible accounting mismatch"
            )
        expected_recall = (
            self.recovered_feasible_count / self.feasible_mask_count
        )
        if self.feasible_recall != expected_recall:
            raise TractableCalibrationError("feasible recall mismatch")
        if not 0.0 <= self.search_exact_evaluation_coverage <= 1.0:
            raise TractableCalibrationError(
                "exact-evaluation coverage outside [0, 1]"
            )
        if self.search_exact_budget_charged != (
            self.search_exact_evaluation_count
        ):
            raise TractableCalibrationError(
                "successful exact charges must equal exact evaluations"
            )
        if self.endpoint1_retained_proportion_gap < 0.0:
            raise TractableCalibrationError(
                "search Endpoint 1 cannot beat exhaustive feasible optimum"
            )
        if self.packing_gap < 0:
            raise TractableCalibrationError(
                "search packing cannot exceed exhaustive feasible optimum"
            )

        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))


SearchProcedure = Callable[[TractableSearchContext], TractableSearchOutput]


@dataclass(frozen=True)
class _EvidenceSpec:
    mask: tuple[int, ...]
    entry: ExactEvaluationEntry
    proposal_reference: str


def _mask_from_pattern(
    profile: TractableFixtureProfile,
    pattern: int,
) -> tuple[int, ...]:
    values = [0] * COMPONENT_COUNT
    for bit_index, component_index in enumerate(profile.free_component_indices):
        if pattern & (1 << bit_index):
            values[component_index] = 1
    return tuple(values)


def _mask_identity(mask: Sequence[int]) -> str:
    retained = tuple(index for index, value in enumerate(mask) if value)
    return canonical_mask_identity(retained)


def _admissible_pattern(
    profile: TractableFixtureProfile,
    raw_mask: Sequence[int],
) -> tuple[tuple[int, ...], int, str]:
    mask = tuple(raw_mask)
    if len(mask) != COMPONENT_COUNT:
        raise TractableCalibrationError(
            "search proposal must use common 516-component mask"
        )
    if any(type(value) is not int or value not in (0, 1) for value in mask):
        raise TractableCalibrationError(
            "search proposal mask must contain integer 0/1 values"
        )

    free = set(profile.free_component_indices)
    if any(value and index not in free for index, value in enumerate(mask)):
        raise TractableCalibrationError(
            "search proposal lies outside declared tractable universe"
        )

    pattern = 0
    for bit_index, component_index in enumerate(profile.free_component_indices):
        if mask[component_index]:
            pattern |= 1 << bit_index

    return mask, pattern, _mask_identity(mask)


def _fixture_fidelity(
    profile: TractableFixtureProfile,
    mask: tuple[int, ...],
) -> float:
    if mask == (1,) * COMPONENT_COUNT:
        return 1.0

    _, pattern, _ = _admissible_pattern(profile, mask)
    return 1.0 if pattern in profile.qualifying_free_patterns else 0.0


def _ledger_hash(
    *,
    label: str,
    builder: ExactLedgerBuilder,
) -> str:
    return canonical_sha256(
        {
            "ledger_hash_contract": "stage12r3-tractable-ledger/v1",
            "label": label,
            "evaluations": [asdict(item) for item in builder.evaluations],
            "proposals": [asdict(item) for item in builder.proposals],
            "sealed": True,
        }
    )


def _evidence(
    *,
    profile: TractableFixtureProfile,
    policy: TechnicalEndpoint2Policy,
    specs: tuple[_EvidenceSpec, ...],
    ledger_reference: str,
    ledger_hash: str,
    prefix: str,
) -> tuple[ExactCandidateEvidence, ...]:
    return tuple(
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
                f"{prefix}:exact:{spec.entry.evaluation_order}:"
                f"{spec.entry.mask_identity}"
            ),
            source_ledger_reference=ledger_reference,
            source_ledger_hash=ledger_hash,
            recomputed_ledger_hash=ledger_hash,
        )
        for spec in specs
    )


def _validate_policy(
    profile: TractableFixtureProfile,
    policy: TechnicalEndpoint2Policy,
) -> None:
    if policy.scientific_data or policy.production_default:
        raise TractableCalibrationError(
            "Part F requires technical non-production Stage 6E policy"
        )
    if policy.component_basis_size != COMPONENT_COUNT:
        raise TractableCalibrationError(
            "Part F requires common 516-component basis"
        )
    if policy.component_basis_reference != profile.component_basis_reference:
        raise TractableCalibrationError(
            "Stage 6E basis does not match tractable profile"
        )
    if policy.fidelity_threshold != profile.fidelity_threshold:
        raise TractableCalibrationError(
            "Stage 6A/6E fidelity thresholds must match"
        )


def run_tractable_calibration(
    *,
    profile: TractableFixtureProfile,
    policy: TechnicalEndpoint2Policy,
    search_procedure: SearchProcedure,
) -> TractableCalibrationResult:
    """Compare injected search against the exact declared feasible region."""

    _validate_policy(profile, policy)

    universe_masks = tuple(
        _mask_from_pattern(profile, pattern)
        for pattern in range(profile.admissible_mask_count)
    )
    universe_identities = tuple(
        sorted(_mask_identity(mask) for mask in universe_masks)
    )

    exhaustive_builder = ExactLedgerBuilder(
        evaluator=lambda mask: _fixture_fidelity(profile, mask),
        fidelity_threshold=profile.fidelity_threshold,
    )
    exhaustive_builder.add_mask(
        (1,) * COMPONENT_COUNT,
        proposal_index=0,
        exact_budget_charge=1,
    )

    exhaustive_specs: list[_EvidenceSpec] = []
    for index, mask in enumerate(universe_masks, start=1):
        entry = exhaustive_builder.add_mask(
            mask,
            proposal_index=index,
            exact_budget_charge=1,
        )
        exhaustive_specs.append(
            _EvidenceSpec(
                mask=mask,
                entry=entry,
                proposal_reference=f"tractable-exhaustive:pattern:{index - 1}",
            )
        )

    exhaustive_entries = exhaustive_builder.seal()
    validate_within_allowance(
        ExactBudgetUsage(
            evaluation_count=len(exhaustive_entries),
            charged_count=len(exhaustive_entries),
        ),
        TechnicalBudgetPolicy(len(exhaustive_entries)),
    )

    exhaustive_endpoint1 = reduce_endpoint1(
        exhaustive_entries,
        termination=TerminationStatus(
            status="completed",
            procedure_censored=False,
        ),
    )

    exhaustive_hash = _ledger_hash(
        label="exhaustive",
        builder=exhaustive_builder,
    )
    exhaustive_evidence = _evidence(
        profile=profile,
        policy=policy,
        specs=tuple(exhaustive_specs),
        ledger_reference=f"stage12r3-tractable-exhaustive:{profile.run_id}",
        ledger_hash=exhaustive_hash,
        prefix="stage12r3-tractable-exhaustive",
    )
    exhaustive_qualification = qualify_and_deduplicate(
        exhaustive_evidence,
        policy,
        model_id=profile.model_id,
        discovery_method_id=profile.discovery_method_id,
        discovery_config_id=profile.discovery_config_id,
    )
    exhaustive_endpoint2 = recompute_endpoint2(
        exhaustive_qualification,
        policy,
    )

    feasible_identities = tuple(
        sorted(
            candidate.mask_identity
            for candidate in exhaustive_qualification.qualified_candidates
        )
    )
    feasible_count = len(feasible_identities)
    if feasible_count == 0:
        raise TractableCalibrationError(
            "declared tractable fixture produced no Stage 6E-feasible masks"
        )

    certificate = CompletenessCertificate(
        exactness_claim="exact",
        exhaustive=True,
        lower_bound=feasible_count,
        upper_bound=feasible_count,
        gap=0,
        certificate_reference=(
            f"stage12r3-tractable-exhaustive:{profile.identity}"
        ),
    )

    context = TractableSearchContext(
        fixture_identity=profile.identity,
        run_id=profile.run_id,
        component_basis_reference=profile.component_basis_reference,
        component_basis_size=COMPONENT_COUNT,
        free_component_indices=profile.free_component_indices,
        admissible_mask_count=profile.admissible_mask_count,
        exact_evaluation_allowance=profile.search_exact_evaluation_allowance,
    )

    search_failed = False
    search_error: str | None = None
    try:
        search_output = search_procedure(context)
        if not isinstance(search_output, TractableSearchOutput):
            raise TractableCalibrationError(
                "search procedure must return TractableSearchOutput"
            )
    except Exception as exc:
        search_failed = True
        search_error = f"{type(exc).__name__}: {exc}"
        search_output = TractableSearchOutput(proposals=())

    search_builder = ExactLedgerBuilder(
        evaluator=lambda mask: _fixture_fidelity(profile, mask),
        fidelity_threshold=profile.fidelity_threshold,
    )
    search_builder.add_mask(
        (1,) * COMPONENT_COUNT,
        proposal_index=0,
        exact_budget_charge=1,
    )

    charged = 1
    failures = 0
    censored = 0
    valid_count = 0
    invalid_count = 0
    duplicate_count = 0
    successful_identities = {
        entry.mask_identity for entry in search_builder.evaluations
    }
    seen_valid_proposals: set[str] = set()
    evaluated_admissible_identities: set[str] = set()

    search_records: list[SearchProposalRecord] = []
    search_specs: list[_EvidenceSpec] = []

    for proposal_index, raw_mask in enumerate(search_output.proposals):
        proposal_reference = (
            f"stage12r3:{profile.run_id}:search-proposal:{proposal_index}"
        )

        try:
            mask, _, mask_identity = _admissible_pattern(profile, raw_mask)
        except Exception as exc:
            invalid_count += 1
            search_records.append(
                SearchProposalRecord(
                    proposal_index=proposal_index,
                    proposal_reference=proposal_reference,
                    mask_identity=None,
                    status="invalid",
                    exact_evaluation_order=None,
                    exact_fidelity=None,
                    qualifies=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        valid_count += 1
        if mask_identity in seen_valid_proposals:
            duplicate_count += 1
        seen_valid_proposals.add(mask_identity)

        duplicate_success = mask_identity in successful_identities
        if (
            not duplicate_success
            and charged >= profile.search_exact_evaluation_allowance
        ):
            censored += 1
            search_records.append(
                SearchProposalRecord(
                    proposal_index=proposal_index,
                    proposal_reference=proposal_reference,
                    mask_identity=mask_identity,
                    status="exact_budget_censored",
                    exact_evaluation_order=None,
                    exact_fidelity=None,
                    qualifies=None,
                    error="search exact-evaluation allowance exhausted",
                )
            )
            continue

        try:
            entry = search_builder.add_mask(
                mask,
                proposal_index=proposal_index + 1,
                exact_budget_charge=1,
            )
        except Exception as exc:
            failures += 1
            search_records.append(
                SearchProposalRecord(
                    proposal_index=proposal_index,
                    proposal_reference=proposal_reference,
                    mask_identity=mask_identity,
                    status="evaluation_failed",
                    exact_evaluation_order=None,
                    exact_fidelity=None,
                    qualifies=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if duplicate_success:
            status = "duplicate_evaluation_reused"
        else:
            status = "evaluated"
            charged += 1
            successful_identities.add(entry.mask_identity)

        evaluated_admissible_identities.add(entry.mask_identity)
        search_records.append(
            SearchProposalRecord(
                proposal_index=proposal_index,
                proposal_reference=proposal_reference,
                mask_identity=entry.mask_identity,
                status=status,
                exact_evaluation_order=entry.evaluation_order,
                exact_fidelity=entry.fidelity,
                qualifies=entry.qualifies,
                error=None,
            )
        )
        search_specs.append(
            _EvidenceSpec(
                mask=mask,
                entry=entry,
                proposal_reference=proposal_reference,
            )
        )

    search_entries = search_builder.seal()
    validate_within_allowance(
        ExactBudgetUsage(
            evaluation_count=len(search_entries),
            charged_count=charged,
        ),
        TechnicalBudgetPolicy(profile.search_exact_evaluation_allowance),
    )

    search_censored = bool(search_failed or failures or censored)
    search_endpoint1 = reduce_endpoint1(
        search_entries,
        termination=TerminationStatus(
            status=(
                "failed"
                if search_failed
                else "censored"
                if search_censored
                else "completed"
            ),
            procedure_censored=search_censored,
        ),
    )

    search_hash = _ledger_hash(
        label="search",
        builder=search_builder,
    )
    search_evidence = _evidence(
        profile=profile,
        policy=policy,
        specs=tuple(search_specs),
        ledger_reference=f"stage12r3-tractable-search:{profile.run_id}",
        ledger_hash=search_hash,
        prefix="stage12r3-tractable-search",
    )
    search_qualification = qualify_and_deduplicate(
        search_evidence,
        policy,
        model_id=profile.model_id,
        discovery_method_id=profile.discovery_method_id,
        discovery_config_id=profile.discovery_config_id,
    )
    search_endpoint2 = recompute_endpoint2(search_qualification, policy)

    recovered = set(feasible_identities) & {
        candidate.mask_identity
        for candidate in search_qualification.qualified_candidates
    }
    recovered_count = len(recovered)
    missed_count = feasible_count - recovered_count

    endpoint1_gap = (
        search_endpoint1.retained_proportion
        - exhaustive_endpoint1.retained_proportion
    )
    if endpoint1_gap < 0 and abs(endpoint1_gap) < 1e-15:
        endpoint1_gap = 0.0

    packing_gap = (
        exhaustive_endpoint2.packing_lower_bound
        - search_endpoint2.packing_lower_bound
    )

    return TractableCalibrationResult(
        profile_identity=profile.identity,
        certificate=certificate,
        admissible_mask_count=profile.admissible_mask_count,
        admissible_mask_identities=universe_identities,
        admissible_universe_hash=canonical_sha256(
            {
                "admissible_mask_identities": list(universe_identities)
            }
        ),
        feasible_mask_count=feasible_count,
        feasible_mask_identities=feasible_identities,
        feasible_inventory_hash=canonical_sha256(
            {
                "feasible_mask_identities": list(feasible_identities)
            }
        ),
        exhaustive_exact_evaluation_count=len(exhaustive_entries),
        exhaustive_endpoint1=exhaustive_endpoint1,
        exhaustive_qualification=exhaustive_qualification,
        exhaustive_endpoint2=exhaustive_endpoint2,
        certified_packing_optimum=(
            exhaustive_endpoint2.packing_lower_bound
        ),
        search_proposals=tuple(search_records),
        search_raw_proposal_count=len(search_output.proposals),
        search_valid_proposal_count=valid_count,
        search_invalid_proposal_count=invalid_count,
        search_duplicate_proposal_count=duplicate_count,
        search_exact_censored_count=censored,
        search_exact_failure_count=failures,
        search_exact_evaluation_allowance=(
            profile.search_exact_evaluation_allowance
        ),
        search_exact_evaluation_count=len(search_entries),
        search_exact_budget_charged=charged,
        search_exact_evaluation_coverage=(
            len(evaluated_admissible_identities)
            / profile.admissible_mask_count
        ),
        search_qualification=search_qualification,
        search_endpoint1=search_endpoint1,
        search_endpoint2=search_endpoint2,
        recovered_feasible_count=recovered_count,
        missed_feasible_count=missed_count,
        feasible_recall=recovered_count / feasible_count,
        endpoint1_retained_proportion_gap=endpoint1_gap,
        packing_gap=packing_gap,
        search_procedure_failed=search_failed,
        search_procedure_error=search_error,
        search_procedure_censored=search_censored,
    )
