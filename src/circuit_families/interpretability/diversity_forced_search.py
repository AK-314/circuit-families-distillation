"""Pure ranking utilities for Stage 12 diversity-forced search."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
    MaskEvaluationMetrics,
    compute_full_model_reference,
    evaluate_component_mask,
)
from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_HOOK_NAME,
    MLP_NEURON_HOOK_NAME,
    ComponentMask,
)
from circuit_families.interpretability.overlap_constraints import (
    is_structurally_distinct,
    maximum_pairwise_jaccard,
    pairwise_jaccard_overlaps,
    validate_jaccard_cutoff,
)
from circuit_families.interpretability.sparse_search import (
    ComponentRanking,
    ExactEvaluationFunction,
    RankingFunction,
    RankingResult,
    SparseSearchResult,
    greedy_sparse_search,
    rank_retained_components,
)
from circuit_families.seeds import numpy_generator, validate_seed
from circuit_families.training import canonical_state_hash

PRIMARY_REUSE_COEFFICIENT = 0.5
MAX_RESTARTS_PER_ALTERNATIVE = 5
NUMERICALLY_INDISTINGUISHABLE_TOLERANCE = 1.0e-12

DIVERSITY_SCORE_DEFINITION = (
    "damage percentile minus reuse coefficient times prior-family reuse rate"
)


@dataclass(frozen=True)
class DerivedSearchSeed:
    """Complete deterministic seed record for one restart."""

    model_seed: int
    checkpoint_index: int
    family_member_index: int
    restart_index: int
    canonical_material: str
    sha256_digest: str
    integer_seed: int
    bit_generator: str = "numpy.random.PCG64"


@dataclass(frozen=True)
class DiversityRankingEntry:
    """Auditable Stage 12 score for one retained component."""

    component_identifier: str
    component_index: int
    component_class: str
    gate_gradient: float
    raw_estimated_removal_damage: float
    damage_percentile: float
    reuse_rate: float
    removal_score: float
    candidate_ordering_score: float
    ranking_position: int


@dataclass(frozen=True)
class DiversityRankingResult:
    """Stage 9-compatible ranking plus raw Stage 12 diagnostics."""

    ranking_result: RankingResult
    entries: tuple[DiversityRankingEntry, ...]


def _validate_positive_index(
    value: int,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def _validate_restart_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("restart_index must be an integer.")

    if not 0 <= value < MAX_RESTARTS_PER_ALTERNATIVE:
        raise ValueError(
            "restart_index must lie between zero and four."
        )

    return value


def _validate_tolerance(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError("tie_tolerance must be numeric.")

    tolerance = float(value)

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "tie_tolerance must be finite and positive."
        )

    return tolerance


def derive_search_seed(
    *,
    model_seed: int,
    checkpoint_index: int,
    family_member_index: int,
    restart_index: int,
) -> DerivedSearchSeed:
    """Derive the committed 32-bit PCG64 restart seed.

    Checkpoint and family-member indices are one-based. Restart indices
    are zero-based.
    """

    model_seed = validate_seed(model_seed)
    checkpoint_index = _validate_positive_index(
        checkpoint_index,
        "checkpoint_index",
    )
    family_member_index = _validate_positive_index(
        family_member_index,
        "family_member_index",
    )
    restart_index = _validate_restart_index(restart_index)

    if family_member_index == 1:
        raise ValueError(
            "C1 has no diversity restart seed."
        )

    material = (
        "circuit-families|stage12-diversity-search|"
        f"model_seed={model_seed}|"
        f"checkpoint_index={checkpoint_index}|"
        f"family_member_index={family_member_index}|"
        f"restart_index={restart_index}"
    )

    digest = hashlib.sha256(
        material.encode("utf-8")
    ).digest()

    return DerivedSearchSeed(
        model_seed=model_seed,
        checkpoint_index=checkpoint_index,
        family_member_index=family_member_index,
        restart_index=restart_index,
        canonical_material=material,
        sha256_digest=digest.hex(),
        integer_seed=int.from_bytes(
            digest[:4],
            byteorder="big",
        ),
    )


def _tolerance_groups(
    entries: Sequence[DiversityRankingEntry],
    *,
    tolerance: float,
) -> tuple[tuple[DiversityRankingEntry, ...], ...]:
    """Group scores by distance from a fixed group anchor."""

    ordered = sorted(
        entries,
        key=lambda value: (
            value.removal_score,
            value.component_index,
        ),
    )

    groups: list[list[DiversityRankingEntry]] = []
    anchors: list[float] = []

    for entry in ordered:
        if (
            not groups
            or entry.removal_score - anchors[-1] > tolerance
        ):
            groups.append([entry])
            anchors.append(entry.removal_score)
        else:
            groups[-1].append(entry)

    return tuple(tuple(group) for group in groups)


def _apply_restart_ordering(
    entries: Sequence[DiversityRankingEntry],
    *,
    restart_index: int,
    seed_record: DerivedSearchSeed | None,
    tie_tolerance: float,
) -> tuple[DiversityRankingEntry, ...]:
    """Perturb ordering only inside committed tolerance groups."""

    restart_index = _validate_restart_index(restart_index)
    tolerance = _validate_tolerance(tie_tolerance)

    if restart_index == 0:
        ordered = sorted(
            entries,
            key=lambda value: (
                value.removal_score,
                value.component_index,
            ),
        )

        return tuple(
            replace(
                value,
                candidate_ordering_score=value.removal_score,
                ranking_position=position,
            )
            for position, value in enumerate(
                ordered,
                start=1,
            )
        )

    if not isinstance(seed_record, DerivedSearchSeed):
        raise TypeError(
            "Later restarts require a DerivedSearchSeed."
        )

    if seed_record.restart_index != restart_index:
        raise ValueError(
            "seed restart index does not match the requested restart."
        )

    groups = _tolerance_groups(
        entries,
        tolerance=tolerance,
    )
    generator = numpy_generator(seed_record.integer_seed)
    output: list[DiversityRankingEntry] = []

    for group_index, group in enumerate(groups):
        stable = sorted(
            group,
            key=lambda value: value.component_index,
        )

        if len(stable) == 1:
            output.append(
                replace(
                    stable[0],
                    candidate_ordering_score=(
                        stable[0].removal_score
                    ),
                )
            )
            continue

        permutation = generator.permutation(len(stable))
        reordered = [
            stable[int(index)]
            for index in permutation
        ]

        anchor = min(
            value.removal_score
            for value in stable
        )

        if group_index + 1 < len(groups):
            next_anchor = min(
                value.removal_score
                for value in groups[group_index + 1]
            )
            span = min(
                tolerance / 2.0,
                (next_anchor - anchor) / 2.0,
            )
        else:
            span = tolerance / 2.0

        step = span / (len(reordered) + 1)

        for position, value in enumerate(reordered):
            output.append(
                replace(
                    value,
                    candidate_ordering_score=(
                        anchor + position * step
                    ),
                )
            )

    ordered_output = sorted(
        output,
        key=lambda value: (
            value.candidate_ordering_score,
            value.component_index,
        ),
    )

    ordering_scores = tuple(
        value.candidate_ordering_score
        for value in ordered_output
    )

    if any(
        later < earlier
        for earlier, later in zip(
            ordering_scores,
            ordering_scores[1:],
            strict=False,
        )
    ):
        raise RuntimeError(
            "Restart perturbation produced an invalid ordering."
        )

    return tuple(
        replace(
            value,
            ranking_position=position,
        )
        for position, value in enumerate(
            ordered_output,
            start=1,
        )
    )


def _validate_rankings(
    rankings: Sequence[ComponentRanking],
) -> tuple[ComponentRanking, ...]:
    values = tuple(rankings)

    if any(
        not isinstance(value, ComponentRanking)
        for value in values
    ):
        raise TypeError(
            "rankings must contain ComponentRanking values."
        )

    identifiers = tuple(
        value.component_identifier
        for value in values
    )
    indices = tuple(
        value.component_index
        for value in values
    )

    if len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "rankings contain duplicate component identifiers."
        )

    if len(indices) != len(set(indices)):
        raise ValueError(
            "rankings contain duplicate component indices."
        )

    for value in values:
        if not math.isfinite(value.estimated_removal_damage):
            raise ValueError(
                "estimated removal damages must be finite."
            )

    return values


def stable_damage_percentiles(
    rankings: Sequence[ComponentRanking],
) -> dict[str, float]:
    """Return stable ordinal percentiles over retained components.

    For n > 1:

        percentile = zero-based ordinal position / (n - 1)

    A singleton receives zero. Equal damage values are ordered using the
    lower stable component index.
    """

    values = _validate_rankings(rankings)

    ordered = sorted(
        values,
        key=lambda value: (
            value.estimated_removal_damage,
            value.component_index,
        ),
    )

    if not ordered:
        return {}

    if len(ordered) == 1:
        return {
            ordered[0].component_identifier: 0.0,
        }

    denominator = len(ordered) - 1

    return {
        value.component_identifier: position / denominator
        for position, value in enumerate(ordered)
    }


def component_reuse_rates(
    retained_component_ids: Sequence[str],
    accepted_family: Sequence[ComponentMask],
) -> dict[str, float]:
    """Return prior-family reuse rates for retained components."""

    identifiers = tuple(retained_component_ids)

    if any(
        not isinstance(identifier, str)
        or not identifier
        for identifier in identifiers
    ):
        raise TypeError(
            "retained component identifiers must be non-empty strings."
        )

    if len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "retained component identifiers must be unique."
        )

    family = tuple(accepted_family)

    if any(
        not isinstance(mask, ComponentMask)
        for mask in family
    ):
        raise TypeError(
            "accepted_family must contain ComponentMask values."
        )

    if not family:
        return {
            identifier: 0.0
            for identifier in identifiers
        }

    retained_sets = tuple(
        set(mask.retained_component_ids)
        for mask in family
    )
    denominator = len(retained_sets)

    return {
        identifier: (
            sum(
                identifier in retained
                for retained in retained_sets
            )
            / denominator
        )
        for identifier in identifiers
    }


def diversity_removal_score(
    *,
    damage_percentile: float,
    reuse_rate: float,
    reuse_coefficient: float = PRIMARY_REUSE_COEFFICIENT,
) -> float:
    """Return the frozen Stage 12 removal score."""

    values = {
        "damage_percentile": damage_percentile,
        "reuse_rate": reuse_rate,
        "reuse_coefficient": reuse_coefficient,
    }

    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(f"{name} must be numeric.")

        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")

    percentile = float(damage_percentile)
    reuse = float(reuse_rate)
    coefficient = float(reuse_coefficient)

    if not 0.0 <= percentile <= 1.0:
        raise ValueError(
            "damage_percentile must lie between zero and one."
        )

    if not 0.0 <= reuse <= 1.0:
        raise ValueError(
            "reuse_rate must lie between zero and one."
        )

    if coefficient < 0.0:
        raise ValueError(
            "reuse_coefficient must be non-negative."
        )

    return percentile - coefficient * reuse


def build_diversity_ranking(
    base_result: RankingResult,
    accepted_family: Sequence[ComponentMask],
    *,
    reuse_coefficient: float = PRIMARY_REUSE_COEFFICIENT,
    restart_index: int = 0,
    seed_record: DerivedSearchSeed | None = None,
    tie_tolerance: float = (
        NUMERICALLY_INDISTINGUISHABLE_TOLERANCE
    ),
) -> DiversityRankingResult:
    """Build a Stage 9-compatible diversity-forced ranking.

    C1 is returned unchanged because no reuse penalty applies.

    For alternatives, the transformed RankingResult stores the active
    diversity removal score in ComponentRanking.estimated_removal_damage,
    which allows the unchanged Stage 9 greedy engine to use it for ordering.
    Raw Stage 9 damage remains available in the accompanying entries.
    """

    if not isinstance(base_result, RankingResult):
        raise TypeError(
            "base_result must be a RankingResult."
        )

    family = tuple(accepted_family)

    if any(
        not isinstance(mask, ComponentMask)
        for mask in family
    ):
        raise TypeError(
            "accepted_family must contain ComponentMask values."
        )

    restart_index = _validate_restart_index(
        restart_index
    )
    tie_tolerance = _validate_tolerance(
        tie_tolerance
    )

    if not family:
        if restart_index != 0 or seed_record is not None:
            raise ValueError(
                "C1 cannot use diversity restarts."
            )

        return DiversityRankingResult(
            ranking_result=base_result,
            entries=(),
        )

    if restart_index > 0:
        if not isinstance(seed_record, DerivedSearchSeed):
            raise TypeError(
                "Later restarts require a DerivedSearchSeed."
            )

        if seed_record.restart_index != restart_index:
            raise ValueError(
                "seed restart index does not match."
            )

    rankings = _validate_rankings(
        base_result.ranked_components
    )

    percentiles = stable_damage_percentiles(rankings)
    reuse_rates = component_reuse_rates(
        tuple(
            value.component_identifier
            for value in rankings
        ),
        family,
    )

    provisional: list[DiversityRankingEntry] = []

    for value in rankings:
        percentile = percentiles[
            value.component_identifier
        ]
        reuse_rate = reuse_rates[
            value.component_identifier
        ]
        score = diversity_removal_score(
            damage_percentile=percentile,
            reuse_rate=reuse_rate,
            reuse_coefficient=reuse_coefficient,
        )

        provisional.append(
            DiversityRankingEntry(
                component_identifier=(
                    value.component_identifier
                ),
                component_index=value.component_index,
                component_class=value.component_class,
                gate_gradient=value.gate_gradient,
                raw_estimated_removal_damage=(
                    value.estimated_removal_damage
                ),
                damage_percentile=percentile,
                reuse_rate=reuse_rate,
                removal_score=score,
                candidate_ordering_score=score,
                ranking_position=0,
            )
        )

    entries = _apply_restart_ordering(
        provisional,
        restart_index=restart_index,
        seed_record=seed_record,
        tie_tolerance=tie_tolerance,
    )

    transformed_rankings = tuple(
        ComponentRanking(
            component_identifier=(
                value.component_identifier
            ),
            component_index=value.component_index,
            component_class=value.component_class,
            gate_gradient=value.gate_gradient,
            estimated_removal_damage=(
                value.candidate_ordering_score
            ),
            ranking_position=value.ranking_position,
        )
        for value in entries
    )

    transformed_result = replace(
        base_result,
        ranked_components=transformed_rankings,
        score_definition=DIVERSITY_SCORE_DEFINITION,
    )

    return DiversityRankingResult(
        ranking_result=transformed_result,
        entries=entries,
    )



@dataclass(frozen=True)
class DiversitySearchExecution:
    """One Stage 12 search through the unchanged Stage 9 engine."""

    result: SparseSearchResult
    ranking_results: tuple[DiversityRankingResult, ...]
    restart_index: int
    seed_record: DerivedSearchSeed | None


def run_diversity_sparse_search(
    *,
    base_ranking_function: RankingFunction,
    exact_evaluation_function: ExactEvaluationFunction,
    initial_metrics: MaskEvaluationMetrics,
    accepted_family: Sequence[ComponentMask],
    fidelity_threshold: float,
    exact_evaluation_budget: int,
    reuse_coefficient: float = PRIMARY_REUSE_COEFFICIENT,
    restart_index: int = 0,
    seed_record: DerivedSearchSeed | None = None,
    tie_tolerance: float = (
        NUMERICALLY_INDISTINGUISHABLE_TOLERANCE
    ),
) -> DiversitySearchExecution:
    """Run diversity ordering through the existing greedy search.

    The wrapper changes candidate ordering only. Exact candidate
    evaluation, within-batch selection, budget accounting and terminal
    deletion checks remain delegated to greedy_sparse_search.
    """

    if not callable(base_ranking_function):
        raise TypeError(
            "base_ranking_function must be callable."
        )

    if not callable(exact_evaluation_function):
        raise TypeError(
            "exact_evaluation_function must be callable."
        )

    family = tuple(accepted_family)

    if any(
        not isinstance(mask, ComponentMask)
        for mask in family
    ):
        raise TypeError(
            "accepted_family must contain ComponentMask values."
        )

    restart_index = _validate_restart_index(
        restart_index
    )
    tie_tolerance = _validate_tolerance(
        tie_tolerance
    )

    if not family:
        if restart_index != 0 or seed_record is not None:
            raise ValueError(
                "C1 cannot use diversity restarts."
            )
    elif restart_index > 0:
        if not isinstance(seed_record, DerivedSearchSeed):
            raise TypeError(
                "Later restarts require a DerivedSearchSeed."
            )

        if seed_record.restart_index != restart_index:
            raise ValueError(
                "seed restart index does not match."
            )

    ranking_results: list[DiversityRankingResult] = []

    def ranking_function(mask: ComponentMask) -> RankingResult:
        base_result = base_ranking_function(mask)

        transformed = build_diversity_ranking(
            base_result,
            family,
            reuse_coefficient=reuse_coefficient,
            restart_index=restart_index,
            seed_record=seed_record,
            tie_tolerance=tie_tolerance,
        )

        ranking_results.append(transformed)
        return transformed.ranking_result

    result = greedy_sparse_search(
        ranking_function=ranking_function,
        exact_evaluation_function=(
            exact_evaluation_function
        ),
        initial_metrics=initial_metrics,
        fidelity_threshold=fidelity_threshold,
        exact_evaluation_budget=exact_evaluation_budget,
    )

    if result.ranking_passes_used != len(ranking_results):
        raise RuntimeError(
            "Recorded ranking passes do not match wrapper diagnostics."
        )

    return DiversitySearchExecution(
        result=result,
        ranking_results=tuple(ranking_results),
        restart_index=restart_index,
        seed_record=seed_record,
    )



FAMILY_TARGET = 10
PER_REQUESTED_CIRCUIT_EXACT_EVALUATION_BUDGET = 10_000
PER_CELL_EXACT_EVALUATION_BUDGET = 50_000

VALID_DISTINCT_CANDIDATE = "valid_distinct_candidate"
FIDELITY_FAILURE = "fidelity_failure"
SPARSITY_FAILURE = "sparsity_failure"
DISTINCTNESS_FAILURE = "distinctness_failure"
SEARCH_FAILURE = "search_failure"
BUDGET_EXHAUSTION = "budget_exhaustion"
INVALID_MASKING_OUTPUT = "invalid_masking_output"
NO_FEASIBLE_CANDIDATE = (
    "no_feasible_candidate_discovered_within_tested_search"
)
RIGHT_CENSORED = "right_censored_at_family_target"


@dataclass(frozen=True)
class FamilyRestartOutcome:
    """One requested-member restart and its final validation."""

    requested_member_index: int
    restart_index: int
    seed_record: DerivedSearchSeed | None
    execution: DiversitySearchExecution
    pairwise_overlaps: tuple[Fraction, ...]
    maximum_pairwise_overlap: Fraction
    outcome_status: str
    accepted_candidate: bool


@dataclass(frozen=True)
class RecoveredFamilyMember:
    """One selected valid and structurally distinct circuit."""

    member_index: int
    selected_restart_index: int
    mask: ComponentMask
    metrics: MaskEvaluationMetrics
    search_result: SparseSearchResult
    pairwise_overlaps: tuple[Fraction, ...]
    maximum_pairwise_overlap: Fraction


@dataclass(frozen=True)
class FamilySearchResult:
    """Complete fixed-budget sequential family-search result."""

    status: str
    fidelity_threshold: float
    distinctness_cutoff: Fraction
    family_target: int
    max_restarts_per_alternative: int
    per_requested_circuit_budget: int
    per_cell_budget: int
    members: tuple[RecoveredFamilyMember, ...]
    restart_outcomes: tuple[FamilyRestartOutcome, ...]
    exact_evaluations_used: int
    budget_remaining: int
    right_censored: bool
    stopping_reason: str

    @property
    def family_size(self) -> int:
        return len(self.members)


def _validate_positive_integer(
    value: int,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def _classify_family_candidate(
    result: SparseSearchResult,
    accepted_family: Sequence[ComponentMask],
    *,
    fidelity_threshold: float,
    distinctness_cutoff: Fraction,
) -> tuple[
    str,
    tuple[Fraction, ...],
    Fraction,
]:
    """Apply exact fidelity, sparsity and distinctness gates."""

    family = tuple(accepted_family)
    overlaps = pairwise_jaccard_overlaps(
        result.final_mask,
        family,
    )
    maximum_overlap = maximum_pairwise_jaccard(
        result.final_mask,
        family,
    )

    if result.status == "invalid_masking_output":
        return (
            INVALID_MASKING_OUTPUT,
            overlaps,
            maximum_overlap,
        )

    if result.status == "ranking_failure":
        return (
            SEARCH_FAILURE,
            overlaps,
            maximum_overlap,
        )

    if result.status == "budget_exhaustion":
        return (
            BUDGET_EXHAUSTION,
            overlaps,
            maximum_overlap,
        )

    if result.status == "fidelity_failure":
        return (
            FIDELITY_FAILURE,
            overlaps,
            maximum_overlap,
        )

    if result.status == (
        "no_feasible_sparse_candidate_discovered_within_budget"
    ):
        return (
            NO_FEASIBLE_CANDIDATE,
            overlaps,
            maximum_overlap,
        )

    if (
        result.status == "valid_but_not_meaningfully_sparse"
        or not result.meaningfully_sparse
    ):
        return (
            SPARSITY_FAILURE,
            overlaps,
            maximum_overlap,
        )

    if result.status != "valid_sparse_circuit":
        return (
            SEARCH_FAILURE,
            overlaps,
            maximum_overlap,
        )

    if result.final_metrics.primary_fidelity < fidelity_threshold:
        return (
            FIDELITY_FAILURE,
            overlaps,
            maximum_overlap,
        )

    if not is_structurally_distinct(
        result.final_mask,
        family,
        cutoff=distinctness_cutoff,
    ):
        return (
            DISTINCTNESS_FAILURE,
            overlaps,
            maximum_overlap,
        )

    return (
        VALID_DISTINCT_CANDIDATE,
        overlaps,
        maximum_overlap,
    )


def select_best_distinct_candidate(
    outcomes: Sequence[FamilyRestartOutcome],
) -> FamilyRestartOutcome | None:
    """Apply the frozen alternative-circuit selection rule."""

    valid = tuple(
        outcome
        for outcome in outcomes
        if outcome.outcome_status
        == VALID_DISTINCT_CANDIDATE
    )

    if not valid:
        return None

    return min(
        valid,
        key=lambda outcome: (
            outcome.execution.result.final_mask.retained_component_count,
            -outcome.execution.result.final_metrics.primary_fidelity,
            outcome.restart_index,
        ),
    )


def _recovered_member(
    outcome: FamilyRestartOutcome,
) -> RecoveredFamilyMember:
    if outcome.outcome_status != VALID_DISTINCT_CANDIDATE:
        raise ValueError(
            "Only a valid distinct candidate may become a member."
        )

    result = outcome.execution.result

    return RecoveredFamilyMember(
        member_index=outcome.requested_member_index,
        selected_restart_index=outcome.restart_index,
        mask=result.final_mask,
        metrics=result.final_metrics,
        search_result=result,
        pairwise_overlaps=outcome.pairwise_overlaps,
        maximum_pairwise_overlap=(
            outcome.maximum_pairwise_overlap
        ),
    )


def _failed_member_status(
    outcomes: Sequence[FamilyRestartOutcome],
    *,
    member_budget_exhausted: bool,
    cell_budget_exhausted: bool,
) -> str:
    statuses = {
        outcome.outcome_status
        for outcome in outcomes
    }

    if (
        member_budget_exhausted
        or cell_budget_exhausted
        or BUDGET_EXHAUSTION in statuses
    ):
        return BUDGET_EXHAUSTION

    priority = (
        INVALID_MASKING_OUTPUT,
        SEARCH_FAILURE,
        DISTINCTNESS_FAILURE,
        SPARSITY_FAILURE,
        FIDELITY_FAILURE,
        NO_FEASIBLE_CANDIDATE,
    )

    for status in priority:
        if status in statuses:
            return status

    return SEARCH_FAILURE


def run_sequential_family_search(
    *,
    base_ranking_function: RankingFunction,
    exact_evaluation_function: ExactEvaluationFunction,
    initial_metrics: MaskEvaluationMetrics,
    fidelity_threshold: float,
    distinctness_cutoff: Fraction | int | float | str,
    model_seed: int,
    checkpoint_index: int,
    family_target: int = FAMILY_TARGET,
    max_restarts_per_alternative: int = (
        MAX_RESTARTS_PER_ALTERNATIVE
    ),
    per_requested_circuit_budget: int = (
        PER_REQUESTED_CIRCUIT_EXACT_EVALUATION_BUDGET
    ),
    per_cell_budget: int = PER_CELL_EXACT_EVALUATION_BUDGET,
    reuse_coefficient: float = PRIMARY_REUSE_COEFFICIENT,
    tie_tolerance: float = (
        NUMERICALLY_INDISTINGUISHABLE_TOLERANCE
    ),
    member_started_callback: (
        Callable[[int], None] | None
    ) = None,
    member_finished_callback: (
        Callable[[int], None] | None
    ) = None,
) -> FamilySearchResult:
    """Recover a fixed-budget structurally distinct circuit family."""

    family_target = _validate_positive_integer(
        family_target,
        "family_target",
    )
    max_restarts_per_alternative = _validate_positive_integer(
        max_restarts_per_alternative,
        "max_restarts_per_alternative",
    )
    per_requested_circuit_budget = _validate_positive_integer(
        per_requested_circuit_budget,
        "per_requested_circuit_budget",
    )
    per_cell_budget = _validate_positive_integer(
        per_cell_budget,
        "per_cell_budget",
    )
    checkpoint_index = _validate_positive_index(
        checkpoint_index,
        "checkpoint_index",
    )

    if family_target > FAMILY_TARGET:
        raise ValueError(
            "family_target may not exceed ten."
        )

    if (
        max_restarts_per_alternative
        > MAX_RESTARTS_PER_ALTERNATIVE
    ):
        raise ValueError(
            "max_restarts_per_alternative may not exceed five."
        )

    if (
        per_requested_circuit_budget
        > PER_REQUESTED_CIRCUIT_EXACT_EVALUATION_BUDGET
    ):
        raise ValueError(
            "per-requested-circuit budget may not exceed 10,000."
        )

    if per_cell_budget > PER_CELL_EXACT_EVALUATION_BUDGET:
        raise ValueError(
            "per-cell budget may not exceed 50,000."
        )

    cutoff = validate_jaccard_cutoff(
        distinctness_cutoff
    )

    members: list[RecoveredFamilyMember] = []
    all_outcomes: list[FamilyRestartOutcome] = []
    total_evaluations = 0
    stopping_status: str | None = None
    stopping_reason: str | None = None

    for member_index in range(1, family_target + 1):
        remaining_cell_budget = (
            per_cell_budget - total_evaluations
        )

        if remaining_cell_budget <= 0:
            stopping_status = BUDGET_EXHAUSTION
            stopping_reason = (
                "cell_budget_exhausted_before_requested_member"
            )
            break

        if member_started_callback is not None:
            member_started_callback(member_index)

        accepted_masks = tuple(
            member.mask
            for member in members
        )

        if member_index == 1:
            restart_plan = (0,)
        else:
            restart_plan = tuple(
                range(max_restarts_per_alternative)
            )

        member_outcomes: list[FamilyRestartOutcome] = []
        member_evaluations = 0

        for restart_index in restart_plan:
            remaining_member_budget = (
                per_requested_circuit_budget
                - member_evaluations
            )
            remaining_cell_budget = (
                per_cell_budget - total_evaluations
            )
            available_budget = min(
                remaining_member_budget,
                remaining_cell_budget,
            )

            if available_budget <= 0:
                break

            seed_record = (
                None
                if member_index == 1
                else derive_search_seed(
                    model_seed=model_seed,
                    checkpoint_index=checkpoint_index,
                    family_member_index=member_index,
                    restart_index=restart_index,
                )
            )

            execution = run_diversity_sparse_search(
                base_ranking_function=base_ranking_function,
                exact_evaluation_function=(
                    exact_evaluation_function
                ),
                initial_metrics=initial_metrics,
                accepted_family=accepted_masks,
                fidelity_threshold=fidelity_threshold,
                exact_evaluation_budget=available_budget,
                reuse_coefficient=reuse_coefficient,
                restart_index=restart_index,
                seed_record=seed_record,
                tie_tolerance=tie_tolerance,
            )

            evaluations_used = (
                execution.result.exact_evaluations_used
            )

            if evaluations_used > available_budget:
                raise RuntimeError(
                    "A restart exceeded its available exact-evaluation "
                    "budget."
                )

            member_evaluations += evaluations_used
            total_evaluations += evaluations_used

            (
                outcome_status,
                overlaps,
                maximum_overlap,
            ) = _classify_family_candidate(
                execution.result,
                accepted_masks,
                fidelity_threshold=fidelity_threshold,
                distinctness_cutoff=cutoff,
            )

            outcome = FamilyRestartOutcome(
                requested_member_index=member_index,
                restart_index=restart_index,
                seed_record=seed_record,
                execution=execution,
                pairwise_overlaps=overlaps,
                maximum_pairwise_overlap=maximum_overlap,
                outcome_status=outcome_status,
                accepted_candidate=False,
            )

            member_outcomes.append(outcome)
            all_outcomes.append(outcome)

            if (
                outcome_status
                == VALID_DISTINCT_CANDIDATE
            ):
                break

        selected = select_best_distinct_candidate(
            member_outcomes
        )

        if selected is not None:
            selected = replace(
                selected,
                accepted_candidate=True,
            )

            member_outcomes[
                member_outcomes.index(
                    select_best_distinct_candidate(
                        member_outcomes
                    )
                )
            ] = selected

            for index in range(
                len(all_outcomes) - len(member_outcomes),
                len(all_outcomes),
            ):
                current = all_outcomes[index]

                if (
                    current.requested_member_index
                    == selected.requested_member_index
                    and current.restart_index
                    == selected.restart_index
                ):
                    all_outcomes[index] = selected
                    break

            members.append(
                _recovered_member(selected)
            )

            if member_finished_callback is not None:
                member_finished_callback(member_index)

            continue

        member_budget_exhausted = (
            member_evaluations
            >= per_requested_circuit_budget
        )
        cell_budget_exhausted = (
            total_evaluations >= per_cell_budget
        )

        stopping_status = _failed_member_status(
            member_outcomes,
            member_budget_exhausted=(
                member_budget_exhausted
            ),
            cell_budget_exhausted=(
                cell_budget_exhausted
            ),
        )
        stopping_reason = (
            f"requested_member_{member_index}_"
            f"{stopping_status}"
        )

        if member_finished_callback is not None:
            member_finished_callback(member_index)

        break

    if len(members) == family_target:
        status = RIGHT_CENSORED
        right_censored = True
        final_reason = "family_target_reached"
    else:
        status = stopping_status or SEARCH_FAILURE
        right_censored = False
        final_reason = (
            stopping_reason
            or "family_construction_stopped"
        )

    if total_evaluations > per_cell_budget:
        raise RuntimeError(
            "Family search exceeded the per-cell budget."
        )

    return FamilySearchResult(
        status=status,
        fidelity_threshold=float(fidelity_threshold),
        distinctness_cutoff=cutoff,
        family_target=family_target,
        max_restarts_per_alternative=(
            max_restarts_per_alternative
        ),
        per_requested_circuit_budget=(
            per_requested_circuit_budget
        ),
        per_cell_budget=per_cell_budget,
        members=tuple(members),
        restart_outcomes=tuple(all_outcomes),
        exact_evaluations_used=total_evaluations,
        budget_remaining=(
            per_cell_budget - total_evaluations
        ),
        right_censored=right_censored,
        stopping_reason=final_reason,
    )



@dataclass(frozen=True)
class CheckpointFamilySearchExecution:
    """Integrity evidence for one checkpoint-backed family search."""

    result: FamilySearchResult
    pseudo_target_sha256: str
    pseudo_target_count: int
    ranking_batch_size: int
    evaluation_batch_size: int
    full_model_reference_sha256: str
    full_model_reference_example_count: int
    full_model_reference_batch_size: int
    model_state_sha256_before: str
    model_state_sha256_after: str
    hook_counts_before: tuple[tuple[str, int], ...]
    hook_counts_after: tuple[tuple[str, int], ...]


def _checkpoint_hook_counts(
    context: CheckpointEvaluationContext,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            hook_name,
            len(
                context.model.hook_dict[
                    hook_name
                ]._forward_hooks
            ),
        )
        for hook_name in (
            ATTENTION_HEAD_HOOK_NAME,
            MLP_NEURON_HOOK_NAME,
        )
    )


def _validate_checkpoint_batch_size(
    value: int,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def run_checkpoint_family_search(
    context: CheckpointEvaluationContext,
    *,
    fidelity_threshold: float,
    distinctness_cutoff: Fraction | int | float | str,
    model_seed: int,
    checkpoint_index: int,
    ranking_batch_size: int,
    evaluation_batch_size: int,
    family_target: int = FAMILY_TARGET,
    max_restarts_per_alternative: int = (
        MAX_RESTARTS_PER_ALTERNATIVE
    ),
    per_requested_circuit_budget: int = (
        PER_REQUESTED_CIRCUIT_EXACT_EVALUATION_BUDGET
    ),
    per_cell_budget: int = PER_CELL_EXACT_EVALUATION_BUDGET,
    reuse_coefficient: float = PRIMARY_REUSE_COEFFICIENT,
    tie_tolerance: float = (
        NUMERICALLY_INDISTINGUISHABLE_TOLERANCE
    ),
    member_started_callback: (
        Callable[[int], None] | None
    ) = None,
    member_finished_callback: (
        Callable[[int], None] | None
    ) = None,
) -> CheckpointFamilySearchExecution:
    """Run Stage 12 through the validated Stage 8 evaluator.

    The full checkpoint reference is computed once. Its predictions are
    used as ranking pseudo-targets, and the same cached logits and
    predictions are supplied to every exact mask evaluation.
    """

    if not isinstance(context, CheckpointEvaluationContext):
        raise TypeError(
            "context must be a CheckpointEvaluationContext."
        )

    ranking_batch_size = _validate_checkpoint_batch_size(
        ranking_batch_size,
        "ranking_batch_size",
    )
    evaluation_batch_size = _validate_checkpoint_batch_size(
        evaluation_batch_size,
        "evaluation_batch_size",
    )

    model_state_before = canonical_state_hash(
        context.model.state_dict()
    )
    hook_counts_before = _checkpoint_hook_counts(context)

    if model_state_before != context.model_state_sha256:
        raise ValueError(
            "Checkpoint context model-state hash does not match "
            "the loaded model."
        )

    full_model_reference = compute_full_model_reference(
        context.model,
        context.inputs,
        context.targets,
        batch_size=evaluation_batch_size,
    )
    pseudo_targets = (
        full_model_reference.predictions.detach().clone()
    )

    pseudo_target_sha256 = canonical_state_hash(
        {"pseudo_targets": pseudo_targets}
    )
    full_model_reference_sha256 = canonical_state_hash(
        {
            "final_logits": full_model_reference.final_logits,
            "predictions": full_model_reference.predictions,
        }
    )

    initial_mask = ComponentMask.all_retained()
    initial_metrics = evaluate_component_mask(
        context.model,
        context.inputs,
        context.targets,
        initial_mask,
        batch_size=evaluation_batch_size,
        full_model_reference=full_model_reference,
    )

    def base_ranking_function(
        mask: ComponentMask,
    ) -> RankingResult:
        return rank_retained_components(
            context.model,
            context.inputs,
            pseudo_targets,
            mask,
            batch_size=ranking_batch_size,
        )

    def exact_evaluation_function(
        mask: ComponentMask,
    ) -> MaskEvaluationMetrics:
        return evaluate_component_mask(
            context.model,
            context.inputs,
            context.targets,
            mask,
            batch_size=evaluation_batch_size,
            full_model_reference=full_model_reference,
        )

    result = run_sequential_family_search(
        base_ranking_function=base_ranking_function,
        exact_evaluation_function=exact_evaluation_function,
        initial_metrics=initial_metrics,
        fidelity_threshold=fidelity_threshold,
        distinctness_cutoff=distinctness_cutoff,
        model_seed=model_seed,
        checkpoint_index=checkpoint_index,
        family_target=family_target,
        max_restarts_per_alternative=(
            max_restarts_per_alternative
        ),
        per_requested_circuit_budget=(
            per_requested_circuit_budget
        ),
        per_cell_budget=per_cell_budget,
        reuse_coefficient=reuse_coefficient,
        tie_tolerance=tie_tolerance,
        member_started_callback=(
            member_started_callback
        ),
        member_finished_callback=(
            member_finished_callback
        ),
    )

    model_state_after = canonical_state_hash(
        context.model.state_dict()
    )
    hook_counts_after = _checkpoint_hook_counts(context)

    if model_state_after != model_state_before:
        raise RuntimeError(
            "Checkpoint family search changed the model-state hash."
        )

    if hook_counts_after != hook_counts_before:
        raise RuntimeError(
            "Checkpoint family search leaked TransformerLens hooks."
        )

    if any(
        parameter.grad is not None
        for parameter in context.model.parameters()
    ):
        raise RuntimeError(
            "Checkpoint family search left parameter gradients populated."
        )

    return CheckpointFamilySearchExecution(
        result=result,
        pseudo_target_sha256=pseudo_target_sha256,
        pseudo_target_count=int(
            pseudo_targets.shape[0]
        ),
        ranking_batch_size=ranking_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        full_model_reference_sha256=(
            full_model_reference_sha256
        ),
        full_model_reference_example_count=(
            full_model_reference.evaluated_example_count
        ),
        full_model_reference_batch_size=(
            full_model_reference.inference_batch_size
        ),
        model_state_sha256_before=model_state_before,
        model_state_sha256_after=model_state_after,
        hook_counts_before=hook_counts_before,
        hook_counts_after=hook_counts_after,
    )
