"""Size/type-matched combinatorial packing calibration.

This layer asks only how much low-overlap multiplicity arises from finite
component-set combinatorics under a declared matching distribution.

Sampled masks are *not* fidelity-qualified circuits.  Stage 6E is reused only
for its retained-Jaccard compatibility graph and exact packing solver.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction

from circuit_families.stage6a.models import canonical_mask_identity
from circuit_families.stage6e.packing import (
    build_compatibility_graph,
    exact_maximum_compatible_subset,
)
from circuit_families.stage12r2.contracts import (
    canonical_sha256,
    validate_technical_record_payload,
)


class CombinatorialFloorError(ValueError):
    """Raised when a combinatorial-floor specification is invalid."""


@dataclass(frozen=True)
class CombinatorialPackingRule:
    """Structural policy consumed by Stage 6E graph/solver functions.

    This is not Stage 6E TechnicalEndpoint2Policy and makes no Endpoint 2 claim.
    """

    component_basis_reference: str
    component_basis_size: int
    max_pairwise_overlap: float
    overlap_rule_reference: str = "jaccard-retained-components/technical-v1"
    solver_reference: str = "exact-maximum-compatible-subset/technical-v1"
    tie_break_reference: str = (
        "lexicographically-smallest-mask-identity-tuple/v1"
    )
    scientific_data: bool = False
    production_eligible: bool = False
    endpoint2_policy: bool = False

    def __post_init__(self) -> None:
        if not self.component_basis_reference:
            raise CombinatorialFloorError(
                "packing basis reference must be non-empty"
            )
        if self.component_basis_size <= 0:
            raise CombinatorialFloorError(
                "packing basis size must be positive"
            )
        if not 0.0 <= self.max_pairwise_overlap <= 1.0:
            raise CombinatorialFloorError(
                "overlap cutoff must lie in [0, 1]"
            )
        if (
            self.overlap_rule_reference
            != "jaccard-retained-components/technical-v1"
        ):
            raise CombinatorialFloorError(
                "Part C requires Stage 6E retained-Jaccard semantics"
            )
        if (
            self.solver_reference
            != "exact-maximum-compatible-subset/technical-v1"
        ):
            raise CombinatorialFloorError(
                "Part C requires the Stage 6E exact packing solver"
            )
        if (
            self.tie_break_reference
            != "lexicographically-smallest-mask-identity-tuple/v1"
        ):
            raise CombinatorialFloorError(
                "Part C requires the Stage 6E deterministic tie-break"
            )
        if self.scientific_data or self.production_eligible:
            raise CombinatorialFloorError(
                "packing rule must remain technical-only"
            )
        if self.endpoint2_policy:
            raise CombinatorialFloorError(
                "combinatorial packing rule is not Endpoint 2 policy"
            )
        validate_technical_record_payload(asdict(self))

    @property
    def policy_hash(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class CombinatorialPackingCandidate:
    """Structural Stage 6E graph input with no fidelity semantics."""

    component_basis_reference: str
    component_basis_size: int
    mask_identity: str
    retained_components: tuple[int, ...]
    draw_references: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.component_basis_reference or not self.mask_identity:
            raise CombinatorialFloorError(
                "packing candidate identities must be non-empty"
            )
        if self.component_basis_size <= 0:
            raise CombinatorialFloorError(
                "packing candidate basis size must be positive"
            )
        if tuple(sorted(set(self.retained_components))) != self.retained_components:
            raise CombinatorialFloorError(
                "packing candidate mask must be sorted and unique"
            )
        if any(
            index < 0 or index >= self.component_basis_size
            for index in self.retained_components
        ):
            raise CombinatorialFloorError(
                "packing candidate index outside supplied basis"
            )
        if not self.draw_references:
            raise CombinatorialFloorError(
                "packing candidate requires draw provenance"
            )
        validate_technical_record_payload(asdict(self))

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SizeTypeMatchingRule:
    """Finite sampling rule.

    A draw first chooses one entry uniformly from ``retained_sizes`` (so
    repeated sizes encode explicit discrete weights), then samples uniformly
    from all masks satisfying that size and, when supplied, the exact
    component-type composition.
    """

    retained_sizes: tuple[int, ...]
    component_type_counts: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self) -> None:
        if not self.retained_sizes:
            raise CombinatorialFloorError(
                "retained_sizes must contain at least one declared size"
            )
        if any(size < 0 for size in self.retained_sizes):
            raise CombinatorialFloorError("retained sizes must be non-negative")

        if self.component_type_counts is not None:
            names = [name for name, _ in self.component_type_counts]
            if len(names) != len(set(names)):
                raise CombinatorialFloorError(
                    "component type-count names must be unique"
                )
            if any(not name for name in names):
                raise CombinatorialFloorError(
                    "component type names must be non-empty"
                )
            if any(count < 0 for _, count in self.component_type_counts):
                raise CombinatorialFloorError(
                    "component type counts must be non-negative"
                )

            total = sum(count for _, count in self.component_type_counts)
            if any(size != total for size in self.retained_sizes):
                raise CombinatorialFloorError(
                    "every declared retained size must equal the requested "
                    "component-type count total"
                )

    @property
    def distribution_semantics(self) -> str:
        if self.component_type_counts is None:
            return (
                "uniform_declared_size_entry_then_uniform_size_matched_mask"
            )
        return (
            "uniform_declared_size_entry_then_uniform_exact_type_matched_mask"
        )


@dataclass(frozen=True)
class CombinatorialFloorProfile:
    profile_id: str
    basis_hash: str
    ordered_component_ids: tuple[str, ...]
    component_types: tuple[str, ...]
    matching_rule: SizeTypeMatchingRule
    batch_count: int
    draws_per_batch: int
    root_seed: int
    seed_stream_id: str
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise CombinatorialFloorError("profile_id must be non-empty")
        if not self.basis_hash:
            raise CombinatorialFloorError("basis_hash must be non-empty")
        if not self.ordered_component_ids:
            raise CombinatorialFloorError(
                "component universe must be non-empty"
            )
        if len(self.ordered_component_ids) != len(self.component_types):
            raise CombinatorialFloorError(
                "component IDs/types must have equal length"
            )
        if len(set(self.ordered_component_ids)) != len(
            self.ordered_component_ids
        ):
            raise CombinatorialFloorError("component IDs must be unique")
        if any(not value for value in self.component_types):
            raise CombinatorialFloorError("component types must be non-empty")
        if self.batch_count <= 0:
            raise CombinatorialFloorError("batch_count must be positive")
        if self.draws_per_batch < 0:
            raise CombinatorialFloorError(
                "draws_per_batch must be non-negative"
            )
        if self.root_seed < 0:
            raise CombinatorialFloorError("root_seed must be non-negative")
        if not self.seed_stream_id:
            raise CombinatorialFloorError(
                "seed_stream_id must be non-empty"
            )
        if self.scientific_data:
            raise CombinatorialFloorError(
                "combinatorial floor requires scientific_data=false"
            )
        if self.production_eligible:
            raise CombinatorialFloorError(
                "combinatorial floor requires production_eligible=false"
            )

        universe_size = len(self.ordered_component_ids)
        if any(
            size > universe_size
            for size in self.matching_rule.retained_sizes
        ):
            raise CombinatorialFloorError(
                "retained size exceeds component universe"
            )

        if self.matching_rule.component_type_counts is not None:
            available = Counter(self.component_types)
            requested = dict(self.matching_rule.component_type_counts)
            for component_type, count in requested.items():
                if component_type not in available:
                    raise CombinatorialFloorError(
                        f"requested component type is absent: {component_type}"
                    )
                if count > available[component_type]:
                    raise CombinatorialFloorError(
                        f"impossible component-type composition: "
                        f"{component_type}"
                    )

        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class MaskProbabilityRecord:
    retained_components: tuple[int, ...]
    numerator: int
    denominator: int


@dataclass(frozen=True)
class CombinatorialDrawRecord:
    batch_index: int
    draw_index: int
    seed: int
    retained_size: int
    retained_components: tuple[int, ...]
    retained_type_counts: tuple[tuple[str, int], ...]
    mask_identity: str
    duplicate_of_draw_index: int | None


@dataclass(frozen=True)
class CombinatorialBatchRecord:
    batch_index: int
    batch_identity: str
    raw_draw_count: int
    unique_mask_count: int
    duplicate_draw_count: int
    packing_statistic: int
    selected_mask_identities: tuple[str, ...]
    compatibility_graph_hash: str


@dataclass(frozen=True)
class PackingFrequencyRecord:
    packing_statistic: int
    batch_count: int


@dataclass(frozen=True)
class CombinatorialFloorResult:
    profile_identity: str
    policy_identity: str
    basis_hash: str
    distribution_semantics: str
    overlap_rule_reference: str
    max_pairwise_overlap: float
    solver_reference: str
    tie_break_reference: str
    draws: tuple[CombinatorialDrawRecord, ...]
    batches: tuple[CombinatorialBatchRecord, ...]
    packing_distribution: tuple[PackingFrequencyRecord, ...]
    raw_draw_count: int
    unique_mask_draw_outcomes: int
    duplicate_draw_count: int
    fidelity_claim: bool = False
    exact_evaluation_count: int = 0
    endpoint2_claim: bool = False
    packing_semantics: str = (
        "combinatorial_overlap_packing_statistic_not_endpoint2"
    )
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.fidelity_claim:
            raise CombinatorialFloorError(
                "combinatorial floor cannot make a fidelity claim"
            )
        if self.exact_evaluation_count != 0:
            raise CombinatorialFloorError(
                "combinatorial floor performs no exact fidelity evaluation"
            )
        if self.endpoint2_claim:
            raise CombinatorialFloorError(
                "combinatorial floor statistic is not Endpoint 2"
            )
        if self.scientific_data or self.production_eligible:
            raise CombinatorialFloorError(
                "combinatorial floor result must remain technical-only"
            )
        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))


def _validate_policy(
    profile: CombinatorialFloorProfile,
    policy: CombinatorialPackingRule,
) -> None:
    if policy.component_basis_reference != profile.basis_hash:
        raise CombinatorialFloorError(
            "Stage 6E policy basis identity does not match combinatorial basis"
        )
    if policy.component_basis_size != len(profile.ordered_component_ids):
        raise CombinatorialFloorError(
            "Stage 6E policy basis size does not match component universe"
        )
    if policy.scientific_data or policy.production_eligible:
        raise CombinatorialFloorError(
            "combinatorial packing rule must remain technical-only"
        )
    if policy.endpoint2_policy:
        raise CombinatorialFloorError(
            "combinatorial packing rule must not claim Endpoint 2 status"
        )


def derive_combinatorial_seed(
    profile: CombinatorialFloorProfile,
    *,
    batch_index: int,
    draw_index: int,
) -> int:
    if batch_index < 0 or draw_index < 0:
        raise CombinatorialFloorError(
            "batch/draw indices must be non-negative"
        )
    digest = canonical_sha256(
        {
            "seed_contract": "stage12r3-combinatorial-seed/v1",
            "profile_identity": profile.identity,
            "root_seed": profile.root_seed,
            "seed_stream_id": profile.seed_stream_id,
            "batch_index": batch_index,
            "draw_index": draw_index,
        }
    )
    return int(digest[:16], 16)


def _type_groups(
    profile: CombinatorialFloorProfile,
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for index, component_type in enumerate(profile.component_types):
        grouped.setdefault(component_type, []).append(index)
    return {
        component_type: tuple(indices)
        for component_type, indices in grouped.items()
    }


def enumerate_matching_masks(
    profile: CombinatorialFloorProfile,
    retained_size: int,
    *,
    max_support: int = 100_000,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate the exact finite matching support for small technical cases."""

    if retained_size not in profile.matching_rule.retained_sizes:
        raise CombinatorialFloorError("retained size is not declared")

    rule = profile.matching_rule
    if rule.component_type_counts is None:
        support_size = __import__("math").comb(
            len(profile.ordered_component_ids),
            retained_size,
        )
        if support_size > max_support:
            raise CombinatorialFloorError(
                "matching support exceeds bounded enumeration limit"
            )
        return tuple(
            itertools.combinations(
                range(len(profile.ordered_component_ids)),
                retained_size,
            )
        )

    grouped = _type_groups(profile)
    requested = dict(rule.component_type_counts)

    choices: list[tuple[tuple[int, ...], ...]] = []
    support_size = 1
    for component_type in sorted(grouped):
        count = requested.get(component_type, 0)
        group_choices = tuple(
            itertools.combinations(grouped[component_type], count)
        )
        support_size *= len(group_choices)
        if support_size > max_support:
            raise CombinatorialFloorError(
                "type-matched support exceeds bounded enumeration limit"
            )
        choices.append(group_choices)

    masks = []
    for pieces in itertools.product(*choices):
        mask = tuple(sorted(itertools.chain.from_iterable(pieces)))
        masks.append(mask)
    return tuple(masks)


def exact_sampling_distribution(
    profile: CombinatorialFloorProfile,
    *,
    max_support: int = 100_000,
) -> tuple[MaskProbabilityRecord, ...]:
    """Return exact probabilities for a bounded small-universe fixture."""

    probability: dict[tuple[int, ...], Fraction] = {}
    size_entry_probability = Fraction(
        1,
        len(profile.matching_rule.retained_sizes),
    )

    for retained_size in profile.matching_rule.retained_sizes:
        support = enumerate_matching_masks(
            profile,
            retained_size,
            max_support=max_support,
        )
        if not support:
            raise CombinatorialFloorError(
                "declared matching rule has empty finite support"
            )
        within_size = Fraction(1, len(support))
        for mask in support:
            probability[mask] = probability.get(mask, Fraction()) + (
                size_entry_probability * within_size
            )

    return tuple(
        MaskProbabilityRecord(
            retained_components=mask,
            numerator=value.numerator,
            denominator=value.denominator,
        )
        for mask, value in sorted(probability.items())
    )


def _sample_mask(
    profile: CombinatorialFloorProfile,
    *,
    seed: int,
) -> tuple[int, ...]:
    rng = random.Random(seed)
    retained_size = profile.matching_rule.retained_sizes[
        rng.randrange(len(profile.matching_rule.retained_sizes))
    ]

    if profile.matching_rule.component_type_counts is None:
        return tuple(
            sorted(
                rng.sample(
                    range(len(profile.ordered_component_ids)),
                    retained_size,
                )
            )
        )

    grouped = _type_groups(profile)
    requested = dict(profile.matching_rule.component_type_counts)
    selected: list[int] = []
    for component_type in sorted(grouped):
        count = requested.get(component_type, 0)
        if count:
            selected.extend(rng.sample(grouped[component_type], count))
    return tuple(sorted(selected))


def _retained_type_counts(
    profile: CombinatorialFloorProfile,
    mask: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    counts = Counter(profile.component_types[index] for index in mask)
    return tuple(sorted(counts.items()))


def _packing_proxy_candidate(
    profile: CombinatorialFloorProfile,
    *,
    mask: tuple[int, ...],
    proposal_references: tuple[str, ...],
) -> CombinatorialPackingCandidate:
    """Build only the structural fields consumed by Stage 6E packing."""

    return CombinatorialPackingCandidate(
        component_basis_reference=profile.basis_hash,
        component_basis_size=len(profile.ordered_component_ids),
        mask_identity=canonical_mask_identity(mask),
        retained_components=mask,
        draw_references=proposal_references,
    )


def packing_statistic_for_masks(
    profile: CombinatorialFloorProfile,
    policy: CombinatorialPackingRule,
    masks: tuple[tuple[int, ...], ...],
) -> tuple[int, tuple[str, ...], str]:
    """Reuse Stage 6E graph/solver on explicit unique set-valued masks."""

    _validate_policy(profile, policy)

    unique: dict[str, tuple[int, ...]] = {}
    references: dict[str, list[str]] = {}

    for index, mask in enumerate(masks):
        if tuple(sorted(set(mask))) != mask:
            raise CombinatorialFloorError(
                "retained component masks must be sorted and unique"
            )
        if any(
            component < 0
            or component >= len(profile.ordered_component_ids)
            for component in mask
        ):
            raise CombinatorialFloorError(
                "retained component index outside basis universe"
            )

        identity = canonical_mask_identity(mask)
        unique.setdefault(identity, mask)
        references.setdefault(identity, []).append(
            f"stage12r3-combinatorial-explicit:{index}"
        )

    candidates = tuple(
        _packing_proxy_candidate(
            profile,
            mask=unique[identity],
            proposal_references=tuple(references[identity]),
        )
        for identity in sorted(unique)
    )

    graph = build_compatibility_graph(candidates, policy)
    selected = exact_maximum_compatible_subset(graph, policy)

    return (
        len(selected),
        selected,
        canonical_sha256(asdict(graph)),
    )


def run_combinatorial_floor(
    profile: CombinatorialFloorProfile,
    policy: CombinatorialPackingRule,
) -> CombinatorialFloorResult:
    """Run deterministic matched-mask batches and Stage 6E packing."""

    _validate_policy(profile, policy)

    all_draws: list[CombinatorialDrawRecord] = []
    batches: list[CombinatorialBatchRecord] = []

    for batch_index in range(profile.batch_count):
        batch_masks: list[tuple[int, ...]] = []
        first_draw: dict[str, int] = {}

        for draw_index in range(profile.draws_per_batch):
            seed = derive_combinatorial_seed(
                profile,
                batch_index=batch_index,
                draw_index=draw_index,
            )
            mask = _sample_mask(profile, seed=seed)
            identity = canonical_mask_identity(mask)

            duplicate_of = first_draw.get(identity)
            if duplicate_of is None:
                first_draw[identity] = draw_index

            all_draws.append(
                CombinatorialDrawRecord(
                    batch_index=batch_index,
                    draw_index=draw_index,
                    seed=seed,
                    retained_size=len(mask),
                    retained_components=mask,
                    retained_type_counts=_retained_type_counts(profile, mask),
                    mask_identity=identity,
                    duplicate_of_draw_index=duplicate_of,
                )
            )
            batch_masks.append(mask)

        packing, selected, graph_hash = packing_statistic_for_masks(
            profile,
            policy,
            tuple(batch_masks),
        )

        unique_count = len(
            {canonical_mask_identity(mask) for mask in batch_masks}
        )
        batch_identity = canonical_sha256(
            {
                "profile_identity": profile.identity,
                "batch_index": batch_index,
                "seed_stream_id": profile.seed_stream_id,
            }
        )
        batches.append(
            CombinatorialBatchRecord(
                batch_index=batch_index,
                batch_identity=batch_identity,
                raw_draw_count=len(batch_masks),
                unique_mask_count=unique_count,
                duplicate_draw_count=len(batch_masks) - unique_count,
                packing_statistic=packing,
                selected_mask_identities=selected,
                compatibility_graph_hash=graph_hash,
            )
        )

    frequencies = Counter(batch.packing_statistic for batch in batches)
    unique_outcomes = len({draw.mask_identity for draw in all_draws})

    result = CombinatorialFloorResult(
        profile_identity=profile.identity,
        policy_identity=canonical_sha256(asdict(policy)),
        basis_hash=profile.basis_hash,
        distribution_semantics=profile.matching_rule.distribution_semantics,
        overlap_rule_reference=policy.overlap_rule_reference,
        max_pairwise_overlap=policy.max_pairwise_overlap,
        solver_reference=policy.solver_reference,
        tie_break_reference=policy.tie_break_reference,
        draws=tuple(all_draws),
        batches=tuple(batches),
        packing_distribution=tuple(
            PackingFrequencyRecord(
                packing_statistic=value,
                batch_count=count,
            )
            for value, count in sorted(frequencies.items())
        ),
        raw_draw_count=len(all_draws),
        unique_mask_draw_outcomes=unique_outcomes,
        duplicate_draw_count=sum(
            batch.duplicate_draw_count for batch in batches
        ),
    )
    return result
