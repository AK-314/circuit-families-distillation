from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, median

from .contrasts import PhaseContrast, TeacherStudentContrast
from .profiles import TechnicalAnalysisProfile


class PopulationAggregationError(ValueError):
    pass


EXPECTED_POPULATION_TEACHER_SEEDS = 5
POPULATION_INPUT_STATES = frozenset(
    {
        "defined",
        "absent",
        "failed",
        "ineligible",
        "unavailable",
        "unresolved",
        "insufficient",
        "incompatible",
        "inapplicable",
    }
)


@dataclass(frozen=True, slots=True, order=True)
class PhasePopulationMetric:
    distillation_condition: str
    method_id: str
    endpoint_id: str
    phase_from: str
    phase_to: str


@dataclass(frozen=True, slots=True, order=True)
class TeacherStudentPopulationMetric:
    phase: str
    distillation_condition: str
    method_id: str
    endpoint_id: str
    protocol_id: str
    fidelity_id: str
    budget_id: str


PopulationMetric = PhasePopulationMetric | TeacherStudentPopulationMetric


@dataclass(frozen=True, slots=True)
class PopulationSummary:
    metric_kind: str
    metric_identity: PopulationMetric
    population_unit: str
    observed_teacher_seeds: tuple[int, ...]
    contributing_teacher_seeds: tuple[int, ...]
    number_defined_teacher_seeds: int
    reducer: str
    aggregated_value: float | None
    failed_seeds: tuple[int, ...]
    ineligible_seeds: tuple[int, ...]
    unavailable_seeds: tuple[int, ...]
    unresolved_seeds: tuple[int, ...]
    insufficient_seeds: tuple[int, ...]
    inapplicable_seeds: tuple[int, ...]
    absent_seeds: tuple[int, ...]
    incompatible_seeds: tuple[int, ...]
    low_population_warning: str | None

    @property
    def defined_teacher_seed_count(self) -> int:
        return self.number_defined_teacher_seeds


def _validate_profile(profile: TechnicalAnalysisProfile) -> str:
    if profile.scientific_data or not profile.synthetic_only:
        raise PopulationAggregationError(
            "Stage 5D population summaries require synthetic-only data"
        )

    if profile.production_eligible:
        raise PopulationAggregationError(
            "Stage 5D population summaries cannot be production eligible"
        )

    if profile.resolves_decisions:
        raise PopulationAggregationError(
            "Stage 5D population summaries may not resolve decisions"
        )

    reducer = profile.settings.population_reducer
    if reducer not in {"mean", "median"}:
        raise PopulationAggregationError(
            f"unsupported population reducer: {reducer}"
        )

    return reducer


def _reduce(values: Sequence[float], reducer: str) -> float:
    if not values:
        raise PopulationAggregationError(
            "population reducer requires at least one teacher-seed value"
        )

    if reducer == "mean":
        return float(fmean(values))
    if reducer == "median":
        return float(median(values))

    raise PopulationAggregationError(
        f"unsupported population reducer: {reducer}"
    )


def _validate_phase_direction(row: PhaseContrast) -> None:
    if (
        row.left_value is None
        or row.right_value is None
        or row.delta is None
    ):
        raise PopulationAggregationError(
            "defined phase contrast requires left, right, and delta values"
        )

    expected = float(row.right_value) - float(row.left_value)
    if not math.isclose(float(row.delta), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise PopulationAggregationError(
            "phase contrast direction must be right phase minus left phase"
        )


def _validate_teacher_student_direction(
    row: TeacherStudentContrast,
) -> None:
    if (
        row.student_summary_value is None
        or row.teacher_value is None
        or row.delta is None
    ):
        raise PopulationAggregationError(
            "defined teacher-student contrast requires student, teacher, "
            "and delta values"
        )

    expected = float(row.student_summary_value) - float(row.teacher_value)
    if not math.isclose(float(row.delta), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise PopulationAggregationError(
            "teacher-student contrast direction must be student summary "
            "minus direct teacher"
        )


def _summarize_group(
    *,
    metric_kind: str,
    metric_identity: PopulationMetric,
    rows: Sequence[PhaseContrast | TeacherStudentContrast],
    reducer: str,
) -> PopulationSummary:
    by_seed: dict[int, PhaseContrast | TeacherStudentContrast] = {}

    for row in rows:
        if row.contribution_unit != "teacher_seed":
            raise PopulationAggregationError(
                "population aggregation accepts teacher-seed contributions only"
            )

        seed = row.key.teacher_seed
        if seed in by_seed:
            raise PopulationAggregationError(
                "duplicate population contribution for teacher seed "
                f"{seed}: metric={metric_identity!r}"
            )

        if row.state not in POPULATION_INPUT_STATES:
            raise PopulationAggregationError(
                f"unsupported population input state: {row.state}"
            )

        if row.state != "defined" and row.delta is not None:
            raise PopulationAggregationError(
                "non-defined contrast may not contribute a numeric delta"
            )

        by_seed[seed] = row

    defined_rows = tuple(
        sorted(
            (
                row
                for row in by_seed.values()
                if row.state == "defined"
            ),
            key=lambda row: row.key.teacher_seed,
        )
    )

    for row in defined_rows:
        if isinstance(row, PhaseContrast):
            _validate_phase_direction(row)
        elif isinstance(row, TeacherStudentContrast):
            _validate_teacher_student_direction(row)
        else:
            raise PopulationAggregationError(
                "population input must be a Part F contrast object"
            )

    contributing_seeds = tuple(
        row.key.teacher_seed
        for row in defined_rows
    )
    values = tuple(
        float(row.delta)
        for row in defined_rows
        if row.delta is not None
    )
    number_defined = len(contributing_seeds)

    def seeds_for(state: str) -> tuple[int, ...]:
        return tuple(
            sorted(
                seed
                for seed, row in by_seed.items()
                if row.state == state
            )
        )

    warning = None
    if number_defined < EXPECTED_POPULATION_TEACHER_SEEDS:
        warning = (
            "low_population_teacher_seed_count:"
            f"defined={number_defined}:"
            f"expected={EXPECTED_POPULATION_TEACHER_SEEDS}"
        )

    return PopulationSummary(
        metric_kind=metric_kind,
        metric_identity=metric_identity,
        population_unit="teacher_seed",
        observed_teacher_seeds=tuple(sorted(by_seed)),
        contributing_teacher_seeds=contributing_seeds,
        number_defined_teacher_seeds=number_defined,
        reducer=reducer,
        aggregated_value=(
            None
            if not values
            else _reduce(values, reducer)
        ),
        failed_seeds=seeds_for("failed"),
        ineligible_seeds=seeds_for("ineligible"),
        unavailable_seeds=seeds_for("unavailable"),
        unresolved_seeds=seeds_for("unresolved"),
        insufficient_seeds=seeds_for("insufficient"),
        inapplicable_seeds=seeds_for("inapplicable"),
        absent_seeds=seeds_for("absent"),
        incompatible_seeds=seeds_for("incompatible"),
        low_population_warning=warning,
    )


def build_phase_population_summaries(
    contrasts: Sequence[PhaseContrast],
    profile: TechnicalAnalysisProfile,
) -> tuple[PopulationSummary, ...]:
    reducer = _validate_profile(profile)
    grouped: dict[PhasePopulationMetric, list[PhaseContrast]] = defaultdict(list)

    for contrast in contrasts:
        if not isinstance(contrast, PhaseContrast):
            raise PopulationAggregationError(
                "phase population input must contain PhaseContrast objects"
            )

        key = contrast.key
        identity = PhasePopulationMetric(
            distillation_condition=key.distillation_condition,
            method_id=key.method_id,
            endpoint_id=key.endpoint_id,
            phase_from=key.phase_from,
            phase_to=key.phase_to,
        )
        grouped[identity].append(contrast)

    return tuple(
        _summarize_group(
            metric_kind="phase_contrast",
            metric_identity=identity,
            rows=grouped[identity],
            reducer=reducer,
        )
        for identity in sorted(grouped)
    )


def build_teacher_student_population_summaries(
    contrasts: Sequence[TeacherStudentContrast],
    profile: TechnicalAnalysisProfile,
) -> tuple[PopulationSummary, ...]:
    reducer = _validate_profile(profile)
    grouped: dict[
        TeacherStudentPopulationMetric,
        list[TeacherStudentContrast],
    ] = defaultdict(list)

    for contrast in contrasts:
        if not isinstance(contrast, TeacherStudentContrast):
            raise PopulationAggregationError(
                "teacher-student population input must contain "
                "TeacherStudentContrast objects"
            )

        key = contrast.key
        identity = TeacherStudentPopulationMetric(
            phase=key.phase,
            distillation_condition=key.distillation_condition,
            method_id=key.method_id,
            endpoint_id=key.endpoint_id,
            protocol_id=key.protocol_id,
            fidelity_id=key.fidelity_id,
            budget_id=key.budget_id,
        )
        grouped[identity].append(contrast)

    return tuple(
        _summarize_group(
            metric_kind="teacher_student_contrast",
            metric_identity=identity,
            rows=grouped[identity],
            reducer=reducer,
        )
        for identity in sorted(grouped)
    )
