from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_SCHEMA_VERSION = "stage5d_technical_analysis_profile_v1"
PROFILE_SET_SCHEMA_VERSION = "stage5d_technical_analysis_profile_set_v1"
TECHNICAL_CLASSIFICATION = "synthetic_technical_only"
TECHNICAL_SELECTION_RULE = (
    "caller_must_inject_profile_id_and_never_select_using_scientific_effects"
)
TECHNICAL_SELECTION_BASIS = "technical_fixture_only_never_scientific_effects"

ALLOWED_CELL_REDUCERS = frozenset({"median", "mean"})
ALLOWED_POPULATION_REDUCERS = frozenset({"median", "mean"})
ALLOWED_DECISION_DEPENDENCIES = frozenset({"UD-004", "UD-011", "UD-012"})


class TechnicalAnalysisProfileError(ValueError):
    pass


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TechnicalAnalysisProfileError(f"{label} must be an object")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TechnicalAnalysisProfileError(
            f"{label} keys mismatch: missing={missing} extra={extra}"
        )


def _expect_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TechnicalAnalysisProfileError(f"{label} must be a non-empty string")
    return value


def _expect_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TechnicalAnalysisProfileError(f"{label} must be a boolean")
    return value


def _expect_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise TechnicalAnalysisProfileError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class TechnicalAnalysisSettings:
    cell_reducer: str
    minimum_eligible_students: int
    phase_pairs: tuple[tuple[str, str], ...]
    population_reducer: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TechnicalAnalysisSettings:
        _expect_exact_keys(
            value,
            {
                "cell_reducer",
                "minimum_eligible_students",
                "phase_pairs",
                "population_reducer",
            },
            "profile.settings",
        )

        cell_reducer = _expect_str(value["cell_reducer"], "cell_reducer")
        if cell_reducer not in ALLOWED_CELL_REDUCERS:
            raise TechnicalAnalysisProfileError(
                f"unsupported technical cell reducer: {cell_reducer}"
            )

        population_reducer = _expect_str(
            value["population_reducer"],
            "population_reducer",
        )
        if population_reducer not in ALLOWED_POPULATION_REDUCERS:
            raise TechnicalAnalysisProfileError(
                f"unsupported technical population reducer: {population_reducer}"
            )

        minimum_eligible_students = _expect_positive_int(
            value["minimum_eligible_students"],
            "minimum_eligible_students",
        )

        raw_pairs = value["phase_pairs"]
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise TechnicalAnalysisProfileError(
                "phase_pairs must be a non-empty list"
            )

        phase_pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for index, raw_pair in enumerate(raw_pairs):
            if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                raise TechnicalAnalysisProfileError(
                    f"phase_pairs[{index}] must contain exactly two phases"
                )

            left = _expect_str(raw_pair[0], f"phase_pairs[{index}][0]")
            right = _expect_str(raw_pair[1], f"phase_pairs[{index}][1]")

            if left == right:
                raise TechnicalAnalysisProfileError(
                    "a technical phase pair cannot compare a phase with itself"
                )

            pair = (left, right)
            if pair in seen:
                raise TechnicalAnalysisProfileError(
                    f"duplicate technical phase pair: {pair}"
                )

            seen.add(pair)
            phase_pairs.append(pair)

        return cls(
            cell_reducer=cell_reducer,
            minimum_eligible_students=minimum_eligible_students,
            phase_pairs=tuple(phase_pairs),
            population_reducer=population_reducer,
        )


@dataclass(frozen=True, slots=True)
class TechnicalAnalysisProfile:
    schema_version: str
    profile_id: str
    classification: str
    synthetic_only: bool
    scientific_data: bool
    production_eligible: bool
    decision_dependencies: tuple[str, ...]
    resolves_decisions: tuple[str, ...]
    selection_basis: str
    settings: TechnicalAnalysisSettings

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TechnicalAnalysisProfile:
        _expect_exact_keys(
            raw,
            {
                "schema_version",
                "profile_id",
                "classification",
                "synthetic_only",
                "scientific_data",
                "production_eligible",
                "decision_dependencies",
                "resolves_decisions",
                "selection_basis",
                "settings",
            },
            "technical analysis profile",
        )

        schema_version = _expect_str(raw["schema_version"], "schema_version")
        if schema_version != PROFILE_SCHEMA_VERSION:
            raise TechnicalAnalysisProfileError(
                f"unsupported profile schema version: {schema_version}"
            )

        profile_id = _expect_str(raw["profile_id"], "profile_id")

        classification = _expect_str(raw["classification"], "classification")
        if classification != TECHNICAL_CLASSIFICATION:
            raise TechnicalAnalysisProfileError(
                "technical profile classification is not synthetic-only"
            )

        synthetic_only = _expect_bool(raw["synthetic_only"], "synthetic_only")
        if not synthetic_only:
            raise TechnicalAnalysisProfileError(
                "Stage 5D profiles must be synthetic-only"
            )

        scientific_data = _expect_bool(raw["scientific_data"], "scientific_data")
        if scientific_data:
            raise TechnicalAnalysisProfileError(
                "Stage 5D profiles cannot permit scientific data"
            )

        production_eligible = _expect_bool(
            raw["production_eligible"],
            "production_eligible",
        )
        if production_eligible:
            raise TechnicalAnalysisProfileError(
                "Stage 5D profiles cannot be production eligible"
            )

        raw_dependencies = raw["decision_dependencies"]
        if not isinstance(raw_dependencies, list):
            raise TechnicalAnalysisProfileError(
                "decision_dependencies must be a list"
            )

        dependencies = tuple(
            _expect_str(value, "decision dependency")
            for value in raw_dependencies
        )

        if len(set(dependencies)) != len(dependencies):
            raise TechnicalAnalysisProfileError(
                "decision_dependencies contains duplicates"
            )

        unsupported = sorted(
            set(dependencies) - ALLOWED_DECISION_DEPENDENCIES
        )
        if unsupported:
            raise TechnicalAnalysisProfileError(
                f"unsupported decision dependencies: {unsupported}"
            )

        raw_resolves = raw["resolves_decisions"]
        if not isinstance(raw_resolves, list):
            raise TechnicalAnalysisProfileError(
                "resolves_decisions must be a list"
            )
        if raw_resolves:
            raise TechnicalAnalysisProfileError(
                "Stage 5D technical profiles may not resolve any UD item"
            )

        selection_basis = _expect_str(
            raw["selection_basis"],
            "selection_basis",
        )
        if selection_basis != TECHNICAL_SELECTION_BASIS:
            raise TechnicalAnalysisProfileError(
                "profile does not explicitly forbid scientific-effect selection"
            )

        settings_raw = _expect_mapping(raw["settings"], "settings")
        settings = TechnicalAnalysisSettings.from_mapping(settings_raw)

        return cls(
            schema_version=schema_version,
            profile_id=profile_id,
            classification=classification,
            synthetic_only=synthetic_only,
            scientific_data=scientific_data,
            production_eligible=production_eligible,
            decision_dependencies=dependencies,
            resolves_decisions=(),
            selection_basis=selection_basis,
            settings=settings,
        )


@dataclass(frozen=True, slots=True)
class TechnicalAnalysisProfileSet:
    schema_version: str
    selection_rule: str
    profiles: tuple[TechnicalAnalysisProfile, ...]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> TechnicalAnalysisProfileSet:
        _expect_exact_keys(
            raw,
            {"schema_version", "selection_rule", "profiles"},
            "technical profile set",
        )

        schema_version = _expect_str(
            raw["schema_version"],
            "profile set schema_version",
        )
        if schema_version != PROFILE_SET_SCHEMA_VERSION:
            raise TechnicalAnalysisProfileError(
                f"unsupported profile-set schema version: {schema_version}"
            )

        selection_rule = _expect_str(
            raw["selection_rule"],
            "selection_rule",
        )
        if selection_rule != TECHNICAL_SELECTION_RULE:
            raise TechnicalAnalysisProfileError(
                "technical profile set must require explicit injection and "
                "forbid selection from scientific effects"
            )

        raw_profiles = raw["profiles"]
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise TechnicalAnalysisProfileError(
                "profile set must contain at least one profile"
            )

        profiles = tuple(
            TechnicalAnalysisProfile.from_mapping(
                _expect_mapping(item, f"profiles[{index}]")
            )
            for index, item in enumerate(raw_profiles)
        )

        ids = [profile.profile_id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise TechnicalAnalysisProfileError(
                "technical profile IDs must be unique"
            )

        return cls(
            schema_version=schema_version,
            selection_rule=selection_rule,
            profiles=profiles,
        )

    def require(self, profile_id: str) -> TechnicalAnalysisProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise TechnicalAnalysisProfileError(
            f"unknown injected technical profile: {profile_id}"
        )


def load_technical_analysis_profile_set(
    path: str | Path,
) -> TechnicalAnalysisProfileSet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return TechnicalAnalysisProfileSet.from_mapping(
        _expect_mapping(raw, "technical profile set")
    )
