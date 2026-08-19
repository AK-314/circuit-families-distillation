"""Versioned technical-only configuration boundary for Stages 5B/C.

These records inject candidate settings into synthetic technical fixtures.
They deliberately do not freeze UD-003, UD-004, UD-005, UD-006, or UD-013.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_SCHEMA_VERSION = "stage5bc-technical-profile/v1"
PROFILE_SET_SCHEMA_VERSION = "stage5bc-technical-profile-set/v1"
TECHNICAL_CLASSIFICATION = "technical_fixture"
TECHNICAL_SETTINGS_STATUS = "technical_candidate_only"

PROFILE_KINDS = (
    "architecture",
    "trainer",
    "adapter",
    "resume",
    "orchestration",
)

ALLOWED_DECISION_DEPENDENCIES = frozenset(
    {
        "UD-003",
        "UD-004",
        "UD-005",
        "UD-006",
        "UD-013",
    }
)

_PROFILE_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "profile_id",
        "profile_kind",
        "classification",
        "scientific_data",
        "production_eligible",
        "settings_status",
        "decision_dependencies",
        "resolves_decisions",
        "settings",
    }
)

_PROFILE_SET_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "profiles",
    }
)


class TechnicalProfileError(ValueError):
    """Raised when a technical profile could masquerade as a frozen choice."""


def _require_exact_keys(
    value: Mapping[str, Any],
    required: frozenset[str],
    *,
    name: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing or extra:
        raise TechnicalProfileError(
            f"{name} keys mismatch: missing={missing!r}, extra={extra!r}"
        )


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TechnicalProfileError(f"{name} must be a non-empty string")
    return value


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return

    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise TechnicalProfileError(f"{path} must not contain non-finite floats")
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TechnicalProfileError(
                    f"{path} mapping keys must be strings"
                )
            _validate_json_value(item, path=f"{path}.{key}")
        return

    raise TechnicalProfileError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


@dataclass(frozen=True)
class TechnicalProfile:
    """One explicit technical candidate profile with no scientific authority."""

    schema_version: str
    profile_id: str
    profile_kind: str
    classification: str
    scientific_data: bool
    production_eligible: bool
    settings_status: str
    decision_dependencies: tuple[str, ...]
    resolves_decisions: tuple[str, ...]
    settings: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TechnicalProfile:
        if not isinstance(value, Mapping):
            raise TechnicalProfileError("technical profile must be a mapping")

        _require_exact_keys(
            value,
            _PROFILE_REQUIRED_KEYS,
            name="technical profile",
        )

        if value["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise TechnicalProfileError(
                "technical profile schema_version must be "
                f"{PROFILE_SCHEMA_VERSION!r}"
            )

        profile_id = _require_nonempty_string(
            value["profile_id"],
            name="profile_id",
        )
        if not profile_id.startswith("technical-") or "/v" not in profile_id:
            raise TechnicalProfileError(
                "profile_id must be explicitly technical and versioned"
            )

        profile_kind = value["profile_kind"]
        if profile_kind not in PROFILE_KINDS:
            raise TechnicalProfileError(
                f"profile_kind must be one of {PROFILE_KINDS!r}"
            )

        if value["classification"] != TECHNICAL_CLASSIFICATION:
            raise TechnicalProfileError(
                "classification must be 'technical_fixture'"
            )

        if value["scientific_data"] is not False:
            raise TechnicalProfileError(
                "technical profile must explicitly set scientific_data=false"
            )

        if value["production_eligible"] is not False:
            raise TechnicalProfileError(
                "technical profile must explicitly set production_eligible=false"
            )

        if value["settings_status"] != TECHNICAL_SETTINGS_STATUS:
            raise TechnicalProfileError(
                "settings_status must be 'technical_candidate_only'"
            )

        dependencies_raw = value["decision_dependencies"]
        if not isinstance(dependencies_raw, list) or not dependencies_raw:
            raise TechnicalProfileError(
                "decision_dependencies must be a non-empty explicit list"
            )

        if any(not isinstance(item, str) for item in dependencies_raw):
            raise TechnicalProfileError(
                "decision_dependencies entries must be strings"
            )

        dependencies = tuple(dependencies_raw)
        if len(set(dependencies)) != len(dependencies):
            raise TechnicalProfileError(
                "decision_dependencies must not contain duplicates"
            )

        unknown = sorted(set(dependencies) - ALLOWED_DECISION_DEPENDENCIES)
        if unknown:
            raise TechnicalProfileError(
                f"unknown unresolved-decision dependencies: {unknown!r}"
            )

        resolves_raw = value["resolves_decisions"]
        if not isinstance(resolves_raw, list):
            raise TechnicalProfileError(
                "resolves_decisions must be an explicit list"
            )
        if resolves_raw:
            raise TechnicalProfileError(
                "technical profiles must resolve no scientific decisions"
            )

        settings_raw = value["settings"]
        if not isinstance(settings_raw, Mapping):
            raise TechnicalProfileError("settings must be an explicit mapping")

        _validate_json_value(settings_raw, path="settings")
        settings = copy.deepcopy(dict(settings_raw))

        return cls(
            schema_version=PROFILE_SCHEMA_VERSION,
            profile_id=profile_id,
            profile_kind=profile_kind,
            classification=TECHNICAL_CLASSIFICATION,
            scientific_data=False,
            production_eligible=False,
            settings_status=TECHNICAL_SETTINGS_STATUS,
            decision_dependencies=dependencies,
            resolves_decisions=(),
            settings=settings,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable explicit record with no filled defaults."""
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_kind": self.profile_kind,
            "classification": self.classification,
            "scientific_data": self.scientific_data,
            "production_eligible": self.production_eligible,
            "settings_status": self.settings_status,
            "decision_dependencies": list(self.decision_dependencies),
            "resolves_decisions": list(self.resolves_decisions),
            "settings": copy.deepcopy(dict(self.settings)),
        }


@dataclass(frozen=True)
class TechnicalProfileSet:
    """Exactly one injected technical profile for each Stage 5B/C concern."""

    schema_version: str
    profiles: tuple[TechnicalProfile, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TechnicalProfileSet:
        if not isinstance(value, Mapping):
            raise TechnicalProfileError("technical profile set must be a mapping")

        _require_exact_keys(
            value,
            _PROFILE_SET_REQUIRED_KEYS,
            name="technical profile set",
        )

        if value["schema_version"] != PROFILE_SET_SCHEMA_VERSION:
            raise TechnicalProfileError(
                "technical profile-set schema_version must be "
                f"{PROFILE_SET_SCHEMA_VERSION!r}"
            )

        raw_profiles = value["profiles"]
        if not isinstance(raw_profiles, list):
            raise TechnicalProfileError("profiles must be an explicit list")

        profiles = tuple(
            TechnicalProfile.from_mapping(item)
            for item in raw_profiles
        )

        ids = [profile.profile_id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise TechnicalProfileError("profile_id values must be unique")

        kinds = [profile.profile_kind for profile in profiles]
        if len(kinds) != len(set(kinds)):
            raise TechnicalProfileError(
                "technical profile set must contain each profile_kind once"
            )

        if set(kinds) != set(PROFILE_KINDS):
            raise TechnicalProfileError(
                "technical profile set must contain exactly these kinds: "
                f"{PROFILE_KINDS!r}"
            )

        return cls(
            schema_version=PROFILE_SET_SCHEMA_VERSION,
            profiles=profiles,
        )

    def by_kind(self, profile_kind: str) -> TechnicalProfile:
        """Return the unique profile for a validated kind."""
        for profile in self.profiles:
            if profile.profile_kind == profile_kind:
                return profile
        raise TechnicalProfileError(
            f"profile kind not present: {profile_kind!r}"
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return deterministic input-order-independent profile-set mapping."""
        ordered = sorted(
            self.profiles,
            key=lambda profile: PROFILE_KINDS.index(profile.profile_kind),
        )
        return {
            "schema_version": self.schema_version,
            "profiles": [
                profile.to_mapping()
                for profile in ordered
            ],
        }


def load_technical_profile_set(path: str | Path) -> TechnicalProfileSet:
    """Load strict JSON with no NaN or implicit scientific defaults."""
    file_path = Path(path)
    try:
        data = json.loads(
            file_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                TechnicalProfileError(
                    f"non-standard JSON constant forbidden: {token}"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise TechnicalProfileError(
            f"invalid technical-profile JSON: {exc}"
        ) from exc

    return TechnicalProfileSet.from_mapping(data)
