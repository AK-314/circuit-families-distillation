"""Versioned technical-only Stage 7A run request and manifest contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from circuit_families.followup_namespace import logical_root
from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage4_seed_derivation import (
    PURPOSE_DEPTHS,
    SEED_DERIVATION_VERSION,
    SeedEvidence,
    SeedInputs,
    derive_seed,
)
from circuit_families.stage7.lifecycle import (
    Stage7Lifecycle,
    build_stage7_lifecycle,
)

RUN_REQUEST_SCHEMA_VERSION: Final = "stage7-technical-run-request/v1"
RUN_MANIFEST_SCHEMA_VERSION: Final = "stage7-technical-run-manifest/v1"
TECHNICAL_CLASSIFICATION: Final = "synthetic_technical_only"
EXCLUSION_REGISTER_PATH: Final = (
    "followup/manifests/stage2_excluded_development_register_v1.json"
)

_REFERENCE_SOURCE_KINDS: Final = frozenset(
    {
        "repository_file",
        "injected_fixture",
        "registered_reference",
    }
)

_REFERENCE_KEYS: Final = frozenset(
    {
        "reference_id",
        "source_kind",
        "source",
        "sha256",
        "selector",
        "classification",
        "scientific_data",
        "production_eligible",
        "resolves_decisions",
    }
)

_SEED_KEYS: Final = frozenset(
    {
        "label",
        "condition_id",
        "purpose",
        "attempt_index",
        "retry_index",
    }
)

_REQUEST_KEYS: Final = frozenset(
    {
        "schema_version",
        "request_id",
        "classification",
        "scientific_data",
        "production_eligible",
        "production_default",
        "teacher_reference",
        "hard_profile_reference",
        "soft_profile_reference",
        "discovery_profile_references",
        "endpoint1_policy_reference",
        "endpoint2_policy_reference",
        "analysis_profile_reference",
        "job_lifecycle_reference",
        "seed_derivation_reference",
        "seed_inputs",
        "output_root",
        "exclusion_register_reference",
        "decision_dependencies",
        "resolves_decisions",
    }
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_ID_RE: Final = re.compile(
    r"technical-[a-z0-9._-]+/v[1-9][0-9]*\Z"
)
_UD_RE: Final = re.compile(r"UD-[0-9]{3}\Z")


class Stage7ContractError(ValueError):
    """Raised when a Stage 7A request could masquerade as production."""


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)

    if missing or extra:
        raise Stage7ContractError(
            f"{label} keys mismatch: missing={missing!r}, extra={extra!r}"
        )


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage7ContractError(
            f"{label} must be a non-empty string"
        )
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the deterministic Stage 7A JSON representation."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Stage7ContractError(
            "Stage 7A records must contain canonical JSON values"
        ) from exc

    return (encoded + "\n").encode("ascii")


@dataclass(frozen=True)
class BoundReference:
    """One injected or repository-bound technical reference."""

    reference_id: str
    source_kind: str
    source: str
    sha256: str
    selector: str | None
    classification: str
    scientific_data: bool
    production_eligible: bool
    resolves_decisions: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BoundReference:
        if not isinstance(value, Mapping):
            raise Stage7ContractError(
                "bound reference must be a mapping"
            )

        _require_exact_keys(
            value,
            _REFERENCE_KEYS,
            label="bound reference",
        )

        reference_id = _nonempty_string(
            value["reference_id"],
            label="reference_id",
        )
        source_kind = value["source_kind"]

        if source_kind not in _REFERENCE_SOURCE_KINDS:
            raise Stage7ContractError(
                f"unsupported reference source_kind: {source_kind!r}"
            )

        source = _nonempty_string(
            value["source"],
            label="source",
        )
        digest = value["sha256"]

        if (
            not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
        ):
            raise Stage7ContractError(
                "bound reference sha256 must be 64 lowercase hex characters"
            )

        selector = value["selector"]

        if selector is not None and (
            not isinstance(selector, str) or not selector
        ):
            raise Stage7ContractError(
                "reference selector must be null or a non-empty string"
            )

        if value["classification"] != "technical_fixture":
            raise Stage7ContractError(
                "every Stage 7A bound reference must be technical_fixture"
            )

        if value["scientific_data"] is not False:
            raise Stage7ContractError(
                "Stage 7A references must set scientific_data=false"
            )

        if value["production_eligible"] is not False:
            raise Stage7ContractError(
                "Stage 7A references must set production_eligible=false"
            )

        resolves = value["resolves_decisions"]

        if not isinstance(resolves, list):
            raise Stage7ContractError(
                "reference resolves_decisions must be an explicit list"
            )

        if resolves:
            raise Stage7ContractError(
                "Stage 7A references may not resolve unresolved decisions"
            )

        if source_kind == "repository_file":
            source_path = PurePosixPath(source)

            if (
                source_path.is_absolute()
                or ".." in source_path.parts
                or not source_path.parts
            ):
                raise Stage7ContractError(
                    "repository reference must use a safe relative path"
                )

        elif source_kind == "injected_fixture":
            if not source.startswith(
                ("synthetic://", "injected://")
            ):
                raise Stage7ContractError(
                    "injected fixtures must use synthetic:// or injected://"
                )

        elif source_kind == "registered_reference":
            if not source.startswith("registered://"):
                raise Stage7ContractError(
                    "registered references must be metadata-only registered:// "
                    "references, not private filesystem paths"
                )

        return cls(
            reference_id=reference_id,
            source_kind=source_kind,
            source=source,
            sha256=digest,
            selector=selector,
            classification="technical_fixture",
            scientific_data=False,
            production_eligible=False,
            resolves_decisions=(),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "source_kind": self.source_kind,
            "source": self.source,
            "sha256": self.sha256,
            "selector": self.selector,
            "classification": self.classification,
            "scientific_data": self.scientific_data,
            "production_eligible": self.production_eligible,
            "resolves_decisions": list(self.resolves_decisions),
        }


@dataclass(frozen=True)
class SeedInputBinding:
    """One explicit input to the accepted Stage 4 seed derivation."""

    label: str
    condition_id: str
    purpose: str
    attempt_index: int
    retry_index: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> SeedInputBinding:
        if not isinstance(value, Mapping):
            raise Stage7ContractError(
                "seed input must be a mapping"
            )

        _require_exact_keys(
            value,
            _SEED_KEYS,
            label="seed input",
        )

        label = _nonempty_string(
            value["label"],
            label="seed label",
        )
        condition_id = _nonempty_string(
            value["condition_id"],
            label="seed condition_id",
        )
        purpose = value["purpose"]

        if purpose not in PURPOSE_DEPTHS:
            raise Stage7ContractError(
                f"unsupported Stage 4 seed purpose: {purpose!r}"
            )

        attempt_index = value["attempt_index"]
        retry_index = value["retry_index"]

        for name, raw in (
            ("attempt_index", attempt_index),
            ("retry_index", retry_index),
        ):
            if (
                isinstance(raw, bool)
                or not isinstance(raw, int)
                or raw < 0
            ):
                raise Stage7ContractError(
                    f"{name} must be a non-negative integer"
                )

        return cls(
            label=label,
            condition_id=condition_id,
            purpose=purpose,
            attempt_index=attempt_index,
            retry_index=retry_index,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "label": self.label,
            "condition_id": self.condition_id,
            "purpose": self.purpose,
            "attempt_index": self.attempt_index,
            "retry_index": self.retry_index,
        }

    def to_seed_inputs(self) -> SeedInputs:
        return SeedInputs(
            condition_id=self.condition_id,
            purpose=self.purpose,
            attempt_index=self.attempt_index,
            retry_index=self.retry_index,
        )


def _validate_output_root(output_root: Any) -> str:
    value = _nonempty_string(
        output_root,
        label="output_root",
    )
    root_path = PurePosixPath(value)
    approved = logical_root("excluded_development")

    if root_path.is_absolute():
        raise Stage7ContractError(
            "Stage 7A request output_root must be repository-relative"
        )

    if ".." in root_path.parts:
        raise Stage7ContractError(
            "Stage 7A request output_root may not traverse directories"
        )

    if (
        root_path == approved
        or root_path.parts[: len(approved.parts)] != approved.parts
    ):
        raise Stage7ContractError(
            "Stage 7A request output_root must be an isolated subdirectory "
            "beneath followup/excluded_development"
        )

    return root_path.as_posix()


def _decision_dependencies(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise Stage7ContractError(
            "decision_dependencies must be a non-empty explicit list"
        )

    if any(
        not isinstance(item, str)
        or not _UD_RE.fullmatch(item)
        for item in value
    ):
        raise Stage7ContractError(
            "decision_dependencies must contain canonical UD-### IDs"
        )

    if len(set(value)) != len(value):
        raise Stage7ContractError(
            "decision_dependencies may not contain duplicates"
        )

    return tuple(value)


@dataclass(frozen=True)
class Stage7TechnicalRunRequest:
    """Fully injected, non-production Stage 7A technical request."""

    schema_version: str
    request_id: str
    classification: str
    scientific_data: bool
    production_eligible: bool
    production_default: bool
    teacher_reference: BoundReference
    hard_profile_reference: BoundReference
    soft_profile_reference: BoundReference
    discovery_profile_references: tuple[BoundReference, ...]
    endpoint1_policy_reference: BoundReference
    endpoint2_policy_reference: BoundReference
    analysis_profile_reference: BoundReference
    job_lifecycle_reference: BoundReference
    seed_derivation_reference: str
    seed_inputs: tuple[SeedInputBinding, ...]
    output_root: str
    exclusion_register_reference: BoundReference
    decision_dependencies: tuple[str, ...]
    resolves_decisions: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> Stage7TechnicalRunRequest:
        if not isinstance(value, Mapping):
            raise Stage7ContractError(
                "Stage 7 technical run request must be a mapping"
            )

        _require_exact_keys(
            value,
            _REQUEST_KEYS,
            label="Stage 7 technical run request",
        )

        if value["schema_version"] != RUN_REQUEST_SCHEMA_VERSION:
            raise Stage7ContractError(
                f"schema_version must be "
                f"{RUN_REQUEST_SCHEMA_VERSION!r}"
            )

        request_id = _nonempty_string(
            value["request_id"],
            label="request_id",
        )

        if not _REQUEST_ID_RE.fullmatch(request_id):
            raise Stage7ContractError(
                "request_id must be explicitly technical and versioned"
            )

        if value["classification"] != TECHNICAL_CLASSIFICATION:
            raise Stage7ContractError(
                f"classification must be "
                f"{TECHNICAL_CLASSIFICATION!r}"
            )

        if value["scientific_data"] is not False:
            raise Stage7ContractError(
                "Stage 7A request must set scientific_data=false"
            )

        if value["production_eligible"] is not False:
            raise Stage7ContractError(
                "Stage 7A request must set production_eligible=false"
            )

        if value["production_default"] is not False:
            raise Stage7ContractError(
                "Stage 7A request may not define a production default"
            )

        teacher = BoundReference.from_mapping(
            value["teacher_reference"]
        )

        if teacher.source_kind not in {
            "injected_fixture",
            "registered_reference",
        }:
            raise Stage7ContractError(
                "teacher_reference must be injected or metadata-only registered"
            )

        hard = BoundReference.from_mapping(
            value["hard_profile_reference"]
        )
        soft = BoundReference.from_mapping(
            value["soft_profile_reference"]
        )

        if hard.reference_id == soft.reference_id:
            raise Stage7ContractError(
                "hard and soft technical profile identities must remain distinct"
            )

        discovery_raw = value[
            "discovery_profile_references"
        ]

        if (
            not isinstance(discovery_raw, list)
            or not discovery_raw
        ):
            raise Stage7ContractError(
                "at least one discovery profile reference is required"
            )

        discovery = tuple(
            BoundReference.from_mapping(item)
            for item in discovery_raw
        )

        discovery_ids = [
            item.reference_id
            for item in discovery
        ]

        if len(set(discovery_ids)) != len(discovery_ids):
            raise Stage7ContractError(
                "discovery profile reference IDs must be unique"
            )

        endpoint1 = BoundReference.from_mapping(
            value["endpoint1_policy_reference"]
        )
        endpoint2 = BoundReference.from_mapping(
            value["endpoint2_policy_reference"]
        )
        analysis = BoundReference.from_mapping(
            value["analysis_profile_reference"]
        )
        job_lifecycle = BoundReference.from_mapping(
            value["job_lifecycle_reference"]
        )

        seed_reference = value[
            "seed_derivation_reference"
        ]

        if seed_reference != SEED_DERIVATION_VERSION:
            raise Stage7ContractError(
                "Stage 7A must reuse the accepted Stage 4 seed derivation"
            )

        seed_raw = value["seed_inputs"]

        if not isinstance(seed_raw, list) or not seed_raw:
            raise Stage7ContractError(
                "Stage 7A requires explicit deterministic seed inputs"
            )

        seed_inputs = tuple(
            SeedInputBinding.from_mapping(item)
            for item in seed_raw
        )
        labels = [
            item.label
            for item in seed_inputs
        ]

        if len(set(labels)) != len(labels):
            raise Stage7ContractError(
                "seed input labels must be unique"
            )

        exclusion = BoundReference.from_mapping(
            value["exclusion_register_reference"]
        )

        if (
            exclusion.source_kind != "repository_file"
            or exclusion.source != EXCLUSION_REGISTER_PATH
        ):
            raise Stage7ContractError(
                "Stage 7A must bind the canonical "
                "excluded-development register"
            )

        resolves = value["resolves_decisions"]

        if not isinstance(resolves, list):
            raise Stage7ContractError(
                "resolves_decisions must be an explicit list"
            )

        if resolves:
            raise Stage7ContractError(
                "Stage 7A may not resolve any unresolved decisions"
            )

        return cls(
            schema_version=RUN_REQUEST_SCHEMA_VERSION,
            request_id=request_id,
            classification=TECHNICAL_CLASSIFICATION,
            scientific_data=False,
            production_eligible=False,
            production_default=False,
            teacher_reference=teacher,
            hard_profile_reference=hard,
            soft_profile_reference=soft,
            discovery_profile_references=discovery,
            endpoint1_policy_reference=endpoint1,
            endpoint2_policy_reference=endpoint2,
            analysis_profile_reference=analysis,
            job_lifecycle_reference=job_lifecycle,
            seed_derivation_reference=SEED_DERIVATION_VERSION,
            seed_inputs=seed_inputs,
            output_root=_validate_output_root(
                value["output_root"]
            ),
            exclusion_register_reference=exclusion,
            decision_dependencies=_decision_dependencies(
                value["decision_dependencies"]
            ),
            resolves_decisions=(),
        )

    def references_by_role(
        self,
    ) -> tuple[tuple[str, BoundReference], ...]:
        discovery = tuple(
            (
                f"discovery_profile[{index}]",
                reference,
            )
            for index, reference in enumerate(
                self.discovery_profile_references
            )
        )

        return (
            ("teacher", self.teacher_reference),
            ("hard_profile", self.hard_profile_reference),
            ("soft_profile", self.soft_profile_reference),
            *discovery,
            (
                "endpoint1_policy",
                self.endpoint1_policy_reference,
            ),
            (
                "endpoint2_policy",
                self.endpoint2_policy_reference,
            ),
            (
                "analysis_profile",
                self.analysis_profile_reference,
            ),
            (
                "job_lifecycle",
                self.job_lifecycle_reference,
            ),
            (
                "exclusion_register",
                self.exclusion_register_reference,
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "classification": self.classification,
            "scientific_data": self.scientific_data,
            "production_eligible": self.production_eligible,
            "production_default": self.production_default,
            "teacher_reference": (
                self.teacher_reference.to_mapping()
            ),
            "hard_profile_reference": (
                self.hard_profile_reference.to_mapping()
            ),
            "soft_profile_reference": (
                self.soft_profile_reference.to_mapping()
            ),
            "discovery_profile_references": [
                item.to_mapping()
                for item in self.discovery_profile_references
            ],
            "endpoint1_policy_reference": (
                self.endpoint1_policy_reference.to_mapping()
            ),
            "endpoint2_policy_reference": (
                self.endpoint2_policy_reference.to_mapping()
            ),
            "analysis_profile_reference": (
                self.analysis_profile_reference.to_mapping()
            ),
            "job_lifecycle_reference": (
                self.job_lifecycle_reference.to_mapping()
            ),
            "seed_derivation_reference": (
                self.seed_derivation_reference
            ),
            "seed_inputs": [
                item.to_mapping()
                for item in self.seed_inputs
            ],
            "output_root": self.output_root,
            "exclusion_register_reference": (
                self.exclusion_register_reference.to_mapping()
            ),
            "decision_dependencies": list(
                self.decision_dependencies
            ),
            "resolves_decisions": list(
                self.resolves_decisions
            ),
        }

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                self.to_mapping()
            )
        ).hexdigest()

    @property
    def run_identity(self) -> str:
        return (
            "stage7-technical-run/v1:"
            f"{self.request_sha256}"
        )


@dataclass(frozen=True)
class RepositoryReferenceEvidence:
    """Integrity evidence for one repository-bound request reference."""

    role: str
    reference_id: str
    source: str
    observed_sha256: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "role": self.role,
            "reference_id": self.reference_id,
            "source": self.source,
            "observed_sha256": self.observed_sha256,
        }


def verify_repository_references(
    request: Stage7TechnicalRunRequest,
    *,
    repository_root: str | Path,
) -> tuple[RepositoryReferenceEvidence, ...]:
    """Verify repository-file bindings without accessing private artifacts."""
    root = Path(
        repository_root
    ).resolve(strict=True)

    evidence: list[
        RepositoryReferenceEvidence
    ] = []

    for role, reference in request.references_by_role():
        if reference.source_kind != "repository_file":
            continue

        candidate = (
            root / reference.source
        ).resolve(strict=True)

        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise Stage7ContractError(
                "repository reference escapes checkout: "
                f"{reference.source}"
            ) from exc

        if not candidate.is_file():
            raise Stage7ContractError(
                "repository reference is not a file: "
                f"{reference.source}"
            )

        observed = hashlib.sha256(
            candidate.read_bytes()
        ).hexdigest()

        if observed != reference.sha256:
            raise Stage7ContractError(
                "repository reference digest mismatch: "
                f"role={role!r} "
                f"source={reference.source!r}"
            )

        evidence.append(
            RepositoryReferenceEvidence(
                role=role,
                reference_id=reference.reference_id,
                source=reference.source,
                observed_sha256=observed,
            )
        )

    return tuple(evidence)


def _seed_evidence_mapping(
    *,
    binding: SeedInputBinding,
    evidence: SeedEvidence,
) -> dict[str, object]:
    return {
        "label": binding.label,
        "condition_id": binding.condition_id,
        "purpose": binding.purpose,
        "attempt_index": binding.attempt_index,
        "retry_index": binding.retry_index,
        "seed_derivation_version": (
            evidence.seed_derivation_version
        ),
        "seed_material": evidence.seed_material,
        "digest_sha256": evidence.digest_sha256,
        "selected_bytes_hex": (
            evidence.selected_bytes_hex
        ),
        "seed_value": evidence.seed_value,
    }


@dataclass(frozen=True)
class Stage7TechnicalRunManifest:
    """Derived immutable technical manifest for one Stage 7A request."""

    request: Stage7TechnicalRunRequest
    lifecycle: Stage7Lifecycle
    seed_evidence: tuple[
        Mapping[str, object],
        ...,
    ]
    repository_reference_evidence: tuple[
        RepositoryReferenceEvidence,
        ...,
    ]

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_identity": self.request.run_identity,
            "request_sha256": self.request.request_sha256,
            "classification": TECHNICAL_CLASSIFICATION,
            "scientific_data": False,
            "production_eligible": False,
            "production_default": False,
            "excluded_development": True,
            "resolves_decisions": [],
            "request": self.request.to_mapping(),
            "seed_evidence": [
                dict(item)
                for item in self.seed_evidence
            ],
            "lifecycle": self.lifecycle.to_mapping(),
            "repository_reference_evidence": [
                item.to_mapping()
                for item in self.repository_reference_evidence
            ],
        }

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                self.to_mapping()
            )
        ).hexdigest()


def build_technical_run_manifest(
    request: Stage7TechnicalRunRequest,
    *,
    stage3: Stage3AvailabilityIndex,
    repository_root: str | Path | None = None,
) -> Stage7TechnicalRunManifest:
    """Derive seeds, lifecycle, and reference evidence deterministically."""
    seed_evidence = tuple(
        _seed_evidence_mapping(
            binding=binding,
            evidence=derive_seed(
                binding.to_seed_inputs(),
                stage3,
            ),
        )
        for binding in request.seed_inputs
    )

    lifecycle = build_stage7_lifecycle(
        output_root=request.output_root,
        job_lifecycle_reference_id=(
            request.job_lifecycle_reference.reference_id
        ),
    )

    reference_evidence = (
        verify_repository_references(
            request,
            repository_root=repository_root,
        )
        if repository_root is not None
        else ()
    )

    return Stage7TechnicalRunManifest(
        request=request,
        lifecycle=lifecycle,
        seed_evidence=seed_evidence,
        repository_reference_evidence=(
            reference_evidence
        ),
    )


def load_technical_run_request(
    source: str | Path,
) -> Stage7TechnicalRunRequest:
    """Load and validate one explicit Stage 7A technical request."""
    source_path = Path(source)

    try:
        raw = json.loads(
            source_path.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage7ContractError(
            "unable to load Stage 7A request: "
            f"{source_path}"
        ) from exc

    return Stage7TechnicalRunRequest.from_mapping(
        raw
    )
