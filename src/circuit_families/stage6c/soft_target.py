"""Technical-only Stage 6C soft-target policy and loss boundary.

The policy and adapters in this module exist only for deterministic synthetic
validation. They select no production soft target, temperature, normalization,
loss, tolerance, argmax requirement, optimizer, schedule, or stopping rule and
do not resolve UD-006.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from circuit_families.interpretability.centred_logit_fidelity import (
    centre_logits_across_classes,
)
from circuit_families.stage4_condition_identity import (
    VERSION_REFERENCE_RE,
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    parse_condition_id,
)
from circuit_families.stage5bc.student_trainer import (
    PreparedTargets,
    SoftTargetAdapter,
)

TECHNICAL_SOFT_POLICY_SCHEMA_VERSION = "stage6c-technical-soft-policy/v1"
TECHNICAL_POLICY_STATUS = "technical_candidate_only"
CENTRING_REF = "per-input-class-mean-centering/v1"
SOFT_LOSS_KIND = "centred_logit_mse"
SOFT_LOSS_REDUCTIONS = ("mean", "sum")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SoftTargetPolicyError(ValueError):
    """Raised when technical policy or cache binding is inconsistent."""


class SoftTargetLossError(ValueError):
    """Raised when the gauge-invariant soft loss inputs are invalid."""


def _require_version_ref(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not VERSION_REFERENCE_RE.fullmatch(value):
        raise SoftTargetPolicyError(
            f"{name} must match the Stage 4 version-reference grammar"
        )
    return value


@dataclass(frozen=True)
class SoftRepresentationMetadata:
    """Injected representation, canonical identity, and ordering evidence."""

    representation_ref: str
    cache_kind: str
    centering_ref: str
    teacher_condition_id: str
    ordering_ref: str
    ordered_input_ids_sha256: str
    temperature_candidate: float | None
    normalization_candidate_ref: str | None

    def __post_init__(self) -> None:
        _require_version_ref(self.representation_ref, name="representation_ref")
        if self.cache_kind != "teacher_logits":
            raise SoftTargetPolicyError(
                "soft representation cache_kind must be teacher_logits"
            )
        if self.centering_ref != CENTRING_REF:
            raise SoftTargetPolicyError(
                f"centering_ref must be the technical candidate {CENTRING_REF!r}"
            )
        if not isinstance(self.teacher_condition_id, str) or not self.teacher_condition_id:
            raise SoftTargetPolicyError("teacher_condition_id must be non-empty")
        _require_version_ref(self.ordering_ref, name="ordering_ref")
        if (
            not isinstance(self.ordered_input_ids_sha256, str)
            or not _SHA256_RE.fullmatch(self.ordered_input_ids_sha256)
        ):
            raise SoftTargetPolicyError(
                "ordered_input_ids_sha256 must be lowercase SHA-256 hex"
            )
        if self.temperature_candidate is not None and (
            isinstance(self.temperature_candidate, bool)
            or not isinstance(self.temperature_candidate, (int, float))
            or not math.isfinite(float(self.temperature_candidate))
            or float(self.temperature_candidate) <= 0.0
        ):
            raise SoftTargetPolicyError(
                "temperature_candidate must be null or a positive finite number"
            )
        if self.normalization_candidate_ref is not None:
            _require_version_ref(
                self.normalization_candidate_ref,
                name="normalization_candidate_ref",
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cache_kind": self.cache_kind,
            "centering_ref": self.centering_ref,
            "normalization_candidate_ref": self.normalization_candidate_ref,
            "ordered_input_ids_sha256": self.ordered_input_ids_sha256,
            "ordering_ref": self.ordering_ref,
            "representation_ref": self.representation_ref,
            "teacher_condition_id": self.teacher_condition_id,
            "temperature_candidate": self.temperature_candidate,
        }


@dataclass(frozen=True)
class TechnicalToleranceMetadata:
    """Nonbinding tolerance candidate retained for later technical comparison."""

    metric_ref: str
    comparison: str
    candidate_value: float
    status: str

    def __post_init__(self) -> None:
        _require_version_ref(self.metric_ref, name="tolerance.metric_ref")
        if self.comparison != "less_than_or_equal":
            raise SoftTargetPolicyError(
                "technical tolerance comparison must be less_than_or_equal"
            )
        if (
            isinstance(self.candidate_value, bool)
            or not isinstance(self.candidate_value, (int, float))
            or not math.isfinite(float(self.candidate_value))
            or float(self.candidate_value) < 0.0
        ):
            raise SoftTargetPolicyError(
                "technical tolerance candidate must be finite and non-negative"
            )
        if self.status != TECHNICAL_POLICY_STATUS:
            raise SoftTargetPolicyError(
                f"tolerance status must be {TECHNICAL_POLICY_STATUS!r}"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_value": float(self.candidate_value),
            "comparison": self.comparison,
            "metric_ref": self.metric_ref,
            "status": self.status,
        }


@dataclass(frozen=True)
class TechnicalArgmaxRequirementMetadata:
    """Optional nonbinding argmax candidate; absence keeps the candidate unset."""

    requirement_ref: str
    candidate_required: bool
    status: str

    def __post_init__(self) -> None:
        _require_version_ref(self.requirement_ref, name="argmax.requirement_ref")
        if not isinstance(self.candidate_required, bool):
            raise SoftTargetPolicyError(
                "argmax candidate_required must be an explicit boolean"
            )
        if self.status != TECHNICAL_POLICY_STATUS:
            raise SoftTargetPolicyError(
                f"argmax status must be {TECHNICAL_POLICY_STATUS!r}"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_required": self.candidate_required,
            "requirement_ref": self.requirement_ref,
            "status": self.status,
        }


@dataclass(frozen=True)
class TechnicalSoftPolicy:
    """Explicit versioned soft policy with no production authority."""

    schema_version: str
    policy_ref: str
    status: str
    scientific_data: bool
    production_eligible: bool
    resolves_ud006: bool
    representation: SoftRepresentationMetadata
    tolerance: TechnicalToleranceMetadata
    argmax_requirement: TechnicalArgmaxRequirementMetadata | None

    def __post_init__(self) -> None:
        if self.schema_version != TECHNICAL_SOFT_POLICY_SCHEMA_VERSION:
            raise SoftTargetPolicyError(
                f"schema_version must be {TECHNICAL_SOFT_POLICY_SCHEMA_VERSION!r}"
            )
        _require_version_ref(self.policy_ref, name="policy_ref")
        if not self.policy_ref.startswith("technical-"):
            raise SoftTargetPolicyError("policy_ref must be explicitly technical")
        if self.status != TECHNICAL_POLICY_STATUS:
            raise SoftTargetPolicyError(
                f"policy status must be {TECHNICAL_POLICY_STATUS!r}"
            )
        if self.scientific_data is not False:
            raise SoftTargetPolicyError("technical policy requires scientific_data=false")
        if self.production_eligible is not False:
            raise SoftTargetPolicyError(
                "technical policy requires production_eligible=false"
            )
        if self.resolves_ud006 is not False:
            raise SoftTargetPolicyError("technical policy must not resolve UD-006")
        if not isinstance(self.representation, SoftRepresentationMetadata):
            raise SoftTargetPolicyError(
                "representation must be SoftRepresentationMetadata"
            )
        if not isinstance(self.tolerance, TechnicalToleranceMetadata):
            raise SoftTargetPolicyError(
                "tolerance must be TechnicalToleranceMetadata"
            )
        if self.argmax_requirement is not None and not isinstance(
            self.argmax_requirement,
            TechnicalArgmaxRequirementMetadata,
        ):
            raise SoftTargetPolicyError(
                "argmax_requirement must be null or technical metadata"
            )

    def to_mapping(self) -> dict[str, Any]:
        record = {
            "argmax_requirement": (
                None
                if self.argmax_requirement is None
                else self.argmax_requirement.to_mapping()
            ),
            "policy_ref": self.policy_ref,
            "production_eligible": self.production_eligible,
            "representation": self.representation.to_mapping(),
            "resolves_ud006": self.resolves_ud006,
            "schema_version": self.schema_version,
            "scientific_data": self.scientific_data,
            "status": self.status,
            "tolerance": self.tolerance.to_mapping(),
        }
        json.dumps(record, allow_nan=False, sort_keys=True)
        return record


def _centre_for_soft_loss(value: Any, *, name: str) -> torch.Tensor:
    try:
        return centre_logits_across_classes(value)
    except (FloatingPointError, TypeError, ValueError) as exc:
        raise SoftTargetLossError(f"invalid {name}: {exc}") from exc


class TechnicalSoftTargetAdapter:
    """Bind the accepted soft adapter to one injected policy and cache identity."""

    cache_kind = "teacher_logits"

    def __init__(
        self,
        *,
        policy: TechnicalSoftPolicy,
        stage3: Stage3AvailabilityIndex,
    ) -> None:
        if not isinstance(policy, TechnicalSoftPolicy):
            raise SoftTargetPolicyError("policy must be TechnicalSoftPolicy")
        if not isinstance(stage3, Stage3AvailabilityIndex):
            raise SoftTargetPolicyError("stage3 must be Stage3AvailabilityIndex")
        try:
            identity = parse_condition_id(
                policy.representation.teacher_condition_id,
                stage3,
            )
        except ConditionIdentityError as exc:
            raise SoftTargetPolicyError(
                f"invalid soft teacher condition identity: {exc}"
            ) from exc
        if identity.depth != 3 or identity.distillation_condition != "soft_target":
            raise SoftTargetPolicyError(
                "soft policy requires a depth-3 soft_target teacher identity"
            )
        self.policy = policy
        self.stage3 = stage3
        self._accepted_adapter = SoftTargetAdapter()

    def __call__(self, cache: Any) -> PreparedTargets:
        manifest_object = getattr(cache, "manifest", None)
        to_mapping = getattr(manifest_object, "to_mapping", None)
        if to_mapping is None or not callable(to_mapping):
            raise SoftTargetPolicyError(
                "soft cache must expose accepted manifest metadata"
            )
        manifest = to_mapping()
        if not isinstance(manifest, Mapping):
            raise SoftTargetPolicyError("soft cache manifest must be a mapping")
        input_order = manifest.get("input_order")
        teacher_reference = manifest.get("teacher_reference")
        if not isinstance(input_order, Mapping) or not isinstance(
            teacher_reference,
            Mapping,
        ):
            raise SoftTargetPolicyError(
                "soft cache manifest lacks identity or ordering metadata"
            )
        representation = self.policy.representation
        if teacher_reference.get("condition_id") != representation.teacher_condition_id:
            raise SoftTargetPolicyError("soft cache teacher identity mismatch")
        if (
            input_order.get("ordering_ref") != representation.ordering_ref
            or input_order.get("ordered_input_ids_sha256")
            != representation.ordered_input_ids_sha256
        ):
            raise SoftTargetPolicyError("soft cache canonical ordering mismatch")
        try:
            prepared = self._accepted_adapter(cache)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SoftTargetPolicyError(
                f"accepted soft target adapter rejected cache: {exc}"
            ) from exc
        centred = _centre_for_soft_loss(prepared.values, name="teacher logits")
        return PreparedTargets(cache_kind=self.cache_kind, values=centred)


class GaugeInvariantSoftLossAdapter:
    """Per-input class-centred MSE through the accepted loss protocol."""

    required_target_cache_kind = "teacher_logits"

    def __init__(self, *, policy: TechnicalSoftPolicy) -> None:
        if not isinstance(policy, TechnicalSoftPolicy):
            raise SoftTargetLossError("policy must be TechnicalSoftPolicy")
        self.policy = policy

    def __call__(
        self,
        *,
        outputs: torch.Tensor,
        targets: PreparedTargets,
        settings: Mapping[str, Any],
    ) -> torch.Tensor:
        if not isinstance(targets, PreparedTargets):
            raise SoftTargetLossError("targets must be PreparedTargets")
        if targets.cache_kind != self.required_target_cache_kind:
            raise SoftTargetLossError(
                "soft loss requires teacher_logits targets; hard targets are forbidden"
            )
        if not isinstance(settings, Mapping):
            raise SoftTargetLossError("soft loss settings must be a mapping")
        if set(settings) != {"loss_kind", "policy", "reduction"}:
            raise SoftTargetLossError(
                "soft loss settings require exactly loss_kind, policy, and reduction"
            )
        if settings["policy"] != self.policy:
            raise SoftTargetLossError("soft loss policy binding mismatch")
        if settings["loss_kind"] != SOFT_LOSS_KIND:
            raise SoftTargetLossError(
                f"loss_kind must be the injected candidate {SOFT_LOSS_KIND!r}"
            )
        reduction = settings["reduction"]
        if reduction not in SOFT_LOSS_REDUCTIONS:
            raise SoftTargetLossError(
                f"soft loss reduction must be one of {SOFT_LOSS_REDUCTIONS!r}"
            )

        student_centred = _centre_for_soft_loss(outputs, name="student logits")
        teacher_centred = _centre_for_soft_loss(
            targets.values,
            name="teacher logits",
        )
        if student_centred.shape != teacher_centred.shape:
            raise SoftTargetLossError(
                "student and teacher logits must have identical shapes"
            )
        squared_error = (student_centred - teacher_centred).square()
        if reduction == "mean":
            return squared_error.mean()
        return squared_error.sum()
