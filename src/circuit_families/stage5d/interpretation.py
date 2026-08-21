from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .outputs import (
    OUTPUT_BUNDLE_SCHEMA_VERSION,
    OUTPUT_CLASSIFICATION,
)

INTERPRETATION_SCHEMA_VERSION = "stage5d_neutral_interpretation_v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "stage5d_barrier1_boundary_audit_v1"
BARRIER_ID = "Barrier 1"
BARRIER_PURPOSE = (
    "deterministic hierarchy exercise on synthetic fixtures only"
)
INTERPRETATION_STATUS = "label_only_not_assessed"


class Stage5DInterpretationError(ValueError):
    pass


@dataclass(frozen=True, slots=True, order=True)
class NeutralOutcomeLabel:
    frozen_order: int
    interpretation_id: str
    neutral_label: str
    status: str = INTERPRETATION_STATUS


@dataclass(frozen=True, slots=True)
class NeutralInterpretationAudit:
    schema_version: str
    source_output_schema_version: str
    source_output_sha256: str
    source_classification: str
    labels: tuple[NeutralOutcomeLabel, ...]
    directional_predictions: tuple[str, ...] = ()
    automatic_conclusions: tuple[str, ...] = ()
    causal_conclusions: tuple[str, ...] = ()
    scientific_claims: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, order=True)
class UnresolvedDecisionBoundary:
    decision_id: str
    decision_family: str
    owner: str
    lane: str
    resolution_stage: str
    status: str = "unresolved"


@dataclass(frozen=True, slots=True)
class Stage5DBoundaryAudit:
    schema_version: str
    barrier_id: str
    purpose: str
    unresolved_decisions: tuple[UnresolvedDecisionBoundary, ...]
    synthetic_only: bool = True
    real_result_ingestion: bool = False
    stage6_execution: bool = False
    production_training: bool = False
    scientific_analysis: bool = False
    final_scientific_freeze: bool = False


FROZEN_OUTCOME_LABELS = (
    NeutralOutcomeLabel(
        frozen_order=1,
        interpretation_id="frozen_outcome_01",
        neutral_label="teacher_phase_across_student_conditions",
    ),
    NeutralOutcomeLabel(
        frozen_order=2,
        interpretation_id="frozen_outcome_02",
        neutral_label="within_teacher_student_realization_variation",
    ),
    NeutralOutcomeLabel(
        frozen_order=3,
        interpretation_id="frozen_outcome_03",
        neutral_label="cross_student_compressibility",
    ),
    NeutralOutcomeLabel(
        frozen_order=4,
        interpretation_id="frozen_outcome_04",
        neutral_label="method_fidelity_sensitivity",
    ),
    NeutralOutcomeLabel(
        frozen_order=5,
        interpretation_id="frozen_outcome_05",
        neutral_label="function_realization_relationship",
    ),
    NeutralOutcomeLabel(
        frozen_order=6,
        interpretation_id="frozen_outcome_06",
        neutral_label="predictive_fidelity_transition_comparison",
    ),
)


STAGE5D_UNRESOLVED_DECISIONS = (
    UnresolvedDecisionBoundary(
        decision_id="UD-004",
        decision_family=(
            "Student replication, attempt cap, replacement and minimum "
            "eligibility"
        ),
        owner="Austin",
        lane="Lane B",
        resolution_stage="Stage 11",
    ),
    UnresolvedDecisionBoundary(
        decision_id="UD-011",
        decision_family=(
            "Primary phase contrasts, cell summary and missing-cell rule"
        ),
        owner="Alex",
        lane="Lane D",
        resolution_stage="Stage 13",
    ),
    UnresolvedDecisionBoundary(
        decision_id="UD-012",
        decision_family="Required analysis tables, figures and manifests",
        owner="Alex",
        lane="Lane D",
        resolution_stage="Stage 13",
    ),
    UnresolvedDecisionBoundary(
        decision_id="UD-014",
        decision_family="Definitive production scope",
        owner="Joint",
        lane="Joint",
        resolution_stage="Stage 14 / Barrier 3",
    ),
)


def frozen_outcome_labels() -> tuple[NeutralOutcomeLabel, ...]:
    return FROZEN_OUTCOME_LABELS


def _require_synthetic_output_bundle(
    output_bundle: Mapping[str, Any],
) -> str:
    if output_bundle.get("schema_version") != OUTPUT_BUNDLE_SCHEMA_VERSION:
        raise Stage5DInterpretationError(
            "neutral interpretation accepts Stage 5D output bundles only"
        )
    if output_bundle.get("classification") != OUTPUT_CLASSIFICATION:
        raise Stage5DInterpretationError(
            "real or production-result ingestion is forbidden"
        )
    if output_bundle.get("scientific_data") is not False:
        raise Stage5DInterpretationError(
            "scientific data cannot enter the Stage 5D interpretation layer"
        )
    if output_bundle.get("production_eligible") is not False:
        raise Stage5DInterpretationError(
            "production results cannot enter the Stage 5D interpretation layer"
        )

    source_sha256 = output_bundle.get("sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise Stage5DInterpretationError(
            "Stage 5D output bundle requires an explicit SHA-256 identity"
        )
    return source_sha256


def build_neutral_interpretation_audit(
    output_bundle: Mapping[str, Any],
) -> NeutralInterpretationAudit:
    source_sha256 = _require_synthetic_output_bundle(output_bundle)
    return NeutralInterpretationAudit(
        schema_version=INTERPRETATION_SCHEMA_VERSION,
        source_output_schema_version=OUTPUT_BUNDLE_SCHEMA_VERSION,
        source_output_sha256=source_sha256,
        source_classification=OUTPUT_CLASSIFICATION,
        labels=FROZEN_OUTCOME_LABELS,
    )


def stage5d_boundary_audit() -> Stage5DBoundaryAudit:
    return Stage5DBoundaryAudit(
        schema_version=BOUNDARY_AUDIT_SCHEMA_VERSION,
        barrier_id=BARRIER_ID,
        purpose=BARRIER_PURPOSE,
        unresolved_decisions=STAGE5D_UNRESOLVED_DECISIONS,
    )
