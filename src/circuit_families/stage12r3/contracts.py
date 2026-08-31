"""Stage 12-R3 technical calibration contracts.

This namespace defines calibration metadata only.  It deliberately reuses:
- Stage 6A for exact qualification / Endpoint 1;
- Stage 6E for overlap, packing, proof, and Endpoint 2;
- Stage 12-R1 discovery-family semantics;
- Stage 12-R2 basis/component identities.

No production policy or scientific-data decision is made here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from circuit_families.stage12r1 import (
    ALGORITHM_FAMILY as STAGE12R1_ALGORITHM_FAMILY,
)
from circuit_families.stage12r1 import (
    UNRESOLVED_PRODUCTION_DECISIONS,
)
from circuit_families.stage12r2.contracts import (
    canonical_sha256,
    validate_technical_record_payload,
)

CONTRACT_VERSION = "stage12r3-calibration-contract/v1"

LayerKind = Literal[
    "combinatorial_floor",
    "ordinary_restart_baseline",
    "local_exact_perturbation",
    "tractable_feasible_region",
]

QualificationSource = Literal[
    "none",
    "stage6a_exact_common_ledger",
]

DiscoveryRelationship = Literal[
    "not_applicable",
    "same_discovery_family_ordinary_restart",
]

ExactnessClaim = Literal[
    "not_applicable",
    "exact",
    "certified_near_exact",
]

EXPECTED_LAYER_KINDS: tuple[LayerKind, ...] = (
    "combinatorial_floor",
    "ordinary_restart_baseline",
    "local_exact_perturbation",
    "tractable_feasible_region",
)

REQUIRED_OPEN_DECISIONS = ("RD-006", "RD-008", "RD-009")

CLAIM_BOUNDARIES: dict[LayerKind, str] = {
    "combinatorial_floor": (
        "expected_overlap_under_declared_size_type_matching_only"
    ),
    "ordinary_restart_baseline": (
        "procedure_relative_same_family_restart_baseline_only"
    ),
    "local_exact_perturbation": (
        "local_exact_fidelity_connectivity_only"
    ),
    "tractable_feasible_region": (
        "technical_fixture_search_gap_only_no_main_scale_transfer"
    ),
}


class Stage12R3ContractError(ValueError):
    """Raised when a Stage 12-R3 technical contract violates its boundary."""


@dataclass(frozen=True)
class NativeWorkBudget:
    """Method-native sampling/search work, separate from exact evaluation."""

    unit: str
    allowance: int | None

    def __post_init__(self) -> None:
        if not self.unit:
            raise Stage12R3ContractError("native work unit must be non-empty")
        if self.allowance is not None and self.allowance < 0:
            raise Stage12R3ContractError(
                "native work allowance must be non-negative"
            )


@dataclass(frozen=True)
class ExactEvaluationBudget:
    """Stage 6A exact-evaluation allowance."""

    allowance: int | None

    def __post_init__(self) -> None:
        if self.allowance is not None and self.allowance < 0:
            raise Stage12R3ContractError(
                "exact evaluation allowance must be non-negative"
            )


@dataclass(frozen=True)
class CompletenessCertificate:
    """Completeness claim for a deliberately tractable technical fixture."""

    exactness_claim: ExactnessClaim
    exhaustive: bool
    lower_bound: int | None
    upper_bound: int | None
    gap: int | None
    certificate_reference: str

    def __post_init__(self) -> None:
        if self.exactness_claim == "not_applicable":
            raise Stage12R3ContractError(
                "tractable certificate cannot be not_applicable"
            )
        if not self.certificate_reference:
            raise Stage12R3ContractError(
                "certificate reference must be non-empty"
            )

        bounds = (self.lower_bound, self.upper_bound)
        if any(value is not None and value < 0 for value in bounds):
            raise Stage12R3ContractError(
                "certificate bounds must be non-negative"
            )

        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise Stage12R3ContractError(
                "certificate lower bound exceeds upper bound"
            )

        if self.exactness_claim == "exact":
            if not self.exhaustive:
                raise Stage12R3ContractError(
                    "exact claim requires exhaustive certification"
                )
            if self.gap not in (0, None):
                raise Stage12R3ContractError(
                    "exact claim cannot have a nonzero gap"
                )
            if (
                self.lower_bound is not None
                and self.upper_bound is not None
                and self.lower_bound != self.upper_bound
            ):
                raise Stage12R3ContractError(
                    "exact claim requires equal certified bounds"
                )

        if self.exactness_claim == "certified_near_exact":
            if self.lower_bound is None or self.upper_bound is None:
                raise Stage12R3ContractError(
                    "near-exact claim requires explicit lower/upper bounds"
                )
            expected_gap = self.upper_bound - self.lower_bound
            if self.gap != expected_gap:
                raise Stage12R3ContractError(
                    "near-exact certificate gap does not match bounds"
                )
            if self.gap == 0:
                raise Stage12R3ContractError(
                    "zero-gap certificate must not be labelled near-exact"
                )


@dataclass(frozen=True)
class CalibrationLayerContract:
    """One semantically distinct Stage 12-R3 calibration layer."""

    layer_id: str
    layer_kind: LayerKind
    basis_hash: str
    ordered_component_ids: tuple[str, ...]
    component_types: tuple[str, ...]
    native_budget: NativeWorkBudget
    exact_budget: ExactEvaluationBudget
    termination_semantics: str
    coverage_semantics: str
    claim_boundary: str

    qualification_source: QualificationSource
    exact_evaluation_boundary: str = "stage6a_exact_evaluation_bridge"
    endpoint_boundary: str = "shared_stage6a_stage6e_reducers"

    discovery_family: str | None = None
    discovery_relationship: DiscoveryRelationship = "not_applicable"
    uses_diversity_pressure: bool = False
    uses_packing_feedback: bool = False
    uses_prior_restart_mask_exclusion: bool = False

    completeness_certificate: CompletenessCertificate | None = None

    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise Stage12R3ContractError("layer_id must be non-empty")
        if not self.basis_hash:
            raise Stage12R3ContractError("basis_hash must be non-empty")
        if not self.ordered_component_ids:
            raise Stage12R3ContractError(
                "ordered component universe must be non-empty"
            )
        if len(self.ordered_component_ids) != len(self.component_types):
            raise Stage12R3ContractError(
                "component IDs and component types must have equal length"
            )
        if len(set(self.ordered_component_ids)) != len(
            self.ordered_component_ids
        ):
            raise Stage12R3ContractError(
                "ordered component IDs must be unique"
            )
        if any(not value for value in self.component_types):
            raise Stage12R3ContractError(
                "component types must be non-empty"
            )

        if self.scientific_data:
            raise Stage12R3ContractError(
                "Stage 12-R3 executable contracts require scientific_data=false"
            )
        if self.production_eligible:
            raise Stage12R3ContractError(
                "Stage 12-R3 executable contracts require "
                "production_eligible=false"
            )

        expected_claim = CLAIM_BOUNDARIES[self.layer_kind]
        if self.claim_boundary != expected_claim:
            raise Stage12R3ContractError(
                f"invalid claim boundary for {self.layer_kind}"
            )

        if self.exact_evaluation_boundary != "stage6a_exact_evaluation_bridge":
            raise Stage12R3ContractError(
                "exact evaluation must remain behind the Stage 6A bridge"
            )
        if self.endpoint_boundary != "shared_stage6a_stage6e_reducers":
            raise Stage12R3ContractError(
                "Endpoint reducers must remain shared Stage 6A/6E reducers"
            )

        if self.layer_kind == "combinatorial_floor":
            if self.qualification_source != "none":
                raise Stage12R3ContractError(
                    "combinatorial floor must not make a fidelity qualification"
                )
            if self.discovery_family is not None:
                raise Stage12R3ContractError(
                    "combinatorial floor is not a discovery family"
                )
            if self.completeness_certificate is not None:
                raise Stage12R3ContractError(
                    "combinatorial floor does not carry tractable exactness"
                )

        elif self.layer_kind == "ordinary_restart_baseline":
            if self.qualification_source != "stage6a_exact_common_ledger":
                raise Stage12R3ContractError(
                    "ordinary restarts require Stage 6A exact qualification"
                )
            if (
                self.discovery_relationship
                != "same_discovery_family_ordinary_restart"
            ):
                raise Stage12R3ContractError(
                    "ordinary restart cannot be relabelled as an "
                    "independent discovery method"
                )
            if not self.discovery_family:
                raise Stage12R3ContractError(
                    "ordinary restart requires its discovery family identity"
                )
            if (
                self.uses_diversity_pressure
                or self.uses_packing_feedback
                or self.uses_prior_restart_mask_exclusion
            ):
                raise Stage12R3ContractError(
                    "ordinary restart baseline cannot use cross-restart "
                    "diversity or packing-aware feedback"
                )
            if self.completeness_certificate is not None:
                raise Stage12R3ContractError(
                    "ordinary restart baseline is not an exactness certificate"
                )

        elif self.layer_kind == "local_exact_perturbation":
            if self.qualification_source != "stage6a_exact_common_ledger":
                raise Stage12R3ContractError(
                    "local perturbations require fresh exact common-ledger "
                    "qualification; surrogate or inherited fidelity is invalid"
                )
            if self.discovery_family is not None:
                raise Stage12R3ContractError(
                    "local perturbation is not an independent discovery family"
                )
            if self.completeness_certificate is not None:
                raise Stage12R3ContractError(
                    "local perturbation does not certify global exactness"
                )

        elif self.layer_kind == "tractable_feasible_region":
            if self.qualification_source != "stage6a_exact_common_ledger":
                raise Stage12R3ContractError(
                    "tractable feasible region requires common exact evaluation"
                )
            if self.completeness_certificate is None:
                raise Stage12R3ContractError(
                    "tractable exact/near-exact claim requires a certificate"
                )
            if self.discovery_family is not None:
                raise Stage12R3ContractError(
                    "feasible-region certification is not a discovery family"
                )

        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class Stage12R3CalibrationContract:
    """Versioned outer contract containing all four calibration layers."""

    profile_id: str
    basis_hash: str
    ordered_component_ids: tuple[str, ...]
    component_types: tuple[str, ...]
    layers: tuple[CalibrationLayerContract, ...]

    contract_version: str = CONTRACT_VERSION
    unresolved_production_decisions: tuple[str, ...] = (
        UNRESOLVED_PRODUCTION_DECISIONS
    )
    scientific_data: bool = False
    production_eligible: bool = False
    production_packing_policy_selected: bool = False
    tractable_calibration_production_choice: bool = False

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise Stage12R3ContractError(
                "unsupported Stage 12-R3 contract version"
            )
        if not self.profile_id:
            raise Stage12R3ContractError("profile_id must be non-empty")
        if self.scientific_data or self.production_eligible:
            raise Stage12R3ContractError(
                "Stage 12-R3 outer contract is technical-only"
            )
        if (
            self.production_packing_policy_selected
            or self.tractable_calibration_production_choice
        ):
            raise Stage12R3ContractError(
                "Stage 12-R3 cannot resolve production packing/calibration policy"
            )

        missing_decisions = set(REQUIRED_OPEN_DECISIONS) - set(
            self.unresolved_production_decisions
        )
        if missing_decisions:
            raise Stage12R3ContractError(
                "required production decisions remain open: "
                + ", ".join(sorted(missing_decisions))
            )

        if len(self.layers) != len(EXPECTED_LAYER_KINDS):
            raise Stage12R3ContractError(
                "outer contract must contain exactly four calibration layers"
            )
        if tuple(layer.layer_kind for layer in self.layers) != EXPECTED_LAYER_KINDS:
            raise Stage12R3ContractError(
                "four calibration layers must remain distinct and ordered"
            )
        if len({layer.layer_id for layer in self.layers}) != len(self.layers):
            raise Stage12R3ContractError(
                "each calibration layer requires a distinct layer identity"
            )

        for layer in self.layers:
            if layer.basis_hash != self.basis_hash:
                raise Stage12R3ContractError(
                    "calibration layers cannot mix basis identities"
                )
            if layer.ordered_component_ids != self.ordered_component_ids:
                raise Stage12R3ContractError(
                    "calibration layers cannot mix component universes/order"
                )
            if layer.component_types != self.component_types:
                raise Stage12R3ContractError(
                    "calibration layers cannot mix component-type identities"
                )

        validate_technical_record_payload(asdict(self))

    @property
    def identity(self) -> str:
        return canonical_sha256(asdict(self))


def _layer_id(
    *,
    profile_id: str,
    basis_hash: str,
    layer_kind: LayerKind,
) -> str:
    return canonical_sha256(
        {
            "contract_version": CONTRACT_VERSION,
            "profile_id": profile_id,
            "basis_hash": basis_hash,
            "layer_kind": layer_kind,
        }
    )


def build_technical_calibration_contract(
    *,
    profile_id: str,
    basis_hash: str,
    ordered_component_ids: tuple[str, ...],
    component_types: tuple[str, ...],
    discovery_family: str = STAGE12R1_ALGORITHM_FAMILY,
    combinatorial_native_allowance: int | None = None,
    restart_native_allowance: int | None = None,
    restart_exact_allowance: int | None = None,
    local_native_allowance: int | None = None,
    local_exact_allowance: int | None = None,
    tractable_native_allowance: int | None = None,
    tractable_exact_allowance: int | None = None,
    tractable_certificate: CompletenessCertificate | None = None,
) -> Stage12R3CalibrationContract:
    """Build a technical-only four-layer calibration contract.

    Allowances are prospective technical inputs, not selected production policy.
    """

    if tractable_certificate is None:
        tractable_certificate = CompletenessCertificate(
            exactness_claim="exact",
            exhaustive=True,
            lower_bound=None,
            upper_bound=None,
            gap=0,
            certificate_reference="technical_complete_mask_enumeration",
        )

    common = {
        "basis_hash": basis_hash,
        "ordered_component_ids": ordered_component_ids,
        "component_types": component_types,
    }

    layers = (
        CalibrationLayerContract(
            layer_id=_layer_id(
                profile_id=profile_id,
                basis_hash=basis_hash,
                layer_kind="combinatorial_floor",
            ),
            layer_kind="combinatorial_floor",
            native_budget=NativeWorkBudget(
                unit="combinatorial_draw",
                allowance=combinatorial_native_allowance,
            ),
            exact_budget=ExactEvaluationBudget(allowance=0),
            termination_semantics="declared_draw_budget_or_finite_exhaustion",
            coverage_semantics="draw_distribution_coverage_only",
            claim_boundary=CLAIM_BOUNDARIES["combinatorial_floor"],
            qualification_source="none",
            **common,
        ),
        CalibrationLayerContract(
            layer_id=_layer_id(
                profile_id=profile_id,
                basis_hash=basis_hash,
                layer_kind="ordinary_restart_baseline",
            ),
            layer_kind="ordinary_restart_baseline",
            native_budget=NativeWorkBudget(
                unit="optimizer_step",
                allowance=restart_native_allowance,
            ),
            exact_budget=ExactEvaluationBudget(
                allowance=restart_exact_allowance
            ),
            termination_semantics="per_restart_native_or_exact_budget_termination",
            coverage_semantics="procedure_relative_recovered_unique_exact_masks",
            claim_boundary=CLAIM_BOUNDARIES["ordinary_restart_baseline"],
            qualification_source="stage6a_exact_common_ledger",
            discovery_family=discovery_family,
            discovery_relationship="same_discovery_family_ordinary_restart",
            **common,
        ),
        CalibrationLayerContract(
            layer_id=_layer_id(
                profile_id=profile_id,
                basis_hash=basis_hash,
                layer_kind="local_exact_perturbation",
            ),
            layer_kind="local_exact_perturbation",
            native_budget=NativeWorkBudget(
                unit="neighborhood_proposal",
                allowance=local_native_allowance,
            ),
            exact_budget=ExactEvaluationBudget(
                allowance=local_exact_allowance
            ),
            termination_semantics="declared_radius_or_native_or_exact_budget",
            coverage_semantics="bounded_local_exact_neighborhood_coverage",
            claim_boundary=CLAIM_BOUNDARIES["local_exact_perturbation"],
            qualification_source="stage6a_exact_common_ledger",
            **common,
        ),
        CalibrationLayerContract(
            layer_id=_layer_id(
                profile_id=profile_id,
                basis_hash=basis_hash,
                layer_kind="tractable_feasible_region",
            ),
            layer_kind="tractable_feasible_region",
            native_budget=NativeWorkBudget(
                unit="mask_universe_state",
                allowance=tractable_native_allowance,
            ),
            exact_budget=ExactEvaluationBudget(
                allowance=tractable_exact_allowance
            ),
            termination_semantics="certificate_defined_completeness",
            coverage_semantics="certified_feasible_region_coverage",
            claim_boundary=CLAIM_BOUNDARIES["tractable_feasible_region"],
            qualification_source="stage6a_exact_common_ledger",
            completeness_certificate=tractable_certificate,
            **common,
        ),
    )

    return Stage12R3CalibrationContract(
        profile_id=profile_id,
        basis_hash=basis_hash,
        ordered_component_ids=ordered_component_ids,
        component_types=component_types,
        layers=layers,
    )
