"""Technical Stage 7A discovery and endpoint bridges.

This module does not implement a discovery algorithm, exact evaluator, Endpoint
1 reducer, or Endpoint 2 packing algorithm. It wires released Stage 7 technical
subjects through the accepted Stage 6D adapters, reconstructs the adapter's
sealed Stage 6A ledger with unchanged captured fidelity values, and passes that
accepted evidence to the accepted Stage 6A and Stage 6E endpoint reducers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from circuit_families.stage6a import (
    COMPONENT_COUNT,
    TerminationStatus,
    canonical_mask_identity,
    reduce_endpoint1,
)
from circuit_families.stage6d import (
    RESOURCE_WARNING,
    UNRESOLVED_DECISIONS,
    DiscoveryRequest,
    DiversityForcedAdapter,
    GreedyDeletionAdapter,
    Stage6AExactEvaluationBridge,
    deterministic_seed_evidence,
    load_technical_profiles,
)
from circuit_families.stage6e import (
    PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS,
    ExactCandidateEvidence,
    load_technical_policy,
    qualify_and_deduplicate,
    recompute_endpoint2,
)
from circuit_families.stage7.contracts import Stage7TechnicalRunRequest

DISCOVERY_ENDPOINT_SCHEMA_VERSION: Final = (
    "stage7-technical-discovery-endpoints/v1"
)

ACCEPTED_DISCOVERY_ADAPTERS: Final = (
    "greedy_deletion",
    "diversity_forced",
)

RELEASED_SUBJECT_ROLES: Final = (
    "direct_teacher",
    "hard_target_student",
    "soft_target_student",
)

BLOCKED_SUBJECT_ROLES: Final = (
    "failed_hard_target_student",
    "failed_soft_target_student",
)


class Stage7DiscoveryEndpointError(ValueError):
    """Raised when Stage 7A discovery or endpoint gating is violated."""


@dataclass(frozen=True)
class TechnicalDiscoverySubject:
    """One model/source identity presented to the discovery bridge."""

    subject_id: str
    role: str
    source_reference_sha256: str
    eligible: bool
    sealed: bool
    release_allowed: bool
    seed_value: int

    def __post_init__(self) -> None:
        if not isinstance(self.subject_id, str) or not self.subject_id:
            raise Stage7DiscoveryEndpointError(
                "subject_id must be a non-empty string"
            )

        if self.role not in {
            *RELEASED_SUBJECT_ROLES,
            *BLOCKED_SUBJECT_ROLES,
        }:
            raise Stage7DiscoveryEndpointError(
                f"unsupported technical discovery subject role: {self.role!r}"
            )

        digest = self.source_reference_sha256

        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Stage7DiscoveryEndpointError(
                "source_reference_sha256 must be lowercase SHA-256 hex"
            )

        for name in (
            "eligible",
            "sealed",
            "release_allowed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise Stage7DiscoveryEndpointError(
                    f"{name} must be boolean"
                )

        if (
            isinstance(self.seed_value, bool)
            or not isinstance(self.seed_value, int)
            or self.seed_value < 0
        ):
            raise Stage7DiscoveryEndpointError(
                "seed_value must be a non-negative integer"
            )


@dataclass(frozen=True)
class _CapturedExactCall:
    mask: tuple[int, ...]
    fidelity: float


class _TechnicalExactCapture:
    """Capture actual adapter evaluator calls without owning discovery."""

    def __init__(self) -> None:
        self.calls: list[_CapturedExactCall] = []

    def __call__(self, mask: tuple[int, ...]) -> float:
        validated = tuple(mask)

        if len(validated) != COMPONENT_COUNT:
            raise Stage7DiscoveryEndpointError(
                "technical exact fixture mask has wrong component count"
            )

        if any(bit not in (0, 1) for bit in validated):
            raise Stage7DiscoveryEndpointError(
                "technical exact fixture mask must be binary"
            )

        retained = sum(validated)
        fidelity = 0.8 + 0.1 * (retained / COMPONENT_COUNT)

        self.calls.append(
            _CapturedExactCall(
                mask=validated,
                fidelity=fidelity,
            )
        )

        return fidelity


def assert_discovery_releasable(
    subject: TechnicalDiscoverySubject,
) -> None:
    """Fail closed before any Stage 6D adapter can be invoked."""
    if subject.role == "direct_teacher":
        if not subject.sealed or not subject.release_allowed:
            raise Stage7DiscoveryEndpointError(
                "direct teacher must carry a released technical reference"
            )
        return

    if subject.role in {
        "hard_target_student",
        "soft_target_student",
    }:
        if (
            not subject.eligible
            or not subject.sealed
            or not subject.release_allowed
        ):
            raise Stage7DiscoveryEndpointError(
                "student discovery requires eligible passed-only sealing"
            )
        return

    raise Stage7DiscoveryEndpointError(
        "failed or unsealed students may not release discovery work"
    )


def _require_distillation_bridge(
    value: Mapping[str, Any],
) -> None:
    required = {
        "scientific_data": False,
        "production_eligible": False,
        "production_default": False,
        "registered_fixture_execution": False,
        "failed_attempts_preserved": True,
        "passed_only_sealing": True,
        "hard_soft_condition_ids_distinct": True,
    }

    for key, expected in required.items():
        if value.get(key) != expected:
            raise Stage7DiscoveryEndpointError(
                f"Part D bridge requirement failed: {key}"
            )

    hard = value.get("hard")
    soft = value.get("soft")

    if not isinstance(hard, Mapping) or not isinstance(soft, Mapping):
        raise Stage7DiscoveryEndpointError(
            "Part D hard/soft summaries are required"
        )

    for name, payload in (
        ("hard", hard),
        ("soft", soft),
    ):
        if payload.get("attempt_count") != 2:
            raise Stage7DiscoveryEndpointError(
                f"{name} attempt accounting is incomplete"
            )

        if payload.get("eligible_count") != 1:
            raise Stage7DiscoveryEndpointError(
                f"{name} fixture requires exactly one eligible attempt"
            )

        if payload.get("training_failure_count") != 1:
            raise Stage7DiscoveryEndpointError(
                f"{name} fixture requires one preserved failure"
            )

        if payload.get("eligible_release_allowed") is not True:
            raise Stage7DiscoveryEndpointError(
                f"{name} eligible attempt did not pass release gate"
            )

        if payload.get("failed_release_allowed") is not False:
            raise Stage7DiscoveryEndpointError(
                f"{name} failed attempt escaped release gate"
            )


def build_technical_discovery_subjects(
    *,
    distillation_result: Mapping[str, Any],
    run_request: Stage7TechnicalRunRequest,
) -> tuple[
    tuple[TechnicalDiscoverySubject, ...],
    tuple[TechnicalDiscoverySubject, ...],
]:
    """Create released and blocked subjects from accepted Part D evidence."""
    if not isinstance(run_request, Stage7TechnicalRunRequest):
        raise Stage7DiscoveryEndpointError(
            "run_request must be Stage7TechnicalRunRequest"
        )

    _require_distillation_bridge(
        distillation_result
    )

    hard = distillation_result["hard"]
    soft = distillation_result["soft"]

    hard_attempts = hard["attempt_record_sha256"]
    soft_attempts = soft["attempt_record_sha256"]

    if (
        not isinstance(hard_attempts, list)
        or len(hard_attempts) != 2
        or not isinstance(soft_attempts, list)
        or len(soft_attempts) != 2
    ):
        raise Stage7DiscoveryEndpointError(
            "Part D attempt references are incomplete"
        )

    released = (
        TechnicalDiscoverySubject(
            subject_id="technical-direct-teacher",
            role="direct_teacher",
            source_reference_sha256=run_request.teacher_reference.sha256,
            eligible=True,
            sealed=True,
            release_allowed=True,
            seed_value=11,
        ),
        TechnicalDiscoverySubject(
            subject_id="technical-hard-eligible-student",
            role="hard_target_student",
            source_reference_sha256=hard["sealed_dense_model_sha256"],
            eligible=True,
            sealed=True,
            release_allowed=True,
            seed_value=13,
        ),
        TechnicalDiscoverySubject(
            subject_id="technical-soft-eligible-student",
            role="soft_target_student",
            source_reference_sha256=soft["sealed_dense_model_sha256"],
            eligible=True,
            sealed=True,
            release_allowed=True,
            seed_value=17,
        ),
    )

    blocked = (
        TechnicalDiscoverySubject(
            subject_id="technical-hard-failed-student",
            role="failed_hard_target_student",
            source_reference_sha256=hard_attempts[1],
            eligible=False,
            sealed=False,
            release_allowed=False,
            seed_value=19,
        ),
        TechnicalDiscoverySubject(
            subject_id="technical-soft-failed-student",
            role="failed_soft_target_student",
            source_reference_sha256=soft_attempts[1],
            eligible=False,
            sealed=False,
            release_allowed=False,
            seed_value=23,
        ),
    )

    for subject in released:
        assert_discovery_releasable(
            subject
        )

    for subject in blocked:
        try:
            assert_discovery_releasable(
                subject
            )
        except Stage7DiscoveryEndpointError:
            continue

        raise Stage7DiscoveryEndpointError(
            "blocked technical student unexpectedly released discovery"
        )

    return released, blocked


def _mask_with_retained(
    *indices: int,
) -> tuple[int, ...]:
    retained = set(indices)

    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= COMPONENT_COUNT
        for index in retained
    ):
        raise Stage7DiscoveryEndpointError(
            "technical retained-component index is invalid"
        )

    return tuple(
        1 if index in retained else 0
        for index in range(COMPONENT_COUNT)
    )


def _proposal_masks(
    *,
    subject_index: int,
    method_name: str,
) -> tuple[tuple[int, ...], ...]:
    base = subject_index * 32

    if method_name == "greedy_deletion":
        return (
            _mask_with_retained(
                base,
                base + 1,
            ),
            _mask_with_retained(
                base + 2,
                base + 3,
            ),
        )

    if method_name == "diversity_forced":
        return (
            _mask_with_retained(
                base + 4,
                base + 5,
            ),
            _mask_with_retained(
                base + 6,
                base + 7,
            ),
        )

    raise Stage7DiscoveryEndpointError(
        f"no accepted adapter for method {method_name!r}"
    )


def _termination_from_discovery(
    stopping_status: str,
) -> TerminationStatus:
    if stopping_status == "completed":
        return TerminationStatus(
            status="completed",
            procedure_censored=False,
        )

    if stopping_status in {
        "native_budget_exhausted",
        "exact_budget_exhausted",
    }:
        return TerminationStatus(
            status="censored",
            procedure_censored=True,
        )

    return TerminationStatus(
        status="failed",
        procedure_censored=False,
    )


def _reconstruct_sealed_stage6a_ledger(
    *,
    captured: _TechnicalExactCapture,
    proposal_masks: tuple[tuple[int, ...], ...],
    fidelity_threshold: float,
    exact_allowance: int,
):
    """Rebuild ledger structure using captured values, never recomputed fidelity."""
    value_by_mask: dict[
        tuple[int, ...],
        float,
    ] = {}

    for call in captured.calls:
        existing = value_by_mask.get(
            call.mask
        )

        if existing is not None and existing != call.fidelity:
            raise Stage7DiscoveryEndpointError(
                "captured exact fidelity is inconsistent for one mask"
            )

        value_by_mask[
            call.mask
        ] = call.fidelity

    replayed_masks: list[
        tuple[int, ...]
    ] = []

    def replay_evaluator(
        mask: tuple[int, ...],
    ) -> float:
        validated = tuple(mask)

        if validated not in value_by_mask:
            raise Stage7DiscoveryEndpointError(
                "ledger reconstruction requested uncaptured fidelity"
            )

        replayed_masks.append(
            validated
        )

        return value_by_mask[
            validated
        ]

    bridge = Stage6AExactEvaluationBridge(
        evaluator=replay_evaluator,
        fidelity_threshold=fidelity_threshold,
        allowance=exact_allowance,
    )

    for proposal_index, mask in enumerate(
        proposal_masks
    ):
        bridge.request(
            mask,
            proposal_index=proposal_index,
        )

    sealed = bridge.terminate()
    evidence = bridge.evidence_record()

    if len(replayed_masks) != len(captured.calls):
        raise Stage7DiscoveryEndpointError(
            "ledger reconstruction did not consume the captured exact transcript"
        )

    for original, replayed in zip(
        captured.calls,
        replayed_masks,
        strict=True,
    ):
        if original.mask != replayed:
            raise Stage7DiscoveryEndpointError(
                "ledger reconstruction changed exact-evaluation order"
            )

    return sealed, evidence


def _stage6e_evidence(
    *,
    subject: TechnicalDiscoverySubject,
    profile,
    policy,
    proposal_masks: tuple[tuple[int, ...], ...],
    sealed_evaluations,
    ledger_hash: str,
    run_id: str,
) -> tuple[ExactCandidateEvidence, ...]:
    by_identity = {
        entry.mask_identity: entry
        for entry in sealed_evaluations
    }

    records = []

    for proposal_index, mask in enumerate(
        proposal_masks
    ):
        retained_components = tuple(
            index
            for index, retained in enumerate(mask)
            if retained
        )

        identity = canonical_mask_identity(
            retained_components
        )

        entry = by_identity.get(
            identity
        )

        if entry is None:
            raise Stage7DiscoveryEndpointError(
                "proposal lacks final exact Stage 6A evidence"
            )

        records.append(
            ExactCandidateEvidence(
                model_id=subject.subject_id,
                discovery_method_id=profile.method_name,
                discovery_config_id=profile.configuration_reference,
                source_budget_reference=policy.source_budget_reference,
                fidelity_metric_reference=policy.fidelity_metric_reference,
                component_basis_reference=policy.component_basis_reference,
                component_basis_size=policy.component_basis_size,
                mask=mask,
                mask_identity=identity,
                exact_fidelity=entry.fidelity,
                proposal_reference=(
                    f"{run_id}:proposal:{proposal_index}"
                ),
                exact_evaluation_reference=(
                    f"{run_id}:exact:{entry.evaluation_order}"
                ),
                source_ledger_reference=(
                    f"{run_id}:stage6a-exact-ledger"
                ),
                source_ledger_hash=ledger_hash,
                recomputed_ledger_hash=ledger_hash,
            )
        )

    return tuple(
        records
    )


def _run_one_adapter(
    *,
    subject: TechnicalDiscoverySubject,
    subject_index: int,
    profile,
    policy,
) -> dict[str, Any]:
    assert_discovery_releasable(
        subject
    )

    if profile.method_name not in ACCEPTED_DISCOVERY_ADAPTERS:
        raise Stage7DiscoveryEndpointError(
            "technical profile names an unaccepted Stage 6D adapter"
        )

    proposal_masks = _proposal_masks(
        subject_index=subject_index,
        method_name=profile.method_name,
    )

    inherited_seen: list[str] = []
    capture = _TechnicalExactCapture()

    if profile.method_name == "greedy_deletion":

        def proposal_source(
            request,
            inherited,
        ):
            inherited_seen.append(
                inherited.__module__
                + "."
                + inherited.__name__
            )
            return proposal_masks

        adapter = GreedyDeletionAdapter(
            proposal_source=proposal_source,
            evaluator=capture,
            fidelity_threshold=policy.fidelity_threshold,
        )

    elif profile.method_name == "diversity_forced":

        def restart_proposal_source(
            request,
            inherited,
        ):
            inherited_seen.append(
                inherited.__module__
                + "."
                + inherited.__name__
            )
            return (
                (
                    0,
                    (
                        proposal_masks[0],
                    ),
                ),
                (
                    1,
                    (
                        proposal_masks[1],
                    ),
                ),
            )

        adapter = DiversityForcedAdapter(
            restart_proposal_source=restart_proposal_source,
            evaluator=capture,
            fidelity_threshold=policy.fidelity_threshold,
        )

    else:
        raise Stage7DiscoveryEndpointError(
            f"unsupported accepted adapter: {profile.method_name!r}"
        )

    run_id = (
        "stage7-technical/"
        f"{subject.subject_id}/"
        f"{profile.profile_id}"
    )

    request = DiscoveryRequest(
        run_id=run_id,
        method_name=profile.method_name,
        method_version=profile.method_version,
        configuration_reference=profile.configuration_reference,
        seed_evidence=deterministic_seed_evidence(
            method_name=profile.method_name,
            method_version=profile.method_version,
            configuration_reference=profile.configuration_reference,
            seed_value=subject.seed_value,
        ),
        native_budget_unit=profile.native_budget_unit,
        native_budget_allowance=profile.native_budget_allowance,
        exact_evaluation_allowance=profile.exact_evaluation_allowance,
        maximum_restarts=profile.maximum_restarts,
        synthetic_fixture=True,
        production_eligible=False,
    )

    result = adapter.run(
        request
    )

    if result.stopping_status != "completed":
        raise Stage7DiscoveryEndpointError(
            "tiny technical discovery fixture did not complete"
        )

    if result.technical_only is not True:
        raise Stage7DiscoveryEndpointError(
            "Stage 6D result lost technical-only status"
        )

    if result.production_eligible is not False:
        raise Stage7DiscoveryEndpointError(
            "Stage 6D result became production eligible"
        )

    if result.unresolved_decisions != UNRESOLVED_DECISIONS:
        raise Stage7DiscoveryEndpointError(
            "Stage 6D result changed unresolved decisions"
        )

    if result.resource_warning != RESOURCE_WARNING:
        raise Stage7DiscoveryEndpointError(
            "Stage 6D result lost native-budget warning"
        )

    if result.native_budget_unit != profile.native_budget_unit:
        raise Stage7DiscoveryEndpointError(
            "Stage 6D native budget unit was relabelled"
        )

    if (
        result.exact_evaluation_allowance
        != profile.exact_evaluation_allowance
    ):
        raise Stage7DiscoveryEndpointError(
            "Stage 6D exact allowance was relabelled"
        )

    if (
        result.exact_ledger_evaluation_count
        != len(capture.calls)
    ):
        raise Stage7DiscoveryEndpointError(
            "captured exact calls do not match accepted ledger count"
        )

    sealed_evaluations, reconstructed = (
        _reconstruct_sealed_stage6a_ledger(
            captured=capture,
            proposal_masks=proposal_masks,
            fidelity_threshold=policy.fidelity_threshold,
            exact_allowance=profile.exact_evaluation_allowance,
        )
    )

    if reconstructed["sha256"] != result.exact_ledger_sha256:
        raise Stage7DiscoveryEndpointError(
            "reconstructed accepted Stage 6A ledger hash mismatch"
        )

    endpoint1 = reduce_endpoint1(
        sealed_evaluations,
        termination=_termination_from_discovery(
            result.stopping_status
        ),
    )

    endpoint2_evidence = _stage6e_evidence(
        subject=subject,
        profile=profile,
        policy=policy,
        proposal_masks=proposal_masks,
        sealed_evaluations=sealed_evaluations,
        ledger_hash=result.exact_ledger_sha256,
        run_id=run_id,
    )

    qualification = qualify_and_deduplicate(
        endpoint2_evidence,
        policy,
        model_id=subject.subject_id,
        discovery_method_id=profile.method_name,
        discovery_config_id=profile.configuration_reference,
    )

    endpoint2 = recompute_endpoint2(
        qualification,
        policy,
    )

    if endpoint2.semantics != PROCEDURE_PACKING_LOWER_BOUND_SEMANTICS:
        raise Stage7DiscoveryEndpointError(
            "Endpoint 2 semantics changed"
        )

    return {
        "run_id": run_id,
        "subject_id": subject.subject_id,
        "subject_role": subject.role,
        "source_reference_sha256": subject.source_reference_sha256,
        "discovery_method": profile.method_name,
        "discovery_method_version": profile.method_version,
        "discovery_configuration_reference": (
            profile.configuration_reference
        ),
        "inherited_entry_point_received": (
            inherited_seen[0]
            if len(inherited_seen) == 1
            else None
        ),
        "native_budget_unit": result.native_budget_unit,
        "native_budget_allowance": result.native_budget_allowance,
        "native_budget_consumed": result.native_budget_consumed,
        "exact_evaluation_allowance": result.exact_evaluation_allowance,
        "exact_evaluation_consumed": result.exact_evaluation_consumed,
        "exact_ledger_sha256": result.exact_ledger_sha256,
        "reconstructed_ledger_sha256": reconstructed["sha256"],
        "ledger_hash_match": (
            reconstructed["sha256"]
            == result.exact_ledger_sha256
        ),
        "exact_fidelity_recomputed": False,
        "proposal_count": result.proposal_count,
        "restart_count": result.restart_count,
        "stopping_status": result.stopping_status,
        "endpoint1": {
            "retained_proportion": endpoint1.retained_proportion,
            "mask_identity": endpoint1.mask_identity,
            "global_minimum_claim": endpoint1.global_minimum_claim,
            "termination_status": endpoint1.termination_status,
            "procedure_censored": endpoint1.procedure_censored,
        },
        "endpoint2": {
            "raw_candidate_count": endpoint2.raw_candidate_count,
            "unique_candidate_count": endpoint2.unique_candidate_count,
            "qualified_candidate_count": endpoint2.qualified_candidate_count,
            "packing_lower_bound": endpoint2.packing_lower_bound,
            "semantics": endpoint2.semantics,
            "result_hash": endpoint2.result_hash,
            "policy_hash": endpoint2.policy_hash,
        },
    }


def run_technical_discovery_endpoint_fixture(
    *,
    distillation_result: Mapping[str, Any],
    run_request: Stage7TechnicalRunRequest,
    discovery_profiles_path: str | Path,
    endpoint2_policy_path: str | Path,
) -> dict[str, Any]:
    """Run only synthetic proposal streams through every accepted adapter."""
    released, blocked = build_technical_discovery_subjects(
        distillation_result=distillation_result,
        run_request=run_request,
    )

    profiles = load_technical_profiles(
        discovery_profiles_path
    )

    profile_methods = tuple(
        profile.method_name
        for profile in profiles
    )

    if profile_methods != ACCEPTED_DISCOVERY_ADAPTERS:
        raise Stage7DiscoveryEndpointError(
            "Stage 7A requires every accepted Stage 6D adapter exactly once"
        )

    policy = load_technical_policy(
        endpoint2_policy_path
    )

    if policy.scientific_data is not False:
        raise Stage7DiscoveryEndpointError(
            "Endpoint 2 technical policy contains scientific data"
        )

    if policy.production_default is not False:
        raise Stage7DiscoveryEndpointError(
            "Endpoint 2 technical policy became a production default"
        )

    if policy.resolves_unresolved_decisions:
        raise Stage7DiscoveryEndpointError(
            "Endpoint 2 technical policy resolved an unresolved decision"
        )

    runs = []

    for subject_index, subject in enumerate(
        released
    ):
        for profile in profiles:
            runs.append(
                _run_one_adapter(
                    subject=subject,
                    subject_index=subject_index,
                    profile=profile,
                    policy=policy,
                )
            )

    expected_run_count = (
        len(released)
        * len(profiles)
    )

    if len(runs) != expected_run_count:
        raise Stage7DiscoveryEndpointError(
            "not every released subject reached every accepted adapter"
        )

    run_subject_ids = {
        record["subject_id"]
        for record in runs
    }

    blocked_ids = {
        subject.subject_id
        for subject in blocked
    }

    if run_subject_ids & blocked_ids:
        raise Stage7DiscoveryEndpointError(
            "failed or unsealed student released discovery work"
        )

    native_units = {
        profile.method_name: profile.native_budget_unit
        for profile in profiles
    }

    if len(set(native_units.values())) != len(native_units):
        raise Stage7DiscoveryEndpointError(
            "method-native units must remain method-specific"
        )

    if any(
        not record["ledger_hash_match"]
        for record in runs
    ):
        raise Stage7DiscoveryEndpointError(
            "Stage 6A ledger linkage is not exact"
        )

    if any(
        record["exact_fidelity_recomputed"]
        for record in runs
    ):
        raise Stage7DiscoveryEndpointError(
            "fidelity was recomputed during endpoint bridging"
        )

    return {
        "schema_version": DISCOVERY_ENDPOINT_SCHEMA_VERSION,
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "production_default": False,
        "resolves_decisions": [],
        "accepted_adapter_methods": list(
            ACCEPTED_DISCOVERY_ADAPTERS
        ),
        "released_subject_ids": [
            subject.subject_id
            for subject in released
        ],
        "blocked_subject_ids": [
            subject.subject_id
            for subject in blocked
        ],
        "released_subject_count": len(released),
        "blocked_subject_count": len(blocked),
        "discovery_run_count": len(runs),
        "expected_discovery_run_count": expected_run_count,
        "native_budget_units": native_units,
        "native_units_resource_equivalent": False,
        "failed_or_unsealed_discovery_runs": 0,
        "all_ledger_hashes_match": True,
        "fidelity_recomputation": False,
        "fidelity_relabelling": False,
        "real_search_execution": False,
        "registered_fixture_execution": False,
        "runs": runs,
    }
