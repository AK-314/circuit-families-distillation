#!/usr/bin/env python3
"""Portable validate-only Stage 12-R3 packing calibration integration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# Repository imports intentionally follow the portable src/ bootstrap above.
# E402 is suppressed only on those delayed imports.

from circuit_families.stage6a import COMPONENT_COUNT, ExactLedgerBuilder  # noqa: E402
from circuit_families.stage6a.models import SealedLedger, TechnicalLedgerProfile  # noqa: E402
from circuit_families.stage6e.records import load_technical_policy  # noqa: E402
from circuit_families.stage12r2.contracts import canonical_sha256  # noqa: E402
from circuit_families.stage12r3 import (  # noqa: E402
    CONTRACT_VERSION,
    CombinatorialFloorProfile,
    CombinatorialPackingRule,
    LocalExactProfile,
    OrdinaryRestartProfile,
    RestartDiscoveryOutput,
    SizeTypeMatchingRule,
    TractableFixtureProfile,
    TractableSearchOutput,
    run_combinatorial_floor,
    run_local_exact_perturbations,
    run_ordinary_restart_baseline,
    run_tractable_calibration,
)

REPORT_VERSION = "stage12r3-validate-report/v1"


def _common_mask(*indices: int) -> tuple[int, ...]:
    values = [0] * COMPONENT_COUNT
    for index in indices:
        values[index] = 1
    return tuple(values)


def _technical_policy():
    return load_technical_policy(
        REPOSITORY_ROOT
        / "followup/configs/stage6e/technical_endpoint2_policy_v1.json"
    )


def _combinatorial_report() -> dict[str, object]:
    ids = ("a0", "a1", "m0", "m1")
    types = ("attention", "attention", "mlp", "mlp")
    basis_hash = canonical_sha256(
        {
            "fixture": "stage12r3-validate-combinatorial",
            "ids": ids,
            "types": types,
        }
    )

    profile = CombinatorialFloorProfile(
        profile_id="stage12r3-validate-combinatorial",
        basis_hash=basis_hash,
        ordered_component_ids=ids,
        component_types=types,
        matching_rule=SizeTypeMatchingRule(
            retained_sizes=(2,),
            component_type_counts=(("attention", 1), ("mlp", 1)),
        ),
        batch_count=2,
        draws_per_batch=6,
        root_seed=1203,
        seed_stream_id="stage12r3-validate",
    )
    packing_rule = CombinatorialPackingRule(
        component_basis_reference=basis_hash,
        component_basis_size=4,
        max_pairwise_overlap=0.5,
    )

    result = run_combinatorial_floor(profile, packing_rule)

    return {
        "layer": "combinatorial_floor",
        "profile_identity": canonical_sha256(asdict(profile)),
        "packing_rule_identity": packing_rule.policy_hash,
        "raw_draw_count": result.raw_draw_count,
        "duplicate_draw_count": result.duplicate_draw_count,
        "packing_statistics": [
            batch.packing_statistic for batch in result.batches
        ],
        "fidelity_claim": result.fidelity_claim,
        "exact_evaluation_count": result.exact_evaluation_count,
        "endpoint2_claim": result.endpoint2_claim,
        "semantics": "size_type_matched_combinatorial_overlap_floor_only",
        "scientific_data": False,
        "production_eligible": False,
    }


def _ordinary_report(policy) -> tuple[dict[str, object], dict[str, object]]:
    masks = (_common_mask(0), _common_mask(0), _common_mask(1))

    profile = OrdinaryRestartProfile(
        profile_id="stage12r3-validate-ordinary",
        run_id="stage12r3-validate-ordinary",
        method_name="technical-ordinary-discovery",
        method_version="v1",
        discovery_config_id="stage12r3-validate-ordinary-config",
        model_id="stage12r3-validate-synthetic-model",
        component_basis_reference=policy.component_basis_reference,
        fidelity_threshold=policy.fidelity_threshold,
        restart_count=3,
        root_seed=1203,
        native_budget_per_restart=5,
        exact_evaluation_allowance=5,
    )

    def procedure(context):
        return RestartDiscoveryOutput(
            proposals=(masks[context.restart_index],),
            native_work_consumed=2,
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=policy,
        evaluator=lambda mask: 1.0,
        discovery_procedure=procedure,
    )

    censored_profile = OrdinaryRestartProfile(
        profile_id="stage12r3-validate-ordinary-censored",
        run_id="stage12r3-validate-ordinary-censored",
        method_name="technical-ordinary-discovery",
        method_version="v1",
        discovery_config_id="stage12r3-validate-ordinary-config",
        model_id="stage12r3-validate-synthetic-model",
        component_basis_reference=policy.component_basis_reference,
        fidelity_threshold=policy.fidelity_threshold,
        restart_count=3,
        root_seed=1204,
        native_budget_per_restart=5,
        exact_evaluation_allowance=2,
    )
    distinct_masks = (_common_mask(0), _common_mask(1), _common_mask(2))

    def censored_procedure(context):
        return RestartDiscoveryOutput(
            proposals=(distinct_masks[context.restart_index],),
            native_work_consumed=2,
        )

    censored = run_ordinary_restart_baseline(
        profile=censored_profile,
        policy=policy,
        evaluator=lambda mask: 1.0,
        discovery_procedure=censored_procedure,
    )

    primary = {
        "layer": "ordinary_restart_baseline",
        "profile_identity": profile.identity,
        "discovery_family": result.discovery_family,
        "discovery_relationship": result.discovery_relationship,
        "restart_count": result.requested_restart_count,
        "raw_proposal_count": result.raw_restart_proposal_count,
        "exact_evaluation_count": result.exact_ledger_evaluation_count,
        "exact_charged": result.exact_budget.charged,
        "qualified_unique_count": (
            result.qualification.qualified_candidate_count
        ),
        "packing_lower_bound": result.packing_lower_bound,
        "duplicate_recovery_present": (
            result.raw_restart_proposal_count
            > result.qualification.unique_candidate_count
        ),
        "procedure_censored": result.procedure_censored,
        "mechanism_count_claim": result.mechanism_count_claim,
        "scientific_data": result.scientific_data,
        "production_eligible": result.production_eligible,
    }
    censored_summary = {
        "exact_censored_count": censored.exact_request_censored_count,
        "procedure_censored": censored.procedure_censored,
        "exact_evaluation_count": censored.exact_ledger_evaluation_count,
        "exact_charged": censored.exact_budget.charged,
    }
    return primary, censored_summary


def _source_seed_ledger(policy, seed: tuple[int, ...]) -> SealedLedger:
    builder = ExactLedgerBuilder(
        evaluator=lambda mask: 1.0,
        fidelity_threshold=policy.fidelity_threshold,
    )
    builder.add_mask((1,) * COMPONENT_COUNT, proposal_index=0)
    builder.add_mask(seed, proposal_index=1)
    entries = builder.seal()

    return SealedLedger(
        profile=TechnicalLedgerProfile(
            profile_version="stage12r3-validate-seed-ledger/v1",
            name="stage12r3-validate-seed-ledger",
            synthetic_only=True,
            scientific_data=False,
            production_eligible=False,
            unresolved_decisions=("RD-006", "RD-008", "RD-009"),
        ),
        evaluations=entries,
        proposals=tuple(builder.proposals),
        has_intact_baseline=True,
        sealed=True,
    )


def _local_report(policy) -> dict[str, object]:
    seed = _common_mask(0)
    empty = _common_mask()
    source = _source_seed_ledger(policy, seed)

    profile = LocalExactProfile(
        profile_id="stage12r3-validate-local",
        run_id="stage12r3-validate-local",
        model_id="stage12r3-validate-synthetic-model",
        discovery_method_id="stage12r3-local-perturbation-v1",
        discovery_config_id="stage12r3-validate-local-config",
        component_basis_reference=policy.component_basis_reference,
        component_types=tuple(
            "attention" if index < 4 else "mlp"
            for index in range(COMPONENT_COUNT)
        ),
        fidelity_threshold=policy.fidelity_threshold,
        exact_evaluation_allowance=4,
        enabled_operations=("drop",),
        max_hamming_distance=1,
    )

    def evaluator(mask):
        if mask == (1,) * COMPONENT_COUNT:
            return 1.0
        if mask == seed:
            return 1.0
        if mask == empty:
            return 0.0
        return 0.0

    result = run_local_exact_perturbations(
        profile=profile,
        policy=policy,
        seed_ledger=source,
        seed_ledger_reference="stage12r3-validate-source-ledger",
        seed_masks=(seed,),
        evaluator=evaluator,
    )

    qualified = sum(
        proposal.qualifies is True for proposal in result.proposals
    )
    nonqualified = sum(
        proposal.qualifies is False for proposal in result.proposals
    )

    return {
        "layer": "local_exact_perturbation",
        "profile_identity": profile.identity,
        "validated_seed_count": len(result.validated_seeds),
        "proposal_count": len(result.proposals),
        "qualified_proposal_count": qualified,
        "nonqualified_proposal_count": nonqualified,
        "exact_evaluation_count": result.exact_ledger_evaluation_count,
        "exact_charged": result.exact_budget_charged,
        "packing_lower_bound": result.packing_lower_bound,
        "inherited_fidelity_used": result.inherited_fidelity_used,
        "surrogate_fidelity_used": result.surrogate_fidelity_used,
        "procedure_censored": result.procedure_censored,
        "discovery_relationship": result.discovery_relationship,
        "scientific_data": result.scientific_data,
        "production_eligible": result.production_eligible,
    }


def _tractable_report(policy) -> tuple[dict[str, object], dict[str, object]]:
    profile = TractableFixtureProfile(
        profile_id="stage12r3-validate-tractable",
        run_id="stage12r3-validate-tractable",
        model_id="stage12r3-validate-synthetic-model",
        discovery_method_id="stage12r3-tractable-search-v1",
        discovery_config_id="stage12r3-validate-tractable-config",
        component_basis_reference=policy.component_basis_reference,
        free_component_indices=(0, 1, 2),
        qualifying_free_patterns=(0, 1, 2),
        fidelity_threshold=policy.fidelity_threshold,
        search_exact_evaluation_allowance=10,
    )

    def mask(pattern: int) -> tuple[int, ...]:
        values = [0] * COMPONENT_COUNT
        for bit_index, component_index in enumerate(
            profile.free_component_indices
        ):
            if pattern & (1 << bit_index):
                values[component_index] = 1
        return tuple(values)

    result = run_tractable_calibration(
        profile=profile,
        policy=policy,
        search_procedure=lambda context: TractableSearchOutput(
            proposals=(mask(1), mask(2))
        ),
    )

    def failed_search(context) -> NoReturn:
        raise RuntimeError("constructed validate-only search failure")

    failed = run_tractable_calibration(
        profile=profile,
        policy=policy,
        search_procedure=failed_search,
    )

    primary = {
        "layer": "tractable_feasible_region",
        "profile_identity": profile.identity,
        "admissible_mask_count": result.admissible_mask_count,
        "feasible_mask_count": result.feasible_mask_count,
        "admissible_universe_hash": result.admissible_universe_hash,
        "feasible_inventory_hash": result.feasible_inventory_hash,
        "certificate_exactness": result.certificate.exactness_claim,
        "certificate_exhaustive": result.certificate.exhaustive,
        "certified_packing_optimum": result.certified_packing_optimum,
        "search_recovered_feasible_count": result.recovered_feasible_count,
        "search_missed_feasible_count": result.missed_feasible_count,
        "feasible_recall": result.feasible_recall,
        "endpoint1_retained_proportion_gap": (
            result.endpoint1_retained_proportion_gap
        ),
        "packing_gap": result.packing_gap,
        "search_exact_evaluation_coverage": (
            result.search_exact_evaluation_coverage
        ),
        "search_procedure_censored": result.search_procedure_censored,
        "teacher_seed_transfer": result.teacher_seed_transfer,
        "main_experiment_transfer": result.main_experiment_transfer,
        "mechanism_count_claim": result.mechanism_count_claim,
        "scientific_data": result.scientific_data,
        "production_eligible": result.production_eligible,
    }
    failed_summary = {
        "search_procedure_failed": failed.search_procedure_failed,
        "search_procedure_censored": failed.search_procedure_censored,
        "search_packing_lower_bound": (
            failed.search_endpoint2.packing_lower_bound
        ),
        "search_feasible_recall": failed.feasible_recall,
        "error_kind": (
            (failed.search_procedure_error or "").split(":", 1)[0]
            or None
        ),
    }
    return primary, failed_summary


def run_validate_only() -> dict[str, object]:
    policy = _technical_policy()

    combinatorial = _combinatorial_report()
    ordinary, ordinary_censored = _ordinary_report(policy)
    local = _local_report(policy)
    tractable, failed_search = _tractable_report(policy)

    report: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "classification": "synthetic_technical_only",
        "layers": {
            "combinatorial_floor": combinatorial,
            "ordinary_restart_baseline": ordinary,
            "local_exact_perturbation": local,
            "tractable_feasible_region": tractable,
        },
        "outcome_preservation": {
            "null": {
                "status": "null",
                "value": None,
                "reason": "no_fidelity_claim_for_combinatorial_floor",
            },
            "negative": {
                "status": "negative",
                "local_nonqualified_neighbor_count": (
                    local["nonqualified_proposal_count"]
                ),
            },
            "zero": {
                "status": "zero",
                "failed_search_packing_lower_bound": (
                    failed_search["search_packing_lower_bound"]
                ),
            },
            "failed": {
                "status": "failed",
                **failed_search,
            },
            "unavailable": {
                "status": "unavailable",
                "subject": "certified_near_exact_variant",
                "reason": "v1_validate_fixture_uses_exact_enumeration_only",
            },
            "censored": {
                "status": "censored",
                **ordinary_censored,
            },
        },
        "claim_boundaries": {
            "ordinary_restart_independent_method_claim": False,
            "local_inherited_fidelity": False,
            "local_surrogate_fidelity": False,
            "tractable_main_scale_transfer": False,
            "mechanism_count_claim": False,
            "production_packing_policy_selected": False,
            "rd_006_open": True,
            "rd_008_open": True,
            "rd_009_open": True,
        },
        "execution": {
            "validate_only": True,
            "outputs_created": False,
            "registered_data_loaded": False,
            "registered_model_execution": False,
            "scientific_execution": False,
            "scientific_data": False,
            "production_eligible": False,
        },
    }

    report["report_hash"] = canonical_sha256(report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Stage 12-R3 technical packing calibration."
    )
    parser.add_argument("--validate-only", action="store_true", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.validate_only:
        raise SystemExit("validate-only mode is required")

    report = run_validate_only()

    if args.json:
        print(
            json.dumps(
                report,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0

    layers = report["layers"]
    outcomes = report["outcome_preservation"]
    execution = report["execution"]

    print("classification=synthetic_technical_only")
    print(f"report_version={report['report_version']}")
    print(f"contract_version={report['contract_version']}")
    print(f"layers={len(layers)}")
    print(
        "combinatorial_floor="
        f"{layers['combinatorial_floor']['raw_draw_count']}"
    )
    print(
        "ordinary_restart_qualified="
        f"{layers['ordinary_restart_baseline']['qualified_unique_count']}"
    )
    print(
        "local_nonqualified_neighbors="
        f"{layers['local_exact_perturbation']['nonqualified_proposal_count']}"
    )
    print(
        "tractable_feasible="
        f"{layers['tractable_feasible_region']['feasible_mask_count']}"
    )
    print(
        "tractable_recall="
        f"{layers['tractable_feasible_region']['feasible_recall']:.6f}"
    )
    print(
        "censored_exact_requests="
        f"{outcomes['censored']['exact_censored_count']}"
    )
    print(
        "failed_search="
        f"{str(outcomes['failed']['search_procedure_failed']).lower()}"
    )
    print(
        "registered_data_loaded="
        f"{str(execution['registered_data_loaded']).lower()}"
    )
    print(
        "scientific_execution="
        f"{str(execution['scientific_execution']).lower()}"
    )
    print(f"report_hash={report['report_hash']}")
    print("STAGE12R3_VALIDATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
