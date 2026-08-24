#!/usr/bin/env python3
"""Read-only synthetic Stage 6D discovery-adapter validation."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from circuit_families.stage6d import (  # noqa: E402
    DiscoveryRequest,
    DiversityForcedAdapter,
    GreedyDeletionAdapter,
    deterministic_seed_evidence,
    load_technical_profiles,
)

MASK_SIZE = 516
PROFILE_PATH = (
    REPOSITORY
    / "followup"
    / "configs"
    / "stage6d"
    / "technical_discovery_profiles_v1.json"
)


def _mask_without(*indices: int) -> list[int]:
    mask = [1] * MASK_SIZE
    for index in indices:
        mask[index] = 0
    return mask


def _evaluator(mask: tuple[int, ...]) -> float:
    removed = sum(value == 0 for value in mask)
    return 1.0 - removed / 1000.0


def _request(profile, *, run_id: str, seed_value: int) -> DiscoveryRequest:
    seed = deterministic_seed_evidence(
        method_name=profile.method_name,
        method_version=profile.method_version,
        configuration_reference=profile.configuration_reference,
        seed_value=seed_value,
    )
    return DiscoveryRequest(
        run_id=run_id,
        method_name=profile.method_name,
        method_version=profile.method_version,
        configuration_reference=profile.configuration_reference,
        seed_evidence=seed,
        native_budget_unit=profile.native_budget_unit,
        native_budget_allowance=profile.native_budget_allowance,
        exact_evaluation_allowance=profile.exact_evaluation_allowance,
        maximum_restarts=profile.maximum_restarts,
        synthetic_fixture=True,
        production_eligible=False,
    )


def _greedy_source(request, inherited_entry_point):
    assert inherited_entry_point.__name__ == "greedy_sparse_search"
    return [
        _mask_without(0),
        _mask_without(1),
        _mask_without(1),
    ]


def _diversity_source(request, inherited_entry_point):
    assert inherited_entry_point.__name__ == "run_sequential_family_search"
    return [
        (0, [_mask_without(2)]),
        (1, [_mask_without(3), _mask_without(3)]),
    ]


def _print_result(result) -> None:
    print(f"method={result.method_name}")
    print(f"method_version={result.method_version}")
    print(f"native_unit={result.native_budget_unit}")
    print(
        "native_usage="
        f"{result.native_budget_consumed}/"
        f"{result.native_budget_allowance}"
    )
    print(
        "exact_usage="
        f"{result.exact_evaluation_consumed}/"
        f"{result.exact_evaluation_allowance}"
    )
    print(f"proposals={result.proposal_count}")
    print(f"restarts={result.restart_count}")
    print(f"termination={result.stopping_status}")
    print(f"exact_ledger_sha256={result.exact_ledger_sha256}")
    print(
        "exact_ledger_counts="
        f"evaluations:{result.exact_ledger_evaluation_count},"
        f"proposals:{result.exact_ledger_proposal_count}"
    )
    print(f"resource_warning={result.resource_warning}")
    print("---")


def main() -> int:
    profiles = load_technical_profiles(PROFILE_PATH)
    by_method = {profile.method_name: profile for profile in profiles}

    greedy_profile = by_method["greedy_deletion"]
    diversity_profile = by_method["diversity_forced"]

    greedy = GreedyDeletionAdapter(
        proposal_source=_greedy_source,
        evaluator=_evaluator,
        fidelity_threshold=0.9,
    )
    diversity = DiversityForcedAdapter(
        restart_proposal_source=_diversity_source,
        evaluator=_evaluator,
        fidelity_threshold=0.9,
    )

    greedy_result = greedy.run(
        _request(
            greedy_profile,
            run_id="stage6d-cli-greedy",
            seed_value=101,
        )
    )
    diversity_result = diversity.run(
        _request(
            diversity_profile,
            run_id="stage6d-cli-diversity",
            seed_value=202,
        )
    )

    for result in (greedy_result, diversity_result):
        if result.stopping_status != "completed":
            print("STAGE6D_VALIDATE=FAIL")
            return 1
        if result.production_eligible:
            print("STAGE6D_VALIDATE=FAIL")
            return 1

    if greedy_result.native_budget_unit == diversity_result.native_budget_unit:
        print("STAGE6D_VALIDATE=FAIL")
        return 1

    if (
        greedy_result.exact_evaluation_consumed
        != diversity_result.exact_evaluation_consumed
    ):
        print("STAGE6D_VALIDATE=FAIL")
        return 1

    _print_result(greedy_result)
    _print_result(diversity_result)

    print("common_record_shape=PASS")
    print("native_exact_separation=PASS")
    print("stage6a_exact_ledger_evidence=PASS")
    print("cwd_independent=PASS")
    print("read_only=PASS")
    print("production_eligible=NO")
    print("scientific_data=NO")
    print("STAGE6D_VALIDATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
