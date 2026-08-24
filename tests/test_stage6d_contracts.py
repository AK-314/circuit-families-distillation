from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from circuit_families.stage6d import (
    RESOURCE_WARNING,
    UNRESOLVED_DECISIONS,
    DiscoveryResult,
    TrajectoryEvent,
    deterministic_seed_evidence,
    load_technical_profiles,
)

PROFILE_PATH = Path(
    "followup/configs/stage6d/technical_discovery_profiles_v1.json"
)


def test_profiles_are_injected_nonproduction_and_resource_imperfect():
    profiles = load_technical_profiles(PROFILE_PATH)

    assert len(profiles) == 2
    assert profiles[0].native_budget_unit != profiles[1].native_budget_unit

    for profile in profiles:
        assert profile.technical_only is True
        assert profile.production_eligible is False
        assert profile.unresolved_decisions == UNRESOLVED_DECISIONS
        assert profile.resource_warning == RESOURCE_WARNING


def test_seed_evidence_is_deterministic_and_configuration_bound():
    kwargs = dict(
        method_name="greedy_deletion",
        method_version="inherited-technical-adapter/v1",
        configuration_reference="fixture-a",
        seed_value=17,
    )

    first = deterministic_seed_evidence(**kwargs)
    second = deterministic_seed_evidence(**kwargs)

    assert first == second

    changed = deterministic_seed_evidence(
        **{**kwargs, "configuration_reference": "fixture-b"}
    )
    assert changed.seed_material_sha256 != first.seed_material_sha256


def test_result_rejects_bad_ledger_hash():
    seed = deterministic_seed_evidence(
        method_name="greedy_deletion",
        method_version="inherited-technical-adapter/v1",
        configuration_reference="fixture",
        seed_value=1,
    )

    with pytest.raises(ValueError, match="exact_ledger_sha256"):
        DiscoveryResult(
            run_id="x",
            method_name="greedy_deletion",
            method_version="inherited-technical-adapter/v1",
            configuration_reference="fixture",
            seed_evidence=seed,
            native_budget_unit="proposals",
            native_budget_allowance=1,
            native_budget_consumed=0,
            native_budget_exhausted=False,
            exact_evaluation_allowance=1,
            exact_evaluation_consumed=1,
            exact_budget_exhausted=True,
            exact_ledger_sha256="bad",
            exact_ledger_evaluation_count=1,
            exact_ledger_proposal_count=0,
            restart_count=0,
            proposal_count=0,
            exact_request_count=0,
            stopping_status="completed",
            trajectory=(),
            technical_only=True,
            production_eligible=False,
            unresolved_decisions=UNRESOLVED_DECISIONS,
            resource_warning=RESOURCE_WARNING,
        )


def test_common_trajectory_shape_is_fixed():
    event = TrajectoryEvent(
        sequence_index=0,
        kind="proposal",
        restart_index=0,
        native_consumed=1,
        exact_requested=0,
        detail={"fixture": True},
    )

    assert tuple(sorted(asdict(event))) == (
        "detail",
        "exact_requested",
        "kind",
        "native_consumed",
        "restart_index",
        "sequence_index",
    )
