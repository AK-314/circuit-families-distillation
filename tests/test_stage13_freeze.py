from __future__ import annotations

import copy
import json

import pytest

from circuit_families.stage13_freeze import Stage13FreezeError
from scripts.validate_stage13_freeze import (
    PATHS,
    load,
    validate,
    validate_analysis,
    validate_array_spec,
    validate_protocol,
)


def test_complete_freeze_recomputes_deterministically() -> None:
    first = validate()
    second = validate()
    assert first == second
    assert first["validation"] == "PASS"
    assert first["logical_job_count"] == 8745
    assert first["registered_or_private_artifacts_accessed"] is False
    assert first["scientific_jobs_executed"] is False


def test_protocol_rejects_postapproval_seed_pooling_and_unknown_defaults() -> None:
    protocol = load(PATHS["protocol"])
    dossier = load(PATHS["dossier"])
    changed_seed = copy.deepcopy(protocol)
    changed_seed["population_and_tasks"]["teacher_seeds"][-1] = 15
    with pytest.raises(Stage13FreezeError, match="post-approval decision edit"):
        validate_protocol(changed_seed, dossier)
    pooled = copy.deepcopy(protocol)
    pooled["interpretation_units"]["hard_soft_pooling"] = True
    with pytest.raises(Stage13FreezeError, match="hard/soft separation"):
        validate_protocol(pooled, dossier)
    unknown = copy.deepcopy(protocol)
    unknown["silent_default"] = 0.99
    with pytest.raises(Stage13FreezeError, match="fields mismatch"):
        validate_protocol(unknown, dossier)


def test_analysis_rejects_missing_fourier_control_and_global_minimum_claim() -> None:
    analysis = load(PATHS["analysis"])
    dossier = load(PATHS["dossier"])
    missing = copy.deepcopy(analysis)
    missing["fourier"]["conditions"].pop()
    with pytest.raises(Stage13FreezeError, match="post-approval analysis edit"):
        validate_analysis(missing, dossier)
    global_claim = copy.deepcopy(analysis)
    global_claim["claim_rules"]["endpoint1"] = "global minimum"
    with pytest.raises(Stage13FreezeError, match="claim boundary"):
        validate_analysis(global_claim, dossier)


def test_manifest_rejects_cycle_identity_collision_and_provider_assumption() -> None:
    arrays = load(PATHS["arrays"])
    seal = load(PATHS["seal"])
    cyclic = copy.deepcopy(arrays)
    cyclic["arrays"][0]["dependencies"] = ["gate-exit"]
    with pytest.raises(Stage13FreezeError, match="cyclic, forward, or missing"):
        validate_array_spec(cyclic, seal)
    duplicate = copy.deepcopy(arrays)
    duplicate["arrays"][1]["array_id"] = duplicate["arrays"][0]["array_id"]
    with pytest.raises(Stage13FreezeError, match="duplicate or invalid"):
        validate_array_spec(duplicate, seal)
    provider = copy.deepcopy(arrays)
    provider["resource_classes"][0]["provider"] = "unverified-cluster"
    with pytest.raises(Stage13FreezeError, match="seal mismatch|provider"):
        validate_array_spec(provider, seal)


def test_synthetic_fixture_contains_rejection_and_failure_paths() -> None:
    fixture = json.loads(PATHS["fixture"].read_text(encoding="utf-8"))
    states = {item["state"] for item in fixture["terminal_cases"]}
    assert {
        "nonfinite_rejected",
        "corrupted_rejected",
        "duplicate_rejected",
        "stale_rejected",
        "conflicting_rejected",
    } <= states
    assert {item["outcome"] for item in fixture["fourier_cases"]} == {
        "winning",
        "tying",
        "losing",
        "failing",
        "incomplete",
    }
