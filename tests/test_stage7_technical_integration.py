from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage4_seed_derivation import derive_seed
from circuit_families.stage7 import (
    JOB_LIFECYCLE_DELEGATE,
    Stage7ContractError,
    Stage7TechnicalRunRequest,
    build_stage7_lifecycle,
    build_technical_run_manifest,
    load_technical_run_request,
    verify_repository_references,
)

ROOT = Path(__file__).resolve().parents[1]

REQUEST_PATH = (
    ROOT
    / "followup/configs/stage7/"
    "technical_run_request_v1.json"
)

STAGE3_PATH = (
    ROOT
    / "followup/manifests/"
    "stage3_teacher_registry_v1.json"
)


def _stage3() -> Stage3AvailabilityIndex:
    registry = json.loads(
        STAGE3_PATH.read_text(
            encoding="utf-8"
        )
    )
    return Stage3AvailabilityIndex.from_registry(
        registry
    )


def _request_mapping() -> dict[str, object]:
    return json.loads(
        REQUEST_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_part_c_request_manifest_and_seed_evidence_are_deterministic() -> None:
    request = load_technical_run_request(
        REQUEST_PATH
    )
    stage3 = _stage3()

    first = build_technical_run_manifest(
        request,
        stage3=stage3,
        repository_root=ROOT,
    )
    second = build_technical_run_manifest(
        request,
        stage3=stage3,
        repository_root=ROOT,
    )

    assert first.to_mapping() == second.to_mapping()
    assert first.manifest_sha256 == second.manifest_sha256
    assert (
        first.request.run_identity
        == second.request.run_identity
    )
    assert len(first.seed_evidence) == 6
    assert len(first.lifecycle.steps) == 10

    direct = derive_seed(
        request.seed_inputs[0].to_seed_inputs(),
        stage3,
    )

    assert (
        first.seed_evidence[0]["seed_value"]
        == direct.seed_value
    )
    assert (
        first.seed_evidence[0]["digest_sha256"]
        == direct.digest_sha256
    )

    hard_conditions = {
        item.condition_id
        for item in request.seed_inputs
        if item.label.startswith("hard_")
    }

    soft_conditions = {
        item.condition_id
        for item in request.seed_inputs
        if item.label.startswith("soft_")
    }

    assert hard_conditions.isdisjoint(
        soft_conditions
    )


def test_part_c_repository_bindings_are_digest_verified() -> None:
    request = load_technical_run_request(
        REQUEST_PATH
    )

    evidence = verify_repository_references(
        request,
        repository_root=ROOT,
    )

    roles = {
        item.role
        for item in evidence
    }

    assert "hard_profile" in roles
    assert "soft_profile" in roles
    assert "endpoint2_policy" in roles
    assert "analysis_profile" in roles
    assert "job_lifecycle" in roles
    assert "exclusion_register" in roles

    assert (
        request.teacher_reference.source_kind
        == "injected_fixture"
    )

    assert all(
        item.observed_sha256
        for item in evidence
    )


def test_part_c_lifecycle_is_ten_step_isolated_and_delegated() -> None:
    request = load_technical_run_request(
        REQUEST_PATH
    )

    lifecycle = build_stage7_lifecycle(
        output_root=request.output_root,
        job_lifecycle_reference_id=(
            request.job_lifecycle_reference.reference_id
        ),
    )

    mapping = lifecycle.to_mapping()
    steps = lifecycle.topological_steps()

    assert (
        mapping["delegates_job_lifecycle_to"]
        == JOB_LIFECYCLE_DELEGATE
    )

    assert [
        step.ordinal
        for step in steps
    ] == list(range(1, 11))

    assert len(
        {
            step.output_root
            for step in steps
        }
    ) == 10

    discovery = next(
        step
        for step in steps
        if step.step_id == "discovery"
    )

    assert discovery.dependencies == (
        "teacher_target_cache",
        "passed_only_sealing",
    )

    assert discovery.condition_roles == (
        "direct_teacher",
        "hard_target",
        "soft_target",
    )

    assert all(
        step.output_root.startswith(
            request.output_root + "/"
        )
        for step in steps
    )


def test_part_c_rejects_production_resolution_and_unsafe_output_root() -> None:
    original = _request_mapping()

    bad_production = copy.deepcopy(
        original
    )
    bad_production["production_default"] = True

    with pytest.raises(
        Stage7ContractError,
        match="production default",
    ):
        Stage7TechnicalRunRequest.from_mapping(
            bad_production
        )

    bad_resolution = copy.deepcopy(
        original
    )
    bad_resolution["resolves_decisions"] = [
        "UD-003"
    ]

    with pytest.raises(
        Stage7ContractError,
        match="may not resolve",
    ):
        Stage7TechnicalRunRequest.from_mapping(
            bad_resolution
        )

    bad_output = copy.deepcopy(
        original
    )
    bad_output["output_root"] = (
        "followup/artifacts/stage7"
    )

    with pytest.raises(
        Stage7ContractError,
        match="excluded_development",
    ):
        Stage7TechnicalRunRequest.from_mapping(
            bad_output
        )


def test_part_c_rejects_bad_binding_and_hard_soft_identity_collision() -> None:
    original = _request_mapping()

    bad_hash = copy.deepcopy(
        original
    )
    endpoint2 = bad_hash[
        "endpoint2_policy_reference"
    ]
    assert isinstance(
        endpoint2,
        dict,
    )
    endpoint2["sha256"] = "0" * 64

    request = Stage7TechnicalRunRequest.from_mapping(
        bad_hash
    )

    with pytest.raises(
        Stage7ContractError,
        match="digest mismatch",
    ):
        verify_repository_references(
            request,
            repository_root=ROOT,
        )

    collision = copy.deepcopy(
        original
    )
    hard = collision[
        "hard_profile_reference"
    ]
    soft = collision[
        "soft_profile_reference"
    ]

    assert isinstance(hard, dict)
    assert isinstance(soft, dict)

    soft["reference_id"] = hard[
        "reference_id"
    ]

    with pytest.raises(
        Stage7ContractError,
        match="hard and soft",
    ):
        Stage7TechnicalRunRequest.from_mapping(
            collision
        )


def _part_d_config():
    from circuit_families.stage7 import TechnicalDistillationFixtureConfig

    return TechnicalDistillationFixtureConfig(
        hard_learning_rate=0.01,
        soft_learning_rate=0.01,
        technical_stop_step=1,
        technical_safety_step_limit=1,
        soft_tolerance=1.0,
    )


def test_part_d_connects_accepted_cache_shared_trainer_and_sealing(tmp_path) -> None:
    from circuit_families.stage7 import (
        SHARED_TRAINER_REFERENCE,
        TARGET_CACHE_BUILDER_REFERENCE,
        TARGET_CACHE_LOADER_REFERENCE,
        run_technical_distillation_fixture,
    )

    result = run_technical_distillation_fixture(
        output_root=tmp_path / "part-d",
        stage3=_stage3(),
        teacher_seed=0,
        phase="stable post-grokking",
        config=_part_d_config(),
    )

    assert result["scientific_data"] is False
    assert result["production_eligible"] is False
    assert result["production_default"] is False
    assert result["resolves_decisions"] == []
    assert result["shared_trainer_reference"] == SHARED_TRAINER_REFERENCE
    assert (
        result["target_cache_builder_reference"]
        == TARGET_CACHE_BUILDER_REFERENCE
    )
    assert (
        result["target_cache_loader_reference"]
        == TARGET_CACHE_LOADER_REFERENCE
    )
    assert result["full_domain_example_count"] == 12_769
    assert result["hard"]["target_cache_kind"] == "teacher_argmax"
    assert result["soft"]["target_cache_kind"] == "teacher_logits"
    assert result["hard"]["eligible_count"] == 1
    assert result["soft"]["eligible_count"] == 1
    assert result["hard"]["eligible_release_allowed"] is True
    assert result["soft"]["eligible_release_allowed"] is True
    assert result["passed_only_sealing"] is True
    assert result["hard_soft_condition_ids_distinct"] is True


def test_part_d_preserves_failed_initializations_and_blocks_release(tmp_path) -> None:
    from circuit_families.stage7 import run_technical_distillation_fixture

    result = run_technical_distillation_fixture(
        output_root=tmp_path / "part-d-failure",
        stage3=_stage3(),
        teacher_seed=0,
        phase="stable post-grokking",
        config=_part_d_config(),
    )

    assert result["hard"]["attempt_count"] == 2
    assert result["soft"]["attempt_count"] == 2
    assert result["hard"]["attempted_initializations"] == [0, 1]
    assert result["soft"]["attempted_initializations"] == [0, 1]
    assert result["hard"]["training_failure_count"] == 1
    assert result["soft"]["training_failure_count"] == 1
    assert result["hard"]["failed_release_allowed"] is False
    assert result["soft"]["failed_release_allowed"] is False
    assert result["failed_attempts_preserved"] is True
    assert result["real_optimization_configuration"] is False
    assert result["registered_fixture_execution"] is False


def _part_e_distillation(tmp_path):
    from circuit_families.stage7 import run_technical_distillation_fixture

    return run_technical_distillation_fixture(
        output_root=tmp_path / "part-e-distillation",
        stage3=_stage3(),
        teacher_seed=0,
        phase="stable post-grokking",
        config=_part_d_config(),
    )


def test_part_e_released_sources_reach_every_accepted_adapter_and_endpoint(tmp_path) -> None:
    from circuit_families.stage7 import (
        ACCEPTED_DISCOVERY_ADAPTERS,
        load_technical_run_request,
        run_technical_discovery_endpoint_fixture,
    )

    result = run_technical_discovery_endpoint_fixture(
        distillation_result=_part_e_distillation(tmp_path),
        run_request=load_technical_run_request(REQUEST_PATH),
        discovery_profiles_path=(
            ROOT
            / "followup/configs/stage6d/"
            "technical_discovery_profiles_v1.json"
        ),
        endpoint2_policy_path=(
            ROOT
            / "followup/configs/stage6e/"
            "technical_endpoint2_policy_v1.json"
        ),
    )

    assert result["accepted_adapter_methods"] == list(
        ACCEPTED_DISCOVERY_ADAPTERS
    )
    assert result["released_subject_count"] == 3
    assert result["blocked_subject_count"] == 2
    assert result["discovery_run_count"] == 6
    assert result["expected_discovery_run_count"] == 6
    assert result["failed_or_unsealed_discovery_runs"] == 0
    assert result["all_ledger_hashes_match"] is True
    assert result["fidelity_recomputation"] is False
    assert result["fidelity_relabelling"] is False
    assert result["real_search_execution"] is False

    units = result["native_budget_units"]

    assert (
        units["greedy_deletion"]
        == "ranked_component_proposals"
    )
    assert (
        units["diversity_forced"]
        == "restart_ranked_proposals"
    )
    assert result["native_units_resource_equivalent"] is False

    for run in result["runs"]:
        assert run["ledger_hash_match"] is True
        assert run["exact_fidelity_recomputed"] is False
        assert run["stopping_status"] == "completed"
        assert run["endpoint1"]["global_minimum_claim"] is False
        assert run["endpoint1"]["termination_status"] == "completed"
        assert run["endpoint1"]["procedure_censored"] is False
        assert run["endpoint2"]["packing_lower_bound"] == 2
        assert (
            run["endpoint2"]["semantics"]
            == "procedure_dependent_packing_lower_bound"
        )


def test_part_e_failed_or_unsealed_student_is_rejected_before_discovery(tmp_path) -> None:
    import pytest

    from circuit_families.stage7 import (
        Stage7DiscoveryEndpointError,
        assert_discovery_releasable,
        build_technical_discovery_subjects,
        load_technical_run_request,
    )

    released, blocked = build_technical_discovery_subjects(
        distillation_result=_part_e_distillation(tmp_path),
        run_request=load_technical_run_request(REQUEST_PATH),
    )

    assert len(released) == 3
    assert len(blocked) == 2

    for subject in blocked:
        with pytest.raises(
            Stage7DiscoveryEndpointError,
            match="may not release discovery",
        ):
            assert_discovery_releasable(subject)


def test_part_e_budget_spaces_and_exact_allowances_remain_separate(tmp_path) -> None:
    from circuit_families.stage7 import (
        load_technical_run_request,
        run_technical_discovery_endpoint_fixture,
    )

    result = run_technical_discovery_endpoint_fixture(
        distillation_result=_part_e_distillation(tmp_path),
        run_request=load_technical_run_request(REQUEST_PATH),
        discovery_profiles_path=(
            ROOT
            / "followup/configs/stage6d/"
            "technical_discovery_profiles_v1.json"
        ),
        endpoint2_policy_path=(
            ROOT
            / "followup/configs/stage6e/"
            "technical_endpoint2_policy_v1.json"
        ),
    )

    by_method = {}

    for run in result["runs"]:
        method = run["discovery_method"]
        signature = (
            run["native_budget_unit"],
            run["native_budget_allowance"],
            run["exact_evaluation_allowance"],
        )

        if method in by_method:
            assert by_method[method] == signature
        else:
            by_method[method] = signature

    assert (
        by_method["greedy_deletion"][0]
        != by_method["diversity_forced"][0]
    )
    assert by_method["greedy_deletion"][1] == 5
    assert by_method["diversity_forced"][1] == 7
    assert by_method["greedy_deletion"][2] == 3
    assert by_method["diversity_forced"][2] == 3


def _part_f_pipeline(tmp_path):
    import json

    from circuit_families.stage7 import (
        build_endpoint_exclusion_records,
        build_part_f_report,
        build_stage5d_analysis_bridge,
        build_teacher_seed_inventory,
        load_technical_run_request,
        run_technical_discovery_endpoint_fixture,
    )

    distillation = _part_e_distillation(
        tmp_path
    )

    discovery = run_technical_discovery_endpoint_fixture(
        distillation_result=distillation,
        run_request=load_technical_run_request(
            REQUEST_PATH
        ),
        discovery_profiles_path=(
            ROOT
            / "followup/configs/stage6d/"
            "technical_discovery_profiles_v1.json"
        ),
        endpoint2_policy_path=(
            ROOT
            / "followup/configs/stage6e/"
            "technical_endpoint2_policy_v1.json"
        ),
    )

    inventory = build_teacher_seed_inventory(
        teacher_seed=0,
        phase="stable post-grokking",
        distillation_result=distillation,
        discovery_result=discovery,
    )

    analysis_bridge = build_stage5d_analysis_bridge(
        repository_root=ROOT,
        inventory=inventory,
    )

    register = json.loads(
        (
            ROOT
            / "followup/manifests/"
            "stage2_excluded_development_register_v1.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    exclusions = build_endpoint_exclusion_records(
        discovery_result=discovery,
        exclusion_register=register,
    )

    report = build_part_f_report(
        inventory=inventory,
        analysis_bridge=analysis_bridge,
        exclusion_entries=exclusions,
    )

    return (
        distillation,
        discovery,
        inventory,
        analysis_bridge,
        exclusions,
        report,
    )


def test_part_f_inventory_retains_teacher_hard_soft_failed_missing_and_endpoints(
    tmp_path,
) -> None:
    (
        _,
        _,
        inventory,
        _,
        _,
        _,
    ) = _part_f_pipeline(tmp_path)

    rows = inventory["rows"]

    assert inventory["population_unit"] == "teacher_seed"
    assert inventory["student_member_unit"] == "student_initialization"
    assert inventory["student_initializations_are_population_replicates"] is False
    assert inventory["hard_soft_pooled"] is False

    assert {
        row["subject_state"]
        for row in rows
    } == {
        "teacher_direct",
        "eligible",
        "failed",
    }

    assert {
        row["distillation_condition"]
        for row in rows
    } == {
        "teacher_direct",
        "hard",
        "soft",
    }

    assert any(
        row["method_state"] == "missing"
        for row in rows
    )

    assert any(
        row["endpoint1_state"] == "missing"
        and row["endpoint2_state"] == "missing"
        for row in rows
    )

    failed = [
        row
        for row in rows
        if row["subject_state"] == "failed"
    ]

    assert failed
    assert all(
        row["method_state"] == "missing"
        for row in failed
    )
    assert all(
        row["endpoint1"] is None
        and row["endpoint2"] is None
        for row in failed
    )


def test_part_f_reuses_stage5d_hierarchy_without_pseudoreplication(tmp_path) -> None:
    (
        _,
        _,
        inventory,
        bridge,
        _,
        report,
    ) = _part_f_pipeline(tmp_path)

    assert bridge["inventory_sha256"] == inventory["sha256"]
    assert bridge["technical_analysis_profile_id"] == "fixture_median_min2"
    assert bridge["hard_soft_tables_separate"] is True
    assert bridge["hard_soft_pooled"] is False
    assert bridge["population_unit"] == "teacher_seed"
    assert bridge["student_member_unit"] == "student_initialization"
    assert (
        bridge["student_initializations_are_population_replicates"]
        is False
    )
    assert bridge["stage5d_reducer_reimplemented"] is False
    assert bridge["resolved_decisions"] == []

    assert report["hard_soft_tables_separate"] is True
    assert report["hard_soft_pooled"] is False
    assert report["population_unit"] == "teacher_seed"
    assert report["student_initializations_are_population_replicates"] is False


def test_part_f_every_endpoint_like_value_is_excluded_and_primary_rejected(
    tmp_path,
) -> None:
    import pytest

    from circuit_families.stage7 import (
        Stage7InventoryReportingError,
        assert_rejected_as_primary_scientific_input,
    )

    (
        _,
        discovery,
        _,
        _,
        exclusions,
        report,
    ) = _part_f_pipeline(tmp_path)

    assert len(exclusions) == 2 * len(discovery["runs"])
    assert len(exclusions) == 12

    for entry in exclusions:
        assert entry["endpoint_values_emitted"] is True
        assert entry["primary_analysis_eligible"] is False
        assert entry["scientific_selection_eligible"] is False
        assert entry["regeneration_required"] is True
        assert entry["disposition"] == "registered_excluded"
        assert entry["promotion_in_place_permitted"] is False

        with pytest.raises(
            Stage7InventoryReportingError,
            match="rejected as primary scientific input",
        ):
            assert_rejected_as_primary_scientific_input(
                entry
            )

    assert report["endpoint_like_fixture_output_count"] == 12
    assert report["excluded_endpoint_like_fixture_output_count"] == 12
    assert report["primary_scientific_acceptance_count"] == 0
    assert report["scientific_selection_acceptance_count"] == 0
    assert report["post_freeze_regeneration_required"] is True
    assert report["registered_fixture_execution"] is False
    assert report["real_scientific_analysis"] is False


def _part_g_inputs(tmp_path):
    import json

    from circuit_families.stage7 import (
        TechnicalDistillationFixtureConfig,
        load_technical_run_request,
    )

    exclusion_register = json.loads(
        (
            ROOT
            / "followup/manifests/"
            "stage2_excluded_development_register_v1.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    return {
        "repository_root": ROOT,
        "stage3": _stage3(),
        "teacher_seed": 0,
        "phase": "stable post-grokking",
        "run_request": load_technical_run_request(
            REQUEST_PATH
        ),
        "distillation_config": TechnicalDistillationFixtureConfig(
            hard_learning_rate=0.01,
            soft_learning_rate=0.01,
            technical_stop_step=1,
            technical_safety_step_limit=1,
            soft_tolerance=1.0,
        ),
        "discovery_profiles_path": (
            ROOT
            / "followup/configs/stage6d/"
            "technical_discovery_profiles_v1.json"
        ),
        "endpoint2_policy_path": (
            ROOT
            / "followup/configs/stage6e/"
            "technical_endpoint2_policy_v1.json"
        ),
        "exclusion_register": exclusion_register,
    }


def test_part_g_resume_skips_completed_rejects_failed_and_blocks_transfer(
    tmp_path,
) -> None:
    from circuit_families.stage7 import (
        build_resume_merge_evidence,
        load_technical_run_request,
        run_technical_discovery_endpoint_fixture,
    )

    distillation = _part_e_distillation(
        tmp_path
    )

    discovery = run_technical_discovery_endpoint_fixture(
        distillation_result=distillation,
        run_request=load_technical_run_request(
            REQUEST_PATH
        ),
        discovery_profiles_path=(
            ROOT
            / "followup/configs/stage6d/"
            "technical_discovery_profiles_v1.json"
        ),
        endpoint2_policy_path=(
            ROOT
            / "followup/configs/stage6e/"
            "technical_endpoint2_policy_v1.json"
        ),
    )

    evidence = build_resume_merge_evidence(
        stage3=_stage3(),
        teacher_seed=0,
        phase="stable post-grokking",
        distillation_result=distillation,
        discovery_result=discovery,
    )

    assert evidence["hard_completed_action"] == "skip_completed"
    assert evidence["soft_completed_action"] == "skip_completed"
    assert evidence["hard_failed_action"] == "reject_failed_attempt"
    assert evidence["soft_failed_action"] == "reject_failed_attempt"
    assert evidence["cross_attempt_transfer_rejected"] is True
    assert evidence["duplicate_merge_rejected"] is True
    assert evidence["merge_entry_count"] == 4
    assert evidence["merge_order_independent"] is True
    assert evidence["before_resume_counts"] == evidence["after_resume_counts"]
    assert evidence["completed_attempts_duplicated"] is False
    assert evidence["failed_attempts_reexecuted"] is False
    assert evidence["native_charges_duplicated"] is False
    assert evidence["exact_evaluations_duplicated"] is False
    assert evidence["endpoints_duplicated"] is False
    assert evidence["resume_state_transferred"] is False


def test_part_g_independent_separate_root_reproduction_matches_everything(
    tmp_path,
) -> None:
    from circuit_families.stage7 import (
        build_pipeline_reproduction_record,
        compare_independent_pipeline_reproduction,
    )

    inputs = _part_g_inputs(
        tmp_path
    )

    source_root = tmp_path / "source-root"
    reproduction_root = tmp_path / "reproduction-root"

    source = build_pipeline_reproduction_record(
        physical_root=source_root,
        **inputs,
    )

    reproduced = build_pipeline_reproduction_record(
        physical_root=reproduction_root,
        **inputs,
    )

    assert source_root != reproduction_root
    assert source_root.is_dir()
    assert reproduction_root.is_dir()

    comparison = compare_independent_pipeline_reproduction(
        source_record=source,
        reproduction_record=reproduced,
    )

    assert comparison["matched"] is True
    assert comparison["mismatch_paths"] == []
    assert (
        comparison["substantive_source_sha256"]
        == comparison["substantive_reproduction_sha256"]
    )
    assert source["run_request_identity"] == reproduced["run_request_identity"]
    assert source["distillation"] == reproduced["distillation"]
    assert source["discovery_runs"] == reproduced["discovery_runs"]
    assert source["inventory"] == reproduced["inventory"]
    assert source["analysis_bridge"] == reproduced["analysis_bridge"]
    assert (
        source["excluded_endpoint_records"]
        == reproduced["excluded_endpoint_records"]
    )
    assert source["report"] == reproduced["report"]
    assert source["resume_merge"] == reproduced["resume_merge"]
    assert (
        source["stage5d_reconstruction"]
        == reproduced["stage5d_reconstruction"]
    )
    assert source["stage5d_reconstruction"]["matched"] is True


def test_part_g_reproduction_mismatch_diagnostic_is_specific(tmp_path) -> None:
    import copy

    from circuit_families.stage7 import (
        build_pipeline_reproduction_record,
        compare_independent_pipeline_reproduction,
    )
    from circuit_families.stage7.reproduction import _sha256

    inputs = _part_g_inputs(
        tmp_path
    )

    source = build_pipeline_reproduction_record(
        physical_root=tmp_path / "source",
        **inputs,
    )

    altered = copy.deepcopy(
        source
    )

    altered["discovery_runs"][0]["endpoint1"][
        "retained_proportion"
    ] = 0.123456

    altered_without_sha = copy.deepcopy(
        altered
    )
    altered_without_sha.pop(
        "sha256"
    )
    altered["sha256"] = _sha256(
        altered_without_sha
    )

    comparison = compare_independent_pipeline_reproduction(
        source_record=source,
        reproduction_record=altered,
    )

    assert comparison["matched"] is False
    assert comparison["specific_mismatch_diagnostics"] is True
    assert any(
        path.startswith(
            "$.discovery_runs[0].endpoint1.retained_proportion:value:"
        )
        for path in comparison["mismatch_paths"]
    )
    assert comparison["stage8_edge_matrix_executed"] is False


def _part_h_config():
    from circuit_families.stage7 import TechnicalDistillationFixtureConfig

    return TechnicalDistillationFixtureConfig(
        hard_learning_rate=0.01,
        soft_learning_rate=0.01,
        technical_stop_step=1,
        technical_safety_step_limit=1,
        soft_tolerance=1.0,
    )


def test_part_h_portable_fixture_runs_all_ten_steps_and_gating(tmp_path) -> None:
    from circuit_families.stage7 import (
        EXPECTED_PIPELINE_STEPS,
        PORTABLE_REPORT_FILENAME,
        run_portable_stage7_fixture,
    )

    output_root = tmp_path / "portable-stage7a"

    report = run_portable_stage7_fixture(
        output_root=output_root,
        repository_root=ROOT,
        teacher_seed=0,
        phase="stable post-grokking",
        distillation_config=_part_h_config(),
    )

    assert report["pipeline_step_count"] == 10
    assert tuple(
        step["step_id"]
        for step in report["pipeline_steps"]
    ) == EXPECTED_PIPELINE_STEPS

    assert all(
        step["status"] == "PASS"
        for step in report["pipeline_steps"]
    )

    assert report["eligible_path_count"] >= 1
    assert report["failed_path_count"] >= 1
    assert report["failed_subject_discovery_count"] == 0

    assert report["excluded_endpoint_output_count"] == 12
    assert report["primary_analysis_eligible_count"] == 0
    assert report["scientific_selection_eligible_count"] == 0
    assert report["post_freeze_regeneration_required"] is True

    assert report["resume_counts_unchanged"] is True
    assert report["reproduction_matched"] is True
    assert report["reproduction_mismatch_paths"] == []

    assert report["source_record_sha256"] == report[
        "reproduction_record_sha256"
    ]

    assert report["scientific_data"] is False
    assert report["production_eligible"] is False
    assert report["registered_fixture_execution"] is False
    assert report["scientific_execution"] is False
    assert report["stage8_execution"] is False

    assert (
        output_root
        / PORTABLE_REPORT_FILENAME
    ).is_file()


def test_part_h_cli_is_cwd_independent_and_writes_only_explicit_temp_root(
    tmp_path,
) -> None:
    import os
    import subprocess
    import sys

    cli = (
        ROOT
        / "scripts/"
        "validate_stage7_technical_integration.py"
    )

    working_directory = (
        tmp_path
        / "unrelated-cwd"
    )

    working_directory.mkdir()

    output_root = (
        tmp_path
        / "cli-output"
    )

    environment = dict(
        os.environ
    )

    environment.pop(
        "PYTHONPATH",
        None,
    )

    environment.pop(
        "STAGE7A_REEXECUTED",
        None,
    )

    before = tuple(
        working_directory.iterdir()
    )

    result = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--output-root",
            str(output_root),
            "--teacher-seed",
            "0",
            "--phase",
            "stable post-grokking",
            "--hard-learning-rate",
            "0.01",
            "--soft-learning-rate",
            "0.01",
            "--technical-stop-step",
            "1",
            "--technical-safety-step-limit",
            "1",
            "--soft-tolerance",
            "1.0",
        ],
        cwd=working_directory,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert (
        "STAGE7A_TECHNICAL_VALIDATION=PASS"
        in result.stdout
    )

    assert (
        "PIPELINE_STEP_COUNT=10"
        in result.stdout
    )

    assert (
        "OUTPUT_WRITTEN=EXPLICIT_TEMP_ROOT_ONLY"
        in result.stdout
    )

    assert (
        "REGISTERED_FIXTURE_EXECUTION=NO"
        in result.stdout
    )

    assert tuple(
        working_directory.iterdir()
    ) == before

    assert output_root.is_dir()


def test_part_h_rejects_repository_output_root_without_writing() -> None:
    import pytest

    from circuit_families.stage7 import (
        Stage7PortableE2EError,
        run_portable_stage7_fixture,
    )

    forbidden = (
        ROOT
        / "forbidden-stage7a-output"
    )

    assert not forbidden.exists()

    with pytest.raises(
        Stage7PortableE2EError,
        match="system temporary",
    ):
        run_portable_stage7_fixture(
            output_root=forbidden,
            repository_root=ROOT,
            teacher_seed=0,
            phase="stable post-grokking",
            distillation_config=_part_h_config(),
        )

    assert not forbidden.exists()
