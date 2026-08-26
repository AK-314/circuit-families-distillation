from __future__ import annotations

import copy
import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage7 import (
    EXPECTED_PIPELINE_STEPS,
    Stage7InventoryReportingError,
    TechnicalDistillationFixtureConfig,
    assert_rejected_as_primary_scientific_input,
    build_pipeline_reproduction_record,
    build_technical_run_manifest,
    compare_reproduction_records,
    load_technical_run_request,
    run_portable_stage7_fixture,
)

ROOT = Path(__file__).resolve().parents[1]

REQUEST_PATH = (
    ROOT
    / "followup/configs/stage7/"
    "technical_run_request_v1.json"
)

STAGE3_REGISTRY_PATH = (
    ROOT
    / "followup/manifests/"
    "stage3_teacher_registry_v1.json"
)

EXCLUSION_REGISTER_PATH = (
    ROOT
    / "followup/manifests/"
    "stage2_excluded_development_register_v1.json"
)

DISCOVERY_PROFILES_PATH = (
    ROOT
    / "followup/configs/stage6d/"
    "technical_discovery_profiles_v1.json"
)

ENDPOINT2_POLICY_PATH = (
    ROOT
    / "followup/configs/stage6e/"
    "technical_endpoint2_policy_v1.json"
)


def _stage3() -> Stage3AvailabilityIndex:
    return Stage3AvailabilityIndex.from_registry(
        json.loads(
            STAGE3_REGISTRY_PATH.read_text(
                encoding="utf-8"
            )
        )
    )


def _technical_config() -> TechnicalDistillationFixtureConfig:
    return TechnicalDistillationFixtureConfig(
        hard_learning_rate=0.01,
        soft_learning_rate=0.01,
        technical_stop_step=1,
        technical_safety_step_limit=1,
        soft_tolerance=1.0,
    )


@pytest.fixture(scope="module")
def integrated_record(
    tmp_path_factory: pytest.TempPathFactory,
):
    root = (
        tmp_path_factory.mktemp(
            "stage7a-part-i-record"
        )
        / "record"
    )

    return build_pipeline_reproduction_record(
        physical_root=root,
        repository_root=ROOT,
        stage3=_stage3(),
        teacher_seed=0,
        phase="stable post-grokking",
        run_request=load_technical_run_request(
            REQUEST_PATH
        ),
        distillation_config=_technical_config(),
        discovery_profiles_path=DISCOVERY_PROFILES_PATH,
        endpoint2_policy_path=ENDPOINT2_POLICY_PATH,
        exclusion_register=json.loads(
            EXCLUSION_REGISTER_PATH.read_text(
                encoding="utf-8"
            )
        ),
    )


@pytest.fixture(scope="module")
def portable_report(
    tmp_path_factory: pytest.TempPathFactory,
):
    output_root = (
        tmp_path_factory.mktemp(
            "stage7a-part-i-portable"
        )
        / "output"
    )

    return run_portable_stage7_fixture(
        output_root=output_root,
        repository_root=ROOT,
        teacher_seed=0,
        phase="stable post-grokking",
        distillation_config=_technical_config(),
    )


def test_part_i_complete_canonical_dag_is_connected_in_order() -> None:
    request = load_technical_run_request(
        REQUEST_PATH
    )

    manifest = build_technical_run_manifest(
        request,
        stage3=_stage3(),
        repository_root=ROOT,
    )

    steps = manifest.lifecycle.topological_steps()

    assert tuple(
        step.step_id
        for step in steps
    ) == EXPECTED_PIPELINE_STEPS

    assert len(steps) == 10

    completed = set()

    for step in steps:
        assert set(
            step.dependencies
        ).issubset(
            completed
        )

        completed.add(
            step.step_id
        )

    assert completed == set(
        EXPECTED_PIPELINE_STEPS
    )


def test_part_i_teacher_direct_hard_soft_and_failed_attempts_remain_distinct(
    integrated_record,
) -> None:
    distillation = integrated_record[
        "distillation"
    ]

    runs = integrated_record[
        "discovery_runs"
    ]

    inventory = integrated_record[
        "inventory"
    ]

    assert distillation["hard"]["attempt_count"] == 2
    assert distillation["soft"]["attempt_count"] == 2

    assert distillation["hard"]["classifications"] == [
        "eligible",
        "training_failure",
    ]

    assert distillation["soft"]["statuses"] == [
        "eligible",
        "failed",
    ]

    assert {
        run["subject_role"]
        for run in runs
    } == {
        "direct_teacher",
        "hard_target_student",
        "soft_target_student",
    }

    assert {
        row["distillation_condition"]
        for row in inventory["rows"]
    } == {
        "teacher_direct",
        "hard",
        "soft",
    }

    assert {
        row["subject_state"]
        for row in inventory["rows"]
    } == {
        "teacher_direct",
        "eligible",
        "failed",
    }

    assert inventory["hard_soft_pooled"] is False


def test_part_i_failed_attempts_are_retained_but_passed_only_discovery_applies(
    integrated_record,
) -> None:
    runs = integrated_record[
        "discovery_runs"
    ]

    inventory = integrated_record[
        "inventory"
    ]

    failed_subject_ids = {
        "technical-hard-failed-student",
        "technical-soft-failed-student",
    }

    assert not (
        failed_subject_ids
        & {
            run["subject_id"]
            for run in runs
        }
    )

    failed_rows = [
        row
        for row in inventory["rows"]
        if row["subject_state"] == "failed"
    ]

    assert len(failed_rows) == 4

    assert all(
        row["method_state"] == "missing"
        for row in failed_rows
    )

    assert all(
        row["endpoint1"] is None
        and row["endpoint2"] is None
        for row in failed_rows
    )


def test_part_i_native_and_exact_budgets_are_separate_and_both_endpoints_exist(
    integrated_record,
) -> None:
    runs = integrated_record[
        "discovery_runs"
    ]

    assert len(runs) == 6

    native_units = {
        run["discovery_method"]: run[
            "native_budget_unit"
        ]
        for run in runs
    }

    assert native_units == {
        "diversity_forced": "restart_ranked_proposals",
        "greedy_deletion": "ranked_component_proposals",
    }

    assert len(
        set(
            native_units.values()
        )
    ) == 2

    for run in runs:
        assert run["native_budget_allowance"] in {
            5,
            7,
        }

        assert run["exact_evaluation_allowance"] == 3

        assert (
            run["exact_ledger_sha256"]
            == run["reconstructed_ledger_sha256"]
        )

        assert run["endpoint1"]["global_minimum_claim"] is False

        assert (
            run["endpoint2"]["semantics"]
            == "procedure_dependent_packing_lower_bound"
        )

        assert run["endpoint2"]["packing_lower_bound"] == 2


def test_part_i_inventory_preserves_teacher_seed_hierarchy(
    integrated_record,
) -> None:
    inventory = integrated_record[
        "inventory"
    ]

    analysis = integrated_record[
        "analysis_bridge"
    ]

    assert inventory["population_unit"] == "teacher_seed"

    assert inventory["student_member_unit"] == (
        "student_initialization"
    )

    assert (
        inventory[
            "student_initializations_are_population_replicates"
        ]
        is False
    )

    assert inventory["hard_soft_pooled"] is False

    assert len(
        inventory["rows"]
    ) == 10

    assert analysis["population_unit"] == "teacher_seed"

    assert analysis["student_member_unit"] == (
        "student_initialization"
    )

    assert analysis["hard_soft_tables_separate"] is True

    assert analysis["hard_soft_pooled"] is False

    assert (
        analysis["student_initializations_are_population_replicates"]
        is False
    )

    assert analysis["stage5d_reducer_reimplemented"] is False


def test_part_i_all_endpoint_like_outputs_are_excluded_and_primary_rejected(
    integrated_record,
) -> None:
    exclusions = integrated_record[
        "excluded_endpoint_records"
    ]

    assert len(exclusions) == 12

    for entry in exclusions:
        assert entry["endpoint_values_emitted"] is True
        assert entry["primary_analysis_eligible"] is False
        assert entry["scientific_selection_eligible"] is False
        assert entry["regeneration_required"] is True
        assert entry["promotion_in_place_permitted"] is False

        with pytest.raises(
            Stage7InventoryReportingError,
            match="rejected as primary scientific input",
        ):
            assert_rejected_as_primary_scientific_input(
                entry
            )


def test_part_i_resume_merge_is_deterministic_and_nonduplicating(
    integrated_record,
) -> None:
    resume = integrated_record[
        "resume_merge"
    ]

    assert resume["hard_completed_action"] == "skip_completed"
    assert resume["soft_completed_action"] == "skip_completed"

    assert resume["hard_failed_action"] == (
        "reject_failed_attempt"
    )

    assert resume["soft_failed_action"] == (
        "reject_failed_attempt"
    )

    assert resume["cross_attempt_transfer_rejected"] is True
    assert resume["duplicate_merge_rejected"] is True
    assert resume["merge_order_independent"] is True

    assert (
        resume["before_resume_counts"]
        == resume["after_resume_counts"]
    )

    assert resume["completed_attempts_duplicated"] is False
    assert resume["failed_attempts_reexecuted"] is False
    assert resume["native_charges_duplicated"] is False
    assert resume["exact_evaluations_duplicated"] is False
    assert resume["endpoints_duplicated"] is False
    assert resume["resume_state_transferred"] is False


def test_part_i_reproduction_success_and_failure_are_both_mismatch_sensitive(
    integrated_record,
    portable_report,
) -> None:
    assert portable_report["reproduction_matched"] is True

    assert portable_report[
        "reproduction_mismatch_paths"
    ] == []

    source = {
        "identity": integrated_record[
            "run_request_identity"
        ],
        "inventory_sha256": integrated_record[
            "inventory"
        ]["sha256"],
        "endpoint2": {
            "packing_lower_bound": integrated_record[
                "discovery_runs"
            ][0]["endpoint2"]["packing_lower_bound"],
        },
    }

    reproduced = copy.deepcopy(
        source
    )

    matched = compare_reproduction_records(
        source,
        reproduced,
    )

    assert matched.matched is True
    assert matched.mismatch_paths == ()

    reproduced[
        "endpoint2"
    ][
        "packing_lower_bound"
    ] += 1

    mismatch = compare_reproduction_records(
        source,
        reproduced,
    )

    assert mismatch.matched is False

    assert any(
        path.startswith(
            "$.endpoint2.packing_lower_bound:value:"
        )
        for path in mismatch.mismatch_paths
    )


def test_part_i_portable_fixture_reports_all_acceptance_boundaries(
    portable_report,
) -> None:
    assert portable_report["pipeline_step_count"] == 10

    assert tuple(
        step["step_id"]
        for step in portable_report[
            "pipeline_steps"
        ]
    ) == EXPECTED_PIPELINE_STEPS

    assert portable_report["eligible_path_count"] == 2
    assert portable_report["failed_path_count"] == 2
    assert portable_report["failed_subject_discovery_count"] == 0

    assert portable_report[
        "excluded_endpoint_output_count"
    ] == 12

    assert portable_report[
        "primary_analysis_eligible_count"
    ] == 0

    assert portable_report[
        "scientific_selection_eligible_count"
    ] == 0

    assert portable_report[
        "post_freeze_regeneration_required"
    ] is True

    assert portable_report["resume_counts_unchanged"] is True

    assert portable_report["registered_fixture_execution"] is False
    assert portable_report["scientific_execution"] is False
    assert portable_report["stage8_execution"] is False


def test_part_i_registered_fixture_contract_is_ready_without_checkpoint_access() -> None:
    request = load_technical_run_request(
        REQUEST_PATH
    )

    teacher = request.teacher_reference

    assert is_dataclass(
        teacher
    )

    changes = {}

    kind_field = None
    uri_field = None

    for field in fields(
        teacher
    ):
        value = getattr(
            teacher,
            field.name,
        )

        if value == "injected_fixture":
            kind_field = field.name
            changes[
                field.name
            ] = "registered_reference"

        if (
            isinstance(
                value,
                str,
            )
            and value.startswith(
                "synthetic://"
            )
        ):
            uri_field = field.name
            changes[
                field.name
            ] = (
                "registered://stage7/"
                "teacher-checkpoint/pending"
            )

    assert kind_field is not None
    assert uri_field is not None

    registered_teacher = replace(
        teacher,
        **changes,
    )

    registered_request = replace(
        request,
        teacher_reference=registered_teacher,
    )

    manifest = build_technical_run_manifest(
        registered_request,
        stage3=_stage3(),
        repository_root=ROOT,
    )

    registered_values = [
        getattr(
            registered_teacher,
            field.name,
        )
        for field in fields(
            registered_teacher
        )
    ]

    assert "registered_reference" in registered_values

    assert any(
        isinstance(
            value,
            str,
        )
        and value.startswith(
            "registered://"
        )
        for value in registered_values
    )

    assert tuple(
        step.step_id
        for step in manifest.lifecycle.topological_steps()
    ) == EXPECTED_PIPELINE_STEPS

    serialized = json.dumps(
        registered_request.to_mapping(),
        sort_keys=True,
    )

    assert "registered://" in serialized

    assert "/Users/" not in serialized
    assert "\\Users\\" not in serialized
    assert "private_checkpoint_access=none" in serialized

    for forbidden_private_evidence in (
        "private_checkpoint_access=granted",
        "private_checkpoint_path",
        "private_checkpoint_bytes",
        "private://",
    ):
        assert forbidden_private_evidence not in serialized

    assert registered_request.request_sha256 != (
        request.request_sha256
    )

    request_mapping = registered_request.to_mapping()

    assert request_mapping["classification"] == "synthetic_technical_only"
    assert request_mapping["scientific_data"] is False
    assert request_mapping["production_eligible"] is False



def test_part_l_validate_only_cli_is_exposed_read_only_and_fixture_free(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    cli = (
        ROOT
        / "scripts/"
        "validate_stage7_technical_integration.py"
    )

    help_result = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--help",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert help_result.returncode == 0
    assert "--validate-only" in help_result.stdout

    before = tuple(
        tmp_path.iterdir()
    )

    result = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--validate-only",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    after = tuple(
        tmp_path.iterdir()
    )

    assert result.returncode == 0, (
        result.stdout
        + result.stderr
    )

    assert "STAGE7A_VALIDATE_ONLY=PASS" in result.stdout
    assert "VALIDATION_MODE=CONTRACT_READINESS_ONLY" in result.stdout
    assert "PIPELINE_STEP_COUNT=10" in result.stdout
    assert "TECHNICAL_FIXTURE_EXECUTION=NO" in result.stdout
    assert "REGISTERED_FIXTURE_EXECUTION=NO" in result.stdout
    assert "OUTPUT_WRITTEN=NO" in result.stdout
    assert "SCIENTIFIC_DATA=NO" in result.stdout
    assert "PRODUCTION_ELIGIBLE=NO" in result.stdout
    assert "PRODUCTION_DEFAULT=NO" in result.stdout
    assert "UD_RESOLUTIONS=0" in result.stdout
    assert "STAGE8_EXECUTION=NO" in result.stdout

    assert before == after


def test_part_l_validate_only_does_not_call_portable_fixture(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib.util

    cli = (
        ROOT
        / "scripts/"
        "validate_stage7_technical_integration.py"
    )

    spec = importlib.util.spec_from_file_location(
        "stage7a_validate_only_cli_test",
        cli,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    def forbidden_fixture_execution(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "portable fixture executed during --validate-only"
        )

    monkeypatch.setattr(
        module,
        "run_portable_stage7_fixture",
        forbidden_fixture_execution,
    )

    assert module.main(
        [
            "--validate-only",
        ]
    ) == 0

    output = capsys.readouterr().out

    assert "STAGE7A_VALIDATE_ONLY=PASS" in output
    assert "TECHNICAL_FIXTURE_EXECUTION=NO" in output
    assert "REGISTERED_FIXTURE_EXECUTION=NO" in output
    assert "OUTPUT_WRITTEN=NO" in output
