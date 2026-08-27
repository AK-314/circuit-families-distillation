from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REQUEST = REPO / "followup/configs/stage7b/registered_fixture_request_v1.json"


def _load() -> dict:
    return json.loads(REQUEST.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage7b_request_is_technical_only_and_stage8_closed() -> None:
    r = _load()
    assert r["technical_only"] is True
    assert r["scientific_data"] is False
    assert r["production_default"] is False
    assert r["ud_resolution"] is False
    assert r["stage8_execution"] is False


def test_stage7b_registered_teacher_identity_is_exact() -> None:
    r = _load()["registered_teacher"]
    assert r["teacher_seed"] == 0
    assert r["phase_label"] == "stable post-grokking"
    assert r["canonical_run_id"] == "stage18-main-training-s0-58b8c1235464"
    assert r["training_step"] == 5900
    assert r["checkpoint_path_relative_to_predecessor"] == (
        "checkpoints/stage18-main-training-s0-58b8c1235464/step_00005900.pt"
    )
    assert r["checkpoint_sha256"] == (
        "f8f0fc43c559aa6f63e796a62627b8657a789a34f558b81ee33fbc6cc8968bc6"
    )
    assert r["complete_domain_size"] == 12769
    assert r["domain_order"] == "lexicographic_modular_addition_inputs"


def test_stage7b_one_hard_one_soft_and_failed_attempt_retention() -> None:
    r = _load()["attempt_roster"]
    assert r["hard_target_attempts"] == 1
    assert r["soft_target_attempts"] == 1
    assert r["attempts_are_unpooled"] is True
    assert r["failed_attempt_is_retained"] is True
    assert r["failed_attempt_is_not_retried_for_eligibility"] is True
    assert r["student_discovery_requires_eligible_sealed_student"] is True
    assert r["teacher_direct_discovery_is_unconditional"] is True


def test_stage7b_minimal_shared_trainer_workload_is_fixed() -> None:
    r = _load()["shared_trainer_workload"]
    assert r["native_positive_work_units_per_attempt"] == 1
    assert r["native_work_unit_safety_ceiling_per_attempt"] == 1
    assert r["smallest_nonzero_supported_integer_workload"] is True
    assert r["reuse_accepted_shared_trainer"] is True
    assert r["training_length_post_outcome_tuning"] is False


def test_stage7b_hard_soft_separation_and_centred_logit_fidelity() -> None:
    r = _load()["targets_and_fidelity"]
    assert r["hard_target"] is True
    assert r["soft_target"] is True
    assert r["hard_soft_pooling"] is False
    assert r["predictive_fidelity"] == "centred_logit_only"
    assert r["complete_domain_exact_evaluation"] is True


def test_stage7b_two_adapters_and_separate_native_exact_budgets() -> None:
    r = _load()["discovery"]
    assert r["accepted_adapter_count"] == 2
    assert r["native_and_exact_budgets_are_separate"] is True
    assert "stage6d" in r["adapter_roster"]
    assert "stage6d" in r["method_native_budgets"]
    assert "stage6a" in r["exact_evaluation_allowances"]


def test_stage7b_accepted_authority_hashes_match() -> None:
    r = _load()["accepted_authority"]
    refs = [
        r["stage7a_technical_request"],
        r["stage6e_endpoint2_policy"],
        *r["stage6a_exact_ledger_endpoint1_authorities"],
        *r["stage6d_discovery_native_budget_authorities"],
    ]
    assert refs
    for ref in refs:
        p = REPO / ref["path"]
        assert p.is_file()
        assert _sha(p) == ref["sha256"]


def test_stage7b_device_batches_and_storage_ceiling_are_predeclared() -> None:
    r = _load()
    e = r["execution_engineering"]
    assert e["device"] == "mps"
    assert e["device_selected_before_endpoint_output"] is True
    assert e["teacher_forward_batch_size"] == 256
    assert e["exact_evaluation_batch_size"] == 256
    assert e["batch_size_semantics"] == (
        "mathematically_invariant_full_domain_partition_only"
    )
    assert e["batch_reduction_allowed_only_after_documented_memory_failure"] is True
    assert e["one_worker"] is True

    s = r["resource_envelope"]
    assert s["predicted_peak_resident_bytes"] == 2147483648
    assert s["predicted_peak_bytes_per_runtime_root"] == 1073741824
    assert s["combined_source_and_reproduction_local_ceiling_bytes"] == 4294967296
    assert s["stop_before_exceeding_combined_local_ceiling"] is True
    assert s["git_lfs_allowed"] is False


def test_stage7b_endpoint_outputs_are_excluded() -> None:
    r = _load()
    assert r["endpoints"]["print_endpoint_values"] is False
    assert r["endpoints"]["interpret_endpoint_values"] is False

    f = r["scientific_firewall"]
    assert f["all_endpoint_like_outputs_lifecycle_state"] == "excluded"
    assert f["primary_input_eligible"] is False
    assert f["regeneration_required_after_definitive_freeze"] is True
    assert f["teacher_seed_population_unit_retained"] is True
    assert f["scientific_interpretation"] is False
    assert f["request_mutation_after_endpoint_output"] is False


def test_stage7b_request_contains_no_private_absolute_path_or_result_value() -> None:
    text = REQUEST.read_text()
    assert "/Users/" not in text
    assert "/private/" not in text
    for pattern in (
        r'"endpoint1_value"\s*:',
        r'"endpoint2_value"\s*:',
        r'"packing_lower_bound_value"\s*:',
        r'"global_minimum_value"\s*:',
    ):
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_stage7b_runtime_boundary_is_local_and_untracked() -> None:
    r = _load()["runtime_boundary"]
    assert r["source_root_relative"].startswith("followup/local/stage7b/")
    assert r["reproduction_root_must_be_separate"] is True
    assert r["runtime_outputs_tracked"] is False
    assert r["predecessor_root_is_runtime_argument"] is True
    assert r["private_absolute_path_in_tracked_content"] is False
