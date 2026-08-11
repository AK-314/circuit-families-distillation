import json

import pytest

from circuit_families.analysis.downstream_finalization import (
    read_last_json_object,
    stage20_training_projection,
    validate_stage18_comparison,
)


def comparison():
    return {
        "stage18_run_id": "run",
        "passed": True,
        "compared_file_count": 818386,
        "deterministic_mismatch_count": 0,
        "mismatches": [],
        "archive_inventory_count": 35,
        "archive_inventory_mismatch_count": 0,
        "normalized_metadata_file_count": 1224,
        "comparison_policy": {"normalized_metadata_wrappers": {}},
    }


def test_read_last_json_object_after_progress(tmp_path):
    path = tmp_path / "comparison.log"
    value = comparison()
    path.write_text("progress\n" + json.dumps(value, indent=2) + "\n", encoding="utf-8")
    assert read_last_json_object(path) == value


def test_validate_stage18_comparison_rejects_mismatch():
    value = comparison()
    value["deterministic_mismatch_count"] = 1
    with pytest.raises(ValueError, match="deterministic_mismatch_count"):
        validate_stage18_comparison(value, run_id="run")


def test_stage20_training_projection_ignores_bookkeeping():
    runs = [
        {
            "model_seed": seed,
            "first_ten_percent_test_step": 100 + seed,
            "stable_post_step": 200 + seed,
            "manifest_sha256": f"ignored-{seed}",
        }
        for seed in range(5)
    ]
    projected = stage20_training_projection({"runs": runs})
    assert projected[3] == {
        "model_seed": 3,
        "first_ten_percent_test_step": 103,
        "stable_post_step": 203,
    }
