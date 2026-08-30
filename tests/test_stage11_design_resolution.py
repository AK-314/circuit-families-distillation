from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from circuit_families.stage11_design_resolution import (
    Stage11DesignResolutionError,
    load_stage11_resolution_record,
    validate_stage11_resolution_record,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "followup" / "manifests" / "stage11_red_team_resolution_v1.json"


def canonical() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def validate(record: dict) -> None:
    validate_stage11_resolution_record(record, repo_root=ROOT)


def test_canonical_stage11_resolution_record_passes() -> None:
    record = load_stage11_resolution_record(
        MANIFEST,
        repo_root=ROOT,
    )
    assert record["coverage"]["red_team_row_count"] == 20
    assert record["coverage"]["rd_count"] == 14
    assert record["scientific_execution"] is False


def test_missing_red_team_row_is_rejected() -> None:
    record = canonical()
    record["red_team_items"].pop()

    with pytest.raises(
        Stage11DesignResolutionError,
        match="missing red-team rows",
    ):
        validate(record)


def test_missing_rd_identifier_is_rejected() -> None:
    record = canonical()
    record["rd_items"] = [item for item in record["rd_items"] if item["decision_id"] != "RD-007"]

    with pytest.raises(
        Stage11DesignResolutionError,
        match="missing RD identifiers",
    ):
        validate(record)


def test_duplicate_rd_identifier_is_rejected() -> None:
    record = canonical()
    record["rd_items"].append(copy.deepcopy(record["rd_items"][0]))

    with pytest.raises(
        Stage11DesignResolutionError,
        match="duplicate RD identifiers",
    ):
        validate(record)


def test_accepted_item_without_implementation_consequence_is_rejected() -> None:
    record = canonical()
    accepted = next(
        item
        for item in record["red_team_items"]
        if item["disposition"].lower().startswith("accepted")
    )
    accepted["required_implementation_packages"] = []

    with pytest.raises(
        Stage11DesignResolutionError,
        match="required_implementation_packages must not be empty",
    ):
        validate(record)


def test_rejected_claim_cannot_be_silently_reintroduced() -> None:
    record = canonical()
    rejected = next(
        item
        for item in record["red_team_items"]
        if item["disposition"].lower().startswith("rejected")
    )
    target = next(item for item in record["red_team_items"] if item is not rejected)
    target["permitted_claim"] = rejected["prohibited_claim"]

    with pytest.raises(
        Stage11DesignResolutionError,
        match="rejected claim silently reintroduced elsewhere",
    ):
        validate(record)


def test_production_blocking_rd_cannot_be_demoted() -> None:
    record = canonical()
    record["rd_items"][0]["blocks_stage15"] = False

    with pytest.raises(
        Stage11DesignResolutionError,
        match="production-blocking and must block Stage 15",
    ):
        validate(record)


def test_resolved_rd_without_resolution_record_is_rejected() -> None:
    record = canonical()
    item = record["rd_items"][0]
    item["status"] = "resolved"

    with pytest.raises(
        Stage11DesignResolutionError,
        match="marked resolved without a resolution record",
    ):
        validate(record)


def test_historical_stage2_hash_mutation_is_rejected() -> None:
    record = canonical()
    record["historical_stage2_guard"]["sha256"] = "0" * 64

    with pytest.raises(
        Stage11DesignResolutionError,
        match="historical Stage 2 register mutation detected",
    ):
        validate(record)


def test_full_factorial_cannot_be_reintroduced() -> None:
    record = canonical()
    record["full_factorial_required"] = True

    with pytest.raises(
        Stage11DesignResolutionError,
        match="literal full factorial must remain rejected",
    ):
        validate(record)


def test_all_rd_ids_are_exactly_once_and_ordered() -> None:
    record = canonical()
    assert [item["decision_id"] for item in record["rd_items"]] == [
        f"RD-{number:03d}" for number in range(1, 15)
    ]


def test_stage2_register_remains_unmodified_in_git_diff() -> None:
    import subprocess

    changed = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--name-only",
            "--",
            "followup/configs/stage2_unresolved_decisions_v1.json",
        ],
        text=True,
    ).strip()

    assert changed == ""


CANDIDATES = ROOT / "followup" / "configs" / "stage11_post_red_team_design_candidates_v1.json"


def candidate_record() -> dict:
    return json.loads(CANDIDATES.read_text(encoding="utf-8"))


def test_stage11_part_c_candidates_pass() -> None:
    from circuit_families.stage11_design_resolution import load_stage11_design_candidates

    record = load_stage11_design_candidates(CANDIDATES)
    assert record["population"]["population_unit"] == "teacher_seed"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("population", "exact_teacher_count"), 15, "exact teacher count must remain open"),
        (("tasks", 1, "modulus"), 113, "Task 2 production modulus must remain open"),
        (("tasks", 2, "formula_terms"), [{"coefficient": 1}], "Task 3 formula must remain open"),
        (
            ("architectures", "exact_family_roster"),
            ["x"],
            "exact architecture roster must remain open",
        ),
        (("component_bases", 2, "block_count"), 8, "MLP block count must remain open"),
        (("component_bases", 3, "rotation_count"), 4, "rotation count must remain open"),
    ],
)
def test_stage11_part_c_rejects_premature_numeric_freeze(path, value, message) -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_design_candidates,
    )

    record = candidate_record()
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(Stage11DesignResolutionError, match=message):
        validate_stage11_design_candidates(record)


def test_stage11_part_c_rejects_full_factorial() -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_design_candidates,
    )

    record = candidate_record()
    record["architectures"]["literal_full_factorial"] = True

    with pytest.raises(Stage11DesignResolutionError, match="literal full factorial"):
        validate_stage11_design_candidates(record)


def test_stage11_part_c_rejects_stage4_identity_mutation() -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_design_candidates,
    )

    record = candidate_record()
    record["historical_stage4_identity_mutated"] = True

    with pytest.raises(Stage11DesignResolutionError, match="Stage 4 identity"):
        validate_stage11_design_candidates(record)


def test_stage11_part_d_candidates_pass() -> None:
    from circuit_families.stage11_design_resolution import load_stage11_complete_candidates

    load_stage11_complete_candidates(CANDIDATES)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "endpoint1",
            "primary_fidelity_threshold",
            0.99,
            "primary fidelity threshold must remain open",
        ),
        ("endpoint2", "component_cap", 258, "Endpoint 2 component_cap must remain open"),
    ],
)
def test_stage11_part_d_rejects_endpoint_numeric_freeze(section, field, value, message) -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_part_d,
    )

    record = candidate_record()
    record["endpoints"][section][field] = value
    with pytest.raises(Stage11DesignResolutionError, match=message):
        validate_stage11_part_d(record)


def test_stage11_part_d_requires_independent_discovery() -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_part_d,
    )

    record = candidate_record()
    record["discovery"]["cosmetic_restart_of_inherited_method_acceptable"] = True
    with pytest.raises(Stage11DesignResolutionError, match="cosmetic restart"):
        validate_stage11_part_d(record)


def test_stage11_part_d_requires_all_four_nulls() -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_part_d,
    )

    record = candidate_record()
    record["packing_calibration"]["required_nulls"].pop()
    with pytest.raises(Stage11DesignResolutionError, match="four packing"):
        validate_stage11_part_d(record)


def test_stage11_part_d_requires_all_fourier_controls() -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_part_d,
    )

    record = candidate_record()
    record["fourier_interchange"]["required_controls"].pop()
    with pytest.raises(Stage11DesignResolutionError, match="Fourier control"):
        validate_stage11_part_d(record)


def test_stage11_part_d_rejects_outcome_conditioned_fourier_pairs() -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_part_d,
    )

    record = candidate_record()
    record["fourier_interchange"]["pair_selection_may_use_candidate_outcomes"] = True
    with pytest.raises(Stage11DesignResolutionError, match="pair selection"):
        validate_stage11_part_d(record)


def test_stage11_part_d_rejects_tier3_preemption() -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_part_d,
    )

    record = candidate_record()
    record["tiering"]["tier3"]["may_run_before_tier1_complete_and_tier2_minimum_secure"] = True
    with pytest.raises(Stage11DesignResolutionError, match="Tier 3"):
        validate_stage11_part_d(record)


def test_stage11_part_d_keeps_resource_numbers_open() -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_part_d,
    )

    record = candidate_record()
    record["resources"]["cluster_concurrency"] = 16
    with pytest.raises(Stage11DesignResolutionError, match="cluster_concurrency must remain open"):
        validate_stage11_part_d(record)


def test_stage11_part_e_report_skeleton_has_no_values() -> None:
    from circuit_families.stage11_design_resolution import (
        validate_stage11_report_skeleton,
    )

    record = candidate_record()
    validate_stage11_report_skeleton(record)


def test_stage11_part_e_rejects_scientific_report_values() -> None:
    from circuit_families.stage11_design_resolution import (
        Stage11DesignResolutionError,
        validate_stage11_report_skeleton,
    )

    record = candidate_record()
    record["planned_report_skeleton"]["scientific_values_present"] = True

    with pytest.raises(
        Stage11DesignResolutionError,
        match="scientific values",
    ):
        validate_stage11_report_skeleton(record)
