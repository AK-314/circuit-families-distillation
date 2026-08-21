from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from circuit_families.stage5d import (
    MISSINGNESS_KINDS,
    TechnicalAnalysisProfileError,
    TechnicalAnalysisProfileSet,
    build_missingness_records,
    build_phase_contrasts,
    build_stage5d_output_bundle,
    build_student_cell_summaries,
    build_teacher_student_contrasts,
    extract_direct_teacher_values,
    load_and_normalize_ingestion,
    load_technical_analysis_profile_set,
)

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = (
    ROOT
    / "tests/fixtures/stage5d/"
    "synthetic_ingestion_envelope_v1.json"
)
PROFILES = (
    ROOT
    / "followup/configs/stage5d/"
    "technical_analysis_profiles_v1.json"
)


def _normalized():
    return load_and_normalize_ingestion(ENVELOPE)


def _profile(profile_id: str = "fixture_median_min2"):
    return load_technical_analysis_profile_set(PROFILES).require(
        profile_id
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            {"production_eligible": True},
            "cannot be production eligible",
        ),
        (
            {"synthetic_only": False},
            "must be synthetic-only",
        ),
        (
            {"classification": "scientific_result"},
            "classification is not synthetic-only",
        ),
        (
            {"selection_basis": "selected_from_observed_effects"},
            "forbid scientific-effect selection",
        ),
    ),
)
def test_adversarial_profile_firewall_mutations_are_rejected(
    mutation,
    message,
) -> None:
    raw = json.loads(PROFILES.read_text(encoding="utf-8"))
    raw["profiles"][0].update(mutation)

    with pytest.raises(TechnicalAnalysisProfileError, match=message):
        TechnicalAnalysisProfileSet.from_mapping(raw)


def test_failed_and_ineligible_sources_never_become_numeric_members() -> None:
    normalized = _normalized()
    bundle = build_stage5d_output_bundle(normalized, _profile())

    failed_attempt_ids = {
        str(record["attempt_id"])
        for record in normalized["student_attempts"]
        if record["outcome"] == "failed"
    }
    ineligible_record_ids = {
        str(record["eligibility_id"])
        for record in normalized["eligibility_records"]
        if record["status"] in {"ineligible", "inapplicable"}
    }
    assert failed_attempt_ids
    assert ineligible_record_ids

    student_rows = (
        bundle["output_objects"]["hard_student_summaries"]["rows"]
        + bundle["output_objects"]["soft_student_summaries"]["rows"]
    )
    numeric_member_sources = {
        source_id
        for row in student_rows
        if row["state"] == "defined"
        for source_id in row["source_record_ids"]
    }
    assert failed_attempt_ids.isdisjoint(numeric_member_sources)
    assert ineligible_record_ids.isdisjoint(numeric_member_sources)

    failure_sources = set(
        bundle["output_objects"]["failure_accounting"][
            "source_record_ids"
        ]
    )
    missingness_sources = set(
        bundle["output_objects"]["missingness_summaries"][
            "source_record_ids"
        ]
    )
    assert failed_attempt_ids <= failure_sources
    assert ineligible_record_ids <= missingness_sources


def test_all_missingness_states_remain_distinct_under_adversarial_inputs() -> None:
    normalized = _normalized()
    profile = _profile()
    cells = list(build_student_cell_summaries(normalized, profile))
    teachers = list(extract_direct_teacher_values(normalized))

    base_phase = build_phase_contrasts(cells, profile)
    base_teacher_student = build_teacher_student_contrasts(cells, teachers)
    base_records = build_missingness_records(
        normalized,
        cells,
        base_phase,
        base_teacher_student,
    )

    min3_profile = _profile("fixture_mean_min3")
    min3_cells = build_student_cell_summaries(normalized, min3_profile)
    min3_records = build_missingness_records(
        normalized,
        min3_cells,
        build_phase_contrasts(min3_cells, min3_profile),
        build_teacher_student_contrasts(min3_cells, teachers),
    )

    missing_teacher = next(
        row
        for row in teachers
        if row.key.teacher_seed == 0
        and row.key.phase == "phase_early"
        and row.key.method_id == "method_beam"
        and row.key.endpoint_id == "endpoint_2"
    )
    teachers.remove(missing_teacher)

    incompatible_index = next(
        index
        for index, row in enumerate(cells)
        if row.key.teacher_seed == 1
        and row.key.phase == "phase_late"
        and row.key.distillation_condition == "hard"
        and row.key.method_id == "method_greedy"
        and row.key.endpoint_id == "endpoint_1"
    )
    original = cells[incompatible_index]
    cells[incompatible_index] = replace(
        original,
        key=replace(
            original.key,
            fidelity_id="adversarial_incompatible_fidelity",
        ),
    )

    adversarial_records = build_missingness_records(
        normalized,
        cells,
        build_phase_contrasts(cells, profile),
        build_teacher_student_contrasts(cells, teachers),
    )
    observed = {
        record.kind
        for record in (
            *base_records,
            *min3_records,
            *adversarial_records,
        )
    }

    assert observed == set(MISSINGNESS_KINDS)
    assert observed == {
        "absent",
        "unavailable",
        "failed",
        "ineligible",
        "insufficient",
        "incompatible",
        "inapplicable",
        "unresolved",
    }


def test_endpoint_boundary_values_survive_deterministic_output_tables() -> None:
    bundle = build_stage5d_output_bundle(_normalized(), _profile())
    objects = bundle["output_objects"]

    teacher_rows = objects["direct_teacher_summaries"]["rows"]
    student_rows = (
        objects["hard_student_summaries"]["rows"]
        + objects["soft_student_summaries"]["rows"]
    )
    member_values = [
        value
        for row in student_rows
        for _, value in row["member_values"]
    ]

    assert any(
        row["key"]["endpoint_id"] == "endpoint_1"
        and row["state"] == "defined"
        and row["value"] == 1.0
        for row in teacher_rows
    ) or 1.0 in member_values
    assert any(
        row["key"]["endpoint_id"] == "endpoint_2"
        and row["state"] == "defined"
        and row["value"] == 0
        for row in teacher_rows
    ) or 0 in member_values


def test_hashseed_does_not_change_normalized_output_hashes_or_tables() -> None:
    code = f"""
from pathlib import Path
from circuit_families.stage5d import (
    build_stage5d_output_bundle,
    canonical_json_bytes,
    load_and_normalize_ingestion,
    load_technical_analysis_profile_set,
    normalized_sha256,
)
normalized = load_and_normalize_ingestion(Path({str(ENVELOPE)!r}))
profile = load_technical_analysis_profile_set(
    Path({str(PROFILES)!r})
).require("fixture_median_min2")
bundle = build_stage5d_output_bundle(normalized, profile)
payload = {{
    "normalized": normalized,
    "normalized_sha256": normalized_sha256(normalized),
    "bundle_sha256": bundle["sha256"],
    "object_hashes": {{
        key: value["sha256"]
        for key, value in bundle["output_objects"].items()
    }},
    "output_tables": {{
        key: value["rows"]
        for key, value in bundle["output_objects"].items()
    }},
}}
print(canonical_json_bytes(payload).decode("utf-8"), end="")
"""
    outputs: list[bytes] = []

    for seed in ("7", "31337"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT.parent,
            env=environment,
            capture_output=True,
            check=True,
        )
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]
    payload = json.loads(outputs[0])
    assert payload["normalized_sha256"]
    assert payload["bundle_sha256"]
    assert payload["object_hashes"]
    assert payload["output_tables"]
