from __future__ import annotations

import copy
import random

import pytest

from circuit_families.stage5bc.job_status import (
    JobStatusReport,
)
from circuit_families.stage5bc.serial_merge import (
    MERGE_ENTRY_STATES,
    SYNTHETIC_REGISTRY_SCHEMA_VERSION,
    SerialMergeError,
    SyntheticMergeEntry,
    canonical_registry_bytes,
    entry_from_job_status,
    merge_entries,
    merge_status_evidence,
    registry_sha256,
    stage3_unavailable_entry,
)


def _report(
    *,
    index: int,
    status: str,
    completion_sha: str | None = None,
    failure_sha: str | None = None,
) -> JobStatusReport:
    return JobStatusReport(
        job_id=f"stage5bc-job/v1::technical_completion::condition-{index}",
        node_type="technical_completion",
        condition_id=f"condition-{index}",
        relative_identity=f"jobs/v1/technical_completion/{index:064x}",
        status=status,
        reason=f"technical_{status}_{index}",
        output_root_exists=status != "planned",
        completion_sha256=completion_sha,
        failure_sha256=failure_sha,
    )


def test_exact_part_q_state_surface() -> None:
    assert MERGE_ENTRY_STATES == (
        "completed",
        "failed",
        "missing",
        "unavailable",
    )


@pytest.mark.parametrize(
    ("status", "expected_state"),
    [
        ("completed", "completed"),
        ("failed", "failed"),
        ("planned", "missing"),
        ("blocked", "missing"),
    ],
)
def test_mergeable_part_p_status_mapping(
    status: str,
    expected_state: str,
) -> None:
    report = _report(
        index=1,
        status=status,
        completion_sha=("a" * 64 if status == "completed" else None),
        failure_sha=("b" * 64 if status == "failed" else None),
    )

    entry = entry_from_job_status(report)

    assert entry.state == expected_state
    assert entry.observed_status == status


@pytest.mark.parametrize(
    "status",
    [
        "running",
        "stale",
        "conflicting",
    ],
)
def test_nonterminal_or_untrusted_statuses_are_rejected(
    status: str,
) -> None:
    with pytest.raises(
        SerialMergeError,
        match=f"{status} job cannot enter",
    ):
        entry_from_job_status(
            _report(
                index=1,
                status=status,
            )
        )


def test_completed_and_failed_require_exact_terminal_hashes() -> None:
    with pytest.raises(
        SerialMergeError,
        match="completion SHA-256",
    ):
        entry_from_job_status(
            _report(
                index=1,
                status="completed",
            )
        )

    with pytest.raises(
        SerialMergeError,
        match="failure SHA-256",
    ):
        entry_from_job_status(
            _report(
                index=2,
                status="failed",
            )
        )


def test_unavailable_stage3_marker_is_canonical_and_portable() -> None:
    entry = stage3_unavailable_entry(
        teacher_seed=0,
        phase="pre-grokking",
        reason="Stage 3 registry marks cell unavailable",
    )

    assert entry.state == "unavailable"
    assert entry.source_kind == "stage3_unavailable"
    assert entry.identity_key == (
        "stage3-unavailable/v1::"
        "teacher_seed=0::"
        "phase=pre-grokking"
    )
    assert entry.record_sha256 is None


def test_merge_preserves_completed_failed_missing_and_unavailable() -> None:
    entries = [
        entry_from_job_status(
            _report(
                index=1,
                status="completed",
                completion_sha="a" * 64,
            )
        ),
        entry_from_job_status(
            _report(
                index=2,
                status="failed",
                failure_sha="b" * 64,
            )
        ),
        entry_from_job_status(
            _report(
                index=3,
                status="planned",
            )
        ),
        stage3_unavailable_entry(
            teacher_seed=0,
            phase="50%",
            reason="Stage 3 registry marks cell unavailable",
        ),
    ]

    registry = merge_entries(entries)

    assert registry["schema_version"] == (
        SYNTHETIC_REGISTRY_SCHEMA_VERSION
    )
    assert registry["scientific_data"] is False
    assert registry["production_eligible"] is False

    assert {
        entry["state"]
        for entry in registry["entries"]
    } == {
        "completed",
        "failed",
        "missing",
        "unavailable",
    }


def test_canonical_identity_order_is_independent_of_input_order() -> None:
    entries = [
        entry_from_job_status(
            _report(
                index=3,
                status="planned",
            )
        ),
        entry_from_job_status(
            _report(
                index=1,
                status="completed",
                completion_sha="a" * 64,
            )
        ),
        entry_from_job_status(
            _report(
                index=2,
                status="failed",
                failure_sha="b" * 64,
            )
        ),
        stage3_unavailable_entry(
            teacher_seed=0,
            phase="pre-grokking",
            reason="unavailable",
        ),
    ]

    baseline = merge_entries(entries)
    baseline_bytes = canonical_registry_bytes(baseline)

    for seed in range(10):
        shuffled = copy.deepcopy(entries)
        random.Random(seed).shuffle(shuffled)

        candidate = merge_entries(shuffled)

        assert candidate == baseline
        assert canonical_registry_bytes(candidate) == baseline_bytes
        assert registry_sha256(candidate) == registry_sha256(baseline)

    identity_keys = [
        entry["identity_key"]
        for entry in baseline["entries"]
    ]

    assert identity_keys == sorted(identity_keys)


def test_duplicate_identity_is_rejected_even_if_record_is_identical() -> None:
    entry = entry_from_job_status(
        _report(
            index=1,
            status="completed",
            completion_sha="a" * 64,
        )
    )

    with pytest.raises(
        SerialMergeError,
        match="duplicate canonical merge identity",
    ):
        merge_entries(
            [entry, copy.deepcopy(entry)]
        )


def test_duplicate_node_condition_coordinate_is_rejected() -> None:
    first = entry_from_job_status(
        _report(
            index=1,
            status="completed",
            completion_sha="a" * 64,
        )
    )

    second = SyntheticMergeEntry(
        identity_key="job::different-job-id",
        source_kind="job_status",
        state="completed",
        observed_status="completed",
        reason="duplicate coordinate",
        job_id="different-job-id",
        node_type=first.node_type,
        condition_id=first.condition_id,
        record_sha256="b" * 64,
    )

    with pytest.raises(
        SerialMergeError,
        match="duplicate job node/condition coordinate",
    ):
        merge_entries([first, second])


def test_registry_hash_changes_with_exact_terminal_hash() -> None:
    first = merge_entries(
        [
            entry_from_job_status(
                _report(
                    index=1,
                    status="completed",
                    completion_sha="a" * 64,
                )
            )
        ]
    )

    second = merge_entries(
        [
            entry_from_job_status(
                _report(
                    index=1,
                    status="completed",
                    completion_sha="b" * 64,
                )
            )
        ]
    )

    assert registry_sha256(first) != registry_sha256(second)


def test_merge_status_evidence_retains_unavailable_markers() -> None:
    registry = merge_status_evidence(
        reports=[
            _report(
                index=1,
                status="completed",
                completion_sha="a" * 64,
            ),
            _report(
                index=2,
                status="blocked",
            ),
        ],
        unavailable_entries=[
            stage3_unavailable_entry(
                teacher_seed=0,
                phase="pre-grokking",
                reason="unavailable",
            ),
            stage3_unavailable_entry(
                teacher_seed=0,
                phase="50%",
                reason="unavailable",
            ),
        ],
    )

    states = [
        entry["state"]
        for entry in registry["entries"]
    ]

    assert states.count("completed") == 1
    assert states.count("missing") == 1
    assert states.count("unavailable") == 2


def test_registry_contains_no_stage5d_summary_surface() -> None:
    registry = merge_entries(
        [
            stage3_unavailable_entry(
                teacher_seed=0,
                phase="pre-grokking",
                reason="unavailable",
            )
        ]
    )

    rendered = canonical_registry_bytes(registry).decode("utf-8")

    assert "stage5d" not in rendered.lower()
    assert "eligibility_summary" not in rendered
    assert "scientific_summary" not in rendered
