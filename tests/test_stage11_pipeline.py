from __future__ import annotations

import tarfile
from fractions import Fraction
from pathlib import Path

import pytest

from circuit_families.analysis.fidelity_calibration import (
    CANDIDATE_THRESHOLDS,
    SOURCE_TRAINING_RUN_ID,
    _safe_archive_member_name,
    load_calibration_source_records,
    parse_stable_post_stage9_row,
)

STAGE9_MANIFEST = Path(
    "manifests/stage9_sparse_stage9-sparse-s1-27fffed087e6.json"
)
STAGE9_TABLE = Path("results/tables/seed_1_stage9_sparse_search.csv")
STAGE9_ARCHIVE = Path(
    "results/archives/stage9-sparse-s1-27fffed087e6.tar.gz"
)
STAGE10_MANIFEST = Path(
    "manifests/stage10_fourier_stage10-fourier-s1-a6f6a5773057.json"
)


def valid_row() -> dict[str, str]:
    return {
        "phase": "stable post-grokking",
        "checkpoint_step": "9050",
        "source_training_run_id": SOURCE_TRAINING_RUN_ID,
        "search_status": "valid_sparse_circuit",
        "threshold_calibration_eligibility": (
            "eligible_for_later_stage11_primary_threshold_calibration"
        ),
        "fidelity_threshold": "0.99",
        "total_retained_components": "146",
        "retained_proportion": str(146 / 516),
        "final_exact_fidelity": str(12642 / 12769),
        "meaningfully_sparse": "True",
        "checkpoint_sha256": "a" * 64,
        "exact_evaluations_used": "6098",
        "final_mask_path": (
            "results/raw/stage9-sparse-s1-27fffed087e6/"
            "step_00009050/threshold_09900/final_mask.json"
        ),
        "final_mask_sha256": "b" * 64,
    }


@pytest.mark.parametrize(
    ("phase", "checkpoint_step"),
    [
        ("pre-grokking", "200"),
        ("50%", "8150"),
        ("stable post-grokking", "8150"),
    ],
)
def test_non_stable_post_rows_are_rejected(
    phase: str,
    checkpoint_step: str,
) -> None:
    row = valid_row()
    row["phase"] = phase
    row["checkpoint_step"] = checkpoint_step

    with pytest.raises(ValueError, match="only"):
        parse_stable_post_stage9_row(row)


def test_stage9_candidate_threshold_outside_grid_is_rejected() -> None:
    row = valid_row()
    row["fidelity_threshold"] = "0.925"

    with pytest.raises(ValueError, match="frozen grid"):
        parse_stable_post_stage9_row(row)


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.json",
        "/absolute.json",
        "safe/../../escape.json",
        r"safe\escape.json",
        "./relative.json",
    ],
)
def test_archive_path_traversal_is_rejected(member_name: str) -> None:
    with pytest.raises(ValueError, match="unsafe archive"):
        _safe_archive_member_name(member_name)


def test_archive_symlink_is_not_accepted_as_regular_file(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        member = tarfile.TarInfo("safe/link.json")
        member.type = tarfile.SYMTYPE
        member.linkname = "../escape.json"
        archive.addfile(member)

    with tarfile.open(archive_path, mode="r") as archive:
        stored = archive.getmember("safe/link.json")
        assert not stored.isfile()


def test_committed_stage9_and_stage10_sources_load() -> None:
    records = load_calibration_source_records(
        stage9_manifest_path=STAGE9_MANIFEST,
        stage9_table_path=STAGE9_TABLE,
        stage9_archive_path=STAGE9_ARCHIVE,
        stage10_manifest_path=STAGE10_MANIFEST,
    )

    assert records.stage9_run_id == "stage9-sparse-s1-27fffed087e6"
    assert records.stage10_run_id == "stage10-fourier-s1-a6f6a5773057"
    assert len(records.circuits) == 6
    assert len(records.fourier_records) == 6
    assert tuple(record.threshold for record in records.circuits) == (
        CANDIDATE_THRESHOLDS
    )
    assert tuple(record.retained_components for record in records.circuits) == (
        146,
        119,
        108,
        82,
        77,
        64,
    )
    assert tuple(record.exact_evaluations for record in records.circuits) == (
        6098,
        6599,
        6780,
        7042,
        7101,
        7328,
    )
    assert all(
        record.checkpoint_step == 9050 for record in records.circuits
    )
    assert all(
        record.search_status == "valid_sparse_circuit"
        for record in records.circuits
    )
    assert all(
        record.classification == "clear_match"
        for record in records.fourier_records
    )
    assert all(
        record.compatible_or_explained
        for record in records.fourier_records
    )
    assert all(
        record.correct_shift_rank == 1
        for record in records.fourier_records
    )


@pytest.mark.parametrize(
    ("threshold", "expected_agreement"),
    [
        (Fraction(99, 100), 12642),
        (Fraction(39, 40), 12450),
        (Fraction(19, 20), 12142),
        (Fraction(9, 10), 11496),
        (Fraction(17, 20), 10869),
        (Fraction(4, 5), 10228),
    ],
)
def test_stage9_integer_agreement_counts_recovered(
    threshold: Fraction,
    expected_agreement: int,
) -> None:
    records = load_calibration_source_records(
        stage9_manifest_path=STAGE9_MANIFEST,
        stage9_table_path=STAGE9_TABLE,
        stage9_archive_path=STAGE9_ARCHIVE,
        stage10_manifest_path=STAGE10_MANIFEST,
    )
    by_threshold = {record.threshold: record for record in records.circuits}

    assert by_threshold[threshold].exact_agreement_count == expected_agreement


def test_loaded_masks_match_stage9_counts_and_hashes() -> None:
    records = load_calibration_source_records(
        stage9_manifest_path=STAGE9_MANIFEST,
        stage9_table_path=STAGE9_TABLE,
        stage9_archive_path=STAGE9_ARCHIVE,
        stage10_manifest_path=STAGE10_MANIFEST,
    )

    for record in records.circuits:
        assert record.mask.retained_component_count == record.retained_components
        assert (
            record.mask.retained_component_proportion
            == record.retained_proportion
        )
        assert len(record.final_mask_sha256) == 64
        assert record.final_mask_member.endswith("/final_mask.json")


def test_sampled_mask_converts_to_canonical_component_mask() -> None:
    from circuit_families.analysis.fidelity_calibration import (
        sample_matched_size_masks,
        sampled_mask_to_component_mask,
    )

    sampled = sample_matched_size_masks(
        Fraction(99, 100),
        retained_count=146,
        replicates=1,
    )[0]
    mask = sampled_mask_to_component_mask(sampled)

    assert mask.retained_component_count == 146
    assert mask.retained_component_ids == sampled.retained_component_identifiers


def test_deterministic_archive_bytes_match_across_writes(
    tmp_path: Path,
) -> None:
    from circuit_families.analysis.fidelity_calibration import (
        file_sha256,
        write_deterministic_tar_gz,
    )

    source = tmp_path / "stage11-test"
    source.mkdir()
    (source / "b.json").write_text('{"b": 2}\n', encoding="utf-8")
    nested = source / "nested"
    nested.mkdir()
    (nested / "a.json").write_text('{"a": 1}\n', encoding="utf-8")

    first = write_deterministic_tar_gz(
        source_directory=source,
        archive_path=tmp_path / "first.tar.gz",
    )
    second = write_deterministic_tar_gz(
        source_directory=source,
        archive_path=tmp_path / "second.tar.gz",
    )

    assert first.read_bytes() == second.read_bytes()
    assert file_sha256(first) == file_sha256(second)


def test_stage11_cli_input_validation_only() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage11_calibration.py",
            "--run-id",
            "stage11-test",
            "--checkpoint-manifest",
            "manifests/checkpoints_seed_1.json",
            "--stage8-manifest",
            "manifests/stage8_masking_s1-5f1bc9dee7ab.json",
            "--stage9-manifest",
            str(STAGE9_MANIFEST),
            "--stage9-table",
            str(STAGE9_TABLE),
            "--stage9-archive",
            str(STAGE9_ARCHIVE),
            "--stage10-manifest",
            str(STAGE10_MANIFEST),
            "--validate-inputs-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "candidate_count: 6" in completed.stdout
    assert "retained_components: 146, 119, 108, 82, 77, 64" in (
        completed.stdout
    )
    assert "input_validation: passed" in completed.stdout



def test_stage11_runner_tracks_required_provenance_and_integrity() -> None:
    source = Path("scripts/run_stage11_calibration.py").read_text(
        encoding="utf-8"
    )

    required_fragments = (
        "--stage8-manifest",
        '"stage8_masking"',
        '"table_sha256"',
        '"archive_sha256"',
        '"source_masks"',
        '"model_state_sha256_before"',
        '"model_state_sha256_after"',
        '"model_state_unchanged"',
        '"hook_counts_before"',
        '"hook_counts_after"',
        '"hook_counts_unchanged"',
        '"metrics": metrics.to_record()',
    )
    for fragment in required_fragments:
        assert fragment in source
