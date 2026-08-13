"""Adversarial tests for the Stage 1 predecessor-link contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from circuit_families.followup_namespace import (
    NamespacePathError,
    validate_followup_output_path,
)
from circuit_families.predecessor_link import (
    PredecessorLinkError,
    compare_commit_blob_populations,
    load_predecessor_link,
    validate_predecessor_link,
    verify_predecessor_link_physical,
    verify_successor_snapshot_physical,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "followup/manifests/predecessor_link_v1.json"


@pytest.fixture
def canonical_record() -> dict:
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


def test_canonical_manifest_passes_strict_validation(
    canonical_record: dict,
) -> None:
    validate_predecessor_link(canonical_record)


def test_canonical_manifest_loads() -> None:
    loaded = load_predecessor_link(CANONICAL)
    assert loaded["schema_version"] == 1
    assert [row["teacher_seed"] for row in loaded["teacher_runs"]] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_canonical_manifest_passes_physical_predecessor_verification(
    canonical_record: dict,
) -> None:
    predecessor = Path(
        "/Users/alexkolesnikov/Projects/circuit-families"
    )
    if not predecessor.is_dir():
        pytest.skip("private predecessor checkout is not installed")

    verify_predecessor_link_physical(
        canonical_record,
        predecessor_root=predecessor,
    )


def test_wrong_but_well_formed_commit_fails_physical_verification(
    canonical_record: dict,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "predecessor"
    repository.mkdir()

    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=repository,
        check=True,
    )
    (repository / "placeholder").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "placeholder"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=repository,
        check=True,
    )

    record = deepcopy(canonical_record)
    record["predecessor"]["analysis_freeze_commit"] = "0" * 40

    with pytest.raises(
        PredecessorLinkError,
        match="Physical predecessor commit mismatch",
    ):
        verify_predecessor_link_physical(
            record,
            predecessor_root=repository,
        )


def test_wrong_but_well_formed_protocol_hash_fails_physical_verification(
    canonical_record: dict,
) -> None:
    predecessor = Path(
        "/Users/alexkolesnikov/Projects/circuit-families"
    )
    if not predecessor.is_dir():
        pytest.skip("private predecessor checkout is not installed")

    record = deepcopy(canonical_record)
    record["predecessor"]["protocol"]["sha256"] = "0" * 64

    with pytest.raises(
        PredecessorLinkError,
        match="Physical predecessor hash mismatch for predecessor.protocol",
    ):
        verify_predecessor_link_physical(
            record,
            predecessor_root=predecessor,
        )


def test_wrong_dataset_run_id_fails_physical_verification(
    canonical_record: dict,
) -> None:
    predecessor = Path(
        "/Users/alexkolesnikov/Projects/circuit-families"
    )
    if not predecessor.is_dir():
        pytest.skip("private predecessor checkout is not installed")

    record = deepcopy(canonical_record)
    record["dataset"]["run_id"] = "wrong-but-well-formed-dataset-run"

    with pytest.raises(
        PredecessorLinkError,
        match="Physical predecessor dataset run ID mismatch",
    ):
        verify_predecessor_link_physical(
            record,
            predecessor_root=predecessor,
        )


def test_wrong_dataset_hash_fails_physical_verification(
    canonical_record: dict,
) -> None:
    predecessor = Path(
        "/Users/alexkolesnikov/Projects/circuit-families"
    )
    if not predecessor.is_dir():
        pytest.skip("private predecessor checkout is not installed")

    record = deepcopy(canonical_record)
    record["dataset"]["dataset_sha256"] = "0" * 64

    with pytest.raises(
        PredecessorLinkError,
        match=(
            "Physical predecessor dataset hash mismatch for "
            "dataset.dataset_sha256"
        ),
    ):
        verify_predecessor_link_physical(
            record,
            predecessor_root=predecessor,
        )


def test_missing_freeze_identity_fails(canonical_record: dict) -> None:
    record = deepcopy(canonical_record)
    del record["predecessor"]["analysis_freeze_manifest"]

    with pytest.raises(
        PredecessorLinkError,
        match="missing required fields: analysis_freeze_manifest",
    ):
        validate_predecessor_link(record)


def test_duplicate_teacher_seed_fails(canonical_record: dict) -> None:
    record = deepcopy(canonical_record)
    record["teacher_runs"][1]["teacher_seed"] = 0

    with pytest.raises(
        PredecessorLinkError,
        match="Duplicate teacher seed",
    ):
        validate_predecessor_link(record)


def test_stage3_cannot_be_marked_resolved(canonical_record: dict) -> None:
    record = deepcopy(canonical_record)
    record["stage3_checkpoint_registry"]["resolved"] = True

    with pytest.raises(
        PredecessorLinkError,
        match="cannot be marked resolved",
    ):
        validate_predecessor_link(record)


def test_stage3_cannot_contain_selection_records(
    canonical_record: dict,
) -> None:
    record = deepcopy(canonical_record)
    record["stage3_checkpoint_registry"]["selection_records"] = [
        {"checkpoint": "invented"}
    ]

    with pytest.raises(
        PredecessorLinkError,
        match="selection_records must be empty",
    ):
        validate_predecessor_link(record)


def test_absolute_private_path_cannot_be_portable_identity(
    canonical_record: dict,
) -> None:
    record = deepcopy(canonical_record)
    record["dataset"]["manifest"]["path"] = (
        "/Users/example/private/predecessor/manifest.json"
    )

    with pytest.raises(
        PredecessorLinkError,
        match="portable repository-relative path",
    ):
        validate_predecessor_link(record)


def test_parent_traversal_cannot_be_portable_identity(
    canonical_record: dict,
) -> None:
    record = deepcopy(canonical_record)
    record["dataset"]["manifest"]["path"] = "../private/manifest.json"

    with pytest.raises(
        PredecessorLinkError,
        match="must not contain parent traversal",
    ):
        validate_predecessor_link(record)


def test_unsupported_schema_version_fails(canonical_record: dict) -> None:
    record = deepcopy(canonical_record)
    record["schema_version"] = 2

    with pytest.raises(
        PredecessorLinkError,
        match="Unsupported predecessor-link schema version",
    ):
        validate_predecessor_link(record)


def test_unknown_root_field_fails(canonical_record: dict) -> None:
    record = deepcopy(canonical_record)
    record["invented_field"] = "not allowed"

    with pytest.raises(
        PredecessorLinkError,
        match="contains unknown fields: invented_field",
    ):
        validate_predecessor_link(record)


def test_exact_predecessor_output_root_fails(tmp_path: Path) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"

    with pytest.raises(
        NamespacePathError,
        match="collides with the immutable predecessor",
    ):
        validate_followup_output_path(
            predecessor,
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_nested_predecessor_output_root_fails(tmp_path: Path) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"

    with pytest.raises(
        NamespacePathError,
        match="collides with the immutable predecessor",
    ):
        validate_followup_output_path(
            predecessor / "results" / "raw" / "collision",
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_symlink_into_predecessor_fails(tmp_path: Path) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"
    successor.mkdir()
    predecessor.mkdir()
    (successor / "followup").mkdir()

    link = successor / "followup" / "predecessor-link"
    link.symlink_to(predecessor, target_is_directory=True)

    with pytest.raises(
        NamespacePathError,
        match="collides with the immutable predecessor",
    ):
        validate_followup_output_path(
            link / "results",
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_missing_teacher_seed_fails(canonical_record: dict) -> None:
    record = deepcopy(canonical_record)
    record["teacher_runs"] = record["teacher_runs"][:-1]

    with pytest.raises(
        PredecessorLinkError,
        match=r"expected_count=5, actual_count=4",
    ):
        validate_predecessor_link(record)


def test_extra_teacher_seed_fails(canonical_record: dict) -> None:
    record = deepcopy(canonical_record)
    extra = deepcopy(record["teacher_runs"][-1])
    extra["teacher_seed"] = 5
    extra["run_id"] = "invented-extra-teacher"
    record["teacher_runs"].append(extra)

    with pytest.raises(
        PredecessorLinkError,
        match=r"expected_count=5, actual_count=6",
    ):
        validate_predecessor_link(record)


def test_out_of_range_teacher_seed_fails(canonical_record: dict) -> None:
    record = deepcopy(canonical_record)
    record["teacher_runs"][-1]["teacher_seed"] = 5

    with pytest.raises(
        PredecessorLinkError,
        match=r"expected_one_of=\[0, 1, 2, 3, 4\], actual=5",
    ):
        validate_predecessor_link(record)


def test_five_teacher_entries_must_collectively_equal_zero_through_four(
    canonical_record: dict,
) -> None:
    record = deepcopy(canonical_record)
    record["teacher_runs"][-1]["teacher_seed"] = 3

    with pytest.raises(
        PredecessorLinkError,
        match="Duplicate teacher seed",
    ):
        validate_predecessor_link(record)


def test_teacher_roster_order_is_canonical(canonical_record: dict) -> None:
    record = deepcopy(canonical_record)
    record["teacher_runs"][0], record["teacher_runs"][1] = (
        record["teacher_runs"][1],
        record["teacher_runs"][0],
    )

    with pytest.raises(
        PredecessorLinkError,
        match=r"deterministic canonical seed ordering",
    ):
        validate_predecessor_link(record)


def test_wrong_but_well_formed_config_file_hash_fails_physical_verification(
    canonical_record: dict,
) -> None:
    predecessor = Path(
        "/Users/alexkolesnikov/Projects/circuit-families"
    )
    if not predecessor.is_dir():
        pytest.skip("private predecessor checkout is not installed")

    record = deepcopy(canonical_record)
    record["architecture"]["model_config"]["file_sha256"] = "0" * 64

    with pytest.raises(
        PredecessorLinkError,
        match="config file hash mismatch",
    ):
        verify_predecessor_link_physical(
            record,
            predecessor_root=predecessor,
        )


def test_wrong_but_well_formed_config_mapping_hash_fails_physical_verification(
    canonical_record: dict,
) -> None:
    predecessor = Path(
        "/Users/alexkolesnikov/Projects/circuit-families"
    )
    if not predecessor.is_dir():
        pytest.skip("private predecessor checkout is not installed")

    record = deepcopy(canonical_record)
    record["architecture"]["training_config"]["mapping_sha256"] = "0" * 64

    with pytest.raises(
        PredecessorLinkError,
        match="config mapping hash mismatch",
    ):
        verify_predecessor_link_physical(
            record,
            predecessor_root=predecessor,
        )

def test_wrong_teacher_run_id_fails_physical_verification(
    canonical_record: dict,
) -> None:
    predecessor = Path(
        "/Users/alexkolesnikov/Projects/circuit-families"
    )
    if not predecessor.is_dir():
        pytest.skip("private predecessor checkout is not installed")

    record = deepcopy(canonical_record)
    record["teacher_runs"][0]["run_id"] = (
        "wrong-but-well-formed-teacher-run"
    )

    with pytest.raises(
        PredecessorLinkError,
        match="Physical predecessor teacher run ID mismatch",
    ):
        verify_predecessor_link_physical(
            record,
            predecessor_root=predecessor,
        )


def test_wrong_teacher_seed_provenance_fails_physical_verification(
    canonical_record: dict,
) -> None:
    predecessor = Path(
        "/Users/alexkolesnikov/Projects/circuit-families"
    )
    if not predecessor.is_dir():
        pytest.skip("private predecessor checkout is not installed")

    record = deepcopy(canonical_record)

    seed_one = deepcopy(record["teacher_runs"][1])
    record["teacher_runs"][0]["run_id"] = seed_one["run_id"]
    record["teacher_runs"][0]["manifest"] = seed_one["manifest"]

    with pytest.raises(
        PredecessorLinkError,
        match="Physical predecessor teacher seed mismatch",
    ):
        verify_predecessor_link_physical(
            record,
            predecessor_root=predecessor,
        )

def test_genuine_repositories_match_frozen_successor_snapshot(
    canonical_record: dict,
) -> None:
    """Verify the frozen 166/166/0 snapshot against genuine repositories."""

    successor_root = ROOT
    predecessor_root = Path(
        "/Users/alexkolesnikov/Projects/circuit-families"
    )

    if not predecessor_root.is_dir():
        pytest.skip(
            "physical predecessor repository is genuinely unavailable"
        )

    evidence = verify_successor_snapshot_physical(
        canonical_record,
        successor_root=successor_root,
        predecessor_root=predecessor_root,
    )

    assert evidence["verified_root_commit"] == (
        "2e8f293ca307370ab213949c4a9490bf5fc2a47c"
    )
    assert evidence["predecessor_comparison_commit"] == (
        "a55509537a70a225fedc5ce3a1c8236110974a6e"
    )
    assert evidence["predecessor_selected_file_count"] == 166
    assert evidence["successor_selected_file_count"] == 167
    assert evidence["overlapping_file_count"] == 166
    assert evidence["byte_identical_overlapping_file_count"] == 166
    assert evidence["changed_overlapping_file_count"] == 0
    assert evidence["changed_paths"] == []

def test_valid_non_root_successor_commit_is_rejected(
    canonical_record: dict,
) -> None:
    """A valid successor commit must still fail if it is not the Git root."""

    record = deepcopy(canonical_record)
    record["successor_snapshot"]["initial_commit"] = (
        "a1630d03a9b739b56eb07f7a393d9df09d004685"
    )

    with pytest.raises(
        PredecessorLinkError,
        match="recorded successor commit is not the root",
    ):
        verify_successor_snapshot_physical(
            record,
            successor_root=ROOT,
            predecessor_root=Path(
                "/Users/alexkolesnikov/Projects/circuit-families"
            ),
        )

def test_nonexistent_successor_commit_is_rejected(
    canonical_record: dict,
) -> None:
    """A well-formed but absent successor commit must be rejected."""

    record = deepcopy(canonical_record)
    nonexistent = "f" * 40

    import subprocess

    probe = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "cat-file",
            "-e",
            f"{nonexistent}^{{commit}}",
        ],
        check=False,
        capture_output=True,
    )
    assert probe.returncode != 0

    record["successor_snapshot"]["initial_commit"] = nonexistent

    with pytest.raises(
        PredecessorLinkError,
        match="successor commit missing",
    ):
        verify_successor_snapshot_physical(
            record,
            successor_root=ROOT,
            predecessor_root=Path(
                "/Users/alexkolesnikov/Projects/circuit-families"
            ),
        )

@pytest.mark.parametrize(
    ("field", "mutated_value", "message"),
    [
        (
            "overlapping_file_count",
            165,
            "successor snapshot overlap count mismatch",
        ),
        (
            "byte_identical_overlapping_file_count",
            165,
            "successor snapshot identical overlap count mismatch",
        ),
        (
            "changed_overlapping_file_count",
            1,
            "successor snapshot changed overlap count mismatch",
        ),
    ],
)
def test_false_successor_snapshot_count_claim_is_rejected(
    canonical_record: dict,
    field: str,
    mutated_value: int,
    message: str,
) -> None:
    """Each frozen overlap-count claim must match physical Git evidence."""

    record = deepcopy(canonical_record)
    record["successor_snapshot"][field] = mutated_value

    with pytest.raises(
        PredecessorLinkError,
        match=message,
    ):
        verify_successor_snapshot_physical(
            record,
            successor_root=ROOT,
            predecessor_root=Path(
                "/Users/alexkolesnikov/Projects/circuit-families"
            ),
        )

def test_changed_overlapping_content_is_rejected_and_reported(
    tmp_path: Path,
) -> None:
    """Changed overlapping Git blobs must fail even when counts match."""

    import subprocess

    predecessor = tmp_path / "predecessor"
    successor = tmp_path / "successor"

    def initialize(repository: Path, content: str) -> str:
        repository.mkdir()
        subprocess.run(
            ["git", "init", "-q"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "config",
                "user.email",
                "fixture@example.invalid",
            ],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Fixture"],
            cwd=repository,
            check=True,
        )
        source = repository / "src"
        source.mkdir()
        (source / "shared.txt").write_text(
            content,
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "add", "src/shared.txt"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "root"],
            cwd=repository,
            check=True,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    predecessor_commit = initialize(predecessor, "predecessor-value\n")
    successor_commit = initialize(successor, "successor-value\n")

    record = {
        "predecessor": {
            "analysis_freeze_commit": predecessor_commit,
        },
        "successor_snapshot": {
            "initial_commit": successor_commit,
            "overlapping_file_count": 1,
            "byte_identical_overlapping_file_count": 0,
            "changed_overlapping_file_count": 1,
        },
    }

    with pytest.raises(
        PredecessorLinkError,
        match="changed overlapping files detected",
    ) as exc_info:
        verify_successor_snapshot_physical(
            record,
            successor_root=successor,
            predecessor_root=predecessor,
        )

    message = str(exc_info.value)
    assert "src/shared.txt" in message
    assert "predecessor-value" not in message
    assert "successor-value" not in message

def test_commit_object_comparison_is_independent_of_later_worktree_state(
    tmp_path: Path,
) -> None:
    """Selected commit objects, not branch/worktree state, define comparison."""

    import subprocess

    predecessor = tmp_path / "predecessor"
    successor = tmp_path / "successor"

    def initialize(repository: Path) -> str:
        repository.mkdir()
        subprocess.run(
            ["git", "init", "-q"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "config",
                "user.email",
                "fixture@example.invalid",
            ],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Fixture"],
            cwd=repository,
            check=True,
        )
        source = repository / "src"
        source.mkdir()
        (source / "shared.txt").write_text(
            "frozen\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "add", "src/shared.txt"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "root"],
            cwd=repository,
            check=True,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    predecessor_commit = initialize(predecessor)
    successor_commit = initialize(successor)

    before = compare_commit_blob_populations(
        predecessor,
        predecessor_commit,
        successor,
        successor_commit,
    )

    # Add a later committed file after the selected successor snapshot.
    later = successor / "src" / "later.txt"
    later.write_text("later commit\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "src/later.txt"],
        cwd=successor,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "later"],
        cwd=successor,
        check=True,
    )

    # Add untracked and dirty tracked worktree state after that later commit.
    (successor / "src" / "untracked.txt").write_text(
        "untracked\n",
        encoding="utf-8",
    )
    (successor / "src" / "shared.txt").write_text(
        "dirty worktree value\n",
        encoding="utf-8",
    )

    after = compare_commit_blob_populations(
        predecessor,
        predecessor_commit,
        successor,
        successor_commit,
    )

    assert before == after
    assert after["predecessor_selected_file_count"] == 1
    assert after["successor_selected_file_count"] == 1
    assert after["overlapping_file_count"] == 1
    assert after["byte_identical_overlapping_file_count"] == 1
    assert after["changed_overlapping_file_count"] == 0
    assert after["changed_paths"] == []
