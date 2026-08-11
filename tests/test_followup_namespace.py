"""Tests for the portable follow-up artifact namespace."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from circuit_families.followup_namespace import (
    APPROVED_LOGICAL_ROOTS,
    FOLLOWUP_ROOT,
    NAMESPACE_VERSION,
    NamespacePathError,
    build_followup_relative_path,
    logical_root,
    resolve_logical_output_path,
    validate_followup_output_path,
    validate_repository_roots,
)


def test_namespace_version_and_roots_are_portable() -> None:
    assert NAMESPACE_VERSION == "circuit-families-distillation/v1"
    assert FOLLOWUP_ROOT == PurePosixPath("followup")
    assert len(APPROVED_LOGICAL_ROOTS) == 13

    for path in APPROVED_LOGICAL_ROOTS.values():
        assert not path.is_absolute()
        assert path.parts[0] == "followup"
        assert ".." not in path.parts


def test_logical_path_construction_is_deterministic() -> None:
    first = build_followup_relative_path(
        "manifests",
        "teacher",
        "seed_0.json",
    )
    second = build_followup_relative_path(
        "manifests",
        "teacher",
        "seed_0.json",
    )

    assert first == second
    assert first == PurePosixPath(
        "followup/manifests/teacher/seed_0.json"
    )


def test_unknown_logical_root_fails_clearly() -> None:
    with pytest.raises(
        NamespacePathError,
        match="Unknown follow-up logical root 'unknown'",
    ):
        logical_root("unknown")


def test_relative_path_rejects_parent_traversal() -> None:
    with pytest.raises(
        NamespacePathError,
        match="traversal is not permitted",
    ):
        build_followup_relative_path(
            "manifests",
            "..",
            "..",
            "results",
            "collision.json",
        )


def test_relative_path_rejects_absolute_injection() -> None:
    with pytest.raises(
        NamespacePathError,
        match="must be repository-relative",
    ):
        build_followup_relative_path(
            "manifests",
            "/tmp/injected.json",
        )


def test_repository_roots_must_be_distinct(tmp_path: Path) -> None:
    root = tmp_path / "same"

    with pytest.raises(
        NamespacePathError,
        match="resolves to the predecessor repository root",
    ):
        validate_repository_roots(
            successor_root=root,
            predecessor_root=root,
        )


def test_successor_cannot_be_inside_predecessor(tmp_path: Path) -> None:
    predecessor = tmp_path / "predecessor"
    successor = predecessor / "nested-successor"

    with pytest.raises(
        NamespacePathError,
        match="Successor repository root must not be inside",
    ):
        validate_repository_roots(
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_predecessor_cannot_be_inside_successor(tmp_path: Path) -> None:
    successor = tmp_path / "successor"
    predecessor = successor / "nested-predecessor"

    with pytest.raises(
        NamespacePathError,
        match="Predecessor repository root must not be inside",
    ):
        validate_repository_roots(
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_valid_synthetic_repository_roots_are_permitted(
    tmp_path: Path,
) -> None:
    successor = tmp_path / "portable-clone-anywhere"
    predecessor = tmp_path / "immutable-source-anywhere"

    actual_successor, actual_predecessor = validate_repository_roots(
        successor_root=successor,
        predecessor_root=predecessor,
    )

    assert actual_successor == successor.resolve()
    assert actual_predecessor == predecessor.resolve()


def test_exact_predecessor_root_is_rejected(tmp_path: Path) -> None:
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


def test_nested_predecessor_path_is_rejected(tmp_path: Path) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"

    with pytest.raises(
        NamespacePathError,
        match="collides with the immutable predecessor",
    ):
        validate_followup_output_path(
            predecessor / "results" / "raw" / "new-run",
            successor_root=successor,
            predecessor_root=predecessor,
        )


@pytest.mark.parametrize(
    "legacy",
    ["results", "checkpoints", "manifests", "figures"],
)
def test_legacy_successor_output_roots_are_rejected(
    tmp_path: Path,
    legacy: str,
) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"

    with pytest.raises(
        NamespacePathError,
        match="inherited legacy output root",
    ):
        validate_followup_output_path(
            successor / legacy / "new-followup-output",
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_successor_root_itself_is_rejected(tmp_path: Path) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"

    with pytest.raises(
        NamespacePathError,
        match="successor repository root itself is not an output root",
    ):
        validate_followup_output_path(
            successor,
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_path_outside_successor_followup_namespace_is_rejected(
    tmp_path: Path,
) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"
    unrelated = tmp_path / "elsewhere" / "output"

    with pytest.raises(
        NamespacePathError,
        match="escapes the authorized namespace",
    ):
        validate_followup_output_path(
            unrelated,
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_relative_traversal_outside_followup_is_rejected_physically(
    tmp_path: Path,
) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"

    with pytest.raises(
        NamespacePathError,
        match="escapes the authorized namespace",
    ):
        validate_followup_output_path(
            "followup/../../outside",
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_symlink_into_predecessor_is_rejected(tmp_path: Path) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"
    successor.mkdir()
    predecessor.mkdir()
    followup = successor / "followup"
    followup.mkdir()

    link = followup / "linked-predecessor"
    link.symlink_to(predecessor, target_is_directory=True)

    with pytest.raises(
        NamespacePathError,
        match="collides with the immutable predecessor",
    ):
        validate_followup_output_path(
            link / "results" / "new-output",
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_symlink_escape_outside_authorized_root_is_rejected(
    tmp_path: Path,
) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"
    outside = tmp_path / "outside"
    successor.mkdir()
    predecessor.mkdir()
    outside.mkdir()
    followup = successor / "followup"
    followup.mkdir()

    link = followup / "outside-link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        NamespacePathError,
        match="escapes the authorized namespace",
    ):
        validate_followup_output_path(
            link / "output",
            successor_root=successor,
            predecessor_root=predecessor,
        )


def test_valid_followup_path_in_temporary_repository(
    tmp_path: Path,
) -> None:
    successor = tmp_path / "portable-successor"
    predecessor = tmp_path / "portable-predecessor"

    resolved = resolve_logical_output_path(
        "teacher_cache",
        "seed_0",
        "cache.bin",
        successor_root=successor,
        predecessor_root=predecessor,
    )

    assert resolved == (
        successor
        / "followup"
        / "artifacts"
        / "teacher_cache"
        / "seed_0"
        / "cache.bin"
    ).resolve()
    assert not resolved.exists()


def test_validation_does_not_create_output_paths(tmp_path: Path) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"

    resolved = resolve_logical_output_path(
        "reviewed_tables",
        "example.csv",
        successor_root=successor,
        predecessor_root=predecessor,
    )

    assert not successor.exists()
    assert not predecessor.exists()
    assert not resolved.exists()


def test_error_reports_supplied_and_resolved_paths(
    tmp_path: Path,
) -> None:
    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"
    supplied = "followup/../../unsafe"

    with pytest.raises(NamespacePathError) as exc_info:
        validate_followup_output_path(
            supplied,
            successor_root=successor,
            predecessor_root=predecessor,
        )

    message = str(exc_info.value)
    assert "supplied=followup/../../unsafe" in message
    assert "resolved=" in message
    assert "authorized_root=" in message
