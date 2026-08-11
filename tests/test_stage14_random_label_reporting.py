"""Tests for deterministic Stage 14 final reporting."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from circuit_families.analysis.stage14_random_label_reporting import (
    FRONTIER_COLUMNS,
    analysis_note_text,
    frontier_rows,
    write_deterministic_archive,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_frontier_schema_is_unique() -> None:
    assert len(FRONTIER_COLUMNS) == len(
        set(FRONTIER_COLUMNS)
    )
    assert "scientific_interpretation" in FRONTIER_COLUMNS


def test_empty_primary_frontier_is_cautious() -> None:
    rows = frontier_rows(
        analysis_run_id="analysis",
        family_rows=(
            {
                "checkpoint_step": "200",
                "status": "sparsity_failure",
                "family_size": "0",
            },
        ),
        restart_rows=(
            {
                "checkpoint_step": "200",
                "restart_used": "True",
                "requested_member_index": "1",
                "restart_index": "0",
                "accepted_candidate": "False",
                "search_status": (
                    "valid_but_not_meaningfully_sparse"
                ),
                "primary_fidelity": "0.99",
                "retained_component_count": "508",
                "exact_evaluations_used": "1580",
            },
        ),
    )

    note = analysis_note_text(
        analysis_run_id="analysis",
        frontier=rows,
    )

    assert len(rows) == 1
    assert rows[0]["family_size"] == 0
    assert "No meaningfully sparse" in note
    assert "does not establish" in note


def test_deterministic_archive_reproduces_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    first = root / "first.txt"
    second = root / "second.txt"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    archive_one = tmp_path / "one.tar.gz"
    archive_two = tmp_path / "two.tar.gz"

    write_deterministic_archive(
        archive_one,
        root=root,
        members=(second, first),
    )
    write_deterministic_archive(
        archive_two,
        root=root,
        members=(first, second),
    )

    assert hashlib.sha256(
        archive_one.read_bytes()
    ).digest() == hashlib.sha256(
        archive_two.read_bytes()
    ).digest()


def test_runner_help_exposes_reporting_mode() -> None:
    completed = subprocess.run(
        [
            "/Users/alexkolesnikov/.local/bin/uv",
            "run",
            "python",
            "scripts/run_stage14_random_label_analysis.py",
            "--help",
        ],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--execute-reporting" in completed.stdout
