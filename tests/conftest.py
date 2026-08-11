"""Portable test selection for the clean follow-up repository.

The predecessor source snapshot includes audit tests that intentionally open
frozen checkpoints, manifests, tables, and archives. Those scientific artifacts
are not published in this repository. The tests remain collected but are skipped
unless the complete predecessor artifact layout has been installed locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]

PREDECESSOR_SENTINELS = (
    REPOSITORY / "manifests" / "stage22_freeze_stage22-freeze-34241335dcf7.json",
    REPOSITORY / "manifests" / "checkpoints_seed_1.json",
    REPOSITORY / "data" / "generated" / "modular_addition_m113.npz",
)

PREDECESSOR_ARTIFACT_TEST_MODULES = frozenset(
    {
        "test_post_stage17_additional_control_seed_freeze.py",
        "test_post_stage17_checkpoint_grid_freeze.py",
        "test_random_label_control.py",
        "test_stage10_pipeline.py",
        "test_stage11_pipeline.py",
        "test_stage12_negative_controls.py",
        "test_stage12_runner.py",
        "test_stage13_pipeline.py",
        "test_stage14_pipeline.py",
        "test_stage14_random_label_analysis_runner.py",
        "test_stage14_random_label_diversity_runner.py",
        "test_stage14_random_label_reporting.py",
        "test_stage14_random_label_sensitivity_runner.py",
        "test_stage14_random_label_transfer_runner.py",
        "test_stage15_unavailable.py",
        "test_stage16_pipeline.py",
        "test_stage17_pipeline.py",
        "test_stage8_validation.py",
        "test_training_data.py",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip predecessor-artifact audits when their immutable inputs are absent."""

    if all(path.is_file() for path in PREDECESSOR_SENTINELS):
        return

    marker = pytest.mark.skip(
        reason=(
            "requires the private predecessor checkpoint/result artifact bundle; "
            "the clean follow-up repository intentionally excludes it"
        )
    )

    for item in items:
        if item.path.name in PREDECESSOR_ARTIFACT_TEST_MODULES:
            item.add_marker(marker)
