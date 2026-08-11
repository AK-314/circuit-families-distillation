"""Validation helpers for post-reproduction downstream finalization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuit_families.training import file_sha256


def read_last_json_object(path: Path) -> dict[str, Any]:
    """Read the final pretty-printed JSON object from a progress log."""
    text = path.read_text(encoding="utf-8")
    start = text.rfind("\n{")
    if start < 0:
        start = text.find("{") - 1
    if start < -1:
        raise ValueError(f"No JSON object found in {path}.")
    value = json.loads(text[start + 1 :])
    if not isinstance(value, dict):
        raise ValueError(f"Final JSON value must be an object: {path}")
    return value


def validate_stage18_comparison(value: Mapping[str, Any], *, run_id: str) -> None:
    """Require the complete successful Stage 18 comparison contract."""
    required = {
        "stage18_run_id": run_id,
        "passed": True,
        "compared_file_count": 818_386,
        "deterministic_mismatch_count": 0,
        "archive_inventory_count": 35,
        "archive_inventory_mismatch_count": 0,
        "normalized_metadata_file_count": 1_224,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(
                f"Stage 18 comparison field {key!r} is {value.get(key)!r}; expected {expected!r}."
            )
    if value.get("mismatches") != []:
        raise ValueError("Stage 18 comparison must contain no mismatches.")
    policy = value.get("comparison_policy")
    if not isinstance(policy, Mapping) or "normalized_metadata_wrappers" not in policy:
        raise ValueError("Stage 18 comparison is missing its normalization policy.")


def verify_manifest_outputs(repository: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Verify every output declared by a stage manifest."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError("Manifest output hash mapping is missing or empty.")
    verified: dict[str, str] = {}
    for relative, expected in outputs.items():
        path = repository / str(relative)
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Output hash mismatch: {relative}")
        verified[str(relative)] = actual
    return verified


def stage20_training_projection(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return exactly the Stage 18 training fields consumed by Stage 20."""
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Stage 18 training manifest is missing runs.")
    projection = tuple(
        {
            "model_seed": int(str(row["model_seed"])),
            "first_ten_percent_test_step": int(str(row["first_ten_percent_test_step"])),
            "stable_post_step": int(str(row["stable_post_step"])),
        }
        for row in runs
    )
    if tuple(row["model_seed"] for row in projection) != (0, 1, 2, 3, 4):
        raise ValueError("Stage 20 training projection must contain seeds 0 through 4.")
    return projection
