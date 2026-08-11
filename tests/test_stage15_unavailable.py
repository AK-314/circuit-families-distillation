"""Repository and boundary tests for the Stage 15 unavailable resolution."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from circuit_families.analysis.no_generalisation_selection import CANDIDATE_FRACTIONS
from circuit_families.stage_status import (
    StageStatus,
    load_stage15_resolution,
    require_stage15_control,
    stage16_may_proceed,
)

ROOT = Path(__file__).resolve().parents[1]
FAILURE = ROOT / "manifests/no_generalisation_control_selection_failure.json"
SELECTION = ROOT / "results/tables/stage13_no_generalisation_selection.csv"
RESOLUTION = ROOT / "manifests/stage15_no_generalisation_unavailable.json"
POST_STAGE17_FREEZE = (
    ROOT / "manifests/post_stage17_checkpoint_grid_and_concurrency_freeze.json"
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_stage13_failure_is_authoritative() -> None:
    failure = _load_json(FAILURE)
    with SELECTION.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert failure["selection_outcome"] == "no_qualifying_fraction"
    assert failure["selected_fraction"] is None
    assert failure["frozen_control_configuration_created"] is False
    assert sum(row["overall_qualification"] == "true" for row in rows) == 0
    assert all(row["overall_qualification"] == "false" for row in rows)
    assert tuple(sorted(float(row["fraction"]) for row in rows)) == CANDIDATE_FRACTIONS
    assert not (ROOT / "configs/controls/no_generalisation.yaml").exists()
    assert not (ROOT / "manifests/no_generalisation_control.json").exists()


def test_stage15_resolution_is_unavailable_and_hashes_reproduce() -> None:
    record = _load_json(RESOLUTION)
    post_stage17_freeze = _load_json(POST_STAGE17_FREEZE)
    control_seed_freeze = _load_json(
        ROOT / "manifests/post_stage17_additional_control_seed_count_freeze.json"
    )
    resolution = load_stage15_resolution(RESOLUTION)

    assert resolution.status is StageStatus.UNAVAILABLE
    assert resolution.status is not StageStatus.COMPLETED_WITH_OUTPUTS
    assert resolution.status is not StageStatus.NOT_STARTED
    assert resolution.family_size is None
    assert resolution.transfer_group_count is None
    assert record["stage13_failure_record"]["sha256"] == _sha256(FAILURE)
    assert record["stage13_selection_table"]["sha256"] == _sha256(SELECTION)
    manifest_source = ROOT / str(record["stage13_manifest"]["path"])
    assert record["stage13_manifest"]["sha256"] == _sha256(manifest_source)
    for name, document in record["source_document_hashes"].items():
        freeze_document = post_stage17_freeze["source_document_hashes"][name]
        assert document["post_resolution_sha256"] == freeze_document["pre_freeze_sha256"]
        control_document = control_seed_freeze["source_document_hashes"][name]
        assert freeze_document["post_freeze_sha256"] == control_document[
            "pre_decision_sha256"
        ]
        assert control_document["post_decision_sha256"] == _sha256(
            ROOT / control_document["path"]
        )


def test_stage15_unavailability_blocks_only_control_dependent_work() -> None:
    resolution = load_stage15_resolution(RESOLUTION)

    with pytest.raises(RuntimeError, match="Stage 15 is unavailable"):
        require_stage15_control(resolution)
    assert stage16_may_proceed(resolution)


def test_resolution_agrees_with_documents_and_creates_no_stage15_scientific_outputs() -> None:
    protocol = (ROOT / "experimental_protocol.md").read_text(encoding="utf-8")
    order = (ROOT / "implementation_order.md").read_text(encoding="utf-8")

    assert "Stage 15 administrative resolution record" in protocol
    assert "unavailable under the frozen protocol" in protocol
    assert "Stage 15-dependent comparisons: `unavailable`" in protocol
    assert "Stage 15 administrative resolution" in order
    assert "administratively resolved — unavailable" in order
    assert "next executable scientific stage: Stage 16" in order
    assert not list((ROOT / "results/tables").glob("stage15_*.csv"))
    assert not list((ROOT / "results/archives").glob("stage15_*.tar.gz"))
    assert not list((ROOT / "figures").glob("stage15_*"))
