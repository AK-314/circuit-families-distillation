"""Stage-boundary status handling for administrative resolutions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class StageStatus(StrEnum):
    """Scientifically meaningful states at a stage boundary."""

    COMPLETED_WITH_OUTPUTS = "completed_with_outputs"
    UNAVAILABLE = "unavailable"
    NOT_STARTED = "not_started"


@dataclass(frozen=True)
class StageResolution:
    """Validated stage state, keeping unavailable metrics undefined."""

    stage: int
    status: StageStatus
    family_size: int | None = None
    transfer_group_count: int | None = None


_STAGE15_EXECUTION_FLAGS = (
    "stage15_circuit_analysis_executed",
    "stage15_control_training_executed",
    "stage15_masking_executed",
    "stage15_sparse_search_executed",
    "stage15_diversity_search_executed",
    "stage15_transfer_analysis_executed",
)


def load_stage15_resolution(path: str | Path) -> StageResolution:
    """Load and validate the administrative Stage 15 unavailable record."""

    record: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("stage") != 15 or record.get("status") != StageStatus.UNAVAILABLE:
        raise ValueError("Stage 15 resolution must have status 'unavailable'.")
    if record.get("stage13_selection_outcome") != "no_qualifying_fraction":
        raise ValueError("Stage 15 unavailability requires the frozen Stage 13 failure.")
    if record.get("selected_fraction") is not None:
        raise ValueError("An unavailable Stage 15 cannot select a control fraction.")
    if record.get("qualifying_candidate_count") != 0:
        raise ValueError("Stage 15 unavailability requires zero qualifying candidates.")
    if record.get("frozen_control_configuration_created") is not False:
        raise ValueError("An unavailable Stage 15 cannot have a frozen control config.")
    if record.get("control_configuration") is not None:
        raise ValueError("An unavailable Stage 15 cannot name a control configuration.")
    if record.get("control_training_run") is not None:
        raise ValueError("An unavailable Stage 15 cannot name a control training run.")
    if any(record.get(field) is not False for field in _STAGE15_EXECUTION_FLAGS):
        raise ValueError("An unavailable Stage 15 cannot record scientific execution.")
    if record.get("replacement_control_introduced") is not False:
        raise ValueError("An unavailable Stage 15 cannot introduce a replacement control.")
    if record.get("qualification_criteria_changed") is not False:
        raise ValueError("Stage 15 resolution cannot change Stage 13 criteria.")
    if record.get("stage16_started") is not False:
        raise ValueError("Stage 16 must be unstarted in the Stage 15 resolution.")
    if record.get("family_size") is not None:
        raise ValueError("Unavailable Stage 15 family size must be undefined, not zero.")
    if record.get("transfer_group_count") is not None:
        raise ValueError("Unavailable Stage 15 transfer-group count must be undefined, not zero.")

    return StageResolution(stage=15, status=StageStatus.UNAVAILABLE)


def require_stage15_control(resolution: StageResolution) -> None:
    """Fail clearly when an analysis requires an actual Stage 15 control."""

    if resolution.status is not StageStatus.COMPLETED_WITH_OUTPUTS:
        raise RuntimeError(
            "A real matched no-generalisation control is required, but Stage 15 is unavailable."
        )


def stage16_may_proceed(resolution: StageResolution) -> bool:
    """Return whether genuine-task functional transfer may proceed."""

    return resolution.status in {
        StageStatus.COMPLETED_WITH_OUTPUTS,
        StageStatus.UNAVAILABLE,
    }
