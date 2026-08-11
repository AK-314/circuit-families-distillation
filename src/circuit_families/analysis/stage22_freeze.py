"""Prediction-table integrity and provisional Stage 22 resolution records."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


PREDICTION_HEADING = "## **6\\. Frozen prediction table**"
CONTROL_HEADING = "### **Control interpretation table**"


def prediction_table_block(protocol: str) -> str:
    after = protocol.split(PREDICTION_HEADING, maxsplit=1)
    if len(after) != 2:
        raise ValueError("Frozen prediction-table heading is missing.")
    section = after[1].split(CONTROL_HEADING, maxsplit=1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    if len(lines) != 11:
        raise ValueError("Frozen prediction table must contain one header and nine data rows.")
    return "\n".join(lines) + "\n"


def prediction_table_sha256(protocol: str) -> str:
    return hashlib.sha256(prediction_table_block(protocol).encode("utf-8")).hexdigest()


def prediction_quantities(protocol: str) -> tuple[str, ...]:
    lines = prediction_table_block(protocol).splitlines()[2:]
    return tuple(line.split("|", maxsplit=2)[1].strip() for line in lines)


def resolution_rows(protocol: str) -> tuple[dict[str, object], ...]:
    resolutions = {
        "Recovered structural family size": (
            "Unresolved",
            "Primary pre-grid families are empty, so the observed change is emergence of sparse "
            "recoverability rather than an ordinary contraction estimate.",
            "stage18_family_summary;stage20_paired_deltas",
        ),
        "Transfer-distinct group count": (
            "Unresolved",
            "Transfer-group counts are undefined for the empty primary pre-grid families.",
            "stage20_seed_checkpoint_metrics",
        ),
        "Circuit size": (
            "Unresolved",
            "Circuit size is undefined where the primary family is empty.",
            "stage20_seed_checkpoint_metrics",
        ),
        "Pairwise structural overlap": (
            "Unresolved",
            "Pairwise overlap is undefined where fewer than two primary circuits exist.",
            "stage20_seed_checkpoint_metrics",
        ),
        "Cross-subset transfer": (
            "Unresolved",
            "Cross-subset transfer is undefined for empty primary families and cannot support a "
            "complete pre-to-post directional comparison.",
            "stage20_seed_checkpoint_metrics;stage21_figure4_transfer_source",
        ),
        "Timing of change": (
            "Unresolved",
            "Emergence timing varies by seed and 35 fixed-grid comparisons are phase-misaligned; "
            "the no-generalisation control is unavailable.",
            "stage20_comparison_registry;stage15_unavailable",
        ),
        "Matched-fidelity result": (
            "Unresolved",
            "The required pre-grid primary circuits do not exist, so matched-fidelity endpoint "
            "comparisons cannot be formed.",
            "stage19_matched_fidelity_summary",
        ),
        "Matched-sparsity result": (
            "Unresolved",
            "The required pre-grid primary circuits do not exist, so matched-sparsity endpoint "
            "comparisons cannot be formed.",
            "stage19_matched_sparsity_summary",
        ),
        "Empty-family transition": (
            "Supported",
            "All 330 empty cells are reported explicitly, never imputed, and excluded only from "
            "metrics that require circuits.",
            "stage19_empty_cells;stage20_paired_deltas",
        ),
    }
    quantities = prediction_quantities(protocol)
    if set(quantities) != set(resolutions):
        raise ValueError("Prediction resolution map does not match the frozen table.")
    return tuple(
        {
            "prediction_index": index,
            "quantity": quantity,
            "resolution_category": resolutions[quantity][0],
            "resolution_reason": resolutions[quantity][1],
            "evidence_sources": resolutions[quantity][2],
            "provisional_pending_stage18_reproduction": True,
        }
        for index, quantity in enumerate(quantities, start=1)
    )


def freeze_rows() -> Sequence[dict[str, object]]:
    return (
        {"field": "included_main_seeds", "value": "0,1,2,3,4"},
        {"field": "checkpoint_steps", "value": "200,3400,7450,8150,8500,8650,9050"},
        {"field": "primary_fidelity", "value": "0.990000"},
        {"field": "fidelity_grid", "value": "0.800,0.850,0.900,0.950,0.975,0.990"},
        {"field": "primary_distinctness_cutoff", "value": "0.50"},
        {"field": "distinctness_grid", "value": "0.25,0.50,0.75"},
        {"field": "transfer_grouping", "value": "complete_linkage_max_distance_0.05"},
        {"field": "matched_fidelity_tolerance", "value": "0.01"},
        {"field": "matched_sparsity_tolerance_components", "value": "5"},
        {"field": "independent_unit", "value": "trained_model_seed"},
        {"field": "control_random_label", "value": "single_seed_descriptive"},
        {"field": "control_no_generalisation", "value": "unavailable_stage15"},
        {"field": "principal_figure_count", "value": "5"},
        {"field": "freeze_status", "value": "provisional_pending_stage18_reproduction"},
    )
