"""Synthetic-only Stage 6A Endpoint 1 reduction."""

from __future__ import annotations

from collections.abc import Sequence

from .models import (
    Endpoint1Result,
    ExactEvaluationEntry,
    TerminationStatus,
)


def reduce_endpoint1(
    evaluations: Sequence[ExactEvaluationEntry],
    *,
    termination: TerminationStatus,
) -> Endpoint1Result:
    """
    Return the procedure-relative smallest qualifying retained proportion.

    This never claims a global minimum.
    """

    if not termination.status:
        raise ValueError("termination status required")

    qualifying = [
        entry
        for entry in evaluations
        if entry.qualifies
    ]

    if not qualifying:
        raise ValueError(
            "Endpoint 1 undefined without a qualifying exact entry"
        )

    best = min(
        qualifying,
        key=lambda entry: (
            entry.retained_proportion,
            entry.evaluation_order,
        ),
    )

    return Endpoint1Result(
        retained_proportion=best.retained_proportion,
        mask_identity=best.mask_identity,
        global_minimum_claim=False,
        termination_status=termination.status,
        procedure_censored=termination.procedure_censored,
    )
