"""Shared Stage 6D adapter helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .models import (
    RESOURCE_WARNING,
    UNRESOLVED_DECISIONS,
    DeterministicSeedEvidence,
    DiscoveryRequest,
    DiscoveryResult,
    TrajectoryEvent,
)


def deterministic_seed_evidence(
    *,
    method_name: str,
    method_version: str,
    configuration_reference: str,
    seed_value: int,
) -> DeterministicSeedEvidence:
    material = {
        "configuration_reference": configuration_reference,
        "method_name": method_name,
        "method_version": method_version,
        "seed_value": seed_value,
    }
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return DeterministicSeedEvidence(
        seed_derivation_version="stage6d-technical-seed/v1",
        seed_material_sha256=hashlib.sha256(encoded).hexdigest(),
        seed_value=seed_value,
    )


def require_request_matches_adapter(
    request: DiscoveryRequest,
    *,
    method_name: str,
    method_version: str,
) -> None:
    if request.method_name != method_name:
        raise ValueError(
            f"request method_name={request.method_name!r} "
            f"does not match adapter {method_name!r}"
        )
    if request.method_version != method_version:
        raise ValueError(
            f"request method_version={request.method_version!r} "
            f"does not match adapter {method_version!r}"
        )


def trajectory_record_shape(event: TrajectoryEvent) -> tuple[str, ...]:
    return tuple(sorted(asdict(event).keys()))


def build_discovery_result(
    *,
    request: DiscoveryRequest,
    native_consumed: int,
    native_exhausted: bool,
    exact_consumed: int,
    exact_exhausted: bool,
    exact_ledger_evidence: Mapping[str, Any],
    restart_count: int,
    proposal_count: int,
    exact_request_count: int,
    stopping_status: str,
    trajectory: list[TrajectoryEvent],
) -> DiscoveryResult:
    return DiscoveryResult(
        run_id=request.run_id,
        method_name=request.method_name,
        method_version=request.method_version,
        configuration_reference=request.configuration_reference,
        seed_evidence=request.seed_evidence,
        native_budget_unit=request.native_budget_unit,
        native_budget_allowance=request.native_budget_allowance,
        native_budget_consumed=native_consumed,
        native_budget_exhausted=native_exhausted,
        exact_evaluation_allowance=request.exact_evaluation_allowance,
        exact_evaluation_consumed=exact_consumed,
        exact_budget_exhausted=exact_exhausted,
        exact_ledger_sha256=str(exact_ledger_evidence["sha256"]),
        exact_ledger_evaluation_count=int(
            exact_ledger_evidence["evaluation_count"]
        ),
        exact_ledger_proposal_count=int(
            exact_ledger_evidence["proposal_count"]
        ),
        restart_count=restart_count,
        proposal_count=proposal_count,
        exact_request_count=exact_request_count,
        stopping_status=stopping_status,
        trajectory=tuple(trajectory),
        technical_only=True,
        production_eligible=False,
        unresolved_decisions=UNRESOLVED_DECISIONS,
        resource_warning=RESOURCE_WARNING,
    )


def event(
    *,
    sequence_index: int,
    kind: str,
    restart_index: int,
    native_consumed: int,
    exact_requested: int,
    detail: Mapping[str, Any],
) -> TrajectoryEvent:
    return TrajectoryEvent(
        sequence_index=sequence_index,
        kind=kind,
        restart_index=restart_index,
        native_consumed=native_consumed,
        exact_requested=exact_requested,
        detail=dict(detail),
    )
