"""Binary proposal extraction and Stage 6A exact-evaluation bridge."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import torch

from circuit_families.stage6a.models import (
    COMPONENT_COUNT,
    ExactEvaluationEntry,
)
from circuit_families.stage6d.budgets import (
    ExactBudgetExhausted,
    Stage6AExactEvaluationBridge,
)

from .gates import (
    GateConfig,
    GateRunIdentity,
    deterministic_gate_values,
    gate_state_record,
    validate_log_alpha,
)

PROPOSAL_RECORD_VERSION = "stage12r1-proposal/v1"
EXACT_BRIDGE_RECORD_VERSION = "stage12r1-exact-bridge/v1"

ExtractionKind = Literal["threshold", "top_k", "stochastic_draw"]


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class ProposalExtractionConfig:
    thresholds: tuple[float, ...] = ()
    top_k_sizes: tuple[int, ...] = ()
    stochastic_draws: int = 0
    max_proposals: int = 16
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.thresholds and not self.top_k_sizes and self.stochastic_draws == 0:
            raise ValueError("at least one proposal extraction rule is required")
        for threshold in self.thresholds:
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("thresholds must lie in [0, 1]")
        for size in self.top_k_sizes:
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("top_k_sizes must contain non-negative integers")
        if (
            isinstance(self.stochastic_draws, bool)
            or not isinstance(self.stochastic_draws, int)
            or self.stochastic_draws < 0
        ):
            raise ValueError("stochastic_draws must be a non-negative integer")
        if (
            isinstance(self.max_proposals, bool)
            or not isinstance(self.max_proposals, int)
            or self.max_proposals < 1
        ):
            raise ValueError("max_proposals must be a positive integer")
        if self.scientific_data:
            raise ValueError("proposal config requires scientific_data=false")
        if self.production_eligible:
            raise ValueError("proposal config requires production_eligible=false")

    def identity_sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class ProposalRecord:
    version: str
    proposal_index: int
    proposal_reference: str
    proposal_identity_sha256: str
    gate_state_sha256: str
    extraction_config_sha256: str
    extraction_kind: ExtractionKind
    extraction_value: float | int
    restart_index: int
    deterministic_identity_sha256: str
    mask: tuple[int, ...]
    mask_sha256: str
    duplicate_of_proposal_index: int | None
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.version != PROPOSAL_RECORD_VERSION:
            raise ValueError("unsupported proposal record version")
        if self.proposal_index < 0:
            raise ValueError("proposal_index must be non-negative")
        if not self.proposal_reference:
            raise ValueError("proposal_reference must be non-empty")
        if any(bit not in (0, 1) for bit in self.mask):
            raise ValueError("proposal mask must be binary")
        if self.scientific_data:
            raise ValueError("proposal records require scientific_data=false")
        if self.production_eligible:
            raise ValueError("proposal records require production_eligible=false")


@dataclass(frozen=True)
class ProposalBatch:
    proposals: tuple[ProposalRecord, ...]
    unique_masks: tuple[tuple[int, ...], ...]
    gate_state_sha256: str
    extraction_config_sha256: str

    @property
    def proposal_count(self) -> int:
        return len(self.proposals)

    @property
    def unique_mask_count(self) -> int:
        return len(self.unique_masks)


@dataclass(frozen=True)
class ExactBridgeResult:
    version: str
    terminal_state: Literal["completed", "exhausted"]
    evaluations: tuple[ExactEvaluationEntry, ...]
    proposal_count: int
    unique_proposal_mask_count: int
    exact_budget_allowance: int
    exact_budget_charged: int
    exact_ledger_sha256: str
    exact_ledger_evaluation_count: int
    exact_ledger_proposal_count: int
    exact_event_kinds: tuple[str, ...]
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.version != EXACT_BRIDGE_RECORD_VERSION:
            raise ValueError("unsupported exact bridge record version")
        if self.scientific_data:
            raise ValueError("exact bridge result requires scientific_data=false")
        if self.production_eligible:
            raise ValueError("exact bridge result requires production_eligible=false")


def _mask_sha256(mask: tuple[int, ...]) -> str:
    return _sha256({"mask": list(mask)})


def _stochastic_mask(
    probabilities: torch.Tensor,
    identity: GateRunIdentity,
) -> tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(identity.torch_seed())
    uniforms = torch.rand(
        probabilities.shape,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    probs = probabilities.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    return tuple(
        int(value)
        for value in (uniforms < probs).tolist()
    )


def extract_binary_proposals(
    *,
    log_alpha: torch.Tensor,
    component_basis_identity: str,
    component_count: int,
    gate_config: GateConfig,
    run_identity: GateRunIdentity,
    extraction_config: ProposalExtractionConfig,
) -> ProposalBatch:
    validate_log_alpha(
        log_alpha,
        component_count=component_count,
    )
    if not component_basis_identity:
        raise ValueError("component_basis_identity must be non-empty")

    for size in extraction_config.top_k_sizes:
        if size > component_count:
            raise ValueError("top-k size exceeds supplied component basis")

    state = gate_state_record(
        log_alpha,
        gate_config,
        component_basis_identity=component_basis_identity,
        component_count=component_count,
    )
    scores = deterministic_gate_values(
        log_alpha,
        gate_config,
    ).detach().cpu()

    raw: list[
        tuple[
            ExtractionKind,
            float | int,
            GateRunIdentity,
            tuple[int, ...],
        ]
    ] = []

    for index, threshold in enumerate(extraction_config.thresholds):
        identity = replace(
            run_identity,
            stream_name=f"proposal/threshold/{index}",
        )
        mask = tuple(
            int(value)
            for value in (scores >= threshold).tolist()
        )
        raw.append(("threshold", threshold, identity, mask))

    ranking = sorted(
        range(component_count),
        key=lambda index: (-float(scores[index]), index),
    )
    for index, size in enumerate(extraction_config.top_k_sizes):
        identity = replace(
            run_identity,
            stream_name=f"proposal/top-k/{index}",
        )
        retained = set(ranking[:size])
        mask = tuple(
            int(component_index in retained)
            for component_index in range(component_count)
        )
        raw.append(("top_k", size, identity, mask))

    for draw_index in range(extraction_config.stochastic_draws):
        identity = replace(
            run_identity,
            stream_name=f"proposal/stochastic/{draw_index}",
        )
        mask = _stochastic_mask(scores, identity)
        raw.append(
            (
                "stochastic_draw",
                draw_index,
                identity,
                mask,
            )
        )

    raw = raw[: extraction_config.max_proposals]

    first_by_mask: dict[tuple[int, ...], int] = {}
    proposals: list[ProposalRecord] = []

    for proposal_index, (
        kind,
        value,
        identity,
        mask,
    ) in enumerate(raw):
        duplicate_of = first_by_mask.get(mask)
        if duplicate_of is None:
            first_by_mask[mask] = proposal_index

        mask_sha = _mask_sha256(mask)
        identity_sha = identity.material_sha256()
        proposal_payload = {
            "version": PROPOSAL_RECORD_VERSION,
            "proposal_index": proposal_index,
            "gate_state_sha256": state.state_sha256,
            "extraction_config_sha256": extraction_config.identity_sha256(),
            "extraction_kind": kind,
            "extraction_value": value,
            "restart_index": run_identity.restart_index,
            "deterministic_identity_sha256": identity_sha,
            "mask_sha256": mask_sha,
            "duplicate_of_proposal_index": duplicate_of,
        }
        proposal_identity = _sha256(proposal_payload)

        proposals.append(
            ProposalRecord(
                version=PROPOSAL_RECORD_VERSION,
                proposal_index=proposal_index,
                proposal_reference=(
                    f"stage12r1-proposal://{proposal_identity}"
                ),
                proposal_identity_sha256=proposal_identity,
                gate_state_sha256=state.state_sha256,
                extraction_config_sha256=(
                    extraction_config.identity_sha256()
                ),
                extraction_kind=kind,
                extraction_value=value,
                restart_index=run_identity.restart_index,
                deterministic_identity_sha256=identity_sha,
                mask=mask,
                mask_sha256=mask_sha,
                duplicate_of_proposal_index=duplicate_of,
            )
        )

    unique_masks = tuple(first_by_mask.keys())

    return ProposalBatch(
        proposals=tuple(proposals),
        unique_masks=unique_masks,
        gate_state_sha256=state.state_sha256,
        extraction_config_sha256=extraction_config.identity_sha256(),
    )


def evaluate_proposals_exact(
    *,
    batch: ProposalBatch,
    evaluator,
    fidelity_threshold: float,
    exact_evaluation_allowance: int,
) -> ExactBridgeResult:
    if not batch.proposals:
        raise ValueError("proposal batch must not be empty")

    for proposal in batch.proposals:
        if len(proposal.mask) != COMPONENT_COUNT:
            raise ValueError(
                "current shared Stage 6A exact bridge requires "
                "the common 516-component basis"
            )

    bridge = Stage6AExactEvaluationBridge(
        evaluator=evaluator,
        fidelity_threshold=fidelity_threshold,
        allowance=exact_evaluation_allowance,
    )

    terminal_state: Literal["completed", "exhausted"] = "completed"

    for proposal in batch.proposals:
        try:
            bridge.request(
                proposal.mask,
                proposal_index=proposal.proposal_index,
            )
        except ExactBudgetExhausted:
            terminal_state = "exhausted"
            break

    if terminal_state == "completed":
        evaluations = bridge.terminate()
    else:
        evaluations = tuple(bridge.builder.evaluations)

    evidence = bridge.evidence_record()

    return ExactBridgeResult(
        version=EXACT_BRIDGE_RECORD_VERSION,
        terminal_state=terminal_state,
        evaluations=tuple(evaluations),
        proposal_count=len(batch.proposals),
        unique_proposal_mask_count=batch.unique_mask_count,
        exact_budget_allowance=exact_evaluation_allowance,
        exact_budget_charged=evidence["charged_count"],
        exact_ledger_sha256=evidence["sha256"],
        exact_ledger_evaluation_count=evidence["evaluation_count"],
        exact_ledger_proposal_count=evidence["proposal_count"],
        exact_event_kinds=tuple(
            event.kind for event in bridge.events
        ),
    )
