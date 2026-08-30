"""Portable synthetic-only Stage 12-R1 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from circuit_families.stage6a.endpoint import reduce_endpoint1
from circuit_families.stage6a.models import (
    TerminationStatus,
    canonical_mask_identity,
)
from circuit_families.stage6e import (
    ExactCandidateEvidence,
    load_technical_policy,
    qualify_and_deduplicate,
    recompute_endpoint2,
)
from circuit_families.stage12r1 import (
    GateConfig,
    GateRunIdentity,
    OptimizerConfig,
    ProposalExtractionConfig,
    evaluate_proposals_exact,
    extract_binary_proposals,
    optimize_gates,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT
    / "followup/configs/stage6e/technical_endpoint2_policy_v1.json"
)


def _sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dense(gates: torch.Tensor, step: int) -> torch.Tensor:
    del step
    return gates


def _objective(output: torch.Tensor, step: int) -> torch.Tensor:
    del step
    target = torch.cat(
        (
            torch.ones(12, dtype=output.dtype, device=output.device),
            torch.zeros(504, dtype=output.dtype, device=output.device),
        )
    )
    return ((output - target) ** 2).mean()


def _gate_config() -> GateConfig:
    return GateConfig(
        temperature=0.7,
        stretch_lower=-0.1,
        stretch_upper=1.1,
    )


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        learning_rate=0.05,
        max_steps=4,
        sparsity_coefficient=0.01,
        checkpoint_every=1,
        checkpoint_retention=2,
    )


def _identity() -> GateRunIdentity:
    return GateRunIdentity(
        method_name="stage12r1_hard_concrete",
        method_version="technical-v1",
        configuration_reference="fixture://stage12r1-part-g",
        run_id="stage12r1-portable-validation",
        condition_identity="synthetic-only",
        restart_index=0,
        seed_value=53,
    )


def _exact_evaluator(mask: tuple[int, ...]) -> float:
    retained = sum(mask)
    if retained == 516:
        return 1.0
    if retained in {6, 8, 10, 12}:
        return 0.95
    return 0.1


def _proposal_for_mask(
    proposal_by_mask: dict[tuple[int, ...], str],
    mask: tuple[int, ...],
) -> str:
    return proposal_by_mask.get(
        mask,
        "stage12r1-proposal://intact-baseline",
    )


def _evidence_records(
    *,
    exact,
    batch,
    policy,
) -> tuple[ExactCandidateEvidence, ...]:
    proposal_by_mask = {
        proposal.mask: proposal.proposal_reference
        for proposal in batch.proposals
    }
    ledger_hash = exact.exact_ledger_sha256
    records = []

    for entry in exact.evaluations:
        if not entry.qualifies:
            continue

        retained = tuple(
            index
            for index, bit in enumerate(
                next(
                    (
                        proposal.mask
                        for proposal in batch.proposals
                        if proposal.mask_sha256
                        and canonical_mask_identity(
                            tuple(
                                idx
                                for idx, value in enumerate(proposal.mask)
                                if value
                            )
                        )
                        == entry.mask_identity
                    ),
                    (1,) * 516,
                )
            )
            if bit
        )
        mask = tuple(
            int(index in set(retained))
            for index in range(516)
        )

        records.append(
            ExactCandidateEvidence(
                model_id="synthetic-stage12r1-model",
                discovery_method_id="stage12r1_hard_concrete",
                discovery_config_id="technical-part-g",
                source_budget_reference=policy.source_budget_reference,
                fidelity_metric_reference=policy.fidelity_metric_reference,
                component_basis_reference=policy.component_basis_reference,
                component_basis_size=516,
                mask=mask,
                mask_identity=entry.mask_identity,
                exact_fidelity=entry.fidelity,
                proposal_reference=_proposal_for_mask(
                    proposal_by_mask,
                    mask,
                ),
                exact_evaluation_reference=(
                    f"stage6a-exact://{entry.evaluation_order}"
                ),
                source_ledger_reference=(
                    "stage12r1-ledger://part-g"
                ),
                source_ledger_hash=ledger_hash,
                recomputed_ledger_hash=ledger_hash,
            )
        )

    return tuple(records)


def run_validation() -> dict[str, object]:
    gate_config = _gate_config()
    optimizer_config = _optimizer_config()
    identity = _identity()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        initial = torch.zeros(516, dtype=torch.float64)

        uninterrupted = optimize_gates(
            initial_log_alpha=initial,
            component_basis_identity="common-516-technical-basis",
            component_count=516,
            gate_config=gate_config,
            run_identity=identity,
            optimizer_config=optimizer_config,
            native_budget_allowance=4,
            dense_mask_adapter=_dense,
            objective_adapter=_objective,
            checkpoint_directory=root / "full",
        )

        interrupted = optimize_gates(
            initial_log_alpha=initial,
            component_basis_identity="common-516-technical-basis",
            component_count=516,
            gate_config=gate_config,
            run_identity=identity,
            optimizer_config=optimizer_config,
            native_budget_allowance=4,
            dense_mask_adapter=_dense,
            objective_adapter=_objective,
            checkpoint_directory=root / "resume",
            interrupt_predicate=lambda next_step: next_step == 2,
        )

        if interrupted.latest_checkpoint is None:
            raise RuntimeError("technical interruption did not produce checkpoint")

        resumed = optimize_gates(
            initial_log_alpha=initial,
            component_basis_identity="common-516-technical-basis",
            component_count=516,
            gate_config=gate_config,
            run_identity=identity,
            optimizer_config=optimizer_config,
            native_budget_allowance=4,
            dense_mask_adapter=_dense,
            objective_adapter=_objective,
            checkpoint_directory=root / "resume",
            resume_from=Path(interrupted.latest_checkpoint),
        )

    if resumed.gate_state_sha256 != uninterrupted.gate_state_sha256:
        raise RuntimeError("resume gate state differs from uninterrupted run")
    if resumed.trajectory != uninterrupted.trajectory:
        raise RuntimeError("resume trajectory differs from uninterrupted run")

    synthetic_gate_state = torch.cat(
        (
            torch.linspace(
                6.0,
                3.0,
                12,
                dtype=torch.float64,
            ),
            torch.full(
                (504,),
                -6.0,
                dtype=torch.float64,
            ),
        )
    )

    extraction_config = ProposalExtractionConfig(
        top_k_sizes=(6, 8, 10, 12),
        max_proposals=4,
    )

    batch = extract_binary_proposals(
        log_alpha=synthetic_gate_state,
        component_basis_identity="common-516-technical-basis",
        component_count=516,
        gate_config=gate_config,
        run_identity=identity,
        extraction_config=extraction_config,
    )

    exact = evaluate_proposals_exact(
        batch=batch,
        evaluator=_exact_evaluator,
        fidelity_threshold=0.9,
        exact_evaluation_allowance=5,
    )

    endpoint1 = reduce_endpoint1(
        exact.evaluations,
        termination=TerminationStatus(
            status="completed",
            procedure_censored=False,
        ),
    )

    policy = load_technical_policy(POLICY_PATH)
    evidence = _evidence_records(
        exact=exact,
        batch=batch,
        policy=policy,
    )

    qualification = qualify_and_deduplicate(
        evidence,
        policy,
        model_id="synthetic-stage12r1-model",
        discovery_method_id="stage12r1_hard_concrete",
        discovery_config_id="technical-part-g",
    )
    endpoint2 = recompute_endpoint2(
        qualification,
        policy,
    )
    endpoint2_again = recompute_endpoint2(
        qualification,
        policy,
    )

    if endpoint2.to_record() != endpoint2_again.to_record():
        raise RuntimeError("Stage 6E recomputation is not deterministic")

    report = {
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "registered_model_access": False,
        "native_budget_unit": resumed.native_budget_unit,
        "native_budget_consumed": resumed.native_budget_consumed,
        "resume_matched": True,
        "proposal_count": batch.proposal_count,
        "unique_mask_count": batch.unique_mask_count,
        "exact_budget_charged": exact.exact_budget_charged,
        "exact_evaluation_count": exact.exact_ledger_evaluation_count,
        "endpoint1_retained_proportion": endpoint1.retained_proportion,
        "endpoint1_mask_identity": endpoint1.mask_identity,
        "stage6e_raw_count": endpoint2.raw_candidate_count,
        "stage6e_unique_count": endpoint2.unique_candidate_count,
        "stage6e_qualified_count": endpoint2.qualified_candidate_count,
        "stage6e_packing_lower_bound": endpoint2.packing_lower_bound,
        "stage6e_record_sha256": _sha256(endpoint2.to_record()),
    }
    report["report_sha256"] = _sha256(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run portable synthetic-only Stage 12-R1 validation."
        )
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        required=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.validate_only:
        return 2

    try:
        report = run_validation()
    except Exception as exc:
        print(
            "STAGE12R1_VALIDATE=FAIL "
            f"{type(exc).__name__}:{exc}"
        )
        return 1

    for key in (
        "classification",
        "native_budget_unit",
        "native_budget_consumed",
        "resume_matched",
        "proposal_count",
        "unique_mask_count",
        "exact_budget_charged",
        "exact_evaluation_count",
        "endpoint1_retained_proportion",
        "endpoint1_mask_identity",
        "stage6e_raw_count",
        "stage6e_unique_count",
        "stage6e_qualified_count",
        "stage6e_packing_lower_bound",
        "stage6e_record_sha256",
        "report_sha256",
    ):
        print(f"{key}={report[key]}")

    print("scientific_data=NO")
    print("production_eligible=NO")
    print("registered_model_access=NO")
    print("STAGE12R1_VALIDATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
