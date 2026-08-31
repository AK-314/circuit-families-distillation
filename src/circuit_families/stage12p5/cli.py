"""Portable validate-only command for the synthetic Stage 12-P5 lifecycle."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from circuit_families.stage12p3.records import canonical_json_bytes, canonical_sha256
from circuit_families.stage12p4.codec import atomic_encode
from circuit_families.stage12p4.records import CodecProfile

from .capacity import account_payload
from .contracts import CONDITIONS, Stage12P5ContractError, trial_from_mapping
from .runner import (
    RunInterrupted,
    build_logical_job,
    reduce_comparison_set,
    run_comparison_set,
    seal_result_with_p3,
)
from .synthetic import build_synthetic_fixture


def _expect_rejection(label: str, operation: Any) -> dict[str, str]:
    try:
        operation()
    except Exception as exc:
        return {"check": label, "result": "rejected", "error": type(exc).__name__}
    raise Stage12P5ContractError(f"adversarial validation did not reject {label}")


def validate(output_root: Path) -> dict[str, Any]:
    output_root = output_root.absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    fixture = build_synthetic_fixture(terminal_paths=True)
    interrupted_root = output_root / "interrupted-resumed"
    interruption_observed = False
    try:
        run_comparison_set(
            trial=fixture.trial,
            payloads=fixture.payloads,
            accounting=fixture.accounting,
            activation_adapter=fixture.recipient_adapter,
            recipient_model_factory=fixture.model_factory,
            outcome_adapter=fixture.outcome_adapter,
            output_root=interrupted_root,
            interrupt_after=2,
        )
    except RunInterrupted:
        interruption_observed = True
    if not interruption_observed:
        raise Stage12P5ContractError("synthetic interruption fixture did not interrupt")
    resumed = run_comparison_set(
        trial=fixture.trial,
        payloads=fixture.payloads,
        accounting=fixture.accounting,
        activation_adapter=fixture.recipient_adapter,
        recipient_model_factory=fixture.model_factory,
        outcome_adapter=fixture.outcome_adapter,
        output_root=interrupted_root,
    )
    uninterrupted = run_comparison_set(
        trial=fixture.trial,
        payloads=fixture.payloads,
        accounting=fixture.accounting,
        activation_adapter=fixture.recipient_adapter,
        recipient_model_factory=fixture.model_factory,
        outcome_adapter=fixture.outcome_adapter,
        output_root=output_root / "uninterrupted",
    )
    if resumed.result_sha256 != uninterrupted.result_sha256:
        raise Stage12P5ContractError("interrupted/resumed result differs from uninterrupted result")
    if tuple(record.condition for record in resumed.records) != CONDITIONS:
        raise Stage12P5ContractError("six-condition inventory did not close")
    lifecycle = {record.condition: record.state for record in resumed.records}
    if not {"failed", "unavailable", "censored"}.issubset(lifecycle.values()):
        raise Stage12P5ContractError("terminal-path fixtures did not cover required states")
    if not any(record.outcome.kind == "nonfinite" for record in resumed.records):
        raise Stage12P5ContractError("nonfinite technical outcome was not retained")

    trial_mapping = fixture.trial.to_mapping()
    outcome_pairing = copy.deepcopy(trial_mapping)
    outcome_pairing["pair"]["candidate_outcomes_consulted"] = True
    missing_control = copy.deepcopy(trial_mapping)
    missing_control["conditions"].pop()
    identity_swap = copy.deepcopy(trial_mapping)
    identity_swap["source_input_id"] = "input-b"
    leaked_payload = fixture.payloads[CONDITIONS[0]]
    adversarial = [
        _expect_rejection(
            "outcome-informed-pairing", lambda: trial_from_mapping(outcome_pairing)
        ),
        _expect_rejection("missing-control", lambda: trial_from_mapping(missing_control)),
        _expect_rejection("identity-swap", lambda: trial_from_mapping(identity_swap)),
        _expect_rejection(
            "capacity-side-channel",
            lambda: _require_eligible(
                account_payload(
                    leaked_payload,
                    fixture.trial.capacity,
                    scalar_precision_bits=64,
                    hidden_metadata={"alignment_matrix": "hidden"},
                )
            ),
        ),
    ]
    job = build_logical_job(fixture.trial)
    p3_evidence = seal_result_with_p3(
        trial=fixture.trial,
        result_root=interrupted_root,
        state_root=output_root / "p3-state",
    )
    reducer = reduce_comparison_set(resumed)
    report = {
        "schema_version": "stage12p5-validation-report/v1",
        "mode": "validate-only",
        "trial_id": fixture.trial.trial_id,
        "comparison_set_id": fixture.trial.comparison_set_id,
        "p3_logical_job_id": job.job_id,
        "p3_campaign_id": p3_evidence["campaign_id"],
        "p3_sealed_manifest_sha256": p3_evidence["sealed_manifest_sha256"],
        "p3_status_complete": p3_evidence["status"]["complete"],
        "alignment_plan_sha256": fixture.alignment_plan_sha256,
        "capacity_sha256": fixture.trial.capacity.capacity_sha256,
        "capacity_comparison_sha256": resumed.capacity_comparison_sha256,
        "sealed_result_sha256": resumed.result_sha256,
        "uninterrupted_result_sha256": uninterrupted.result_sha256,
        "resume_equivalent": True,
        "condition_inventory": list(CONDITIONS),
        "lifecycle_by_condition": lifecycle,
        "nonfinite_retained": True,
        "adversarial_rejections": adversarial,
        "reducer_sha256": canonical_sha256(reducer),
        "production_defaults_present": False,
        "scientific_claims_present": False,
        "registered_artifacts_accessed": False,
        "network_accessed": False,
        "scientific_data": False,
        "production_eligible": False,
    }
    profile = CodecProfile("codec/stage12p5-json/v1", "none", None)
    atomic_encode(
        output_root / "validation-report.json",
        [canonical_json_bytes(report)],
        profile,
    )
    return report


def _require_eligible(record: Any) -> None:
    if not record.eligible:
        raise Stage12P5ContractError(record.ineligibility_reason or "capacity-ineligible")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate(args.output_root)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
