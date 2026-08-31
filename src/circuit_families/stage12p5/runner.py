"""Deterministic, resumable, outcome-neutral Stage 12-P5 runner."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from circuit_families.stage12p3 import (
    CampaignController,
    CampaignManifest,
    ExpectedArtifact,
    HashBoundReference,
    LogicalJobSpec,
    OutputContract,
    PriorityClass,
    ResourceClass,
    RetryPolicy,
    canonical_json_bytes,
    canonical_sha256,
    compile_campaign,
)
from circuit_families.stage12p4 import CodecProfile, LedgerField, MetricSchema, write_ledger
from circuit_families.stage12p4.codec import atomic_encode

from .capacity import CapacityAccounting, validate_comparison_capacity
from .contracts import CONDITIONS, Stage12P5ContractError, TrialContract
from .fourier import ActivationAdapter, InterventionPayload


class OutcomeAdapter(Protocol):
    adapter_ref: str

    def observe(self, model: Any, *, input_id: str, condition: str) -> float: ...


class CensoredOutcome(RuntimeError):
    """Injected technical policy marks an observation censored."""


class UnavailableCondition(RuntimeError):
    """A condition is structurally unavailable and remains explicit."""


class RunInterrupted(RuntimeError):
    """Synthetic interruption after a sealed condition boundary."""


def _one_line(value: BaseException) -> str:
    text = " ".join(str(value).splitlines()).strip()
    return text or type(value).__name__


@dataclass(frozen=True)
class FailureRecord:
    category: str
    detail: str
    phase: str
    retryable: bool
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.category or not self.detail or "\n" in self.detail:
            raise Stage12P5ContractError("failure category/detail must be one non-empty line")
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P5ContractError("failure records must remain technical-only")


@dataclass(frozen=True)
class OutcomeObservation:
    outcome_adapter_ref: str
    kind: str
    value: float | None
    finite: bool | None
    raw_token: str | None
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"finite", "nonfinite", "absent"}:
            raise Stage12P5ContractError("unknown raw outcome kind")
        if self.kind == "finite" and (
            self.value is None or self.finite is not True or not math.isfinite(self.value)
        ):
            raise Stage12P5ContractError("finite outcome observation is inconsistent")
        if self.kind == "nonfinite" and (self.value is not None or self.finite is not False):
            raise Stage12P5ContractError("nonfinite outcome must use a preserved raw token")
        if self.kind == "absent" and (self.value is not None or self.finite is not None):
            raise Stage12P5ContractError("absent outcome observation is inconsistent")
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P5ContractError("outcome observations must remain technical-only")


@dataclass(frozen=True)
class ConditionExecutionRecord:
    trial_id: str
    comparison_set_id: str
    condition: str
    condition_index: int
    state: str
    payload_sha256: str
    capacity_accounting_sha256: str
    execution_evidence: Mapping[str, Any] | None
    outcome: OutcomeObservation
    failure: FailureRecord | None
    attempt_index: int = 0
    retry_index: int = 0
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS or CONDITIONS[self.condition_index] != self.condition:
            raise Stage12P5ContractError("condition identity/order mismatch")
        if self.state not in {"complete", "failed", "unavailable", "censored"}:
            raise Stage12P5ContractError("condition is not in a sealed terminal state")
        if (self.failure is None) != (self.state == "complete"):
            if self.state != "complete" and self.failure is None:
                raise Stage12P5ContractError("noncomplete condition requires failure accounting")
            if self.state == "complete" and self.failure is not None:
                raise Stage12P5ContractError("complete condition cannot carry a failure")
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P5ContractError("execution records must remain technical-only")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "stage12p5-condition-result/v1",
            "trial_id": self.trial_id,
            "comparison_set_id": self.comparison_set_id,
            "condition": self.condition,
            "condition_index": self.condition_index,
            "state": self.state,
            "payload_sha256": self.payload_sha256,
            "capacity_accounting_sha256": self.capacity_accounting_sha256,
            "execution_evidence": (
                None if self.execution_evidence is None else dict(self.execution_evidence)
            ),
            "outcome": asdict(self.outcome),
            "failure": None if self.failure is None else asdict(self.failure),
            "attempt_index": self.attempt_index,
            "retry_index": self.retry_index,
            "scientific_data": False,
            "production_eligible": False,
        }

    @property
    def record_sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True)
class ComparisonSetResult:
    trial_id: str
    comparison_set_id: str
    capacity_comparison_sha256: str
    records: tuple[ConditionExecutionRecord, ...]
    sealed: bool
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        conditions = tuple(record.condition for record in self.records)
        if self.sealed and conditions != CONDITIONS:
            raise Stage12P5ContractError("sealed comparison set lacks complete inventory")
        if not self.sealed and conditions != CONDITIONS[: len(conditions)]:
            raise Stage12P5ContractError("partial comparison inventory is not a canonical prefix")
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P5ContractError("comparison results must remain technical-only")

    @property
    def result_sha256(self) -> str:
        return canonical_sha256(self.to_mapping(include_hash=False))

    def to_mapping(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": "stage12p5-comparison-result/v1",
            "trial_id": self.trial_id,
            "comparison_set_id": self.comparison_set_id,
            "capacity_comparison_sha256": self.capacity_comparison_sha256,
            "condition_inventory": list(CONDITIONS),
            "records": [record.to_mapping() for record in self.records],
            "sealed": self.sealed,
            "scientific_data": False,
            "production_eligible": False,
        }
        if include_hash:
            value["result_sha256"] = canonical_sha256(value)
        return value


def build_logical_job(trial: TrialContract) -> LogicalJobSpec:
    """Bind the P5 trial to the existing P3 logical-job identity."""
    trial_ref = HashBoundReference(
        reference=f"stage12p5://trial/{trial.trial_id}",
        sha256=canonical_sha256(trial.to_mapping()),
        interface_version="stage12p5-trial-contract/v1",
    )
    return LogicalJobSpec(
        family="stage12p5-fourier-interchange",
        producer_interface_version="stage12p5-runner/v1",
        dependencies=(),
        expected_inputs=(trial_ref,),
        payload_reference=trial_ref,
        config_reference=HashBoundReference(
            reference=f"stage12p5://capacity/{trial.capacity.capacity_sha256}",
            sha256=trial.capacity.capacity_sha256,
            interface_version="stage12p5-capacity/v1",
        ),
        output_contract=OutputContract(
            manifest_relative_path="manifests/stage12p5-output.json",
            manifest_schema_version="stage12p5-comparison-result/v1",
            artifacts=(
                ExpectedArtifact("artifacts/conditions.ledger.gz", "application/gzip"),
                ExpectedArtifact("artifacts/technical-report.json", "application/json"),
            ),
        ),
        resource_class_reference="resource/injected-stage12p5/v1",
        priority_class_reference="priority/injected-stage12p5/v1",
        protected_tier="tier/injected-stage12p5/v1",
        retry_seed_namespace_reference=trial.seed_namespace_ref,
    )


def _codec(codec: str = "none") -> CodecProfile:
    return CodecProfile(
        reference=f"codec/stage12p5-{codec}/v1",
        codec=codec,
        compression_level=6 if codec == "gzip" else None,
    )


def _record_path(root: Path, condition_index: int) -> Path:
    return root / "sealed-conditions" / f"{condition_index:02d}.json"


def _write_record(path: Path, record: ConditionExecutionRecord) -> None:
    value = record.to_mapping()
    value["record_sha256"] = record.record_sha256
    atomic_encode(path, [canonical_json_bytes(value)], _codec())


def _load_record(
    path: Path,
    *,
    trial: TrialContract,
    payload: InterventionPayload,
    accounting: CapacityAccounting,
    condition_index: int,
) -> ConditionExecutionRecord:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage12P5ContractError("sealed condition record is corrupt") from exc
    required = {
        "schema_version",
        "trial_id",
        "comparison_set_id",
        "condition",
        "condition_index",
        "state",
        "payload_sha256",
        "capacity_accounting_sha256",
        "execution_evidence",
        "outcome",
        "failure",
        "attempt_index",
        "retry_index",
        "scientific_data",
        "production_eligible",
        "record_sha256",
    }
    if set(value) != required:
        raise Stage12P5ContractError("sealed condition fields mismatch")
    if value["schema_version"] != "stage12p5-condition-result/v1":
        raise Stage12P5ContractError("sealed condition schema mismatch")
    outcome = OutcomeObservation(**value["outcome"])
    failure = None if value["failure"] is None else FailureRecord(**value["failure"])
    record = ConditionExecutionRecord(
        trial_id=value["trial_id"],
        comparison_set_id=value["comparison_set_id"],
        condition=value["condition"],
        condition_index=value["condition_index"],
        state=value["state"],
        payload_sha256=value["payload_sha256"],
        capacity_accounting_sha256=value["capacity_accounting_sha256"],
        execution_evidence=value["execution_evidence"],
        outcome=outcome,
        failure=failure,
        attempt_index=value["attempt_index"],
        retry_index=value["retry_index"],
        scientific_data=value["scientific_data"],
        production_eligible=value["production_eligible"],
    )
    if value["record_sha256"] != record.record_sha256:
        raise Stage12P5ContractError("sealed condition record hash mismatch")
    if (
        record.trial_id != trial.trial_id
        or record.comparison_set_id != trial.comparison_set_id
        or record.condition_index != condition_index
        or record.payload_sha256 != payload.payload_sha256
        or record.capacity_accounting_sha256 != accounting.accounting_sha256
    ):
        raise Stage12P5ContractError("stale or cross-trial sealed condition evidence")
    return record


def _observe(
    adapter: OutcomeAdapter,
    model: Any,
    *,
    trial: TrialContract,
    condition: str,
) -> OutcomeObservation:
    value = float(
        adapter.observe(model, input_id=trial.recipient_input_id, condition=condition)
    )
    if math.isfinite(value):
        return OutcomeObservation(adapter.adapter_ref, "finite", value, True, None)
    token = "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    return OutcomeObservation(adapter.adapter_ref, "nonfinite", None, False, token)


def run_comparison_set(
    *,
    trial: TrialContract,
    payloads: Mapping[str, InterventionPayload],
    accounting: Mapping[str, CapacityAccounting],
    activation_adapter: ActivationAdapter,
    recipient_model_factory: Callable[[], Any],
    outcome_adapter: OutcomeAdapter,
    output_root: str | Path,
    interrupt_after: int | None = None,
) -> ComparisonSetResult:
    """Execute in canonical order and resume only exact sealed condition records."""
    if tuple(payloads) != CONDITIONS:
        raise Stage12P5ContractError("execution requires all six payloads before aligned runs")
    capacity_hash = validate_comparison_capacity(accounting, trial.capacity)
    root = Path(output_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    records: list[ConditionExecutionRecord] = []
    newly_executed = 0
    for index, condition in enumerate(CONDITIONS):
        payload = payloads[condition]
        capacity_record = accounting[condition]
        path = _record_path(root, index)
        if path.exists():
            records.append(
                _load_record(
                    path,
                    trial=trial,
                    payload=payload,
                    accounting=capacity_record,
                    condition_index=index,
                )
            )
            continue
        model = recipient_model_factory()
        try:
            evidence = activation_adapter.write(
                model,
                input_id=trial.recipient_input_id,
                location=trial.location,
                state=payload.ordinary_state,
            )
            outcome = _observe(outcome_adapter, model, trial=trial, condition=condition)
            record = ConditionExecutionRecord(
                trial_id=trial.trial_id,
                comparison_set_id=trial.comparison_set_id,
                condition=condition,
                condition_index=index,
                state="complete",
                payload_sha256=payload.payload_sha256,
                capacity_accounting_sha256=capacity_record.accounting_sha256,
                execution_evidence=evidence,
                outcome=outcome,
                failure=None,
            )
        except CensoredOutcome as exc:
            record = ConditionExecutionRecord(
                trial.trial_id,
                trial.comparison_set_id,
                condition,
                index,
                "censored",
                payload.payload_sha256,
                capacity_record.accounting_sha256,
                None,
                OutcomeObservation(outcome_adapter.adapter_ref, "absent", None, None, None),
                FailureRecord("outcome_censored", _one_line(exc), "outcome", False),
            )
        except UnavailableCondition as exc:
            record = ConditionExecutionRecord(
                trial.trial_id,
                trial.comparison_set_id,
                condition,
                index,
                "unavailable",
                payload.payload_sha256,
                capacity_record.accounting_sha256,
                None,
                OutcomeObservation(outcome_adapter.adapter_ref, "absent", None, None, None),
                FailureRecord("condition_unavailable", _one_line(exc), "intervention", False),
            )
        except Exception as exc:  # explicit technical failure accounting boundary
            record = ConditionExecutionRecord(
                trial.trial_id,
                trial.comparison_set_id,
                condition,
                index,
                "failed",
                payload.payload_sha256,
                capacity_record.accounting_sha256,
                None,
                OutcomeObservation(outcome_adapter.adapter_ref, "absent", None, None, None),
                FailureRecord(
                    type(exc).__name__,
                    _one_line(exc),
                    "execution",
                    True,
                ),
            )
        _write_record(path, record)
        records.append(record)
        newly_executed += 1
        if interrupt_after is not None and newly_executed >= interrupt_after and index < 5:
            raise RunInterrupted(f"synthetic interruption after condition index {index}")
    result = ComparisonSetResult(
        trial.trial_id,
        trial.comparison_set_id,
        capacity_hash,
        tuple(records),
        True,
    )
    write_compact_result(root, result, accounting)
    return result


def record_unavailable_comparison_set(
    *, trial: TrialContract, reason: str, output_root: str | Path
) -> ComparisonSetResult:
    """Close all six conditions as unavailable after structural control failure."""
    if not isinstance(reason, str) or not reason or "\n" in reason:
        raise Stage12P5ContractError("unavailability reason must be one non-empty line")
    records = []
    for index, condition in enumerate(CONDITIONS):
        payload_sha = canonical_sha256(
            {
                "schema_version": "stage12p5-unconstructed-payload/v1",
                "trial_id": trial.trial_id,
                "condition": condition,
                "reason": reason,
            }
        )
        accounting_sha = canonical_sha256(
            {
                "schema_version": "stage12p5-capacity-ineligible/v1",
                "capacity_sha256": trial.capacity.capacity_sha256,
                "condition": condition,
                "reason": reason,
            }
        )
        record = ConditionExecutionRecord(
            trial.trial_id,
            trial.comparison_set_id,
            condition,
            index,
            "unavailable",
            payload_sha,
            accounting_sha,
            None,
            OutcomeObservation(trial.outcome_adapter_ref, "absent", None, None, None),
            FailureRecord("comparison_set_unavailable", reason, "control_construction", False),
        )
        _write_record(_record_path(Path(output_root).absolute(), index), record)
        records.append(record)
    result = ComparisonSetResult(
        trial.trial_id,
        trial.comparison_set_id,
        canonical_sha256(
            {
                "schema_version": "stage12p5-capacity-unavailable/v1",
                "capacity_sha256": trial.capacity.capacity_sha256,
                "reason": reason,
            }
        ),
        tuple(records),
        True,
    )
    write_compact_result(Path(output_root).absolute(), result, None)
    return result


def seal_result_with_p3(
    *,
    trial: TrialContract,
    result_root: str | Path,
    state_root: str | Path,
) -> dict[str, Any]:
    """Publish the two P4 artifacts through the existing P3 sealed-output lifecycle."""
    job = build_logical_job(trial)
    resource = ResourceClass(
        "resource/injected-stage12p5/v1",
        cpu_units=1,
        accelerator_capability=None,
        memory_bytes=1,
        scratch_bytes=1,
        walltime_seconds=1,
    )
    priority = PriorityClass("priority/injected-stage12p5/v1", 0)
    manifest = CampaignManifest(
        manifest_reference=HashBoundReference(
            "stage12p5://manifest/synthetic",
            canonical_sha256(
                {
                    "trial_id": trial.trial_id,
                    "job_id": job.job_id,
                    "scientific_data": False,
                    "production_eligible": False,
                }
            ),
            "stage12p5-synthetic-manifest/v1",
        ),
        jobs=(job,),
        resource_classes=(resource,),
        priority_classes=(priority,),
    )
    campaign = compile_campaign(manifest)
    controller = CampaignController(
        campaign,
        state_root,
        RetryPolicy("retry/stage12p5-synthetic/v1", 1, (), 60),
    )
    attempt = controller.claim(job.job_id, worker_id="stage12p5-synthetic-worker", now=0)
    controller.heartbeat(
        job.job_id,
        worker_id="stage12p5-synthetic-worker",
        claim_token=attempt["claim_token"],
        now=1,
    )
    source_root = Path(result_root).absolute()
    for artifact in job.output_contract.artifacts:
        controller.write_artifact(
            job.job_id,
            attempt["attempt_index"],
            artifact.relative_path,
            (source_root / artifact.relative_path).read_bytes(),
        )
    sealed_manifest = controller.publish_output_manifest(
        job.job_id,
        worker_id="stage12p5-synthetic-worker",
        claim_token=attempt["claim_token"],
    )
    sealed_sha256 = controller.complete(
        job.job_id,
        worker_id="stage12p5-synthetic-worker",
        claim_token=attempt["claim_token"],
        now=2,
    )
    return {
        "campaign_id": campaign.campaign_id,
        "logical_job_id": job.job_id,
        "attempt_index": attempt["attempt_index"],
        "retry_index": attempt["retry_index"],
        "seed_evidence": attempt["seed_evidence"],
        "sealed_manifest_sha256": sealed_sha256,
        "sealed_manifest": sealed_manifest,
        "status": controller.status(),
        "scientific_data": False,
        "production_eligible": False,
    }


def write_compact_result(
    root: Path,
    result: ComparisonSetResult,
    accounting: Mapping[str, CapacityAccounting] | None,
) -> None:
    """Write canonical report plus P4 compact metric ledger."""
    artifacts = root / "artifacts"
    schema = MetricSchema(
        "schema/stage12p5-condition-ledger/v1",
        (
            LedgerField("condition_index", "integer", allow_negative=False),
            LedgerField("condition", "string"),
            LedgerField("status", "string"),
            LedgerField("outcome_kind", "string"),
            LedgerField("outcome_value", "number", nullable=True),
            LedgerField("record_sha256", "string"),
            LedgerField("capacity_accounting_sha256", "string"),
        ),
        ("condition_index",),
    )
    rows = [
        {
            "condition_index": record.condition_index,
            "condition": record.condition,
            "status": record.state,
            "outcome_kind": record.outcome.kind,
            "outcome_value": record.outcome.value,
            "record_sha256": record.record_sha256,
            "capacity_accounting_sha256": record.capacity_accounting_sha256,
        }
        for record in result.records
    ]
    evidence = write_ledger(
        artifacts / "conditions.ledger.gz",
        rows,
        schema=schema,
        context={
            "trial_id": result.trial_id,
            "comparison_set_id": result.comparison_set_id,
            "capacity_comparison_sha256": result.capacity_comparison_sha256,
            "scientific_data": False,
            "production_eligible": False,
        },
        profile=_codec("gzip"),
    )
    report = result.to_mapping()
    report["compact_ledger"] = {
        "logical_byte_length": evidence.logical_byte_length,
        "logical_sha256": evidence.logical_sha256,
        "compact_byte_length": evidence.compact_byte_length,
        "compact_sha256": evidence.compact_sha256,
        "row_count": evidence.row_count,
    }
    atomic_encode(
        artifacts / "technical-report.json",
        [canonical_json_bytes(report)],
        _codec(),
    )


def reduce_comparison_set(result: ComparisonSetResult) -> dict[str, Any]:
    """Reconstruct an outcome-neutral six-condition table; no superiority rule."""
    if not result.sealed or tuple(record.condition for record in result.records) != CONDITIONS:
        raise Stage12P5ContractError("reducer requires a sealed complete comparison set")
    counts: dict[str, int] = {}
    for record in result.records:
        counts[record.state] = counts.get(record.state, 0) + 1
    return {
        "schema_version": "stage12p5-technical-reducer/v1",
        "trial_id": result.trial_id,
        "comparison_set_id": result.comparison_set_id,
        "capacity_comparison_sha256": result.capacity_comparison_sha256,
        "condition_rows": [record.to_mapping() for record in result.records],
        "lifecycle_counts": dict(sorted(counts.items())),
        "claim_rule_present": False,
        "superiority_threshold_present": False,
        "scientific_data": False,
        "production_eligible": False,
    }
