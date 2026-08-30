"""Portable validate-only Stage 12-P3 synthetic campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapters import GenericJobArrayAdapter, LocalFixtureAdapter
from .contracts import SchedulerObservation, validate_operational_status
from .dag import compile_campaign
from .policy import (
    ConcurrencyProfile,
    RetryPolicy,
    SheddingPolicy,
    TierRule,
    WorkerCapabilities,
)
from .records import (
    CampaignManifest,
    ExpectedArtifact,
    HashBoundReference,
    LogicalJobSpec,
    OutputContract,
    PriorityClass,
    ResourceClass,
    canonical_json_bytes,
    canonical_sha256,
)
from .state import CampaignController

_FIXTURE_SHA = "a" * 64


def _reference(name: str) -> HashBoundReference:
    return HashBoundReference(f"synthetic://{name}", _FIXTURE_SHA, f"{name}/v1")


def _job(
    family: str,
    *,
    dependencies: tuple[str, ...] = (),
    tier: str = "tier1-technical",
    priority: str = "priority/primary-technical/v1",
) -> LogicalJobSpec:
    return LogicalJobSpec(
        family=family,
        producer_interface_version=f"{family}/v1",
        dependencies=dependencies,
        expected_inputs=(_reference(f"{family}-input"),),
        payload_reference=_reference(f"{family}-payload"),
        config_reference=_reference(f"{family}-config"),
        output_contract=OutputContract(
            f"manifests/{family}.json",
            f"{family}-fixture-output/v1",
            (ExpectedArtifact(f"artifacts/{family}.txt", "text/plain"),),
        ),
        resource_class_reference="resource/local-technical/v1",
        priority_class_reference=priority,
        protected_tier=tier,
        retry_seed_namespace_reference="stage12p3-fixture-seed/v1",
    )


def build_validate_only_campaign():
    p1 = _job("p1-like-producer")
    direct = _job("independent-direct-teacher-branch")
    retryable = _job("forced-retryable-fixture")
    stale = _job("forced-stale-claim-fixture")
    terminal = _job("forced-terminal-fixture")
    optional = _job(
        "optional-fixture",
        tier="tier3-optional",
        priority="priority/optional-technical/v1",
    )
    p2 = _job("p2-like-consumer", dependencies=(p1.job_id,))
    reducer = _job("sealed-ledger-fan-in-reducer", dependencies=(p2.job_id, direct.job_id))
    failed_dependant = _job("dependency-failure-fixture", dependencies=(terminal.job_id,))
    jobs = (p1, direct, retryable, stale, terminal, optional, p2, reducer, failed_dependant)
    manifest = CampaignManifest(
        manifest_reference=_reference("validate-only-campaign"),
        jobs=jobs,
        resource_classes=(
            ResourceClass(
                "resource/local-technical/v1",
                cpu_units=1,
                accelerator_capability=None,
                memory_bytes=1024,
                scratch_bytes=1024,
                walltime_seconds=10,
            ),
        ),
        priority_classes=(
            PriorityClass("priority/primary-technical/v1", 0),
            PriorityClass("priority/optional-technical/v1", 100),
        ),
    )
    return compile_campaign(manifest), {job.family: job for job in jobs}


def _fixture_worker(job: LogicalJobSpec, _seed: object) -> dict[str, bytes]:
    artifact = job.output_contract.artifacts[0]
    return {artifact.relative_path: f"technical fixture: {job.family}\n".encode("ascii")}


def run_validate_only(output_root: Path) -> dict[str, object]:
    campaign, jobs = build_validate_only_campaign()
    retry_policy = RetryPolicy(
        "retry/validate-only/v1",
        maximum_attempts=2,
        retryable_categories=("interruption", "stale_claim", "worker_error"),
        lease_seconds=5,
    )
    controller = CampaignController(campaign, output_root, retry_policy)
    p1 = jobs["p1-like-producer"]
    partial_claim = controller.claim(p1.job_id, worker_id="partial-worker", now=0)
    controller.write_artifact(
        p1.job_id,
        partial_claim["attempt_index"],
        p1.output_contract.artifacts[0].relative_path,
        b"technical fixture: p1-like-producer\n",
    )
    partial_did_not_unblock = (
        controller.store.read()["jobs"][jobs["p2-like-consumer"].job_id]["state"] == "blocked"
    )
    controller.publish_output_manifest(
        p1.job_id,
        worker_id="partial-worker",
        claim_token=partial_claim["claim_token"],
    )
    controller.complete(
        p1.job_id,
        worker_id="partial-worker",
        claim_token=partial_claim["claim_token"],
        now=1,
    )

    retry_job = jobs["forced-retryable-fixture"]
    retry_claim = controller.claim(retry_job.job_id, worker_id="retry-worker", now=0)
    controller.fail(
        retry_job.job_id,
        worker_id="retry-worker",
        claim_token=retry_claim["claim_token"],
        category="worker_error",
        detail="forced validate-only retry",
        now=1,
    )
    controller.retry(retry_job.job_id)

    stale_job = jobs["forced-stale-claim-fixture"]
    controller.claim(stale_job.job_id, worker_id="stale-worker", now=0)
    stale_result = controller.reconcile(now=6)
    controller.retry(stale_job.job_id)

    terminal_job = jobs["forced-terminal-fixture"]
    terminal_claim = controller.claim(terminal_job.job_id, worker_id="terminal-worker", now=0)
    controller.fail(
        terminal_job.job_id,
        worker_id="terminal-worker",
        claim_token=terminal_claim["claim_token"],
        category="validation_failure",
        detail="forced validate-only terminal failure",
        now=1,
    )

    shedding = SheddingPolicy(
        "shedding/validate-only/v1",
        (
            TierRule("tier1-technical", protected=True, optional=False, shedding_rank=None),
            TierRule("tier3-optional", protected=False, optional=True, shedding_rank=100),
        ),
        "validate-only-capacity-shortfall/v1",
    )
    shedding_result = controller.apply_shedding(
        maximum_retained_jobs=len(campaign.manifest.jobs) - 1,
        policy=shedding,
    )

    capabilities = WorkerCapabilities("resource/local-technical/v1", 1, (), 1024, 1024)
    concurrency = ConcurrencyProfile(
        "concurrency/validate-only/v1", (("resource/local-technical/v1", 1),)
    )
    adapter = LocalFixtureAdapter(
        {family: _fixture_worker for family in jobs if family != "forced-terminal-fixture"}
    )
    observations = []
    for now in range(10, 30):
        observation = adapter.launch_one(
            controller,
            worker_id="local-fixture-worker",
            capabilities=capabilities,
            concurrency=concurrency,
            now=now,
        )
        observations.append(observation.result)
        if observation.job_id is None:
            break

    array_adapter = GenericJobArrayAdapter()
    array_plan = array_adapter.render(
        controller, tuple(job.job_id for job in campaign.manifest.jobs)
    )
    submission = array_adapter.submission(
        array_plan,
        backend_job_id="synthetic-array-1",
        array_index=0,
        controller=controller,
    )
    scheduler_result = array_adapter.ingest(
        submission,
        SchedulerObservation(
            logical_job_id=submission.logical_job_id,
            backend_job_id=submission.backend_job_id,
            scheduler_state="finished",
            observed_sequence=1,
        ),
    )
    status = controller.status()
    validate_operational_status(status)
    state = controller.store.read()
    inventory = []
    for job_id in campaign.topological_job_ids:
        job = campaign.jobs_by_id[job_id]
        record = state["jobs"][job_id]
        inventory.append(
            {
                "job_id": job_id,
                "family": job.family,
                "protected_tier": job.protected_tier,
                "state": record["state"],
                "attempt_count": len(record["attempts"]),
                "failure_categories": [
                    attempt["failure_category"]
                    for attempt in record["attempts"]
                    if attempt["failure_category"] is not None
                ],
                "sealed_output_manifest_sha256": record["sealed_output_manifest_sha256"],
                "shed_reason": record["shed_reason"],
            }
        )
    report: dict[str, object] = {
        "schema_version": "stage12p3-validate-only-report/v1",
        "campaign_id": campaign.campaign_id,
        "partial_output_did_not_unblock": partial_did_not_unblock,
        "sealed_output_did_unblock": state["jobs"][jobs["p2-like-consumer"].job_id]["state"]
        == "succeeded",
        "forced_stale_claim_count": stale_result["stale_failures"],
        "shedding_result": shedding_result,
        "local_observations": observations,
        "generic_array_mapping": [entry.__dict__ for entry in array_plan.entries],
        "scheduler_finished_changed_sealed_state": scheduler_result["sealed_job_state_changed"],
        "status": status,
        "inventory": inventory,
        "rd_012_resolved": False,
        "rd_013_resolved": False,
        "rd_014_resolved": False,
        "scientific_data": False,
        "production_eligible": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(report))
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = run_validate_only(args.output_root.absolute())
    if args.report is not None:
        _write_report(args.report.absolute(), report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
