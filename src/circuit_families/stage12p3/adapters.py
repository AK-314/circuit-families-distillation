"""Deterministic local fixture and generic scheduler-array adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import SchedulerCancellation, SchedulerObservation, SchedulerSubmission
from .policy import ConcurrencyProfile, WorkerCapabilities
from .records import LogicalJobSpec, Stage12P3ContractError, require_reference
from .state import CampaignController, CampaignStateError

FixtureWorker = Callable[[LogicalJobSpec, Mapping[str, Any]], Mapping[str, bytes]]


class FixtureWorkerFailure(RuntimeError):
    """Controlled technical-fixture failure with an explicit taxonomy."""

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(detail)
        self.category = category
        self.detail = detail


@dataclass(frozen=True)
class LocalRunObservation:
    job_id: str | None
    result: str
    failure_category: str | None
    scientific_data: bool = False
    production_eligible: bool = False


class LocalFixtureAdapter:
    """In-process adapter authorized only for injected harmless fixture workers."""

    def __init__(self, workers_by_family: Mapping[str, FixtureWorker]) -> None:
        if not workers_by_family:
            raise Stage12P3ContractError("local adapter requires fixture workers")
        self._workers = dict(workers_by_family)

    def launch_one(
        self,
        controller: CampaignController,
        *,
        worker_id: str,
        capabilities: WorkerCapabilities,
        concurrency: ConcurrencyProfile,
        now: int,
    ) -> LocalRunObservation:
        claim = controller.claim_next(
            worker_id=worker_id,
            capabilities=capabilities,
            concurrency=concurrency,
            now=now,
        )
        if claim is None:
            return LocalRunObservation(None, "no_ready_satisfiable_job", None)
        job_id = claim["job_id"]
        job = controller.campaign.jobs_by_id[job_id]
        try:
            worker = self._workers[job.family]
        except KeyError:
            controller.fail(
                job_id,
                worker_id=worker_id,
                claim_token=claim["claim_token"],
                category="validation_failure",
                detail="no injected technical worker for declared family",
                now=now,
            )
            return LocalRunObservation(job_id, "failed", "validation_failure")
        controller.heartbeat(
            job_id,
            worker_id=worker_id,
            claim_token=claim["claim_token"],
            now=now,
        )
        try:
            outputs = worker(job, claim["seed_evidence"])
            expected = {artifact.relative_path for artifact in job.output_contract.artifacts}
            if set(outputs) != expected:
                raise FixtureWorkerFailure(
                    "validation_failure", "fixture worker output inventory mismatch"
                )
            for relative_path, data in sorted(outputs.items()):
                controller.write_artifact(job_id, claim["attempt_index"], relative_path, data)
            controller.publish_output_manifest(
                job_id,
                worker_id=worker_id,
                claim_token=claim["claim_token"],
            )
            controller.complete(
                job_id,
                worker_id=worker_id,
                claim_token=claim["claim_token"],
                now=now,
            )
        except FixtureWorkerFailure as exc:
            controller.fail(
                job_id,
                worker_id=worker_id,
                claim_token=claim["claim_token"],
                category=exc.category,
                detail=exc.detail,
                now=now,
            )
            return LocalRunObservation(job_id, "failed", exc.category)
        except (CampaignStateError, OSError, ValueError) as exc:
            controller.fail(
                job_id,
                worker_id=worker_id,
                claim_token=claim["claim_token"],
                category="worker_error",
                detail=f"technical worker error: {type(exc).__name__}",
                now=now,
            )
            return LocalRunObservation(job_id, "failed", "worker_error")
        return LocalRunObservation(job_id, "succeeded", None)

    @staticmethod
    def status(controller: CampaignController) -> dict[str, Any]:
        return controller.status()

    @staticmethod
    def stop() -> dict[str, object]:
        return {
            "operation": "stop_dispatch_requested",
            "active_claims_cancelled": False,
            "scientific_data": False,
            "production_eligible": False,
        }

    @staticmethod
    def resume(controller: CampaignController, *, now: int) -> dict[str, int]:
        return controller.reconcile(now=now)


@dataclass(frozen=True)
class JobArrayEntry:
    array_index: int
    logical_job_id: str
    resource_class_reference: str
    priority_class_reference: str


@dataclass(frozen=True)
class GenericJobArrayPlan:
    adapter_reference: str
    entries: tuple[JobArrayEntry, ...]
    script: str
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        require_reference(self.adapter_reference, label="array adapter reference")
        indices = tuple(entry.array_index for entry in self.entries)
        if indices != tuple(range(len(self.entries))):
            raise Stage12P3ContractError("array indices must be contiguous from zero")
        identities = tuple(entry.logical_job_id for entry in self.entries)
        if len(set(identities)) != len(identities):
            raise Stage12P3ContractError("array mapping contains duplicate jobs")
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P3ContractError("generic array plan must remain technical-only")

    def to_mapping(self) -> dict[str, object]:
        return {
            "adapter_reference": self.adapter_reference,
            "entries": [entry.__dict__ for entry in self.entries],
            "script": self.script,
            "scientific_data": False,
            "production_eligible": False,
        }


class GenericJobArrayAdapter:
    """Render portable array metadata without selecting a real scheduler."""

    def __init__(self, adapter_reference: str = "generic-job-array/v1") -> None:
        require_reference(adapter_reference, label="adapter_reference")
        self.adapter_reference = adapter_reference
        self._last_observation: dict[tuple[str, str], int] = {}

    def render(
        self,
        controller: CampaignController,
        job_ids: tuple[str, ...],
    ) -> GenericJobArrayPlan:
        if len(set(job_ids)) != len(job_ids):
            raise Stage12P3ContractError("array input contains duplicate job IDs")
        jobs = []
        for job_id in sorted(job_ids):
            try:
                jobs.append(controller.campaign.jobs_by_id[job_id])
            except KeyError as exc:
                raise Stage12P3ContractError("array input contains unknown job ID") from exc
        entries = tuple(
            JobArrayEntry(
                array_index=index,
                logical_job_id=job.job_id,
                resource_class_reference=job.resource_class_reference,
                priority_class_reference=job.priority_class_reference,
            )
            for index, job in enumerate(jobs)
        )
        cases = "\n".join(
            f"  {entry.array_index}) LOGICAL_JOB_ID='{entry.logical_job_id}' ;;"
            for entry in entries
        )
        script = (
            "#!/bin/sh\nset -eu\n"
            ': "${GENERIC_ARRAY_INDEX:?inject GENERIC_ARRAY_INDEX}"\n'
            'case "$GENERIC_ARRAY_INDEX" in\n'
            f"{cases}\n"
            "  *) echo 'invalid array index' >&2; exit 64 ;;\n"
            "esac\n"
            'exec stage12p3-worker --logical-job-id "$LOGICAL_JOB_ID"\n'
        )
        return GenericJobArrayPlan(self.adapter_reference, entries, script)

    def submission(
        self,
        plan: GenericJobArrayPlan,
        *,
        backend_job_id: str,
        array_index: int,
        controller: CampaignController,
    ) -> SchedulerSubmission:
        if array_index < 0 or array_index >= len(plan.entries):
            raise Stage12P3ContractError("array submission index is out of range")
        entry = plan.entries[array_index]
        submission = SchedulerSubmission(
            logical_job_id=entry.logical_job_id,
            backend_name=self.adapter_reference,
            backend_job_id=backend_job_id,
            array_index=array_index,
        )
        submission.validate_for(controller.campaign.jobs_by_id[entry.logical_job_id])
        return submission

    def ingest(
        self,
        submission: SchedulerSubmission,
        observation: SchedulerObservation,
    ) -> dict[str, object]:
        if observation.logical_job_id != submission.logical_job_id:
            raise Stage12P3ContractError("scheduler observation logical job mismatch")
        if observation.backend_job_id != submission.backend_job_id:
            raise Stage12P3ContractError("scheduler observation backend job mismatch")
        key = (submission.backend_job_id, submission.logical_job_id)
        previous = self._last_observation.get(key, -1)
        if observation.observed_sequence <= previous:
            raise Stage12P3ContractError("stale scheduler observation")
        self._last_observation[key] = observation.observed_sequence
        return {
            "logical_job_id": observation.logical_job_id,
            "scheduler_state": observation.scheduler_state,
            "sealed_job_state_changed": False,
            "scientific_data": False,
            "production_eligible": False,
        }

    @staticmethod
    def cancellation(
        submission: SchedulerSubmission,
        *,
        cancellation_request_id: str,
    ) -> SchedulerCancellation:
        return SchedulerCancellation(
            logical_job_id=submission.logical_job_id,
            backend_job_id=submission.backend_job_id,
            cancellation_request_id=cancellation_request_id,
        )
