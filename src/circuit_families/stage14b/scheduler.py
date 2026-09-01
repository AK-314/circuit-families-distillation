"""Provider-neutral placement, local, Slurm-class, and offline Mac adapters."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .records import (
    Stage14BError,
    atomic_write,
    canonical_json_bytes,
    canonical_sha256,
    with_boundary,
)

_AUTHORITY_NONCE = object()


class VerifiedLaunchAuthorization:
    """Non-serializable capability issued only after the full launch gate passes."""

    __slots__ = ("authorization_sha256", "stage14_sha", "_nonce", "_consumed")

    def __init__(self, authorization_sha256: str, stage14_sha: str, nonce: object) -> None:
        if nonce is not _AUTHORITY_NONCE:
            raise Stage14BError("launch capabilities cannot be constructed directly")
        self.authorization_sha256 = authorization_sha256
        self.stage14_sha = stage14_sha
        self._nonce = nonce
        self._consumed = False

    def consume(self) -> None:
        if self._nonce is not _AUTHORITY_NONCE or self._consumed:
            raise Stage14BError("invalid or replayed launch capability")
        self._consumed = True


def verify_launch_authorization(
    authorization: Mapping[str, Any],
    *,
    current_sha: str,
    expected_bindings: Mapping[str, str],
    operator_confirmation: str,
) -> VerifiedLaunchAuthorization:
    required = {
        "schema_version",
        "authority",
        "approved",
        "stage14_sha",
        "bindings",
        "authorization_nonce",
        "scientific_data",
        "production_eligible",
        "definitive_execution_started",
    }
    if set(authorization) != required:
        raise Stage14BError("Alex 6 launch authorization fields mismatch")
    if authorization["schema_version"] != "alex6-stage15-launch-authorization/v1":
        raise Stage14BError("wrong launch authorization schema")
    if authorization["authority"] != "Alex 6" or authorization["approved"] is not True:
        raise Stage14BError("launch authority is absent or did not approve")
    if authorization["stage14_sha"] != current_sha:
        raise Stage14BError("launch authorization does not bind the exact Stage 14 SHA")
    if authorization["bindings"] != dict(expected_bindings):
        raise Stage14BError("launch authorization binding mismatch")
    if authorization["scientific_data"] is not False:
        raise Stage14BError("launch authorization boundary mismatch")
    if authorization["production_eligible"] is not True:
        raise Stage14BError("launch authorization must explicitly grant production eligibility")
    if authorization["definitive_execution_started"] is not False:
        raise Stage14BError("authorization cannot claim execution already started")
    if operator_confirmation != f"LAUNCH_STAGE15_AT_{current_sha}":
        raise Stage14BError("explicit exact-SHA operator confirmation missing")
    digest = canonical_sha256(dict(authorization))
    return VerifiedLaunchAuthorization(digest, current_sha, _AUTHORITY_NONCE)


@dataclass(frozen=True)
class WorkerCapabilities:
    pool_reference: str
    qualified_resource_classes: tuple[str, ...]
    cpu_cores: int
    memory_bytes: int
    accelerator_count: int
    scratch_bytes: int
    maximum_wall_seconds: int
    availability_intervals: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.pool_reference or self.cpu_cores <= 0 or self.memory_bytes <= 0:
            raise Stage14BError("invalid worker capability record")
        if self.accelerator_count < 0 or self.scratch_bytes <= 0 or self.maximum_wall_seconds <= 0:
            raise Stage14BError("invalid worker capacity")


def match_capability(job: Mapping[str, Any], worker: WorkerCapabilities) -> bool:
    requirements = job["requirements"]
    interval = requirements.get("availability_interval")
    interval_ok = interval is None or any(
        start <= interval[0] and interval[1] <= end for start, end in worker.availability_intervals
    )
    return (
        requirements["resource_class"] in worker.qualified_resource_classes
        and worker.cpu_cores >= requirements["cpu_cores"]
        and worker.memory_bytes >= requirements["memory_bytes"]
        and worker.accelerator_count >= requirements["accelerators"]
        and worker.scratch_bytes >= requirements["scratch_bytes"]
        and worker.maximum_wall_seconds >= requirements["wall_seconds"]
        and interval_ok
    )


def deterministic_placement_plan(
    jobs: Sequence[Mapping[str, Any]], workers: Sequence[WorkerCapabilities]
) -> dict[str, Any]:
    placements = []
    for job in sorted(jobs, key=lambda item: (-int(item["priority"]), item["logical_job_id"])):
        eligible = [worker for worker in workers if match_capability(job, worker)]
        placements.append(
            {
                "logical_job_id": job["logical_job_id"],
                "pool_reference": eligible[0].pool_reference if eligible else None,
                "status": "PLACED" if eligible else "UNPLACED",
            }
        )
    record = with_boundary(
        {
            "schema_version": "stage14b-placement-plan/v1",
            "placements": placements,
            "all_jobs_placed": all(item["status"] == "PLACED" for item in placements),
            "logical_identity_changed": False,
        }
    )
    record["placement_sha256"] = canonical_sha256(record)
    return record


@dataclass(frozen=True)
class SlurmConfig:
    adapter_reference: str
    submit_command: str
    status_command: str
    accounting_command: str
    cancel_command: str
    account: str
    partition: str
    worker_command: str

    def __post_init__(self) -> None:
        for value in (
            self.adapter_reference,
            self.submit_command,
            self.status_command,
            self.accounting_command,
            self.cancel_command,
            self.account,
            self.partition,
            self.worker_command,
        ):
            if not value or any(character in value for character in "\n\r\x00"):
                raise Stage14BError("Slurm fields must be injected non-empty single-line values")


class SlurmClassAdapter:
    """Renders Slurm-class plans; submission always traverses the launch capability."""

    def __init__(
        self,
        config: SlurmConfig,
        *,
        executor: Callable[[list[str]], str] | None = None,
    ) -> None:
        self.config = config
        self._executor = executor

    def dry_run(
        self,
        logical_job_ids: Sequence[str],
        *,
        dependency_provider_ids: Sequence[str] = (),
        concurrency_cap: int,
    ) -> dict[str, Any]:
        if not logical_job_ids or len(logical_job_ids) != len(set(logical_job_ids)):
            raise Stage14BError("Slurm array requires unique logical jobs")
        if concurrency_cap <= 0:
            raise Stage14BError("Slurm concurrency cap must be positive")
        entries = [
            {"array_index": index, "logical_job_id": job_id}
            for index, job_id in enumerate(logical_job_ids)
        ]
        dependency = ":".join(dependency_provider_ids) if dependency_provider_ids else None
        return with_boundary(
            {
                "schema_version": "stage14b-slurm-array-plan/v1",
                "adapter_reference": self.config.adapter_reference,
                "account": self.config.account,
                "partition": self.config.partition,
                "array": f"0-{len(entries) - 1}%{concurrency_cap}",
                "dependency_provider_ids": list(dependency_provider_ids),
                "dependency_expression": f"afterok:{dependency}" if dependency else None,
                "entries": entries,
                "worker_command": self.config.worker_command,
                "provider_job_id": None,
                "dry_run": True,
            }
        )

    def submit(
        self, plan: Mapping[str, Any], *, capability: VerifiedLaunchAuthorization | None
    ) -> dict[str, Any]:
        if capability is None:
            raise Stage14BError("production submission rejected: Alex 6 authorization absent")
        capability.consume()
        if self._executor is None:
            raise Stage14BError("no authorized scheduler executor was injected")
        if plan.get("dry_run") is not True or plan.get("provider_job_id") is not None:
            raise Stage14BError("Slurm submission plan is stale or not a verified dry run")
        command = [
            self.config.submit_command,
            "--parsable",
            "--account",
            self.config.account,
            "--partition",
            self.config.partition,
            "--array",
            str(plan["array"]),
        ]
        dependency = plan.get("dependency_expression")
        if dependency:
            command.extend(["--dependency", str(dependency)])
        command.extend(["--wrap", self.config.worker_command])
        provider_job_id = self._executor(command).strip()
        if not provider_job_id or any(character.isspace() for character in provider_job_id):
            raise Stage14BError("scheduler returned an invalid provider job ID")
        return {
            "schema_version": "stage14b-slurm-submission/v1",
            "adapter_reference": self.config.adapter_reference,
            "provider_job_id": provider_job_id,
            "logical_job_ids": [item["logical_job_id"] for item in plan["entries"]],
            "authorization_sha256": capability.authorization_sha256,
            "stage14_sha": capability.stage14_sha,
            "scientific_data": False,
            "production_eligible": True,
            "definitive_execution_started": True,
        }


def build_mac_shard_bundle(
    jobs: Sequence[Mapping[str, Any]], destination: Path, *, environment_sha256: str
) -> dict[str, Any]:
    """Seal bounded eligible jobs for an outbound/offline school-Mac workflow."""
    entries = []
    for job in sorted(jobs, key=lambda item: item["logical_job_id"]):
        if job["requirements"]["accelerators"] > 0 and not job["requirements"].get("mps_eligible"):
            raise Stage14BError("Mac shard contains an ineligible accelerator job")
        entries.append(dict(job))
    manifest = with_boundary(
        {
            "schema_version": "stage14b-mac-shard/v1",
            "environment_sha256": environment_sha256,
            "jobs": entries,
            "job_count": len(entries),
            "inbound_service_required": False,
            "daemon_required": False,
            "administrator_access_required": False,
            "credentials_present": False,
        }
    )
    manifest["shard_sha256"] = canonical_sha256(manifest)
    atomic_write(destination, canonical_json_bytes(manifest))
    return manifest


def verify_mac_result_bundle(shard: Mapping[str, Any], result_path: Path) -> dict[str, Any]:
    try:
        result = json.loads(result_path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage14BError("Mac result bundle is unreadable") from exc
    expected_jobs = {job["logical_job_id"] for job in shard["jobs"]}
    actual_jobs = {item.get("logical_job_id") for item in result.get("results", [])}
    if expected_jobs != actual_jobs or result.get("shard_sha256") != shard["shard_sha256"]:
        raise Stage14BError("Mac result bundle identity or completeness mismatch")
    if result.get("scientific_data") is not False or result.get("production_eligible") is not False:
        raise Stage14BError("Mac result crossed technical boundary")
    for item in result["results"]:
        if item.get("state") not in {"sealed_success", "failed", "unavailable", "censored"}:
            raise Stage14BError("Mac result has invalid terminal state")
    return with_boundary(
        {
            "schema_version": "stage14b-mac-result-verification/v1",
            "shard_sha256": shard["shard_sha256"],
            "job_count": len(actual_jobs),
            "verification": "PASS",
        }
    )


class LocalTechnicalScheduler:
    """Deterministic dependency scheduler restricted to injected fixture workers."""

    def run(
        self,
        jobs: Sequence[Mapping[str, Any]],
        worker: Any,
        *,
        interrupted_after: int | None = None,
    ) -> dict[str, Any]:
        by_id = {job["logical_job_id"]: job for job in jobs}
        states = {job_id: "planned" for job_id in by_id}
        attempts = {job_id: 0 for job_id in by_id}
        events = []
        while True:
            ready = [
                job_id
                for job_id, job in by_id.items()
                if states[job_id] == "planned"
                and all(states.get(dep) == "sealed_success" for dep in job["dependencies"])
            ]
            if not ready:
                break
            job_id = sorted(ready)[0]
            attempts[job_id] += 1
            outcome = worker(by_id[job_id], attempts[job_id])
            states[job_id] = outcome
            events.append({"job_id": job_id, "attempt": attempts[job_id], "outcome": outcome})
            if interrupted_after is not None and len(events) == interrupted_after:
                break
        return with_boundary(
            {
                "schema_version": "stage14b-local-scheduler-state/v1",
                "states": states,
                "attempts": attempts,
                "events": events,
                "interrupted": interrupted_after is not None and len(events) == interrupted_after,
            }
        )
