"""Durable atomic campaign state, claiming, leases, and sealed output evidence."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import secrets
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .dag import CompiledCampaign
from .policy import (
    FAILURE_CATEGORIES,
    ConcurrencyProfile,
    RetryPolicy,
    SheddingPolicy,
    WorkerCapabilities,
)
from .records import (
    Stage12P3ContractError,
    canonical_json_bytes,
    canonical_sha256,
    safe_relative_path,
)

STATE_SCHEMA_VERSION: Final = "stage12p3-state/v1"
SEALED_OUTPUT_SCHEMA_VERSION: Final = "stage12p3-sealed-output/v1"
JOB_STATES: Final = frozenset(
    {
        "planned",
        "blocked",
        "ready",
        "claimed",
        "running",
        "succeeded",
        "retryable_failure",
        "terminal_failure",
        "shed_unavailable",
    }
)
ACTIVE_STATES: Final = frozenset({"claimed", "running"})
TERMINAL_STATES: Final = frozenset({"succeeded", "terminal_failure", "shed_unavailable"})

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class CampaignStateError(Stage12P3ContractError):
    """Raised when durable state or a transition violates the campaign contract."""


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_no_symlink(root: Path, candidate: Path) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise CampaignStateError("path escapes campaign state root") from exc
    current = root
    if current.is_symlink():
        raise CampaignStateError("campaign state root must not be a symlink")
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise CampaignStateError("path crosses a symlink component")


def _state_digest(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "state_sha256"}
    return canonical_sha256(payload)


class DurableStateStore:
    """One compact JSON state file guarded by thread and filesystem locks."""

    def __init__(self, root: str | Path, campaign: CompiledCampaign) -> None:
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise CampaignStateError("state root must not be a symlink")
        self.state_path = self.root / "state.json"
        self.lock_path = self.root / "state.lock"
        self.campaign = campaign
        if not self.state_path.exists():
            initial = {
                "schema_version": STATE_SCHEMA_VERSION,
                "campaign_id": campaign.campaign_id,
                "scientific_data": False,
                "production_eligible": False,
                "incomplete_campaign_reason": None,
                "jobs": {
                    job.job_id: {
                        "job_id": job.job_id,
                        "job_spec_sha256": canonical_sha256(job.to_mapping()),
                        "state": "planned",
                        "blocked_reason": None,
                        "attempts": [],
                        "sealed_output_manifest_sha256": None,
                        "shed_reason": None,
                    }
                    for job in campaign.manifest.jobs
                },
            }
            initial["state_sha256"] = _state_digest(initial)
            with self._locked():
                if not self.state_path.exists():
                    _atomic_write(self.state_path, canonical_json_bytes(initial))
        self.read()

    @contextmanager
    def _locked(self):
        with _thread_lock(self.lock_path):
            with self.lock_path.open("a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="ascii"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStateError("state store is unreadable") from exc
        self._validate(value)
        return value

    def _validate(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise CampaignStateError("state store must contain a mapping")
        if value.get("schema_version") != STATE_SCHEMA_VERSION:
            raise CampaignStateError("state schema mismatch")
        if value.get("state_sha256") != _state_digest(value):
            raise CampaignStateError("state store integrity hash mismatch")
        if value.get("campaign_id") != self.campaign.campaign_id:
            raise CampaignStateError("state campaign hash mismatch")
        if (
            value.get("scientific_data") is not False
            or value.get("production_eligible") is not False
        ):
            raise CampaignStateError("state store crossed the scientific boundary")
        expected = {
            job.job_id: canonical_sha256(job.to_mapping()) for job in self.campaign.manifest.jobs
        }
        jobs = value.get("jobs")
        if not isinstance(jobs, dict) or set(jobs) != set(expected):
            raise CampaignStateError("state job identity set mismatch")
        for job_id, record in jobs.items():
            if not isinstance(record, dict) or record.get("job_id") != job_id:
                raise CampaignStateError("state job record identity mismatch")
            if record.get("job_spec_sha256") != expected[job_id]:
                raise CampaignStateError("state job hash mismatch")
            if record.get("state") not in JOB_STATES:
                raise CampaignStateError("state contains an invalid lifecycle value")
            if not isinstance(record.get("attempts"), list):
                raise CampaignStateError("state attempts must be a list")
            for index, attempt in enumerate(record["attempts"]):
                if not isinstance(attempt, dict) or attempt.get("attempt_index") != index:
                    raise CampaignStateError("attempt history is not append-only canonical order")

    def read(self) -> dict[str, Any]:
        with self._locked():
            return copy.deepcopy(self._read_unlocked())

    def mutate(self, operation: Callable[[dict[str, Any]], Any]) -> Any:
        with self._locked():
            state = self._read_unlocked()
            result = operation(state)
            state["state_sha256"] = _state_digest(state)
            self._validate(state)
            _atomic_write(self.state_path, canonical_json_bytes(state))
            return result


class CampaignController:
    """Validated transition engine over a durable campaign store."""

    def __init__(
        self,
        campaign: CompiledCampaign,
        state_root: str | Path,
        retry_policy: RetryPolicy,
    ) -> None:
        self.campaign = campaign
        self.retry_policy = retry_policy
        self.store = DurableStateStore(state_root, campaign)
        self._refresh_readiness()

    def _refresh_readiness(self) -> None:
        by_id = self.campaign.jobs_by_id

        def operation(state: dict[str, Any]) -> None:
            for job_id in self.campaign.topological_job_ids:
                record = state["jobs"][job_id]
                if record["state"] in ACTIVE_STATES | TERMINAL_STATES | {"retryable_failure"}:
                    continue
                dependencies = [state["jobs"][item]["state"] for item in by_id[job_id].dependencies]
                if all(item == "succeeded" for item in dependencies):
                    record["state"] = "ready"
                    record["blocked_reason"] = None
                elif any(item in {"terminal_failure", "shed_unavailable"} for item in dependencies):
                    record["state"] = "blocked"
                    record["blocked_reason"] = "dependency_failure"
                else:
                    record["state"] = "blocked" if dependencies else "ready"
                    record["blocked_reason"] = "dependencies_not_sealed" if dependencies else None

        self.store.mutate(operation)

    @staticmethod
    def _seed_evidence(
        job_id: str, attempt_index: int, retry_index: int, namespace: str
    ) -> dict[str, Any]:
        material = (
            "stage12p3-seed:v1\n"
            f"namespace={namespace}\njob_id={job_id}\n"
            f"attempt_index={attempt_index}\nretry_index={retry_index}\n"
        )
        digest = hashlib.sha256(material.encode("ascii")).hexdigest()
        return {
            "namespace": namespace,
            "material": material,
            "sha256": digest,
            "value": int(digest[:16], 16),
        }

    def claim(self, job_id: str, *, worker_id: str, now: int) -> dict[str, Any]:
        if not isinstance(worker_id, str) or not worker_id:
            raise CampaignStateError("worker_id must be non-empty")
        if isinstance(now, bool) or not isinstance(now, int) or now < 0:
            raise CampaignStateError("now must be a non-negative integer")
        job = self.campaign.jobs_by_id.get(job_id)
        if job is None:
            raise CampaignStateError("cannot claim an unknown job")

        def operation(state: dict[str, Any]) -> dict[str, Any]:
            record = state["jobs"][job_id]
            if record["state"] != "ready":
                raise CampaignStateError(f"job is not ready: {record['state']}")
            attempt_index = len(record["attempts"])
            retry_index = attempt_index
            if attempt_index >= self.retry_policy.maximum_attempts:
                raise CampaignStateError("retry policy attempt bound exhausted")
            token = secrets.token_hex(32)
            attempt = {
                "attempt_index": attempt_index,
                "retry_index": retry_index,
                "state": "claimed",
                "worker_id": worker_id,
                "claim_token": token,
                "claimed_at": now,
                "heartbeat_at": now,
                "lease_deadline": now + self.retry_policy.lease_seconds,
                "ended_at": None,
                "failure_category": None,
                "failure_detail": None,
                "output_manifest_sha256": None,
                "seed_evidence": self._seed_evidence(
                    job_id, attempt_index, retry_index, job.retry_seed_namespace_reference
                ),
            }
            record["attempts"].append(attempt)
            record["state"] = "claimed"
            return copy.deepcopy(attempt)

        return self.store.mutate(operation)

    def claim_next(
        self,
        *,
        worker_id: str,
        capabilities: WorkerCapabilities,
        concurrency: ConcurrencyProfile,
        now: int,
    ) -> dict[str, Any] | None:
        """Atomically select and claim the highest-priority satisfiable job."""
        resources = {
            resource.reference: resource for resource in self.campaign.manifest.resource_classes
        }
        priorities = {
            priority.reference: priority for priority in self.campaign.manifest.priority_classes
        }

        def operation(state: dict[str, Any]) -> dict[str, Any] | None:
            active_counts: dict[str, int] = {}
            for active_job_id, record in state["jobs"].items():
                if record["state"] in ACTIVE_STATES:
                    reference = self.campaign.jobs_by_id[active_job_id].resource_class_reference
                    active_counts[reference] = active_counts.get(reference, 0) + 1

            candidates = []
            for candidate_id, record in state["jobs"].items():
                if record["state"] != "ready":
                    continue
                job = self.campaign.jobs_by_id[candidate_id]
                resource = resources[job.resource_class_reference]
                if not capabilities.satisfies(resource):
                    continue
                if active_counts.get(resource.reference, 0) >= concurrency.limit_for(
                    resource.reference
                ):
                    continue
                candidates.append(job)
            if not candidates:
                return None
            selected = min(
                candidates,
                key=lambda job: (
                    priorities[job.priority_class_reference].dispatch_rank,
                    job.job_id,
                ),
            )
            record = state["jobs"][selected.job_id]
            attempt_index = len(record["attempts"])
            if attempt_index >= self.retry_policy.maximum_attempts:
                raise CampaignStateError("retry policy attempt bound exhausted")
            token = secrets.token_hex(32)
            attempt = {
                "attempt_index": attempt_index,
                "retry_index": attempt_index,
                "state": "claimed",
                "worker_id": worker_id,
                "claim_token": token,
                "claimed_at": now,
                "heartbeat_at": now,
                "lease_deadline": now + self.retry_policy.lease_seconds,
                "ended_at": None,
                "failure_category": None,
                "failure_detail": None,
                "output_manifest_sha256": None,
                "seed_evidence": self._seed_evidence(
                    selected.job_id,
                    attempt_index,
                    attempt_index,
                    selected.retry_seed_namespace_reference,
                ),
            }
            record["attempts"].append(attempt)
            record["state"] = "claimed"
            return {"job_id": selected.job_id, **copy.deepcopy(attempt)}

        return self.store.mutate(operation)

    def apply_shedding(
        self,
        *,
        maximum_retained_jobs: int,
        policy: SheddingPolicy,
    ) -> dict[str, Any]:
        """Shed only optional unstarted work, preserving every inventory row."""
        if (
            isinstance(maximum_retained_jobs, bool)
            or not isinstance(maximum_retained_jobs, int)
            or maximum_retained_jobs < 0
        ):
            raise CampaignStateError("maximum_retained_jobs must be non-negative")
        rules = {
            job.job_id: policy.rule_for(job.protected_tier) for job in self.campaign.manifest.jobs
        }

        def operation(state: dict[str, Any]) -> dict[str, Any]:
            retained = [
                job_id
                for job_id, record in state["jobs"].items()
                if record["state"] != "shed_unavailable"
            ]
            required = max(0, len(retained) - maximum_retained_jobs)
            candidates = [
                job_id
                for job_id in retained
                if rules[job_id].optional
                and not rules[job_id].protected
                and state["jobs"][job_id]["state"] in {"planned", "blocked", "ready"}
                and not state["jobs"][job_id]["attempts"]
            ]
            candidates.sort(
                key=lambda job_id: (
                    -int(rules[job_id].shedding_rank),
                    job_id,
                )
            )
            shed = candidates[:required]
            for job_id in shed:
                record = state["jobs"][job_id]
                record["state"] = "shed_unavailable"
                record["shed_reason"] = policy.reason
            unsatisfied = required - len(shed)
            if shed or unsatisfied:
                state["incomplete_campaign_reason"] = policy.reason
            return {
                "shed_job_ids": shed,
                "requested_shed_count": required,
                "unsatisfied_scarcity_count": unsatisfied,
                "protected_jobs_shed": 0,
                "reason": policy.reason if shed or unsatisfied else None,
            }

        result = self.store.mutate(operation)
        self._refresh_readiness()
        return result

    def heartbeat(self, job_id: str, *, worker_id: str, claim_token: str, now: int) -> None:
        def operation(state: dict[str, Any]) -> None:
            record = state["jobs"][job_id]
            if record["state"] not in ACTIVE_STATES or not record["attempts"]:
                raise CampaignStateError("job has no active claim")
            attempt = record["attempts"][-1]
            if attempt["worker_id"] != worker_id or attempt["claim_token"] != claim_token:
                raise CampaignStateError("heartbeat worker or claim token mismatch")
            if now > attempt["lease_deadline"]:
                raise CampaignStateError("cannot heartbeat an expired lease")
            if now < attempt["heartbeat_at"]:
                raise CampaignStateError("heartbeat time cannot move backwards")
            attempt["heartbeat_at"] = now
            attempt["lease_deadline"] = now + self.retry_policy.lease_seconds
            attempt["state"] = "running"
            record["state"] = "running"

        self.store.mutate(operation)

    def attempt_root(self, job_id: str, attempt_index: int) -> Path:
        if job_id not in self.campaign.jobs_by_id:
            raise CampaignStateError("unknown job output root")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise CampaignStateError("attempt index must be non-negative")
        path = self.store.root / "jobs" / job_id / "attempts" / f"attempt-{attempt_index:06d}"
        _ensure_no_symlink(self.store.root, path)
        path.mkdir(parents=True, exist_ok=True)
        _ensure_no_symlink(self.store.root, path)
        return path

    def write_artifact(
        self, job_id: str, attempt_index: int, relative_path: str, data: bytes
    ) -> str:
        try:
            safe_relative_path(relative_path, label="artifact path")
        except Stage12P3ContractError as exc:
            raise CampaignStateError(str(exc)) from exc
        if not isinstance(data, bytes):
            raise CampaignStateError("artifact data must be bytes")
        root = self.attempt_root(job_id, attempt_index)
        path = root.joinpath(*PurePosixPath(relative_path).parts)
        _ensure_no_symlink(root, path)
        if path.exists():
            existing = path.read_bytes()
            if existing != data:
                raise CampaignStateError("conflicting artifact publication")
        else:
            _atomic_write(path, data)
        return hashlib.sha256(data).hexdigest()

    def publish_output_manifest(
        self,
        job_id: str,
        *,
        worker_id: str,
        claim_token: str,
    ) -> dict[str, Any]:
        state = self.store.read()
        record = state["jobs"][job_id]
        if record["state"] not in ACTIVE_STATES or not record["attempts"]:
            raise CampaignStateError("job has no active attempt to publish")
        attempt = record["attempts"][-1]
        if attempt["worker_id"] != worker_id or attempt["claim_token"] != claim_token:
            raise CampaignStateError("publication worker or claim token mismatch")
        job = self.campaign.jobs_by_id[job_id]
        root = self.attempt_root(job_id, attempt["attempt_index"])
        artifacts = []
        for expected in job.output_contract.artifacts:
            path = root.joinpath(*PurePosixPath(expected.relative_path).parts)
            _ensure_no_symlink(root, path)
            if not path.is_file():
                raise CampaignStateError(f"expected output is missing: {expected.relative_path}")
            data = path.read_bytes()
            artifacts.append(
                {
                    "relative_path": expected.relative_path,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "media_type": expected.media_type,
                }
            )
        manifest = {
            "schema_version": SEALED_OUTPUT_SCHEMA_VERSION,
            "declared_output_schema_version": job.output_contract.manifest_schema_version,
            "campaign_id": self.campaign.campaign_id,
            "job_id": job_id,
            "attempt_index": attempt["attempt_index"],
            "retry_index": attempt["retry_index"],
            "artifacts": artifacts,
            "scientific_data": False,
            "production_eligible": False,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        path = root.joinpath(*PurePosixPath(job.output_contract.manifest_relative_path).parts)
        _ensure_no_symlink(root, path)
        if path.exists() and path.read_bytes() != manifest_bytes:
            raise CampaignStateError("conflicting output manifest publication")
        if not path.exists():
            _atomic_write(path, manifest_bytes)
        return manifest

    def _validated_manifest(
        self, job_id: str, attempt: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str]:
        job = self.campaign.jobs_by_id[job_id]
        root = self.attempt_root(job_id, attempt["attempt_index"])
        path = root.joinpath(*PurePosixPath(job.output_contract.manifest_relative_path).parts)
        _ensure_no_symlink(root, path)
        try:
            raw = path.read_bytes()
            manifest = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStateError("sealed output manifest is missing or invalid") from exc
        expected_scalar = {
            "schema_version": SEALED_OUTPUT_SCHEMA_VERSION,
            "declared_output_schema_version": job.output_contract.manifest_schema_version,
            "campaign_id": self.campaign.campaign_id,
            "job_id": job_id,
            "attempt_index": attempt["attempt_index"],
            "retry_index": attempt["retry_index"],
            "scientific_data": False,
            "production_eligible": False,
        }
        for key, value in expected_scalar.items():
            if manifest.get(key) != value:
                raise CampaignStateError(f"sealed output manifest {key} mismatch")
        artifacts = manifest.get("artifacts")
        expected_paths = {item.relative_path: item for item in job.output_contract.artifacts}
        if not isinstance(artifacts, list) or {
            item.get("relative_path") for item in artifacts
        } != set(expected_paths):
            raise CampaignStateError("sealed output artifact inventory mismatch")
        for item in artifacts:
            relative = safe_relative_path(item["relative_path"], label="sealed artifact path")
            path = root.joinpath(*PurePosixPath(relative).parts)
            _ensure_no_symlink(root, path)
            data = path.read_bytes()
            if item.get("sha256") != hashlib.sha256(data).hexdigest() or item.get(
                "size_bytes"
            ) != len(data):
                raise CampaignStateError("sealed output artifact hash or size mismatch")
            if item.get("media_type") != expected_paths[relative].media_type:
                raise CampaignStateError("sealed output artifact media type mismatch")
        canonical = canonical_json_bytes(manifest)
        if raw != canonical:
            raise CampaignStateError("sealed output manifest is not canonical")
        return manifest, hashlib.sha256(raw).hexdigest()

    def complete(
        self,
        job_id: str,
        *,
        worker_id: str,
        claim_token: str,
        now: int,
    ) -> str:
        def operation(state: dict[str, Any]) -> str:
            record = state["jobs"][job_id]
            if record["state"] == "succeeded":
                attempt = record["attempts"][-1]
                if attempt["worker_id"] != worker_id or attempt["claim_token"] != claim_token:
                    raise CampaignStateError("completion worker or claim token mismatch")
                return record["sealed_output_manifest_sha256"]
            if record["state"] not in ACTIVE_STATES or not record["attempts"]:
                raise CampaignStateError("job has no active attempt to complete")
            attempt = record["attempts"][-1]
            if attempt["worker_id"] != worker_id or attempt["claim_token"] != claim_token:
                raise CampaignStateError("completion worker or claim token mismatch")
            _, digest = self._validated_manifest(job_id, attempt)
            attempt["state"] = "succeeded"
            attempt["ended_at"] = now
            attempt["output_manifest_sha256"] = digest
            record["state"] = "succeeded"
            record["sealed_output_manifest_sha256"] = digest
            return digest

        digest = self.store.mutate(operation)
        self._refresh_readiness()
        return digest

    def fail(
        self,
        job_id: str,
        *,
        worker_id: str,
        claim_token: str,
        category: str,
        detail: str,
        now: int,
    ) -> str:
        if category not in FAILURE_CATEGORIES:
            raise CampaignStateError("unknown failure category")
        if not isinstance(detail, str) or not detail or "\n" in detail or "\r" in detail:
            raise CampaignStateError("failure detail must be one non-empty line")

        def operation(state: dict[str, Any]) -> str:
            record = state["jobs"][job_id]
            if record["state"] in {"retryable_failure", "terminal_failure"}:
                previous = record["attempts"][-1]
                if previous["worker_id"] == worker_id and previous["claim_token"] == claim_token:
                    return record["state"]
                raise CampaignStateError("failure report conflicts with terminal attempt")
            if record["state"] not in ACTIVE_STATES:
                raise CampaignStateError("job has no active attempt to fail")
            attempt = record["attempts"][-1]
            if attempt["worker_id"] != worker_id or attempt["claim_token"] != claim_token:
                raise CampaignStateError("failure worker or claim token mismatch")
            retryable = (
                category in self.retry_policy.retryable_categories
                and len(record["attempts"]) < self.retry_policy.maximum_attempts
            )
            resulting = "retryable_failure" if retryable else "terminal_failure"
            attempt["state"] = resulting
            attempt["ended_at"] = now
            attempt["failure_category"] = category
            attempt["failure_detail"] = detail
            record["state"] = resulting
            return resulting

        resulting = self.store.mutate(operation)
        self._refresh_readiness()
        return resulting

    def retry(self, job_id: str) -> None:
        def operation(state: dict[str, Any]) -> None:
            record = state["jobs"][job_id]
            if record["state"] != "retryable_failure":
                raise CampaignStateError("only retryable failures may be readied")
            record["state"] = "ready"

        self.store.mutate(operation)

    def reconcile(self, *, now: int) -> dict[str, int]:
        recovered_successes = 0
        stale_failures = 0
        state = self.store.read()
        for job_id in self.campaign.topological_job_ids:
            record = state["jobs"][job_id]
            if record["state"] not in ACTIVE_STATES:
                continue
            attempt = record["attempts"][-1]
            try:
                self._validated_manifest(job_id, attempt)
            except CampaignStateError:
                if now <= attempt["lease_deadline"]:
                    continue
                self.fail(
                    job_id,
                    worker_id=attempt["worker_id"],
                    claim_token=attempt["claim_token"],
                    category="stale_claim",
                    detail="lease expired without sealed output",
                    now=now,
                )
                stale_failures += 1
            else:
                self.complete(
                    job_id,
                    worker_id=attempt["worker_id"],
                    claim_token=attempt["claim_token"],
                    now=now,
                )
                recovered_successes += 1
        self._refresh_readiness()
        return {"recovered_successes": recovered_successes, "stale_failures": stale_failures}

    def status(self) -> dict[str, Any]:
        state = self.store.read()
        counts: dict[str, int] = {}
        by_family: dict[str, dict[str, int]] = {}
        by_tier: dict[str, dict[str, int]] = {}
        by_resource: dict[str, dict[str, int]] = {}
        by_failure_category: dict[str, int] = {}
        for job_id, record in state["jobs"].items():
            lifecycle = record["state"]
            job = self.campaign.jobs_by_id[job_id]
            counts[lifecycle] = counts.get(lifecycle, 0) + 1
            for bucket, key in (
                (by_family, job.family),
                (by_tier, job.protected_tier),
                (by_resource, job.resource_class_reference),
            ):
                bucket.setdefault(key, {})[lifecycle] = (
                    bucket.setdefault(key, {}).get(lifecycle, 0) + 1
                )
            for attempt in record["attempts"]:
                category = attempt["failure_category"]
                if category is not None:
                    by_failure_category[category] = by_failure_category.get(category, 0) + 1
        return {
            "schema_version": "stage12p3-operational-status/v1",
            "campaign_id": self.campaign.campaign_id,
            "counts": dict(sorted(counts.items())),
            "by_family": {
                key: dict(sorted(value.items())) for key, value in sorted(by_family.items())
            },
            "by_tier": {key: dict(sorted(value.items())) for key, value in sorted(by_tier.items())},
            "by_resource_class": {
                key: dict(sorted(value.items())) for key, value in sorted(by_resource.items())
            },
            "by_failure_category": dict(sorted(by_failure_category.items())),
            "retained_job_ids": sorted(
                job_id
                for job_id, record in state["jobs"].items()
                if record["state"] != "shed_unavailable"
            ),
            "complete": all(
                record["state"] in TERMINAL_STATES for record in state["jobs"].values()
            ),
            "incomplete_campaign_reason": state["incomplete_campaign_reason"],
            "scientific_data": False,
            "production_eligible": False,
        }
