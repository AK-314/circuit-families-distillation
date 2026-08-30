"""Deterministic validation and compilation of Stage 12-P3 campaign DAGs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .records import CampaignManifest, LogicalJobSpec, Stage12P3ContractError


class DagValidationError(Stage12P3ContractError):
    """Raised when a campaign dependency graph is not closed and deterministic."""


@dataclass(frozen=True)
class CompiledCampaign:
    manifest: CampaignManifest
    topological_job_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if set(self.topological_job_ids) != {job.job_id for job in self.manifest.jobs}:
            raise DagValidationError("compiled topological identities mismatch manifest")

    @property
    def campaign_id(self) -> str:
        return self.manifest.campaign_id

    @property
    def jobs_by_id(self) -> dict[str, LogicalJobSpec]:
        return {job.job_id: job for job in self.manifest.jobs}

    def topological_jobs(self) -> tuple[LogicalJobSpec, ...]:
        by_id = self.jobs_by_id
        return tuple(by_id[job_id] for job_id in self.topological_job_ids)

    def ready_job_ids(self, succeeded_job_ids: set[str]) -> tuple[str, ...]:
        return tuple(
            job.job_id
            for job in self.topological_jobs()
            if job.job_id not in succeeded_job_ids
            and set(job.dependencies).issubset(succeeded_job_ids)
        )


def compile_campaign(manifest: CampaignManifest) -> CompiledCampaign:
    """Validate closure, outputs, resource references and acyclicity."""
    jobs = manifest.jobs
    job_ids = [job.job_id for job in jobs]
    duplicates = sorted(job_id for job_id, count in Counter(job_ids).items() if count > 1)
    if duplicates:
        raise DagValidationError(f"duplicate logical jobs: {duplicates!r}")

    by_id = {job.job_id: job for job in jobs}
    resource_refs = [item.reference for item in manifest.resource_classes]
    priority_refs = [item.reference for item in manifest.priority_classes]
    if len(set(resource_refs)) != len(resource_refs):
        raise DagValidationError("duplicate resource class references")
    if len(set(priority_refs)) != len(priority_refs):
        raise DagValidationError("duplicate priority class references")

    output_owners: dict[str, str] = {}
    for job in jobs:
        if job.job_id in job.dependencies:
            raise DagValidationError("self-dependency is forbidden")
        dangling = sorted(set(job.dependencies) - set(by_id))
        if dangling:
            raise DagValidationError(f"dangling dependencies for {job.job_id}: {dangling!r}")
        if job.resource_class_reference not in resource_refs:
            raise DagValidationError("job references unknown resource class")
        if job.priority_class_reference not in priority_refs:
            raise DagValidationError("job references unknown priority class")
        for artifact in job.output_contract.artifacts:
            owner = output_owners.setdefault(artifact.relative_path, job.job_id)
            if owner != job.job_id:
                raise DagValidationError(
                    f"duplicate campaign output {artifact.relative_path!r} owned by {owner}"
                )

    indegree = {job_id: len(job.dependencies) for job_id, job in by_id.items()}
    children = {job_id: [] for job_id in by_id}
    for job in jobs:
        for dependency in job.dependencies:
            children[dependency].append(job.job_id)
    ready = sorted(job_id for job_id, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        job_id = ready.pop(0)
        ordered.append(job_id)
        for child in sorted(children[job_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(ordered) != len(jobs):
        cyclic = sorted(job_id for job_id, degree in indegree.items() if degree > 0)
        raise DagValidationError(f"cycle detected: {cyclic!r}")
    return CompiledCampaign(manifest=manifest, topological_job_ids=tuple(ordered))
