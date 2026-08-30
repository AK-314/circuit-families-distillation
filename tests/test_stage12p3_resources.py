from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_stage12p3_state import make_job, ref

from circuit_families.stage12p3 import (
    CampaignController,
    CampaignManifest,
    ConcurrencyProfile,
    PriorityClass,
    ResourceClass,
    RetryPolicy,
    SheddingPolicy,
    Stage12P3ContractError,
    TierRule,
    WorkerCapabilities,
    compile_campaign,
)


def campaign_with(jobs, priorities, resources=None):
    return compile_campaign(
        CampaignManifest(
            manifest_reference=ref("resources-manifest"),
            jobs=tuple(jobs),
            resource_classes=tuple(resources)
            if resources is not None
            else (ResourceClass("resource/small/v1", 1, None, 1024, 1024, 10),),
            priority_classes=tuple(priorities),
        )
    )


def retry() -> RetryPolicy:
    return RetryPolicy("retry/technical/v1", 2, ("worker_error",), 10)


def capabilities(reference: str = "resource/small/v1") -> WorkerCapabilities:
    return WorkerCapabilities(reference, 2, (), 2048, 2048)


def concurrency(limit: int = 1) -> ConcurrencyProfile:
    return ConcurrencyProfile("concurrency/technical/v1", (("resource/small/v1", limit),))


def test_dispatch_uses_injected_priority_then_stable_job_identity(tmp_path: Path) -> None:
    high = replace(make_job("high"), priority_class_reference="priority/high/v1")
    low = replace(make_job("low"), priority_class_reference="priority/low/v1")
    campaign = campaign_with(
        (low, high),
        (PriorityClass("priority/high/v1", 0), PriorityClass("priority/low/v1", 10)),
    )
    active = CampaignController(campaign, tmp_path, retry())
    claim = active.claim_next(
        worker_id="worker", capabilities=capabilities(), concurrency=concurrency(), now=0
    )
    assert claim is not None and claim["job_id"] == high.job_id

    tied_a = replace(make_job("tied-a"), priority_class_reference="priority/tied/v1")
    tied_b = replace(make_job("tied-b"), priority_class_reference="priority/tied/v1")
    tied_campaign = campaign_with((tied_b, tied_a), (PriorityClass("priority/tied/v1", 0),))
    tied = CampaignController(tied_campaign, tmp_path / "tied", retry())
    tied_claim = tied.claim_next(
        worker_id="worker", capabilities=capabilities(), concurrency=concurrency(), now=0
    )
    assert tied_claim is not None
    assert tied_claim["job_id"] == min(tied_a.job_id, tied_b.job_id)


def test_unlike_resource_classes_are_never_relabelled_or_equated(tmp_path: Path) -> None:
    gpu_resource = ResourceClass("resource/gpu/v1", 1, "cuda/v1", 1024, 1024, 10)
    gpu_job = replace(make_job("gpu"), resource_class_reference="resource/gpu/v1")
    campaign = campaign_with(
        (gpu_job,), (PriorityClass("priority/primary/v1", 0),), (gpu_resource,)
    )
    active = CampaignController(campaign, tmp_path, retry())
    cpu_with_large_numbers = WorkerCapabilities("resource/cpu/v1", 100, ("cuda/v1",), 9999, 9999)
    profile = ConcurrencyProfile("concurrency/technical/v1", (("resource/gpu/v1", 1),))
    assert (
        active.claim_next(
            worker_id="worker",
            capabilities=cpu_with_large_numbers,
            concurrency=profile,
            now=0,
        )
        is None
    )


def test_atomic_concurrency_limit_prevents_overcommit(tmp_path: Path) -> None:
    first = make_job("first")
    second = make_job("second")
    campaign = campaign_with((first, second), (PriorityClass("priority/primary/v1", 0),))
    active = CampaignController(campaign, tmp_path, retry())
    one = active.claim_next(
        worker_id="worker-1", capabilities=capabilities(), concurrency=concurrency(1), now=0
    )
    two = active.claim_next(
        worker_id="worker-2", capabilities=capabilities(), concurrency=concurrency(1), now=0
    )
    assert one is not None
    assert two is None
    assert active.status()["counts"] == {"claimed": 1, "ready": 1}


def test_shedding_is_reverse_ranked_and_never_removes_protected_jobs(tmp_path: Path) -> None:
    protected = replace(make_job("protected"), protected_tier="tier1")
    optional_early = replace(make_job("optional-early"), protected_tier="tier3")
    optional_late = replace(make_job("optional-late"), protected_tier="tier2-unprotected")
    campaign = campaign_with(
        (protected, optional_early, optional_late),
        (PriorityClass("priority/primary/v1", 0),),
    )
    active = CampaignController(campaign, tmp_path, retry())
    policy = SheddingPolicy(
        "shedding/technical/v1",
        (
            TierRule("tier1", protected=True, optional=False, shedding_rank=None),
            TierRule("tier2-unprotected", protected=False, optional=True, shedding_rank=10),
            TierRule("tier3", protected=False, optional=True, shedding_rank=20),
        ),
        "technical-capacity-shortfall/v1",
    )
    first = active.apply_shedding(maximum_retained_jobs=2, policy=policy)
    assert first["shed_job_ids"] == [optional_early.job_id]
    second = active.apply_shedding(maximum_retained_jobs=1, policy=policy)
    assert second["shed_job_ids"] == [optional_late.job_id]
    state = active.store.read()["jobs"]
    assert state[protected.job_id]["state"] == "ready"
    assert state[optional_early.job_id]["state"] == "shed_unavailable"
    assert state[optional_late.job_id]["state"] == "shed_unavailable"
    assert len(state) == 3
    status = active.status()
    assert status["counts"] == {"ready": 1, "shed_unavailable": 2}
    assert status["incomplete_campaign_reason"] == "technical-capacity-shortfall/v1"


def test_scarcity_cannot_force_protected_shedding(tmp_path: Path) -> None:
    protected = replace(make_job("protected"), protected_tier="tier1")
    campaign = campaign_with((protected,), (PriorityClass("priority/primary/v1", 0),))
    active = CampaignController(campaign, tmp_path, retry())
    policy = SheddingPolicy(
        "shedding/technical/v1",
        (TierRule("tier1", protected=True, optional=False, shedding_rank=None),),
        "technical-capacity-shortfall/v1",
    )
    result = active.apply_shedding(maximum_retained_jobs=0, policy=policy)
    assert result["shed_job_ids"] == []
    assert result["unsatisfied_scarcity_count"] == 1
    assert active.store.read()["jobs"][protected.job_id]["state"] == "ready"


def test_invalid_protected_shedding_and_metric_driven_priority_are_rejected() -> None:
    with pytest.raises(Stage12P3ContractError, match="protected tiers"):
        TierRule("tier1", protected=True, optional=False, shedding_rank=1)
    with pytest.raises(TypeError):
        PriorityClass(
            reference="priority/invalid/v1",
            dispatch_rank=0,
            output_metric=0.9,  # type: ignore[call-arg]
        )
