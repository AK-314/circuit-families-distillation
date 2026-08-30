from __future__ import annotations

from pathlib import Path

import pytest
from test_stage12p3_state import make_campaign, make_job, seal_active

from circuit_families.stage12p3 import (
    FAILURE_CATEGORIES,
    CampaignController,
    CampaignStateError,
    RetryPolicy,
)


def retry_policy(
    maximum_attempts: int = 2,
    categories: tuple[str, ...] | None = None,
) -> RetryPolicy:
    return RetryPolicy(
        reference="retry/all-technical/v1",
        maximum_attempts=maximum_attempts,
        retryable_categories=categories or tuple(sorted(FAILURE_CATEGORIES)),
        lease_seconds=10,
    )


@pytest.mark.parametrize("category", sorted(FAILURE_CATEGORIES))
def test_every_failure_category_is_explicit_and_terminal_at_zero_retries(
    tmp_path: Path,
    category: str,
) -> None:
    logical = make_job("producer")
    active = CampaignController(make_campaign(logical), tmp_path, retry_policy(1))
    claim = active.claim(logical.job_id, worker_id="worker", now=0)
    result = active.fail(
        logical.job_id,
        worker_id="worker",
        claim_token=claim["claim_token"],
        category=category,
        detail=f"forced {category}",
        now=1,
    )
    assert result == "terminal_failure"
    record = active.store.read()["jobs"][logical.job_id]
    assert record["state"] == "terminal_failure"
    assert record["attempts"][0]["failure_category"] == category


def test_retry_seed_changes_but_logical_identity_and_prior_attempt_remain(
    tmp_path: Path,
) -> None:
    logical = make_job("producer")
    active = CampaignController(make_campaign(logical), tmp_path, retry_policy())
    first = active.claim(logical.job_id, worker_id="worker-a", now=0)
    assert (
        active.fail(
            logical.job_id,
            worker_id="worker-a",
            claim_token=first["claim_token"],
            category="worker_error",
            detail="forced retry",
            now=1,
        )
        == "retryable_failure"
    )
    active.retry(logical.job_id)
    second = active.claim(logical.job_id, worker_id="worker-b", now=2)
    assert second["attempt_index"] == 1
    assert second["retry_index"] == 1
    assert second["seed_evidence"]["sha256"] != first["seed_evidence"]["sha256"]
    assert logical.job_id == active.campaign.topological_job_ids[0]
    attempts = active.store.read()["jobs"][logical.job_id]["attempts"]
    assert [attempt["state"] for attempt in attempts] == ["retryable_failure", "claimed"]


def test_nonretryable_failure_is_terminal_even_with_attempt_capacity(
    tmp_path: Path,
) -> None:
    logical = make_job("producer")
    policy = retry_policy(3, ("interruption",))
    active = CampaignController(make_campaign(logical), tmp_path, policy)
    claim = active.claim(logical.job_id, worker_id="worker", now=0)
    assert (
        active.fail(
            logical.job_id,
            worker_id="worker",
            claim_token=claim["claim_token"],
            category="validation_failure",
            detail="invalid sealed evidence",
            now=1,
        )
        == "terminal_failure"
    )


def test_downstream_remains_blocked_and_failed_dependency_is_retained(
    tmp_path: Path,
) -> None:
    parent = make_job("producer")
    child = make_job("consumer", (parent.job_id,))
    active = CampaignController(make_campaign(parent, child), tmp_path, retry_policy(1))
    claim = active.claim(parent.job_id, worker_id="worker", now=0)
    active.fail(
        parent.job_id,
        worker_id="worker",
        claim_token=claim["claim_token"],
        category="unavailable_input",
        detail="fixture input unavailable",
        now=1,
    )
    state = active.store.read()["jobs"]
    assert state[parent.job_id]["state"] == "terminal_failure"
    assert state[child.job_id]["state"] == "blocked"
    assert state[child.job_id]["blocked_reason"] == "dependency_failure"
    assert state[parent.job_id]["attempts"][0]["failure_category"] == "unavailable_input"


def test_restart_preserves_live_claim_then_reconciles_expiry(tmp_path: Path) -> None:
    logical = make_job("producer")
    campaign = make_campaign(logical)
    original = CampaignController(campaign, tmp_path, retry_policy())
    claim = original.claim(logical.job_id, worker_id="worker", now=0)
    restarted = CampaignController(campaign, tmp_path, retry_policy())
    assert restarted.status()["counts"] == {"claimed": 1}
    assert restarted.reconcile(now=10)["stale_failures"] == 0
    assert restarted.reconcile(now=11)["stale_failures"] == 1
    record = restarted.store.read()["jobs"][logical.job_id]
    assert record["attempts"][0]["claim_token"] == claim["claim_token"]


def test_restart_skips_success_and_retains_exact_manifest_hash(tmp_path: Path) -> None:
    logical = make_job("producer")
    campaign = make_campaign(logical)
    original = CampaignController(campaign, tmp_path, retry_policy())
    claim = original.claim(logical.job_id, worker_id="worker", now=0)
    digest = seal_active(original, logical, claim)
    restarted = CampaignController(campaign, tmp_path, retry_policy())
    assert restarted.status()["counts"] == {"succeeded": 1}
    assert restarted.store.read()["jobs"][logical.job_id]["sealed_output_manifest_sha256"] == digest
    with pytest.raises(CampaignStateError, match="not ready"):
        restarted.claim(logical.job_id, worker_id="other", now=5)


def test_uninterrupted_and_interrupted_resumed_end_with_complete_inventory(
    tmp_path: Path,
) -> None:
    logical = make_job("producer")
    campaign = make_campaign(logical)
    uninterrupted = CampaignController(campaign, tmp_path / "straight", retry_policy())
    straight_claim = uninterrupted.claim(logical.job_id, worker_id="worker", now=0)
    seal_active(uninterrupted, logical, straight_claim)

    resumed = CampaignController(campaign, tmp_path / "resumed", retry_policy())
    interrupted_claim = resumed.claim(logical.job_id, worker_id="worker", now=0)
    resumed.fail(
        logical.job_id,
        worker_id="worker",
        claim_token=interrupted_claim["claim_token"],
        category="interruption",
        detail="forced controller stop",
        now=1,
    )
    resumed = CampaignController(campaign, tmp_path / "resumed", retry_policy())
    resumed.retry(logical.job_id)
    resumed_claim = resumed.claim(logical.job_id, worker_id="worker", now=2)
    seal_active(resumed, logical, resumed_claim, now=3)

    assert uninterrupted.status()["complete"] is True
    assert resumed.status()["complete"] is True
    assert uninterrupted.status()["counts"] == resumed.status()["counts"] == {"succeeded": 1}
    resumed_attempts = resumed.store.read()["jobs"][logical.job_id]["attempts"]
    assert [attempt["state"] for attempt in resumed_attempts] == [
        "retryable_failure",
        "succeeded",
    ]
