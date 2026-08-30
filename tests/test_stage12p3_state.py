from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from circuit_families.stage12p3 import (
    CampaignController,
    CampaignManifest,
    CampaignStateError,
    ExpectedArtifact,
    HashBoundReference,
    LogicalJobSpec,
    OutputContract,
    PriorityClass,
    ResourceClass,
    RetryPolicy,
    compile_campaign,
)

SHA = "a" * 64


def ref(name: str) -> HashBoundReference:
    return HashBoundReference(f"synthetic://{name}", SHA, f"{name}/v1")


def make_job(family: str, dependencies: tuple[str, ...] = ()) -> LogicalJobSpec:
    return LogicalJobSpec(
        family=family,
        producer_interface_version=f"{family}/v1",
        dependencies=dependencies,
        expected_inputs=(ref(f"input-{family}"),),
        payload_reference=ref(f"payload-{family}"),
        config_reference=ref(f"config-{family}"),
        output_contract=OutputContract(
            f"manifests/{family}.json",
            f"{family}-output/v1",
            (ExpectedArtifact(f"artifacts/{family}.txt", "text/plain"),),
        ),
        resource_class_reference="resource/small/v1",
        priority_class_reference="priority/primary/v1",
        protected_tier="tier1-technical",
        retry_seed_namespace_reference="stage12p3-technical-seed/v1",
    )


def make_campaign(*jobs: LogicalJobSpec):
    manifest = CampaignManifest(
        manifest_reference=ref("manifest"),
        jobs=jobs,
        resource_classes=(ResourceClass("resource/small/v1", 1, None, 1024, 1024, 10),),
        priority_classes=(PriorityClass("priority/primary/v1", 0),),
    )
    return compile_campaign(manifest)


def policy(maximum_attempts: int = 2) -> RetryPolicy:
    return RetryPolicy(
        reference="retry/technical/v1",
        maximum_attempts=maximum_attempts,
        retryable_categories=("interruption", "stale_claim", "worker_error"),
        lease_seconds=10,
    )


def controller(
    tmp_path: Path, maximum_attempts: int = 2
) -> tuple[CampaignController, LogicalJobSpec]:
    logical = make_job("producer")
    return CampaignController(make_campaign(logical), tmp_path, policy(maximum_attempts)), logical


def seal_active(
    controller: CampaignController, job: LogicalJobSpec, claim: dict, now: int = 2
) -> str:
    controller.write_artifact(
        job.job_id, claim["attempt_index"], f"artifacts/{job.family}.txt", b"ok\n"
    )
    controller.publish_output_manifest(
        job.job_id, worker_id=claim["worker_id"], claim_token=claim["claim_token"]
    )
    return controller.complete(
        job.job_id,
        worker_id=claim["worker_id"],
        claim_token=claim["claim_token"],
        now=now,
    )


def test_competing_claimants_have_exactly_one_winner(tmp_path: Path) -> None:
    active, logical = controller(tmp_path)
    successes: list[dict] = []
    failures: list[str] = []
    barrier = threading.Barrier(2)

    def compete(worker: str) -> None:
        barrier.wait()
        try:
            successes.append(active.claim(logical.job_id, worker_id=worker, now=0))
        except CampaignStateError as exc:
            failures.append(str(exc))

    threads = [threading.Thread(target=compete, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(successes) == 1
    assert len(failures) == 1
    assert active.status()["counts"] == {"claimed": 1}


def test_zero_exit_without_sealed_manifest_cannot_succeed(tmp_path: Path) -> None:
    active, logical = controller(tmp_path)
    claim = active.claim(logical.job_id, worker_id="worker", now=0)
    with pytest.raises(CampaignStateError, match="manifest is missing"):
        active.complete(logical.job_id, worker_id="worker", claim_token=claim["claim_token"], now=1)
    assert active.status()["counts"] == {"claimed": 1}


def test_sealed_completion_is_idempotent_and_unblocks_dependant(tmp_path: Path) -> None:
    parent = make_job("producer")
    child = make_job("consumer", (parent.job_id,))
    active = CampaignController(make_campaign(parent, child), tmp_path, policy())
    assert active.status()["counts"] == {"blocked": 1, "ready": 1}
    claim = active.claim(parent.job_id, worker_id="worker", now=0)
    digest = seal_active(active, parent, claim)
    repeated = active.complete(
        parent.job_id,
        worker_id="worker",
        claim_token=claim["claim_token"],
        now=3,
    )
    assert repeated == digest
    with pytest.raises(CampaignStateError, match="token mismatch"):
        active.complete(
            parent.job_id,
            worker_id="worker",
            claim_token="wrong",
            now=3,
        )
    assert active.status()["counts"] == {"ready": 1, "succeeded": 1}


def test_partial_tampered_and_conflicting_outputs_are_rejected(tmp_path: Path) -> None:
    active, logical = controller(tmp_path)
    claim = active.claim(logical.job_id, worker_id="worker", now=0)
    with pytest.raises(CampaignStateError, match="expected output is missing"):
        active.publish_output_manifest(
            logical.job_id, worker_id="worker", claim_token=claim["claim_token"]
        )

    active.write_artifact(logical.job_id, 0, "artifacts/producer.txt", b"first")
    with pytest.raises(CampaignStateError, match="conflicting artifact"):
        active.write_artifact(logical.job_id, 0, "artifacts/producer.txt", b"second")
    active.publish_output_manifest(
        logical.job_id, worker_id="worker", claim_token=claim["claim_token"]
    )
    artifact = active.attempt_root(logical.job_id, 0) / "artifacts/producer.txt"
    artifact.write_bytes(b"tampered")
    with pytest.raises(CampaignStateError, match="hash or size mismatch"):
        active.complete(logical.job_id, worker_id="worker", claim_token=claim["claim_token"], now=1)


def test_wrong_token_live_and_expired_lease_paths(tmp_path: Path) -> None:
    active, logical = controller(tmp_path)
    claim = active.claim(logical.job_id, worker_id="worker", now=0)
    with pytest.raises(CampaignStateError, match="token mismatch"):
        active.heartbeat(logical.job_id, worker_id="worker", claim_token="wrong", now=1)
    active.heartbeat(logical.job_id, worker_id="worker", claim_token=claim["claim_token"], now=5)
    assert active.reconcile(now=14) == {"recovered_successes": 0, "stale_failures": 0}
    assert active.reconcile(now=16) == {"recovered_successes": 0, "stale_failures": 1}
    assert active.status()["counts"] == {"retryable_failure": 1}


def test_crash_after_manifest_publication_recovers_as_success(tmp_path: Path) -> None:
    active, logical = controller(tmp_path)
    claim = active.claim(logical.job_id, worker_id="worker", now=0)
    active.write_artifact(logical.job_id, 0, "artifacts/producer.txt", b"ok")
    active.publish_output_manifest(
        logical.job_id, worker_id="worker", claim_token=claim["claim_token"]
    )
    restarted = CampaignController(make_campaign(logical), tmp_path, policy())
    assert restarted.reconcile(now=1) == {"recovered_successes": 1, "stale_failures": 0}
    assert restarted.status()["counts"] == {"succeeded": 1}


def test_retry_boundary_and_attempt_history_are_retained(tmp_path: Path) -> None:
    active, logical = controller(tmp_path, maximum_attempts=1)
    claim = active.claim(logical.job_id, worker_id="worker", now=0)
    assert (
        active.fail(
            logical.job_id,
            worker_id="worker",
            claim_token=claim["claim_token"],
            category="interruption",
            detail="forced",
            now=1,
        )
        == "terminal_failure"
    )
    state = active.store.read()["jobs"][logical.job_id]
    assert state["attempts"][0]["failure_category"] == "interruption"
    assert state["state"] == "terminal_failure"


def test_state_hash_traversal_and_symlink_escape_are_rejected(tmp_path: Path) -> None:
    active, logical = controller(tmp_path)
    active.claim(logical.job_id, worker_id="worker", now=0)
    with pytest.raises(CampaignStateError, match="escape"):
        active.write_artifact(logical.job_id, 0, "../escape", b"bad")

    attempt_root = active.attempt_root(logical.job_id, 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    (attempt_root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CampaignStateError, match="symlink"):
        active.write_artifact(logical.job_id, 0, "linked/escape", b"bad")

    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="ascii"))
    state["jobs"][logical.job_id]["state"] = "succeeded"
    state_path.write_text(json.dumps(state), encoding="ascii")
    with pytest.raises(CampaignStateError, match="integrity hash mismatch"):
        active.store.read()


def test_resume_rejects_changed_campaign_inputs(tmp_path: Path) -> None:
    active, logical = controller(tmp_path)
    claim = active.claim(logical.job_id, worker_id="worker", now=0)
    seal_active(active, logical, claim)
    changed = replace(
        logical, config_reference=HashBoundReference("synthetic://changed", "b" * 64, "changed/v1")
    )
    with pytest.raises(CampaignStateError, match="campaign hash mismatch"):
        CampaignController(make_campaign(changed), tmp_path, policy())
