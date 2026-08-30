from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_stage12p3_state import make_job, ref

from circuit_families.stage12p3 import (
    CampaignController,
    CampaignManifest,
    ConcurrencyProfile,
    GenericJobArrayAdapter,
    LocalFixtureAdapter,
    PriorityClass,
    ResourceClass,
    RetryPolicy,
    SchedulerObservation,
    Stage12P3ContractError,
    WorkerCapabilities,
    compile_campaign,
)
from circuit_families.stage12p3.cli import run_validate_only

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def controller(tmp_path: Path):
    first = make_job("first")
    second = make_job("second")
    campaign = compile_campaign(
        CampaignManifest(
            manifest_reference=ref("adapter-manifest"),
            jobs=(first, second),
            resource_classes=(ResourceClass("resource/small/v1", 1, None, 1024, 1024, 10),),
            priority_classes=(PriorityClass("priority/primary/v1", 0),),
        )
    )
    active = CampaignController(
        campaign,
        tmp_path,
        RetryPolicy("retry/technical/v1", 2, ("worker_error",), 10),
    )
    return active, first, second


def test_local_fixture_adapter_executes_only_injected_worker_outputs(tmp_path: Path) -> None:
    active, first, second = controller(tmp_path)

    def worker(job, _seed):
        return {job.output_contract.artifacts[0].relative_path: b"fixture\n"}

    adapter = LocalFixtureAdapter({"first": worker, "second": worker})
    capabilities = WorkerCapabilities("resource/small/v1", 1, (), 1024, 1024)
    concurrency = ConcurrencyProfile("concurrency/technical/v1", (("resource/small/v1", 1),))
    observed = []
    for now in range(3):
        observed.append(
            adapter.launch_one(
                active,
                worker_id="worker",
                capabilities=capabilities,
                concurrency=concurrency,
                now=now,
            ).result
        )
    assert observed == ["succeeded", "succeeded", "no_ready_satisfiable_job"]
    assert active.status()["counts"] == {"succeeded": 2}
    assert {first.job_id, second.job_id} == set(active.store.read()["jobs"])


def test_generic_array_mapping_and_observations_are_strict(tmp_path: Path) -> None:
    active, first, second = controller(tmp_path)
    adapter = GenericJobArrayAdapter()
    plan = adapter.render(active, (second.job_id, first.job_id))
    assert [entry.array_index for entry in plan.entries] == [0, 1]
    assert [entry.logical_job_id for entry in plan.entries] == sorted((first.job_id, second.job_id))
    assert "GENERIC_ARRAY_INDEX" in plan.script
    submission = adapter.submission(
        plan, backend_job_id="backend-1", array_index=0, controller=active
    )
    observation = SchedulerObservation(
        submission.logical_job_id, submission.backend_job_id, "finished", 1
    )
    result = adapter.ingest(submission, observation)
    assert result["sealed_job_state_changed"] is False
    assert active.status()["counts"] == {"ready": 2}
    cancellation = adapter.cancellation(submission, cancellation_request_id="cancel-request-1")
    assert cancellation.logical_job_id == submission.logical_job_id
    assert cancellation.backend_job_id == submission.backend_job_id

    with pytest.raises(Stage12P3ContractError, match="stale"):
        adapter.ingest(submission, observation)
    with pytest.raises(Stage12P3ContractError, match="logical job mismatch"):
        adapter.ingest(
            submission,
            replace(observation, logical_job_id=second.job_id),
        )
    with pytest.raises(Stage12P3ContractError, match="out of range"):
        adapter.submission(plan, backend_job_id="backend-1", array_index=2, controller=active)


def test_validate_only_campaign_forces_required_operational_paths(tmp_path: Path) -> None:
    report = run_validate_only(tmp_path)
    assert report["scientific_data"] is False
    assert report["production_eligible"] is False
    assert report["partial_output_did_not_unblock"] is True
    assert report["sealed_output_did_unblock"] is True
    assert report["forced_stale_claim_count"] == 1
    assert report["scheduler_finished_changed_sealed_state"] is False
    assert report["status"]["counts"] == {
        "blocked": 1,
        "shed_unavailable": 1,
        "succeeded": 6,
        "terminal_failure": 1,
    }
    assert report["status"]["complete"] is False
    assert all(report[f"rd_{number}_resolved"] is False for number in ("012", "013", "014"))


def _run_cli(cwd: Path, output_root: Path, hash_seed: str) -> dict:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONHASHSEED"] = hash_seed
    completed = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts/validate_stage12p3.py"),
            "--validate-only",
            "--output-root",
            str(output_root),
        ],
        cwd=cwd,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def test_cli_is_cwd_portable_and_hash_seed_invariant(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    first = _run_cli(ROOT, tmp_path / "root-run", "1")
    second = _run_cli(unrelated, tmp_path / "unrelated-run", "987654")
    assert first["report_sha256"] == second["report_sha256"]
    assert first == second
