from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.job_dag import (
    TechnicalJobRegistry,
    build_job_node,
)
from circuit_families.stage5bc.job_outputs import (
    atomic_write_job_file,
    bind_job_output_root,
    write_job_completion,
)
from circuit_families.stage5bc.job_status import (
    JOB_STATUSES,
    JobStatusError,
    decide_attempt_resume,
    inspect_registry_statuses,
    status_mapping,
    write_job_failure,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)
IDENTITY_SPEC_PATH = (
    ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
)


@pytest.fixture(scope="module")
def stage3() -> Stage3AvailabilityIndex:
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return Stage3AvailabilityIndex.from_registry(raw)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _technical_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()

    _git("init", "-q", cwd=repo)

    (repo / ".gitignore").write_text(
        "scratch/\n",
        encoding="utf-8",
    )
    (repo / "tracked.txt").write_text(
        "tracked\n",
        encoding="utf-8",
    )
    _git(
        "add",
        ".gitignore",
        "tracked.txt",
        cwd=repo,
    )

    scratch = repo / "scratch"
    scratch.mkdir()

    return repo, scratch


def _nodes(stage3: Stage3AvailabilityIndex):
    spec = json.loads(
        IDENTITY_SPEC_PATH.read_text(encoding="utf-8")
    )

    cache_id = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="hard_target",
        ),
        stage3,
    )
    student_id = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="hard_target",
            student_initialization=0,
        ),
        stage3,
    )
    complete_id = spec["synthetic_test_vectors"]["complete_a"]

    cache = build_job_node(
        stage3=stage3,
        node_type="teacher_cache",
        condition_id=cache_id,
    )
    training = build_job_node(
        stage3=stage3,
        node_type="training",
        condition_id=student_id,
        dependencies=(cache.job_id,),
    )
    completion = build_job_node(
        stage3=stage3,
        node_type="technical_completion",
        condition_id=student_id,
        dependencies=(training.job_id,),
    )
    eligibility = build_job_node(
        stage3=stage3,
        node_type="future_eligibility",
        condition_id=student_id,
        dependencies=(completion.job_id,),
    )
    discovery = build_job_node(
        stage3=stage3,
        node_type="discovery",
        condition_id=complete_id,
        dependencies=(eligibility.job_id,),
    )
    endpoint = build_job_node(
        stage3=stage3,
        node_type="endpoint",
        condition_id=complete_id,
        dependencies=(discovery.job_id,),
    )
    merge = build_job_node(
        stage3=stage3,
        node_type="merge",
        condition_id=complete_id,
        dependencies=(endpoint.job_id,),
    )
    analysis = build_job_node(
        stage3=stage3,
        node_type="analysis",
        condition_id=complete_id,
        dependencies=(merge.job_id,),
    )

    return (
        cache,
        training,
        completion,
        eligibility,
        discovery,
        endpoint,
        merge,
        analysis,
    )


def _registry(stage3: Stage3AvailabilityIndex):
    nodes = _nodes(stage3)
    return TechnicalJobRegistry(
        stage3=stage3,
        nodes=nodes,
    ), nodes


def _statuses(
    *,
    registry,
    repo: Path,
    scratch: Path,
):
    return inspect_registry_statuses(
        registry=registry,
        repository_root=repo,
        scratch_root=scratch,
    )


def _complete(
    *,
    repo: Path,
    scratch: Path,
    node,
    payload: bytes = b"technical\n",
):
    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=node,
    )
    artifact = atomic_write_job_file(
        root,
        "artifacts/result.bin",
        payload,
    )
    write_job_completion(
        root,
        artifacts=(artifact,),
    )
    return root, artifact


def test_exact_status_vocabulary() -> None:
    assert JOB_STATUSES == (
        "planned",
        "blocked",
        "running",
        "completed",
        "failed",
        "stale",
        "conflicting",
    )


def test_initial_dag_is_planned_then_blocked(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    reports = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )
    mapping = status_mapping(reports)

    assert mapping[nodes[0].job_id] == "planned"
    assert mapping[nodes[1].job_id] == "blocked"
    assert mapping[nodes[2].job_id] == "blocked"

    for node in nodes[3:]:
        assert mapping[node.job_id] == "blocked"


def test_allocated_executable_root_without_terminal_record_is_running(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=nodes[0],
    )

    mapping = status_mapping(
        _statuses(
            registry=registry,
            repo=repo,
            scratch=scratch,
        )
    )

    assert mapping[nodes[0].job_id] == "running"
    assert mapping[nodes[1].job_id] == "blocked"


def test_valid_completion_unlocks_next_job(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    _complete(
        repo=repo,
        scratch=scratch,
        node=nodes[0],
    )

    reports = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )
    mapping = status_mapping(reports)

    assert mapping[nodes[0].job_id] == "completed"
    assert mapping[nodes[1].job_id] == "planned"
    assert mapping[nodes[2].job_id] == "blocked"


def test_valid_failure_reports_failed_and_does_not_unlock_child(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=nodes[0],
    )
    write_job_failure(
        root,
        reason="technical fixture failure",
    )

    reports = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )
    mapping = status_mapping(reports)

    assert mapping[nodes[0].job_id] == "failed"
    assert mapping[nodes[1].job_id] == "blocked"


def test_missing_completed_artifact_reports_stale(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    root, artifact = _complete(
        repo=repo,
        scratch=scratch,
        node=nodes[0],
    )
    (root.path / artifact.relative_path).unlink()

    reports = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )

    cache_report = reports[0]

    assert cache_report.status == "stale"
    assert cache_report.reason == "completion_artifact_missing"


def test_mismatched_completed_artifact_hash_reports_stale(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    root, artifact = _complete(
        repo=repo,
        scratch=scratch,
        node=nodes[0],
    )

    (root.path / artifact.relative_path).write_bytes(
        b"tampered bytes\n"
    )

    cache_report = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )[0]

    assert cache_report.status == "stale"
    assert cache_report.reason == "completion_artifact_hash_mismatch"


def test_incomplete_atomic_temp_write_reports_stale(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=nodes[0],
    )

    temp = root.path / ".result.bin.interrupted.tmp"
    temp.write_bytes(b"partial")

    report = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )[0]

    assert report.status == "stale"
    assert report.reason == "incomplete_temporary_write_present"


def test_duplicate_completion_reports_conflicting(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    root, _ = _complete(
        repo=repo,
        scratch=scratch,
        node=nodes[0],
    )

    duplicate = root.path / "completion.duplicate.json"
    duplicate.write_bytes(
        (root.path / "completion.json").read_bytes()
    )

    report = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )[0]

    assert report.status == "conflicting"
    assert report.reason == "duplicate_completion_records_present"


def test_completion_plus_failure_reports_conflicting(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    root, _ = _complete(
        repo=repo,
        scratch=scratch,
        node=nodes[0],
    )

    failure_record = {
        "schema_version": "stage5bc-job-failure/v1",
        "scientific_data": False,
        "production_eligible": False,
        "failure_state": "failed",
        "job_id": root.job_id,
        "node_type": root.node_type,
        "condition_id": root.condition_id,
        "relative_identity": root.relative_identity,
        "reason": "contradictory terminal evidence",
    }

    (root.path / "failure.json").write_text(
        json.dumps(
            failure_record,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    report = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )[0]

    assert report.status == "conflicting"
    assert report.reason == (
        "completion_and_failure_records_both_present"
    )


def test_inert_identity_root_remains_blocked_if_empty(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=nodes[3],
    )

    report = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )[3]

    assert report.status == "blocked"
    assert report.output_root_exists is True


def test_inert_placeholder_with_runtime_file_is_stale(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    registry, nodes = _registry(stage3)

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=nodes[3],
    )

    (root.path / "forbidden.txt").write_text(
        "forbidden\n",
        encoding="utf-8",
    )

    report = _statuses(
        registry=registry,
        repo=repo,
        scratch=scratch,
    )[3]

    assert report.status == "stale"
    assert report.reason == (
        "inert_placeholder_has_runtime_output_evidence"
    )


def test_status_output_is_independent_of_registry_input_order(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    nodes = _nodes(stage3)

    first = TechnicalJobRegistry(
        stage3=stage3,
        nodes=nodes,
    )
    second = TechnicalJobRegistry(
        stage3=stage3,
        nodes=reversed(nodes),
    )

    first_reports = _statuses(
        registry=first,
        repo=repo,
        scratch=scratch,
    )
    second_reports = _statuses(
        registry=second,
        repo=repo,
        scratch=scratch,
    )

    assert first_reports == second_reports


def _attempt(
    stage3: Stage3AvailabilityIndex,
    *,
    initialization: int = 0,
    attempt: int = 0,
    retry: int = 0,
):
    return build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=initialization,
        attempt_index=attempt,
        retry_index=retry,
    )


def _report_for_status(
    *,
    node,
    status: str,
):
    from circuit_families.stage5bc.job_status import JobStatusReport

    return JobStatusReport(
        job_id=node.job_id,
        node_type=node.node_type,
        condition_id=node.condition_id,
        relative_identity="jobs/v1/training/synthetic",
        status=status,
        reason=f"technical_{status}",
        output_root_exists=status != "planned",
    )


def test_completed_attempt_is_skipped_not_duplicated(
    stage3: Stage3AvailabilityIndex,
) -> None:
    training = _nodes(stage3)[1]
    identity = _attempt(stage3)

    decision = decide_attempt_resume(
        node=training,
        status_report=_report_for_status(
            node=training,
            status="completed",
        ),
        requested_attempt_identity=identity,
    )

    assert decision.action == "skip_completed"


def test_running_attempt_requires_exact_checkpoint_identity(
    stage3: Stage3AvailabilityIndex,
) -> None:
    training = _nodes(stage3)[1]
    identity = _attempt(stage3)

    decision = decide_attempt_resume(
        node=training,
        status_report=_report_for_status(
            node=training,
            status="running",
        ),
        requested_attempt_identity=identity,
        checkpoint_attempt_identity=identity,
    )

    assert decision.action == "resume_existing"


@pytest.mark.parametrize(
    "checkpoint_identity",
    [
        "different_condition",
        "different_attempt",
        "different_retry",
    ],
)
def test_resume_state_cannot_transfer_to_another_identity(
    stage3: Stage3AvailabilityIndex,
    checkpoint_identity: str,
) -> None:
    training = _nodes(stage3)[1]
    requested = _attempt(stage3)

    if checkpoint_identity == "different_condition":
        checkpoint = _attempt(
            stage3,
            initialization=1,
        )
    elif checkpoint_identity == "different_attempt":
        checkpoint = _attempt(
            stage3,
            attempt=1,
        )
    else:
        checkpoint = _attempt(
            stage3,
            retry=1,
        )

    with pytest.raises(
        JobStatusError,
        match="state transfer",
    ):
        decide_attempt_resume(
            node=training,
            status_report=_report_for_status(
                node=training,
                status="running",
            ),
            requested_attempt_identity=requested,
            checkpoint_attempt_identity=checkpoint,
        )


def test_failed_attempt_cannot_be_reexecuted_as_resume(
    stage3: Stage3AvailabilityIndex,
) -> None:
    training = _nodes(stage3)[1]
    identity = _attempt(stage3)

    decision = decide_attempt_resume(
        node=training,
        status_report=_report_for_status(
            node=training,
            status="failed",
        ),
        requested_attempt_identity=identity,
    )

    assert decision.action == "reject_failed_attempt"


def test_planned_attempt_rejects_foreign_preexisting_checkpoint(
    stage3: Stage3AvailabilityIndex,
) -> None:
    training = _nodes(stage3)[1]
    identity = _attempt(stage3)

    with pytest.raises(
        JobStatusError,
        match="planned job cannot consume",
    ):
        decide_attempt_resume(
            node=training,
            status_report=_report_for_status(
                node=training,
                status="planned",
            ),
            requested_attempt_identity=identity,
            checkpoint_attempt_identity=identity,
        )
