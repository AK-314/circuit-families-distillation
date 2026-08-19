from __future__ import annotations

import copy
import hashlib
import json
import runpy
import subprocess
import time
from pathlib import Path

import pytest
import torch

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.attempt_records import (
    attempt_record_sha256,
    canonical_attempt_record_bytes,
    emit_technical_attempt_record,
    outcome_from_training_result,
)
from circuit_families.stage5bc.job_dag import (
    TechnicalJobRegistry,
    build_job_node,
)
from circuit_families.stage5bc.job_outputs import (
    JobArtifactEvidence,
    atomic_write_job_file,
    bind_job_output_root,
    write_job_completion,
)
from circuit_families.stage5bc.job_status import (
    decide_attempt_resume,
    inspect_registry_statuses,
)
from circuit_families.stage5bc.serial_merge import (
    canonical_registry_bytes,
    merge_status_evidence,
    registry_sha256,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
)
from circuit_families.stage5bc.student_trainer import (
    HardTargetAdapter,
    SoftTargetAdapter,
    TechnicalInterruption,
    TechnicalLoopSnapshot,
    TrainerLifecycle,
)
from circuit_families.stage5bc.target_cache import (
    TargetCacheBatch,
    build_target_cache,
    load_target_cache,
)
from circuit_families.stage5bc.technical_checkpoint import (
    load_technical_resume_checkpoint,
    save_technical_resume_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)
CACHE_FIXTURE_PATH = (
    ROOT / "tests/fixtures/stage5bc/technical_cache_manifest_v1.json"
)

TRAINING_HELPERS = runpy.run_path(
    str(ROOT / "tests/test_stage5bc_training_loop.py")
)


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


def _stage3() -> Stage3AvailabilityIndex:
    return Stage3AvailabilityIndex.from_registry(
        json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    )


def _condition_ids(
    stage3: Stage3AvailabilityIndex,
    *,
    condition: str,
) -> tuple[str, str]:
    cache_condition = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition=condition,
        ),
        stage3,
    )
    student_condition = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition=condition,
            student_initialization=0,
        ),
        stage3,
    )

    return cache_condition, student_condition


def _branch_nodes(
    stage3: Stage3AvailabilityIndex,
    *,
    condition: str,
):
    cache_condition, student_condition = _condition_ids(
        stage3,
        condition=condition,
    )

    cache = build_job_node(
        stage3=stage3,
        node_type="teacher_cache",
        condition_id=cache_condition,
    )
    training = build_job_node(
        stage3=stage3,
        node_type="training",
        condition_id=student_condition,
        dependencies=(cache.job_id,),
    )
    completion = build_job_node(
        stage3=stage3,
        node_type="technical_completion",
        condition_id=student_condition,
        dependencies=(training.job_id,),
    )

    return cache, training, completion


def _manifest_template() -> dict:
    return json.loads(
        CACHE_FIXTURE_PATH.read_text(encoding="utf-8")
    )


def _manifest_id(
    template: dict,
    *,
    condition: str,
) -> str:
    base = template.get(
        "manifest_id",
        "stage5bc-technical-cache/v1",
    )

    if not isinstance(base, str) or not base:
        raise AssertionError("fixture manifest_id must be non-empty")

    if base.endswith("/v1"):
        return base[:-3] + f"-part-r-{condition}/v1"

    return base + f"-part-r-{condition}"


def _ordering_ref(template: dict) -> str:
    direct = template.get("ordering_ref")

    if isinstance(direct, str) and direct:
        return direct

    ordering = template.get("example_ordering")

    if isinstance(ordering, dict):
        nested = ordering.get("ordering_ref")

        if isinstance(nested, str) and nested:
            return nested

    return "stage5bc-part-r-order/v1"


def _technical_training_components():
    settings = TRAINING_HELPERS["_settings"]()
    inputs = TRAINING_HELPERS["_inputs"]().detach().clone()

    model_constructor = TRAINING_HELPERS["_model_constructor"]
    optimizer_factory = TRAINING_HELPERS["_optimizer_factory"]
    hard_loss = TRAINING_HELPERS["_hard_loss"]
    soft_loss = TRAINING_HELPERS["_soft_loss"]
    stop_rule = TRAINING_HELPERS["_stop_rule"]

    probe = model_constructor(
        seed=0,
        device=torch.device("cpu"),
        settings=settings.model,
    )

    with torch.no_grad():
        probe_logits = probe(inputs)

    assert isinstance(probe_logits, torch.Tensor)
    assert probe_logits.ndim == 2
    assert probe_logits.shape[0] == inputs.shape[0]
    assert probe_logits.shape[1] >= 2

    return {
        "settings": settings,
        "inputs": inputs,
        "model_constructor": model_constructor,
        "optimizer_factory": optimizer_factory,
        "hard_loss": hard_loss,
        "soft_loss": soft_loss,
        "stop_rule": stop_rule,
        "class_count": int(probe_logits.shape[1]),
    }


def _teacher_logits(
    *,
    example_count: int,
    class_count: int,
) -> torch.Tensor:
    values = torch.arange(
        example_count * class_count,
        dtype=torch.float32,
    ).reshape(example_count, class_count)

    return (
        values / 7.0
        + torch.linspace(
            -0.3,
            0.3,
            steps=class_count,
            dtype=torch.float32,
        )
    )


def _file_evidence(
    *,
    output_root,
    path: Path,
) -> JobArtifactEvidence:
    raw = path.read_bytes()

    return JobArtifactEvidence(
        relative_path=path.relative_to(
            output_root.path
        ).as_posix(),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def _build_loaded_cache(
    *,
    output_root,
    condition_id: str,
    condition: str,
    input_count: int,
    class_count: int,
):
    template = _manifest_template()

    teacher_reference = copy.deepcopy(
        template["teacher_reference"]
    )
    teacher_reference["condition_id"] = condition_id

    provenance_hashes = copy.deepcopy(
        template["provenance_hashes"]
    )

    input_ids = tuple(
        f"part-r-input-{index}"
        for index in range(input_count)
    )

    logits = _teacher_logits(
        example_count=input_count,
        class_count=class_count,
    )
    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    built = build_target_cache(
        output_root=output_root.path,
        manifest_relative_path="cache/manifest.json",
        payload_relative_path="cache/payload.bin",
        completion_relative_path=(
            "cache/target-cache-completion.json"
        ),
        manifest_id=_manifest_id(
            template,
            condition=condition,
        ),
        ordering_ref=_ordering_ref(template),
        expected_example_count=input_count,
        expected_class_count=class_count,
        teacher_reference=teacher_reference,
        provenance_hashes=provenance_hashes,
        batches=(
            TargetCacheBatch(
                input_ids=input_ids,
                raw_logits=logits,
                probabilities=probabilities,
            ),
        ),
        technical_fixture=True,
        stage4_record_serializable=False,
        expected_input_ids=input_ids,
    )

    loaded = load_target_cache(
        output_root=output_root.path,
        manifest_relative_path="cache/manifest.json",
        expected_input_ids=input_ids,
        expected_teacher_reference=teacher_reference,
        expected_provenance_hashes=provenance_hashes,
        expected_stage4_cache_kind=(
            "teacher_argmax"
            if condition == "hard_target"
            else "teacher_logits"
        ),
    )

    cache_artifacts = (
        _file_evidence(
            output_root=output_root,
            path=built.manifest_path,
        ),
        _file_evidence(
            output_root=output_root,
            path=built.payload_path,
        ),
        _file_evidence(
            output_root=output_root,
            path=built.completion_path,
        ),
    )

    cache_completion = write_job_completion(
        output_root,
        artifacts=cache_artifacts,
    )

    return {
        "built": built,
        "loaded": loaded,
        "cache_completion": cache_completion,
        "manifest_sha256": hashlib.sha256(
            built.manifest_path.read_bytes()
        ).hexdigest(),
    }


def _lifecycle(
    *,
    condition: str,
    components: dict,
    events: list,
) -> TrainerLifecycle:
    def recorder(event) -> None:
        events.append(event)

    return TrainerLifecycle(
        model_constructor=components["model_constructor"],
        target_adapter=(
            HardTargetAdapter()
            if condition == "hard_target"
            else SoftTargetAdapter()
        ),
        loss_adapter=(
            components["hard_loss"]
            if condition == "hard_target"
            else components["soft_loss"]
        ),
        optimizer_schedule_factory=components[
            "optimizer_factory"
        ],
        stop_rule=components["stop_rule"],
        recorder=recorder,
    )


def _configuration_refs(condition: str) -> dict[str, str]:
    return {
        "trainer": "technical-trainer/v1",
        "adapter": (
            "technical-hard-adapter/v1"
            if condition == "hard_target"
            else "technical-soft-adapter/v1"
        ),
    }


def _configuration_hashes(
    refs: dict[str, str],
) -> dict[str, str]:
    return {
        key: hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()
        for key, value in refs.items()
    }


def _training_log_bytes(result) -> bytes:
    payload = {
        "technical_fixture": True,
        "scientific_data": False,
        "production_eligible": False,
        "terminal_status": result.terminal_status,
        "terminal_reason": result.terminal_reason,
        "updates_completed": result.updates_completed,
        "target_cache_kind": result.target_cache_kind,
        "trajectory": [
            {
                "step": progress.step,
                "updates_completed": progress.updates_completed,
                "metrics": dict(progress.metrics),
            }
            for progress in result.trajectory
        ],
    }

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _attempt_cache_reference(
    *,
    condition_id: str,
    record_sha256: str,
) -> dict[str, object]:
    return {
        "record_type": "teacher_output_cache",
        "schema_version": "teacher_output_cache/v1",
        "condition_id": condition_id,
        "record_sha256": record_sha256,
    }


def _artifact_reference(
    *,
    path: str,
    sha256: str,
    storage_class: str,
) -> dict[str, str]:
    return {
        "path": path,
        "sha256": sha256,
        "storage_class": storage_class,
    }


def _status_for(reports, job_id: str):
    matches = [
        report
        for report in reports
        if report.job_id == job_id
    ]

    assert len(matches) == 1

    return matches[0]


def _run_branch(
    *,
    repo: Path,
    scratch: Path,
    registry: TechnicalJobRegistry,
    stage3: Stage3AvailabilityIndex,
    condition: str,
    nodes,
    components: dict,
) -> dict:
    started = time.perf_counter()

    cache_node, training_node, completion_node = nodes

    cache_root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=cache_node,
    )

    cache_info = _build_loaded_cache(
        output_root=cache_root,
        condition_id=cache_node.condition_id,
        condition=condition,
        input_count=int(
            components["inputs"].shape[0]
        ),
        class_count=components["class_count"],
    )

    after_cache = inspect_registry_statuses(
        registry=registry,
        repository_root=repo,
        scratch_root=scratch,
    )

    assert (
        _status_for(
            after_cache,
            cache_node.job_id,
        ).status
        == "completed"
    )
    assert (
        _status_for(
            after_cache,
            training_node.job_id,
        ).status
        == "planned"
    )

    identity = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition=condition,
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )

    training_root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=training_node,
    )

    events: list = []
    lifecycle = _lifecycle(
        condition=condition,
        components=components,
        events=events,
    )

    prepared = lifecycle.prepare(
        cache=cache_info["loaded"],
        model_seed=identity.training_seed.seed_value,
        device="cpu",
        settings=components["settings"],
    )

    refs = _configuration_refs(condition)
    config_hashes = _configuration_hashes(refs)

    checkpoint_dir = training_root.path / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resume_path = checkpoint_dir / "interrupted.pt"
    saved_resume = None

    def snapshot_callback(snapshot) -> None:
        nonlocal saved_resume

        if (
            snapshot.updates_completed == 1
            and saved_resume is None
        ):
            saved_resume = save_technical_resume_checkpoint(
                resume_path,
                prepared=prepared,
                snapshot=snapshot,
                attempt_identity=identity,
                stage3=stage3,
                configuration_hashes=config_hashes,
                target_cache_manifest_sha256=cache_info[
                    "manifest_sha256"
                ],
            )

    outer_training_mode = prepared.model.training

    with pytest.raises(TechnicalInterruption) as interrupted:
        lifecycle.run_technical(
            prepared=prepared,
            training_inputs=components["inputs"],
            configuration_refs=refs,
            technical_safety_step_limit=8,
            snapshot_callback=snapshot_callback,
            interrupt_after_updates=1,
        )

    assert interrupted.value.snapshot.updates_completed == 1
    assert saved_resume is not None
    assert saved_resume.updates_completed == 1

    during_interruption = inspect_registry_statuses(
        registry=registry,
        repository_root=repo,
        scratch_root=scratch,
    )
    training_running = _status_for(
        during_interruption,
        training_node.job_id,
    )

    assert training_running.status == "running"

    resume_decision = decide_attempt_resume(
        node=training_node,
        status_report=training_running,
        requested_attempt_identity=identity,
        checkpoint_attempt_identity=identity,
    )

    assert resume_decision.action == "resume_existing"

    resumed_events: list = []
    resumed_lifecycle = _lifecycle(
        condition=condition,
        components=components,
        events=resumed_events,
    )

    resumed_prepared = resumed_lifecycle.prepare(
        cache=cache_info["loaded"],
        model_seed=identity.training_seed.seed_value,
        device="cpu",
        settings=components["settings"],
    )

    restored_snapshot = load_technical_resume_checkpoint(
        resume_path,
        prepared=resumed_prepared,
        expected_attempt_identity=identity,
        stage3=stage3,
        expected_configuration_hashes=config_hashes,
        expected_target_cache_manifest_sha256=cache_info[
            "manifest_sha256"
        ],
        expected_file_sha256=saved_resume.file_sha256,
    )

    assert restored_snapshot.updates_completed == 1

    result = resumed_lifecycle.run_technical(
        prepared=resumed_prepared,
        training_inputs=components["inputs"],
        configuration_refs=refs,
        technical_safety_step_limit=8,
        resume_snapshot=restored_snapshot,
    )

    outcome_kind, failure_detail = outcome_from_training_result(
        result
    )

    assert outcome_kind == "succeeded"
    assert failure_detail is None
    assert result.terminal_status == "stop_rule_met"
    assert result.updates_completed > 1

    sealed_path = checkpoint_dir / "sealed.pt"
    sealed_snapshot = TechnicalLoopSnapshot(
        updates_completed=result.updates_completed,
        trajectory=result.trajectory,
        outer_training_mode=outer_training_mode,
    )

    sealed = save_technical_resume_checkpoint(
        sealed_path,
        prepared=resumed_prepared,
        snapshot=sealed_snapshot,
        attempt_identity=identity,
        stage3=stage3,
        configuration_hashes=config_hashes,
        target_cache_manifest_sha256=cache_info[
            "manifest_sha256"
        ],
    )

    reload_events: list = []
    reload_lifecycle = _lifecycle(
        condition=condition,
        components=components,
        events=reload_events,
    )
    reload_prepared = reload_lifecycle.prepare(
        cache=cache_info["loaded"],
        model_seed=identity.training_seed.seed_value,
        device="cpu",
        settings=components["settings"],
    )

    reloaded_terminal = load_technical_resume_checkpoint(
        sealed_path,
        prepared=reload_prepared,
        expected_attempt_identity=identity,
        stage3=stage3,
        expected_configuration_hashes=config_hashes,
        expected_target_cache_manifest_sha256=cache_info[
            "manifest_sha256"
        ],
        expected_file_sha256=sealed.file_sha256,
    )

    assert (
        reloaded_terminal.updates_completed
        == result.updates_completed
    )
    assert reloaded_terminal.trajectory == result.trajectory

    training_log = atomic_write_job_file(
        training_root,
        "logs/training.json",
        _training_log_bytes(result),
    )

    interruption_checkpoint = _file_evidence(
        output_root=training_root,
        path=resume_path,
    )
    sealed_checkpoint = _file_evidence(
        output_root=training_root,
        path=sealed_path,
    )

    training_completion = write_job_completion(
        training_root,
        artifacts=(
            training_log,
            interruption_checkpoint,
            sealed_checkpoint,
        ),
    )

    after_training = inspect_registry_statuses(
        registry=registry,
        repository_root=repo,
        scratch_root=scratch,
    )

    training_completed = _status_for(
        after_training,
        training_node.job_id,
    )
    technical_planned = _status_for(
        after_training,
        completion_node.job_id,
    )

    assert training_completed.status == "completed"
    assert technical_planned.status == "planned"

    completed_decision = decide_attempt_resume(
        node=training_node,
        status_report=training_completed,
        requested_attempt_identity=identity,
    )

    assert completed_decision.action == "skip_completed"

    completion_root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=completion_node,
    )

    target_cache_reference = _attempt_cache_reference(
        condition_id=cache_node.condition_id,
        record_sha256=cache_info[
            "manifest_sha256"
        ],
    )

    attempt_record = emit_technical_attempt_record(
        stage3=stage3,
        attempt_identity=identity,
        target_cache_reference=target_cache_reference,
        outcome_kind=outcome_kind,
        student_architecture_ref="technical-architecture/v1",
        replication_policy_ref="technical-replication/v1",
        training_config_ref="technical-training/v1",
        training_log=_artifact_reference(
            path=(
                training_root.relative_identity
                + "/logs/training.json"
            ),
            sha256=training_log.sha256,
            storage_class="external_log",
        ),
        model_checkpoint=_artifact_reference(
            path=(
                training_root.relative_identity
                + "/checkpoints/sealed.pt"
            ),
            sha256=sealed.file_sha256,
            storage_class="external_checkpoint",
        ),
        failure_detail=None,
    )

    attempt_bytes = canonical_attempt_record_bytes(
        attempt_record
    )

    attempt_artifact = atomic_write_job_file(
        completion_root,
        "attempt/record.json",
        attempt_bytes,
    )

    assert (
        attempt_artifact.sha256
        == attempt_record_sha256(attempt_record)
    )

    reloaded_attempt = json.loads(
        (
            completion_root.path
            / attempt_artifact.relative_path
        ).read_text(encoding="utf-8")
    )

    assert reloaded_attempt == attempt_record

    attempt_completion = write_job_completion(
        completion_root,
        artifacts=(attempt_artifact,),
    )

    final_branch_status = inspect_registry_statuses(
        registry=registry,
        repository_root=repo,
        scratch_root=scratch,
    )

    assert (
        _status_for(
            final_branch_status,
            cache_node.job_id,
        ).status
        == "completed"
    )
    assert (
        _status_for(
            final_branch_status,
            training_node.job_id,
        ).status
        == "completed"
    )
    assert (
        _status_for(
            final_branch_status,
            completion_node.job_id,
        ).status
        == "completed"
    )

    branch_paths = [
        path
        for path in (
            cache_root.path,
            training_root.path,
            completion_root.path,
        )
        if path.exists()
    ]

    peak_artifact_bytes = max(
        (
            path.stat().st_size
            for root in branch_paths
            for path in root.rglob("*")
            if path.is_file()
        ),
        default=0,
    )

    runtime = time.perf_counter() - started

    return {
        "condition": condition,
        "adapter": (
            "HardTargetAdapter"
            if condition == "hard_target"
            else "SoftTargetAdapter"
        ),
        "cache_kind": result.target_cache_kind,
        "identity": identity,
        "cache_manifest_sha256": cache_info[
            "manifest_sha256"
        ],
        "cache_completion_sha256": cache_info[
            "cache_completion"
        ].sha256,
        "resume_checkpoint_sha256": saved_resume.file_sha256,
        "sealed_checkpoint_sha256": sealed.file_sha256,
        "sealed_model_state_sha256": sealed.model_state_sha256,
        "training_completion_sha256": training_completion.sha256,
        "attempt_record_sha256": attempt_artifact.sha256,
        "attempt_completion_sha256": attempt_completion.sha256,
        "updates_completed": result.updates_completed,
        "runtime_seconds": runtime,
        "peak_artifact_bytes": peak_artifact_bytes,
        "scientific_data": False,
        "production_eligible": False,
    }


def test_hard_and_soft_end_to_end_technical_fixture(
    tmp_path: Path,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    stage3 = _stage3()
    components = _technical_training_components()

    hard_nodes = _branch_nodes(
        stage3,
        condition="hard_target",
    )
    soft_nodes = _branch_nodes(
        stage3,
        condition="soft_target",
    )

    registry = TechnicalJobRegistry(
        stage3=stage3,
        nodes=(
            *hard_nodes,
            *soft_nodes,
        ),
    )

    hard = _run_branch(
        repo=repo,
        scratch=scratch,
        registry=registry,
        stage3=stage3,
        condition="hard_target",
        nodes=hard_nodes,
        components=components,
    )

    soft = _run_branch(
        repo=repo,
        scratch=scratch,
        registry=registry,
        stage3=stage3,
        condition="soft_target",
        nodes=soft_nodes,
        components=components,
    )

    final_reports = inspect_registry_statuses(
        registry=registry,
        repository_root=repo,
        scratch_root=scratch,
    )

    assert len(final_reports) == 6
    assert all(
        report.status == "completed"
        for report in final_reports
    )

    merged = merge_status_evidence(
        reports=final_reports,
    )
    merged_reversed = merge_status_evidence(
        reports=reversed(final_reports),
    )

    registry_bytes = canonical_registry_bytes(merged)

    assert (
        canonical_registry_bytes(merged_reversed)
        == registry_bytes
    )
    assert registry_sha256(merged_reversed) == registry_sha256(
        merged
    )

    assert hard["cache_kind"] == "teacher_argmax"
    assert soft["cache_kind"] == "teacher_logits"

    assert (
        hard["identity"].condition_id
        != soft["identity"].condition_id
    )
    assert (
        hard["identity"].training_seed.seed_value
        != soft["identity"].training_seed.seed_value
    )
    assert (
        hard["identity"].tie_breaking_seed.seed_value
        != soft["identity"].tie_breaking_seed.seed_value
    )

    for branch in (hard, soft):
        assert branch["scientific_data"] is False
        assert branch["production_eligible"] is False

        print(
            "E2E_BRANCH "
            f"condition={branch['condition']} "
            f"adapter={branch['adapter']} "
            f"cache_kind={branch['cache_kind']} "
            f"updates={branch['updates_completed']} "
            f"runtime_seconds={branch['runtime_seconds']:.6f} "
            f"peak_artifact_bytes={branch['peak_artifact_bytes']}"
        )
        print(
            "E2E_IDENTITY "
            f"condition={branch['condition']} "
            f"condition_id={branch['identity'].condition_id} "
            f"attempt_index={branch['identity'].attempt_index} "
            f"retry_index={branch['identity'].retry_index} "
            f"training_seed={branch['identity'].training_seed.seed_value} "
            f"tie_breaking_seed="
            f"{branch['identity'].tie_breaking_seed.seed_value}"
        )
        print(
            "E2E_HASHES "
            f"condition={branch['condition']} "
            f"cache_manifest={branch['cache_manifest_sha256']} "
            f"cache_completion={branch['cache_completion_sha256']} "
            f"resume_checkpoint={branch['resume_checkpoint_sha256']} "
            f"sealed_checkpoint={branch['sealed_checkpoint_sha256']} "
            f"sealed_model_state={branch['sealed_model_state_sha256']} "
            f"training_completion={branch['training_completion_sha256']} "
            f"attempt_record={branch['attempt_record_sha256']} "
            f"attempt_completion={branch['attempt_completion_sha256']}"
        )

    print(
        "E2E_MERGE "
        f"entry_count={len(merged['entries'])} "
        f"registry_bytes={len(registry_bytes)} "
        f"registry_sha256={registry_sha256(merged)} "
        "input_order_independent=yes"
    )
    print("E2E_FIREWALL scientific_data=false")
    print("E2E_FIREWALL production_eligible=false")
    print("E2E_FIREWALL full_domain_eligibility=no")
    print("E2E_FIREWALL real_teacher_outputs=no")
    print("E2E_FIREWALL real_student_training=no")
    print("E2E_FIREWALL stage5d_summary=no")
