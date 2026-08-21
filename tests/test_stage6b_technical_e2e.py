from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage4_schema_common import CommonSchemaContract
from circuit_families.stage4_schema_records import validate_part_m_record
from circuit_families.stage5bc.attempt_records import (
    attempt_record_sha256,
    emit_technical_attempt_record,
    outcome_from_training_result,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
)
from circuit_families.stage5bc.student_trainer import (
    HardTargetAdapter,
    OptimizerScheduleBundle,
    TechnicalLoopSnapshot,
    TrainerLifecycle,
    TrainerSettingsBundle,
)
from circuit_families.stage5bc.target_cache import (
    FULL_DOMAIN_EXAMPLE_COUNT,
    TargetCacheBatch,
    build_target_cache,
    load_target_cache,
)
from circuit_families.stage5bc.technical_checkpoint import (
    load_technical_resume_checkpoint,
    save_technical_resume_checkpoint,
)
from circuit_families.stage5bc.technical_profiles import (
    load_technical_profile_set,
)
from circuit_families.stage6b import (
    CanonicalDecisionVector,
    HardAttemptLedger,
    HardLabelLossAdapter,
    assess_hard_attempt,
    canonical_decision_bytes,
    circuit_release_gate,
    evaluate_hard_target_eligibility,
    generate_hard_sealing_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
VOCAB_PATH = ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
IDENTITY_PATH = ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
PROFILE_PATH = ROOT / "tests/fixtures/stage5bc/technical_profile_set_v1.json"
ORDERING_REF = "technical-stage6b-full-domain-order/v1"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(*, path: str, sha256: str, storage_class: str) -> dict[str, str]:
    return {
        "path": path,
        "sha256": sha256,
        "storage_class": storage_class,
    }


def _condition_id(
    stage3: Stage3AvailabilityIndex,
    *,
    condition: str,
    initialization: int | None = None,
) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition=condition,
            student_initialization=initialization,
        ),
        stage3,
    )


class _TechnicalReplayModel(torch.nn.Module):
    """One-parameter fixture model that replays binary labels from its input."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(4.0, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        labels = inputs[:, 0]
        return self.scale * torch.stack((1.0 - labels, labels), dim=-1)


class _ObservedHardLoss(HardLabelLossAdapter):
    def __init__(self) -> None:
        self.calls = 0
        self.cache_kinds: list[str] = []

    def __call__(self, *, outputs, targets, settings):
        self.calls += 1
        self.cache_kinds.append(targets.cache_kind)
        return super().__call__(
            outputs=outputs,
            targets=targets,
            settings=settings,
        )


def _model_constructor(*, seed: int, device: torch.device, settings):
    assert settings == {
        "profile_id": "technical-architecture-fixture/v1",
        "technical_fixture": True,
    }
    torch.manual_seed(seed)
    model = _TechnicalReplayModel()
    model.eval()
    return model.to(device)


def _optimizer_factory(
    *,
    model: torch.nn.Module,
    settings,
) -> OptimizerScheduleBundle:
    assert settings == {
        "technical_fixture": True,
        "technical_learning_rate": 0.01,
    }
    return OptimizerScheduleBundle(
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=float(settings["technical_learning_rate"]),
        ),
        scheduler=None,
    )


def _stop_rule(*, progress, settings) -> bool:
    assert settings == {
        "technical_fixture": True,
        "technical_stop_step": 1,
    }
    return progress.updates_completed >= settings["technical_stop_step"]


def _settings() -> TrainerSettingsBundle:
    return TrainerSettingsBundle(
        model={
            "profile_id": "technical-architecture-fixture/v1",
            "technical_fixture": True,
        },
        loss={
            "loss_kind": "cross_entropy",
            "reduction": "mean",
        },
        optimizer_schedule={
            "technical_fixture": True,
            "technical_learning_rate": 0.01,
        },
        stop={
            "technical_fixture": True,
            "technical_stop_step": 1,
        },
    )


def _configuration_refs() -> dict[str, str]:
    return {
        "architecture_profile": "technical-architecture-fixture/v1",
        "trainer_profile": "technical-trainer-fixture/v1",
        "adapter_profile": "technical-adapter-fixture/v1",
    }


def _configuration_hashes() -> dict[str, str]:
    return {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in _configuration_refs().items()
    }


def _build_and_load_cache(output_root: Path, stage3: Stage3AvailabilityIndex):
    decisions = torch.arange(FULL_DOMAIN_EXAMPLE_COUNT, dtype=torch.int64) % 2
    logits = torch.stack(
        (
            torch.where(decisions == 0, 3.0, -3.0),
            torch.where(decisions == 1, 3.0, -3.0),
        ),
        dim=-1,
    ).to(torch.float32)
    input_ids = tuple(
        f"technical-stage6b-input-{index:05d}"
        for index in range(FULL_DOMAIN_EXAMPLE_COUNT)
    )
    teacher_reference = {
        "record_type": "teacher_reference",
        "schema_version": "teacher_reference/v1",
        "condition_id": _condition_id(stage3, condition="hard_target"),
        "record_sha256": "5" * 64,
    }
    provenance_hashes = {
        "dataset_sha256": "6" * 64,
        "split_sha256": "7" * 64,
        "task_config_sha256": "8" * 64,
        "model_config_sha256": "9" * 64,
        "training_config_sha256": "a" * 64,
        "component_basis_sha256": "b" * 64,
    }
    built = build_target_cache(
        output_root=output_root,
        manifest_relative_path="cache/manifest.json",
        payload_relative_path="cache/payload.bin",
        completion_relative_path="cache/completion.json",
        manifest_id="technical-stage6b-part-f-cache/v1",
        ordering_ref=ORDERING_REF,
        expected_example_count=FULL_DOMAIN_EXAMPLE_COUNT,
        expected_class_count=2,
        teacher_reference=teacher_reference,
        provenance_hashes=provenance_hashes,
        batches=(TargetCacheBatch(input_ids=input_ids, raw_logits=logits),),
        technical_fixture=True,
        stage4_record_serializable=False,
        expected_input_ids=input_ids,
    )
    loaded = load_target_cache(
        output_root=output_root,
        manifest_relative_path="cache/manifest.json",
        expected_input_ids=input_ids,
        expected_teacher_reference=teacher_reference,
        expected_provenance_hashes=provenance_hashes,
        expected_stage4_cache_kind="teacher_argmax",
    )
    return built, loaded


def _training_log_sha256(result) -> str:
    value = {
        "scientific_data": False,
        "production_eligible": False,
        "target_cache_kind": result.target_cache_kind,
        "terminal_reason": result.terminal_reason,
        "terminal_status": result.terminal_status,
        "updates_completed": result.updates_completed,
        "trajectory": [
            {
                "metrics": dict(progress.metrics),
                "step": progress.step,
                "updates_completed": progress.updates_completed,
            }
            for progress in result.trajectory
        ],
    }
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_attempt(
    *,
    output_root: Path,
    stage3: Stage3AvailabilityIndex,
    loaded_cache,
    cache_manifest_sha256: str,
    retry_index: int,
    lifecycle: TrainerLifecycle,
):
    identity = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=retry_index,
    )
    prepared = lifecycle.prepare(
        cache=loaded_cache,
        model_seed=identity.training_seed.seed_value,
        device="cpu",
        settings=_settings(),
    )
    inputs = loaded_cache.argmax.to(torch.float32).unsqueeze(-1)
    result = lifecycle.run_technical(
        prepared=prepared,
        training_inputs=inputs,
        configuration_refs=_configuration_refs(),
        technical_safety_step_limit=1,
    )
    outcome_kind, failure_detail = outcome_from_training_result(result)
    assert outcome_kind == "succeeded"
    assert failure_detail is None

    checkpoint_path = output_root / f"checkpoints/attempt-r{retry_index}.pt"
    checkpoint = save_technical_resume_checkpoint(
        checkpoint_path,
        prepared=prepared,
        snapshot=TechnicalLoopSnapshot(
            updates_completed=result.updates_completed,
            trajectory=result.trajectory,
            outer_training_mode=prepared.model.training,
        ),
        attempt_identity=identity,
        stage3=stage3,
        configuration_hashes=_configuration_hashes(),
        target_cache_manifest_sha256=cache_manifest_sha256,
    )
    restored = load_technical_resume_checkpoint(
        checkpoint_path,
        prepared=prepared,
        expected_attempt_identity=identity,
        stage3=stage3,
        expected_configuration_hashes=_configuration_hashes(),
        expected_target_cache_manifest_sha256=cache_manifest_sha256,
        expected_file_sha256=checkpoint.file_sha256,
    )
    assert restored.updates_completed == result.updates_completed

    with torch.no_grad():
        decisions = prepared.model(inputs).argmax(dim=-1).to(torch.int64)

    attempt_record = emit_technical_attempt_record(
        stage3=stage3,
        attempt_identity=identity,
        target_cache_reference={
            "record_type": "teacher_output_cache",
            "schema_version": "teacher_output_cache/v1",
            "condition_id": _condition_id(stage3, condition="hard_target"),
            "record_sha256": cache_manifest_sha256,
        },
        outcome_kind=outcome_kind,
        student_architecture_ref="technical-architecture-fixture/v1",
        replication_policy_ref="technical-orchestration-fixture/v1",
        training_config_ref="technical-trainer-fixture/v1",
        training_log=_artifact(
            path=f"technical/stage6b/attempt-r{retry_index}.json",
            sha256=_training_log_sha256(result),
            storage_class="external_log",
        ),
        model_checkpoint=_artifact(
            path=f"technical/stage6b/attempt-r{retry_index}.pt",
            sha256=checkpoint.file_sha256,
            storage_class="external_checkpoint",
        ),
        failure_detail=failure_detail,
    )
    return identity, prepared, result, checkpoint, attempt_record, tuple(decisions.tolist())


def _validate_stage4_record(
    record,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
    registry,
    registry_sha256: str,
) -> None:
    validate_part_m_record(
        record,
        contract=contract,
        stage3=stage3,
        stage3_registry=registry,
        stage3_registry_sha256=registry_sha256,
    )


def run_stage6b_part_f_fixture(output_root: Path) -> dict[str, object]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_sha256 = _file_sha256(REGISTRY_PATH)
    stage3 = Stage3AvailabilityIndex.from_registry(registry)
    contract = CommonSchemaContract.from_specs(
        json.loads(VOCAB_PATH.read_text(encoding="utf-8")),
        json.loads(IDENTITY_PATH.read_text(encoding="utf-8")),
    )
    profile_set = load_technical_profile_set(PROFILE_PATH)
    orchestration = profile_set.by_kind("orchestration")
    assert orchestration.settings["technical_worker_limit"] == 1
    assert all(not profile.resolves_decisions for profile in profile_set.profiles)

    built_cache, loaded_cache = _build_and_load_cache(output_root, stage3)
    cache_manifest_sha256 = _file_sha256(built_cache.manifest_path)
    manifest = loaded_cache.manifest.to_mapping()
    ordering_sha256 = manifest["input_order"]["ordered_input_ids_sha256"]
    teacher_decisions = tuple(loaded_cache.argmax.tolist())

    loss_adapter = _ObservedHardLoss()
    events = []
    lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=loss_adapter,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_stop_rule,
        recorder=events.append,
    )
    positive_data = _run_attempt(
        output_root=output_root,
        stage3=stage3,
        loaded_cache=loaded_cache,
        cache_manifest_sha256=cache_manifest_sha256,
        retry_index=0,
        lifecycle=lifecycle,
    )
    negative_data = _run_attempt(
        output_root=output_root,
        stage3=stage3,
        loaded_cache=loaded_cache,
        cache_manifest_sha256=cache_manifest_sha256,
        retry_index=1,
        lifecycle=lifecycle,
    )
    _, _, positive_result, positive_checkpoint, positive_attempt, positive_values = (
        positive_data
    )
    _, _, negative_result, negative_checkpoint, negative_attempt, negative_values = (
        negative_data
    )
    assert positive_values == teacher_decisions
    assert negative_values == teacher_decisions

    mutated_values = list(negative_values)
    mutated_values[-1] = 1 - mutated_values[-1]
    assert sum(
        teacher != student
        for teacher, student in zip(
            teacher_decisions,
            mutated_values,
            strict=True,
        )
    ) == 1
    teacher_vector = CanonicalDecisionVector(
        role="direct_teacher",
        condition_id=_condition_id(stage3, condition="direct_teacher"),
        ordering_ref=ORDERING_REF,
        ordered_input_ids_sha256=ordering_sha256,
        decisions=teacher_decisions,
    )

    positive_attempt = copy.deepcopy(positive_attempt)
    positive_attempt["record_status"] = "sealed"
    positive_evaluation = evaluate_hard_target_eligibility(
        teacher=teacher_vector,
        student=CanonicalDecisionVector(
            role="hard_target_student",
            condition_id=positive_attempt["condition_id"],
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ordering_sha256,
            decisions=positive_values,
        ),
        stage3=stage3,
    )
    negative_evaluation = evaluate_hard_target_eligibility(
        teacher=teacher_vector,
        student=CanonicalDecisionVector(
            role="hard_target_student",
            condition_id=negative_attempt["condition_id"],
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ordering_sha256,
            decisions=tuple(mutated_values),
        ),
        stage3=stage3,
    )
    positive = assess_hard_attempt(
        attempt_record=positive_attempt,
        stage3=stage3,
        evaluation=positive_evaluation,
    )
    negative = assess_hard_attempt(
        attempt_record=negative_attempt,
        stage3=stage3,
        evaluation=negative_evaluation,
    )
    ledger = HardAttemptLedger()
    ledger.add(positive)
    ledger.add(negative)

    dense_output_sha256 = hashlib.sha256(
        canonical_decision_bytes(positive_values)
    ).hexdigest()
    sealing = generate_hard_sealing_evidence(
        assessment=positive,
        stage3=stage3,
        checkpoint=positive_attempt["payload"]["model_checkpoint"],
        dense_output=_artifact(
            path="technical/stage6b/positive-dense-output.bin",
            sha256=dense_output_sha256,
            storage_class="external_large_object",
        ),
        architecture_ref="technical-architecture-fixture/v1",
    )
    positive_gate = circuit_release_gate(assessment=positive, sealing=sealing)
    negative_gate = circuit_release_gate(assessment=negative, sealing=None)

    for record in (
        positive.attempt_record,
        negative.attempt_record,
        positive.eligibility.stage4_record,
        negative.eligibility.stage4_record,
        sealing.stage4_record,
    ):
        _validate_stage4_record(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha256=registry_sha256,
        )

    return {
        "accounting": {
            "attempt_count": ledger.attempt_count,
            "eligible_count": ledger.classification_count("eligible"),
            "negative_attempt_retained": negative in ledger.assessments(),
            "subperfect_count": ledger.classification_count(
                "subperfect_agreement"
            ),
        },
        "cache": {
            "cache_kind": lifecycle.target_cache_kind,
            "example_count": manifest["example_count"],
            "loaded_by_accepted_interface": True,
            "manifest_sha256": cache_manifest_sha256,
        },
        "negative": {
            "agreement_count": negative_evaluation.agreement_count,
            "attempt_outcome": negative.attempt_record["payload"][
                "attempt_outcome"
            ],
            "attempt_record_sha256": attempt_record_sha256(
                negative.attempt_record
            ),
            "checkpoint_file_sha256": negative_checkpoint.file_sha256,
            "checkpoint_state_sha256": negative_checkpoint.model_state_sha256,
            "eligibility_record_sha256": negative.eligibility.stage4_record_sha256,
            "eligibility_status": "ineligible",
            "failure_kind": negative.classification,
            "gate_allowed": negative_gate.allowed,
            "gate_reason": negative_gate.reason,
            "sealing_status": "unsealed",
            "student_decisions_sha256": (
                negative_evaluation.student_decisions_sha256
            ),
            "teacher_decisions_sha256": (
                negative_evaluation.teacher_decisions_sha256
            ),
            "total_count": negative_evaluation.total_count,
            "updates_completed": negative_result.updates_completed,
        },
        "positive": {
            "agreement_count": positive_evaluation.agreement_count,
            "attempt_outcome": positive.attempt_record["payload"][
                "attempt_outcome"
            ],
            "attempt_record_sha256": attempt_record_sha256(
                positive.attempt_record
            ),
            "checkpoint_file_sha256": positive_checkpoint.file_sha256,
            "checkpoint_state_sha256": positive_checkpoint.model_state_sha256,
            "dense_output_sha256": sealing.dense_output_sha256,
            "eligibility_record_sha256": positive.eligibility.stage4_record_sha256,
            "eligibility_status": "eligible",
            "failure_kind": None,
            "gate_allowed": positive_gate.allowed,
            "gate_reason": positive_gate.reason,
            "sealed_model_record_sha256": sealing.stage4_record_sha256,
            "sealing_status": "sealed_hash_consistent",
            "student_decisions_sha256": (
                positive_evaluation.student_decisions_sha256
            ),
            "teacher_decisions_sha256": (
                positive_evaluation.teacher_decisions_sha256
            ),
            "total_count": positive_evaluation.total_count,
            "updates_completed": positive_result.updates_completed,
        },
        "shared_trainer": {
            "events_recorded": len(events),
            "hard_loss_calls": loss_adapter.calls,
            "loss_cache_kinds": loss_adapter.cache_kinds,
            "lifecycle_class": type(lifecycle).__name__,
            "one_worker": True,
        },
        "scientific_data": False,
        "production_eligible": False,
        "records": {
            "attempt_records_validated": 2,
            "eligibility_records_validated": 2,
            "sealed_model_records_validated": 1,
            "stage4_compatible": True,
        },
    }


def test_stage6b_part_f_technical_end_to_end(tmp_path: Path) -> None:
    result = run_stage6b_part_f_fixture(tmp_path)

    assert result["cache"] == {
        "cache_kind": "teacher_argmax",
        "example_count": 12_769,
        "loaded_by_accepted_interface": True,
        "manifest_sha256": result["cache"]["manifest_sha256"],
    }
    assert result["shared_trainer"] == {
        "events_recorded": 4,
        "hard_loss_calls": 2,
        "loss_cache_kinds": ["teacher_argmax", "teacher_argmax"],
        "lifecycle_class": "TrainerLifecycle",
        "one_worker": True,
    }
    assert result["positive"]["agreement_count"] == 12_769
    assert result["positive"]["attempt_outcome"] == "succeeded"
    assert result["positive"]["eligibility_status"] == "eligible"
    assert result["positive"]["sealing_status"] == "sealed_hash_consistent"
    assert result["positive"]["gate_allowed"] is True
    assert result["negative"]["agreement_count"] == 12_768
    assert result["negative"]["attempt_outcome"] == "succeeded"
    assert result["negative"]["failure_kind"] == "subperfect_agreement"
    assert result["negative"]["sealing_status"] == "unsealed"
    assert result["negative"]["gate_allowed"] is False
    assert result["accounting"] == {
        "attempt_count": 2,
        "eligible_count": 1,
        "negative_attempt_retained": True,
        "subperfect_count": 1,
    }
    assert result["scientific_data"] is False
    assert result["production_eligible"] is False
    assert result["records"] == {
        "attempt_records_validated": 2,
        "eligibility_records_validated": 2,
        "sealed_model_records_validated": 1,
        "stage4_compatible": True,
    }


def _run_with_hash_seed(seed: int) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(seed)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(ROOT / "src"), existing_pythonpath)
        if part
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--emit-fixture"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_stage6b_part_f_is_pythonhashseed_deterministic() -> None:
    seed_11 = _run_with_hash_seed(11)
    seed_29 = _run_with_hash_seed(29)

    assert seed_11 == seed_29


def _main() -> int:
    if sys.argv[1:] != ["--emit-fixture"]:
        return 2
    with tempfile.TemporaryDirectory(prefix="stage6b-part-f-") as temporary:
        result = run_stage6b_part_f_fixture(Path(temporary))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
