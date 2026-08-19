from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from circuit_families.stage4_condition_identity import (
    Stage3AvailabilityIndex,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
)
from circuit_families.stage5bc.student_trainer import (
    HardTargetAdapter,
    OptimizerScheduleBundle,
    PreparedTargets,
    TechnicalInterruption,
    TrainerLifecycle,
    TrainerSettingsBundle,
)
from circuit_families.stage5bc.technical_checkpoint import (
    TechnicalCheckpointError,
    load_technical_resume_checkpoint,
    save_technical_resume_checkpoint,
)
from circuit_families.training.checkpoints import canonical_state_hash

REGISTRY = Path(
    "followup/manifests/stage3_teacher_registry_v1.json"
)

CONFIG_HASHES = {
    "architecture_profile_sha256": "1" * 64,
    "trainer_profile_sha256": "2" * 64,
    "adapter_profile_sha256": "3" * 64,
    "resume_profile_sha256": "4" * 64,
}
TARGET_HASH = "5" * 64


class _TechnicalCache:
    def __init__(self) -> None:
        self.argmax = torch.tensor(
            [1, 0, 2, 1],
            dtype=torch.int64,
        )

    def stage4_view(self, cache_kind: str) -> torch.Tensor:
        if cache_kind != "teacher_argmax":
            raise AssertionError(cache_kind)
        return self.argmax


def _stage3() -> Stage3AvailabilityIndex:
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return Stage3AvailabilityIndex.from_registry(raw)


def _identity(stage3: Stage3AvailabilityIndex):
    return build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )


def _inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float32,
    )


def _settings() -> TrainerSettingsBundle:
    return TrainerSettingsBundle(
        model={"technical_width": 2},
        loss={"purpose": "technical_resume_fixture"},
        optimizer_schedule={
            "technical_learning_rate": 0.05,
            "technical_gamma": 0.8,
        },
        stop={"technical_stop_step": 4},
    )


def _model_constructor(
    *,
    seed: int,
    device: torch.device,
    settings,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Linear(2, 3)
    model.eval()
    return model.to(device)


def _optimizer_factory(
    *,
    model: torch.nn.Module,
    settings,
) -> OptimizerScheduleBundle:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(settings["technical_learning_rate"]),
        momentum=0.25,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=float(settings["technical_gamma"]),
    )
    return OptimizerScheduleBundle(
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _loss(
    *,
    outputs: torch.Tensor,
    targets: PreparedTargets,
    settings,
) -> torch.Tensor:
    return functional.cross_entropy(
        outputs,
        targets.values,
    )


def _stop_rule(*, progress, settings) -> bool:
    return progress.step >= int(settings["technical_stop_step"])


def _lifecycle(events: list) -> TrainerLifecycle:
    return TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_stop_rule,
        recorder=events.append,
    )


def _prepared(
    lifecycle: TrainerLifecycle,
    identity,
):
    return lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=identity.training_seed.seed_value,
        device="cpu",
        settings=_settings(),
    )


def test_forced_interruption_resume_matches_uninterrupted_exactly(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    identity = _identity(stage3)

    baseline_events = []
    baseline_lifecycle = _lifecycle(baseline_events)
    baseline = _prepared(
        baseline_lifecycle,
        identity,
    )

    baseline_result = baseline_lifecycle.run_technical(
        prepared=baseline,
        training_inputs=_inputs(),
        configuration_refs={
            "trainer": "technical-trainer-fixture/v1",
        },
        technical_safety_step_limit=6,
    )

    baseline_model_hash = canonical_state_hash(
        baseline.model.state_dict()
    )
    baseline_optimizer_hash = canonical_state_hash(
        baseline.optimizer_schedule.optimizer.state_dict()
    )
    baseline_scheduler_hash = canonical_state_hash(
        baseline.optimizer_schedule.scheduler.state_dict()
    )

    interrupted_events = []
    interrupted_lifecycle = _lifecycle(interrupted_events)
    interrupted = _prepared(
        interrupted_lifecycle,
        identity,
    )

    checkpoint_path = tmp_path / "resume.pt"
    evidence_holder = {}

    def save_snapshot(snapshot) -> None:
        if snapshot.updates_completed != 2:
            return

        evidence_holder["evidence"] = save_technical_resume_checkpoint(
            checkpoint_path,
            prepared=interrupted,
            snapshot=snapshot,
            attempt_identity=identity,
            stage3=stage3,
            configuration_hashes=CONFIG_HASHES,
            target_cache_manifest_sha256=TARGET_HASH,
        )

    with pytest.raises(TechnicalInterruption) as caught:
        interrupted_lifecycle.run_technical(
            prepared=interrupted,
            training_inputs=_inputs(),
            configuration_refs={
                "trainer": "technical-trainer-fixture/v1",
            },
            technical_safety_step_limit=6,
            snapshot_callback=save_snapshot,
            interrupt_after_updates=2,
        )

    assert caught.value.snapshot.updates_completed == 2
    assert checkpoint_path.is_file()

    resume_events = []
    resume_lifecycle = _lifecycle(resume_events)
    resumed = _prepared(
        resume_lifecycle,
        identity,
    )

    evidence = evidence_holder["evidence"]

    snapshot = load_technical_resume_checkpoint(
        checkpoint_path,
        prepared=resumed,
        expected_attempt_identity=identity,
        stage3=stage3,
        expected_configuration_hashes=CONFIG_HASHES,
        expected_target_cache_manifest_sha256=TARGET_HASH,
        expected_file_sha256=evidence.file_sha256,
    )

    resumed_result = resume_lifecycle.run_technical(
        prepared=resumed,
        training_inputs=_inputs(),
        configuration_refs={
            "trainer": "technical-trainer-fixture/v1",
        },
        technical_safety_step_limit=6,
        resume_snapshot=snapshot,
    )

    assert baseline_result.terminal_status == "stop_rule_met"
    assert resumed_result.terminal_status == "stop_rule_met"
    assert baseline_result.updates_completed == 4
    assert resumed_result.updates_completed == 4
    assert baseline_result.trajectory == resumed_result.trajectory

    assert canonical_state_hash(
        resumed.model.state_dict()
    ) == baseline_model_hash

    assert canonical_state_hash(
        resumed.optimizer_schedule.optimizer.state_dict()
    ) == baseline_optimizer_hash

    assert canonical_state_hash(
        resumed.optimizer_schedule.scheduler.state_dict()
    ) == baseline_scheduler_hash

    assert resumed.model.training is False


def test_wrong_attempt_identity_is_rejected(tmp_path: Path) -> None:
    stage3 = _stage3()
    identity = _identity(stage3)

    lifecycle = _lifecycle([])
    prepared = _prepared(lifecycle, identity)

    checkpoint_path = tmp_path / "wrong-identity.pt"

    holder = {}

    def save(snapshot) -> None:
        if snapshot.updates_completed == 1:
            holder["evidence"] = save_technical_resume_checkpoint(
                checkpoint_path,
                prepared=prepared,
                snapshot=snapshot,
                attempt_identity=identity,
                stage3=stage3,
                configuration_hashes=CONFIG_HASHES,
                target_cache_manifest_sha256=TARGET_HASH,
            )

    with pytest.raises(TechnicalInterruption):
        lifecycle.run_technical(
            prepared=prepared,
            training_inputs=_inputs(),
            configuration_refs={"trainer": "technical/v1"},
            technical_safety_step_limit=4,
            snapshot_callback=save,
            interrupt_after_updates=1,
        )

    wrong = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=1,
        attempt_index=0,
        retry_index=0,
    )

    fresh_lifecycle = _lifecycle([])
    fresh = _prepared(fresh_lifecycle, wrong)

    with pytest.raises(
        TechnicalCheckpointError,
        match="attempt identity mismatch",
    ):
        load_technical_resume_checkpoint(
            checkpoint_path,
            prepared=fresh,
            expected_attempt_identity=wrong,
            stage3=stage3,
            expected_configuration_hashes=CONFIG_HASHES,
            expected_target_cache_manifest_sha256=TARGET_HASH,
        )


@pytest.mark.parametrize(
    "stale",
    [
        "configuration",
        "target",
    ],
)
def test_stale_configuration_or_target_hash_is_rejected(
    tmp_path: Path,
    stale: str,
) -> None:
    stage3 = _stage3()
    identity = _identity(stage3)

    lifecycle = _lifecycle([])
    prepared = _prepared(lifecycle, identity)

    checkpoint_path = tmp_path / f"stale-{stale}.pt"

    def save(snapshot) -> None:
        if snapshot.updates_completed == 1:
            save_technical_resume_checkpoint(
                checkpoint_path,
                prepared=prepared,
                snapshot=snapshot,
                attempt_identity=identity,
                stage3=stage3,
                configuration_hashes=CONFIG_HASHES,
                target_cache_manifest_sha256=TARGET_HASH,
            )

    with pytest.raises(TechnicalInterruption):
        lifecycle.run_technical(
            prepared=prepared,
            training_inputs=_inputs(),
            configuration_refs={"trainer": "technical/v1"},
            technical_safety_step_limit=4,
            snapshot_callback=save,
            interrupt_after_updates=1,
        )

    fresh_lifecycle = _lifecycle([])
    fresh = _prepared(fresh_lifecycle, identity)

    configs = copy.deepcopy(CONFIG_HASHES)
    target = TARGET_HASH

    if stale == "configuration":
        configs["trainer_profile_sha256"] = "e" * 64
    else:
        target = "e" * 64

    pattern = (
        "configuration hashes"
        if stale == "configuration"
        else "target-cache hash"
    )

    with pytest.raises(
        TechnicalCheckpointError,
        match=pattern,
    ):
        load_technical_resume_checkpoint(
            checkpoint_path,
            prepared=fresh,
            expected_attempt_identity=identity,
            stage3=stage3,
            expected_configuration_hashes=configs,
            expected_target_cache_manifest_sha256=target,
        )


def test_physical_checkpoint_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    identity = _identity(stage3)

    lifecycle = _lifecycle([])
    prepared = _prepared(lifecycle, identity)

    checkpoint_path = tmp_path / "tampered.pt"
    holder = {}

    def save(snapshot) -> None:
        if snapshot.updates_completed == 1:
            holder["evidence"] = save_technical_resume_checkpoint(
                checkpoint_path,
                prepared=prepared,
                snapshot=snapshot,
                attempt_identity=identity,
                stage3=stage3,
                configuration_hashes=CONFIG_HASHES,
                target_cache_manifest_sha256=TARGET_HASH,
            )

    with pytest.raises(TechnicalInterruption):
        lifecycle.run_technical(
            prepared=prepared,
            training_inputs=_inputs(),
            configuration_refs={"trainer": "technical/v1"},
            technical_safety_step_limit=4,
            snapshot_callback=save,
            interrupt_after_updates=1,
        )

    payload = bytearray(checkpoint_path.read_bytes())
    payload[-1] ^= 1
    checkpoint_path.write_bytes(payload)

    fresh_lifecycle = _lifecycle([])
    fresh = _prepared(fresh_lifecycle, identity)

    with pytest.raises(
        TechnicalCheckpointError,
        match="physical SHA-256 mismatch",
    ):
        load_technical_resume_checkpoint(
            checkpoint_path,
            prepared=fresh,
            expected_attempt_identity=identity,
            stage3=stage3,
            expected_configuration_hashes=CONFIG_HASHES,
            expected_target_cache_manifest_sha256=TARGET_HASH,
            expected_file_sha256=holder["evidence"].file_sha256,
        )


def test_internal_model_state_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    identity = _identity(stage3)

    lifecycle = _lifecycle([])
    prepared = _prepared(lifecycle, identity)

    checkpoint_path = tmp_path / "internal-tamper.pt"

    def save(snapshot) -> None:
        if snapshot.updates_completed == 1:
            save_technical_resume_checkpoint(
                checkpoint_path,
                prepared=prepared,
                snapshot=snapshot,
                attempt_identity=identity,
                stage3=stage3,
                configuration_hashes=CONFIG_HASHES,
                target_cache_manifest_sha256=TARGET_HASH,
            )

    with pytest.raises(TechnicalInterruption):
        lifecycle.run_technical(
            prepared=prepared,
            training_inputs=_inputs(),
            configuration_refs={"trainer": "technical/v1"},
            technical_safety_step_limit=4,
            snapshot_callback=save,
            interrupt_after_updates=1,
        )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    first_key = next(iter(payload["model_state"]))
    payload["model_state"][first_key] = (
        payload["model_state"][first_key].clone()
    )
    payload["model_state"][first_key].view(-1)[0] += 1.0
    torch.save(payload, checkpoint_path)

    fresh_lifecycle = _lifecycle([])
    fresh = _prepared(fresh_lifecycle, identity)

    with pytest.raises(
        TechnicalCheckpointError,
        match="internal state hash mismatch",
    ):
        load_technical_resume_checkpoint(
            checkpoint_path,
            prepared=fresh,
            expected_attempt_identity=identity,
            stage3=stage3,
            expected_configuration_hashes=CONFIG_HASHES,
            expected_target_cache_manifest_sha256=TARGET_HASH,
        )


def test_checkpoint_contains_scheduler_rng_and_seed_evidence(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    identity = _identity(stage3)

    lifecycle = _lifecycle([])
    prepared = _prepared(lifecycle, identity)

    checkpoint_path = tmp_path / "contents.pt"

    def save(snapshot) -> None:
        if snapshot.updates_completed == 1:
            save_technical_resume_checkpoint(
                checkpoint_path,
                prepared=prepared,
                snapshot=snapshot,
                attempt_identity=identity,
                stage3=stage3,
                configuration_hashes=CONFIG_HASHES,
                target_cache_manifest_sha256=TARGET_HASH,
            )

    with pytest.raises(TechnicalInterruption):
        lifecycle.run_technical(
            prepared=prepared,
            training_inputs=_inputs(),
            configuration_refs={"trainer": "technical/v1"},
            technical_safety_step_limit=4,
            snapshot_callback=save,
            interrupt_after_updates=1,
        )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert payload["scientific_data"] is False
    assert payload["production_eligible"] is False
    assert payload["scheduler_state"] is not None
    assert isinstance(payload["torch_rng_state"], torch.Tensor)
    assert "training_seed" in payload["attempt_identity"]
    assert "tie_breaking_seed" in payload["attempt_identity"]
    assert payload["snapshot"]["updates_completed"] == 1
