#!/usr/bin/env python3
"""Validate the Stage 12-P1 policy-neutral teacher foundation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F

from circuit_families.stage12p1.phase import (
    build_teacher_trajectory,
    select_teacher_phases,
)
from circuit_families.stage12p1.records import (
    build_foundation_record,
    foundation_record_sha256,
    validate_foundation_record,
)
from circuit_families.stage12p1.tasks import (
    TASK_CONFIG_SCHEMA_VERSION,
    ModularAdditionImplementation,
    build_task_record,
)
from circuit_families.stage12p1.teacher import (
    TeacherTrainingAdapter,
    TeacherTrainingRequest,
    build_technical_tabular_teacher_data,
    emit_unavailable_teacher,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _task() -> dict:
    implementation = ModularAdditionImplementation()

    return build_task_record(
        {
            "schema_version": TASK_CONFIG_SCHEMA_VERSION,
            "task_id": "technical-stage12p1-validation",
            "implementation": implementation.name,
            "implementation_version": implementation.version,
            "modulus": 5,
            "input_domains": [
                list(range(5)),
                list(range(5)),
            ],
            "parameters": {},
            "split_identity": {
                "kind": "technical-validation-split",
                "version": "v1",
                "train_indices": list(range(20)),
                "test_indices": list(range(20, 25)),
            },
            "architecture_compatibility": {
                "input_arity": 2,
                "output_class_count": 5,
                "classification": "technical-fixture-only",
            },
            "scientific_data": False,
            "production_eligible": False,
        }
    )


def _request(
    output_root: Path,
    *,
    resume_id: str,
) -> TeacherTrainingRequest:
    return TeacherTrainingRequest(
        task_record=_task(),
        architecture_config={
            "kind": "tiny-linear-validation-fixture",
            "input_dim": 2,
            "output_dim": 5,
        },
        training_config={
            "loss": "cross_entropy",
            "classification": "technical-fixture-only",
        },
        optimizer_config={
            "kind": "sgd",
            "learning_rate": 0.01,
        },
        scheduler_config={
            "kind": "none",
        },
        stopping_config={
            "technical_stop_step": 0,
        },
        backend_qualification={
            "backend_id": "cpu-technical-validation",
            "device": "cpu",
            "qualified": True,
            "exact_resume_supported": True,
            "qualification_ref": "stage12p1-validate-only/v1",
        },
        model_seed_id="technical-validation-model-seed/v1",
        model_seed=101,
        training_seed_id="technical-validation-training-seed/v1",
        training_seed=202,
        max_technical_updates=2,
        checkpoint_interval=1,
        checkpoint_retention=1,
        output_root=output_root,
        resume_id=resume_id,
    )


def _model_constructor(
    *,
    task_record,
    architecture_config,
    seed,
    device,
):
    del task_record
    torch.manual_seed(seed)
    return torch.nn.Linear(
        int(architecture_config["input_dim"]),
        int(architecture_config["output_dim"]),
    ).to(device)


def _loss(*, outputs, targets, config):
    if config["loss"] != "cross_entropy":
        raise ValueError("unexpected validation loss configuration")
    return F.cross_entropy(outputs, targets)


def _optimizer(*, model, config):
    if config["kind"] != "sgd":
        raise ValueError("unexpected validation optimizer configuration")
    return torch.optim.SGD(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )


def _scheduler(*, optimizer, config):
    del optimizer
    if config["kind"] != "none":
        raise ValueError("unexpected validation scheduler configuration")
    return None


def _stop(*, step, metrics, config):
    if set(metrics) != {
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
    }:
        raise ValueError("validation metric surface changed")
    return step >= int(config["technical_stop_step"])


def _trajectory_rows() -> list[dict]:
    tests = {
        0: 0.00,
        50: 0.02,
        100: 0.15,
        150: 0.50,
        200: 0.90,
        250: 0.990,
        300: 0.995,
        350: 0.997,
        400: 0.999,
        450: 1.000,
    }

    rows = []
    for step, test_accuracy in tests.items():
        rows.append(
            {
                "training_step": step,
                "train_accuracy": (
                    0.5
                    if step == 0
                    else 1.0
                ),
                "test_accuracy": test_accuracy,
                "train_loss": 0.1,
                "test_loss": 1.0,
                "checkpoint_path": (
                    "checkpoints/technical-validation/"
                    f"step_{step:08d}.pt"
                ),
                "checkpoint_sha256": _sha(
                    f"technical-validation-checkpoint-{step}"
                ),
            }
        )

    return rows


def run_validate_only() -> dict[str, object]:
    with tempfile.TemporaryDirectory(
        prefix="stage12p1-validate-"
    ) as temporary:
        output_root = Path(temporary)

        request = _request(
            output_root,
            resume_id="completed",
        )

        adapter = TeacherTrainingAdapter(
            model_constructor=_model_constructor,
            loss_fn=_loss,
            optimizer_factory=_optimizer,
            scheduler_factory=_scheduler,
            stop_rule=_stop,
        )

        data = build_technical_tabular_teacher_data(
            request.task_record
        )

        completed = adapter.run(
            request=request,
            data=data,
        )

        trajectory = build_teacher_trajectory(
            teacher_seed_id="technical-validation-teacher-seed/v1",
            teacher_seed=request.model_seed,
            task_identity_sha256=request.task_record["hashes"][
                "task_identity_sha256"
            ],
            teacher_artifact_sha256=completed.artifact_sha256,
            records=_trajectory_rows(),
        )

        phases = select_teacher_phases(trajectory)

        completed_record = build_foundation_record(
            request=request,
            result=completed,
            phase_selection=phases,
        )
        validate_foundation_record(completed_record)

        unavailable_request = _request(
            output_root,
            resume_id="unavailable",
        )
        unavailable = emit_unavailable_teacher(
            unavailable_request,
            reason="technical validation unavailable fixture",
        )
        unavailable_record = build_foundation_record(
            request=unavailable_request,
            result=unavailable,
        )
        validate_foundation_record(unavailable_record)

        return {
            "stage12p1_validate_only": "PASS",
            "scientific_data": False,
            "production_eligible": False,
            "production_teachers_trained": False,
            "completed_record_sha256": foundation_record_sha256(
                completed_record
            ),
            "unavailable_record_sha256": foundation_record_sha256(
                unavailable_record
            ),
            "task_identity_sha256": request.task_record["hashes"][
                "task_identity_sha256"
            ],
            "phase_rule_id": phases["selection_rule"]["rule_id"],
            "phase_rule_version": phases["selection_rule"][
                "rule_version"
            ],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Stage 12-P1 technical-only task/teacher "
            "foundation."
        )
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run portable synthetic validation only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.validate_only:
        print(
            "ERROR: this CLI exposes validate-only operation",
            file=sys.stderr,
        )
        return 2

    report = run_validate_only()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
