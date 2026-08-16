from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from circuit_families.analysis.stage3_teacher_inputs import CANONICAL_TEACHERS
from circuit_families.analysis.stage3_teacher_registry import (
    RegistryTeacherProvenance,
    Stage3RegistryError,
    build_phase_selection_table,
    build_registry,
    serialize_phase_selection_table_csv,
    serialize_registry_json,
)
from circuit_families.analysis.stage3_teacher_registry_verify import (
    Stage3RegistryVerificationError,
    verify_registry_physical,
    verify_registry_structure,
    verify_resolution_linkage,
)
from circuit_families.analysis.stage3_teacher_selection import (
    extract_all_teacher_candidates,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_stage3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import circuit_families.analysis.stage3_teacher_inputs as inputs

    steps = tuple(range(0, 501, 50))
    monkeypatch.setattr(inputs, "EXPECTED_STEPS", steps)

    successor = tmp_path / "successor"
    predecessor = tmp_path / "predecessor"
    successor.mkdir()
    predecessor.mkdir()

    teacher_runs = []

    for teacher in CANONICAL_TEACHERS:
        run = teacher.run_id
        checkpoint_dir = predecessor / "checkpoints" / run
        checkpoint_dir.mkdir(parents=True)

        rows = []
        tests = {
            0: 0.01,
            50: 0.051 if teacher.seed == 0 else 0.04,
            100: 0.10,
            150: 0.30,
            200: 0.49,
            250: 0.80,
            300: 0.990,
            350: 0.991,
            400: 0.992,
            450: 0.993,
            500: 0.994,
        }

        for step in steps:
            checkpoint = checkpoint_dir / f"step_{step:08d}.pt"
            checkpoint.write_bytes(
                f"seed={teacher.seed};step={step}".encode("utf-8")
            )
            rows.append(
                {
                    "run_id": run,
                    "training_step": step,
                    "train_accuracy": 0.5 if step == 0 else 1.0,
                    "test_accuracy": tests[step],
                    "train_loss": 0.01,
                    "test_loss": 1.0,
                    "checkpoint_path": checkpoint.relative_to(
                        predecessor
                    ).as_posix(),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )

        metrics_rel = f"results/raw/{run}/metrics.jsonl"
        metrics_path = predecessor / metrics_rel
        metrics_path.parent.mkdir(parents=True)
        metrics_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        manifest = {
            "run_id": run,
            "model_seed": teacher.seed,
            "evaluation_interval": 50,
            "checkpoint_interval": 50,
            "output_paths": {
                "metrics_jsonl": metrics_rel,
                "checkpoint_directory": f"checkpoints/{run}",
            },
            "hashes": {
                "metrics_jsonl_sha256": sha256(metrics_path),
            },
        }

        manifest_path = predecessor / teacher.manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )

        teacher_runs.append(
            {
                "teacher_seed": teacher.seed,
                "run_id": run,
                "training_manifest": {
                    "path": teacher.manifest_path,
                    "sha256": sha256(manifest_path),
                },
            }
        )

    link_path = (
        successor
        / "followup"
        / "manifests"
        / "predecessor_link_v1.json"
    )
    link_path.parent.mkdir(parents=True)
    link_path.write_text(
        json.dumps({"teacher_runs": teacher_runs}, sort_keys=True),
        encoding="utf-8",
    )

    decisions_path = (
        successor
        / "followup"
        / "configs"
        / "stage2_unresolved_decisions_v1.json"
    )
    decisions_path.parent.mkdir(parents=True)
    decisions_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {"decision_id": "UD-001", "status": "unresolved"},
                    {"decision_id": "UD-002", "status": "unresolved"},
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    candidates = extract_all_teacher_candidates(successor, predecessor)

    provenance = {}
    for candidate in candidates:
        validated = candidate.validated_input
        provenance[candidate.seed] = RegistryTeacherProvenance(
            teacher_seed=candidate.seed,
            canonical_run_id=candidate.run_id,
            training_manifest_path=validated.manifest_path,
            metrics_path=validated.metrics_path,
            training_manifest_sha256=validated.manifest_sha256,
            metrics_sha256=validated.metrics_sha256,
            training_interval=1,
            evaluation_interval=validated.evaluation_interval,
            checkpoint_interval=validated.checkpoint_interval,
            run_max_step=validated.last_step,
            predecessor_analysis_freeze_commit="a" * 40,
            training_code_commit="b" * 40,
            model_identity={"architecture": "synthetic"},
            training_config_identity={"synthetic": True},
            dataset_identity={"dataset": "synthetic"},
            split_identity={"split": "synthetic"},
            selected_dense_output_status="not-generated",
        )

    registry = build_registry(candidates, provenance)
    table = build_phase_selection_table(registry)

    return {
        "successor": successor,
        "predecessor": predecessor,
        "candidates": candidates,
        "provenance": provenance,
        "registry": registry,
        "table": table,
    }


def test_full_synthetic_pipeline_portable_and_physical(
    synthetic_stage3,
) -> None:
    registry = synthetic_stage3["registry"]

    structural = verify_registry_structure(registry)
    assert structural["structural_status"] == "PASS"
    assert structural["physical_status"] == "SKIPPED"
    assert structural["selected_cell_count"] == 13
    assert structural["unavailable_cell_count"] == 2

    physical = verify_registry_physical(
        registry,
        synthetic_stage3["successor"],
        synthetic_stage3["predecessor"],
    )
    assert physical["physical_status"] == "PASS"
    assert physical["source_hash_status"] == "PASS"
    assert physical["checkpoint_hash_status"] == "PASS"
    assert physical["selection_recomputation_status"] == "PASS"
    assert physical["rule_margin_recomputation_status"] == "PASS"


def test_registry_and_table_bytes_are_deterministic(
    synthetic_stage3,
) -> None:
    registry_a = synthetic_stage3["registry"]
    candidates = synthetic_stage3["candidates"]
    provenance = synthetic_stage3["provenance"]

    registry_b = build_registry(list(reversed(candidates)), provenance)

    json_a = serialize_registry_json(registry_a)
    json_b = serialize_registry_json(registry_b)
    assert json_a == json_b

    csv_a = serialize_phase_selection_table_csv(
        build_phase_selection_table(registry_a)
    )
    csv_b = serialize_phase_selection_table_csv(
        build_phase_selection_table(registry_b)
    )
    assert csv_a == csv_b

    assert b"/Users/" not in json_a
    assert b"generated_at" not in json_a
    assert b"timestamp" not in json_a
    assert b"local_root" not in json_a


def test_registry_builder_rejects_absolute_checkpoint_path(
    synthetic_stage3,
) -> None:
    candidate = synthetic_stage3["candidates"][1]
    provenance = synthetic_stage3["provenance"][1]

    broken = copy.deepcopy(candidate.transition_50.record)
    assert broken is not None
    broken["checkpoint_path"] = "/private/checkpoint.pt"

    from circuit_families.analysis.stage3_teacher_selection import (
        PhaseCandidate,
        TeacherCandidates,
    )

    broken_transition = PhaseCandidate(
        phase_label="50%",
        availability_status="selected",
        record=broken,
        unavailable_reason=None,
        transition_target=0.50,
        transition_absolute_distance=abs(
            float(broken["test_accuracy"]) - 0.50
        ),
    )
    broken_teacher = TeacherCandidates(
        seed=candidate.seed,
        run_id=candidate.run_id,
        validated_input=candidate.validated_input,
        pre=candidate.pre,
        transition_landmarks=candidate.transition_landmarks,
        transition_50=broken_transition,
        stable=candidate.stable,
    )

    with pytest.raises(Stage3RegistryError):
        build_registry(
            [
                synthetic_stage3["candidates"][0],
                broken_teacher,
                *synthetic_stage3["candidates"][2:],
            ],
            synthetic_stage3["provenance"],
        )


def test_structural_verifier_rejects_omission_and_wrong_order(
    synthetic_stage3,
) -> None:
    registry = synthetic_stage3["registry"]

    omitted = copy.deepcopy(registry)
    omitted["records"].pop()
    with pytest.raises(Stage3RegistryVerificationError):
        verify_registry_structure(omitted)

    reordered = copy.deepcopy(registry)
    reordered["records"][0], reordered["records"][1] = (
        reordered["records"][1],
        reordered["records"][0],
    )
    with pytest.raises(Stage3RegistryVerificationError):
        verify_registry_structure(reordered)


def test_resolution_linkage_preserves_historical_unresolved_state(
    synthetic_stage3,
) -> None:
    registry_bytes = serialize_registry_json(synthetic_stage3["registry"])
    table_bytes = serialize_phase_selection_table_csv(
        synthetic_stage3["table"]
    )

    resolution = {
        "resolution_schema_version": "1",
        "registry_namespace": (
            "circuit-families-distillation/stage3-teacher-registry"
        ),
        "resolution_status": "resolved",
        "resolves_decision_ids": ["UD-001", "UD-002"],
        "historical_stage2_register_mutated": False,
        "resolution_source": "Stage 3 sealed teacher registry",
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "phase_selection_table_sha256": hashlib.sha256(
            table_bytes
        ).hexdigest(),
    }

    report = verify_resolution_linkage(
        resolution,
        successor_root=synthetic_stage3["successor"],
        registry_sha256=resolution["registry_sha256"],
        table_sha256=resolution["phase_selection_table_sha256"],
    )

    assert report["resolution_linkage_status"] == "PASS"
    assert report["historical_stage2_status"] == "PASS"


def test_physical_verifier_rejects_tampered_checkpoint_hash(
    synthetic_stage3,
) -> None:
    registry = copy.deepcopy(synthetic_stage3["registry"])

    selected = next(
        record
        for record in registry["records"]
        if record["availability_status"] == "selected"
    )
    selected["checkpoint_sha256"] = "f" * 64

    with pytest.raises(
        Stage3RegistryVerificationError,
        match="physical frozen-rule recomputation",
    ):
        verify_registry_physical(
            registry,
            synthetic_stage3["successor"],
            synthetic_stage3["predecessor"],
        )
