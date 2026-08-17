from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from circuit_families.analysis.phase_detection import (
    find_pre_grokking_checkpoint,
    find_stable_post_sequence,
    select_transition_landmarks,
)
from circuit_families.analysis.stage3_teacher_inputs import (
    CanonicalTeacher,
    LinkedTeacher,
    Stage3InputError,
    validate_teacher_input,
)


def row(step: int, test: float, train: float = 1.0) -> dict[str, object]:
    return {
        "training_step": step,
        "train_accuracy": train,
        "test_accuracy": test,
        "train_loss": 0.01,
        "test_loss": 1.0,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_no_valid_pre_is_explicitly_missing() -> None:
    rows = [
        row(0, 0.01, 0.50),
        row(50, 0.051, 1.0),
        row(100, 0.10, 1.0),
    ]
    assert find_pre_grokking_checkpoint(rows) is None


def test_no_stable_sequence_is_rejected() -> None:
    rows = [
        row(0, 0.99),
        row(50, 0.99),
        row(100, 0.99),
        row(150, 0.99),
        row(200, 0.98),
    ]
    with pytest.raises(ValueError):
        find_stable_post_sequence(rows)


def test_no_interior_checkpoint_is_rejected() -> None:
    rows = [row(0, 0.01), row(100, 0.99)]
    with pytest.raises(ValueError):
        select_transition_landmarks(
            rows,
            pre_step=0,
            stable_post_step=100,
        )


def test_exact_transition_tie_chooses_earlier_step() -> None:
    rows = [
        row(0, 0.01),
        row(50, 0.49),
        row(100, 0.51),
        row(150, 0.99),
    ]
    result = select_transition_landmarks(
        rows,
        pre_step=0,
        stable_post_step=150,
    )
    assert result["50%"]["training_step"] == 50


def test_first_ten_percent_boundary_is_exclusive_for_pre() -> None:
    rows = [
        row(0, 0.01, 0.50),
        row(50, 0.05, 0.999),
        row(100, 0.10, 1.0),
        row(150, 0.01, 1.0),
    ]
    selected = find_pre_grokking_checkpoint(rows)
    assert selected is not None
    assert selected["training_step"] == 50


def test_five_checkpoint_stable_boundary_selects_fifth() -> None:
    rows = [
        row(0, 0.98),
        row(50, 0.99),
        row(100, 0.99),
        row(150, 0.99),
        row(200, 0.99),
        row(250, 0.99),
    ]
    sequence, selected = find_stable_post_sequence(rows)
    assert [x["training_step"] for x in sequence] == [50, 100, 150, 200, 250]
    assert selected["training_step"] == 250


def _build_physical_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LinkedTeacher, Path, Path]:
    import circuit_families.analysis.stage3_teacher_inputs as mod

    run_id = "stage18-main-training-s0-58b8c1235464"
    teacher = CanonicalTeacher(
        seed=0,
        run_id=run_id,
        manifest_path=f"manifests/training_{run_id}.json",
    )

    steps = (0, 50, 100)
    monkeypatch.setattr(mod, "EXPECTED_STEPS", steps)

    checkpoint_dir = tmp_path / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True)

    metrics_rows = []
    for step in steps:
        checkpoint = checkpoint_dir / f"step_{step:08d}.pt"
        checkpoint.write_bytes(f"checkpoint-{step}".encode())
        metrics_rows.append(
            {
                "run_id": run_id,
                "training_step": step,
                "train_accuracy": 1.0,
                "test_accuracy": 0.01,
                "train_loss": 0.01,
                "test_loss": 1.0,
                "checkpoint_path": checkpoint.relative_to(tmp_path).as_posix(),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )

    metrics_rel = f"results/raw/{run_id}/metrics.jsonl"
    metrics_path = tmp_path / metrics_rel
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "".join(json.dumps(x) + "\n" for x in metrics_rows),
        encoding="utf-8",
    )

    manifest = {
        "run_id": run_id,
        "model_seed": 0,
        "evaluation_interval": 50,
        "checkpoint_interval": 50,
        "output_paths": {
            "metrics_jsonl": metrics_rel,
            "checkpoint_directory": f"checkpoints/{run_id}",
        },
        "hashes": {
            "metrics_jsonl_sha256": sha256(metrics_path),
        },
    }

    manifest_path = tmp_path / teacher.manifest_path
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    linked = LinkedTeacher(
        teacher=teacher,
        manifest_sha256=sha256(manifest_path),
    )
    return linked, tmp_path, metrics_path


def test_duplicate_steps_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked, root, metrics_path = _build_physical_fixture(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in metrics_path.read_text().splitlines()]
    rows[2]["training_step"] = 50
    metrics_path.write_text(
        "".join(json.dumps(x) + "\n" for x in rows),
        encoding="utf-8",
    )

    manifest_path = root / linked.teacher.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"]["metrics_jsonl_sha256"] = sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    linked = LinkedTeacher(linked.teacher, sha256(manifest_path))

    with pytest.raises(Stage3InputError):
        validate_teacher_input(linked, root, verify_checkpoint_hashes=True)


def test_unsorted_steps_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked, root, metrics_path = _build_physical_fixture(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in metrics_path.read_text().splitlines()]
    rows[1], rows[2] = rows[2], rows[1]
    metrics_path.write_text(
        "".join(json.dumps(x) + "\n" for x in rows),
        encoding="utf-8",
    )

    manifest_path = root / linked.teacher.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"]["metrics_jsonl_sha256"] = sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    linked = LinkedTeacher(linked.teacher, sha256(manifest_path))

    with pytest.raises(Stage3InputError):
        validate_teacher_input(linked, root, verify_checkpoint_hashes=True)


def test_mismatched_checkpoint_filename_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked, root, metrics_path = _build_physical_fixture(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in metrics_path.read_text().splitlines()]
    rows[1]["checkpoint_path"] = rows[2]["checkpoint_path"]
    metrics_path.write_text(
        "".join(json.dumps(x) + "\n" for x in rows),
        encoding="utf-8",
    )

    manifest_path = root / linked.teacher.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"]["metrics_jsonl_sha256"] = sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    linked = LinkedTeacher(linked.teacher, sha256(manifest_path))

    with pytest.raises(Stage3InputError, match="filename/step mismatch"):
        validate_teacher_input(linked, root, verify_checkpoint_hashes=True)


def test_missing_checkpoint_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked, root, _ = _build_physical_fixture(tmp_path, monkeypatch)
    missing = (
        root
        / "checkpoints"
        / linked.teacher.run_id
        / "step_00000050.pt"
    )
    missing.unlink()

    with pytest.raises(Stage3InputError, match="missing checkpoint"):
        validate_teacher_input(linked, root, verify_checkpoint_hashes=True)


def test_changed_metrics_hash_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked, root, metrics_path = _build_physical_fixture(tmp_path, monkeypatch)
    metrics_path.write_text(
        metrics_path.read_text() + " ",
        encoding="utf-8",
    )

    with pytest.raises(Stage3InputError, match="metrics SHA-256 mismatch"):
        validate_teacher_input(linked, root, verify_checkpoint_hashes=True)


def test_wrong_run_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked, root, metrics_path = _build_physical_fixture(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in metrics_path.read_text().splitlines()]
    rows[1]["run_id"] = "wrong-run"
    metrics_path.write_text(
        "".join(json.dumps(x) + "\n" for x in rows),
        encoding="utf-8",
    )

    manifest_path = root / linked.teacher.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"]["metrics_jsonl_sha256"] = sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    linked = LinkedTeacher(linked.teacher, sha256(manifest_path))

    with pytest.raises(Stage3InputError, match="run_id"):
        validate_teacher_input(linked, root, verify_checkpoint_hashes=True)


def test_wrong_seed_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked, root, _ = _build_physical_fixture(tmp_path, monkeypatch)

    manifest_path = root / linked.teacher.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["model_seed"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    linked = LinkedTeacher(linked.teacher, sha256(manifest_path))

    with pytest.raises(Stage3InputError, match="model_seed"):
        validate_teacher_input(linked, root, verify_checkpoint_hashes=True)


def test_checkpoint_hash_mismatch_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked, root, _ = _build_physical_fixture(tmp_path, monkeypatch)
    checkpoint = (
        root
        / "checkpoints"
        / linked.teacher.run_id
        / "step_00000050.pt"
    )
    checkpoint.write_bytes(b"changed-checkpoint")

    with pytest.raises(Stage3InputError, match="checkpoint SHA-256 mismatch"):
        validate_teacher_input(linked, root, verify_checkpoint_hashes=True)
