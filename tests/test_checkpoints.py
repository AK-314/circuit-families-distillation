"""Tests for checkpoint integrity, restoration, and reevaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from circuit_families.config import (
    load_model_config,
    load_training_config,
)
from circuit_families.models.transformer import build_transformer
from circuit_families.training.checkpoints import (
    canonical_state_hash,
    checkpoint_filename,
    file_sha256,
    load_checkpoint_payload,
    reload_and_reevaluate,
    restore_checkpoint,
    save_checkpoint,
)
from circuit_families.training.metrics import evaluate_model
from circuit_families.training.trainer import (
    build_optimizer,
    train_full_batch_step,
)

DATASET_HASHES = {
    "dataset_sha256": "a" * 64,
    "split_sha256": "b" * 64,
}


def _model_and_optimizer(seed: int = 0):
    model_config = load_model_config("configs/model.yaml")
    training_config = load_training_config("configs/training.yaml")
    model = build_transformer(
        model_config,
        seed=seed,
        device="cpu",
    )
    optimizer = build_optimizer(model, training_config)
    return model, optimizer, model_config, training_config


def _train_and_test_data() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    train_inputs = torch.tensor(
        [
            [0, 0, 113],
            [1, 2, 113],
            [56, 57, 113],
            [112, 112, 113],
        ],
        dtype=torch.long,
    )
    train_targets = torch.tensor(
        [0, 3, 0, 111],
        dtype=torch.long,
    )
    test_inputs = torch.tensor(
        [
            [4, 9, 113],
            [17, 22, 113],
            [88, 103, 113],
        ],
        dtype=torch.long,
    )
    test_targets = torch.tensor(
        [13, 39, 78],
        dtype=torch.long,
    )
    return train_inputs, train_targets, test_inputs, test_targets


def _saved_checkpoint(tmp_path: Path):
    model, optimizer, model_config, training_config = (
        _model_and_optimizer()
    )
    (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = _train_and_test_data()

    train_full_batch_step(
        model,
        optimizer,
        train_inputs,
        train_targets,
    )

    metrics = {
        "train": evaluate_model(
            model,
            train_inputs,
            train_targets,
        ),
        "test": evaluate_model(
            model,
            test_inputs,
            test_targets,
        ),
    }

    saved = save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        step=1,
        model_config=model_config,
        training_config=training_config,
        model_seed=0,
        dataset_hashes=DATASET_HASHES,
        metrics=metrics,
        repository_root=".",
        device=torch.device("cpu"),
    )

    return (
        saved,
        model,
        optimizer,
        metrics,
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    )


def test_checkpoint_filename_is_exact_and_zero_padded() -> None:
    assert checkpoint_filename(0) == "step_00000000.pt"
    assert checkpoint_filename(50) == "step_00000050.pt"
    assert checkpoint_filename(40_000) == "step_00040000.pt"


def test_checkpoint_contains_required_reproducibility_fields(
    tmp_path: Path,
) -> None:
    saved, *_ = _saved_checkpoint(tmp_path)
    payload = load_checkpoint_payload(saved.path)

    assert saved.path.name == "step_00000001.pt"
    assert payload["training_step"] == 1
    assert payload["model_seed"] == 0
    assert payload["model_config"]["model"]["d_vocab_out"] == 113
    assert payload["training_config"]["optimizer"]["name"] == "AdamW"
    assert payload["dataset_hashes"] == DATASET_HASHES
    assert payload["git_commit"]
    assert payload["metrics"]["train"]
    assert payload["metrics"]["test"]
    assert payload["package_versions"]["torch"]
    assert payload["package_versions"]["transformer-lens"]
    assert payload["device"]["selected_device"] == "cpu"


def test_physical_and_canonical_hashes_are_stable(
    tmp_path: Path,
) -> None:
    saved, model, optimizer, *_ = _saved_checkpoint(tmp_path)

    assert file_sha256(saved.path) == saved.file_sha256
    assert file_sha256(saved.path) == file_sha256(saved.path)
    assert (
        canonical_state_hash(model.state_dict())
        == saved.model_state_sha256
    )
    assert (
        canonical_state_hash(optimizer.state_dict())
        == saved.optimizer_state_sha256
    )


def test_canonical_state_hash_changes_when_tensor_changes() -> None:
    model, _, _, _ = _model_and_optimizer()

    first_state = deepcopy(model.state_dict())
    second_state = deepcopy(model.state_dict())

    first_name = next(iter(second_state))
    second_state[first_name].view(-1)[0] += 1.0

    assert canonical_state_hash(first_state) != canonical_state_hash(
        second_state
    )


def test_restore_recovers_model_and_optimizer_state(
    tmp_path: Path,
) -> None:
    saved, original_model, original_optimizer, *_ = (
        _saved_checkpoint(tmp_path)
    )
    restored_model, restored_optimizer, _, _ = (
        _model_and_optimizer(seed=9)
    )

    resume_state = restore_checkpoint(
        saved.path,
        model=restored_model,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
    )

    assert resume_state.training_step == 1
    assert resume_state.model_seed == 0
    assert (
        canonical_state_hash(restored_model.state_dict())
        == canonical_state_hash(original_model.state_dict())
    )
    assert (
        canonical_state_hash(restored_optimizer.state_dict())
        == canonical_state_hash(original_optimizer.state_dict())
    )


def test_restored_optimizer_produces_identical_next_update(
    tmp_path: Path,
) -> None:
    (
        saved,
        original_model,
        original_optimizer,
        _,
        train_inputs,
        train_targets,
        _,
        _,
    ) = _saved_checkpoint(tmp_path)

    restored_model, restored_optimizer, _, _ = (
        _model_and_optimizer(seed=9)
    )
    restore_checkpoint(
        saved.path,
        model=restored_model,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
    )

    train_full_batch_step(
        original_model,
        original_optimizer,
        train_inputs,
        train_targets,
    )
    train_full_batch_step(
        restored_model,
        restored_optimizer,
        train_inputs,
        train_targets,
    )

    assert canonical_state_hash(
        original_model.state_dict()
    ) == canonical_state_hash(restored_model.state_dict())


def test_reload_and_reevaluate_matches_recorded_metrics(
    tmp_path: Path,
) -> None:
    (
        saved,
        _,
        _,
        metrics,
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = _saved_checkpoint(tmp_path)

    model, optimizer, _, training_config = _model_and_optimizer(seed=4)
    tolerance = training_config["validation"]

    result = reload_and_reevaluate(
        saved.path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        absolute_tolerance=tolerance[
            "reload_metric_absolute_tolerance"
        ],
        relative_tolerance=tolerance[
            "reload_metric_relative_tolerance"
        ],
    )

    assert result["train"] == metrics["train"]
    assert result["test"] == metrics["test"]


def test_reload_validation_rejects_incorrect_recorded_metric(
    tmp_path: Path,
) -> None:
    (
        saved,
        _,
        _,
        _,
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = _saved_checkpoint(tmp_path)

    payload = load_checkpoint_payload(saved.path)
    payload["metrics"]["train"]["cross_entropy"] += 1.0
    torch.save(payload, saved.path)

    model, optimizer, _, _ = _model_and_optimizer(seed=4)

    with pytest.raises(
        ValueError,
        match="train.cross_entropy",
    ):
        reload_and_reevaluate(
            saved.path,
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            absolute_tolerance=1.0e-6,
            relative_tolerance=1.0e-6,
        )
