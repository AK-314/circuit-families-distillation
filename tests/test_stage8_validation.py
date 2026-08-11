"""Integration tests for Stage 8 checkpoint and provenance validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from circuit_families.interpretability.fidelity import (
    load_checkpoint_evaluation_context,
)
from circuit_families.interpretability.masks import (
    ComponentMask,
    save_component_mask,
)
from circuit_families.training import file_sha256

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_mask.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_mask_script",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


RUN_ID = "modular-addition-training-s1-5f1bc9dee7ab"
CHECKPOINT_MANIFEST = Path("manifests/checkpoints_seed_1.json")


def _context(step: int):
    return load_checkpoint_evaluation_context(
        repository_root=".",
        run_id=RUN_ID,
        checkpoint_manifest_path=CHECKPOINT_MANIFEST,
        checkpoint_step=step,
        device_override="cpu",
    )


def test_context_reuses_validated_model_checkpoint_and_dataset() -> None:
    context = _context(9050)

    assert context.run_id == RUN_ID
    assert context.checkpoint_phase == "stable post-grokking"
    assert context.checkpoint_step == 9050
    assert context.checkpoint_sha256 == (
        "5b449db5ff9a62d5b621450c013bc259"
        "49499ed767e6db1723561a4e87ab8d70"
    )
    assert context.inputs.shape == (12_769, 3)
    assert context.targets.shape == (12_769,)
    assert context.example_ordering == "lexicographic"
    assert context.model.cfg.d_vocab_out == 113
    assert context.device.type == "cpu"


def test_full_model_context_is_checkpoint_specific() -> None:
    pre = _context(200)
    post = _context(9050)

    assert pre.checkpoint_sha256 != post.checkpoint_sha256
    assert pre.model_state_sha256 != post.model_state_sha256
    assert pre.checkpoint_step != post.checkpoint_step


def test_checkpoint_manifest_hash_mismatch_fails_clearly(
    tmp_path: Path,
) -> None:
    record = json.loads(
        CHECKPOINT_MANIFEST.read_text(encoding="utf-8")
    )
    record["selected_stable_post_checkpoint"][
        "checkpoint_sha256"
    ] = "0" * 64

    path = tmp_path / "checkpoints.json"
    path.write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="hash mismatch between manifests",
    ):
        load_checkpoint_evaluation_context(
            repository_root=".",
            run_id=RUN_ID,
            checkpoint_manifest_path=path,
            checkpoint_step=9050,
            device_override="cpu",
        )


def test_single_mask_record_is_deterministic_and_complete(
    tmp_path: Path,
) -> None:
    context = _context(9050)
    context = context.__class__(
        **{
            **context.__dict__,
            "inputs": context.inputs[:17],
            "targets": context.targets[:17],
        }
    )
    mask_path = save_component_mask(
        tmp_path / "all_retained.json",
        ComponentMask.all_retained(),
    )

    first = MODULE.build_evaluation_record(
        repository_root=".",
        context=context,
        mask=ComponentMask.all_retained(),
        mask_path=mask_path,
        batch_size=5,
    )
    second = MODULE.build_evaluation_record(
        repository_root=".",
        context=context,
        mask=ComponentMask.all_retained(),
        mask_path=mask_path,
        batch_size=5,
    )

    assert first == second
    assert first["metrics"]["primary_fidelity"] == 1.0
    assert first["metrics"]["evaluated_example_count"] == 17
    assert first["evaluated_sequence_position"] == -1
    assert first["output_classes"]["maximum"] == 112
    assert not first["output_classes"]["equals_token_eligible"]
    assert first["full_model_reference"] == {
        "method": "computed_live",
        "checkpoint_specific": True,
        "cached": False,
    }

    first_path = MODULE.write_evaluation_record(
        tmp_path / "first.json",
        first,
    )
    second_path = MODULE.write_evaluation_record(
        tmp_path / "second.json",
        second,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert file_sha256(first_path) == file_sha256(second_path)


VALIDATION_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_component_masking.py"
)
VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "validate_component_masking_script",
    VALIDATION_SCRIPT_PATH,
)
assert VALIDATION_SPEC is not None
assert VALIDATION_SPEC.loader is not None

VALIDATION_MODULE = importlib.util.module_from_spec(
    VALIDATION_SPEC
)
VALIDATION_SPEC.loader.exec_module(VALIDATION_MODULE)


def test_stage8_validation_case_plan_is_exact() -> None:
    assert VALIDATION_MODULE.build_validation_case_keys() == (
        (200, "all_retained"),
        (3400, "all_retained"),
        (7450, "all_retained"),
        (8150, "all_retained"),
        (8500, "all_retained"),
        (8650, "all_retained"),
        (9050, "all_retained"),
        (9050, "all_ablated"),
        (9050, "head_H0_ablated"),
        (9050, "neuron_N0_ablated"),
        (9050, "saved_arbitrary_reloaded"),
    )


def test_stage8_output_paths_are_deterministic(
    tmp_path: Path,
) -> None:
    outputs = VALIDATION_MODULE.stage8_output_paths(
        tmp_path,
        seed=1,
        combined_config_sha256=(
            "5f1bc9dee7ab55c53b19f7750e1e4c57"
            "1a0cb84981d5fccdba88158c2b1e36e2"
        ),
    )

    assert outputs["stage8_run_id"] == (
        "stage8-masking-s1-5f1bc9dee7ab"
    )
    assert Path(outputs["mask_directory"]) == (
        tmp_path
        / "results"
        / "raw"
        / "stage8-masking-s1-5f1bc9dee7ab"
        / "masks"
    )
    assert Path(outputs["result_table"]) == (
        tmp_path
        / "results"
        / "tables"
        / "seed_1_stage8_mask_validation.csv"
    )
    assert Path(outputs["manifest"]) == (
        tmp_path
        / "manifests"
        / "stage8_masking_s1-5f1bc9dee7ab.json"
    )


def test_validation_table_serialization_is_deterministic(
    tmp_path: Path,
) -> None:
    rows = []

    for step, mask_type in (
        (200, "all_retained"),
        (9050, "all_ablated"),
    ):
        row = {
            field: ""
            for field in VALIDATION_MODULE.VALIDATION_TABLE_FIELDS
        }
        row.update(
            {
                "run_id": RUN_ID,
                "checkpoint_phase": "test",
                "checkpoint_step": step,
                "checkpoint_sha256": "a" * 64,
                "model_state_sha256": "b" * 64,
                "mask_id": "component-mask-test",
                "mask_type": mask_type,
                "mask_path": "mask.json",
                "retained_attention_heads": 4,
                "retained_mlp_neurons": 512,
                "retained_components": 516,
                "retained_proportion": 1.0,
                "prediction_agreement_count": 17,
                "primary_fidelity": 1.0,
                "full_accuracy": 1.0,
                "masked_accuracy": 1.0,
                "accuracy_change": 0.0,
                "full_cross_entropy": 0.1,
                "masked_cross_entropy": 0.1,
                "cross_entropy_change": 0.0,
                "mean_kl_divergence": 0.0,
                "mean_jensen_shannon_divergence": 0.0,
                "maximum_absolute_logit_difference": 0.0,
                "evaluated_examples": 17,
                "evaluation_batch_size": 5,
                "validation_status": "passed",
            }
        )
        rows.append(row)

    first = VALIDATION_MODULE.write_validation_table(
        tmp_path / "first.csv",
        rows,
    )
    second = VALIDATION_MODULE.write_validation_table(
        tmp_path / "second.csv",
        rows,
    )

    assert first.read_bytes() == second.read_bytes()
    assert file_sha256(first) == file_sha256(second)

    header = first.read_text(
        encoding="utf-8"
    ).splitlines()[0]

    assert header.split(",") == list(
        VALIDATION_MODULE.VALIDATION_TABLE_FIELDS
    )
