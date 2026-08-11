"""Pipeline and provenance tests for the Stage 14 control."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from circuit_families.config import (
    load_config,
    load_training_config,
)
from circuit_families.training.data import load_training_data
from circuit_families.training.random_label import (
    FINAL_STEP,
    MODEL_SEED,
    RANDOM_LABEL_SEED,
    load_random_label_training_data,
    validate_stage14_training_settings,
)

REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPOSITORY / "scripts/run_stage14_random_label.py"


def load_runner():
    """Load the Stage 14 runner as a test module."""

    spec = importlib.util.spec_from_file_location(
        "stage14_runner_test_module",
        RUNNER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Stage 14 runner.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def test_random_label_loader_changes_only_targets() -> None:
    task_config = load_config(REPOSITORY / "configs/task.yaml")

    primary = load_training_data(
        archive_path=(REPOSITORY / "data/generated/modular_addition_m113.npz"),
        metadata_path=(REPOSITORY / "data/generated/modular_addition_m113.metadata.json"),
        manifest_path=(
            REPOSITORY / "manifests/dataset_modular-addition-dataset-s0-7ef9c73ff18f.json"
        ),
        task_config=task_config,
        device="cpu",
    )

    control = load_random_label_training_data(
        archive_path=(REPOSITORY / "data/generated/modular_addition_m113.npz"),
        metadata_path=(REPOSITORY / "data/generated/modular_addition_m113.metadata.json"),
        manifest_path=(
            REPOSITORY / "manifests/dataset_modular-addition-dataset-s0-7ef9c73ff18f.json"
        ),
        task_config_path=REPOSITORY / "configs/task.yaml",
        task_config=task_config,
        device="cpu",
    )

    assert torch.equal(control.train_inputs, primary.train_inputs)
    assert torch.equal(control.test_inputs, primary.test_inputs)
    assert control.train_count == primary.train_count == 3_830
    assert control.test_count == primary.test_count == 8_939
    assert control.total_count == primary.total_count == 12_769
    assert not torch.equal(
        control.train_targets,
        primary.train_targets,
    )
    assert not torch.equal(
        control.full_targets,
        primary.full_targets,
    )

    with np.load(
        REPOSITORY / "data/generated/modular_addition_m113.npz",
        allow_pickle=False,
    ) as archive:
        random_labels = torch.as_tensor(
            archive["random_labels"],
            dtype=torch.long,
        )
        train_indices = torch.as_tensor(
            archive["train_indices"],
            dtype=torch.long,
        )
        test_indices = torch.as_tensor(
            archive["test_indices"],
            dtype=torch.long,
        )

    assert torch.equal(control.full_targets, random_labels)
    assert torch.equal(
        control.train_targets,
        random_labels.index_select(0, train_indices),
    )
    assert torch.equal(
        control.test_targets,
        random_labels.index_select(0, test_indices),
    )


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("optimizer", "learning_rate", 0.002),
        ("optimizer", "weight_decay", 0.0),
        ("optimizer", "name", "Adam"),
        ("schedule", "warmup_steps", 1),
        ("training", "batch_mode", "mini_batch"),
        ("training", "precision", "float64"),
        ("training", "evaluation_interval", 100),
        ("training", "checkpoint_interval", 100),
        ("training", "evaluate_step_zero", False),
    ],
)
def test_training_policy_rejects_drift(
    section: str,
    name: str,
    value: object,
) -> None:
    config = load_training_config(REPOSITORY / "configs/training.yaml")
    changed = deepcopy(config)
    changed[section][name] = value

    with pytest.raises(ValueError):
        validate_stage14_training_settings(
            changed,
            model_seed=MODEL_SEED,
            random_label_seed=RANDOM_LABEL_SEED,
            final_step=FINAL_STEP,
        )


def test_training_policy_rejects_mps_priority() -> None:
    config = load_training_config(REPOSITORY / "configs/training.yaml")
    changed = deepcopy(config)
    changed["device"]["priority"] = ["cuda", "mps", "cpu"]

    with pytest.raises(ValueError, match="device priority|MPS"):
        validate_stage14_training_settings(
            changed,
            model_seed=MODEL_SEED,
            random_label_seed=RANDOM_LABEL_SEED,
            final_step=FINAL_STEP,
        )


def test_runner_has_no_overwrite_option() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "--overwrite" not in source
    assert "overwrite=False" in source


def test_forbidden_output_detection(tmp_path: Path) -> None:
    runner = load_runner()
    output_file = tmp_path / "results/tables/seed_0_stage14_random_label_sparse_search.csv"
    output_file.parent.mkdir(parents=True)
    output_file.write_text("forbidden\n", encoding="utf-8")

    matches = runner.find_forbidden_later_stage_outputs(tmp_path)

    assert matches == ("results/tables/seed_0_stage14_random_label_sparse_search.csv",)


def test_run_identity_is_deterministic() -> None:
    runner = load_runner()

    class Validation:
        archive_sha256 = "a" * 64
        manifest_sha256 = "b" * 64
        canonical_dataset_sha256 = "c" * 64
        split_sha256 = "d" * 64
        random_labels_sha256 = "e" * 64
        random_label_permutation_sha256 = "f" * 64

    identity_one = runner.build_run_identity(
        implementation_commit="1" * 40,
        dataset_validation=Validation(),
        task_config_sha256="2" * 64,
        model_config_sha256="3" * 64,
        training_config_sha256="4" * 64,
        main_checkpoint_manifest_sha256="5" * 64,
        stage8_manifest_sha256="6" * 64,
    )
    identity_two = runner.build_run_identity(
        implementation_commit="1" * 40,
        dataset_validation=Validation(),
        task_config_sha256="2" * 64,
        model_config_sha256="3" * 64,
        training_config_sha256="4" * 64,
        main_checkpoint_manifest_sha256="5" * 64,
        stage8_manifest_sha256="6" * 64,
    )

    assert identity_one == identity_two
    assert runner.identity_sha256(identity_one) == (runner.identity_sha256(identity_two))
    assert identity_one["model_seed"] == 0
    assert identity_one["random_label_seed"] == 1
    assert identity_one["random_labels"] is True
    assert identity_one["true_labels"] is False


def test_validate_only_does_not_create_outputs(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--validate-inputs-only",
        "--repository-root",
        str(REPOSITORY),
        "--output-root",
        str(tmp_path),
        "--device",
        "cpu",
    ]

    before = sorted(str(file_path.relative_to(tmp_path)) for file_path in tmp_path.rglob("*"))

    result = subprocess.run(
        command,
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )

    after = sorted(str(file_path.relative_to(tmp_path)) for file_path in tmp_path.rglob("*"))

    assert before == after == []
    assert "input_validation: passed" in result.stdout
    assert "validate_only_outputs_created: false" in result.stdout
    assert "stage15_started: false" in result.stdout


def test_seed_one_checkpoint_grid_is_exact() -> None:
    payload = json.loads(
        (REPOSITORY / "manifests/checkpoints_seed_1.json").read_text(encoding="utf-8")
    )

    observed = [
        payload["pre_checkpoint"]["training_step"],
        payload["formal_transition_checkpoints"]["10%"]["training_step"],
        payload["formal_transition_checkpoints"]["25%"]["training_step"],
        payload["formal_transition_checkpoints"]["50%"]["training_step"],
        payload["formal_transition_checkpoints"]["75%"]["training_step"],
        payload["formal_transition_checkpoints"]["90%"]["training_step"],
        payload["selected_stable_post_checkpoint"]["training_step"],
    ]

    assert observed == [
        200,
        3_400,
        7_450,
        8_150,
        8_500,
        8_650,
        9_050,
    ]


def load_selection_module():
    """Load the Stage 14 checkpoint-selection script."""

    script_path = REPOSITORY / "scripts/select_stage14_checkpoints.py"
    spec = importlib.util.spec_from_file_location(
        "stage14_selection_test_module",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the checkpoint-selection script.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def test_all_saved_checkpoints_are_reload_verified_by_runner() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))

    assignments = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            assignments.extend(target.id for target in node.targets if isinstance(target, ast.Name))

        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments.append(node.target.id)

    assert "CHECKPOINT_VERIFICATION_STEPS" not in assignments

    training_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_training"
    ]

    assert len(training_calls) == 1

    keywords = {
        keyword.arg: keyword.value
        for keyword in training_calls[0].keywords
        if keyword.arg is not None
    }

    value = keywords["checkpoint_verification_steps"]

    assert isinstance(value, ast.Constant)
    assert value.value is None


def test_flat_metric_record_normalisation() -> None:
    module = load_selection_module()

    row = module.normalise_metric_record(
        {
            "training_step": 50,
            "train_accuracy": 0.75,
            "test_accuracy": 0.01,
            "train_loss": 1.25,
            "test_loss": 4.75,
            "gradient_norm": 2.5,
        },
        stage14_run_id="stage14-test",
    )

    assert row == {
        "stage14_run_id": "stage14-test",
        "training_step": 50,
        "train_accuracy": 0.75,
        "test_accuracy": 0.01,
        "train_cross_entropy": 1.25,
        "test_cross_entropy": 4.75,
        "gradient_norm": 2.5,
    }


def test_metric_trajectory_requires_all_182_steps() -> None:
    module = load_selection_module()

    rows = [
        {
            "training_step": step,
        }
        for step in range(0, 9051, 50)
    ]

    module.validate_metric_trajectory(rows)

    with pytest.raises(ValueError):
        module.validate_metric_trajectory(rows[:-1])


def test_manifest_file_hash_maps_to_checkpoint_sha256() -> None:
    from circuit_families.analysis.random_label_control import (
        MAIN_MODEL_REFERENCE_CHECKPOINTS,
        build_exact_checkpoint_matches,
    )

    checkpoints = [
        {
            "training_step": step,
            "path": f"checkpoints/step_{step}.pt",
            "file_sha256": f"file-{step}",
            "model_state_sha256": f"model-{step}",
            "optimizer_state_sha256": f"optimizer-{step}",
            "reload_verified": True,
        }
        for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS
    ]

    metrics = [
        {
            "training_step": step,
            "train_accuracy": 0.5,
            "test_accuracy": 0.01,
            "train_cross_entropy": 1.0,
            "test_cross_entropy": 5.0,
        }
        for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS
    ]

    rows = build_exact_checkpoint_matches(
        checkpoint_records=checkpoints,
        metric_records=metrics,
    )

    assert rows[0]["checkpoint_sha256"] == "file-200"
    assert rows[-1]["checkpoint_sha256"] == "file-9050"


def load_masking_module():
    """Load the Stage 14 masking-validation script."""

    script_path = REPOSITORY / "scripts/validate_stage14_masking.py"
    spec = importlib.util.spec_from_file_location(
        "stage14_masking_test_module",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Stage 14 masking validator.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def test_stage14_masking_case_grid_is_exact() -> None:
    module = load_masking_module()

    assert module.validation_case_keys() == (
        (200, "all_retained"),
        (3_400, "all_retained"),
        (7_450, "all_retained"),
        (8_150, "all_retained"),
        (8_500, "all_retained"),
        (8_650, "all_retained"),
        (9_050, "all_retained"),
        (9_050, "all_ablated"),
        (9_050, "head_H0_ablated"),
        (9_050, "neuron_N0_ablated"),
        (9_050, "saved_arbitrary_reloaded"),
    )


def test_masking_validator_reuses_stage8_evaluator() -> None:
    source = (REPOSITORY / "scripts/validate_stage14_masking.py").read_text(encoding="utf-8")

    assert "stage8._evaluate_without_mutation" in source
    assert "stage8._validate_all_retained" in source
    assert "stage8._hook_counts" in source
    assert "def evaluate_component_mask" not in source
    assert "def masked_model_logits" not in source


def test_recorded_path_can_resolve_external_output_root(
    tmp_path: Path,
) -> None:
    module = load_masking_module()
    repository = tmp_path / "repository"
    output_root = tmp_path / "output"
    repository.mkdir()
    output_file = output_root / "manifests/example.json"
    output_file.parent.mkdir(parents=True)
    output_file.write_text("{}\n", encoding="utf-8")

    resolved = module.resolve_recorded_path(
        repository=repository,
        output_root=output_root,
        value="manifests/example.json",
    )

    assert resolved == output_file.resolve()


def test_masking_table_serialisation_is_deterministic(
    tmp_path: Path,
) -> None:
    module = load_masking_module()
    row = {column: None for column in module.TABLE_COLUMNS}
    row.update(
        {
            "stage14_run_id": "stage14-test",
            "checkpoint_step": 200,
            "mask_type": "all_retained",
            "primary_fidelity": 1.0,
            "validation_status": "passed",
        }
    )

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    module.write_validation_table(first, [row])
    module.write_validation_table(second, [row])

    assert first.read_bytes() == second.read_bytes()


def test_parameter_gradient_count_detects_gradients() -> None:
    module = load_masking_module()
    model = torch.nn.Linear(2, 1)

    assert module.parameter_gradient_count(model) == 0

    model(torch.ones(1, 2)).sum().backward()

    assert module.parameter_gradient_count(model) == 2


def test_hook_count_zero_validation_is_recursive() -> None:
    module = load_masking_module()

    assert module.hook_counts_are_zero(
        {
            "forward": 0,
            "backward": [0, 0],
            "nested": {"count": 0},
        }
    )
    assert not module.hook_counts_are_zero({"forward": 1, "backward": 0})


def test_output_vocabulary_excludes_equals_token() -> None:
    from circuit_families.config import load_model_config

    model_config = load_model_config(REPOSITORY / "configs/model.yaml")

    assert model_config["model"]["d_vocab_out"] == 113
    assert model_config["model"]["d_vocab"] == 114


def test_fidelity_evaluator_uses_final_position_logits() -> None:
    import inspect

    from circuit_families.interpretability.fidelity import (
        evaluate_component_mask,
    )

    source = inspect.getsource(evaluate_component_mask)

    assert "final_position_logits" in source
    assert "masked_sequence_logits" in source


def load_reproduction_module():
    """Load the Stage 14 reproduction auditor."""

    script_path = REPOSITORY / "scripts/verify_stage14_reproduction.py"
    spec = importlib.util.spec_from_file_location(
        "stage14_reproduction_test_module",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Stage 14 reproduction auditor.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def test_masking_table_paths_use_artifact_root() -> None:
    source = (REPOSITORY / "scripts/validate_stage14_masking.py").read_text(encoding="utf-8")

    assert "artifact_root: Path" in source
    assert source.count("artifact_root=output_root") == 6
    assert ("relative_or_absolute(\n            Path.cwd(),") not in source


def test_reproduction_roots_must_be_independent(
    tmp_path: Path,
) -> None:
    module = load_reproduction_module()
    primary = tmp_path / "primary"
    primary.mkdir()

    with pytest.raises(ValueError, match="must differ"):
        module.require_distinct_output_roots(
            primary,
            primary,
        )

    nested = primary / "nested"
    nested.mkdir()

    with pytest.raises(ValueError, match="inside"):
        module.require_distinct_output_roots(
            primary,
            nested,
        )


def test_training_manifest_normalisation_removes_only_timestamp() -> None:
    module = load_reproduction_module()
    manifest = {
        "timestamp_utc": "2026-07-22T12:00:00Z",
        "run_id": "stage14-test",
        "final_metrics": {"training_step": 9050},
    }

    normalised = module.normalise_training_manifest(manifest)

    assert normalised == {
        "run_id": "stage14-test",
        "final_metrics": {"training_step": 9050},
    }
    assert "timestamp_utc" in manifest


def test_selection_manifest_normalisation_removes_source_hash() -> None:
    module = load_reproduction_module()
    manifest = {
        "source_training_manifest": {
            "path": "manifests/training_example.json",
            "sha256": "a" * 64,
        },
        "classification": {"label": "memorisation_control"},
    }

    normalised = module.normalise_selection_manifest(manifest)

    assert normalised == {
        "source_training_manifest": {
            "path": "manifests/training_example.json",
        },
        "classification": {"label": "memorisation_control"},
    }
    assert manifest["source_training_manifest"]["sha256"] == "a" * 64


def test_checkpoint_hash_vectors_require_complete_trajectory() -> None:
    module = load_reproduction_module()

    checkpoints = [
        {
            "training_step": step,
            "file_sha256": f"file-{step}",
            "model_state_sha256": f"model-{step}",
            "optimizer_state_sha256": (f"optimizer-{step}"),
            "reload_verified": True,
        }
        for step in range(0, 9051, 50)
    ]

    vectors = module.checkpoint_hash_vectors({"checkpoints": checkpoints})

    assert len(vectors["training_steps"]) == 182
    assert vectors["training_steps"][0] == 0
    assert vectors["training_steps"][-1] == 9050
    assert vectors["file_sha256"][0] == "file-0"
    assert vectors["model_state_sha256"][-1] == "model-9050"

    changed = [dict(record) for record in checkpoints]
    changed[-1]["reload_verified"] = False

    with pytest.raises(ValueError, match="reload-verified"):
        module.checkpoint_hash_vectors({"checkpoints": changed})


def test_reproduction_json_writer_is_deterministic(
    tmp_path: Path,
) -> None:
    module = load_reproduction_module()
    payload = {
        "stage": 14,
        "status": "passed",
        "values": [3, 2, 1],
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    module.write_json(first, payload)
    module.write_json(second, payload)

    assert first.read_bytes() == second.read_bytes()


def test_byte_comparison_detects_difference(
    tmp_path: Path,
) -> None:
    module = load_reproduction_module()
    primary = tmp_path / "primary"
    reproduction = tmp_path / "reproduction"
    relative = Path("results/example.csv")

    (primary / relative).parent.mkdir(parents=True)
    (reproduction / relative).parent.mkdir(parents=True)

    (primary / relative).write_text(
        "same\n",
        encoding="utf-8",
    )
    (reproduction / relative).write_text(
        "same\n",
        encoding="utf-8",
    )

    result = module.compare_file_bytes(
        primary_root=primary,
        reproduction_root=reproduction,
        relative_file=relative,
    )

    assert result["byte_identical"] is True

    (reproduction / relative).write_text(
        "different\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="results/example.csv",
    ):
        module.compare_file_bytes(
            primary_root=primary,
            reproduction_root=reproduction,
            relative_file=relative,
        )


def test_validate_only_permits_existing_stage14_analysis_outputs(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--validate-inputs-only",
        "--repository-root",
        str(REPOSITORY),
        "--output-root",
        str(tmp_path),
        "--device",
        "cpu",
    ]

    before = tuple(
        sorted(
            str(file_path.relative_to(tmp_path))
            for file_path in tmp_path.rglob("*")
        )
    )

    result = subprocess.run(
        command,
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )

    after = tuple(
        sorted(
            str(file_path.relative_to(tmp_path))
            for file_path in tmp_path.rglob("*")
        )
    )

    assert before == after == ()
    assert "input_validation: passed" in result.stdout
    assert (
        "validate_only_outputs_created: false"
        in result.stdout
    )
    assert (
        "random_label_sparse_search_started: false"
        in result.stdout
    )
    assert "diversity_search_started: false" in result.stdout
    assert "stage15_started: false" in result.stdout
