"""Dataset-prefix and pipeline invariants for Stage 13."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from circuit_families.config import (
    load_config,
    load_model_config,
    load_training_config,
)
from circuit_families.training.data import TrainingData
from circuit_families.training.logging import read_jsonl
from circuit_families.training.no_generalisation import (
    CANDIDATE_FRACTIONS,
    FROZEN_SUBSET_RECORDS,
    control_array_name,
    frozen_example_count,
    load_and_validate_control_subsets,
    load_no_generalisation_training_data,
    subset_by_fraction,
)
from circuit_families.training.run import (
    build_execution_plan,
    run_training,
)

ARCHIVE_PATH = "data/generated/modular_addition_m113.npz"
METADATA_PATH = (
    "data/generated/modular_addition_m113.metadata.json"
)
MANIFEST_PATH = (
    "manifests/"
    "dataset_modular-addition-dataset-s0-7ef9c73ff18f.json"
)


def _load_subsets():
    return load_and_validate_control_subsets(
        archive_path=ARCHIVE_PATH,
        metadata_path=METADATA_PATH,
        manifest_path=MANIFEST_PATH,
        task_config=load_config("configs/task.yaml"),
    )


def test_exactly_five_candidate_fractions_are_recognised() -> None:
    assert CANDIDATE_FRACTIONS == (
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    )
    assert FROZEN_SUBSET_RECORDS == (
        (0.05, "05pct", 638),
        (0.10, "10pct", 1_276),
        (0.15, "15pct", 1_915),
        (0.20, "20pct", 2_553),
        (0.25, "25pct", 3_192),
    )


def test_control_array_names_are_exact() -> None:
    assert control_array_name(0.05) == (
        "control_train_indices_05pct"
    )
    assert control_array_name(0.25) == (
        "control_train_indices_25pct"
    )


def test_example_counts_are_exact_and_not_recomputed() -> None:
    assert [
        frozen_example_count(fraction)
        for fraction in CANDIDATE_FRACTIONS
    ] == [
        638,
        1_276,
        1_915,
        2_553,
        3_192,
    ]


def test_real_frozen_control_subsets_validate() -> None:
    subsets = _load_subsets()

    assert len(subsets) == 5
    assert [
        subset.fraction
        for subset in subsets
    ] == list(CANDIDATE_FRACTIONS)
    assert [
        subset.exact_example_count
        for subset in subsets
    ] == [
        638,
        1_276,
        1_915,
        2_553,
        3_192,
    ]

    assert all(
        len(subset.subset_sha256) == 64
        for subset in subsets
    )
    assert len(
        {
            subset.source_permutation_sha256
            for subset in subsets
        }
    ) == 1


def test_all_subsets_are_nested() -> None:
    subsets = _load_subsets()

    for smaller, larger in zip(
        subsets,
        subsets[1:],
        strict=False,
    ):
        assert np.array_equal(
            larger.indices[: smaller.indices.size],
            smaller.indices,
        )


def test_every_subset_lies_within_original_training_partition() -> None:
    subsets = _load_subsets()

    with np.load(
        ARCHIVE_PATH,
        allow_pickle=False,
    ) as archive:
        primary_train = archive["train_indices"]

    for subset in subsets:
        assert np.array_equal(
            subset.indices,
            primary_train[: subset.exact_example_count],
        )


def test_no_subset_overlaps_test_partition() -> None:
    subsets = _load_subsets()

    with np.load(
        ARCHIVE_PATH,
        allow_pickle=False,
    ) as archive:
        test_indices = archive["test_indices"]

    for subset in subsets:
        assert np.intersect1d(
            subset.indices,
            test_indices,
        ).size == 0


def test_subset_lookup_rejects_new_fraction() -> None:
    subsets = _load_subsets()

    with pytest.raises(ValueError, match="outside the frozen"):
        subset_by_fraction(subsets, 0.30)


def test_loader_uses_true_labels_and_unchanged_test_set() -> None:
    task_config = load_config("configs/task.yaml")
    subset = subset_by_fraction(_load_subsets(), 0.05)

    data = load_no_generalisation_training_data(
        archive_path=ARCHIVE_PATH,
        metadata_path=METADATA_PATH,
        manifest_path=MANIFEST_PATH,
        task_config=task_config,
        device="cpu",
        subset=subset,
    )

    with np.load(
        ARCHIVE_PATH,
        allow_pickle=False,
    ) as archive:
        expected_train_targets = torch.as_tensor(
            archive["true_labels"][subset.indices],
            dtype=torch.long,
        )
        expected_test_targets = torch.as_tensor(
            archive["true_labels"][archive["test_indices"]],
            dtype=torch.long,
        )
        expected_test_inputs = torch.as_tensor(
            archive["inputs"][archive["test_indices"]],
            dtype=torch.long,
        )

    assert data.train_count == 638
    assert data.test_count == 8_939
    assert torch.equal(
        data.train_targets,
        expected_train_targets,
    )
    assert torch.equal(
        data.test_targets,
        expected_test_targets,
    )
    assert torch.equal(
        data.test_inputs,
        expected_test_inputs,
    )
    assert data.training_subset is not None
    assert data.training_subset["true_labels"] is True
    assert data.training_subset["random_labels"] is False


def test_loader_rejects_forged_subset_record() -> None:
    task_config = load_config("configs/task.yaml")
    subset = subset_by_fraction(_load_subsets(), 0.05)
    forged = replace(
        subset,
        subset_sha256="0" * 64,
    )

    with pytest.raises(
        ValueError,
        match="validated frozen prefix",
    ):
        load_no_generalisation_training_data(
            archive_path=ARCHIVE_PATH,
            metadata_path=METADATA_PATH,
            manifest_path=MANIFEST_PATH,
            task_config=task_config,
            device="cpu",
            subset=forged,
        )


def _tiny_stage13_training_data(
    device: str | torch.device,
) -> TrainingData:
    selected_device = torch.device(device)

    return TrainingData(
        train_inputs=torch.tensor(
            [
                [0, 0, 113],
                [1, 2, 113],
                [56, 57, 113],
                [112, 112, 113],
            ],
            dtype=torch.long,
            device=selected_device,
        ),
        train_targets=torch.tensor(
            [0, 3, 0, 111],
            dtype=torch.long,
            device=selected_device,
        ),
        test_inputs=torch.tensor(
            [
                [4, 9, 113],
                [17, 22, 113],
                [88, 103, 113],
            ],
            dtype=torch.long,
            device=selected_device,
        ),
        test_targets=torch.tensor(
            [13, 39, 78],
            dtype=torch.long,
            device=selected_device,
        ),
        dataset_hashes={
            "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64,
            "control_subset_sha256": "c" * 64,
            "source_permutation_sha256": "d" * 64,
        },
        archive_path=Path("dummy/dataset.npz"),
        metadata_path=Path("dummy/metadata.json"),
        manifest_path=Path("dummy/manifest.json"),
        archive_sha256="e" * 64,
        metadata_sha256="f" * 64,
        total_count=7,
        train_count=4,
        test_count=3,
        training_subset={
            "fraction": 0.05,
            "array_name": "control_train_indices_05pct",
            "subset_identifier": "frozen_05pct_training_prefix",
            "exact_example_count": 4,
            "subset_sha256": "c" * 64,
            "source_permutation_sha256": "d" * 64,
            "nested_prefix": True,
            "true_labels": True,
            "random_labels": False,
        },
    )


def test_stage13_execution_plan_uses_exact_matched_horizon() -> None:
    plan = build_execution_plan(
        load_training_config("configs/training.yaml"),
        smoke=False,
        max_steps_override=9_050,
    )

    assert plan.mode == "full"
    assert plan.max_steps == 9_050
    assert plan.evaluation_interval == 50
    assert plan.checkpoint_interval == 50
    assert plan.evaluate_step_zero
    assert plan.checkpoint_step_zero
    assert plan.checkpoint_final_step


def test_execution_plan_rejects_invalid_step_override() -> None:
    training_config = load_training_config(
        "configs/training.yaml"
    )

    with pytest.raises(TypeError, match="must be an integer"):
        build_execution_plan(
            training_config,
            smoke=False,
            max_steps_override=True,
        )

    with pytest.raises(
        ValueError,
        match="no greater than",
    ):
        build_execution_plan(
            training_config,
            smoke=False,
            max_steps_override=40_001,
        )


def test_subset_training_run_records_identity_and_selected_checks(
    tmp_path: Path,
) -> None:
    run_identity = {
        "stage": 13,
        "candidate_fraction": 0.05,
        "exact_training_example_count": 4,
        "subset_sha256": "c" * 64,
        "model_seed": 0,
        "matched_horizon": 9_050,
        "implementation_commit": "1" * 40,
    }

    result = run_training(
        repository_root=".",
        task_config_path="configs/task.yaml",
        model_config_path="configs/model.yaml",
        training_config_path="configs/training.yaml",
        dataset_archive_path="unused.npz",
        dataset_metadata_path="unused.json",
        dataset_manifest_path="unused-manifest.json",
        model_seed=0,
        smoke=True,
        device_override="cpu",
        output_root=tmp_path,
        training_data=_tiny_stage13_training_data("cpu"),
        experiment_type_override=(
            "stage13_no_generalisation_training"
        ),
        run_identity=run_identity,
        checkpoint_verification_steps=(0, 5),
    )

    assert result.run_id.startswith(
        "stage13-no-generalisation-training-smoke-s0-"
    )

    records = read_jsonl(result.metrics_path)
    assert [record["training_step"] for record in records] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]

    manifest = json.loads(
        result.manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["run_identity"] == run_identity
    assert manifest["dataset"]["training_subset"][
        "subset_identifier"
    ] == "frozen_05pct_training_prefix"
    assert manifest["dataset"]["canonical_hashes"][
        "control_subset_sha256"
    ] == "c" * 64
    assert manifest["acceptance"] == {
        "checkpoint_reload_verification": "passed",
        "verified_checkpoint_count": 2,
        "verified_checkpoint_steps": [0, 5],
    }

    verified_steps = [
        checkpoint["training_step"]
        for checkpoint in manifest["checkpoints"]
        if checkpoint["reload_verified"]
    ]
    assert verified_steps == [0, 5]


def test_run_identity_enters_deterministic_run_id(
    tmp_path: Path,
) -> None:
    base_identity = {
        "stage": 13,
        "candidate_fraction": 0.05,
        "subset_sha256": "c" * 64,
        "implementation_commit": "1" * 40,
    }
    changed_identity = {
        **base_identity,
        "candidate_fraction": 0.10,
        "subset_sha256": "d" * 64,
    }

    common = {
        "repository_root": ".",
        "task_config_path": "configs/task.yaml",
        "model_config_path": "configs/model.yaml",
        "training_config_path": "configs/training.yaml",
        "dataset_archive_path": "unused.npz",
        "dataset_metadata_path": "unused.json",
        "dataset_manifest_path": "unused-manifest.json",
        "model_seed": 0,
        "smoke": True,
        "device_override": "cpu",
        "training_data": _tiny_stage13_training_data("cpu"),
        "experiment_type_override": (
            "stage13_no_generalisation_training"
        ),
        "checkpoint_verification_steps": (0, 5),
    }

    first = run_training(
        **common,
        output_root=tmp_path / "first",
        run_identity=base_identity,
    )
    repeated = run_training(
        **common,
        output_root=tmp_path / "repeated",
        run_identity=base_identity,
    )
    changed = run_training(
        **common,
        output_root=tmp_path / "changed",
        run_identity=changed_identity,
    )

    assert first.run_id == repeated.run_id
    assert first.combined_config_sha256 == (
        repeated.combined_config_sha256
    )
    assert changed.run_id != first.run_id
    assert changed.combined_config_sha256 != (
        first.combined_config_sha256
    )


def test_stage13_execution_order_is_descending() -> None:
    from circuit_families.training.no_generalisation import (
        STAGE13_EXECUTION_ORDER,
        validate_requested_fractions,
    )

    assert validate_requested_fractions(
        (0.05, 0.10, 0.15, 0.20, 0.25)
    ) == (
        0.25,
        0.20,
        0.15,
        0.10,
        0.05,
    )
    assert STAGE13_EXECUTION_ORDER == (
        0.25,
        0.20,
        0.15,
        0.10,
        0.05,
    )


@pytest.mark.parametrize(
    "fractions",
    [
        (0.05, 0.10, 0.15, 0.20),
        (0.05, 0.10, 0.15, 0.20, 0.30),
        (0.05, 0.10, 0.15, 0.20, 0.20),
    ],
)
def test_stage13_rejects_changed_candidate_grid(
    fractions: tuple[float, ...],
) -> None:
    from circuit_families.training.no_generalisation import (
        validate_requested_fractions,
    )

    with pytest.raises(ValueError):
        validate_requested_fractions(fractions)


def test_stage13_training_settings_match_primary_config() -> None:
    from circuit_families.training.no_generalisation import (
        STAGE13_CHECKPOINT_VALIDATION_STEPS,
        validate_stage13_training_settings,
    )

    training_config = load_training_config(
        "configs/training.yaml"
    )

    validate_stage13_training_settings(
        training_config,
        model_seed=0,
        final_step=9_050,
    )

    assert STAGE13_CHECKPOINT_VALIDATION_STEPS == (
        0,
        200,
        3_400,
        4_050,
        5_000,
        7_450,
        8_150,
        8_500,
        8_650,
        9_050,
    )


def test_stage13_rejects_weight_decay_change() -> None:
    from copy import deepcopy

    from circuit_families.training.no_generalisation import (
        validate_stage13_training_settings,
    )

    training_config = deepcopy(
        load_training_config("configs/training.yaml")
    )
    training_config["optimizer"]["weight_decay"] = 0.0

    with pytest.raises(
        ValueError,
        match="weight_decay",
    ):
        validate_stage13_training_settings(
            training_config,
            model_seed=0,
            final_step=9_050,
        )


def test_stage13_rejects_changed_model_seed_or_horizon() -> None:
    from circuit_families.training.no_generalisation import (
        validate_stage13_training_settings,
    )

    training_config = load_training_config(
        "configs/training.yaml"
    )

    with pytest.raises(ValueError, match="seed must equal 0"):
        validate_stage13_training_settings(
            training_config,
            model_seed=1,
            final_step=9_050,
        )

    with pytest.raises(ValueError, match="matched horizon"):
        validate_stage13_training_settings(
            training_config,
            model_seed=0,
            final_step=9_000,
        )


def test_stage13_run_id_is_deterministic_and_commit_sensitive() -> None:
    from circuit_families.training.no_generalisation import (
        deterministic_stage13_run_id,
    )

    configuration = {
        "source_dataset_manifest_sha256": "a" * 64,
        "candidate_subset_sha256s": ["b" * 64, "c" * 64],
        "model_config_sha256": "d" * 64,
        "training_config_sha256": "e" * 64,
        "model_seed": 0,
        "matched_horizon": 9_050,
        "implementation_commit": "1" * 40,
    }

    first = deterministic_stage13_run_id(configuration)
    repeated = deterministic_stage13_run_id(configuration)
    changed = deterministic_stage13_run_id(
        {
            **configuration,
            "implementation_commit": "2" * 40,
        }
    )

    assert first == repeated
    assert first.startswith(
        "stage13-no-generalisation-s0-"
    )
    assert changed != first


def _load_stage13_runner_module():
    import importlib.util

    path = Path(
        "scripts/run_stage13_no_generalisation_pilots.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_stage13_no_generalisation_pilots",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Stage 13 runner module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_training_run_ids_are_deterministic_and_unique() -> None:
    module = _load_stage13_runner_module()
    task_config = load_config("configs/task.yaml")
    model_config = load_model_config("configs/model.yaml")
    training_config = load_training_config(
        "configs/training.yaml"
    )
    subsets = _load_subsets()

    configuration = {
        "source_dataset_manifest": {
            "path": MANIFEST_PATH,
            "sha256": "a" * 64,
        },
        "model_config": {
            "sha256": "b" * 64,
        },
        "training_config": {
            "sha256": "c" * 64,
        },
        "model_seed": 0,
        "matched_horizon": 9_050,
        "implementation_commit": "1" * 40,
    }
    stage13_run_id = "stage13-no-generalisation-s0-" + "2" * 12

    run_ids = []

    for subset in subsets:
        identity = module.candidate_run_identity(
            stage13_run_id=stage13_run_id,
            subset=subset,
            configuration=configuration,
        )
        first, first_hash = module.candidate_training_run_id(
            task_config=task_config,
            model_config=model_config,
            training_config=training_config,
            run_identity=identity,
        )
        repeated, repeated_hash = (
            module.candidate_training_run_id(
                task_config=task_config,
                model_config=model_config,
                training_config=training_config,
                run_identity=identity,
            )
        )

        assert first == repeated
        assert first_hash == repeated_hash
        run_ids.append(first)

    assert len(set(run_ids)) == 5


def test_candidate_output_preflight_rejects_existing_output(
    tmp_path: Path,
) -> None:
    module = _load_stage13_runner_module()
    training_config = load_training_config(
        "configs/training.yaml"
    )
    run_id = "stage13-no-generalisation-training-s0-" + "a" * 12

    paths = module.expected_output_paths(
        output_root=tmp_path,
        training_config=training_config,
        run_id=run_id,
    )
    paths[1].mkdir(parents=True)

    with pytest.raises(
        FileExistsError,
        match="refuses to overwrite",
    ):
        module.validate_absent_candidate_outputs(
            output_root=tmp_path,
            training_config=training_config,
            run_ids=(run_id,),
        )


def _synthetic_stage13_metric_rows():
    from circuit_families.plotting.no_generalisation import (
        STAGE13_METRICS_COLUMNS,
    )

    rows = []

    for fraction, _, count in FROZEN_SUBSET_RECORDS:
        for step in range(0, 9_051, 50):
            row = {
                "fraction": fraction,
                "exact_training_example_count": count,
                "subset_identifier": (
                    f"frozen_{int(fraction * 100):02d}pct_"
                    "training_prefix"
                ),
                "subset_sha256": (
                    f"{int(fraction * 100):02d}" * 32
                ),
                "run_id": (
                    "stage13-no-generalisation-training-s0-"
                    f"{int(fraction * 100):012d}"
                ),
                "training_git_commit": "1" * 40,
                "device": "cpu",
                "training_step": step,
                "learning_rate": 0.001,
                "weight_norm": 30.0,
                "gradient_norm": (
                    None
                    if step == 0
                    else 1.0
                ),
                "train_loss": 5.0,
                "test_loss": 5.0,
                "train_accuracy": (
                    0.998
                    if step < 5_000
                    else 0.999
                ),
                "test_accuracy": 0.05,
                "checkpoint_path": (
                    f"checkpoints/{fraction:.2f}/"
                    f"step_{step:08d}.pt"
                ),
                "checkpoint_sha256": "a" * 64,
                "model_state_sha256": "b" * 64,
                "optimizer_state_sha256": "c" * 64,
            }

            assert set(row) == set(STAGE13_METRICS_COLUMNS)
            rows.append(row)

    return rows


def test_stage13_metrics_csv_round_trip_is_deterministic(
    tmp_path: Path,
) -> None:
    from circuit_families.plotting.no_generalisation import (
        read_stage13_metrics_csv,
        validate_stage13_metrics_rows,
        write_stage13_metrics_csv,
    )

    rows = _synthetic_stage13_metric_rows()
    first = write_stage13_metrics_csv(
        rows,
        tmp_path / "first.csv",
    )
    second = write_stage13_metrics_csv(
        rows,
        tmp_path / "second.csv",
    )

    assert first.read_bytes() == second.read_bytes()

    loaded = read_stage13_metrics_csv(first)
    validate_stage13_metrics_rows(loaded)

    assert len(loaded) == 5 * 182
    assert loaded[0]["training_step"] == 0
    assert loaded[-1]["training_step"] == 9_050


def test_stage13_metrics_validation_rejects_missing_checkpoint() -> None:
    from circuit_families.plotting.no_generalisation import (
        validate_stage13_metrics_rows,
    )

    rows = _synthetic_stage13_metric_rows()
    rows.pop(100)

    with pytest.raises(
        ValueError,
        match="complete saved-evaluation grid",
    ):
        validate_stage13_metrics_rows(rows)


def test_stage13_plot_regenerates_from_saved_metrics(
    tmp_path: Path,
) -> None:
    from circuit_families.plotting.no_generalisation import (
        plot_stage13_training_curves,
        read_stage13_metrics_csv,
        stage13_figure_hashes,
        write_stage13_figure_caption,
        write_stage13_metrics_csv,
    )

    metrics_path = write_stage13_metrics_csv(
        _synthetic_stage13_metric_rows(),
        tmp_path / "metrics.csv",
    )
    rows = read_stage13_metrics_csv(metrics_path)

    png_path, pdf_path = plot_stage13_training_curves(
        rows,
        selected_fraction=0.20,
        png_path=tmp_path / "curves.png",
        pdf_path=tmp_path / "curves.pdf",
    )
    caption_path = write_stage13_figure_caption(
        tmp_path / "caption.txt",
        selected_fraction=0.20,
    )

    hashes = stage13_figure_hashes(
        png_path=png_path,
        pdf_path=pdf_path,
        caption_path=caption_path,
    )

    assert png_path.is_file()
    assert pdf_path.is_file()
    assert caption_path.is_file()
    assert "20%" in caption_path.read_text(encoding="utf-8")
    assert all(
        len(value) == 64
        for value in hashes.values()
    )


def _load_stage13_selector_module():
    import importlib.util

    path = Path(
        "scripts/select_no_generalisation_control.py"
    )
    spec = importlib.util.spec_from_file_location(
        "select_no_generalisation_control",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load Stage 13 selector module."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_metric_selection_chooses_largest_candidate() -> None:
    module = _load_stage13_selector_module()

    qualifications, selected = module.evaluate_saved_metrics(
        _synthetic_stage13_metric_rows()
    )

    assert len(qualifications) == 5
    assert selected == 0.25
    assert qualifications[0].candidate_fraction == 0.25
    assert all(
        qualification.overall_qualification
        for qualification in qualifications
    )


def test_stage13_selection_csv_is_deterministic(
    tmp_path: Path,
) -> None:
    module = _load_stage13_selector_module()
    qualifications, selected = module.evaluate_saved_metrics(
        _synthetic_stage13_metric_rows()
    )

    first = module.write_selection_csv(
        qualifications,
        selected_fraction=selected,
        path=tmp_path / "first.csv",
    )
    second = module.write_selection_csv(
        qualifications,
        selected_fraction=selected,
        path=tmp_path / "second.csv",
    )

    assert first.read_bytes() == second.read_bytes()
    text = first.read_text(encoding="utf-8")
    assert "selected_control" in text
    assert "0.25" in text
    assert "true" in text


def test_selector_rejects_circuit_family_metric_field() -> None:
    module = _load_stage13_selector_module()
    record = {
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_sha256": "a" * 64,
        "gradient_norm": None,
        "learning_rate": 0.001,
        "mode": "full",
        "model_state_sha256": "b" * 64,
        "optimizer_state_sha256": "c" * 64,
        "run_id": "run",
        "schema_version": 1,
        "test_accuracy": 0.05,
        "test_loss": 5.0,
        "train_accuracy": 1.0,
        "train_loss": 0.1,
        "training_step": 0,
        "weight_norm": 30.0,
        "circuit_size": 12,
    }

    with pytest.raises(
        ValueError,
        match="curve-only schema",
    ):
        module.validate_metric_record_fields(record)


def test_selector_refuses_existing_artifact(
    tmp_path: Path,
) -> None:
    module = _load_stage13_selector_module()
    paths = module.output_paths(
        tmp_path,
        "stage13-no-generalisation-s0-test",
    )
    paths["selection_table"].parent.mkdir(
        parents=True,
    )
    paths["selection_table"].write_text(
        "existing\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="refuses to overwrite",
    ):
        module.validate_absent_outputs(paths)
