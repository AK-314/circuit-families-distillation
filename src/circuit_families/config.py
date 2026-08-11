"""Load, validate, and identify experiment configurations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing or invalid."""


ConfigValidator = Callable[[Mapping[str, Any]], None]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping.")
    return value


def _require(
    mapping: Mapping[str, Any],
    key: str,
    expected: Any,
    section: str,
) -> None:
    if key not in mapping:
        raise ConfigError(f"Missing required field: {section}.{key}")

    actual = mapping[key]
    if actual != expected:
        raise ConfigError(
            f"{section}.{key} must be {expected!r}; received {actual!r}."
        )


def _require_non_empty_string(
    mapping: Mapping[str, Any],
    key: str,
    section: str,
) -> str:
    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section}.{key} must be a non-empty string.")

    return value


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {config_path}: {exc}") from exc

    return dict(_mapping(loaded, "configuration"))


def load_validated_config(
    path: str | Path,
    validator: ConfigValidator,
) -> dict[str, Any]:
    """Load a YAML mapping and validate it with the supplied validator."""

    config = _load_yaml_mapping(path)
    validator(config)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen task configuration.

    This name is retained for compatibility with the Stage 3-4 code.
    """

    return load_validated_config(path, validate_task_config)


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen model configuration."""

    return load_validated_config(path, validate_model_config)


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen training configuration."""

    return load_validated_config(path, validate_training_config)


def validate_task_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen modular-addition dataset configuration."""

    root = _mapping(config, "configuration")

    _require(root, "schema_version", 1, "configuration")
    _require(
        root,
        "experiment_type",
        "modular_addition_dataset",
        "configuration",
    )

    task = _mapping(root.get("task"), "task")
    _require(task, "name", "modular_addition", "task")
    _require(task, "modulus", 113, "task")
    _require(task, "pair_order", "lexicographic", "task")
    _require(task, "equals_token_id", 113, "task")
    _require(task, "expected_pair_count", 12_769, "task")

    split = _mapping(root.get("split"), "split")
    _require(split, "generator", "PCG64", "split")
    _require(split, "seed", 0, "split")
    _require(split, "primary_train_fraction", 0.30, "split")
    _require(split, "primary_train_count", 3_830, "split")
    _require(split, "primary_test_count", 8_939, "split")
    _require(
        split,
        "control_train_fractions",
        [0.05, 0.10, 0.15, 0.20, 0.25],
        "split",
    )

    random_labels = _mapping(root.get("random_labels"), "random_labels")
    _require(random_labels, "generator", "PCG64", "random_labels")
    _require(random_labels, "seed", 1, "random_labels")
    _require(
        random_labels,
        "method",
        "permute_complete_true_label_vector",
        "random_labels",
    )

    outputs = _mapping(root.get("outputs"), "outputs")
    for key in (
        "data_directory",
        "manifest_directory",
        "dataset_filename",
        "metadata_filename",
    ):
        _require_non_empty_string(outputs, key, "outputs")

    modulus = task["modulus"]
    pair_count = task["expected_pair_count"]
    train_count = split["primary_train_count"]
    test_count = split["primary_test_count"]

    if pair_count != modulus**2:
        raise ConfigError(
            "task.expected_pair_count must equal task.modulus squared."
        )

    if train_count + test_count != pair_count:
        raise ConfigError(
            "split.primary_train_count plus split.primary_test_count "
            "must equal task.expected_pair_count."
        )


def validate_model_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen one-layer TransformerLens model configuration."""

    root = _mapping(config, "configuration")

    _require(root, "schema_version", 1, "configuration")
    _require(
        root,
        "experiment_type",
        "modular_addition_transformer_model",
        "configuration",
    )

    model = _mapping(root.get("model"), "model")

    frozen_values = {
        "implementation": "transformer_lens",
        "n_layers": 1,
        "n_ctx": 3,
        "d_model": 128,
        "n_heads": 4,
        "d_head": 32,
        "d_mlp": 512,
        "act_fn": "relu",
        "positional_embedding_type": "standard",
        "attention_dir": "causal",
        "normalization_type": None,
        "dropout": 0.0,
        "tie_word_embeddings": False,
        "d_vocab": 114,
        "d_vocab_out": 113,
        "dtype": "float32",
        "init_weights": True,
        "init_mode": "gpt2",
        "default_prepend_bos": False,
    }

    for key, expected in frozen_values.items():
        _require(model, key, expected, "model")

    if model["n_heads"] * model["d_head"] != model["d_model"]:
        raise ConfigError(
            "model.n_heads multiplied by model.d_head must equal "
            "model.d_model."
        )

    if model["d_vocab_out"] >= model["d_vocab"]:
        raise ConfigError(
            "model.d_vocab_out must be smaller than model.d_vocab so "
            "the equals token cannot be predicted."
        )


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen full-batch AdamW training configuration."""

    root = _mapping(config, "configuration")

    _require(root, "schema_version", 1, "configuration")
    _require(
        root,
        "experiment_type",
        "modular_addition_training",
        "configuration",
    )

    optimizer = _mapping(root.get("optimizer"), "optimizer")
    optimizer_values = {
        "name": "AdamW",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.98,
        "epsilon": 1.0e-8,
        "weight_decay": 1.0,
    }
    for key, expected in optimizer_values.items():
        _require(optimizer, key, expected, "optimizer")

    schedule = _mapping(root.get("schedule"), "schedule")
    _require(schedule, "name", "constant", "schedule")
    _require(schedule, "warmup_steps", 0, "schedule")

    training = _mapping(root.get("training"), "training")
    training_values = {
        "max_steps": 40_000,
        "batch_mode": "full_training_set",
        "precision": "float32",
        "gradient_clipping": None,
        "evaluation_interval": 50,
        "checkpoint_interval": 50,
        "evaluate_step_zero": True,
        "checkpoint_step_zero": True,
        "checkpoint_final_step": True,
    }
    for key, expected in training_values.items():
        _require(training, key, expected, "training")

    device = _mapping(root.get("device"), "device")
    _require(device, "override", None, "device")
    _require(device, "priority", ["cuda", "cpu"], "device")

    validation = _mapping(root.get("validation"), "validation")
    for key in (
        "reload_metric_absolute_tolerance",
        "reload_metric_relative_tolerance",
    ):
        value = validation.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
        ):
            raise ConfigError(
                f"validation.{key} must be a non-negative number."
            )

    smoke = _mapping(root.get("smoke"), "smoke")
    _require(smoke, "steps", 5, "smoke")
    _require(smoke, "evaluation_interval", 1, "smoke")
    _require(smoke, "checkpoint_interval", 1, "smoke")

    outputs = _mapping(root.get("outputs"), "outputs")
    for key in (
        "checkpoint_directory",
        "results_directory",
        "manifest_directory",
    ):
        _require_non_empty_string(outputs, key, "outputs")


def canonical_mapping_json(config: Mapping[str, Any]) -> str:
    """Return any validated mapping in stable canonical JSON form."""

    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def mapping_hash(config: Mapping[str, Any]) -> str:
    """Return a SHA-256 hash for a configuration mapping."""

    canonical = canonical_mapping_json(config).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_config_json(config: Mapping[str, Any]) -> str:
    """Return the task configuration in stable canonical JSON form."""

    validate_task_config(config)
    return canonical_mapping_json(config)


def config_hash(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 hash of the validated task configuration."""

    validate_task_config(config)
    return mapping_hash(config)


def combined_config_hash(
    named_configs: Mapping[str, Mapping[str, Any]],
) -> str:
    """Hash multiple named configurations in a stable order."""

    if not named_configs:
        raise ConfigError("named_configs must not be empty.")

    canonical = {
        name: json.loads(canonical_mapping_json(config))
        for name, config in sorted(named_configs.items())
    }
    return mapping_hash(canonical)


def stable_run_id(
    experiment_type: str,
    seed: int,
    config: Mapping[str, Any],
) -> str:
    """Derive a stable run ID from experiment type, seed, and config."""

    return stable_run_id_from_hash(
        experiment_type,
        seed,
        config_hash(config),
    )


def stable_run_id_from_hash(
    experiment_type: str,
    seed: int,
    configuration_sha256: str,
) -> str:
    """Derive a stable run ID from an experiment type, seed, and hash."""

    if not isinstance(experiment_type, str) or not experiment_type.strip():
        raise ConfigError("experiment_type must be a non-empty string.")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("seed must be an integer.")

    if seed < 0:
        raise ConfigError("seed must be non-negative.")

    if (
        not isinstance(configuration_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", configuration_sha256)
    ):
        raise ConfigError(
            "configuration_sha256 must be a lowercase SHA-256 hex digest."
        )

    slug = re.sub(r"[^a-z0-9]+", "-", experiment_type.lower()).strip("-")
    if not slug:
        raise ConfigError(
            "experiment_type must contain an alphanumeric character."
        )

    return f"{slug}-s{seed}-{configuration_sha256[:12]}"
