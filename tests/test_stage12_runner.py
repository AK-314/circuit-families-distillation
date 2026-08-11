"""Tests for the integrated Stage 12 runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from circuit_families.interpretability.masks import (
    ComponentMask,
)

RUNNER_PATH = Path(
    "scripts/run_stage12_diversity.py"
)
STAGE9_ARCHIVE = Path(
    "results/archives/"
    "stage9-sparse-s1-27fffed087e6.tar.gz"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_stage12_diversity",
        RUNNER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load Stage 12 runner."
        )

    module = importlib.util.module_from_spec(
        spec
    )
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def search_config() -> dict:
    value = yaml.safe_load(
        Path("configs/search.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def test_stage9_reference_cell_is_exact() -> None:
    runner = load_runner()
    reference = (
        runner.load_stage9_reference_cell(
            STAGE9_ARCHIVE
        )
    )
    runner.validate_stage9_reference(
        reference
    )

    mask = ComponentMask.from_record(
        reference.final_mask_record
    )
    assert mask.retained_component_count == 146
    assert (
        reference.summary["search"][
            "exact_evaluations_used"
        ]
        == 6098
    )
    assert len(reference.final_mask_sha256) == 64
    assert (
        len(
            reference
            .accepted_removals_sha256
        )
        == 64
    )
    assert (
        len(
            reference
            .candidate_evaluations_sha256
        )
        == 64
    )


def test_stage12_run_id_is_deterministic() -> None:
    runner = load_runner()
    configuration = {
        "model_seed": 1,
        "checkpoint_step": 9050,
        "cutoffs": [0.5, 0.25, 0.75],
    }

    first = (
        runner.deterministic_stage12_run_id(
            configuration
        )
    )
    second = (
        runner.deterministic_stage12_run_id(
            dict(reversed(
                list(configuration.items())
            ))
        )
    )

    assert first == second
    assert first.startswith(
        "stage12-diversity-s1-"
    )
    assert len(first.rsplit("-", 1)[1]) == 12


def test_stage12_output_paths_match_config(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    paths = runner.build_output_paths(
        tmp_path,
        stage12_run_id="fixture-run",
        search_config=search_config(),
    )

    assert paths.raw_directory == (
        tmp_path
        / "results"
        / "raw"
        / "fixture-run"
    )
    assert paths.family_summary == (
        tmp_path
        / "results"
        / "tables"
        / "seed_1_stage12_family_summary.csv"
    )
    assert paths.archive == (
        tmp_path
        / "results"
        / "archives"
        / "fixture-run.tar.gz"
    )
    assert paths.manifest == (
        tmp_path
        / "manifests"
        / "stage12_diversity_fixture-run.json"
    )


def test_stage12_runner_refuses_overwrite(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    paths = runner.build_output_paths(
        tmp_path,
        stage12_run_id="fixture-run",
        search_config=search_config(),
    )
    paths.family_summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.family_summary.write_text(
        "stale",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="refuses to overwrite",
    ):
        runner.validate_absent_outputs(paths)
