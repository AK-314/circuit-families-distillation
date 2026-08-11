"""Tests for the frozen Stage 12 search configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from circuit_families.config import mapping_hash
from circuit_families.interpretability.diversity_forced_search import (
    FAMILY_TARGET,
    MAX_RESTARTS_PER_ALTERNATIVE,
    NUMERICALLY_INDISTINGUISHABLE_TOLERANCE,
    PER_CELL_EXACT_EVALUATION_BUDGET,
    PER_REQUESTED_CIRCUIT_EXACT_EVALUATION_BUDGET,
    PRIMARY_REUSE_COEFFICIENT,
)
from circuit_families.interpretability.masks import (
    SEARCHABLE_COMPONENT_COUNT,
)
from circuit_families.interpretability.sparse_search import (
    CANDIDATE_BATCH_SIZE,
    MEANINGFULLY_SPARSE_MAX_COMPONENTS,
)

SEARCH_CONFIG_PATH = Path("configs/search.yaml")


def load_search_config() -> dict:
    value = yaml.safe_load(
        SEARCH_CONFIG_PATH.read_text(encoding="utf-8")
    )

    assert isinstance(value, dict)
    return value


def test_search_config_matches_frozen_scientific_values() -> None:
    config = load_search_config()

    assert config["source"]["checkpoint_step"] == 9050
    assert config["source"]["model_seed"] == 1

    assert config["fidelity"]["primary_threshold"] == 0.99
    assert config["fidelity"]["threshold_numerator"] == 99
    assert config["fidelity"]["threshold_denominator"] == 100
    assert (
        config["fidelity"]["primary_sparsity_max_components"]
        == MEANINGFULLY_SPARSE_MAX_COMPONENTS
        == 258
    )

    assert (
        config["components"]["total_count"]
        == SEARCHABLE_COMPONENT_COUNT
        == 516
    )
    assert (
        config["components"]["candidate_batch_size"]
        == CANDIDATE_BATCH_SIZE
        == 16
    )

    assert config["distinctness"]["primary_cutoff"] == 0.5
    assert config["distinctness"]["sensitivity_grid"] == [
        0.25,
        0.5,
        0.75,
    ]
    assert (
        config["distinctness"]["definitive_execution_order"]
        == [0.5, 0.25, 0.75]
    )

    assert (
        config["ranking"]["reuse_coefficient"]
        == PRIMARY_REUSE_COEFFICIENT
        == 0.5
    )


def test_search_config_matches_implementation_limits() -> None:
    config = load_search_config()

    assert (
        config["ranking"][
            "numerically_indistinguishable_tolerance"
        ]
        == NUMERICALLY_INDISTINGUISHABLE_TOLERANCE
        == 1.0e-12
    )

    assert (
        config["restarts"][
            "maximum_per_requested_alternative"
        ]
        == MAX_RESTARTS_PER_ALTERNATIVE
        == 5
    )
    assert (
        config["budgets"][
            "per_requested_circuit_exact_evaluations"
        ]
        == PER_REQUESTED_CIRCUIT_EXACT_EVALUATION_BUDGET
        == 10_000
    )
    assert (
        config["budgets"]["per_cell_exact_evaluations"]
        == PER_CELL_EXACT_EVALUATION_BUDGET
        == 50_000
    )
    assert (
        config["budgets"]["family_target"]
        == FAMILY_TARGET
        == 10
    )

    assert (
        config["seed_derivation"]["digest_algorithm"]
        == "sha256"
    )
    assert (
        config["seed_derivation"]["integer_seed_derivation"]
        == "first_4_digest_bytes_interpreted_as_unsigned_"
        "big_endian_integer"
    )
    assert (
        config["seed_derivation"]["bit_generator"]
        == "numpy.random.PCG64"
    )

    assert (
        config["execution"]["definitive_checkpoint_steps"]
        == [9050]
    )
    assert (
        config["execution"]["pre_grokking_family_search"]
        == "prohibited"
    )
    assert (
        config["execution"]["transition_family_search"]
        == "prohibited"
    )
    assert config["execution"]["stage13_started"] is False

    assert (
        config["archive"][
            "runtime_excluded_from_deterministic_scientific_hashes"
        ]
        is True
    )


def test_search_config_hash_is_deterministic() -> None:
    first = load_search_config()
    second = load_search_config()

    first_hash = mapping_hash(first)
    second_hash = mapping_hash(second)

    assert first == second
    assert first_hash == second_hash
    assert len(first_hash) == 64
