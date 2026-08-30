"""Focused/adversarial tests for the Stage 12-P1 task protocol."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from circuit_families.stage12p1.tasks import (
    TASK_CONFIG_SCHEMA_VERSION,
    ModularAdditionImplementation,
    ModularMultiplicationImplementation,
    ModularPolynomialImplementation,
    TaskImplementationRegistry,
    TaskProtocolError,
    TaskRegistry,
    build_task_record,
    validate_task_record,
)


def _base_config(
    *,
    task_id: str,
    implementation: str,
    implementation_version: str,
    parameters: Mapping[str, Any],
    modulus: int = 7,
    domains: list[list[int]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": TASK_CONFIG_SCHEMA_VERSION,
        "task_id": task_id,
        "implementation": implementation,
        "implementation_version": implementation_version,
        "modulus": modulus,
        "input_domains": (
            [list(range(modulus)), list(range(modulus))]
            if domains is None
            else domains
        ),
        "parameters": copy.deepcopy(dict(parameters)),
        "split_identity": {
            "kind": "technical-fixture-split",
            "version": "v1",
            "train_indices": [0, 2, 4],
            "test_indices": [1, 3, 5],
        },
        "architecture_compatibility": {
            "input_arity": 2,
            "output_class_count": modulus,
            "classification": "technical-fixture-only",
        },
        "scientific_data": False,
        "production_eligible": False,
    }


def _addition() -> dict[str, Any]:
    implementation = ModularAdditionImplementation()
    return _base_config(
        task_id="technical-fixture-addition-m7",
        implementation=implementation.name,
        implementation_version=implementation.version,
        parameters={},
    )


def _multiplication() -> dict[str, Any]:
    implementation = ModularMultiplicationImplementation()
    return _base_config(
        task_id="technical-fixture-multiplication-m7",
        implementation=implementation.name,
        implementation_version=implementation.version,
        parameters={},
    )


def _polynomial() -> dict[str, Any]:
    implementation = ModularPolynomialImplementation()
    return _base_config(
        task_id="technical-fixture-polynomial-m7",
        implementation=implementation.name,
        implementation_version=implementation.version,
        parameters={
            "terms": [
                {"coefficient": 3, "exponents": [0, 0]},
                {"coefficient": 1, "exponents": [0, 2]},
                {"coefficient": 2, "exponents": [1, 0]},
                {"coefficient": 4, "exponents": [1, 1]},
            ]
        },
    )


def test_same_generic_consumer_builds_all_three_technical_fixtures() -> None:
    implementation_registry = TaskImplementationRegistry()

    records = [
        implementation_registry.build(config)
        for config in (_addition(), _multiplication(), _polynomial())
    ]

    assert [record["example_order"]["example_count"] for record in records] == [
        49,
        49,
        49,
    ]
    assert all(record["scientific_data"] is False for record in records)
    assert all(record["production_eligible"] is False for record in records)
    assert len(
        {
            record["hashes"]["target_sha256"]
            for record in records
        }
    ) == 3

    registry = TaskRegistry(
        records,
        implementation_registry=implementation_registry,
    )
    assert registry.to_mapping()["task_count"] == 3
    assert registry.to_mapping()["ordering"] == (
        "task-id-lexicographic-no-priority/v1"
    )


def test_build_is_deterministic_and_round_trip_validates() -> None:
    first = build_task_record(_polynomial())
    second = build_task_record(copy.deepcopy(_polynomial()))

    assert first == second
    assert validate_task_record(first) == first


def test_target_change_changes_target_dataset_config_and_identity_hashes() -> None:
    original = build_task_record(_polynomial())
    changed_config = _polynomial()
    changed_config["parameters"]["terms"][-1]["coefficient"] = 5
    changed = build_task_record(changed_config)

    for name in (
        "task_config_sha256",
        "target_sha256",
        "dataset_sha256",
        "task_identity_sha256",
    ):
        assert original["hashes"][name] != changed["hashes"][name]

    assert (
        original["hashes"]["domain_sha256"]
        == changed["hashes"]["domain_sha256"]
    )
    assert (
        original["hashes"]["example_order_sha256"]
        == changed["hashes"]["example_order_sha256"]
    )


def test_domain_change_changes_domain_order_dataset_and_identity_hashes() -> None:
    original = build_task_record(_addition())
    changed_config = _addition()
    changed_config["input_domains"][1] = [0, 1, 2, 3]
    changed_config["architecture_compatibility"]["output_class_count"] = 7
    changed = build_task_record(changed_config)

    for name in (
        "task_config_sha256",
        "domain_sha256",
        "example_order_sha256",
        "dataset_sha256",
        "task_identity_sha256",
    ):
        assert original["hashes"][name] != changed["hashes"][name]


@pytest.mark.parametrize(
    "domain",
    (
        [0, 2, 1],
        [0, 1, 1],
        [0, 1, 7],
        [],
    ),
)
def test_noncanonical_or_invalid_domains_are_rejected(domain: list[int]) -> None:
    config = _addition()
    config["input_domains"][0] = domain

    with pytest.raises(TaskProtocolError):
        build_task_record(config)


@pytest.mark.parametrize("modulus", (True, 1, 0, -7, 7.0, "7"))
def test_invalid_modulus_is_rejected(modulus: object) -> None:
    config = _addition()
    config["modulus"] = modulus

    with pytest.raises(TaskProtocolError, match="modulus"):
        build_task_record(config)


def test_polynomial_terms_must_be_canonical_and_unique() -> None:
    config = _polynomial()
    config["parameters"]["terms"][1], config["parameters"]["terms"][2] = (
        config["parameters"]["terms"][2],
        config["parameters"]["terms"][1],
    )

    with pytest.raises(TaskProtocolError, match="lexicographic"):
        build_task_record(config)


def test_noncanonical_polynomial_coefficient_is_rejected() -> None:
    config = _polynomial()
    config["parameters"]["terms"][0]["coefficient"] = 0

    with pytest.raises(TaskProtocolError, match="non-zero residue"):
        build_task_record(config)


def test_nonserializable_definition_is_rejected() -> None:
    config = _addition()
    config["architecture_compatibility"]["bad"] = {1, 2, 3}

    with pytest.raises(TaskProtocolError, match="nonserializable"):
        build_task_record(config)


def test_tampered_record_hash_is_rejected() -> None:
    record = build_task_record(_addition())
    record["hashes"]["dataset_sha256"] = "f" * 64

    with pytest.raises(TaskProtocolError, match="inconsistent"):
        validate_task_record(record)


def test_tampered_output_vocabulary_is_rejected() -> None:
    record = build_task_record(_addition())
    record["output_vocabulary"]["values"][-1] = 99

    with pytest.raises(TaskProtocolError, match="inconsistent"):
        validate_task_record(record)


def test_duplicate_task_id_is_rejected() -> None:
    first = build_task_record(_addition())
    second_config = _multiplication()
    second_config["task_id"] = first["task_definition"]["task_id"]
    second = build_task_record(second_config)

    with pytest.raises(TaskProtocolError, match="duplicate task_id"):
        TaskRegistry([first, second])


def test_duplicate_underlying_identity_is_rejected_even_with_new_name() -> None:
    first = build_task_record(_addition())
    second_config = _addition()
    second_config["task_id"] = "technical-fixture-addition-alias-m7"
    second = build_task_record(second_config)

    # task_id is part of task_config_sha256, so construct an exact duplicate
    # record instead; a registry must still reject duplicate task identity.
    second = copy.deepcopy(first)
    second["task_definition"]["task_id"] = first["task_definition"]["task_id"]

    with pytest.raises(TaskProtocolError, match="duplicate task_id"):
        TaskRegistry([first, second])


@dataclass(frozen=True)
class _BadRangeImplementation:
    name: str = "technical_bad_range"
    version: str = "technical-bad-range/v1"

    def normalize_parameters(
        self,
        parameters: Any,
        *,
        arity: int,
        modulus: int,
    ) -> Mapping[str, Any]:
        del modulus
        if arity != 1 or parameters != {}:
            raise TaskProtocolError("invalid technical bad-range fixture")
        return {}

    def target(
        self,
        example: tuple[int, ...],
        *,
        parameters: Mapping[str, Any],
        modulus: int,
    ) -> int:
        del example, parameters
        return modulus


def test_registered_implementation_cannot_emit_out_of_range_target() -> None:
    bad = _BadRangeImplementation()
    registry = TaskImplementationRegistry([bad])
    config = _base_config(
        task_id="technical-fixture-invalid-target",
        implementation=bad.name,
        implementation_version=bad.version,
        parameters={},
        modulus=5,
        domains=[[0, 1, 2]],
    )
    config["architecture_compatibility"]["input_arity"] = 1

    with pytest.raises(TaskProtocolError, match="outside output range"):
        registry.build(config)


def test_duplicate_implementation_identity_is_rejected() -> None:
    implementation = ModularAdditionImplementation()
    with pytest.raises(TaskProtocolError, match="duplicate task implementation"):
        TaskImplementationRegistry([implementation, implementation])


def test_wrong_implementation_version_is_rejected() -> None:
    config = _addition()
    config["implementation_version"] = "modular-addition/v999"

    with pytest.raises(TaskProtocolError, match="version mismatch"):
        build_task_record(config)


def test_registry_serialization_is_order_independent() -> None:
    records = [
        build_task_record(_addition()),
        build_task_record(_multiplication()),
        build_task_record(_polynomial()),
    ]

    forward = TaskRegistry(records)
    reverse = TaskRegistry(list(reversed(records)))

    assert forward.canonical_bytes() == reverse.canonical_bytes()
    assert forward.sha256() == reverse.sha256()


@dataclass(frozen=True)
class _TargetOrderFixtureImplementation:
    """Same declared implementation identity; optional target-order permutation."""

    reverse_assignment: bool
    name: str = "technical_target_order_fixture"
    version: str = "technical-target-order-fixture/v1"

    def normalize_parameters(
        self,
        parameters: Any,
        *,
        arity: int,
        modulus: int,
    ) -> Mapping[str, Any]:
        del modulus
        if arity != 1 or parameters != {}:
            raise TaskProtocolError("invalid target-order fixture configuration")
        return {}

    def target(
        self,
        example: tuple[int, ...],
        *,
        parameters: Mapping[str, Any],
        modulus: int,
    ) -> int:
        del parameters
        value = example[0]
        return (modulus - 1 - value) if self.reverse_assignment else value


def test_target_order_change_with_same_declared_config_is_detected() -> None:
    original_impl = _TargetOrderFixtureImplementation(reverse_assignment=False)
    reordered_impl = _TargetOrderFixtureImplementation(reverse_assignment=True)

    original_registry = TaskImplementationRegistry([original_impl])
    reordered_registry = TaskImplementationRegistry([reordered_impl])

    config = _base_config(
        task_id="technical-fixture-target-order",
        implementation=original_impl.name,
        implementation_version=original_impl.version,
        parameters={},
        modulus=5,
        domains=[[0, 1, 2, 3, 4]],
    )
    config["architecture_compatibility"]["input_arity"] = 1

    original = original_registry.build(config)
    reordered = reordered_registry.build(config)

    # Both implementations emit exactly the same multiset {0,1,2,3,4},
    # but attach those targets to inputs in a different canonical order.
    assert original["hashes"]["task_config_sha256"] == (
        reordered["hashes"]["task_config_sha256"]
    )
    assert original["hashes"]["domain_sha256"] == (
        reordered["hashes"]["domain_sha256"]
    )
    assert original["hashes"]["example_order_sha256"] == (
        reordered["hashes"]["example_order_sha256"]
    )
    assert original["hashes"]["target_sha256"] != (
        reordered["hashes"]["target_sha256"]
    )
    assert original["hashes"]["dataset_sha256"] != (
        reordered["hashes"]["dataset_sha256"]
    )
    assert original["hashes"]["task_identity_sha256"] != (
        reordered["hashes"]["task_identity_sha256"]
    )

    with pytest.raises(TaskProtocolError, match="inconsistent"):
        validate_task_record(
            original,
            implementation_registry=reordered_registry,
        )
