"""Policy-neutral modular-task protocol and registry for Stage 12-P1.

The historical repository configuration freezes one specific modular-addition
experiment. This module deliberately does not weaken that contract. Instead it
provides a small versioned task interface for technical fixtures and later
injected production configuration.

No built-in implementation is a production roster entry. Registry order is
lexicographic metadata order only and carries no scientific priority.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

TASK_RECORD_SCHEMA_VERSION = "stage12p1-task-record/v1"
TASK_REGISTRY_SCHEMA_VERSION = "stage12p1-task-registry/v1"
TASK_CONFIG_SCHEMA_VERSION = "stage12p1-task-config/v1"
EXAMPLE_ORDER_VERSION = "lexicographic-domain-product/v1"
TECHNICAL_CLASSIFICATION = "technical_fixture"

_TASK_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "implementation",
        "implementation_version",
        "modulus",
        "input_domains",
        "parameters",
        "split_identity",
        "architecture_compatibility",
        "scientific_data",
        "production_eligible",
    }
)

_HASH_KEYS = frozenset(
    {
        "task_config_sha256",
        "domain_sha256",
        "example_order_sha256",
        "target_sha256",
        "dataset_sha256",
        "split_identity_sha256",
        "architecture_compatibility_sha256",
        "task_identity_sha256",
    }
)


class TaskProtocolError(ValueError):
    """Raised when a task definition or registry violates the Stage 12-P1 contract."""


def _canonical_json_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise TaskProtocolError(f"{path} contains a non-finite float")
        return value

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in value:
            if not isinstance(key, str) or not key:
                raise TaskProtocolError(
                    f"{path} mapping keys must be non-empty strings"
                )
        for key in sorted(value):
            normalized[key] = _canonical_json_value(
                value[key],
                path=f"{path}.{key}",
            )
        return normalized

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    raise TaskProtocolError(
        f"{path} contains unsupported/nonserializable type "
        f"{type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic compact JSON bytes with one final LF."""
    normalized = _canonical_json_value(value)
    try:
        text = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TaskProtocolError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc
    return (text + "\n").encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TaskProtocolError(f"{name} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise TaskProtocolError(f"{name} may not contain control characters")
    return value


def _require_modulus(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise TaskProtocolError("modulus must be an integer >= 2")
    return value


def _normalize_domains(value: Any, *, modulus: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise TaskProtocolError("input_domains must be a sequence of domains")

    domains: list[tuple[int, ...]] = []
    for domain_index, raw_domain in enumerate(value):
        if not isinstance(raw_domain, Sequence) or isinstance(
            raw_domain,
            (str, bytes, bytearray),
        ):
            raise TaskProtocolError(
                f"input_domains[{domain_index}] must be a sequence"
            )

        domain = tuple(raw_domain)
        if not domain:
            raise TaskProtocolError(
                f"input_domains[{domain_index}] must not be empty"
            )

        previous: int | None = None
        for item_index, item in enumerate(domain):
            if isinstance(item, bool) or not isinstance(item, int):
                raise TaskProtocolError(
                    f"input_domains[{domain_index}][{item_index}] "
                    "must be an integer"
                )
            if not 0 <= item < modulus:
                raise TaskProtocolError(
                    f"input_domains[{domain_index}][{item_index}] "
                    "must lie in [0, modulus)"
                )
            if previous is not None and item <= previous:
                raise TaskProtocolError(
                    f"input_domains[{domain_index}] must be in strictly "
                    "increasing canonical order with no duplicates"
                )
            previous = item

        domains.append(domain)

    if not domains:
        raise TaskProtocolError("at least one input domain is required")

    return tuple(domains)


def _normalize_optional_identity(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise TaskProtocolError(f"{name} must be null or a non-empty mapping")
    normalized = _canonical_json_value(value, path=name)
    assert isinstance(normalized, Mapping)
    return normalized


def _normalize_architecture_compatibility(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise TaskProtocolError(
            "architecture_compatibility must be a non-empty mapping"
        )
    normalized = _canonical_json_value(
        value,
        path="architecture_compatibility",
    )
    assert isinstance(normalized, Mapping)
    return normalized


class TaskImplementation(Protocol):
    """Registered deterministic target implementation."""

    name: str
    version: str

    def normalize_parameters(
        self,
        parameters: Any,
        *,
        arity: int,
        modulus: int,
    ) -> Mapping[str, Any]:
        ...

    def target(
        self,
        example: tuple[int, ...],
        *,
        parameters: Mapping[str, Any],
        modulus: int,
    ) -> int:
        ...


@dataclass(frozen=True)
class ModularAdditionImplementation:
    name: str = "modular_addition"
    version: str = "modular-addition/v1"

    def normalize_parameters(
        self,
        parameters: Any,
        *,
        arity: int,
        modulus: int,
    ) -> Mapping[str, Any]:
        del modulus
        if arity != 2:
            raise TaskProtocolError(
                "modular_addition requires exactly two input domains"
            )
        if not isinstance(parameters, Mapping) or dict(parameters):
            raise TaskProtocolError(
                "modular_addition parameters must be an empty mapping"
            )
        return {}

    def target(
        self,
        example: tuple[int, ...],
        *,
        parameters: Mapping[str, Any],
        modulus: int,
    ) -> int:
        del parameters
        return (example[0] + example[1]) % modulus


@dataclass(frozen=True)
class ModularMultiplicationImplementation:
    name: str = "modular_multiplication"
    version: str = "modular-multiplication/v1"

    def normalize_parameters(
        self,
        parameters: Any,
        *,
        arity: int,
        modulus: int,
    ) -> Mapping[str, Any]:
        del modulus
        if arity != 2:
            raise TaskProtocolError(
                "modular_multiplication requires exactly two input domains"
            )
        if not isinstance(parameters, Mapping) or dict(parameters):
            raise TaskProtocolError(
                "modular_multiplication parameters must be an empty mapping"
            )
        return {}

    def target(
        self,
        example: tuple[int, ...],
        *,
        parameters: Mapping[str, Any],
        modulus: int,
    ) -> int:
        del parameters
        return (example[0] * example[1]) % modulus


@dataclass(frozen=True)
class ModularPolynomialImplementation:
    """Canonical explicit multivariate polynomial modulo ``modulus``.

    Parameters are represented as::

        {
          "terms": [
            {"coefficient": 3, "exponents": [0, 0]},
            {"coefficient": 2, "exponents": [1, 0]},
            {"coefficient": 1, "exponents": [0, 2]}
          ]
        }

    Terms must already be in strictly increasing lexicographic exponent order.
    Coefficients are canonical non-zero residues. An empty term list is the
    canonical representation of the zero polynomial.

    This representation is an implementation capability only; it does not
    choose or freeze any production Task-3 formula.
    """

    name: str = "modular_polynomial"
    version: str = "modular-polynomial-terms/v1"

    def normalize_parameters(
        self,
        parameters: Any,
        *,
        arity: int,
        modulus: int,
    ) -> Mapping[str, Any]:
        if not isinstance(parameters, Mapping) or set(parameters) != {"terms"}:
            raise TaskProtocolError(
                "modular_polynomial parameters must contain exactly 'terms'"
            )

        raw_terms = parameters["terms"]
        if not isinstance(raw_terms, Sequence) or isinstance(
            raw_terms,
            (str, bytes, bytearray),
        ):
            raise TaskProtocolError("polynomial terms must be a sequence")

        normalized_terms: list[dict[str, Any]] = []
        previous_exponents: tuple[int, ...] | None = None

        for term_index, raw_term in enumerate(raw_terms):
            if not isinstance(raw_term, Mapping) or set(raw_term) != {
                "coefficient",
                "exponents",
            }:
                raise TaskProtocolError(
                    f"polynomial term {term_index} must contain exactly "
                    "'coefficient' and 'exponents'"
                )

            coefficient = raw_term["coefficient"]
            if (
                isinstance(coefficient, bool)
                or not isinstance(coefficient, int)
                or not 1 <= coefficient < modulus
            ):
                raise TaskProtocolError(
                    f"polynomial term {term_index} coefficient must be a "
                    "canonical non-zero residue"
                )

            raw_exponents = raw_term["exponents"]
            if not isinstance(raw_exponents, Sequence) or isinstance(
                raw_exponents,
                (str, bytes, bytearray),
            ):
                raise TaskProtocolError(
                    f"polynomial term {term_index} exponents must be a sequence"
                )

            exponents = tuple(raw_exponents)
            if len(exponents) != arity:
                raise TaskProtocolError(
                    f"polynomial term {term_index} exponent arity mismatch"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in exponents
            ):
                raise TaskProtocolError(
                    f"polynomial term {term_index} exponents must be "
                    "non-negative integers"
                )

            if previous_exponents is not None and exponents <= previous_exponents:
                raise TaskProtocolError(
                    "polynomial terms must be in strictly increasing "
                    "lexicographic exponent order with no duplicates"
                )
            previous_exponents = exponents

            normalized_terms.append(
                {
                    "coefficient": coefficient,
                    "exponents": list(exponents),
                }
            )

        return {"terms": normalized_terms}

    def target(
        self,
        example: tuple[int, ...],
        *,
        parameters: Mapping[str, Any],
        modulus: int,
    ) -> int:
        total = 0
        for term in parameters["terms"]:
            value = int(term["coefficient"])
            for coordinate, exponent in zip(
                example,
                term["exponents"],
                strict=True,
            ):
                value = (value * pow(coordinate, int(exponent), modulus)) % modulus
            total = (total + value) % modulus
        return total


DEFAULT_IMPLEMENTATIONS: tuple[TaskImplementation, ...] = (
    ModularAdditionImplementation(),
    ModularMultiplicationImplementation(),
    ModularPolynomialImplementation(),
)


class TaskImplementationRegistry:
    """Implementation-name registry with explicit version matching."""

    def __init__(
        self,
        implementations: Sequence[TaskImplementation] = DEFAULT_IMPLEMENTATIONS,
    ) -> None:
        self._implementations: dict[str, TaskImplementation] = {}

        for implementation in implementations:
            name = _require_nonempty_string(
                getattr(implementation, "name", None),
                name="implementation.name",
            )
            version = _require_nonempty_string(
                getattr(implementation, "version", None),
                name=f"implementation[{name}].version",
            )
            if name in self._implementations:
                raise TaskProtocolError(
                    f"duplicate task implementation identity: {name!r}"
                )
            self._implementations[name] = implementation

            if implementation.version != version:
                raise TaskProtocolError(
                    f"implementation {name!r} exposes unstable version metadata"
                )

    def implementation(
        self,
        name: str,
        version: str,
    ) -> TaskImplementation:
        try:
            implementation = self._implementations[name]
        except KeyError as exc:
            raise TaskProtocolError(
                f"unknown task implementation: {name!r}"
            ) from exc

        if implementation.version != version:
            raise TaskProtocolError(
                f"implementation version mismatch for {name!r}: "
                f"expected {implementation.version!r}, received {version!r}"
            )
        return implementation

    def build(self, config: Mapping[str, Any]) -> dict[str, Any]:
        return build_task_record(config, implementation_registry=self)


def _normalized_config(
    config: Mapping[str, Any],
    *,
    implementation_registry: TaskImplementationRegistry,
) -> tuple[dict[str, Any], TaskImplementation]:
    if not isinstance(config, Mapping):
        raise TaskProtocolError("task config must be a mapping")

    if set(config) != _TASK_CONFIG_KEYS:
        missing = sorted(_TASK_CONFIG_KEYS - set(config))
        extra = sorted(set(config) - _TASK_CONFIG_KEYS)
        raise TaskProtocolError(
            f"task config keys mismatch: missing={missing!r}, extra={extra!r}"
        )

    if config["schema_version"] != TASK_CONFIG_SCHEMA_VERSION:
        raise TaskProtocolError("task config schema_version mismatch")

    if config["scientific_data"] is not False:
        raise TaskProtocolError("task config must declare scientific_data=false")
    if config["production_eligible"] is not False:
        raise TaskProtocolError(
            "task config must declare production_eligible=false"
        )

    task_id = _require_nonempty_string(config["task_id"], name="task_id")
    implementation_name = _require_nonempty_string(
        config["implementation"],
        name="implementation",
    )
    implementation_version = _require_nonempty_string(
        config["implementation_version"],
        name="implementation_version",
    )
    modulus = _require_modulus(config["modulus"])
    domains = _normalize_domains(config["input_domains"], modulus=modulus)

    implementation = implementation_registry.implementation(
        implementation_name,
        implementation_version,
    )
    parameters = implementation.normalize_parameters(
        config["parameters"],
        arity=len(domains),
        modulus=modulus,
    )

    split_identity = _normalize_optional_identity(
        config["split_identity"],
        name="split_identity",
    )
    architecture_compatibility = _normalize_architecture_compatibility(
        config["architecture_compatibility"]
    )

    normalized = {
        "schema_version": TASK_CONFIG_SCHEMA_VERSION,
        "task_id": task_id,
        "implementation": implementation.name,
        "implementation_version": implementation.version,
        "modulus": modulus,
        "input_domains": [list(domain) for domain in domains],
        "parameters": copy.deepcopy(dict(parameters)),
        "split_identity": (
            None
            if split_identity is None
            else copy.deepcopy(dict(split_identity))
        ),
        "architecture_compatibility": copy.deepcopy(
            dict(architecture_compatibility)
        ),
        "scientific_data": False,
        "production_eligible": False,
    }
    return normalized, implementation


def _compute_hashes(
    config: Mapping[str, Any],
    implementation: TaskImplementation,
) -> tuple[dict[str, str], int]:
    modulus = int(config["modulus"])
    domains = tuple(
        tuple(int(item) for item in domain)
        for domain in config["input_domains"]
    )
    parameters = config["parameters"]

    domain_material = {
        "ordering_version": EXAMPLE_ORDER_VERSION,
        "input_domains": [list(domain) for domain in domains],
    }
    domain_sha256 = canonical_sha256(domain_material)

    example_order_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    dataset_digest = hashlib.sha256()
    example_count = 0

    for example_count, example in enumerate(
        itertools.product(*domains),
        start=1,
    ):
        target = implementation.target(
            tuple(int(value) for value in example),
            parameters=parameters,
            modulus=modulus,
        )
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or not 0 <= target < modulus
        ):
            raise TaskProtocolError(
                "registered implementation emitted target outside output range"
            )

        input_record = {
            "index": example_count - 1,
            "input": list(example),
        }
        target_record = {
            "index": example_count - 1,
            "target": target,
        }
        dataset_record = {
            "index": example_count - 1,
            "input": list(example),
            "target": target,
        }

        example_order_digest.update(canonical_json_bytes(input_record))
        target_digest.update(canonical_json_bytes(target_record))
        dataset_digest.update(canonical_json_bytes(dataset_record))

    split_material: Any
    if config["split_identity"] is None:
        split_material = {
            "kind": "no-split-declared",
            "example_order_version": EXAMPLE_ORDER_VERSION,
        }
    else:
        split_material = config["split_identity"]

    architecture_material = config["architecture_compatibility"]

    hashes = {
        "task_config_sha256": canonical_sha256(config),
        "domain_sha256": domain_sha256,
        "example_order_sha256": example_order_digest.hexdigest(),
        "target_sha256": target_digest.hexdigest(),
        "dataset_sha256": dataset_digest.hexdigest(),
        "split_identity_sha256": canonical_sha256(split_material),
        "architecture_compatibility_sha256": canonical_sha256(
            architecture_material
        ),
    }

    identity_material = {
        "identity_version": "stage12p1-task-identity/v1",
        "implementation": config["implementation"],
        "implementation_version": config["implementation_version"],
        "modulus": modulus,
        "task_config_sha256": hashes["task_config_sha256"],
        "domain_sha256": hashes["domain_sha256"],
        "example_order_sha256": hashes["example_order_sha256"],
        "target_sha256": hashes["target_sha256"],
        "dataset_sha256": hashes["dataset_sha256"],
        "split_identity_sha256": hashes["split_identity_sha256"],
        "architecture_compatibility_sha256": hashes[
            "architecture_compatibility_sha256"
        ],
    }
    hashes["task_identity_sha256"] = canonical_sha256(identity_material)
    return hashes, example_count


def build_task_record(
    config: Mapping[str, Any],
    *,
    implementation_registry: TaskImplementationRegistry | None = None,
) -> dict[str, Any]:
    """Build one compact deterministic technical task record."""
    registry = (
        TaskImplementationRegistry()
        if implementation_registry is None
        else implementation_registry
    )

    normalized, implementation = _normalized_config(
        config,
        implementation_registry=registry,
    )
    hashes, example_count = _compute_hashes(normalized, implementation)

    modulus = int(normalized["modulus"])
    condition_identity_material = {
        "identity_version": "stage12p1-task-condition-material/v1",
        "task_id": normalized["task_id"],
        "implementation": normalized["implementation"],
        "implementation_version": normalized["implementation_version"],
        "modulus": modulus,
        "example_order_version": EXAMPLE_ORDER_VERSION,
        "example_count": example_count,
        **copy.deepcopy(hashes),
    }

    return {
        "schema_version": TASK_RECORD_SCHEMA_VERSION,
        "classification": TECHNICAL_CLASSIFICATION,
        "scientific_data": False,
        "production_eligible": False,
        "task_definition": normalized,
        "output_vocabulary": {
            "kind": "modular_residue_classes",
            "modulus": modulus,
            "size": modulus,
            "values": list(range(modulus)),
        },
        "example_order": {
            "version": EXAMPLE_ORDER_VERSION,
            "example_count": example_count,
        },
        "hashes": hashes,
        "condition_identity_material": condition_identity_material,
    }


def validate_task_record(
    record: Mapping[str, Any],
    *,
    implementation_registry: TaskImplementationRegistry | None = None,
) -> dict[str, Any]:
    """Rebuild and compare every deterministic field in a task record."""
    if not isinstance(record, Mapping):
        raise TaskProtocolError("task record must be a mapping")

    required = {
        "schema_version",
        "classification",
        "scientific_data",
        "production_eligible",
        "task_definition",
        "output_vocabulary",
        "example_order",
        "hashes",
        "condition_identity_material",
    }
    if set(record) != required:
        raise TaskProtocolError("task record keys mismatch")

    if record["schema_version"] != TASK_RECORD_SCHEMA_VERSION:
        raise TaskProtocolError("task record schema_version mismatch")
    if record["classification"] != TECHNICAL_CLASSIFICATION:
        raise TaskProtocolError("task record must be a technical fixture")
    if record["scientific_data"] is not False:
        raise TaskProtocolError("task record must declare scientific_data=false")
    if record["production_eligible"] is not False:
        raise TaskProtocolError(
            "task record must declare production_eligible=false"
        )

    hashes = record["hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != _HASH_KEYS:
        raise TaskProtocolError("task record hash inventory mismatch")

    rebuilt = build_task_record(
        record["task_definition"],
        implementation_registry=implementation_registry,
    )

    if _canonical_json_value(record) != rebuilt:
        raise TaskProtocolError(
            "task record content/hash identity is inconsistent with definition"
        )

    return copy.deepcopy(rebuilt)


class TaskRegistry:
    """Canonical collection of unique technical task records."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        implementation_registry: TaskImplementationRegistry | None = None,
    ) -> None:
        self._implementation_registry = (
            TaskImplementationRegistry()
            if implementation_registry is None
            else implementation_registry
        )

        validated = [
            validate_task_record(
                record,
                implementation_registry=self._implementation_registry,
            )
            for record in records
        ]

        by_task_id: dict[str, dict[str, Any]] = {}
        by_identity: dict[str, str] = {}

        for record in validated:
            task_id = record["task_definition"]["task_id"]
            identity = record["hashes"]["task_identity_sha256"]

            if task_id in by_task_id:
                raise TaskProtocolError(
                    f"duplicate task_id in registry: {task_id!r}"
                )
            if identity in by_identity:
                raise TaskProtocolError(
                    "duplicate task identity in registry: "
                    f"{task_id!r} duplicates {by_identity[identity]!r}"
                )

            by_task_id[task_id] = record
            by_identity[identity] = task_id

        self._records = tuple(
            by_task_id[task_id]
            for task_id in sorted(by_task_id)
        )

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(record) for record in self._records)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_REGISTRY_SCHEMA_VERSION,
            "classification": TECHNICAL_CLASSIFICATION,
            "scientific_data": False,
            "production_eligible": False,
            "ordering": "task-id-lexicographic-no-priority/v1",
            "task_count": len(self._records),
            "records": [
                copy.deepcopy(record)
                for record in self._records
            ],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
