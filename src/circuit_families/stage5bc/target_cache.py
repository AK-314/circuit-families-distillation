"""Teacher-target-cache contract and canonical manifest utilities.

Part F defines metadata and validation only. Payload construction/loading is
implemented later. No teacher checkpoint is loaded here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch

CACHE_MANIFEST_SCHEMA_VERSION = "stage5bc-target-cache-manifest/v1"
FULL_DOMAIN_EXAMPLE_COUNT = 12_769
STAGE4_CACHE_KINDS = ("teacher_argmax", "teacher_logits")
CENTRING_SEMANTICS = "subtract_per_input_class_mean"
ATOMIC_COMPLETION_PROTOCOL = "write-temp-fsync-replace/v1"

_FLOAT_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "bfloat16",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID_RE = re.compile(r"^cfdid:v1:d3\|.+$")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "technical_fixture",
        "stage4_record_serializable",
        "stage4_cache_kinds",
        "example_count",
        "class_count",
        "input_order",
        "representations",
        "teacher_reference",
        "provenance_hashes",
        "payload",
        "completion",
    }
)

_INPUT_ORDER_KEYS = frozenset(
    {
        "ordering_ref",
        "ordered_input_ids_sha256",
        "example_count",
        "exact_order_required",
    }
)

_REPRESENTATION_KEYS = frozenset(
    {
        "raw_logits",
        "centred_logits",
        "argmax",
        "probabilities",
    }
)

_LOGIT_KEYS = frozenset(
    {
        "present",
        "representation",
        "shape",
        "dtype",
        "sha256",
    }
)

_ARGMAX_KEYS = frozenset(
    {
        "present",
        "representation",
        "shape",
        "dtype",
        "sha256",
    }
)

_TEACHER_REFERENCE_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "condition_id",
        "record_sha256",
    }
)

_PROVENANCE_HASH_KEYS = frozenset(
    {
        "dataset_sha256",
        "split_sha256",
        "task_config_sha256",
        "model_config_sha256",
        "training_config_sha256",
        "component_basis_sha256",
    }
)

_PAYLOAD_KEYS = frozenset(
    {
        "path",
        "sha256",
        "storage_class",
    }
)

_COMPLETION_KEYS = frozenset(
    {
        "atomic_write_protocol",
        "completion_state",
        "completion_record_path",
        "completion_record_sha256",
    }
)


class TargetCacheContractError(ValueError):
    """Raised when cache metadata violates the frozen technical boundary."""


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetCacheContractError(f"{name} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    required: frozenset[str],
    *,
    name: str,
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        raise TargetCacheContractError(
            f"{name} keys mismatch: missing={missing!r}, extra={extra!r}"
        )


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TargetCacheContractError(
            f"{name} must be a positive integer"
        )
    return value


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TargetCacheContractError(
            f"{name} must be a non-empty string"
        )
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TargetCacheContractError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _require_portable_relative_path(value: Any, *, name: str) -> str:
    text = _require_nonempty_string(value, name=name)

    if "\\" in text:
        raise TargetCacheContractError(
            f"{name} must use POSIX separators"
        )

    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise TargetCacheContractError(
            f"{name} must be a portable relative path"
        )

    return text


def _require_shape(
    value: Any,
    *,
    expected: tuple[int, ...],
    name: str,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TargetCacheContractError(f"{name} must be a list")

    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise TargetCacheContractError(
            f"{name} must contain positive integers"
        )

    shape = tuple(value)
    if shape != expected:
        raise TargetCacheContractError(
            f"{name} mismatch: expected={expected!r}, actual={shape!r}"
        )

    return shape


def _validate_logit_representation(
    value: Any,
    *,
    name: str,
    expected_representation: str,
    example_count: int,
    class_count: int,
) -> None:
    mapping = _require_mapping(value, name=name)
    _require_exact_keys(mapping, _LOGIT_KEYS, name=name)

    if mapping["present"] is not True:
        raise TargetCacheContractError(f"{name}.present must be true")

    if mapping["representation"] != expected_representation:
        raise TargetCacheContractError(
            f"{name}.representation must be "
            f"{expected_representation!r}"
        )

    _require_shape(
        mapping["shape"],
        expected=(example_count, class_count),
        name=f"{name}.shape",
    )

    if mapping["dtype"] not in _FLOAT_DTYPES:
        raise TargetCacheContractError(
            f"{name}.dtype must be an explicit floating dtype"
        )

    _require_sha256(mapping["sha256"], name=f"{name}.sha256")


def _validate_argmax_representation(
    value: Any,
    *,
    example_count: int,
) -> None:
    name = "representations.argmax"
    mapping = _require_mapping(value, name=name)
    _require_exact_keys(mapping, _ARGMAX_KEYS, name=name)

    if mapping["present"] is not True:
        raise TargetCacheContractError(f"{name}.present must be true")

    if mapping["representation"] != "argmax_from_raw_logits":
        raise TargetCacheContractError(
            f"{name}.representation must be 'argmax_from_raw_logits'"
        )

    _require_shape(
        mapping["shape"],
        expected=(example_count,),
        name=f"{name}.shape",
    )

    if mapping["dtype"] != "int64":
        raise TargetCacheContractError(
            f"{name}.dtype must be 'int64'"
        )

    _require_sha256(mapping["sha256"], name=f"{name}.sha256")


def _validate_probability_representation(
    value: Any,
    *,
    example_count: int,
    class_count: int,
) -> None:
    name = "representations.probabilities"
    mapping = _require_mapping(value, name=name)

    if "present" not in mapping or not isinstance(mapping["present"], bool):
        raise TargetCacheContractError(
            f"{name}.present must be an explicit boolean"
        )

    if mapping["present"] is False:
        if set(mapping) != {"present"}:
            raise TargetCacheContractError(
                f"{name} must contain only 'present' when absent"
            )
        return

    _require_exact_keys(mapping, _LOGIT_KEYS, name=name)

    if mapping["representation"] != "teacher_probabilities":
        raise TargetCacheContractError(
            f"{name}.representation must be 'teacher_probabilities'"
        )

    _require_shape(
        mapping["shape"],
        expected=(example_count, class_count),
        name=f"{name}.shape",
    )

    if mapping["dtype"] not in _FLOAT_DTYPES:
        raise TargetCacheContractError(
            f"{name}.dtype must be an explicit floating dtype"
        )

    _require_sha256(mapping["sha256"], name=f"{name}.sha256")


def _validate_json_finite(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise TargetCacheContractError(
                f"{path} contains a non-finite float"
            )
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_finite(item, path=f"{path}[{index}]")
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TargetCacheContractError(
                    f"{path} contains a non-string mapping key"
                )
            _validate_json_finite(item, path=f"{path}.{key}")
        return

    raise TargetCacheContractError(
        f"{path} contains unsupported JSON type "
        f"{type(value).__name__}"
    )


class TargetCacheManifest:
    """Validated immutable-by-copy target-cache manifest boundary."""

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        validated = self._validate(mapping)
        self._mapping = validated

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> TargetCacheManifest:
        return cls(mapping)

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
    ) -> TargetCacheManifest:
        file_path = Path(path)
        try:
            value = json.loads(
                file_path.read_text(encoding="utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    TargetCacheContractError(
                        f"non-standard JSON constant forbidden: {token}"
                    )
                ),
            )
        except json.JSONDecodeError as exc:
            raise TargetCacheContractError(
                f"invalid target-cache manifest JSON: {exc}"
            ) from exc

        return cls(value)

    @staticmethod
    def _validate(
        raw: Mapping[str, Any],
    ) -> dict[str, Any]:
        mapping = _require_mapping(raw, name="target-cache manifest")
        _require_exact_keys(
            mapping,
            _TOP_LEVEL_KEYS,
            name="target-cache manifest",
        )
        _validate_json_finite(mapping)

        if mapping["schema_version"] != CACHE_MANIFEST_SCHEMA_VERSION:
            raise TargetCacheContractError(
                "unsupported target-cache manifest schema_version"
            )

        manifest_id = _require_nonempty_string(
            mapping["manifest_id"],
            name="manifest_id",
        )
        if not manifest_id.startswith("technical-") and not manifest_id.startswith(
            "cache-"
        ):
            raise TargetCacheContractError(
                "manifest_id must identify a technical or cache artifact"
            )
        if "/v" not in manifest_id:
            raise TargetCacheContractError(
                "manifest_id must be explicitly versioned"
            )

        technical_fixture = mapping["technical_fixture"]
        if not isinstance(technical_fixture, bool):
            raise TargetCacheContractError(
                "technical_fixture must be an explicit boolean"
            )

        stage4_serializable = mapping["stage4_record_serializable"]
        if not isinstance(stage4_serializable, bool):
            raise TargetCacheContractError(
                "stage4_record_serializable must be an explicit boolean"
            )

        cache_kinds = mapping["stage4_cache_kinds"]
        if not isinstance(cache_kinds, list):
            raise TargetCacheContractError(
                "stage4_cache_kinds must be an explicit list"
            )
        if tuple(cache_kinds) != STAGE4_CACHE_KINDS:
            raise TargetCacheContractError(
                "stage4_cache_kinds must be exactly "
                f"{STAGE4_CACHE_KINDS!r}"
            )

        example_count = _require_positive_int(
            mapping["example_count"],
            name="example_count",
        )
        class_count = _require_positive_int(
            mapping["class_count"],
            name="class_count",
        )

        input_order = _require_mapping(
            mapping["input_order"],
            name="input_order",
        )
        _require_exact_keys(
            input_order,
            _INPUT_ORDER_KEYS,
            name="input_order",
        )

        _require_nonempty_string(
            input_order["ordering_ref"],
            name="input_order.ordering_ref",
        )
        _require_sha256(
            input_order["ordered_input_ids_sha256"],
            name="input_order.ordered_input_ids_sha256",
        )

        if input_order["example_count"] != example_count:
            raise TargetCacheContractError(
                "input_order.example_count must equal manifest example_count"
            )

        if input_order["exact_order_required"] is not True:
            raise TargetCacheContractError(
                "input_order.exact_order_required must be true"
            )

        representations = _require_mapping(
            mapping["representations"],
            name="representations",
        )
        _require_exact_keys(
            representations,
            _REPRESENTATION_KEYS,
            name="representations",
        )

        _validate_logit_representation(
            representations["raw_logits"],
            name="representations.raw_logits",
            expected_representation="raw_final_position_logits",
            example_count=example_count,
            class_count=class_count,
        )
        _validate_logit_representation(
            representations["centred_logits"],
            name="representations.centred_logits",
            expected_representation=CENTRING_SEMANTICS,
            example_count=example_count,
            class_count=class_count,
        )
        _validate_argmax_representation(
            representations["argmax"],
            example_count=example_count,
        )
        _validate_probability_representation(
            representations["probabilities"],
            example_count=example_count,
            class_count=class_count,
        )

        teacher_reference = _require_mapping(
            mapping["teacher_reference"],
            name="teacher_reference",
        )
        _require_exact_keys(
            teacher_reference,
            _TEACHER_REFERENCE_KEYS,
            name="teacher_reference",
        )

        if teacher_reference["record_type"] != "teacher_reference":
            raise TargetCacheContractError(
                "teacher_reference.record_type must be 'teacher_reference'"
            )
        if teacher_reference["schema_version"] != "teacher_reference/v1":
            raise TargetCacheContractError(
                "teacher_reference.schema_version must be "
                "'teacher_reference/v1'"
            )
        condition_id = teacher_reference["condition_id"]
        if (
            not isinstance(condition_id, str)
            or not _CONDITION_ID_RE.fullmatch(condition_id)
        ):
            raise TargetCacheContractError(
                "teacher_reference.condition_id must be a depth-3 "
                "canonical-condition-shaped ID"
            )
        _require_sha256(
            teacher_reference["record_sha256"],
            name="teacher_reference.record_sha256",
        )

        provenance = _require_mapping(
            mapping["provenance_hashes"],
            name="provenance_hashes",
        )
        _require_exact_keys(
            provenance,
            _PROVENANCE_HASH_KEYS,
            name="provenance_hashes",
        )
        for field in sorted(_PROVENANCE_HASH_KEYS):
            _require_sha256(
                provenance[field],
                name=f"provenance_hashes.{field}",
            )

        payload = _require_mapping(mapping["payload"], name="payload")
        _require_exact_keys(payload, _PAYLOAD_KEYS, name="payload")
        _require_portable_relative_path(
            payload["path"],
            name="payload.path",
        )
        _require_sha256(payload["sha256"], name="payload.sha256")

        if payload["storage_class"] not in {
            "technical_fixture_small_file",
            "external_large_object",
        }:
            raise TargetCacheContractError(
                "payload.storage_class is invalid"
            )

        completion = _require_mapping(
            mapping["completion"],
            name="completion",
        )
        _require_exact_keys(
            completion,
            _COMPLETION_KEYS,
            name="completion",
        )

        if (
            completion["atomic_write_protocol"]
            != ATOMIC_COMPLETION_PROTOCOL
        ):
            raise TargetCacheContractError(
                "completion.atomic_write_protocol mismatch"
            )

        if completion["completion_state"] != "complete":
            raise TargetCacheContractError(
                "completion.completion_state must be 'complete'"
            )

        _require_portable_relative_path(
            completion["completion_record_path"],
            name="completion.completion_record_path",
        )
        _require_sha256(
            completion["completion_record_sha256"],
            name="completion.completion_record_sha256",
        )

        if example_count != FULL_DOMAIN_EXAMPLE_COUNT:
            if not technical_fixture:
                raise TargetCacheContractError(
                    "sub-full-domain caches must be marked technical_fixture"
                )
            if stage4_serializable:
                raise TargetCacheContractError(
                    "sub-full-domain technical fixtures cannot be serialized "
                    "as Stage 4 production cache records"
                )

        if technical_fixture:
            if stage4_serializable:
                raise TargetCacheContractError(
                    "technical fixtures can never be Stage 4 record serializable"
                )
            if payload["storage_class"] != "technical_fixture_small_file":
                raise TargetCacheContractError(
                    "technical fixtures must use technical small-file storage"
                )
        else:
            if example_count != FULL_DOMAIN_EXAMPLE_COUNT:
                raise TargetCacheContractError(
                    "non-technical caches must cover the full 12,769 inputs"
                )
            if not stage4_serializable:
                raise TargetCacheContractError(
                    "non-technical full-domain cache contract must declare "
                    "Stage 4 record serializability explicitly"
                )
            if payload["storage_class"] != "external_large_object":
                raise TargetCacheContractError(
                    "full-domain Stage 4 cache payload must use "
                    "external_large_object storage"
                )

        return copy.deepcopy(dict(mapping))

    @property
    def example_count(self) -> int:
        return int(self._mapping["example_count"])

    @property
    def class_count(self) -> int:
        return int(self._mapping["class_count"])

    @property
    def technical_fixture(self) -> bool:
        return bool(self._mapping["technical_fixture"])

    @property
    def stage4_record_serializable(self) -> bool:
        return bool(self._mapping["stage4_record_serializable"])

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(self._mapping)

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self._mapping,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


_PAYLOAD_MAGIC = b"CFDTCACHE1\n"
_PAYLOAD_SCHEMA_VERSION = "stage5bc-target-cache-payload/v1"
_COMPLETION_SCHEMA_VERSION = "stage5bc-target-cache-completion/v1"

_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class TargetCacheBatch:
    """One ordered batch supplied to the streaming technical cache builder."""

    input_ids: Sequence[str]
    raw_logits: torch.Tensor
    probabilities: torch.Tensor | None = None


@dataclass(frozen=True)
class BuiltTargetCache:
    """Paths and validated manifest emitted by one cache build."""

    manifest: TargetCacheManifest
    manifest_path: Path
    payload_path: Path
    completion_path: Path


@dataclass(frozen=True)
class LoadedTargetCache:
    """Strictly verified target-cache contents."""

    manifest: TargetCacheManifest
    input_ids: tuple[str, ...]
    raw_logits: torch.Tensor
    centred_logits: torch.Tensor
    argmax: torch.Tensor
    probabilities: torch.Tensor | None

    def stage4_view(self, cache_kind: str) -> torch.Tensor:
        """Return the Stage 4 hard-label or teacher-logit cache view."""
        if cache_kind == "teacher_argmax":
            return self.argmax
        if cache_kind == "teacher_logits":
            return self.centred_logits
        raise TargetCacheContractError(
            f"unsupported Stage 4 cache kind: {cache_kind!r}"
        )


def centre_logits(raw_logits: torch.Tensor) -> torch.Tensor:
    """Subtract each input's mean across output classes."""
    if not isinstance(raw_logits, torch.Tensor):
        raise TargetCacheContractError("raw_logits must be a torch.Tensor")
    if raw_logits.ndim != 2:
        raise TargetCacheContractError(
            "raw_logits must have shape [examples, classes]"
        )
    if raw_logits.shape[0] <= 0 or raw_logits.shape[1] <= 0:
        raise TargetCacheContractError(
            "raw_logits dimensions must be non-empty"
        )
    if not raw_logits.is_floating_point():
        raise TargetCacheContractError(
            "raw_logits must use a floating dtype"
        )
    if not bool(torch.isfinite(raw_logits).all()):
        raise TargetCacheContractError(
            "raw_logits must contain only finite values"
        )
    return raw_logits - raw_logits.mean(dim=-1, keepdim=True)


def _torch_dtype_name(dtype: torch.dtype) -> str:
    for name, candidate in _TORCH_DTYPES.items():
        if dtype == candidate:
            return name
    raise TargetCacheContractError(
        f"unsupported cache floating dtype: {dtype}"
    )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().to(device="cpu").contiguous()
    return cpu.view(torch.uint8).numpy().tobytes(order="C")


def _tensor_from_bytes(
    payload: bytes,
    *,
    dtype_name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    try:
        dtype = _TORCH_DTYPES[dtype_name]
    except KeyError as exc:
        raise TargetCacheContractError(
            f"unsupported payload dtype: {dtype_name!r}"
        ) from exc

    expected_items = math.prod(shape)
    expected_bytes = expected_items * torch.empty(
        (),
        dtype=dtype,
    ).element_size()

    if len(payload) != expected_bytes:
        raise TargetCacheContractError(
            "payload tensor byte length does not match declared shape/dtype"
        )

    writable = bytearray(payload)
    tensor = torch.frombuffer(writable, dtype=dtype).clone()

    return tensor.reshape(shape)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolve_output_path(
    output_root: Path,
    relative_path: str,
    *,
    name: str,
) -> Path:
    portable = _require_portable_relative_path(
        relative_path,
        name=name,
    )
    posix = PurePosixPath(portable)
    return output_root.joinpath(*posix.parts)


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, path)
        temporary_name = None
        _fsync_parent(path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _update_ordered_input_hash(
    digest: Any,
    input_id: str,
) -> bytes:
    if not isinstance(input_id, str) or not input_id:
        raise TargetCacheContractError(
            "every canonical input ID must be a non-empty string"
        )
    encoded = input_id.encode("utf-8")
    digest.update(struct.pack(">Q", len(encoded)))
    digest.update(encoded)
    return encoded


def _validate_build_batch(
    batch: TargetCacheBatch,
    *,
    expected_class_count: int,
    expected_dtype_name: str | None,
    probabilities_present: bool,
    expected_probability_dtype_name: str | None,
) -> tuple[str, str | None]:
    if not isinstance(batch, TargetCacheBatch):
        raise TargetCacheContractError(
            "cache batches must be TargetCacheBatch instances"
        )

    logits = batch.raw_logits
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise TargetCacheContractError(
            "batch raw_logits must be a rank-2 torch.Tensor"
        )

    if logits.shape[0] != len(batch.input_ids):
        raise TargetCacheContractError(
            "batch input_ids length must match raw_logits rows"
        )

    if logits.shape[0] <= 0:
        raise TargetCacheContractError(
            "cache batches must not be empty"
        )

    if logits.shape[1] != expected_class_count:
        raise TargetCacheContractError(
            "batch class count differs from expected_class_count"
        )

    dtype_name = _torch_dtype_name(logits.dtype)

    if expected_dtype_name is not None and dtype_name != expected_dtype_name:
        raise TargetCacheContractError(
            "raw-logit dtype changed across streaming batches"
        )

    if not bool(torch.isfinite(logits).all()):
        raise TargetCacheContractError(
            "raw_logits contain non-finite values"
        )

    if probabilities_present != (batch.probabilities is not None):
        raise TargetCacheContractError(
            "probability presence changed across streaming batches"
        )

    probability_dtype_name: str | None = None
    if batch.probabilities is not None:
        probabilities = batch.probabilities

        if (
            not isinstance(probabilities, torch.Tensor)
            or probabilities.shape != logits.shape
        ):
            raise TargetCacheContractError(
                "probabilities must be a tensor matching raw_logits shape"
            )

        probability_dtype_name = _torch_dtype_name(
            probabilities.dtype
        )

        if (
            expected_probability_dtype_name is not None
            and probability_dtype_name
            != expected_probability_dtype_name
        ):
            raise TargetCacheContractError(
                "probability dtype changed across streaming batches"
            )

        if not bool(torch.isfinite(probabilities).all()):
            raise TargetCacheContractError(
                "probabilities contain non-finite values"
            )

    return dtype_name, probability_dtype_name



def _validated_expected_input_ids(
    value: Sequence[str],
    *,
    expected_count: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TargetCacheContractError(
            "expected_input_ids must be an explicit sequence of strings"
        )

    items = tuple(value)

    if len(items) != expected_count:
        raise TargetCacheContractError(
            "expected_input_ids length must equal expected example count"
        )

    if any(not isinstance(item, str) or not item for item in items):
        raise TargetCacheContractError(
            "expected_input_ids entries must be non-empty strings"
        )

    if len(set(items)) != len(items):
        raise TargetCacheContractError(
            "expected_input_ids must not contain duplicates"
        )

    return items


def _stage4_kind_for_condition_id(condition_id: str) -> str | None:
    if "distillation_condition=hard_target" in condition_id:
        return "teacher_argmax"
    if "distillation_condition=soft_target" in condition_id:
        return "teacher_logits"
    return None

def build_target_cache(
    *,
    output_root: str | Path,
    manifest_relative_path: str,
    payload_relative_path: str,
    completion_relative_path: str,
    manifest_id: str,
    ordering_ref: str,
    expected_example_count: int,
    expected_class_count: int,
    teacher_reference: Mapping[str, Any],
    provenance_hashes: Mapping[str, str],
    batches: Iterable[TargetCacheBatch],
    technical_fixture: bool,
    stage4_record_serializable: bool,
    expected_input_ids: Sequence[str] | None = None,
) -> BuiltTargetCache:
    """Stream one cache payload and atomically publish its metadata."""
    example_count = _require_positive_int(
        expected_example_count,
        name="expected_example_count",
    )
    class_count = _require_positive_int(
        expected_class_count,
        name="expected_class_count",
    )

    expected_order = None
    if expected_input_ids is not None:
        expected_order = _validated_expected_input_ids(
            expected_input_ids,
            expected_count=example_count,
        )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = _resolve_output_path(
        root,
        manifest_relative_path,
        name="manifest_relative_path",
    )
    payload_path = _resolve_output_path(
        root,
        payload_relative_path,
        name="payload_relative_path",
    )
    completion_path = _resolve_output_path(
        root,
        completion_relative_path,
        name="completion_relative_path",
    )

    if len({manifest_path, payload_path, completion_path}) != 3:
        raise TargetCacheContractError(
            "manifest, payload and completion paths must be distinct"
        )

    for candidate in (
        manifest_path,
        payload_path,
        completion_path,
    ):
        if candidate.exists():
            raise TargetCacheContractError(
                f"cache output already exists: {candidate}"
            )

    iterator: Iterator[TargetCacheBatch] = iter(batches)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise TargetCacheContractError(
            "cache construction requires at least one batch"
        ) from exc

    if not isinstance(first, TargetCacheBatch):
        raise TargetCacheContractError(
            "cache batches must be TargetCacheBatch instances"
        )

    first_probability_present = first.probabilities is not None

    first_dtype_name, first_probability_dtype_name = _validate_build_batch(
        first,
        expected_class_count=class_count,
        expected_dtype_name=None,
        probabilities_present=first_probability_present,
        expected_probability_dtype_name=None,
    )

    header = {
        "schema_version": _PAYLOAD_SCHEMA_VERSION,
        "expected_example_count": example_count,
        "class_count": class_count,
        "logit_dtype": first_dtype_name,
        "probabilities_present": first_probability_present,
        "probability_dtype": first_probability_dtype_name,
    }
    header_bytes = _canonical_json_bytes(header)

    payload_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_input_digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    centred_digest = hashlib.sha256()
    argmax_digest = hashlib.sha256()
    probability_digest = (
        hashlib.sha256() if first_probability_present else None
    )

    seen_ids: set[str] = set()
    observed_count = 0

    temporary_name: str | None = None

    def all_batches() -> Iterator[TargetCacheBatch]:
        yield first
        yield from iterator

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{payload_path.name}.",
            suffix=".tmp",
            dir=payload_path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name

            handle.write(_PAYLOAD_MAGIC)
            handle.write(struct.pack(">Q", len(header_bytes)))
            handle.write(header_bytes)

            for batch in all_batches():
                dtype_name, probability_dtype_name = _validate_build_batch(
                    batch,
                    expected_class_count=class_count,
                    expected_dtype_name=first_dtype_name,
                    probabilities_present=first_probability_present,
                    expected_probability_dtype_name=(
                        first_probability_dtype_name
                    ),
                )

                if dtype_name != first_dtype_name:
                    raise TargetCacheContractError(
                        "streaming raw-logit dtype mismatch"
                    )

                if (
                    probability_dtype_name
                    != first_probability_dtype_name
                ):
                    raise TargetCacheContractError(
                        "streaming probability dtype mismatch"
                    )

                raw_cpu = (
                    batch.raw_logits.detach()
                    .to(device="cpu")
                    .contiguous()
                )
                centred_cpu = centre_logits(raw_cpu)
                argmax_cpu = raw_cpu.argmax(dim=-1).to(torch.int64)

                probabilities_cpu = None
                if batch.probabilities is not None:
                    probabilities_cpu = (
                        batch.probabilities.detach()
                        .to(device="cpu")
                        .contiguous()
                    )

                for row_index, input_id in enumerate(batch.input_ids):
                    if input_id in seen_ids:
                        raise TargetCacheContractError(
                            f"duplicate canonical input ID: {input_id!r}"
                        )

                    if (
                        expected_order is not None
                        and input_id != expected_order[observed_count]
                    ):
                        raise TargetCacheContractError(
                            "canonical input order mismatch: "
                            f"index={observed_count}, "
                            f"expected={expected_order[observed_count]!r}, "
                            f"actual={input_id!r}"
                        )

                    seen_ids.add(input_id)

                    encoded_id = _update_ordered_input_hash(
                        ordered_input_digest,
                        input_id,
                    )

                    raw_row = raw_cpu[row_index]
                    centred_row = centred_cpu[row_index]
                    argmax_row = argmax_cpu[row_index : row_index + 1]

                    raw_bytes = _tensor_bytes(raw_row)
                    centred_bytes = _tensor_bytes(centred_row)
                    argmax_bytes = _tensor_bytes(argmax_row)

                    raw_digest.update(raw_bytes)
                    centred_digest.update(centred_bytes)
                    argmax_digest.update(argmax_bytes)

                    handle.write(struct.pack(">I", len(encoded_id)))
                    handle.write(encoded_id)
                    handle.write(raw_bytes)

                    if probabilities_cpu is not None:
                        probability_bytes = _tensor_bytes(
                            probabilities_cpu[row_index]
                        )
                        assert probability_digest is not None
                        probability_digest.update(probability_bytes)
                        handle.write(probability_bytes)

                    observed_count += 1

            if observed_count != example_count:
                raise TargetCacheContractError(
                    "streamed example count mismatch: "
                    f"expected={example_count}, observed={observed_count}"
                )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, payload_path)
        temporary_name = None
        _fsync_parent(payload_path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass

    payload_sha256 = _file_sha256(payload_path)
    ordered_input_sha256 = ordered_input_digest.hexdigest()
    raw_sha256 = raw_digest.hexdigest()
    centred_sha256 = centred_digest.hexdigest()
    argmax_sha256 = argmax_digest.hexdigest()

    probability_record: dict[str, Any]
    probability_sha256: str | None

    if probability_digest is None:
        probability_record = {"present": False}
        probability_sha256 = None
    else:
        probability_sha256 = probability_digest.hexdigest()
        probability_record = {
            "present": True,
            "representation": "teacher_probabilities",
            "shape": [example_count, class_count],
            "dtype": first_probability_dtype_name,
            "sha256": probability_sha256,
        }

    completion_record = {
        "schema_version": _COMPLETION_SCHEMA_VERSION,
        "payload_path": payload_relative_path,
        "payload_sha256": payload_sha256,
        "example_count": example_count,
        "ordered_input_ids_sha256": ordered_input_sha256,
        "representation_hashes": {
            "raw_logits_sha256": raw_sha256,
            "centred_logits_sha256": centred_sha256,
            "argmax_sha256": argmax_sha256,
            "probabilities_sha256": probability_sha256,
        },
    }
    completion_bytes = _canonical_json_bytes(completion_record)
    completion_sha256 = hashlib.sha256(completion_bytes).hexdigest()

    _atomic_write_bytes(completion_path, completion_bytes)

    storage_class = (
        "technical_fixture_small_file"
        if technical_fixture
        else "external_large_object"
    )

    manifest_mapping = {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "technical_fixture": technical_fixture,
        "stage4_record_serializable": stage4_record_serializable,
        "stage4_cache_kinds": list(STAGE4_CACHE_KINDS),
        "example_count": example_count,
        "class_count": class_count,
        "input_order": {
            "ordering_ref": ordering_ref,
            "ordered_input_ids_sha256": ordered_input_sha256,
            "example_count": example_count,
            "exact_order_required": True,
        },
        "representations": {
            "raw_logits": {
                "present": True,
                "representation": "raw_final_position_logits",
                "shape": [example_count, class_count],
                "dtype": first_dtype_name,
                "sha256": raw_sha256,
            },
            "centred_logits": {
                "present": True,
                "representation": CENTRING_SEMANTICS,
                "shape": [example_count, class_count],
                "dtype": first_dtype_name,
                "sha256": centred_sha256,
            },
            "argmax": {
                "present": True,
                "representation": "argmax_from_raw_logits",
                "shape": [example_count],
                "dtype": "int64",
                "sha256": argmax_sha256,
            },
            "probabilities": probability_record,
        },
        "teacher_reference": copy.deepcopy(dict(teacher_reference)),
        "provenance_hashes": copy.deepcopy(dict(provenance_hashes)),
        "payload": {
            "path": payload_relative_path,
            "sha256": payload_sha256,
            "storage_class": storage_class,
        },
        "completion": {
            "atomic_write_protocol": ATOMIC_COMPLETION_PROTOCOL,
            "completion_state": "complete",
            "completion_record_path": completion_relative_path,
            "completion_record_sha256": completion_sha256,
        },
    }

    manifest = TargetCacheManifest.from_mapping(manifest_mapping)
    _atomic_write_bytes(manifest_path, manifest.canonical_bytes())

    return BuiltTargetCache(
        manifest=manifest,
        manifest_path=manifest_path,
        payload_path=payload_path,
        completion_path=completion_path,
    )


def _read_exact(handle: Any, count: int, *, name: str) -> bytes:
    payload = handle.read(count)
    if len(payload) != count:
        raise TargetCacheContractError(
            f"truncated target-cache payload while reading {name}"
        )
    return payload


def load_target_cache(
    *,
    output_root: str | Path,
    manifest_relative_path: str,
    expected_input_ids: Sequence[str] | None = None,
    expected_teacher_reference: Mapping[str, Any] | None = None,
    expected_provenance_hashes: Mapping[str, str] | None = None,
    expected_stage4_cache_kind: str | None = None,
) -> LoadedTargetCache:
    """Strictly load and verify one completed cache against all manifest hashes."""
    root = Path(output_root)
    manifest_path = _resolve_output_path(
        root,
        manifest_relative_path,
        name="manifest_relative_path",
    )

    if not manifest_path.is_file():
        raise TargetCacheContractError(
            f"target-cache manifest does not exist: {manifest_path}"
        )

    manifest = TargetCacheManifest.from_json_file(manifest_path)
    mapping = manifest.to_mapping()

    expected_order = None
    if expected_input_ids is not None:
        expected_order = _validated_expected_input_ids(
            expected_input_ids,
            expected_count=manifest.example_count,
        )

    if expected_teacher_reference is not None:
        expected_teacher = _require_mapping(
            expected_teacher_reference,
            name="expected_teacher_reference",
        )
        if dict(expected_teacher) != mapping["teacher_reference"]:
            raise TargetCacheContractError(
                "teacher-reference context mismatch"
            )

    if expected_provenance_hashes is not None:
        expected_provenance = _require_mapping(
            expected_provenance_hashes,
            name="expected_provenance_hashes",
        )
        if dict(expected_provenance) != mapping["provenance_hashes"]:
            raise TargetCacheContractError(
                "cache provenance context mismatch"
            )

    if expected_stage4_cache_kind is not None:
        if expected_stage4_cache_kind not in STAGE4_CACHE_KINDS:
            raise TargetCacheContractError(
                "expected_stage4_cache_kind is unsupported"
            )

        condition_kind = _stage4_kind_for_condition_id(
            mapping["teacher_reference"]["condition_id"]
        )

        if (
            condition_kind is not None
            and condition_kind != expected_stage4_cache_kind
        ):
            raise TargetCacheContractError(
                "hard/soft cache-kind context mismatch"
            )

    payload_path = _resolve_output_path(
        root,
        mapping["payload"]["path"],
        name="payload.path",
    )
    completion_path = _resolve_output_path(
        root,
        mapping["completion"]["completion_record_path"],
        name="completion.completion_record_path",
    )

    if not payload_path.is_file():
        raise TargetCacheContractError(
            "target-cache payload is missing"
        )
    if not completion_path.is_file():
        raise TargetCacheContractError(
            "target-cache completion record is missing"
        )

    actual_payload_sha = _file_sha256(payload_path)
    if actual_payload_sha != mapping["payload"]["sha256"]:
        raise TargetCacheContractError(
            "target-cache payload SHA-256 mismatch"
        )

    completion_bytes = completion_path.read_bytes()
    actual_completion_sha = hashlib.sha256(
        completion_bytes
    ).hexdigest()

    if (
        actual_completion_sha
        != mapping["completion"]["completion_record_sha256"]
    ):
        raise TargetCacheContractError(
            "target-cache completion-record SHA-256 mismatch"
        )

    try:
        completion = json.loads(
            completion_bytes.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                TargetCacheContractError(
                    f"non-standard JSON constant forbidden: {token}"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise TargetCacheContractError(
            "target-cache completion record is invalid JSON"
        ) from exc

    expected_completion = {
        "schema_version": _COMPLETION_SCHEMA_VERSION,
        "payload_path": mapping["payload"]["path"],
        "payload_sha256": mapping["payload"]["sha256"],
        "example_count": mapping["example_count"],
        "ordered_input_ids_sha256": mapping["input_order"][
            "ordered_input_ids_sha256"
        ],
        "representation_hashes": {
            "raw_logits_sha256": mapping["representations"][
                "raw_logits"
            ]["sha256"],
            "centred_logits_sha256": mapping["representations"][
                "centred_logits"
            ]["sha256"],
            "argmax_sha256": mapping["representations"][
                "argmax"
            ]["sha256"],
            "probabilities_sha256": (
                mapping["representations"]["probabilities"].get("sha256")
                if mapping["representations"]["probabilities"]["present"]
                else None
            ),
        },
    }

    if completion != expected_completion:
        raise TargetCacheContractError(
            "target-cache completion record disagrees with manifest"
        )

    example_count = manifest.example_count
    class_count = manifest.class_count
    raw_dtype_name = mapping["representations"]["raw_logits"]["dtype"]
    probability_record = mapping["representations"]["probabilities"]
    probabilities_present = probability_record["present"]
    probability_dtype_name = (
        probability_record["dtype"]
        if probabilities_present
        else None
    )

    raw_item_size = torch.empty(
        (),
        dtype=_TORCH_DTYPES[raw_dtype_name],
    ).element_size()

    probability_item_size = 0
    if probability_dtype_name is not None:
        probability_item_size = torch.empty(
            (),
            dtype=_TORCH_DTYPES[probability_dtype_name],
        ).element_size()

    input_ids: list[str] = []
    raw_rows: list[torch.Tensor] = []
    probability_rows: list[torch.Tensor] = []

    with payload_path.open("rb") as handle:
        magic = _read_exact(
            handle,
            len(_PAYLOAD_MAGIC),
            name="magic",
        )
        if magic != _PAYLOAD_MAGIC:
            raise TargetCacheContractError(
                "target-cache payload magic mismatch"
            )

        header_length = struct.unpack(
            ">Q",
            _read_exact(handle, 8, name="header length"),
        )[0]

        if header_length <= 0 or header_length > 1024 * 1024:
            raise TargetCacheContractError(
                "target-cache payload header length is invalid"
            )

        header_bytes = _read_exact(
            handle,
            header_length,
            name="header",
        )

        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TargetCacheContractError(
                "target-cache payload header is invalid JSON"
            ) from exc

        expected_header = {
            "schema_version": _PAYLOAD_SCHEMA_VERSION,
            "expected_example_count": example_count,
            "class_count": class_count,
            "logit_dtype": raw_dtype_name,
            "probabilities_present": probabilities_present,
            "probability_dtype": probability_dtype_name,
        }

        if header != expected_header:
            raise TargetCacheContractError(
                "target-cache payload header disagrees with manifest"
            )

        for index in range(example_count):
            input_length = struct.unpack(
                ">I",
                _read_exact(
                    handle,
                    4,
                    name=f"input-id length at example {index}",
                ),
            )[0]

            if input_length <= 0 or input_length > 1024 * 1024:
                raise TargetCacheContractError(
                    "target-cache input-ID length is invalid"
                )

            input_bytes = _read_exact(
                handle,
                input_length,
                name=f"input ID at example {index}",
            )

            try:
                input_id = input_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise TargetCacheContractError(
                    "target-cache input ID is not UTF-8"
                ) from exc

            if not input_id:
                raise TargetCacheContractError(
                    "target-cache input ID must not be empty"
                )

            raw_bytes = _read_exact(
                handle,
                class_count * raw_item_size,
                name=f"raw logits at example {index}",
            )

            raw_row = _tensor_from_bytes(
                raw_bytes,
                dtype_name=raw_dtype_name,
                shape=(class_count,),
            )

            input_ids.append(input_id)
            raw_rows.append(raw_row)

            if probabilities_present:
                probability_bytes = _read_exact(
                    handle,
                    class_count * probability_item_size,
                    name=f"probabilities at example {index}",
                )
                probability_rows.append(
                    _tensor_from_bytes(
                        probability_bytes,
                        dtype_name=probability_dtype_name,
                        shape=(class_count,),
                    )
                )

        if handle.read(1) != b"":
            raise TargetCacheContractError(
                "target-cache payload has trailing bytes"
            )

    if len(set(input_ids)) != len(input_ids):
        raise TargetCacheContractError(
            "target-cache payload contains duplicate input IDs"
        )

    if (
        expected_order is not None
        and tuple(input_ids) != expected_order
    ):
        raise TargetCacheContractError(
            "canonical input sequence does not match expected_input_ids"
        )

    raw_logits = torch.stack(raw_rows, dim=0)

    if not bool(torch.isfinite(raw_logits).all()):
        raise TargetCacheContractError(
            "loaded raw logits contain non-finite values"
        )

    centred_logits = centre_logits(raw_logits)
    argmax = raw_logits.argmax(dim=-1).to(torch.int64)

    probabilities: torch.Tensor | None = None
    if probabilities_present:
        probabilities = torch.stack(probability_rows, dim=0)
        if not bool(torch.isfinite(probabilities).all()):
            raise TargetCacheContractError(
                "loaded probabilities contain non-finite values"
            )

    ordered_digest = hashlib.sha256()
    for input_id in input_ids:
        _update_ordered_input_hash(ordered_digest, input_id)

    if (
        ordered_digest.hexdigest()
        != mapping["input_order"]["ordered_input_ids_sha256"]
    ):
        raise TargetCacheContractError(
            "target-cache canonical input-order hash mismatch"
        )

    if (
        hashlib.sha256(_tensor_bytes(raw_logits)).hexdigest()
        != mapping["representations"]["raw_logits"]["sha256"]
    ):
        raise TargetCacheContractError(
            "target-cache raw-logit hash mismatch"
        )

    if (
        hashlib.sha256(_tensor_bytes(centred_logits)).hexdigest()
        != mapping["representations"]["centred_logits"]["sha256"]
    ):
        raise TargetCacheContractError(
            "target-cache centred-logit hash mismatch"
        )

    if (
        hashlib.sha256(_tensor_bytes(argmax)).hexdigest()
        != mapping["representations"]["argmax"]["sha256"]
    ):
        raise TargetCacheContractError(
            "target-cache argmax hash mismatch"
        )

    if probabilities is not None:
        if (
            hashlib.sha256(_tensor_bytes(probabilities)).hexdigest()
            != probability_record["sha256"]
        ):
            raise TargetCacheContractError(
                "target-cache probability hash mismatch"
            )

    return LoadedTargetCache(
        manifest=manifest,
        input_ids=tuple(input_ids),
        raw_logits=raw_logits,
        centred_logits=centred_logits,
        argmax=argmax,
        probabilities=probabilities,
    )
