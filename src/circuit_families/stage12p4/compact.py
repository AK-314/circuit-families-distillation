"""Deterministic bit-packed masks and streaming schema-bound ledgers."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import struct
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final

from circuit_families.stage12p3.records import canonical_json_bytes, require_reference

from .codec import atomic_encode, decode_bytes
from .records import CodecProfile, Stage12P4Error, sha256_file, validate_technical_payload

MASK_MAGIC: Final = b"S12P4M1\n"
LEDGER_MAGIC: Final = b"S12P4L1\n"
MASK_VERSION: Final = "stage12p4-compact-mask/v1"
LEDGER_VERSION: Final = "stage12p4-compact-ledger/v1"
LEDGER_SCHEMA_VERSION: Final = "stage12p4-ledger-schema/v1"
FIELD_TYPES: Final = frozenset({"string", "integer", "number", "boolean"})
ROW_STATES: Final = frozenset({"complete", "failed", "unavailable", "censored"})


@dataclass(frozen=True)
class CompactObjectEvidence:
    path: Path
    logical_byte_length: int
    logical_sha256: str
    compact_byte_length: int
    compact_sha256: str
    row_count: int | None = None


@dataclass(frozen=True)
class LedgerField:
    name: str
    field_type: str
    nullable: bool = False
    allow_negative: bool = True

    def __post_init__(self) -> None:
        require_reference(self.name, label="ledger field name")
        if self.field_type not in FIELD_TYPES:
            raise Stage12P4Error("unsupported ledger field type")
        if self.allow_negative is not True and self.field_type not in {"integer", "number"}:
            raise Stage12P4Error("negative policy applies only to numeric fields")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MetricSchema:
    reference: str
    fields: tuple[LedgerField, ...]
    key_fields: tuple[str, ...]
    schema_version: str = LEDGER_SCHEMA_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != LEDGER_SCHEMA_VERSION:
            raise Stage12P4Error("unsupported ledger schema")
        require_reference(self.reference, label="ledger schema reference")
        if not self.fields:
            raise Stage12P4Error("ledger schema requires fields")
        names = tuple(field.name for field in self.fields)
        if len(set(names)) != len(names):
            raise Stage12P4Error("ledger field names must be unique")
        if not self.key_fields or len(set(self.key_fields)) != len(self.key_fields):
            raise Stage12P4Error("ledger key fields must be non-empty and unique")
        if not set(self.key_fields).issubset(names):
            raise Stage12P4Error("ledger key fields must be declared fields")
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P4Error("ledger schema must remain technical-only")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "reference": self.reference,
            "fields": [field.to_mapping() for field in self.fields],
            "key_fields": list(self.key_fields),
            "scientific_data": False,
            "production_eligible": False,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MetricSchema:
        if set(value) != {
            "schema_version",
            "reference",
            "fields",
            "key_fields",
            "scientific_data",
            "production_eligible",
        }:
            raise Stage12P4Error("ledger schema fields mismatch")
        raw_fields = value["fields"]
        if not isinstance(raw_fields, list) or any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "field_type", "nullable", "allow_negative"}
            for item in raw_fields
        ):
            raise Stage12P4Error("ledger field descriptors mismatch")
        raw_keys = value["key_fields"]
        if not isinstance(raw_keys, list):
            raise Stage12P4Error("ledger key fields must be a list")
        return cls(
            reference=str(value["reference"]),
            fields=tuple(LedgerField(**dict(item)) for item in raw_fields),
            key_fields=tuple(str(item) for item in raw_keys),
            schema_version=str(value["schema_version"]),
            scientific_data=value["scientific_data"],
            production_eligible=value["production_eligible"],
        )

    def validate_row(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        names = tuple(field.name for field in self.fields)
        if set(row) != set(names):
            raise Stage12P4Error("ledger row fields do not match closed schema")
        values: list[Any] = []
        for field in self.fields:
            value = row[field.name]
            if value is None:
                if not field.nullable:
                    raise Stage12P4Error(f"ledger field {field.name!r} is not nullable")
                values.append(None)
                continue
            if field.field_type == "string":
                valid = isinstance(value, str)
            elif field.field_type == "boolean":
                valid = isinstance(value, bool)
            elif field.field_type == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not valid:
                raise Stage12P4Error(f"ledger field {field.name!r} has wrong type")
            if field.field_type in {"integer", "number"}:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise Stage12P4Error("non-finite ledger values are forbidden")
                if numeric < 0 and not field.allow_negative:
                    raise Stage12P4Error(f"negative ledger field {field.name!r} is forbidden")
            if field.name == "status" and value not in ROW_STATES:
                raise Stage12P4Error("ledger status must preserve an explicit supported state")
            values.append(value)
        validate_technical_payload(row)
        return tuple(values)

    def key_for(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row[name] for name in self.key_fields)


def _frame(magic: bytes, header: Mapping[str, Any], body: bytes) -> bytes:
    header_bytes = canonical_json_bytes(header)
    return magic + struct.pack(">I", len(header_bytes)) + header_bytes + body


def _unframe(raw: bytes, magic: bytes) -> tuple[dict[str, Any], bytes]:
    if len(raw) < len(magic) + 4 or raw[: len(magic)] != magic:
        raise Stage12P4Error("compact object magic/version mismatch")
    length = struct.unpack(">I", raw[len(magic) : len(magic) + 4])[0]
    start = len(magic) + 4
    end = start + length
    if end > len(raw):
        raise Stage12P4Error("compact object header is truncated")
    try:
        header = json.loads(raw[start:end])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage12P4Error("compact object header is invalid") from exc
    if not isinstance(header, dict) or canonical_json_bytes(header) != raw[start:end]:
        raise Stage12P4Error("compact object header is not canonical")
    return header, raw[end:]


def _pack_bits(values: Sequence[int]) -> bytes:
    packed = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if isinstance(value, bool) or value not in (0, 1):
            raise Stage12P4Error("mask values must be integer zero or one")
        if value:
            packed[index // 8] |= 1 << (7 - (index % 8))
    return bytes(packed)


def _logical_mask_bytes(
    values: Sequence[int], component_universe: str, basis_identity: str
) -> bytes:
    return canonical_json_bytes(
        {
            "component_universe": component_universe,
            "basis_identity": basis_identity,
            "values": list(values),
        }
    )


def write_mask(
    path: Path,
    values: Sequence[int],
    *,
    component_universe: str,
    basis_identity: str,
    profile: CodecProfile,
) -> CompactObjectEvidence:
    require_reference(component_universe, label="component universe")
    require_reference(basis_identity, label="basis identity")
    logical = _logical_mask_bytes(values, component_universe, basis_identity)
    body = _pack_bits(values)
    header = {
        "schema_version": MASK_VERSION,
        "storage_encoding": "bitpack-msb0/v1",
        "component_universe": component_universe,
        "basis_identity": basis_identity,
        "component_count": len(values),
        "padding_rule": "zero-low-order-padding-bits/v1",
        "logical_mask_sha256": hashlib.sha256(logical).hexdigest(),
        "packed_byte_length": len(body),
        "packed_sha256": hashlib.sha256(body).hexdigest(),
        "scientific_data": False,
        "production_eligible": False,
    }
    raw = _frame(MASK_MAGIC, header, body)
    atomic_encode(path, (raw,), profile)
    compact_size, compact_hash = sha256_file(path)
    return CompactObjectEvidence(
        path,
        len(logical),
        hashlib.sha256(logical).hexdigest(),
        compact_size,
        compact_hash,
    )


def read_mask(
    path: Path,
    *,
    profile: CodecProfile,
    expected_component_universe: str | None = None,
    expected_basis_identity: str | None = None,
) -> tuple[int, ...]:
    raw = decode_bytes(path.read_bytes(), profile)
    header, body = _unframe(raw, MASK_MAGIC)
    required = {
        "schema_version",
        "storage_encoding",
        "component_universe",
        "basis_identity",
        "component_count",
        "padding_rule",
        "logical_mask_sha256",
        "packed_byte_length",
        "packed_sha256",
        "scientific_data",
        "production_eligible",
    }
    if set(header) != required or header["schema_version"] != MASK_VERSION:
        raise Stage12P4Error("compact mask header schema mismatch")
    if header["storage_encoding"] != "bitpack-msb0/v1":
        raise Stage12P4Error("compact mask encoding ambiguity")
    if header["padding_rule"] != "zero-low-order-padding-bits/v1":
        raise Stage12P4Error("unsupported mask padding rule")
    if header["scientific_data"] is not False or header["production_eligible"] is not False:
        raise Stage12P4Error("compact mask crossed the technical boundary")
    if expected_component_universe is not None and (
        header["component_universe"] != expected_component_universe
    ):
        raise Stage12P4Error("component universe mismatch")
    if expected_basis_identity is not None and header["basis_identity"] != expected_basis_identity:
        raise Stage12P4Error("component basis mismatch")
    count = header["component_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise Stage12P4Error("invalid mask component count")
    if len(body) != (count + 7) // 8 or header["packed_byte_length"] != len(body):
        raise Stage12P4Error("packed mask length mismatch")
    if header["packed_sha256"] != hashlib.sha256(body).hexdigest():
        raise Stage12P4Error("packed mask hash mismatch")
    padding = len(body) * 8 - count
    if padding and body and body[-1] & ((1 << padding) - 1):
        raise Stage12P4Error("packed mask padding is nonzero")
    values = tuple((body[index // 8] >> (7 - index % 8)) & 1 for index in range(count))
    logical = _logical_mask_bytes(
        values, str(header["component_universe"]), str(header["basis_identity"])
    )
    if header["logical_mask_sha256"] != hashlib.sha256(logical).hexdigest():
        raise Stage12P4Error("logical mask hash mismatch")
    return values


class MetricLedgerWriter:
    """Append rows to a bounded-memory partial, then atomically seal one object."""

    def __init__(
        self,
        path: Path,
        schema: MetricSchema,
        *,
        context: Mapping[str, Any],
        profile: CodecProfile,
    ) -> None:
        validate_technical_payload(context)
        self.path = path
        self.schema = schema
        self.context = dict(context)
        self.profile = profile
        self.partial_path = path.with_name(f".{path.name}.rows.partial")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.partial_path.open("xb")
        self._digest = hashlib.sha256()
        self._byte_length = 0
        self._row_count = 0
        self._previous_key: tuple[Any, ...] | None = None
        self._finalized = False

    def append(self, row: Mapping[str, Any]) -> None:
        if self._finalized:
            raise Stage12P4Error("cannot append to finalized ledger")
        values = self.schema.validate_row(row)
        key = self.schema.key_for(row)
        if self._previous_key is not None and key <= self._previous_key:
            raise Stage12P4Error("ledger rows must be in strict canonical key order")
        encoded = (
            json.dumps(
                values,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
        self._handle.write(encoded)
        self._digest.update(encoded)
        self._byte_length += len(encoded)
        self._row_count += 1
        self._previous_key = key

    def finalize(self) -> CompactObjectEvidence:
        if self._finalized:
            raise Stage12P4Error("ledger is already finalized")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        header = {
            "schema_version": LEDGER_VERSION,
            "storage_encoding": "canonical-jsonl-row-array/v1",
            "schema": self.schema.to_mapping(),
            "context": self.context,
            "row_count": self._row_count,
            "row_byte_length": self._byte_length,
            "row_content_sha256": self._digest.hexdigest(),
            "scientific_data": False,
            "production_eligible": False,
        }
        header_bytes = canonical_json_bytes(header)

        def chunks() -> Iterator[bytes]:
            yield LEDGER_MAGIC
            yield struct.pack(">I", len(header_bytes))
            yield header_bytes
            with self.partial_path.open("rb") as rows:
                while block := rows.read(1024 * 1024):
                    yield block

        try:
            atomic_encode(self.path, chunks(), self.profile)
        finally:
            if self.partial_path.exists():
                self.partial_path.unlink()
        self._finalized = True
        compact_size, compact_hash = sha256_file(self.path)
        return CompactObjectEvidence(
            self.path,
            self._byte_length,
            self._digest.hexdigest(),
            compact_size,
            compact_hash,
            self._row_count,
        )

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()


@contextmanager
def _decoded_handle(path: Path, profile: CodecProfile) -> Iterator[BinaryIO]:
    raw = path.open("rb")
    try:
        if profile.codec == "gzip":
            decoded = gzip.GzipFile(fileobj=raw, mode="rb")
            try:
                yield decoded
            except (EOFError, gzip.BadGzipFile, zlib.error) as exc:
                raise Stage12P4Error("compressed ledger is corrupt or truncated") from exc
            finally:
                decoded.close()
        else:
            yield raw
    finally:
        raw.close()
def _read_ledger_header(handle: BinaryIO) -> tuple[dict[str, Any], MetricSchema]:
    if handle.read(len(LEDGER_MAGIC)) != LEDGER_MAGIC:
        raise Stage12P4Error("compact ledger magic/version mismatch")
    length_bytes = handle.read(4)
    if len(length_bytes) != 4:
        raise Stage12P4Error("compact ledger header length is truncated")
    length = struct.unpack(">I", length_bytes)[0]
    raw = handle.read(length)
    if len(raw) != length:
        raise Stage12P4Error("compact ledger header is truncated")
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage12P4Error("compact ledger header is invalid") from exc
    required = {
        "schema_version",
        "storage_encoding",
        "schema",
        "context",
        "row_count",
        "row_byte_length",
        "row_content_sha256",
        "scientific_data",
        "production_eligible",
    }
    if not isinstance(header, dict) or set(header) != required:
        raise Stage12P4Error("compact ledger header schema mismatch")
    if canonical_json_bytes(header) != raw or header["schema_version"] != LEDGER_VERSION:
        raise Stage12P4Error("compact ledger header is not canonical")
    if header["storage_encoding"] != "canonical-jsonl-row-array/v1":
        raise Stage12P4Error("compact ledger encoding ambiguity")
    if header["scientific_data"] is not False or header["production_eligible"] is not False:
        raise Stage12P4Error("compact ledger crossed the technical boundary")
    if not isinstance(header["schema"], Mapping):
        raise Stage12P4Error("compact ledger schema descriptor is invalid")
    return header, MetricSchema.from_mapping(header["schema"])


def read_ledger_header(path: Path, *, profile: CodecProfile) -> dict[str, Any]:
    with _decoded_handle(path, profile) as handle:
        header, _ = _read_ledger_header(handle)
        return header


def iter_ledger_rows(
    path: Path,
    *,
    profile: CodecProfile,
    expected_schema: MetricSchema | None = None,
) -> Iterator[dict[str, Any]]:
    with _decoded_handle(path, profile) as handle:
        header, schema = _read_ledger_header(handle)
        if expected_schema is not None and schema.to_mapping() != expected_schema.to_mapping():
            raise Stage12P4Error("ledger schema drift detected")
        digest = hashlib.sha256()
        byte_length = 0
        row_count = 0
        previous_key: tuple[Any, ...] | None = None
        names = tuple(field.name for field in schema.fields)
        for raw in handle:
            digest.update(raw)
            byte_length += len(raw)
            try:
                values = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Stage12P4Error("ledger row is invalid JSON") from exc
            if not isinstance(values, list) or len(values) != len(names):
                raise Stage12P4Error("ledger row width mismatch")
            row = dict(zip(names, values, strict=True))
            schema.validate_row(row)
            key = schema.key_for(row)
            if previous_key is not None and key <= previous_key:
                raise Stage12P4Error("ledger row order is not canonical")
            previous_key = key
            row_count += 1
            yield row
        if row_count != header["row_count"]:
            raise Stage12P4Error("ledger row count mismatch")
        if byte_length != header["row_byte_length"]:
            raise Stage12P4Error("ledger row byte length mismatch")
        if digest.hexdigest() != header["row_content_sha256"]:
            raise Stage12P4Error("ledger row content hash mismatch")


def write_ledger(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: MetricSchema,
    context: Mapping[str, Any],
    profile: CodecProfile,
) -> CompactObjectEvidence:
    writer = MetricLedgerWriter(path, schema, context=context, profile=profile)
    try:
        for row in rows:
            writer.append(row)
        return writer.finalize()
    except Exception:
        writer.abort()
        raise
