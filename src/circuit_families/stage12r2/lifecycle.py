"""Versioned Stage 12-R2 technical lifecycle records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from circuit_families.stage12r2.contracts import (
    canonical_sha256,
    validate_technical_record_payload,
)

LIFECYCLE_VERSION = "stage12r2-lifecycle/v1"
TECHNICAL_CLASSIFICATION = "synthetic_technical_only"


@dataclass(frozen=True)
class Stage12R2LifecycleRecord:
    basis_hash: str
    model_identity: str
    exact_ledger_reference: str
    transform_hash: str | None = None
    partition_hash: str | None = None
    rd004_basis_panel: None = None
    rd004_partition_seed: None = None
    rd004_rotation_seed: None = None
    rd004_model_assignment: None = None
    classification: str = TECHNICAL_CLASSIFICATION
    scientific_data: bool = False
    production_eligible: bool = False
    record_version: str = LIFECYCLE_VERSION

    def __post_init__(self) -> None:
        if self.record_version != LIFECYCLE_VERSION:
            raise ValueError("unsupported Stage 12-R2 lifecycle version")
        if self.classification != TECHNICAL_CLASSIFICATION:
            raise ValueError("Stage 12-R2 lifecycle must remain technical-only")
        if self.scientific_data:
            raise ValueError("Stage 12-R2 requires scientific_data=false")
        if self.production_eligible:
            raise ValueError("Stage 12-R2 requires production_eligible=false")
        for name in ("basis_hash", "model_identity", "exact_ledger_reference"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if any(
            value is not None
            for value in (
                self.rd004_basis_panel,
                self.rd004_partition_seed,
                self.rd004_rotation_seed,
                self.rd004_model_assignment,
            )
        ):
            raise ValueError("RD-004 production fields must remain unresolved")

    def payload_without_hash(self) -> dict[str, object]:
        return {
            "record_version": self.record_version,
            "classification": self.classification,
            "scientific_data": self.scientific_data,
            "production_eligible": self.production_eligible,
            "basis_hash": self.basis_hash,
            "model_identity": self.model_identity,
            "exact_ledger_reference": self.exact_ledger_reference,
            "transform_hash": self.transform_hash,
            "partition_hash": self.partition_hash,
            "rd004": {
                "basis_panel": self.rd004_basis_panel,
                "partition_seed": self.rd004_partition_seed,
                "rotation_seed": self.rd004_rotation_seed,
                "model_assignment": self.rd004_model_assignment,
            },
        }

    @property
    def record_hash(self) -> str:
        return canonical_sha256(self.payload_without_hash())

    def to_record(self) -> dict[str, object]:
        payload = self.payload_without_hash()
        validate_technical_record_payload(payload)
        payload["record_hash"] = self.record_hash
        return payload


def lifecycle_record_from_mapping(
    record: Mapping[str, Any],
) -> Stage12R2LifecycleRecord:
    expected = {
        "record_version",
        "classification",
        "scientific_data",
        "production_eligible",
        "basis_hash",
        "model_identity",
        "exact_ledger_reference",
        "transform_hash",
        "partition_hash",
        "rd004",
        "record_hash",
    }
    if set(record) != expected:
        raise ValueError("lifecycle record fields do not match contract")
    validate_technical_record_payload(record)

    rd004 = record["rd004"]
    if not isinstance(rd004, Mapping):
        raise ValueError("rd004 must be a mapping")
    if set(rd004) != {
        "basis_panel",
        "partition_seed",
        "rotation_seed",
        "model_assignment",
    }:
        raise ValueError("RD-004 fields do not match unresolved contract")

    obj = Stage12R2LifecycleRecord(
        record_version=str(record["record_version"]),
        classification=str(record["classification"]),
        scientific_data=bool(record["scientific_data"]),
        production_eligible=bool(record["production_eligible"]),
        basis_hash=str(record["basis_hash"]),
        model_identity=str(record["model_identity"]),
        exact_ledger_reference=str(record["exact_ledger_reference"]),
        transform_hash=(
            None if record["transform_hash"] is None else str(record["transform_hash"])
        ),
        partition_hash=(
            None if record["partition_hash"] is None else str(record["partition_hash"])
        ),
        rd004_basis_panel=rd004["basis_panel"],
        rd004_partition_seed=rd004["partition_seed"],
        rd004_rotation_seed=rd004["rotation_seed"],
        rd004_model_assignment=rd004["model_assignment"],
    )
    if str(record["record_hash"]) != obj.record_hash:
        raise ValueError("lifecycle record hash is stale or incorrect")
    return obj
