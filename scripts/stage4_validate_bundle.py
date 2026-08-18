#!/usr/bin/env python3
"""Validate a Stage 4 record bundle without generating or modifying artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from circuit_families.stage4_condition_identity import (  # noqa: E402
    Stage3AvailabilityIndex,
)
from circuit_families.stage4_schema_common import (  # noqa: E402
    CommonSchemaContract,
    Stage4SchemaError,
)
from circuit_families.stage4_schema_graph import (  # noqa: E402
    validate_stage4_record_graph,
)

VOCAB_PATH = (
    REPO_ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
)
IDENTITY_SPEC_PATH = (
    REPO_ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
)
REGISTRY_PATH = (
    REPO_ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)
DECISIONS_PATH = (
    REPO_ROOT / "followup/configs/stage2_unresolved_decisions_v1.json"
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path}: line={exc.lineno} column={exc.colno}"
        ) from exc


def _current_unresolved_decisions(
    decisions: Any,
) -> set[str]:
    wanted = {f"UD-{number:03d}" for number in range(3, 15)}
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            ident = value.get("decision_id")

            if (
                ident in wanted
                and value.get("status") == "unresolved"
            ):
                found.add(ident)

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(decisions)

    if found != wanted:
        missing = sorted(wanted - found)
        extra = sorted(found - wanted)
        raise ValueError(
            "Stage 4 authority requires UD-003 through UD-014 unresolved: "
            f"missing={missing} extra={extra}"
        )

    return found


def _records_from_bundle(bundle: Any) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(bundle, dict):
        raise ValueError("bundle root must be a JSON object")

    records = bundle.get("records_by_sha256")

    if not isinstance(records, dict):
        raise ValueError(
            "bundle must contain object field 'records_by_sha256'"
        )

    return records


def validate_bundle_path(bundle_path: Path) -> tuple[int, int]:
    vocab = _load_json(VOCAB_PATH)
    identity_spec = _load_json(IDENTITY_SPEC_PATH)
    registry = _load_json(REGISTRY_PATH)
    decisions = _load_json(DECISIONS_PATH)
    bundle = _load_json(bundle_path)

    stage3 = Stage3AvailabilityIndex.from_registry(registry)
    contract = CommonSchemaContract.from_specs(
        vocab,
        identity_spec,
    )

    registry_sha256 = hashlib.sha256(
        REGISTRY_PATH.read_bytes()
    ).hexdigest()

    unresolved = _current_unresolved_decisions(decisions)
    records = _records_from_bundle(bundle)

    validate_stage4_record_graph(
        records,
        contract=contract,
        stage3=stage3,
        stage3_registry=registry,
        stage3_registry_sha256=registry_sha256,
        current_unresolved_decision_ids=unresolved,
    )

    record_types = {
        record["record_type"]
        for record in records.values()
    }

    return len(records), len(record_types)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Stage 4 record bundle against the frozen common "
            "identity/schema/graph contract. This command is read-only."
        )
    )

    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Path to a JSON bundle containing records_by_sha256.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        record_count, record_type_count = validate_bundle_path(
            args.bundle
        )
    except (
        OSError,
        ValueError,
        Stage4SchemaError,
    ) as exc:
        print(
            f"FAIL stage4_bundle_validation reason={exc}",
            file=sys.stderr,
        )
        return 2

    print(
        "PASS stage4_bundle_validation "
        f"records={record_count} "
        f"record_types={record_type_count} "
        "scientific_computation=NO "
        "artifact_generation=NO"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
