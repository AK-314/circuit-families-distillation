"""Validate the Stage 1 follow-up namespace and predecessor link only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from circuit_families.followup_namespace import (
    APPROVED_LOGICAL_ROOTS,
    FOLLOWUP_ROOT,
    NAMESPACE_VERSION,
)
from circuit_families.predecessor_link import (
    PredecessorLinkError,
    load_predecessor_link,
    verify_predecessor_link_physical,
)

DEFAULT_MANIFEST = Path("followup/manifests/predecessor_link_v1.json")
DEFAULT_SCHEMA = Path("schemas/predecessor_link_v1.schema.json")
DEFAULT_NAMESPACE_DOC = Path(
    "docs/distillation_followup/followup_namespace.md"
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_namespace_spec(repository: Path) -> None:
    document = repository / DEFAULT_NAMESPACE_DOC
    if not document.is_file():
        raise ValueError(f"Namespace specification is missing: {document}")

    text = document.read_text(encoding="utf-8")

    if NAMESPACE_VERSION not in text:
        raise ValueError(
            "Namespace specification does not contain the canonical "
            f"namespace version {NAMESPACE_VERSION!r}."
        )

    if f"`{FOLLOWUP_ROOT}/`" not in text:
        raise ValueError(
            "Namespace specification does not declare the follow-up root."
        )

    missing = [
        path.as_posix()
        for path in APPROVED_LOGICAL_ROOTS.values()
        if f"`{path.as_posix()}/`" not in text
    ]
    if missing:
        raise ValueError(
            "Namespace specification is missing approved logical roots: "
            + ", ".join(sorted(missing))
        )

    print("namespace_spec: PASS")
    print(f"namespace_version: {NAMESPACE_VERSION}")
    print(f"approved_logical_root_count: {len(APPROVED_LOGICAL_ROOTS)}")


def _validate_schema_file(repository: Path) -> None:
    path = repository / DEFAULT_SCHEMA
    if not path.is_file():
        raise ValueError(f"Predecessor-link schema is missing: {path}")

    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Predecessor-link schema is invalid JSON: {exc}") from exc

    expected_id = "circuit-families-distillation/predecessor-link/v1"
    if schema.get("$id") != expected_id:
        raise ValueError(
            "Predecessor-link schema ID mismatch: "
            f"expected={expected_id!r}, actual={schema.get('$id')!r}"
        )

    if schema.get("additionalProperties") is not False:
        raise ValueError(
            "Predecessor-link root schema must reject unknown fields."
        )

    version = schema.get("properties", {}).get("schema_version", {}).get("const")
    if version != 1:
        raise ValueError(
            f"Predecessor-link schema version must be 1; received {version!r}."
        )

    print("schema_file: PASS")
    print(f"schema_id: {expected_id}")
    print("schema_additional_properties: REJECTED")


def _validate_manifest(repository: Path) -> dict:
    path = repository / DEFAULT_MANIFEST
    record = load_predecessor_link(path)

    print("canonical_manifest: PASS")
    print(f"manifest_path: {DEFAULT_MANIFEST.as_posix()}")
    print(
        "teacher_seeds: "
        + str([row["teacher_seed"] for row in record["teacher_runs"]])
    )
    print(
        "stage3_checkpoint_registry_resolved: "
        + str(record["stage3_checkpoint_registry"]["resolved"])
    )
    print(
        "scientific_execution: "
        + str(record["metadata"]["scientific_execution"])
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate only the Stage 1 follow-up namespace and predecessor "
            "provenance contract. No scientific computation is performed."
        )
    )
    parser.add_argument(
        "--predecessor-root",
        type=Path,
        default=None,
        help=(
            "Optional physical predecessor checkout. If omitted, portable "
            "schema/manifest validation still runs and physical verification "
            "is explicitly skipped."
        ),
    )
    args = parser.parse_args()

    repository = _repository_root()

    print("===== STAGE 1 VALIDATE-ONLY =====")
    print(f"repository: {repository}")

    try:
        _validate_namespace_spec(repository)
        _validate_schema_file(repository)
        record = _validate_manifest(repository)

        if args.predecessor_root is None:
            print(
                "physical_predecessor_verification: SKIPPED "
                "(WARN: --predecessor-root not supplied)"
            )
            print("scientific_computation: NO")
            print("files_written: NO")
            print("overall: PASS_WITH_WARNING")
            return 0

        verify_predecessor_link_physical(
            record,
            predecessor_root=args.predecessor_root,
        )
        print("physical_predecessor_verification: PASS")
        print(f"physical_predecessor_root: {args.predecessor_root.resolve()}")
        print("scientific_computation: NO")
        print("files_written: NO")
        print("overall: PASS")
        return 0

    except (ValueError, PredecessorLinkError, OSError) as exc:
        print("overall: FAIL")
        print(f"reason: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
