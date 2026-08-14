#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from circuit_families.stage2_method_development_firewall import (  # noqa: E402
    load_firewall_register,
)
from circuit_families.stage2_scientific_skeleton import (  # noqa: E402
    load_scientific_skeleton,
)
from circuit_families.stage2_unresolved_decisions import (  # noqa: E402
    load_unresolved_decisions,
)


EXPECTED_BASE_COMMIT = "9118ecd239753c54fa5c66766e5d80b54d2a6259"

SKELETON = Path("followup/manifests/stage2_scientific_skeleton_freeze_v1.json")
UNRESOLVED = Path("followup/configs/stage2_unresolved_decisions_v1.json")
FIREWALL = Path("followup/manifests/stage2_excluded_development_register_v1.json")

CANONICAL_TEXT_RECORDS = [
    Path("docs/distillation_followup/distillation_experimental_protocol_draft.md"),
    Path("docs/distillation_followup/stage2_scientific_skeleton.md"),
    Path("docs/distillation_followup/stage2_method_development_firewall.md"),
    UNRESOLVED,
    SKELETON,
    FIREWALL,
]

AUTHORITY_SOURCES = {
    "implementation_master": Path(
        "docs/distillation_followup/distillation_implementation_master.md"
    ),
    "protocol": Path(
        "docs/distillation_followup/distillation_experimental_protocol_draft.md"
    ),
    "workflow": Path("workflow.md"),
    "workstream_A": Path(
        "docs/distillation_followup/workstreams/ws_a_protocol_registry.md"
    ),
    "predecessor_link": Path("followup/manifests/predecessor_link_v1.json"),
}

PRIVATE_PATH_PATTERNS = (
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\"),
)


class PortableValidationError(RuntimeError):
    pass


def repo_path(relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise PortableValidationError(
            f"non-portable repository path requested: {relative}"
        )
    return REPO_ROOT / relative


def hash_object(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "hash-object", str(path)],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise PortableValidationError(
            f"cannot compute Git blob identity for {path}"
        ) from exc


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_authority(records: list[dict]) -> None:
    expected = {}

    for authority_id, relative in AUTHORITY_SOURCES.items():
        physical = repo_path(relative)

        if not physical.is_file():
            raise PortableValidationError(
                f"missing authority source: {relative}"
            )

        expected[authority_id] = {
            "repository_path": relative.as_posix(),
            "git_blob": hash_object(physical),
            "sha256": sha256(physical),
        }

    for index, record in enumerate(records):
        authority = {
            entry["authority_id"]: entry
            for entry in record["authority"]
        }

        if set(authority) != set(expected):
            raise PortableValidationError(
                f"record {index} authority roster mismatch: "
                f"expected={sorted(expected)} actual={sorted(authority)}"
            )

        for authority_id, want in expected.items():
            got = authority[authority_id]

            for field in ("repository_path", "git_blob", "sha256"):
                if got[field] != want[field]:
                    raise PortableValidationError(
                        f"record {index} stale authority "
                        f"{authority_id}.{field}: "
                        f"recorded={got[field]!r} actual={want[field]!r}"
                    )


def validate_portable_content() -> None:
    violations = []

    for relative in CANONICAL_TEXT_RECORDS:
        physical = repo_path(relative)

        if not physical.is_file():
            raise PortableValidationError(
                f"missing canonical Stage 2 record: {relative}"
            )

        text = physical.read_text(encoding="utf-8")

        for pattern in PRIVATE_PATH_PATTERNS:
            if pattern.search(text):
                violations.append(
                    {
                        "repository_path": relative.as_posix(),
                        "pattern": pattern.pattern,
                    }
                )

    if violations:
        raise PortableValidationError(
            f"private absolute path found in canonical Stage 2 records: "
            f"{violations}"
        )


def validate_boundary(
    skeleton: dict,
    unresolved: dict,
    firewall: dict,
) -> None:
    if skeleton["metadata"]["created_from_commit"] != EXPECTED_BASE_COMMIT:
        raise PortableValidationError(
            "skeleton created_from_commit does not match Stage 2 prerequisite"
        )

    if skeleton["metadata"]["scientific_execution"] is not False:
        raise PortableValidationError(
            "Stage 2 skeleton must not authorize scientific execution"
        )

    if skeleton["freeze_scope"] != {
        "is_partial_freeze": True,
        "numeric_protocol_fully_frozen": False,
        "stage3_started": False,
        "teacher_registry_frozen": False,
    }:
        raise PortableValidationError("Stage 2 partial-freeze boundary drifted")

    claims = skeleton["claims_boundary"]

    for key in (
        "endpoint_outputs_present",
        "full_numeric_protocol_frozen",
        "production_ready",
        "scientific_results_present",
        "stage3_authorized",
    ):
        if claims[key] is not False:
            raise PortableValidationError(
                f"Stage 2 claims boundary violated: {key}={claims[key]!r}"
            )

    if any(d["status"] != "unresolved" for d in unresolved["decisions"]):
        raise PortableValidationError(
            "Stage 2 unresolved decision was prematurely resolved"
        )

    if unresolved["coverage"]["all_decisions_unresolved"] is not True:
        raise PortableValidationError(
            "all_decisions_unresolved flag must remain true"
        )

    if unresolved["coverage"]["recommended_values_are_nonbinding"] is not True:
        raise PortableValidationError(
            "recommended values must remain nonbinding"
        )

    if firewall["firewall"]["primary_analysis_eligible"] is not False:
        raise PortableValidationError(
            "development output cannot be primary-analysis eligible"
        )

    if firewall["firewall"]["scientific_selection_eligible"] is not False:
        raise PortableValidationError(
            "development output cannot select scientific protocol values"
        )

    if firewall["firewall"]["post_freeze_regeneration_required"] is not True:
        raise PortableValidationError(
            "post-freeze regeneration must remain required"
        )


def validate_excluded_output_absence() -> None:
    root = repo_path(Path("followup/excluded_development"))

    if not root.exists():
        return

    files = [p for p in root.rglob("*") if p.is_file()]

    if files:
        raise PortableValidationError(
            "unexpected Stage 2 excluded-development output exists: "
            + ", ".join(
                p.relative_to(REPO_ROOT).as_posix()
                for p in files
            )
        )


def main() -> int:
    print("stage=2")
    print(f"repository_root={REPO_ROOT}")
    print(f"invocation_cwd={Path.cwd()}")
    print("scientific_execution=NO")

    skeleton_path = repo_path(SKELETON)
    unresolved_path = repo_path(UNRESOLVED)
    firewall_path = repo_path(FIREWALL)

    try:
        skeleton = load_scientific_skeleton(skeleton_path)
        print("PASS scientific skeleton validates")

        unresolved = load_unresolved_decisions(unresolved_path)
        print("PASS unresolved decision register validates")

        firewall = load_firewall_register(firewall_path)
        print("PASS method-development firewall validates")

        validate_authority([skeleton, unresolved, firewall])
        print("PASS authority identities match current repository content")

        validate_boundary(skeleton, unresolved, firewall)
        print("PASS partial-freeze and no-scientific-execution boundary holds")

        validate_portable_content()
        print("PASS canonical records contain no private absolute paths")

        validate_excluded_output_absence()
        print("PASS no endpoint-producing excluded-development output exists")

    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}")
        print("STAGE2_PORTABLE_VALIDATION: FAIL")
        return 1

    print("STAGE2_PORTABLE_VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
