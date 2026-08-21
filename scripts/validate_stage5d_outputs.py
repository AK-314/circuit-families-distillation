#!/usr/bin/env python3
"""Validate and reconstruct synthetic Stage 5D outputs deterministically."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_ROOT = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_ROOT / "bin/python"
SRC_ROOT = REPO_ROOT / "src"


def _ensure_repository_runtime() -> None:
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_parts = (
        []
        if not existing_pythonpath
        else existing_pythonpath.split(os.pathsep)
    )
    if str(SRC_ROOT) not in pythonpath_parts:
        pythonpath_parts.insert(0, str(SRC_ROOT))
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    if Path(sys.prefix).resolve() == VENV_ROOT.resolve():
        return
    if os.environ.get("STAGE5D_REEXECUTED") == "1":
        raise RuntimeError("Stage 5D CLI could not enter repository virtualenv")
    if not VENV_PYTHON.is_file():
        raise RuntimeError(f"repository virtualenv is missing: {VENV_PYTHON}")

    environment = dict(os.environ)
    environment["STAGE5D_REEXECUTED"] = "1"
    os.execve(
        VENV_PYTHON,
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


_ensure_repository_runtime()
sys.dont_write_bytecode = True
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from circuit_families.stage5d import (  # noqa: E402
    Stage5DOutputError,
    build_stage5d_output_bundle,
    load_and_normalize_ingestion,
    load_technical_analysis_profile_set,
    validate_stage5d_output_bundle,
    write_stage5d_output_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate deterministic Stage 5D synthetic outputs. The command "
            "is read-only unless --output-root names a temporary directory."
        )
    )
    parser.add_argument("--ingestion", required=True, type=Path)
    parser.add_argument("--profile-set", required=True, type=Path)
    parser.add_argument(
        "--profile-id",
        required=True,
        help="Explicit technical fixture profile; no profile is selected by default.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional dedicated directory under the system temporary root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized = load_and_normalize_ingestion(args.ingestion)
        profile = load_technical_analysis_profile_set(
            args.profile_set
        ).require(args.profile_id)
        bundle = build_stage5d_output_bundle(normalized, profile)
        validate_stage5d_output_bundle(bundle, normalized, profile)

        written = "NO"
        if args.output_root is not None:
            write_stage5d_output_bundle(bundle, args.output_root)
            written = "YES"
    except (OSError, RuntimeError, ValueError, Stage5DOutputError) as exc:
        print(f"STAGE5D_OUTPUT_VALIDATION=FAIL reason={exc}", file=sys.stderr)
        return 2

    print(
        "STAGE5D_OUTPUT_VALIDATION=PASS "
        f"PROFILE_ID={profile.profile_id} "
        f"BUNDLE_SHA256={bundle['sha256']} "
        f"OUTPUT_WRITTEN={written} "
        "SCIENTIFIC_DATA=NO"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
