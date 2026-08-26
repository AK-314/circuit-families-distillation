#!/usr/bin/env python3
"""Validate the complete portable synthetic Stage 7A integration."""

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
        else existing_pythonpath.split(
            os.pathsep
        )
    )

    if str(SRC_ROOT) not in pythonpath_parts:
        pythonpath_parts.insert(
            0,
            str(SRC_ROOT),
        )

    os.environ["PYTHONPATH"] = os.pathsep.join(
        pythonpath_parts
    )

    os.environ[
        "PYTHONDONTWRITEBYTECODE"
    ] = "1"

    if Path(
        sys.prefix
    ).resolve() == VENV_ROOT.resolve():
        return

    if not VENV_PYTHON.is_file():
        return

    if os.environ.get(
        "STAGE7A_REEXECUTED"
    ) == "1":
        raise RuntimeError(
            "Stage 7A CLI could not enter repository virtualenv"
        )

    environment = dict(
        os.environ
    )

    environment[
        "STAGE7A_REEXECUTED"
    ] = "1"

    os.execve(
        VENV_PYTHON,
        [
            str(VENV_PYTHON),
            str(
                Path(__file__).resolve()
            ),
            *sys.argv[1:],
        ],
        environment,
    )


_ensure_repository_runtime()

sys.dont_write_bytecode = True

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

from circuit_families.stage7 import (  # noqa: E402
    Stage7PortableE2EError,
    TechnicalDistillationFixtureConfig,
    run_portable_stage7_fixture,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Stage 7A synthetic technical integration. "
            "Runtime writes are restricted to one explicit system-temporary root."
        )
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help=(
            "Dedicated non-existing directory beneath the system temporary root."
        ),
    )

    parser.add_argument(
        "--teacher-seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--phase",
        required=True,
    )

    parser.add_argument(
        "--hard-learning-rate",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--soft-learning-rate",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--technical-stop-step",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--technical-safety-step-limit",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--soft-tolerance",
        required=True,
        type=float,
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(
        argv
    )

    try:
        config = TechnicalDistillationFixtureConfig(
            hard_learning_rate=args.hard_learning_rate,
            soft_learning_rate=args.soft_learning_rate,
            technical_stop_step=args.technical_stop_step,
            technical_safety_step_limit=(
                args.technical_safety_step_limit
            ),
            soft_tolerance=args.soft_tolerance,
        )

        report = run_portable_stage7_fixture(
            output_root=args.output_root,
            repository_root=REPO_ROOT,
            teacher_seed=args.teacher_seed,
            phase=args.phase,
            distillation_config=config,
        )

    except (
        OSError,
        RuntimeError,
        ValueError,
        Stage7PortableE2EError,
    ) as exc:
        print(
            "STAGE7A_TECHNICAL_VALIDATION=FAIL "
            f"reason={exc}",
            file=sys.stderr,
        )
        return 2

    print(
        "STAGE7A_TECHNICAL_VALIDATION=PASS"
    )

    print(
        f"PIPELINE_STEP_COUNT={report['pipeline_step_count']}"
    )

    for step in report[
        "pipeline_steps"
    ]:
        print(
            "PIPELINE_STEP="
            f"{step['ordinal']}/10|"
            f"{step['step_id']}|"
            f"STATUS={step['status']}|"
            f"SHA256={step['sha256']}"
        )

    print(
        f"REQUEST_SHA256={report['request_sha256']}"
    )

    print(
        f"MANIFEST_SHA256={report['manifest_sha256']}"
    )

    print(
        f"RUN_IDENTITY={report['run_identity']}"
    )

    print(
        f"ELIGIBLE_PATH_COUNT={report['eligible_path_count']}"
    )

    print(
        f"FAILED_PATH_COUNT={report['failed_path_count']}"
    )

    print(
        "FAILED_SUBJECT_DISCOVERY_COUNT="
        f"{report['failed_subject_discovery_count']}"
    )

    print(
        "EXCLUDED_ENDPOINT_OUTPUT_COUNT="
        f"{report['excluded_endpoint_output_count']}"
    )

    print(
        "PRIMARY_ANALYSIS_ELIGIBLE_COUNT="
        f"{report['primary_analysis_eligible_count']}"
    )

    print(
        "SCIENTIFIC_SELECTION_ELIGIBLE_COUNT="
        f"{report['scientific_selection_eligible_count']}"
    )

    print(
        "POST_FREEZE_REGENERATION_REQUIRED="
        f"{report['post_freeze_regeneration_required']}"
    )

    print(
        "RESUME_COUNTS_UNCHANGED="
        f"{report['resume_counts_unchanged']}"
    )

    print(
        "REPRODUCTION_MATCHED="
        f"{report['reproduction_matched']}"
    )

    print(
        "REPRODUCTION_MISMATCH_PATHS="
        f"{report['reproduction_mismatch_paths']}"
    )

    print(
        "SOURCE_RECORD_SHA256="
        f"{report['source_record_sha256']}"
    )

    print(
        "REPRODUCTION_RECORD_SHA256="
        f"{report['reproduction_record_sha256']}"
    )

    print(
        "SUBSTANTIVE_SHA256="
        f"{report['substantive_sha256']}"
    )

    print(
        f"OUTPUT_ROOT={args.output_root.resolve()}"
    )

    print(
        "OUTPUT_WRITTEN=EXPLICIT_TEMP_ROOT_ONLY"
    )

    print(
        "SCIENTIFIC_DATA=NO"
    )

    print(
        "PRODUCTION_ELIGIBLE=NO"
    )

    print(
        "REGISTERED_FIXTURE_EXECUTION=NO"
    )

    print(
        "STAGE8_EXECUTION=NO"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
