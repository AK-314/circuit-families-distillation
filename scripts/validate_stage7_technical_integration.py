#!/usr/bin/env python3
"""Validate the complete portable synthetic Stage 7A integration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields, replace
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

from circuit_families.stage4_condition_identity import (  # noqa: E402
    Stage3AvailabilityIndex,
)
from circuit_families.stage7 import (  # noqa: E402
    EXPECTED_PIPELINE_STEPS,
    Stage7PortableE2EError,
    TechnicalDistillationFixtureConfig,
    build_technical_run_manifest,
    load_technical_run_request,
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
        "--validate-only",
        action="store_true",
        help=(
            "Validate Stage 7A contracts and registered-reference readiness "
            "without executing the synthetic technical integration."
        ),
    )

    parser.add_argument(
        "--output-root",
        required=False,
        type=Path,
        help=(
            "Dedicated non-existing directory beneath the system temporary root."
        ),
    )

    parser.add_argument(
        "--teacher-seed",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--phase",
        required=False,
    )

    parser.add_argument(
        "--hard-learning-rate",
        required=False,
        type=float,
    )

    parser.add_argument(
        "--soft-learning-rate",
        required=False,
        type=float,
    )

    parser.add_argument(
        "--technical-stop-step",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--technical-safety-step-limit",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--soft-tolerance",
        required=False,
        type=float,
    )

    return parser




def _run_validate_only() -> int:
    request = load_technical_run_request(
        REPO_ROOT
        / "followup/configs/stage7/"
        "technical_run_request_v1.json"
    )

    stage3 = Stage3AvailabilityIndex.from_registry(
        json.loads(
            (
                REPO_ROOT
                / "followup/manifests/"
                "stage3_teacher_registry_v1.json"
            ).read_text(
                encoding="utf-8"
            )
        )
    )

    manifest = build_technical_run_manifest(
        request,
        stage3=stage3,
        repository_root=REPO_ROOT,
    )

    request_mapping = request.to_mapping()
    manifest_mapping = manifest.to_mapping()

    if (
        request_mapping["classification"]
        != "synthetic_technical_only"
    ):
        raise ValueError(
            "validate-only requires synthetic_technical_only classification"
        )

    for key in (
        "scientific_data",
        "production_eligible",
        "production_default",
    ):
        if request_mapping[key] is not False:
            raise ValueError(
                f"request firewall failed for {key}"
            )

        if manifest_mapping[key] is not False:
            raise ValueError(
                f"manifest firewall failed for {key}"
            )

    if request_mapping["resolves_decisions"] != []:
        raise ValueError(
            "validate-only may not resolve UD decisions"
        )

    step_ids = tuple(
        step.step_id
        for step in manifest.lifecycle.topological_steps()
    )

    if step_ids != EXPECTED_PIPELINE_STEPS:
        raise ValueError(
            "validate-only lifecycle differs from canonical pipeline"
        )

    teacher = request.teacher_reference
    teacher_changes: dict[str, str] = {}

    for field in fields(
        teacher
    ):
        value = getattr(
            teacher,
            field.name,
        )

        if value == "injected_fixture":
            teacher_changes[
                field.name
            ] = "registered_reference"

        if (
            isinstance(
                value,
                str,
            )
            and value.startswith(
                "synthetic://"
            )
        ):
            teacher_changes[
                field.name
            ] = (
                "registered://stage7/"
                "teacher-checkpoint/pending"
            )

    if len(
        teacher_changes
    ) < 2:
        raise ValueError(
            "registered-reference readiness binding unavailable"
        )

    registered_teacher = replace(
        teacher,
        **teacher_changes,
    )

    registered_request = replace(
        request,
        teacher_reference=registered_teacher,
    )

    registered_manifest = build_technical_run_manifest(
        registered_request,
        stage3=stage3,
        repository_root=REPO_ROOT,
    )

    registered_steps = tuple(
        step.step_id
        for step in registered_manifest.lifecycle.topological_steps()
    )

    if registered_steps != EXPECTED_PIPELINE_STEPS:
        raise ValueError(
            "registered-reference readiness changed canonical lifecycle"
        )

    registered_mapping = registered_request.to_mapping()

    if (
        registered_mapping["classification"]
        != "synthetic_technical_only"
    ):
        raise ValueError(
            "registered-reference readiness changed classification"
        )

    for key in (
        "scientific_data",
        "production_eligible",
        "production_default",
    ):
        if registered_mapping[key] is not False:
            raise ValueError(
                f"registered-reference firewall failed for {key}"
            )

    serialized = json.dumps(
        registered_mapping,
        sort_keys=True,
    )

    if "private_checkpoint_access=none" not in serialized:
        raise ValueError(
            "explicit no-private-checkpoint declaration missing"
        )

    for forbidden in (
        "private_checkpoint_access=granted",
        "private_checkpoint_path",
        "private_checkpoint_bytes",
        "private://",
        "/Users/",
        "\\Users\\",
    ):
        if forbidden in serialized:
            raise ValueError(
                "private checkpoint evidence present: "
                + forbidden
            )

    registered_uri = next(
        value
        for value in (
            getattr(
                registered_teacher,
                field.name,
            )
            for field in fields(
                registered_teacher
            )
        )
        if (
            isinstance(
                value,
                str,
            )
            and value.startswith(
                "registered://"
            )
        )
    )

    print(
        "STAGE7A_VALIDATE_ONLY=PASS"
    )
    print(
        "VALIDATION_MODE=CONTRACT_READINESS_ONLY"
    )
    print(
        f"PIPELINE_STEP_COUNT={len(step_ids)}"
    )
    print(
        f"REQUEST_SHA256={request.request_sha256}"
    )
    print(
        f"MANIFEST_SHA256={manifest.manifest_sha256}"
    )
    print(
        "REGISTERED_REQUEST_SHA256="
        f"{registered_request.request_sha256}"
    )
    print(
        "REGISTERED_MANIFEST_SHA256="
        f"{registered_manifest.manifest_sha256}"
    )
    print(
        f"REGISTERED_REFERENCE_URI={registered_uri}"
    )
    print(
        "TECHNICAL_FIXTURE_EXECUTION=NO"
    )
    print(
        "REGISTERED_FIXTURE_EXECUTION=NO"
    )
    print(
        "OUTPUT_WRITTEN=NO"
    )
    print(
        "SCIENTIFIC_DATA=NO"
    )
    print(
        "PRODUCTION_ELIGIBLE=NO"
    )
    print(
        "PRODUCTION_DEFAULT=NO"
    )
    print(
        "UD_RESOLUTIONS=0"
    )
    print(
        "STAGE8_EXECUTION=NO"
    )

    return 0

def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(
        argv
    )

    fixture_argument_names = (
        "output_root",
        "teacher_seed",
        "phase",
        "hard_learning_rate",
        "soft_learning_rate",
        "technical_stop_step",
        "technical_safety_step_limit",
        "soft_tolerance",
    )

    if args.validate_only:
        supplied_fixture_arguments = [
            name
            for name in fixture_argument_names
            if getattr(
                args,
                name,
            ) is not None
        ]

        if supplied_fixture_arguments:
            parser.error(
                "--validate-only cannot be combined with technical-fixture "
                "arguments: "
                + ", ".join(
                    supplied_fixture_arguments
                )
            )

        try:
            return _run_validate_only()
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(
                "STAGE7A_VALIDATE_ONLY=FAIL "
                f"reason={exc}",
                file=sys.stderr,
            )
            return 2

    missing_fixture_arguments = [
        name
        for name in fixture_argument_names
        if getattr(
            args,
            name,
        ) is None
    ]

    if missing_fixture_arguments:
        parser.error(
            "full synthetic technical integration requires: "
            + ", ".join(
                "--"
                + name.replace(
                    "_",
                    "-",
                )
                for name in missing_fixture_arguments
            )
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
