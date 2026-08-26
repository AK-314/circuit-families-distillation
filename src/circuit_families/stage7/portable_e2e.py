"""Portable synthetic Stage 7A end-to-end validation fixture.

The fixture composes the already accepted Stage 7A integration layers. It
writes runtime evidence only beneath one explicit system-temporary root and
contains no registered checkpoint, scientific execution, or production choice.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage7.contracts import (
    build_technical_run_manifest,
    load_technical_run_request,
)
from circuit_families.stage7.distillation import (
    TechnicalDistillationFixtureConfig,
)
from circuit_families.stage7.reproduction import (
    build_pipeline_reproduction_record,
    compare_independent_pipeline_reproduction,
)

PORTABLE_E2E_SCHEMA_VERSION: Final = "stage7-portable-technical-e2e/v1"
PORTABLE_REPORT_FILENAME: Final = "stage7a_validation_report.json"

EXPECTED_PIPELINE_STEPS: Final = (
    "teacher_target_cache",
    "student_attempts",
    "eligibility",
    "passed_only_sealing",
    "discovery",
    "exact_ledgers_endpoint1",
    "endpoint2",
    "teacher_seed_inventory",
    "analysis_report",
    "independent_reproduction",
)


class Stage7PortableE2EError(ValueError):
    """Raised when the portable Stage 7A technical fixture is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(
        _canonical_bytes(value)
    ).hexdigest()


def _file_sha256(source: Path) -> str:
    if not source.is_file():
        raise Stage7PortableE2EError(
            f"expected technical runtime file is missing: {source.name}"
        )

    return hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def _validate_temporary_output_root(
    *,
    output_root: str | Path,
    repository_root: str | Path,
) -> Path:
    repository = Path(
        repository_root
    ).resolve(strict=True)

    temporary = Path(
        tempfile.gettempdir()
    ).resolve(strict=True)

    candidate = Path(
        output_root
    ).expanduser().resolve(strict=False)

    if candidate == temporary:
        raise Stage7PortableE2EError(
            "output root must be a dedicated child of the system temporary root"
        )

    try:
        candidate.relative_to(
            temporary
        )
    except ValueError as exc:
        raise Stage7PortableE2EError(
            "output root must be beneath the system temporary root"
        ) from exc

    try:
        candidate.relative_to(
            repository
        )
    except ValueError:
        pass
    else:
        raise Stage7PortableE2EError(
            "portable validation may not write inside the repository"
        )

    try:
        repository.relative_to(
            candidate
        )
    except ValueError:
        pass
    else:
        raise Stage7PortableE2EError(
            "portable output root may not contain the repository"
        )

    if candidate.exists():
        raise Stage7PortableE2EError(
            "portable output root must not already exist"
        )

    return candidate


def _step_record(
    *,
    ordinal: int,
    step_id: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "ordinal": ordinal,
        "step_id": step_id,
        "status": "PASS",
        "evidence": dict(
            evidence
        ),
    }

    record["sha256"] = _sha256(
        record
    )

    return record


def _pipeline_step_records(
    *,
    source_record: Mapping[str, Any],
    reproduction_comparison: Mapping[str, Any],
    source_physical_root: Path,
) -> list[dict[str, Any]]:
    distillation = source_record[
        "distillation"
    ]

    hard = distillation[
        "hard"
    ]

    soft = distillation[
        "soft"
    ]

    runs = source_record[
        "discovery_runs"
    ]

    inventory = source_record[
        "inventory"
    ]

    analysis = source_record[
        "analysis_bridge"
    ]

    exclusions = source_record[
        "excluded_endpoint_records"
    ]

    report = source_record[
        "report"
    ]

    resume = source_record[
        "resume_merge"
    ]

    hard_classifications = list(
        hard["classifications"]
    )

    soft_statuses = list(
        soft["statuses"]
    )

    eligible_count = (
        hard_classifications.count(
            "eligible"
        )
        + soft_statuses.count(
            "eligible"
        )
    )

    failed_count = (
        hard_classifications.count(
            "training_failure"
        )
        + soft_statuses.count(
            "failed"
        )
    )

    failed_subject_ids = {
        "technical-hard-failed-student",
        "technical-soft-failed-student",
    }

    observed_discovery_subject_ids = {
        run["subject_id"]
        for run in runs
    }

    failed_discovery_count = len(
        failed_subject_ids
        & observed_discovery_subject_ids
    )

    ledger_hashes = sorted(
        run["exact_ledger_sha256"]
        for run in runs
    )

    endpoint1_hashes = sorted(
        _sha256(
            run["endpoint1"]
        )
        for run in runs
    )

    endpoint2_hashes = sorted(
        run["endpoint2"]["result_hash"]
        for run in runs
    )

    step_evidence = {
        "teacher_target_cache": {
            "cache_count": 2,
            "hard_manifest_sha256": _file_sha256(
                source_physical_root
                / "distillation/hard_target/cache/manifest.json"
            ),
            "soft_manifest_sha256": _file_sha256(
                source_physical_root
                / "distillation/soft_target/cache/manifest.json"
            ),
        },
        "student_attempts": {
            "attempt_count": (
                hard["attempt_count"]
                + soft["attempt_count"]
            ),
            "hard_attempt_record_sha256": list(
                hard["attempt_record_sha256"]
            ),
            "soft_attempt_record_sha256": list(
                soft["attempt_record_sha256"]
            ),
        },
        "eligibility": {
            "eligible_count": eligible_count,
            "failed_count": failed_count,
            "hard_classifications": hard_classifications,
            "soft_statuses": soft_statuses,
        },
        "passed_only_sealing": {
            "sealed_eligible_count": 2,
            "hard_sealed_reference_sha256": hard[
                "sealed_dense_model_sha256"
            ],
            "soft_sealed_reference_sha256": soft[
                "sealed_dense_model_sha256"
            ],
            "failed_subject_discovery_count": failed_discovery_count,
        },
        "discovery": {
            "run_count": len(
                runs
            ),
            "subject_roles": sorted(
                {
                    run["subject_role"]
                    for run in runs
                }
            ),
            "methods": sorted(
                {
                    run["discovery_method"]
                    for run in runs
                }
            ),
            "native_charge_count": resume[
                "before_resume_counts"
            ]["native_charges"],
        },
        "exact_ledgers_endpoint1": {
            "ledger_count": len(
                ledger_hashes
            ),
            "ledger_sha256": ledger_hashes,
            "exact_evaluation_count": resume[
                "before_resume_counts"
            ]["exact_evaluations"],
            "endpoint1_count": len(
                endpoint1_hashes
            ),
            "endpoint1_sha256": endpoint1_hashes,
        },
        "endpoint2": {
            "endpoint2_count": len(
                endpoint2_hashes
            ),
            "endpoint2_result_sha256": endpoint2_hashes,
            "packing_lower_bounds": [
                run["endpoint2"][
                    "packing_lower_bound"
                ]
                for run in runs
            ],
        },
        "teacher_seed_inventory": {
            "inventory_sha256": inventory[
                "sha256"
            ],
            "row_count": len(
                inventory["rows"]
            ),
            "population_unit": inventory[
                "population_unit"
            ],
            "hard_soft_pooled": inventory[
                "hard_soft_pooled"
            ],
        },
        "analysis_report": {
            "stage5d_output_bundle_sha256": analysis[
                "stage5d_output_bundle_sha256"
            ],
            "report_sha256": report[
                "sha256"
            ],
            "excluded_endpoint_output_count": len(
                exclusions
            ),
            "primary_analysis_eligible_count": sum(
                entry[
                    "primary_analysis_eligible"
                ]
                for entry in exclusions
            ),
            "scientific_selection_eligible_count": sum(
                entry[
                    "scientific_selection_eligible"
                ]
                for entry in exclusions
            ),
        },
        "independent_reproduction": {
            "matched": reproduction_comparison[
                "matched"
            ],
            "source_sha256": reproduction_comparison[
                "source_sha256"
            ],
            "reproduction_sha256": reproduction_comparison[
                "reproduction_sha256"
            ],
            "substantive_source_sha256": reproduction_comparison[
                "substantive_source_sha256"
            ],
            "substantive_reproduction_sha256": reproduction_comparison[
                "substantive_reproduction_sha256"
            ],
            "mismatch_paths": list(
                reproduction_comparison[
                    "mismatch_paths"
                ]
            ),
        },
    }

    return [
        _step_record(
            ordinal=ordinal,
            step_id=step_id,
            evidence=step_evidence[
                step_id
            ],
        )
        for ordinal, step_id in enumerate(
            EXPECTED_PIPELINE_STEPS,
            start=1,
        )
    ]


def run_portable_stage7_fixture(
    *,
    output_root: str | Path,
    repository_root: str | Path,
    teacher_seed: int,
    phase: str,
    distillation_config: TechnicalDistillationFixtureConfig,
) -> dict[str, Any]:
    """Run the complete portable technical pipeline twice and compare it."""
    repository = Path(
        repository_root
    ).resolve(strict=True)

    physical_root = _validate_temporary_output_root(
        output_root=output_root,
        repository_root=repository,
    )

    if not isinstance(
        distillation_config,
        TechnicalDistillationFixtureConfig,
    ):
        raise Stage7PortableE2EError(
            "distillation_config must be TechnicalDistillationFixtureConfig"
        )

    request = load_technical_run_request(
        repository
        / "followup/configs/stage7/"
        "technical_run_request_v1.json"
    )

    stage3 = Stage3AvailabilityIndex.from_registry(
        json.loads(
            (
                repository
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
        repository_root=repository,
    )

    lifecycle_ids = tuple(
        step.step_id
        for step in manifest.lifecycle.topological_steps()
    )

    if lifecycle_ids != EXPECTED_PIPELINE_STEPS:
        raise Stage7PortableE2EError(
            "portable fixture does not match the canonical ten-step lifecycle"
        )

    exclusion_register = json.loads(
        (
            repository
            / "followup/manifests/"
            "stage2_excluded_development_register_v1.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    physical_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    source_root = physical_root / "source"

    reproduction_root = (
        physical_root
        / "reproduction"
    )

    common = {
        "repository_root": repository,
        "stage3": stage3,
        "teacher_seed": teacher_seed,
        "phase": phase,
        "run_request": request,
        "distillation_config": distillation_config,
        "discovery_profiles_path": (
            repository
            / "followup/configs/stage6d/"
            "technical_discovery_profiles_v1.json"
        ),
        "endpoint2_policy_path": (
            repository
            / "followup/configs/stage6e/"
            "technical_endpoint2_policy_v1.json"
        ),
        "exclusion_register": exclusion_register,
    }

    source = build_pipeline_reproduction_record(
        physical_root=source_root,
        **common,
    )

    reproduction = (
        build_pipeline_reproduction_record(
            physical_root=reproduction_root,
            **common,
        )
    )

    comparison = (
        compare_independent_pipeline_reproduction(
            source_record=source,
            reproduction_record=reproduction,
        )
    )

    if comparison["matched"] is not True:
        raise Stage7PortableE2EError(
            "independent portable reproduction mismatch: "
            f"{comparison['mismatch_paths']}"
        )

    steps = _pipeline_step_records(
        source_record=source,
        reproduction_comparison=comparison,
        source_physical_root=source_root,
    )

    if tuple(
        step["step_id"]
        for step in steps
    ) != EXPECTED_PIPELINE_STEPS:
        raise Stage7PortableE2EError(
            "portable step-report order changed"
        )

    if any(
        step["status"] != "PASS"
        for step in steps
    ):
        raise Stage7PortableE2EError(
            "portable pipeline contains a non-PASS step"
        )

    eligibility = steps[2][
        "evidence"
    ]

    sealing = steps[3][
        "evidence"
    ]

    if eligibility["eligible_count"] < 1:
        raise Stage7PortableE2EError(
            "portable fixture lacks an eligible path"
        )

    if eligibility["failed_count"] < 1:
        raise Stage7PortableE2EError(
            "portable fixture lacks a failed path"
        )

    if sealing["failed_subject_discovery_count"] != 0:
        raise Stage7PortableE2EError(
            "failed path released downstream discovery"
        )

    exclusions = source[
        "excluded_endpoint_records"
    ]

    primary_eligible = sum(
        entry["primary_analysis_eligible"]
        for entry in exclusions
    )

    scientific_selection_eligible = sum(
        entry["scientific_selection_eligible"]
        for entry in exclusions
    )

    if primary_eligible != 0:
        raise Stage7PortableE2EError(
            "excluded development entered primary analysis eligibility"
        )

    if scientific_selection_eligible != 0:
        raise Stage7PortableE2EError(
            "excluded development entered scientific selection eligibility"
        )

    resume = source[
        "resume_merge"
    ]

    if resume["before_resume_counts"] != resume["after_resume_counts"]:
        raise Stage7PortableE2EError(
            "resume changed completed technical counts"
        )

    if any(
        (
            resume["completed_attempts_duplicated"],
            resume["native_charges_duplicated"],
            resume["exact_evaluations_duplicated"],
            resume["endpoints_duplicated"],
            resume["resume_state_transferred"],
        )
    ):
        raise Stage7PortableE2EError(
            "portable resume/no-duplication gate failed"
        )

    report = {
        "schema_version": PORTABLE_E2E_SCHEMA_VERSION,
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "production_default": False,
        "registered_fixture_execution": False,
        "scientific_execution": False,
        "stage8_execution": False,
        "request_sha256": request.request_sha256,
        "run_identity": request.run_identity,
        "manifest_sha256": manifest.manifest_sha256,
        "pipeline_step_count": len(
            steps
        ),
        "pipeline_steps": steps,
        "eligible_path_count": eligibility[
            "eligible_count"
        ],
        "failed_path_count": eligibility[
            "failed_count"
        ],
        "failed_subject_discovery_count": sealing[
            "failed_subject_discovery_count"
        ],
        "excluded_endpoint_output_count": len(
            exclusions
        ),
        "primary_analysis_eligible_count": primary_eligible,
        "scientific_selection_eligible_count": (
            scientific_selection_eligible
        ),
        "post_freeze_regeneration_required": all(
            entry["regeneration_required"]
            for entry in exclusions
        ),
        "resume_counts_unchanged": (
            resume["before_resume_counts"]
            == resume["after_resume_counts"]
        ),
        "reproduction_matched": comparison[
            "matched"
        ],
        "reproduction_mismatch_paths": list(
            comparison[
                "mismatch_paths"
            ]
        ),
        "source_record_sha256": source[
            "sha256"
        ],
        "reproduction_record_sha256": reproduction[
            "sha256"
        ],
    }

    report["substantive_sha256"] = _sha256(
        report
    )

    report_path = (
        physical_root
        / PORTABLE_REPORT_FILENAME
    )

    report_path.write_bytes(
        _canonical_bytes(
            report
        )
    )

    if _file_sha256(
        report_path
    ) != hashlib.sha256(
        _canonical_bytes(
            report
        )
    ).hexdigest():
        raise Stage7PortableE2EError(
            "portable report write verification failed"
        )

    return report
