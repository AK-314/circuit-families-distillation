"""Deterministic final reporting for Stage 14 random-label analysis."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from circuit_families.analysis.fidelity_calibration import (
    write_csv_records,
)
from circuit_families.analysis.random_label_circuit_analysis import (
    load_frozen_analysis_configuration,
)
from circuit_families.analysis.stage14_random_label_runner import (
    find_stage15_artifacts,
    output_contract,
    validate_analysis_inputs,
)

FRONTIER_COLUMNS = (
    "analysis_run_id",
    "checkpoint_step",
    "family_status",
    "family_size",
    "accepted_circuit",
    "representative_search_status",
    "representative_primary_fidelity",
    "representative_retained_component_count",
    "representative_exact_evaluations_used",
    "primary_fidelity_threshold",
    "meaningfully_sparse_max_components",
    "scientific_interpretation",
)

DETERMINISTIC_REPORT_OUTPUTS = (
    "sparse_search_table",
    "family_summary_table",
    "circuits_table",
    "pairwise_overlap_table",
    "restart_table",
    "frontier_table",
    "fidelity_sensitivity_table",
    "distinctness_sensitivity_table",
    "transfer_table",
    "analysis_note",
)

REQUIRED_PRE_REPORT_OUTPUTS = {
    "raw_output_directory",
    "sparse_search_table",
    "family_summary_table",
    "circuits_table",
    "pairwise_overlap_table",
    "restart_table",
    "fidelity_sensitivity_table",
    "distinctness_sensitivity_table",
    "transfer_table",
    "runtime_table",
}


@dataclass(frozen=True)
class ReportingWorkloadResult:
    """Final deterministic Stage 14 reporting outputs."""

    analysis_run_id: str
    implementation_commit: str
    frontier_table: Path
    analysis_note: Path
    manifest: Path
    archive: Path


def _sha256(file_name: Path) -> str:
    return hashlib.sha256(file_name.read_bytes()).hexdigest()


def _stable_json(
    file_name: Path,
    value: Mapping[str, Any],
) -> Path:
    file_name.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    file_name.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return file_name


def _read_csv(
    file_name: Path,
) -> list[dict[str, str]]:
    with file_name.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def frontier_rows(
    *,
    analysis_run_id: str,
    family_rows: Sequence[Mapping[str, str]],
    restart_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Build one descriptive primary-result row per checkpoint."""

    restarts_by_step = {
        int(row["checkpoint_step"]): row
        for row in restart_rows
        if row["restart_used"].lower() == "true"
        and int(row["requested_member_index"]) == 1
        and int(row["restart_index"]) == 0
    }
    rows: list[dict[str, object]] = []

    for family in sorted(
        family_rows,
        key=lambda row: int(
            row["checkpoint_step"]
        ),
    ):
        step = int(family["checkpoint_step"])
        restart = restarts_by_step.get(step)

        if restart is None:
            raise ValueError(
                f"Missing primary C1 restart row at step {step}."
            )

        family_size = int(family["family_size"])
        accepted = (
            restart["accepted_candidate"].lower()
            == "true"
        )

        if family_size == 0:
            interpretation = (
                "No meaningfully sparse random-label circuit "
                "was recovered under the frozen threshold, "
                "search method and exact-evaluation budget."
            )
        else:
            interpretation = (
                "At least one meaningfully sparse random-label "
                "circuit was recovered under the frozen analysis."
            )

        rows.append(
            {
                "analysis_run_id": analysis_run_id,
                "checkpoint_step": step,
                "family_status": family["status"],
                "family_size": family_size,
                "accepted_circuit": accepted,
                "representative_search_status": (
                    restart["search_status"]
                ),
                "representative_primary_fidelity": (
                    restart["primary_fidelity"]
                ),
                "representative_retained_component_count": (
                    int(
                        restart[
                            "retained_component_count"
                        ]
                    )
                ),
                "representative_exact_evaluations_used": (
                    int(
                        restart[
                            "exact_evaluations_used"
                        ]
                    )
                ),
                "primary_fidelity_threshold": 0.99,
                "meaningfully_sparse_max_components": 258,
                "scientific_interpretation": (
                    interpretation
                ),
            }
        )

    return rows


def analysis_note_text(
    *,
    analysis_run_id: str,
    frontier: Sequence[Mapping[str, object]],
) -> str:
    """Return the final cautious Stage 14 interpretation note."""

    family_sizes = tuple(
        int(row["family_size"])
        for row in frontier
    )
    all_empty = all(
        value == 0
        for value in family_sizes
    )

    if all_empty:
        result_text = (
            "No meaningfully sparse random-label circuit was "
            "recovered at any matched checkpoint under the "
            "frozen 0.99 fidelity threshold, search method and "
            "exact-evaluation budget."
        )
    else:
        result_text = (
            "At least one matched checkpoint produced a "
            "meaningfully sparse random-label circuit under "
            "the frozen analysis."
        )

    trajectory = " → ".join(
        str(value)
        for value in family_sizes
    )

    return (
        "# Stage 14 random-label circuit analysis\n\n"
        f"- Analysis run: `{analysis_run_id}`\n"
        "- Control classification: `memorisation_control`\n"
        "- Primary fidelity threshold: `0.99`\n"
        "- Primary distinctness cutoff: `0.50`\n"
        "- Meaningfully sparse boundary: at most `258 / 516` "
        "searchable components\n"
        f"- Primary family-size trajectory: `{trajectory}`\n\n"
        "## Primary result\n\n"
        f"{result_text}\n\n"
        "This is a bounded search result. It does not establish "
        "that sparse random-label circuits do not exist outside "
        "the frozen threshold, search procedure or evaluation "
        "budget.\n\n"
        "Primary-family transfer and grouping outputs preserve "
        "empty-family outcomes rather than imputing profiles or "
        "groups. Subset-discovery searches are reported separately "
        "from the primary global family.\n"
    )


def write_deterministic_archive(
    archive: Path,
    *,
    root: Path,
    members: Sequence[Path],
) -> Path:
    """Write a byte-deterministic gzip-compressed tar archive."""

    archive.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    buffer = io.BytesIO()

    with tarfile.open(
        fileobj=buffer,
        mode="w",
        format=tarfile.PAX_FORMAT,
    ) as tar:
        for file_name in sorted(
            (
                member.resolve()
                for member in members
                if member.is_file()
            ),
            key=lambda item: item.relative_to(
                root.resolve()
            ).as_posix(),
        ):
            relative = file_name.relative_to(
                root.resolve()
            ).as_posix()
            data = file_name.read_bytes()
            info = tarfile.TarInfo(
                name=relative
            )
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            tar.addfile(
                info,
                io.BytesIO(data),
            )

    with archive.open("wb") as handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=handle,
            mtime=0,
        ) as compressed:
            compressed.write(
                buffer.getvalue()
            )

    return archive


def execute_reporting_workload(
    *,
    repository_root: str | Path,
    expected_implementation_commit: str,
    output_root: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> ReportingWorkloadResult:
    """Generate the final frontier, note, manifest and archive."""

    repository = Path(repository_root).resolve()
    selected_output_root = (
        repository
        if output_root is None
        else Path(output_root).resolve()
    )

    validation = validate_analysis_inputs(
        repository_root=repository,
        expected_implementation_commit=(
            expected_implementation_commit
        ),
        output_root=selected_output_root,
        require_clean_repository=False,
        require_outputs_absent=False,
        verify_checkpoint_hashes=True,
    )

    if (
        validation.current_commit
        != expected_implementation_commit
    ):
        raise RuntimeError(
            "Validated implementation commit differs."
        )

    if find_stage15_artifacts(repository):
        raise FileExistsError(
            "Stage 15 artifacts exist before reporting."
        )

    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    resolved = dict(
        output_contract(configuration).resolve(
            selected_output_root
        )
    )
    observed_existing = {
        name
        for name, file_name in resolved.items()
        if file_name.exists()
    }

    if observed_existing != REQUIRED_PRE_REPORT_OUTPUTS:
        raise FileExistsError(
            "Reporting requires the exact completed "
            "scientific output state."
        )

    frontier_table = resolved["frontier_table"]
    analysis_note = resolved["analysis_note"]
    manifest = resolved["manifest"]
    archive = resolved["archive"]

    for file_name in (
        frontier_table,
        analysis_note,
        manifest,
        archive,
    ):
        if file_name.exists():
            raise FileExistsError(file_name)

    generated = (
        frontier_table,
        analysis_note,
        manifest,
        archive,
    )

    try:
        if progress_callback is not None:
            progress_callback(
                "building deterministic frontier"
            )

        family_rows = _read_csv(
            resolved["family_summary_table"]
        )
        restart_rows = _read_csv(
            resolved["restart_table"]
        )
        frontier = frontier_rows(
            analysis_run_id=(
                configuration.analysis_run_id
            ),
            family_rows=family_rows,
            restart_rows=restart_rows,
        )
        write_csv_records(
            frontier_table,
            fieldnames=FRONTIER_COLUMNS,
            rows=frontier,
        )

        analysis_note.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        analysis_note.write_text(
            analysis_note_text(
                analysis_run_id=(
                    configuration.analysis_run_id
                ),
                frontier=frontier,
            ),
            encoding="utf-8",
        )

        deterministic_records = []

        for output_name in (
            *DETERMINISTIC_REPORT_OUTPUTS[:-1],
            "analysis_note",
        ):
            file_name = resolved[
                output_name
            ]

            if not file_name.is_file():
                raise FileNotFoundError(
                    file_name
                )

            deterministic_records.append(
                {
                    "output_name": output_name,
                    "path": (
                        file_name.resolve()
                        .relative_to(
                            selected_output_root
                        )
                        .as_posix()
                    ),
                    "sha256": _sha256(
                        file_name
                    ),
                    "included_in_deterministic_scientific_hashes": True,
                }
            )

        runtime_table = resolved[
            "runtime_table"
        ]
        manifest_payload = {
            "schema_version": 1,
            "experiment_type": (
                "stage14_random_label_circuit_analysis"
            ),
            "analysis_run_id": (
                configuration.analysis_run_id
            ),
            "analysis_identity_sha256": (
                configuration.analysis_identity_sha256
            ),
            "analysis_configuration_path": (
                configuration.path
                .relative_to(repository)
                .as_posix()
            ),
            "analysis_configuration_sha256": (
                configuration.sha256
            ),
            "implementation_git_commit": (
                expected_implementation_commit
            ),
            "control_classification": (
                configuration.payload["source"][
                    "control_classification"
                ]
            ),
            "checkpoint_steps": list(
                configuration.checkpoint_steps
            ),
            "deterministic_outputs": (
                deterministic_records
            ),
            "runtime_output": {
                "path": (
                    runtime_table.resolve()
                    .relative_to(
                        selected_output_root
                    )
                    .as_posix()
                ),
                "sha256": _sha256(
                    runtime_table
                ),
                "included_in_deterministic_scientific_hashes": False,
            },
            "primary_family_sizes": [
                int(row["family_size"])
                for row in frontier
            ],
            "stage15_started": False,
        }
        _stable_json(
            manifest,
            manifest_payload,
        )

        archive_members = [
            *(
                file_name
                for file_name in resolved[
                    "raw_output_directory"
                ].rglob("*")
                if file_name.is_file()
            ),
            *(
                resolved[name]
                for name in (
                    "sparse_search_table",
                    "family_summary_table",
                    "circuits_table",
                    "pairwise_overlap_table",
                    "restart_table",
                    "frontier_table",
                    "fidelity_sensitivity_table",
                    "distinctness_sensitivity_table",
                    "transfer_table",
                    "runtime_table",
                    "analysis_note",
                    "manifest",
                )
            ),
        ]
        write_deterministic_archive(
            archive,
            root=selected_output_root,
            members=archive_members,
        )

        if progress_callback is not None:
            progress_callback(
                "deterministic reporting complete"
            )

        return ReportingWorkloadResult(
            analysis_run_id=(
                configuration.analysis_run_id
            ),
            implementation_commit=(
                expected_implementation_commit
            ),
            frontier_table=frontier_table,
            analysis_note=analysis_note,
            manifest=manifest,
            archive=archive,
        )

    except Exception:
        for file_name in generated:
            file_name.unlink(
                missing_ok=True
            )

        raise
