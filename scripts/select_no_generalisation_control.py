"""Apply the frozen Stage 13 curve-only selection rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from circuit_families.analysis.no_generalisation_selection import (
    DESCENDING_CANDIDATE_FRACTIONS,
    MATCHED_HORIZON,
    CandidateQualification,
    evaluate_candidate,
    select_largest_qualifying_fraction,
)
from circuit_families.manifests import (
    utc_timestamp,
    write_manifest,
)
from circuit_families.plotting.no_generalisation import (
    STAGE13_METRICS_COLUMNS,
    plot_stage13_training_curves,
    read_stage13_metrics_csv,
    stage13_figure_hashes,
    validate_stage13_metrics_rows,
    write_stage13_figure_caption,
    write_stage13_metrics_csv,
)
from circuit_families.training.checkpoints import file_sha256
from circuit_families.training.logging import read_jsonl
from circuit_families.training.no_generalisation import (
    STAGE13_CHECKPOINT_VALIDATION_STEPS,
)

METRICS_TABLE = Path(
    "results/tables/"
    "stage13_no_generalisation_training_metrics.csv"
)
SELECTION_TABLE = Path(
    "results/tables/"
    "stage13_no_generalisation_selection.csv"
)
SELECTION_NOTE = Path(
    "results/notes/"
    "stage13_no_generalisation_selection.md"
)
FIGURE_PNG = Path(
    "figures/"
    "stage13_no_generalisation_training_curves.png"
)
FIGURE_PDF = Path(
    "figures/"
    "stage13_no_generalisation_training_curves.pdf"
)
FIGURE_CAPTION = Path(
    "figures/"
    "stage13_no_generalisation_training_curves_caption.txt"
)
PROTOCOL_PATH = Path("experimental_protocol.md")

DATA_COVERAGE_LIMITATION = (
    "The matched no-generalisation control changes the fraction "
    "of the original training partition available to the model. "
    "It therefore controls for training time, architecture, "
    "optimiser and regularisation while deliberately changing "
    "data coverage. Any later difference between the main "
    "condition and this control cannot be attributed solely to "
    "generalisation without acknowledging the reduced-data "
    "intervention."
)

EXPECTED_METRIC_FIELDS = frozenset(
    {
        "checkpoint_path",
        "checkpoint_sha256",
        "gradient_norm",
        "learning_rate",
        "mode",
        "model_state_sha256",
        "optimizer_state_sha256",
        "run_id",
        "schema_version",
        "test_accuracy",
        "test_loss",
        "train_accuracy",
        "train_loss",
        "training_step",
        "weight_norm",
    }
)

SELECTION_COLUMNS = (
    "fraction",
    "exact_training_example_count",
    "required_horizon",
    "selection_rank",
    "first_train_accuracy_step",
    "criterion1_reached_by_step_5000",
    "persistence_numerator",
    "persistence_denominator",
    "persistence_proportion",
    "criterion2_persistence",
    "maximum_test_accuracy",
    "maximum_test_accuracy_step",
    "criterion3_test_accuracy_ceiling",
    "mean_test_accuracy_final_window",
    "mean_test_accuracy_preceding_window",
    "test_accuracy_window_difference",
    "criterion4_test_accuracy_plateau",
    "test_cross_entropy_at_4050",
    "test_cross_entropy_at_9050",
    "test_cross_entropy_fractional_fall",
    "criterion5_test_loss_plateau",
    "overall_qualification",
    "selected_control",
)


def parse_args() -> argparse.Namespace:
    """Parse Stage 13 selection arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Stage 13 curve-only "
            "selection artifacts."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--matched-horizon",
        type=int,
        default=MATCHED_HORIZON,
    )
    parser.add_argument(
        "--expected-implementation-commit",
    )
    return parser.parse_args()


def resolve(repository: Path, value: str | Path) -> Path:
    """Resolve a repository-relative path."""

    path = Path(value)
    return path if path.is_absolute() else repository / path


def relative_path(repository: Path, path: Path) -> str:
    """Render a repository-relative POSIX path."""

    return path.resolve().relative_to(
        repository.resolve()
    ).as_posix()


def git_output(repository: Path, *arguments: str) -> str:
    """Run Git and return stripped standard output."""

    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_unchanged_implementation(
    repository: Path,
    expected_commit: str | None,
) -> str:
    """Permit only untracked definitive Stage 13 pilot outputs."""

    if subprocess.run(
        ("git", "diff", "--quiet"),
        cwd=repository,
        check=False,
    ).returncode:
        raise RuntimeError(
            "Tracked implementation files changed before selection."
        )

    if subprocess.run(
        ("git", "diff", "--cached", "--quiet"),
        cwd=repository,
        check=False,
    ).returncode:
        raise RuntimeError(
            "Staged changes exist before Stage 13 selection."
        )

    commit = git_output(repository, "rev-parse", "HEAD")

    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            "Implementation commit mismatch: expected "
            f"{expected_commit}, found {commit}."
        )

    status = git_output(
        repository,
        "status",
        "--short",
        "--untracked-files=all",
    )
    allowed_prefixes = (
        "checkpoints/"
        "stage13-no-generalisation-training-s0-",
        "manifests/"
        "training_stage13-no-generalisation-training-s0-",
        "results/raw/"
        "stage13-no-generalisation-training-s0-",
        "results/raw/"
        "stage13-no-generalisation-s0-",
    )
    forbidden: list[str] = []

    for line in status.splitlines():
        if not line.startswith("?? "):
            forbidden.append(line)
            continue

        path = line[3:]

        if not path.startswith(allowed_prefixes):
            forbidden.append(line)

    if forbidden:
        raise RuntimeError(
            "Unexpected repository changes exist before "
            "Stage 13 selection:\n"
            + "\n".join(forbidden)
        )

    return commit


def require_later_stage_outputs_absent(
    repository: Path,
) -> None:
    """Reject Stage 14 or Stage 15 scientific outputs."""

    patterns = (
        "manifests/*stage14*",
        "manifests/*stage15*",
        "results/**/*stage14*",
        "results/**/*stage15*",
        "figures/**/*stage14*",
        "figures/**/*stage15*",
    )
    found = sorted(
        {
            relative_path(repository, path)
            for pattern in patterns
            for path in repository.glob(pattern)
            if path.is_file()
        }
    )

    if found:
        raise RuntimeError(
            "Stage 14 or Stage 15 outputs already exist: "
            + ", ".join(found)
        )


def load_json_object(
    path: Path,
    label: str,
) -> dict[str, Any]:
    """Load a required JSON object."""

    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object.")

    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(
            f"{name} must be a lowercase SHA-256 digest."
        )

    return value


def validate_metric_record_fields(
    record: Mapping[str, Any],
) -> None:
    """Reject every field outside the frozen curve schema."""

    if set(record) != EXPECTED_METRIC_FIELDS:
        unexpected = sorted(
            set(record).difference(EXPECTED_METRIC_FIELDS)
        )
        missing = sorted(
            EXPECTED_METRIC_FIELDS.difference(record)
        )
        raise ValueError(
            "Training metric fields differ from the frozen "
            f"curve-only schema; unexpected={unexpected}, "
            f"missing={missing}."
        )


def validate_registry(
    registry: Mapping[str, Any],
    *,
    implementation_commit: str,
    matched_horizon: int,
) -> None:
    """Validate the complete deterministic five-candidate registry."""

    if registry.get("implementation_commit") != (
        implementation_commit
    ):
        raise ValueError(
            "Candidate registry implementation commit mismatch."
        )

    if registry.get("matched_horizon") != matched_horizon:
        raise ValueError(
            "Candidate registry matched horizon mismatch."
        )

    if registry.get("model_seed") != 0:
        raise ValueError(
            "Candidate registry model seed must equal 0."
        )

    if registry.get("candidate_execution_order") != list(
        DESCENDING_CANDIDATE_FRACTIONS
    ):
        raise ValueError(
            "Candidate registry execution order does not match "
            "the frozen descending order."
        )

    if (
        registry.get("no_control_circuit_metric_inspected")
        is not True
    ):
        raise ValueError(
            "Candidate registry does not affirm the "
            "curve-only evidence boundary."
        )

    if (
        registry.get("stage14_started") is not False
        or registry.get("stage15_started") is not False
    ):
        raise ValueError(
            "Candidate registry indicates a later stage began."
        )

    candidates = registry.get("candidates")

    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ValueError(
            "Candidate registry must contain exactly five runs."
        )

    fractions = [
        float(candidate["fraction"])
        for candidate in candidates
    ]

    if fractions != list(DESCENDING_CANDIDATE_FRACTIONS):
        raise ValueError(
            "Candidate registry rows are not in the frozen "
            "descending order."
        )

    if len(set(fractions)) != 5:
        raise ValueError(
            "Candidate registry fractions must be unique."
        )


def _validate_checkpoint_records(
    *,
    repository: Path,
    manifest: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    checkpoints = manifest.get("checkpoints")

    if not isinstance(checkpoints, list):
        raise ValueError(
            "Training manifest checkpoints must be a list."
        )

    expected_steps = list(range(0, MATCHED_HORIZON + 1, 50))
    actual_steps = [
        checkpoint["training_step"]
        for checkpoint in checkpoints
    ]

    if actual_steps != expected_steps:
        raise ValueError(
            "Training manifest checkpoint grid is incomplete."
        )

    required_verified = set(
        STAGE13_CHECKPOINT_VALIDATION_STEPS
    )
    actual_verified = {
        checkpoint["training_step"]
        for checkpoint in checkpoints
        if checkpoint["reload_verified"]
    }

    if actual_verified != required_verified:
        raise ValueError(
            "Reload-verified checkpoint set does not match "
            "the frozen Stage 13 validation set."
        )

    by_step: dict[int, Mapping[str, Any]] = {}

    for checkpoint in checkpoints:
        path = resolve(repository, checkpoint["path"])

        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {path}"
            )

        if file_sha256(path) != checkpoint["file_sha256"]:
            raise ValueError(
                "Checkpoint physical hash mismatch at step "
                f"{checkpoint['training_step']}."
            )

        by_step[int(checkpoint["training_step"])] = checkpoint

    return by_step


def load_candidate_metric_rows(
    *,
    repository: Path,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load and validate all candidate manifests and curves."""

    stage13_run_id = str(registry["stage13_run_id"])
    implementation_commit = str(
        registry["implementation_commit"]
    )
    rows: list[dict[str, Any]] = []

    for candidate in registry["candidates"]:
        fraction = float(candidate["fraction"])
        exact_count = int(
            candidate["exact_training_example_count"]
        )
        run_id = str(candidate["candidate_run_id"])
        manifest_path = resolve(
            repository,
            candidate["training_manifest_path"],
        )
        metrics_path = resolve(
            repository,
            candidate["metrics_path"],
        )

        if file_sha256(manifest_path) != _require_sha256(
            candidate["training_manifest_sha256"],
            "training_manifest_sha256",
        ):
            raise ValueError(
                f"Training manifest hash mismatch for {fraction:.2f}."
            )

        if file_sha256(metrics_path) != _require_sha256(
            candidate["metrics_sha256"],
            "metrics_sha256",
        ):
            raise ValueError(
                f"Metrics hash mismatch for {fraction:.2f}."
            )

        manifest = load_json_object(
            manifest_path,
            "candidate training manifest",
        )

        if manifest.get("run_id") != run_id:
            raise ValueError(
                "Candidate run ID differs between registry "
                "and training manifest."
            )

        if manifest.get("git_commit") != implementation_commit:
            raise ValueError(
                "Candidate training manifest implementation "
                "commit mismatch."
            )

        if manifest.get("mode") != "full":
            raise ValueError(
                "Candidate training run was not a full run."
            )

        if manifest.get("seed", {}).get("value") != 0:
            raise ValueError(
                "Candidate training model seed must equal 0."
            )

        execution = manifest.get("execution", {})

        if (
            execution.get("max_steps") != MATCHED_HORIZON
            or execution.get("evaluation_interval") != 50
            or execution.get("checkpoint_interval") != 50
        ):
            raise ValueError(
                "Candidate execution schedule differs from "
                "the frozen Stage 13 schedule."
            )

        device = manifest.get("device", {}).get(
            "selected_device"
        )

        if device not in {"cpu", "cuda"}:
            raise ValueError(
                "Candidate training used an impermissible device."
            )

        subset = manifest.get("dataset", {}).get(
            "training_subset"
        )

        if not isinstance(subset, Mapping):
            raise ValueError(
                "Candidate manifest lacks training-subset provenance."
            )

        expected_subset = {
            "fraction": fraction,
            "subset_identifier": candidate[
                "subset_identifier"
            ],
            "exact_example_count": exact_count,
            "subset_sha256": candidate["subset_sha256"],
            "source_permutation_sha256": candidate[
                "source_permutation_sha256"
            ],
            "true_labels": True,
            "random_labels": False,
        }

        for name, expected in expected_subset.items():
            if subset.get(name) != expected:
                raise ValueError(
                    "Candidate subset provenance mismatch for "
                    f"{fraction:.2f}: {name}."
                )

        dataset = manifest["dataset"]

        if (
            dataset.get("train_count") != exact_count
            or dataset.get("test_count") != 8_939
        ):
            raise ValueError(
                "Candidate train or test example count mismatch."
            )

        identity = manifest.get("run_identity")

        if not isinstance(identity, Mapping):
            raise ValueError(
                "Candidate manifest lacks Stage 13 run identity."
            )

        if (
            identity.get("stage13_run_id") != stage13_run_id
            or identity.get("candidate_fraction") != fraction
            or identity.get("implementation_commit")
            != implementation_commit
            or identity.get("true_labels") is not True
            or identity.get("random_labels") is not False
            or identity.get("test_set_unchanged") is not True
        ):
            raise ValueError(
                "Candidate Stage 13 run identity is inconsistent."
            )

        acceptance = manifest.get("acceptance", {})

        if (
            acceptance.get(
                "checkpoint_reload_verification"
            )
            != "passed"
            or acceptance.get(
                "verified_checkpoint_count"
            )
            != len(STAGE13_CHECKPOINT_VALIDATION_STEPS)
            or acceptance.get(
                "verified_checkpoint_steps"
            )
            != list(STAGE13_CHECKPOINT_VALIDATION_STEPS)
        ):
            raise ValueError(
                "Candidate checkpoint reload validation is incomplete."
            )

        checkpoints = _validate_checkpoint_records(
            repository=repository,
            manifest=manifest,
        )
        metric_records = read_jsonl(metrics_path)

        if len(metric_records) != 182:
            raise ValueError(
                "Candidate metrics must contain exactly 182 records."
            )

        for metric in metric_records:
            validate_metric_record_fields(metric)
            step = int(metric["training_step"])

            if metric["run_id"] != run_id:
                raise ValueError(
                    "Metric record run ID mismatch."
                )

            if metric["mode"] != "full":
                raise ValueError(
                    "Metric record mode must equal full."
                )

            checkpoint = checkpoints.get(step)

            if checkpoint is None:
                raise ValueError(
                    "Metric record has no matching checkpoint."
                )

            comparisons = {
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["file_sha256"],
                "model_state_sha256": checkpoint[
                    "model_state_sha256"
                ],
                "optimizer_state_sha256": checkpoint[
                    "optimizer_state_sha256"
                ],
            }

            for name, expected in comparisons.items():
                if metric[name] != expected:
                    raise ValueError(
                        "Metric/checkpoint provenance mismatch "
                        f"for {name} at step {step}."
                    )

            rows.append(
                {
                    "fraction": fraction,
                    "exact_training_example_count": exact_count,
                    "subset_identifier": candidate[
                        "subset_identifier"
                    ],
                    "subset_sha256": candidate[
                        "subset_sha256"
                    ],
                    "run_id": run_id,
                    "training_git_commit": (
                        implementation_commit
                    ),
                    "device": device,
                    "training_step": step,
                    "learning_rate": metric["learning_rate"],
                    "weight_norm": metric["weight_norm"],
                    "gradient_norm": metric["gradient_norm"],
                    "train_loss": metric["train_loss"],
                    "test_loss": metric["test_loss"],
                    "train_accuracy": metric[
                        "train_accuracy"
                    ],
                    "test_accuracy": metric["test_accuracy"],
                    "checkpoint_path": metric[
                        "checkpoint_path"
                    ],
                    "checkpoint_sha256": metric[
                        "checkpoint_sha256"
                    ],
                    "model_state_sha256": metric[
                        "model_state_sha256"
                    ],
                    "optimizer_state_sha256": metric[
                        "optimizer_state_sha256"
                    ],
                }
            )

    if any(
        set(row) != set(STAGE13_METRICS_COLUMNS)
        for row in rows
    ):
        raise RuntimeError(
            "Combined metric rows differ from the frozen schema."
        )

    validate_stage13_metrics_rows(rows)
    return rows


def evaluate_saved_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[CandidateQualification], float | None]:
    """Evaluate all five candidates from the saved curve table."""

    validate_stage13_metrics_rows(rows)
    qualifications: list[CandidateQualification] = []

    for fraction in DESCENDING_CANDIDATE_FRACTIONS:
        candidate_rows = sorted(
            (
                row
                for row in rows
                if float(row["fraction"]) == fraction
            ),
            key=lambda row: int(row["training_step"]),
        )
        counts = {
            int(row["exact_training_example_count"])
            for row in candidate_rows
        }

        if len(counts) != 1:
            raise ValueError(
                "Candidate exact example count is inconsistent."
            )

        qualifications.append(
            evaluate_candidate(
                training_steps=[
                    int(row["training_step"])
                    for row in candidate_rows
                ],
                train_accuracy=[
                    float(row["train_accuracy"])
                    for row in candidate_rows
                ],
                test_accuracy=[
                    float(row["test_accuracy"])
                    for row in candidate_rows
                ],
                test_cross_entropy=[
                    float(row["test_loss"])
                    for row in candidate_rows
                ],
                candidate_fraction=fraction,
                training_example_count=counts.pop(),
                required_horizon=MATCHED_HORIZON,
            )
        )

    selected = select_largest_qualifying_fraction(
        qualifications
    )
    return qualifications, selected


def _serialise_csv(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Selection CSV cannot contain non-finite values."
            )
        return repr(value)

    return str(value)


def write_selection_csv(
    qualifications: Sequence[CandidateQualification],
    *,
    selected_fraction: float | None,
    path: str | Path,
) -> Path:
    """Write the complete deterministic qualification table."""

    if len(qualifications) != 5:
        raise ValueError(
            "Selection table requires exactly five candidates."
        )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SELECTION_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()

        for qualification in qualifications:
            row = qualification.as_row(
                selected_fraction=selected_fraction
            )

            if tuple(row) != SELECTION_COLUMNS:
                raise RuntimeError(
                    "Qualification row field order changed."
                )

            writer.writerow(
                {
                    name: _serialise_csv(row[name])
                    for name in SELECTION_COLUMNS
                }
            )

    return output


def write_selection_note(
    path: str | Path,
    *,
    stage13_run_id: str,
    qualifications: Sequence[CandidateQualification],
    selected_fraction: float | None,
) -> Path:
    """Write the human-readable mechanical selection record."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    outcome = (
        "`no_qualifying_fraction`"
        if selected_fraction is None
        else f"`{selected_fraction:.2f}`"
    )
    lines = [
        "# Stage 13 no-generalisation selection",
        "",
        f"- Stage 13 run ID: `{stage13_run_id}`",
        f"- Matched horizon: `{MATCHED_HORIZON}`",
        "- Candidate execution and selection order: "
        "`0.25, 0.20, 0.15, 0.10, 0.05`",
        "- Selection rule: choose the largest candidate satisfying "
        "all five frozen qualification criteria.",
        f"- Selection outcome: {outcome}",
        "- Permitted evidence: saved training accuracy, test "
        "accuracy and test cross-entropy curves only.",
        "- Control circuit-family metrics inspected: `false`",
        "- Stage 14 started: `false`",
        "- Stage 15 started: `false`",
        "",
        "## Qualification summary",
        "",
        "| Fraction | Count | First ≥99.9% step | Persistence | "
        "Max test accuracy | Loss fall | Qualified | Selected |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]

    for qualification in qualifications:
        first_step = (
            "none"
            if qualification.first_train_accuracy_step is None
            else str(
                qualification.first_train_accuracy_step
            )
        )
        selected = (
            selected_fraction
            == qualification.candidate_fraction
        )
        lines.append(
            "| "
            f"{qualification.candidate_fraction:.2f} | "
            f"{qualification.training_example_count} | "
            f"{first_step} | "
            f"{qualification.persistence_numerator}/"
            f"{qualification.persistence_denominator} "
            f"({qualification.persistence_proportion:.12g}) | "
            f"{qualification.maximum_test_accuracy:.12g} | "
            f"{qualification.test_cross_entropy_fractional_fall:.12g} | "
            f"{str(qualification.overall_qualification).lower()} | "
            f"{str(selected).lower()} |"
        )

    lines.extend(
        [
            "",
            "## Resolved interval ambiguity",
            "",
            "The frozen mathematical inequalities are followed: "
            "`8050 < step <= 9050` gives checkpoints "
            "`8100, 8150, ..., 9050`, while "
            "`7050 < step <= 8050` gives "
            "`7100, 7150, ..., 8050`. Both windows contain "
            "20 disjoint saved checkpoints.",
            "",
            "## Data-coverage limitation",
            "",
            DATA_COVERAGE_LIMITATION,
            "",
        ]
    )

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return output


def output_paths(
    repository: Path,
    stage13_run_id: str,
) -> dict[str, Path]:
    """Return every deterministic Stage 13 selection artifact."""

    return {
        "metrics_table": repository / METRICS_TABLE,
        "selection_table": repository / SELECTION_TABLE,
        "selection_note": repository / SELECTION_NOTE,
        "figure_png": repository / FIGURE_PNG,
        "figure_pdf": repository / FIGURE_PDF,
        "figure_caption": repository / FIGURE_CAPTION,
        "manifest": (
            repository
            / "manifests"
            / f"stage13_no_generalisation_"
            f"{stage13_run_id}.json"
        ),
    }


def validate_absent_outputs(
    paths: Mapping[str, Path],
) -> None:
    """Refuse all colliding or partial selection outputs."""

    existing = [
        path
        for path in paths.values()
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Stage 13 selection refuses to overwrite outputs:\n"
            + "\n".join(str(path) for path in existing)
        )


def generate_selection_artifacts(
    *,
    repository: Path,
    registry_path: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    """Generate every curve-only Stage 13 selection artifact."""

    registry = load_json_object(
        registry_path,
        "candidate registry",
    )
    validate_registry(
        registry,
        implementation_commit=implementation_commit,
        matched_horizon=MATCHED_HORIZON,
    )
    stage13_run_id = str(registry["stage13_run_id"])
    paths = output_paths(repository, stage13_run_id)
    validate_absent_outputs(paths)

    metric_rows = load_candidate_metric_rows(
        repository=repository,
        registry=registry,
    )

    with TemporaryDirectory(
        prefix=".stage13-selection-",
        dir=repository,
    ) as temporary:
        temporary_root = Path(temporary)
        temporary_paths = {
            name: temporary_root / path.name
            for name, path in paths.items()
        }

        write_stage13_metrics_csv(
            metric_rows,
            temporary_paths["metrics_table"],
        )
        saved_rows = read_stage13_metrics_csv(
            temporary_paths["metrics_table"]
        )
        qualifications, selected_fraction = (
            evaluate_saved_metrics(saved_rows)
        )
        write_selection_csv(
            qualifications,
            selected_fraction=selected_fraction,
            path=temporary_paths["selection_table"],
        )
        write_selection_note(
            temporary_paths["selection_note"],
            stage13_run_id=stage13_run_id,
            qualifications=qualifications,
            selected_fraction=selected_fraction,
        )
        plot_stage13_training_curves(
            saved_rows,
            selected_fraction=selected_fraction,
            png_path=temporary_paths["figure_png"],
            pdf_path=temporary_paths["figure_pdf"],
        )
        write_stage13_figure_caption(
            temporary_paths["figure_caption"],
            selected_fraction=selected_fraction,
        )

        artifact_records = {
            name: {
                "path": relative_path(
                    repository,
                    paths[name],
                ),
                "sha256": file_sha256(
                    temporary_paths[name]
                ),
            }
            for name in (
                "metrics_table",
                "selection_table",
                "selection_note",
                "figure_png",
                "figure_pdf",
                "figure_caption",
            )
        }
        figure_hashes = stage13_figure_hashes(
            png_path=temporary_paths["figure_png"],
            pdf_path=temporary_paths["figure_pdf"],
            caption_path=temporary_paths[
                "figure_caption"
            ],
        )
        qualification_rows = [
            qualification.as_row(
                selected_fraction=selected_fraction
            )
            for qualification in qualifications
        ]
        manifest = {
            "schema_version": 1,
            "experiment_type": (
                "stage13_no_generalisation_selection"
            ),
            "creation_timestamp_utc": utc_timestamp(),
            "stage13_run_id": stage13_run_id,
            "implementation_commit": implementation_commit,
            "source_protocol_pre_freeze_sha256": (
                file_sha256(repository / PROTOCOL_PATH)
            ),
            "source_candidate_registry": {
                "path": relative_path(
                    repository,
                    registry_path,
                ),
                "sha256": file_sha256(registry_path),
            },
            "matched_horizon": MATCHED_HORIZON,
            "model_seed": 0,
            "candidate_execution_order": list(
                DESCENDING_CANDIDATE_FRACTIONS
            ),
            "selection_rule": (
                "Select the largest candidate fraction "
                "satisfying all five prespecified criteria."
            ),
            "selection_evidence": (
                "training_and_test_curves_only"
            ),
            "selection_outcome": (
                "no_qualifying_fraction"
                if selected_fraction is None
                else "selected"
            ),
            "selected_fraction": selected_fraction,
            "candidate_qualification_rows": (
                qualification_rows
            ),
            "artifacts": artifact_records,
            "figure_hashes": figure_hashes,
            "checkpoint_validation": {
                "required_steps": list(
                    STAGE13_CHECKPOINT_VALIDATION_STEPS
                ),
                "all_candidates_passed": True,
                "all_checkpoint_file_hashes_verified": True,
            },
            "no_control_circuit_metric_inspected": True,
            "stage14_started": False,
            "stage15_started": False,
            "data_coverage_limitation": (
                DATA_COVERAGE_LIMITATION
            ),
            "resolved_protocol_ambiguity": {
                "formula_followed": True,
                "final_window_steps": (
                    "8100, 8150, ..., 9050"
                ),
                "preceding_window_steps": (
                    "7100, 7150, ..., 8050"
                ),
                "window_count_each": 20,
            },
        }
        write_manifest(
            temporary_paths["manifest"],
            manifest,
        )

        for name, destination in paths.items():
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.move(
                str(temporary_paths[name]),
                destination,
            )

    return {
        "stage13_run_id": stage13_run_id,
        "selected_fraction": selected_fraction,
        "selection_outcome": manifest[
            "selection_outcome"
        ],
        "qualification_rows": qualification_rows,
        "paths": paths,
    }


def main() -> None:
    """Generate definitive curve-only Stage 13 artifacts."""

    args = parse_args()
    repository = args.repository_root.resolve()

    if args.matched_horizon != MATCHED_HORIZON:
        raise ValueError(
            "Stage 13 matched horizon must remain 9050."
        )

    implementation_commit = (
        require_unchanged_implementation(
            repository,
            args.expected_implementation_commit,
        )
    )
    require_later_stage_outputs_absent(repository)
    registry_path = resolve(
        repository,
        args.candidate_registry,
    )
    result = generate_selection_artifacts(
        repository=repository,
        registry_path=registry_path,
        implementation_commit=implementation_commit,
    )

    print(
        "stage13_run_id: "
        f"{result['stage13_run_id']}"
    )
    print(
        "selection_outcome: "
        f"{result['selection_outcome']}"
    )
    selected = result["selected_fraction"]
    print(
        "selected_fraction: "
        + (
            "none"
            if selected is None
            else f"{selected:.2f}"
        )
    )

    for row in result["qualification_rows"]:
        print(
            "qualification: "
            f"fraction={row['fraction']:.2f} "
            f"qualified={str(row['overall_qualification']).lower()} "
            f"selected={str(row['selected_control']).lower()} "
            f"first_train_step={row['first_train_accuracy_step']} "
            f"persistence={row['persistence_numerator']}/"
            f"{row['persistence_denominator']} "
            f"max_test_accuracy={row['maximum_test_accuracy']} "
            f"loss_fall={row['test_cross_entropy_fractional_fall']}"
        )

    for name, path in result["paths"].items():
        print(
            f"{name}: {relative_path(repository, path)}"
        )
        print(
            f"{name}_sha256: {file_sha256(path)}"
        )

    print("no_control_circuit_metric_inspected: true")
    print("stage14_started: false")
    print("stage15_started: false")
    print("stage13_curve_only_selection: passed")


if __name__ == "__main__":
    main()
