"""Prepare the final Stage 22 analysis freeze and prediction resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuit_families.analysis.stage18_scaling import stable_json, write_csv
from circuit_families.analysis.stage22_freeze import (
    freeze_rows,
    prediction_table_sha256,
    resolution_rows,
)
from circuit_families.training import file_sha256

ORIGINAL_PROTOCOL_COMMIT = "6ca148b7916e16d121edb4827948a7af73197db2"
STAGE21_RUN_ID = "stage21-figures-e50a93433996"
STAGE21_MANIFEST = Path(f"manifests/stage21_figures_{STAGE21_RUN_ID}.json")
STAGE21_FINALIZATION = Path(f"manifests/stage21_finalization_{STAGE21_RUN_ID}.json")
OUTPUTS = {
    "prediction_resolution": Path("results/tables/stage22_prediction_resolution.csv"),
    "analysis_freeze": Path("results/tables/stage22_analysis_freeze.csv"),
    "exploratory_register": Path("results/tables/stage22_exploratory_analysis_register.csv"),
}
NOTE = Path("results/notes/stage22_analysis_freeze.md")


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def validate_inputs(repository: Path) -> tuple[str, str, str, bool, dict[str, str]]:
    if _git(repository, "branch", "--show-current").strip() != "main":
        raise ValueError("Stage 22 requires branch main.")
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if subprocess.run(("git", *arguments), cwd=repository, check=False).returncode:
            raise ValueError("Stage 22 requires a clean tracked repository.")
    commit = _git(repository, "rev-parse", "HEAD").strip()
    finalization_path = repository / STAGE21_FINALIZATION
    finalization = _object(finalization_path)
    if finalization.get("stage21_run_id") != STAGE21_RUN_ID:
        raise ValueError("Stage 21 finalization run ID mismatch.")
    if finalization.get("status") != "finalized":
        raise ValueError("Stage 21 must be finalized before Stage 22.")
    if finalization.get("stage21_finalized") is not True:
        raise ValueError("Stage 21 finalization flag is missing.")
    if finalization.get("stage18_reproduction_status") != "passed":
        raise ValueError("Stage 18 reproduction must pass before Stage 22.")
    if finalization.get("passed") is not True:
        raise ValueError("Stage 21 revalidation did not pass.")
    source = finalization.get("provisional_source", {})
    if source.get("manifest_path") != str(STAGE21_MANIFEST):
        raise ValueError("Stage 21 finalization source path mismatch.")
    manifest_path = repository / STAGE21_MANIFEST
    manifest = _object(manifest_path)
    if manifest.get("stage21_run_id") != STAGE21_RUN_ID:
        raise ValueError("Stage 21 source run ID mismatch.")
    manifest_hash = file_sha256(manifest_path)
    if source.get("manifest_sha256") != manifest_hash:
        raise ValueError("Stage 21 finalization source hash mismatch.")
    verified_outputs = finalization.get("revalidation", {}).get("verified_outputs")
    if verified_outputs != manifest.get("outputs"):
        raise ValueError("Stage 21 finalization output inventory mismatch.")
    hashes = {
        str(STAGE21_FINALIZATION): file_sha256(finalization_path),
        str(STAGE21_MANIFEST): manifest_hash,
    }
    for relative, expected in manifest["outputs"].items():
        actual = file_sha256(repository / relative)
        if actual != expected:
            raise ValueError(f"Stage 21 output hash mismatch: {relative}")
        hashes[relative] = actual
    current_protocol = (repository / "experimental_protocol.md").read_text(encoding="utf-8")
    original_protocol = _git(
        repository, "show", f"{ORIGINAL_PROTOCOL_COMMIT}:experimental_protocol.md"
    )
    current_hash = prediction_table_sha256(current_protocol)
    original_hash = prediction_table_sha256(original_protocol)
    if current_hash != original_hash:
        raise ValueError("The frozen prediction table changed after project initiation.")
    hashes["experimental_protocol.md"] = file_sha256(repository / "experimental_protocol.md")
    return (
        commit,
        hashes[str(STAGE21_FINALIZATION)],
        current_hash,
        True,
        hashes,
    )


def deterministic_run_id(stage21_hash: str, commit: str, prediction_hash: str) -> str:
    value = f"stage22|{stage21_hash}|{commit}|{prediction_hash}|analysis-freeze-v1"
    return f"stage22-freeze-{hashlib.sha256(value.encode('ascii')).hexdigest()[:12]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Stage 22 analysis freeze.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-inputs-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()
    commit, stage21_hash, prediction_hash, reproduced, sources = validate_inputs(repository)
    run_id = deterministic_run_id(stage21_hash, commit, prediction_hash)
    if args.validate_inputs_only:
        print("stage22_validate_inputs_only: passed")
        print(f"implementation_commit: {commit}")
        print(f"stage22_run_id: {run_id}")
        print(f"prediction_table_sha256: {prediction_hash}")
        print("prediction_table_unchanged: true")
        print(f"stage18_reproduction_passed: {str(reproduced).lower()}")
        return
    manifest_path = repository / f"manifests/stage22_freeze_{run_id}.json"
    candidates = [repository / path for path in OUTPUTS.values()]
    candidates.extend((repository / NOTE, manifest_path))
    if any(path.exists() for path in candidates):
        raise FileExistsError("Stage 22 outputs already exist.")
    protocol = (repository / "experimental_protocol.md").read_text(encoding="utf-8")
    rows = resolution_rows(protocol)
    outputs = [
        write_csv(repository / OUTPUTS["prediction_resolution"], rows),
        write_csv(repository / OUTPUTS["analysis_freeze"], freeze_rows()),
        write_csv(
            repository / OUTPUTS["exploratory_register"],
            (
                {
                    "entry_id": "none_recorded",
                    "status": "none",
                    "description": "No post-result exploratory analysis added through Stage 22.",
                },
            ),
        ),
    ]
    status = "analysis_frozen"
    note_path = repository / NOTE
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "# Stage 22 analysis freeze\n\n"
        f"Status: `{status}`.\n\n"
        f"The nine-row prediction table is unchanged from `{ORIGINAL_PROTOCOL_COMMIT}` "
        f"with SHA-256 `{prediction_hash}`. Eight directional predictions remain "
        "unresolved because primary pre-grid families are empty or their dependent "
        "metrics are undefined. The prespecified empty-family handling prediction is "
        "supported. The final analysis freeze was released after Stage 18 independent "
        "reproduction passed and Stages 19--21 were revalidated.\n",
        encoding="utf-8",
    )
    outputs.append(note_path)
    stable_json(
        manifest_path,
        {
            "schema_version": 1,
            "experiment_stage": 22,
            "stage22_run_id": run_id,
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "analysis_freeze_finalized": True,
            "analysis_frozen": True,
            "stage18_reproduction_passed": True,
            "stage18_reproduction_status": "passed",
            "implementation_commit": commit,
            "upstream_stage21": {
                "run_id": STAGE21_RUN_ID,
                "finalization_path": str(STAGE21_FINALIZATION),
                "finalization_sha256": stage21_hash,
                "status": "finalized",
            },
            "prediction_table": {
                "original_protocol_commit": ORIGINAL_PROTOCOL_COMMIT,
                "original_and_current_sha256": prediction_hash,
                "unchanged": True,
                "rewritten": False,
            },
            "prediction_resolution_counts": {"Supported": 1, "Unresolved": 8},
            "source_artifacts": sources,
            "outputs": {str(path.relative_to(repository)): file_sha256(path) for path in outputs},
        },
    )
    print(f"stage22_run_id: {run_id}")
    print(f"status: {status}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
