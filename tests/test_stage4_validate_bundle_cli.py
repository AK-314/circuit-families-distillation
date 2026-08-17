from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/stage4_validate_bundle.py"

VALID_BUNDLE = (
    ROOT / "tests/fixtures/stage4/synthetic_valid_bundle_v1.json"
)
INVALID_BUNDLE = (
    ROOT / "tests/fixtures/stage4/synthetic_invalid_bundle_v1.json"
)


def run_cli(bundle: Path, *, cwd: Path | None = None):
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--bundle",
            str(bundle),
        ],
        cwd=cwd,
        text=True,
        capture_output=True,
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_valid_bundle_passes():
    proc = run_cli(VALID_BUNDLE)

    assert proc.returncode == 0
    assert "PASS stage4_bundle_validation" in proc.stdout
    assert "records=15" in proc.stdout
    assert "record_types=14" in proc.stdout
    assert "scientific_computation=NO" in proc.stdout
    assert "artifact_generation=NO" in proc.stdout
    assert proc.stderr == ""


def test_cli_invalid_bundle_fails():
    proc = run_cli(INVALID_BUNDLE)

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "FAIL stage4_bundle_validation" in proc.stderr
    assert "endpoint_1 does not reconstruct" in proc.stderr


def test_cli_missing_bundle_fails_boundedly():
    proc = run_cli(ROOT / "tests/fixtures/stage4/does-not-exist.json")

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "FAIL stage4_bundle_validation" in proc.stderr
    assert "file not found" in proc.stderr


def test_cli_malformed_json_fails_boundedly():
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "malformed.json"
        path.write_text("{not-json", encoding="utf-8")

        proc = run_cli(path)

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "FAIL stage4_bundle_validation" in proc.stderr
    assert "invalid JSON" in proc.stderr


def test_cli_missing_records_mapping_fails():
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "wrong-shape.json"
        path.write_text(
            json.dumps({"not_records": {}}),
            encoding="utf-8",
        )

        proc = run_cli(path)

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "records_by_sha256" in proc.stderr


def test_cli_is_cwd_independent():
    with tempfile.TemporaryDirectory() as raw:
        proc = run_cli(
            VALID_BUNDLE.resolve(),
            cwd=Path(raw),
        )

    assert proc.returncode == 0
    assert "PASS stage4_bundle_validation" in proc.stdout


def test_cli_does_not_modify_valid_bundle():
    before = file_sha256(VALID_BUNDLE)

    proc = run_cli(VALID_BUNDLE)

    after = file_sha256(VALID_BUNDLE)

    assert proc.returncode == 0
    assert before == after


def test_cli_does_not_modify_invalid_bundle():
    before = file_sha256(INVALID_BUNDLE)

    proc = run_cli(INVALID_BUNDLE)

    after = file_sha256(INVALID_BUNDLE)

    assert proc.returncode == 2
    assert before == after


def test_cli_source_contains_no_generation_entrypoint():
    text = CLI.read_text()

    prohibited = (
        "torch.save(",
        "np.save(",
        "subprocess.Popen(",
        "model.train(",
        "optimizer.step(",
    )

    for token in prohibited:
        assert token not in text
