from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from circuit_families.stage7b.registered_fixture import (
    RegisteredFixtureBindings,
    RegisteredFixtureError,
    run_registered_fixture,
    validate_registered_fixture_identity,
)

REPO = Path(__file__).resolve().parents[1]
SOURCE_REQUEST = (
    REPO / "followup/configs/stage7b/registered_fixture_request_v1.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _fixture_identity(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    repo = tmp_path / "successor"
    pred = tmp_path / "predecessor"

    manifest = (
        pred
        / "manifests"
        / "training_stage18-main-training-s0-58b8c1235464.json"
    )
    checkpoint = (
        pred
        / "checkpoints"
        / "stage18-main-training-s0-58b8c1235464"
        / "step_00005900.pt"
    )

    _write_json(
        manifest,
        {
            "model_seed": 0,
            "modulus": 113,
            "training_steps": 40000,
            "dataset_identity": "temporary-portable-stage7b-fixture",
        },
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"portable-stage7b-checkpoint-v1")

    request = json.loads(SOURCE_REQUEST.read_text())
    request["registered_teacher"]["checkpoint_sha256"] = _sha(checkpoint)

    registry = {
        "schema_version": "temporary-stage7b-registry",
        "records": [
            {
                "canonical_run_id":
                    "stage18-main-training-s0-58b8c1235464",
                "teacher_seed": 0,
                "phase_label": "stable post-grokking",
                "training_step": 5900,
                "checkpoint_path":
                    "checkpoints/"
                    "stage18-main-training-s0-58b8c1235464/"
                    "step_00005900.pt",
                "checkpoint_sha256": _sha(checkpoint),
                "training_manifest_path": str(manifest.relative_to(pred)),
                "training_manifest_sha256": _sha(manifest),
            }
        ],
    }

    _write_json(
        repo / "followup/manifests/stage3_teacher_registry_v1.json",
        registry,
    )

    request_path = (
        repo
        / "followup/configs/stage7b/registered_fixture_request_v1.json"
    )
    _write_json(request_path, request)

    return repo, pred, request_path, request


class FakeAcceptedBindings:
    def __init__(
        self,
        *,
        hard_eligible: bool = False,
        soft_eligible: bool = True,
        primary_eligible: bool = False,
        endpoint_nonce: str = "stable",
    ) -> None:
        self.hard_eligible = hard_eligible
        self.soft_eligible = soft_eligible
        self.primary_eligible = primary_eligible
        self.endpoint_nonce = endpoint_nonce
        self.events: list[tuple[Any, ...]] = []
        self.endpoint_payloads: list[dict[str, Any]] = []
        self.batch_sizes: list[int] = []

    def load_checkpoint_payload(self, path: Path) -> dict[str, Any]:
        self.events.append(("checkpoint_payload", path.name))
        return {
            "training_step": 5900,
            "model_seed": 0,
            "model_state": {"temporary_weight": "portable"},
            "model_state_sha256": "0" * 64,
        }

    def restore_model(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_payload: dict[str, Any],
        device: str,
    ) -> dict[str, Any]:
        self.events.append(("restore", device))
        return {"registered_teacher": True}

    def evaluate_teacher(
        self,
        *,
        model: Any,
        domain_inputs: np.ndarray,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        self.events.append(("teacher_eval", len(domain_inputs), device))
        self.batch_sizes.append(batch_size)

        a = domain_inputs[:, 0].astype(np.float64)
        b = domain_inputs[:, 1].astype(np.float64)
        logits = np.stack(
            (
                a + b,
                a - b,
                b - a,
                (a * 0.5) + (b * 0.25),
                np.zeros_like(a),
            ),
            axis=1,
        )
        return logits

    def build_target_caches(
        self,
        *,
        domain_inputs: np.ndarray,
        hard_targets: np.ndarray,
        centred_teacher_logits: np.ndarray,
        output_root: Path,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.events.append(
            (
                "caches",
                len(domain_inputs),
                hard_targets.shape,
                centred_teacher_logits.shape,
            )
        )
        return {
            "hard": {"kind": "hard", "count": len(domain_inputs)},
            "soft": {"kind": "soft", "count": len(domain_inputs)},
        }

    def run_student_attempt(
        self,
        *,
        target_kind: str,
        target_cache: Any,
        attempt_index: int,
        work_units: int,
        safety_ceiling: int,
        request: dict[str, Any],
        output_root: Path,
    ) -> dict[str, Any]:
        self.events.append(
            (
                "attempt",
                target_kind,
                attempt_index,
                work_units,
                safety_ceiling,
            )
        )
        assert attempt_index == 0
        assert work_units == 1
        assert safety_ceiling == 1
        return {
            "target_kind": target_kind,
            "attempt_index": attempt_index,
        }

    def assess_student_attempt(
        self,
        *,
        target_kind: str,
        attempt_result: Any,
        teacher_hard_targets: np.ndarray,
        teacher_centred_logits: np.ndarray,
        domain_inputs: np.ndarray,
        request: dict[str, Any],
        output_root: Path,
    ) -> dict[str, Any]:
        eligible = (
            self.hard_eligible
            if target_kind == "hard"
            else self.soft_eligible
        )
        self.events.append(("assessment", target_kind, eligible))

        return {
            "status": "eligible" if eligible else "failed",
            "eligible": eligible,
            "sealed": eligible,
            "sealed_subject": (
                {"student_kind": target_kind}
                if eligible
                else None
            ),
            "failure_retained": not eligible,
            "imputed": False,
        }

    def run_discovery(
        self,
        *,
        adapter_name: str,
        subject_kind: str,
        subject: Any,
        request: dict[str, Any],
        output_root: Path,
    ) -> dict[str, Any]:
        self.events.append(("discovery", subject_kind, adapter_name))
        return {
            "adapter_name": adapter_name,
            "subject_kind": subject_kind,
            "proposal_source": "injected-real-adapter-stand-in",
            "native_budget_authority": "stage6d",
            "nonce": self.endpoint_nonce,
        }

    def run_exact_endpoints(
        self,
        *,
        adapter_name: str,
        subject_kind: str,
        discovery_result: Any,
        teacher_centred_logits: np.ndarray,
        domain_inputs: np.ndarray,
        request: dict[str, Any],
        output_root: Path,
    ) -> dict[str, Any]:
        payload = {
            "adapter_name": adapter_name,
            "subject_kind": subject_kind,
            "native_budget_authority": "stage6d",
            "exact_budget_authority": "stage6a",
            "native_and_exact_budgets_separate": True,
            "endpoint1": {
                "intact_model_evaluated": True,
                "definition": "accepted-stage6a",
                "nonce": self.endpoint_nonce,
            },
            "endpoint2": {
                "qualified_count": 0,
                "packing_lower_bound": 0,
                "definition": "accepted-stage6e",
                "nonce": self.endpoint_nonce,
            },
        }
        self.endpoint_payloads.append(payload)
        self.events.append(("exact", subject_kind, adapter_name))
        return payload

    def build_excluded_outputs(
        self,
        *,
        identity: Any,
        request: dict[str, Any],
        source_code_sha: str,
        attempt_assessments: dict[str, dict[str, Any]],
        discovery_records: list[dict[str, Any]],
        endpoint_records: list[dict[str, Any]],
        output_root: Path,
    ) -> dict[str, Any]:
        self.events.append(("exclude", len(endpoint_records)))

        records = []
        for index, endpoint in enumerate(endpoint_records):
            records.append(
                {
                    "record_id": f"excluded-{index}",
                    "lifecycle_state": "excluded",
                    "primary_input_eligible": self.primary_eligible,
                    "regeneration_required_after_definitive_freeze": True,
                    "source_code_sha": source_code_sha,
                    "teacher_seed": identity.teacher_seed,
                    "canonical_run_id": identity.canonical_run_id,
                    "endpoint_record_hash": hashlib.sha256(
                        json.dumps(
                            endpoint,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    "scientific_interpretation": False,
                }
            )

        return {
            "exclusion_records": records,
            "report": {
                "schema_version": "temporary-stage7b-report",
                "teacher_seed": identity.teacher_seed,
                "hard_soft_pooled": False,
                "stage8_status": "NOT_STARTED",
                "scientific_data": False,
            },
        }

    def bindings(self) -> RegisteredFixtureBindings:
        return RegisteredFixtureBindings(
            restore_model=self.restore_model,
            evaluate_teacher=self.evaluate_teacher,
            build_target_caches=self.build_target_caches,
            run_student_attempt=self.run_student_attempt,
            assess_student_attempt=self.assess_student_attempt,
            run_discovery=self.run_discovery,
            run_exact_endpoints=self.run_exact_endpoints,
            build_excluded_outputs=self.build_excluded_outputs,
            load_checkpoint_payload=self.load_checkpoint_payload,
            discovery_adapter_names=(
                "greedy_deletion",
                "diversity_forced",
            ),
        )


def _run(
    tmp_path: Path,
    *,
    name: str,
    hard_eligible: bool = False,
    soft_eligible: bool = True,
    primary_eligible: bool = False,
    endpoint_nonce: str = "stable",
):
    repo, pred, request_path, request = _fixture_identity(tmp_path)
    fake = FakeAcceptedBindings(
        hard_eligible=hard_eligible,
        soft_eligible=soft_eligible,
        primary_eligible=primary_eligible,
        endpoint_nonce=endpoint_nonce,
    )
    output = repo / "followup/local/stage7b" / name
    result = run_registered_fixture(
        repository_root=repo,
        predecessor_root=pred,
        output_root=output,
        request_path=request_path,
        bindings=fake.bindings(),
    )
    return repo, pred, request_path, request, output, fake, result


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("teacher_seed", 1),
        ("phase_label", "transition"),
        ("training_step", 5850),
        (
            "checkpoint_path_relative_to_predecessor",
            "checkpoints/wrong/step_00005900.pt",
        ),
        ("checkpoint_sha256", "f" * 64),
    ],
)
def test_wrong_registered_identity_fields_are_rejected(
    tmp_path: Path,
    field: str,
    bad_value: Any,
) -> None:
    repo, pred, request_path, request = _fixture_identity(tmp_path)
    request["registered_teacher"][field] = bad_value
    _write_json(request_path, request)

    with pytest.raises(
        RegisteredFixtureError,
        match="registered teacher identity did not bind exactly one",
    ):
        validate_registered_fixture_identity(
            repository_root=repo,
            predecessor_root=pred,
            request=request,
        )


def test_training_manifest_provenance_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    repo, pred, request_path, request = _fixture_identity(tmp_path)
    manifest = next((pred / "manifests").glob("*.json"))
    manifest.write_text('{"tampered": true}\n')

    with pytest.raises(
        RegisteredFixtureError,
        match="training-manifest hash mismatch",
    ):
        validate_registered_fixture_identity(
            repository_root=repo,
            predecessor_root=pred,
            request=request,
        )


def test_checkpoint_payload_step_and_seed_mismatch_are_rejected_before_restore(
    tmp_path: Path,
) -> None:
    repo, pred, request_path, request = _fixture_identity(tmp_path)

    restore_called = False

    def load_bad(_: Path) -> dict[str, Any]:
        return {
            "training_step": 5800,
            "model_seed": 0,
            "model_state": {"w": 1},
            "model_state_sha256": "0" * 64,
        }

    def restore(**_: Any) -> Any:
        nonlocal restore_called
        restore_called = True
        raise AssertionError("restore must not execute")

    fake = FakeAcceptedBindings()
    bindings = fake.bindings()
    bindings = RegisteredFixtureBindings(
        restore_model=restore,
        evaluate_teacher=bindings.evaluate_teacher,
        build_target_caches=bindings.build_target_caches,
        run_student_attempt=bindings.run_student_attempt,
        assess_student_attempt=bindings.assess_student_attempt,
        run_discovery=bindings.run_discovery,
        run_exact_endpoints=bindings.run_exact_endpoints,
        build_excluded_outputs=bindings.build_excluded_outputs,
        load_checkpoint_payload=load_bad,
        discovery_adapter_names=bindings.discovery_adapter_names,
    )

    with pytest.raises(
        RegisteredFixtureError,
        match="training-step mismatch",
    ):
        run_registered_fixture(
            repository_root=repo,
            predecessor_root=pred,
            output_root=repo / "followup/local/stage7b/run",
            request_path=request_path,
            bindings=bindings,
        )

    assert restore_called is False


def test_one_hard_one_soft_attempt_and_passed_only_student_discovery(
    tmp_path: Path,
) -> None:
    _, _, _, _, output, fake, result = _run(
        tmp_path,
        name="run",
        hard_eligible=False,
        soft_eligible=True,
    )

    attempts = [
        event for event in fake.events
        if event[0] == "attempt"
    ]
    assert attempts == [
        ("attempt", "hard", 0, 1, 1),
        ("attempt", "soft", 0, 1, 1),
    ]

    discoveries = [
        event for event in fake.events
        if event[0] == "discovery"
    ]
    assert discoveries == [
        ("discovery", "teacher", "greedy_deletion"),
        ("discovery", "teacher", "diversity_forced"),
        ("discovery", "soft_student", "greedy_deletion"),
        ("discovery", "soft_student", "diversity_forced"),
    ]

    hard = json.loads((output / "attempts/hard.json").read_text())
    soft = json.loads((output / "attempts/soft.json").read_text())

    assert hard["status"] == "failed"
    assert hard["failure_retained"] is True
    assert hard["imputed"] is False
    assert soft["status"] == "eligible"

    assert result.teacher_discovery_release_count == 2
    assert result.student_discovery_release_count == 2


def test_teacher_direct_discovery_runs_when_both_students_fail(
    tmp_path: Path,
) -> None:
    _, _, _, _, _, fake, result = _run(
        tmp_path,
        name="run",
        hard_eligible=False,
        soft_eligible=False,
    )

    discoveries = [
        event for event in fake.events
        if event[0] == "discovery"
    ]
    assert discoveries == [
        ("discovery", "teacher", "greedy_deletion"),
        ("discovery", "teacher", "diversity_forced"),
    ]
    assert result.teacher_discovery_release_count == 2
    assert result.student_discovery_release_count == 0


def test_real_adapter_injection_and_separate_budget_accounting(
    tmp_path: Path,
) -> None:
    _, _, _, _, _, fake, _ = _run(
        tmp_path,
        name="run",
        hard_eligible=False,
        soft_eligible=False,
    )

    assert fake.endpoint_payloads
    for payload in fake.endpoint_payloads:
        assert payload["adapter_name"] in {
            "greedy_deletion",
            "diversity_forced",
        }
        assert payload["native_budget_authority"] == "stage6d"
        assert payload["exact_budget_authority"] == "stage6a"
        assert payload["native_and_exact_budgets_separate"] is True


def test_intact_endpoint1_and_zero_endpoint2_packing_are_retained(
    tmp_path: Path,
) -> None:
    _, _, _, _, _, fake, result = _run(
        tmp_path,
        name="run",
        hard_eligible=False,
        soft_eligible=False,
    )

    assert len(fake.endpoint_payloads) == 2
    assert len(result.endpoint_record_hashes) == 2

    for payload in fake.endpoint_payloads:
        assert payload["endpoint1"]["intact_model_evaluated"] is True
        assert payload["endpoint1"]["definition"] == "accepted-stage6a"
        assert payload["endpoint2"]["qualified_count"] == 0
        assert payload["endpoint2"]["packing_lower_bound"] == 0
        assert payload["endpoint2"]["definition"] == "accepted-stage6e"


def test_every_endpoint_has_exclusion_and_primary_eligibility_is_zero(
    tmp_path: Path,
) -> None:
    _, _, _, _, _, fake, result = _run(
        tmp_path,
        name="run",
        hard_eligible=False,
        soft_eligible=True,
    )

    assert result.exclusion_record_count == len(fake.endpoint_payloads)
    assert result.primary_eligible_count == 0


def test_primary_eligible_excluded_output_is_rejected(
    tmp_path: Path,
) -> None:
    repo, pred, request_path, _ = _fixture_identity(tmp_path)
    fake = FakeAcceptedBindings(primary_eligible=True)

    with pytest.raises(
        RegisteredFixtureError,
        match="primary-eligible",
    ):
        run_registered_fixture(
            repository_root=repo,
            predecessor_root=pred,
            output_root=repo / "followup/local/stage7b/run",
            request_path=request_path,
            bindings=fake.bindings(),
        )


def test_independent_runs_match_deterministic_artifact_identities(
    tmp_path: Path,
) -> None:
    repo, pred, request_path, _ = _fixture_identity(tmp_path)

    first_fake = FakeAcceptedBindings()
    first = run_registered_fixture(
        repository_root=repo,
        predecessor_root=pred,
        output_root=repo / "followup/local/stage7b/source",
        request_path=request_path,
        bindings=first_fake.bindings(),
    )

    second_fake = FakeAcceptedBindings()
    second = run_registered_fixture(
        repository_root=repo,
        predecessor_root=pred,
        output_root=repo / "followup/local/stage7b/reproduction",
        request_path=request_path,
        bindings=second_fake.bindings(),
    )

    deterministic_fields = (
        "provenance_status",
        "hard_attempt_status",
        "soft_attempt_status",
        "teacher_discovery_release_count",
        "student_discovery_release_count",
        "discovery_result_count",
        "endpoint_record_hashes",
        "exclusion_record_count",
        "primary_eligible_count",
        "runtime_file_count",
        "runtime_total_bytes",
        "report_sha256",
        "inventory_sha256",
    )

    mismatch = {
        field: (getattr(first, field), getattr(second, field))
        for field in deterministic_fields
        if getattr(first, field) != getattr(second, field)
    }
    assert mismatch == {}


def test_independent_run_mismatch_is_diagnosable_by_artifact_field(
    tmp_path: Path,
) -> None:
    repo, pred, request_path, _ = _fixture_identity(tmp_path)

    first = run_registered_fixture(
        repository_root=repo,
        predecessor_root=pred,
        output_root=repo / "followup/local/stage7b/source",
        request_path=request_path,
        bindings=FakeAcceptedBindings(endpoint_nonce="source").bindings(),
    )
    second = run_registered_fixture(
        repository_root=repo,
        predecessor_root=pred,
        output_root=repo / "followup/local/stage7b/reproduction",
        request_path=request_path,
        bindings=FakeAcceptedBindings(endpoint_nonce="changed").bindings(),
    )

    deterministic_fields = (
        "endpoint_record_hashes",
        "report_sha256",
        "inventory_sha256",
    )
    mismatch = {
        field: {
            "source": getattr(first, field),
            "reproduction": getattr(second, field),
        }
        for field in deterministic_fields
        if getattr(first, field) != getattr(second, field)
    }

    assert "endpoint_record_hashes" in mismatch
    assert "inventory_sha256" in mismatch
    assert mismatch["endpoint_record_hashes"]["source"] != (
        mismatch["endpoint_record_hashes"]["reproduction"]
    )


def test_discovery_and_endpoint_order_is_deterministic(
    tmp_path: Path,
) -> None:
    _, _, _, _, _, fake, _ = _run(
        tmp_path,
        name="run",
        hard_eligible=True,
        soft_eligible=True,
    )

    discoveries = [
        event[1:]
        for event in fake.events
        if event[0] == "discovery"
    ]

    assert discoveries == [
        ("teacher", "greedy_deletion"),
        ("teacher", "diversity_forced"),
        ("hard_student", "greedy_deletion"),
        ("hard_student", "diversity_forced"),
        ("soft_student", "greedy_deletion"),
        ("soft_student", "diversity_forced"),
    ]


def test_runner_help_is_cwd_independent(tmp_path: Path) -> None:
    runner = REPO / "scripts/run_stage7b_registered_fixture.py"
    proc = subprocess.run(
        [sys.executable, str(runner), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--predecessor-root" in proc.stdout
    assert "--output-root" in proc.stdout


def test_portable_execution_writes_only_under_injected_runtime_root(
    tmp_path: Path,
) -> None:
    repo, _, _, _, output, _, _ = _run(
        tmp_path,
        name="run",
    )

    files = sorted(
        p.relative_to(repo)
        for p in repo.rglob("*")
        if p.is_file()
    )

    allowed_prefixes = (
        Path("followup/manifests"),
        Path("followup/configs/stage7b"),
        Path("followup/local/stage7b/run"),
    )

    for rel in files:
        assert any(
            rel == prefix or prefix in rel.parents
            for prefix in allowed_prefixes
        ), rel

    assert output.is_dir()


def test_fixed_batch_size_is_passed_to_teacher_evaluation(
    tmp_path: Path,
) -> None:
    _, _, _, request, _, fake, _ = _run(
        tmp_path,
        name="run",
    )
    assert fake.batch_sizes == [
        request["execution_engineering"]["teacher_forward_batch_size"]
    ]


def test_bridge_source_has_no_synthetic_proposal_fallback() -> None:
    text = (
        REPO
        / "src/circuit_families/stage7b/registered_fixture.py"
    ).read_text()

    assert "proposal_stream" not in text
    assert "GreedyDeletionAdapter" in text
    assert "DiversityForcedAdapter" in text
    assert "torch.optim." not in text


def test_accepted_stage7_reproduction_module_contains_explicit_mismatch_logic() -> None:
    text = (
        REPO
        / "src/circuit_families/stage7/reproduction.py"
    ).read_text().lower()

    assert "compare_independent_pipeline_reproduction" in text
    assert "mismatch" in text
    assert "reproduction" in text


def test_generic_training_step_is_run_metadata_not_selected_checkpoint(
    tmp_path: Path,
) -> None:
    repo, pred, request_path, request = _fixture_identity(tmp_path)
    manifest = next((pred / "manifests").glob("*.json"))

    _write_json(
        manifest,
        {
            "model_seed": 0,
            "modulus": 113,
            "training_step": 40000,
            "training_steps": 40000,
            "dataset_identity": "temporary-portable-stage7b-fixture",
        },
    )

    registry_path = (
        repo / "followup/manifests/stage3_teacher_registry_v1.json"
    )
    registry = json.loads(registry_path.read_text())
    registry["records"][0]["training_manifest_sha256"] = _sha(manifest)
    _write_json(registry_path, registry)

    identity, checkpoint, resolved_manifest, record = (
        validate_registered_fixture_identity(
            repository_root=repo,
            predecessor_root=pred,
            request=request,
        )
    )

    assert identity.training_step == 5900
    assert checkpoint.is_file()
    assert resolved_manifest == manifest
    assert record["training_step"] == 5900


def test_explicit_checkpoint_step_mismatch_remains_blocking(
    tmp_path: Path,
) -> None:
    repo, pred, request_path, request = _fixture_identity(tmp_path)
    manifest = next((pred / "manifests").glob("*.json"))

    _write_json(
        manifest,
        {
            "model_seed": 0,
            "modulus": 113,
            "training_steps": 40000,
            "selected_checkpoint_step": 5850,
        },
    )

    registry_path = (
        repo / "followup/manifests/stage3_teacher_registry_v1.json"
    )
    registry = json.loads(registry_path.read_text())
    registry["records"][0]["training_manifest_sha256"] = _sha(manifest)
    _write_json(registry_path, registry)

    with pytest.raises(
        RegisteredFixtureError,
        match="training-manifest explicit selected-checkpoint-step mismatch",
    ):
        validate_registered_fixture_identity(
            repository_root=repo,
            predecessor_root=pred,
            request=request,
        )


def test_checkpoint_cadence_and_final_step_do_not_override_stage3_selection(
    tmp_path: Path,
) -> None:
    repo, pred, request_path, request = _fixture_identity(tmp_path)
    manifest = next((pred / "manifests").glob("*.json"))

    _write_json(
        manifest,
        {
            "model_seed": 0,
            "modulus": 113,
            "training_step": 40000,
            "training_steps": 40000,
            "final_step": 40000,
            "checkpoint_every_steps": 50,
            "checkpoint_step_interval": 50,
            "save_every_steps": 50,
        },
    )

    registry_path = (
        repo / "followup/manifests/stage3_teacher_registry_v1.json"
    )
    registry = json.loads(registry_path.read_text())
    registry["records"][0]["training_manifest_sha256"] = _sha(manifest)
    _write_json(registry_path, registry)

    identity, checkpoint, resolved_manifest, record = (
        validate_registered_fixture_identity(
            repository_root=repo,
            predecessor_root=pred,
            request=request,
        )
    )

    assert identity.training_step == 5900
    assert checkpoint.is_file()
    assert resolved_manifest == manifest
    assert record["training_step"] == 5900


def test_unambiguous_registered_checkpoint_step_mismatch_is_still_rejected(
    tmp_path: Path,
) -> None:
    repo, pred, request_path, request = _fixture_identity(tmp_path)
    manifest = next((pred / "manifests").glob("*.json"))

    _write_json(
        manifest,
        {
            "model_seed": 0,
            "modulus": 113,
            "training_steps": 40000,
            "registered_checkpoint_step": 5850,
        },
    )

    registry_path = (
        repo / "followup/manifests/stage3_teacher_registry_v1.json"
    )
    registry = json.loads(registry_path.read_text())
    registry["records"][0]["training_manifest_sha256"] = _sha(manifest)
    _write_json(registry_path, registry)

    with pytest.raises(
        RegisteredFixtureError,
        match="training-manifest explicit selected-checkpoint-step mismatch",
    ):
        validate_registered_fixture_identity(
            repository_root=repo,
            predecessor_root=pred,
            request=request,
        )


def test_production_restore_uses_accepted_repository_constructor() -> None:
    text = (
        REPO
        / "src/circuit_families/stage7b/registered_fixture.py"
    ).read_text()

    start = text.index("    def restore_model(")
    end = text.index("    def evaluate_teacher(", start)
    restore = text[start:end]

    assert "from circuit_families.models import build_transformer" in restore
    assert "build_transformer(" in restore
    assert "model_config," in restore
    assert "model.load_state_dict(" in restore
    assert "strict=True" in restore
    assert "canonical_state_hash(model.state_dict())" in restore
    assert "HookedTransformerConfig" not in restore
    assert "HookedTransformer(" not in restore
