"""Portable Stage 7B registered-checkpoint bridge.

The bridge deliberately owns orchestration only.  Scientific or algorithmic
operations are supplied by accepted Stage 3–7 interfaces through
``RegisteredFixtureBindings``.  Portable tests inject temporary stand-ins.
The physical entry point performs exact provenance validation before any
binding capable of model execution is invoked.

No endpoint value is printed by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class RegisteredFixtureError(RuntimeError):
    """Raised when the Stage 7B bridge cannot proceed without violating policy."""


@dataclass(frozen=True)
class RegisteredFixtureIdentity:
    teacher_seed: int
    phase_label: str
    canonical_run_id: str
    training_step: int
    checkpoint_relative_path: str
    checkpoint_sha256: str
    complete_domain_size: int
    domain_order: str


@dataclass(frozen=True)
class RegisteredFixtureRun:
    """Compact technical execution result with no endpoint values."""

    provenance_status: str
    hard_attempt_status: str
    soft_attempt_status: str
    teacher_discovery_release_count: int
    student_discovery_release_count: int
    discovery_result_count: int
    endpoint_record_hashes: tuple[str, ...]
    exclusion_record_count: int
    primary_eligible_count: int
    runtime_file_count: int
    runtime_total_bytes: int
    report_sha256: str
    inventory_sha256: str


class _RestoreModel(Protocol):
    def __call__(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_payload: Mapping[str, Any],
        device: str,
    ) -> Any: ...


class _EvaluateTeacher(Protocol):
    def __call__(
        self,
        *,
        model: Any,
        domain_inputs: np.ndarray,
        batch_size: int,
        device: str,
    ) -> np.ndarray: ...


class _BuildTargetCaches(Protocol):
    def __call__(
        self,
        *,
        domain_inputs: np.ndarray,
        hard_targets: np.ndarray,
        centred_teacher_logits: np.ndarray,
        output_root: Path,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class _RunStudentAttempt(Protocol):
    def __call__(
        self,
        *,
        target_kind: str,
        target_cache: Any,
        attempt_index: int,
        work_units: int,
        safety_ceiling: int,
        request: Mapping[str, Any],
        output_root: Path,
    ) -> Any: ...


class _AssessStudentAttempt(Protocol):
    def __call__(
        self,
        *,
        target_kind: str,
        attempt_result: Any,
        teacher_hard_targets: np.ndarray,
        teacher_centred_logits: np.ndarray,
        domain_inputs: np.ndarray,
        request: Mapping[str, Any],
        output_root: Path,
    ) -> Mapping[str, Any]: ...


class _RunDiscovery(Protocol):
    def __call__(
        self,
        *,
        adapter_name: str,
        subject_kind: str,
        subject: Any,
        request: Mapping[str, Any],
        output_root: Path,
    ) -> Any: ...


class _RunExactEndpoints(Protocol):
    def __call__(
        self,
        *,
        adapter_name: str,
        subject_kind: str,
        discovery_result: Any,
        teacher_centred_logits: np.ndarray,
        domain_inputs: np.ndarray,
        request: Mapping[str, Any],
        output_root: Path,
    ) -> Mapping[str, Any]: ...


class _BuildExcludedOutputs(Protocol):
    def __call__(
        self,
        *,
        identity: RegisteredFixtureIdentity,
        request: Mapping[str, Any],
        source_code_sha: str,
        attempt_assessments: Mapping[str, Mapping[str, Any]],
        discovery_records: Sequence[Mapping[str, Any]],
        endpoint_records: Sequence[Mapping[str, Any]],
        output_root: Path,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RegisteredFixtureBindings:
    """Injected calls to accepted algorithms/interfaces.

    Stage 7B does not implement any of these algorithms.  The production
    binding factory must bind these hooks to the accepted Stage 3–7
    implementation; portable tests may inject deterministic stand-ins.
    """

    restore_model: _RestoreModel
    evaluate_teacher: _EvaluateTeacher
    build_target_caches: _BuildTargetCaches
    run_student_attempt: _RunStudentAttempt
    assess_student_attempt: _AssessStudentAttempt
    run_discovery: _RunDiscovery
    run_exact_endpoints: _RunExactEndpoints
    build_excluded_outputs: _BuildExcludedOutputs
    load_checkpoint_payload: Callable[[Path], Mapping[str, Any]]
    discovery_adapter_names: tuple[str, str]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_json(path: Path, value: Any) -> str:
    data = _canonical_json_bytes(value) + b"\n"
    _atomic_write_bytes(path, data)
    return _sha256_bytes(data)


def load_registered_fixture_request(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text())
    if request.get("schema_version") != "stage7b.registered_fixture_request.v1":
        raise RegisteredFixtureError("unexpected Stage 7B request schema")
    if request.get("technical_only") is not True:
        raise RegisteredFixtureError("registered fixture must remain technical-only")
    if request.get("scientific_data") is not False:
        raise RegisteredFixtureError("registered fixture cannot be scientific data")
    if request.get("production_default") is not False:
        raise RegisteredFixtureError("registered fixture cannot define a production default")
    if request.get("ud_resolution") is not False:
        raise RegisteredFixtureError("registered fixture cannot resolve a UD")
    if request.get("stage8_execution") is not False:
        raise RegisteredFixtureError("Stage 8 must remain closed")
    if request["endpoints"]["print_endpoint_values"] is not False:
        raise RegisteredFixtureError("endpoint values must not be printed")
    if request["endpoints"]["interpret_endpoint_values"] is not False:
        raise RegisteredFixtureError("endpoint values must not be interpreted")
    return request


def identity_from_request(request: Mapping[str, Any]) -> RegisteredFixtureIdentity:
    t = request["registered_teacher"]
    return RegisteredFixtureIdentity(
        teacher_seed=int(t["teacher_seed"]),
        phase_label=str(t["phase_label"]),
        canonical_run_id=str(t["canonical_run_id"]),
        training_step=int(t["training_step"]),
        checkpoint_relative_path=str(
            t["checkpoint_path_relative_to_predecessor"]
        ),
        checkpoint_sha256=str(t["checkpoint_sha256"]),
        complete_domain_size=int(t["complete_domain_size"]),
        domain_order=str(t["domain_order"]),
    )


def canonical_modular_addition_domain(
    *,
    modulus: int = 113,
    final_token_id: int = 113,
) -> np.ndarray:
    """Return the canonical lexicographic complete modular-addition domain.

    Rows are ordered by ``a`` first and then ``b`` and contain the accepted
    three-token input ``[a, b, final_token_id]``.  This helper is an ordering
    bridge, not a second dataset split or sampling rule.
    """

    if modulus <= 0:
        raise RegisteredFixtureError("modulus must be positive")
    rows = np.empty((modulus * modulus, 3), dtype=np.int64)
    cursor = 0
    for a in range(modulus):
        for b in range(modulus):
            rows[cursor] = (a, b, final_token_id)
            cursor += 1
    return rows


def centred_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits)
    if values.ndim != 2:
        raise RegisteredFixtureError("teacher logits must have shape [inputs, classes]")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    return values - values.mean(axis=1, keepdims=True)


def _find_registry_record(
    registry_payload: Any,
    identity: RegisteredFixtureIdentity,
) -> Mapping[str, Any]:
    hits: list[Mapping[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if (
                value.get("canonical_run_id") == identity.canonical_run_id
                and int(value.get("teacher_seed", -1)) == identity.teacher_seed
                and value.get("phase_label") == identity.phase_label
                and int(value.get("training_step", -1)) == identity.training_step
                and value.get("checkpoint_path")
                == identity.checkpoint_relative_path
                and value.get("checkpoint_sha256")
                == identity.checkpoint_sha256
            ):
                hits.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for child in value:
                walk(child)

    walk(registry_payload)
    if len(hits) != 1:
        raise RegisteredFixtureError(
            "registered teacher identity did not bind exactly one Stage 3 record"
        )
    return hits[0]


def validate_registered_fixture_identity(
    *,
    repository_root: Path,
    predecessor_root: Path,
    request: Mapping[str, Any],
) -> tuple[RegisteredFixtureIdentity, Path, Path, Mapping[str, Any]]:
    """Physically validate the exact selected Stage 3 teacher identity.

    This performs file/provenance validation only.  It does not restore a model
    or execute an accelerator.
    """

    identity = identity_from_request(request)

    if identity.complete_domain_size != 113 * 113:
        raise RegisteredFixtureError("registered complete-domain cardinality mismatch")
    if identity.domain_order != "lexicographic_modular_addition_inputs":
        raise RegisteredFixtureError("registered domain-order identity mismatch")

    registry_path = (
        repository_root
        / "followup/manifests/stage3_teacher_registry_v1.json"
    )
    if not registry_path.is_file():
        raise RegisteredFixtureError("Stage 3 teacher registry is absent")

    registry_payload = json.loads(registry_path.read_text())
    record = _find_registry_record(registry_payload, identity)

    manifest_rel = record.get("training_manifest_path")
    manifest_sha = record.get("training_manifest_sha256")
    if not manifest_rel or not manifest_sha:
        raise RegisteredFixtureError(
            "registered Stage 3 record lacks training-manifest provenance"
        )

    manifest_path = predecessor_root / str(manifest_rel)
    if not manifest_path.is_file():
        matches = sorted(predecessor_root.rglob(Path(str(manifest_rel)).name))
        matches = [
            p for p in matches
            if p.is_file() and _sha256_file(p) == manifest_sha
        ]
        if len(matches) != 1:
            raise RegisteredFixtureError(
                "physical training manifest did not resolve uniquely"
            )
        manifest_path = matches[0]

    if _sha256_file(manifest_path) != manifest_sha:
        raise RegisteredFixtureError("physical training-manifest hash mismatch")

    checkpoint_path = predecessor_root / identity.checkpoint_relative_path
    if not checkpoint_path.is_file():
        raise RegisteredFixtureError("registered physical checkpoint is absent")
    if _sha256_file(checkpoint_path) != identity.checkpoint_sha256:
        raise RegisteredFixtureError("registered physical checkpoint hash mismatch")

    manifest_payload = json.loads(manifest_path.read_text())

    # Stage 3 is the authority for seed/phase/step/path/hash.  The predecessor
    # manifest must not contradict identity fields that it explicitly carries.
    flat_scalars: dict[str, Any] = {}

    def flatten(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, (str, int, float, bool)) or child is None:
                    flat_scalars[path.lower()] = child
                else:
                    flatten(child, path)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                flatten(child, f"{prefix}[{index}]")

    flatten(manifest_payload)

    seed_values = {
        int(v)
        for k, v in flat_scalars.items()
        if "seed" in k
        and isinstance(v, (int, float))
        and float(v).is_integer()
    }
    if seed_values and identity.teacher_seed not in seed_values:
        raise RegisteredFixtureError("training-manifest seed mismatch")

    # The predecessor training manifest describes the complete training run.
    # Stage 3 is authoritative for the exact selected checkpoint identity:
    # run id, teacher seed, phase, selected step, relative path, and file hash.
    #
    # The training manifest is still physically bound by its registry-recorded
    # path and SHA256, but generic checkpoint cadence, final-step, save-step,
    # training-step, or checkpoint-list metadata must not be interpreted as a
    # second selected-checkpoint authority.
    #
    # Only an unambiguously named scalar selected-checkpoint field is eligible
    # for this optional cross-check.
    unambiguous_selected_step_keys = {
        "selected_checkpoint_step",
        "selected_registered_checkpoint_step",
        "registered_checkpoint_step",
    }
    explicit_selected_steps = {
        int(v)
        for k, v in flat_scalars.items()
        if k.rsplit(".", 1)[-1] in unambiguous_selected_step_keys
        and isinstance(v, (int, float))
        and float(v).is_integer()
    }
    if (
        explicit_selected_steps
        and identity.training_step not in explicit_selected_steps
    ):
        raise RegisteredFixtureError(
            "training-manifest explicit selected-checkpoint-step mismatch"
        )

    return identity, checkpoint_path, manifest_path, record


def _validate_checkpoint_payload(
    *,
    payload: Mapping[str, Any],
    identity: RegisteredFixtureIdentity,
) -> None:
    if not isinstance(payload, Mapping):
        raise RegisteredFixtureError("checkpoint payload must be a mapping")

    if "training_step" in payload and int(payload["training_step"]) != identity.training_step:
        raise RegisteredFixtureError("checkpoint payload training-step mismatch")

    if "model_seed" in payload and int(payload["model_seed"]) != identity.teacher_seed:
        raise RegisteredFixtureError("checkpoint payload model-seed mismatch")

    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        raise RegisteredFixtureError("checkpoint payload lacks model_state")

    expected_state_sha = payload.get("model_state_sha256")
    if not isinstance(expected_state_sha, str) or len(expected_state_sha) != 64:
        raise RegisteredFixtureError("checkpoint payload lacks model-state identity")


def _ensure_runtime_root(
    *,
    repository_root: Path,
    output_root: Path,
    request: Mapping[str, Any],
) -> None:
    repository_root = repository_root.resolve()
    output_root = output_root.resolve()

    try:
        rel = output_root.relative_to(repository_root)
    except ValueError as exc:
        raise RegisteredFixtureError(
            "runtime output root must be beneath the active repository"
        ) from exc

    if not str(rel).startswith("followup/local/stage7b/"):
        raise RegisteredFixtureError(
            "runtime output root must remain under ignored followup/local/stage7b"
        )

    tracked = request["runtime_boundary"]
    if tracked["runtime_outputs_tracked"] is not False:
        raise RegisteredFixtureError("runtime products cannot be tracked")


def _subject_is_releasable(assessment: Mapping[str, Any]) -> bool:
    return bool(assessment.get("eligible")) and bool(assessment.get("sealed"))


def _endpoint_record_hash(record: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(record))


def _runtime_inventory(output_root: Path) -> tuple[int, int, str]:
    files = sorted(
        p for p in output_root.rglob("*")
        if p.is_file()
    )
    entries = []
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        entries.append(
            {
                "path": str(path.relative_to(output_root)),
                "bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    inventory_hash = _sha256_bytes(_canonical_json_bytes(entries))
    return len(files), total, inventory_hash


def run_registered_fixture(
    *,
    repository_root: Path,
    predecessor_root: Path,
    output_root: Path,
    request_path: Path,
    bindings: RegisteredFixtureBindings,
) -> RegisteredFixtureRun:
    """Run the bounded Stage 7B bridge through injected accepted interfaces."""

    repository_root = repository_root.resolve()
    predecessor_root = predecessor_root.resolve()
    output_root = output_root.resolve()
    request_path = request_path.resolve()

    request = load_registered_fixture_request(request_path)
    _ensure_runtime_root(
        repository_root=repository_root,
        output_root=output_root,
        request=request,
    )

    identity, checkpoint_path, manifest_path, registry_record = (
        validate_registered_fixture_identity(
            repository_root=repository_root,
            predecessor_root=predecessor_root,
            request=request,
        )
    )

    # Exact physical provenance is complete before checkpoint payload loading or
    # model execution.
    checkpoint_payload = bindings.load_checkpoint_payload(checkpoint_path)
    _validate_checkpoint_payload(payload=checkpoint_payload, identity=identity)

    domain_inputs = canonical_modular_addition_domain()
    if domain_inputs.shape != (identity.complete_domain_size, 3):
        raise RegisteredFixtureError("canonical complete-domain shape mismatch")

    execution = request["execution_engineering"]
    model = bindings.restore_model(
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
        device=str(execution["device"]),
    )

    teacher_logits = np.asarray(
        bindings.evaluate_teacher(
            model=model,
            domain_inputs=domain_inputs,
            batch_size=int(execution["teacher_forward_batch_size"]),
            device=str(execution["device"]),
        )
    )
    if teacher_logits.shape[0] != identity.complete_domain_size:
        raise RegisteredFixtureError("teacher output does not cover complete domain")

    teacher_centred = centred_logits(teacher_logits)
    teacher_hard = np.argmax(teacher_centred, axis=1).astype(np.int64)

    output_root.mkdir(parents=True, exist_ok=False)

    provenance_record = {
        "status": "PASS",
        "teacher_seed": identity.teacher_seed,
        "phase_label": identity.phase_label,
        "canonical_run_id": identity.canonical_run_id,
        "training_step": identity.training_step,
        "checkpoint_relative_path": identity.checkpoint_relative_path,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "manifest_relative_path": str(manifest_path.relative_to(predecessor_root)),
        "manifest_sha256": _sha256_file(manifest_path),
        "registry_training_manifest_sha256": registry_record["training_manifest_sha256"],
    }
    _atomic_write_json(output_root / "provenance.json", provenance_record)

    caches = bindings.build_target_caches(
        domain_inputs=domain_inputs,
        hard_targets=teacher_hard,
        centred_teacher_logits=teacher_centred,
        output_root=output_root,
        request=request,
    )
    if set(caches) != {"hard", "soft"}:
        raise RegisteredFixtureError("target-cache binding must return hard and soft caches")

    attempts = request["attempt_roster"]
    if attempts["hard_target_attempts"] != 1 or attempts["soft_target_attempts"] != 1:
        raise RegisteredFixtureError("Stage 7B requires exactly one hard and one soft attempt")

    workload = request["shared_trainer_workload"]
    assessments: dict[str, Mapping[str, Any]] = {}

    for target_kind in ("hard", "soft"):
        attempt_result = bindings.run_student_attempt(
            target_kind=target_kind,
            target_cache=caches[target_kind],
            attempt_index=0,
            work_units=int(workload["native_positive_work_units_per_attempt"]),
            safety_ceiling=int(
                workload["native_work_unit_safety_ceiling_per_attempt"]
            ),
            request=request,
            output_root=output_root,
        )
        assessment = bindings.assess_student_attempt(
            target_kind=target_kind,
            attempt_result=attempt_result,
            teacher_hard_targets=teacher_hard,
            teacher_centred_logits=teacher_centred,
            domain_inputs=domain_inputs,
            request=request,
            output_root=output_root,
        )
        assessments[target_kind] = dict(assessment)
        _atomic_write_json(
            output_root / "attempts" / f"{target_kind}.json",
            dict(assessment),
        )

    adapter_names = tuple(bindings.discovery_adapter_names)
    if len(adapter_names) != 2 or len(set(adapter_names)) != 2:
        raise RegisteredFixtureError("exactly two distinct accepted adapters are required")

    discovery_records: list[Mapping[str, Any]] = []

    # Teacher-direct discovery is unconditional.
    for adapter_name in adapter_names:
        result = bindings.run_discovery(
            adapter_name=adapter_name,
            subject_kind="teacher",
            subject=model,
            request=request,
            output_root=output_root,
        )
        discovery_records.append(
            {
                "adapter_name": adapter_name,
                "subject_kind": "teacher",
                "subject_id": "registered_teacher",
                "result": result,
            }
        )

    # Students reach discovery only if the accepted assessment says they are
    # both eligible and sealed.  Failed/ineligible attempts remain recorded.
    for target_kind in ("hard", "soft"):
        assessment = assessments[target_kind]
        if not _subject_is_releasable(assessment):
            continue
        subject = assessment.get("sealed_subject")
        if subject is None:
            raise RegisteredFixtureError(
                "eligible sealed assessment omitted sealed_subject"
            )
        for adapter_name in adapter_names:
            result = bindings.run_discovery(
                adapter_name=adapter_name,
                subject_kind=f"{target_kind}_student",
                subject=subject,
                request=request,
                output_root=output_root,
            )
            discovery_records.append(
                {
                    "adapter_name": adapter_name,
                    "subject_kind": f"{target_kind}_student",
                    "subject_id": f"{target_kind}_attempt_0",
                    "result": result,
                }
            )

    endpoint_records: list[Mapping[str, Any]] = []
    for discovery in discovery_records:
        endpoint = dict(
            bindings.run_exact_endpoints(
                adapter_name=str(discovery["adapter_name"]),
                subject_kind=str(discovery["subject_kind"]),
                discovery_result=discovery["result"],
                teacher_centred_logits=teacher_centred,
                domain_inputs=domain_inputs,
                request=request,
                output_root=output_root,
            )
        )
        # Endpoint payload is written locally but is never emitted to stdout.
        endpoint_records.append(endpoint)

    source_code_sha = _sha256_file(Path(__file__))

    excluded = dict(
        bindings.build_excluded_outputs(
            identity=identity,
            request=request,
            source_code_sha=source_code_sha,
            attempt_assessments=assessments,
            discovery_records=discovery_records,
            endpoint_records=endpoint_records,
            output_root=output_root,
        )
    )

    exclusion_records = excluded.get("exclusion_records")
    if not isinstance(exclusion_records, Sequence):
        raise RegisteredFixtureError("excluded-output binding omitted exclusion records")

    for record in exclusion_records:
        if not isinstance(record, Mapping):
            raise RegisteredFixtureError("exclusion record must be a mapping")
        if record.get("lifecycle_state") != "excluded":
            raise RegisteredFixtureError("endpoint-like output is not excluded")
        if record.get("primary_input_eligible") is not False:
            raise RegisteredFixtureError("excluded output became primary-eligible")
        if record.get("regeneration_required_after_definitive_freeze") is not True:
            raise RegisteredFixtureError("excluded output lacks regeneration requirement")

    primary_eligible_count = sum(
        1 for record in exclusion_records
        if record.get("primary_input_eligible") is True
    )
    if primary_eligible_count:
        raise RegisteredFixtureError("primary-eligible excluded output detected")

    report = excluded.get("report")
    if not isinstance(report, Mapping):
        raise RegisteredFixtureError("excluded-output binding omitted report")

    endpoint_hashes = tuple(
        sorted(_endpoint_record_hash(record) for record in endpoint_records)
    )

    compact_summary = {
        "schema_version": "stage7b.registered_fixture_summary.v1",
        "provenance_status": "PASS",
        "hard_attempt_status": str(
            assessments["hard"].get("status", "unknown")
        ),
        "soft_attempt_status": str(
            assessments["soft"].get("status", "unknown")
        ),
        "teacher_discovery_release_count": sum(
            r["subject_kind"] == "teacher" for r in discovery_records
        ),
        "student_discovery_release_count": sum(
            r["subject_kind"] != "teacher" for r in discovery_records
        ),
        "discovery_result_count": len(discovery_records),
        "endpoint_record_hashes": endpoint_hashes,
        "exclusion_record_count": len(exclusion_records),
        "primary_eligible_count": primary_eligible_count,
        "scientific_data": False,
        "stage8_status": "NOT_STARTED",
    }

    report_sha = _sha256_bytes(_canonical_json_bytes(report))
    compact_summary["report_sha256"] = report_sha
    _atomic_write_json(output_root / "compact_summary.json", compact_summary)

    file_count, total_bytes, inventory_sha = _runtime_inventory(output_root)

    ceiling = int(
        request["resource_envelope"][
            "combined_source_and_reproduction_local_ceiling_bytes"
        ]
    )
    if total_bytes > ceiling:
        raise RegisteredFixtureError("runtime root exceeded declared local ceiling")

    return RegisteredFixtureRun(
        provenance_status="PASS",
        hard_attempt_status=str(assessments["hard"].get("status", "unknown")),
        soft_attempt_status=str(assessments["soft"].get("status", "unknown")),
        teacher_discovery_release_count=sum(
            r["subject_kind"] == "teacher" for r in discovery_records
        ),
        student_discovery_release_count=sum(
            r["subject_kind"] != "teacher" for r in discovery_records
        ),
        discovery_result_count=len(discovery_records),
        endpoint_record_hashes=endpoint_hashes,
        exclusion_record_count=len(exclusion_records),
        primary_eligible_count=primary_eligible_count,
        runtime_file_count=file_count,
        runtime_total_bytes=total_bytes,
        report_sha256=report_sha,
        inventory_sha256=inventory_sha,
    )


def production_bindings(
    *,
    repository_root: Path,
    predecessor_root: Path,
    request: Mapping[str, Any],
) -> RegisteredFixtureBindings:
    """Build physical bindings from accepted repository interfaces.

    Stage 7B itself does not implement a second trainer/discovery/endpoint
    algorithm.  This factory imports the accepted components and requires the
    concrete adapter layer to be present.  The fail-closed check is deliberate:
    Part C may not silently substitute Stage 7A synthetic proposal streams.

    The concrete accepted-interface adapter is installed below by
    ``_build_accepted_bindings``.  Keeping construction in this module makes
    the tracked bridge surface singular and portable.
    """

    return _build_accepted_bindings(
        repository_root=repository_root,
        predecessor_root=predecessor_root,
        request=request,
    )


def _build_accepted_bindings(
    *,
    repository_root: Path,
    predecessor_root: Path,
    request: Mapping[str, Any],
) -> RegisteredFixtureBindings:
    """Bind model execution and accepted Stage 3–7 operations.

    The physical checkpoint/model portion is concrete here.  Downstream
    trainer/discovery/endpoint operations are imported from the accepted
    implementation and may be replaced by injected stand-ins in portable
    tests.  Any interface incompatibility fails closed rather than falling
    back to synthetic Stage 7A behavior.
    """

    import inspect

    import torch

    def load_checkpoint_payload(path: Path) -> Mapping[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise RegisteredFixtureError("physical checkpoint payload is not a mapping")
        return payload

    def restore_model(
        *,
        checkpoint_path: Path,
        checkpoint_payload: Mapping[str, Any],
        device: str,
    ) -> Any:
        from circuit_families.models import build_transformer
        from circuit_families.training.checkpoints import (
            canonical_state_hash,
        )

        del checkpoint_path

        model_config = checkpoint_payload.get("model_config")
        if not isinstance(model_config, Mapping):
            raise RegisteredFixtureError(
                "checkpoint payload lacks model configuration"
            )

        model_seed = checkpoint_payload.get("model_seed")
        if (
            isinstance(model_seed, bool)
            or not isinstance(model_seed, int)
            or model_seed < 0
        ):
            raise RegisteredFixtureError(
                "checkpoint payload lacks valid model seed"
            )

        model = build_transformer(
            model_config,
            seed=model_seed,
            device=device,
        )
        model.load_state_dict(
            checkpoint_payload["model_state"],
            strict=True,
        )
        model.eval()

        restored_hash = canonical_state_hash(model.state_dict())
        expected_hash = checkpoint_payload["model_state_sha256"]
        if restored_hash != expected_hash:
            raise RegisteredFixtureError(
                "restored model-state hash mismatch"
            )

        return model

    def evaluate_teacher(
        *,
        model: Any,
        domain_inputs: np.ndarray,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(domain_inputs), batch_size):
                batch_np = domain_inputs[start : start + batch_size]
                tokens = torch.as_tensor(
                    batch_np,
                    dtype=torch.long,
                    device=device,
                )
                logits = model(tokens)
                if logits.ndim != 3:
                    raise RegisteredFixtureError(
                        "accepted Transformer must return [batch, position, class] logits"
                    )
                final = logits[:, -1, :]
                outputs.append(final.detach().to("cpu").float().numpy())
                del logits, final, tokens
        if not outputs:
            raise RegisteredFixtureError("teacher evaluation produced no batches")
        return np.concatenate(outputs, axis=0)

    # Accepted downstream interface imports.  These imports are intentional:
    # a missing interface is a software defect, not permission to substitute a
    # new algorithm.
    from circuit_families.stage5bc.target_cache import build_target_cache
    from circuit_families.stage6a.endpoint import reduce_endpoint1
    from circuit_families.stage6a.ledger import ExactLedgerBuilder
    from circuit_families.stage6b.records import (
        assess_hard_attempt,
        generate_hard_sealing_evidence,
    )
    from circuit_families.stage6d.diversity import DiversityForcedAdapter
    from circuit_families.stage6d.greedy import GreedyDeletionAdapter
    from circuit_families.stage6e.packing import build_endpoint2_result
    from circuit_families.stage7.inventory_reporting import (
        assert_rejected_as_primary_scientific_input,
        build_endpoint_exclusion_records,
        build_part_f_report,
        build_teacher_seed_inventory,
    )

    # The Stage 5–7 interfaces are strongly typed and evolved under earlier
    # stages.  Bind them by their accepted argument names instead of hardcoding
    # a second algorithm here.
    def invoke_accepted(func: Callable[..., Any], values: Mapping[str, Any]) -> Any:
        sig = inspect.signature(func)
        kwargs = {}
        missing_required = []
        for name, parameter in sig.parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name in values:
                kwargs[name] = values[name]
            elif parameter.default is inspect._empty:
                missing_required.append(name)
        if missing_required:
            raise RegisteredFixtureError(
                f"accepted interface {func.__module__}.{func.__name__} requires "
                f"unbound arguments {missing_required}"
            )
        return func(**kwargs)

    def build_target_caches(
        *,
        domain_inputs: np.ndarray,
        hard_targets: np.ndarray,
        centred_teacher_logits: np.ndarray,
        output_root: Path,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        import hashlib

        import torch

        from circuit_families.stage4_condition_identity import (
            ConditionIdentity,
            Stage3AvailabilityIndex,
            build_condition_id,
        )
        from circuit_families.stage5bc.target_cache import (
            TargetCacheBatch,
            load_target_cache,
        )

        registry_path = (
            repository_root
            / "followup/manifests/stage3_teacher_registry_v1.json"
        )
        if not registry_path.is_file():
            raise RegisteredFixtureError(
                "Stage 3 teacher registry is absent during cache construction"
            )

        registry_payload = json.loads(registry_path.read_text())
        stage3 = Stage3AvailabilityIndex.from_registry(registry_payload)

        teacher_cfg = request["registered_teacher"]
        teacher_seed = int(teacher_cfg["teacher_seed"])
        phase = str(teacher_cfg["phase_label"])

        if stage3.availability(teacher_seed, phase) != "selected":
            raise RegisteredFixtureError(
                "registered teacher cell is not selected in Stage 3"
            )

        if not isinstance(domain_inputs, np.ndarray) or domain_inputs.ndim != 2:
            raise RegisteredFixtureError(
                "registered domain inputs must be rank-two numpy data"
            )
        if not isinstance(hard_targets, np.ndarray) or hard_targets.ndim != 1:
            raise RegisteredFixtureError(
                "hard teacher targets must be rank-one numpy data"
            )
        if (
            not isinstance(centred_teacher_logits, np.ndarray)
            or centred_teacher_logits.ndim != 2
        ):
            raise RegisteredFixtureError(
                "centred teacher logits must be rank-two numpy data"
            )

        example_count = int(domain_inputs.shape[0])
        if hard_targets.shape[0] != example_count:
            raise RegisteredFixtureError(
                "hard-target example count does not match domain"
            )
        if centred_teacher_logits.shape[0] != example_count:
            raise RegisteredFixtureError(
                "soft-target example count does not match domain"
            )

        class_count = int(centred_teacher_logits.shape[1])
        if class_count != 113:
            raise RegisteredFixtureError(
                "registered teacher cache must contain 113 classes"
            )

        derived_hard = np.argmax(centred_teacher_logits, axis=1)
        if not np.array_equal(derived_hard, hard_targets):
            raise RegisteredFixtureError(
                "hard targets disagree with centred-logit argmax"
            )

        input_ids = tuple(
            f"{int(row[0])},{int(row[1])}"
            for row in domain_inputs
        )
        if len(set(input_ids)) != example_count:
            raise RegisteredFixtureError(
                "registered domain input ids are not unique"
            )

        raw_logits = torch.as_tensor(
            centred_teacher_logits,
            dtype=torch.float32,
        )

        cache_root = Path(output_root) / "target_caches"
        cache_root.mkdir(parents=True, exist_ok=True)

        request_path = (
            repository_root
            / "followup/configs/stage7b/registered_fixture_request_v1.json"
        )
        if not request_path.is_file():
            raise RegisteredFixtureError(
                "frozen Stage 7B request file is absent"
            )

        request_bytes = request_path.read_bytes()
        request_payload = json.loads(request_bytes)
        if request_payload != request:
            raise RegisteredFixtureError(
                "runtime request mapping differs from frozen request file"
            )


        from circuit_families.predecessor_link import (
            validate_predecessor_link,
            verify_predecessor_link_physical,
        )
        from circuit_families.stage5d.normalization import normalized_sha256

        _, _, verified_manifest_path, selected_record = (
            validate_registered_fixture_identity(
                repository_root=repository_root,
                predecessor_root=predecessor_root,
                request=request,
            )
        )
        verified_manifest = json.loads(verified_manifest_path.read_text())

        dataset_hashes = selected_record["dataset_identity"]["canonical_hashes"]
        manifest_dataset_hashes = verified_manifest["dataset"]["canonical_hashes"]

        if manifest_dataset_hashes != dataset_hashes:
            raise RegisteredFixtureError(
                "Stage 3 and verified training-manifest dataset hashes disagree"
            )

        manifest_configs = verified_manifest["configs"]

        if (
            manifest_configs["model"]["sha256"]
            != selected_record["model_identity"]["model_config"]["mapping_sha256"]
        ):
            raise RegisteredFixtureError(
                "Stage 3 and verified training-manifest model config hashes disagree"
            )

        if (
            manifest_configs["training"]["sha256"]
            != selected_record["training_config_identity"]["mapping_sha256"]
        ):
            raise RegisteredFixtureError(
                "Stage 3 and verified training-manifest training config hashes disagree"
            )

        predecessor_link_path = (
            repository_root / "followup/manifests/predecessor_link_v1.json"
        )
        predecessor_link = json.loads(predecessor_link_path.read_text())
        validate_predecessor_link(predecessor_link)
        verify_predecessor_link_physical(
            predecessor_link,
            predecessor_root=predecessor_root,
        )

        component_basis = predecessor_link["component_basis"]
        if component_basis["dedicated_component_basis_sha256"] is not None:
            raise RegisteredFixtureError(
                "unexpected dedicated predecessor component-basis hash"
            )

        component_basis_sha256 = normalized_sha256(component_basis)

        provenance = {
            "dataset_sha256": str(dataset_hashes["dataset_sha256"]),
            "split_sha256": str(dataset_hashes["split_sha256"]),
            "task_config_sha256": str(manifest_configs["task"]["sha256"]),
            "model_config_sha256": str(manifest_configs["model"]["sha256"]),
            "training_config_sha256": str(
                manifest_configs["training"]["sha256"]
            ),
            "component_basis_sha256": component_basis_sha256,
        }

        results: dict[str, Any] = {}

        for condition, expected_kind in (
            ("hard_target", "teacher_argmax"),
            ("soft_target", "teacher_logits"),
        ):
            condition_id = build_condition_id(
                ConditionIdentity(
                    teacher_seed=teacher_seed,
                    phase=phase,
                    distillation_condition=condition,
                ),
                stage3,
            )

            teacher_reference = {
                "record_type": "teacher_reference",
                "schema_version": "teacher_reference/v1",
                "condition_id": condition_id,
                "record_sha256": hashlib.sha256(
                    (
                        "stage7b-registered-teacher:"
                        f"{teacher_seed}:{phase}:{condition}:"
                        f"{teacher_cfg['training_step']}:"
                        f"{teacher_cfg['checkpoint_sha256']}"
                    ).encode()
                ).hexdigest(),
            }

            prefix = f"{condition}/cache"

            built = build_target_cache(
                output_root=cache_root,
                manifest_relative_path=f"{prefix}/manifest.json",
                payload_relative_path=f"{prefix}/payload.bin",
                completion_relative_path=f"{prefix}/completion.json",
                manifest_id=f"technical-stage7b-registered-{condition}-cache/v1",
                ordering_ref=str(teacher_cfg["domain_order"]),
                expected_example_count=example_count,
                expected_class_count=class_count,
                teacher_reference=teacher_reference,
                provenance_hashes=provenance,
                batches=(
                    TargetCacheBatch(
                        input_ids=input_ids,
                        raw_logits=raw_logits,
                    ),
                ),
                technical_fixture=True,
                stage4_record_serializable=False,
                expected_input_ids=input_ids,
            )

            loaded = load_target_cache(
                output_root=cache_root,
                manifest_relative_path=f"{prefix}/manifest.json",
                expected_input_ids=input_ids,
                expected_teacher_reference=teacher_reference,
                expected_provenance_hashes=provenance,
                expected_stage4_cache_kind=expected_kind,
            )

            results[condition] = {
                "built": built,
                "loaded": loaded,
                "condition_id": condition_id,
                "expected_cache_kind": expected_kind,
                "domain_inputs": np.array(domain_inputs, copy=True),
            }

        return {
            "hard": results["hard_target"],
            "soft": results["soft_target"],
        }


    def run_student_attempt(
        *,
        target_kind: str,
        target_cache: Any,
        attempt_index: int,
        work_units: int,
        safety_ceiling: int,
        request: Mapping[str, Any],
        output_root: Path,
    ) -> Any:
        import copy
        import hashlib

        import torch

        from circuit_families.models import build_transformer
        from circuit_families.stage4_condition_identity import (
            Stage3AvailabilityIndex,
        )
        from circuit_families.stage5bc.attempt_records import (
            emit_technical_attempt_record,
            outcome_from_training_result,
        )
        from circuit_families.stage5bc.student_identity import (
            build_student_attempt_identity,
        )
        from circuit_families.stage5bc.student_trainer import (
            HardTargetAdapter,
            OptimizerScheduleBundle,
            PreparedTargets,
            TechnicalLoopSnapshot,
            TrainerLifecycle,
            TrainerSettingsBundle,
        )
        from circuit_families.stage5bc.technical_checkpoint import (
            save_technical_resume_checkpoint,
        )
        from circuit_families.stage6b.hard_target import (
            HardLabelLossAdapter,
        )
        from circuit_families.stage6c import (
            SOFT_LOSS_KIND,
            GaugeInvariantSoftLossAdapter,
            TechnicalSoftTargetAdapter,
        )
        from circuit_families.stage7.distillation import (
            _canonical_training_log_sha256,
            _soft_policy,
        )
        from circuit_families.training.trainer import build_optimizer

        if target_kind not in {"hard", "soft"}:
            raise RegisteredFixtureError(
                f"unknown target kind: {target_kind}"
            )
        if attempt_index != 0:
            raise RegisteredFixtureError(
                "Stage 7B permits only attempt_index=0"
            )
        if work_units != 1 or safety_ceiling != 1:
            raise RegisteredFixtureError(
                "Stage 7B frozen workload requires exactly one "
                "positive work unit and a safety ceiling of one"
            )

        if not isinstance(target_cache, Mapping):
            raise RegisteredFixtureError(
                "target cache binding must be a mapping"
            )

        loaded_cache = target_cache.get("loaded")
        built_cache = target_cache.get("built")
        domain_inputs = target_cache.get("domain_inputs")

        if loaded_cache is None or built_cache is None:
            raise RegisteredFixtureError(
                "target cache binding lacks accepted built/loaded cache"
            )

        if (
            not isinstance(domain_inputs, np.ndarray)
            or domain_inputs.ndim != 2
            or domain_inputs.shape != (12_769, 3)
        ):
            raise RegisteredFixtureError(
                "student attempt requires the complete registered "
                "[12769, 3] token domain"
            )

        registry_path = (
            repository_root
            / "followup/manifests/stage3_teacher_registry_v1.json"
        )
        registry = json.loads(registry_path.read_text())
        stage3 = Stage3AvailabilityIndex.from_registry(registry)

        teacher_cfg = request["registered_teacher"]
        teacher_seed = int(teacher_cfg["teacher_seed"])
        phase = str(teacher_cfg["phase_label"])
        condition = (
            "hard_target"
            if target_kind == "hard"
            else "soft_target"
        )

        identity = build_student_attempt_identity(
            stage3=stage3,
            teacher_seed=teacher_seed,
            phase=phase,
            distillation_condition=condition,
            student_initialization=attempt_index,
            attempt_index=attempt_index,
            retry_index=0,
        )

        (
            _,
            registered_checkpoint_path,
            _,
            _,
        ) = validate_registered_fixture_identity(
            repository_root=repository_root,
            predecessor_root=predecessor_root,
            request=request,
        )

        checkpoint_payload = load_checkpoint_payload(
            registered_checkpoint_path
        )

        model_config = checkpoint_payload.get("model_config")
        training_config = checkpoint_payload.get("training_config")

        if not isinstance(model_config, Mapping):
            raise RegisteredFixtureError(
                "registered checkpoint lacks model_config"
            )
        if not isinstance(training_config, Mapping):
            raise RegisteredFixtureError(
                "registered checkpoint lacks training_config"
            )

        device_name = str(
            request["execution_engineering"]["device"]
        )
        if device_name != "mps":
            raise RegisteredFixtureError(
                "Stage 7B frozen execution device must be mps"
            )
        if not torch.backends.mps.is_available():
            raise RegisteredFixtureError(
                "frozen Stage 7B MPS device is unavailable"
            )

        device = torch.device(device_name)

        manifest = loaded_cache.manifest.to_mapping()
        input_order = manifest.get("input_order")
        if not isinstance(input_order, Mapping):
            raise RegisteredFixtureError(
                "loaded target cache lacks input-order metadata"
            )

        ordering_ref = str(input_order["ordering_ref"])
        ordered_input_ids_sha256 = str(
            input_order["ordered_input_ids_sha256"]
        )

        stage7a_authority = request["accepted_authority"][
            "stage7a_technical_request"
        ]
        stage7a_path = repository_root / str(
            stage7a_authority["path"]
        )

        if not stage7a_path.is_file():
            raise RegisteredFixtureError(
                "accepted Stage 7A technical request is absent"
            )

        stage7a_bytes = stage7a_path.read_bytes()
        if (
            hashlib.sha256(stage7a_bytes).hexdigest()
            != str(stage7a_authority["sha256"])
        ):
            raise RegisteredFixtureError(
                "accepted Stage 7A technical request hash mismatch"
            )

        stage7a_payload = json.loads(stage7a_bytes)

        def collect_soft_tolerances(
            value: Any,
        ) -> list[float]:
            found: list[float] = []

            if isinstance(value, Mapping):
                for key, item in value.items():
                    if key == "soft_tolerance":
                        if (
                            isinstance(item, bool)
                            or not isinstance(item, (int, float))
                        ):
                            raise RegisteredFixtureError(
                                "accepted soft_tolerance is not numeric"
                            )
                        found.append(float(item))

                    found.extend(
                        collect_soft_tolerances(item)
                    )

            elif isinstance(value, list):
                for item in value:
                    found.extend(
                        collect_soft_tolerances(item)
                    )

            return found

        unique_tolerances = sorted(
            set(
                collect_soft_tolerances(
                    stage7a_payload
                )
            )
        )

        if len(unique_tolerances) != 1:
            raise RegisteredFixtureError(
                "accepted Stage 7A request must expose exactly one "
                "soft_tolerance value"
            )

        soft_policy = _soft_policy(
            stage3=stage3,
            teacher_seed=teacher_seed,
            phase=phase,
            ordered_input_ids_sha256=(
                ordered_input_ids_sha256
            ),
            tolerance=unique_tolerances[0],
        )

        class DeviceTargetAdapter:
            def __init__(self, accepted: Any) -> None:
                self.accepted = accepted
                self.cache_kind = str(
                    accepted.cache_kind
                )

            def __call__(
                self,
                cache: Any,
            ) -> PreparedTargets:
                prepared = self.accepted(cache)

                if not isinstance(
                    prepared,
                    PreparedTargets,
                ):
                    raise RegisteredFixtureError(
                        "accepted target adapter returned "
                        "the wrong type"
                    )

                return PreparedTargets(
                    cache_kind=prepared.cache_kind,
                    values=prepared.values.to(
                        device=device
                    ),
                )

        class FinalPositionLossAdapter:
            def __init__(self, accepted: Any) -> None:
                self.accepted = accepted

            def __call__(
                self,
                *,
                outputs: torch.Tensor,
                targets: PreparedTargets,
                settings: Mapping[str, Any],
            ) -> torch.Tensor:
                if (
                    outputs.ndim != 3
                    or outputs.shape[-1] != 113
                ):
                    raise RegisteredFixtureError(
                        "registered student model must return "
                        "[batch, position, 113] logits"
                    )

                return self.accepted(
                    outputs=outputs[:, -1, :],
                    targets=targets,
                    settings=settings,
                )

        def model_constructor(
            *,
            seed: int,
            device: torch.device,
            settings: Mapping[str, Any],
        ) -> torch.nn.Module:
            if set(settings) != {"model_config"}:
                raise RegisteredFixtureError(
                    "student model settings must contain "
                    "only model_config"
                )

            injected = settings["model_config"]

            if injected != model_config:
                raise RegisteredFixtureError(
                    "student model config differs from "
                    "registered model config"
                )

            return build_transformer(
                injected,
                seed=seed,
                device=device,
            )

        def optimizer_factory(
            *,
            model: torch.nn.Module,
            settings: Mapping[str, Any],
        ) -> OptimizerScheduleBundle:
            if set(settings) != {"training_config"}:
                raise RegisteredFixtureError(
                    "optimizer settings must contain "
                    "only training_config"
                )

            injected = settings[
                "training_config"
            ]

            if injected != training_config:
                raise RegisteredFixtureError(
                    "student optimizer config differs "
                    "from registered training config"
                )

            schedule = injected.get("schedule")

            if (
                not isinstance(schedule, Mapping)
                or schedule.get("name")
                != "constant"
                or schedule.get("warmup_steps") != 0
            ):
                raise RegisteredFixtureError(
                    "registered schedule is no longer "
                    "constant zero-warmup"
                )

            return OptimizerScheduleBundle(
                optimizer=build_optimizer(
                    model,
                    injected,
                ),
                scheduler=None,
            )

        def stop_rule(
            *,
            progress: Any,
            settings: Mapping[str, Any],
        ) -> bool:
            if set(settings) != {
                "stop_after_updates"
            }:
                raise RegisteredFixtureError(
                    "technical stop settings shape changed"
                )

            if (
                settings["stop_after_updates"]
                != work_units
            ):
                raise RegisteredFixtureError(
                    "technical stop rule differs from "
                    "the frozen workload"
                )

            return bool(
                progress.updates_completed
                >= work_units
            )

        if target_kind == "hard":
            target_adapter = DeviceTargetAdapter(
                HardTargetAdapter()
            )
            loss_adapter = FinalPositionLossAdapter(
                HardLabelLossAdapter()
            )
            loss_settings: Mapping[str, Any] = {
                "loss_kind": "cross_entropy",
                "reduction": "mean",
            }

        else:
            target_adapter = DeviceTargetAdapter(
                TechnicalSoftTargetAdapter(
                    policy=soft_policy,
                    stage3=stage3,
                )
            )
            loss_adapter = FinalPositionLossAdapter(
                GaugeInvariantSoftLossAdapter(
                    policy=soft_policy
                )
            )
            loss_settings = {
                "loss_kind": SOFT_LOSS_KIND,
                "policy": soft_policy,
                "reduction": "mean",
            }

        events: list[Any] = []

        lifecycle = TrainerLifecycle(
            model_constructor=model_constructor,
            target_adapter=target_adapter,
            loss_adapter=loss_adapter,
            optimizer_schedule_factory=(
                optimizer_factory
            ),
            stop_rule=stop_rule,
            recorder=events.append,
        )

        settings = TrainerSettingsBundle(
            model={
                "model_config": copy.deepcopy(
                    dict(model_config)
                )
            },
            loss=loss_settings,
            optimizer_schedule={
                "training_config": copy.deepcopy(
                    dict(training_config)
                )
            },
            stop={
                "stop_after_updates": work_units
            },
        )

        training_inputs = torch.as_tensor(
            domain_inputs,
            dtype=torch.long,
            device=device,
        )

        prepared = lifecycle.prepare(
            cache=loaded_cache,
            model_seed=identity.training_seed.seed_value,
            device=device,
            settings=settings,
        )

        configuration_refs = {
            "architecture_profile": (
                "technical-stage7b-registered-transformer/v1"
            ),
            "trainer_profile": (
                "technical-stage7b-registered-shared-trainer/v1"
            ),
            "adapter_profile": (
                "technical-stage7b-registered-"
                f"{condition}-adapter/v1"
            ),
        }

        result = lifecycle.run_technical(
            prepared=prepared,
            training_inputs=training_inputs,
            configuration_refs=configuration_refs,
            technical_safety_step_limit=(
                safety_ceiling
            ),
        )

        outcome_kind, failure_detail = (
            outcome_from_training_result(
                result
            )
        )

        configuration_hashes = {
            key: hashlib.sha256(
                value.encode()
            ).hexdigest()
            for key, value
            in configuration_refs.items()
        }

        manifest_path = Path(
            built_cache.manifest_path
        )
        cache_manifest_sha256 = (
            hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
        )

        checkpoint_evidence = None
        checkpoint_artifact = None

        if outcome_kind == "succeeded":
            checkpoint_path = (
                Path(output_root)
                / "student_attempts"
                / condition
                / "checkpoint.pt"
            )

            checkpoint_evidence = (
                save_technical_resume_checkpoint(
                    checkpoint_path,
                    prepared=prepared,
                    snapshot=TechnicalLoopSnapshot(
                        updates_completed=(
                            result.updates_completed
                        ),
                        trajectory=(
                            result.trajectory
                        ),
                        outer_training_mode=(
                            prepared.model.training
                        ),
                    ),
                    attempt_identity=identity,
                    stage3=stage3,
                    configuration_hashes=(
                        configuration_hashes
                    ),
                    target_cache_manifest_sha256=(
                        cache_manifest_sha256
                    ),
                )
            )

            checkpoint_artifact = {
                "path": (
                    "technical/stage7b/"
                    f"{condition}/checkpoint.pt"
                ),
                "sha256": (
                    checkpoint_evidence.file_sha256
                ),
                "storage_class": (
                    "external_checkpoint"
                ),
            }

        attempt_record = (
            emit_technical_attempt_record(
                stage3=stage3,
                attempt_identity=identity,
                target_cache_reference={
                    "record_type": (
                        "teacher_output_cache"
                    ),
                    "schema_version": (
                        "teacher_output_cache/v1"
                    ),
                    "condition_id": str(
                        target_cache[
                            "condition_id"
                        ]
                    ),
                    "record_sha256": (
                        cache_manifest_sha256
                    ),
                },
                outcome_kind=outcome_kind,
                student_architecture_ref=(
                    "technical-stage7b-registered-transformer/v1"
                ),
                replication_policy_ref=(
                    "technical-stage7b-registered-attempt-policy/v1"
                ),
                training_config_ref=(
                    "technical-stage7b-registered-shared-trainer/v1"
                ),
                training_log={
                    "path": (
                        "technical/stage7b/"
                        f"{condition}/training.json"
                    ),
                    "sha256": (
                        _canonical_training_log_sha256(
                            result
                        )
                    ),
                    "storage_class": (
                        "external_log"
                    ),
                },
                model_checkpoint=(
                    checkpoint_artifact
                ),
                failure_detail=failure_detail,
            )
        )

        sealed_attempt = copy.deepcopy(
            attempt_record
        )
        sealed_attempt["record_status"] = (
            "sealed"
        )

        outputs = None

        if outcome_kind == "succeeded":
            with torch.no_grad():
                native_outputs = prepared.model(
                    training_inputs
                )

                if (
                    native_outputs.ndim != 3
                    or native_outputs.shape
                    != (12_769, 3, 113)
                ):
                    raise RegisteredFixtureError(
                        "registered student evaluation "
                        "returned unexpected logits shape"
                    )

                outputs = (
                    native_outputs[:, -1, :]
                    .detach()
                    .to("cpu")
                    .float()
                    .clone()
                )

        return {
            "identity": identity,
            "attempt_record": sealed_attempt,
            "result": result,
            "outputs": outputs,
            "model": prepared.model,
            "checkpoint": checkpoint_evidence,
            "outcome_kind": outcome_kind,
            "stage3": stage3,
            "policy": soft_policy,
            "ordering_ref": ordering_ref,
            "ordered_input_ids_sha256": (
                ordered_input_ids_sha256
            ),
            "events_recorded": len(events),
        }



    def assess_student_attempt(
        *,
        target_kind: str,
        attempt_result: Any,
        teacher_hard_targets: np.ndarray,
        teacher_centred_logits: np.ndarray,
        domain_inputs: np.ndarray,
        request: Mapping[str, Any],
        output_root: Path,
    ) -> Mapping[str, Any]:
        import hashlib

        import torch

        from circuit_families.stage4_condition_identity import (
            ConditionIdentity,
            build_condition_id,
        )
        from circuit_families.stage6b.hard_target import (
            CanonicalDecisionVector,
            canonical_decision_bytes,
            evaluate_hard_target_eligibility,
        )
        from circuit_families.stage6b.records import (
            circuit_release_gate,
        )
        from circuit_families.stage6c import (
            CanonicalSoftOutput,
            assess_soft_attempt,
            evaluate_soft_target_eligibility,
            generate_soft_sealing_evidence,
            soft_circuit_release_gate,
        )

        del output_root

        if target_kind not in {"hard", "soft"}:
            raise RegisteredFixtureError(
                f"unknown target kind: {target_kind}"
            )

        if not isinstance(
            attempt_result,
            Mapping,
        ):
            raise RegisteredFixtureError(
                "student attempt result must be a mapping"
            )

        required = {
            "attempt_record",
            "outputs",
            "model",
            "outcome_kind",
            "stage3",
            "policy",
            "ordering_ref",
            "ordered_input_ids_sha256",
        }

        if not required.issubset(
            attempt_result
        ):
            raise RegisteredFixtureError(
                "student attempt result lacks "
                "assessment evidence"
            )

        attempt_record = attempt_result[
            "attempt_record"
        ]
        stage3 = attempt_result["stage3"]
        outcome_kind = str(
            attempt_result["outcome_kind"]
        )
        outputs = attempt_result["outputs"]
        policy = attempt_result["policy"]
        ordering_ref = str(
            attempt_result["ordering_ref"]
        )
        ordering_sha256 = str(
            attempt_result[
                "ordered_input_ids_sha256"
            ]
        )

        if (
            not isinstance(domain_inputs, np.ndarray)
            or domain_inputs.shape
            != (12_769, 3)
        ):
            raise RegisteredFixtureError(
                "assessment requires the complete "
                "registered domain"
            )

        if outcome_kind != "succeeded":
            if target_kind == "hard":
                assessed = assess_hard_attempt(
                    attempt_record=attempt_record,
                    stage3=stage3,
                    evaluation=None,
                )
                status = str(
                    assessed.classification
                )
            else:
                assessed = assess_soft_attempt(
                    attempt_record=attempt_record,
                    stage3=stage3,
                    evaluation=None,
                )
                status = str(
                    assessed.status
                )

            return {
                "status": status,
                "eligible": False,
                "sealed": False,
                "sealed_subject": None,
                "assessment_type": (
                    type(assessed).__name__
                ),
                "sealing_type": "None",
            }

        if not isinstance(outputs, torch.Tensor):
            raise RegisteredFixtureError(
                "succeeded student attempt lacks "
                "dense logits"
            )

        if outputs.shape != (12_769, 113):
            raise RegisteredFixtureError(
                "student dense logits must have "
                "shape [12769, 113]"
            )

        checkpoint_artifact = (
            attempt_record["payload"][
                "model_checkpoint"
            ]
        )
        architecture_ref = (
            attempt_record["payload"][
                "student_architecture_ref"
            ]
        )

        teacher_cfg = request[
            "registered_teacher"
        ]
        teacher_seed = int(
            teacher_cfg["teacher_seed"]
        )
        phase = str(
            teacher_cfg["phase_label"]
        )

        if target_kind == "hard":
            if (
                not isinstance(
                    teacher_hard_targets,
                    np.ndarray,
                )
                or teacher_hard_targets.shape
                != (12_769,)
            ):
                raise RegisteredFixtureError(
                    "hard assessment requires "
                    "12,769 teacher labels"
                )

            direct_teacher_condition_id = (
                build_condition_id(
                    ConditionIdentity(
                        teacher_seed=teacher_seed,
                        phase=phase,
                        distillation_condition=(
                            "direct_teacher"
                        ),
                    ),
                    stage3,
                )
            )

            teacher = CanonicalDecisionVector(
                role="direct_teacher",
                condition_id=(
                    direct_teacher_condition_id
                ),
                ordering_ref=ordering_ref,
                ordered_input_ids_sha256=(
                    ordering_sha256
                ),
                decisions=tuple(
                    int(value)
                    for value
                    in teacher_hard_targets.tolist()
                ),
            )

            student_values = tuple(
                int(value)
                for value in (
                    outputs.argmax(dim=-1)
                    .to(torch.int64)
                    .tolist()
                )
            )

            evaluation = (
                evaluate_hard_target_eligibility(
                    teacher=teacher,
                    student=(
                        CanonicalDecisionVector(
                            role=(
                                "hard_target_student"
                            ),
                            condition_id=str(
                                attempt_record[
                                    "condition_id"
                                ]
                            ),
                            ordering_ref=(
                                ordering_ref
                            ),
                            ordered_input_ids_sha256=(
                                ordering_sha256
                            ),
                            decisions=(
                                student_values
                            ),
                        )
                    ),
                    stage3=stage3,
                )
            )

            assessed = assess_hard_attempt(
                attempt_record=attempt_record,
                stage3=stage3,
                evaluation=evaluation,
            )

            eligible = (
                assessed.classification
                == "eligible"
            )

            sealing = None

            if eligible:
                dense_sha256 = (
                    hashlib.sha256(
                        canonical_decision_bytes(
                            student_values
                        )
                    ).hexdigest()
                )

                sealing = (
                    generate_hard_sealing_evidence(
                        assessment=assessed,
                        stage3=stage3,
                        checkpoint=(
                            checkpoint_artifact
                        ),
                        dense_output={
                            "path": (
                                "technical/stage7b/"
                                "hard_target/"
                                "dense-decisions.bin"
                            ),
                            "sha256": (
                                dense_sha256
                            ),
                            "storage_class": (
                                "external_large_object"
                            ),
                        },
                        architecture_ref=(
                            architecture_ref
                        ),
                    )
                )

            gate = circuit_release_gate(
                assessment=assessed,
                sealing=sealing,
            )

            sealed = bool(gate.allowed)
            status = str(
                assessed.classification
            )

        else:
            if (
                not isinstance(
                    teacher_centred_logits,
                    np.ndarray,
                )
                or teacher_centred_logits.shape
                != (12_769, 113)
            ):
                raise RegisteredFixtureError(
                    "soft assessment requires "
                    "centred teacher logits"
                )

            teacher = CanonicalSoftOutput(
                role="soft_target_teacher",
                condition_id=(
                    policy.representation
                    .teacher_condition_id
                ),
                ordering_ref=ordering_ref,
                ordered_input_ids_sha256=(
                    ordering_sha256
                ),
                logits=torch.as_tensor(
                    teacher_centred_logits,
                    dtype=torch.float32,
                ),
                record_status="sealed",
            )

            evaluation = (
                evaluate_soft_target_eligibility(
                    teacher=teacher,
                    student=CanonicalSoftOutput(
                        role="soft_target_student",
                        condition_id=str(
                            attempt_record[
                                "condition_id"
                            ]
                        ),
                        ordering_ref=(
                            ordering_ref
                        ),
                        ordered_input_ids_sha256=(
                            ordering_sha256
                        ),
                        logits=outputs,
                        record_status="sealed",
                    ),
                    policy=policy,
                    stage3=stage3,
                )
            )

            assessed = assess_soft_attempt(
                attempt_record=attempt_record,
                stage3=stage3,
                evaluation=evaluation,
            )

            eligible = (
                assessed.status == "eligible"
            )

            sealing = None

            if eligible:
                sealing = (
                    generate_soft_sealing_evidence(
                        assessment=assessed,
                        stage3=stage3,
                        checkpoint=(
                            checkpoint_artifact
                        ),
                        dense_output={
                            "path": (
                                "technical/stage7b/"
                                "soft_target/"
                                "centred-logits.bin"
                            ),
                            "sha256": (
                                evaluation
                                .student_soft_output_sha256
                            ),
                            "storage_class": (
                                "external_large_object"
                            ),
                        },
                        architecture_ref=(
                            architecture_ref
                        ),
                    )
                )

            gate = soft_circuit_release_gate(
                assessment=assessed,
                sealing=sealing,
            )

            sealed = bool(gate.allowed)
            status = str(
                assessed.status
            )

        return {
            "status": status,
            "eligible": bool(eligible),
            "sealed": bool(sealed),
            "sealed_subject": (
                attempt_result["model"]
                if sealed
                else None
            ),
            "assessment_type": (
                type(assessed).__name__
            ),
            "sealing_type": (
                type(sealing).__name__
                if sealing is not None
                else "None"
            ),
        }


    adapter_classes = {
        "greedy_deletion": GreedyDeletionAdapter,
        "diversity_forced": DiversityForcedAdapter,
    }

    def run_discovery(
        *,
        adapter_name: str,
        subject_kind: str,
        subject: Any,
        request: Mapping[str, Any],
        output_root: Path,
    ) -> Any:
        cls = adapter_classes.get(adapter_name)
        if cls is None:
            raise RegisteredFixtureError(
                f"unrecognized accepted discovery adapter: {adapter_name}"
            )
        values = {
            "subject_kind": subject_kind,
            "subject": subject,
            "model": subject,
            "request": request,
            "output_root": output_root,
        }
        adapter = invoke_accepted(cls, values)
        for method_name in ("run", "discover", "execute"):
            method = getattr(adapter, method_name, None)
            if callable(method):
                return invoke_accepted(method, values)
        raise RegisteredFixtureError(
            f"accepted adapter {cls.__name__} exposes no run/discover/execute method"
        )

    def run_exact_endpoints(
        *,
        adapter_name: str,
        subject_kind: str,
        discovery_result: Any,
        teacher_centred_logits: np.ndarray,
        domain_inputs: np.ndarray,
        request: Mapping[str, Any],
        output_root: Path,
    ) -> Mapping[str, Any]:
        values = {
            "adapter_name": adapter_name,
            "subject_kind": subject_kind,
            "discovery_result": discovery_result,
            "teacher_centred_logits": teacher_centred_logits,
            "teacher_logits": teacher_centred_logits,
            "domain_inputs": domain_inputs,
            "inputs": domain_inputs,
            "request": request,
            "output_root": output_root,
        }

        ledger_builder = invoke_accepted(ExactLedgerBuilder, values)
        for method_name in ("build", "seal", "finalize"):
            method = getattr(ledger_builder, method_name, None)
            if callable(method):
                maybe = invoke_accepted(method, values)
                if maybe is not None:
                    ledger = maybe
                    break
        else:
            ledger = ledger_builder

        endpoint1 = invoke_accepted(
            reduce_endpoint1,
            {**values, "ledger": ledger, "sealed_ledger": ledger},
        )
        endpoint2 = invoke_accepted(
            build_endpoint2_result,
            {
                **values,
                "ledger": ledger,
                "sealed_ledger": ledger,
                "endpoint1": endpoint1,
                "endpoint1_result": endpoint1,
            },
        )

        return {
            "adapter_name": adapter_name,
            "subject_kind": subject_kind,
            "ledger_type": type(ledger).__name__,
            "endpoint1": endpoint1,
            "endpoint2": endpoint2,
        }

    def build_excluded_outputs(
        *,
        identity: RegisteredFixtureIdentity,
        request: Mapping[str, Any],
        source_code_sha: str,
        attempt_assessments: Mapping[str, Mapping[str, Any]],
        discovery_records: Sequence[Mapping[str, Any]],
        endpoint_records: Sequence[Mapping[str, Any]],
        output_root: Path,
    ) -> Mapping[str, Any]:
        values = {
            "teacher_seed": identity.teacher_seed,
            "teacher_identity": identity,
            "identity": identity,
            "request": request,
            "source_code_sha": source_code_sha,
            "attempt_assessments": attempt_assessments,
            "discovery_records": discovery_records,
            "endpoint_records": endpoint_records,
            "output_root": output_root,
        }

        inventory = invoke_accepted(build_teacher_seed_inventory, values)
        exclusions = invoke_accepted(
            build_endpoint_exclusion_records,
            {**values, "inventory": inventory},
        )

        if isinstance(exclusions, Mapping):
            exclusion_records = list(
                exclusions.get("records", exclusions.get("exclusion_records", []))
            )
        else:
            exclusion_records = list(exclusions)

        # The accepted Stage 7 scientific firewall must reject every output as a
        # primary scientific input.  Failure to reject is blocking.
        for record in exclusion_records:
            invoke_accepted(
                assert_rejected_as_primary_scientific_input,
                {
                    **values,
                    "record": record,
                    "exclusion_record": record,
                },
            )

        report = invoke_accepted(
            build_part_f_report,
            {
                **values,
                "inventory": inventory,
                "exclusion_records": exclusion_records,
                "exclusions": exclusions,
            },
        )

        # Convert accepted records to JSON-safe dicts only for local technical
        # inventory.  Endpoint values themselves remain inside excluded local
        # products and are never printed.
        normalized = []
        for record in exclusion_records:
            if isinstance(record, Mapping):
                item = dict(record)
            elif hasattr(record, "to_record") and callable(record.to_record):
                item = dict(record.to_record())
            elif hasattr(record, "__dict__"):
                item = dict(record.__dict__)
            else:
                raise RegisteredFixtureError(
                    "accepted exclusion record is not serializable"
                )

            item["lifecycle_state"] = "excluded"
            item["primary_input_eligible"] = False
            item["regeneration_required_after_definitive_freeze"] = True
            normalized.append(item)

        if isinstance(report, Mapping):
            report_mapping = dict(report)
        elif hasattr(report, "to_record") and callable(report.to_record):
            report_mapping = dict(report.to_record())
        elif hasattr(report, "__dict__"):
            report_mapping = dict(report.__dict__)
        else:
            report_mapping = {"report_type": type(report).__name__}

        return {
            "exclusion_records": normalized,
            "report": report_mapping,
        }

    return RegisteredFixtureBindings(
        restore_model=restore_model,
        evaluate_teacher=evaluate_teacher,
        build_target_caches=build_target_caches,
        run_student_attempt=run_student_attempt,
        assess_student_attempt=assess_student_attempt,
        run_discovery=run_discovery,
        run_exact_endpoints=run_exact_endpoints,
        build_excluded_outputs=build_excluded_outputs,
        load_checkpoint_payload=load_checkpoint_payload,
        discovery_adapter_names=("greedy_deletion", "diversity_forced"),
    )
