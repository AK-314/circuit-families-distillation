"""Policy-neutral technical teacher training for Stage 12-P1.

This module owns mechanics only:

- task/config/seed/backend identity binding;
- injected model, loss, optimizer, scheduler, and stopping interfaces;
- deterministic technical execution;
- atomic rolling checkpoints with bounded retention;
- exact continuation from a bound checkpoint;
- explicit technical terminal states;
- compact metrics and sealed technical artifacts.

It chooses no production task, architecture, teacher seed, optimiser, schedule,
stopping threshold, training budget, or backend qualification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import torch

from circuit_families.seeds import seed_everything
from circuit_families.stage12p1.tasks import (
    TaskImplementationRegistry,
    canonical_json_bytes,
    canonical_sha256,
    validate_task_record,
)
from circuit_families.training.checkpoints import (
    canonical_state_hash,
    file_sha256,
)

TEACHER_CHECKPOINT_SCHEMA_VERSION = "stage12p1-teacher-checkpoint/v1"
TEACHER_ARTIFACT_SCHEMA_VERSION = "stage12p1-teacher-artifact/v1"
TEACHER_DATA_SCHEMA_VERSION = "stage12p1-teacher-data/v1"

TEACHER_STATES = (
    "completed",
    "failed",
    "interrupted",
    "numerical-failure",
    "unavailable",
)

_TERMINAL_NO_RESUME = frozenset(
    {
        "completed",
        "failed",
        "numerical-failure",
        "unavailable",
    }
)

_RESUME_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class TeacherProtocolError(ValueError):
    """Raised when technical teacher mechanics violate their contract."""


class TeacherModelConstructor(Protocol):
    def __call__(
        self,
        *,
        task_record: Mapping[str, Any],
        architecture_config: Mapping[str, Any],
        seed: int,
        device: torch.device,
    ) -> torch.nn.Module:
        ...


class TeacherLoss(Protocol):
    def __call__(
        self,
        *,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        config: Mapping[str, Any],
    ) -> torch.Tensor:
        ...


class TeacherOptimizerFactory(Protocol):
    def __call__(
        self,
        *,
        model: torch.nn.Module,
        config: Mapping[str, Any],
    ) -> torch.optim.Optimizer:
        ...


class TeacherSchedulerFactory(Protocol):
    def __call__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        config: Mapping[str, Any],
    ) -> object | None:
        ...


class TeacherStopRule(Protocol):
    def __call__(
        self,
        *,
        step: int,
        metrics: Mapping[str, float],
        config: Mapping[str, Any],
    ) -> bool:
        ...


@dataclass(frozen=True)
class TeacherDataBundle:
    """Explicit train/test tensors bound to one task identity."""

    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    test_inputs: torch.Tensor
    test_targets: torch.Tensor
    task_identity_sha256: str
    dataset_sha256: str
    split_identity_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "train_inputs",
            "train_targets",
            "test_inputs",
            "test_targets",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TeacherProtocolError(f"{name} must be a torch.Tensor")

        if self.train_inputs.ndim < 1 or self.test_inputs.ndim < 1:
            raise TeacherProtocolError("teacher inputs must have a batch dimension")

        if self.train_targets.ndim != 1 or self.test_targets.ndim != 1:
            raise TeacherProtocolError("teacher targets must be rank-1 class IDs")

        if self.train_targets.dtype != torch.int64:
            raise TeacherProtocolError("train_targets must use torch.int64")
        if self.test_targets.dtype != torch.int64:
            raise TeacherProtocolError("test_targets must use torch.int64")

        if self.train_inputs.shape[0] != self.train_targets.shape[0]:
            raise TeacherProtocolError("train input/target counts disagree")
        if self.test_inputs.shape[0] != self.test_targets.shape[0]:
            raise TeacherProtocolError("test input/target counts disagree")

        if self.train_targets.numel() == 0 or self.test_targets.numel() == 0:
            raise TeacherProtocolError(
                "technical teacher train/test partitions must be non-empty"
            )

        for name in (
            "task_identity_sha256",
            "dataset_sha256",
            "split_identity_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise TeacherProtocolError(f"{name} must be lowercase SHA-256")


@dataclass(frozen=True)
class TeacherTrainingRequest:
    """All policy-bearing teacher settings must be supplied explicitly."""

    task_record: Mapping[str, Any]
    architecture_config: Mapping[str, Any]
    training_config: Mapping[str, Any]
    optimizer_config: Mapping[str, Any]
    scheduler_config: Mapping[str, Any]
    stopping_config: Mapping[str, Any]
    backend_qualification: Mapping[str, Any]
    model_seed_id: str
    model_seed: int
    training_seed_id: str
    training_seed: int
    max_technical_updates: int
    checkpoint_interval: int
    checkpoint_retention: int
    output_root: Path
    resume_id: str

    def __post_init__(self) -> None:
        validate_task_record(self.task_record)

        for name in (
            "architecture_config",
            "training_config",
            "optimizer_config",
            "scheduler_config",
            "stopping_config",
            "backend_qualification",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TeacherProtocolError(f"{name} must be a mapping")
            canonical_json_bytes(value)

        for name in ("model_seed_id", "training_seed_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TeacherProtocolError(f"{name} must be a non-empty string")

        for name in ("model_seed", "training_seed"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 2**32 - 1
            ):
                raise TeacherProtocolError(
                    f"{name} must be an integer in [0, 2**32 - 1]"
                )

        for name in (
            "max_technical_updates",
            "checkpoint_interval",
            "checkpoint_retention",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TeacherProtocolError(f"{name} must be a positive integer")

        if not isinstance(self.output_root, Path):
            raise TeacherProtocolError("output_root must be pathlib.Path")

        if (
            not isinstance(self.resume_id, str)
            or _RESUME_ID_RE.fullmatch(self.resume_id) is None
        ):
            raise TeacherProtocolError(
                "resume_id must be a portable 1-128 character identifier"
            )

        qualification = self.backend_qualification
        required = {
            "backend_id",
            "device",
            "qualified",
            "exact_resume_supported",
            "qualification_ref",
        }
        if set(qualification) != required:
            raise TeacherProtocolError(
                "backend_qualification keys must be exactly "
                f"{sorted(required)!r}"
            )
        if qualification["qualified"] is not True:
            raise TeacherProtocolError(
                "backend must be explicitly qualified before teacher execution"
            )
        if qualification["exact_resume_supported"] is not True:
            raise TeacherProtocolError(
                "backend must explicitly support exact resume"
            )
        for field in ("backend_id", "device", "qualification_ref"):
            if (
                not isinstance(qualification[field], str)
                or not qualification[field]
            ):
                raise TeacherProtocolError(
                    f"backend_qualification.{field} must be non-empty"
                )


@dataclass(frozen=True)
class TeacherRunResult:
    status: str
    updates_completed: int
    artifact_path: Path
    artifact_sha256: str
    checkpoint_paths: tuple[Path, ...]
    resumed_from_checkpoint_sha256: str | None
    reused_terminal_artifact: bool = False

    def __post_init__(self) -> None:
        if self.status not in TEACHER_STATES:
            raise TeacherProtocolError(f"invalid teacher status {self.status!r}")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    metadata = canonical_json_bytes(
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def teacher_data_sha256(bundle: TeacherDataBundle) -> str:
    return canonical_sha256(
        {
            "schema_version": TEACHER_DATA_SCHEMA_VERSION,
            "task_identity_sha256": bundle.task_identity_sha256,
            "dataset_sha256": bundle.dataset_sha256,
            "split_identity_sha256": bundle.split_identity_sha256,
            "train_inputs_sha256": _tensor_sha256(bundle.train_inputs),
            "train_targets_sha256": _tensor_sha256(bundle.train_targets),
            "test_inputs_sha256": _tensor_sha256(bundle.test_inputs),
            "test_targets_sha256": _tensor_sha256(bundle.test_targets),
        }
    )


def build_technical_tabular_teacher_data(
    task_record: Mapping[str, Any],
    *,
    implementation_registry: TaskImplementationRegistry | None = None,
) -> TeacherDataBundle:
    """Materialize a tiny technical task using raw coordinate vectors.

    This helper is deliberately technical-fixture-only. Production
    representations remain injected rather than implied by this convenience
    path.
    """
    validated = validate_task_record(
        task_record,
        implementation_registry=implementation_registry,
    )

    if validated["classification"] != "technical_fixture":
        raise TeacherProtocolError(
            "tabular teacher-data helper is technical-fixture-only"
        )
    if validated["scientific_data"] is not False:
        raise TeacherProtocolError("scientific task data are forbidden here")
    if validated["production_eligible"] is not False:
        raise TeacherProtocolError("production-eligible task data are forbidden here")

    definition = validated["task_definition"]
    split = definition["split_identity"]
    if (
        not isinstance(split, Mapping)
        or "train_indices" not in split
        or "test_indices" not in split
    ):
        raise TeacherProtocolError(
            "technical tabular helper requires explicit train_indices/test_indices"
        )

    registry = (
        TaskImplementationRegistry()
        if implementation_registry is None
        else implementation_registry
    )
    implementation = registry.implementation(
        definition["implementation"],
        definition["implementation_version"],
    )

    import itertools

    domains = tuple(
        tuple(int(value) for value in domain)
        for domain in definition["input_domains"]
    )
    examples = tuple(itertools.product(*domains))
    targets = tuple(
        implementation.target(
            tuple(int(value) for value in example),
            parameters=definition["parameters"],
            modulus=int(definition["modulus"]),
        )
        for example in examples
    )

    def normalize_indices(raw: Any, *, name: str) -> tuple[int, ...]:
        if not isinstance(raw, Sequence) or isinstance(
            raw,
            (str, bytes, bytearray),
        ):
            raise TeacherProtocolError(f"{name} must be a sequence")
        result = tuple(raw)
        if not result:
            raise TeacherProtocolError(f"{name} must not be empty")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(examples)
            for index in result
        ):
            raise TeacherProtocolError(f"{name} contains invalid index")
        if len(set(result)) != len(result):
            raise TeacherProtocolError(f"{name} contains duplicates")
        return result

    train_indices = normalize_indices(
        split["train_indices"],
        name="split_identity.train_indices",
    )
    test_indices = normalize_indices(
        split["test_indices"],
        name="split_identity.test_indices",
    )

    if set(train_indices) & set(test_indices):
        raise TeacherProtocolError("technical train/test indices overlap")

    all_inputs = torch.tensor(examples, dtype=torch.float32)
    all_targets = torch.tensor(targets, dtype=torch.int64)

    return TeacherDataBundle(
        train_inputs=all_inputs[list(train_indices)].clone(),
        train_targets=all_targets[list(train_indices)].clone(),
        test_inputs=all_inputs[list(test_indices)].clone(),
        test_targets=all_targets[list(test_indices)].clone(),
        task_identity_sha256=validated["hashes"]["task_identity_sha256"],
        dataset_sha256=validated["hashes"]["dataset_sha256"],
        split_identity_sha256=validated["hashes"]["split_identity_sha256"],
    )


def _request_hashes(
    request: TeacherTrainingRequest,
    data: TeacherDataBundle | None,
) -> dict[str, str]:
    hashes = {
        "task_record_sha256": canonical_sha256(request.task_record),
        "task_identity_sha256": request.task_record["hashes"][
            "task_identity_sha256"
        ],
        "architecture_config_sha256": canonical_sha256(
            request.architecture_config
        ),
        "training_config_sha256": canonical_sha256(request.training_config),
        "optimizer_config_sha256": canonical_sha256(request.optimizer_config),
        "scheduler_config_sha256": canonical_sha256(request.scheduler_config),
        "stopping_config_sha256": canonical_sha256(request.stopping_config),
        "backend_qualification_sha256": canonical_sha256(
            request.backend_qualification
        ),
    }
    if data is not None:
        hashes["teacher_data_sha256"] = teacher_data_sha256(data)
    return hashes


def _environment_record(
    request: TeacherTrainingRequest,
) -> dict[str, Any]:
    record = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "backend_id": request.backend_qualification["backend_id"],
        "device": request.backend_qualification["device"],
        "qualification_ref": request.backend_qualification["qualification_ref"],
    }
    return {
        "record": record,
        "sha256": canonical_sha256(record),
    }


def _run_root(request: TeacherTrainingRequest) -> Path:
    return request.output_root / request.resume_id


def _relative_to_run(run_root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(run_root)
    except ValueError as exc:
        raise TeacherProtocolError("teacher artifact escaped run root") from exc
    return PurePosixPath(relative.as_posix()).as_posix()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, path)
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _atomic_publish_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise TeacherProtocolError(
                f"sealed teacher artifact already exists: {path}"
            ) from exc

        Path(temporary_name).unlink()
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "cpu": torch.get_rng_state().clone(),
    }

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise TeacherProtocolError("CUDA backend requested but unavailable")
        state["cuda"] = [
            item.clone()
            for item in torch.cuda.get_rng_state_all()
        ]
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        getter = getattr(mps, "get_rng_state", None) if mps is not None else None
        if getter is None or not callable(getter):
            raise TeacherProtocolError(
                "qualified MPS exact resume requires torch.mps.get_rng_state()"
            )
        state["mps"] = getter().clone()

    return state


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    if "cpu" not in state or not isinstance(state["cpu"], torch.Tensor):
        raise TeacherProtocolError("checkpoint lacks CPU RNG state")
    torch.set_rng_state(state["cpu"])

    if device.type == "cuda":
        values = state.get("cuda")
        if not isinstance(values, list):
            raise TeacherProtocolError("checkpoint lacks CUDA RNG states")
        torch.cuda.set_rng_state_all(values)
    elif device.type == "mps":
        value = state.get("mps")
        mps = getattr(torch, "mps", None)
        setter = getattr(mps, "set_rng_state", None) if mps is not None else None
        if not isinstance(value, torch.Tensor) or setter is None or not callable(setter):
            raise TeacherProtocolError("checkpoint lacks restorable MPS RNG state")
        setter(value)


def _scheduler_state(scheduler: object | None) -> Any:
    if scheduler is None:
        return None
    method = getattr(scheduler, "state_dict", None)
    if method is None or not callable(method):
        raise TeacherProtocolError(
            "injected scheduler must expose state_dict() for resume"
        )
    return copy.deepcopy(method())


def _checkpoint_payload(
    *,
    request: TeacherTrainingRequest,
    request_hashes: Mapping[str, str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    device: torch.device,
    updates_completed: int,
    metric_snapshots: Sequence[Mapping[str, Any]],
    resume_lineage: Sequence[str],
) -> dict[str, Any]:
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = _scheduler_state(scheduler)
    rng_state = _capture_rng_state(device)

    return {
        "schema_version": TEACHER_CHECKPOINT_SCHEMA_VERSION,
        "scientific_data": False,
        "production_eligible": False,
        "resume_id": request.resume_id,
        "request_hashes": copy.deepcopy(dict(request_hashes)),
        "model_seed_id": request.model_seed_id,
        "model_seed": request.model_seed,
        "training_seed_id": request.training_seed_id,
        "training_seed": request.training_seed,
        "updates_completed": updates_completed,
        "metric_snapshots": copy.deepcopy(list(metric_snapshots)),
        "resume_lineage": list(resume_lineage),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng_state": rng_state,
        "state_hashes": {
            "model_state_sha256": canonical_state_hash(model_state),
            "optimizer_state_sha256": canonical_state_hash(optimizer_state),
            "scheduler_state_sha256": canonical_state_hash(scheduler_state),
            "rng_state_sha256": canonical_state_hash(rng_state),
            "metrics_sha256": canonical_sha256(list(metric_snapshots)),
        },
    }


def _atomic_save_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise TeacherProtocolError(
            f"teacher checkpoint already exists: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())

        os.link(temporary_name, path)
        Path(temporary_name).unlink()
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass

    return file_sha256(path)


def _checkpoint_paths(run_root: Path) -> tuple[Path, ...]:
    root = run_root / "checkpoints"
    if not root.is_dir():
        return ()
    return tuple(sorted(root.glob("step_*.pt"), key=lambda path: path.name))


def _write_latest_pointer(
    run_root: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    updates_completed: int,
) -> None:
    record = {
        "schema_version": "stage12p1-teacher-latest-checkpoint/v1",
        "scientific_data": False,
        "production_eligible": False,
        "checkpoint_path": _relative_to_run(run_root, checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "updates_completed": updates_completed,
    }
    _atomic_replace_bytes(
        run_root / "latest_checkpoint.json",
        canonical_json_bytes(record),
    )


def _apply_checkpoint_retention(
    run_root: Path,
    *,
    retention: int,
) -> tuple[Path, ...]:
    checkpoints = list(_checkpoint_paths(run_root))
    while len(checkpoints) > retention:
        oldest = checkpoints.pop(0)
        oldest.unlink()
    _fsync_directory(run_root / "checkpoints")
    return tuple(checkpoints)


def _save_rolling_checkpoint(
    *,
    run_root: Path,
    request: TeacherTrainingRequest,
    request_hashes: Mapping[str, str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    device: torch.device,
    updates_completed: int,
    metric_snapshots: Sequence[Mapping[str, Any]],
    resume_lineage: Sequence[str],
) -> tuple[Path, str, tuple[Path, ...]]:
    path = (
        run_root
        / "checkpoints"
        / f"step_{updates_completed:08d}.pt"
    )
    payload = _checkpoint_payload(
        request=request,
        request_hashes=request_hashes,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        updates_completed=updates_completed,
        metric_snapshots=metric_snapshots,
        resume_lineage=resume_lineage,
    )
    digest = _atomic_save_checkpoint(path, payload)
    _write_latest_pointer(run_root, path, digest, updates_completed)
    retained = _apply_checkpoint_retention(
        run_root,
        retention=request.checkpoint_retention,
    )
    return path, digest, retained


def _latest_checkpoint_record(run_root: Path) -> Mapping[str, Any] | None:
    path = run_root / "latest_checkpoint.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeacherProtocolError(
            "latest checkpoint pointer is unreadable"
        ) from exc

    required = {
        "schema_version",
        "scientific_data",
        "production_eligible",
        "checkpoint_path",
        "checkpoint_sha256",
        "updates_completed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise TeacherProtocolError("latest checkpoint pointer keys mismatch")
    if value["scientific_data"] is not False:
        raise TeacherProtocolError("latest checkpoint pointer authority mismatch")
    if value["production_eligible"] is not False:
        raise TeacherProtocolError("latest checkpoint pointer authority mismatch")
    if not isinstance(value["checkpoint_path"], str):
        raise TeacherProtocolError("latest checkpoint path is invalid")
    if (
        not isinstance(value["checkpoint_sha256"], str)
        or _SHA256_RE.fullmatch(value["checkpoint_sha256"]) is None
    ):
        raise TeacherProtocolError("latest checkpoint SHA-256 is invalid")
    return value


def _load_checkpoint(
    *,
    run_root: Path,
    request: TeacherTrainingRequest,
    request_hashes: Mapping[str, str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], list[str], str] | None:
    pointer = _latest_checkpoint_record(run_root)
    if pointer is None:
        return None

    relative = PurePosixPath(pointer["checkpoint_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise TeacherProtocolError("latest checkpoint path is not portable")
    path = run_root.joinpath(*relative.parts)

    if not path.is_file():
        raise TeacherProtocolError("latest checkpoint file is missing")

    actual_file_hash = file_sha256(path)
    if actual_file_hash != pointer["checkpoint_sha256"]:
        raise TeacherProtocolError("latest checkpoint physical hash mismatch")

    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except Exception as exc:
        raise TeacherProtocolError("teacher checkpoint could not be decoded") from exc

    required = {
        "schema_version",
        "scientific_data",
        "production_eligible",
        "resume_id",
        "request_hashes",
        "model_seed_id",
        "model_seed",
        "training_seed_id",
        "training_seed",
        "updates_completed",
        "metric_snapshots",
        "resume_lineage",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "state_hashes",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise TeacherProtocolError("teacher checkpoint keys mismatch")

    if payload["schema_version"] != TEACHER_CHECKPOINT_SCHEMA_VERSION:
        raise TeacherProtocolError("teacher checkpoint schema mismatch")
    if payload["scientific_data"] is not False:
        raise TeacherProtocolError("teacher checkpoint scientific_data must be false")
    if payload["production_eligible"] is not False:
        raise TeacherProtocolError(
            "teacher checkpoint production_eligible must be false"
        )
    if payload["resume_id"] != request.resume_id:
        raise TeacherProtocolError("teacher checkpoint resume identity mismatch")
    if dict(payload["request_hashes"]) != dict(request_hashes):
        raise TeacherProtocolError("teacher checkpoint request hashes are stale")
    if payload["model_seed_id"] != request.model_seed_id:
        raise TeacherProtocolError("teacher checkpoint model seed ID mismatch")
    if payload["model_seed"] != request.model_seed:
        raise TeacherProtocolError("teacher checkpoint model seed mismatch")
    if payload["training_seed_id"] != request.training_seed_id:
        raise TeacherProtocolError("teacher checkpoint training seed ID mismatch")
    if payload["training_seed"] != request.training_seed:
        raise TeacherProtocolError("teacher checkpoint training seed mismatch")

    actual_state_hashes = {
        "model_state_sha256": canonical_state_hash(payload["model_state"]),
        "optimizer_state_sha256": canonical_state_hash(payload["optimizer_state"]),
        "scheduler_state_sha256": canonical_state_hash(payload["scheduler_state"]),
        "rng_state_sha256": canonical_state_hash(payload["rng_state"]),
        "metrics_sha256": canonical_sha256(payload["metric_snapshots"]),
    }
    if dict(payload["state_hashes"]) != actual_state_hashes:
        raise TeacherProtocolError("teacher checkpoint internal state hash mismatch")

    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])

    saved_scheduler = payload["scheduler_state"]
    if scheduler is None and saved_scheduler is not None:
        raise TeacherProtocolError(
            "checkpoint contains scheduler state but request has no scheduler"
        )
    if scheduler is not None and saved_scheduler is None:
        raise TeacherProtocolError("checkpoint is missing scheduler state")
    if scheduler is not None:
        loader = getattr(scheduler, "load_state_dict", None)
        if loader is None or not callable(loader):
            raise TeacherProtocolError(
                "injected scheduler must expose load_state_dict()"
            )
        loader(saved_scheduler)

    _restore_rng_state(payload["rng_state"], device)

    updates = payload["updates_completed"]
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates < 0
        or updates > request.max_technical_updates
    ):
        raise TeacherProtocolError("checkpoint update count is invalid")

    metric_snapshots = payload["metric_snapshots"]
    if not isinstance(metric_snapshots, list):
        raise TeacherProtocolError("checkpoint metrics must be a list")

    lineage = payload["resume_lineage"]
    if not isinstance(lineage, list) or any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in lineage
    ):
        raise TeacherProtocolError("checkpoint resume lineage is invalid")

    return (
        updates,
        copy.deepcopy(metric_snapshots),
        [*lineage, actual_file_hash],
        actual_file_hash,
    )


def _evaluate(
    *,
    model: torch.nn.Module,
    loss_fn: TeacherLoss,
    loss_config: Mapping[str, Any],
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            train_outputs = model(train_inputs)
            test_outputs = model(test_inputs)

            if train_outputs.ndim != 2 or test_outputs.ndim != 2:
                raise TeacherProtocolError(
                    "teacher model outputs must be rank-2 class logits"
                )

            train_loss = loss_fn(
                outputs=train_outputs,
                targets=train_targets,
                config=loss_config,
            )
            test_loss = loss_fn(
                outputs=test_outputs,
                targets=test_targets,
                config=loss_config,
            )

            if not isinstance(train_loss, torch.Tensor) or train_loss.ndim != 0:
                raise TeacherProtocolError("teacher loss must return scalar tensor")
            if not isinstance(test_loss, torch.Tensor) or test_loss.ndim != 0:
                raise TeacherProtocolError("teacher loss must return scalar tensor")

            values = {
                "train_loss": float(train_loss.detach().item()),
                "test_loss": float(test_loss.detach().item()),
                "train_accuracy": float(
                    (train_outputs.argmax(dim=-1) == train_targets)
                    .float()
                    .mean()
                    .item()
                ),
                "test_accuracy": float(
                    (test_outputs.argmax(dim=-1) == test_targets)
                    .float()
                    .mean()
                    .item()
                ),
            }
    finally:
        model.train(was_training)

    import math

    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("nonfinite_evaluation_metrics")

    return values


def _artifact_payload(
    *,
    request: TeacherTrainingRequest,
    request_hashes: Mapping[str, str],
    environment: Mapping[str, Any],
    status: str,
    reason: str,
    updates_completed: int,
    metric_snapshots: Sequence[Mapping[str, Any]],
    checkpoint_paths: Sequence[Path],
    checkpoint_hashes: Mapping[str, str],
    resumed_from_checkpoint_sha256: str | None,
    resume_lineage: Sequence[str],
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
) -> dict[str, Any]:
    if status not in TEACHER_STATES:
        raise TeacherProtocolError(f"unsupported teacher artifact status {status!r}")

    run_root = _run_root(request)

    state_hashes: dict[str, str] | None = None
    if model is not None and optimizer is not None:
        state_hashes = {
            "model_state_sha256": canonical_state_hash(model.state_dict()),
            "optimizer_state_sha256": canonical_state_hash(
                optimizer.state_dict()
            ),
            "scheduler_state_sha256": canonical_state_hash(
                _scheduler_state(scheduler)
            ),
            "metrics_sha256": canonical_sha256(list(metric_snapshots)),
        }

    payload = {
        "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
        "classification": "technical_fixture",
        "scientific_data": False,
        "production_eligible": False,
        "status": status,
        "reason": reason,
        "resume_id": request.resume_id,
        "task_id": request.task_record["task_definition"]["task_id"],
        "task_identity_sha256": request.task_record["hashes"][
            "task_identity_sha256"
        ],
        "request_hashes": copy.deepcopy(dict(request_hashes)),
        "seed_identity": {
            "model_seed_id": request.model_seed_id,
            "model_seed": request.model_seed,
            "training_seed_id": request.training_seed_id,
            "training_seed": request.training_seed,
        },
        "backend_qualification": copy.deepcopy(
            dict(request.backend_qualification)
        ),
        "environment": copy.deepcopy(dict(environment)),
        "technical_budget": {
            "max_technical_updates": request.max_technical_updates,
            "checkpoint_interval": request.checkpoint_interval,
            "checkpoint_retention": request.checkpoint_retention,
        },
        "updates_completed": updates_completed,
        "metric_snapshots": copy.deepcopy(list(metric_snapshots)),
        "checkpoint_inventory": [
            {
                "path": _relative_to_run(run_root, path),
                "sha256": checkpoint_hashes[_relative_to_run(run_root, path)],
            }
            for path in checkpoint_paths
        ],
        "resumed_from_checkpoint_sha256": resumed_from_checkpoint_sha256,
        "resume_lineage": list(resume_lineage),
        "state_hashes": state_hashes,
    }

    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def validate_teacher_artifact(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TeacherProtocolError("teacher artifact must be a mapping")

    required = {
        "schema_version",
        "classification",
        "scientific_data",
        "production_eligible",
        "status",
        "reason",
        "resume_id",
        "task_id",
        "task_identity_sha256",
        "request_hashes",
        "seed_identity",
        "backend_qualification",
        "environment",
        "technical_budget",
        "updates_completed",
        "metric_snapshots",
        "checkpoint_inventory",
        "resumed_from_checkpoint_sha256",
        "resume_lineage",
        "state_hashes",
        "content_sha256",
    }
    if set(record) != required:
        raise TeacherProtocolError("teacher artifact keys mismatch")

    if record["schema_version"] != TEACHER_ARTIFACT_SCHEMA_VERSION:
        raise TeacherProtocolError("teacher artifact schema mismatch")
    if record["classification"] != "technical_fixture":
        raise TeacherProtocolError("teacher artifact classification mismatch")
    if record["scientific_data"] is not False:
        raise TeacherProtocolError("teacher artifact scientific_data must be false")
    if record["production_eligible"] is not False:
        raise TeacherProtocolError(
            "teacher artifact production_eligible must be false"
        )
    if record["status"] not in TEACHER_STATES:
        raise TeacherProtocolError("teacher artifact status is invalid")

    stored_hash = record["content_sha256"]
    if not isinstance(stored_hash, str) or _SHA256_RE.fullmatch(stored_hash) is None:
        raise TeacherProtocolError("teacher artifact content hash is invalid")

    material = copy.deepcopy(dict(record))
    material.pop("content_sha256")
    if canonical_sha256(material) != stored_hash:
        raise TeacherProtocolError("teacher artifact content hash mismatch")

    return copy.deepcopy(dict(record))


def _artifact_path(
    *,
    request: TeacherTrainingRequest,
    status: str,
    updates_completed: int,
) -> Path:
    return (
        _run_root(request)
        / "artifacts"
        / f"{status}-{updates_completed:08d}.json"
    )


def _publish_artifact(
    *,
    request: TeacherTrainingRequest,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    validated = validate_teacher_artifact(payload)
    path = _artifact_path(
        request=request,
        status=validated["status"],
        updates_completed=validated["updates_completed"],
    )
    raw = canonical_json_bytes(validated)
    _atomic_publish_once(path, raw)
    return path, hashlib.sha256(raw).hexdigest()


def _artifact_files(run_root: Path) -> tuple[Path, ...]:
    root = run_root / "artifacts"
    if not root.is_dir():
        return ()
    return tuple(sorted(root.glob("*.json"), key=lambda path: path.name))


def _existing_terminal_result(
    request: TeacherTrainingRequest,
) -> TeacherRunResult | None:
    run_root = _run_root(request)

    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in _artifact_files(run_root):
        try:
            record = validate_teacher_artifact(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, TeacherProtocolError) as exc:
            raise TeacherProtocolError(
                f"existing teacher artifact is invalid: {path.name}"
            ) from exc

        if record["resume_id"] != request.resume_id:
            raise TeacherProtocolError("existing artifact resume identity mismatch")

        expected_request_hashes = _request_hashes(request, None)
        stored_request_hashes = record["request_hashes"]
        for field, expected in expected_request_hashes.items():
            if stored_request_hashes.get(field) != expected:
                raise TeacherProtocolError(
                    "existing terminal artifact request hashes are stale"
                )

        if record["status"] in _TERMINAL_NO_RESUME:
            candidates.append((path, record))

    if not candidates:
        return None
    if len(candidates) != 1:
        raise TeacherProtocolError(
            "multiple terminal teacher artifacts exist for one resume identity"
        )

    path, record = candidates[0]
    return TeacherRunResult(
        status=record["status"],
        updates_completed=record["updates_completed"],
        artifact_path=path,
        artifact_sha256=file_sha256(path),
        checkpoint_paths=_checkpoint_paths(run_root),
        resumed_from_checkpoint_sha256=record[
            "resumed_from_checkpoint_sha256"
        ],
        reused_terminal_artifact=True,
    )


def emit_unavailable_teacher(
    request: TeacherTrainingRequest,
    *,
    reason: str,
) -> TeacherRunResult:
    """Seal one explicit unavailable technical teacher state."""
    if not isinstance(reason, str) or not reason or "\n" in reason:
        raise TeacherProtocolError(
            "unavailable reason must be a non-empty single line"
        )

    existing = _existing_terminal_result(request)
    if existing is not None:
        return existing

    run_root = _run_root(request)
    run_root.mkdir(parents=True, exist_ok=True)

    hashes = _request_hashes(request, None)
    environment = _environment_record(request)
    payload = _artifact_payload(
        request=request,
        request_hashes=hashes,
        environment=environment,
        status="unavailable",
        reason=reason,
        updates_completed=0,
        metric_snapshots=(),
        checkpoint_paths=(),
        checkpoint_hashes={},
        resumed_from_checkpoint_sha256=None,
        resume_lineage=(),
        model=None,
        optimizer=None,
        scheduler=None,
    )
    path, digest = _publish_artifact(request=request, payload=payload)
    return TeacherRunResult(
        status="unavailable",
        updates_completed=0,
        artifact_path=path,
        artifact_sha256=digest,
        checkpoint_paths=(),
        resumed_from_checkpoint_sha256=None,
    )


class TeacherTrainingAdapter:
    """Injected policy-neutral teacher execution adapter."""

    def __init__(
        self,
        *,
        model_constructor: TeacherModelConstructor,
        loss_fn: TeacherLoss,
        optimizer_factory: TeacherOptimizerFactory,
        scheduler_factory: TeacherSchedulerFactory,
        stop_rule: TeacherStopRule,
    ) -> None:
        for name, value in (
            ("model_constructor", model_constructor),
            ("loss_fn", loss_fn),
            ("optimizer_factory", optimizer_factory),
            ("scheduler_factory", scheduler_factory),
            ("stop_rule", stop_rule),
        ):
            if not callable(value):
                raise TeacherProtocolError(f"{name} must be callable")

        self.model_constructor = model_constructor
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.stop_rule = stop_rule

    def run(
        self,
        *,
        request: TeacherTrainingRequest,
        data: TeacherDataBundle,
        interrupt_after_updates: int | None = None,
    ) -> TeacherRunResult:
        """Run or resume one exact technical teacher identity."""
        if not isinstance(request, TeacherTrainingRequest):
            raise TeacherProtocolError("request must be TeacherTrainingRequest")
        if not isinstance(data, TeacherDataBundle):
            raise TeacherProtocolError("data must be TeacherDataBundle")

        task_hash = request.task_record["hashes"]["task_identity_sha256"]
        if data.task_identity_sha256 != task_hash:
            raise TeacherProtocolError("teacher data task identity mismatch")
        if data.dataset_sha256 != request.task_record["hashes"]["dataset_sha256"]:
            raise TeacherProtocolError("teacher data dataset hash mismatch")
        if (
            data.split_identity_sha256
            != request.task_record["hashes"]["split_identity_sha256"]
        ):
            raise TeacherProtocolError("teacher data split hash mismatch")

        if interrupt_after_updates is not None:
            if (
                isinstance(interrupt_after_updates, bool)
                or not isinstance(interrupt_after_updates, int)
                or interrupt_after_updates <= 0
                or interrupt_after_updates > request.max_technical_updates
            ):
                raise TeacherProtocolError(
                    "interrupt_after_updates must lie within technical budget"
                )

        existing = _existing_terminal_result(request)
        if existing is not None:
            return existing

        run_root = _run_root(request)
        run_root.mkdir(parents=True, exist_ok=True)

        request_hashes = _request_hashes(request, data)
        environment = _environment_record(request)
        device = torch.device(request.backend_qualification["device"])

        model: torch.nn.Module | None = None
        optimizer: torch.optim.Optimizer | None = None
        scheduler: object | None = None
        updates_completed = 0
        metric_snapshots: list[dict[str, Any]] = []
        resume_lineage: list[str] = []
        resumed_from: str | None = None
        retained_paths: tuple[Path, ...] = _checkpoint_paths(run_root)
        checkpoint_hashes: dict[str, str] = {
            _relative_to_run(run_root, path): file_sha256(path)
            for path in retained_paths
        }

        try:
            seed_everything(request.model_seed, deterministic_torch=True)
            model = self.model_constructor(
                task_record=request.task_record,
                architecture_config=request.architecture_config,
                seed=request.model_seed,
                device=device,
            )
            if not isinstance(model, torch.nn.Module):
                raise TeacherProtocolError(
                    "model_constructor must return torch.nn.Module"
                )
            model.to(device)

            optimizer = self.optimizer_factory(
                model=model,
                config=request.optimizer_config,
            )
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TeacherProtocolError(
                    "optimizer_factory must return torch.optim.Optimizer"
                )

            scheduler = self.scheduler_factory(
                optimizer=optimizer,
                config=request.scheduler_config,
            )

            seed_everything(request.training_seed, deterministic_torch=True)

            train_inputs = data.train_inputs.to(device)
            train_targets = data.train_targets.to(device)
            test_inputs = data.test_inputs.to(device)
            test_targets = data.test_targets.to(device)

            restored = _load_checkpoint(
                run_root=run_root,
                request=request,
                request_hashes=request_hashes,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

            if restored is None:
                initial_metrics = _evaluate(
                    model=model,
                    loss_fn=self.loss_fn,
                    loss_config=request.training_config,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    test_inputs=test_inputs,
                    test_targets=test_targets,
                )
                metric_snapshots.append(
                    {
                        "step": 0,
                        **initial_metrics,
                    }
                )
            else:
                (
                    updates_completed,
                    metric_snapshots,
                    resume_lineage,
                    resumed_from,
                ) = restored

            latest_metrics = dict(metric_snapshots[-1])
            latest_metrics.pop("step", None)

            stop_now = self.stop_rule(
                step=updates_completed,
                metrics=latest_metrics,
                config=request.stopping_config,
            )
            if not isinstance(stop_now, bool):
                raise TeacherProtocolError("stop_rule must return bool")

            reason = "injected_stop_rule_before_update" if stop_now else ""

            while not stop_now and updates_completed < request.max_technical_updates:
                model.train()
                optimizer.zero_grad(set_to_none=True)

                outputs = model(train_inputs)
                if not torch.isfinite(outputs).all():
                    raise FloatingPointError("nonfinite_model_outputs")

                loss = self.loss_fn(
                    outputs=outputs,
                    targets=train_targets,
                    config=request.training_config,
                )
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                    raise TeacherProtocolError(
                        "loss_fn must return a scalar torch.Tensor"
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite_training_loss")

                loss.backward()

                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(
                        parameter.grad
                    ).all():
                        raise FloatingPointError("nonfinite_gradients")

                optimizer.step()

                for parameter in model.parameters():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError("nonfinite_parameters")

                if scheduler is not None:
                    step_method = getattr(scheduler, "step", None)
                    if step_method is None or not callable(step_method):
                        raise TeacherProtocolError(
                            "injected scheduler must expose callable step()"
                        )
                    step_method()

                updates_completed += 1

                latest_metrics = _evaluate(
                    model=model,
                    loss_fn=self.loss_fn,
                    loss_config=request.training_config,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    test_inputs=test_inputs,
                    test_targets=test_targets,
                )

                stop_now = self.stop_rule(
                    step=updates_completed,
                    metrics=latest_metrics,
                    config=request.stopping_config,
                )
                if not isinstance(stop_now, bool):
                    raise TeacherProtocolError("stop_rule must return bool")

                should_snapshot = (
                    updates_completed % request.checkpoint_interval == 0
                    or stop_now
                    or updates_completed == request.max_technical_updates
                    or interrupt_after_updates == updates_completed
                )

                if should_snapshot:
                    if (
                        not metric_snapshots
                        or metric_snapshots[-1]["step"] != updates_completed
                    ):
                        metric_snapshots.append(
                            {
                                "step": updates_completed,
                                **latest_metrics,
                            }
                        )

                    (
                        _checkpoint_path,
                        checkpoint_digest,
                        retained_paths,
                    ) = _save_rolling_checkpoint(
                        run_root=run_root,
                        request=request,
                        request_hashes=request_hashes,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        device=device,
                        updates_completed=updates_completed,
                        metric_snapshots=metric_snapshots,
                        resume_lineage=resume_lineage,
                    )
                    checkpoint_hashes = {
                        _relative_to_run(run_root, path): file_sha256(path)
                        for path in retained_paths
                    }

                if interrupt_after_updates == updates_completed:
                    payload = _artifact_payload(
                        request=request,
                        request_hashes=request_hashes,
                        environment=environment,
                        status="interrupted",
                        reason="forced_technical_interruption",
                        updates_completed=updates_completed,
                        metric_snapshots=metric_snapshots,
                        checkpoint_paths=retained_paths,
                        checkpoint_hashes=checkpoint_hashes,
                        resumed_from_checkpoint_sha256=resumed_from,
                        resume_lineage=resume_lineage,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    path, digest = _publish_artifact(
                        request=request,
                        payload=payload,
                    )
                    return TeacherRunResult(
                        status="interrupted",
                        updates_completed=updates_completed,
                        artifact_path=path,
                        artifact_sha256=digest,
                        checkpoint_paths=retained_paths,
                        resumed_from_checkpoint_sha256=resumed_from,
                    )

            if stop_now:
                reason = (
                    reason
                    if reason
                    else "injected_stop_rule"
                )
            else:
                reason = "max_technical_budget_reached"

            if (
                not metric_snapshots
                or metric_snapshots[-1]["step"] != updates_completed
            ):
                metric_snapshots.append(
                    {
                        "step": updates_completed,
                        **latest_metrics,
                    }
                )

            if updates_completed > 0:
                final_checkpoint = (
                    run_root
                    / "checkpoints"
                    / f"step_{updates_completed:08d}.pt"
                )
                if not final_checkpoint.exists():
                    (
                        _checkpoint_path,
                        _checkpoint_digest,
                        retained_paths,
                    ) = _save_rolling_checkpoint(
                        run_root=run_root,
                        request=request,
                        request_hashes=request_hashes,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        device=device,
                        updates_completed=updates_completed,
                        metric_snapshots=metric_snapshots,
                        resume_lineage=resume_lineage,
                    )

            retained_paths = _checkpoint_paths(run_root)
            checkpoint_hashes = {
                _relative_to_run(run_root, path): file_sha256(path)
                for path in retained_paths
            }

            payload = _artifact_payload(
                request=request,
                request_hashes=request_hashes,
                environment=environment,
                status="completed",
                reason=reason,
                updates_completed=updates_completed,
                metric_snapshots=metric_snapshots,
                checkpoint_paths=retained_paths,
                checkpoint_hashes=checkpoint_hashes,
                resumed_from_checkpoint_sha256=resumed_from,
                resume_lineage=resume_lineage,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            path, digest = _publish_artifact(
                request=request,
                payload=payload,
            )
            return TeacherRunResult(
                status="completed",
                updates_completed=updates_completed,
                artifact_path=path,
                artifact_sha256=digest,
                checkpoint_paths=retained_paths,
                resumed_from_checkpoint_sha256=resumed_from,
            )

        except FloatingPointError as exc:
            retained_paths = _checkpoint_paths(run_root)
            checkpoint_hashes = {
                _relative_to_run(run_root, path): file_sha256(path)
                for path in retained_paths
            }
            payload = _artifact_payload(
                request=request,
                request_hashes=request_hashes,
                environment=environment,
                status="numerical-failure",
                reason=str(exc),
                updates_completed=updates_completed,
                metric_snapshots=metric_snapshots,
                checkpoint_paths=retained_paths,
                checkpoint_hashes=checkpoint_hashes,
                resumed_from_checkpoint_sha256=resumed_from,
                resume_lineage=resume_lineage,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            path, digest = _publish_artifact(
                request=request,
                payload=payload,
            )
            return TeacherRunResult(
                status="numerical-failure",
                updates_completed=updates_completed,
                artifact_path=path,
                artifact_sha256=digest,
                checkpoint_paths=retained_paths,
                resumed_from_checkpoint_sha256=resumed_from,
            )

        except TeacherProtocolError:
            raise

        except Exception as exc:
            retained_paths = _checkpoint_paths(run_root)
            checkpoint_hashes = {
                _relative_to_run(run_root, path): file_sha256(path)
                for path in retained_paths
            }
            payload = _artifact_payload(
                request=request,
                request_hashes=request_hashes,
                environment=environment,
                status="failed",
                reason=f"{type(exc).__name__}:{exc}",
                updates_completed=updates_completed,
                metric_snapshots=metric_snapshots,
                checkpoint_paths=retained_paths,
                checkpoint_hashes=checkpoint_hashes,
                resumed_from_checkpoint_sha256=resumed_from,
                resume_lineage=resume_lineage,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            path, digest = _publish_artifact(
                request=request,
                payload=payload,
            )
            return TeacherRunResult(
                status="failed",
                updates_completed=updates_completed,
                artifact_path=path,
                artifact_sha256=digest,
                checkpoint_paths=retained_paths,
                resumed_from_checkpoint_sha256=resumed_from,
            )
