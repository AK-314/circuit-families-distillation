"""Frozen registries, validation, and lifecycle contracts for Stage 18."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from circuit_families.config import mapping_hash
from circuit_families.training import file_sha256

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


CONFIGURATION_PATH = Path("configs/stage18_scaling.json")
PRIMARY_MAIN_SEEDS = (0, 1, 2, 3, 4)
PILOT_SEED = 1
FRESH_TRAINING_SEEDS = (0, 2, 3, 4)
RESERVE_SEEDS = (5, 6, 7, 8, 9)
CHECKPOINT_STEPS = (200, 3400, 7450, 8150, 8500, 8650, 9050)
FIDELITY_GRID = (
    Fraction(4, 5),
    Fraction(17, 20),
    Fraction(9, 10),
    Fraction(19, 20),
    Fraction(39, 40),
    Fraction(99, 100),
)
FIDELITY_DISPLAYS = ("0.800", "0.850", "0.900", "0.950", "0.975", "0.990")
DISTINCTNESS_GRID = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
DISTINCTNESS_DISPLAYS = ("0.25", "0.50", "0.75")
PRIMARY_CELL = (Fraction(99, 100), Fraction(1, 2))
PRODUCTION_WORKERS = 12
THREADS_PER_WORKER = 1
COMPUTE_ONLY_CEILING = 14
TOTAL_CELL_COUNT = 630
FRESH_CELL_COUNT = 612
REFERENCE_CELL_COUNT = 18
STAGE17_RUN_ID = "stage17-sensitivity-s1-7801e7938531"
STAGE17_MANIFEST = "manifests/stage17_sensitivity_stage17-sensitivity-s1-7801e7938531.json"
STAGE17_ARCHIVE = "results/archives/stage17-sensitivity-s1-7801e7938531.tar.gz"
PERMITTED_USER_FILE = "stage17_inspection.md"
PERMITTED_USER_FILE_BASELINE_SHA256 = (
    "fc8cd5cc791715bb558cb47472ffc4b821254723591e458e894318659dc86533"
)

ExecutionMode = Literal["fresh_execution", "reference_existing_result"]


@dataclass(frozen=True)
class FrozenStage18Configuration:
    path: Path
    sha256: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class Stage18Cell:
    global_cell_index: int
    model_seed: int
    checkpoint_index: int
    checkpoint_step: int
    fidelity: Fraction
    fidelity_display: str
    distinctness: Fraction
    distinctness_display: str
    family_search_execution_mode: ExecutionMode
    transfer_execution_mode: ExecutionMode
    source_stage: int | None
    source_run_id: str | None
    source_manifest: str | None
    source_cell_id: str | None
    fresh_cell_index: int | None
    worker_id: str | None
    output_root: str
    cell_id: str

    @property
    def is_primary(self) -> bool:
        return (self.fidelity, self.distinctness) == PRIMARY_CELL

    def to_record(self) -> dict[str, object]:
        return {
            "global_cell_index": self.global_cell_index,
            "cell_id": self.cell_id,
            "model_seed": self.model_seed,
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_sha256": None,
            "fidelity_numerator": self.fidelity.numerator,
            "fidelity_denominator": self.fidelity.denominator,
            "displayed_fidelity": self.fidelity_display,
            "distinctness_numerator": self.distinctness.numerator,
            "distinctness_denominator": self.distinctness.denominator,
            "displayed_cutoff": self.distinctness_display,
            "primary_cell": self.is_primary,
            "family_search_execution_mode": self.family_search_execution_mode,
            "transfer_execution_mode": self.transfer_execution_mode,
            "source_stage": self.source_stage,
            "source_run_id": self.source_run_id,
            "source_manifest": self.source_manifest,
            "source_cell_id": self.source_cell_id,
            "fresh_cell_index": self.fresh_cell_index,
            "worker_id": self.worker_id,
            "output_root": self.output_root,
            "status": "not_executed",
            "stopping_reason": None,
            "family_size": None,
            "transfer_group_status": "not_evaluated",
        }


@dataclass(frozen=True)
class WorkerShard:
    worker_id: str
    cells: tuple[Stage18Cell, ...]

    @property
    def shard_sha256(self) -> str:
        payload = {
            "worker_id": self.worker_id,
            "cell_ids": [cell.cell_id for cell in self.cells],
            "thread_settings": {"intra_op": 1, "inter_op": 1},
        }
        return mapping_hash(payload)

    def to_record(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "cell_count": len(self.cells),
            "ordered_cell_ids": json.dumps([cell.cell_id for cell in self.cells]),
            "shard_sha256": self.shard_sha256,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "omp_num_threads": "1",
            "veclib_maximum_threads": "1",
            "output_root": f"results/raw/{{stage18_run_id}}/workers/{self.worker_id}",
        }


@dataclass(frozen=True)
class Stage18InputValidation:
    repository: Path
    configuration: FrozenStage18Configuration
    implementation_commit: str
    repository_clean: bool
    permitted_user_file_sha256: str
    cells: tuple[Stage18Cell, ...]
    shards: tuple[WorkerShard, ...]
    source_hashes: dict[str, str]


def _load_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return payload


def load_stage18_configuration(
    repository_root: str | Path,
    path: str | Path = CONFIGURATION_PATH,
) -> FrozenStage18Configuration:
    repository = Path(repository_root).resolve()
    candidate = Path(path)
    config_path = candidate if candidate.is_absolute() else repository / candidate
    payload = _load_object(config_path, "Stage 18 configuration")
    expected = {
        "main_seeds": list(PRIMARY_MAIN_SEEDS),
        "pilot_seed": PILOT_SEED,
        "fresh_training_seeds": list(FRESH_TRAINING_SEEDS),
        "reserve_seeds": list(RESERVE_SEEDS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Stage 18 {key} differs from the frozen value.")
    concurrency = payload.get("concurrency", {})
    if concurrency.get("production_workers") != PRODUCTION_WORKERS:
        raise ValueError("Stage 18 production worker count must be exactly 12.")
    if concurrency.get("intra_op_threads_per_worker") != 1:
        raise ValueError("Stage 18 intra-op threads per worker must be one.")
    if concurrency.get("inter_op_threads_per_worker") != 1:
        raise ValueError("Stage 18 inter-op threads per worker must be one.")
    if concurrency.get("compute_only_ceiling_workers") != COMPUTE_ONLY_CEILING:
        raise ValueError("Stage 18 compute-only ceiling must remain 14.")
    if payload.get("additional_controls", {}).get("additional_random_label_seed_count") != 0:
        raise ValueError("Stage 18 additional random-label seed count must be zero.")
    return FrozenStage18Configuration(
        path=config_path,
        sha256=file_sha256(config_path),
        payload=payload,
    )


def deterministic_cell_id(
    model_seed: int,
    checkpoint_step: int,
    fidelity: Fraction,
    distinctness: Fraction,
) -> str:
    return (
        f"s{model_seed}-step{checkpoint_step}-"
        f"f{fidelity.numerator}of{fidelity.denominator}-"
        f"d{distinctness.numerator}of{distinctness.denominator}"
    )


def build_stage18_registry(
    analysis_seeds: Sequence[int] = PRIMARY_MAIN_SEEDS,
) -> tuple[Stage18Cell, ...]:
    selected_seeds = tuple(analysis_seeds)
    if len(selected_seeds) != 5 or len(set(selected_seeds)) != 5:
        raise ValueError("Stage 18 analysis requires exactly five unique model seeds.")
    if PILOT_SEED not in selected_seeds:
        raise ValueError("Stage 18 analysis seeds must retain pilot seed 1.")
    if any(seed not in PRIMARY_MAIN_SEEDS + RESERVE_SEEDS for seed in selected_seeds):
        raise ValueError("Stage 18 analysis seed is outside the frozen primary/reserve lists.")
    cells: list[Stage18Cell] = []
    fresh_index = 0
    for model_seed in selected_seeds:
        for checkpoint_index, checkpoint_step in enumerate(CHECKPOINT_STEPS, start=1):
            for fidelity, fidelity_display in zip(FIDELITY_GRID, FIDELITY_DISPLAYS, strict=True):
                for distinctness, distinctness_display in zip(
                    DISTINCTNESS_GRID, DISTINCTNESS_DISPLAYS, strict=True
                ):
                    reference = model_seed == 1 and checkpoint_step == 9050
                    cell_id = deterministic_cell_id(
                        model_seed, checkpoint_step, fidelity, distinctness
                    )
                    worker_id = None
                    assigned_fresh_index = None
                    if not reference:
                        assigned_fresh_index = fresh_index
                        worker_id = f"worker_{fresh_index % PRODUCTION_WORKERS:02d}"
                        fresh_index += 1
                    cells.append(
                        Stage18Cell(
                            global_cell_index=len(cells) + 1,
                            model_seed=model_seed,
                            checkpoint_index=checkpoint_index,
                            checkpoint_step=checkpoint_step,
                            fidelity=fidelity,
                            fidelity_display=fidelity_display,
                            distinctness=distinctness,
                            distinctness_display=distinctness_display,
                            family_search_execution_mode=(
                                "reference_existing_result" if reference else "fresh_execution"
                            ),
                            transfer_execution_mode=(
                                "reference_existing_result" if reference else "fresh_execution"
                            ),
                            source_stage=17 if reference else None,
                            source_run_id=STAGE17_RUN_ID if reference else None,
                            source_manifest=STAGE17_MANIFEST if reference else None,
                            source_cell_id=cell_id if reference else None,
                            fresh_cell_index=assigned_fresh_index,
                            worker_id=worker_id,
                            output_root=(
                                f"results/raw/{{stage18_run_id}}/references/{cell_id}"
                                if reference
                                else (
                                    "results/raw/{stage18_run_id}/workers/"
                                    f"{worker_id}/cells/{cell_id}"
                                )
                            ),
                            cell_id=cell_id,
                        )
                    )
    registry = tuple(cells)
    validate_stage18_registry(registry)
    return registry


def validate_stage18_registry(cells: Sequence[Stage18Cell]) -> None:
    if len(cells) != TOTAL_CELL_COUNT:
        raise ValueError(f"Stage 18 registry must contain {TOTAL_CELL_COUNT} cells.")
    if tuple(cell.global_cell_index for cell in cells) != tuple(range(1, TOTAL_CELL_COUNT + 1)):
        raise ValueError("Stage 18 global cell indices must be consecutive and one-based.")
    if len({cell.cell_id for cell in cells}) != TOTAL_CELL_COUNT:
        raise ValueError("Stage 18 cell identifiers must be unique.")
    references = [
        cell for cell in cells if cell.family_search_execution_mode == "reference_existing_result"
    ]
    fresh = [cell for cell in cells if cell.family_search_execution_mode == "fresh_execution"]
    if len(references) != REFERENCE_CELL_COUNT or len(fresh) != FRESH_CELL_COUNT:
        raise ValueError("Stage 18 fresh/reference execution identity failed.")
    if any(cell.model_seed != 1 or cell.checkpoint_step != 9050 for cell in references):
        raise ValueError("Only seed 1 step 9050 may reference Stage 17.")
    if tuple(cell.fresh_cell_index for cell in fresh) != tuple(range(FRESH_CELL_COUNT)):
        raise ValueError("Fresh-cell indices must be consecutive and zero-based.")
    if any(cell.worker_id != f"worker_{cell.fresh_cell_index % 12:02d}" for cell in fresh):
        raise ValueError("Stage 18 modulo-12 worker assignment mismatch.")


def build_worker_shards(cells: Sequence[Stage18Cell] | None = None) -> tuple[WorkerShard, ...]:
    registry = build_stage18_registry() if cells is None else tuple(cells)
    validate_stage18_registry(registry)
    shards = tuple(
        WorkerShard(
            worker_id=f"worker_{index:02d}",
            cells=tuple(cell for cell in registry if cell.worker_id == f"worker_{index:02d}"),
        )
        for index in range(PRODUCTION_WORKERS)
    )
    if len(shards) != 12 or any(len(shard.cells) != 51 for shard in shards):
        raise ValueError("Stage 18 must contain twelve disjoint 51-cell shards.")
    assigned = [cell.cell_id for shard in shards for cell in shard.cells]
    if len(assigned) != FRESH_CELL_COUNT or len(set(assigned)) != FRESH_CELL_COUNT:
        raise ValueError("Stage 18 worker shards are not disjoint and complete.")
    return shards


def build_main_seed_registry() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "primary_seed": seed,
            "training_status": "complete" if seed == PILOT_SEED else "not_started",
            "training_execution": (
                "reference_existing_result" if seed == PILOT_SEED else "fresh_execution"
            ),
            "training_run_id": (
                "modular-addition-training-s1-5f1bc9dee7ab" if seed == PILOT_SEED else None
            ),
            "training_eligibility": ("eligible" if seed == PILOT_SEED else "not_evaluated"),
            "grokking_classification": (
                "complete_grokking_seed" if seed == PILOT_SEED else "not_evaluated"
            ),
            "final_training_step": 40000 if seed == PILOT_SEED else None,
            "replacement_status": "not_replaced",
            "replacement_reason": None,
            "selected_checkpoint_status": ("complete" if seed == PILOT_SEED else "not_evaluated"),
            "complete_analysis_status": "not_started",
        }
        for seed in PRIMARY_MAIN_SEEDS
    )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_repository_cleanliness(
    repository: Path,
    *,
    allow_permitted_user_file_absent: bool = False,
) -> tuple[str, bool, str]:
    if _git(repository, "branch", "--show-current") != "main":
        raise ValueError("Stage 18 requires branch main.")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    entries = tuple(line for line in status.splitlines() if line)
    allowed = (f"?? {PERMITTED_USER_FILE}",)
    if entries == () and allow_permitted_user_file_absent:
        return _git(repository, "rev-parse", "HEAD"), True, "absent_in_reproduction"
    if entries != allowed:
        raise ValueError(
            "Tracked repository must be clean and the sole untracked path must be "
            f"{PERMITTED_USER_FILE!r}; received {entries!r}."
        )
    user_path = repository / PERMITTED_USER_FILE
    digest = file_sha256(user_path)
    if digest != PERMITTED_USER_FILE_BASELINE_SHA256:
        raise ValueError("Permitted user-file SHA-256 changed.")
    return _git(repository, "rev-parse", "HEAD"), True, digest


def validate_stage18_inputs(
    repository_root: str | Path,
    *,
    expected_implementation_commit: str | None = None,
    reproduction_mode: bool = False,
) -> Stage18InputValidation:
    repository = Path(repository_root).resolve()
    configuration = load_stage18_configuration(repository)
    commit, clean, user_hash = validate_repository_cleanliness(
        repository,
        allow_permitted_user_file_absent=reproduction_mode,
    )
    if expected_implementation_commit is not None and commit != expected_implementation_commit:
        raise ValueError("Current commit does not match expected Stage 18 implementation commit.")
    references = configuration.payload["references"]
    pinned = {
        "freeze_manifest": references["freeze_manifest_sha256"],
        "freeze_note": references["freeze_note_sha256"],
        "benchmark_summary": references["benchmark_summary_sha256"],
        "stage17_manifest": references["stage17_manifest_sha256"],
        "stage17_archive": references["stage17_archive_sha256"],
    }
    source_hashes: dict[str, str] = {}
    for key, expected in pinned.items():
        path = repository / references[key]
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Pinned Stage 18 source hash mismatch for {key}.")
        source_hashes[key] = actual
    for key in ("control_seed_freeze", "stage16_manifest", "stage15_manifest"):
        path = repository / references[key]
        source_hashes[key] = file_sha256(path)
    if source_hashes["control_seed_freeze"] != references["control_seed_freeze_sha256"]:
        raise ValueError("Pinned additional-control seed-count freeze hash mismatch.")
    stage15 = _load_object(repository / references["stage15_manifest"], "Stage 15 manifest")
    if stage15.get("status") != "unavailable":
        raise ValueError("Stage 15 must remain unavailable.")
    stage17 = _load_object(repository / references["stage17_manifest"], "Stage 17 manifest")
    registry = stage17.get("registry")
    if not isinstance(registry, list) or len(registry) != 18:
        raise ValueError("Stage 17 reference registry must contain exactly 18 cells.")
    if any(row.get("model_seed") != 1 or row.get("checkpoint_step") != 9050 for row in registry):
        raise ValueError("Stage 17 reference cells must all be seed 1 step 9050.")
    cells = build_stage18_registry()
    shards = build_worker_shards(cells)
    return Stage18InputValidation(
        repository=repository,
        configuration=configuration,
        implementation_commit=commit,
        repository_clean=clean,
        permitted_user_file_sha256=user_hash,
        cells=cells,
        shards=shards,
        source_hashes=source_hashes,
    )


def stable_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    if not rows:
        raise ValueError("Cannot infer fields for an empty Stage 18 table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_preexecution_registries(repository_root: str | Path) -> tuple[Path, ...]:
    repository = Path(repository_root).resolve()
    cells = build_stage18_registry()
    shards = build_worker_shards(cells)
    seeds = build_main_seed_registry()
    outputs = (
        write_csv(
            repository / "results/tables/stage18_main_seed_registry_pre_execution.csv",
            seeds,
        ),
        write_csv(
            repository / "results/tables/stage18_cell_registry_pre_execution.csv",
            tuple(cell.to_record() for cell in cells),
        ),
        write_csv(
            repository / "results/tables/stage18_worker_shards_pre_execution.csv",
            tuple(shard.to_record() for shard in shards),
        ),
    )
    shard_root = repository / "manifests/stage18_worker_shards"
    for shard in shards:
        outputs += (
            stable_json(
                shard_root / f"{shard.worker_id}.json",
                {
                    "schema_version": 1,
                    "worker_id": shard.worker_id,
                    "cell_count": len(shard.cells),
                    "ordered_cell_ids": [cell.cell_id for cell in shard.cells],
                    "shard_sha256": shard.shard_sha256,
                    "thread_settings": {
                        "OMP_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1",
                        "torch_intra_op_threads": 1,
                        "torch_inter_op_threads": 1,
                    },
                    "output_root": (f"results/raw/{{stage18_run_id}}/workers/{shard.worker_id}"),
                },
            ),
        )
    return outputs


def deterministic_stage18_run_id(config_sha256: str, implementation_commit: str) -> str:
    digest = hashlib.sha256(
        f"stage18|{config_sha256}|{implementation_commit}".encode("ascii")
    ).hexdigest()[:12]
    return f"stage18-scaling-{digest}"
