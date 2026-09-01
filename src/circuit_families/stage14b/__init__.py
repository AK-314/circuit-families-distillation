"""Provider-neutral Stage 14-B qualification and rehearsal package."""

from .environment import capture_environment, verify_environment
from .feasibility import solve_feasibility
from .inputs import plan_input_bundle, stage_input_bundle, verify_input_root
from .monitoring import (
    FINAL_WINDOW_SECONDS,
    gate_15_1,
    gate_15_2,
    gate_15_3,
    model_storage,
    monitor_campaign,
)
from .records import Stage14BError, canonical_json_bytes, canonical_sha256
from .rehearsal import compare_rehearsals, reduced_rehearsal_manifest, run_rehearsal
from .resources import inventory_resource_pool, qualification_policy, qualify_backend
from .scheduler import (
    LocalTechnicalScheduler,
    SlurmClassAdapter,
    SlurmConfig,
    VerifiedLaunchAuthorization,
    WorkerCapabilities,
    build_mac_shard_bundle,
    deterministic_placement_plan,
    verify_launch_authorization,
    verify_mac_result_bundle,
)

__all__ = [
    "FINAL_WINDOW_SECONDS",
    "LocalTechnicalScheduler",
    "SlurmClassAdapter",
    "SlurmConfig",
    "Stage14BError",
    "VerifiedLaunchAuthorization",
    "WorkerCapabilities",
    "build_mac_shard_bundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "capture_environment",
    "compare_rehearsals",
    "deterministic_placement_plan",
    "gate_15_1",
    "gate_15_2",
    "gate_15_3",
    "inventory_resource_pool",
    "model_storage",
    "monitor_campaign",
    "plan_input_bundle",
    "qualification_policy",
    "qualify_backend",
    "reduced_rehearsal_manifest",
    "run_rehearsal",
    "solve_feasibility",
    "stage_input_bundle",
    "verify_environment",
    "verify_input_root",
    "verify_launch_authorization",
    "verify_mac_result_bundle",
]
