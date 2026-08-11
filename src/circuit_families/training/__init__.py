"""Training, evaluation, device, and checkpoint utilities."""

from circuit_families.training.checkpoints import (
    ResumeState,
    SavedCheckpoint,
    canonical_state_hash,
    checkpoint_filename,
    checkpoint_path,
    file_sha256,
    load_checkpoint_payload,
    reload_and_reevaluate,
    restore_checkpoint,
    save_checkpoint,
)
from circuit_families.training.data import (
    TrainingData,
    load_training_data,
)
from circuit_families.training.device import (
    device_record,
    resolve_device,
)
from circuit_families.training.logging import (
    append_jsonl,
    read_jsonl,
)
from circuit_families.training.metrics import (
    OUTPUT_CLASS_COUNT,
    classification_accuracy,
    cross_entropy_loss,
    evaluate_model,
    final_position_logits,
    gradient_norm,
    parameter_norm,
)
from circuit_families.training.run import (
    ExecutionPlan,
    TrainingRunResult,
    build_execution_plan,
    run_training,
)
from circuit_families.training.trainer import (
    build_optimizer,
    train_full_batch_step,
)

__all__ = [
    "ExecutionPlan",
    "OUTPUT_CLASS_COUNT",
    "ResumeState",
    "SavedCheckpoint",
    "TrainingData",
    "TrainingRunResult",
    "append_jsonl",
    "build_execution_plan",
    "build_optimizer",
    "canonical_state_hash",
    "checkpoint_filename",
    "checkpoint_path",
    "classification_accuracy",
    "cross_entropy_loss",
    "device_record",
    "evaluate_model",
    "file_sha256",
    "final_position_logits",
    "gradient_norm",
    "load_checkpoint_payload",
    "load_training_data",
    "parameter_norm",
    "read_jsonl",
    "reload_and_reevaluate",
    "resolve_device",
    "restore_checkpoint",
    "run_training",
    "save_checkpoint",
    "train_full_batch_step",
]
