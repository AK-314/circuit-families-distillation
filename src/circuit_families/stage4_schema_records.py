"""Stage 4 record-specific validators for Part M schema types."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from circuit_families.stage4_condition_identity import (
    VERSION_REFERENCE_RE,
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    parse_condition_id,
)
from circuit_families.stage4_schema_common import (
    CommonSchemaContract,
    Stage4SchemaError,
    validate_common_envelope,
)
from circuit_families.stage4_seed_derivation import (
    SeedDerivationError,
    SeedEvidence,
    verify_seed_evidence,
)

STAGE3_REGISTRY_PATH = "followup/manifests/stage3_teacher_registry_v1.json"
STAGE3_REGISTRY_SHA256 = (
    "36656fe848f7cb980cd9178f1d48cbbee74bb410135746e079cae11440b6ff0d"
)
STAGE3_REGISTRY_NAMESPACE = (
    "circuit-families-distillation/stage3-teacher-registry"
)
STAGE3_RECORD_SCHEMA_VERSION = "1"
FULL_DATASET_EXAMPLE_COUNT = 12769

COMPONENT_COUNT = 516
COMPONENT_BASIS_STATUS = "reused_predecessor_definition"

MASKS_SOURCE = {
    "path": "src/circuit_families/interpretability/masks.py",
    "sha256": "2e3ef77f82547c8505d5d6ebc52d6d7e17e426c0fe743bcaa7190d45849f8bc5",
}
COMPONENT_ABLATION_SOURCE = {
    "path": "src/circuit_families/interpretability/component_ablation.py",
    "sha256": "2a75d55462fc8346c4eba8cfe3f5cf0c400effdb91d81d26fa39899af92993fd",
}
STAGE8_MASKING_MANIFEST = {
    "path": "manifests/stage8_masking_s1-5f1bc9dee7ab.json",
    "sha256": "ed6aca8d20d43ea7618936b962c8e865859c21894e5d809580106ffe73a8d4e5",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_M_RECORD_TYPES = frozenset(
    {
        "teacher_reference",
        "teacher_output_cache",
        "student_attempt",
        "student_eligibility",
        "sealed_dense_model",
    }
)


def _error(message: str) -> None:
    raise Stage4SchemaError(message)


def _exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str] | None = None,
    *,
    label: str,
) -> None:
    optional = optional or set()
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)

    if missing or extra:
        _error(
            f"{label} keys mismatch: "
            f"missing={missing!r} extra={extra!r}"
        )


def _validate_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _error(f"{label} must be 64 lowercase hexadecimal characters")


def _validate_version_ref(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not VERSION_REFERENCE_RE.fullmatch(value):
        _error(f"{label} must match version-reference grammar")


def _validate_uint(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _error(f"{label} must be a non-negative integer")


def _validate_portable_artifact(
    value: Any,
    *,
    label: str,
    required_storage_class: str | None = None,
) -> None:
    if not isinstance(value, Mapping):
        _error(f"{label} must be an object")

    _exact_keys(
        value,
        {"path", "sha256", "storage_class"},
        label=label,
    )

    path = value["path"]

    if not isinstance(path, str) or not path:
        _error(f"{label}.path must be a non-empty relative POSIX path")

    if (
        path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        _error(f"{label}.path must be a portable relative POSIX path")

    _validate_sha256(value["sha256"], label=f"{label}.sha256")

    storage_class = value["storage_class"]

    allowed = {
        "external_checkpoint",
        "external_large_object",
        "external_log",
    }

    if storage_class not in allowed:
        _error(f"{label}.storage_class is invalid: {storage_class!r}")

    if (
        required_storage_class is not None
        and storage_class != required_storage_class
    ):
        _error(
            f"{label}.storage_class must be "
            f"{required_storage_class!r}"
        )


def _reference_identity(
    reference: Any,
    *,
    expected_record_type: str,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
    require_hash: bool,
):
    if not isinstance(reference, Mapping):
        _error("record reference must be an object")

    required = {"record_type", "schema_version", "condition_id"}
    optional = {"record_sha256"}

    _exact_keys(
        reference,
        required,
        optional,
        label="record reference",
    )

    if reference["record_type"] != expected_record_type:
        _error(
            f"record reference must target {expected_record_type!r}"
        )

    expected_schema = contract.schema_versions[expected_record_type]

    if reference["schema_version"] != expected_schema:
        _error(
            "record reference schema_version does not match "
            f"{expected_record_type!r}"
        )

    if require_hash and "record_sha256" not in reference:
        _error("record reference requires record_sha256")

    if "record_sha256" in reference:
        _validate_sha256(
            reference["record_sha256"],
            label="record reference record_sha256",
        )

    try:
        identity = parse_condition_id(
            reference["condition_id"],
            stage3,
        )
    except ConditionIdentityError as exc:
        _error(f"invalid referenced condition_id: {exc}")

    expected_depth = contract.record_type_required_depths[
        expected_record_type
    ]

    if identity.depth != expected_depth:
        _error(
            "record reference condition depth mismatch: "
            f"actual={identity.depth} expected={expected_depth}"
        )

    return identity


def _same_teacher_phase(left, right) -> None:
    if (
        left.teacher_seed != right.teacher_seed
        or left.phase != right.phase
    ):
        _error("referenced record does not share teacher seed/phase")


def _same_condition(left, right) -> None:
    if (
        left.teacher_seed != right.teacher_seed
        or left.phase != right.phase
        or left.distillation_condition != right.distillation_condition
    ):
        _error("referenced record does not share distillation condition")


def _same_student_identity(left, right) -> None:
    _same_condition(left, right)

    if left.student_initialization != right.student_initialization:
        _error("referenced record does not share student initialization")


def _find_stage3_record(
    *,
    teacher_seed: int,
    phase: str,
    registry: Mapping[str, Any],
):
    matches = [
        item
        for item in registry["records"]
        if (
            item["teacher_seed"] == teacher_seed
            and item["phase_label"] == phase
        )
    ]

    if len(matches) != 1:
        _error(
            "Stage 3 registry must contain exactly one matching "
            "teacher-phase record"
        )

    return matches[0]


def _validate_teacher_reference(
    record: Mapping[str, Any],
    identity,
    *,
    stage3_registry: Mapping[str, Any],
    stage3_registry_sha256: str,
) -> None:
    if identity.distillation_condition != "direct_teacher":
        _error("teacher_reference requires direct_teacher condition")

    if record["provenance"]["producer_lane"] != "lane_a":
        _error("teacher_reference producer_lane must be lane_a")

    payload = record["payload"]

    if not isinstance(payload, Mapping):
        _error("teacher_reference payload must be an object")

    _exact_keys(
        payload,
        {
            "stage3_registry_path",
            "stage3_registry_sha256",
            "stage3_registry_namespace",
            "stage3_record_schema_version",
            "canonical_run_id",
            "checkpoint",
            "training_step",
        },
        label="teacher_reference payload",
    )

    if payload["stage3_registry_path"] != STAGE3_REGISTRY_PATH:
        _error("teacher_reference stage3_registry_path mismatch")

    if stage3_registry_sha256 != STAGE3_REGISTRY_SHA256:
        _error("provided Stage 3 registry SHA-256 is not frozen authority")

    if payload["stage3_registry_sha256"] != stage3_registry_sha256:
        _error("teacher_reference Stage 3 registry SHA-256 mismatch")

    if payload["stage3_registry_namespace"] != STAGE3_REGISTRY_NAMESPACE:
        _error("teacher_reference Stage 3 namespace mismatch")

    if payload["stage3_record_schema_version"] != STAGE3_RECORD_SCHEMA_VERSION:
        _error("teacher_reference Stage 3 record schema version mismatch")

    source = _find_stage3_record(
        teacher_seed=identity.teacher_seed,
        phase=identity.phase,
        registry=stage3_registry,
    )

    if source["availability_status"] != "selected":
        _error("teacher_reference requires selected Stage 3 cell")

    if payload["canonical_run_id"] != source["canonical_run_id"]:
        _error("teacher_reference canonical_run_id mismatch")

    _validate_uint(
        payload["training_step"],
        label="teacher_reference training_step",
    )

    if payload["training_step"] != source["training_step"]:
        _error("teacher_reference training_step mismatch")

    _validate_portable_artifact(
        payload["checkpoint"],
        label="teacher_reference checkpoint",
        required_storage_class="external_checkpoint",
    )

    expected_checkpoint = {
        "path": source["checkpoint_path"],
        "sha256": source["checkpoint_sha256"],
        "storage_class": "external_checkpoint",
    }

    if dict(payload["checkpoint"]) != expected_checkpoint:
        _error("teacher_reference checkpoint does not match Stage 3 record")


def _validate_teacher_output_cache(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if identity.distillation_condition not in {"hard_target", "soft_target"}:
        _error(
            "teacher_output_cache requires hard_target or soft_target"
        )

    if record["provenance"]["producer_lane"] != "lane_b":
        _error("teacher_output_cache producer_lane must be lane_b")

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "teacher_reference",
            "cache_kind",
            "example_ordering_ref",
            "example_count",
            "artifact",
        },
        label="teacher_output_cache payload",
    )

    teacher_identity = _reference_identity(
        payload["teacher_reference"],
        expected_record_type="teacher_reference",
        contract=contract,
        stage3=stage3,
        require_hash=True,
    )

    if teacher_identity.distillation_condition != "direct_teacher":
        _error("cache teacher_reference must target direct_teacher")

    _same_teacher_phase(identity, teacher_identity)

    expected_kind = (
        "teacher_argmax"
        if identity.distillation_condition == "hard_target"
        else "teacher_logits"
    )

    if payload["cache_kind"] != expected_kind:
        _error(
            f"{identity.distillation_condition} cache_kind must be "
            f"{expected_kind!r}"
        )

    _validate_version_ref(
        payload["example_ordering_ref"],
        label="example_ordering_ref",
    )

    if payload["example_count"] != FULL_DATASET_EXAMPLE_COUNT:
        _error(
            "teacher_output_cache example_count must equal frozen "
            "dataset size 12769"
        )

    _validate_portable_artifact(
        payload["artifact"],
        label="teacher_output_cache artifact",
        required_storage_class="external_large_object",
    )


def _seed_evidence(value: Any, *, label: str) -> SeedEvidence:
    if not isinstance(value, Mapping):
        _error(f"{label} must be an object")

    required = {
        "seed_derivation_version",
        "seed_material",
        "digest_sha256",
        "selected_bytes_hex",
        "seed_value",
    }

    _exact_keys(value, required, label=label)

    try:
        return SeedEvidence(**dict(value))
    except TypeError as exc:
        _error(f"{label} is malformed: {exc}")


def _validate_student_attempt(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if identity.distillation_condition not in {"hard_target", "soft_target"}:
        _error("student_attempt requires hard_target or soft_target")

    if record["provenance"]["producer_lane"] != "lane_b":
        _error("student_attempt producer_lane must be lane_b")

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "target_cache",
            "attempt_index",
            "retry_index",
            "attempt_outcome",
            "student_architecture_ref",
            "replication_policy_ref",
            "training_config_ref",
            "training_seed",
            "tie_breaking_seed",
            "training_log",
        },
        {"model_checkpoint", "failure_reason"},
        label="student_attempt payload",
    )

    cache_identity = _reference_identity(
        payload["target_cache"],
        expected_record_type="teacher_output_cache",
        contract=contract,
        stage3=stage3,
        require_hash=True,
    )

    _same_condition(identity, cache_identity)

    _validate_uint(payload["attempt_index"], label="attempt_index")
    _validate_uint(payload["retry_index"], label="retry_index")

    if payload["attempt_outcome"] not in {"succeeded", "failed"}:
        _error("student_attempt attempt_outcome is invalid")

    for field in (
        "student_architecture_ref",
        "replication_policy_ref",
        "training_config_ref",
    ):
        _validate_version_ref(payload[field], label=field)

    _validate_portable_artifact(
        payload["training_log"],
        label="student_attempt training_log",
        required_storage_class="external_log",
    )

    if payload["attempt_outcome"] == "succeeded":
        if "model_checkpoint" not in payload:
            _error("succeeded student_attempt requires model_checkpoint")
        if "failure_reason" in payload:
            _error("succeeded student_attempt cannot include failure_reason")

        _validate_portable_artifact(
            payload["model_checkpoint"],
            label="student_attempt model_checkpoint",
            required_storage_class="external_checkpoint",
        )

    else:
        if "failure_reason" not in payload:
            _error("failed student_attempt requires failure_reason")
        if "model_checkpoint" in payload:
            _error("failed student_attempt cannot include model_checkpoint")
        if (
            not isinstance(payload["failure_reason"], str)
            or not payload["failure_reason"].strip()
        ):
            _error("failure_reason must be a non-empty string")

    training = _seed_evidence(
        payload["training_seed"],
        label="training_seed",
    )
    tie = _seed_evidence(
        payload["tie_breaking_seed"],
        label="tie_breaking_seed",
    )

    for expected_purpose, evidence in (
        ("training", training),
        ("tie_breaking", tie),
    ):
        try:
            inputs = verify_seed_evidence(evidence, stage3)
        except SeedDerivationError as exc:
            _error(f"invalid {expected_purpose} seed evidence: {exc}")

        if inputs.condition_id != record["condition_id"]:
            _error(
                f"{expected_purpose} seed condition_id mismatch"
            )
        if inputs.purpose != expected_purpose:
            _error(
                f"{expected_purpose} seed purpose mismatch"
            )
        if inputs.attempt_index != payload["attempt_index"]:
            _error(
                f"{expected_purpose} seed attempt_index mismatch"
            )
        if inputs.retry_index != payload["retry_index"]:
            _error(
                f"{expected_purpose} seed retry_index mismatch"
            )


def _validate_student_eligibility(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if identity.distillation_condition not in {"hard_target", "soft_target"}:
        _error("student_eligibility requires hard_target or soft_target")

    if record["provenance"]["producer_lane"] != "lane_b":
        _error("student_eligibility producer_lane must be lane_b")

    payload = record["payload"]

    common_required = {
        "attempt_reference",
        "attempt_index",
        "retry_index",
        "eligibility_status",
        "criterion",
    }

    if identity.distillation_condition == "hard_target":
        required = common_required | {
            "evaluation_example_count",
            "teacher_argmax_agreement_count",
        }
        optional: set[str] = set()
    else:
        required = common_required | {"soft_policy_ref"}
        optional = set()

    _exact_keys(
        payload,
        required,
        optional,
        label="student_eligibility payload",
    )

    attempt_identity = _reference_identity(
        payload["attempt_reference"],
        expected_record_type="student_attempt",
        contract=contract,
        stage3=stage3,
        require_hash=True,
    )

    _same_student_identity(identity, attempt_identity)

    _validate_uint(payload["attempt_index"], label="attempt_index")
    _validate_uint(payload["retry_index"], label="retry_index")

    if payload["eligibility_status"] not in {
        "passed",
        "failed",
        "pending_policy",
    }:
        _error("student_eligibility eligibility_status is invalid")

    if identity.distillation_condition == "hard_target":
        if payload["criterion"] != "exact_teacher_argmax_agreement":
            _error(
                "hard_target eligibility criterion must be "
                "'exact_teacher_argmax_agreement'"
            )

        if payload["eligibility_status"] == "pending_policy":
            _error("hard_target eligibility cannot be pending_policy")

        if payload["evaluation_example_count"] != FULL_DATASET_EXAMPLE_COUNT:
            _error(
                "hard_target eligibility evaluation_example_count "
                "must be 12769"
            )

        agreement = payload["teacher_argmax_agreement_count"]

        _validate_uint(
            agreement,
            label="teacher_argmax_agreement_count",
        )

        if agreement > FULL_DATASET_EXAMPLE_COUNT:
            _error(
                "teacher_argmax_agreement_count cannot exceed 12769"
            )

        expected_status = (
            "passed"
            if agreement == FULL_DATASET_EXAMPLE_COUNT
            else "failed"
        )

        if payload["eligibility_status"] != expected_status:
            _error(
                "hard_target eligibility status must equal exact "
                "12769/12769 teacher-argmax criterion"
            )

    else:
        if payload["criterion"] != "soft_policy_reference":
            _error(
                "soft_target eligibility criterion must be "
                "'soft_policy_reference'"
            )

        _validate_version_ref(
            payload["soft_policy_ref"],
            label="soft_policy_ref",
        )


def _validate_component_basis(value: Any) -> None:
    if not isinstance(value, Mapping):
        _error("component_basis must be an object")

    _exact_keys(
        value,
        {
            "component_count",
            "status",
            "masks_source",
            "component_ablation_source",
            "stage8_masking_manifest",
        },
        label="component_basis",
    )

    if value["component_count"] != COMPONENT_COUNT:
        _error("component_basis component_count must be 516")

    if value["status"] != COMPONENT_BASIS_STATUS:
        _error(
            "component_basis status must be "
            "'reused_predecessor_definition'"
        )

    expected = {
        "masks_source": MASKS_SOURCE,
        "component_ablation_source": COMPONENT_ABLATION_SOURCE,
        "stage8_masking_manifest": STAGE8_MASKING_MANIFEST,
    }

    for field, expected_value in expected.items():
        actual = value[field]
        if not isinstance(actual, Mapping):
            _error(f"component_basis {field} must be an object")
        if dict(actual) != expected_value:
            _error(f"component_basis {field} does not match frozen authority")


def _validate_sealed_dense_model(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if identity.distillation_condition not in {"hard_target", "soft_target"}:
        _error("sealed_dense_model requires hard_target or soft_target")

    if record["provenance"]["producer_lane"] != "lane_b":
        _error("sealed_dense_model producer_lane must be lane_b")

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "eligibility_reference",
            "eligibility_status",
            "architecture_ref",
            "component_basis",
            "model_checkpoint",
        },
        label="sealed_dense_model payload",
    )

    eligibility_identity = _reference_identity(
        payload["eligibility_reference"],
        expected_record_type="student_eligibility",
        contract=contract,
        stage3=stage3,
        require_hash=True,
    )

    _same_student_identity(identity, eligibility_identity)

    if payload["eligibility_status"] != "passed":
        _error("sealed_dense_model requires passing eligibility")

    _validate_version_ref(
        payload["architecture_ref"],
        label="architecture_ref",
    )

    _validate_component_basis(payload["component_basis"])

    _validate_portable_artifact(
        payload["model_checkpoint"],
        label="sealed_dense_model model_checkpoint",
        required_storage_class="external_checkpoint",
    )


def validate_part_m_record(
    record: Mapping[str, Any],
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
    stage3_registry: Mapping[str, Any],
    stage3_registry_sha256: str,
) -> None:
    """Validate one Part M record including cross-field semantics."""
    validate_common_envelope(
        record,
        contract=contract,
        stage3=stage3,
    )

    record_type = record["record_type"]

    if record_type not in _M_RECORD_TYPES:
        _error(f"record_type is not a Part M schema: {record_type!r}")

    try:
        identity = parse_condition_id(record["condition_id"], stage3)
    except ConditionIdentityError as exc:
        _error(f"invalid Part M condition_id: {exc}")

    if record_type == "teacher_reference":
        _validate_teacher_reference(
            record,
            identity,
            stage3_registry=stage3_registry,
            stage3_registry_sha256=stage3_registry_sha256,
        )
    elif record_type == "teacher_output_cache":
        _validate_teacher_output_cache(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )
    elif record_type == "student_attempt":
        _validate_student_attempt(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )
    elif record_type == "student_eligibility":
        _validate_student_eligibility(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )
    elif record_type == "sealed_dense_model":
        _validate_sealed_dense_model(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )
