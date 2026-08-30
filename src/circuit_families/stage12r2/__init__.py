"""Technical Stage 12-R2 basis-sensitivity machinery."""

from circuit_families.stage12r2.accounting import (
    BasisAccounting,
    TypeAccounting,
    account_basis_mask,
)
from circuit_families.stage12r2.blocks import (
    BLOCK_BASIS_FAMILY,
    BLOCK_PARTITION_VERSION,
    BalancedBlockPartition,
    balanced_block_basis,
    build_balanced_partition,
    expand_block_mask_to_parent_values,
)
from circuit_families.stage12r2.comparison import (
    BasisSensitivitySummary,
    CrossBasisComparisonRequest,
    ExactLedgerEvidence,
    comparison_values,
    validate_cross_basis_comparison,
)
from circuit_families.stage12r2.contracts import (
    BASIS_CONTRACT_VERSION,
    BASIS_MASK_VERSION,
    BasisComponentDescriptor,
    BasisContract,
    BasisMask,
    BasisRelationship,
    canonical_json_bytes,
    canonical_sha256,
    validate_relationship,
    validate_technical_record_payload,
)
from circuit_families.stage12r2.lifecycle import (
    LIFECYCLE_VERSION,
    TECHNICAL_CLASSIFICATION,
    Stage12R2LifecycleRecord,
    lifecycle_record_from_mapping,
)
from circuit_families.stage12r2.protocols import (
    ActivationInterceptionCapability,
    BasisMaskApplicationCapability,
    BasisMetadataProvider,
)
from circuit_families.stage12r2.rotation import (
    ROTATED_BASIS_FAMILY,
    ROTATION_ALGORITHM_VERSION,
    RotationSpec,
    apply_rotated_coordinate_mask,
    build_rotation_matrix,
    build_rotation_spec,
    rotated_basis,
    validate_rotation_matrix,
)

__all__ = [
    "validate_cross_basis_comparison",
    "lifecycle_record_from_mapping",
    "comparison_values",
    "TECHNICAL_CLASSIFICATION",
    "Stage12R2LifecycleRecord",
    "LIFECYCLE_VERSION",
    "ExactLedgerEvidence",
    "CrossBasisComparisonRequest",
    "BasisSensitivitySummary",
    "ROTATED_BASIS_FAMILY",
    "ROTATION_ALGORITHM_VERSION",
    "RotationSpec",
    "apply_rotated_coordinate_mask",
    "build_rotation_matrix",
    "build_rotation_spec",
    "rotated_basis",
    "validate_rotation_matrix",
    "BASIS_CONTRACT_VERSION",
    "BASIS_MASK_VERSION",
    "ActivationInterceptionCapability",
    "BasisComponentDescriptor",
    "BasisContract",
    "BasisMask",
    "BasisMaskApplicationCapability",
    "BasisMetadataProvider",
    "BasisRelationship",
    "canonical_json_bytes",
    "canonical_sha256",
    "validate_relationship",
    "validate_technical_record_payload",
]

from circuit_families.stage12r2.attention import (
    ATTENTION_COORDINATE_DEFINITION,
    ATTENTION_COORDINATE_FAMILY,
    AttentionCoordinateSpec,
    apply_attention_coordinate_mask,
    attention_coordinate_basis,
    parent_head_mask_to_coordinate_mask,
)
from circuit_families.stage12r2.canonical import (
    CANONICAL_BASIS_FAMILY,
    CANONICAL_COORDINATE_DEFINITION,
    basis_mask_to_component_mask,
    canonical_basis_contract,
    component_mask_to_basis_mask,
)

__all__ += [
    "ATTENTION_COORDINATE_DEFINITION",
    "ATTENTION_COORDINATE_FAMILY",
    "AttentionCoordinateSpec",
    "CANONICAL_BASIS_FAMILY",
    "CANONICAL_COORDINATE_DEFINITION",
    "apply_attention_coordinate_mask",
    "attention_coordinate_basis",
    "basis_mask_to_component_mask",
    "canonical_basis_contract",
    "component_mask_to_basis_mask",
    "parent_head_mask_to_coordinate_mask",
]


__all__ += [
    "BLOCK_BASIS_FAMILY",
    "BLOCK_PARTITION_VERSION",
    "BalancedBlockPartition",
    "BasisAccounting",
    "TypeAccounting",
    "account_basis_mask",
    "balanced_block_basis",
    "build_balanced_partition",
    "expand_block_mask_to_parent_values",
]
