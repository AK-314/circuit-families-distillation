"""Fixed deterministic orthogonal rotations for technical activation views."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from circuit_families.stage12r2.contracts import (
    BasisComponentDescriptor,
    BasisContract,
    BasisMask,
    BasisRelationship,
    canonical_sha256,
)

ROTATION_ALGORITHM_VERSION = "stage12r2-seeded-qr-sign-normalized/v1"
ROTATED_BASIS_FAMILY = "fixed_orthogonal_rotated_view"


@dataclass(frozen=True)
class RotationSpec:
    parent_basis_hash: str
    parent_model_identity: str
    subspace_identity: str
    dimension: int
    seed: int
    dtype: str
    matrix_hash: str
    inverse_convention: str = "transpose"
    algorithm_version: str = ROTATION_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if self.algorithm_version != ROTATION_ALGORITHM_VERSION:
            raise ValueError("unsupported rotation algorithm version")
        if not self.parent_basis_hash or len(self.parent_basis_hash) != 64:
            raise ValueError("parent_basis_hash must be a SHA-256 digest")
        if not self.parent_model_identity:
            raise ValueError("parent_model_identity must be non-empty")
        if not self.subspace_identity:
            raise ValueError("subspace_identity must be non-empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.inverse_convention != "transpose":
            raise ValueError("orthogonal inverse convention must be transpose")
        if len(self.matrix_hash) != 64:
            raise ValueError("matrix_hash must be a SHA-256 digest")

    def identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_version": self.algorithm_version,
            "parent_basis_hash": self.parent_basis_hash,
            "parent_model_identity": self.parent_model_identity,
            "subspace_identity": self.subspace_identity,
            "dimension": self.dimension,
            "seed": self.seed,
            "dtype": self.dtype,
            "matrix_hash": self.matrix_hash,
            "inverse_convention": self.inverse_convention,
        }

    @property
    def rotation_hash(self) -> str:
        return canonical_sha256(self.identity_payload())


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def matrix_sha256(matrix: torch.Tensor) -> str:
    """Hash matrix shape, dtype, and exact CPU bytes without JSON embedding."""
    if matrix.ndim != 2:
        raise ValueError("rotation matrix must be rank 2")
    cpu = matrix.detach().contiguous().cpu()
    payload = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "bytes_sha256": canonical_sha256(list(cpu.numpy().tobytes())),
    }
    return canonical_sha256(payload)


def _stable_matrix_hash(matrix: torch.Tensor) -> str:
    """Compact exact hash without serializing matrix values into records."""
    import hashlib

    cpu = matrix.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(str(cpu.dtype).encode())
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def build_rotation_matrix(
    *,
    dimension: int,
    seed: int,
    dtype: str,
    identity: bool = False,
) -> torch.Tensor:
    """Construct a deterministic orthogonal matrix with fixed QR sign convention."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    torch_dtype = _torch_dtype(dtype)
    if identity:
        return torch.eye(dimension, dtype=torch_dtype)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    source = torch.randn(
        (dimension, dimension),
        generator=generator,
        dtype=torch_dtype,
    )
    q, r = torch.linalg.qr(source)

    diagonal = torch.diagonal(r)
    signs = torch.where(
        diagonal < 0,
        torch.tensor(-1.0, dtype=torch_dtype),
        torch.tensor(1.0, dtype=torch_dtype),
    )
    q = q * signs.unsqueeze(0)
    return q.contiguous()


def build_rotation_spec(
    *,
    parent_basis: BasisContract,
    subspace_identity: str,
    dimension: int,
    seed: int,
    dtype: str,
    identity: bool = False,
) -> tuple[RotationSpec, torch.Tensor]:
    matrix = build_rotation_matrix(
        dimension=dimension,
        seed=seed,
        dtype=dtype,
        identity=identity,
    )
    spec = RotationSpec(
        parent_basis_hash=parent_basis.basis_hash,
        parent_model_identity=parent_basis.parent_model_identity,
        subspace_identity=subspace_identity,
        dimension=dimension,
        seed=seed,
        dtype=dtype,
        matrix_hash=_stable_matrix_hash(matrix),
    )
    return spec, matrix


def validate_rotation_matrix(
    matrix: torch.Tensor,
    *,
    spec: RotationSpec,
    parent_basis: BasisContract,
    atol: float = 1e-6,
) -> None:
    if spec.parent_basis_hash != parent_basis.basis_hash:
        raise ValueError("rotation spec belongs to the wrong basis")
    if spec.parent_model_identity != parent_basis.parent_model_identity:
        raise ValueError("rotation spec belongs to the wrong model")
    if matrix.ndim != 2 or tuple(matrix.shape) != (spec.dimension, spec.dimension):
        raise ValueError("rotation matrix has the wrong dimensions")
    if matrix.dtype != _torch_dtype(spec.dtype):
        raise ValueError("rotation matrix has the wrong dtype")
    if _stable_matrix_hash(matrix) != spec.matrix_hash:
        raise ValueError("rotation matrix hash is stale or incorrect")

    identity = torch.eye(spec.dimension, dtype=matrix.dtype, device=matrix.device)
    error = torch.max(torch.abs(matrix.T @ matrix - identity)).item()
    if error > atol:
        raise ValueError("rotation matrix is not orthogonal within tolerance")


def rotated_basis(
    *,
    parent_basis: BasisContract,
    spec: RotationSpec,
    component_type: str,
    intervention_location: str,
    parameter_weights: tuple[int, ...],
) -> BasisContract:
    if spec.parent_basis_hash != parent_basis.basis_hash:
        raise ValueError("rotation spec does not belong to parent basis")
    if len(parameter_weights) != spec.dimension:
        raise ValueError("parameter-weight metadata length does not match dimension")

    components = tuple(
        BasisComponentDescriptor(
            component_id=f"ROT_{index:04d}",
            component_type=component_type,
            source_subspace=spec.subspace_identity,
            intervention_location=intervention_location,
            parameter_weight=parameter_weights[index],
            coordinate_identity=(
                f"rotation={spec.rotation_hash}/coordinate={index}"
            ),
        )
        for index in range(spec.dimension)
    )

    return BasisContract(
        parent_model_identity=parent_basis.parent_model_identity,
        parent_component_basis_identity=parent_basis.parent_component_basis_identity,
        basis_family=ROTATED_BASIS_FAMILY,
        coordinate_definition="fixed orthogonal rotated coordinates/v1",
        components=components,
        intervention_location=intervention_location,
        intervention_semantics=(
            "activation maps into fixed orthogonal coordinates, binary mask applies, "
            "then transpose inverse maps back before downstream computation"
        ),
        parameter_weight_denominator_definition=(
            "sum explicitly declared rotated-coordinate parameter weights"
        ),
        raw_component_denominator_definition="rotated coordinate count",
        relationship=BasisRelationship(
            kind="rotated_view",
            parent_basis_hash=parent_basis.basis_hash,
            mapping_identity=spec.rotation_hash,
        ),
        rotation_subspace_identity=spec.subspace_identity,
        display_label="fixed orthogonal rotated view",
    )


def apply_rotated_coordinate_mask(
    activation: torch.Tensor,
    *,
    mask: BasisMask,
    basis: BasisContract,
    spec: RotationSpec,
    matrix: torch.Tensor,
    atol: float = 1e-6,
) -> torch.Tensor:
    """Rotate -> mask -> inverse-rotate on the final activation axis."""
    mask.validate_for(basis)
    if basis.basis_family != ROTATED_BASIS_FAMILY:
        raise ValueError("basis is not a rotated view")
    if basis.rotation_subspace_identity != spec.subspace_identity:
        raise ValueError("basis and rotation subspace identities do not match")
    if activation.shape[-1] != spec.dimension:
        raise ValueError("activation final dimension does not match rotation")
    if activation.dtype != _torch_dtype(spec.dtype):
        raise ValueError("activation dtype does not match rotation dtype")

    if spec.parent_model_identity != basis.parent_model_identity:
        raise ValueError("rotation spec belongs to the wrong model")
    if basis.relationship is None:
        raise ValueError("rotated basis is missing its parent relationship")
    if basis.relationship.parent_basis_hash != spec.parent_basis_hash:
        raise ValueError("rotation spec belongs to the wrong parent basis")
    if matrix.ndim != 2 or tuple(matrix.shape) != (spec.dimension, spec.dimension):
        raise ValueError("rotation matrix has the wrong dimensions")
    if matrix.dtype != _torch_dtype(spec.dtype):
        raise ValueError("rotation matrix has the wrong dtype")
    if _stable_matrix_hash(matrix) != spec.matrix_hash:
        raise ValueError("rotation matrix hash is stale or incorrect")
    identity = torch.eye(spec.dimension, dtype=matrix.dtype, device=matrix.device)
    if torch.max(torch.abs(matrix.T @ matrix - identity)).item() > atol:
        raise ValueError("rotation matrix is not orthogonal within tolerance")

    matrix_on_device = matrix.to(device=activation.device)
    rotated = activation @ matrix_on_device
    mask_tensor = torch.tensor(
        mask.values,
        dtype=activation.dtype,
        device=activation.device,
    )
    masked_rotated = rotated * mask_tensor
    return masked_rotated @ matrix_on_device.T
