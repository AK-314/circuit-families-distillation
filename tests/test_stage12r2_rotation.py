from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from circuit_families.stage12r2.canonical import canonical_basis_contract
from circuit_families.stage12r2.contracts import BasisMask
from circuit_families.stage12r2.rotation import (
    _stable_matrix_hash,
    apply_rotated_coordinate_mask,
    build_rotation_matrix,
    build_rotation_spec,
    rotated_basis,
    validate_rotation_matrix,
)


def parent_basis():
    return canonical_basis_contract(
        parent_model_identity="technical-model:fixture-v1",
        parent_component_basis_identity="stage4-component-basis:v1",
        attention_parameter_weight=32,
        mlp_parameter_weight=129,
    )


def fixture(*, seed: int = 19, identity: bool = False, dtype: str = "float64"):
    parent = parent_basis()
    spec, matrix = build_rotation_spec(
        parent_basis=parent,
        subspace_identity="technical-subspace:layer0:width4",
        dimension=4,
        seed=seed,
        dtype=dtype,
        identity=identity,
    )
    basis = rotated_basis(
        parent_basis=parent,
        spec=spec,
        component_type="technical_activation_coordinate",
        intervention_location="technical.activation",
        parameter_weights=(1, 1, 1, 1),
    )
    return parent, spec, matrix, basis


def test_rotation_is_orthogonal_and_deterministic() -> None:
    parent, spec1, matrix1, _ = fixture(seed=123)
    _, spec2, matrix2, _ = fixture(seed=123)
    validate_rotation_matrix(matrix1, spec=spec1, parent_basis=parent)
    assert torch.equal(matrix1, matrix2)
    assert spec1.matrix_hash == spec2.matrix_hash
    assert spec1.rotation_hash == spec2.rotation_hash


def test_seed_and_subspace_change_rotation_identity() -> None:
    _, first, _, _ = fixture(seed=1)
    _, second, _, _ = fixture(seed=2)

    parent = parent_basis()
    third, _ = build_rotation_spec(
        parent_basis=parent,
        subspace_identity="technical-subspace:different",
        dimension=4,
        seed=1,
        dtype="float64",
    )
    assert len({first.rotation_hash, second.rotation_hash, third.rotation_hash}) == 3


def test_all_on_rotated_intervention_is_dense_identity() -> None:
    _, spec, matrix, basis = fixture()
    activation = torch.randn((3, 5, 4), dtype=torch.float64)
    before = activation.clone()
    result = apply_rotated_coordinate_mask(
        activation,
        mask=BasisMask(basis_hash=basis.basis_hash, values=(1, 1, 1, 1)),
        basis=basis,
        spec=spec,
        matrix=matrix,
        atol=1e-10,
    )
    assert torch.allclose(result, activation, atol=1e-10, rtol=1e-10)
    assert torch.equal(activation, before)


def test_identity_rotation_matches_parent_coordinate_mask() -> None:
    _, spec, matrix, basis = fixture(identity=True)
    activation = torch.arange(16, dtype=torch.float64).reshape(2, 2, 4)
    mask = BasisMask(basis_hash=basis.basis_hash, values=(1, 0, 1, 0))
    result = apply_rotated_coordinate_mask(
        activation,
        mask=mask,
        basis=basis,
        spec=spec,
        matrix=matrix,
    )
    expected = activation * torch.tensor((1, 0, 1, 0), dtype=torch.float64)
    assert torch.equal(result, expected)


def test_full_round_trip_is_numerically_stable() -> None:
    _, spec, matrix, basis = fixture()
    activation = torch.randn((8, 4), dtype=torch.float64)
    result = apply_rotated_coordinate_mask(
        activation,
        mask=BasisMask(basis_hash=basis.basis_hash, values=(1,) * 4),
        basis=basis,
        spec=spec,
        matrix=matrix,
        atol=1e-10,
    )
    assert torch.max(torch.abs(result - activation)).item() < 1e-10


def test_nonorthogonal_matrix_rejects_even_with_matching_hash() -> None:
    parent, spec, matrix, _ = fixture()
    bad = matrix.clone()
    bad[0, 0] += 0.2
    forged = replace(spec, matrix_hash=_stable_matrix_hash(bad))
    with pytest.raises(ValueError, match="not orthogonal"):
        validate_rotation_matrix(bad, spec=forged, parent_basis=parent)


def test_wrong_dimension_rejects() -> None:
    parent, spec, _, _ = fixture()
    wrong = build_rotation_matrix(
        dimension=3,
        seed=spec.seed,
        dtype=spec.dtype,
    )
    forged = replace(
        spec,
        dimension=3,
        matrix_hash=_stable_matrix_hash(wrong),
    )
    validate_rotation_matrix(wrong, spec=forged, parent_basis=parent)


def test_stale_hash_rejects() -> None:
    parent, spec, matrix, _ = fixture()
    stale = replace(spec, matrix_hash="0" * 64)
    with pytest.raises(ValueError, match="hash"):
        validate_rotation_matrix(matrix, spec=stale, parent_basis=parent)


def test_wrong_dtype_rejects() -> None:
    parent, spec, matrix, _ = fixture(dtype="float64")
    wrong = matrix.float()
    with pytest.raises(ValueError, match="dtype"):
        validate_rotation_matrix(wrong, spec=spec, parent_basis=parent)


def test_wrong_basis_rejects() -> None:
    _, spec, matrix, _ = fixture()
    other = canonical_basis_contract(
        parent_model_identity="technical-model:other",
        parent_component_basis_identity="stage4-component-basis:v1",
        attention_parameter_weight=32,
        mlp_parameter_weight=129,
    )
    with pytest.raises(ValueError, match="wrong basis|wrong model"):
        validate_rotation_matrix(matrix, spec=spec, parent_basis=other)


def test_partial_mask_changes_activation_without_input_mutation() -> None:
    _, spec, matrix, basis = fixture()
    activation = torch.randn((2, 3, 4), dtype=torch.float64)
    before = activation.clone()
    result = apply_rotated_coordinate_mask(
        activation,
        mask=BasisMask(basis_hash=basis.basis_hash, values=(1, 0, 1, 0)),
        basis=basis,
        spec=spec,
        matrix=matrix,
    )
    assert not torch.allclose(result, activation)
    assert torch.equal(activation, before)


def test_rotation_record_contains_metadata_not_matrix_payload() -> None:
    _, spec, _, _ = fixture()
    payload = spec.identity_payload()
    assert payload["algorithm_version"]
    assert payload["matrix_hash"]
    assert payload["inverse_convention"] == "transpose"
    assert "matrix" not in payload
