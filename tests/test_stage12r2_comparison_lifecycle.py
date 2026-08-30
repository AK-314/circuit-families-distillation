from __future__ import annotations

import copy

import pytest

from circuit_families.stage12r2.accounting import (
    BasisAccounting,
    TypeAccounting,
)
from circuit_families.stage12r2.canonical import canonical_basis_contract
from circuit_families.stage12r2.comparison import (
    BasisSensitivitySummary,
    CrossBasisComparisonRequest,
    ExactLedgerEvidence,
    comparison_values,
    validate_cross_basis_comparison,
)
from circuit_families.stage12r2.contracts import BasisContract, BasisRelationship
from circuit_families.stage12r2.lifecycle import (
    Stage12R2LifecycleRecord,
    lifecycle_record_from_mapping,
)


def canonical() -> BasisContract:
    return canonical_basis_contract(
        parent_model_identity="technical-model:fixture-v1",
        parent_component_basis_identity="stage4-component-basis:v1",
        attention_parameter_weight=32,
        mlp_parameter_weight=129,
    )


def related_view(parent: BasisContract) -> BasisContract:
    component = parent.components[0]
    return BasisContract(
        parent_model_identity=parent.parent_model_identity,
        parent_component_basis_identity=parent.parent_component_basis_identity,
        basis_family="technical-related-view",
        coordinate_definition="technical related coordinates",
        components=(component,),
        intervention_location=component.intervention_location,
        intervention_semantics="technical",
        parameter_weight_denominator_definition="shared-parameter-denominator/v1",
        raw_component_denominator_definition="related-view count",
        relationship=BasisRelationship(
            kind="rotated_view",
            parent_basis_hash=parent.basis_hash,
            mapping_identity="technical-transform:v1",
        ),
        rotation_subspace_identity="technical-subspace:v1",
    )


def evidence(
    basis: BasisContract,
    *,
    fidelity: float | None = -0.25,
    state: str = "evaluated",
    model: str = "technical-model:fixture-v1",
    domain: str = "technical-domain:v1",
    intact_mask: bool = False,
) -> ExactLedgerEvidence:
    return ExactLedgerEvidence(
        ledger_reference="exact-ledger:technical:001",
        mask_identity="mask:001",
        basis_hash=basis.basis_hash,
        model_identity=model,
        dense_reference_identity="dense-reference:001",
        fidelity_definition_identity="centred-logit-predictive-fidelity/v1",
        evaluation_domain_identity=domain,
        intervention_protocol_identity="technical-intervention:v1",
        state=state,
        exact_fidelity=fidelity,
        intact_mask=intact_mask,
        failure_reason="technical failure" if state == "failed" else None,
    )


def accounting(
    basis: BasisContract,
    *,
    denominator_definition: str = "shared-parameter-denominator/v1",
    types: tuple[TypeAccounting, ...] | None = None,
    parent_total: int | None = 10,
    parent_retained: int | None = 5,
) -> BasisAccounting:
    return BasisAccounting(
        basis_hash=basis.basis_hash,
        raw_retained_count=1,
        raw_total_count=basis.component_count,
        parameter_weight_retained=1,
        parameter_weight_total=2,
        parameter_weight_denominator_definition=denominator_definition,
        type_accounting=types
        or (
            TypeAccounting(
                component_type="attention_head",
                retained_count=1,
                total_count=1,
            ),
        ),
        parent_neuron_retained_count=parent_retained,
        parent_neuron_total_count=parent_total,
    )


def summary(
    basis: BasisContract,
    **kwargs: object,
) -> BasisSensitivitySummary:
    return BasisSensitivitySummary(
        evidence=evidence(basis),
        accounting=accounting(basis, **kwargs),
    )


def test_negative_exact_fidelity_is_preserved_unchanged() -> None:
    parent = canonical()
    row = summary(parent)
    assert row.evidence.exact_fidelity == -0.25


def test_failed_mask_and_intact_mask_state_are_preserved() -> None:
    parent = canonical()
    failed = evidence(parent, state="failed", fidelity=None, intact_mask=True)
    assert failed.state == "failed"
    assert failed.intact_mask is True
    assert failed.failure_reason == "technical failure"


def test_canonical_result_relabelled_as_other_basis_rejects() -> None:
    parent = canonical()
    child = related_view(parent)
    left = summary(parent)
    request = CrossBasisComparisonRequest(
        left=left,
        right=summary(child),
        left_basis=child,
        right_basis=child,
        measure="parameter_weighted_proportion",
    )
    with pytest.raises(ValueError, match="relabeled"):
        validate_cross_basis_comparison(request)


def test_raw_count_cross_granularity_rejects() -> None:
    parent = canonical()
    child = related_view(parent)
    request = CrossBasisComparisonRequest(
        left=summary(parent),
        right=summary(child),
        left_basis=parent,
        right_basis=child,
        measure="raw_component_proportion",
    )
    with pytest.raises(ValueError, match="raw component"):
        validate_cross_basis_comparison(request)


def test_parameter_denominator_substitution_rejects() -> None:
    parent = canonical()
    child = related_view(parent)
    request = CrossBasisComparisonRequest(
        left=summary(parent),
        right=summary(child, denominator_definition="different-denominator/v1"),
        left_basis=parent,
        right_basis=child,
        measure="parameter_weighted_proportion",
    )
    with pytest.raises(ValueError, match="denominator"):
        validate_cross_basis_comparison(request)


def test_component_type_omission_rejects() -> None:
    parent = canonical()
    child = related_view(parent)
    request = CrossBasisComparisonRequest(
        left=summary(parent),
        right=summary(
            child,
            types=(
                TypeAccounting(
                    component_type="different_type",
                    retained_count=1,
                    total_count=1,
                ),
            ),
        ),
        left_basis=parent,
        right_basis=child,
        measure="type_stratified",
    )
    with pytest.raises(ValueError, match="strata"):
        validate_cross_basis_comparison(request)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("model_identity", "other-model", "model_identity"),
        ("evaluation_domain_identity", "other-domain", "evaluation_domain_identity"),
    ],
)
def test_cross_model_or_domain_rejects(
    field: str,
    value: str,
    match: str,
) -> None:
    parent = canonical()
    child = related_view(parent)
    right_evidence = evidence(child)
    object.__setattr__(right_evidence, field, value)
    request = CrossBasisComparisonRequest(
        left=summary(parent),
        right=BasisSensitivitySummary(
            evidence=right_evidence,
            accounting=accounting(child),
        ),
        left_basis=parent,
        right_basis=child,
        measure="parameter_weighted_proportion",
    )
    with pytest.raises(ValueError, match=match):
        validate_cross_basis_comparison(request)


def test_valid_parameter_weighted_comparison_is_reconstructable() -> None:
    parent = canonical()
    child = related_view(parent)
    request = CrossBasisComparisonRequest(
        left=summary(parent),
        right=summary(child),
        left_basis=parent,
        right_basis=child,
        measure="parameter_weighted_proportion",
    )
    assert comparison_values(request) == (0.5, 0.5)


def test_lifecycle_keeps_rd004_unresolved_and_hashes_provenance() -> None:
    record = Stage12R2LifecycleRecord(
        basis_hash="a" * 64,
        model_identity="technical-model:fixture-v1",
        exact_ledger_reference="exact-ledger:technical:001",
        transform_hash="b" * 64,
        partition_hash="c" * 64,
    )
    mapping = record.to_record()
    assert mapping["rd004"] == {
        "basis_panel": None,
        "partition_seed": None,
        "rotation_seed": None,
        "model_assignment": None,
    }
    assert lifecycle_record_from_mapping(mapping) == record


def test_stale_transform_or_partition_record_hash_rejects() -> None:
    record = Stage12R2LifecycleRecord(
        basis_hash="a" * 64,
        model_identity="technical-model:fixture-v1",
        exact_ledger_reference="exact-ledger:technical:001",
        transform_hash="b" * 64,
        partition_hash="c" * 64,
    ).to_record()
    tampered = copy.deepcopy(record)
    tampered["transform_hash"] = "d" * 64
    with pytest.raises(ValueError, match="record hash"):
        lifecycle_record_from_mapping(tampered)


def test_technical_profile_cannot_be_relabelled_production() -> None:
    record = Stage12R2LifecycleRecord(
        basis_hash="a" * 64,
        model_identity="technical-model:fixture-v1",
        exact_ledger_reference="exact-ledger:technical:001",
    ).to_record()
    record["production_eligible"] = True
    with pytest.raises(ValueError, match="production_eligible"):
        lifecycle_record_from_mapping(record)


def test_rd004_cannot_be_filled() -> None:
    with pytest.raises(ValueError, match="RD-004"):
        Stage12R2LifecycleRecord(
            basis_hash="a" * 64,
            model_identity="technical-model:fixture-v1",
            exact_ledger_reference="exact-ledger:technical:001",
            rd004_partition_seed=7,
        )
