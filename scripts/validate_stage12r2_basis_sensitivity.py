"""Portable validate-only Stage 12-R2 technical integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import torch

from circuit_families.interpretability.masks import ComponentMask
from circuit_families.stage12r2.accounting import account_basis_mask
from circuit_families.stage12r2.attention import (
    AttentionCoordinateSpec,
    apply_attention_coordinate_mask,
    attention_coordinate_basis,
)
from circuit_families.stage12r2.blocks import (
    balanced_block_basis,
    build_balanced_partition,
    expand_block_mask_to_parent_values,
)
from circuit_families.stage12r2.canonical import (
    basis_mask_to_component_mask,
    canonical_basis_contract,
    component_mask_to_basis_mask,
)
from circuit_families.stage12r2.comparison import (
    BasisSensitivitySummary,
    CrossBasisComparisonRequest,
    ExactLedgerEvidence,
    validate_cross_basis_comparison,
)
from circuit_families.stage12r2.contracts import BasisMask, canonical_json_bytes
from circuit_families.stage12r2.lifecycle import Stage12R2LifecycleRecord
from circuit_families.stage12r2.rotation import (
    apply_rotated_coordinate_mask,
    build_rotation_spec,
    rotated_basis,
)

CLASSIFICATION = "synthetic_technical_only"


def _sha(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_report() -> dict[str, object]:
    parent = canonical_basis_contract(
        parent_model_identity="technical-model:stage12r2-validator-v1",
        parent_component_basis_identity="stage4-component-basis:technical-v1",
        attention_parameter_weight=32,
        mlp_parameter_weight=129,
    )

    canonical_mask = ComponentMask.from_ablated_identifiers(
        ["H1", "N0", "N7"],
    )
    bound_canonical = component_mask_to_basis_mask(canonical_mask, parent)
    canonical_round_trip = (
        basis_mask_to_component_mask(bound_canonical, parent) == canonical_mask
    )

    attention_spec = AttentionCoordinateSpec(layer=0, n_heads=4, d_head=3)
    attention_basis = attention_coordinate_basis(
        parent_basis=parent,
        spec=attention_spec,
        parameter_weight_per_coordinate=(1,) * 12,
    )
    attention_activation = torch.arange(
        24,
        dtype=torch.float64,
    ).reshape(1, 2, 4, 3)
    attention_all_on = apply_attention_coordinate_mask(
        attention_activation,
        mask=BasisMask(
            basis_hash=attention_basis.basis_hash,
            values=(1,) * 12,
        ),
        basis=attention_basis,
        spec=attention_spec,
    )
    attention_identity = torch.equal(attention_all_on, attention_activation)

    partition_a = build_balanced_partition(
        parent_basis=parent,
        layer_identity="blocks.0.mlp.hook_post",
        partition_seed=11,
        block_count=8,
    )
    partition_b = build_balanced_partition(
        parent_basis=parent,
        layer_identity="blocks.0.mlp.hook_post",
        partition_seed=29,
        block_count=8,
    )
    grouped = balanced_block_basis(
        parent_basis=parent,
        partition=partition_a,
    )
    grouped_mask = BasisMask(
        basis_hash=grouped.basis_hash,
        values=(1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0),
    )
    expanded = expand_block_mask_to_parent_values(
        block_mask=grouped_mask,
        block_basis=grouped,
        parent_basis=parent,
        partition=partition_a,
    )
    grouped_accounting = account_basis_mask(
        basis=grouped,
        mask=grouped_mask,
        partition=partition_a,
    )

    rotation_identity_spec, rotation_identity_matrix = build_rotation_spec(
        parent_basis=parent,
        subspace_identity="technical-validation-subspace:width4",
        dimension=4,
        seed=0,
        dtype="float64",
        identity=True,
    )
    identity_rotated_basis = rotated_basis(
        parent_basis=parent,
        spec=rotation_identity_spec,
        component_type="technical_activation_coordinate",
        intervention_location="technical.validation.activation",
        parameter_weights=(1, 1, 1, 1),
    )
    rotation_input = torch.arange(16, dtype=torch.float64).reshape(2, 2, 4)
    identity_rotation_result = apply_rotated_coordinate_mask(
        rotation_input,
        mask=BasisMask(
            basis_hash=identity_rotated_basis.basis_hash,
            values=(1, 0, 1, 0),
        ),
        basis=identity_rotated_basis,
        spec=rotation_identity_spec,
        matrix=rotation_identity_matrix,
    )

    rotation_spec, rotation_matrix = build_rotation_spec(
        parent_basis=parent,
        subspace_identity="technical-validation-subspace:width4",
        dimension=4,
        seed=71,
        dtype="float64",
    )
    nontrivial_rotated_basis = rotated_basis(
        parent_basis=parent,
        spec=rotation_spec,
        component_type="technical_activation_coordinate",
        intervention_location="technical.validation.activation",
        parameter_weights=(1, 1, 1, 1),
    )
    nontrivial_all_on = apply_rotated_coordinate_mask(
        rotation_input,
        mask=BasisMask(
            basis_hash=nontrivial_rotated_basis.basis_hash,
            values=(1, 1, 1, 1),
        ),
        basis=nontrivial_rotated_basis,
        spec=rotation_spec,
        matrix=rotation_matrix,
        atol=1e-10,
    )

    evidence = ExactLedgerEvidence(
        ledger_reference="stage6a-exact-ledger:technical-validator",
        mask_identity="technical-mask:validator",
        basis_hash=grouped.basis_hash,
        model_identity=parent.parent_model_identity,
        dense_reference_identity="technical-dense-reference:v1",
        fidelity_definition_identity="centred-logit-predictive-fidelity/v1",
        evaluation_domain_identity="technical-domain:validator-v1",
        intervention_protocol_identity="technical-intervention:v1",
        state="evaluated",
        exact_fidelity=-0.125,
        intact_mask=False,
    )
    summary = BasisSensitivitySummary(
        evidence=evidence,
        accounting=grouped_accounting,
    )

    lifecycle = Stage12R2LifecycleRecord(
        basis_hash=grouped.basis_hash,
        model_identity=parent.parent_model_identity,
        exact_ledger_reference=evidence.ledger_reference,
        partition_hash=partition_a.partition_hash,
    )

    invalid_comparison_rejected = False
    try:
        validate_cross_basis_comparison(
            CrossBasisComparisonRequest(
                left=summary,
                right=summary,
                left_basis=grouped,
                right_basis=grouped,
                measure="raw_component_proportion",
            )
        )
        relabelled = BasisSensitivitySummary(
            evidence=ExactLedgerEvidence(
                ledger_reference=evidence.ledger_reference,
                mask_identity=evidence.mask_identity,
                basis_hash=parent.basis_hash,
                model_identity=evidence.model_identity,
                dense_reference_identity=evidence.dense_reference_identity,
                fidelity_definition_identity=evidence.fidelity_definition_identity,
                evaluation_domain_identity=evidence.evaluation_domain_identity,
                intervention_protocol_identity=evidence.intervention_protocol_identity,
                state=evidence.state,
                exact_fidelity=evidence.exact_fidelity,
                intact_mask=evidence.intact_mask,
            ),
            accounting=grouped_accounting,
        )
        validate_cross_basis_comparison(
            CrossBasisComparisonRequest(
                left=relabelled,
                right=summary,
                left_basis=parent,
                right_basis=grouped,
                measure="raw_component_proportion",
            )
        )
    except ValueError:
        invalid_comparison_rejected = True

    report: dict[str, object] = {
        "classification": CLASSIFICATION,
        "scientific_data": False,
        "production_eligible": False,
        "rd004_resolved": False,
        "registered_model_access": False,
        "canonical": {
            "basis_hash": parent.basis_hash,
            "component_count": parent.component_count,
            "round_trip_mask_exact": canonical_round_trip,
        },
        "attention_refinement": {
            "basis_hash": attention_basis.basis_hash,
            "component_count": attention_basis.component_count,
            "all_on_identity": attention_identity,
            "location": "blocks.0.attn.hook_z",
        },
        "partitions": {
            "first_hash": partition_a.partition_hash,
            "second_hash": partition_b.partition_hash,
            "distinct": partition_a.partition_hash != partition_b.partition_hash,
            "expanded_parent_retained": sum(expanded),
        },
        "rotations": {
            "identity_hash": rotation_identity_spec.rotation_hash,
            "nontrivial_hash": rotation_spec.rotation_hash,
            "identity_mask_checksum": float(identity_rotation_result.sum().item()),
            "nontrivial_all_on_identity": bool(
                torch.allclose(
                    nontrivial_all_on,
                    rotation_input,
                    atol=1e-10,
                    rtol=1e-10,
                )
            ),
        },
        "accounting": {
            "raw_retained_count": grouped_accounting.raw_retained_count,
            "raw_total_count": grouped_accounting.raw_total_count,
            "raw_retained_proportion": grouped_accounting.raw_retained_proportion,
            "parameter_weight_retained": grouped_accounting.parameter_weight_retained,
            "parameter_weight_total": grouped_accounting.parameter_weight_total,
            "parameter_weighted_proportion": (
                grouped_accounting.parameter_weight_retained_proportion
            ),
            "type_strata": [
                {
                    "component_type": row.component_type,
                    "retained_count": row.retained_count,
                    "total_count": row.total_count,
                    "retained_proportion": row.retained_proportion,
                }
                for row in grouped_accounting.type_accounting
            ],
            "parent_neuron_retained_count": (
                grouped_accounting.parent_neuron_retained_count
            ),
            "parent_neuron_total_count": grouped_accounting.parent_neuron_total_count,
        },
        "exact_ledger_consumption": {
            "reference": evidence.ledger_reference,
            "exact_fidelity_preserved": evidence.exact_fidelity,
            "state": evidence.state,
            "intact_mask": evidence.intact_mask,
        },
        "invalid_cross_basis_comparison_rejected": invalid_comparison_rejected,
        "lifecycle": lifecycle.to_record(),
    }

    report["report_hash"] = _sha(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report()

    checks = {
        "classification": report["classification"] == CLASSIFICATION,
        "scientific_data": report["scientific_data"] is False,
        "production_eligible": report["production_eligible"] is False,
        "canonical_round_trip": report["canonical"]["round_trip_mask_exact"],
        "attention_all_on": report["attention_refinement"]["all_on_identity"],
        "partition_distinct": report["partitions"]["distinct"],
        "rotation_all_on": report["rotations"]["nontrivial_all_on_identity"],
        "invalid_comparison_rejected": report[
            "invalid_cross_basis_comparison_rejected"
        ],
        "negative_fidelity_preserved": (
            report["exact_ledger_consumption"]["exact_fidelity_preserved"] == -0.125
        ),
    }
    if not all(checks.values()):
        print(json.dumps(checks, sort_keys=True), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(f"classification={report['classification']}")
        print("scientific_data=false")
        print("production_eligible=false")
        print(f"canonical_basis_hash={report['canonical']['basis_hash']}")
        print(f"attention_basis_hash={report['attention_refinement']['basis_hash']}")
        print(f"partition_a_hash={report['partitions']['first_hash']}")
        print(f"partition_b_hash={report['partitions']['second_hash']}")
        print(f"rotation_hash={report['rotations']['nontrivial_hash']}")
        print(
            "invalid_cross_basis_comparison_rejected="
            f"{str(report['invalid_cross_basis_comparison_rejected']).lower()}"
        )
        print(
            "exact_fidelity_preserved="
            f"{report['exact_ledger_consumption']['exact_fidelity_preserved']}"
        )
        print(f"report_hash={report['report_hash']}")
        print("STAGE12R2_VALIDATE=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
