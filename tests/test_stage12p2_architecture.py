"""Tests for policy-neutral Stage 12-P2 architecture records."""

from __future__ import annotations

from dataclasses import replace

import pytest

from circuit_families.stage12p2.architecture import (
    ArchitectureContractError,
    ArchitectureRecord,
    ArchitectureRegistry,
    BuilderDescriptor,
    validate_task_architecture_compatibility,
)
from circuit_families.stage12p2.builders import (
    CanonicalPredecessorBuilder,
    TechnicalTransformerBuilder,
    canonical_predecessor_record,
    default_technical_architecture_registry,
    technical_transformer_record,
    transformer_parameter_count,
)

BUILDER_HASH = "1" * 64


class StubTechnicalTransformerBuilder:
    descriptor = BuilderDescriptor(
        builder_ref="technical-transformer-builder/v1",
        implementation_sha256=BUILDER_HASH,
    )

    def validate_record(self, record: ArchitectureRecord) -> None:
        if record.activation not in {"relu", "gelu"}:
            raise ValueError("unsupported activation")
        if record.normalization not in {None, "layer_norm"}:
            raise ValueError("unsupported normalization")
        if record.positional_embedding_type not in {
            None,
            "standard",
        }:
            raise ValueError("unsupported positional embedding")

    def build(
        self,
        *,
        record: ArchitectureRecord,
        seed: int,
        device,
    ):
        self.validate_record(record)
        return {
            "architecture_ref": record.architecture_ref,
            "seed": seed,
            "device": str(device),
        }


def architecture_record(
    *,
    family: str = "predecessor",
    name: str = "matched",
    version: str = "v1",
    d_model: int = 128,
    n_heads: int = 4,
    d_head: int = 32,
    parameter_count: int = 227_313,
    searchable_component_count: int = 516,
    component_type_counts: dict[str, int] | None = None,
    activation: str = "relu",
    builder_sha256: str = BUILDER_HASH,
    scientific_data: bool = False,
    production_eligible: bool = False,
) -> ArchitectureRecord:
    if component_type_counts is None:
        component_type_counts = {
            "attention_head": 4,
            "mlp_neuron": 512,
        }

    return ArchitectureRecord(
        family=family,
        name=name,
        version=version,
        compatibility={
            "task_protocol": "stage12p1-task/v1",
            "input_representation": "token_sequence",
        },
        dimensions={
            "n_layers": 1,
            "n_ctx": 3,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_head": d_head,
            "d_mlp": 512,
            "d_vocab": 114,
            "d_vocab_out": 113,
        },
        activation=activation,
        normalization=None,
        positional_embedding_type="standard",
        parameter_count=parameter_count,
        searchable_component_count=searchable_component_count,
        component_type_counts=component_type_counts,
        initialization_ref="technical-gpt2-init/v1",
        builder_ref="technical-transformer-builder/v1",
        builder_sha256=builder_sha256,
        scientific_data=scientific_data,
        production_eligible=production_eligible,
    )


def registry() -> ArchitectureRegistry:
    builder = StubTechnicalTransformerBuilder()
    return ArchitectureRegistry(
        builders={builder.descriptor.builder_ref: builder},
    )


def test_record_exposes_explicit_accounting_and_technical_flags() -> None:
    record = architecture_record()
    mapping = record.to_mapping()

    assert record.architecture_ref == "predecessor-matched/v1"
    assert mapping["parameter_count"] == 227_313
    assert mapping["searchable_component_count"] == 516
    assert mapping["component_type_counts"] == {
        "attention_head": 4,
        "mlp_neuron": 512,
    }
    assert mapping["scientific_data"] is False
    assert mapping["production_eligible"] is False
    assert len(mapping["record_sha256"]) == 64


def test_registry_rejects_duplicate_architecture_reference() -> None:
    value = architecture_record()
    values = registry()
    values.register(value)

    with pytest.raises(
        ArchitectureContractError,
        match="duplicate architecture reference",
    ):
        values.register(value)


def test_record_rejects_invalid_head_divisibility_relation() -> None:
    with pytest.raises(
        ArchitectureContractError,
        match="n_heads multiplied by d_head",
    ):
        architecture_record(
            d_model=128,
            n_heads=3,
            d_head=32,
        )


def test_record_rejects_component_count_mismatch() -> None:
    with pytest.raises(
        ArchitectureContractError,
        match="must sum",
    ):
        architecture_record(
            searchable_component_count=516,
            component_type_counts={
                "attention_head": 4,
                "mlp_neuron": 511,
            },
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scientific_data", True),
        ("production_eligible", True),
    ],
)
def test_record_rejects_scientific_or_production_claims(
    field: str,
    value: bool,
) -> None:
    kwargs = {field: value}

    with pytest.raises(
        ArchitectureContractError,
        match=field,
    ):
        architecture_record(**kwargs)


def test_registry_rejects_builder_hash_mismatch() -> None:
    with pytest.raises(
        ArchitectureContractError,
        match="builder hash",
    ):
        registry().register(architecture_record(builder_sha256="2" * 64))


def test_builder_rejects_unsupported_architecture_combination() -> None:
    with pytest.raises(
        ArchitectureContractError,
        match="builder rejected architecture record",
    ):
        registry().register(architecture_record(activation="unsupported_activation"))


def test_registry_does_not_freeze_a_five_family_roster() -> None:
    values = registry()

    for index in range(6):
        record = architecture_record(
            family=f"technical{index}",
            name="variant",
            parameter_count=227_313 + index,
        )
        values.register(record)

    assert len(values.records()) == 6


def test_mapping_is_defensive_against_input_mutation() -> None:
    compatibility = {
        "task_protocol": "stage12p1-task/v1",
    }
    dimensions = {
        "n_layers": 1,
        "d_model": 128,
        "n_heads": 4,
        "d_head": 32,
    }
    counts = {
        "attention_head": 4,
        "mlp_neuron": 512,
    }

    record = ArchitectureRecord(
        family="technical",
        name="copy-check",
        version="v1",
        compatibility=compatibility,
        dimensions=dimensions,
        activation="relu",
        normalization=None,
        positional_embedding_type="standard",
        parameter_count=227_313,
        searchable_component_count=516,
        component_type_counts=counts,
        initialization_ref="technical-gpt2-init/v1",
        builder_ref="technical-transformer-builder/v1",
        builder_sha256=BUILDER_HASH,
    )

    compatibility["task_protocol"] = "mutated"
    dimensions["d_model"] = 999
    counts["mlp_neuron"] = 0

    assert record.compatibility["task_protocol"] == "stage12p1-task/v1"
    assert record.dimensions["d_model"] == 128
    assert record.component_type_counts["mlp_neuron"] == 512


def test_replacing_version_changes_architecture_identity() -> None:
    first = architecture_record()
    second = replace(first, version="v2")

    assert first.architecture_ref != second.architecture_ref


def test_registry_build_dispatches_through_bound_builder() -> None:
    value = architecture_record()
    values = registry()
    values.register(value)

    built = values.build(
        value.architecture_ref,
        seed=7,
        device="cpu",
    )

    assert built == {
        "architecture_ref": value.architecture_ref,
        "seed": 7,
        "device": "cpu",
    }


def test_task_compatibility_requires_literal_required_fields() -> None:
    record = architecture_record()

    validate_task_architecture_compatibility(
        record,
        {
            "task_protocol": "stage12p1-task/v1",
            "input_representation": "token_sequence",
        },
    )

    with pytest.raises(
        ArchitectureContractError,
        match="compatibility mismatch",
    ):
        validate_task_architecture_compatibility(
            record,
            {
                "input_representation": "different",
            },
        )

    with pytest.raises(
        ArchitectureContractError,
        match="missing required compatibility fields",
    ):
        validate_task_architecture_compatibility(
            record,
            {
                "unadvertised_requirement": True,
            },
        )


def test_parameter_formula_matches_frozen_predecessor_count() -> None:
    assert (
        transformer_parameter_count(
            n_layers=1,
            n_ctx=3,
            d_model=128,
            n_heads=4,
            d_head=32,
            d_mlp=512,
            d_vocab=114,
            d_vocab_out=113,
        )
        == 227_313
    )


def test_canonical_adapter_constructs_exact_frozen_predecessor() -> None:
    record = canonical_predecessor_record()
    builder = CanonicalPredecessorBuilder()
    builder.validate_record(record)

    model = builder.build(
        record=record,
        seed=3,
        device="cpu",
    )

    assert model.cfg.n_layers == 1
    assert model.cfg.d_model == 128
    assert model.cfg.n_heads == 4
    assert model.cfg.d_mlp == 512
    assert model.cfg.d_vocab == 114
    assert model.cfg.d_vocab_out == 113
    assert sum(parameter.numel() for parameter in model.parameters()) == 227_313


def test_canonical_adapter_rejects_mutated_architecture() -> None:
    canonical = canonical_predecessor_record()
    mutated = ArchitectureRecord(
        family=canonical.family,
        name=canonical.name,
        version=canonical.version,
        compatibility=canonical.compatibility,
        dimensions={
            **dict(canonical.dimensions),
            "d_model": 64,
            "n_heads": 4,
            "d_head": 16,
        },
        activation=canonical.activation,
        normalization=canonical.normalization,
        positional_embedding_type=canonical.positional_embedding_type,
        parameter_count=100_000,
        searchable_component_count=canonical.searchable_component_count,
        component_type_counts=canonical.component_type_counts,
        initialization_ref=canonical.initialization_ref,
        builder_ref=canonical.builder_ref,
        builder_sha256=canonical.builder_sha256,
    )

    with pytest.raises(
        ArchitectureContractError,
        match="canonical predecessor record mismatch",
    ):
        CanonicalPredecessorBuilder().validate_record(mutated)


def test_two_distinct_technical_variants_construct_from_one_builder() -> None:
    shared_compatibility = {
        "implementation": "transformer_lens",
        "input_representation": "token_sequence",
        "output_class_count": 113,
    }

    shallow_narrow = technical_transformer_record(
        family="technical",
        name="shallow-narrow",
        version="v1",
        n_layers=1,
        n_ctx=3,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_mlp=128,
        d_vocab=114,
        d_vocab_out=113,
        compatibility=shared_compatibility,
    )
    deeper = technical_transformer_record(
        family="technical",
        name="deeper",
        version="v1",
        n_layers=2,
        n_ctx=3,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_mlp=128,
        d_vocab=114,
        d_vocab_out=113,
        compatibility=shared_compatibility,
    )

    builder = TechnicalTransformerBuilder()
    first_model = builder.build(
        record=shallow_narrow,
        seed=5,
        device="cpu",
    )
    second_model = builder.build(
        record=deeper,
        seed=5,
        device="cpu",
    )

    assert shallow_narrow.architecture_ref != deeper.architecture_ref
    assert shallow_narrow.parameter_count != deeper.parameter_count
    assert shallow_narrow.searchable_component_count == 132
    assert deeper.searchable_component_count == 264
    assert first_model.cfg.n_layers == 1
    assert second_model.cfg.n_layers == 2


def test_technical_builder_rejects_record_count_tampering() -> None:
    record = technical_transformer_record(
        family="technical",
        name="tamper-check",
        version="v1",
        n_layers=2,
        n_ctx=3,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_mlp=128,
        d_vocab=114,
        d_vocab_out=113,
        compatibility={
            "implementation": "transformer_lens",
        },
    )

    tampered = ArchitectureRecord(
        family=record.family,
        name=record.name,
        version=record.version,
        compatibility=record.compatibility,
        dimensions=record.dimensions,
        activation=record.activation,
        normalization=record.normalization,
        positional_embedding_type=record.positional_embedding_type,
        parameter_count=record.parameter_count + 1,
        searchable_component_count=record.searchable_component_count,
        component_type_counts=record.component_type_counts,
        initialization_ref=record.initialization_ref,
        builder_ref=record.builder_ref,
        builder_sha256=record.builder_sha256,
    )

    with pytest.raises(
        ArchitectureContractError,
        match="parameter count does not match",
    ):
        TechnicalTransformerBuilder().validate_record(tampered)


def test_default_registry_has_builders_but_no_frozen_roster() -> None:
    values = default_technical_architecture_registry()

    assert values.records() == ()

    canonical = canonical_predecessor_record()
    values.register(canonical)
    built = values.build(
        canonical.architecture_ref,
        seed=0,
        device="cpu",
    )

    assert built.cfg.n_layers == 1
