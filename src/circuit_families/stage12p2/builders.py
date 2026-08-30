"""Stage 12-P2 canonical adapter and technical transformer variants."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from circuit_families.config import load_model_config
from circuit_families.models.transformer import (
    EXPECTED_PARAMETER_COUNT,
    build_transformer,
    parameter_count,
)
from circuit_families.seeds import seed_everything, validate_seed
from circuit_families.stage12p2.architecture import (
    ArchitectureContractError,
    ArchitectureRecord,
    BuilderDescriptor,
)

CANONICAL_PREDECESSOR_BUILDER_REF = "stage12p2-canonical-predecessor-builder/v1"
TECHNICAL_TRANSFORMER_BUILDER_REF = "stage12p2-technical-transformer-builder/v1"
CANONICAL_INITIALIZATION_REF = "predecessor-gpt2-initialization/v1"
TECHNICAL_INITIALIZATION_REF = "technical-gpt2-initialization/v1"


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CANONICAL_PREDECESSOR_BUILDER_SHA256 = _semantic_sha256(
    {
        "builder_ref": CANONICAL_PREDECESSOR_BUILDER_REF,
        "implementation": (
            "delegate_without_config_relaxation_to_"
            "circuit_families.models.transformer.build_transformer"
        ),
        "model_config": "configs/model.yaml",
        "record_contract": "stage12p2-architecture/v1",
    }
)

TECHNICAL_TRANSFORMER_BUILDER_SHA256 = _semantic_sha256(
    {
        "builder_ref": TECHNICAL_TRANSFORMER_BUILDER_REF,
        "implementation": "technical_transformerlens_dimension_adapter",
        "attention_direction": "causal",
        "dtype": "float32",
        "dropout": 0.0,
        "tie_word_embeddings": False,
        "default_prepend_bos": False,
        "init_mode": "gpt2",
        "record_contract": "stage12p2-architecture/v1",
        "scientific_data": False,
        "production_eligible": False,
    }
)


def transformer_parameter_count(
    *,
    n_layers: int,
    n_ctx: int,
    d_model: int,
    n_heads: int,
    d_head: int,
    d_mlp: int,
    d_vocab: int,
    d_vocab_out: int,
) -> int:
    """Return the exact no-normalisation TransformerLens parameter count."""
    values = {
        "n_layers": n_layers,
        "n_ctx": n_ctx,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_head": d_head,
        "d_mlp": d_mlp,
        "d_vocab": d_vocab,
        "d_vocab_out": d_vocab_out,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ArchitectureContractError(f"{name} must be a positive integer")

    if n_heads * d_head != d_model:
        raise ArchitectureContractError("n_heads multiplied by d_head must equal d_model")

    embedding_parameters = d_vocab * d_model
    positional_parameters = n_ctx * d_model

    attention_parameters_per_layer = 4 * n_heads * d_model * d_head + 3 * n_heads * d_head + d_model
    mlp_parameters_per_layer = 2 * d_model * d_mlp + d_mlp + d_model

    unembedding_parameters = d_model * d_vocab_out + d_vocab_out

    return (
        embedding_parameters
        + positional_parameters
        + n_layers * (attention_parameters_per_layer + mlp_parameters_per_layer)
        + unembedding_parameters
    )


def _expected_component_counts(
    dimensions: Mapping[str, int],
) -> dict[str, int]:
    n_layers = dimensions["n_layers"]
    n_heads = dimensions["n_heads"]
    d_mlp = dimensions["d_mlp"]
    return {
        "attention_head": n_layers * n_heads,
        "mlp_neuron": n_layers * d_mlp,
    }


def canonical_predecessor_record() -> ArchitectureRecord:
    """Return the canonical predecessor architecture through a new adapter."""
    return ArchitectureRecord(
        family="predecessor",
        name="matched",
        version="v1",
        compatibility={
            "implementation": "transformer_lens",
            "input_representation": "token_sequence",
            "output_class_count": 113,
        },
        dimensions={
            "n_layers": 1,
            "n_ctx": 3,
            "d_model": 128,
            "n_heads": 4,
            "d_head": 32,
            "d_mlp": 512,
            "d_vocab": 114,
            "d_vocab_out": 113,
        },
        activation="relu",
        normalization=None,
        positional_embedding_type="standard",
        parameter_count=EXPECTED_PARAMETER_COUNT,
        searchable_component_count=516,
        component_type_counts={
            "attention_head": 4,
            "mlp_neuron": 512,
        },
        initialization_ref=CANONICAL_INITIALIZATION_REF,
        builder_ref=CANONICAL_PREDECESSOR_BUILDER_REF,
        builder_sha256=CANONICAL_PREDECESSOR_BUILDER_SHA256,
        scientific_data=False,
        production_eligible=False,
    )


def technical_transformer_record(
    *,
    family: str,
    name: str,
    version: str,
    n_layers: int,
    n_ctx: int,
    d_model: int,
    n_heads: int,
    d_head: int,
    d_mlp: int,
    d_vocab: int,
    d_vocab_out: int,
    compatibility: Mapping[str, Any],
    activation: str = "relu",
    normalization: str | None = None,
    positional_embedding_type: str | None = "standard",
) -> ArchitectureRecord:
    """Construct one technical-only transformer architecture record.

    This helper defines no roster. Callers may create any number of technical
    variants that satisfy the builder's mechanics-only contract.
    """
    dimensions = {
        "n_layers": n_layers,
        "n_ctx": n_ctx,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_head": d_head,
        "d_mlp": d_mlp,
        "d_vocab": d_vocab,
        "d_vocab_out": d_vocab_out,
    }

    component_counts = _expected_component_counts(dimensions)
    searchable_count = sum(component_counts.values())

    count = transformer_parameter_count(
        n_layers=n_layers,
        n_ctx=n_ctx,
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        d_mlp=d_mlp,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out,
    )

    return ArchitectureRecord(
        family=family,
        name=name,
        version=version,
        compatibility=compatibility,
        dimensions=dimensions,
        activation=activation,
        normalization=normalization,
        positional_embedding_type=positional_embedding_type,
        parameter_count=count,
        searchable_component_count=searchable_count,
        component_type_counts=component_counts,
        initialization_ref=TECHNICAL_INITIALIZATION_REF,
        builder_ref=TECHNICAL_TRANSFORMER_BUILDER_REF,
        builder_sha256=TECHNICAL_TRANSFORMER_BUILDER_SHA256,
        scientific_data=False,
        production_eligible=False,
    )


class CanonicalPredecessorBuilder:
    """Adapter that preserves the frozen predecessor builder unchanged."""

    descriptor = BuilderDescriptor(
        builder_ref=CANONICAL_PREDECESSOR_BUILDER_REF,
        implementation_sha256=CANONICAL_PREDECESSOR_BUILDER_SHA256,
    )

    def validate_record(self, record: ArchitectureRecord) -> None:
        expected = canonical_predecessor_record()

        fields = (
            "family",
            "name",
            "version",
            "compatibility",
            "dimensions",
            "activation",
            "normalization",
            "positional_embedding_type",
            "parameter_count",
            "searchable_component_count",
            "component_type_counts",
            "initialization_ref",
            "builder_ref",
            "builder_sha256",
            "scientific_data",
            "production_eligible",
        )
        mismatches = [
            field for field in fields if getattr(record, field) != getattr(expected, field)
        ]
        if mismatches:
            raise ArchitectureContractError(
                "canonical predecessor record mismatch for fields: " + ", ".join(mismatches)
            )

    def build(
        self,
        *,
        record: ArchitectureRecord,
        seed: int,
        device: str | torch.device,
    ) -> HookedTransformer:
        self.validate_record(record)
        repo_root = Path(__file__).resolve().parents[3]
        config = load_model_config(repo_root / "configs/model.yaml")
        return build_transformer(
            config,
            seed=seed,
            device=device,
        )


class TechnicalTransformerBuilder:
    """Mechanics-only TransformerLens builder for heterogeneous fixtures."""

    descriptor = BuilderDescriptor(
        builder_ref=TECHNICAL_TRANSFORMER_BUILDER_REF,
        implementation_sha256=TECHNICAL_TRANSFORMER_BUILDER_SHA256,
    )

    def validate_record(self, record: ArchitectureRecord) -> None:
        if not isinstance(record, ArchitectureRecord):
            raise ArchitectureContractError("record must be ArchitectureRecord")
        if record.builder_ref != self.descriptor.builder_ref:
            raise ArchitectureContractError("technical record builder_ref mismatch")
        if record.builder_sha256 != self.descriptor.implementation_sha256:
            raise ArchitectureContractError("technical record builder hash mismatch")
        if record.initialization_ref != TECHNICAL_INITIALIZATION_REF:
            raise ArchitectureContractError("technical initialization reference mismatch")
        if record.activation not in {"relu", "gelu"}:
            raise ArchitectureContractError("unsupported technical activation")
        if record.normalization is not None:
            raise ArchitectureContractError(
                "technical builder currently supports normalization=None only"
            )
        if record.positional_embedding_type != "standard":
            raise ArchitectureContractError(
                "technical builder requires standard positional embeddings"
            )

        required_dimensions = {
            "n_layers",
            "n_ctx",
            "d_model",
            "n_heads",
            "d_head",
            "d_mlp",
            "d_vocab",
            "d_vocab_out",
        }
        if set(record.dimensions) != required_dimensions:
            raise ArchitectureContractError("technical transformer dimensions keys mismatch")

        dimensions = record.dimensions

        if dimensions["n_heads"] * dimensions["d_head"] != dimensions["d_model"]:
            raise ArchitectureContractError("n_heads multiplied by d_head must equal d_model")
        if dimensions["d_vocab_out"] >= dimensions["d_vocab"]:
            raise ArchitectureContractError("d_vocab_out must be smaller than d_vocab")

        expected_components = _expected_component_counts(dimensions)
        if dict(record.component_type_counts) != expected_components:
            raise ArchitectureContractError(
                "component type counts do not match architecture dimensions"
            )
        if record.searchable_component_count != sum(expected_components.values()):
            raise ArchitectureContractError("searchable component count does not match dimensions")

        expected_parameter_count = transformer_parameter_count(
            **{
                key: dimensions[key]
                for key in (
                    "n_layers",
                    "n_ctx",
                    "d_model",
                    "n_heads",
                    "d_head",
                    "d_mlp",
                    "d_vocab",
                    "d_vocab_out",
                )
            }
        )
        if record.parameter_count != expected_parameter_count:
            raise ArchitectureContractError(
                "parameter count does not match architecture dimensions"
            )

    def build(
        self,
        *,
        record: ArchitectureRecord,
        seed: int,
        device: str | torch.device,
    ) -> HookedTransformer:
        self.validate_record(record)
        seed = validate_seed(seed)
        selected_device = torch.device(device)
        seed_everything(seed)

        dimensions = record.dimensions

        config = HookedTransformerConfig(
            n_layers=dimensions["n_layers"],
            n_ctx=dimensions["n_ctx"],
            d_model=dimensions["d_model"],
            n_heads=dimensions["n_heads"],
            d_head=dimensions["d_head"],
            d_mlp=dimensions["d_mlp"],
            act_fn=record.activation,
            positional_embedding_type=record.positional_embedding_type,
            attention_dir="causal",
            normalization_type=record.normalization,
            d_vocab=dimensions["d_vocab"],
            d_vocab_out=dimensions["d_vocab_out"],
            dtype=torch.float32,
            device=str(selected_device),
            seed=seed,
            init_weights=True,
            init_mode="gpt2",
            default_prepend_bos=False,
            tie_word_embeddings=False,
        )

        model = HookedTransformer(config)
        actual_parameter_count = parameter_count(model)
        if actual_parameter_count != record.parameter_count:
            raise ArchitectureContractError(
                "constructed model parameter count does not match record: "
                f"record={record.parameter_count}, "
                f"actual={actual_parameter_count}"
            )
        return model


def default_technical_architecture_registry():
    """Return a registry with builders only; it freezes no architecture roster."""
    from circuit_families.stage12p2.architecture import ArchitectureRegistry

    canonical = CanonicalPredecessorBuilder()
    technical = TechnicalTransformerBuilder()
    return ArchitectureRegistry(
        builders={
            canonical.descriptor.builder_ref: canonical,
            technical.descriptor.builder_ref: technical,
        }
    )
