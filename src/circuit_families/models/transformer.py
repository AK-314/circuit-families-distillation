"""Construction of the frozen modular-addition transformer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from circuit_families.config import validate_model_config
from circuit_families.seeds import seed_everything, validate_seed

EXPECTED_PARAMETER_COUNT = 227_313


def build_transformer(
    config: Mapping[str, Any],
    *,
    seed: int,
    device: str | torch.device,
) -> HookedTransformer:
    """Build the frozen one-layer TransformerLens model.

    The model has 114 input embeddings, including the equals token, but only
    113 output logits. Token 113 is therefore structurally impossible as an
    answer prediction.
    """

    validate_model_config(config)
    seed = validate_seed(seed)
    selected_device = torch.device(device)

    seed_everything(seed)

    model_values = config["model"]

    transformer_config = HookedTransformerConfig(
        n_layers=model_values["n_layers"],
        n_ctx=model_values["n_ctx"],
        d_model=model_values["d_model"],
        n_heads=model_values["n_heads"],
        d_head=model_values["d_head"],
        d_mlp=model_values["d_mlp"],
        act_fn=model_values["act_fn"],
        positional_embedding_type=model_values[
            "positional_embedding_type"
        ],
        attention_dir=model_values["attention_dir"],
        normalization_type=model_values["normalization_type"],
        d_vocab=model_values["d_vocab"],
        d_vocab_out=model_values["d_vocab_out"],
        dtype=torch.float32,
        device=str(selected_device),
        seed=seed,
        init_weights=model_values["init_weights"],
        init_mode=model_values["init_mode"],
        default_prepend_bos=model_values["default_prepend_bos"],
        tie_word_embeddings=model_values["tie_word_embeddings"],
    )

    model = HookedTransformer(transformer_config)

    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "Unexpected model parameter count: "
            f"expected {EXPECTED_PARAMETER_COUNT}, "
            f"received {parameter_count(model)}."
        )

    return model


def parameter_count(model: torch.nn.Module) -> int:
    """Return the total number of model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())
