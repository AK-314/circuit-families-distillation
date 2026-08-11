"""Exact zero ablation of frozen attention heads and MLP neurons."""

from __future__ import annotations

from typing import Any

import torch
from transformer_lens import HookedTransformer

from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_COUNT,
    ATTENTION_HEAD_HOOK_NAME,
    MLP_NEURON_COUNT,
    MLP_NEURON_HOOK_NAME,
    ComponentMask,
    validate_frozen_architecture,
)
from circuit_families.training.metrics import OUTPUT_CLASS_COUNT


def _require_component_mask(mask: ComponentMask) -> ComponentMask:
    if not isinstance(mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask.")

    return mask


def apply_attention_head_mask(
    activation: torch.Tensor,
    mask: ComponentMask,
) -> torch.Tensor:
    """Multiply complete per-head output vectors by a binary mask."""

    _require_component_mask(mask)

    if not isinstance(activation, torch.Tensor):
        raise TypeError("attention-head activation must be a tensor.")

    if activation.ndim != 4:
        raise ValueError(
            "attention-head activation must have shape "
            "(batch, position, head, d_head)."
        )

    if activation.shape[2] != ATTENTION_HEAD_COUNT:
        raise ValueError(
            "attention-head activation must contain exactly "
            f"{ATTENTION_HEAD_COUNT} heads."
        )

    values = torch.tensor(
        mask.attention_head_mask,
        dtype=activation.dtype,
        device=activation.device,
    ).view(1, 1, ATTENTION_HEAD_COUNT, 1)

    return activation * values


def apply_mlp_neuron_mask(
    activation: torch.Tensor,
    mask: ComponentMask,
) -> torch.Tensor:
    """Multiply individual post-ReLU neuron activations by a binary mask."""

    _require_component_mask(mask)

    if not isinstance(activation, torch.Tensor):
        raise TypeError("MLP-neuron activation must be a tensor.")

    if activation.ndim != 3:
        raise ValueError(
            "MLP-neuron activation must have shape "
            "(batch, position, d_mlp)."
        )

    if activation.shape[2] != MLP_NEURON_COUNT:
        raise ValueError(
            "MLP-neuron activation must contain exactly "
            f"{MLP_NEURON_COUNT} neurons."
        )

    values = torch.tensor(
        mask.mlp_neuron_mask,
        dtype=activation.dtype,
        device=activation.device,
    ).view(1, 1, MLP_NEURON_COUNT)

    return activation * values


def validate_mask_model(model: HookedTransformer) -> None:
    """Validate the frozen architecture and required hook locations."""

    if not isinstance(model, HookedTransformer):
        raise TypeError("model must be a HookedTransformer.")

    validate_frozen_architecture(
        n_heads=model.cfg.n_heads,
        d_mlp=model.cfg.d_mlp,
    )

    if model.cfg.n_layers != 1:
        raise ValueError("Masked model must contain exactly one layer.")

    required_hooks = {
        ATTENTION_HEAD_HOOK_NAME,
        MLP_NEURON_HOOK_NAME,
    }
    missing = sorted(required_hooks.difference(model.hook_dict))

    if missing:
        raise ValueError(
            "Masked model is missing required hooks: "
            + ", ".join(missing)
        )


def _validate_inputs(inputs: torch.Tensor) -> None:
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a PyTorch tensor.")

    if inputs.ndim != 2:
        raise ValueError(
            "inputs must have shape (batch_size, sequence_length)."
        )

    if inputs.shape[0] == 0:
        raise ValueError("inputs batch dimension must not be empty.")

    if inputs.shape[1] == 0:
        raise ValueError("inputs sequence dimension must not be empty.")

    if inputs.dtype != torch.long:
        raise TypeError("inputs must have dtype torch.long.")


def masked_model_logits(
    model: HookedTransformer,
    inputs: torch.Tensor,
    mask: ComponentMask,
) -> torch.Tensor:
    """Evaluate one model under exact head and neuron zero ablation."""

    validate_mask_model(model)
    _validate_inputs(inputs)
    _require_component_mask(mask)

    def head_hook(
        activation: torch.Tensor,
        hook: Any,
    ) -> torch.Tensor:
        if hook.name != ATTENTION_HEAD_HOOK_NAME:
            raise RuntimeError("Attention-head hook name changed.")
        return apply_attention_head_mask(activation, mask)

    def neuron_hook(
        activation: torch.Tensor,
        hook: Any,
    ) -> torch.Tensor:
        if hook.name != MLP_NEURON_HOOK_NAME:
            raise RuntimeError("MLP-neuron hook name changed.")
        return apply_mlp_neuron_mask(activation, mask)

    was_training = model.training
    model.eval()

    try:
        with model.hooks(
            fwd_hooks=[
                (ATTENTION_HEAD_HOOK_NAME, head_hook),
                (MLP_NEURON_HOOK_NAME, neuron_hook),
            ]
        ):
            with torch.inference_mode():
                logits = model(inputs)
    finally:
        model.train(was_training)

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError(
            "Masked model must return three-dimensional logits."
        )

    if logits.shape[2] != OUTPUT_CLASS_COUNT:
        raise ValueError(
            f"Masked logits must contain exactly {OUTPUT_CLASS_COUNT} "
            "output classes."
        )

    return logits
