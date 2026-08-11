"""Deterministic training-device selection."""

from __future__ import annotations

from typing import Any

import torch

DEVICE_PRIORITY = ("cuda", "cpu")
SUPPORTED_DEVICES = frozenset(DEVICE_PRIORITY)


def mps_is_available() -> bool:
    """Return whether PyTorch reports an available Apple MPS device."""

    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def resolve_device(override: str | None = None) -> torch.device:
    """Resolve a requested device or choose CUDA, then CPU."""

    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise ValueError("device override must be a non-empty string.")

        requested = override.strip().lower()

        if requested not in SUPPORTED_DEVICES:
            raise ValueError(
                "device override must be one of: cuda, cpu."
            )

        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested but is not available."
            )

        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def device_record(device: torch.device) -> dict[str, Any]:
    """Return JSON-safe information about a selected PyTorch device."""

    selected = torch.device(device)

    cuda_devices = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]

    record: dict[str, Any] = {
        "selected_device": str(selected),
        "device_type": selected.type,
        "device_index": selected.index,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
        "mps_available": mps_is_available(),
    }

    if selected.type == "cuda":
        index = selected.index
        if index is None:
            index = torch.cuda.current_device()
        record["selected_device_name"] = torch.cuda.get_device_name(index)
    elif selected.type == "mps":
        record["selected_device_name"] = "Apple Metal Performance Shaders"
    else:
        record["selected_device_name"] = "CPU"

    return record
