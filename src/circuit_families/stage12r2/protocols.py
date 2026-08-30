"""Capability-based Stage 12-R2 interfaces.

These protocols deliberately avoid any concrete teacher/student model class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from circuit_families.stage12r2.contracts import BasisContract, BasisMask


@runtime_checkable
class BasisMetadataProvider(Protocol):
    def stage12r2_basis_contract(self) -> BasisContract:
        """Return the auditable basis contract exposed by this capability."""


@runtime_checkable
class ActivationInterceptionCapability(Protocol):
    def stage12r2_activation_shape(
        self,
        *,
        subspace_identity: str,
    ) -> tuple[int, ...]:
        """Return the eligible activation shape without executing science."""


@runtime_checkable
class BasisMaskApplicationCapability(Protocol):
    def stage12r2_apply_basis_mask(
        self,
        *,
        basis: BasisContract,
        mask: BasisMask,
    ) -> None:
        """Apply a validated technical mask at the basis-defined intervention point."""
