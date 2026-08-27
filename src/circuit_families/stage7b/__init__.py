"""Stage 7B registered-checkpoint bridge.

This package contains only the narrow bridge from an exact registered physical
teacher identity into accepted Stage 3–7 interfaces.  It defines no scientific
default, discovery method, trainer, fidelity metric, endpoint policy, or Stage 8
behavior.
"""

from .registered_fixture import (
    RegisteredFixtureBindings,
    RegisteredFixtureError,
    RegisteredFixtureIdentity,
    RegisteredFixtureRun,
    canonical_modular_addition_domain,
    centred_logits,
    load_registered_fixture_request,
    run_registered_fixture,
    validate_registered_fixture_identity,
)

__all__ = [
    "RegisteredFixtureBindings",
    "RegisteredFixtureError",
    "RegisteredFixtureIdentity",
    "RegisteredFixtureRun",
    "canonical_modular_addition_domain",
    "centred_logits",
    "load_registered_fixture_request",
    "run_registered_fixture",
    "validate_registered_fixture_identity",
]
