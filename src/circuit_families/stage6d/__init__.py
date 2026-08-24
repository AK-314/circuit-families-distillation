"""Stage 6D discovery adapters and budget-ledger contracts."""

from .adapters import deterministic_seed_evidence
from .budgets import (
    ExactBudgetEvent,
    ExactBudgetExhausted,
    NativeBudgetEvent,
    NativeBudgetExhausted,
    NativeBudgetLedger,
    Stage6AExactEvaluationBridge,
)
from .diversity import DiversityForcedAdapter
from .greedy import GreedyDeletionAdapter
from .models import (
    PRODUCTION_ELIGIBLE,
    RESOURCE_WARNING,
    TECHNICAL_ONLY,
    UNRESOLVED_DECISIONS,
    DeterministicSeedEvidence,
    DiscoveryAdapter,
    DiscoveryRequest,
    DiscoveryResult,
    TechnicalDiscoveryProfile,
    TrajectoryEvent,
)
from .profiles import load_technical_profiles, technical_profile_from_record

__all__ = [
    "PRODUCTION_ELIGIBLE",
    "RESOURCE_WARNING",
    "TECHNICAL_ONLY",
    "UNRESOLVED_DECISIONS",
    "DeterministicSeedEvidence",
    "Stage6AExactEvaluationBridge",
    "NativeBudgetLedger",
    "NativeBudgetExhausted",
    "NativeBudgetEvent",
    "ExactBudgetExhausted",
    "ExactBudgetEvent",
    "DiscoveryAdapter",
    "deterministic_seed_evidence",
    "DiversityForcedAdapter",
    "GreedyDeletionAdapter",
    "DiscoveryRequest",
    "DiscoveryResult",
    "TechnicalDiscoveryProfile",
    "TrajectoryEvent",
    "load_technical_profiles",
    "technical_profile_from_record",
]
