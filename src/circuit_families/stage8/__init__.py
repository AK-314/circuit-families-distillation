"""Stage 8 technical scientific-edge-case validation."""

from .edge_cases import (
    EDGE_CASE_SCHEMA_VERSION,
    EdgeCaseMatrixError,
    run_edge_case_matrix,
)

__all__ = [
    "EDGE_CASE_SCHEMA_VERSION",
    "EdgeCaseMatrixError",
    "run_edge_case_matrix",
]
