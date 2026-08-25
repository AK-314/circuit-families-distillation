#!/usr/bin/env python3
"""Read-only Stage 6E technical Endpoint 2 validator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from circuit_families.stage6a.models import canonical_mask_identity  # noqa: E402
from circuit_families.stage6e import (  # noqa: E402
    ExactCandidateEvidence,
    build_compatibility_graph,
    load_technical_policy,
    qualify_and_deduplicate,
    recompute_endpoint2,
)

POLICY_PATH = ROOT / "followup/configs/stage6e/technical_endpoint2_policy_v1.json"


def _mask(indices: tuple[int, ...]) -> tuple[int, ...]:
    retained = set(indices)
    return tuple(1 if i in retained else 0 for i in range(516))


def _evidence(name: str, indices: tuple[int, ...], fidelity: float):
    policy = load_technical_policy(POLICY_PATH)
    mask = _mask(indices)
    return ExactCandidateEvidence(
        model_id="synthetic-model",
        discovery_method_id="synthetic-method",
        discovery_config_id="synthetic-config",
        source_budget_reference=policy.source_budget_reference,
        fidelity_metric_reference=policy.fidelity_metric_reference,
        component_basis_reference=policy.component_basis_reference,
        component_basis_size=516,
        mask=mask,
        mask_identity=canonical_mask_identity(indices),
        exact_fidelity=fidelity,
        proposal_reference=f"proposal-{name}",
        exact_evaluation_reference=f"eval-{name}",
        source_ledger_reference=f"ledger-{name}",
        source_ledger_hash="a" * 64,
        recomputed_ledger_hash="a" * 64,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", required=True)
    args = parser.parse_args()

    if not args.validate_only:
        return 2

    policy = load_technical_policy(POLICY_PATH)
    evidence = (
        _evidence("a", (0, 1), 0.8),
        _evidence("b", (2, 3), 0.8),
        _evidence("c", (1, 2), 0.8),
        _evidence("a-dup", (0, 1), 0.8),
    )

    qualification = qualify_and_deduplicate(
        evidence,
        policy,
        model_id="synthetic-model",
        discovery_method_id="synthetic-method",
        discovery_config_id="synthetic-config",
    )
    result = recompute_endpoint2(qualification, policy)
    recomputed = recompute_endpoint2(qualification, policy)

    passed = result.to_record() == recomputed.to_record()

    print(f"policy_hash={policy.policy_hash}")
    print(f"input_hash={result.input_hash}")
    print(f"graph_hash={result.graph_hash}")
    print(f"raw_count={result.raw_candidate_count}")
    print(f"unique_count={result.unique_candidate_count}")
    print(f"qualified_count={result.qualified_candidate_count}")
    graph = build_compatibility_graph(
        qualification.qualified_candidates,
        policy,
    )
    print(f"edge_count={len(graph.compatible_edges)}")
    print(
        "selected_members="
        + ",".join(member.mask_identity for member in result.selected_members)
    )
    print(f"packing_lower_bound={result.packing_lower_bound}")
    print(f"recomputation={'PASS' if passed else 'FAIL'}")
    print("semantics=procedure_dependent_packing_lower_bound")
    print("production_default=false")
    print("scientific_data=false")
    print(f"STATUS={'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
