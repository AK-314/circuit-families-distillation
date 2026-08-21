from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_families.stage6a.endpoint import reduce_endpoint1
from circuit_families.stage6a.models import (
    ExactEvaluationEntry,
    TerminationStatus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only synthetic Stage 6A Endpoint 1 recompute."
    )
    parser.add_argument("input", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = json.loads(args.input.read_text())

    evaluations = tuple(
        ExactEvaluationEntry(
            mask_identity=str(item["mask_identity"]),
            retained_count=int(item["retained_count"]),
            retained_proportion=float(item["retained_proportion"]),
            fidelity=float(item["fidelity"]),
            qualifies=bool(item["qualifies"]),
            evaluation_order=int(item["evaluation_order"]),
            exact_budget_charge=int(item["exact_budget_charge"]),
        )
        for item in payload["evaluations"]
    )

    termination = TerminationStatus(
        status=str(payload["termination"]["status"]),
        procedure_censored=bool(payload["termination"]["procedure_censored"]),
    )

    result = reduce_endpoint1(
        evaluations,
        termination=termination,
    )

    print(json.dumps({
        "retained_proportion": result.retained_proportion,
        "mask_identity": result.mask_identity,
        "global_minimum_claim": result.global_minimum_claim,
        "termination_status": result.termination_status,
        "procedure_censored": result.procedure_censored,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
