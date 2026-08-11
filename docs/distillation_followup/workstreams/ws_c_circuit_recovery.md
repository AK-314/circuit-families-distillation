# Workstream C — Predictive Fidelity, Discovery, and Endpoints

## Mission

Generalize the validated masking/search substrate to centred-logit fidelity, multiple discovery methods, method-aware budget accounting, and the two frozen endpoints.

## Implementation order

1. Add per-input class centring and a streaming, numerically stable centred-logit fidelity accumulator.
2. Validate invariance to per-input additive-logit shifts.
3. Generalize mask-evaluation records so primary fidelity is named and versioned rather than assumed to be top-one agreement.
4. Insert and exactly evaluate the intact mask before every search.
5. Implement endpoint 1 as the minimum qualifying proportion over the exact-evaluation ledger plus the intact mask.
6. Define one discovery adapter interface covering proposals, native budget, exact-evaluation allowance, restarts, termination, and trajectory records.
7. Adapt the existing greedy deletion machinery to the new objective.
8. Adapt the diversity-forced machinery and any additional frozen method without sharing method-native budget units.
9. Deduplicate valid circuit masks before packing.
10. Construct the compatibility graph under the frozen overlap cutoff and compute the frozen deterministic maximum separated subset.
11. Implement endpoint 2 with an explicit zero outcome.
12. Add failure, exhaustion, cap, and recomputation tests.

## Required invariants

- The intact mask has fidelity 1.0 and proportion 1.0.
- Endpoint 1 is always defined for a valid dense reference.
- Only exact full-domain evaluations can qualify a mask for either endpoint.
- Endpoint 2 includes only masks at or below the frozen component-proportion cap.
- The packing result is invariant to discovery proposal order.
- Native optimization and exact evaluation counts are stored separately.
- Every method uses the same final exact evaluator but retains its own native budget.
- Phase comparisons never depend on a method silently changing budgets between cells.

## Deliverables

- Centred-logit fidelity implementation.
- Versioned exact-evaluation ledger.
- Discovery adapter and method integrations.
- Endpoint reducers.
- Packing solver and overlap validation.
- Budget and failure tests.

## Acceptance gate

Synthetic and real-model fixtures reproduce both endpoints from ledgers alone, including the endpoint-1 value 1.0 and endpoint-2 value zero in forced failure cases.

## Interfaces

- Consumes sealed teacher/student models and schemas from A and B.
- Supplies immutable search ledgers and endpoint records to D.
