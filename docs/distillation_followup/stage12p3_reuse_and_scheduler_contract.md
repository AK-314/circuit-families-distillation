# Stage 12-P3 reuse audit and scheduler-neutral contract

**Classification:** synthetic technical orchestration only
**Scientific data:** false
**Production eligible:** false

## Reuse classification

| Requirement | Classification | Authority and Stage 12-P3 treatment |
|---|---|---|
| Canonical condition identity | direct reuse | Stage 4 condition IDs remain opaque producer inputs; no competing serializer is introduced. |
| Attempt/retry seed coordinates | adapter | Stage 4 keeps attempt and retry outside logical identity. P3 carries an injected seed-namespace reference and records attempt evidence in execution state. |
| Job dependency validation | direct reuse and extension | Stage 5C topological, closure, collision, and deterministic-order semantics are retained; P3 generalizes job families and hash-bound producer references. |
| Isolated output roots and atomic publication | direct reuse and extension | Stage 5C path-safety and sealed-completion rules remain the model. P3 adds per-attempt roots and a durable claim token. |
| Lifecycle, resume, reproduction, inventory | adapter | Stage 7's explicit failed/unavailable inventory and portable roots are preserved through P3 operational projections and reconciliation. |
| P1 teacher/cache/seal outputs | adapter | Producers enter as opaque reference, interface-version, and SHA-256 triples. P3 never imports a concrete trainer. |
| P2 checkpoint/dense-output/release outputs | adapter | The discovery-release and sealed-student hashes are expected-input evidence, not scheduler state. |
| Stage 6A/6E ledgers and endpoints | direct consumer contract | Reducer jobs consume sealed ledger references; P3 never duplicates discovery or derives scientific endpoint direction. |
| Resource classes, priority classes, claims, leases, shedding | new policy-neutral code | Values are injected. The package defines validation and mechanics only. |
| Stage 15 roster, numeric profiles, Symbolica details, quotas | deferred | RD-012, RD-013, and RD-014 remain open for Stages 13–14. |

## Contract separation

`LogicalJobSpec` contains immutable producer interface versions, ordered dependency
identities, hash-bound opaque inputs/config, expected sealed outputs, resource and
priority references, protected tier, and a seed-namespace reference. Attempt,
retry, worker, claim, lease, backend job ID, and array index are absent from its
identity payload.

Scheduler submissions and observations are execution metadata. An adapter must
echo the logical identity unchanged. A scheduler `finished` observation is not
sealed success; only a validated, atomically published output manifest can cause
that state transition.

The durable lifecycle is `planned`, `blocked`, `ready`, `claimed`, `running`,
`succeeded`, `retryable_failure`, `terminal_failure`, or `shed_unavailable`.
`claimed` is distinct from `running` so a crash before the first heartbeat is
visible. Every attempt retains worker, token, lease, heartbeat, failure, seed,
and sealed-manifest evidence while remaining outside logical identity. State is
published atomically under a filesystem lock and an integrity hash; attempt
artifacts live beneath a validated `jobs/<job-id>/attempts/<attempt-index>` root.

Resource, priority, concurrency, retry, lease, tier, and shedding values are
injected technical records. A worker must match the exact native resource-class
reference in addition to satisfying generic quantities. Equal-priority jobs use
logical-job identity as the stable tie-break. Shedding can mark only unstarted,
explicitly optional work; protected jobs remain in the inventory even when a
capacity target cannot be met.

The local adapter executes only injected in-process fixture workers. The generic
array adapter emits a contiguous index mapping and portable script, keeps backend
job IDs outside logical identity, rejects stale observations, and represents
cancellation separately. Symbolica-specific submission fields remain deferred
injected adapter data for Stage 14; none are fabricated here.

Operational status permits counts by family, tier, lifecycle state, resource
class, and failure category. It rejects effect direction, endpoint values, and
scientific rankings. Every executable profile and report must retain
`scientific_data=false` and `production_eligible=false`.

## Downstream interfaces

Austin 4 may use logical job IDs, attempt roots, and sealed output manifests as
the compact-storage/export boundary. Alex 5 may inject the definitive job list,
tier rules, and policy references without modifying the scheduler core. Stage 14
may wrap `SchedulerSubmission`, `SchedulerObservation`, and
`SchedulerCancellation` with actual backend fields after qualification.
