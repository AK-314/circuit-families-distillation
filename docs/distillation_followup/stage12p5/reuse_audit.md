# Stage 12-P5 Fourier interchange reuse audit

**Implementation base:** `7dd9c9ec7357bc70b0fe8e1f275a3be750d85d48`

**Scientific data:** false

**Production eligible:** false

## Boundary

This package is a policy-neutral technical runner. It does not select a
production pair, location, representation, mode, alignment, capacity rule,
outcome, trial count, aggregation, threshold, seed roster, or success rule.
RD-011 remains open. The validation command uses only tiny in-memory arrays and
does not load registered checkpoints, dense-output caches, or private artifacts.

## Reused producers and consumers

| Requirement | Existing authority reused | Stage 12-P5 use |
|---|---|---|
| Canonical JSON and hashes | `stage12p3.records.canonical_json_bytes` and `canonical_sha256` | Hashes pair, trial, comparison, alignment, payload, accounting, result, and report material without a parallel serializer. |
| Model and architecture references | Stage 12-P2 `ArchitectureRecord`, builder, checkpoint, component, and dense-output contracts | P5 stores opaque versioned references and their hashes; architecture compatibility is explicit at typed locations. |
| Activation locations | Stage 12-P2 component `hook_name`, `activation_axis`, architecture reference, and builder compatibility surfaces | The injected P5 activation adapter requires declared architecture, location, hook, layout, and shape. Equal tensor shape is insufficient. |
| Root identities and seeds | Stage 4 condition identity and `seed-derivation/v1`; shared `seeds.numpy_generator` | The trial stores the upstream root seed namespace. P5 derives domain-separated sub-seeds from the canonical trial identity and uses the shared PCG64 constructor. |
| Attempt, retry, dependency, and sealed-output identity | Stage 12-P3 `LogicalJobSpec`, `HashBoundReference`, `OutputContract`, and durable sealed-output conventions | `build_logical_job` binds a P5 trial to one P3 job. Execution records retain attempt/retry coordinates outside scientific identity. |
| Compact records and deterministic publication | Stage 12-P4 `MetricSchema`, `MetricLedgerWriter`/`write_ledger`, `CodecProfile`, and atomic codec | Final condition rows use a P4 compressed ledger; canonical reports and resume records use the P4 atomic codec. |
| Failure and lifecycle semantics | Stage 7 failure/inventory patterns and P3/P4 terminal states | Failed, unavailable, censored, nonfinite, interrupted, resumed, and complete conditions remain explicit. |
| Exact evidence patterns | Stage 6A exact ledger and Stage 6E result/proof separation | Capacity accounting, execution evidence, raw outcomes, and reduction are separate hash-bound records. |
| Historical Fourier semantics | `analysis/fourier_sanity_check.py` | NumPy FFT normalization and explicit convention metadata are reused where valid. Diagnostic classifications and scientific interpretations are not reused. |

## New Stage 12-P5 interfaces

- `PairContract` retains source-teacher and recipient-student roles plus
  outcome-independent selection evidence.
- `TrialContract` binds input, location, Fourier, mode, alignment, capacity,
  outcome-adapter, seed, and complete six-condition identities.
- `ExtractedCoordinateState` records values separately from complete provenance,
  shapes, dtypes, norms, and hashes.
- `AlignmentPlan` is an explicit pair-bound matrix with fit boundary, ranks,
  residual, and hash; it is never an implicit tensor cast.
- `CapacityContract` defines an operational allowance over real degrees of
  freedom, support, rank, precision, side information, external identifiers,
  recipient shape, and write budget. It explicitly is not a universal
  information-theoretic channel-capacity claim.
- `InterventionPayload` records the values actually inserted and the one
  permitted condition-specific difference.
- `ConditionExecutionRecord` separates hook/write evidence, raw technical
  outcome, lifecycle state, and failure evidence.
- `ComparisonSetResult` seals only a complete aligned-plus-five-controls
  inventory. Structural control failure produces six explicit unavailable
  records rather than a reduced comparison.

## Control invariants

The builder constructs all six payloads before execution:

1. aligned Fourier interchange uses the pair-bound alignment plan;
2. wrong-mode control requires a valid distinct injected mode and never falls
   back to the aligned mode;
3. shuffled coefficients use a deterministic non-identity permutation while
   recording marginal-value and norm preservation;
4. mismatched input uses a deterministic derangement and proves no accidental
   source/recipient match;
5. equal-norm random state uses a domain-separated trial seed, explicit
   tolerance, and deterministic zero-norm behavior;
6. ordinary activation patching records its injected shape adapter and applies
   no Fourier alignment.

Every payload is independently checked against the same capacity-contract hash.
Hidden mode labels, input identities, alignment matrices, coordinate indices,
random seeds, or payload lengths are treated as undeclared side channels.

## Outcome neutrality

The reducer reconstructs condition rows and lifecycle counts only. It contains
no Stage 13 superiority rule, statistical test, aggregation, threshold,
directional label, shared-abstraction claim, or uniqueness claim.
