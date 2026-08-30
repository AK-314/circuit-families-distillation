# Alex 2 — Stage 12-R1 independent sparse-mask discovery

## Paste this entire document into one fresh Chat-mode task

Repository: `AK-314/circuit-families-distillation`

Local clone convention: `~/Projects/circuit-families-distillation`

Required scientific authority floor:

```text
d36f1b442ab7b783f3211377303a2981fc0d00e3
```

Required branch:

```text
feat/stage-12r1-independent-discovery
```

Create the branch from the current `origin/main` after this handoff is merged.
Part A must prove that `d36f1b442ab7b783f3211377303a2981fc0d00e3`
is an ancestor, that current local `main` exactly matches `origin/main`, and
that both Barrier 1 packages are present. Record current `origin/main` as the
implementation base.

## Mission

Implement and validate an algorithmically independent continuous/stochastic
sparse-mask discovery family. It must integrate with the existing exact mask
evaluation and endpoint ledgers without pretending that its native optimization
budget is equivalent to the inherited greedy/diversity family.

This is technical Stage 12 implementation. It does not select the final
production algorithm/version, regularization, thresholds, optimization budget,
restart count, or exact-evaluation allowance. Those remain Stage 13 decisions.

Austin simultaneously implements multi-architecture student support. This task
must consume shared dense-model/component/evaluation protocols rather than
assuming only one concrete student architecture.

## Authorities — read completely before acting

1. `docs/distillation_followup/stage11_post_red_team_design_resolution.md`
2. `followup/configs/stage11_post_red_team_design_candidates_v1.json`
3. `followup/manifests/stage11_red_team_resolution_v1.json`
4. `docs/distillation_followup/post_red_team_protocol_amendment.md`
5. `docs/distillation_followup/distillation_implementation_post_red_team.md`
6. `docs/distillation_followup/stage10_discovery_compute_benchmark.md`
7. existing Stage 5A, 6A, 6D, 6E, 7, and 10 implementation/tests;
8. `docs/distillation_followup/handoffs/post_red_team/handoff_sequence.md`.

The Stage 11 record requires two genuinely different discovery families but
leaves the production algorithm and numbers open. The inherited discrete method
must not be rewritten here.

## Scientific boundary

Permitted:

- synthetic logits, masks, component models, and objective surfaces;
- tiny technical neural networks and toy domains;
- differentiability, optimization, resume, determinism, runtime, memory, and
  storage diagnostics;
- method-recovery tests where the optimum is constructed in advance;
- explicitly excluded technical fixture endpoints.

Prohibited:

- running any registered teacher or student checkpoint;
- comparing phases, hard/soft conditions, architectures, or teacher seeds;
- choosing method settings from scientific recoverability;
- freezing RD-005, RD-006, RD-007, RD-008, RD-009, RD-012, or RD-014;
- describing technical recovery as evidence for the project hypothesis.

Every executable profile must say `scientific_data=false` and
`production_eligible=false`.

## Chat protocol

- Stay in Chat mode for the entire handoff; do not mix Chat and Work modes.
- Complete Parts A–H in order.
- Every operational response contains exactly one fenced terminal block.
- Briefly state what the block changes or inspects and which diagnostics it
  prints.
- Alex returns complete stdout/stderr before the next block.
- A part may require multiple one-block turns. Never collapse an implementation
  part into one unexplained bulk command.
- Never answer only “Part complete” when the next block is available.
- Use focused tests during implementation and one exact-SHA integration gate at
  the end; do not rerun the full historical suite after each edit.
- Preserve user-owned untracked files.
- End every response with:

```text
HANDOFF=ALEX_02_STAGE12R1
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact implementation base recorded in Part A>
HEAD=<exact current SHA>
WAITING_FOR=<NONE or exact blocker>
SCIENTIFIC_DATA=NO
```

## Required technical method

Implement at least one complete technical candidate based on a differentiable
stochastic gate parameterization, provisionally hard-concrete/L0-style gates.
The implementation must make the following separations explicit:

1. **Surrogate native optimization:** differentiable fidelity/sparsity objective
   used to update continuous gate parameters.
2. **Mask proposal:** deterministic or seeded-stochastic conversion of learned
   gates into binary component masks.
3. **Exact qualification:** full-domain exact evaluation through the common
   Stage 6A ledger; surrogate values never qualify a circuit.
4. **Endpoint reduction:** existing Stage 6A/6E reducers consume exact ledgers;
   the method does not implement private endpoint definitions.

The method may expose technical alternatives through injected profiles, but the
task must not proliferate several half-implemented optimizers.

## Expected implementation surface

Prefer one isolated namespace following repository conventions, such as
`stage12r1`, containing:

- method configuration and validation;
- gate distribution/parameterization;
- seeded sampling and deterministic mask extraction;
- native optimizer, trajectory, checkpoint, and resume state;
- proposal and exact-evaluation bridge;
- method-native budget and failure records;
- validate-only CLI;
- focused/adversarial tests.

Reuse common records/utilities rather than duplicating Stage 6A/6D. Explain the
selected paths after the Part A/B audit and before file creation.

## Part A — Exact-base, scope, and collision guard

The first block is read-only. It must print and verify:

- repository root, remote, branch, HEAD, local `main`, and `origin/main`;
- authority-floor ancestry and local-main equality;
- merged PR #17 and #18 identities;
- exact hashes of the Stage 11 candidate and resolution records;
- tracked cleanliness and separately listed untracked files;
- absence of Stage 12-R1 and Stage 15 artifacts/processes;
- available Stage 5A/6A/6D/6E/7/10 APIs and focused tests;
- installed PyTorch/backend versions relevant to differentiable gates;
- no need to access the private predecessor.

After output diagnosis, a second block may create the required feature branch.

**Part A passes when:** the branch starts from recorded shared main, both
Barrier 1 packages are in its ancestry, the tracked tree is clean, and no
scientific artifact has been accessed.

## Part B — Reuse audit and independent-method contract

Inspect the existing discovery stack and record:

- the common discovery request/result/proposal types;
- exact-evaluation ledger and budget APIs;
- dense-model/component-mask abstraction;
- centred-logit objective/evaluator interface;
- inherited greedy/diversity adapter boundaries;
- Stage 6E packing consumer requirements;
- Stage 7 end-to-end integration points;
- Stage 10 benchmark assumptions;
- serialization, hash, identity, and seed utilities.

Define one method contract that is demonstrably independent of discrete greedy
deletion:

- continuous gate parameters optimized jointly or stochastically;
- differentiable sparsity pressure;
- seeded sampling/reparameterization;
- proposal extraction from learned gate state;
- no component-by-component greedy deletion hidden in the optimizer;
- explicit native budget unit;
- exact evaluation requested only through the shared bridge.

Write a short algorithmic-independence rationale and adversarial contract test
that rejects an adapter which merely wraps the inherited method with different
restart labels.

**Part B passes when:** the new method can share outer ledgers while retaining
different proposal dynamics and non-equivalent native accounting.

## Part C — Gate parameterization and deterministic identity

Implement and validate:

- one gate parameter per component under a supplied component basis;
- hard-concrete or equivalently documented reparameterized stochastic gates;
- valid temperature/stretch/clamp behavior supplied through technical config;
- expected-L0 or equivalent differentiable sparsity statistic;
- deterministic evaluation gates and seeded stochastic training samples;
- explicit RNG streams derived from the complete method/run identity;
- CPU device/dtype handling and CUDA-capable tensors without hard-coded device;
- finite-value and shape validation;
- stable state serialization and hashes.

Required tests include:

- same identity/seed gives identical CPU samples and records;
- changed seed or complete condition identity changes the stream;
- deterministic extraction is unaffected by proposal iteration order;
- invalid temperature, bounds, shapes, nonfinite parameters, and basis mismatch
  reject clearly;
- saturated gates remain representable;
- all-on and all-off masks are handled without accidental qualification.

No technical temperature or initialization becomes a production value.

**Part C passes when:** gate behavior, identity, serialization, and failure
boundaries are independently testable without a real model.

## Part D — Native optimization, trajectory, checkpoint, and resume

Implement a method-native optimizer that accepts injected:

- differentiable dense-model mask adapter;
- reference output/objective adapter;
- gate configuration;
- optimizer/schedule configuration;
- sparsity coefficient or schedule;
- minibatch/full-domain technical policy;
- maximum native budget;
- checkpoint/resume/output identity.

Record native work in a method-specific unit such as optimizer steps plus
objective/sample counts. Do not label it equivalent to greedy proposal counts.

Required behavior:

- bounded optimization with explicit completed/exhausted/interrupted/numerical-
  failure states;
- atomic rolling checkpoint and bounded retention;
- uninterrupted/resumed equivalence under the supported deterministic rule;
- rejection of stale/tampered/mismatched resume state;
- compact trajectories sufficient to audit objective, sparsity, gates, and
  budget without per-example verbose JSON;
- no scientific early stopping based on exact endpoint effects.

Tests use constructed differentiable fixtures with known relevant components,
including zero-gradient, exploding/nonfinite, interruption, and budget-zero
cases.

**Part D passes when:** native state can train, interrupt, resume, terminate, and
reproduce its declared technical record without exact endpoint shortcuts.

## Part E — Binary proposals and common exact-ledger bridge

Implement proposal extraction that can emit a bounded, deduplicated candidate
set from learned gate state using injected technical rules, for example:

- deterministic probability/score thresholds;
- top-k sizes supplied prospectively;
- seeded stochastic draws;
- mandatory intact mask handled by the common evaluator, not charged as a
  discovered sparse success.

Every proposal must record provenance to gate state, extraction rule, restart,
proposal order, and deterministic identity. Duplicate binary masks retain
proposal references while the common exact ledger charges/scientifically stores
the unique mask under the frozen accounting contract.

Bridge to Stage 6A so that:

- only exact full-domain evaluation creates qualifying mask records;
- the exact allowance is separate from native optimization;
- exhaustion, evaluator failure, duplicate proposals, and intact-baseline
  behavior remain auditable;
- Endpoint 1 remains defined through the intact mask;
- Stage 6E can consume the same exact ledger unchanged.

Required tests force surrogate/exact disagreement and prove the surrogate never
qualifies a mask.

**Part E passes when:** Stage 6A and 6E reducers can reconstruct technical
endpoints from the common ledger without method-specific branches.

## Part F — Records, budgets, failures, and technical profiles

Provide versioned records for:

- method/config identity and algorithm family;
- native budget definition/consumption;
- gate/restart/trajectory/checkpoint identity;
- proposal provenance and deduplication;
- exact-evaluation requests and bridge results;
- terminal state and failure taxonomy;
- technical profile classification.

Validate that:

- production algorithm/version/budget/restarts/exact allowance remain null or
  explicitly unresolved;
- native units are not called equivalent across methods;
- failed optimization still permits the common intact-mask Endpoint 1 rule;
- no mask is omitted because its exact fidelity is negative or unfavorable;
- production eligibility cannot be asserted by a technical profile;
- RD-005–RD-009/RD-012/RD-014 remain unresolved;
- records contain no private paths or large payloads.

Add adversarial corruptions for budget transfer, result relabeling, stale hashes,
method-family confusion, and silently clipped fidelity.

**Part F passes when:** the complete method lifecycle is reconstructable and
cannot masquerade as a frozen production run.

## Part G — Validate-only integration and compatibility gate

Build a portable validate-only CLI exercising:

1. a tiny differentiable component model;
2. seeded gate optimization;
3. interruption/resume or a separately tested resume fixture;
4. bounded proposal extraction;
5. exact evaluation through Stage 6A;
6. Endpoint 1 reconstruction;
7. Stage 6E packing consumption where multiple masks qualify;
8. explicit technical boundary output.

Run from repository root and an unrelated working directory. Test at least two
`PYTHONHASHSEED` values for deterministic records. Run:

- Stage 12-R1 focused/adversarial tests;
- Stage 5A centred-logit compatibility;
- Stage 6A ledger/Endpoint 1 compatibility;
- Stage 6D adapter/budget compatibility;
- Stage 6E packing compatibility;
- Stage 7 technical integration where relevant;
- Ruff on changed Python;
- diff, private-path, secret, large-file, binary, checkpoint, and LFS hygiene.

Do not require CUDA for portable tests. CUDA qualification belongs to Stage 14.

**Part G passes when:** the technical method completes the common pipeline
portably and deterministically without accessing a registered model.

## Part H — Commit, exact-SHA double-check, PR, and stop

Inspect the full surface and confirm only Stage 12-R1 implementation,
tests/validation, and necessary technical documentation changed. Create
coherent commits without amend or force-push. Push and open a PR against `main`.

At the final exact SHA, use a fresh detached checkout to rerun focused,
adversarial, compatibility, portable-CLI, Ruff, diff, cleanliness, and artifact
hygiene checks. Classify findings as blocking, nonblocking, or question. Repair
blocking findings only through descendant commits and repeat against the new
exact SHA.

Do not merge from this handoff unless the master task explicitly authorizes it.

Final report:

- base, branch, parent, final SHA, and PR;
- exact changed files;
- algorithmic-independence explanation;
- native/exact budget separation;
- proposal/exact-ledger behavior;
- resume/determinism and failure evidence;
- test totals and artifact sizes;
- unresolved production choices;
- internal findings/repairs;
- no scientific/private execution;
- interfaces exported to Alex 4/5 and Austin 3;
- explicit stop before Alex 3 and Stage 15.

Final status:

```text
ALEX_02_STAGE12R1_STATUS=COMPLETE_AT_HANDOFF_GATE
SCIENTIFIC_DATA=NO
PRODUCTION_METHOD_SELECTED=NO
STAGE15_STARTED=NO
```

## Prohibited shortcuts

- Do not rename the inherited discrete method and call it independent.
- Do not qualify masks using surrogate loss.
- Do not implement private endpoint or packing reducers.
- Do not freeze technical hyperparameters because one fixture recovers well.
- Do not run registered teachers/students.
- Do not begin basis sensitivity, packing-null calibration, Stage 13 freeze, or
  Stage 15 from this task.
