# Alex 4 — Stage 12-R3 packing nulls and tractable calibration

## Paste this entire document into one fresh Chat-mode task

Repository: `AK-314/circuit-families-distillation`

Local clone convention: `~/Projects/circuit-families-distillation`

Required scientific authority floor:

```text
d36f1b442ab7b783f3211377303a2981fc0d00e3
```

Required Stage 12-R1/R2 integration floor:

```text
4e47821a86585006f21920a13ef075437bb732a0
```

Required branch:

```text
feat/stage-12r3-packing-calibration
```

Create the branch from current `origin/main` after this handoff is merged. Part
A must prove both required floors are ancestors, local `main` exactly matches
`origin/main`, and the Stage 12-R1/R2 and Stage 6E contracts are present. Record
the current `origin/main` SHA as the implementation base.

## Mission

Implement the four distinct technical calibration layers required to interpret
the procedure-relative circuit packing lower bound:

1. a size-and-component-type-matched combinatorial floor;
2. an ordinary independent-restart discovery baseline;
3. local fidelity-retaining perturbations around exactly qualified circuits;
4. an exact or certified near-exact feasible-region calibration on a deliberately
   tractable synthetic problem.

The package must compare recovered search outputs with known or certified
technical feasible regions while preserving the existing Stage 6E definition
of Endpoint 2. It must not redefine packing, treat packing as a mechanism count,
or infer main-scale search optimality from a toy calibration.

This is technical Stage 12 implementation. It does not select production null
algorithms, draw counts, budgets, reportable coverage, calibration model, mask
universe, fidelity threshold, component cap, overlap rule, packing solver, or
Endpoint 2 status. RD-006, RD-008, and RD-009 remain open for Stage 13.

Austin simultaneously implements scheduler-neutral orchestration. This task
must export deterministic job/config/result identities and compact reports that
can later be scheduled, but it must not depend on Austin's concrete scheduler.

## Authorities — read completely before acting

1. `docs/distillation_followup/stage11_post_red_team_design_resolution.md`
2. `followup/configs/stage11_post_red_team_design_candidates_v1.json`
3. `followup/manifests/stage11_red_team_resolution_v1.json`
4. `followup/configs/post_red_team_open_decisions_v1.json`
5. `docs/distillation_followup/post_red_team_protocol_amendment.md`
6. merged Stage 12-R1 discovery contracts, exact bridge, records, and tests;
7. merged Stage 12-R2 basis/accounting contracts where type matching is needed;
8. Stage 6A exact ledger and Endpoint 1 contracts;
9. Stage 6E packing graph, solver, proof, record, and recomputation contracts;
10. relevant Stage 7 lifecycle/inventory compatibility;
11. Phase I size-matched Jaccard-null utilities only where their semantics
    genuinely apply;
12. `docs/distillation_followup/handoffs/post_red_team/handoff_sequence.md`.

The four calibration layers answer different questions and may not be collapsed
into one generic “null.” Existing exact-ledger and packing reducers must be
reused rather than reimplemented under new names.

## Scientific boundary

Permitted:

- synthetic component universes, masks, logits, fidelity surfaces, and overlap
  graphs;
- tiny technical models/domains where complete mask enumeration is possible;
- constructed feasible sets and search algorithms with known behavior;
- deterministic sampling, optimization, certification, runtime, memory, and
  coverage diagnostics;
- explicitly excluded technical calibration reports.

Prohibited:

- loading or evaluating registered teacher/student checkpoints;
- using Phase I or Phase II scientific endpoint values to select algorithms or
  settings;
- running definitive discovery, packing, phase, condition, architecture, or
  basis comparisons;
- calling an ordinary restart a second independent discovery family;
- qualifying perturbations with a surrogate rather than exact evaluation;
- resolving RD-006, RD-008, RD-009, or any other production decision;
- claiming a true mechanism count, true main-scale packing number, global
  circuit minimum, or guaranteed transfer from the tractable fixture.

Every executable profile and result must state `scientific_data=false` and
`production_eligible=false`.

## Chat protocol

- Stay in Chat mode for the entire handoff; do not mix Chat and Work modes.
- Complete Parts A–H in order. Use multiple one-block turns inside a part when
  genuinely needed.
- Every operational response contains exactly one fenced terminal block.
- Briefly state what the block changes or inspects and which diagnostics it
  prints.
- Alex returns complete stdout/stderr before the next block.
- Never emit only a status footer when the next block is available.
- Never claim to await output from a block that was not supplied.
- Use focused tests during implementation and one exact-SHA integration gate at
  the end; do not repeatedly run the full historical suite.
- Preserve user-owned untracked files.
- End every response with:

```text
HANDOFF=ALEX_04_STAGE12R3
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact implementation base recorded in Part A>
HEAD=<exact current SHA>
WAITING_FOR=<NONE or exact blocker>
SCIENTIFIC_DATA=NO
```

## Core calibration distinctions

### Combinatorial floor

This asks how much low-overlap multiplicity arises from component-set
combinatorics after matching declared size and component-type composition. It
does not establish that sampled masks are faithful circuits.

### Ordinary-restart baseline

This asks what the same discovery procedure recovers under independent ordinary
starts without diversity pressure or packing-aware proposal modification. It is
a search baseline, not an algorithmically independent discovery family.

### Local fidelity-retaining perturbation

This asks whether exactly qualified circuits sit inside locally connected
regions containing other exactly qualified, sufficiently separated masks. Every
accepted neighbor requires common exact evaluation.

### Tractable feasible-region calibration

This asks how much a chosen search procedure misses when the complete feasible
mask set—or a certified near-exact bound—is available on a small technical
problem. Transfer claims are explicitly limited.

## Expected implementation surface

Prefer one cohesive namespace, such as `stage12r3`, containing:

- shared calibration identities/profiles and compact records;
- size/type-matched combinatorial sampler;
- ordinary-restart baseline adapter;
- local perturbation generator and exact-evaluation bridge;
- tractable mask-universe enumerator or certified near-exact solver;
- recovered-versus-feasible comparison and coverage reducers;
- deterministic validate-only CLI;
- focused/adversarial tests.

Reuse Stage 6A/6E/12-R1 records and evaluators. Calibration-specific metadata
may wrap them, but must not fork fidelity, overlap, or packing definitions.

## Part A — Exact-base, ancestry, and scope guard

The first block is read-only. It must print and verify:

- repository root, remote, branch, HEAD, local `main`, and `origin/main`;
- authority-floor and R1/R2-integration-floor ancestry;
- merged PR #20/#24 identities and expected implementation namespaces;
- Stage 11 candidate/resolution hashes and open RD-006/RD-008/RD-009 status;
- tracked cleanliness and separately listed untracked files;
- Stage 6A/6E, Stage 12-R1/R2, and relevant Phase I null APIs/tests;
- absence of Stage 12-R3 and Stage 15 implementation/processes;
- Python/backend capabilities needed for bounded combinatorial tests;
- no private predecessor or registered-model dependency.

After output diagnosis, a second block may create the required branch.

**Part A passes when:** the branch starts from shared main containing the merged
R1/R2 contracts, tracked state is clean, user artifacts are preserved, and no
scientific/private artifact has been touched.

## Part B — Reuse audit and four-layer calibration contract

Inspect and map:

- Stage 6A exact-mask entries, intact baseline, qualification, and budget rules;
- Stage 6E overlap graph, packing solver/proof, zero-packing, and recomputation;
- Stage 12-R1 proposal provenance, restarts, method-native budget, exact bridge,
  and lifecycle records;
- Stage 12-R2 component types, basis identity, grouping, and accounting;
- Phase I size-matched sampling/seed logic that can be reused without importing
  old top-one or scientific semantics;
- canonical JSON, hash, seed, failure, and compact-record utilities.

Define one versioned outer contract with distinct layer identities, inputs,
budgets, termination, coverage, and claim boundaries. It must keep native
sampling/search work separate from exact mask-evaluation allowance.

Adversarial contract tests must reject:

- collapsing the four layers under one result type without layer identity;
- relabeling an ordinary restart as an independent discovery method;
- substituting surrogate fidelity for exact entries;
- mixing basis/component-type identities;
- claiming exactness without a valid certificate;
- setting `scientific_data=true` or `production_eligible=true`.

**Part B passes when:** all four layers share common ledgers/reducers but remain
semantically and computationally distinguishable.

## Part C — Size/type-matched combinatorial floor

Implement deterministic sampling of mask collections from a supplied component
universe while matching prospectively supplied:

- retained component count or declared size distribution;
- component-type counts where requested;
- basis identity and ordered component universe;
- draw/batch identity and explicit seed stream;
- overlap metric/cutoff reference consumed by Stage 6E.

The sampler must be uniform under its documented finite sampling rule or clearly
state its alternative distribution. It must not use model outputs, fidelity,
discovery scores, observed effect directions, or accepted packing members to
bias draws.

For each draw/batch, reuse Stage 6E to produce the packing statistic implied by
the sampled masks and store compact summary/distribution records. Preserve
duplicates as auditable draw outcomes while applying explicit unique-mask rules
before packing where required by the declared contract.

Tests include exact small-universe distribution checks, size/type matching,
seed/order determinism, impossible compositions, duplicate handling, basis
mismatch, overlap extremes, empty/zero-packing cases, and no fidelity claim.

**Part C passes when:** the combinatorial floor is reproducible and answers only
the expected-overlap question under its declared matching distribution.

## Part D — Ordinary independent-restart baseline

Implement an adapter that accepts an injected discovery procedure and runs
ordinary independent restarts under the same declared method/config and
method-native budget per restart. It must:

- derive restart seeds from complete run/method/restart identity;
- prohibit diversity pressure, packing-aware feedback, or prior-restart mask
  exclusion in the ordinary baseline;
- retain every restart's proposals, exact requests/results, failures, and
  termination state;
- deduplicate masks only at the declared combined-ledger boundary;
- use the common exact-evaluation accounting rule;
- feed exactly qualified unique masks to Stage 6E;
- report coverage and packing as procedure-relative baseline outcomes;
- remain explicitly the same discovery family, not a second method.

Use constructed deterministic discovery callables and Stage 12-R1 technical
adapters. Test repeated masks, disjoint masks, failed/exhausted restarts, changed
seed identity, exact-budget exhaustion, restart-order invariance, and rejection
of cross-restart information leakage.

**Part D passes when:** the contribution of ordinary restart diversity can be
measured without changing discovery dynamics or misrepresenting compute.

## Part E — Local exact-fidelity-retaining perturbations

Implement bounded local proposal generation around supplied exactly qualified
seed masks using injected technical neighborhoods such as add, drop, swap, or
type-preserving swap. The record must state the neighborhood, distance, parent
mask, basis/component universe, seed, proposal order, and native work.

Required behavior:

- only exactly qualified common-ledger entries may seed a fidelity-retaining
  neighborhood;
- generated masks are deduplicated with complete parent/proposal provenance;
- every candidate is evaluated through the common exact bridge;
- negative/unfavorable fidelity and failed evaluations remain recorded;
- only exact qualifiers enter the perturbation packing graph;
- local search cannot silently expand beyond supplied radius/budget;
- intact-mask Endpoint 1 fallback and zero Endpoint 2 remain valid;
- no surrogate or parent fidelity is copied to a neighbor.

Tests use synthetic fidelity surfaces with isolated optima, connected plateaus,
deceptive surrogate neighborhoods, asymmetric type constraints, exhausted
budgets, and evaluator failures.

**Part E passes when:** the local layer distinguishes isolated recovered masks
from local feasible families without manufacturing qualification.

## Part F — Tractable exact or certified near-exact feasible region

Implement a deliberately small technical calibration interface with explicit:

- task/model/domain and dense-reference identity;
- finite component basis and complete mask universe definition;
- common fidelity, size, and exact-evaluation rules;
- exact enumeration or documented certified near-exact algorithm;
- stopping/completeness certificate;
- full feasible-mask inventory or compact hash-bound representation;
- exact feasible Endpoint 1 and feasible packing under Stage 6E rules;
- resource/coverage accounting and transfer limitations.

The default portable fixture should be small enough for complete enumeration.
If a near-exact path is also implemented, its certificate must expose lower and
upper bounds/gap and never use the word exact when the gap is nonzero.

Compare one or more injected technical search outputs with the known feasible
region using metrics such as:

- feasible-mask recall/coverage;
- recovered Endpoint 1 gap to feasible minimum;
- recovered packing lower bound versus feasible packing;
- duplicate/invalid proposal rate;
- exact-evaluation coverage and censoring;
- failure to recover despite feasibility.

Do not call the fixture a teacher-seed replicate or infer main-scale error
bounds. Tests adversarially corrupt universe, evaluator, certificate, feasible
hash, packing proof, and exact/near-exact labels.

**Part F passes when:** search suboptimality is measurable on a certified small
instance and every limitation of transfer is machine-readable.

## Part G — Integrated reports and portable compatibility gate

Build a validate-only CLI that exercises all four layers on deterministic
synthetic fixtures and emits compact, versioned, outcome-neutral reports. It
must include at least:

1. combinatorial floor with size/type matching;
2. ordinary restarts with duplicate and distinct recovery;
3. local perturbations containing qualified and nonqualified neighbors;
4. complete feasible-region enumeration;
5. recovered-versus-feasible Endpoint 1 and packing comparison;
6. zero/failed/censored outcome preservation;
7. explicit procedure-relative and no-transfer/no-mechanism-count boundaries.

Run from repository root and an unrelated cwd. Test at least two
`PYTHONHASHSEED` values with identical canonical report hashes. Run:

- Stage 12-R3 focused/adversarial tests;
- Stage 6A exact-ledger/Endpoint 1 compatibility;
- Stage 6E packing/proof/recomputation compatibility;
- Stage 12-R1 discovery/proposal/exact-bridge compatibility;
- Stage 12-R2 component-type/basis compatibility;
- relevant Stage 7 lifecycle/inventory compatibility;
- Ruff on changed Python;
- diff, private-path, secret, large-file, binary, checkpoint, and LFS hygiene.

Technical reports must preserve null, negative, failed, unavailable, and
censored outcomes without directional interpretation.

**Part G passes when:** all four calibration layers are portable, deterministic,
compact, and independently recomputable without registered data.

## Part H — Commit, exact-SHA double-check, PR, and stop

Inspect the surface and ensure it is limited to Stage 12-R3 code,
tests/validation, and necessary technical documentation. Create coherent
commits without amend or force-push. Push and open a PR against `main`.

At the final exact SHA, run a fresh detached-checkout double-check covering
focused/adversarial and compatibility tests, unrelated-cwd CLI, multiple hash
seeds, Ruff, diff, tracked cleanliness, artifact sizes, and Git/LFS surface.
Classify findings as blocking, nonblocking, or question. Repair blockers only
through descendant commits and repeat against the new exact SHA.

Do not merge inside this handoff without master-task authorization.

Final report:

- base, branch, parent, final SHA, and PR;
- exact changed files and artifact sizes;
- four calibration-layer identities and semantics;
- budget/exact-evaluation separation;
- determinism, failure, and coverage evidence;
- tractable exact/near-exact certificate and search-gap evidence;
- Stage 6A/6E/R1/R2 compatibility and test totals;
- unresolved RD-006/RD-008/RD-009 production choices;
- internal findings and descendant repairs;
- confirmation of no registered/private/scientific execution;
- interfaces exported to Alex 5, Austin 3/4, and Stage 14;
- explicit stop before Stage 13 and Stage 15.

Final status:

```text
ALEX_04_STAGE12R3_STATUS=COMPLETE_AT_HANDOFF_GATE
SCIENTIFIC_DATA=NO
PRODUCTION_PACKING_POLICY_SELECTED=NO
TRACTABLE_CALIBRATION_PRODUCTION_CHOICE=NO
STAGE15_STARTED=NO
```

## Prohibited shortcuts

- Do not merge the four calibration layers into one undifferentiated null.
- Do not treat ordinary restarts as an independent discovery family.
- Do not accept local neighbors using surrogate or inherited parent fidelity.
- Do not duplicate Stage 6A fidelity or Stage 6E packing definitions.
- Do not claim exactness without an exhaustive/certified stopping record.
- Do not tune technical algorithms from registered scientific outcomes.
- Do not infer a true main-scale packing number or mechanism count.
- Do not run registered teachers, students, discovery, or endpoint analysis.
- Do not begin Stage 13 freeze, Stage 14, or Stage 15.
