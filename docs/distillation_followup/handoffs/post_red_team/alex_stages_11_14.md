# Alex handoff — Stages 11–14 scientific freeze and recovery machinery

## Mission

Prepare the scientific and recovery side of the post-red-team experiment for
one automated Stage 15 Symbolica campaign. Complete real work through the Stage
14 launch gate; do not run definitive scientific jobs from this task.

This task runs in parallel with Austin's Stages 11–14 production-machinery
task. The two tasks synchronize only at the four barriers listed below.

## Authoritative files

Read completely before modifying anything:

1. `docs/distillation_followup/post_red_team_protocol_amendment.md`
2. `docs/distillation_followup/distillation_implementation_post_red_team.md`
3. `docs/distillation_followup/red_team/red_team_resolution_matrix.md`
4. `followup/configs/post_red_team_open_decisions_v1.json`
5. `docs/distillation_followup/stage9_training_backend_benchmark.md`
6. `docs/distillation_followup/stage10_discovery_compute_benchmark.md`

The Stage 2 protocol and original implementation master remain historical
authorities for completed Stages 1–10. They must not be rewritten.

## Required starting state — shared Barrier 0

- Work in the local clone, conventionally
  `~/Projects/circuit-families-distillation`.
- The post-red-team planning commit/PR must be present in `origin/main` before
  this implementation branch is created.
- Record the exact `origin/main` SHA in Part A and treat it as immutable base.
- Branch name: `feat/stages-11-14-scientific-freeze`.
- The worktree must be clean except for explicitly identified user-owned
  untracked documents. Never add the presentation or red-team DOCX by accident.
- No predecessor checkpoint or scientific output is authorized before Stage
  15.

If the amended planning files are not in `origin/main`, stop with
`WAITING_FOR_POST_RED_TEAM_PLAN_MERGE`. Do not reconstruct them independently.

## Chat operating protocol

- Use Chat mode for the whole task. If Work mode is chosen instead, start a new
  task; never mix modes inside this task.
- Work through Parts A–F in order. These parts represent real lifecycle
  boundaries; do not invent additional alphabetic parts.
- Give exactly one fenced terminal block per operational response.
- Before the block, state briefly what it changes and what evidence it prints.
- Alex pastes the complete terminal output back before the next block.
- Every block must print useful diagnostics: path, branch, exact SHA, changed
  files, test counts, hashes, failures, or explicit PASS/FAIL labels.
- Do not reply with a prose-only “Part complete.” The next response must contain
  the next executable block unless genuinely waiting at a synchronization
  barrier.
- Do not repeat the full repository suite after every small edit. Use focused
  tests while developing, compatibility tests at integration, and one full
  gate near the end.
- Keep responses compact. At the end of every response print:

```text
WORKSTREAM=ALEX_11_14
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact SHA>
HEAD=<exact SHA>
WAITING_FOR=<NONE or barrier>
SCIENTIFIC_DATA=NO
```

## Ownership

Alex is primary owner of:

- Stage 11 scientific resolution;
- teacher/task/architecture/basis assignment design;
- the independent discovery method contract and implementation;
- basis sensitivity and packing calibration machinery;
- Endpoint 1/Endpoint 2 hierarchy and analysis freeze;
- the exact Stage 13 job manifest;
- Stage 14 launch authorization.

Austin owns trainer/architecture production support, resource-neutral
orchestration, compact export, Fourier execution plumbing, and cluster
rehearsal tooling. Do not duplicate Austin's implementation unless a barrier
report establishes that it is missing.

## Part A — Guard, inventory, and branch

One read-only block must establish:

- repository and remote identity;
- exact local/main/origin/main SHA;
- presence and hashes of all four amended authorities;
- completed Stage 1–10 state;
- absence of Stage 15 scientific artifacts;
- current tracked and untracked status;
- available tests/modules relevant to discovery, bases, packing, and analysis;
- Austin branch/PR state if it already exists.

Then create the feature branch only after the guard passes. Do not alter or
stage user-owned untracked files.

**Part A gate:** exact shared base recorded; no scientific execution; branch
clean and isolated.

## Part B — Stage 11 scientific resolution

Resolve the design at the level needed for implementation. Do not freeze
unsupported numeric values yet.

Produce versioned candidate records for:

1. primary mod-113 task and approximately 15-teacher construction rule;
2. reduced Task 2 modular-multiplication replication;
3. Task 3 modular-polynomial capacity-contingent replication;
4. resource branches:
   - Symbolica available;
   - Symbolica unavailable and school Macs available;
   - Symbolica incomplete with school-Mac recovery;
5. canonical architecture plus the sparse multi-architecture panel;
6. canonical basis plus limited attention-coordinate, block, rotation, and
   accounting sensitivities;
7. Endpoint 1 primary and Endpoint 2 key-secondary status;
8. two genuinely different discovery families;
9. four packing calibration layers;
10. tractable exact/near-exact calibration;
11. registered Fourier key secondary;
12. protected Tier 1, protected Tier 2 minimum, Tier 3, and fixed workload
    shedding order;
13. school-Mac breadth priority: primary recovery, reproduction, Task 2, then
    Task 3 only after an operational—not outcome-based—gate;
14. bounded post-submission extension/rebuttal-reserve menu.

Update the additive RD register through a new resolution-candidate record; do
not mutate the historical Stage 2 UD register.

No comparative follow-up endpoint may be generated or consulted.

**Synchronization Barrier 1:** commit and push the candidate scientific
interface. Send Austin the exact SHA and a concise list of interfaces that are
now safe to implement. Austin does not need a ceremonial review of prose; he
must flag only contradictions that make implementation impossible.

## Part C — Stage 12 recovery and calibration implementation

Implement the Alex-owned missing machinery with synthetic/excluded fixtures:

### C1. Independent discovery

- continuous/stochastic sparse-mask optimizer genuinely distinct from greedy
  deletion/diversity restarts;
- shared exact-evaluation ledger integration;
- method-native budget and common exact allowance kept separate;
- restart, resume, failure, deduplication, and deterministic identity tests;
- CPU/CUDA-capable interface without claiming cross-backend bitwise identity.

### C2. Basis sensitivity

- pre-output-projection attention coordinates;
- seeded balanced MLP blocks;
- a limited fixed orthogonal-rotation interface;
- parameter-weighted and type-stratified accounting;
- identity, intervention, round-trip, and invalid-cross-basis tests.

### C3. Packing calibration

- size/type-matched combinatorial floor;
- ordinary-restart baseline;
- local fidelity-retaining perturbation null;
- tractable feasible-region calibration interface;
- exact or certified near-exact search on a small fixture;
- deterministic reducers and null-result reporting.

Reuse Stage 5A/6A/6D/6E contracts rather than creating parallel endpoint
formats. All outputs are synthetic or registered technical exclusions.

**Part C gate:** focused tests pass; no production profile or scientific output
exists; changes are committed as coherent implementation commits.

## Part D — Stage 13 scientific freeze and complete production manifest

Wait for Austin's Stage 12 implementation SHA/PR, integrate it, and run the
cross-lane compatibility checks before freezing.

Resolve every production-blocking RD decision using:

- the accepted red-team design;
- Stage 9/10 technical benchmarks;
- synthetic and excluded technical fixtures;
- the actual announced resource envelope if available;
- no comparative scientific endpoint evidence.

Generate and seal:

- exact teacher/task/phase rules;
- architecture and sparse-assignment matrices;
- hard/soft training, eligibility, replication, and attempt rules;
- component-basis definitions and assignment;
- fidelity threshold/frontier;
- methods, budgets, nulls, packing policy, and calibration;
- Fourier pairs/interventions/controls/outcome;
- hierarchical analysis and missing-cell rules;
- Task 2 and Task 3 reduced protocols;
- resource-contingent workload branches;
- full declarative Stage 15 DAG with canonical identities, dependencies,
  resource classes, priorities, retries, terminal states, outputs, and hashes;
- required reports and claim-resolution table.

Generate the complete planned report once from synthetic records.

**Synchronization Barrier 2:** both owners inspect the integrated exact SHA.
Classify findings only as blocking, nonblocking, or question. Repair blocking
findings before the freeze commit. No long independent-review ritual is
required.

**Part D gate:** all production-blocking RDs resolved by additive records;
manifest and configs immutable; excluded development outputs enumerated;
Stage 15 not started.

## Part E — Stage 14 integrated dress rehearsal

Use Austin's cluster package to run one complete reduced DAG on the available
technical backend. The rehearsal must exercise:

- teacher/target/student/eligibility flow;
- both discovery methods;
- canonical and at least one alternate architecture/basis path;
- all packing-null interfaces and tractable calibration;
- Fourier aligned path plus every control;
- interruption, retry, duplicate claim, ineligible student, search failure,
  budget exhaustion, incomplete export, and resume;
- deterministic serial merge and independent endpoint recomputation;
- compact storage, quotas, compression, transfer, and destination-hash checks;
- CPU/CUDA qualification or a clearly recorded waiting condition if Symbolica
  hardware is not yet accessible.

The final resource grant may change concurrency and the amount of lower-tier
work, but not Tier 1 estimands or inclusion rules.

**Synchronization Barrier 3:** Alex and Austin perform one fresh-checkout
double-check of the exact integrated head. Run focused/adversarial suites, one
compatibility suite, repository-wide Ruff, diff hygiene, manifest validation,
and one full portable suite. Do not rerun redundant historical review packages.

## Part F — Launch authorization and shared Stage 15 handoff

Produce one final report containing:

- base and exact final SHA;
- resolved/open RD accounting;
- frozen manifest and config hashes;
- tests and dress-rehearsal evidence;
- actual or required Symbolica resource envelope;
- container/image identity;
- input staging and export paths;
- protected tier completion projections;
- three human gates;
- abort and workload-shedding rules;
- confirmation that no scientific job has started.

Push the exact head and open the integration PR. Merge only after the internal
double-check is recorded against the final exact SHA.

**Synchronization Barrier 4:** after the integrated Stage 11–14 commit is in
`main`, materialize one shared Stage 15 handoff from
`shared_stage_15_symbolica_campaign_skeleton.md`, replacing every placeholder
with frozen hashes and actual cluster details. Neither owner launches Stage 15
from this task.

Final state:

```text
STAGES_11_14_STATUS=READY_FOR_SINGLE_STAGE15_CAMPAIGN
SCIENTIFIC_DATA=NO
STAGE15_STARTED=NO
```

## Prohibited work

- No definitive teacher or student training.
- No endpoint-producing production search.
- No selection based on apparent phase/condition effects.
- No mutation of predecessor or hash-anchored Stage 2 authorities.
- No literal full factorial.
- No separate Stage 15 teacher/student/discovery chats.
- No automatic merge before the exact-SHA double-check.
