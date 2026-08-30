# Austin handoff — Stages 11–14 production machinery and cluster rehearsal

> **SUPERSEDED — DO NOT PASTE INTO A STAGE TASK.** This combined handoff
> compressed several independent implementations into oversized parts. Use
> `handoff_sequence.md` and the numbered Austin handoffs instead. It is retained
> only to preserve the planning history.

## Mission

Prepare the training, orchestration, compact-storage, Fourier-execution, and
cluster side of the post-red-team experiment for one automated Stage 15
Symbolica campaign. Use synthetic or explicitly excluded technical fixtures
only. Do not run definitive scientific jobs from this task.

This task runs in parallel with Alex's scientific-freeze and recovery-machinery
task.

## Authoritative files

Read completely before modifying anything:

1. `docs/distillation_followup/post_red_team_protocol_amendment.md`
2. `docs/distillation_followup/distillation_implementation_post_red_team.md`
3. `docs/distillation_followup/red_team/red_team_resolution_matrix.md`
4. `followup/configs/post_red_team_open_decisions_v1.json`
5. `docs/distillation_followup/stage9_training_backend_benchmark.md`
6. `docs/distillation_followup/stage10_discovery_compute_benchmark.md`

The old protocol/master remain historical authorities for completed Stages
1–10 and must not be rewritten.

## Required starting state — shared Barrier 0

- Clone or update the repository only after the post-red-team planning PR is in
  `origin/main`.
- Record the exact `origin/main` SHA in Part A and branch from it.
- Branch name: `feat/stages-11-14-production-machinery`.
- Use no private predecessor/checkpoint artifact during Stages 11–14.
- Do not create a substitute scientific design if Alex's Barrier 1 interface is
  not yet available.

If amended planning is missing, stop with
`WAITING_FOR_POST_RED_TEAM_PLAN_MERGE`. If planning is present but Alex's
Barrier 1 interface is not, Part A and the noncommittal audit portion of Part B
may proceed; scientific-policy-dependent implementation waits.

## Chat operating protocol

- Use Chat mode for the whole task. If Work mode is chosen, use a fresh task;
  never switch modes inside this one.
- Complete Parts A–F in order. Do not manufacture A–Z parts.
- Give exactly one fenced terminal block per operational response.
- Austin returns the complete stdout/stderr before receiving the next block.
- Each block prints paths, exact SHAs, changed files, test totals, hashes,
  failures, and explicit PASS/FAIL diagnostics.
- Never answer “Part complete; proceed” without also giving the next executable
  block unless waiting at a named synchronization barrier.
- Use focused tests during development and one integrated/full gate near the
  end; do not repeatedly run the entire historical suite.
- End every response with:

```text
WORKSTREAM=AUSTIN_11_14
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact SHA>
HEAD=<exact SHA>
WAITING_FOR=<NONE or barrier>
SCIENTIFIC_DATA=NO
```

## Ownership

Austin is primary owner of:

- expanded teacher-training and phase-registry production support;
- multi-architecture student construction/training support;
- interruption-safe resource-neutral DAG execution;
- compact ledgers, checkpoints, inventories, and exports;
- Fourier interchange execution plumbing and controls;
- CPU/CUDA qualification tooling;
- Stage 14 cluster dress rehearsal.

Alex owns the scientific choices, independent discovery implementation, basis
definitions, packing calibration, analysis freeze, exact production manifest,
and final launch authority. Do not silently resolve Alex-owned RD decisions.

## Part A — Guard, inventory, and branch

One read-only block must establish:

- repository/remote identity and exact base SHA;
- presence and hashes of amended authorities;
- clean tracked state and any untracked paths;
- Stage 5B/5C/6B/6C/7/9 orchestration and trainer interfaces available for
  reuse;
- environment and backend inventory;
- absence of definitive Stage 15 artifacts;
- Alex branch/interface availability if published.

Create the feature branch only after the guard passes.

**Part A gate:** exact common base recorded; no private artifact access; no
scientific computation.

## Part B — Stage 12 gap audit and stable scaffolding

Before Alex's Barrier 1 interface, perform a read-only/technical gap audit for:

- teacher generation and per-seed phase selection;
- architecture registry and shared trainer adaptability;
- job DAG dependency/resource/priority semantics;
- compact mask and metric ledgers;
- rolling checkpoint/resume;
- storage quota and export integrity;
- Fourier interchange integration points;
- CUDA qualification and permitted numerical drift.

Implement only policy-neutral scaffolding whose interface is already fixed by
Stages 4–10. Do not choose rosters, budgets, thresholds, task formulas, or
scientific assignments.

After Alex publishes Barrier 1, rebase/merge that exact interface commit and
show the resulting lineage.

**Synchronization Barrier 1:** record Alex's exact interface SHA and list any
blocking implementation contradiction. Questions that do not block code do not
pause the task.

## Part C — Expanded teacher and architecture production support

Implement with technical fixtures:

- declarative task registry supporting mod-113 addition, modular
  multiplication, and one prospectively configured modular-polynomial task;
- teacher construction/training/resume without embedding a production roster;
- per-seed phase-selection adapter driven only by frozen training metrics;
- architecture registry with canonical and additional student builders;
- shared hard/soft trainer compatibility across architectures;
- parameter count, searchable component count, and component-type accounting;
- full-domain cache and eligibility compatibility;
- bounded rolling checkpoints and atomic terminal records;
- deterministic identities/seeds from the complete condition identity;
- clean unavailable/failure outcomes when a task does not meet the frozen
  phase or eligibility rule.

No real task roster is trained definitively here. Fixture outcomes are excluded
technical data.

**Part C gate:** architecture/task/trainer focused and compatibility tests pass;
no production profile is selected.

## Part D — Resource-neutral DAG, compact storage, and Fourier execution

### D1. Campaign orchestration

- materialize scheduler jobs from the frozen-style declarative DAG;
- support local, generic job-array, and pluggable Symbolica scheduler adapters;
- enforce dependencies, isolated outputs, atomic completion, bounded retries,
  terminal failure, resume, priorities, and workload shedding;
- ensure direct-teacher discovery can overlap student training;
- derive reducer sensitivities from sealed ledgers without duplicate search;
- expose status by protected tier without displaying comparative effect
  directions at the operational gate.

### D2. Compact artifacts

- bit-pack or equivalently compact masks;
- use compressed/columnar ledgers where suitable;
- prohibit verbose per-step JSON and dense checkpoint history in production;
- enforce per-job scratch quotas;
- merge deterministically and serially;
- export, resume export, and independently verify destination hashes;
- detect partial, stale, duplicate, and conflicting objects.

### D3. Fourier execution plumbing

- implement the registered alignment/intervention runner interface;
- support matched information capacity;
- implement wrong-mode, shuffled-coefficient, mismatched-input, equal-norm
  random-state, and unaligned ordinary-patching controls;
- ensure aligned and control jobs share the same trial/capacity contract;
- emit a result even when aligned interchange fails to outperform controls.

Use Alex's scientific specifications when available; keep numeric/configurable
choices injected rather than hard-coded.

**Part D gate:** forced interruption/resume, quota, incomplete export,
dependency, priority, Fourier-control, and determinism tests pass on technical
fixtures.

## Part E — Stage 14 cluster package and integration delivery

Produce:

- locked environment/container definition;
- input staging and hash-verification command;
- hardware/CPU/CUDA inventory and qualification command;
- one-command campaign launch, status, pause/stop, resume, recompute, inventory,
  compact, and export operations;
- generic scheduler adapter plus documented Symbolica-specific fields still
  awaiting actual access details;
- reduced complete DAG fixture;
- monitoring/heartbeat and concise alert output;
- capacity projection from an actual resource manifest;
- final-window reservation mechanism;
- operator guide for the three human gates.

Commit and push the exact implementation SHA. Open a PR but do not merge before
Alex integrates the scientific/recovery branch.

**Synchronization Barrier 2:** send Alex the exact SHA, changed-file list,
focused test totals, technical-fixture hashes, and any unresolved cluster facts.

## Part F — Integrated rehearsal and exact-SHA double-check

After Alex produces the integrated Stage 13 freeze SHA:

1. create a fresh checkout of that exact SHA;
2. materialize the reduced frozen DAG;
3. run the complete technical rehearsal;
4. force interruption, retry, failure, and incomplete-transfer cases;
5. run CPU/CUDA qualification if the actual backend is accessible;
6. recompute outputs independently;
7. verify compact export and destination hashes;
8. run focused/adversarial suites, compatibility, Ruff, diff hygiene, manifest
   validation, and one full portable suite;
9. record findings as blocking, nonblocking, or question;
10. verify repairs on the final exact SHA.

This is the internal double-check replacing a separate ceremonial review. It
does not authorize Stage 15 by itself.

**Synchronization Barrier 3:** deliver the exact-SHA rehearsal/double-check
report to Alex. Alex owns the launch decision.

At Barrier 4, verify that the integrated Stage 11–14 commit is in `main` and the
materialized shared Stage 15 handoff contains actual hashes/resources. Stop;
do not launch the campaign from this task.

Final state:

```text
AUSTIN_STAGES_11_14_STATUS=COMPLETE_AT_LAUNCH_HANDOFF
SCIENTIFIC_DATA=NO
STAGE15_STARTED=NO
```

## Prohibited work

- No definitive teacher/student training or circuit search.
- No private predecessor/checkpoint access.
- No scientific roster, threshold, budget, or interpretation selected from
  technical fixture outcomes.
- No mutation of historical Stage 2 authorities.
- No independent competing schemas or condition identities.
- No verbose production artifact design known to recreate the predecessor's
  storage explosion.
- No merge or Stage 15 launch without the integrated exact-SHA gate.
