# Austin 3 — Stage 12-P3 resource-neutral campaign DAG and resumption

## Paste this entire document into one fresh Chat-mode task

Repository: `AK-314/circuit-families-distillation`

Local clone convention: `~/Projects/circuit-families-distillation`

Required scientific authority floor:

```text
d36f1b442ab7b783f3211377303a2981fc0d00e3
```

Required Stage 12-P1/P2 integration floor:

```text
3002acebf7172146b86dafc2c885f5ec67f16909
```

Required branch:

```text
feat/stage-12p3-resource-neutral-dag
```

Create the branch from current `origin/main` after this handoff is merged. Part
A must prove both required floors are ancestors, local `main` exactly matches
`origin/main`, and the Stage 12-P1/P2 producer contracts are present. Record the
current `origin/main` SHA as the implementation base.

## Mission

Implement a scheduler-neutral, interruption-safe campaign DAG that can later
orchestrate the frozen Stage 15 job families without containing the scientific
freeze itself.

The package must provide:

1. deterministic declarative job and dependency identities;
2. dependency-aware readiness and topological validation;
3. isolated per-job/per-attempt outputs;
4. atomic claiming, leases, heartbeats, and stale-claim recovery;
5. bounded retries, terminal failures, and exact resumption;
6. injected resource classes and priorities;
7. protected-tier-aware workload shedding without using effect directions;
8. local and generic job-array adapters with operational-only status.

This task builds orchestration machinery. It does not freeze the Stage 15 DAG,
job matrix, protected tiers, concurrency, retry counts, timeouts, resources,
Symbolica scheduler details, storage quotas, or scientific configurations.
RD-012, RD-013, and RD-014 remain open for Stages 13–14.

Alex simultaneously implements basis sensitivity. Do not import Alex's
concrete implementation. Jobs must carry opaque, hash-bound input/config
references and producer interface versions so later Stage 13 integration can
compose all work families without changing the scheduler core.

## Authorities — read completely before acting

1. `docs/distillation_followup/stage11_post_red_team_design_resolution.md`
2. `followup/configs/stage11_post_red_team_design_candidates_v1.json`
3. `followup/manifests/stage11_red_team_resolution_v1.json`
4. `followup/configs/post_red_team_open_decisions_v1.json`
5. merged Stage 12-P1 task/teacher contracts and validation;
6. merged Stage 12-P2 architecture/student/seal/release contracts;
7. `src/circuit_families/stage5bc/job_dag.py` and its tests;
8. existing Stage 7 lifecycle, integration, reproduction, and inventory code;
9. Stage 4 condition identity and seed-derivation authorities;
10. Stage 6A/6E ledger and endpoint lifecycle states;
11. `docs/distillation_followup/handoffs/post_red_team/austin_stages_11_14.md`;
12. `docs/distillation_followup/handoffs/post_red_team/handoff_sequence.md`.

Extend existing orchestration through explicit adapters. Do not replace working
Stage 5C/7 lifecycle semantics merely to create a new namespace.

## Scientific boundary

Permitted:

- synthetic job specifications, identities, dependency graphs, and payload
  references;
- tiny local commands or in-process workers that create technical fixture
  outputs only;
- forced interruption, retry, lease expiry, failure, shedding, and resume;
- runtime, scheduling, state-store, serialization, and determinism diagnostics;
- explicitly excluded technical campaign reports.

Prohibited:

- launching registered teacher/student/discovery/Fourier work;
- accessing private predecessor artifacts or registered checkpoints;
- materializing the definitive Stage 15 scientific matrix;
- selecting production tiers, priority values, resource shapes, concurrency,
  retry limits, lease durations, or scheduler backend from fixture outcomes;
- displaying phase/condition/architecture effect directions in operational
  status;
- resolving RD-012, RD-013, RD-014, or another production decision;
- beginning compact export, Fourier execution, Stage 14, or Stage 15.

Every executable profile and report must state `scientific_data=false` and
`production_eligible=false`.

## Chat protocol

- Stay in Chat mode for the entire handoff; do not mix Chat and Work modes.
- Complete Parts A–H in order. A part may use several one-block turns where
  implementation genuinely requires them.
- Every operational response contains exactly one fenced terminal block.
- Briefly state what the block changes or inspects and which diagnostics it
  prints.
- Austin returns complete stdout/stderr before the next block.
- Never emit only a status footer when the next block is available.
- Never claim to await output from a block that was not actually supplied.
- Use focused tests during implementation and one exact-SHA integration gate at
  the end; do not repeatedly run the whole historical suite.
- Preserve unrelated or user-owned untracked files.
- End every response with:

```text
HANDOFF=AUSTIN_03_STAGE12P3
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact implementation base recorded in Part A>
HEAD=<exact current SHA>
WAITING_FOR=<NONE or exact blocker>
SCIENTIFIC_DATA=NO
```

## Core orchestration invariants

### Identity is distinct from execution state

A logical job identity covers immutable scientific/technical inputs,
producer/consumer interface versions, and dependency identities. Attempt,
retry, worker, claim, lease, and scheduler-array coordinates are execution
metadata and must not silently change the logical job.

### Success requires sealed evidence

Exit code zero alone is not success. A job becomes successful only when its
declared output manifest exists, validates against job identity, and is sealed
with expected hashes. Partial or stale output cannot unblock dependants.

### Status is operational only

Status may expose counts by job family, protected tier, lifecycle state,
resource class, and failure category. It must not expose comparative endpoint
directions or rank scientific conditions during collection.

### Resumption is exact and idempotent

Restarting the controller must preserve successful jobs, reconcile active
claims, retry only eligible failures, and never duplicate a logical output or
erase failure/unavailable records.

## Expected implementation surface

Prefer one cohesive namespace, such as `stage12p3`, containing:

- versioned job, dependency, resource, priority, and campaign records;
- deterministic DAG compiler/validator;
- durable compact state store and transition validator;
- atomic claim/lease/heartbeat logic;
- retry, terminal-failure, resume, and reconciliation engine;
- workload-shedding policy adapter;
- local and generic job-array adapters;
- operational status/inventory reporter;
- validate-only CLI and focused/adversarial tests.

Use injected paths/root directories in executors, while persistent records use
portable relative object identities and hashes. Do not store private absolute
paths or large payloads in Git.

## Part A — Exact-base, ancestry, and scope guard

The first block is read-only. It must print and verify:

- repository root, remote, branch, HEAD, local `main`, and `origin/main`;
- authority-floor and Stage 12-P1/P2 integration-floor ancestry;
- merged PR #17, #22, and exact reviewed P2 head identities;
- Stage 11 candidate/resolution hashes and unresolved RD-012–014 status;
- tracked cleanliness and separately listed untracked files;
- Stage 5C/7 orchestration APIs and Stage 12-P1/P2 producer interfaces;
- absence of Stage 12-P3 implementation and Stage 15 scientific outputs/
  processes;
- Python/backend/filesystem capabilities needed for atomic local tests;
- no need for private predecessor or registered model access.

After output diagnosis, a second block may create the required branch.

**Part A passes when:** the branch starts from shared main containing the merged
P1/P2 contracts, tracked state is clean, and no scientific/private artifact has
been touched.

## Part B — Reuse audit and scheduler-neutral contract

Inspect and map:

- Stage 5C job-DAG request, dependency, seed, and execution records;
- Stage 7 lifecycle, resume/merge, reproduction, and inventory contracts;
- Stage 4 canonical identities and attempt/retry seed semantics;
- Stage 12-P1 teacher/checkpoint/cache/seal outputs;
- Stage 12-P2 student/checkpoint/dense-output/release outputs;
- Stage 6A/6E exact-ledger completion/failure consumers;
- atomic write, canonical JSON, hash, path-safety, and lock utilities;
- current local-execution assumptions that must become adapters.

Classify requirements as direct reuse, adapter, new policy-neutral code, or
deferred to Stage 13/14. Define a scheduler-neutral contract separating:

- immutable logical job specification;
- dependency and expected-input references;
- requested resource class and priority class;
- execution attempt/claim/lease data;
- expected sealed-output contract;
- scheduler submission/observation/cancellation metadata;
- operational status projection.

Adversarial contract tests must reject a backend adapter that changes logical
job identity, treats scheduler completion as sealed success, or embeds
scientific result directions in status.

**Part B passes when:** existing lifecycle semantics are reused and the core
DAG has no dependency on local, Symbolica, Slurm, CUDA, or another scheduler.

## Part C — Declarative DAG compilation and deterministic identity

Implement versioned records and compilation for:

- campaign identity and immutable manifest reference;
- logical job family/type and producer interface version;
- ordered dependency identities and expected input hashes;
- opaque payload/config references and expected output contract;
- resource-class and priority-class references;
- protected-tier label without scientific direction fields;
- attempt/retry seed namespace reference;
- production/scientific boundary flags.

The compiler must:

- produce deterministic job identities and canonical ordering independent of
  input dictionary/set order;
- validate complete dependency closure;
- reject cycles, self-dependencies, duplicate jobs, duplicate outputs, dangling
  inputs, invalid relative paths, and identity/hash mismatches;
- permit independent job families to become ready concurrently;
- permit direct-teacher discovery to be independent of student training when
  its declared inputs are ready;
- represent reducer-only jobs as consumers of sealed ledgers rather than
  duplicate discovery work;
- leave the definitive Stage 15 job list and numeric settings injected.

Test tiny fan-out/fan-in graphs, disconnected valid subgraphs, cycle/dangling
failures, order invariance, changed config hashes, and job-family collisions.

**Part C passes when:** the same declarative technical manifest always compiles
to the same validated DAG and identities without executing a job.

## Part D — Durable state, isolated outputs, and atomic claiming

Implement a compact durable state store with validated transitions among at
least:

- planned;
- blocked on dependencies;
- ready;
- claimed/running;
- succeeded with sealed evidence;
- retryable failure;
- terminal failure;
- deliberately shed/unavailable.

Define any additional states narrowly and document them.

Required claiming behavior:

- one active owner/token per job attempt;
- atomic claim and compare-and-swap style transition under the supported local
  filesystem rule;
- lease deadline and heartbeat evidence;
- stale-lease detection without stealing a live lease;
- idempotent repeated completion/failure reports;
- rejection of wrong worker/token, backwards transitions, and state tampering;
- job/attempt output isolation using validated relative paths;
- atomic output-manifest publication;
- successful transition only after output identity/schema/hash verification.

Tests must exercise competing claimants, crash before/after output publication,
partial/stale/conflicting outputs, expired/live leases, duplicate completion,
and traversal/symlink escape attempts.

**Part D passes when:** concurrent technical workers cannot claim or publish the
same logical attempt twice and no partial artifact is treated as success.

## Part E — Retry, terminal failure, resume, and reconciliation

Implement policy-injected bounded retry and restart-safe reconciliation.

Requirements:

- retry eligibility and maximum attempts supplied by an unresolved technical
  profile;
- retry index and derived seed evidence distinct from logical job identity;
- explicit failure taxonomy for worker error, numerical failure, validation
  failure, resource exhaustion, interruption, stale claim, dependency failure,
  and unavailable input;
- terminal failure after exhausted/nonretryable cases;
- downstream blocking that retains the failed dependency record;
- restart skips sealed successes and retains their exact hashes;
- restart reconciles running/claimed jobs from leases and output manifests;
- interrupted attempts remain visible rather than overwritten;
- resumed execution cannot change inputs, resource/priority references, or
  expected outputs;
- campaign completeness computed from states, never inferred from absence.

Required tests compare uninterrupted and interrupted/resumed technical
campaigns, force every failure class, retry at boundaries including zero, and
reject stale state-store/campaign/job hashes.

**Part E passes when:** a controller crash at each supported boundary can be
recovered deterministically without duplicate execution or hidden failure.

## Part F — Resource classes, priorities, and workload shedding

Implement validated injected records for:

- generic resource requirements such as CPU, accelerator capability, memory,
  scratch, walltime, and optional affinity labels;
- priority classes and stable tie-breaking;
- protected Tier 1/Tier 2/optional Tier 3 labels;
- maximum concurrency references without choosing production values;
- reverse-priority workload-shedding rules;
- incomplete-campaign reason and retained planned-job inventory.

The scheduler must:

- dispatch only ready jobs whose declared generic capability is satisfied;
- never claim unlike native resource units are equal;
- order equal-priority work deterministically;
- shed optional work before protected work under an injected policy;
- prohibit protected work from being silently shed;
- never use observed scientific effect size/direction or endpoint values for
  priority, retry, or shedding;
- report planned/completed/failed/shed counts by tier and family without
  scientific comparisons.

Adversarial tests reject priority derived from output metrics, resource-class
relabeling, protected-job shedding, nondeterministic ties, capacity overcommit,
and silent deletion of shed jobs.

**Part F passes when:** resource scarcity changes only declared operational
scope under an auditable policy and never silently changes scientific results
or identities.

## Part G — Local/generic adapters and portable campaign validation

Implement:

1. a deterministic local/in-process adapter for technical fixtures;
2. a generic job-array adapter that renders submission metadata/scripts and
   ingests scheduler observations without requiring a real scheduler;
3. operational launch/status/stop/resume/reconcile interfaces suitable for
   later Stage 14 wrapping.

The generic adapter must keep backend job/array IDs outside logical job
identity, validate array-index mapping, distinguish scheduler state from sealed
job state, and reject stale/mismatched observations. Symbolica-specific fields
remain explicit placeholders or injected adapter data until Stage 14 access is
known; do not fabricate them.

Build a validate-only CLI using a tiny synthetic DAG that:

- contains P1-like producer, P2-like consumer, independent branch, fan-in
  reducer, and optional job families;
- executes only harmless technical fixture workers;
- forces one interruption, one retryable failure, one terminal/dependency
  failure, one stale claim, and one shedding decision;
- resumes and reconciles to a deterministic final inventory;
- proves sealed outputs unblock dependants and partial outputs do not;
- prints operational-only status and explicit no-science boundaries.

Run from repository root and an unrelated cwd. Test at least two
`PYTHONHASHSEED` values with identical canonical report hashes. Run:

- Stage 12-P3 focused/adversarial tests;
- Stage 5C job-DAG compatibility;
- Stage 7 lifecycle/resume/inventory compatibility;
- Stage 12-P1/P2 producer-interface compatibility;
- Stage 4 identity/seed compatibility;
- relevant Stage 6A/6E lifecycle compatibility;
- Ruff on changed Python;
- diff, private-path, secret, large-file, binary, checkpoint, and LFS hygiene.

**Part G passes when:** the complete technical campaign can launch, fail,
resume, shed, reconcile, and report portably without a scheduler or registered
artifact.

## Part H — Commit, exact-SHA double-check, PR, and stop

Inspect the surface and ensure it is limited to Stage 12-P3 code,
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
- DAG/job/dependency identity design;
- claim/lease/output-isolation evidence;
- retry/resume/reconciliation and failure evidence;
- resource/priority/shedding boundary evidence;
- adapter and operational-status behavior;
- deterministic portable validation and test totals;
- unresolved RD-012/RD-013/RD-014 production choices;
- internal findings and descendant repairs;
- confirmation of no registered/private/scientific execution;
- interfaces exported to Austin 4, Alex 5, and Stage 14;
- explicit stop before Austin 4 and Stage 15.

Final status:

```text
AUSTIN_03_STAGE12P3_STATUS=COMPLETE_AT_HANDOFF_GATE
SCIENTIFIC_DATA=NO
PRODUCTION_DAG_FROZEN=NO
PRODUCTION_SCHEDULER_SELECTED=NO
STAGE15_STARTED=NO
```

## Prohibited shortcuts

- Do not equate scheduler completion with sealed job success.
- Do not put retry/worker/backend IDs into logical job identity.
- Do not infer campaign completeness from missing files.
- Do not delete failed, unavailable, or shed planned jobs from inventory.
- Do not display or use scientific effect directions in operational control.
- Do not hard-code Symbolica, Slurm, CUDA, concurrency, retry, lease, resource,
  tier, or shedding production values.
- Do not execute registered teachers, students, discovery, endpoints, or
  Fourier jobs.
- Do not begin compact export, Fourier execution, Stage 14, or Stage 15.
