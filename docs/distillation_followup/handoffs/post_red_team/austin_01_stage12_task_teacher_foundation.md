# Austin 1 — Stage 12-P1 task registry, teacher generation, and phase selection

## Paste this entire document into one fresh Chat-mode task

Repository: `AK-314/circuit-families-distillation`

Local clone convention: `~/Projects/circuit-families-distillation`

Required exact base:

```text
52b8f602614cb2b830ecc31f9c0200cbdcb4462e
```

Required branch:

```text
feat/stage-12p1-task-teacher-foundation
```

## Mission

Implement the policy-neutral task and teacher-production foundation needed by
the post-red-team design. This task may run immediately in parallel with Alex's
Stage 11 design-resolution task because it does not choose the production task
roster, teacher seeds, phase cells, budgets, or scientific interpretations.

Build configurable interfaces and validate them on tiny synthetic/excluded
fixtures. Do not train definitive teachers or access private predecessor
checkpoints.

## Authorities — read completely before acting

1. `docs/distillation_followup/post_red_team_protocol_amendment.md`
2. `docs/distillation_followup/distillation_implementation_post_red_team.md`
3. `docs/distillation_followup/red_team/red_team_resolution_matrix.md`
4. `followup/configs/post_red_team_open_decisions_v1.json`
5. `docs/distillation_followup/stage9_training_backend_benchmark.md`
6. `docs/distillation_followup/handoffs/post_red_team/handoff_sequence.md`
7. existing Stage 3 teacher registry/phase-selection implementation;
8. existing Stage 5B/5C target-cache, trainer, and orchestration implementation.

The old Stage 2 authorities and UD register remain immutable. Alex's Stage 11
candidate interface may be integrated later if published during this task, but
do not wait for it to implement policy-neutral abstractions.

## Scientific boundary

Permitted:

- source/API inspection;
- synthetic modular domains and tiny toy models;
- runtime, memory, resume, serialization, and deterministic-identity tests;
- training-metric-only phase-selection fixtures;
- failure/unavailable fixtures;
- interface compatibility checks.

Prohibited:

- production teacher roster selection;
- definitive mod-113, multiplication, or polynomial teacher training;
- private predecessor/checkpoint access;
- circuit discovery or endpoint computation;
- choosing task formulas or phase rules according to apparent scientific
  effects;
- resolving RD-001, RD-002, RD-003, RD-012, RD-013, or RD-014.

All emitted fixtures must be explicitly technical/non-production.

## Chat protocol

- Stay in Chat mode for this handoff; never mix Chat and Work mode.
- Complete Parts A–G in order.
- Each operational response contains exactly one fenced terminal block.
- Explain briefly what the block does and what diagnostics it prints.
- Austin returns complete stdout/stderr before the next block.
- A part may need multiple one-block turns; one successful command does not
  automatically complete a part.
- Never respond with only “Part complete” when the next block is available.
- Use focused tests during development and one compatibility/full portable gate
  at the end.
- End every response with:

```text
HANDOFF=AUSTIN_01_STAGE12P1
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=52b8f602614cb2b830ecc31f9c0200cbdcb4462e
HEAD=<exact current SHA>
WAITING_FOR=<NONE or exact blocker>
SCIENTIFIC_DATA=NO
```

## Design constraints

- Reuse existing dataset/config/hash/seed/manifest utilities.
- Reuse Stage 3 phase-selection semantics through an adapter; do not fork its
  logic invisibly.
- Reuse the Stage 5B/5C training and target-cache contracts.
- Task definitions are data/config objects, not `if task == ...` branches spread
  through training code.
- Production rosters and numeric budgets remain injected configuration.
- Every job is resumable and writes atomically.
- A teacher/task/phase can terminate unavailable without being converted to a
  zero or silently replaced.
- No absolute private paths or large artifacts enter Git.

## Expected implementation surface

Prefer one cohesive namespace following repository conventions, for example:

- task/domain/target protocol and registry;
- teacher construction/training/resume adapter;
- training-metric trajectory record;
- phase-selection adapter and unavailable status;
- sealed technical teacher artifact record;
- validate-only CLI;
- focused and adversarial tests.

Do not force these filenames if existing modules provide a cleaner home. Before
creating files, present the reuse audit and explain the selected locations.

## Part A — Exact-base and portability guard

The first block is read-only and must print:

- repository root and remote;
- current branch, HEAD, `main`, and `origin/main`;
- exact equality with required base `52b8f602...`;
- clean tracked state and separately listed untracked files;
- authority hashes;
- Python/environment identity;
- presence of reusable Stage 3 and Stage 5B/5C modules/tests;
- absence of Stage 12-P1 collisions and Stage 15 artifacts;
- confirmation that no private predecessor root is needed.

After returned output is diagnosed, a second block may create the branch.

**Part A passes when:** branch starts at the exact base, tracked tree is clean,
and no scientific/private artifact has been accessed.

## Part B — Reuse and gap audit

Inspect existing code and produce a concise, evidence-backed map of:

- modular-addition dataset/task assumptions currently hard-coded;
- model-construction assumptions;
- teacher-training and checkpoint/resume support;
- Stage 3 trajectory and phase-selection inputs;
- target-cache identity dependencies;
- deterministic seed derivation;
- artifact schemas and sealing;
- generic versus task-specific orchestration;
- portable CLI/test patterns.

Classify each requirement as:

- direct reuse;
- adapter required;
- new policy-neutral implementation;
- dependent on Alex's Stage 11 interface;
- intentionally deferred to Austin 2–6 or Alex-owned work.

Do not modify code until the audit identifies exact reuse points and collision
risks.

**Part B passes when:** the chosen design extends existing contracts rather than
creating a competing training/registry system.

## Part C — Configurable modular-task protocol and registry

Implement a versioned task protocol capable of representing, without choosing
a production roster:

- ordered finite input domains;
- output vocabulary/modulus;
- deterministic target computation;
- canonical example ordering;
- train/test split identity where applicable;
- task configuration and implementation version;
- dataset/domain/target hashes;
- architecture compatibility requirements;
- complete condition-identity material.

It must support technical fixtures corresponding to:

- modular addition;
- modular multiplication;
- a configurable modular polynomial supplied as explicit coefficients or an
  equivalently canonical representation.

The polynomial fixture is not a frozen Task 3 formula. Registry ordering or
names must not imply scientific priority.

Tests must reject duplicate identities, noncanonical domains, invalid modulus,
inconsistent hashes, target/order changes, unsupported output ranges, and
nonserializable task definitions.

**Part C passes when:** the same generic consumer can build and hash all task
fixtures without task-specific branches outside registered implementations.

## Part D — Teacher construction, training, and resume adapter

Implement a teacher-production adapter that accepts injected:

- task record;
- architecture/config record;
- model/training seed identities;
- optimizer/schedule/stopping configuration;
- maximum technical budget;
- checkpoint cadence/retention policy;
- output root and resume identity.

Required behavior:

- deterministic initialization under the qualified backend contract;
- atomic rolling checkpoint plus bounded retention;
- resume without duplicating completed updates;
- uninterrupted/resumed equivalence under the supported deterministic rule;
- trajectory logging limited to metrics needed for technical monitoring and
  later phase selection;
- terminal completed, failed, interrupted, numerical-failure, and unavailable
  states;
- no dense history or verbose JSON storage explosion;
- sealed final technical artifact with config, task, seed, environment, and
  content hashes.

Use tiny models/domains and very short training only. Any fixture accuracy is
technical and must not be interpreted scientifically.

**Part D passes when:** a synthetic teacher can start, interrupt, resume, seal,
reload, and reproduce its declared artifact without production policy.

## Part E — Training-metric trajectory and phase-selection adapter

Adapt the Stage 3 phase-selection machinery so it can consume a generic sealed
teacher trajectory while preserving:

- selection from training/test metrics only;
- per-seed phase selection;
- explicit pre/transition/stable-post identities;
- unavailable phase outcomes;
- no circuit/distillation/endpoint inputs;
- immutable checkpoint hashes;
- canonical table and registry records.

Do not silently alter the historical Stage 3 rule. If the expanded-teacher rule
must differ, expose it as injected versioned policy to be frozen by Alex 5,
with the historical rule available as a separate implementation.

Adversarial tests must prove that circuit, student, packing, or Fourier fields
cannot enter selection; missing bounds remain unavailable; dirty/unsealed
checkpoints cannot be selected; and common training steps are not relabeled as
common functional phases without qualification.

**Part E passes when:** synthetic trajectories produce deterministic selected
or unavailable records independent of any mechanistic outcome.

## Part F — Records, portable CLI, integration, and validation

Provide versioned records and a validate-only CLI covering:

- task identity and hashes;
- teacher attempt identity;
- trajectory/checkpoint inventory;
- resume lineage;
- phase-selection result;
- sealed technical teacher;
- terminal failure/unavailable status;
- explicit `scientific_data=false` and `production_eligible=false` boundary.

Validate from repository root and an unrelated working directory. Test at
least two `PYTHONHASHSEED` values where deterministic serialization is claimed.

Run:

- focused task/teacher/phase tests;
- Stage 3 registry compatibility;
- Stage 5B/5C cache/trainer compatibility;
- resume and failure adversarial tests;
- Ruff on changed Python;
- diff hygiene and private-path/secret/large-file checks.

Do not access the private predecessor even if it exists on the current machine.

**Part F passes when:** the complete technical lifecycle validates portably,
records are compact and deterministic, and historical consumers still work.

## Part G — Commit, exact-SHA double-check, PR, and handoff

Before committing, inspect the complete surface and confirm it contains only
the task/teacher foundation plus tests/docs/validation. Ensure no real
checkpoint, dense output, cache, or scientific table is staged.

Create coherent commits without amend or force-push. Push and open a PR against
`main`. In a fresh checkout at the final exact SHA:

- rerun focused/adversarial and compatibility tests;
- run the validate-only CLI from an unrelated cwd;
- verify tracked cleanliness;
- inspect artifact sizes and Git/LFS surface;
- record blocking, nonblocking, and question findings;
- repair blockers only through descendant commits and recheck the new exact
  SHA.

Do not merge from this handoff unless the master task explicitly authorizes it.

Final report:

- base, branch, final SHA, parent, and PR;
- exact changed files;
- reuse decisions;
- test/validation totals;
- resume/determinism evidence;
- artifact-size evidence;
- internal findings/repairs;
- confirmation of no private/scientific work;
- interfaces exported to Austin 2 and Alex 5;
- explicit stop before Austin 2 or Stage 15.

Final status:

```text
AUSTIN_01_STAGE12P1_STATUS=COMPLETE_AT_HANDOFF_GATE
SCIENTIFIC_DATA=NO
PRODUCTION_TEACHERS_TRAINED=NO
STAGE15_STARTED=NO
```

## Prohibited shortcuts

- Do not hard-code the final three tasks or teacher roster into training.
- Do not treat fixture convergence as task-selection evidence.
- Do not rewrite Stage 3 to make expanded teachers appear historically frozen.
- Do not begin multi-architecture student work, campaign orchestration, compact
  export, Fourier execution, or Stage 14 cluster packaging in this task.
- Do not access or copy private predecessor artifacts.
