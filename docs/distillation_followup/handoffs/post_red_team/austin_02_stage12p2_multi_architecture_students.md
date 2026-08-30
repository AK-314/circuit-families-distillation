# Austin 2 — Stage 12-P2 multi-architecture student training and eligibility

## Paste this entire document into one fresh Chat-mode task

Repository: `AK-314/circuit-families-distillation`

Local clone convention: `~/Projects/circuit-families-distillation`

Required scientific authority floor:

```text
d36f1b442ab7b783f3211377303a2981fc0d00e3
```

Required branch:

```text
feat/stage-12p2-multi-architecture-students
```

Create the branch from current `origin/main` after this handoff is merged. Part
A must prove the authority floor is an ancestor, local `main` exactly matches
`origin/main`, and both Barrier 1 packages are present. Record current
`origin/main` as the implementation base.

## Mission

Implement policy-neutral multi-architecture student construction, shared
hard/soft training compatibility, component accounting, eligibility, sealing,
resume, and failure records.

The Stage 11 interface permits a canonical predecessor-matched family plus a
balanced sparse panel of up to five candidate families, but the exact roster,
assignments, training settings, soft tolerance, replication, and production
eligibility policies remain Stage 13 decisions. This task builds the mechanism;
it does not select those values.

Alex simultaneously implements the independent discovery family. Provide a
clean dense-model/component interface so later discovery code does not depend
on one concrete architecture.

## Authorities — read completely before acting

1. `docs/distillation_followup/stage11_post_red_team_design_resolution.md`
2. `followup/configs/stage11_post_red_team_design_candidates_v1.json`
3. `followup/manifests/stage11_red_team_resolution_v1.json`
4. Stage 12-P1 task/teacher source, tests, and validate-only CLI;
5. existing Stage 5B/5C trainer, identity, cache, checkpoint, and DAG contracts;
6. existing Stage 6B hard eligibility and Stage 6C soft eligibility contracts;
7. Stage 4 common condition/schema/seed authorities;
8. `docs/distillation_followup/handoffs/post_red_team/handoff_sequence.md`.

Do not alter the Stage 11 candidate records or Stage 12-P1 foundation merely to
make this implementation easier. Extend through explicit adapters.

## Scientific boundary

Permitted:

- tiny technical architectures and synthetic task records;
- synthetic teacher target caches;
- one/few-update training fixtures;
- hard/soft loss, eligibility, resume, accounting, serialization, runtime, and
  memory tests;
- constructed passed/failed/unavailable cases.

Prohibited:

- definitive student training;
- registered teacher checkpoints or private predecessor access;
- selecting the production architecture roster or sparse assignment;
- selecting hard/soft optimizers, budgets, soft tolerance, argmax rule,
  replication, or attempt cap from outcomes;
- circuit discovery or endpoint calculation;
- resolving RD-002, RD-003, RD-004, RD-010, RD-012, RD-013, or RD-014.

All executable profiles remain technical and non-production.

## Chat protocol

- Use Chat mode for the entire handoff; never mix Chat and Work modes.
- Complete Parts A–H in order.
- Give exactly one fenced terminal block per operational response.
- Briefly explain what it changes/inspects and which diagnostics it prints.
- Austin returns complete stdout/stderr before the next block.
- A part may take multiple one-block turns; do not collapse a major
  implementation into one opaque command.
- Never provide only a prose completion when the next block is ready.
- Use focused tests while developing and one exact-SHA integration gate at the
  end.
- Preserve user-owned/unrelated untracked files.
- End every response with:

```text
HANDOFF=AUSTIN_02_STAGE12P2
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact implementation base recorded in Part A>
HEAD=<exact current SHA>
WAITING_FOR=<NONE or exact blocker>
SCIENTIFIC_DATA=NO
```

## Design constraints

- Architecture is a versioned injected record, never inferred from a checkpoint
  filename or condition label.
- The canonical family must reproduce the predecessor-matched student
  construction through an explicit compatibility adapter.
- Additional technical architectures exercise depth/width/head/MLP variation,
  but fixture choices do not freeze the production panel.
- Hard and soft students remain separate estimands and record types.
- Hard eligibility remains exact full-domain argmax agreement.
- Soft matching remains gauge-invariant; tolerance and additional argmax rule
  are injected unresolved policy.
- Circuit fidelity will later be relative to the student's own dense outputs.
- Parameter count, searchable component count, and component-type counts are
  explicit per architecture.
- Ineligible/failed attempts remain records and never enter discovery queues.

## Expected implementation surface

Prefer one cohesive namespace, such as `stage12p2`, containing:

- architecture/config records and registry;
- model-builder adapters;
- searchable-component accounting and dense-model descriptor;
- architecture-aware student identity and initialization;
- shared trainer adapters for hard and soft conditions;
- architecture-aware eligibility/sealing/failure records;
- portable validate-only CLI;
- focused/adversarial tests.

Reuse Stage 5B/5C/6B/6C APIs. Explain selected locations after the Part B audit
before creating files.

## Part A — Exact-base, ancestry, and scope guard

The first block is read-only and must print:

- repository/remote identity, branch, HEAD, local `main`, and `origin/main`;
- authority-floor ancestry and exact local-main equality;
- merged PR #17/#18 identities and expected files;
- Stage 11 candidate hashes and Stage 12-P1 validate-only status;
- clean tracked state and separately listed untracked files;
- relevant Stage 4/5B/5C/6B/6C modules/tests;
- no Stage 12-P2 collision or Stage 15 artifact/process;
- environment/backend identity;
- no dependency on private predecessor artifacts.

After returned output diagnosis, a second block may create the feature branch.

**Part A passes when:** the branch starts from shared main containing both first
packages, tracked state is clean, and no private/scientific artifact is touched.

## Part B — Reuse audit and cross-architecture contract

Inspect and map:

- current canonical model config/construction;
- Stage 5B/5C student identity and initialization;
- shared trainer loss/optimizer/checkpoint interfaces;
- Stage 6B/6C hard/soft loss and eligibility adapters;
- Stage 4 condition identities and sealed-dense-model records;
- Stage 12-P1 task/teacher/cache identities;
- current component-mask assumptions in interpretability/discovery code;
- parameter/component counting utilities;
- architecture-dependent activation hooks and dense-output generation.

Classify each requirement as direct reuse, adapter, new policy-neutral code, or
deferred. Define the minimum architecture-neutral interfaces consumed later by:

- target/cache and trainer code;
- eligibility and sealing;
- component-basis construction;
- dense model exact evaluation;
- discovery and analysis.

Do not make later discovery import a Stage 12-P2 concrete model class.

**Part B passes when:** the new layer extends existing contracts and makes all
architecture assumptions explicit.

## Part C — Architecture records, registry, and model construction

Implement versioned architecture records containing at least:

- family/name/version;
- task/vocabulary/context compatibility;
- layer, model-width, head, and MLP dimensions;
- activation/normalization/positional settings needed for exact construction;
- parameter count;
- searchable component types/counts;
- initialization policy reference;
- builder implementation identity/hash;
- production-selection and scientific-data flags.

Implement a registry/model-builder protocol supporting:

- the canonical predecessor-matched architecture through an explicit adapter;
- several tiny technical transformer variants sufficient to prove depth, width,
  head, and MLP differences are representable;
- rejection of duplicate identities, invalid divisibility/dimensions,
  inconsistent parameter/component counts, unsupported task/vocabulary
  combinations, and builder-record mismatch.

Do not freeze a five-family roster. Technical fixtures demonstrate capability,
not inclusion.

**Part C passes when:** construction is deterministic from a validated record
and architecture identity fully explains the resulting model shape/counts.

## Part D — Searchable-component accounting and dense-model descriptor

Implement architecture-aware accounting for:

- parameter count and optionally parameters per component;
- attention-head count by layer;
- MLP-neuron count by layer;
- canonical searchable component count;
- component type/layer/index identity;
- hooks/intervention targets needed by later basis packages;
- dense-output reference identity;
- component-basis compatibility hash.

The interface must support multi-layer architectures without flattening away
layer identity. It must reject component descriptors from another architecture,
invalid masks, duplicate components, count/hash mismatch, and attempts to
compare raw component proportions without denominator metadata.

Do not implement Alex's alternate bases here. Export enough canonical metadata
for Alex 3 to construct them separately.

**Part D passes when:** every technical model has a reproducible component
inventory and later discovery can operate through the descriptor rather than
hard-coded 4-head/512-neuron assumptions.

## Part E — Architecture-aware student identity, initialization, and training

Extend existing student identity/trainer adapters so the complete identity
includes task, teacher, phase, condition, architecture, initialization,
training policy, and backend identity.

Required behavior:

- deterministic initialization under the supported backend rule;
- hard and soft adapters share model construction/bookkeeping;
- task/teacher caches validate against architecture/task output contracts;
- optimizer/schedule/stopping/budget remain injected profiles;
- atomic bounded rolling checkpoints;
- interruption/resume without duplicate attempts;
- explicit completed/failed/interrupted/numerical-failure/unavailable statuses;
- compact trajectory and checkpoint inventory;
- no architecture silently substituted after resume.

Use tiny synthetic caches/tasks and minimal updates. Test canonical and at least
two structurally distinct technical architectures.

**Part E passes when:** hard/soft technical attempts train, interrupt, resume,
and seal through one shared engine across different architectures.

## Part F — Eligibility, sealing, failure accounting, and queue boundary

Integrate Stage 6B/6C semantics across architectures:

- hard exact agreement over the supplied complete technical domain;
- gauge-invariant soft comparison;
- injected unresolved soft tolerance and optional argmax requirement;
- dense output/checkpoint/config/task/teacher/architecture hashes;
- passed, ineligible, optimization-failed, numerical-failed, interrupted, and
  unavailable attempt records;
- no imputation or hidden replacement;
- passed-only discovery release record.

Adversarial tests must prove:

- one hard mismatch blocks eligibility;
- gauge shifts do not alter soft assessment;
- architecture/task/cache/checkpoint hash mismatch rejects;
- failed/ineligible attempts remain counted;
- a sealed model cannot change architecture after eligibility;
- ineligible students never enter the discovery queue;
- hard/soft records cannot be pooled or relabeled;
- production eligibility cannot be asserted under a technical profile.

**Part F passes when:** eligibility is architecture-agnostic but preserves the
separate hard/soft scientific boundaries and complete failure accounting.

## Part G — Portable end-to-end validation and compatibility

Build a validate-only CLI that, using synthetic fixtures only:

1. creates multiple task and architecture records;
2. constructs distinct student models;
3. builds synthetic teacher targets/caches;
4. runs minimal hard/soft updates;
5. interrupts/resumes one attempt;
6. evaluates passed and failed eligibility cases;
7. seals eligible technical models and blocks ineligible discovery release;
8. prints parameter/component accounting and scientific-boundary status.

Run from repository root and an unrelated cwd. Test deterministic records under
multiple `PYTHONHASHSEED` values. Run:

- Stage 12-P2 focused/adversarial tests;
- Stage 12-P1 task/teacher compatibility;
- Stage 4 schema/identity compatibility;
- Stage 5B/5C trainer/cache/resume compatibility;
- Stage 6B hard and Stage 6C soft compatibility;
- relevant Stage 7 technical integration;
- Ruff on changed Python;
- diff, private-path, secret, large-file, binary, checkpoint, and LFS audits.

No real checkpoint, dense output, or cache may enter Git.

**Part G passes when:** multiple technical architectures complete the same
student lifecycle portably without resolving a production roster or policy.

## Part H — Commit, exact-SHA double-check, PR, and stop

Inspect the surface and ensure it is limited to Stage 12-P2 code,
tests/validation, and necessary technical documentation. Create coherent
commits without amend/force-push. Push and open a PR against `main`.

At the final exact SHA, run a fresh detached-checkout double-check covering
focused/adversarial and compatibility tests, unrelated-cwd CLI, Ruff, diff,
tracked cleanliness, artifact sizes, and Git/LFS surface. Classify findings as
blocking, nonblocking, or question; repair blockers only in descendants and
repeat against the new exact SHA.

Do not merge inside this handoff without master-task authorization.

Final report:

- base, branch, parent, final SHA, and PR;
- exact changed files;
- architecture/model/component interfaces;
- hard/soft trainer and eligibility evidence;
- resume/determinism/failure evidence;
- test totals and artifact sizes;
- unresolved production choices;
- internal findings/repairs;
- no scientific/private execution;
- interfaces exported to Austin 3, Alex 3/5, and discovery consumers;
- explicit stop before Austin 3 and Stage 15.

Final status:

```text
AUSTIN_02_STAGE12P2_STATUS=COMPLETE_AT_HANDOFF_GATE
SCIENTIFIC_DATA=NO
PRODUCTION_ARCHITECTURE_SELECTED=NO
PRODUCTION_STUDENTS_TRAINED=NO
STAGE15_STARTED=NO
```

## Prohibited shortcuts

- Do not hard-code a five-architecture production panel.
- Do not fork separate hard/soft model construction or bookkeeping systems.
- Do not hide layer identity in a flat component index.
- Do not choose soft tolerance, optimizer, attempt cap, or architecture from
  fixture success.
- Do not run registered teachers/students or circuit search.
- Do not begin Austin 3 orchestration, compact export, Fourier, Stage 14, or
  Stage 15 from this task.
