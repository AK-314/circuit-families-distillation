# Alex 3 — Stage 12-R2 basis re-granulation and orientation sensitivity

## Paste this entire document into one fresh Chat-mode task

Repository: `AK-314/circuit-families-distillation`

Local clone convention: `~/Projects/circuit-families-distillation`

Required scientific authority floor:

```text
d36f1b442ab7b783f3211377303a2981fc0d00e3
```

Required Stage 12-R1 floor:

```text
841269830cd132d2d90e4e1405bd9029816eb4d7
```

Required branch:

```text
feat/stage-12r2-basis-sensitivity
```

Create the branch from the current `origin/main` after this handoff is merged.
Part A must prove that both required floors are ancestors, that local `main`
exactly matches `origin/main`, and that the tracked tree is clean. Record the
current `origin/main` SHA as the implementation base.

## Mission

Implement and validate the technical machinery needed to test whether sparse
circuit recovery changes under a limited, prospectively specified set of
component re-granulations and coordinate orientations.

The package must support:

1. the existing canonical head-plus-neuron basis without changing its meaning;
2. pre-output-projection attention coordinates;
3. seeded balanced blocks of MLP neurons;
4. fixed seeded orthogonal rotations in eligible activation subspaces;
5. raw-component, parameter-weighted, and component-type-stratified accounting;
6. explicit compatibility rules that reject invalid cross-basis comparisons.

This is technical Stage 12 implementation. It does not select the production
basis panel, coordinate definition, block count, partition seed, rotation count,
rotation seed, model assignment, or reporting emphasis. RD-004 remains open for
Stage 13.

Austin may simultaneously implement Stage 12-P2 multi-architecture students.
Do not import an Austin-specific concrete model class. Define capability-based
interfaces so later architecture implementations can expose compatible
activation and component metadata.

## Authorities — read completely before acting

1. `docs/distillation_followup/stage11_post_red_team_design_resolution.md`
2. `followup/configs/stage11_post_red_team_design_candidates_v1.json`
3. `followup/manifests/stage11_red_team_resolution_v1.json`
4. `followup/configs/post_red_team_open_decisions_v1.json`
5. `docs/distillation_followup/post_red_team_protocol_amendment.md`
6. `docs/distillation_followup/distillation_implementation_post_red_team.md`
7. existing Stage 4 component-basis and sealed-model contracts;
8. existing Stage 5A centred-logit evaluator and Stage 6A/6E endpoint ledgers;
9. merged Stage 12-R1 contracts, records, exact bridge, and tests;
10. `docs/distillation_followup/handoffs/post_red_team/handoff_sequence.md`.

The Stage 11 authorities require a bounded basis-sensitivity panel but leave
its production values unresolved. They prohibit universal basis-invariance
claims and raw-count comparisons that ignore changed granularity.

## Scientific boundary

Permitted:

- synthetic tensors, toy attention/MLP modules, masks, and logits;
- tiny technical networks and constructed activation subspaces;
- exact algebraic tests of masking, grouping, rotation, inverse mapping, and
  accounting;
- deterministic identity, serialization, runtime, and memory diagnostics;
- explicitly excluded technical fixture endpoints.

Prohibited:

- loading or executing registered teacher or student checkpoints;
- comparing phases, hard/soft conditions, architectures, teacher seeds, or
  scientific endpoint values;
- selecting a basis, block partition, or rotation because it gives a favorable
  recovery result;
- changing the canonical basis definition retrospectively;
- resolving RD-004 or any other production-blocking RD item;
- presenting technical invariance or sensitivity as a project result.

Every executable profile and result must state `scientific_data=false` and
`production_eligible=false`.

## Chat protocol

- Stay in Chat mode for the entire handoff; do not mix Chat and Work modes.
- Complete Parts A–G in order. The letters reflect real gates, not a target
  count; use multiple one-block turns inside a part when needed.
- Every operational response contains exactly one fenced terminal block.
- Briefly state what the block changes or inspects and which diagnostics it
  prints.
- Alex returns complete stdout/stderr before the next block.
- Never answer only “Part complete” when the next block is available.
- Use focused tests during implementation and one exact-SHA integration gate at
  the end; do not rerun the full historical suite after each edit.
- Preserve user-owned untracked files, especially the presentation directory
  and red-team dossier.
- End every response with:

```text
HANDOFF=ALEX_03_STAGE12R2
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact implementation base recorded in Part A>
HEAD=<exact current SHA>
WAITING_FOR=<NONE or exact blocker>
SCIENTIFIC_DATA=NO
```

## Core invariants

### Basis identity

Every basis must have a versioned identity and content hash covering at least:

- parent model/component-basis identity;
- basis family and coordinate definition;
- ordered component descriptors and types;
- intervention location and semantics;
- grouping/partition identity where applicable;
- rotation/subspace identity where applicable;
- component-count and parameter-weight denominator definitions.

Display labels must never substitute for hashes. A mask from one basis cannot
be evaluated or reduced under another merely because dimensions coincide.

### Functional intervention

All-on intervention in every supported basis must reproduce the unmasked
technical model within an explicitly tested numerical tolerance. All-off and
partial masks must act at the documented location. A coordinate transform must
not alter model parameters or silently change the dense reference.

### Comparison boundary

Raw component proportions are meaningful only within an identified basis.
Cross-basis reporting must state the accounting measure and compatibility
relationship. Parameter-weighted and type-stratified summaries supplement;
they do not make different bases causally identical.

## Expected implementation surface

Prefer an isolated `stage12r2` namespace containing:

- basis/component descriptors and canonical serialization;
- capability protocols for activation interception and mask application;
- canonical-basis compatibility adapter;
- attention-coordinate adapter;
- deterministic balanced-block partitioning;
- deterministic orthogonal-rotation construction and intervention;
- component/parameter/type accounting;
- comparison validation and technical reducers;
- validate-only CLI and focused/adversarial tests.

Reuse Stage 4/5A/6A/6E/12-R1 identities and ledgers rather than defining new
scientific endpoints. Explain selected paths after the Part A/B audit and
before creating files.

## Part A — Exact-base, scope, and collision guard

The first block is read-only. It must print and verify:

- repository root, remote, branch, HEAD, local `main`, and `origin/main`;
- authority-floor and Stage 12-R1-floor ancestry;
- exact hashes of the Stage 11 candidate/resolution records;
- merged PR #18 and #20 identities;
- tracked cleanliness and separately listed untracked files;
- absence of Stage 12-R2 and Stage 15 artifacts/processes;
- existing component-basis, intervention, evaluator, endpoint, packing, and
  Stage 12-R1 APIs/tests;
- installed PyTorch/backend versions needed by matrix operations;
- confirmation that no private predecessor access is required.

After diagnosing that output, a second block may create the required feature
branch.

**Part A passes when:** the branch starts from recorded shared main, required
floors are in its ancestry, the tracked tree is clean, user artifacts are
preserved, and no scientific artifact has been accessed.

## Part B — Reuse audit and versioned basis contract

Inspect the existing source and establish which contracts can be reused for:

- canonical component descriptors and ordering;
- sealed dense-model and component-basis identity;
- mask shape/basis validation;
- exact centred-logit evaluation;
- Endpoint 1/2 ledger consumption;
- hashing, seed derivation, canonical JSON, and failure records.

Implement a versioned basis contract and capability protocols that do not
assume one model class. Require explicit metadata for component type, source
subspace, intervention location, parameter weight, parent basis, and ordered
coordinate identity. Define compatibility relations such as same basis,
refinement/coarsening, and rotated view without treating them as equivalence of
scientific interpretation.

Required adversarial tests reject:

- duplicate or reordered component identities under a stale hash;
- identical display names with different basis hashes;
- masks carrying the wrong basis identity;
- malformed parent/refinement relationships;
- absolute private paths or large tensor payloads in records;
- `scientific_data=true` or `production_eligible=true`.

**Part B passes when:** later adapters can share one strict identity contract
without importing registered models or selecting production basis values.

## Part C — Canonical compatibility and attention-coordinate refinement

First wrap the existing canonical head-plus-individual-neuron basis without
changing its component order, mask meaning, denominator, or exact-evaluation
behavior. Add regression tests against existing technical fixtures.

Then implement a capability-based pre-output-projection attention-coordinate
view. The adapter must make explicit:

- the tensor and axis on which coordinates are defined;
- head/layer/coordinate ordering;
- whether masks act before the output projection and how broadcasting works;
- mapping from refined coordinates to their parent attention head;
- parameter-weight accounting policy supplied as metadata, not inferred from a
  favorable result;
- unsupported-model and shape failures.

Required tests include:

- all-on equivalence to the same dense technical output;
- all-off and single-coordinate effects on a constructed attention fixture;
- exact reconstruction of a parent-head mask by its full coordinate group;
- deterministic identities across process/hash seeds;
- rejection of post-projection tensors mislabeled as pre-projection;
- no private Endpoint 1/2 or fidelity implementation.

**Part C passes when:** canonical and refined attention masks use the common
evaluation path while retaining distinct, auditable basis identities.

## Part D — Seeded balanced neuron blocks and accounting

Implement deterministic balanced partitions of eligible MLP neurons using the
complete model/basis/layer identity plus an injected partition seed and block
count. The algorithm must:

- cover each eligible neuron exactly once;
- produce no empty, duplicate, or overlapping membership;
- keep block sizes differing by at most one where mathematically possible;
- define stable block ordering independent of set/dictionary iteration;
- distinguish different seeds, policies, layers, and parent bases in identity;
- reject impossible block counts and stale/tampered membership hashes.

Block masks must expand deterministically to parent-neuron masks. Prove all-on
equivalence and exact agreement between a block mask and its expanded parent
mask on technical fixtures.

Implement accounting records for:

- raw component count/proportion in the active basis;
- parameter-weight total and retained proportion with a declared denominator;
- component-type counts and retained proportions;
- parent-neuron coverage for grouped bases.

Do not claim that parameter weighting corrects all granularity dependence.

**Part D passes when:** grouping and accounting are complete, deterministic,
reconstructable, and cannot be mistaken for canonical raw-component counts.

## Part E — Fixed orthogonal rotations and intervention semantics

Implement fixed seeded orthogonal rotations for explicitly eligible activation
subspaces. Use a documented deterministic construction, including sign or
orientation normalization so the same identity reproduces the same matrix.
Record algorithm/version, dtype, dimension, complete subspace identity, seed,
matrix hash, and inverse/transpose convention.

The rotated intervention must follow an explicit sequence:

1. map eligible activations into the rotated coordinates;
2. apply the binary coordinate mask there;
3. map back before downstream computation.

Required tests include:

- orthogonality and deterministic matrix/hash reproduction;
- changed seed/subspace identity changes the rotation identity;
- all-on rotated intervention equals the same unmasked technical model;
- identity rotation matches the parent-coordinate mask;
- full mask round-trip is numerically stable under declared tolerance;
- non-orthogonal, wrong-dimensional, stale-hash, wrong-dtype, and wrong-basis
  matrices reject;
- model parameters and dense reference outputs remain unchanged;
- no rotation is fitted or selected using scientific endpoint behavior.

Large matrices must not be embedded in normal JSON records. Store only compact
reconstruction metadata/hashes unless a tiny excluded fixture explicitly tests
serialization.

**Part E passes when:** orientation changes are fixed, reproducible technical
views with correct intervention behavior rather than learned transformations.

## Part F — Comparison guards, stratified reducers, and lifecycle records

Provide technical reducers/records that can consume exact-ledger outputs and
report, without redefining endpoints:

- basis-specific component proportion;
- parameter-weighted retained proportion;
- component-type-stratified retained counts/proportions;
- parent coverage for refinement/coarsening views;
- failure/unavailable/censored state with intact-mask behavior preserved.

Implement an explicit cross-basis comparison request. It must validate model,
dense-reference, fidelity-definition, evaluation-domain, intervention protocol,
and basis relationship identities. Reject raw comparisons when denominators or
accounting definitions are silently mixed. Preserve unfavorable and negative
exact fidelity values unchanged.

Versioned lifecycle records must retain transform/partition provenance,
technical profile classification, exact-ledger references, and unresolved
production fields. RD-004 and its exact basis/seed/assignment choices must
remain null or explicitly unresolved.

Adversarial tests include:

- canonical result relabeled as rotated/refined;
- raw count copied across different granularity;
- parameter denominator substitution;
- component-type omission;
- stale transform/partition hash;
- cross-model or cross-domain comparison;
- clipped fidelity or dropped failed mask;
- technical profile altered to production eligible.

**Part F passes when:** valid sensitivity summaries are reconstructable and
invalid basis-invariance claims cannot be manufactured by relabeling records.

## Part G — Portable integration, exact-SHA double-check, PR, and stop

Build a portable validate-only CLI exercising one tiny technical model through:

1. canonical basis compatibility;
2. attention-coordinate refinement where supported by the fixture;
3. two deterministic balanced-block partitions;
4. identity plus at least one nontrivial fixed rotation;
5. common exact evaluation and endpoint-ledger consumption;
6. raw, parameter-weighted, and type-stratified summaries;
7. a deliberately rejected invalid cross-basis comparison;
8. explicit technical/no-science boundary output.

Run from repository root and an unrelated working directory. Test at least two
`PYTHONHASHSEED` values and require identical canonical report hashes. Run:

- all Stage 12-R2 focused/adversarial tests;
- canonical Stage 4 component-basis compatibility;
- Stage 5A fidelity and Stage 6A/6E endpoint compatibility;
- Stage 12-R1 contract/ledger compatibility where relevant;
- Ruff on changed Python;
- diff, private-path, secret, large-file, binary, checkpoint, and LFS hygiene.

Inspect the full surface and confirm only Stage 12-R2 implementation,
tests/validation, and necessary technical documentation changed. Create
coherent commits without amend or force-push. Push and open a PR against `main`.

At the final exact SHA, use a fresh detached checkout to rerun focused,
adversarial, compatibility, CLI, Ruff, diff, cleanliness, and artifact-hygiene
checks. Classify findings as blocking, nonblocking, or question. Repair blocking
findings only through descendant commits and repeat against the new exact SHA.

Do not merge from this handoff unless the master task explicitly authorizes it.

Final report:

- base, branch, parent, final SHA, and PR;
- exact changed files and artifact sizes;
- basis identity and intervention design;
- canonical compatibility evidence;
- attention-refinement, block, rotation, and accounting evidence;
- comparison-guard and adversarial results;
- deterministic/portable validation and test totals;
- unresolved RD-004 production choices;
- internal findings and repairs;
- confirmation of no registered/private/scientific execution;
- interfaces exported to Alex 4/5 and Austin 5;
- explicit stop before Alex 4 and Stage 15.

Final status:

```text
ALEX_03_STAGE12R2_STATUS=COMPLETE_AT_HANDOFF_GATE
SCIENTIFIC_DATA=NO
PRODUCTION_BASIS_SELECTED=NO
STAGE15_STARTED=NO
```

## Prohibited shortcuts

- Do not rename components without changing basis identity.
- Do not treat equal vector length as basis compatibility.
- Do not compare raw component counts across changed granularity as though they
  shared a denominator.
- Do not learn rotations from scientific outcomes.
- Do not duplicate fidelity, Endpoint 1, or Endpoint 2 definitions.
- Do not freeze technical partition/rotation values because a fixture behaves
  well.
- Do not run registered teachers or students.
- Do not begin packing-null calibration, Stage 13 freeze, or Stage 15.
