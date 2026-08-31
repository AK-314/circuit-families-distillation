# Austin 5 — Stage 12-P5 Fourier causal-interchange runner and controls

## Paste this entire document into one fresh Chat-mode task

Repository: `AK-314/circuit-families-distillation`

Local clone convention: `~/Projects/circuit-families-distillation`

Required Stage 12-P4 merge/base:

```text
a5f1a33abc77ae3fd1b06917cbe5941ff12a7b7b
```

Required Stage 12-P4 implementation head:

```text
464904c19c3f4bafdd0fc424aa7be4fb3350a831
```

Required Stage 11 scientific-interface head:

```text
6e027a8c22e5228dadad8707f3a262e78028f855
```

Required Stage 12-P2 architecture-interface head:

```text
c4065c2977f2d4e0cd09a54014f60f993f08aceb
```

Required branch:

```text
feat/stage-12p5-fourier-interchange-runner
```

Create the branch from current `origin/main` after this handoff is merged. Part
A must prove that all four required SHAs are ancestors of `origin/main`, that
PR #29 is merged at the stated implementation head, and that the Stage 11,
P2, P3, and P4 interfaces required below are physically present. Record the
then-current `origin/main` SHA as the implementation base.

## Mission

Implement the policy-neutral technical runner for the registered key-secondary
Fourier causal-interchange experiment. The package must provide:

1. typed trial, pair, location, Fourier-coordinate, alignment, intervention,
   control, capacity, outcome, and failure contracts;
2. injected adapters for extracting Fourier coordinates and applying aligned
   state interventions across compatible teacher/student architectures;
3. explicit, auditable information-capacity accounting shared by the aligned
   intervention and every control;
4. aligned interchange plus all five required controls:
   wrong Fourier mode, shuffled coefficients, mismatched input, equal-norm
   random state, and unaligned ordinary activation patching;
5. deterministic trial identities, seed derivation, execution order,
   interruption/resume, and complete failure/unavailable/censored accounting;
6. outcome-neutral records and reducers that cannot silently turn a technical
   result into a scientific claim;
7. a portable synthetic validate-only command exercising the complete runner.

The work is technical infrastructure. It must not choose the production pair
roster, intervention location, Fourier representation, mode-identification
rule, alignment algorithm, capacity rule, outcome, trial count, aggregation,
success threshold, or seed roster. RD-011 remains open until Stage 13.

## Authorities — read completely before acting

1. `docs/distillation_followup/stage11_post_red_team_design_resolution.md`
2. `followup/configs/stage11_post_red_team_design_candidates_v1.json`
3. `followup/manifests/stage11_red_team_resolution_v1.json`
4. `followup/configs/post_red_team_open_decisions_v1.json`
5. `docs/distillation_followup/post_red_team_protocol_amendment.md`
6. `docs/distillation_followup/distillation_experimental_protocol_draft.md`
7. `docs/distillation_followup/workstreams/ws_e_fourier_secondary.md`
8. `docs/distillation_followup/distillation_implementation_post_red_team.md`
9. merged Stage 12-P2 architecture registry, component accounting, student
   builders, eligibility, checkpoint, and activation-hook interfaces;
10. merged Stage 12-P3 logical-job, dependency, claim, attempt, retry, resume,
    and sealed-output interfaces;
11. merged Stage 12-P4 compact records, manifests, inventories, quotas, merge,
    export, and verification interfaces;
12. Stage 4 canonical identities, seeds, hashes, schemas, and envelopes;
13. Stage 5B–5C cache/trainer/job-output and atomic-publication interfaces;
14. Stage 6A exact-evaluation ledger and Stage 6E packing proof/result
    contracts where shared evidence patterns apply;
15. Stage 7 lifecycle, reproduction, inventory, and failure records;
16. `src/circuit_families/analysis/fourier_sanity_check.py` and its tests, as
    historical diagnostic semantics to audit and reuse where valid—not as a
    causal-interchange implementation or scientific authority;
17. `docs/distillation_followup/handoffs/post_red_team/handoff_sequence.md`.

Reuse canonical identities, serializers, seed derivation, sealed-output,
failure, architecture, and compact-storage contracts. Do not create a parallel
identity or result system merely because Fourier tensors have different shapes.

## Scientific and operational boundary

Permitted:

- tiny deterministic synthetic models, activations, Fourier states, inputs,
  labels, outcomes, and pair rosters;
- injected technical profiles for representation, mode, location, alignment,
  capacity, control construction, outcome computation, trial count, and seeds;
- deliberately successful, null, failed, unavailable, censored, nonfinite,
  interrupted, and resumed technical fixtures;
- exact tensor/array comparisons, hashes, byte counts, ranks, dimensions,
  norms, and declared capacity diagnostics;
- local deterministic execution and compact technical reports.

Prohibited:

- loading or evaluating registered teacher/student checkpoints or dense-output
  caches;
- selecting pairs using candidate interchange outcomes;
- choosing any production RD-011 value from synthetic results;
- omitting, weakening, or resource-shedding any required control;
- treating equal tensor shape, equal norm, or equal byte length alone as proof
  of equal information capacity;
- claiming a shared causal abstraction from the runner itself;
- claiming uniqueness among all possible algorithms under any result;
- starting a registered Fourier run, Stage 13 freeze, Stage 14 qualification,
  or Stage 15 production;
- adding private artifacts, checkpoints, archives, large generated outputs,
  credentials, or LFS objects to Git.

Every executable profile, trial, result, failure, manifest, and validation
report must state `scientific_data=false` and `production_eligible=false`.

## Non-negotiable scientific semantics

- Fourier interchange is executed in production regardless of the direction of
  the primary recoverability results, but this handoff performs no production
  execution.
- Production pair selection may not use candidate scientific outcomes.
- Every aligned trial belongs to one complete comparison set containing all
  five controls under the same frozen trial and capacity contract.
- Evidence for a shared causal abstraction is eligible for interpretation only
  if aligned interchange outperforms every prespecified capacity-matched
  control under the later frozen comparison rule.
- This can never establish a uniquely identified algorithm.
- A failed aligned intervention, a control outperforming aligned interchange,
  or an entirely unavailable comparison set is a valid recordable outcome; the
  runner must not suppress it.

## Chat protocol

- Stay in Chat mode for the entire handoff; do not mix Chat and Work modes.
- Complete Parts A–H in order. Use extra one-block turns inside a part only when
  a real failure or diagnostic branch requires them.
- Every operational response contains exactly one fenced terminal block.
- Keep each terminal block short enough to render reliably; split a genuine
  substep rather than emitting a huge invisible block.
- Briefly state what that block changes or inspects and which diagnostics it
  prints.
- Austin returns complete stdout/stderr before the next block.
- Never emit only a status footer when the next block is available.
- Never claim to await output from a block that was not supplied.
- Use the repository virtual environment when present. Do not silently fall
  back to an incompatible system interpreter.
- In zsh snippets, do not use reserved shell variable names such as `status`.
- Use focused tests while implementing and one broad exact-SHA double-check at
  the end. Do not repeatedly run the full historical suite.
- Preserve user-owned untracked files and unrelated active work.
- End every response with:

```text
HANDOFF=AUSTIN_05_STAGE12P5
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact implementation base recorded in Part A>
HEAD=<exact current SHA>
WAITING_FOR=<NONE or exact blocker>
SCIENTIFIC_DATA=NO
```

## Required design separations

Keep these objects distinct:

1. **Pair contract** — model/architecture roles and pre-outcome selection
   evidence.
2. **Trial contract** — input, location, representation, mode, seeds, outcome,
   and complete comparison-set identity.
3. **Extracted coordinate state** — source values plus provenance and shape.
4. **Alignment plan** — an auditable mapping, never an implicit tensor cast.
5. **Capacity contract** — what information may cross the intervention
   boundary and how equivalence is verified.
6. **Intervention payload** — values actually inserted after validation.
7. **Control construction** — one of exactly five registered control kinds.
8. **Execution evidence** — hooks invoked, identities checked, and intervention
   applied or explicitly failed.
9. **Outcome observation** — raw technical measurement under an injected
   outcome adapter.
10. **Comparison-set result** — complete aligned-plus-controls inventory with
    no claim-generation logic.

Model weights, architecture metadata, host paths, device, worker, retry,
scheduler, and storage metadata must not leak into the scientific pair/trial
identity except through explicitly declared canonical references.

## Part A — Exact base, ancestry, and scope guard

The first block is read-only. It must print and verify:

- repository root, remote, branch, HEAD, local `main`, and `origin/main`;
- exact ancestry of `a5f1a33...`, `464904c...`, `6e027a8...`, and
  `c4065c2...`;
- PR #29 merged state and exact head;
- Stage 11 Fourier interface and open RD-011 record;
- Stage 12-P2 architecture registry/component/activation-hook surfaces;
- Stage 12-P3 logical-job, dependency, attempt, retry, resume, and sealed-output
  interfaces;
- Stage 12-P4 compact result, manifest, merge, and export interfaces;
- relevant Stage 4/5B–C/6A/6E/7 identity and failure consumers;
- the historical Fourier diagnostic implementation and tests;
- tracked cleanliness and separately listed untracked files;
- absence of Stage 12-P5, Stage 13, Stage 14, and Stage 15 work;
- available numerical dependencies without installing or upgrading anything;
- no registered/private-artifact dependency or active scientific process.

After diagnosing that output, a second block may create the required branch.

**Part A passes when:** the branch starts from shared main containing all
required inputs, the tracked state is clean, unrelated files are preserved,
RD-011 remains open, and no scientific/private artifact has been accessed.

## Part B — Reuse audit and versioned trial contract

Map every producer and consumer before implementation. Identify exactly which
existing code supplies:

- canonical model/task/checkpoint/student/architecture/input identities;
- deterministic seed derivation and attempt/retry identity;
- activation read/write hooks and architecture-specific location adapters;
- Fourier transforms, coordinate conventions, mode identities, and gauges;
- atomic outputs, lifecycle states, failures, compact storage, and sealing;
- reusable raw-outcome and exact-evidence patterns.

Define one versioned, schema-closed outer contract containing:

- pair-selection evidence created without candidate outcomes;
- source and recipient roles plus canonical architecture/model references;
- input-set and individual input identity;
- intervention location and representation-profile reference;
- source/target Fourier-mode identity and coordinate convention;
- alignment-profile and capacity-profile references;
- outcome-adapter and comparison-set references;
- deterministic root seed and derived seed namespace;
- exactly one aligned condition and exactly the five registered controls;
- lifecycle/failure state and explicit scientific/production boundaries.

The interface may accept injected technical choices but must not ship a
production profile or a default that masquerades as one. Unknown fields,
ambiguous identities, outcome-informed pair evidence, hard/soft role confusion,
unsupported architecture/location combinations, missing controls,
`scientific_data=true`, and `production_eligible=true` must reject.

Add adversarial tests for identity collisions, reversed roles, duplicate or
missing conditions, mismatched inputs, stale profile references, seed-domain
collisions, unknown fields, and relabeled scientific payloads.

**Part B passes when:** the complete experiment can be described without model
execution, every later choice is injected and auditable, and no new identity
system duplicates existing canonical semantics.

## Part C — Fourier coordinates and cross-model alignment adapters

Implement policy-neutral adapters for:

1. reading the declared ordinary activation at a typed architecture location;
2. mapping it into declared Fourier coordinates;
3. selecting declared modes/coefficients without consulting outcomes;
4. fitting or loading an injected cross-model alignment plan using only its
   declared alignment-data boundary;
5. validating and applying the mapping into the recipient coordinate space;
6. reconstructing the recipient intervention state and writing it through the
   typed hook;
7. emitting complete provenance, shapes, dtypes, gauges, ranks, norms, hashes,
   and invertibility/reconstruction diagnostics where applicable.

Separate Fourier transform, mode selection, alignment fitting, alignment
application, and intervention writing. A model adapter must declare compatible
locations and layouts; never infer compatibility from equal tensor shape.
Alignment data may not overlap a prohibited outcome boundary, and trial
outcomes may not refit alignment.

Support tiny injected technical implementations sufficient to exercise
identity, sign/phase, permutation, scaling, degenerate-mode, rank-deficient,
complex/real encoding, additive-gauge, reconstruction, and architecture-layout
cases. Reuse historical Fourier conventions only after verifying them against
the current declared representation.

Tests must reject wrong input/model/location/mode identity, ambiguous sign or
phase convention, unrecorded conjugate handling, rank mismatch, nonfinite
coordinates, silent dtype loss, reused fitted alignment across incompatible
pairs, and intervention through an unsupported hook.

**Part C passes when:** a synthetic state can be extracted, aligned,
reconstructed, intervened, and exactly audited across at least two compatible
technical architecture adapters without freezing a scientific alignment rule.

## Part D — Matched information-capacity contract

Implement an explicit capacity contract shared by aligned interchange and all
five controls. It must declare and verify, as applicable:

- coordinate universe and allowed selected coordinates;
- real degrees of freedom, including complex-number accounting;
- rank and support constraints;
- scalar precision/quantization assumptions when capacity depends on them;
- side information available to sender and recipient;
- identifiers or indices transmitted outside the numeric payload;
- norm/energy constraints as a separate diagnostic, not a capacity synonym;
- recipient location, shape, and write budget;
- deterministic padding, truncation, or rejection rule;
- a canonical capacity-accounting record and hash.

Provide no universal claim that these fields constitute information-theoretic
channel capacity. The contract is the prespecified operational information
allowance used to match conditions. Its limitations must be explicit.

Every condition in a comparison set must independently validate against the
same capacity contract. A difference must either be a declared condition
property permitted by that contract or a hard failure. Matching only shape,
bytes, coefficient count, or norm is insufficient.

Adversarial tests must expose hidden mode labels, indices, alignment matrices,
input identity, random seeds, or metadata as side channels; complex-versus-real
degree mismatch; rank inflation; precision mismatch; padding leakage; and
state-dependent variable-length payloads.

**Part D passes when:** the runner can prove that all six conditions obey the
same declared operational information allowance or record exactly why a
condition is ineligible.

## Part E — Aligned interchange and all five controls

Implement one comparison-set builder and executor containing exactly:

1. `aligned_fourier_interchange`;
2. `wrong_fourier_mode`;
3. `shuffled_coefficients`;
4. `mismatched_input`;
5. `equal_norm_random_state`;
6. `unaligned_ordinary_activation_patching`.

All six conditions must reuse the same pair, recipient input, intervention
location, outcome adapter, comparison-set root, and capacity contract. Any
allowed difference must be identified by the condition record.

Required control semantics:

- **Wrong Fourier mode:** select a valid, distinct mode under a deterministic
  injected rule; never silently fall back to the aligned mode; match the
  declared capacity.
- **Shuffled coefficients:** use a deterministic permutation and declared
  preservation rule for support, marginal values, norm, and capacity; record
  the permutation identity.
- **Mismatched input:** use a deterministic derangement over an eligible input
  set, prove no source/recipient accidental match, and retain the same capacity.
- **Equal-norm random state:** derive randomness from the trial seed, match the
  declared norm within an explicit tolerance, obey the same support/rank/
  precision/capacity rules, and record zero-norm behavior.
- **Unaligned ordinary activation patching:** patch the declared ordinary state
  at the same recipient location and write budget without Fourier alignment;
  record any required shape adapter as part of the control, not hidden setup.

Build every control before allowing aligned execution to be reported complete.
Do not replace a failed control with a more favorable draw. If a control is
structurally unavailable, the complete comparison set is unavailable under a
declared rule rather than silently reduced to five conditions.

Tests must distinguish all condition identities and force wrong-mode scarcity,
single-input derangement failure, repeated coefficients, zero norm,
rank-deficient random draws, layout mismatch, control construction failure,
and attempted omission or substitution.

**Part E passes when:** one synthetic comparison set executes all six
capacity-validated conditions, and every control-specific invariant is
verified rather than inferred from a common wrapper.

## Part F — Deterministic runner, failures, resume, and outcome-neutral records

Implement a deterministic runner that:

- enumerates pair/comparison/trial/condition identities canonically;
- derives disjoint seeds for pair, input, alignment, condition, shuffle,
  mismatch, random state, and retry operations;
- separates proposal/planning from model execution;
- executes conditions in a declared order while making results independent of
  worker completion order;
- supports interruption after any condition and exact resume without rerunning
  already sealed valid outputs;
- rejects stale, cross-trial, cross-condition, and partially sealed evidence;
- records planned, running, complete, failed, unavailable, and censored states;
- preserves nonfinite outcomes under an injected technical policy;
- emits raw observations and diagnostics without directional labels such as
  “supportive,” “successful abstraction,” or “negative result”;
- seals comparison sets only when the full registered inventory and failure
  accounting is closed;
- writes through P3/P4 lifecycle, compact-record, manifest, and verification
  adapters instead of inventing a second orchestration/storage layer.

Provide a reducer that reconstructs the six-condition table, checks capacity
and inventory closure, and computes only injected technical summaries. It must
not contain the later Stage 13 superiority rule, statistical test, threshold,
aggregation, or scientific claim.

Tests must force crashes before/after intervention and outcome measurement,
duplicate completion, conflicting results, corrupted manifests, changed
alignment/capacity profiles, unordered worker returns, retry exhaustion,
nonfinite outcomes, total aligned failure, a control exceeding aligned, and
all conditions tied. These are record states, not reasons to rewrite the trial.

**Part F passes when:** interrupted and uninterrupted runs produce the same
canonical sealed result inventory, and unfavorable or incomplete evidence is
retained without interpretation.

## Part G — Integrated portable validation

Build one validate-only CLI using deterministic synthetic fixtures that runs:

1. two compatible technical architecture adapters with different internal
   layouts;
2. Fourier extraction, declared-mode selection, alignment, reconstruction, and
   intervention;
3. explicit capacity accounting including complex/real and side-information
   checks;
4. one complete aligned-plus-five-controls comparison set;
5. one failed, one unavailable, one nonfinite, and one censored path;
6. interruption after a subset of conditions and exact resume;
7. compact P4 record/manifest sealing and deterministic reconstruction;
8. adversarial rejection of outcome-informed pairing, missing controls,
   capacity side channels, identity swaps, and stale evidence;
9. an outcome-neutral technical report with no production defaults or claims.

Run from repository root and an unrelated cwd under at least two
`PYTHONHASHSEED` values. Canonical reports and sealed result hashes must be
identical. Run:

- Stage 12-P5 focused/adversarial tests;
- Stage 12-P2 architecture/component/eligibility compatibility;
- Stage 12-P3 job/attempt/retry/resume/output compatibility;
- Stage 12-P4 compact-record/manifest/merge/export compatibility;
- Stage 11 resolution and RD-011-open validation;
- Stage 4 identity/seed/hash/schema compatibility;
- Stage 5B–C cache/trainer/output-publication compatibility;
- Stage 6A/6E ledger/result/proof compatibility where consumed;
- Stage 7 lifecycle/inventory/reproduction compatibility;
- historical Fourier diagnostic tests to prove no regression;
- Ruff on changed Python;
- diff, private-path, secret, large-file, binary, checkpoint, archive, and LFS
  hygiene.

The CLI may write only beneath an explicit temporary/output root. It must not
contact a network service, inspect registered artifacts, discover private
checkpoints, or execute a real model.

**Part G passes when:** the complete synthetic Fourier-interchange lifecycle is
portable, deterministic, capacity-audited, control-complete, resume-safe, and
scientifically neutral.

## Part H — Commit, exact-SHA double-check, PR, and stop

Inspect the changed surface and ensure it is limited to Stage 12-P5 code,
tests/validation, and necessary technical documentation. Create coherent
commits without amend or force-push. Push and open a PR against `main`.

At the final exact SHA, run a fresh detached-checkout internal double-check
covering focused/adversarial and compatibility tests, root/unrelated-cwd CLI
execution, multiple hash seeds, deterministic report/result hashes, all six
condition inventories, capacity-accounting equality, forced interruption and
resume, complete failure accounting, Ruff, diff, tracked cleanliness, artifact
sizes, and Git/LFS surface. Classify findings as blocking, nonblocking, or
question. Repair blockers only through descendant commits and repeat against
the new exact SHA.

Do not merge inside this handoff without master-task authorization.

Final report:

- base, branch, parent, final SHA, and PR;
- exact changed files and artifact sizes;
- reused versus new identities/interfaces;
- pair/trial/location/Fourier/alignment/intervention/control contracts;
- operational capacity definition, side-information accounting, equality
  evidence, and explicit limitations;
- all five control invariants and complete comparison-set evidence;
- deterministic seeds, execution, interruption/resume, failures, and sealing;
- outcome-neutral report and canonical hashes;
- Stage 11/P2/P3/P4 and Stage 4/5B–C/6A/6E/7 compatibility totals;
- open RD-011 production choices;
- internal findings and descendant repairs;
- confirmation of no registered/private/scientific execution or Git/LFS
  artifact upload;
- interfaces exported to Alex 5/6, Austin 6, and Stages 13–15;
- explicit stop before production Fourier execution, Stage 13, Stage 14, and
  Stage 15.

Final status:

```text
AUSTIN_05_STAGE12P5_STATUS=COMPLETE_AT_HANDOFF_GATE
SCIENTIFIC_DATA=NO
PRODUCTION_PAIR_ROSTER_SELECTED=NO
PRODUCTION_INTERVENTION_PROFILE_SELECTED=NO
PRODUCTION_CAPACITY_RULE_SELECTED=NO
RD_011_RESOLVED=NO
STAGE13_STARTED=NO
STAGE15_STARTED=NO
```

## Prohibited shortcuts

- Do not select pairs after inspecting candidate outcomes.
- Do not call activation similarity, Fourier correlation, or diagnostic
  reconstruction a causal interchange result.
- Do not treat equal shape or equal norm as matched information capacity.
- Do not hide mode labels, input identity, alignment matrices, indices, seeds,
  or variable payload length as uncounted side information.
- Do not omit, replace, pool, or weaken an inconvenient required control.
- Do not reroll a failed control until it becomes favorable.
- Do not refit alignment using trial outcomes.
- Do not conflate hard- and soft-target student identities or outcomes.
- Do not drop failed, unavailable, censored, nonfinite, tied, or control-winning
  comparison sets.
- Do not embed the scientific success criterion in the technical reducer.
- Do not claim a shared abstraction or unique algorithm from synthetic tests.
- Do not embed private paths, credentials, large fixtures, archives,
  checkpoints, or LFS objects in Git.
- Do not resolve RD-011 or begin Stage 13, Stage 14, or Stage 15.
