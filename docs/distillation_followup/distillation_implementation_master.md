# Distilled-Realization Follow-up: Detailed Implementation Order

## Governing principle

Preserve the completed predecessor study, freeze the new scientific design before new endpoint-producing work, and then implement the follow-up through parallel technical lanes that synchronize at explicit barriers.

The predecessor's `experimental_protocol.md`, `implementation_order.md`, analysis freeze, and scientific artifacts remain unchanged. This follow-up receives separate configuration, run, result, manifest, table, figure, and archive namespaces.

The unit of population inference is the teacher seed. Physical jobs may run at finer granularity, but production completion, aggregation, and reproduction must roll up through teacher seed.

## Why this order differs from the predecessor

The predecessor began with an unimplemented model-training and circuit-search pipeline. This follow-up begins with five trained teacher runs, dense checkpoints, masking code, search code, manifests, and a completed analysis. The implementation order therefore has three kinds of work:

1. **Reuse and verification:** preserve validated teacher and masking substrate.
2. **New core machinery:** student distillation, predictive fidelity, method-aware budgets, and new endpoint reducers.
3. **New experimental orchestration:** nested teacher–phase–condition–student analysis without treating lower-level repetitions as independent samples.

Parallelism starts only after common condition identities and artifact schemas exist. Otherwise the lanes will create incompatible records that are expensive to reconcile.

## Scope hierarchy

### Core — cannot be removed

- Five existing teacher seeds as the population-level units.
- A prospectively frozen, per-teacher functional phase grid.
- Direct evaluation of every selected teacher checkpoint.
- Hard-target students with exact full-domain decision reproduction.
- Soft-target students with gauge-invariant matching and a frozen eligibility rule.
- Multiple independently initialized student attempts per teacher–phase–condition cell.
- The common 516-component basis and identical masking semantics.
- Primary predictive fidelity based on per-input centred logits.
- Smallest recovered component proportion, always defined up to 1.0.
- Circuit packing lower bound, including zero.
- At least the existing discovery procedure; any additional method must be frozen.
- Method-specific native budgets and a common exact-evaluation allowance.
- Teacher-seed-level aggregation, separate hard/soft analyses, and failure accounting.
- Deterministic manifests, inventory sealing, independent reproduction, and analysis freeze.

### Important but reducible before freeze

- Number of eligible students targeted per cell.
- Number of discovery methods beyond the existing method.
- Size of the fidelity, component-cap, and overlap sensitivity grids.
- Seven-phase teacher-only analysis.
- Seven-phase student distillation.
- Number of independently reproduced teacher-seed shards.

### Secondary and gated

- Fourier interchange.
- Entropy estimation.
- Cross-task breadth.
- General theory.
- Conditional atlases.

These items cannot delay complete execution of the core experiment.

## Parallel ownership lanes

| Lane | Ownership | Workstream brief |
|---|---|---|
| A | Protocol, teacher registry, schemas, provenance | `workstreams/ws_a_protocol_registry.md` |
| B | Teacher targets, hard/soft distillation, eligibility | `workstreams/ws_b_distillation.md` |
| C | Predictive fidelity, discovery adapters, endpoints | `workstreams/ws_c_circuit_recovery.md` |
| D | Orchestration, inventories, hierarchical analysis, reporting | `workstreams/ws_d_orchestration_analysis.md` |
| E | Fourier interchange after the core freeze | `workstreams/ws_e_fourier_secondary.md` |

Ownership is by interface, not experimental condition. Do not create separate hard and soft codebases, separate phase pipelines, or method-specific exact evaluators.

---

# Stage 1 — Preserve the predecessor and create a follow-up namespace

## Implement

- Record the predecessor analysis-freeze commit.
- Leave its protocol, implementation order, configs, results, and amendment log unchanged.
- Create dedicated follow-up paths for configs, checkpoints, results, manifests, tables, figures, archives, and excluded development output.
- Make follow-up commands reject predecessor output roots.
- Record whether implementation occurs on a dedicated branch or in a successor repository.
- Add a predecessor-link manifest listing every reused dataset, architecture, teacher run, and code component.

## Deliverables

- Follow-up namespace specification.
- Predecessor-link manifest schema and first manifest.
- Output-root collision tests.
- Written declaration of previously visible predecessor results.

## Acceptance gate

No follow-up command can overwrite or be mistaken for a predecessor artifact.

---

# Stage 2 — Freeze the scientific skeleton

## Implement

Freeze the choices already settled in the follow-up protocol:

- research question;
- experimental hierarchy;
- teacher seed as the population unit;
- hard and soft conditions as separate estimands;
- direct teacher evaluation;
- endpoint definitions and permitted language;
- outcome categories rather than directional predictions;
- Fourier interchange as secondary;
- explicitly gated extensions.

Keep numeric or roster decisions unresolved only where the protocol freeze register identifies them explicitly.

## Deliverables

- Reviewed follow-up protocol draft.
- Freeze register with an owner and resolution stage for every open item.
- Method-development firewall and exclusion rule.

## Acceptance gate

No implementation lane is free to invent a scientific estimand, eligibility rule, or interpretation.

---

# Stage 3 — Select and seal the teacher phase registry

## Rationale

The predecessor's scaled grid used common training steps. Those steps do not represent the same function-level phase in every seed. The new design treats phase/function as a repeated condition within teacher, so phase selection must occur separately within each teacher.

## Implement

For each of teacher seeds 0–4, select using training/test metrics only:

1. the frozen-rule pre-grokking checkpoint;
2. the nearest eligible 50% transition checkpoint;
3. the frozen-rule stable post-grokking checkpoint.

Recommended definitive grid: these three function-defined phases, producing 15 teacher checkpoints.

For each selected checkpoint, record:

- teacher seed;
- phase label and selection rule;
- training step;
- achieved train/test accuracy and loss;
- checkpoint path and SHA-256;
- training run and manifest;
- architecture, dataset, and split hashes;
- dense-output reference status.

Do not label a common training step as a common phase without verifying the per-seed functional criterion.

## Optional prospective expansion

A seven-landmark, per-teacher function-defined teacher-only grid may be frozen as secondary. Seven-phase student distillation remains gated by the Stage 14 compute projection.

## Deliverables

- Canonical teacher registry.
- Phase-selection table.
- Registry verifier and tests.
- List of any teacher seed lacking an eligible phase checkpoint.

## Acceptance gate

Every planned teacher function has one immutable checkpoint hash selected without using new circuit or distillation endpoints.

---

# Stage 4 — Freeze condition identities, schemas, and seed derivation

## Implement

Define the canonical condition hierarchy:

```text
teacher_seed / phase / distillation_condition / student_initialization /
discovery_method / fidelity_setting / component_cap / overlap_setting
```

Implement versioned schemas for:

- teacher reference;
- teacher output cache;
- student attempt;
- student eligibility;
- sealed dense model;
- discovery run;
- native budget ledger;
- exact mask-evaluation ledger;
- endpoint record;
- student-cell summary;
- teacher-seed inventory;
- excluded development output;
- reproduction comparison;
- analysis freeze.

Derive all training, tie-breaking, and discovery seeds from the complete canonical identity. Store both the derivation material and integer seed.

## Cross-field validation

- Hard and soft records cannot share a condition identity.
- Student circuit work requires a sealed eligibility record.
- Every dense model must declare the same architecture and component-basis hash.
- Every search record must name its primary fidelity definition and budget version.
- Every endpoint must trace to an exact-evaluation ledger.
- Failed attempts count against the frozen attempt cap.

## Deliverables

- Schemas and validators.
- Canonical ID builder.
- Deterministic seed derivation.
- Synthetic records covering valid and invalid cases.

## Acceptance gate — Barrier 0

Lanes B–D can exchange schema-valid synthetic artifacts without scientific computation.

After Barrier 0, Stages 5A–5D may proceed in parallel.

---

# Stage 5A — Implement centred-logit predictive fidelity

**Owner:** Lane C
**May run in parallel with:** Stages 5B, 5C, and 5D

## Implement

- Centre logits across classes independently for every input.
- Accumulate numerator and denominator over all 12,769 inputs in deterministic order.
- Use a frozen numerical precision and reduction order.
- Implement and test the denominator guard.
- Version the fidelity definition in every evaluation record.
- Retain top-one agreement, KL, Jensen–Shannon, accuracy, and cross-entropy as diagnostics.
- Do not reuse the predecessor's top-one fidelity field as if it were predictive fidelity.

## Required tests

- Dense model compared with itself returns exactly or tolerance-equivalently 1.0.
- Adding a different scalar to all class logits for each input does not change fidelity.
- Batched and unbatched evaluation agree within the frozen tolerance.
- Reordered inputs give the same aggregate when restored to canonical ordering.
- Negative fidelity values remain representable rather than clipped silently.

## Deliverables

- Centred-logit reference type.
- Streaming predictive-fidelity evaluator.
- Versioned metric record.
- Numerical and gauge-invariance tests.

## Acceptance gate

The evaluator is deterministic, gauge-invariant, and independently reproducible on a sealed teacher checkpoint.

---

# Stage 5B — Implement the teacher target cache and shared student trainer

**Owner:** Lane B
**May run in parallel with:** Stages 5A, 5C, and 5D

## Implement

For each teacher checkpoint, cache over the complete input universe:

- raw final-position logits;
- per-input centred logits;
- probabilities if required by a frozen secondary metric;
- argmax decisions;
- canonical input ordering;
- teacher/checkpoint and dataset hashes.

Build one shared student-training engine with adapters for hard and soft losses. Record initialization seed, optimizer state, training trajectory, stopping reason, dense output, and final checkpoint.

## Required tests

- Target cache reproduces directly recomputed teacher outputs.
- Cache hash changes if teacher, checkpoint, dataset, input order, or target representation changes.
- Student initialization reproduces from the condition ID.
- Resume and uninterrupted training produce equivalent results under the frozen deterministic rule.
- Hard and soft adapters use the same model construction and bookkeeping.

## Deliverables

- Teacher target-cache builder and loader.
- Shared student trainer.
- Attempt manifest writer.
- Resume and reproduction tests.

## Acceptance gate

A synthetic attempt can train, resume, seal, reload, and reproduce without condition-specific trainer drift.

---

# Stage 5C — Implement resumable orchestration and isolated outputs

**Owner:** Lane D
**May run in parallel with:** Stages 5A, 5B, and 5D

## Implement

- Build a declarative job DAG from the condition registry.
- Separate target-cache, training, eligibility, discovery, endpoint, merge, and analysis nodes.
- Assign every job an isolated writable root.
- Write atomically completed artifacts and immutable completion records.
- Detect missing, duplicate, stale, and hash-conflicting jobs.
- Resume without duplicating attempts or transferring unused budgets.
- Merge worker outputs deterministically and serially.

## Deliverables

- DAG builder.
- Job registry and status report.
- Isolated-output convention.
- Deterministic merge and resume tests.

## Acceptance gate

Synthetic jobs can be interrupted, resumed, and merged without collision, duplication, or loss of failure records.

---

# Stage 5D — Implement hierarchical analysis on synthetic fixtures

**Owner:** Lane D
**May run in parallel with:** Stages 5A, 5B, and 5C

## Implement

Using synthetic endpoint records only, implement:

- direct teacher values by seed, phase, and method;
- student-cell summaries across eligible initializations;
- within-cell range and median absolute deviation;
- within-seed phase contrasts;
- direct teacher–student contrasts;
- separate hard and soft tables;
- method-stratified summaries;
- missing and unresolved cell handling;
- attempt and eligibility failure reporting;
- teacher-seed-level population summaries.

## Required protections

- Students cannot be treated as population replicates.
- Circuits, thresholds, and methods cannot inflate sample size.
- Hard and soft conditions cannot be pooled.
- A cell below the frozen eligible-student minimum becomes unresolved.
- Zero packing and endpoint-1 value 1.0 remain ordinary reportable outcomes.

## Deliverables

- Analysis data model.
- Synthetic tables and figure skeletons.
- Anti-pseudoreplication tests.

## Acceptance gate — Barrier 1

The planned analysis can consume schema-valid records without special cases added after real results become visible.

---

# Stage 6A — Implement exact-evaluation ledgers and endpoint 1

**Owner:** Lane C
**Depends on:** Stage 5A
**May run in parallel with:** Stages 6B–6E

## Implement

- Insert the intact mask as a mandatory exact evaluation for every dense model and method.
- Store every exactly evaluated unique mask, fidelity, retained count, retained proportion, evaluation order, and budget charge.
- Define endpoint 1 as the smallest qualifying proportion in the ledger plus the intact mask.
- Record the smallest qualifying mask rather than only the search's nominal terminal mask.
- Record termination and failure status separately.

## Required edge cases

- No sub-full qualifying mask: endpoint equals 1.0.
- Search optimization failure: endpoint still equals the smallest qualifying exact evaluation, at worst 1.0.
- Budget exhaustion: endpoint remains defined and is explicitly procedure-censored.
- Duplicate mask proposals: one scientific mask record with auditable proposal references and frozen budget accounting.

## Deliverables

- Exact-evaluation ledger.
- Endpoint-1 reducer.
- Recompute command and tests.

## Acceptance gate

Endpoint 1 can be reconstructed from the ledger alone and is defined for every valid dense model.

---

# Stage 6B — Implement hard-target eligibility

**Owner:** Lane B
**Depends on:** Stage 5B
**May run in parallel with:** Stages 6A and 6C–6E

## Implement

- Train against the teacher's complete argmax target vector.
- Evaluate exact decision agreement over all 12,769 inputs.
- Require 12,769/12,769 agreement for eligibility.
- Seal eligible student checkpoint and dense-output hashes.
- Record unsuccessful optimization, numerical failure, and sub-100% agreement as distinct failed-attempt statuses.
- Prevent circuit jobs from being created for ineligible attempts.

## Deliverables

- Hard-target loss adapter.
- Full-domain eligibility evaluator.
- Failure taxonomy and attempt records.
- Exact agreement tests.

## Acceptance gate

Changing one student decision makes the attempt ineligible, and the failed attempt remains permanently counted.

---

# Stage 6C — Implement soft-target training and eligibility

**Owner:** Lane B
**Depends on:** Stage 5B
**May run in parallel with:** Stages 6A, 6B, 6D, and 6E

## Implement

- Implement the selected gauge-invariant soft loss, provisionally centred-logit error.
- Support the prospectively frozen temperature or normalization if any.
- Evaluate the frozen full-domain soft tolerance.
- Evaluate and store teacher–student argmax agreement.
- Enforce the frozen rule on whether 100% argmax agreement is additionally required.
- Seal eligible checkpoints and dense outputs.

## Required tests

- Per-input additive-logit shifts do not change targets, loss, or eligibility.
- Acceptance is computed over the complete universe.
- Tolerance boundary behavior is deterministic.
- Soft and hard eligibility records cannot be confused.

## Deliverables

- Soft-target loss adapter.
- Soft eligibility evaluator.
- Gauge and tolerance tests.

## Acceptance gate

Soft eligibility is fully determined by sealed teacher/student outputs and the frozen rule.

---

# Stage 6D — Implement discovery-method adapters and budget ledgers

**Owner:** Lane C
**Depends on:** Stage 5A
**May run in parallel with:** Stages 6A–6C and 6E

## Implement

Define one adapter interface that records:

- method name, version, and config;
- native optimization budget and unit;
- common exact-evaluation allowance;
- proposals and restarts;
- exact-evaluation requests;
- stopping and failure status;
- deterministic seeds;
- search trajectory.

Adapt the existing greedy deletion and diversity-forced machinery to the new fidelity evaluator. Add any genuinely different discovery method only after its algorithm and native budget can be specified prospectively.

## Budget rule

Native budgets are method-specific and not declared equivalent. Final exact mask evaluations use one common allowance and counting rule. Phase and student contrasts are within method. Cross-method raw packing counts receive an explicit resource-imperfect warning.

## Deliverables

- Discovery adapter protocol.
- Existing-method integrations.
- Native and exact budget ledgers.
- Exhaustion, restart, and deterministic-seed tests.

## Acceptance gate

Every method produces the same ledger format without pretending its native optimization primitives are equal.

---

# Stage 6E — Implement endpoint 2 and deterministic packing

**Owner:** Lane C
**Depends on:** Stage 4 schemas; may use synthetic ledgers initially
**May run in parallel with:** Stages 6A–6D

## Implement

- Filter exact-evaluated masks by frozen predictive fidelity and component-proportion cap.
- Deduplicate identical masks.
- Compute pairwise overlap in the common 516-component basis.
- Build the compatibility graph under the frozen overlap cutoff.
- Compute the frozen exact or deterministic maximum mutually compatible subset among recovered masks.
- Record zero when no mask qualifies.
- Store the selected packing members and proof/recomputation metadata.

For the expected small recovered families, exact maximum-clique or exhaustive subset selection is preferred over a proposal-order-dependent greedy packer.

## Deliverables

- Packing reducer.
- Compatibility graph record.
- Proposal-order invariance and zero-result tests.

## Acceptance gate — Barrier 2

Both endpoints can be deterministically reconstructed from method-agnostic exact-evaluation ledgers.

---

# Stage 7 — Integrate one technical end-to-end fixture

## Purpose

Exercise the complete real-model interface before freezing numeric design choices. This is method development, not a scientific pilot.

## Implement

Using one registered teacher checkpoint and the minimum attempts needed to exercise hard and soft paths:

1. build the teacher target cache;
2. train hard and soft student attempts;
3. evaluate eligibility;
4. seal eligible dense models where obtained;
5. run every discovery adapter on the teacher and eligible fixture students;
6. build exact ledgers and both endpoints;
7. generate a teacher-seed-style inventory;
8. run synthetic-plus-fixture analysis and reports;
9. reproduce all deterministic artifacts independently.

## Firewall

Any scientific endpoint value emitted is entered in the excluded-development-output register. It may not select phase checkpoints, thresholds, component caps, overlap cutoffs, student counts, method roster, or budgets based on apparent phase or condition effects. Definitive results are regenerated after freeze.

## Acceptance gate

The complete pipeline works, and all fixture endpoint outputs are explicitly excluded.

---

# Stage 8 — Validate the scientific edge cases

## Implement

Force and verify:

- hard student with exactly one mismatched input;
- soft student immediately above and below tolerance;
- no sparse qualifying circuit;
- no packing-eligible circuit;
- method optimization failure;
- exact-evaluation budget exhaustion;
- duplicate proposals;
- missing student cell;
- cell below the minimum eligible-student count;
- interrupted and resumed worker;
- conflicting worker inventory;
- additive-logit gauge shift;
- result-order permutation.

## Deliverables

- Edge-case fixture suite.
- Expected status and endpoint table.
- Failure-to-reporting integration tests.

## Acceptance gate

No scientific failure mode is represented as missing data accidentally or assigned an interpretation after results are visible.

---

# Stage 9 — Benchmark student training

## Permitted evidence

Only convergence feasibility, runtime, memory, storage, numerical stability, and eligibility implementation may be inspected. Phase or condition effects on circuit endpoints are prohibited evidence.

## Implement

- Measure runtime per training step and per attempt.
- Measure checkpoint and dense-output storage.
- Characterize whether the proposed hard and soft training budgets can reach their frozen-style acceptance checks on technical fixtures.
- Verify deterministic resume.
- Estimate worst-case cost under the proposed attempt cap.
- Identify numerical failures without changing the scientific target.

## Deliverables

- Training benchmark table.
- Worst-case attempt-cost projection.
- Proposed optimizer, schedule, stopping, and attempt budgets.

## Acceptance gate

The proposed training configuration is technically usable without inspecting new circuit comparisons.

---

# Stage 10 — Benchmark circuit discovery by method

## Permitted evidence

Only runtime, memory, proposal counts, exact-evaluation counts, storage, failure mechanics, and worker throughput may determine budgets. Scientific phase and student differences are prohibited evidence.

## Implement

For each candidate discovery method:

- benchmark native optimization operations;
- benchmark exact full-domain mask evaluation;
- project runs per dense model;
- estimate sensitivity-grid reuse from the same ledger;
- determine whether proposal trajectories can support several thresholds without rerunning search;
- establish a common final exact-evaluation allowance;
- establish method-specific native budgets;
- validate worker isolation and feasible concurrency.

## Deliverables

- Per-method budget proposal.
- Common exact-evaluation proposal.
- Runtime/storage/concurrency table.
- Method inclusion or exclusion recommendation based only on technical feasibility.

## Acceptance gate

Every retained method has a usable, auditable budget definition.

---

# Stage 11 — Freeze distillation parameters

## Freeze

For hard students:

- loss;
- optimizer and schedule;
- stopping rule;
- maximum training budget;
- exact 100% argmax eligibility.

For soft students:

- target representation;
- loss and any temperature;
- optimizer and schedule;
- stopping rule;
- maximum training budget;
- numerical tolerance;
- whether 100% argmax agreement is additionally required.

For both:

- planned eligible students per cell;
- maximum attempted initializations per cell;
- minimum eligible students for cell summaries;
- replacement/stop rule;
- failure taxonomy.

## Recommended replication rule for evaluation

Target three eligible students per teacher × phase × condition cell, allow at most six fixed initialization attempts, stop when three eligible students are obtained or six attempts are exhausted, and retain every failure. This would imply:

- 5 teachers × 3 phases × 2 conditions × 3 eligible students = 90 eligible students if every cell succeeds;
- at most 5 × 3 × 2 × 6 = 180 student attempts.

This is a recommendation until frozen after the compute projection.

## Acceptance gate

Eligibility and replication cannot be changed in response to the observed recoverability of successful students.

---

# Stage 12 — Freeze fidelity, packing, and discovery parameters

## Freeze

- exact centred-logit formula implementation and numerical precision;
- primary predictive-fidelity threshold;
- any fidelity sensitivity grid;
- maximum component proportion for packing;
- overlap metric and cutoff;
- component-cap and overlap sensitivity settings;
- discovery-method roster and version;
- method-specific native budgets;
- common final exact-evaluation allowance;
- restart and termination rules;
- packing subset algorithm;
- cross-method interpretation limitation.

The proposed 0.99 predictive-fidelity threshold is not frozen until this stage is committed.

## Acceptance gate

No threshold, cap, overlap rule, method, or budget can be selected to maximize the observed phase or distillation contrast.

---

# Stage 13 — Freeze analysis and missing-cell rules

## Freeze

- primary phase contrasts;
- direct teacher–student contrast;
- student-cell summary, provisionally median;
- realization-dispersion summaries, provisionally range and median absolute deviation;
- minimum eligible-student count;
- unresolved-cell handling;
- student-attempt failure summaries;
- teacher-seed population summaries;
- method-stratified reporting;
- hard/soft separation;
- required tables and figures;
- sensitivity interpretation;
- outcome-category resolution rules.

## Acceptance gate

Synthetic fixtures generate the complete planned report, including failures and unresolved cells, before definitive data exist.

---

# Stage 14 — Project total compute and freeze the definitive scope

## Base design arithmetic

Under the recommended three-phase, three-eligible-student design:

- direct teacher models: 5 × 3 = 15;
- eligible student models: 5 × 3 × 2 × 3 = 90;
- total dense models receiving circuit analysis: 105;
- method-model cells: 105 multiplied by the number of discovery methods;
- maximum training attempts: 180 under a six-attempt cap.

Any sensitivity grid multiplies or reuses these cells depending on whether endpoints can be recomputed from shared ledgers. The projection must not assume reuse unless the implementation proves it.

## Implement

Project:

- training compute and storage;
- teacher output caching;
- exact mask evaluations;
- method-native optimization;
- raw proposal and ledger files;
- inventory and archive sizes;
- independent reproduction;
- expected wall-clock time under safe concurrency;
- operational headroom and merge cost.

## Prospective fallback order

If the complete plan is infeasible, reduce scope before freeze in this order:

1. remove optional sensitivity cells that require rerunning search;
2. remove additional discovery methods that lack defensible budgets;
3. keep the three-phase core and reduce eligible-student target only if realization sensitivity remains estimable;
4. reduce teacher-only secondary trajectories;
5. do not reduce the five teacher seeds before removing optional breadth.

## Deliverables

- Definitive compute projection.
- Frozen scope and concurrency manifest.
- Explicit list of gated extensions.

## Acceptance gate — Barrier 3: final protocol freeze

- Every freeze-register item is resolved.
- Protocol and exact configs are committed.
- Excluded development outputs are enumerated.
- No definitive endpoint-producing job has started.

---

# Stage 15 — Run definitive direct teacher evaluation

## Implement

For all 15 primary teacher checkpoints:

1. verify checkpoint and dense-output hashes;
2. compute the frozen full-domain predictive-fidelity reference;
3. run every frozen discovery method under its budget;
4. seal exact-evaluation ledgers;
5. compute both endpoints;
6. record all budgets, failures, and stopping reasons.

This stage may run in five parallel shards, one per teacher seed. It may also overlap with Stage 16 once the relevant teacher target cache is sealed.

## Acceptance gate

Every selected teacher function has a complete direct result for every method or an explicit, protocol-valid failure record.

---

# Stage 16 — Train definitive hard and soft student attempts

## Implement

Within each teacher-seed shard, for every selected phase:

- build and seal the teacher target cache;
- run predeclared hard attempts;
- run predeclared soft attempts;
- stop only under the frozen attempt rule;
- retain all failed attempts;
- seal all completed checkpoints and dense outputs.

Attempt jobs may run concurrently by phase, condition, and initialization in isolated output roots.

## Acceptance gate

Every planned attempt slot is either completed with a sealed artifact or assigned an explicit terminal failure status.

---

# Stage 17 — Seal student eligibility and cell completeness

## Implement

- Recompute hard decision agreement from sealed outputs.
- Recompute soft tolerance and argmax criteria from sealed outputs.
- Seal eligible-student records.
- Prevent any ineligible model from entering the circuit queue.
- Count eligible and failed attempts per teacher–phase–condition cell.
- Mark cells satisfying the frozen minimum as complete.
- Mark insufficient cells unresolved without imputation.

## Deliverables

- Eligibility table.
- Attempt-failure table.
- Cell-completeness registry.
- Circuit-job queue containing eligible students only.

## Acceptance gate

Every student circuit job traces to a sealed eligibility record, and every failed attempt remains visible.

---

# Stage 18 — Run definitive student circuit discovery

## Implement

For every eligible student and frozen method:

1. verify model, teacher, phase, condition, and eligibility hashes;
2. compute the student's own dense reference;
3. exactly evaluate the intact mask;
4. run discovery under method-specific native and common exact budgets;
5. seal the proposal trajectory and exact-evaluation ledger;
6. compute endpoint 1 and endpoint 2;
7. store failure and censoring status.

Jobs may run in parallel below teacher seed, but no workers share writable raw directories, ledgers, tables, manifests, or archives.

## Acceptance gate

Every eligible student has a method-complete endpoint record or an explicit method failure, with endpoint 1 still defined.

---

# Stage 19 — Seal teacher-seed inventories

## Implement

For each teacher seed, account for:

- every selected phase;
- every direct teacher evaluation;
- every hard and soft attempt;
- every eligibility decision;
- every expected discovery job;
- every exact ledger and endpoint;
- every failure, exclusion, and unresolved cell;
- every budget and config hash.

Merge worker output deterministically and serially. Do not aggregate scientifically yet.

## Acceptance gate — Barrier 4

All five teacher-seed inventories are sealed, mutually schema-consistent, and contain no unaccounted planned jobs.

---

# Stage 20 — Independent reproduction and integrity audit

## Implement

- Select the reproduction scope under the frozen rule.
- Recompute teacher targets and eligibility.
- Reproduce complete discovery ledgers for the selected cells or shards.
- Compare deterministic artifacts, endpoint values, hashes, and inventory membership.
- Recompute intact-mask invariants and budget totals globally.
- Verify no predecessor or excluded-development artifact entered production.

## Deliverables

- Reproduction archives.
- Machine-readable comparison.
- Integrity exception register.

## Acceptance gate

All deterministic comparisons pass, or discrepancies are resolved and every affected production artifact regenerated before analysis.

---

# Stage 21 — Recompute primary endpoints from sealed ledgers

## Implement

Independently of the search runners:

- recompute endpoint 1 from each exact ledger plus intact mask;
- recompute eligible mask sets;
- recompute overlap graphs and endpoint 2;
- compare recomputed and stored values;
- apply frozen sensitivity settings from the same ledgers where valid;
- label budget exhaustion and procedure censoring.

## Acceptance gate

Every reported endpoint is a deterministic property of a sealed ledger and frozen configuration, not an unchecked search summary.

---

# Stage 22 — Aggregate within the experimental hierarchy

## Implement

In this order:

1. retain raw results by student realization;
2. summarize eligible students within teacher–phase–condition–method cells;
3. calculate within-cell realization dispersion;
4. calculate within-teacher phase contrasts;
5. calculate direct teacher–student contrasts;
6. summarize those teacher-level quantities across five seeds;
7. repeat separately for hard and soft conditions and for each method.

## Report

- raw teacher-seed trajectories;
- raw student-cell distributions;
- phase paired changes by teacher seed;
- sign consistency;
- median and mean paired changes where useful;
- full seed-level range;
- eligibility failure rates;
- unresolved cells;
- budget and failure patterns.

Students, circuits, methods, thresholds, and checkpoints are never pooled as independent population replicates.

## Acceptance gate

Every population summary has at most one appropriately constructed contribution per teacher seed for the relevant contrast.

---

# Stage 23 — Run frozen sensitivity and method-dependence analyses

## Implement

- Apply frozen fidelity sensitivity.
- Apply frozen component-cap and overlap sensitivity.
- Compare fidelity definitions as prespecified.
- Examine method dependence within the method-specific budget caveat.
- Determine whether conclusions reverse, weaken, or remain stable.
- Do not promote a favorable sensitivity result over the primary setting.

## Acceptance gate

Protocol dependence is reported as a result rather than treated as a reason to retune the primary procedure.

---

# Stage 24 — Produce principal tables and figures

## Required outputs

1. Teacher phase registry and dense training trajectories.
2. Student attempt, eligibility, and failure accounting.
3. Endpoint-1 teacher and student phase trajectories, separated by hard/soft and method.
4. Endpoint-2 teacher and student phase trajectories, separated by hard/soft and method.
5. Within-function realization dispersion.
6. Direct teacher-versus-distilled contrasts.
7. Fidelity, cap, overlap, and method sensitivity.
8. Budget consumption, search failures, and unresolved cells.

Every plot must display or link to teacher-seed-level values rather than only pooled summaries.

## Acceptance gate

Tables and figures regenerate from sealed inventories and frozen analysis configs only.

---

# Stage 25 — Resolve the frozen outcome categories

## Implement

Resolve, without rewriting, whether the observed results fit each category:

- teacher phase effect persists in both student conditions;
- students of one teacher vary materially;
- all distilled students become similarly compressible;
- results depend strongly on discovery method or fidelity definition;
- stable functions enable recovery while realization controls packing;
- the predecessor transition disappears under predictive fidelity.

Categories may overlap. None is a binary success criterion.

## Acceptance gate

Every interpretation is traceable to frozen endpoints and the teacher-seed hierarchy, with no global-minimum or true-packing claim.

---

# Stage 26 — Freeze the primary analysis

## Freeze

- included teacher seeds and phases;
- all student attempts and eligibility decisions;
- unresolved cells;
- methods, budgets, and exact-evaluation allowances;
- primary and sensitivity settings;
- sealed inventories and reproduction comparison;
- aggregation outputs;
- final tables and figures;
- outcome-category resolution;
- exploratory analysis register;
- final analysis commit.

## Acceptance gate — Barrier 5

The primary analysis is immutable before interpretive paper drafting.

---

# Stage 27 — Write the core paper

## Order

1. Methods from the frozen protocol.
2. Complete attempt and eligibility accounting.
3. Direct teacher results.
4. Hard-target student results.
5. Soft-target student results.
6. Realization sensitivity.
7. Method and fidelity dependence.
8. Outcome-category interpretation.
9. Limitations and prohibited overclaims.

## Permitted language

- procedure-relative smallest recovered proportion;
- upper bound on unknown globally minimal sufficient proportion;
- packing lower bound under the specified procedure;
- teacher-seed-level evidence about phase/function;
- conditional realization sensitivity;
- distillation- and method-dependent recoverability.

## Prohibited language

- minimum circuit size;
- globally minimal circuit;
- enumeration of all mechanisms;
- true packing number;
- independent replication from students or circuits;
- equivalence of hard and soft function definitions;
- perfectly equal resources across unlike methods;
- blinded confirmation of the predecessor finding.

---

# Optional secondary sequence — Fourier interchange

This sequence begins only if the core run cannot be compromised.

## Stage E1 — Freeze the causal estimand and pair-selection rule

- Select teacher/student pairs without interchange outcomes.
- Freeze intervention layer/location, Fourier representation, and outcome metric.

## Stage E2 — Implement alignment and information-capacity matching

- Implement cross-model Fourier alignment.
- Freeze dimensionality, norm, bandwidth, and information-capacity constraints.

## Stage E3 — Implement all controls before aligned execution

- Wrong Fourier mode.
- Shuffled coefficients.
- Mismatched input.
- Equal-norm random state.
- Unaligned ordinary activation patching.

## Stage E4 — Validate the intervention pipeline

- Verify input identity, source/recipient identities, norm matching, dimensionality, and causal intervention placement.

## Stage E5 — Execute under one frozen manifest

- Run aligned interchange and all controls under matched capacity and trial rules.

## Stage E6 — Analyze and report

- Support a shared causal abstraction only if aligned interchange outperforms every control.
- Do not claim uniqueness among all possible algorithms.

---

# Condensed execution map

```text
Stages 1–4: governance, teacher phases, shared schemas
    |
    +-- 5A fidelity ------------------ 6A endpoint 1 ----+
    |                                 6D methods --------+--+
    |                                 6E packing --------+  |
    +-- 5B trainer ------------------- 6B hard ------------+-- Stage 7 integration
    |                                 6C soft ------------+   Stage 8 edge cases
    +-- 5C orchestration --------------------------------+   Stages 9–10 benchmarks
    +-- 5D synthetic analysis ---------------------------+          |
                                                                     v
Stages 11–14: freeze distillation, search, analysis, compute scope
    |
    +-- Stage 15 direct teachers --+
    +-- Stages 16–17 students -----+-- Stage 18 discovery
                                           |
Stages 19–21: inventories, reproduction, endpoint recomputation
    |
Stages 22–26: hierarchical analysis, sensitivity, figures, freeze
    |
Stage 27: paper
    |
Optional E1–E6: Fourier interchange
```

## Final integrity rules

1. The predecessor protocol and results remain immutable.
2. Phase checkpoints are selected separately within teacher by function-level rules.
3. Teacher seed is the population unit.
4. Hard and soft students remain separate.
5. Failed student attempts remain reported and count against the attempt cap.
6. Ineligible students never enter circuit analysis.
7. Student circuit fidelity is relative to the student's own dense outputs.
8. The intact mask makes endpoint 1 defined up to 1.0.
9. If no packing-eligible circuit is recovered, endpoint 2 is recorded as zero.
10. Only full-domain exact evaluations qualify masks for endpoints.
11. Method-native budgets remain method-specific; exact-evaluation allowance is common.
12. Raw cross-method packing counts are not described as perfectly resource-matched.
13. Definitive production starts only after the final protocol freeze.
14. Analysis recomputes endpoints from sealed ledgers.
15. Population summaries never pool students, methods, thresholds, or circuits as independent replicates.
16. Fourier interchange and other extensions cannot reduce core coverage.
