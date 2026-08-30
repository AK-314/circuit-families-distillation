# Distilled-Realization Follow-up

## Post-red-team implementation order

**Current state:** Stages 1--10 complete; no definitive scientific run started

**Supersedes:** unexecuted Stages 11--27 and optional E1--E6 in
`distillation_implementation_master.md`

**Does not supersede:** the completed Stage 1--10 technical record

## Operating principle

The next four stages prepare and freeze one automated production campaign.
Stage 15 executes that campaign as a scheduler-managed dependency graph with
three human gates. It is not split into manual chats for teacher evaluation,
training, eligibility, discovery, and secondary experiments. Those remain
internal job classes because their dependencies and failure rules differ.

After production, Stages 16--18 perform integrity closure, locked analysis, and
paper construction.

```text
Stages 11--12: scientific resolution and missing implementation
       ↓
Stage 13: prospective protocol + exact job-manifest freeze
       ↓
Stage 14: cluster dress rehearsal and launch authorization
       ↓
Stage 15: one automated Symbolica production campaign
       ↓
Stage 16: post-run integrity and completeness audit
       ↓
Stage 17: locked analysis, figures, and claim resolution
       ↓
Stage 18: paper, limitations, disclosure, and release package
```

## Stage 11 — Resolve the post-red-team scientific design

### Purpose

Turn the accepted red-team repairs into a coherent experimental design before
additional production machinery is written.

### Implement and decide

1. Record every red-team criticism as accepted, accepted with modification,
   rejected, or gated, with a reason and a concrete protocol consequence.
2. Construct the candidate expanded teacher roster without inspecting new
   follow-up endpoints.
3. Define the candidate student architecture panel and its interpretation
   limits.
4. Define the canonical basis and the limited basis-sensitivity families.
5. Specify the tiered crossing and protected primary matrix.
6. Establish Endpoint 1 as primary and Endpoint 2 as key secondary unless
   preproduction calibration earns co-primary status.
7. Specify the fidelity frontier, four packing nulls, tractable calibration,
   and Fourier key-secondary analysis.
8. Define unconditional attempt/eligibility accounting.

### Deliverables

- red-team resolution matrix;
- reviewed `followup/configs/post_red_team_open_decisions_v1.json` with every
  item still honestly marked open or explicitly resolved by a prospective
  Stage 11 record;
- candidate teacher and architecture rosters;
- candidate sparse assignment matrix;
- endpoint hierarchy and claims table;
- explicit unresolved-decision list for Stages 12--13;
- compute-envelope targets for the protected tiers.

### Gate

Every accepted red-team repair has an owner, implementation location, analysis
consequence, and freeze stage. No definitive teacher, student, or circuit
outcome is generated.

## Stage 12 — Complete the missing scientific machinery

### Purpose

Implement and test everything newly required by the amended design, using
synthetic or explicitly excluded technical fixtures only.

### Work packages

#### 12A — Expanded teacher and architecture support

- teacher training and per-seed phase-selection pipeline;
- architecture registry and construction adapters;
- parameter/component accounting across architectures;
- balanced assignment generator for the sparse panel.

#### 12B — Independent discovery method

- implement an algorithmically distinct continuous/stochastic sparse-mask
  optimizer;
- integrate common exact-evaluation ledgers;
- preserve method-native budget accounting;
- test failure, restart, deduplication, and resume behavior.

#### 12C — Basis sensitivity

- pre-output-projection attention coordinates;
- seeded balanced MLP blocks;
- fixed orthogonal rotations;
- parameter-weighted and component-type-stratified reducers.

#### 12D — Packing calibration and tractable model

- combinatorial size/type-matched floor;
- ordinary-restart baseline;
- local fidelity-retaining perturbation null;
- exact or near-exact feasible-region calibration model;
- calibration reports that compare recovered versus feasible solutions.

#### 12E — Fourier causal interchange

- alignment and intervention API;
- capacity matching;
- all five required controls;
- frozen-style pair-selection and outcome reducer;
- negative and null-result reporting.

#### 12F — Compact production records

- bit-packed or equivalently compact masks;
- columnar/compressed ledgers where appropriate;
- rolling resume checkpoints rather than dense checkpoint histories;
- deterministic inventories and export bundles;
- storage-quota and incomplete-copy detection.

### Parallelism

12A--12F may proceed in parallel after Stage 11 fixes their interfaces. They
must converge on one schema and job identity system. A work package is not a
separate scientific stage and does not require an A--Z conversation.

### Gate

The amended pipeline completes a reduced synthetic end-to-end run, including
both discovery families, at least one alternate architecture, each required
basis family, the packing nulls, tractable calibration, Fourier interchange,
compact export, and deterministic recomputation. All emitted endpoints remain
excluded technical output.

## Stage 13 — Freeze the protocol, analysis, and exact job manifest

### Purpose

Resolve the scientific choices and turn them into one immutable production
manifest before any definitive outcome is generated.

### Freeze

- teacher roster, phase rule, and unavailable-cell handling;
- student architecture roster and sparse assignment;
- hard/soft training, eligibility, replication, and attempt caps;
- basis definitions and sensitivity assignment;
- fidelity implementation, primary threshold, and frontier;
- endpoint hierarchy, caps, overlap, packing solver, and censoring;
- discovery methods, versions, native budgets, exact allowance, and restarts;
- packing null definitions and draw counts;
- tractable calibration model and exactness criterion;
- primary contrasts, population model, dispersion, and missing-cell rules;
- Fourier pairs, alignment, intervention, controls, and outcome;
- protected Tier 1, protected Tier 2 minimum, and Tier 3 priority;
- required tables, figures, failure reports, and claim-resolution rules.

### Production manifest

Generate the complete declarative DAG before launch. Every planned job must
have:

- canonical identity and deterministic seeds;
- prerequisite artifact hashes;
- resource class and estimated runtime/memory/storage;
- output root and expected records;
- retry cap and terminal failure state;
- tier and scheduling priority;
- exact budget and relevant method-native budget;
- merge, recomputation, and retention rule.

Fidelity, cap, and overlap reducer settings that can reuse a ledger are not
duplicated as discovery jobs.

### Gate — prospective scientific freeze

- no open decision needed to interpret a production result;
- complete planned report generated from synthetic records;
- frozen manifest hashes committed;
- excluded technical outputs enumerated;
- no definitive production job started.

## Stage 14 — Cluster dress rehearsal and production authorization

### Purpose

Prove that the frozen campaign can run on the actual or equivalent cluster
without changing the scientific design.

### Prepare before the allocation

- versioned container and locked dependencies;
- data/checkpoint staging and hash verification;
- CPU and CUDA qualification suite;
- job-array submission and dependency handling;
- monitoring dashboard and automatic failure alerts;
- bounded retries and resume tests;
- compact ledger/storage tests;
- deterministic serial merge;
- export to persistent storage plus independent hash verification;
- one-command launch, status, stop, recompute, and export operations.

### Dress rehearsal

Run a deliberately tiny, non-scientific copy of the complete dependency graph
on the intended backend or a technically equivalent rental machine. Force at
least one interruption, retry, ineligible student, failed search, budget
exhaustion, and incomplete transfer.

### Capacity manifest

Record the granted resources rather than assuming a particular Symbolica
inventory. The planning request is approximately:

- 96 continuous hours;
- 128--256 modern CPU cores;
- 8--16 identical CUDA GPUs;
- at least 512 GB aggregate RAM;
- 4 TB fast scratch and 1 TB persistent/export capacity.

The exact grant determines only how far down the frozen priority list the
campaign can run. It cannot change Tier 1 estimands or inclusion rules.

### Gate — launch authorization

- container and hardware qualification pass;
- projected Tier 1 completion fits the conservative granted envelope;
- protected Tier 2 minimum has a declared capacity trigger;
- monitoring and stop paths work;
- final 6--12 hours are reserved for audit and export;
- production manifest and configs are immutable.

## Stage 15 — Automated Symbolica production campaign

### Nature of this stage

Stage 15 is one production stage and normally one operational conversation. It
contains scheduler subphases, not manually approved scientific stages. One
launch command submits or activates the frozen dependency graph.

Human interaction is limited to monitoring, responding to infrastructure
alerts, and the three gates below. Software may retry jobs only within the
frozen rules. No one may tune scientific settings after comparative outcomes
become visible.

### Internal dependency graph

```text
hardware and container qualification
              ↓
teacher training / registry extension / phase selection
              ↓
teacher target caches
              ↓
student training, eligibility, retries, and sealing
              ↓
direct-teacher and eligible-student discovery
              ↓
primary endpoints and core packing calibration
              ↓
architecture, basis, method, null, calibration, and Fourier jobs
              ↓
exact recomputation, inventories, compression, and export
```

Independent branches of the graph run concurrently when their prerequisites
are satisfied. Direct-teacher discovery need not wait for student training.
Ledger-derived thresholds and overlap settings do not rerun discovery.

### Gate 15.1 — launch

Run on the actual allocation:

1. inspect hardware and filesystem quotas;
2. verify container, inputs, clocks, and hashes;
3. execute a tiny complete pipeline;
4. compare the result with the Stage 14 reference;
5. authorize the protected job arrays.

Failure pauses the campaign for infrastructure repair. It does not authorize a
scientific redesign.

### Protected scheduling order

1. qualification and input integrity;
2. teacher and student eligibility prerequisites;
3. complete Tier 1 direct-teacher and canonical student matrix;
4. independent discovery method coverage;
5. packing calibration and null minimum;
6. tractable search calibration;
7. architecture panel minimum;
8. Fourier interchange and all controls;
9. core basis re-granulation and orientation sensitivity;
10. extra partitions, rotations, restarts, and exploratory breadth.

The Stage 13 manifest must state the exact minimum at each numbered level and
whether partial lower-priority work is reportable.

### Gate 15.2 — primary completeness

After eligibility and enough primary jobs have resolved, inspect only
operational completeness and predeclared health measures:

- planned versus terminal jobs;
- eligibility and terminal-failure counts;
- runtime, memory, storage, and retry rates;
- projected completion of protected tiers;
- hash and inventory integrity.

Do not inspect phase-effect direction to reallocate resources. If Tier 1 is at
risk, stop Tier 3 and then unprotected Tier 2 work according to the frozen
priority list. Release remaining capacity only after the protected plan is
secure.

### Gate 15.3 — exit

No later than the reserved final window:

- stop launching optional jobs;
- allow protected jobs to finish or record terminal states;
- recompute primary endpoints independently from sealed ledgers;
- close teacher-seed inventories;
- validate attempt and eligibility accounting;
- verify budgets, exact-evaluation counts, and failure records;
- compact and compress artifacts;
- copy to persistent/off-cluster storage;
- verify destination hashes before deleting nothing from scratch;
- emit one campaign completion report.

### Acceptance gate

Every protected planned cell has either a sealed result or a protocol-valid
terminal status. All exported artifacts match the frozen manifest and verified
hashes. No scientific aggregation is required to close Stage 15.

## Stage 16 — Post-run integrity and completeness audit

### Purpose

Separate "jobs finished" from "data are scientifically analysable."

### Audit

- reproduce target caches and eligibility from sealed outputs;
- independently recompute primary endpoints and packing graphs;
- verify teacher/student/architecture/basis/method identities;
- confirm all attempt failures and unresolved cells are present;
- compare campaign inventory with the Stage 13 manifest;
- audit exact and native budgets;
- verify deterministic artifacts and quantify permitted backend drift;
- verify no excluded development artifact entered production;
- reconcile partial Tier 2/3 execution against frozen capacity rules;
- produce an exception register.

Repairs may regenerate corrupted artifacts from frozen inputs and configs.
Repairs may not change thresholds, methods, inclusion rules, or estimands.

### Gate

All blocking integrity exceptions are resolved or the affected cells are
assigned their predeclared terminal status. The immutable analysis input bundle
is sealed before comparative analysis begins.

## Stage 17 — Locked analysis, figures, and claim resolution

### Order

1. attempt, eligibility, missing-cell, and failure accounting;
2. direct-teacher primary Endpoint 1;
3. hard-target student Endpoint 1;
4. soft-target student Endpoint 1;
5. within-teacher realization variation and teacher--student contrasts;
6. discovery-method dependence;
7. packing and its null/calibration results;
8. architecture external-validity panel;
9. basis granularity/orientation sensitivity;
10. fidelity frontier and other frozen sensitivities;
11. Fourier causal interchange and controls;
12. outcome-category and claim-resolution table.

Population summaries are constructed at teacher-seed level. Every main figure
must expose or link to seed-level values and failure accounting. Null draws,
students, masks, thresholds, architectures, and methods do not inflate the
population sample size.

### Gate — analysis freeze

- all planned tables and figures regenerate from the sealed bundle;
- primary and sensitivity outputs use only frozen configs;
- endpoint recomputation matches Stage 16;
- claim language respects the protocol amendment;
- exploratory work is separately registered;
- analysis commit and release manifest are sealed.

## Stage 18 — Paper and release package

### Paper order

1. predecessor result and its limitations;
2. prospective red-team-driven design amendment;
3. experimental hierarchy and failure accounting;
4. primary recoverability result;
5. function versus realization evidence;
6. method, packing calibration, architecture, and basis qualifications;
7. Fourier causal evidence;
8. limitations, null results, and prohibited inferences;
9. complete compute and reproducibility statement.

### Release

- protocol amendment and frozen configs;
- code and environment lock;
- compact analysis bundle and manifests;
- tables, figures, and figure-source data;
- reproduction commands;
- artifact availability statement;
- honest disclosure of substantial LLM assistance in project design, code,
  review, and writing as required by the submission venue.

### Gate

Every paper claim maps to a frozen analysis output or is labeled speculation.
The released package can reproduce the reported tables and figures without the
private working directories.

## Operational rhythm

Stages 11--14 may use ordinary implementation/review conversations, with parts
only where they identify genuine lifecycle boundaries. Stage 15 uses one
campaign conversation and one machine-generated status report; it does not use
alphabetical parts for individual job families. Stages 16--18 each receive one
self-contained handoff after the preceding gate is actually complete.

Independent review is useful but not ceremonial. When a second human reviewer
is unavailable, an internal double-check must use a fresh checkout, exact
committed SHA, adversarial tests, and an explicit findings log. Scientific
redesign decisions remain human-owned.
