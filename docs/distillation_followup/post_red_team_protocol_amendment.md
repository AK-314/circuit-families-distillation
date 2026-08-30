# Distilled-Realization Circuit Recoverability

## Post-red-team prospective protocol amendment

**Status:** design amendment to be resolved and frozen in Stages 11--14

**Date:** 29 August 2026

**Scientific execution under this amendment:** not started

**Completed technical foundation:** Stages 1--10

**Production campaign:** Stage 15, only after the Stage 14 launch gate

## 1. Authority and supersession

The Stage 2 scientific skeleton and the completed technical work through Stage
10 remain part of the audit trail. Their files and manifests are not rewritten.

This amendment changes the prospective design in response to scientific
red-teaming conducted before definitive follow-up data collection. For
unexecuted work, it supersedes:

- Sections 2--13 and 15 of
  `distillation_experimental_protocol_draft.md` where this amendment states a
  different design;
- Stages 11--27 and the optional E1--E6 sequence in
  `distillation_implementation_master.md`;
- the dates, ownership assumptions, and M5-Max-only production schedule in
  `broad_timeline.md` and `compute_and_two_person_collaboration.md`.

Supersession is prospective. It does not retroactively alter the meaning or
acceptance of Stages 1--10, nor does it pretend that decisions marked open in
the Stage 2 register were already resolved.

## 2. Amended research question

> Under predeclared component bases, interventions, evaluation rules, and
> discovery procedures, does the grokking-associated change in recoverable
> sparse circuitry follow the teacher function, vary across distilled
> realizations of that function, depend on architecture or basis, or arise from
> the distillation and discovery procedures themselves?

The project studies **procedure-relative sparse-circuit recoverability**. It
does not identify a unique mechanism, enumerate all circuits, or measure the
globally minimal sufficient subnetwork.

## 3. Experimental hierarchy and inference

```text
teacher seed
  └── phase checkpoint/function
        └── distillation condition: hard or soft
              └── student architecture
                    └── student initialization
                          └── component basis condition
                                └── discovery method
                                      └── endpoint settings and recovered masks
```

- Teacher seed remains the population-level unit for phase/function claims.
- Phase is a repeated condition within teacher seed.
- Student architecture is a predeclared experimental factor, not a population
  replicate.
- Student initialization estimates realization sensitivity conditional on a
  teacher, phase, condition, and architecture.
- Basis conditions, discovery methods, thresholds, and recovered masks are
  repeated measurements.
- Hard and soft students remain separate estimands.
- Direct teachers are evaluated under every primary protocol condition that is
  meaningful for them.
- The effective population sample size for a teacher-phase contrast is the
  number of eligible matched teacher seeds, not the number of students, masks,
  methods, basis settings, or search jobs.

The teacher roster should be expanded beyond the five predecessor seeds if the
prospective Stage 11 construction and technical checks permit it. The working
target is approximately 15 matched teachers, but the exact roster, phase
availability, and inclusion rule remain to be frozen.

## 4. Distillation conditions and architecture panel

### 4.1 Hard-target students

- Targets are the teacher argmax decisions over all 12,769 inputs.
- Eligibility requires exact 12,769/12,769 agreement.
- Failed attempts remain outcomes and count against the attempt cap.
- Circuit fidelity is relative to each student's own dense outputs.

### 4.2 Soft-target students

- Targets and losses must be invariant to the per-input additive-logit gauge.
- The target representation, loss, temperature or normalization, tolerance,
  and argmax requirement are frozen before production.
- Failed attempts remain outcomes and count against the attempt cap.
- Circuit fidelity is relative to each student's own dense outputs.

### 4.3 Architecture panel

The original one-layer architecture remains the canonical matched student.
The amended study includes a systematic, prospectively specified panel of
additional student architectures. Its purpose is external validity: to test
whether conclusions are confined to the predecessor architecture.

The panel is not interpreted as a clean causal estimate of depth, width, or any
single architectural property unless architectures differ only in that one
property. Parameter count, searchable component count, component types, and
training eligibility must be reported by architecture. Raw component counts
are not compared across unlike bases without an explicit denominator and
parameter-weighted sensitivity.

The working maximum is five student architecture families. The exact roster
and which teacher/phase/condition cells receive each architecture are frozen in
Stages 11 and 13 using a sparse, tiered design rather than a full factorial.

## 5. Component bases and intervention sensitivity

The canonical primary basis remains the predecessor basis where it is defined:
attention heads plus individual MLP neurons, intervened on by activation
zeroing. The phrase "under a fixed component basis" is no longer a universal
claim. Instead, the study tests whether the result survives selected
predeclared changes of granularity and orientation.

Required basis-sensitivity families are:

1. **Canonical basis:** predecessor heads and individual neurons.
2. **Attention refinement:** split each attention head into coordinates before
   the output projection, under one frozen coordinate definition.
3. **Coarse MLP blocks:** prospectively seeded, balanced random partitions of
   neurons into blocks.
4. **Orientation sensitivity:** a limited set of fixed orthogonal rotations in
   eligible activation subspaces.
5. **Accounting sensitivity:** parameter-weighted and component-type-stratified
   summaries alongside raw proportions.

Only a limited, prospectively selected subset is required on the full primary
matrix. Additional random partitions or rotations are lower-priority
sensitivity work. Conclusions are limited to the tested bases and partitions.

## 6. Discovery procedures

At least two algorithmically distinct search families are required for the
protected experiment:

- the inherited discrete greedy/diversity machinery, adapted to the new exact
  fidelity and ledger contract;
- a continuous or stochastic sparse-mask method, provisionally an L0-style or
  hard-concrete mask optimizer, with independent proposal dynamics.

The second method cannot be a cosmetic restart policy over the first. Each
method receives:

- a versioned algorithm and configuration;
- its own native optimization budget and unit;
- the common exact-mask-evaluation allowance;
- frozen restart, failure, and termination rules;
- deterministic identity and seeds where the backend permits;
- complete proposal and exact-evaluation ledgers.

Native budgets are not declared equivalent across unlike methods. Scientific
comparisons are primarily within method; cross-method agreement or
disagreement is itself a result.

## 7. Endpoint hierarchy

### 7.1 Primary endpoint

The primary endpoint remains the smallest exactly evaluated component
proportion satisfying the frozen centred-logit fidelity rule, with the intact
mask included so the endpoint is always defined up to 1.0.

It is a procedure-relative recovered upper bound, not "the minimum circuit
size."

### 7.2 Key secondary endpoint

The circuit packing lower bound becomes a **key secondary endpoint** unless
Stages 12--14 demonstrate that its search, null calibration, and interpretation
are sufficiently stable to justify co-primary status before production.

It is called a procedure-relative packing lower bound. It is not a count of
mechanisms or the true packing number.

### 7.3 Fidelity frontier

The primary fidelity setting remains prospectively frozen, but conclusions
must also be shown across a predeclared centred-logit fidelity frontier. Exact
ledger reuse should derive the frontier without rerunning search wherever the
proposal set supports it. A single favorable threshold cannot carry the paper.

## 8. Packing calibration and nulls

Packing requires layered calibration. The following nulls have distinct roles
and must not be collapsed into one:

1. **Combinatorial floor:** overlap distribution for size- and type-matched
   masks under the declared component universe.
2. **Ordinary-restart baseline:** repeated discovery without an explicit
   diversity pressure.
3. **Local perturbation null:** fidelity-retaining perturbations around a
   recovered mask, under a frozen perturbation proposal.
4. **Tractable feasible-region calibration:** exact or near-exact exploration
   on a small model where the qualifying mask region can be characterized much
   more completely.

These distinguish combinatorial separation, optimizer multiplicity, local
flatness, and broader feasible-region multiplicity. None turns the recovered
packing into an estimate of the total number of mechanisms.

## 9. Tractable search-calibration model

A smaller prospectively specified model/task instance is included to calibrate
how far the search procedures fall below an exact or near-exact solution. It
must be small enough for exhaustive enumeration, branch-and-bound, or a clearly
audited near-exact alternative over the relevant mask space.

This calibration is not treated as another population replicate of the main
experiment. Its purposes are:

- estimate search suboptimality for Endpoint 1;
- compare recovered and feasible packing for Endpoint 2;
- test whether method agreement is informative;
- expose failure modes of the proposed nulls.

## 10. Fourier causal interchange

The Fourier interchange experiment is promoted from an optional extension to a
**registered key secondary analysis**. It is executed regardless of whether
the primary recoverability results are favorable.

The intervention location, alignment, pair selection, capacity matching,
outcome, and trial roster are frozen before production. Required controls are:

- wrong Fourier mode;
- shuffled coefficients;
- mismatched input;
- equal-norm random state;
- unaligned ordinary activation patching.

Aligned interchange supports a shared causal abstraction only if it
outperforms every capacity-matched control. It does not establish uniqueness
among all possible algorithms.

## 11. Analysis principles

- Primary contrasts are paired within teacher seed.
- Every eligible and failed attempt is reported; eligibility is not conditioned
  away in the headline accounting.
- Hard and soft estimands are analyzed separately.
- Architecture, basis, and method effects are reported as conditional repeated
  factors.
- Missing or underpowered cells are unresolved under a frozen rule; they are
  not imputed.
- The primary endpoint is analyzed first. Packing, architecture, basis, null,
  and Fourier results qualify its mechanistic interpretation.
- Fidelity, cap, and overlap sensitivity are reported prospectively, not used
  to choose the most favorable story.
- Teacher-seed values, matched differences, eligibility failures, search
  failures, and budget censoring remain visible.

The analysis may use hierarchical models or randomization procedures, but the
model specification must respect teacher seed as the population unit and must
be frozen before definitive outcomes are inspected.

## 12. Sparse tiered crossing

A literal factorial crossing every teacher, phase, condition, initialization,
architecture, basis, discovery method, fidelity threshold, component cap, and
overlap cutoff is prohibited as the default design. It wastes compute and
creates pseudo-replication without improving the number of teacher-level
replicates.

The production manifest must instead define tiers:

### Tier 1 — protected primary matrix

- full eligible teacher-seed roster and primary phase contrasts;
- direct teachers;
- canonical matched student architecture;
- hard and soft students;
- frozen student replication and failure accounting;
- canonical basis;
- both required discovery families;
- primary Endpoint 1 and key-secondary Endpoint 2;
- core packing calibration.

### Tier 2 — prespecified external-validity and interpretation panel

- additional student architectures on balanced, predeclared cells;
- selected basis re-granulations and rotations;
- full fidelity frontier from reusable ledgers;
- tractable exact/near-exact calibration;
- registered Fourier interchange and controls.

### Tier 3 — optional breadth

- extra architectures beyond the protected panel;
- additional partitions, rotations, null draws, or restarts;
- broader phase trajectories;
- exploratory task-structured descriptive metrics.

Tier 3 may run only after Tier 1 completeness and the protected Tier 2 minimum
are secure. Unused allocation cannot justify changing the primary estimand.

## 13. Production and stopping rules

Definitive production is one Stage 15 campaign driven by a frozen dependency
graph. It is not a sequence of manual stage chats. The campaign must:

- validate every input and configuration hash before launch;
- isolate writable roots by job identity;
- submit jobs only when their prerequisites have passed;
- retry only under predeclared limits;
- retain terminal failures and eligibility failures;
- never transfer unused budgets across scientific cells;
- checkpoint and resume without duplicating attempts;
- emit compact, analysis-ready ledgers rather than verbose JSON traces;
- stop optional work before protected work when capacity becomes scarce;
- reserve the final allocation window for exact recomputation, inventory
  closure, compression, and off-cluster copying.

There are exactly three human decision gates during the campaign:

1. **Launch gate:** hardware/backend qualification plus a tiny end-to-end run.
2. **Primary-completeness gate:** eligibility health and protected Tier 1
   completeness before the remaining allocation is released to lower tiers.
3. **Exit gate:** no new optional jobs; recompute, audit, compact, and export.

Routine monitoring and automatic alerts do not create additional scientific
gates. A human may stop unsafe or malfunctioning execution at any time, but may
not change scientific settings after seeing comparative outcomes.

## 14. Open freeze decisions

Stages 11--14 must resolve at least the following before production:

The machine-readable planning register is
`followup/configs/post_red_team_open_decisions_v1.json`. It extends rather than
mutates the historical Stage 2 UD register.

- expanded teacher roster and phase availability rule;
- student architecture roster and balanced sparse assignment;
- hard and soft training, eligibility, replication, and attempt caps;
- canonical and sensitivity component bases;
- centred-logit implementation, primary threshold, and fidelity frontier;
- Endpoint 2 status, cap, overlap, and packing solver;
- inherited and independent discovery method specifications;
- method-native and common exact-evaluation budgets;
- all four packing null specifications and draw counts;
- tractable calibration model and exactness criterion;
- primary contrasts, summaries, missing-cell rule, and statistical model;
- Fourier pair selection, intervention, controls, and outcome;
- protected Tier 1, protected Tier 2 minimum, Tier 3 priority order;
- backend qualification and reproducibility rule;
- cluster resources, concurrency, retries, storage, export, and abort rules;
- required tables, figures, and claim-resolution rules.

Technical values previously used in Stages 5--10 are nonbinding until adopted
by the prospective freeze.

## 15. Claims boundary

Permitted language includes:

- procedure-relative smallest recovered component proportion;
- recovered upper bound on unknown global sparsity;
- procedure-relative packing lower bound;
- teacher-seed-level evidence about phase/function;
- conditional realization, architecture, basis, and method dependence;
- evidence for a shared causal abstraction under the registered interchange
  and controls.

Prohibited language includes:

- globally minimum or uniquely correct circuit;
- enumeration or number of mechanisms;
- true packing number;
- population replication from students, architectures, methods, thresholds,
  masks, or null draws;
- causal effect of architecture from a non-isolated architecture panel;
- basis invariance beyond tested bases;
- equal compute across unlike discovery methods;
- uniquely identified algorithm from Fourier interchange;
- blinded confirmation of the predecessor result.

## 16. Explicitly gated extensions

Generic entropy estimation, broad cross-task expansion, general theory, and
conditional atlases remain outside the required campaign. A task-structured
descriptive function metric may be included only if it is specified before
production and is not presented as a causal mediator.
