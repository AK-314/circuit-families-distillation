# Distilled-Realization Circuit Recoverability

## Follow-up experimental protocol — draft for prospective freeze

**Protocol status:** Stage 2 scientific skeleton frozen; full numeric, roster, analysis, and production protocol remains draft and prospectively unresolved
**Relationship to predecessor:** Prospective follow-up to `experimental_protocol.md`
**Results already visible:** The predecessor study's training and circuit-family results
**Results that must not be generated before this protocol is frozen:** Distillation outcomes used for scientific comparison, centred-logit circuit endpoints, and Fourier-interchange outcomes

## 1. Governance and relationship to the predecessor study

The predecessor protocol and its amendment history remain immutable. This document does not amend or replace them. It defines a new experiment that reuses their task, trained teacher models, checkpoint-selection rules, component basis, masking semantics, artifact conventions, and integrity machinery where explicitly stated.

This is not globally pre-results: the original circuit-family results are known. It is prospective only with respect to the new distillation experiment and the new primary endpoints. Existing results must be disclosed as prior evidence when the follow-up is reported.

The Stage 2 scientific skeleton is frozen, but before endpoint-producing scientific execution the remaining freeze-register choices must be resolved prospectively and the full protocol frozen. Any method-development output produced before the relevant freeze must be registered, excluded from the primary analysis, and regenerated after freeze.

## 2. Frozen research question

> Under a fixed component basis, intervention and evaluation protocol, does the grokking-associated change in sparse-circuit recoverability follow the teacher function, vary across distilled realizations of that function, or arise from the distillation and discovery procedures themselves?

## 3. Experimental hierarchy and unit of inference

```text
teacher seed
  └── phase checkpoint/function
        └── distillation condition: hard or soft
              └── student initialization
                    └── discovery method
                          └── thresholds and circuits
```

- The independently trained teacher seed is the population-level unit.
- Phase is a repeated condition within teacher seed.
- Student initialization estimates realization sensitivity conditional on a teacher, phase, and distillation condition.
- Discovery methods, thresholds, and recovered circuits are repeated measurements, not independent replications.
- Hard- and soft-target students are separate estimands and must never be pooled.
- Every selected teacher checkpoint is evaluated directly under the same component, masking, fidelity, and exact-evaluation protocol used for students.

## 4. Reused fixed substrate

Unless the freeze register records a prospective change, the follow-up reuses:

- modular addition modulo 113 and all 12,769 ordered inputs;
- the existing five teacher training runs and their densely saved checkpoints;
- the one-layer transformer architecture;
- the component basis of four attention heads and 512 MLP neurons;
- the existing activation-zeroing intervention and mask representation;
- full-domain exact evaluation in deterministic input order;
- artifact hashes, run manifests, checkpoint integrity checks, and deterministic seed derivation.

A student must use the same architecture and searchable component basis as its teacher unless a different choice is frozen before any student training. This equality is necessary for component proportions to be directly comparable.

## 5. Distillation conditions

### 5.1 Hard-target students

- Targets are the teacher's argmax decisions on the complete 12,769-input universe.
- Eligibility requires reproducing 100% of those teacher decisions on the complete universe.
- An ineligible student is recorded as a failed realization attempt and is excluded from circuit endpoint analysis.
- Failure counts and rates remain reported; they must not be hidden by replacement.
- Circuit fidelity is always relative to the eligible student's own dense outputs, not directly to the teacher.

### 5.2 Soft-target students

- The primary matching target must be frozen as either per-input centred logits or probabilities. Centred logits are the recommended primary target because they remove the additive-logit gauge and align with the proposed circuit-fidelity metric.
- The loss, any temperature, normalization, and full-domain acceptance tolerance must be frozen prospectively.
- Whether 100% teacher–student argmax agreement is an eligibility requirement must be frozen prospectively. The recommended rule is to require it in addition to the soft tolerance.
- Circuit fidelity is relative to the eligible student's own dense outputs.

### 5.3 Attempt accounting

The number of planned student initializations, maximum attempts per cell, replacement rule, and minimum number of eligible students required for a cell must be frozen. All attempted initializations remain in the realization-attempt registry even when they are ineligible.

## 6. Primary predictive fidelity

For dense model (M), component mask (C), input (x), and final-position logits (z), define per-input class-centred logits

\[
\tilde z(x)=z(x)-\frac{1}{K}\sum_{k=1}^{K}z_k(x).
\]

The proposed primary circuit fidelity is

\[
F_{\mathrm{logit}}(C)
=
1-
\frac{\sum_x\lVert\tilde z_C(x)-\tilde z_M(x)\rVert_2^2}
{\sum_x\lVert\tilde z_M(x)\rVert_2^2}.
\]

The exact summation precision, denominator guard, and threshold must be frozen. A threshold of 0.99 is the recommended default, but it is not frozen merely by appearing in this draft.

Top-one agreement, KL divergence, Jensen–Shannon divergence, cross-entropy change, and accuracy are secondary diagnostics. The predecessor study's top-one fidelity is not silently substituted for the new primary predictive fidelity.

## 7. Primary endpoint 1: smallest recovered component proportion

For teacher or eligible student model (M), discovery method (d), and frozen fidelity threshold (\tau), define

\[
S(M,d;\tau)
=
\min_{C\in\mathcal E(M,d)\cup\{C_{\mathrm{full}}\}}
\left\{
\frac{|C|}{516}:F_{\mathrm{logit}}(C)\geq\tau
\right\},
\]

where \(\mathcal E(M,d)\) is the set of masks evaluated exactly by method \(d\) under its frozen budget.

- The intact mask is inserted as a mandatory qualifying baseline and has proportion 1.0.
- Values up to and including 1.0 are allowed, so the endpoint remains defined when sparse recovery fails.
- The recorded endpoint is the smallest qualifying exactly evaluated mask, not merely the method's nominal terminal mask.
- The quantity is procedure-relative and an upper bound on the unknown globally minimal sufficient proportion.
- It must be called neither “minimum circuit size” nor evidence of global optimality.
- Search termination status is recorded separately: completed, budget exhausted, optimization failed, or no sub-full qualifying mask recovered.

## 8. Primary endpoint 2: circuit packing lower bound

At frozen predictive fidelity (\tau), maximum component proportion (p_{\max}), maximum pairwise overlap (o_{\max}), and method-specific search budget (B_d), define the recovered packing count as the largest mutually separated subset among the valid circuits returned by method (d):

\[
P(M,d;\tau,p_{\max},o_{\max},B_d).
\]

- Every included circuit must satisfy the fidelity and maximum-proportion rules by final exact evaluation.
- Every pair must satisfy the frozen overlap rule in the common 516-component basis.
- Zero is recorded when no qualifying circuit is recovered.
- The quantity is a packing lower bound under the specified procedure, not the true packing number.
- If the method has generated more valid circuits than can be packed greedily without a guarantee of maximal subset selection, the final packing subset must be computed by a frozen exact or deterministic combinatorial rule.

Discovery methods may use different optimization primitives. Each method therefore receives a prospectively frozen native optimization budget plus a common allowance of final exact mask evaluations. Phase and student effects are interpreted within method. Raw packing counts across methods are not described as perfectly resource-matched.

## 9. Discovery methods and budget accounting

The method roster must be frozen before endpoint-producing search. Each method specification must include:

- algorithm version and configuration;
- native optimization budget and its unit;
- common final exact-evaluation allowance;
- restart count and deterministic seed derivation;
- candidate deduplication;
- stopping and failure rules;
- complete trajectory or proposal logging sufficient to recompute both endpoints.

The existing greedy deletion and diversity-forced machinery may be reused, but reuse does not exempt it from the new centred-logit objective, exact-evaluation allowance, or method-specific budget declaration.

## 10. Primary contrasts and summaries

For endpoint (Y\), teacher seed (s\), phase (p\), condition (c\), student initialization (r\), and method (d\):

- teacher direct result: \(Y^{T}_{s,p,d}\);
- student result: \(Y^{S}_{s,p,c,r,d}\);
- teacher-cell student summary: the prospectively frozen robust summary across eligible \(r\), recommended to be the median;
- phase contrast: within-seed change in that summary between prospectively frozen phase landmarks;
- distillation contrast: student cell summary minus the directly evaluated teacher value for the same seed, phase, and method;
- realization sensitivity: within-cell dispersion across eligible student initializations, reported by range and median absolute deviation.

Population-level summaries are across teacher seeds. Student runs and circuits must not be pooled to inflate the sample size. With five teacher seeds, raw seed-level trajectories, paired changes, sign consistency, ranges, and robust summaries carry the argument; conventional significance thresholds do not determine the conclusion.

Eligibility failure rates are reported by teacher seed, phase, and distillation condition. Cells below the frozen minimum eligible-student count are unresolved, not imputed.

Hard and soft conditions receive separate tables, figures, summaries, and conclusions.

## 11. Frozen outcome interpretations

These are outcome categories, not directional predictions.

- **Teacher phase effect persists in both student conditions:** evidence that properties of the teacher function contribute.
- **Students of one teacher vary materially:** internal realization contributes independently.
- **All distilled students become similarly compressible:** distillation-induced representation dominates.
- **Results depend strongly on discovery method or fidelity definition:** protocol-dependent phenomenon.
- **Stable functions enable recovery, while within-function realization controls packing:** interaction between function structure and implementation.
- **The original transition disappears under predictive fidelity:** the predecessor result was substantially driven by top-one margins or decision-level tolerance.

No single category is a success criterion, and multiple categories may apply.

## 12. Secondary Fourier interchange experiment

Fourier interchange is secondary and outside the critical path of the core study. It may start only after its intervention, alignment rule, information-capacity matching, and controls are frozen.

Required controls are:

- wrong Fourier mode;
- shuffled coefficients;
- mismatched input;
- equal-norm random state;
- unaligned ordinary activation patching.

Success supports a shared causal abstraction only when aligned interchange outperforms every control under matched information capacity. It does not establish uniqueness among all possible algorithms.

## 13. Scope gates

The core paper requires teacher/student distillation, direct teacher evaluation, both primary endpoints, method-aware budget accounting, hierarchical analysis, and reproducible reporting.

The following are explicitly gated extensions and cannot delay the core paper:

- entropy estimation;
- cross-task breadth;
- general theory;
- conditional atlases;
- Fourier interchange beyond the prespecified secondary experiment.

## 14. Method-development and pilot firewall

Before the definitive run, a technical pilot may evaluate:

- convergence and runtime;
- artifact sizes and worker isolation;
- numerical correctness of centred-logit evaluation;
- intact-mask invariants;
- training acceptance implementation;
- interface compatibility between trainers and search methods.

The pilot may not be used to select phase landmarks, endpoint thresholds, sparsity cutoffs, overlap cutoffs, student counts, or search methods based on apparent phase or condition effects. Any endpoint values emitted incidentally are registered as excluded method-development output and regenerated after freeze.

## 15. Freeze register

The Stage 2 skeleton freeze does not resolve the register below. The full protocol cannot move from draft to frozen until these are resolved:

| Decision | Required frozen value |
|---|---|
| Teacher registry | Exact five seeds, per-seed checkpoint hashes, and phase labels |
| Primary phase grid | Exact checkpoints/functions used for definitive distillation |
| Student architecture | Exact config and initialization rule |
| Student replication | Planned initializations, max attempts, minimum eligible count |
| Hard training | Loss, optimizer, schedule, stopping rule, attempt budget |
| Soft target | Centred logits or probabilities |
| Soft training | Loss, temperature if any, optimizer, schedule, stopping rule |
| Soft eligibility | Numeric tolerance and argmax rule |
| Primary fidelity | Formula implementation, threshold, numerical precision |
| Endpoint 2 size cap | Maximum component proportion |
| Endpoint 2 overlap | Metric and cutoff |
| Discovery roster | Exact methods and versions |
| Method budgets | Native units and values per method |
| Exact evaluation allowance | Common value and counting rule |
| Phase contrasts | Exact primary contrasts |
| Student cell summary | Median or other fixed summary |
| Missing-cell rule | Minimum eligible students and unresolved-cell handling |
| Production concurrency | Worker count, threading, output isolation, merge rule |
| Analysis outputs | Required tables, figures, and manifests |

## 16. Prohibited claims

The study must not claim:

- recovery of the globally minimum circuit;
- enumeration of all circuits;
- a true packing number rather than a lower bound;
- independent replication from students, methods, thresholds, or circuits;
- equivalence of hard- and soft-target functions;
- perfectly equal compute across unlike discovery methods;
- uniquely identified causal abstraction from Fourier interchange;
- blinded confirmation of the predecessor result.
