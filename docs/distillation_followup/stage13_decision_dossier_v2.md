# Stage 13 revised decision dossier — approval gate

**Status:** Packages A, B, and C pending Alex approval; Parts C–H not begun  
**Implementation base:** `7976c98cc83a6df098ae0ef8c59b56027a7f4899`  
**Scientific data / production eligibility:** false / false  
**Machine-readable authority:** `followup/decisions/stage13_decision_dossier_v2.json`  
**Supersedes:** unapproved v1 SHA-256 `8bce11d84632e6bc8b3ae4f5abf30c0348ff91be651984fdea984a383d0dc5b7`

This revision uses only admitted Phase I context, accepted Stage 11/12 contracts,
technical benchmark records, and constructed non-scientific timing evidence. It
does not use registered/private model artifacts, scientific endpoint direction,
or unpublished comparative results.

## Compact decision table

| Decision | Package A recommendation | Package B / C distinction |
|---|---|---|
| RD-001 | Task 1 seeds `0–14`, fixed without replacement; pre/50%/stable; primary matched pre vs stable. Task 2 multiplication mod 59; Task 3 `(x²+xy+y²) mod 59`. | B keeps the roster. C trains `0–19`, taking first 15 complete by seed order. |
| RD-002 | Canonical plus depth-2, 8-head, MLP-256, MLP-1024; every alternate on shared anchors `0–4`, primary phases, hard/soft. | B keeps depth-2 and 8-head on every anchor. C uses A. |
| RD-003 | Tier 1: 3 eligible of 6 attempts; Tier 2: 2 of 4. Hard exact argmax; soft centred-logit MSE ≤`1e-4` plus exact argmax. | B: 2 of 4 Tier 1, 1 of 2 external, `1e-3`. C uses A. |
| RD-004 | Canonical plus attention coordinates, one seeded 32-neuron block partition, two rotations; every basis on the same seed-`0–4` direct/hard/soft checkpoints. | B retains attention, blocks, rotation-0 on the same panel. C uses A. |
| RD-005 | Fidelity `0.99` primary; frontier `0.90, 0.95, 0.975, 0.99, 0.995`; class-centred logits, float64 accumulation, exact-zero rejection. | Identical. |
| RD-006 | Endpoint 1 sole primary. Packing key secondary at cap `0.25`, Jaccard ≤`0.25`. Full `3×4` grid reducer-only. | Identical definitions. |
| RD-007 | Greedy: 516 proposals. Hard-concrete: `4×5000`; 256 exact total per method ledger, intact included. | B uses `2×2500`, 128 exact. C uses A. |
| RD-008 | Fixed 60-slot anchor null panel; 10,000 draws, 16 ordinary restarts, 128 local exact requests; no replacement. | B uses 2,000 / 8 / 64. C uses A. |
| RD-009 | Addition mod 7, 18 components, exact `2^18 = 262,144` enumeration and certificate. | Identical. |
| RD-010 | Teacher-seed paired mean; median student cell; raw/MAD/range; 10,000 bootstrap; sign-flip/Holm; no imputation. | Identical. |
| RD-011 | Internal `blocks.0.mlp.hook_post`; diagonal modes `(1,1)…(4,4)`; counterfactual centred-logit displacement fidelity; 256 trials; all five controls; no uniqueness. | Identical test. |
| RD-012 | Full canonical Task 1 Tier 1; connected architecture/basis panels, Task 2, frontier, exact calibration, Fourier controls protected. | B reduces only predeclared breadth. C changes only roster acquisition. |
| RD-014 | Two infrastructure retries; rolling retention 2; 12-hour audit; complete E16/E8/CPU256 branches selected by frozen thresholds. | Package switching or shedding is not an operational fallback. |

RD-013 remains wholly deferred to Stage 14 provider/backend qualification.

## Full job-count projection

Counts are maximum logical slots unless labelled lower/central/conservative.
Rows marked “subset” are already included in discovery/exact totals.

| Work class | Package A | Package B | Package C |
|---|---:|---:|---:|
| Teacher training jobs | 26 | 26 | 31 |
| Student attempt jobs, lower / central / conservative | 490 / 690 / 980 | 260 / 390 / 520 | 490 / 690 / 980 |
| Architecture student attempts, subset | 160 / 240 / 320 | 40 / 60 / 80 | 160 / 240 / 320 |
| Greedy discovery ledger jobs | 675 | 415 | 675 |
| Hard-concrete discovery ledger jobs | 675 | 415 | 675 |
| Hard-concrete native restart runs | 2,700 | 830 | 2,700 |
| Ordinary-restart profile jobs / subjobs | 60 / 960 | 60 / 480 | 60 / 960 |
| Discovery exact evaluations, intact included | 345,600 | 106,240 | 345,600 |
| Ordinary-restart exact evaluations, intact included | 15,360 | 7,680 | 15,360 |
| Local-null exact evaluations | 7,680 | 3,840 | 7,680 |
| Combinatorial-null draws | 600,000 | 120,000 | 600,000 |
| Basis-panel discovery ledgers, both methods, subset | 240 | 180 | 240 |
| Architecture-panel discovery ledgers, both methods, subset | 320 | 80 | 320 |
| Fourier sets / condition jobs / input-condition trials | 60 / 360 / 92,160 | 60 / 360 / 92,160 | 60 / 360 / 92,160 |
| Exact calibration jobs / masks | 1 / 262,144 | 1 / 262,144 | 1 / 262,144 |
| All exact evaluations including calibration | 630,784 | 379,904 | 630,784 |
| Merge jobs / verified export jobs | 1 / 1 | 1 / 1 | 1 / 1 |

A/C have 675 ledgers per method: 315 canonical Task 1 direct/student models,
160 architecture students, 120 alternative-basis models, 50 Task 2 models, and
30 Task 3 models. B has 415: 225 + 40 + 90 + 30 + 30. Null calibration uses
`seeds 0–4 × {pre,stable} × {direct,lowest-hard,lowest-soft} × two methods = 60`
slots. Unavailable slots remain unavailable; the projection never replaces them.

## Compute, memory, storage, and wall time

The envelope is 96 hours, 256 CPU cores, 16 CUDA GPUs, 4 TiB scratch, and
1 TiB persistent. The final 12 hours are exclusively for audit/export, leaving
an 84-hour scientific ceiling. The grant itself is unverified.

| Pkg | Scenario | Attempts | GPU-h | CPU-core-h | Science h | Total h | GPU-worker GiB | 256 CPU GiB | Persist GiB | Scratch GiB | Fits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | lower | 490 | 169.070 | 13.136 | 13.492 | 25.492 | 0.861 | 123.392 | 12.054 | 18.081 | yes |
| A | central | 690 | 771.103 | 25.628 | 72.991 | 84.991 | 1.720 | 192.000 | 24.711 | 49.421 | yes, narrow |
| A | conservative | 980 | 2,964.659 | 80.683 | 383.213 | 395.213 | 3.440 | 256.000 | 57.584 | 172.753 | **no** |
| B | lower | 260 | 33.822 | 6.963 | 3.519 | 15.519 | 0.861 | 123.392 | 5.672 | 8.509 | yes |
| B | central | 390 | 189.992 | 19.453 | 21.072 | 33.072 | 1.720 | 192.000 | 12.231 | 24.462 | yes |
| B | conservative | 520 | 1,121.352 | 74.483 | 152.751 | 164.751 | 3.440 | 256.000 | 29.131 | 87.394 | **no** |
| C | lower | 490 | 169.199 | 13.136 | 13.501 | 25.501 | 0.861 | 123.392 | 12.093 | 18.140 | yes |
| C | central | 690 | 772.075 | 25.628 | 73.078 | 85.078 | 1.720 | 192.000 | 24.763 | 49.526 | yes, narrow |
| C | conservative | 980 | 2,972.521 | 80.683 | 384.195 | 396.195 | 3.440 | 256.000 | 57.689 | 173.066 | **no** |

“GPU-worker GiB” is a planning proxy, not a CUDA VRAM measurement. CPU
aggregate assumes 256 resident workers; Stage 14 must lower concurrency if host
memory requires it.

### Central additive compute breakdown

| Work class | A | B | C | Unit |
|---|---:|---:|---:|---|
| Teacher training | 5.056 | 5.056 | 6.028 | GPU-h |
| Canonical student training | 87.500 | 64.167 | 87.500 | GPU-h |
| Architecture student training | 60.214 | 16.742 | 60.214 | GPU-h |
| Greedy ranking | 0.025 | 0.015 | 0.025 | CPU-core-h |
| Main hard-concrete | 525.000 | 80.694 | 525.000 | GPU-h |
| Ordinary-restart hard-concrete | 93.333 | 23.333 | 93.333 | GPU-h |
| Discovery exact evaluation | 8.491 | 2.610 | 8.491 | CPU-core-h |
| Ordinary exact evaluation | 0.377 | 0.189 | 0.377 | CPU-core-h |
| Local-null exact evaluation | 0.189 | 0.094 | 0.189 | CPU-core-h |
| Combinatorial null | 0.003 | 0.001 | 0.003 | CPU-core-h |
| Exact `2^18` calibration | 0.364 | 0.364 | 0.364 | CPU-core-h |
| Fourier capture/alignment/intervention | 0.179 | 0.179 | 0.179 | CPU-core-h |
| Merge and verified export | 16.000 | 16.000 | 16.000 | CPU-core-h |

Non-additive panel attribution:

| Subset | A | B | C |
|---|---:|---:|---:|
| Basis ledgers / exact evals / hard-concrete steps | 240 / 61,440 / 2,400,000 | 180 / 23,040 / 450,000 | 240 / 61,440 / 2,400,000 |
| Architecture ledgers / exact evals / hard-concrete steps | 320 / 81,920 / 3,200,000 | 80 / 10,240 / 200,000 | 320 / 81,920 / 3,200,000 |

All lower/central/conservative category values and formulas are in
`followup/manifests/stage13_package_resource_projection_v2.json`.

## Benchmark evidence and unknowns

The constructed benchmark accessed no registered model. Three Package-A-size
native hard-concrete runs (516 gates, 5,000 steps, checkpoint 50, retention 2)
had median 1.905668 seconds, or 2,623.751 native steps/s. Three 16-restart
ledger-bridge runs with a 256 allowance had median 0.667390 seconds. These time
mechanics, checkpointing, deduplication, qualification, and packing—not model
forward/backward cost. Stage 10 measured full-domain exact evaluation at
0.088445375 seconds on its technical CPU fixture.

Explicitly unbenchmarked: CUDA training throughput/VRAM; model-in-loop
hard-concrete throughput/VRAM; concurrent scheduler/filesystem efficiency;
alternate-architecture speed beyond parameter scaling; Fourier throughput;
tiny mod-7 enumeration throughput; merge/export throughput; actual hardware,
quotas, paths, queue latency, and grant.

Scenarios use 5k/20k/40k realized updates per training attempt;
0.018544916/0.035/0.141523208 seconds per canonical-equivalent update;
0.035/0.14/0.28 seconds per model-in-loop hard-concrete step; and
85%/70%/50% efficiency. Lower/central CUDA rates are hypotheses. Conservative
training uses the measured CPU reference, not a CUDA observation.

## Frozen Stage 14 feasibility branches

Stage 14 must time the actual backend three times after one discarded warmup
and use the minimum. Complete branches are E16 (16 CUDA + up to 256 CPU), E8
(8 CUDA + up to 256 CPU), and CPU256 recovery. Every branch contains every job,
seed, method, control, sensitivity, and exact allowance in the approved package.

Pass equation:

`training_updates/(workers × measured_training_rate × 0.70) + hard_concrete_steps/(workers × measured_HC_rate × 0.70) + measured_CPU_critical_path + 4 h ≤ 84 h`.

Central bindings are A `15,713,440` training-equivalent updates and
`15,900,000` hard-concrete steps; B `8,842,000` and `2,675,000`; C
`15,813,440` and `15,900,000`. Reference E16 floors are 28.571 training
updates/s/GPU, 7.143 model-loop hard-concrete steps/s/GPU, and 11.306 exact
evaluations/s/CPU worker. E8 GPU floors double. Queue and merge/export critical
paths are included. Memory/storage projections must be ≤80% of verified
capacity, backend checks must pass, and the audit reserve cannot be consumed.
Select E16, then E8, then CPU256 only if it passes. Otherwise launch is blocked
pending a new Alex-approved amendment.

## Changed decision entries

### RD-002 — connected architecture panel

Every alternate uses anchors `{0,1,2,3,4}`, both primary phases, hard/soft.
Permitted contrast: alternate minus canonical student-cell median within the
identical teacher seed, phase, and condition, aggregated over complete pairs.
Missing cells remain missing. No isolated causal architecture claim is allowed.

### RD-004 — paired basis panel

Each available seed-`0–4` pre/stable direct teacher, lowest eligible hard
student, and lowest eligible soft student receives canonical, attention,
32-neuron-block, rotation-0, and rotation-1 bases. Contrasts pair each basis to
canonical on the same model/checkpoint/method. Counts include raw,
parameter-weighted, and component-type denominators.

### RD-006 — packing grid

Endpoint 1 is primary. Packing at cap `0.25`, Jaccard `0.25` is the
procedure-relative key secondary. Caps `{0.125,0.25,0.375}` cross overlaps
`{0,0.125,0.25,0.5}`; the other 11 cells are descriptive. Every cell is
reducer-only over sealed-ledger masks/fidelities: no discovery, inference,
exact evaluation, or mutation. An absent mask stays absent.

### RD-007/RD-008 — measured mechanics and null panel

A/C retain two discovery families, `4×5000` hard-concrete steps, and 256 exact
evaluations including intact. The null panel is exactly seeds `0–4 ×
pre/stable × direct/lowest-hard/lowest-soft × two methods`. Ordinary restarts
have no diversity/packing feedback and remain the same family. B's reductions
are a complete alternative, never an adaptive cut.

### RD-011 — internal causal Fourier interchange

Primary location is `blocks.0.mlp.hook_post`, after nonlinearity and before
`W_out`. A function `g(x+y mod p)` has 2-D Fourier support on `(k,k)`, so primary
modes are `(1,1),(2,2),(3,3),(4,4)`. The wrong-mode control is separate,
disjoint, capacity-matched support `(1,2),(2,1),(1,-1),(2,-2)` and never informs
fitting.

The interchanged variable is the internal modular-sum code. For base `(x,y)`
and donor sum `s_d`, the recipient task-code projection is replaced by the
aligned donor projection. Target behavior is the intact recipient on
`(x,(s_d-x) mod p)`: change from `s` to `s_d` while preserving the recipient's
own counterfactual centred-logit pattern. Primary outcome:

`1 - ||C(logits_intervened)-C(logits_counterfactual)||² / ||C(logits_base)-C(logits_counterfactual)||²`.

`C` subtracts the class mean. One is exact counterfactual behavior, zero is no
movement, negatives remain, zero denominator is unavailable. Aligned must beat
wrong mode, shuffled coefficients, mismatched input, equal-norm random state,
and unaligned ordinary patching; counterfactual argmax and no-change
preservation must pass. This tests recipient downstream use of a donor internal
sum variable, not final-output compatibility. Uniqueness claims are prohibited.

### RD-012/RD-014 — launch blocker

Protected architecture/basis panels use the connected assignments above.
Central A/C are close to the 84-hour ceiling and every conservative scenario
exceeds it. Stage 14 may bind provider facts and select only a passing complete
branch. It may not change scope, calibration, priority, or audit reserve.

## Approval gate

No package is approved. No production config, scientific manifest, campaign,
claim registry, or export bundle in Parts C–H may be created. Alex may approve
one complete package by identifier and this dossier's SHA-256, reject all, or
request another prospective amendment.
