# Stage 13 final amended decision dossier — approval gate

**Status:** pending Alex approval  
**Package:** `stage13-package-a-protected-core-optional-five-task/v3`  
**Implementation base:** `7976c98cc83a6df098ae0ef8c59b56027a7f4899`  
**Machine authority:** `followup/decisions/stage13_decision_dossier_v3.json`  
**Dossier SHA-256:** `642118fe30e1c435fc05c656ac6167446c76a445c9a1efd376d5a7c23410b1f4`

The v3 amendment changes RD-001, RD-012, RD-014, and only the backend-authority
sentence of RD-007 needed for the requested qualified school-Mac MPS branch.
All other decision content is preserved from v2 and hash-bound in the machine
dossier. Parts C–H have not begun.

## Final scope

The required launch scope is **Tier 1 plus protected Tier 2**:

- Task 1 modular addition, fixed teacher seeds `0–14`, with no replacement;
- Task 2 modular multiplication, fixed seeds `0–4`, pre/stable phases;
- the accepted connected architecture and basis panels;
- both discovery methods, Endpoint 1, packing and reducer sensitivities;
- protected null calibration and exact `2^18` calibration;
- the accepted Fourier intervention and all five controls.

Task 3, Task 4, and Task 5 are optional Tier 3 increments. They are excluded
from launch feasibility and run only in that order after protected completion
and the 12-hour audit reserve are secure. A failed task, seed, phase, teacher,
or student slot is retained and is never replaced. Tasks are task-indexed
external-validity conditions, not independent population replicates.

## Exact five-task panel

| Task | Status | Exact definition | Fixed panel |
|---|---|---|---|
| 1 | Protected Tier 1 | `(x+y) mod 113`; full `113×113`; seeds `0–14` | Accepted Package A panel |
| 2 | Protected Tier 2 | `(x*y) mod 59`; full `59×59`; seeds `0–4` | Pre/stable; accepted protected Task 2 design |
| 3 | Optional 1 | `(x²+xy+y²) mod 59` | Seeds `0–4`; pre/stable; direct + lowest eligible hard/soft; canonical architecture/basis; both methods |
| 4 | Optional 2 | `(x³+y²) mod 59` | Same fixed optional panel |
| 5 | Optional 3 | `(x²y+xy²) mod 59 = xy(x+y) mod 59` | Same fixed optional panel |

Task 4 is asymmetric, separable, mixed degree, and nonlinear; it is not modular
subtraction or a linear relabelling of Task 1. Task 5 is a coupled homogeneous
cubic with no pure monomial. It differs from separable Task 4, symmetric
quadratic Task 3, and single-monomial multiplication.

Tasks 3–5 share a fixed full-domain split: rank all 3,481 `(x,y)` examples by
`SHA256("stage13/reduced-task-split/v1\0" + decimal(x) + "\0" + decimal(y))`,
break ties lexicographically, assign the first 1,044 to training and the
remaining 2,437 to test.

| Increment manifest | Manifest SHA-256 | Task identity SHA-256 |
|---|---|---|
| Task 3 | `b8797e72cceec09c5be3e1870dab9909c4140fc4e08df9ed4e76050cff9b88ff` | `df5e9285cd2c9985ae2866282d534f738b598e3caf13bdc14dc84828360733eb` |
| Task 4 | `7490e47485204d82745f2bda47acd6d3ca37c28e85c153e82694bcbdf53c4758` | `fef2f01e151ce8324f149acee17e5d5d7f9038db65e4e9c3a20d226ca7d6bc86` |
| Task 5 | `cf784b526eb801c8a6032cc0addb45c8376190139e5a324c9c37d9c0db6e1b87` | `59ee026362f1a21e991cb4c5957b60b23d320847d01f0e613f74434e5620a619` |

## Corrected protected-core compute projection

`h_support` is the Stage 14 measured/reserved number of host CPU cores per
active accelerator. Device-hours do not include this host reservation.

| Scenario | GPU/accelerator device-h | Host support CPU-core-h | Standalone CPU-core-h | Weak/serial CPU h | Max useful CPU | Ideal wall at max concurrency | Efficiency-adjusted wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lower | 162.593 | `162.593 × h_support` | 12.759 | 2 | 256 | 2.112 h | 2.484 h |
| Central | 740.964 | `740.964 × h_support` | 25.249 | 5 | 256 | 5.640 h | 8.057 h |
| Conservative | 2,847.230 | `2,847.230 × h_support` | 80.343 | 12 | 256 | 15.762 h | 31.523 h |

Protected job counts are 21 teacher jobs; 470/660/940 student attempts;
645 greedy and 645 hard-concrete ledgers; 2,580 hard-concrete restart runs;
960 ordinary-restart subjobs; 615,424 exact evaluations; 92,160 Fourier
input-condition trials; and one protected merge/export.

The ideal wall calculation uses phase-specific useful concurrency rather than
dividing all work by one invented machine count. The efficiency-adjusted value
uses the scenario efficiency only as a planning sensitivity. Stage 14 replaces
all rates, efficiencies, machine counts, and availability with measurements.

## Optional increment compute projection

Tasks 3, 4, and 5 have identical planned job counts but require separate
task-specific Stage 14 throughput qualification. Each has 5 teacher jobs,
20/30/40 student attempts, 30 ledgers per method, 120 hard-concrete restarts,
15,360 exact evaluations, and one incremental merge/export.

| Each increment | Device-h | Host support CPU-core-h | Standalone CPU-core-h | Weak/serial CPU h | Max useful CPU | Ideal wall | Adjusted wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lower | 6.477 | `6.477 × h_support` | 0.628 | 0.5 | 60 | 0.600 h | 0.706 h |
| Central | 30.139 | `30.139 × h_support` | 1.378 | 1 | 60 | 1.583 h | 2.262 h |
| Conservative | 117.428 | `117.428 × h_support` | 4.378 | 3 | 60 | 6.534 h | 13.068 h |

## Corrected storage projection

| Scope | Scenario | Retained compact | Peak active scratch | Uncompressed/staging/retry worst | Persistent quota request | Scratch quota request |
|---|---|---:|---|---:|---:|---:|
| Protected core | Lower | 6.334 GiB | `0.25G + 0.05C + merge staging` | 39.828 GiB | 8 GiB | 64 GiB |
| Protected core | Central | 14.658 GiB | `0.50G + 0.10C + merge staging` | 83.750 GiB | 32 GiB | 128 GiB |
| Protected core | Conservative | 41.272 GiB | `1.00G + 0.25C + merge staging` | 203.731 GiB | 64 GiB | 256 GiB |
| Each optional task | Lower | 0.368 GiB | `0.25G + 0.05C + merge staging` | 2.042 GiB | +1 GiB | +4 GiB |
| Each optional task | Central | 0.807 GiB | `0.50G + 0.10C + merge staging` | 4.215 GiB | +2 GiB | +8 GiB |
| Each optional task | Conservative | 1.976 GiB | `1.00G + 0.25C + merge staging` | 9.417 GiB | +4 GiB | +16 GiB |

`G` and `C` are the actual simultaneously active accelerator and standalone
CPU workers bound by Stage 14. Safety quotas are rounded to the next power of
two above demand divided by 0.80. If measured artifacts are larger, Stage 14
must recompute upward; it may not reduce retained evidence.

## Stage 14 actual-resource binding

No provider, institution, 16-GPU pool, 256-core pool, MPS qualification, or
uninterrupted window is presumed. Stage 14 must bind:

- actual machine count and hardware class;
- qualified CPU/CUDA/MPS status and minimum-of-three post-warmup throughputs;
- host CPU support reservation per accelerator;
- memory, VRAM, active scratch, persistent and scratch quotas;
- availability intervals, permitted hours, queue/preemption behavior;
- measured scheduling, merge, and export efficiency.

Let `H_total` be the sum of verified permitted availability intervals and
`H_science = max(0, H_total−12h)`. The dependency-closed protected schedule must
fit `H_science`; availability is integrated by interval rather than treated as
continuous.

For a school-Mac branch, Stage 14 binds machine count `M`, class counts `M_k`,
CPU cores/RAM/disk per class, allowed intervals, CPU training/hard-concrete/
exact rates, and MPS rates only when that class passes exact and semantic MPS
qualification. Host support reservations are removed before standalone CPU
capacity is calculated. Unqualified or absent MPS falls back to measured CPU
rates, not an assumed speedup.

If no verified configuration completes the protected core within
`H_science`, launch blocks pending a prospective amendment. Once protected
completion and audit capacity are secure, Task 3 is admitted only if its whole
increment fits; the same rule then applies to Task 4 and Task 5.

## Approval gate

Exact approval sentence:

> I approve `stage13-package-a-protected-core-optional-five-task/v3` exactly as described in the Stage 13 v3 decision dossier with SHA-256 `642118fe30e1c435fc05c656ac6167446c76a445c9a1efd376d5a7c23410b1f4`; Parts C–H may proceed subject to the Stage 14 protected-core launch gate.

Until that sentence is supplied by Alex, approval remains pending and Parts
C–H remain blocked.
