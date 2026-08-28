# Stage 9 student-training backend benchmark

Stage 9 benchmarks one full-domain optimizer update for hard and soft target
conditions on the frozen 227,313-parameter architecture. Each available
backend/condition pair is repeated twice from the same initialization. The
comparison records byte-level state/output equality, parameter and output
drift, and full-domain argmax disagreement.

The benchmark also reports checkpoint/dense-output storage and simple linear
runtime scenarios. Those scenarios are planning bounds, not a scientific
training configuration. Optimizer, schedule, stopping, attempt cap, soft
tolerance, and definitive backend remain unresolved until scientific
red-teaming and the later freeze stages.

Qualification rule:

- CPU may be a deterministic reference candidate only if both hard and soft
  repeats are byte-identical.
- MPS remains development-only for definitive training even when semantic
  outputs agree, because the accepted workload exercises operations for which
  PyTorch reports nondeterministic MPS implementations.
- CUDA is unmeasured locally and must pass this same conformance benchmark on
  the eventual machine before it can be used.
