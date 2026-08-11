# Workstream E — Secondary Fourier Interchange

## Status

Gated. This workstream is not on the core implementation critical path and must not consume resources needed for complete primary coverage.

## Mission

Test whether aligned teacher–student Fourier-state interchange supports a shared causal abstraction under matched information capacity.

## Preconditions

- Core protocol frozen.
- Teacher/student pair-selection rule frozen without interchange outcomes.
- Fourier mode identification and alignment algorithm frozen.
- Patched state location, scale, and information-capacity matching frozen.
- All five controls implemented before aligned results are inspected.

## Implementation order

1. Define the causal estimand and intervention location.
2. Implement Fourier coefficient extraction and cross-model alignment.
3. Define the matched information-capacity budget.
4. Implement aligned interchange.
5. Implement wrong-mode, shuffled-coefficient, mismatched-input, equal-norm-random-state, and unaligned-activation controls.
6. Validate norm, dimensionality, input identity, and intervention bookkeeping.
7. Freeze pair sampling, trial count, outcome metric, and comparison rule.
8. Execute controls and aligned interchange under one immutable manifest.
9. Report aligned performance against every control without uniqueness claims.

## Acceptance gate

An aligned-interchange claim is permitted only if it outperforms every prespecified control under the same frozen information-capacity rule.
