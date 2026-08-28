# Pre-scientific-red-team gate

## Status

Stages 8--10 are technically complete at a linear local head. No Stage 11
configuration has been selected, no definitive production job has started,
and the scientific-method red-team has not begun.

This gate uses an internal exact-head double-check under the temporary
single-owner waiver. It is not an independent review and must not be described
as one.

## Stage 8: forced edge cases

All 14 prospectively listed cases pass through accepted Stage 5--7 interfaces:
one hard mismatch, both sides of the soft tolerance, no sparse circuit, zero
packing, method failure, exact-budget exhaustion, duplicate proposals, missing
and under-populated cells, resume, conflicting inventory, logit-gauge shift,
and result-order permutation.

The gauge-shift case preserves eligibility with a discrepancy difference of
approximately `1.4e-15`. Byte hashes of re-centred floating-point arrays are
not treated as gauge-invariant after arbitrary offsets; the scientific
eligibility decision is tolerance-based.

## Stage 9: student training and backend qualification

The complete 12,769-input domain and the frozen 227,313-parameter architecture
were used for two hard and two soft one-update repetitions per backend.

- CPU: exact checkpoint and dense-output reproduction in both conditions.
- MPS: not byte-reproducible; differences were confined to `embed.W_E`, with
  relative parameter L2 drift around `1.4e-9` to `1.7e-8`, dense-output drift
  around `3.3e-7` to `3.6e-7`, and `0/12,769` argmax disagreements.
- Therefore CPU is the verified deterministic reference candidate. MPS is
  development-only for definitive training. CUDA remains unmeasured and must
  pass the same conformance test before use.
- One final checkpoint plus dense output is about 8.5 MB per attempt; 180
  attempts are about 1.53 GB. Retaining legacy checkpoints every 50 updates
  would instead approach 99 GB at 10,000 updates or 393 GB at 40,000 updates,
  so checkpoint retention must be deliberately frozen later.

These measurements do not freeze an optimizer, schedule, stopping rule,
training length, attempt cap, eligibility tolerance, or production backend.

## Stage 10: discovery and exact-evaluation mechanics

One physically verified registered teacher was used without recording endpoint
values.

- One exact full-domain mask evaluation: about 0.088 seconds steady-state.
- One complete 516-component ranking pass: about 0.131 seconds steady-state.
- Greedy technical fixture: about 1.00 second steady-state.
- Diversity-forced technical fixture: about 1.18 seconds steady-state.
- Both adapters reproduced identical evidence hashes across two runs.
- Measured peak resident memory was about 518 MB; the default remains one
  worker until production-scale monitoring justifies more.

The technical budgets are deliberately tiny and are not production budgets.
Method-native units remain non-equivalent. Endpoint threshold, component-cap,
overlap, and packing sensitivity can reuse masks already in a sealed ledger;
a threshold that changes the discovery trajectory may require a new search.

## Combined verification

- Full repository: 1,383 passed, 270 expected private-artifact skips.
- Affected exact-head suite from a detached worktree: 127 passed.
- Stage 8 executable matrix: 14/14 passed.
- Repository-wide Ruff and diff hygiene: passed.
- Machine reports are hash-bound to the exact benchmark modules.
- Tracked changed surface contains no private path, checkpoint, bulk array,
  archive, Git LFS object, scientific conclusion, or production default.

## Decisions reserved for scientific red-teaming and later freeze

- UD-003: student architecture and initialization.
- UD-004: replication, attempt cap, replacement, and minimum eligibility.
- UD-005: hard-target training configuration.
- UD-006: soft target, loss, temperature, training, and eligibility.
- UD-007: fidelity implementation details, threshold, and precision.
- UD-008: component cap, overlap rule, and cutoff.
- UD-009: discovery roster, versions, and native budgets.
- UD-010: common exact-evaluation allowance.
- UD-011: phase contrasts, cell summary, and missing-cell rule.
- UD-012: analysis tables, figures, and manifests.
- UD-013: production concurrency, isolation, and merge rule.
- UD-014: definitive production scope.

The next activity is scientific-method red-teaming. Stage 11--14 freeze work
must wait until critiques are classified, resolved, and recorded.
