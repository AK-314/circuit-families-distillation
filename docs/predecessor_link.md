# Predecessor Link

This follow-up reuses code and selected scientific artifacts from the completed
`circuit-families` predecessor while preserving that study as an immutable,
separate record.

## Canonical machine-readable record

- Schema: `schemas/predecessor_link_v1.schema.json`
- Manifest: `followup/manifests/predecessor_link_v1.json`
- Namespace: `circuit-families-distillation/v1`

## Source

- Repository: `https://github.com/AK-314/circuit-families`
- Analysis-freeze commit: `a55509537a70a225fedc5ce3a1c8236110974a6e`
- Analysis-freeze manifest: `manifests/stage22_freeze_stage22-freeze-34241335dcf7.json`
- Analysis-freeze manifest SHA-256: `248a111f7c328fac81ebe5be5cbd45582487635d43e31848c6a0cd6f06a29f30`
- Frozen predecessor protocol SHA-256: `39a2852052a2e6c0e28f722d8c644be2a6897b2444f969342789385d048b47a7`
- Frozen predecessor implementation order SHA-256: `416e40146eee8aaf8a42a8e6c28b1e9219c33cdc90f46b01789ce5903fed1e34`

## Snapshot scope

The successor source snapshot is recorded at
`2c261552b0553e29cdf1b07544371cf34129a799` in `https://github.com/AK-314/circuit-families-distillation`.

Across the audited `src/`, `tests/`, `scripts/`, and `configs/` snapshot,
166 files overlap with the predecessor and all 166 are byte-identical. The
successor adds its portable-test configuration without importing predecessor
scientific outputs.

The successor deliberately excludes predecessor:

- model checkpoints;
- raw scientific results;
- compressed scientific archives;
- result tables and figures;
- predecessor manifests as local successor artifacts; and
- predecessor Git history.

The predecessor-link manifest records the reused dataset, architecture,
component/masking definitions, and the five known teacher training-run
identities for seeds 0--4.

The definitive per-teacher checkpoint registry is **not** selected here.
Phase-selection outcomes, selected steps and paths, checkpoint SHA-256 values,
achieved metrics, and dense-output identities remain explicitly deferred to
Stage 3.

No absolute private filesystem path is a canonical artifact identity.
