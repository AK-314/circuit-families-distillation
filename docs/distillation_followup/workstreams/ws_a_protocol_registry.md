# Workstream A — Protocol, Registry, and Provenance

## Mission

Turn the follow-up protocol into machine-enforced condition identities, schemas, manifests, and freeze records while preserving the predecessor audit trail.

## Inputs

- Follow-up protocol draft and freeze register.
- Predecessor analysis-freeze commit.
- Existing teacher checkpoint manifests and component-basis constants.

## Implementation order

1. Create the follow-up namespace and predecessor-link manifest.
2. Build the canonical teacher registry with per-seed, function-defined checkpoint paths, hashes, phase labels, achieved accuracies, architecture hash, dataset hash, and predecessor commit. Reject any registry that labels a common step as a common phase without verifying the per-seed phase criterion.
3. Define condition IDs and deterministic seed derivation.
4. Define schemas for student attempts, eligibility, search jobs, exact evaluations, endpoints, cell summaries, inventories, and exclusions.
5. Add cross-field validators: hard/soft separation, eligible-before-search, identical 516-component basis, and no predecessor output roots.
6. Add protocol-freeze and amendment manifests.
7. Build inventory comparison and deterministic merge validation.
8. Freeze exact configs after the integration pilot.

## Deliverables

- Teacher registry.
- Versioned schemas and validators.
- Follow-up config loader.
- Freeze/amendment/exclusion manifests.
- Teacher-seed inventory format.
- Provenance and collision tests.

## Acceptance gate

Every planned artifact has one canonical identity, validates independently, traces to a teacher checkpoint, and cannot overwrite predecessor output.

## Interfaces

- Supplies IDs and schemas to Workstreams B–D.
- Receives no scientific endpoint values before the final freeze.
