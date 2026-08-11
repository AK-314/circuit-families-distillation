# Workstream D — Orchestration, Hierarchical Analysis, and Reporting

## Mission

Run the frozen design safely at scale and ensure analysis follows the teacher-seed hierarchy rather than the physical job layout.

## Implementation order

1. Build a declarative DAG from the canonical condition registry.
2. Separate teacher-output, student-training, eligibility, discovery, endpoint, merge, and analysis nodes.
3. Give every job an isolated output root and atomic completion record.
4. Implement resumability without duplicate attempt IDs or budget transfer.
5. Build deterministic teacher-seed inventory merge and missing-job detection.
6. Generate synthetic hard/soft tables and figures before real endpoint data exist.
7. Implement teacher-direct records and robust student-cell summaries.
8. Implement within-teacher phase contrasts, direct teacher–student contrasts, and realization dispersion.
9. Implement separate hard and soft reports.
10. Add method-stratified reporting and resource-imperfect cross-method warnings.
11. Recompute endpoints from ledgers during QC.
12. Produce the final outcome-category table and analysis-freeze manifest.

## Production layout

```text
teacher seed
  ├── direct teacher evaluation for every selected phase
  ├── hard students for every phase and initialization
  └── soft students for every phase and initialization
        └── discovery-method jobs after eligibility
```

Physical workers may split lower-level jobs, but completion and analysis roll up through the teacher seed.

## Required analysis outputs

- All raw teacher-level values.
- All student attempts and eligibility outcomes.
- Student-cell median or other frozen summary.
- Within-cell range and median absolute deviation.
- Within-seed phase contrasts.
- Within-seed direct teacher–student contrasts.
- Seed-level sign consistency and full ranges.
- Method-specific budget consumption and failure classifications.
- Unresolved-cell table.
- Separate hard and soft figures.

## Acceptance gate

No report treats students, methods, thresholds, or circuits as population replicates; every displayed aggregate can be reconstructed from sealed teacher-seed inventories.

## Interfaces

- Consumes registries and schemas from A, eligibility records from B, and ledgers/endpoints from C.
- Controls production only after the final protocol freeze.
