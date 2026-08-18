# Stage 4 Common Vocabulary v1

This freezes structural vocabulary only. It does not freeze any numeric or scientific choice governed by UD-003–UD-014.

## Namespace and versions

- namespace: `circuit-families-distillation`
- vocabulary version: `common-vocabulary/v1`
- condition identity version: `condition-identity/v1`
- schema family version: `common-schema/v1`
- seed derivation version: `seed-derivation/v1`

## Canonical hierarchy

`teacher_seed / phase / distillation_condition / student_initialization / discovery_method / fidelity_setting / component_cap / overlap_setting`

## Stage 3 vocabulary preservation

- Stage 3 canonical phase order: `pre-grokking`, `50%`, `stable post-grokking`
- Stage 3 `phase_label` maps explicitly to common Stage 4 `phase`.
- Stage 3 `availability_status` maps explicitly to common Stage 4 `availability_state`.
- These upstream mappings are bridges only; Stage 4 common records do not silently accept alternate field spellings.

## Frozen enumerations

- distillation condition: `direct_teacher`, `hard_target`, `soft_target`
- record status: `draft`, `sealed`, `superseded`
- availability state: `planned`, `selected`, `unavailable`, `failed`, `missing`
- attempt outcome: `succeeded`, `failed`
- seed purpose: `training`, `tie_breaking`, `discovery`
- not-applicable token: `na`

## Prefix depths

- `teacher_seed` = `1`
- `teacher_phase` = `2`
- `distillation_condition` = `3`
- `student_initialization` = `4`
- `discovery_method` = `5`
- `fidelity_setting` = `6`
- `component_cap` = `7`
- `overlap_setting` = `8`

A record must use its record-type-specific required depth. Missing levels may not be silently omitted from a complete-width identity. The only frozen not-applicable spelling is `na`, and later cross-field validators must reject it wherever that hierarchy level is semantically required.

## Alias policy

Canonical values are exact and case-sensitive. Aliases are rejected rather than normalized silently. In particular, bare `hard` and `soft` are invalid distillation conditions; use `hard_target` and `soft_target`.

## Record types and schema versions

- `teacher_reference` → `teacher_reference/v1`
- `teacher_output_cache` → `teacher_output_cache/v1`
- `student_attempt` → `student_attempt/v1`
- `student_eligibility` → `student_eligibility/v1`
- `sealed_dense_model` → `sealed_dense_model/v1`
- `discovery_run` → `discovery_run/v1`
- `native_budget_ledger` → `native_budget_ledger/v1`
- `exact_mask_evaluation_ledger` → `exact_mask_evaluation_ledger/v1`
- `endpoint_record` → `endpoint_record/v1`
- `student_cell_summary` → `student_cell_summary/v1`
- `teacher_seed_inventory` → `teacher_seed_inventory/v1`
- `excluded_development_output` → `excluded_development_output/v1`
- `reproduction_comparison` → `reproduction_comparison/v1`
- `analysis_freeze` → `analysis_freeze/v1`

## Scientific-value boundary

- UD-003–UD-014 remain `unresolved`.
- Stage 4 may freeze typed/versioned references and placeholders.
- Stage 4 may not choose unresolved numeric values through schema defaults.

PART_F_VOCABULARY_STATUS: PASS
