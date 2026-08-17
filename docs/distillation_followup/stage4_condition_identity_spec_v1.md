# Stage 4 Condition Identity Specification v1

This specification freezes the identity grammar only. The production builder/parser is implemented in Part H.

## Wire format

`cfdid:v1:d<depth>|<field1>=<canonical-value>|...|<fieldN>=<canonical-value>`

The explicit depth is part of the ID. A prefix identity therefore cannot collide with a deeper identity that happens to begin with the same fields.

## Canonical hierarchy

`teacher_seed / phase / distillation_condition / student_initialization / discovery_method / fidelity_setting / component_cap / overlap_setting`

## Serialization

- fields appear strictly in canonical hierarchy order;
- the number of serialized fields equals the explicit depth;
- strings must already be Unicode NFC;
- strings serialize as UTF-8 with RFC3986 percent-encoding;
- percent escapes use uppercase hexadecimal;
- only ASCII letters, digits, `-`, `.`, `_`, and `~` remain literal;
- integer components are unsigned base-10 with no sign or leading zeros;
- empty values are forbidden;
- reordered, missing, or extra fields are invalid;
- `na` never substitutes for an omitted hierarchy suffix inside a canonical condition ID; validated prefix depth handles earlier artifacts.

## Required record depths

- `teacher_reference`: depth `3`
- `teacher_output_cache`: depth `3`
- `student_attempt`: depth `4`
- `student_eligibility`: depth `4`
- `sealed_dense_model`: depth `4`
- `discovery_run`: depth `8`
- `native_budget_ledger`: depth `8`
- `exact_mask_evaluation_ledger`: depth `8`
- `endpoint_record`: depth `8`
- `student_cell_summary`: depth `3`
- `teacher_seed_inventory`: depth `2`
- `excluded_development_output`: depth `3`
- `reproduction_comparison`: depth `3`
- `analysis_freeze`: depth `2`

## Availability boundary

- all 15 planned teacher-phase cells have valid depth-2 identities;
- the two unavailable seed-0 cells stop at depth 2;
- any depth >=3 identity requires a selected Stage 3 cell;
- no cache, student, dense-model, discovery, ledger, or endpoint identity can therefore descend from either unavailable cell.

## Condition rules

- depth 3 distinguishes `direct_teacher`, `hard_target`, and `soft_target`;
- student initialization is a non-negative integer;
- attempt/retry identity is intentionally not part of the canonical condition ID;
- depth >=4 forbids `direct_teacher` and requires `hard_target` or `soft_target`;
- discovery method, fidelity setting, component cap, and overlap setting are typed version references matching `^[a-z][a-z0-9._-]*/v[1-9][0-9]*$`;
- raw numeric values at those later hierarchy levels are not valid Stage 4 identities.

## Synthetic uniqueness vectors

- `teacher_phase` → `cfdid:v1:d2|teacher_seed=1|phase=stable%20post-grokking`
- `direct_teacher` → `cfdid:v1:d3|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=direct_teacher`
- `hard_condition` → `cfdid:v1:d3|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target`
- `soft_condition` → `cfdid:v1:d3|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=soft_target`
- `hard_init_0` → `cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0`
- `hard_init_1` → `cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=1`
- `complete_a` → `cfdid:v1:d8|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0|discovery_method=synthetic-method-a%2Fv1|fidelity_setting=synthetic-fidelity-a%2Fv1|component_cap=synthetic-cap-a%2Fv1|overlap_setting=synthetic-overlap-a%2Fv1`
- `complete_method_changed` → `cfdid:v1:d8|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0|discovery_method=synthetic-method-b%2Fv1|fidelity_setting=synthetic-fidelity-a%2Fv1|component_cap=synthetic-cap-a%2Fv1|overlap_setting=synthetic-overlap-a%2Fv1`
- `complete_fidelity_changed` → `cfdid:v1:d8|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0|discovery_method=synthetic-method-a%2Fv1|fidelity_setting=synthetic-fidelity-b%2Fv1|component_cap=synthetic-cap-a%2Fv1|overlap_setting=synthetic-overlap-a%2Fv1`
- `complete_cap_changed` → `cfdid:v1:d8|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0|discovery_method=synthetic-method-a%2Fv1|fidelity_setting=synthetic-fidelity-a%2Fv1|component_cap=synthetic-cap-b%2Fv1|overlap_setting=synthetic-overlap-a%2Fv1`
- `complete_overlap_changed` → `cfdid:v1:d8|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0|discovery_method=synthetic-method-a%2Fv1|fidelity_setting=synthetic-fidelity-a%2Fv1|component_cap=synthetic-cap-a%2Fv1|overlap_setting=synthetic-overlap-b%2Fv1`

Every vector above is synthetic and non-scientific. Changing condition type, student initialization, discovery method, fidelity reference, component-cap reference, or overlap reference changes the canonical ID.

## Unresolved-value boundary

- UD-003–UD-014 remain unresolved.
- The synthetic `/v1` references above demonstrate identity structure only and are not frozen scientific settings.

PART_G_IDENTITY_SPEC_STATUS: PASS
