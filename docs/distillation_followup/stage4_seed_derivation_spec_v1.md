# Stage 4 Seed Derivation Specification v1

This freezes the deterministic derivation algorithm, not the later roster of student or discovery jobs.

## Inputs

- namespace: `circuit-families-distillation`
- seed derivation version: `seed-derivation/v1`
- canonical `condition-identity/v1` ID, reused verbatim
- explicit purpose: `training`, `tie_breaking`, or `discovery`
- explicit non-negative `attempt_index`
- explicit non-negative `retry_index`

Training and tie-breaking use the complete depth-4 student condition identity. Discovery uses the complete depth-8 discovery condition identity. Attempt/retry coordinates are separate from student initialization and are always explicit.

## Exact material

The derivation material is ASCII (therefore byte-identical UTF-8), uses LF line endings, and includes a final LF:

```text
cfdseed:v1
namespace=circuit-families-distillation
seed_derivation_version=seed-derivation/v1
condition_id=<canonical-condition-id>
purpose=<canonical-purpose>
attempt_index=<canonical-uint>
retry_index=<canonical-uint>
```

No JSON serializer, Python object hash, machine path, username, timestamp, or process-specific value participates.

## Digest and integer extraction

- digest: SHA-256 over the exact material bytes;
- select digest bytes `0:8`;
- interpret those eight bytes as an unsigned big-endian 64-bit integer;
- final seed: `raw_u64 & 0x7FFFFFFFFFFFFFFF`;
- range: `0` through `9223372036854775807` inclusive;
- stored evidence includes derivation version, exact material, full SHA-256 hex digest, selected-byte hex, and final integer seed.

## Purpose depth contract

- `training`: exact depth 4;
- `tie_breaking`: exact depth 4;
- `discovery`: exact depth 8;

Any future stochastic purpose requires an explicit vocabulary/version extension rather than a free-form purpose string.

## Known independently calculable vectors

### `training_attempt0_retry0`

- condition ID: `cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0`
- purpose: `training`
- attempt index: `0`
- retry index: `0`
- material with LF shown as `\n`: `cfdseed:v1\nnamespace=circuit-families-distillation\nseed_derivation_version=seed-derivation/v1\ncondition_id=cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0\npurpose=training\nattempt_index=0\nretry_index=0\n`
- SHA-256: `15cc19f298db90994c9afffa9a2ad587a06e8dd59314e6cb0f690891efc7a308`
- selected bytes: `15cc19f298db9099`
- raw unsigned 64-bit integer: `1570658899782766745`
- final seed: `1570658899782766745`

### `tie_breaking_attempt0_retry0`

- condition ID: `cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0`
- purpose: `tie_breaking`
- attempt index: `0`
- retry index: `0`
- material with LF shown as `\n`: `cfdseed:v1\nnamespace=circuit-families-distillation\nseed_derivation_version=seed-derivation/v1\ncondition_id=cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0\npurpose=tie_breaking\nattempt_index=0\nretry_index=0\n`
- SHA-256: `3e170463ad6e72369f812bfa1d31218af07ec4ba682024c2ec23a1a1c497c77b`
- selected bytes: `3e170463ad6e7236`
- raw unsigned 64-bit integer: `4474049580973847094`
- final seed: `4474049580973847094`

### `training_attempt1_retry0`

- condition ID: `cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0`
- purpose: `training`
- attempt index: `1`
- retry index: `0`
- material with LF shown as `\n`: `cfdseed:v1\nnamespace=circuit-families-distillation\nseed_derivation_version=seed-derivation/v1\ncondition_id=cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0\npurpose=training\nattempt_index=1\nretry_index=0\n`
- SHA-256: `2959768e0af93b799323b51662331f5ac804ece9c57cc930103ebe1577a747e2`
- selected bytes: `2959768e0af93b79`
- raw unsigned 64-bit integer: `2979542980923833209`
- final seed: `2979542980923833209`

### `training_attempt0_retry1`

- condition ID: `cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0`
- purpose: `training`
- attempt index: `0`
- retry index: `1`
- material with LF shown as `\n`: `cfdseed:v1\nnamespace=circuit-families-distillation\nseed_derivation_version=seed-derivation/v1\ncondition_id=cfdid:v1:d4|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0\npurpose=training\nattempt_index=0\nretry_index=1\n`
- SHA-256: `d7ad8ab6d1304282fa6ae64e471ad152950a74e922cb06c64b60257d46c6fbb2`
- selected bytes: `d7ad8ab6d1304282`
- raw unsigned 64-bit integer: `15541230406923731586`
- final seed: `6317858370068955778`

### `discovery_attempt0_retry0`

- condition ID: `cfdid:v1:d8|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0|discovery_method=synthetic-method-a%2Fv1|fidelity_setting=synthetic-fidelity-a%2Fv1|component_cap=synthetic-cap-a%2Fv1|overlap_setting=synthetic-overlap-a%2Fv1`
- purpose: `discovery`
- attempt index: `0`
- retry index: `0`
- material with LF shown as `\n`: `cfdseed:v1\nnamespace=circuit-families-distillation\nseed_derivation_version=seed-derivation/v1\ncondition_id=cfdid:v1:d8|teacher_seed=1|phase=stable%20post-grokking|distillation_condition=hard_target|student_initialization=0|discovery_method=synthetic-method-a%2Fv1|fidelity_setting=synthetic-fidelity-a%2Fv1|component_cap=synthetic-cap-a%2Fv1|overlap_setting=synthetic-overlap-a%2Fv1\npurpose=discovery\nattempt_index=0\nretry_index=0\n`
- SHA-256: `8f2e9b6b4755bc8f5d5f956c694907fbacecfa0d6fe5a8cff911cf619f0ef88d`
- selected bytes: `8f2e9b6b4755bc8f`
- raw unsigned 64-bit integer: `10317354681412992143`
- final seed: `1093982644558216335`

These vectors require only the material above, SHA-256, big-endian integer conversion, and the stated 63-bit mask; they do not require project code.

## Scientific boundary

- UD-003–UD-014 remain unresolved.
- No optimizer, tolerance, fidelity threshold, component cap, overlap cutoff, discovery budget, or production roster is selected here.

PART_J_SEED_SPEC_STATUS: PASS
