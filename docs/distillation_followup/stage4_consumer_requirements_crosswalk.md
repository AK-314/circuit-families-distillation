# Stage 4 Consumer Requirements Crosswalk

This document is a Stage 4 interface inventory. It freezes no unresolved scientific numeric value. Producer/consumer assignments describe the shared record contract only; UD-003–UD-014 remain unresolved at their later prescribed stages.

## Source fingerprints

- `handoff`: `followup/local/scratch/handoffs/alex_stage_04_common_contracts.md`; sha256 `957fa8d03162789975784b0bc576bd066c3347b2c14866aa09edfb846f685b93`; tracked `no`
- `austin`: `followup/local/scratch/handoffs/austin_stage_04_barrier0_review.md`; sha256 `e1d573f635238e0dc05aecfefa8e0e5b67262c40c630dee234b530f2420914cc`; tracked `no`
- `master`: `docs/distillation_followup/distillation_implementation_master.md`; sha256 `2aac7951e85cb26a2d081a368819965cf7b4110145fdf13a83abd802a4ffebd4`; tracked `yes`
- `protocol`: `docs/distillation_followup/distillation_experimental_protocol_draft.md`; sha256 `92441a8234d58273cca33b8f716c116c60015666628d55459a3843104a6e49be`; tracked `yes`
- `ws_b`: `docs/distillation_followup/workstreams/ws_b_distillation.md`; sha256 `f462583aa0b141b7934fc0fdd955e5c261b2426de77d0b9870fa6f1bc269fc6e`; tracked `yes`
- `ws_c`: `docs/distillation_followup/workstreams/ws_c_circuit_recovery.md`; sha256 `115fcd74736f0ef0b9033f1c33a3cfaa3684afb250b2644a512aeb980a733a67`; tracked `yes`
- `ws_d`: `docs/distillation_followup/workstreams/ws_d_orchestration_analysis.md`; sha256 `c04ca8617e803e2e7fe356f06d6ca85618ed6810c55728c7f0da76221441d94b`; tracked `yes`
- `workflow`: `workflow.md`; sha256 `9c2587b90ee0f278c1ed142da2e84e78617bf5c70956bbb76ec21b39d8e2f1c5`; tracked `yes`

## Canonical hierarchy

`teacher_seed / phase / distillation_condition / student_initialization / discovery_method / fidelity_setting / component_cap / overlap_setting`

Required identity depth is explicit per record type. Deeper levels are not silently omitted; later Stage 4 identity rules must encode validated prefix depth or the single frozen not-applicable representation.

## Shared schema crosswalk

| Schema | Producer | Consumer(s) | Creation stage | Required identity depth | Large-artifact storage class | Cross-field dependencies |
|---|---|---|---|---|---|---|
| teacher reference | Lane A / Stage 3 canonical registry bridge | Lane B; Lane C; Lane D | Stage 4 contract; instantiated from selected Stage 3 cells | teacher_seed/phase/distillation_condition | metadata-only; checkpoint by portable reference + SHA-256 | Stage 3 selected/unavailable state; checkpoint/source hashes; direct-teacher versus student condition distinction |
| teacher output cache | Lane B | Lane B; Lane C; Lane D | Stage 5B+ after Barrier 0 | teacher_seed/phase/distillation_condition | large object external; record stores portable reference + hash | selected teacher reference; deterministic ordering; hard/soft target identity separation; no unavailable cells |
| student attempt | Lane B | Lane B; Lane D | Stage 5B+ technical work; definitive roster later | teacher_seed/phase/distillation_condition/student_initialization | metadata record; model/log payloads external by reference + hash | teacher/cache identity; initialization identity; attempt/retry identity distinct from initialization; training/tie seed evidence; UD-003–UD-006 config references |
| student eligibility | Lane B | Lane C; Lane D | Stage 11 freeze / later eligible students | teacher_seed/phase/distillation_condition/student_initialization | small metadata/decision record | sealed student attempt; hard exact 12,769/12,769 teacher-argmax agreement; soft policy structurally representable but UD-006 numeric tolerance unresolved until Stage 11 |
| sealed dense model | Lane B | Lane C; Lane D | after passing eligibility | teacher_seed/phase/distillation_condition/student_initialization | checkpoint external; seal record stores portable reference + hash | passing sealed eligibility; frozen architecture identity; 516-component-basis identity; model hash |
| discovery run | Lane C | Lane C; Lane D | Stage 12+ after eligibility | teacher_seed/phase/distillation_condition/student_initialization/discovery_method/fidelity_setting/component_cap/overlap_setting | metadata record; search payloads external where large | sealed dense model; discovery method + method-budget version; fidelity-definition version; component-cap and overlap references; discovery seed |
| native budget ledger | Lane C | Lane C; Lane D | Stage 12+ discovery | teacher_seed/phase/distillation_condition/student_initialization/discovery_method/fidelity_setting/component_cap/overlap_setting | small append/sealed ledger metadata | discovery run; method-budget version; native optimization counts kept separate from exact mask evaluations |
| exact mask-evaluation ledger | Lane C | Lane C; Lane D | Stage 12+ exact evaluation | teacher_seed/phase/distillation_condition/student_initialization/discovery_method/fidelity_setting/component_cap/overlap_setting | ledger metadata; masks/results external if large, hashed | sealed dense model; fidelity-definition version; exact-evaluation allowance reference UD-010; exact versus native count separation; intact baseline |
| endpoint record | Lane C | Lane D | Stage 12+ from sealed exact ledger | teacher_seed/phase/distillation_condition/student_initialization/discovery_method/fidelity_setting/component_cap/overlap_setting | small sealed metadata record | sealed exact-evaluation ledger; Endpoint 1 in [0,1] and may equal 1.0 without global-minimum claim; Endpoint 2 >=0 and labelled packing lower bound |
| student-cell summary | Lane D | Lane D / primary analysis | Stage 13+ | teacher_seed/phase/distillation_condition | small summary metadata | teacher seed as population unit; hard/soft conditions never mixed; eligible/failed/missing attempt visibility; UD-011 summary rules |
| teacher-seed inventory | Lane D | Lane D; production orchestration/review | Stage 13+ inventory; contract frozen Stage 4 | teacher_seed/phase | small inventory metadata | all planned phase cells retained; selected/unavailable/failed/missing states explicit; seed-0 pre-grokking and 50% unavailable cells cannot be silently omitted |
| excluded development output | Lanes B/C/D as applicable | Lane D firewall/audit only | technical development before definitive production | teacher_seed/phase/distillation_condition | metadata + external payload references where needed | method-development firewall; explicit excluded/pilot lifecycle; must never become primary input |
| reproduction comparison | Lane D / reproducing lane | Lane D; Barrier/review audit | reproduction/verification stages | teacher_seed/phase/distillation_condition | small comparison metadata | source identity/hash and reproduced identity/hash both required; semantic comparison result; portable references |
| analysis freeze | Lane D / joint analysis freeze | Barrier review; definitive analysis | later analysis freeze, not production-ready in Stage 4 | teacher_seed/phase | small sealed metadata record | teacher seed population unit; firewall state; required upstream records; cannot claim production readiness while UD-003–UD-014 remain unresolved |

## Austin consumer/interface evidence

Austin's local Stage 4 review handoff is machine-local scratch and is not canonical project data. The following bounded line references were used only to ensure this crosswalk includes his Lane B/Lane D consumer review requirements:

- line 1: # HANDOFF — Stage 4-R: Barrier 0 consumer and cross-laptop validation
- line 5: authorize Austin to create competing schemas or begin Stage 5 implementation.
- line 9: - **Owner:** Austin / future Lane B and Lane D consumer
- line 18: `origin/feat/stage-04-common-identities-schemas`
- line 21: schemas, and synthetic artifact exchange are identical across both laptops
- line 25: Austin may immediately audit consumer requirements and the current authority.
- line 27: Lane B/D production code until the Stage 4 PR is jointly approved, merged, and
- line 42: review, but no canonical schema or implementation commit.
- line 48: - invent alternative field names, IDs, seed derivation, or schemas;
- line 57: ## Required consumer perspective
- line 59: Review whether the common contracts give future Lane B/D code enough structure
- line 87: ### Part C — Lane B/D consumer checklist
- line 91: Stage 4 structure from later numeric/scientific choices. Do not propose a schema.
- line 108: Write an ignored local memo under `followup/local/scratch/` containing consumer
- line 149: ### Part M — Teacher/cache/student schema review
- line 152: model contracts against Lane B needs. Require explicit failures, 12,769-domain
- line 170: ### Part P — Cross-schema DAG review
- line 172: Run valid and invalid graph bundles. Confirm references, versions, hashes,

## Contract conclusions

- One shared Stage 4 schema family is required; no lane-private parallel record format is permitted.
- Large arrays, checkpoints, caches, masks, and ledger payloads are not embedded in shared JSON records; records carry portable relative/object-store references and hashes.
- Teacher/cache/student records must preserve direct-teacher, hard-target, and soft-target identity separation.
- Student initialization identity is distinct from attempt/retry identity.
- Discovery records require method, method-budget, fidelity-definition, component-cap, and overlap-setting version references without freezing UD-007–UD-010 numeric choices.
- Teacher-seed inventories retain all 15 planned Stage 3 phase cells, including the two explicit seed-0 unavailable cells.
- Analysis uses teacher seed as the population unit.
- Excluded development outputs remain firewalled from primary inputs.
- Analysis-freeze records cannot claim production readiness while UD-003–UD-014 remain unresolved.

PART_D_CROSSWALK_STATUS: PASS
