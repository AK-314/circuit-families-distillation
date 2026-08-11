# Workstream B — Hard and Soft Distillation

## Mission

Create independently initialized student realizations of each selected teacher function, enforce full-domain eligibility, and preserve every attempt.

## Implementation order

1. Load and verify a teacher checkpoint from the canonical registry.
2. Cache final-position logits, per-input centred logits, probabilities if required, and argmax decisions over all 12,769 inputs.
3. Implement deterministic student initialization from the canonical condition ID.
4. Implement one shared training loop with condition-specific loss adapters.
5. Implement hard-target full-domain training and exact 12,769/12,769 eligibility.
6. Implement soft-target gauge-invariant loss and frozen tolerance evaluation.
7. Implement the frozen soft argmax eligibility rule.
8. Record all attempts, including numerical failures and eligibility failures.
9. Seal eligible student checkpoints, dense outputs, metrics, and hashes.
10. Add resume, reproduction, and attempted-initialization accounting tests.

## Required invariants

- Hard labels equal the teacher's cached argmax vector byte-for-byte.
- Adding any constant to every class logit for one input does not change centred-logit targets or soft loss.
- Eligibility is evaluated on all inputs, not a held-out sample.
- A student cannot be passed to circuit recovery before its eligibility record is sealed.
- Student circuit fidelity is later computed relative to that student's own sealed dense outputs.
- Failed attempts remain visible and count against the frozen attempt cap.

## Deliverables

- Teacher-target cache builder.
- Hard and soft loss adapters.
- Student trainer and checkpoint loader.
- Eligibility evaluator.
- Attempt registry writer.
- Reproduction and gauge-invariance tests.

## Acceptance gate

For a technical fixture, hard and soft attempts reproduce deterministically, eligibility can be recomputed from sealed outputs, and failed attempts cannot be silently replaced.

## Interfaces

- Consumes teacher registry and schemas from Workstream A.
- Supplies only sealed dense models and references to Workstream C.
- Supplies attempt and eligibility records to Workstream D.
