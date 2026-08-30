# Circuit Families: Distilled Realizations

Research code and prospective protocol for testing whether grokking-associated
changes in sparse-circuit recoverability follow a teacher function, vary across
distilled realizations of that function, or arise from distillation and circuit
discovery procedures.

## Status

Technical implementation through Stage 10 is complete. Scientific red-teaming
has changed the proposed production design, but no definitive follow-up
training or endpoint-producing search has begun. Stages 11--14 now prepare and
freeze the amended design; Stage 15 is one automated production campaign.

## Start here

- `workflow.md`
- `docs/distillation_followup/post_red_team_protocol_amendment.md`
- `docs/distillation_followup/distillation_implementation_post_red_team.md`
- `docs/distillation_followup/red_team/red_team_resolution_matrix.md`
- `docs/distillation_followup/handoffs/post_red_team/`
- `docs/distillation_followup/distillation_experimental_protocol_draft.md`
- `docs/distillation_followup/distillation_implementation_master.md`
- `docs/distillation_followup/workstreams/`

The original protocol draft and implementation master are preserved Stage 2
authorities. For unexecuted work from Stage 11 onward, the post-red-team
amendment and implementation order take precedence. The older broad timeline
and two-person plan are historical planning documents, not the current
execution schedule.

## Predecessor

This repository begins from a clean source snapshot of the completed
`AK-314/circuit-families` study. It deliberately excludes the predecessor's
checkpoints, raw outputs, archives, result tables, figures, manifests, and Git
history. Reused scientific artifacts must be identified by immutable hashes in
the predecessor-link and teacher-registry records.

See `docs/predecessor_link.md` for the source commit and freeze hashes.

## Environment

- Python 3.11
- PyTorch
- TransformerLens 3.5.1
- `uv` dependency management

Install the locked environment:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

The default clean-repository run executes portable unit and synthetic-integration
tests. Predecessor artifact-audit tests are retained but skipped unless the
complete private predecessor checkpoint/result bundle is installed at its
original repository-relative paths.

Run lint checks:

```bash
uv run ruff check .
```

## Artifact rule

Git contains source, tests, configs, protocols, schemas, small manifests, and
reviewed summaries. Checkpoints, dense output caches, raw proposal/evaluation
ledgers, and archives belong in the external artifact store. Commit a manifest
only after the corresponding remote object and SHA-256 have been verified.
