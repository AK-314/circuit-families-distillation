# Austin 4 — Stage 12-P4 compact artifacts, quotas, and verified export

## Paste this entire document into one fresh Chat-mode task

Repository: `AK-314/circuit-families-distillation`

Local clone convention: `~/Projects/circuit-families-distillation`

Required Stage 12-P3 merge/base:

```text
cc707edc965d6d01c6c187c50c2e021019ac0d3b
```

Required Stage 12-P3 implementation head:

```text
955d357609f8916797d2ea929f924e8f120e2334
```

Required branch:

```text
feat/stage-12p4-compact-storage-export
```

Create the branch from current `origin/main` after this handoff is merged. Part
A must prove that both required SHAs are ancestors of `origin/main`, that PR
#25 is merged, and that the Stage 12-P3 logical-job, attempt-root, and sealed
manifest interfaces are present. Record the current `origin/main` SHA as the
implementation base.

## Mission

Implement the compact artifact and transfer layer needed by the future
production campaign without selecting production quotas or touching registered
scientific artifacts. The package must provide:

1. compact, schema-bound mask and metric ledgers;
2. bounded rolling-checkpoint retention and explicit quota accounting;
3. deterministic shard merge and inventory sealing;
4. deterministic compression and export bundles;
5. resumable copy with destination-side hash verification;
6. rejection or explicit accounting of partial, stale, duplicate, and
   conflicting objects;
7. a measured technical size report on synthetic fixtures.

The work exists to prevent the predecessor's verbose many-file storage pattern.
It must integrate with Stage 12-P3 job identities and sealed outputs while
remaining independent of any particular scheduler, object-storage provider, or
production quota. RD-014 remains open for Stages 13–14.

## Authorities — read completely before acting

1. `docs/distillation_followup/stage11_post_red_team_design_resolution.md`
2. `followup/configs/stage11_post_red_team_design_candidates_v1.json`
3. `followup/manifests/stage11_red_team_resolution_v1.json`
4. `followup/configs/post_red_team_open_decisions_v1.json`
5. `docs/distillation_followup/post_red_team_protocol_amendment.md`
6. `docs/distillation_followup/distillation_implementation_post_red_team.md`
7. `docs/distillation_followup/compute_and_two_person_collaboration.md`
8. merged Stage 12-P3 contracts, records, state store, adapters, CLI, and tests;
9. Stage 4 canonical identities, paths, hashes, schemas, and seed contracts;
10. Stage 5B–5C cache, job-output, atomic publication, and DAG contracts;
11. Stage 6A exact-evaluation ledger and Stage 6E packing record/proof contracts;
12. Stage 7 lifecycle, reproduction, inventory, and failure records;
13. merged Stage 12-P1/P2 checkpoint, dense-output, sealing, and eligibility
    interfaces;
14. merged Stage 12-R1/R2 compact proposal, exact-ledger, mask, component-basis,
    and failure records where available;
15. `docs/distillation_followup/handoffs/post_red_team/handoff_sequence.md`.

Reuse canonical serializers, identity fields, masks, ledgers, manifests, and
failure taxonomies. Do not create a second scientific record system disguised
as a storage format.

## Scientific and operational boundary

Permitted:

- synthetic masks, metrics, checkpoints, ledgers, manifests, and job outputs;
- technical byte counts, compression ratios, file counts, throughput, checksum,
  interruption, resume, and quota diagnostics;
- injected technical quota, retention, compression, chunk, and transfer
  profiles;
- local-filesystem source and destination adapters;
- deterministic corrupt/partial/stale/conflicting fixtures;
- compact technical reports with no comparative scientific endpoints.

Prohibited:

- loading or evaluating registered teacher/student checkpoints;
- copying private predecessor artifacts into the repository or test fixtures;
- selecting production scratch/persistent quotas, retention cadence, codec,
  object store, credentials, or Symbolica paths;
- deleting source artifacts after export;
- treating scheduler completion as sealed storage success;
- silently overwriting a destination object with a different hash;
- placing large outputs, checkpoints, archives, credentials, or LFS objects in
  Git;
- resolving RD-014 or beginning Fourier execution, Stage 13, Stage 14, or
  Stage 15.

Every executable profile, manifest, inventory, size report, and validation
result must state `scientific_data=false` and `production_eligible=false`.

## Chat protocol

- Stay in Chat mode for the entire handoff; do not mix Chat and Work modes.
- Complete Parts A–H in order. Use extra one-block turns inside a part only when
  a real failure or diagnostic branch requires them.
- Every operational response contains exactly one fenced terminal block.
- Briefly state what that block changes or inspects and which diagnostics it
  prints.
- Austin returns complete stdout/stderr before the next block.
- Never emit only a status footer when the next block is available.
- Never claim to await output from a block that was not supplied.
- Use focused tests while implementing and one broad exact-SHA double-check at
  the end. Do not repeatedly run the full historical suite.
- Preserve user-owned untracked files and unrelated active work.
- End every response with:

```text
HANDOFF=AUSTIN_04_STAGE12P4
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact implementation base recorded in Part A>
HEAD=<exact current SHA>
WAITING_FOR=<NONE or exact blocker>
SCIENTIFIC_DATA=NO
```

## Required design separations

Keep these objects distinct:

1. **Logical scientific/technical record** — canonical content and identity.
2. **Storage encoding** — bit packing, row encoding, compression, and chunks.
3. **Scratch inventory** — what exists locally, including partial attempts.
4. **Sealed artifact manifest** — exact accepted files, sizes, and hashes.
5. **Export transfer state** — resumable progress, destination, attempts, and
   verification.
6. **Destination verification** — independently reread bytes and compare the
   sealed manifest.

Compression, path, transfer attempt, and destination metadata must not change
logical job, circuit, mask, metric, or endpoint identity.

## Part A — Exact base, ancestry, and scope guard

The first block is read-only. It must print and verify:

- repository root, remote, branch, HEAD, local `main`, and `origin/main`;
- exact ancestry of `cc707edc...` and `955d3576...`;
- PR #25 merged state and exact head;
- Stage 12-P3 logical job IDs, attempt roots, output contracts, state store,
  sealed manifests, and operational status interfaces;
- relevant Stage 4/5B–5C/6A/6E/7 and Stage 12-P1/P2/R1/R2 storage consumers;
- Stage 11 authority hashes and open RD-014 status;
- tracked cleanliness and separately listed untracked files;
- absence of Stage 12-P4, Fourier-runner, Stage 13, Stage 14, and Stage 15 work;
- available standard-library and installed compression/checksum capabilities;
- no registered/private-artifact dependency or active scientific process.

After diagnosing that output, a second block may create the required branch.

**Part A passes when:** the branch starts from shared main containing P3, the
tracked state is clean, unrelated files are preserved, and no scientific or
private artifact has been accessed.

## Part B — Reuse audit and versioned compact-storage contract

Map every producer and consumer before implementing:

- P3 logical job, attempt, expected-output, sealing, and reconciliation records;
- P1/P2 checkpoint, dense-output, eligibility, and release records;
- R1 proposals, masks, exact-evaluation ledgers, failures, and trajectories;
- R2 basis/component identities and parameter-weighted accounting;
- Stage 6A/6E endpoint and proof records;
- Stage 7 inventory/reproduction status;
- Stage 4 canonical paths, hashes, JSON records, and identity envelopes.

Define one versioned outer storage contract covering:

- artifact class and producer interface version;
- logical job/attempt/output-contract references;
- schema and ordered-field identity;
- source byte length and SHA-256;
- storage encoding, codec, chunking, and deterministic parameters;
- compact-object byte length and SHA-256;
- scratch quota and retention profile references;
- lifecycle state: planned, partial, complete, sealed, exporting, exported,
  verified, conflict, or failed;
- explicit scientific and production boundaries.

All numeric values are injected technical profiles. Do not freeze production
values. Reject unknown fields where the canonical contract requires closure.

Adversarial tests must reject cross-job output reuse, schema drift, hash/length
mismatch, unsafe paths, missing producer evidence, encoding ambiguity,
scientific payload relabeling, and `production_eligible=true`.

**Part B passes when:** storage metadata wraps canonical records without
changing their scientific identity or duplicating their semantics.

## Part C — Compact masks and metric ledgers

Implement deterministic compact writers/readers for the high-volume record
families needed by discovery and analysis:

- bit-packed masks bound to an ordered component universe, basis identity,
  component count, padding rule, and mask SHA-256;
- appendable metric/exact-evaluation rows with an explicit closed schema,
  canonical ordering, numeric representation, row count, and content hash;
- compact failure/unavailable/censored rows rather than missing records;
- streaming operation so complete data need not be held in RAM;
- deterministic finalization into a sealed object and manifest.

Use a repository-supported, portable codec. If several codecs are exposed,
keep selection injected and test each supported encoding. Do not add a heavy
dependency merely to call output “columnar.” JSONL/CSV plus deterministic
compression is acceptable when it is measurably compact and schema-bound.

Tests must cover mask round trips across byte boundaries, all-zero/all-one and
empty masks where valid, padding corruption, universe/basis mismatch, negative
and nonfinite metric policy, row-order determinism, interrupted finalization,
large streaming fixtures, and legacy-record reconstruction.

**Part C passes when:** compact objects round-trip exactly to canonical logical
records, are deterministic across working directories/hash seeds, and retain
failure/null/censored states.

## Part D — Rolling retention and quota enforcement

Implement a policy-neutral scratch manager that accepts injected:

- per-job hard quota and warning threshold;
- reserved bytes required for atomic finalization;
- checkpoint cadence and maximum retained generations;
- protected artifact classes that cannot be evicted;
- temporary/partial cleanup eligibility;
- failure behavior when safe completion cannot fit.

Integrate through adapters rather than rewriting P1/P2/R1 checkpoint semantics.
Quota accounting must include partial files, compression staging, manifests,
and atomic-rename reserve. A quota failure must produce an explicit terminal or
retryable operational record; it must never truncate a sealed object or delete
the only valid checkpoint.

Rolling retention must be deterministic, preserve the newest valid bounded set
plus declared protected generations, and record every deletion decision. Tests
must include exact-boundary bytes, insufficient finalization reserve, stale
partials, interrupted cleanup, protected generations, corrupted newest
checkpoint, concurrent claim rejection, and restart reconciliation.

**Part D passes when:** technical fixtures remain within injected quotas or
fail explicitly while always retaining a valid recovery boundary.

## Part E — Deterministic merge, deduplication, and conflict handling

Implement serial deterministic merge for sealed shard objects. The merger must:

- require compatible schema, logical campaign, basis, and producer versions;
- order shards and rows by declared canonical keys independent of arrival time;
- verify every source hash before consumption;
- distinguish byte-identical duplicates from semantic-key conflicts;
- apply only a declared deduplication rule and record duplicate provenance;
- reject conflicting content for the same unique key;
- preserve failed, unavailable, censored, and empty shards in inventory;
- emit a sealed merged manifest with counts, hashes, source closure, and
  recomputation evidence;
- support deterministic restart after an interrupted merge.

Never resolve a conflict using file modification time, scheduler completion
order, “latest wins,” or effect direction. Tests must shuffle arrival order,
inject duplicate and conflicting rows, omit shards, corrupt hashes, repeat
resume, and compare output bytes across hash seeds.

**Part E passes when:** identical logical shard sets yield identical merged
bytes and manifests, and every inconsistency becomes an auditable failure.

## Part F — Deterministic bundle, resumable export, and destination verification

Implement a provider-neutral export interface with a complete local-filesystem
adapter. It must:

- build a deterministic bundle/inventory without timestamps, host paths, uid,
  gid, or filesystem-order leakage;
- compress using fixed, recorded deterministic parameters;
- split into deterministic chunks when the injected profile requests it;
- stage destination objects under partial identities;
- resume only after verifying already copied bytes/chunks;
- atomically publish the destination manifest last;
- independently reopen destination objects and verify sizes and SHA-256 hashes;
- detect truncated, stale, extra, duplicate, and conflicting destination
  objects;
- refuse silent overwrite when a destination identity has different content;
- retain source artifacts after successful verification.

Transfer state and destination paths remain outside scientific and logical job
identity. No cloud SDK, credentials, or provider-specific policy is required.

Tests must force interruption at multiple byte/chunk boundaries, corrupted
partial files, stale transfer state, already-complete destinations, source
mutation after planning, conflicting published objects, extra objects,
read-after-write failure, and repeated idempotent verification.

**Part F passes when:** a stopped export can resume to byte-identical verified
destination content without deleting or mutating the source.

## Part G — Integrated portable validation and measured size report

Build one validate-only CLI using deterministic synthetic fixtures that runs:

1. compact mask and metric-ledger creation;
2. rolling checkpoints under a small injected quota;
3. one quota-warning and one explicit quota-failure path;
4. deterministic shard merge with duplicate accounting;
5. interrupted export and resume;
6. destination-side verification;
7. corrupt/partial/stale/conflicting-object rejection;
8. reconstruction of canonical records from compact outputs;
9. a measured verbose-fixture-versus-compact size/file-count report.

The size report must state fixture identities, logical row/mask counts, raw and
compact bytes, file counts, peak streaming buffer where measurable, and
compression ratio. It is technical evidence only and must not claim the final
production footprint or freeze a quota/codec.

Run from repository root and an unrelated cwd under at least two
`PYTHONHASHSEED` values. Canonical reports and exported bundle hashes must be
identical. Run:

- Stage 12-P4 focused/adversarial tests;
- Stage 12-P3 identity/state/output/adapter compatibility;
- Stage 12-P1/P2 checkpoint, sealing, and failure compatibility;
- Stage 12-R1/R2 mask/ledger/basis compatibility;
- Stage 4 path/hash/schema compatibility;
- Stage 5B–5C output-publication and DAG compatibility;
- Stage 6A/6E ledger/endpoint/proof compatibility;
- Stage 7 lifecycle/inventory/reproduction compatibility;
- Ruff on changed Python;
- diff, private-path, secret, large-file, binary, checkpoint, archive, and LFS
  hygiene.

The validate-only command may write only beneath an explicit temporary/output
root. It must never contact a network service or inspect registered artifacts.

**Part G passes when:** the complete compact–quota–merge–export lifecycle is
portable, deterministic, interruption-safe, and measurably bounded on technical
fixtures.

## Part H — Commit, exact-SHA double-check, PR, and stop

Inspect the changed surface and ensure it is limited to Stage 12-P4 code,
tests/validation, and necessary technical documentation. Create coherent
commits without amend or force-push. Push and open a PR against `main`.

At the final exact SHA, run a fresh detached-checkout double-check covering
focused/adversarial and compatibility tests, root/unrelated-cwd CLI execution,
multiple hash seeds, deterministic bundle/report hashes, forced interruption
and resume, quota boundaries, conflict detection, Ruff, diff, tracked
cleanliness, artifact sizes, and Git/LFS surface. Classify findings as blocking,
nonblocking, or question. Repair blockers only through descendant commits and
repeat against the new exact SHA.

Do not merge inside this handoff without master-task authorization.

Final report:

- base, branch, parent, final SHA, and PR;
- exact changed files and artifact sizes;
- compact mask/ledger formats and canonical round-trip evidence;
- quota, reserve, retention, and recovery-boundary behavior;
- deterministic merge/deduplication/conflict evidence;
- bundle, interruption/resume, and independent destination verification;
- measured technical size/file-count report and limitations;
- P1/P2/P3, R1/R2, Stage 4/5B–C/6A/6E/7 compatibility totals;
- deterministic CLI, report, and bundle hashes;
- unresolved RD-014 production choices;
- internal findings and descendant repairs;
- confirmation of no registered/private/scientific execution or Git/LFS
  artifact upload;
- interfaces exported to Austin 5/6, Alex 5/6, and Stages 14–15;
- explicit stop before Fourier execution, Stage 13, Stage 14, and Stage 15.

Final status:

```text
AUSTIN_04_STAGE12P4_STATUS=COMPLETE_AT_HANDOFF_GATE
SCIENTIFIC_DATA=NO
PRODUCTION_STORAGE_PROFILE_SELECTED=NO
PRODUCTION_EXPORT_DESTINATION_SELECTED=NO
RD_014_RESOLVED=NO
STAGE15_STARTED=NO
```

## Prohibited shortcuts

- Do not put retry, worker, backend, path, codec, or transfer-attempt metadata
  into logical scientific identity.
- Do not serialize one verbose JSON document per metric step or mask.
- Do not assume scheduler completion means sealed or exported success.
- Do not evict the only valid checkpoint or any declared protected artifact.
- Do not use “latest wins” or filesystem order to resolve conflicts.
- Do not call an unverified copy exported.
- Do not delete source data after destination verification.
- Do not embed absolute private paths, credentials, large fixtures, archives,
  checkpoints, or LFS objects in Git.
- Do not freeze RD-014 values from synthetic size ratios.
- Do not begin Austin 5 Fourier work, Stage 13, Stage 14, or Stage 15.
