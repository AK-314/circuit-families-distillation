# Austin 6 — Stage 14-B cluster package, qualification, and exact-SHA rehearsal

## Paste this entire document into one fresh Chat-mode task

Repository: `AK-314/circuit-families-distillation`

Local clone convention: `~/Projects/circuit-families-distillation`

Required Stage 13 merge/base:

```text
015e67a60db64e41713f8493d5394ce90c129e69
```

Required Stage 13 implementation head:

```text
51fc2147ff4e123ebbfcabc4206520ca72f8e24f
```

Required Stage 13 bindings:

```text
approved package:
stage13-package-a-protected-core-optional-five-task/v3

approval record SHA-256:
acd34de34ee94f943ddfb2d088572196d74c5bde9eae8474e691b5f8ef89d336

campaign root SHA-256:
b369c35de0af04afc33f9ed1777fbf0db1542a32b0a3e0c830d25d361d6c66a6

expanded canonical members SHA-256:
adbfb30694bb984de4d8ba582cee0efb468b8f9a2fce01f6a3654b5b78b1927b

logical jobs:
8745 total = 7884 protected + 287 Task 3 + 287 Task 4 + 287 Task 5
```

Required branch:

```text
feat/stage-14b-cluster-package-rehearsal
```

Create the branch from current `origin/main` after this handoff is merged. Part
A must prove that the Stage 13 merge and implementation head are ancestors of
`origin/main`, verify every frozen hash above from physical bytes, and record
the then-current `origin/main` SHA as the implementation base.

## Current resource facts — do not improve or invent them

- Alex intends to request approximately 96 continuous hours on Symbolica.
- The actual Symbolica scheduler, queue, GPUs, CPUs, RAM, storage, quotas,
  container support, paths, credentials, network, and grant are unknown.
- Eton approval is pending. The number, model, memory, storage, availability,
  administrative restrictions, and MPS suitability of school Macs are unknown.
- Symbolica is the intended protected-core backend if granted and qualified.
- Qualified Eton Macs are intended primarily for optional Tasks 3–5 and
  overflow/reproduction. They may become a protected-core fallback only if the
  complete frozen feasibility equation passes on the measured available pool.
- If no verified resource configuration fits the protected core, launch is
  blocked. Austin must not shrink the protocol or activate a salvage package.

## Mission

Turn the frozen Stage 13 campaign into a provider-neutral, reproducible,
operator-safe execution package and prove the complete machinery on a reduced
non-scientific rehearsal. Austin 6 must provide:

1. a locked environment/container specification and build verification;
2. canonical input-bundle staging and independent hash verification;
3. hardware/resource inventory plus CPU, CUDA, and optional MPS qualification;
4. scheduler-neutral job-array planning with local and Slurm-class adapters;
5. an interruption-tolerant school-Mac worker package without inbound services;
6. one-command qualify, plan, dry-run, status, pause, stop, resume, audit,
   recompute, compact, export, and verify operations;
7. monitoring that exposes completeness, failures, resources, storage, and
   integrity without exposing comparative scientific direction;
8. the frozen 12-hour final-window reservation and three human-gate machinery;
9. a complete reduced synthetic DAG rehearsal with forced failures and resume;
10. an exact-SHA internal double-check and a resource-binding report for Alex 6.

This task builds and rehearses the launch package. It does not authorize or
start Stage 15, and it must make accidental production launch impossible while
the Alex 6 authorization artifact is absent.

## Authorities — read completely before acting

1. `docs/distillation_followup/stage13_protocol_manifest_freeze.md`
2. `followup/decisions/stage13_approval_v1.json`
3. `followup/configs/stage13/frozen_scientific_protocol_v1.json`
4. `followup/configs/stage13/production_profiles_v1.json`
5. `followup/configs/stage13/analysis_report_plan_v1.json`
6. `followup/manifests/stage13_campaign_root_v1.json`
7. `followup/manifests/stage13_job_array_spec_v1.json`
8. `followup/manifests/stage13_expanded_manifest_seal_v1.json`
9. `followup/manifests/stage13_scope_resource_projection_v3.json`
10. `followup/manifests/stage13_excluded_evidence_v1.json`
11. all three `followup/manifests/stage13_optional_tasks/` increments;
12. `scripts/materialize_stage13_manifest.py` and
    `scripts/validate_stage13_freeze.py`;
13. merged Stage 12-P3 DAG/state/claim/retry/scheduler interfaces;
14. merged Stage 12-P4 compact storage/quota/merge/export interfaces;
15. merged Stage 12-P1/P2/R1/R2/R3/P5 producer, discovery, calibration, basis,
    Fourier, result, and failure interfaces;
16. Stage 4 identities/seeds/hashes and Stage 5–7 lifecycle/reproduction
    contracts;
17. `docs/distillation_followup/distillation_implementation_post_red_team.md`;
18. `docs/distillation_followup/handoffs/post_red_team/shared_stage_15_symbolica_campaign_skeleton.md`;
19. `pyproject.toml`, `uv.lock`, `.python-version`, and repository-supported
    installation/test commands;
20. `docs/distillation_followup/handoffs/post_red_team/handoff_sequence.md`.

The Stage 13 protocol, profiles, analysis, approval, job identities, priorities,
retry semantics, optional-task order, gates, and hashes are immutable inputs.
Stage 14 may bind only the explicitly deferred resource/backend facts and the
prospectively defined qualification results.

## Scientific and operational boundary

Permitted:

- synthetic models, inputs, checkpoints, caches, masks, ledgers, failures,
  Fourier states, manifests, bundles, and report fixtures;
- actual hardware inventory, non-scientific throughput, memory, disk, network,
  interruption, scheduler, container, and numerical-equivalence measurements;
- exact resource facts supplied by Symbolica/Eton or observed read-only on an
  authorized machine;
- local temporary roots and explicitly authorized remote scratch/persistent
  roots;
- bounded downloads needed to build the locked environment, only with normal
  authorization and no secrets printed;
- generated credentials templates containing names only, never values.

Prohibited:

- accessing a registered/private teacher, student, checkpoint, dense output,
  endpoint, or Phase I result bundle;
- running a definitive teacher/student/discovery/null/calibration/Fourier job;
- changing any Stage 13 scientific value, identity, tier, priority, retry,
  failure, report, or claim rule;
- admitting Tasks 3–5 before protected completion is operationally secure;
- treating MPS, CUDA, or CPU as qualified without the frozen technical suite;
- assuming identical results across different hardware classes;
- storing credentials, tokens, private keys, host-specific secrets, absolute
  private paths, large artifacts, containers, or result bundles in Git/LFS;
- deleting scratch/source data before destination verification and Alex 6
  custody approval;
- issuing a production scheduler submission or Stage 15 launch.

Every rehearsal artifact and report must state `scientific_data=false`,
`production_eligible=false`, and `definitive_execution_started=false`.

## Chat protocol

- Stay in Chat mode for the entire handoff; do not mix Chat and Work modes.
- Complete Parts A–H in order. Use additional one-block turns only for a real
  failure, diagnostic branch, or the explicit resource-facts waiting gate.
- Every operational response contains exactly one fenced terminal block.
- Keep blocks short enough to render reliably and print useful diagnostics.
- Briefly state what the block changes or inspects.
- Austin returns complete stdout/stderr before the next block.
- Never emit only a status footer when the next block is available.
- Never claim to await output from a block that was not supplied.
- Use the repository virtual environment when present; do not silently use an
  incompatible system interpreter.
- In zsh snippets, do not use reserved names such as `status`.
- Do not print environment variables wholesale; secrets may be present.
- Preserve unrelated tracked changes and user-owned untracked files.
- Use focused checks during implementation and one broad exact-SHA
  double-check at the end.
- End every response with:

```text
HANDOFF=AUSTIN_06_STAGE14B
COMPLETED_PARTS=<...>
NEXT_PART=<...>
BASE=<exact implementation base recorded in Part A>
HEAD=<exact current SHA>
WAITING_FOR=<NONE or exact resource/access blocker>
SCIENTIFIC_DATA=NO
```

## Required separations

Keep these independent and hash-bound:

1. **Frozen scientific manifest** — immutable Stage 13 jobs and meanings.
2. **Environment identity** — source SHA, lockfile, interpreter, packages,
   container recipe, built image digest, and platform.
3. **Input bundle** — exact public/config/schema/code inputs and their hashes;
   production private/registered data are absent until separately staged.
4. **Resource inventory** — observed machines, hardware classes, availability,
   quotas, paths, and scheduler capabilities.
5. **Qualification record** — numerical, resume, throughput, memory, and
   storage evidence for one hardware/environment class.
6. **Placement plan** — mapping immutable jobs to qualified resource classes;
   placement never changes logical identity.
7. **Scheduler submission** — provider IDs, arrays, dependencies, retries, and
   runtime metadata outside scientific identity.
8. **Operational state** — claims, heartbeats, failures, completion, and gates.
9. **Storage/export state** — scratch, compact outputs, chunks, destinations,
   and independent verification.
10. **Launch authorization** — a separate Alex 6 artifact that does not exist
    during Austin 6 and is mandatory for any definitive submission.

## Part A — Exact base, frozen-hash verification, and integration inventory

The first block is read-only. It must print and verify:

- repository root, remote, branch, HEAD, local `main`, and `origin/main`;
- ancestry of `015e67a...` and `51fc214...`;
- PR #35 merged state and exact head;
- physical SHA-256 values for every Stage 13 frozen artifact listed in its
  freeze document, including approval, campaign root, array spec, expanded
  seal, resource projection, exclusions, optional increments, and report;
- independent materialization of all 8,745 ordered job identities and equality
  with `adbfb306...` without storing a verbose tracked expansion;
- 7,884 protected jobs and three ordered 287-job optional increments;
- importability and version identity of all P1–P5/R1–R3 consumers;
- `uv.lock`, Python, torch, build/container tools, scheduler commands, CUDA/MPS
  visibility, disk space, and current host facts—absence is reported, not fixed;
- tracked cleanliness and separately listed untracked files;
- absence of Stage 14-B package artifacts and active Stage 15/scientific work;
- no access to registered/private artifacts.

After diagnosing that output, a second block may create the required branch.
Then create a compatibility inventory mapping every Stage 13 job family to its
runner, resource class, expected outputs, lifecycle handler, compact serializer,
recompute path, and rehearsal fixture. Missing executors or semantic mismatches
are blocking implementation findings.

**Part A passes when:** the complete freeze is physically intact and executable
interfaces are accounted for without touching scientific inputs.

## Part B — Locked environment, container, and input-bundle contract

Implement one versioned environment contract containing:

- repository commit and dirty-state prohibition;
- `.python-version`, interpreter implementation/version, platform, and ABI;
- `uv.lock` SHA and exact resolved package inventory;
- torch/CUDA/MPS/runtime/library versions and build flags;
- deterministic environment variables and thread settings;
- container recipe SHA, build command, image ID, immutable digest, and supported
  runtime (`Docker`, `Apptainer`, or the provider-supported equivalent);
- build provenance and network/dependency source inventory;
- compatibility rules for CPU, CUDA, and MPS pools;
- explicit non-portable or unqualified fields.

Provide a minimal versioned container recipe using the existing lock rather
than inventing a second dependency list. Never add a container image to Git.
If the local host cannot build the provider image, validate syntax and produce
a reproducible build plan; the actual digest remains an explicit waiting field
until built on an authorized compatible system.

Implement an input-bundle planner and verifier that:

- stages only committed code/config/schema/manifests plus later explicitly
  supplied registered input objects;
- binds every object by relative canonical path, size, SHA-256, role, and
  provenance;
- rejects extra, missing, stale, duplicate, conflicting, unsafe, symlinked,
  world-writable, or mutated inputs;
- uses deterministic bundle/chunk rules and verifies after transfer;
- separates credential/path bindings from the content manifest;
- never discovers private inputs by scanning a home directory.

Tests must cover lock drift, package drift, dirty code, image-tag mutation,
platform mismatch, missing runtime libraries, corrupt/stale/extra inputs,
unsafe paths, symlinks, transfer interruption, and destination re-verification.

**Part B passes when:** environment and input bytes can be independently
reconstructed and verified, while unavailable container/provider facts remain
explicit rather than fabricated.

## Part C — Resource inventory and CPU/CUDA/MPS qualification

Implement a read-only inventory command producing one closed record per
resource pool. It must capture, without secrets:

- provider/site, scheduler/version, queue/partition capabilities, account
  placeholder identity, wall limits, preemption, arrays, dependencies, and job
  limits;
- host count, CPU model/architecture, physical/logical cores, RAM, NUMA where
  relevant, accelerator count/model/VRAM/driver/runtime, and interconnect;
- local/shared scratch and persistent capacity, quota, inode/file limits,
  throughput, atomic-rename semantics, and permitted roots;
- network/transfer policy and interruption behavior;
- permitted availability intervals, especially school evenings/weekends;
- environment/container support and observed clock/source metadata;
- homogeneous hardware-class fingerprints and maximum eligible concurrency.

Implement a qualification suite using excluded deterministic fixtures only:

- CPU reference output and rerun determinism;
- CUDA and MPS output comparison against the CPU reference under a prospective
  technical tolerance defined before inspection;
- hard and soft training steps, eligibility computation, greedy ranking,
  model-in-loop hard-concrete, exact full-domain evaluation, packing, exact
  calibration shard, Fourier interchange, serialization, checkpoint/resume,
  compact merge, and export verification;
- three measured repeats after one discarded warmup;
- training updates/s, hard-concrete steps/s, exact evaluations/s, Fourier
  trials/s, memory/VRAM peaks, host-support cores, scratch/output rate, startup,
  queue/preemption, and serial merge/export critical path;
- uninterrupted versus interrupted/resumed semantic equality;
- same-class repeatability and cross-class drift.

Qualification is per hardware/environment class. Unqualified MPS contributes
zero MPS capacity; a Mac may still qualify for CPU work. Mixed Mac classes are
separate pools. Do not aggregate RAM across machines as if one job can use it.

For Eton, provide a no-admin smoke bundle that can be run on one machine first.
It inventories hardware and executes bounded CPU/MPS tests without installing a
daemon, exposing an inbound port, reading other users' files, or exceeding an
injected CPU/memory/disk/time ceiling.

**Resource-facts waiting gate:** if Symbolica/Eton access is unavailable, Part C
may finish the implementation and synthetic tests, record
`WAITING_FOR_AUTHORIZED_RESOURCE_FACTS`, and continue through the provider-
neutral rehearsal. It must not mark any real pool qualified.

**Part C passes when:** every available pool is measured honestly or explicitly
waiting, and no backend gains production authority from assumptions.

## Part D — Scheduler adapters, placement, and guarded operator commands

Reuse P3 logical identities and state machine. Implement:

- a deterministic local technical scheduler for rehearsal;
- a generic job-array contract;
- a Slurm-class adapter with injected command names/fields and no hard-coded
  Symbolica account, queue, partition, path, or credential;
- an offline/interruption-tolerant school-Mac shard package that receives a
  sealed job bundle, runs bounded eligible jobs, and returns a sealed result
  bundle for independent verification;
- capability-based resource matching for CPU, CUDA, qualified MPS, memory,
  storage, wall time, and availability intervals;
- dependency-aware arrays, concurrency caps, priorities, claims, heartbeats,
  bounded retries, terminal failures, and restart reconciliation;
- provider IDs and worker placement stored only as operational metadata.

Provide one cohesive operator CLI with subcommands equivalent to:

```text
qualify
stage-inputs
plan
launch --dry-run
status
pause
stop
resume
audit
recompute
compact
export
verify-export
```

Every command must print exact campaign/environment/resource/manifest identity,
scope, dry-run/production state, changed objects, and next safe action. Status
must report counts, failures, retries, runtime, resources, storage, hashes, and
gates without highlighting phase effects or other comparative direction.

The production `launch` path must require all of:

- exact final Stage 14 SHA;
- verified Stage 13 freeze hashes;
- qualified resource/environment/input manifests;
- passing protected-core feasibility record;
- an Alex 6 launch-authorization artifact with exact schema and hash;
- an explicit non-dry-run operator confirmation.

Because the Alex 6 artifact is absent, production launch must deterministically
reject throughout Austin 6. Tests must prove that flags, environment variables,
edited JSON, direct adapter calls, and replayed technical tokens cannot bypass
the guard.

**Part D passes when:** the immutable DAG can be mapped onto local, Slurm-class,
and bounded Mac workers without changing logical identity, while real launch
remains impossible.

## Part E — Monitoring, gates, final window, and verified custody

Implement concise operational monitoring and gate records for:

### Gate 15.1 — launch readiness

- exact SHA/environment/input/resource/manifest verification;
- qualification status and tiny full-pipeline comparison;
- protected-core capacity and storage pass;
- production launch remains unauthorized during Austin 6.

### Gate 15.2 — primary completeness

- planned/terminal/eligible/failed/retry counts by protected family;
- hash, dependency, quota, and inventory integrity;
- projected protected completion under remaining verified intervals;
- no comparative endpoint direction;
- optional Task 3, then 4, then 5 admission only as whole sealed increments
  after protected completion is secure under the frozen rule.

### Gate 15.3 — exit

- automatic final-window start at `H_total - 12 hours`;
- no new optional work after the boundary;
- protected terminal-state closure and independent recomputation;
- deterministic merge/compact/bundle/export;
- destination reread and size/hash verification;
- source preservation and custody manifest.

Monitoring must distinguish scheduler running from logical success and sealed
output. It must detect stale heartbeats, duplicate claims, queue stalls,
preemption, storage warnings, quota breach, partial objects, transfer failure,
clock anomalies, and final-window risk. Alerts should contain identifiers and
actions, never secret values or scientific directions.

Model storage explicitly as active scratch, rolling recovery, staging/retry
worst case, retained compact output, destination transfer, and verified
persistent custody. Enforce measured quotas with headroom. Git/LFS is never an
artifact destination.

**Part E passes when:** operators can safely observe, pause, resume, stop, audit,
and export the campaign, and the 12-hour audit window cannot be consumed by
scientific scheduling.

## Part F — Complete reduced synthetic rehearsal and forced failures

Materialize a deliberately tiny but topology-complete rehearsal manifest from
the frozen 8,745-job graph. It must include at least one representative of:

- all five tasks and protected/optional admission gates;
- teacher training, phase availability, hard/soft target caching, student
  training, eligibility, sealing, and failure;
- canonical and alternate architecture/basis paths;
- greedy and hard-concrete discovery, exact-ledger bridge, both endpoints,
  frontier/packing reducer reuse, all four packing/calibration layers, and the
  exact-calibration sharding/certificate path;
- aligned Fourier interchange and every one of the five controls;
- report reduction, recomputation, compact merge, export, and verification;
- all three human gates with production release disabled.

Force and recover from:

- process interruption before and after checkpoint publication;
- retryable worker/resource interruption;
- nonretryable scientific/validation failure;
- ineligible student and unavailable phase;
- search numerical failure and budget exhaustion;
- duplicate/stale claim and conflicting output;
- missing dependency, orphan output, corrupted ledger/manifest;
- quota warning/failure, interrupted merge, incomplete/corrupt transfer;
- simulated queue/preemption delay and final-window boundary;
- optional-task admission rejection;
- unauthorized production-launch attempt.

Run uninterrupted and interrupted/resumed rehearsals. Independently recompute
the synthetic report, manifests, compact bundle, and destination inventory.
Canonical scientific/technical content hashes must match; runtime telemetry may
differ only in explicitly excluded fields.

Run from repository root and an unrelated cwd under at least two
`PYTHONHASHSEED` values. The rehearsal may write only beneath explicit temporary
roots and must contact no real scheduler or network service unless separately
authorized for a technical adapter smoke test.

**Part F passes when:** every lifecycle path works, failures remain visible, and
the complete reduced DAG is reproducible without any scientific execution.

## Part G — Bind actual resources and prove protected-core feasibility

When authorized resource access is available, bind only observed facts:

- Symbolica grant, scheduler/account/queue, hardware pools, permitted 96-hour
  interval, quotas, paths, container support, transfer destination, and measured
  rates; and/or
- Eton approved machines, per-class inventory, allowed nights/weekends,
  interruption rules, storage ceilings, CPU/MPS qualification, and measured
  rates.

Use Stage 13's frozen equations and the measured resource records to construct
a dependency- and interval-aware schedule. Report separately:

- accelerator device-hours;
- host-support CPU-core-hours;
- standalone CPU-core-hours;
- serial/weak CPU critical path;
- maximum useful versus actually available concurrency;
- ideal and efficiency-adjusted wall time;
- retained, active scratch, staging/retry worst case, and requested quotas;
- protected core and each whole optional task increment.

The feasibility solver must honor dependencies, heterogeneous pools, per-job
memory, unavailable intervals, queue/preemption efficiency, final-window
reservation, and optional ordering. It must never divide total work by all
cores when jobs are serial, GPU-bound, unavailable, or not concurrently ready.

Pass only if one verified configuration completes all 7,884 protected jobs
within `H_science = H_total - 12 hours`, stays within 80% of verified memory and
storage limits, passes environment/backend/resume equivalence, and leaves the
audit/export path feasible. Optional tasks are assessed afterward and cannot
make the protected pass appear easier.

If Symbolica or Eton is unavailable, record the exact missing facts. If no
configuration passes, report `PROTECTED_CORE_LAUNCH_BLOCKED` and stop before
Alex 6 authorization. Do not select a reduced package; Alex owns any prospective
salvage amendment.

**Part G passes when:** real resource facts support a complete protected-core
schedule, or the absence/failure is honestly and reproducibly blocking.

## Part H — Commit, exact-SHA double-check, PR, and Stage 14-B handoff

Inspect the changed surface. It should contain only Stage 14-B environment,
container recipe, staging, qualification, resource, scheduler, operator,
monitoring, rehearsal, validation, tests, and necessary documentation. Stage 13
scientific artifacts and Stage 12 semantic implementations remain unchanged
unless a demonstrated integration blocker requires a narrow descendant repair.

Create coherent commits without amend or force-push. Push and open a PR against
`main`; do not merge inside this handoff without master-task authorization.

At the final exact SHA, run a fresh detached-checkout internal double-check:

- physically verify every Stage 13 input hash and all 8,745 job identities;
- rebuild or independently verify the locked environment/container evidence;
- rerun input staging and destination verification;
- rerun available CPU/CUDA/MPS qualifications and resource projections;
- execute uninterrupted and forced-failure/resume reduced rehearsals;
- exercise every operator command, monitoring state, gate, final-window, and
  production-launch rejection;
- run focused/adversarial, consumed compatibility, Ruff, diff, schema,
  manifest, cwd/hash-seed determinism, tracked-cleanliness, size, private-path,
  secret, binary, archive, checkpoint, container-image, and LFS hygiene checks;
- classify findings as blocking, nonblocking, or question;
- repair blockers only through descendant commits and repeat at the new SHA.

Final report:

- base, branch, parent, final SHA, and PR;
- Stage 13 input paths/hashes and exact manifest reproduction;
- environment lock, recipe, built image digest or explicit build blocker;
- input-bundle identity and staging verification;
- available resource inventories and per-class qualification evidence;
- scheduler/Mac adapters and operator command surface;
- reduced rehearsal counts, hashes, forced failures, resume/recompute/export;
- monitoring, gates, 12-hour reserve, and launch-guard evidence;
- protected-core and optional-task feasibility projections using real facts;
- remaining Symbolica/Eton/provider/container/path/credential blockers;
- focused/compatibility totals and internal findings;
- confirmation of no registered/private/scientific execution;
- explicit statement that Stage 15 was not authorized or launched;
- exact handoff inputs for Alex 6.

Two valid terminal states exist:

```text
AUSTIN_06_STAGE14B_STATUS=COMPLETE_AT_ALEX6_HANDOFF
RESOURCE_BINDING=PASS
PROTECTED_CORE_FEASIBLE=YES
SCIENTIFIC_DATA=NO
DEFINITIVE_EXECUTION_STARTED=NO
STAGE15_STARTED=NO
```

or, when facts/access are genuinely unavailable or insufficient:

```text
AUSTIN_06_STAGE14B_STATUS=TECHNICAL_PACKAGE_READY_WAITING
RESOURCE_BINDING=WAITING_OR_FAILED
PROTECTED_CORE_FEASIBLE=UNRESOLVED_OR_NO
SCIENTIFIC_DATA=NO
DEFINITIVE_EXECUTION_STARTED=NO
STAGE15_STARTED=NO
```

The second state is a correct waiting/blocking result, not permission to invent
resources, reduce scope, or launch.

## Prohibited shortcuts

- Do not mutate or regenerate Stage 13 scientific choices under new identities.
- Do not use a floating dependency, container tag, unverified image, or dirty
  checkout as production evidence.
- Do not hard-code guessed Symbolica/Eton accounts, queues, paths, credentials,
  machine counts, or availability.
- Do not call MPS/CUDA/CPU qualified from device detection alone.
- Do not mix hardware classes without separate qualification and placement.
- Do not aggregate RAM or disk across hosts as if one job can consume it.
- Do not estimate wall time by dividing all work by every advertised core.
- Do not expose comparative scientific direction in monitoring or capacity
  decisions.
- Do not admit an optional task before protected completion is secure.
- Do not retry a scientific failure with changed settings.
- Do not treat scheduler completion as sealed output success.
- Do not delete source/scratch data before verified export and custody handoff.
- Do not put credentials, images, checkpoints, bundles, private paths, or large
  outputs in Git/LFS.
- Do not bypass the absent Alex 6 launch authorization.
- Do not start Stage 15.
