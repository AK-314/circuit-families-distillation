# Stage 14-B provider-neutral cluster package and reduced rehearsal

## Boundary and current result

This package implements the Austin 6 technical launch machinery without
starting Stage 15. It uses only committed public/config/schema/code inputs and
excluded deterministic fixtures. It has not accessed a registered teacher,
student, checkpoint, dense output, endpoint, credential, provider path, or
private result.

The implementation base is
`19393dc345556fcec1564ef3918650d25b2b88ec`. The Stage 13 merge
`015e67a60db64e41713f8493d5394ce90c129e69` and implementation head
`51fc2147ff4e123ebbfcabc4206520ca72f8e24f` are ancestors of that base. The
frozen campaign still expands to 8,745 jobs: 7,884 protected and three ordered
287-job optional increments. The canonical member hash remains
`adbfb30694bb984de4d8ba582cee0efb468b8f9a2fce01f6a3654b5b78b1927b`.

The current terminal state is technically ready but waiting. The local host
passed the bounded CPU technical fixture, but it is not a production pool.
CUDA and usable MPS are absent, as are Docker/Podman/Apptainer and Slurm
commands. No Symbolica or Eton pool has been qualified. The immutable container
digest, provider paths, quotas, scheduler details, transfer destination, and
96-hour availability interval therefore remain unbound. Protected-core launch
is blocked; no smaller or salvage package is selected.

```text
AUSTIN_06_STAGE14B_STATUS=TECHNICAL_PACKAGE_READY_WAITING
RESOURCE_BINDING=WAITING_OR_FAILED
PROTECTED_CORE_FEASIBLE=UNRESOLVED_OR_NO
SCIENTIFIC_DATA=NO
DEFINITIVE_EXECUTION_STARTED=NO
STAGE15_STARTED=NO
```

## Package separations

The implementation keeps these identities independent:

1. the immutable Stage 13 manifest and its physical hashes;
2. source, lock, interpreter, package, platform, recipe, image, and runtime
   environment identity;
3. exact staged input bytes and separate credential/path bindings;
4. secret-free resource inventories;
5. per-hardware/environment-class qualification records;
6. capability-based placement that never changes logical job identity;
7. provider job IDs and array indices as operational metadata;
8. claims, heartbeats, retries, failures, gates, and final-window state;
9. deterministic compact/export state with destination reread verification;
10. the absent Alex 6 launch authorization.

The compatibility inventory maps all 17 frozen Stage 13 job families to their
runner, resource class, expected output, lifecycle handler, compact serializer,
recompute path, and reduced rehearsal fixture. No missing executor or semantic
mismatch is recorded.

## Locked environment and container

`followup/configs/stage14b/environment_build_plan_v1.json` binds the
implementation base, `.python-version`, `uv.lock`, and the container recipe.
The recipe consumes the existing lock with `uv sync --frozen`; it does not
invent a second dependency list. Both base images must be supplied as immutable
`name@sha256:digest` references. A mutable default tag is intentionally absent.

The current host has no supported container runtime, so syntax and contract
checks are available but no image ID or immutable digest is claimed. An
authorized compatible build must fill the explicit waiting fields and capture
its source inventory and provenance.

The input planner selects exact tracked code/config/schema/manifest/fixture
bytes. Each object is bound by portable relative path, byte length, SHA-256,
role, and provenance. Staging rejects untracked extras, missing/corrupt/stale
objects, duplicates, unsafe paths, symlinks, world-writable objects, conflicting
destinations, mutation during copy, and incomplete transfers. It never scans a
home directory for registered inputs.

## Inventory and qualification

Resource inventory records omit serial numbers, UUIDs, credentials, absolute
private paths, and secret values. They keep scheduler capabilities, host-local
CPU/RAM/accelerator facts, storage/inode/quota facts, permitted roots,
availability intervals, interruption/network policy, container support, and a
homogeneous class fingerprint distinct from production authority.

The qualification policy was fixed before inspection at absolute tolerance
`1e-6` and relative tolerance `1e-5`. It discards one warmup and measures three
repeats. The excluded fixture covers hard/soft steps, eligibility, greedy
ranking, hard-concrete, full-domain evaluation, packing, an exact calibration
shard, Fourier interchange, serialization, checkpoint/resume, compact merge,
and export verification. Device discovery alone cannot qualify CPU, CUDA, or
MPS. Mixed classes require separate records, and memory is enforced per host.

`scripts/stage14b_eton_smoke.py` is the bounded single-machine school-Mac smoke
bundle. It needs no administrator access, daemon, inbound port, or network
service and accepts explicit memory, disk, and time ceilings. It qualifies CPU
and, only when requested and available, separately evaluates MPS.

## Scheduler, operator, monitoring, and custody

The package provides a deterministic local technical scheduler, a generic
capability matcher, a Slurm-class array adapter with only injected provider
fields, and an outbound/offline Mac shard bundle. The Slurm adapter cannot
construct or replay a launch capability. A future submission requires an exact
Alex 6 authorization record, exact current Stage 14 SHA, matching frozen
bindings, a one-use in-process capability, an injected scheduler executor, and
the literal exact-SHA operator confirmation.

All operator commands emit campaign, manifest, environment/resource status,
scope, dry-run state, changed objects, and the next safe action. They are:

```text
.venv/bin/python -m circuit_families.stage14b.cli qualify --backend cpu --output-root <technical-root>
.venv/bin/python -m circuit_families.stage14b.cli stage-inputs --destination <empty-staging-root>
.venv/bin/python -m circuit_families.stage14b.cli plan
.venv/bin/python -m circuit_families.stage14b.cli launch --dry-run
.venv/bin/python -m circuit_families.stage14b.cli status --state-root <state-root>
.venv/bin/python -m circuit_families.stage14b.cli pause --state-root <state-root>
.venv/bin/python -m circuit_families.stage14b.cli stop --state-root <state-root>
.venv/bin/python -m circuit_families.stage14b.cli resume --state-root <state-root>
.venv/bin/python -m circuit_families.stage14b.cli audit
.venv/bin/python -m circuit_families.stage14b.cli recompute
.venv/bin/python -m circuit_families.stage14b.cli compact --source-root <sealed-root> --relative-path <path> --bundle-root <bundle-root> --bundle-reference <reference>
.venv/bin/python -m circuit_families.stage14b.cli export --bundle-root <bundle-root> --destination <destination> --transfer-state <state-file> --destination-reference <reference>
.venv/bin/python -m circuit_families.stage14b.cli verify-export --destination <destination> --expected-manifest-sha256 <sha256>
```

`launch` without `--dry-run` deterministically rejects while the Alex 6
artifact is absent. Flags, environment variables, edited JSON, direct adapter
calls, and replayed capabilities do not bypass the guard.

Monitoring reports logical lifecycle counts, terminal failures, retries,
resources, storage, integrity hashes, alerts, and human gates. It contains no
endpoint value, effect direction, comparison, or scientific ranking. Scheduler
completion never counts as sealed success. The exact final-window rule reserves
43,200 seconds at `H_total - 12 hours`, stops new optional work, closes and
recomputes protected state, serially compacts/exports, rereads destination
bytes, and preserves every source pending Alex 6 custody approval.

## Reduced rehearsal

The 48-job excluded synthetic DAG includes all five tasks, protected and
optional gates, teacher/phase/cache/student/eligibility/seal/failure paths,
canonical and alternate architecture/basis paths, both discovery methods and
their exact bridge, both endpoints, frontier and packing reuse, all four
calibration layers and certificate, aligned Fourier interchange and all five
controls, report/recompute/compact/export/verification, and all three human
gates with production release disabled.

Uninterrupted and interrupted/resumed executions use different ready-job order
and hash seeds. Their canonical job outputs, independent recomputation,
deterministic compact bundle, and verified destination inventory match.
Telemetry differences are restricted to explicitly excluded execution-order,
hash-seed, interruption, retry, and timing fields.

The forced matrix contains 20 visible cases: interruption before and after a
checkpoint, worker/resource interruption, nonretryable validation failure,
ineligible student, unavailable phase, numerical search failure, budget
exhaustion, duplicate/stale claim, conflicting output, missing dependency,
orphan output, corrupt ledger/manifest, quota warning/failure, interrupted
merge, incomplete/corrupt transfer, queue/preemption delay, final-window
boundary, optional-admission rejection, and unauthorized production launch.

## Validation and Alex 6 inputs

The exact-SHA validator is cwd-independent and writes only below an explicit
temporary output root:

```text
.venv/bin/python -m scripts.validate_stage14b --validate-only --output-root <empty-technical-root>
```

It revalidates every frozen Stage 13 hash and all 8,745 identities, captures and
verifies the clean locked environment, stages and rereads the input bundle,
runs the available CPU technical qualification, runs both rehearsals, verifies
the compatibility surface, exercises the launch rejection, and emits a
resource-binding waiting report. Alex 6 must receive the exact merged Stage 14
SHA, container digest/provenance, staged input identity, real resource and
qualification records, feasibility result, scheduler/path bindings, rehearsal
report, and this explicit blocker inventory before deciding whether to issue a
launch authorization.
