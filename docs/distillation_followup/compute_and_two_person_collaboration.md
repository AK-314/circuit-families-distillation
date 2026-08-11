# Compute and Two-Person Collaboration Plan

## 1. What the predecessor actually cost

Measurements from the existing project on the M5 Max MacBook Pro:

| Quantity | Observed predecessor value |
|---|---:|
| Hardware | Apple M5 Max, 18 cores, 48 GB RAM |
| Definitive Stage 18 wall time | 257,027.8 seconds = 71.4 hours |
| Production workers | 12 isolated one-thread workers |
| Stage 18 search cells | 630 |
| Exact mask evaluations | 14,172,542 |
| Ranking passes | 846,226 |
| Stage 18 raw output | approximately 91 GB |
| Stage 18 compressed archives | approximately 9.8 GB |
| Whole current project results | approximately 106 GB |
| Whole current project checkpoints | approximately 15 GB |
| Result-file count | approximately 866,000 |
| Checkpoint-file count | approximately 5,900 |

The predecessor therefore averaged approximately:

- 22,496 exact mask evaluations per search cell;
- 144 MB of raw output per search cell under the old file layout;
- 1.36 worker-hours per search cell, inferred from 71.4 wall hours × 12 workers / 630 cells.

These are scaling anchors, not guaranteed costs for the new fidelity or methods.

## 2. New core-design size

Under the recommended starting design:

- 5 teacher seeds;
- 3 function-defined phases per teacher;
- 2 distillation conditions;
- 3 eligible students per teacher–phase–condition cell;
- up to 6 attempted initializations per cell.

This gives:

| Item | Count |
|---|---:|
| Direct teacher models | 5 × 3 = 15 |
| Eligible student models if every cell succeeds | 5 × 3 × 2 × 3 = 90 |
| Dense models receiving circuit analysis | 105 |
| Maximum student attempts | 5 × 3 × 2 × 6 = 180 |
| Search cells for one method and one primary setting | 105 |
| Search cells for two methods and one primary setting | 210 |

## 3. Student-training estimate

A local throughput check using the existing architecture and all 12,769 inputs measured:

- approximately 0.069 seconds per full-domain hard-target training step;
- approximately 0.023 seconds per full-domain forward pass.

The convergence length is unknown until the technical pilot. At the maximum 180 attempts:

| Training budget | Serial training time at measured throughput | Practical multi-worker order of magnitude |
|---|---:|---:|
| 1,000 steps per attempt | 3.5 hours | under 1 hour to roughly 2 hours |
| 5,000 steps per attempt | 17.3 hours | roughly 2–5 hours |
| 10,000 steps per attempt | 34.5 hours | roughly 4–10 hours |

This excludes checkpoint writing, eligibility evaluation, retries, throttling, and a slower second laptop. Student training is likely smaller than circuit search unless convergence requires many tens of thousands of steps.

Do not save dense student checkpoints every 50 steps as in teacher training. Keep an atomic resume checkpoint, the final checkpoint, metrics, and sealed dense outputs. Dense checkpointing across 180 attempts could add tens to hundreds of gigabytes without scientific value.

## 4. Circuit-search estimate

If the new method has approximately the predecessor's average cell cost:

| Production scope | Inferred worker-hours | Inferred 12-worker wall time | Conservative planning range |
|---|---:|---:|---:|
| 105 cells, one method | 143 | 11.9 hours | 12–36 hours |
| 210 cells, two methods | 286 | 23.8 hours | 1–3 days |
| Full independent repeat of two-method production | another 286 | another 23.8 hours | another 1–3 days |

The conservative range allows for centred-logit fidelity, harder student functions, packing search, uneven cells, and operational interruptions.

A naïve 18-setting sensitivity grid would produce 1,890 search cells per method. At predecessor scaling that is about 2,571 worker-hours, or roughly 8.9 days wall time on 12 comparable workers, per method. Two methods plus full repetition could move into the 3–5 week range. It should not be the default.

Where scientifically valid, component-cap and overlap sensitivity should be recomputed from one sealed proposal/evaluation ledger. Fidelity settings that change the search trajectory may still require separate searches and must be costed explicitly.

## 5. Storage estimate

Under the predecessor's raw layout:

| Scope | Approximate raw search storage |
|---|---:|
| 105 cells, one method | 15 GB |
| 210 cells, two methods | 30 GB |
| 1,890 cells, one full 18-setting grid | 273 GB |
| Two methods with full 18-setting grids | 546 GB |

The new implementation should not preserve the predecessor's “many tiny files” layout. Use one append-only exact-evaluation ledger during a job, then seal one compressed archive per model–method cell with a small JSON manifest and SHA-256. This should reduce file count by orders of magnitude and likely reduce storage materially, but no reduction should be assumed in the frozen compute projection until measured.

Practical free-space reservations:

- one method, core only: at least 100 GB across active work, staging, and reproduction;
- two methods, core only: at least 200 GB;
- broad rerun-based sensitivity: 500 GB or more;
- predecessor plus follow-up retained locally: 350–500 GB is a sensible minimum working reserve.

The current M5 Max machine has approximately 1.1 TiB free, so local capacity is adequate for the recommended core design.

## 6. Git is for code and metadata, not scientific bulk data

The existing repository already has a GitHub `origin`, but local `main` is 80 commits ahead. Its reachable Git history contains tracked archives of approximately 527 MB and 106 MB. GitHub blocks regular Git files above 100 MiB and recommends keeping repositories ideally below 1 GB and strongly below 5 GB.

Therefore, do not push the current 80-commit branch directly as the collaboration baseline.

Recommended approach:

1. Preserve the current repository unchanged as the completed predecessor record.
2. Create a new private repository for the distillation follow-up from a clean working-tree snapshot, or create a history-filtered clone in a separate directory.
3. Include source code, tests, configs, protocol documents, schemas, small manifests, small summary tables, and figure source code.
4. Exclude checkpoints, raw outputs, archives, dense output caches, temporary ledgers, and local environments.
5. Record the predecessor freeze commit and hashes of reused artifacts in a predecessor-link manifest.
6. Store a Git bundle of the predecessor history in durable artifact storage if an off-laptop archival copy is required.

GitHub's current documentation:

- [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [Managing access to personal repositories](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration)

Git LFS is not recommended for the primary artifact store. GitHub Free currently includes 10 GiB of LFS storage and 10 GiB of monthly bandwidth, well below the existing 121 GB working set, and each changed LFS object version consumes its full size again.

## 7. Recommended collaboration architecture

```text
Private GitHub repository
  ├── code, tests, locked environment
  ├── protocols, configs, schemas
  ├── job-assignment registries
  ├── small manifests and hashes
  └── pull requests and review history

Private versioned object-storage bucket
  ├── selected teacher checkpoints
  ├── sealed student checkpoints and outputs
  ├── search-cell archives
  ├── teacher-seed inventory archives
  └── reproduction bundles

Each laptop
  ├── independent clone and virtual environment
  ├── local scratch/output root
  ├── assigned immutable job list
  └── upload/download tool using personal credentials
```

Use an S3-compatible object store or another central versioned object store. A simple manifest-aware upload/download wrapper is preferable to bidirectional synchronization of live result directories. DVC is optional; the existing project already has strong manifests and hashes, so DVC should be adopted only if it simplifies rather than duplicates that system.

Do not use Dropbox, iCloud Drive, or Syncthing as the authoritative live result directory. They can copy partially written files and create conflict versions. Seal and hash a cell first, then upload its immutable archive.

## 8. GitHub working practice

- Both collaborators use their own GitHub and artifact-store accounts.
- Keep `main` releasable and never run unfinished production configs from it.
- Create one branch per issue or bounded stage, such as `feat/centred-logit-fidelity`.
- Push small commits at least daily and after every passing implementation unit.
- Open a pull request; the other collaborator reviews interface and scientific-contract changes.
- Require tests and schema compatibility before merging.
- Tag protocol and analysis freezes, for example `distillation-protocol-freeze-v1`.
- Never force-push shared branches.
- Never commit credentials; use environment configuration or the operating-system keychain.
- Commit artifact manifests only after the referenced object has uploaded and its hash has been verified from the remote.

A private GitHub repository supports remote collaboration regardless of Wi-Fi network. GitHub Free currently permits unlimited collaborators on private repositories. An organization-owned repository is preferable for shared ownership and clearer roles; a personal private repository is adequate for two trusted collaborators.

## 9. Two-person implementation division

### Person 1 — Existing-project, circuit-method, and heavy-production owner

Recommended for the person who has the predecessor project and checkpoints:

- Lane A: predecessor link, teacher registry, phase selection, protocol and schemas.
- Lane C: centred-logit fidelity, exact ledgers, endpoint reducers, discovery adapters, budget accounting.
- Execution of definitive direct-teacher and student circuit search on the stronger machine.
- Scientific aggregation and final interpretation.
- Maintainer of the canonical protocol/config freeze.

Primary code/scientific stages: 1–4, 5A, 6A, 6D, 6E, 12–15, 18, 20–27.

### Person 2 — Distillation and execution-system owner

- Lane B: teacher target caches, shared trainer, hard/soft losses, eligibility, attempt accounting.
- Lane D implementation: job DAG, isolated output roots, resume, deterministic merge, synthetic analysis fixtures.
- Student-production and upload tooling.
- Local execution limited by hardware: technical fixtures, one training worker, eligibility, manifests, compact analysis, and optional assigned student attempts after cross-machine validation.

Primary code stages: 5B–5D, 6B–6C, 7–11, 16–17, and 19. Person 2 does not need to execute Stage 18 definitive circuit search merely because they own its orchestration interfaces.

### Shared decisions and reviews

Both people must approve:

- Stage 4 schemas and canonical identities;
- Stage 7 integration output and exclusion register;
- Stages 11–14 numeric freezes;
- any protocol amendment;
- production job assignment;
- reproduction discrepancies;
- Stage 26 analysis freeze.

Each person reviews the other's pull requests at the interfaces between B/C and A/D. Neither person should both introduce and unilaterally approve a scientific definition change.

## 10. Two-laptop production allocation

### Expected footprint on the weaker laptop

The existing environment is about 1.1 GB; source, tests, and configs are under 10 MB; a complete model checkpoint is about 2.75 MB; and all 15 selected teacher checkpoints should be only about 40–50 MB. One dense 12,769 × 113 float32 logit array is about 5.5 MiB.

If Person 2 processes one attempt at a time, uploads sealed artifacts, and removes verified local scratch copies, expected concurrent use is approximately:

| Item | Working allowance |
|---|---:|
| Clean repository and environment | 2–4 GB |
| Selected teachers and target caches | under 1 GB |
| One active training attempt and resume/final artifacts | under 1–2 GB |
| One optional search/integration fixture | 1–5 GB |
| Logs, downloads, temporary archive, and safety margin | 5–10 GB |

Approximately 20 GB free is a practical minimum; 40 GB free is comfortable. If the internal disk cannot provide that, place local scratch, downloaded artifacts, and archives on a 500 GB or 1 TB external SSD. Keep the Git checkout and environment internal if convenient.

Secondary storage does not increase RAM or CPU performance. Use one worker and minibatched training on an 8 GB machine; do not run multiple concurrent PyTorch workers. The full-domain training benchmark used about 0.85 GB peak RSS on the M5 Max, but allow 1–2 GB for platform and library variation.

Before sharing production:

1. lock Python and dependency versions;
2. record both hardware/software environments;
3. run the same technical fixture on both laptops;
4. compare teacher outputs, fidelity, eligibility, and endpoint records;
5. freeze numerical tolerances for any expected hardware differences.

With an 8 GB / 256 GB weaker laptop, do not allocate two definitive search shards to it by default. Run definitive circuit search on the M5 Max. Person 2 may run assigned student-training attempts sequentially after cross-machine validation, because the model is small, then upload sealed checkpoints and outputs. Alternatively, the M5 Max can execute all definitive numerical work while Person 2 owns implementation, review, orchestration, manifests, and compact analysis.

If later benchmarking shows the weaker laptop is fast enough and has an external SSD, it may receive a complete teacher-seed search shard. That is an optimization, not the baseline plan.

If the machines differ materially, do not confound hard versus soft condition with machine. Either run definitive numerical work on one hardware class, or balance both conditions and phases across machines under a frozen assignment. A small cross-machine reproduction subset should be mandatory.

Every assignment is generated before production and committed as a small job-registry file. Workers do not claim work informally in chat. Completed cell archives upload to object storage; verified manifests return through pull requests or a dedicated manifest branch.

## 11. Recommended near-term sequence

1. Create the clean private follow-up repository.
2. Invite the collaborator and configure branch/PR practice.
3. Create the private object-storage bucket and per-user credentials.
4. Upload only the 15 selected teacher checkpoints and their registry, not the entire 15 GB checkpoint tree.
5. Merge the common schemas and condition-ID implementation.
6. Let Person 1 begin fidelity/endpoints while Person 2 begins distillation/orchestration.
7. Run the cross-laptop integration fixture.
8. Benchmark both machines.
9. Freeze numeric parameters and the job assignment.
10. Begin definitive production.
