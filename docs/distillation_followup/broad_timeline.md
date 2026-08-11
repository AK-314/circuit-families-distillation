# Distillation Follow-up: Broad Two-Person Timeline

## Planning basis

**Calendar start:** 11 August 2026
**Recommended core completion:** 9 October 2026
**Protected contingency reserve:** 12–23 October 2026
**People:** Alex and Austin
**Heavy production machine:** Alex's M5 Max MacBook Pro
**Austin's machine:** development, tests, one-worker fixtures/training, manifests, and compact analysis

This is an eight-week core schedule with two additional reserve weeks. It assumes approximately 10–15 focused person-hours per person per week during implementation, prompt pull-request review, and permission for the M5 Max to run long jobs overnight. At approximately five person-hours per week, expect 10–12 weeks instead.

The schedule targets the three-phase, three-eligible-student core with one fully supported discovery procedure. A second method and broad rerun-based sensitivity may extend the calendar unless their cost is lower than the conservative projection.

## Responsibility summary

### Alex

- Preserve and link the predecessor study.
- Select and verify the 15 teacher checkpoints.
- Own the protocol, schemas, and scientific freeze.
- Implement centred-logit fidelity and both endpoint reducers.
- Adapt discovery methods and budget ledgers.
- Run heavy teacher and student circuit production on the M5 Max.
- Lead endpoint recomputation, hierarchical aggregation, and final interpretation.

### Austin

- Implement teacher-output caching and the shared student trainer.
- Implement hard and soft losses and eligibility.
- Implement attempt accounting and sealed student artifacts.
- Implement the job DAG, resume behavior, isolated outputs, uploads, and deterministic merge.
- Build synthetic analysis/reporting fixtures.
- Run technical fixtures and optional sequential student attempts, but not definitive circuit search by default.

### Shared

- Review all interface pull requests.
- Approve protocol/schema changes.
- Approve numeric freeze decisions.
- Review excluded development output.
- Resolve reproduction discrepancies.
- Approve the analysis freeze.

## Week 0 — Repository and collaboration setup

**Dates:** 11–16 August 2026

### Alex

- Preserve the completed predecessor repository.
- Create a clean private follow-up GitHub repository.
- Create the follow-up object-storage bucket.
- Move the follow-up documents into the project repository.
- Prepare the predecessor-link manifest.
- Identify the candidate per-seed pre, 50%, and stable-post checkpoints.

### Austin

- Create GitHub and artifact-store access.
- Clone the clean repository.
- Install the locked environment.
- Run the existing unit tests.
- Record laptop hardware, operating system, Python, PyTorch, and TransformerLens versions.

### Meetings

- **Kickoff, 60 minutes:** repository structure, ownership, security, branch rules, artifact rules, and deadline.
- **Environment check, 20 minutes:** confirm both machines can run the same smoke test.

### Gate

Both people can clone, test, push a branch, open a pull request, and upload/download one test artifact.

## Week 1 — Teacher registry, schemas, and interfaces

**Dates:** 17–23 August 2026

### Alex

- Finalize the 15-checkpoint teacher registry using accuracy-only phase rules.
- Implement canonical condition IDs and deterministic seed derivation.
- Draft schemas for dense models, exact-evaluation ledgers, and endpoints.
- Begin centred-logit fidelity implementation.

### Austin

- Draft schemas for teacher target caches, student attempts, eligibility, and job completion.
- Scaffold the shared trainer and output-cache loader.
- Scaffold the DAG/job registry and isolated output roots.
- Build synthetic valid and invalid records.

### Joint handoff

Merge one common schema package. Neither person proceeds with independent record formats.

### Meetings

- **Monday planning, 20 minutes.**
- **Wednesday interface review, 30 minutes.**
- **Friday schema gate, 45 minutes.**

### Gate

Synthetic records pass the same validators on both laptops.

## Week 2 — Parallel core implementation

**Dates:** 24–30 August 2026

### Alex

- Finish centred-logit predictive fidelity and gauge tests.
- Implement the intact-mask baseline and endpoint 1.
- Implement endpoint 2's compatibility graph and deterministic packing reducer.
- Define the discovery adapter and budget ledger.

### Austin

- Finish the teacher target cache.
- Implement hard-target training and 12,769/12,769 eligibility.
- Implement soft-target centred-logit loss and configurable eligibility tolerance.
- Finish resumable training and sealed attempt records.

### Meetings

- **Monday planning, 20 minutes.**
- **Wednesday B/C interface integration, 30 minutes.**
- **Friday demo and merge gate, 45 minutes.**

### Gate

Both endpoint reducers work from synthetic ledgers, and both distillation conditions produce schema-valid technical attempts.

## Week 3 — End-to-end technical pilot and compute freeze evidence

**Dates:** 31 August–6 September 2026

### Alex

- Adapt the existing discovery code to the new fidelity and ledger interface.
- Run one direct-teacher technical search fixture.
- Benchmark exact evaluation, method-native work, storage, and concurrency.
- Record all emitted pilot endpoints as excluded development output.

### Austin

- Run one hard and one soft technical attempt.
- Validate eligibility boundaries and failed-attempt accounting.
- Integrate trainer outputs with the discovery queue.
- Validate upload, interruption, resume, and deterministic merge.

### Shared

- Run the same cross-laptop fixture.
- Compare outputs and agree numerical tolerances.
- Review all forced edge cases.
- Produce the definitive compute and storage projection.

### Meetings

- **Monday pilot launch, 20 minutes.**
- **Wednesday discrepancy triage, up to 45 minutes.**
- **Friday integration gate, 60 minutes.**

### Gate

The whole pipeline works; pilot endpoints are excluded; remaining decisions can be frozen using technical evidence only.

## Week 4 — Final protocol freeze and production launch

**Dates:** 7–13 September 2026

### Alex

- Freeze predictive fidelity, component cap, overlap, methods, and budgets.
- Freeze the teacher registry and analysis rules.
- Build and verify direct-teacher production jobs.
- Launch direct-teacher circuit searches on the M5 Max.

### Austin

- Freeze hard/soft optimizer, loss, tolerance, stopping, attempt, and failure rules.
- Generate definitive student-attempt jobs.
- Launch assigned student training sequentially if cross-machine validation passed; otherwise prepare jobs for the M5 Max.
- Monitor manifest and upload correctness.

### Required joint meeting

- **Protocol-freeze meeting, 90 minutes:** resolve every freeze-register item, review the excluded pilot register, approve configs, and sign off the production assignment.

### Gate

The protocol and configs are committed and tagged before any definitive endpoint-producing job begins.

## Week 5 — Definitive student generation and teacher searches

**Dates:** 14–20 September 2026

### Alex

- Continue and finish direct-teacher search.
- Run student training not safely assigned to Austin.
- Monitor storage, throughput, and failures without inspecting comparative endpoints.

### Austin

- Monitor or run assigned hard/soft attempts one at a time.
- Recompute and seal student eligibility.
- Maintain attempt, failure, and cell-completeness registries.
- Upload verified student artifacts and open manifest pull requests.

### Meetings

- **Monday production check, 20 minutes.**
- **Friday completeness review, 30 minutes.**
- Use asynchronous GitHub status updates between meetings; no daily call.

### Gate

Every planned attempt is complete or has a terminal failure record, and the eligible-student circuit queue is sealed.

## Week 6 — Definitive student circuit search

**Dates:** 21–27 September 2026

### Alex

- Run definitive student circuit discovery on the M5 Max.
- Seal exact-evaluation ledgers and both endpoints.
- Maintain isolated output roots and upload completed archives.

### Austin

- Verify incoming manifests and remote hashes.
- Run deterministic inventory merging.
- Detect missing, duplicate, or conflicting jobs.
- Prepare compact analysis inputs without inspecting or interpreting phase contrasts.

### Meetings

- **Monday search launch/check, 20 minutes.**
- **Midweek only if a production blocker occurs.**
- **Friday inventory review, 30 minutes.**

### Gate

All five teacher-seed inventories account for every teacher, student attempt, eligibility decision, method job, endpoint, and failure.

## Week 7 — Reproduction, endpoint audit, and hierarchical analysis

**Dates:** 28 September–4 October 2026

### Alex

- Recompute both endpoints independently from sealed ledgers.
- Run the frozen reproduction scope.
- Aggregate within student cells and then within teacher seeds.
- Generate primary and sensitivity tables.

### Austin

- Independently verify selected caches, eligibility records, manifests, and inventories.
- Re-run compact analysis from sealed summaries.
- Compare reproduction artifacts and register discrepancies.
- Generate initial figures from frozen tables.

### Meetings

- **Monday audit allocation, 20 minutes.**
- **Wednesday discrepancy review, 45 minutes.**
- **Friday analysis gate, 60 minutes.**

### Gate

All deterministic comparisons pass and every population summary respects the teacher-seed hierarchy.

## Week 8 — Figures, outcome resolution, and analysis freeze

**Dates:** 5–11 October 2026

### Alex

- Finalize teacher/student endpoint tables and figures.
- Resolve the frozen outcome categories.
- Draft methods and primary results.
- Prepare the analysis-freeze manifest.

### Austin

- Audit figures against source tables.
- Check hard/soft separation, failure reporting, labels, and captions.
- Review methods for implementation accuracy.
- Reproduce the final report from the clean repository and sealed artifacts.

### Required joint meetings

- **Tuesday interpretation review, 60 minutes.**
- **Friday analysis-freeze meeting, 90 minutes.**

### Gate

Primary analysis is frozen and the core methods/results draft exists by 9 October, with the weekend available for administrative cleanup.

## Reserve Weeks 9–10 — Contingency, not planned scope

**Dates:** 12–23 October 2026

Use this reserve only for:

- failed or interrupted production jobs;
- numerical or reproduction discrepancies;
- missing eligible-student cells under the frozen attempt rule;
- figure/report corrections;
- environment or storage failures;
- final paper integration.

Do not automatically fill reserve time with extensions before the core analysis freeze.

## Regular meeting cadence

The default cadence should remain light:

- **Monday, 20 minutes:** select the week's bounded issues and confirm dependencies.
- **Wednesday, 30 minutes during Weeks 1–3 and 7 only:** interface or discrepancy review.
- **Friday, 30–45 minutes:** demonstrate merged work and decide whether the stage gate passes.
- **Asynchronous daily update:** one GitHub issue comment stating completed, next, and blocked.

Formal meetings that both people must attend:

1. kickoff;
2. schema gate;
3. integration gate;
4. protocol freeze;
5. production inventory gate;
6. analysis gate;
7. analysis freeze.

## Expansion ladder if the project finishes early

The design is expandable, but expansion choices must be frozen before the relevant results are inspected. Use technical completion and resource evidence, not favorable outcomes.

### Expansion Gate A — decided at the Week 3 compute review

If the core is projected to finish by the end of Week 7 with at least one reserve week intact:

1. increase eligible student realizations uniformly from three to five per cell; or
2. widen the independent reproduction scope.

Additional realizations are the preferred first expansion because they directly improve estimation of realization sensitivity.

### Expansion Gate B — decided at the Week 4 protocol freeze

If method benchmarking supports it without threatening the reserve:

1. add the second discovery method under its own frozen budget;
2. add sensitivity settings recoverable from the same sealed ledgers;
3. add the seven-landmark direct-teacher trajectory.

### Expansion Gate C — activated only after the core analysis freeze

In order:

1. seven-phase student distillation on a prospectively selected uniform grid;
2. the prespecified Fourier interchange experiment;
3. broader sensitivity requiring new searches;
4. entropy estimation;
5. cross-task breadth, theory, and conditional atlases.

Do not add an extension simply because a primary result looks interesting. The trigger must be schedule/compute capacity or a predeclared post-core sequence.

## What changes the calendar materially

Add approximately:

- **1–2 weeks** if hard/soft eligibility requires major optimizer work;
- **1–3 weeks** for a second computationally expensive discovery method;
- **1–2 weeks** for five rather than three eligible students per cell;
- **1–2 weeks** for full rather than sampled independent reproduction;
- **2–4 weeks** for a broad rerun-based sensitivity grid;
- **2–4 weeks** for the Fourier interchange extension.

The eight-week core remains realistic only if the definitive scope is frozen by the end of Week 4 and heavy search remains on the M5 Max.
