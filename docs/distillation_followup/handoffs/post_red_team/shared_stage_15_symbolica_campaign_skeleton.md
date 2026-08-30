# Shared Stage 15 Symbolica campaign handoff — skeleton only

## Do not execute this file yet

This is a template for the single Stage 15 production task. Stages 11–14 must
materialize a final handoff with every placeholder replaced by committed,
verified values. The materialized handoff must be reviewed at the exact
integrated SHA and merged to `main` before launch.

Required substitutions:

```text
<FINAL_STAGE14_SHA>
<PROTOCOL_FREEZE_HASH>
<PRODUCTION_MANIFEST_PATH_AND_HASH>
<CONTAINER_ID_AND_DIGEST>
<INPUT_BUNDLE_PATH_AND_HASH>
<RESOURCE_MANIFEST_PATH_AND_HASH>
<SCHEDULER_ADAPTER_AND_QUEUE>
<SCRATCH_ROOT>
<PERSISTENT_EXPORT_ROOT>
<TIER1_COMPLETENESS_RULE>
<TIER2_MINIMUM_RULE>
<FINAL_WINDOW_START_RULE>
<STATUS_COMMAND>
<STOP_COMMAND>
<EXPORT_AND_VERIFY_COMMAND>
```

## Mission

Run one automated, scheduler-managed scientific campaign. Teacher generation,
student training, eligibility, discovery, nulls, calibration, architecture,
basis, Fourier, recomputation, and export are internal DAG job families—not
separate stage conversations.

## Operating protocol

- Use one Chat-mode task for all four days.
- Give exactly one terminal block for each of the three gates or a genuine
  incident response.
- Routine jobs run unattended through the scheduler.
- Status updates report completeness, failures, resources, and storage without
  highlighting comparative effect directions before the primary-completeness
  gate.
- No setting in the frozen manifest may be changed because of observed
  scientific outcomes.
- Infrastructure repairs may restart frozen jobs; scientific redesign requires
  aborting Stage 15 and issuing a prospective amendment.

## Gate 15.1 — launch

Verify exact SHA, clean checkout, manifest/config/container/input hashes,
hardware, quotas, scheduler, and export destination. Run the tiny complete
pipeline and compare it to the Stage 14 reference. Only then release protected
arrays.

Required output:

```text
GATE_15_1=PASS
PRODUCTION_RELEASED=YES
```

## Automated campaign

```text
qualification
  ↓
teacher training / phase registry
  ↓
target caches
  ↓
student training / eligibility / sealing
  ↓
direct-teacher + eligible-student discovery
  ↓
primary endpoints + packing calibration
  ↓
architecture / basis / method / tractable / Fourier jobs
  ↓
independent recomputation / inventories / compact export
```

Frozen scheduling priority:

1. qualification and integrity;
2. teacher/student prerequisites;
3. complete protected mod-113 Tier 1;
4. independent method coverage;
5. packing calibration/null minimum;
6. tractable calibration;
7. architecture minimum;
8. Fourier and all controls;
9. core basis sensitivity;
10. optional breadth.

## Gate 15.2 — primary completeness

Inspect planned/terminal counts, eligibility/failure accounting, runtime,
memory, storage, retries, and projected protected completion. Do not reallocate
according to effect direction. Shed work only in the frozen reverse-priority
order.

Required output:

```text
GATE_15_2=PASS
TIER1_SECURE=YES
REMAINING_CAPACITY_RELEASE=<frozen decision>
```

## Gate 15.3 — exit

At the frozen final-window boundary, stop new optional jobs, close protected
terminal states, independently recompute endpoints, seal inventories, compact,
export, and verify destination hashes. Delete nothing from scratch until the
destination verification passes and Stage 16 accepts custody.

Required output:

```text
GATE_15_3=PASS
STAGE15_STATUS=COMPLETE_AT_AUDIT_HANDOFF
EXPORTED_ARTIFACTS_VERIFIED=YES
STAGE16_STARTED=NO
```

## Failure boundary

Infrastructure failure may produce `PAUSED` or a protocol-valid terminal job
state. It must never silently shrink the protected primary estimand. If the
Tier 1 completeness rule cannot be met, Stage 15 closes as incomplete and the
resource-contingent school-Mac recovery branch activates under its already
frozen manifest.
