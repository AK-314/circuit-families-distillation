# Adam handoff — Stage 14 Symbolica practice-node probe

## Purpose

Run one bounded, synthetic technical probe on one CUDA practice node before Alex
receives the two-node allocation. This is not a scientific run and cannot launch
Stage 15.

The probe answers four practical questions:

1. Can the locked repository environment install and see the allocated GPU?
2. Do CPU and CUDA pass the prospective numerical correctness checks?
3. How quickly does this hardware execute an actual full-domain student update
   and an actual full-domain, model-in-the-loop 516-gate objective step?
4. How quickly does one CPU process perform an actual full-domain exact mask
   evaluation, and what does that imply for the provisional 16-GPU/64-CPU
   protected-core schedule?

It uses freshly initialized deterministic models and the public modular-addition
domain. It does not read teachers, students, checkpoints, registered outputs, or
private predecessor files.

## What Adam receives

- The public GitHub repository URL.
- One exact commit SHA supplied by Alex.
- One terminal block supplied by Alex.

No separate input archive, credentials file, model checkpoint, or dataset is
required.

## What the command does

- Clones and checks out the exact source commit.
- Verifies the tracked checkout is clean.
- Records the node, GPU, driver, disk and Python facts.
- Installs the repository's frozen environment with `uv.lock`.
- Runs CPU/CUDA correctness probes.
- Runs short representative training, model-in-loop gate, and exact-evaluation
  timings.
- Writes a hash-authenticated JSON report and complete terminal log.
- Produces one archive named `stage14-symbolica-probe-output.tar.gz`.

The old Stage 14-B matrix microfixture is retained only as a numerical
correctness check. Its speed is not used as campaign throughput.

## Expected resource envelope

- Preferred node: one H100 GPU with ordinary host CPUs.
- Dependency download: approximately 3–6 GB on a cold cache.
- Installed environment: approximately 8–15 GB.
- Probe working/output data beyond the environment: normally well below 100 MB.
- Returned archive: normally below 5 MB because it contains JSON and text only.
- Expected elapsed time: usually 15–45 minutes on a fresh node, most of which is
  environment installation. A slow network or package cache can extend this.

## What Adam returns

Exactly one file:

`stage14-symbolica-probe-output.tar.gz`

Adam should also copy the final four terminal lines containing the exit status,
archive path, SHA-256, and byte count. If the command fails, the same archive
still contains the complete failure log and should be returned unchanged.

## Interpretation boundary

Passing the practice probe means the software and one-node hardware path work
and provides planning measurements. It does not prove two-node scaling, queue or
filesystem behaviour, long-run throughput, or protected-core feasibility. Alex
must inspect the returned evidence and complete the Stage 14 launch gate before
any definitive execution.

