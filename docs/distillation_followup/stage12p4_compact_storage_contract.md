# Stage 12-P4 compact storage and verified-export contract

**Classification:** synthetic technical storage only
**Scientific data:** false
**Production eligible:** false
**Production quota, codec, retention, destination, and credentials:** unresolved

## Reuse audit

Stage 12-P4 adds a storage and transfer layer; it does not add a scientific
record system. The implementation reuses the following merged interfaces.

| Producer or consumer | Reused identity/evidence | P4 treatment |
|---|---|---|
| Stage 12-P3 | `LogicalJobSpec.job_id`, attempt root, `OutputContract`, and `stage12p3-sealed-output/v1` manifest | `ProducerEvidence` binds the exact job, attempt, output-contract hash, sealed-manifest hash, and declared relative source path. Worker, retry, scheduler, backend, destination, and transfer-attempt fields remain outside logical identity. |
| Stage 12-P1 | teacher attempt, checkpoint, cache, trajectory, phase, and sealed teacher records | Compact objects wrap canonical records and hashes. P4 does not load registered teachers or change phase/teacher identity. |
| Stage 12-P2 | architecture-bound rolling checkpoints, dense outputs, eligibility, failures, and discovery-release seal | The scratch manager is an adapter over existing checkpoint semantics. It never evicts the only valid recovery boundary or a protected final artifact. |
| Stage 12-R1 | proposal identity, binary masks, exact-ledger bridge, optimizer checkpoints, native-budget/failure records | Masks may be bit-packed only after binding their component universe and basis. Proposal, exact-evaluation, failure, unavailable, and censored semantics are retained as rows. |
| Stage 12-R2 | basis hash, ordered component descriptors, `BasisMask`, rotation/block relationships, parameter weights | Mask storage records the basis identity, ordered universe, component count, padding rule, and logical mask hash. Cross-basis reuse is rejected. |
| Stage 4 | canonical paths, SHA-256, closed schemas, identity envelopes, and seed separation | P4 uses portable relative POSIX paths, canonical JSON, closed fields, and SHA-256. It does not modify Stage 4 identity or seed contracts. |
| Stage 5B-C | cache/checkpoint evidence, atomic job outputs, DAG closure, serial merge, failure/status records | P4 follows atomic partial-to-final publication and deterministic ordering. Scheduler completion never implies a sealed or exported artifact. |
| Stage 6A | exact-evaluation row semantics, mask identity, exact-budget charge, termination/censoring | The ledger schema can reconstruct the canonical row mapping exactly. Negative-value policy is explicit and nonfinite values are rejected. |
| Stage 6E | basis-bound exact evidence, deduplication, packing proof and endpoint records | Merge deduplicates only byte-identical rows for the same declared key. It never uses effect direction, arrival time, or “latest wins.” |
| Stage 7 | lifecycle, explicit unavailable/failed inventory, reproduction and integrity evidence | Scratch, merge, export, and verification reports retain explicit terminal states and technical boundaries. |

RD-014 remains open for Stages 13-14. Every quota, warning threshold, reserve,
retention cadence, codec level, chunk size, and destination reference used here
is injected synthetic technical configuration.

## Required separations

1. Canonical logical records and identities are producer-owned.
2. Storage encoding covers bit packing, row arrays, compression, and chunks.
3. Scratch inventory includes complete, partial, stale, failed, and protected
   local objects.
4. A sealed artifact manifest contains exact accepted paths, byte lengths, and
   SHA-256 hashes.
5. Transfer state records verified prefixes, destination reference, attempts,
   and publication state outside logical identity.
6. Destination verification independently rereads the published manifest and
   every object, then compares an exact inventory.

## Versioned outer contract

`stage12p4-storage-object/v1` is a closed wrapper containing:

- artifact class and producer interface version;
- exact P3 logical-job, attempt, output-contract, sealed-manifest, and source
  references;
- logical schema and ordered fields;
- source byte length and SHA-256;
- one unambiguous storage encoding and injected codec/chunk references;
- compact byte length and SHA-256;
- injected scratch quota and retention references;
- lifecycle state from planned through verified/conflict/failed;
- `scientific_data=false` and `production_eligible=false`.

Unknown fields, unsafe paths, cross-job reuse, missing producer evidence,
hash/length disagreement, schema drift, ambiguous encoding, scientific
relabeling, and production eligibility are errors.

## Compact formats

### Masks

`stage12p4-compact-mask/v1` stores bits in declared component order, most
significant bit first. Low-order padding bits in the final byte must be zero.
The header binds component-universe reference, basis identity, component count,
padding rule, packed length/hash, and canonical logical-mask hash. Empty masks
are supported where the producer contract permits them.

### Metric and exact-evaluation ledgers

`stage12p4-compact-ledger/v1` contains one canonical header followed by JSONL
row arrays in declared field order. Field name/type/nullability/negative policy
and unique key order are closed by `stage12p4-ledger-schema/v1`. Writers spool
rows to a file-backed partial, maintain byte/hash/count evidence incrementally,
and atomically finalize one deterministic object. Readers stream rows and
verify count, byte length, content hash, field types, and strict key order.

The supported explicit row states are `complete`, `failed`, `unavailable`, and
`censored`; absence is not used as a substitute for those states.

### Compression

The portable implementation supports no compression and deterministic gzip.
Gzip uses a fixed header with zero timestamp, no filename/comment, fixed OS
marker, injected level, raw DEFLATE, and deterministic CRC/length trailer. No
third-party compression dependency is required.

## Quota and rolling retention

Quota projection includes every current regular file (including partials and
manifests), new staging bytes, new manifest bytes, and the injected atomic
reserve. Exact-boundary completion is allowed. Unsafe completion returns an
explicit retryable `insufficient_finalization_reserve` record before writing a
sealed object.

Retention deterministically keeps the newest valid injected count plus every
explicitly protected generation/class. An invalid newest checkpoint never
displaces an older valid recovery boundary. Declared stale partials and
unprotected excess/invalid generations are eligible for cleanup, and each
decision is recorded. A nonblocking claim rejects concurrent cleanup.

## Merge and export

Serial merge requires a closed shard list and identical schema, logical
campaign, basis, and producer versions. Every source object hash is verified
before consumption. A stable k-way merge orders by declared row keys. Identical
rows are deduplicated with source provenance; different rows sharing a key are
conflicts. Empty and unavailable shards remain in the source closure.

Bundles use sorted fixed-metadata USTAR, injected deterministic compression,
and optional deterministic chunks. They contain no timestamp, host path, uid,
gid, or filesystem-order metadata. Local export verifies any existing partial
prefix byte-for-byte, stages under `.partial`, refuses conflicting published
objects, and publishes the destination manifest last. Verification rereads all
destination bytes and rejects missing, truncated, stale, extra, duplicate, or
conflicting identities. Source artifacts are never deleted.

## Validation boundary

`scripts/validate_stage12p4.py --validate-only --output-root <explicit-root>`
runs only deterministic synthetic fixtures beneath the supplied root. It does
not use network services, credentials, registered artifacts, teacher/student
checkpoints, private predecessor data, or production settings. Its measured
size report is technical evidence and neither predicts production footprint
nor freezes RD-014.
