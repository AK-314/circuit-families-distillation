# Follow-up namespace

Namespace version: `circuit-families-distillation/v1`

This namespace belongs only to the `circuit-families-distillation` successor
study. It does not rename, relocate, or reinterpret any artifact from the
`circuit-families` predecessor.

## Portability rule

Canonical identities must not depend on one collaborator's absolute filesystem
path. Repository identity, immutable Git commit, namespace version, relative
path, run identity, and cryptographic hash are the portable identifiers.

Absolute local paths may appear only in local verification output or machine
records. They are never the sole canonical scientific identity of an artifact.

## Successor namespace root

All new follow-up commands that create study artifacts must write beneath the
dedicated repository-relative root:

`followup/`

The inherited top-level predecessor-style roots are legacy locations and are
not approved output roots for new follow-up commands:

- `checkpoints/`
- `results/`
- `manifests/`
- `figures/`

Historical predecessor-compatible code may continue to understand those roots
for regression and provenance purposes. Stage 1 does not retrofit or redirect
that historical machinery. New follow-up commands must use the namespace
validator introduced for this successor study.

## Logical roots

| Logical name | Repository-relative root | Classification | Git policy |
|---|---|---|---|
| follow-up configs | `followup/configs/` | frozen or prospective small configuration records | tracked |
| local scratch | `followup/local/scratch/` | disposable machine-local intermediate state | ignored / never canonical |
| teacher caches | `followup/artifacts/teacher_cache/` | large reusable teacher outputs and caches | external / ignored |
| student checkpoints | `followup/artifacts/student_checkpoints/` | large sealed or intermediate model checkpoints | external / ignored |
| student outputs | `followup/artifacts/student_outputs/` | large dense student outputs and model-derived caches | external / ignored |
| raw discovery output | `followup/artifacts/discovery_raw/` | raw proposals, masks, ledgers, worker output | external / ignored |
| manifests | `followup/manifests/` | small canonical provenance, freeze, inventory, and link records | tracked when reviewable and non-secret |
| reviewed endpoint tables | `followup/reviewed/tables/` | small reviewed tables produced only at authorized later stages | tracked when explicitly approved |
| notes | `followup/reviewed/notes/` | reviewed methodological, audit, and analysis notes | tracked |
| figures | `followup/reviewed/figures/` | reviewed publication or audit figures produced only at authorized later stages | tracked when small and explicitly approved |
| archives | `followup/artifacts/archives/` | sealed large archives and archive payloads | external / ignored |
| excluded development output | `followup/excluded_development/` | development outputs that must never enter primary production analysis | ignored except for small tracked exclusion registers/manifests |
| reproduction bundles | `followup/artifacts/reproduction_bundles/` | large isolated reproduction packages | external / ignored |

## Storage classes

### Git-tracked small records

Git may contain:

- follow-up configs;
- schemas and validators;
- small manifests and inventories;
- predecessor-link records;
- exclusion registers;
- reviewed notes;
- reviewed small tables and figures;
- documentation and tests.

Tracked records must use repository-relative portable identities where a path is
needed and cryptographic hashes where immutability matters.

### Local or externally stored large artifacts

Git must not contain:

- checkpoints;
- teacher or student dense-output caches;
- raw discovery proposals;
- exact-evaluation ledgers when large;
- bulk arrays;
- compressed scientific archives;
- reproduction bundles;
- disposable scratch data.

Stage 1 defines their logical namespace only. It does not configure, select, or
require any remote object-storage provider.

## Preservation boundary

The predecessor repository is immutable and is never an authorized follow-up
output root. A follow-up output path is invalid if it:

1. is the predecessor repository root;
2. is inside the predecessor repository;
3. resolves through a symlink into the predecessor repository;
4. escapes the authorized follow-up root by traversal or symlink;
5. points to an inherited predecessor-style output root for a new follow-up
   command;
6. otherwise causes a successor artifact to masquerade as a predecessor
   artifact.

The successor source repository and the predecessor repository are distinct
identities even if they contain byte-identical inherited source files.

## Temporary test roots

Tests may construct an isolated synthetic successor repository in a temporary
directory. The same namespace version and collision rules apply there.

Temporary test roots are permitted so long as they are explicitly supplied as
the synthetic successor root and do not resolve into the real predecessor or
outside the authorized synthetic follow-up namespace.

## Stage boundary

This specification defines storage and provenance boundaries only.

It does not define:

- teacher phase selections;
- condition identities;
- student schemas;
- predictive fidelity;
- distillation parameters;
- circuit-discovery rules;
- endpoint definitions;
- artifact-store provider configuration;
- any scientific comparison or new analysis.

Those remain governed by their later implementation stages.
