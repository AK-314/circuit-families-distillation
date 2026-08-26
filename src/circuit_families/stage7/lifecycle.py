"""High-level Stage 7 technical lifecycle.

This module owns only the Stage 7 integration order and isolated step roots.
It deliberately does not reimplement Stage 5B/C job identities, dependency
validation, status handling, resume semantics, or atomic job completion.
Those mechanics remain delegated to TechnicalJobRegistry and its accepted
Stage 5B/C companions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

from circuit_families.followup_namespace import logical_root

LIFECYCLE_SCHEMA_VERSION: Final = "stage7-technical-lifecycle/v1"
JOB_LIFECYCLE_DELEGATE: Final = (
    "circuit_families.stage5bc.job_dag.TechnicalJobRegistry"
)

CONDITION_ROLES: Final = frozenset(
    {
        "direct_teacher",
        "hard_target",
        "soft_target",
    }
)


class Stage7LifecycleError(ValueError):
    """Raised when the high-level Stage 7 lifecycle is invalid."""


@dataclass(frozen=True)
class LifecycleStep:
    """One high-level canonical Stage 7 pipeline step."""

    ordinal: int
    step_id: str
    dependencies: tuple[str, ...]
    condition_roles: tuple[str, ...]
    delegates_to: tuple[str, ...]
    output_subdir: str
    output_root: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "step_id": self.step_id,
            "dependencies": list(self.dependencies),
            "condition_roles": list(self.condition_roles),
            "delegates_to": list(self.delegates_to),
            "output_subdir": self.output_subdir,
            "output_root": self.output_root,
        }


_STEP_SPECS: Final = (
    (
        1,
        "teacher_target_cache",
        (),
        ("direct_teacher",),
        ("circuit_families.stage5bc.target_cache",),
        "01-teacher-target-cache",
    ),
    (
        2,
        "student_attempts",
        ("teacher_target_cache",),
        ("hard_target", "soft_target"),
        (
            "circuit_families.stage5bc.student_trainer",
            "circuit_families.stage5bc.student_identity",
            JOB_LIFECYCLE_DELEGATE,
        ),
        "02-student-attempts",
    ),
    (
        3,
        "eligibility",
        ("student_attempts",),
        ("hard_target", "soft_target"),
        (
            "circuit_families.stage6b.records",
            "circuit_families.stage6c.eligibility",
            "circuit_families.stage6c.records",
        ),
        "03-eligibility",
    ),
    (
        4,
        "passed_only_sealing",
        ("eligibility",),
        ("hard_target", "soft_target"),
        (
            "circuit_families.stage6b.records",
            "circuit_families.stage6c.records",
            "circuit_families.stage5bc.job_outputs",
            "circuit_families.stage5bc.job_status",
        ),
        "04-passed-only-sealing",
    ),
    (
        5,
        "discovery",
        ("teacher_target_cache", "passed_only_sealing"),
        ("direct_teacher", "hard_target", "soft_target"),
        (
            "circuit_families.stage6d.adapters",
            "circuit_families.stage6d.greedy",
            "circuit_families.stage6d.diversity",
            "circuit_families.stage6d.budgets",
            JOB_LIFECYCLE_DELEGATE,
        ),
        "05-discovery",
    ),
    (
        6,
        "exact_ledgers_endpoint1",
        ("discovery",),
        ("direct_teacher", "hard_target", "soft_target"),
        (
            "circuit_families.stage6a.ledger",
            "circuit_families.stage6a.budget",
            "circuit_families.stage6a.endpoint",
        ),
        "06-exact-ledgers-endpoint1",
    ),
    (
        7,
        "endpoint2",
        ("exact_ledgers_endpoint1",),
        ("direct_teacher", "hard_target", "soft_target"),
        (
            "circuit_families.stage6e.packing",
            "circuit_families.stage6e.records",
        ),
        "07-endpoint2",
    ),
    (
        8,
        "teacher_seed_inventory",
        ("eligibility", "endpoint2"),
        ("direct_teacher", "hard_target", "soft_target"),
        (
            "circuit_families.stage4_schema_analysis",
            "circuit_families.stage4_schema_graph",
            "circuit_families.stage5bc.serial_merge",
        ),
        "08-teacher-seed-inventory",
    ),
    (
        9,
        "analysis_report",
        ("teacher_seed_inventory",),
        ("direct_teacher", "hard_target", "soft_target"),
        (
            "circuit_families.stage5d.cells",
            "circuit_families.stage5d.contrasts",
            "circuit_families.stage5d.population",
            "circuit_families.stage5d.outputs",
        ),
        "09-analysis-report",
    ),
    (
        10,
        "independent_reproduction",
        ("analysis_report",),
        ("direct_teacher", "hard_target", "soft_target"),
        (
            "circuit_families.stage5d.outputs",
            "circuit_families.stage4_schema_analysis",
        ),
        "10-independent-reproduction",
    ),
)


def _validate_excluded_output_root(output_root: str) -> PurePosixPath:
    if not isinstance(output_root, str) or not output_root:
        raise Stage7LifecycleError("output_root must be a non-empty string")

    root_path = PurePosixPath(output_root)
    approved = logical_root("excluded_development")

    if root_path.is_absolute():
        raise Stage7LifecycleError(
            "Stage 7A output_root must be portable and repository-relative"
        )

    if ".." in root_path.parts:
        raise Stage7LifecycleError(
            "Stage 7A output_root may not contain path traversal"
        )

    if root_path == approved:
        raise Stage7LifecycleError(
            "Stage 7A requires an isolated subdirectory beneath "
            "followup/excluded_development"
        )

    if root_path.parts[: len(approved.parts)] != approved.parts:
        raise Stage7LifecycleError(
            "Stage 7A output_root must remain beneath "
            "followup/excluded_development"
        )

    return root_path


@dataclass(frozen=True)
class Stage7Lifecycle:
    """Validated deterministic high-level Stage 7 technical DAG."""

    output_root: str
    job_lifecycle_reference_id: str
    steps: tuple[LifecycleStep, ...]

    def __post_init__(self) -> None:
        _validate_excluded_output_root(self.output_root)

        if (
            not isinstance(self.job_lifecycle_reference_id, str)
            or not self.job_lifecycle_reference_id
        ):
            raise Stage7LifecycleError(
                "job_lifecycle_reference_id must be non-empty"
            )

        if len(self.steps) != 10:
            raise Stage7LifecycleError(
                "Stage 7 lifecycle requires exactly ten canonical steps"
            )

        step_ids = tuple(step.step_id for step in self.steps)

        if len(set(step_ids)) != len(step_ids):
            raise Stage7LifecycleError(
                "duplicate Stage 7 step IDs are forbidden"
            )

        expected_ordinals = tuple(range(1, 11))
        actual_ordinals = tuple(step.ordinal for step in self.steps)

        if actual_ordinals != expected_ordinals:
            raise Stage7LifecycleError(
                "Stage 7 step ordinals must be exactly 1..10"
            )

        seen: set[str] = set()
        roots: set[str] = set()

        for step in self.steps:
            if not step.condition_roles:
                raise Stage7LifecycleError(
                    f"{step.step_id} must declare condition roles"
                )

            unknown_roles = set(step.condition_roles) - CONDITION_ROLES

            if unknown_roles:
                raise Stage7LifecycleError(
                    f"{step.step_id} has unknown roles: "
                    f"{sorted(unknown_roles)!r}"
                )

            for dependency in step.dependencies:
                if dependency not in seen:
                    raise Stage7LifecycleError(
                        f"{step.step_id} dependency is not an earlier step: "
                        f"{dependency!r}"
                    )

            if step.output_root in roots:
                raise Stage7LifecycleError(
                    "Stage 7 lifecycle output roots must be isolated"
                )

            expected_root = (
                PurePosixPath(self.output_root) / step.output_subdir
            ).as_posix()

            if step.output_root != expected_root:
                raise Stage7LifecycleError(
                    f"{step.step_id} output root is not deterministic"
                )

            roots.add(step.output_root)
            seen.add(step.step_id)

    def topological_steps(self) -> tuple[LifecycleStep, ...]:
        """Return the validated canonical dependency order."""
        return self.steps

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "scientific_data": False,
            "production_eligible": False,
            "excluded_development": True,
            "delegates_job_lifecycle_to": JOB_LIFECYCLE_DELEGATE,
            "job_lifecycle_reference_id": self.job_lifecycle_reference_id,
            "output_root": self.output_root,
            "steps": [step.to_mapping() for step in self.steps],
        }


def build_stage7_lifecycle(
    *,
    output_root: str,
    job_lifecycle_reference_id: str,
) -> Stage7Lifecycle:
    """Build the fixed high-level integration DAG without duplicating Stage 5C."""
    root_path = _validate_excluded_output_root(output_root)

    steps = tuple(
        LifecycleStep(
            ordinal=ordinal,
            step_id=step_id,
            dependencies=dependencies,
            condition_roles=condition_roles,
            delegates_to=delegates_to,
            output_subdir=output_subdir,
            output_root=(root_path / output_subdir).as_posix(),
        )
        for (
            ordinal,
            step_id,
            dependencies,
            condition_roles,
            delegates_to,
            output_subdir,
        ) in _STEP_SPECS
    )

    return Stage7Lifecycle(
        output_root=root_path.as_posix(),
        job_lifecycle_reference_id=job_lifecycle_reference_id,
        steps=steps,
    )
