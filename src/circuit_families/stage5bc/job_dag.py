"""Declarative synthetic Stage 5B/C job dependency graph.

Part N describes dependencies only. It does not allocate output roots, inspect
runtime status, execute jobs, select eligibility, or enter later scientific
stages.

Only teacher-cache, training and technical-completion nodes are marked
executable. All later nodes are inert placeholders.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from circuit_families.stage4_condition_identity import (
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    parse_condition_id,
)

JOB_NODE_TYPES = (
    "teacher_cache",
    "training",
    "technical_completion",
    "future_eligibility",
    "discovery",
    "endpoint",
    "merge",
    "analysis",
)

EXECUTABLE_JOB_NODE_TYPES = frozenset(
    {
        "teacher_cache",
        "training",
        "technical_completion",
    }
)

INERT_PLACEHOLDER_NODE_TYPES = frozenset(
    set(JOB_NODE_TYPES) - EXECUTABLE_JOB_NODE_TYPES
)

_EXPECTED_DEPTH = {
    "teacher_cache": 3,
    "training": 4,
    "technical_completion": 4,
    "future_eligibility": 4,
    "discovery": 8,
    "endpoint": 8,
    "merge": 8,
    "analysis": 8,
}

_ALLOWED_DEPENDENCY_TYPES = {
    "teacher_cache": frozenset(),
    "training": frozenset({"teacher_cache"}),
    "technical_completion": frozenset({"training"}),
    "future_eligibility": frozenset({"technical_completion"}),
    "discovery": frozenset({"future_eligibility"}),
    "endpoint": frozenset({"discovery"}),
    "merge": frozenset({"endpoint"}),
    "analysis": frozenset({"merge"}),
}

_SINGLE_DEPENDENCY_TYPES = frozenset(
    {
        "training",
        "technical_completion",
        "future_eligibility",
        "discovery",
        "endpoint",
    }
)

_MULTI_DEPENDENCY_TYPES = frozenset(
    {
        "merge",
        "analysis",
    }
)

_DEPTH8_VERSION_FIELDS = (
    "discovery_method",
    "fidelity_setting",
    "component_cap",
    "overlap_setting",
)


class JobDagError(ValueError):
    """Raised when the declarative technical DAG is invalid."""


def canonical_job_id(
    node_type: str,
    condition_id: str,
) -> str:
    """Build the deterministic non-path job identity used by Part N."""
    if node_type not in JOB_NODE_TYPES:
        raise JobDagError(
            f"unsupported job node type: {node_type!r}"
        )

    if not isinstance(condition_id, str) or not condition_id:
        raise JobDagError(
            "condition_id must be a non-empty string"
        )

    return f"stage5bc-job/v1::{node_type}::{condition_id}"


@dataclass(frozen=True)
class TechnicalJobNode:
    """One schema-valid synthetic dependency node."""

    node_type: str
    condition_id: str
    dependencies: tuple[str, ...]

    @property
    def job_id(self) -> str:
        return canonical_job_id(
            self.node_type,
            self.condition_id,
        )

    @property
    def execution_allowed(self) -> bool:
        return self.node_type in EXECUTABLE_JOB_NODE_TYPES

    @property
    def inert_placeholder(self) -> bool:
        return self.node_type in INERT_PLACEHOLDER_NODE_TYPES

    @property
    def scientific_data(self) -> bool:
        return False

    @property
    def production_eligible(self) -> bool:
        return False


def _parse_node_identity(
    *,
    stage3: Stage3AvailabilityIndex,
    node_type: str,
    condition_id: str,
):
    try:
        identity = parse_condition_id(
            condition_id,
            stage3,
        )
    except ConditionIdentityError as exc:
        raise JobDagError(
            f"invalid or unavailable job ancestry: {exc}"
        ) from exc

    expected_depth = _EXPECTED_DEPTH[node_type]

    if identity.depth != expected_depth:
        raise JobDagError(
            f"{node_type} requires condition depth "
            f"{expected_depth}, found {identity.depth}"
        )

    if identity.distillation_condition not in {
        "hard_target",
        "soft_target",
    }:
        raise JobDagError(
            f"{node_type} requires hard_target or soft_target ancestry"
        )

    if identity.depth == 8:
        for field in _DEPTH8_VERSION_FIELDS:
            value = getattr(identity, field)

            if (
                not isinstance(value, str)
                or not value.startswith("synthetic-")
                or "/v" not in value
            ):
                raise JobDagError(
                    "Part N depth-8 placeholders must use explicit "
                    f"synthetic version references: field={field}"
                )

    return identity


def build_job_node(
    *,
    stage3: Stage3AvailabilityIndex,
    node_type: str,
    condition_id: str,
    dependencies: Iterable[str] = (),
) -> TechnicalJobNode:
    """Validate and construct one mechanics-only synthetic DAG node."""
    if node_type not in JOB_NODE_TYPES:
        raise JobDagError(
            f"unsupported job node type: {node_type!r}"
        )

    _parse_node_identity(
        stage3=stage3,
        node_type=node_type,
        condition_id=condition_id,
    )

    if isinstance(dependencies, (str, bytes)):
        raise JobDagError(
            "dependencies must be an iterable of job IDs, not a string"
        )

    dependency_tuple = tuple(dependencies)

    if any(
        not isinstance(item, str) or not item
        for item in dependency_tuple
    ):
        raise JobDagError(
            "dependency IDs must be non-empty strings"
        )

    if len(set(dependency_tuple)) != len(dependency_tuple):
        raise JobDagError(
            "a node cannot list a dependency more than once"
        )

    node = TechnicalJobNode(
        node_type=node_type,
        condition_id=condition_id,
        dependencies=dependency_tuple,
    )

    if node.job_id in dependency_tuple:
        raise JobDagError(
            "a node cannot directly depend on itself"
        )

    return node


def _lineage_prefix(identity: Any) -> tuple[Any, ...]:
    return (
        identity.teacher_seed,
        identity.phase,
        identity.distillation_condition,
    )


def _student_prefix(identity: Any) -> tuple[Any, ...]:
    return (
        identity.teacher_seed,
        identity.phase,
        identity.distillation_condition,
        identity.student_initialization,
    )


def _complete_identity(identity: Any) -> tuple[Any, ...]:
    return (
        identity.teacher_seed,
        identity.phase,
        identity.distillation_condition,
        identity.student_initialization,
        identity.discovery_method,
        identity.fidelity_setting,
        identity.component_cap,
        identity.overlap_setting,
    )


class TechnicalJobRegistry:
    """Validated immutable-in-interface synthetic Stage 5B/C DAG."""

    def __init__(
        self,
        *,
        stage3: Stage3AvailabilityIndex,
        nodes: Iterable[TechnicalJobNode],
    ) -> None:
        node_tuple = tuple(nodes)

        if not node_tuple:
            raise JobDagError(
                "technical job registry requires at least one node"
            )

        if any(
            not isinstance(node, TechnicalJobNode)
            for node in node_tuple
        ):
            raise JobDagError(
                "registry nodes must be TechnicalJobNode instances"
            )

        ids = [node.job_id for node in node_tuple]
        duplicate_ids = sorted(
            job_id
            for job_id, count in Counter(ids).items()
            if count > 1
        )

        if duplicate_ids:
            raise JobDagError(
                "duplicate job IDs are forbidden: "
                f"{duplicate_ids!r}"
            )

        by_id = {
            node.job_id: node
            for node in node_tuple
        }

        for node in node_tuple:
            for dependency_id in node.dependencies:
                if dependency_id not in by_id:
                    raise JobDagError(
                        "dangling job dependency: "
                        f"node={node.job_id!r} "
                        f"dependency={dependency_id!r}"
                    )

        self._nodes = by_id
        self._stage3 = stage3

        topological_ids = self._compute_topological_order()
        self._validate_dependency_contracts()
        self._topological_ids = topological_ids

    def _compute_topological_order(self) -> tuple[str, ...]:
        indegree = {
            job_id: len(node.dependencies)
            for job_id, node in self._nodes.items()
        }

        children = {
            job_id: []
            for job_id in self._nodes
        }

        for node in self._nodes.values():
            for dependency_id in node.dependencies:
                children[dependency_id].append(node.job_id)

        ready = sorted(
            job_id
            for job_id, degree in indegree.items()
            if degree == 0
        )
        ordered = []

        while ready:
            job_id = ready.pop(0)
            ordered.append(job_id)

            for child_id in sorted(children[job_id]):
                indegree[child_id] -= 1

                if indegree[child_id] == 0:
                    ready.append(child_id)
                    ready.sort()

        if len(ordered) != len(self._nodes):
            cyclic = sorted(
                job_id
                for job_id, degree in indegree.items()
                if degree > 0
            )
            raise JobDagError(
                "cycle detected in technical job DAG: "
                f"{cyclic!r}"
            )

        return tuple(ordered)

    def _validate_dependency_contracts(self) -> None:
        parsed = {
            job_id: _parse_node_identity(
                stage3=self._stage3,
                node_type=node.node_type,
                condition_id=node.condition_id,
            )
            for job_id, node in self._nodes.items()
        }

        for job_id, node in self._nodes.items():
            allowed_parent_types = _ALLOWED_DEPENDENCY_TYPES[
                node.node_type
            ]

            if node.node_type == "teacher_cache":
                if node.dependencies:
                    raise JobDagError(
                        "teacher_cache nodes must be DAG roots"
                    )
            elif node.node_type in _SINGLE_DEPENDENCY_TYPES:
                if len(node.dependencies) != 1:
                    raise JobDagError(
                        f"{node.node_type} requires exactly one dependency"
                    )
            elif node.node_type in _MULTI_DEPENDENCY_TYPES:
                if not node.dependencies:
                    raise JobDagError(
                        f"{node.node_type} requires at least one dependency"
                    )

            child_identity = parsed[job_id]

            for dependency_id in node.dependencies:
                parent = self._nodes[dependency_id]

                if parent.node_type not in allowed_parent_types:
                    raise JobDagError(
                        f"{node.node_type} cannot depend on "
                        f"{parent.node_type}"
                    )

                parent_identity = parsed[dependency_id]

                if (
                    child_identity.distillation_condition
                    != parent_identity.distillation_condition
                ):
                    raise JobDagError(
                        "hard/soft collision across dependency edge"
                    )

                if (
                    _lineage_prefix(child_identity)
                    != _lineage_prefix(parent_identity)
                ):
                    raise JobDagError(
                        "dependency ancestry does not share "
                        "teacher seed/phase/distillation condition"
                    )

                if (
                    child_identity.depth >= 4
                    and parent_identity.depth >= 4
                    and _student_prefix(child_identity)
                    != _student_prefix(parent_identity)
                ):
                    raise JobDagError(
                        "dependency ancestry does not share "
                        "student initialization"
                    )

                if (
                    child_identity.depth == 8
                    and parent_identity.depth == 8
                    and _complete_identity(child_identity)
                    != _complete_identity(parent_identity)
                ):
                    raise JobDagError(
                        "depth-8 placeholder dependency "
                        "does not share complete synthetic identity"
                    )

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def topological_nodes(self) -> tuple[TechnicalJobNode, ...]:
        """Return deterministic dependency order."""
        return tuple(
            copy.deepcopy(self._nodes[job_id])
            for job_id in self._topological_ids
        )

    def executable_nodes(self) -> tuple[TechnicalJobNode, ...]:
        """Return only nodes authorized for technical execution in this package."""
        return tuple(
            node
            for node in self.topological_nodes()
            if node.execution_allowed
        )

    def inert_placeholder_nodes(self) -> tuple[TechnicalJobNode, ...]:
        """Return later-stage nodes, which carry no execution authority."""
        return tuple(
            node
            for node in self.topological_nodes()
            if node.inert_placeholder
        )

    def get(self, job_id: str) -> TechnicalJobNode:
        """Return a defensive copy of one declared node."""
        try:
            node = self._nodes[job_id]
        except KeyError as exc:
            raise JobDagError(
                f"unknown job ID: {job_id!r}"
            ) from exc

        return copy.deepcopy(node)

    def to_mapping(self) -> dict[str, Any]:
        """Return a deterministic metadata-only registry representation."""
        return {
            "schema_version": "stage5bc-technical-job-dag/v1",
            "scientific_data": False,
            "production_eligible": False,
            "nodes": [
                {
                    "job_id": node.job_id,
                    "node_type": node.node_type,
                    "condition_id": node.condition_id,
                    "dependencies": list(node.dependencies),
                    "execution_allowed": node.execution_allowed,
                    "inert_placeholder": node.inert_placeholder,
                }
                for node in self.topological_nodes()
            ],
        }
