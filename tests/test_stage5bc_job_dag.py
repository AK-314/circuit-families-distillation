from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.job_dag import (
    EXECUTABLE_JOB_NODE_TYPES,
    INERT_PLACEHOLDER_NODE_TYPES,
    JOB_NODE_TYPES,
    JobDagError,
    TechnicalJobRegistry,
    build_job_node,
    canonical_job_id,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)
IDENTITY_SPEC_PATH = (
    ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
)


@pytest.fixture(scope="module")
def stage3() -> Stage3AvailabilityIndex:
    registry = json.loads(
        REGISTRY_PATH.read_text(encoding="utf-8")
    )
    return Stage3AvailabilityIndex.from_registry(registry)


@pytest.fixture(scope="module")
def synthetic_complete_hard() -> str:
    spec = json.loads(
        IDENTITY_SPEC_PATH.read_text(encoding="utf-8")
    )
    return spec["synthetic_test_vectors"]["complete_a"]


def _condition(
    stage3: Stage3AvailabilityIndex,
    *,
    condition: str,
    initialization: int | None = None,
) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition=condition,
            student_initialization=initialization,
        ),
        stage3,
    )


def _complete_soft(stage3: Stage3AvailabilityIndex) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="soft_target",
            student_initialization=0,
            discovery_method="synthetic-method-a/v1",
            fidelity_setting="synthetic-fidelity-a/v1",
            component_cap="synthetic-cap-a/v1",
            overlap_setting="synthetic-overlap-a/v1",
        ),
        stage3,
    )


def _chain(
    stage3: Stage3AvailabilityIndex,
    complete_id: str,
    *,
    condition: str = "hard_target",
):
    cache_id = _condition(
        stage3,
        condition=condition,
    )
    student_id = _condition(
        stage3,
        condition=condition,
        initialization=0,
    )

    cache = build_job_node(
        stage3=stage3,
        node_type="teacher_cache",
        condition_id=cache_id,
    )
    training = build_job_node(
        stage3=stage3,
        node_type="training",
        condition_id=student_id,
        dependencies=(cache.job_id,),
    )
    completion = build_job_node(
        stage3=stage3,
        node_type="technical_completion",
        condition_id=student_id,
        dependencies=(training.job_id,),
    )
    eligibility = build_job_node(
        stage3=stage3,
        node_type="future_eligibility",
        condition_id=student_id,
        dependencies=(completion.job_id,),
    )
    discovery = build_job_node(
        stage3=stage3,
        node_type="discovery",
        condition_id=complete_id,
        dependencies=(eligibility.job_id,),
    )
    endpoint = build_job_node(
        stage3=stage3,
        node_type="endpoint",
        condition_id=complete_id,
        dependencies=(discovery.job_id,),
    )
    merge = build_job_node(
        stage3=stage3,
        node_type="merge",
        condition_id=complete_id,
        dependencies=(endpoint.job_id,),
    )
    analysis = build_job_node(
        stage3=stage3,
        node_type="analysis",
        condition_id=complete_id,
        dependencies=(merge.job_id,),
    )

    return (
        cache,
        training,
        completion,
        eligibility,
        discovery,
        endpoint,
        merge,
        analysis,
    )


def test_exact_part_n_node_type_surface() -> None:
    assert JOB_NODE_TYPES == (
        "teacher_cache",
        "training",
        "technical_completion",
        "future_eligibility",
        "discovery",
        "endpoint",
        "merge",
        "analysis",
    )

    assert EXECUTABLE_JOB_NODE_TYPES == {
        "teacher_cache",
        "training",
        "technical_completion",
    }

    assert INERT_PLACEHOLDER_NODE_TYPES == {
        "future_eligibility",
        "discovery",
        "endpoint",
        "merge",
        "analysis",
    }


def test_valid_full_synthetic_chain(
    stage3: Stage3AvailabilityIndex,
    synthetic_complete_hard: str,
) -> None:
    nodes = _chain(
        stage3,
        synthetic_complete_hard,
    )

    registry = TechnicalJobRegistry(
        stage3=stage3,
        nodes=nodes,
    )

    assert registry.node_count == 8

    assert tuple(
        node.node_type
        for node in registry.topological_nodes()
    ) == JOB_NODE_TYPES

    assert tuple(
        node.node_type
        for node in registry.executable_nodes()
    ) == (
        "teacher_cache",
        "training",
        "technical_completion",
    )

    assert tuple(
        node.node_type
        for node in registry.inert_placeholder_nodes()
    ) == (
        "future_eligibility",
        "discovery",
        "endpoint",
        "merge",
        "analysis",
    )

    assert all(
        node.scientific_data is False
        and node.production_eligible is False
        for node in registry.topological_nodes()
    )


def test_registry_serialization_marks_later_nodes_inert(
    stage3: Stage3AvailabilityIndex,
    synthetic_complete_hard: str,
) -> None:
    registry = TechnicalJobRegistry(
        stage3=stage3,
        nodes=_chain(stage3, synthetic_complete_hard),
    )

    mapping = registry.to_mapping()

    assert mapping["scientific_data"] is False
    assert mapping["production_eligible"] is False

    for node in mapping["nodes"]:
        if node["node_type"] in EXECUTABLE_JOB_NODE_TYPES:
            assert node["execution_allowed"] is True
            assert node["inert_placeholder"] is False
        else:
            assert node["execution_allowed"] is False
            assert node["inert_placeholder"] is True


def test_hard_and_soft_job_ids_are_distinct(
    stage3: Stage3AvailabilityIndex,
) -> None:
    hard_cache = _condition(
        stage3,
        condition="hard_target",
    )
    soft_cache = _condition(
        stage3,
        condition="soft_target",
    )

    assert canonical_job_id(
        "teacher_cache",
        hard_cache,
    ) != canonical_job_id(
        "teacher_cache",
        soft_cache,
    )


def test_hard_soft_dependency_collision_is_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    soft_cache_id = _condition(
        stage3,
        condition="soft_target",
    )
    hard_student_id = _condition(
        stage3,
        condition="hard_target",
        initialization=0,
    )

    soft_cache = build_job_node(
        stage3=stage3,
        node_type="teacher_cache",
        condition_id=soft_cache_id,
    )
    hard_training = build_job_node(
        stage3=stage3,
        node_type="training",
        condition_id=hard_student_id,
        dependencies=(soft_cache.job_id,),
    )

    with pytest.raises(
        JobDagError,
        match="hard/soft collision",
    ):
        TechnicalJobRegistry(
            stage3=stage3,
            nodes=(soft_cache, hard_training),
        )


def test_unavailable_ancestry_is_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    unavailable = (
        "cfdid:v1:d3|teacher_seed=0|"
        "phase=pre-grokking|"
        "distillation_condition=hard_target"
    )

    with pytest.raises(
        JobDagError,
        match="invalid or unavailable job ancestry",
    ):
        build_job_node(
            stage3=stage3,
            node_type="teacher_cache",
            condition_id=unavailable,
        )


def test_wrong_condition_depth_is_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    student_id = _condition(
        stage3,
        condition="hard_target",
        initialization=0,
    )

    with pytest.raises(
        JobDagError,
        match="teacher_cache requires condition depth 3",
    ):
        build_job_node(
            stage3=stage3,
            node_type="teacher_cache",
            condition_id=student_id,
        )


def test_depth8_placeholder_must_remain_explicitly_synthetic(
    stage3: Stage3AvailabilityIndex,
) -> None:
    nonsynthetic = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="hard_target",
            student_initialization=0,
            discovery_method="method-a/v1",
            fidelity_setting="fidelity-a/v1",
            component_cap="cap-a/v1",
            overlap_setting="overlap-a/v1",
        ),
        stage3,
    )

    with pytest.raises(
        JobDagError,
        match="explicit synthetic version references",
    ):
        build_job_node(
            stage3=stage3,
            node_type="discovery",
            condition_id=nonsynthetic,
        )


def test_dangling_dependency_is_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    student_id = _condition(
        stage3,
        condition="hard_target",
        initialization=0,
    )

    training = build_job_node(
        stage3=stage3,
        node_type="training",
        condition_id=student_id,
        dependencies=("missing-job",),
    )

    with pytest.raises(
        JobDagError,
        match="dangling job dependency",
    ):
        TechnicalJobRegistry(
            stage3=stage3,
            nodes=(training,),
        )


def test_duplicate_job_id_is_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    cache_id = _condition(
        stage3,
        condition="hard_target",
    )

    cache = build_job_node(
        stage3=stage3,
        node_type="teacher_cache",
        condition_id=cache_id,
    )

    with pytest.raises(
        JobDagError,
        match="duplicate job IDs",
    ):
        TechnicalJobRegistry(
            stage3=stage3,
            nodes=(cache, cache),
        )


def test_cycle_is_rejected_before_execution_semantics(
    stage3: Stage3AvailabilityIndex,
    synthetic_complete_hard: str,
) -> None:
    merge_id = canonical_job_id(
        "merge",
        synthetic_complete_hard,
    )
    analysis_id = canonical_job_id(
        "analysis",
        synthetic_complete_hard,
    )

    merge = build_job_node(
        stage3=stage3,
        node_type="merge",
        condition_id=synthetic_complete_hard,
        dependencies=(analysis_id,),
    )
    analysis = build_job_node(
        stage3=stage3,
        node_type="analysis",
        condition_id=synthetic_complete_hard,
        dependencies=(merge_id,),
    )

    with pytest.raises(
        JobDagError,
        match="cycle detected",
    ):
        TechnicalJobRegistry(
            stage3=stage3,
            nodes=(merge, analysis),
        )



def test_wrong_dependency_type_is_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    cache_id = _condition(
        stage3,
        condition="hard_target",
    )
    student_id = _condition(
        stage3,
        condition="hard_target",
        initialization=0,
    )

    cache = build_job_node(
        stage3=stage3,
        node_type="teacher_cache",
        condition_id=cache_id,
    )
    training = build_job_node(
        stage3=stage3,
        node_type="training",
        condition_id=student_id,
        dependencies=(cache.job_id,),
    )

    # future_eligibility normally requires technical_completion.
    # Supplying a valid training node keeps all other nodes well-formed so
    # this test reaches the intended wrong-edge-type validation.
    eligibility = build_job_node(
        stage3=stage3,
        node_type="future_eligibility",
        condition_id=student_id,
        dependencies=(training.job_id,),
    )

    with pytest.raises(
        JobDagError,
        match="future_eligibility cannot depend on training",
    ):
        TechnicalJobRegistry(
            stage3=stage3,
            nodes=(cache, training, eligibility),
        )


def test_complete_depth8_ancestry_mismatch_is_rejected(
    stage3: Stage3AvailabilityIndex,
    synthetic_complete_hard: str,
) -> None:
    spec = json.loads(
        IDENTITY_SPEC_PATH.read_text(encoding="utf-8")
    )
    changed = spec["synthetic_test_vectors"]["complete_method_changed"]

    # Build the complete valid ancestry through discovery first. This ensures
    # the endpoint edge itself is what is being tested, rather than failing
    # earlier because discovery lacks its required eligibility dependency.
    valid_chain = _chain(
        stage3,
        synthetic_complete_hard,
    )
    cache, training, completion, eligibility, discovery = valid_chain[:5]

    endpoint = build_job_node(
        stage3=stage3,
        node_type="endpoint",
        condition_id=changed,
        dependencies=(discovery.job_id,),
    )

    with pytest.raises(
        JobDagError,
        match="complete synthetic identity",
    ):
        TechnicalJobRegistry(
            stage3=stage3,
            nodes=(
                cache,
                training,
                completion,
                eligibility,
                discovery,
                endpoint,
            ),
        )

def test_topological_order_is_independent_of_input_order(
    stage3: Stage3AvailabilityIndex,
    synthetic_complete_hard: str,
) -> None:
    nodes = _chain(
        stage3,
        synthetic_complete_hard,
    )

    forward = TechnicalJobRegistry(
        stage3=stage3,
        nodes=nodes,
    )
    reverse = TechnicalJobRegistry(
        stage3=stage3,
        nodes=reversed(nodes),
    )

    assert (
        forward.to_mapping()
        == reverse.to_mapping()
    )


def test_soft_branch_can_form_separate_synthetic_chain(
    stage3: Stage3AvailabilityIndex,
) -> None:
    soft_complete = _complete_soft(stage3)

    registry = TechnicalJobRegistry(
        stage3=stage3,
        nodes=_chain(
            stage3,
            soft_complete,
            condition="soft_target",
        ),
    )

    assert registry.node_count == 8
    assert all(
        "distillation_condition=soft_target"
        in node.condition_id
        for node in registry.topological_nodes()
    )
