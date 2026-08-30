from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from circuit_families.stage12p3 import (
    CampaignManifest,
    ExpectedArtifact,
    HashBoundReference,
    LogicalJobSpec,
    OutputContract,
    PriorityClass,
    ResourceClass,
    SchedulerObservation,
    SchedulerSubmission,
    Stage12P3ContractError,
    compile_campaign,
    logical_job_from_mapping,
    validate_operational_status,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def reference(name: str, sha256: str = SHA_A) -> HashBoundReference:
    return HashBoundReference(
        reference=f"synthetic://{name}",
        sha256=sha256,
        interface_version=f"{name}/v1",
    )


def resource() -> ResourceClass:
    return ResourceClass(
        reference="resource/technical-small/v1",
        cpu_units=1,
        accelerator_capability=None,
        memory_bytes=1024,
        scratch_bytes=1024,
        walltime_seconds=10,
    )


def priority() -> PriorityClass:
    return PriorityClass(reference="priority/technical/v1", dispatch_rank=10)


def job(
    family: str,
    *,
    dependencies: tuple[str, ...] = (),
    config_sha256: str = SHA_A,
) -> LogicalJobSpec:
    return LogicalJobSpec(
        family=family,
        producer_interface_version=f"{family}/v1",
        dependencies=dependencies,
        expected_inputs=(reference(f"{family}-input"),),
        payload_reference=reference(f"{family}-payload"),
        config_reference=reference(f"{family}-config", config_sha256),
        output_contract=OutputContract(
            manifest_relative_path=f"manifests/{family}.json",
            manifest_schema_version=f"{family}-output/v1",
            artifacts=(ExpectedArtifact(f"artifacts/{family}.json", "application/json"),),
        ),
        resource_class_reference="resource/technical-small/v1",
        priority_class_reference="priority/technical/v1",
        protected_tier="tier1-technical",
        retry_seed_namespace_reference="seed-derivation/v1",
    )


def manifest(*jobs: LogicalJobSpec) -> CampaignManifest:
    return CampaignManifest(
        manifest_reference=reference("campaign-manifest"),
        jobs=jobs,
        resource_classes=(resource(),),
        priority_classes=(priority(),),
    )


def test_logical_identity_excludes_execution_coordinates_and_is_order_stable() -> None:
    left = job("producer")
    right = replace(
        left,
        expected_inputs=tuple(reversed(left.expected_inputs)),
        dependencies=tuple(reversed(left.dependencies)),
    )
    assert left.job_id == right.job_id
    assert "attempt" not in left.identity_payload()
    assert "worker" not in left.identity_payload()
    assert "backend" not in left.identity_payload()


def test_config_hash_and_family_change_logical_identity() -> None:
    original = job("producer")
    assert job("producer", config_sha256=SHA_B).job_id != original.job_id
    assert job("other-producer").job_id != original.job_id


def test_serialized_identity_and_hash_mismatches_are_rejected() -> None:
    original = job("producer")
    mapping = original.to_mapping()
    assert logical_job_from_mapping(mapping) == original
    wrong_id = copy.deepcopy(mapping)
    wrong_id["job_id"] = "0" * 64
    with pytest.raises(Stage12P3ContractError, match="identity/hash mismatch"):
        logical_job_from_mapping(wrong_id)
    stale_id = copy.deepcopy(mapping)
    stale_id["config_reference"]["sha256"] = SHA_B
    with pytest.raises(Stage12P3ContractError, match="identity/hash mismatch"):
        logical_job_from_mapping(stale_id)


def test_tiny_fan_out_fan_in_and_disconnected_subgraph_compile_deterministically() -> None:
    root = job("p1-like-producer")
    left = job("p2-like-consumer", dependencies=(root.job_id,))
    right = job("independent-direct-teacher-discovery")
    reducer = job("sealed-ledger-reducer", dependencies=(left.job_id, right.job_id))
    forward = compile_campaign(manifest(root, left, right, reducer))
    reverse = compile_campaign(manifest(reducer, right, left, root))
    assert forward.campaign_id == reverse.campaign_id
    assert forward.topological_job_ids == reverse.topological_job_ids
    assert set(forward.ready_job_ids(set())) == {root.job_id, right.job_id}
    assert reducer.job_id in forward.ready_job_ids({root.job_id, left.job_id, right.job_id})


def test_dangling_dependency_and_duplicate_output_are_rejected() -> None:
    dangling = job("dangling", dependencies=("0" * 64,))
    with pytest.raises(Stage12P3ContractError, match="dangling"):
        compile_campaign(manifest(dangling))

    first = job("first")
    second = replace(job("second"), output_contract=first.output_contract)
    with pytest.raises(Stage12P3ContractError, match="duplicate campaign output"):
        compile_campaign(manifest(first, second))


def test_paths_and_boundary_flags_are_strict() -> None:
    with pytest.raises(Stage12P3ContractError, match="escape"):
        ExpectedArtifact("../escape.json", "application/json")
    with pytest.raises(Stage12P3ContractError, match="scientific_data=false"):
        replace(resource(), scientific_data=True)


def test_backend_cannot_change_identity_or_equate_finished_with_sealed_success() -> None:
    logical = job("producer")
    valid = SchedulerSubmission(
        logical_job_id=logical.job_id,
        backend_name="generic-array/v1",
        backend_job_id="array-42",
        array_index=0,
    )
    valid.validate_for(logical)

    changed = replace(valid, logical_job_id="f" * 64)
    with pytest.raises(Stage12P3ContractError, match="changed logical job identity"):
        changed.validate_for(logical)

    observation = SchedulerObservation(
        logical_job_id=logical.job_id,
        backend_job_id="array-42",
        scheduler_state="finished",
        observed_sequence=1,
    )
    assert observation.sealed_success is False


def test_operational_status_rejects_scientific_direction_and_rank() -> None:
    validate_operational_status(
        {"counts": {"tier1-technical": {"succeeded": 1}}, "scientific_data": False}
    )
    with pytest.raises(Stage12P3ContractError, match="forbidden scientific field"):
        validate_operational_status({"families": {"producer": {"effect_direction": "up"}}})
    with pytest.raises(Stage12P3ContractError, match="forbidden scientific field"):
        validate_operational_status({"condition_comparison": ["a", "b"]})
