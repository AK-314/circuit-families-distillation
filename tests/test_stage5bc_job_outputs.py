from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.job_dag import (
    TechnicalJobRegistry,
    build_job_node,
)
from circuit_families.stage5bc.job_outputs import (
    JOB_COMPLETION_FILENAME,
    JobArtifactEvidence,
    JobOutputError,
    atomic_write_job_file,
    bind_job_output_root,
    canonical_job_relative_identity,
    validate_ignored_scratch_root,
    write_job_completion,
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
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return Stage3AvailabilityIndex.from_registry(raw)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _technical_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()

    _git("init", "-q", cwd=repo)

    (repo / ".gitignore").write_text(
        "scratch/\n"
        "private-predecessor/\n"
        "scratch-real/\n"
        "scratch-link/\n",
        encoding="utf-8",
    )
    (repo / "tracked.txt").write_text(
        "tracked\n",
        encoding="utf-8",
    )

    _git(
        "add",
        ".gitignore",
        "tracked.txt",
        cwd=repo,
    )

    scratch = repo / "scratch"
    scratch.mkdir()

    return repo, scratch


def _conditions(
    stage3: Stage3AvailabilityIndex,
) -> tuple[str, str, str]:
    spec = json.loads(
        IDENTITY_SPEC_PATH.read_text(encoding="utf-8")
    )

    cache = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="hard_target",
        ),
        stage3,
    )
    student = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="hard_target",
            student_initialization=0,
        ),
        stage3,
    )
    complete = spec["synthetic_test_vectors"]["complete_a"]

    return cache, student, complete


def _all_nodes(
    stage3: Stage3AvailabilityIndex,
):
    cache_id, student_id, complete_id = _conditions(stage3)

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


def test_explicit_scratch_must_exist_and_be_git_ignored(
    tmp_path: Path,
) -> None:
    repo, scratch = _technical_repo(tmp_path)

    assert validate_ignored_scratch_root(
        repository_root=repo,
        scratch_root=scratch,
    ) == scratch

    not_ignored = repo / "not-ignored"
    not_ignored.mkdir()

    with pytest.raises(
        JobOutputError,
        match="explicitly Git-ignored",
    ):
        validate_ignored_scratch_root(
            repository_root=repo,
            scratch_root=not_ignored,
        )

    missing = repo / "scratch" / "missing"

    with pytest.raises(
        JobOutputError,
        match="must already exist",
    ):
        validate_ignored_scratch_root(
            repository_root=repo,
            scratch_root=missing,
        )



def test_repository_parent_alias_does_not_false_positive_as_escape(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()

    repo, scratch = _technical_repo(real_parent)

    alias_parent = tmp_path / "alias-parent"
    os.symlink(
        real_parent,
        alias_parent,
    )

    alias_repo = alias_parent / "repo"
    alias_scratch = alias_repo / "scratch"

    validated = validate_ignored_scratch_root(
        repository_root=alias_repo,
        scratch_root=alias_scratch,
    )

    assert validated == scratch.resolve(strict=True)
    assert validated.is_dir()
    assert not validated.is_symlink()

def test_all_declared_jobs_receive_distinct_canonical_relative_identities(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    nodes = _all_nodes(stage3)

    registry = TechnicalJobRegistry(
        stage3=stage3,
        nodes=nodes,
    )

    roots = [
        bind_job_output_root(
            repository_root=repo,
            scratch_root=scratch,
            node=node,
        )
        for node in registry.topological_nodes()
    ]

    assert len(roots) == 8
    assert len(
        {
            root.relative_identity
            for root in roots
        }
    ) == 8
    assert len(
        {
            root.path
            for root in roots
        }
    ) == 8

    for node, root in zip(
        registry.topological_nodes(),
        roots,
        strict=True,
    ):
        assert (
            root.relative_identity
            == canonical_job_relative_identity(node)
        )
        assert root.path.is_dir()
        assert root.path.is_relative_to(scratch)


def test_rebinding_same_job_root_is_collision(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    node = _all_nodes(stage3)[0]

    bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=node,
    )

    with pytest.raises(
        JobOutputError,
        match="root collision",
    ):
        bind_job_output_root(
            repository_root=repo,
            scratch_root=scratch,
            node=node,
        )


def test_atomic_write_is_write_once_and_hashes_bytes(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    node = _all_nodes(stage3)[0]

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=node,
    )

    payload = b"technical artifact\n"

    evidence = atomic_write_job_file(
        root,
        "artifacts/result.bin",
        payload,
    )

    assert evidence.sha256 == hashlib.sha256(payload).hexdigest()
    assert evidence.size_bytes == len(payload)
    assert (
        root.path / evidence.relative_path
    ).read_bytes() == payload

    with pytest.raises(
        JobOutputError,
        match="collision",
    ):
        atomic_write_job_file(
            root,
            "artifacts/result.bin",
            b"replacement forbidden\n",
        )

    assert not list(
        (root.path / "artifacts").glob("*.tmp")
    )


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape.bin",
        "nested/../../escape.bin",
        "/Users/private/predecessor/checkpoint.pt",
        "/tmp/private-predecessor/checkpoint.pt",
        "C:\\private\\checkpoint.pt",
    ],
)
def test_output_path_escape_and_absolute_private_paths_are_rejected(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
    bad_path: str,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    node = _all_nodes(stage3)[0]

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=node,
    )

    with pytest.raises(JobOutputError):
        atomic_write_job_file(
            root,
            bad_path,
            b"must-not-write",
        )


def test_symlink_escape_inside_job_root_is_rejected(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    node = _all_nodes(stage3)[0]

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=node,
    )

    outside = tmp_path / "outside"
    outside.mkdir()

    os.symlink(
        outside,
        root.path / "escape-link",
    )

    with pytest.raises(
        JobOutputError,
        match="symlink component",
    ):
        atomic_write_job_file(
            root,
            "escape-link/forbidden.bin",
            b"must-not-escape",
        )

    assert not (outside / "forbidden.bin").exists()


def test_scratch_root_symlink_is_rejected(
    tmp_path: Path,
) -> None:
    repo, _ = _technical_repo(tmp_path)

    real = repo / "scratch-real"
    real.mkdir()

    link = repo / "scratch-link"
    os.symlink(real, link)

    with pytest.raises(
        JobOutputError,
        match="symlink",
    ):
        validate_ignored_scratch_root(
            repository_root=repo,
            scratch_root=link,
        )


def test_tracked_namespace_cannot_be_scratch_root(
    tmp_path: Path,
) -> None:
    repo, _ = _technical_repo(tmp_path)

    src = repo / "src"
    src.mkdir()
    (src / "tracked.py").write_text(
        "x = 1\n",
        encoding="utf-8",
    )
    _git("add", "src/tracked.py", cwd=repo)

    with pytest.raises(
        JobOutputError,
        match="tracked repository namespace",
    ):
        validate_ignored_scratch_root(
            repository_root=repo,
            scratch_root=src,
        )


def test_protected_predecessor_namespace_overlap_is_rejected(
    tmp_path: Path,
) -> None:
    repo, _ = _technical_repo(tmp_path)

    predecessor = repo / "private-predecessor"
    predecessor.mkdir()
    nested_scratch = predecessor / "scratch"
    nested_scratch.mkdir()

    with pytest.raises(
        JobOutputError,
        match="protected predecessor namespace",
    ):
        validate_ignored_scratch_root(
            repository_root=repo,
            scratch_root=nested_scratch,
            protected_roots=(predecessor,),
        )


def test_inert_placeholder_can_have_identity_root_but_cannot_write(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    analysis = _all_nodes(stage3)[-1]

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=analysis,
    )

    assert root.execution_allowed is False

    with pytest.raises(
        JobOutputError,
        match="inert placeholder",
    ):
        atomic_write_job_file(
            root,
            "analysis.json",
            b"forbidden",
        )

    with pytest.raises(
        JobOutputError,
        match="inert placeholder",
    ):
        write_job_completion(
            root,
            artifacts=(),
        )


def test_completion_record_is_canonical_verified_and_immutable(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    training = _all_nodes(stage3)[1]

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=training,
    )

    first = atomic_write_job_file(
        root,
        "logs/train.json",
        b'{"technical":true}\n',
    )
    second = atomic_write_job_file(
        root,
        "checkpoints/state.bin",
        b"tiny-state",
    )

    completion = write_job_completion(
        root,
        artifacts=(second, first),
    )

    raw = completion.path.read_bytes()
    record = json.loads(raw.decode("utf-8"))

    assert completion.path.name == JOB_COMPLETION_FILENAME
    assert completion.sha256 == hashlib.sha256(raw).hexdigest()
    assert record["scientific_data"] is False
    assert record["production_eligible"] is False
    assert record["completion_state"] == "complete"
    assert record["job_id"] == root.job_id
    assert record["relative_identity"] == root.relative_identity
    assert [
        item["relative_path"]
        for item in record["artifacts"]
    ] == [
        "checkpoints/state.bin",
        "logs/train.json",
    ]

    with pytest.raises(
        JobOutputError,
        match="collision",
    ):
        write_job_completion(
            root,
            artifacts=(first, second),
        )


def test_completion_rejects_forged_artifact_hash(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    node = _all_nodes(stage3)[0]

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=node,
    )

    evidence = atomic_write_job_file(
        root,
        "cache/payload.bin",
        b"payload",
    )

    forged = JobArtifactEvidence(
        relative_path=evidence.relative_path,
        sha256="e" * 64,
        size_bytes=evidence.size_bytes,
    )

    with pytest.raises(
        JobOutputError,
        match="hash mismatch",
    ):
        write_job_completion(
            root,
            artifacts=(forged,),
        )


def test_general_writer_cannot_claim_completion_filename(
    tmp_path: Path,
    stage3: Stage3AvailabilityIndex,
) -> None:
    repo, scratch = _technical_repo(tmp_path)
    node = _all_nodes(stage3)[0]

    root = bind_job_output_root(
        repository_root=repo,
        scratch_root=scratch,
        node=node,
    )

    with pytest.raises(
        JobOutputError,
        match="reserved",
    ):
        atomic_write_job_file(
            root,
            JOB_COMPLETION_FILENAME,
            b"forged completion",
        )
