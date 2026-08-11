from __future__ import annotations

from pathlib import Path

import pytest

from circuit_families.analysis.fourier_sanity_check import (
    STABLE_POST_CHECKPOINT_STEP,
    STABLE_POST_THRESHOLDS,
    _safe_archive_member_name,
    load_stable_post_stage9_circuits,
)
from circuit_families.interpretability.masks import (
    SEARCHABLE_COMPONENT_IDS,
)

REPOSITORY = Path(__file__).resolve().parents[1]
STAGE9_MANIFEST = (
    REPOSITORY
    / "manifests"
    / "stage9_sparse_stage9-sparse-s1-27fffed087e6.json"
)
STAGE9_TABLE = (
    REPOSITORY
    / "results"
    / "tables"
    / "seed_1_stage9_sparse_search.csv"
)
STAGE9_ARCHIVE = (
    REPOSITORY
    / "results"
    / "archives"
    / "stage9-sparse-s1-27fffed087e6.tar.gz"
)


def test_loads_all_and_only_six_stable_post_circuits() -> None:
    circuits = load_stable_post_stage9_circuits(
        stage9_manifest_path=STAGE9_MANIFEST,
        stage9_table_path=STAGE9_TABLE,
        stage9_archive_path=STAGE9_ARCHIVE,
    )

    assert len(circuits) == 6
    assert tuple(
        circuit.fidelity_threshold for circuit in circuits
    ) == STABLE_POST_THRESHOLDS
    assert {
        circuit.checkpoint_step for circuit in circuits
    } == {STABLE_POST_CHECKPOINT_STEP}


def test_component_ordering_remains_frozen() -> None:
    assert SEARCHABLE_COMPONENT_IDS[:4] == (
        "H0",
        "H1",
        "H2",
        "H3",
    )
    assert SEARCHABLE_COMPONENT_IDS[4] == "N0"
    assert SEARCHABLE_COMPONENT_IDS[-1] == "N511"
    assert len(SEARCHABLE_COMPONENT_IDS) == 516


@pytest.mark.parametrize(
    "name",
    (
        "../escape.json",
        "/absolute/path.json",
        "safe/../../escape.json",
    ),
)
def test_archive_path_traversal_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_archive_member_name(name)


def test_stage9_archive_and_mask_hashes_are_verified(
    tmp_path: Path,
) -> None:
    damaged = tmp_path / "damaged.tar.gz"
    damaged.write_bytes(STAGE9_ARCHIVE.read_bytes() + b"damage")

    with pytest.raises(ValueError, match="archive SHA-256"):
        load_stable_post_stage9_circuits(
            stage9_manifest_path=STAGE9_MANIFEST,
            stage9_table_path=STAGE9_TABLE,
            stage9_archive_path=damaged,
        )


def test_stage9_circuit_masks_have_expected_counts() -> None:
    circuits = load_stable_post_stage9_circuits(
        stage9_manifest_path=STAGE9_MANIFEST,
        stage9_table_path=STAGE9_TABLE,
        stage9_archive_path=STAGE9_ARCHIVE,
    )

    assert tuple(
        circuit.retained_components for circuit in circuits
    ) == (146, 119, 108, 82, 77, 64)

    assert all(circuit.retained_heads == 4 for circuit in circuits)


def test_all_stable_post_retention_flags_have_six_entries() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        retained_flags,
    )

    circuits = load_stable_post_stage9_circuits(
        stage9_manifest_path=STAGE9_MANIFEST,
        stage9_table_path=STAGE9_TABLE,
        stage9_archive_path=STAGE9_ARCHIVE,
    )
    flags = retained_flags(circuits)

    assert set(flags) == set(SEARCHABLE_COMPONENT_IDS)
    assert all(len(value) == 6 for value in flags.values())
    assert all(flags[head] == (True,) * 6 for head in ("H0", "H1", "H2", "H3"))


def test_stage10_configuration_does_not_select_threshold() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        stage10_configuration_record,
    )

    configuration = stage10_configuration_record(
        source_training_run_id="training-run",
        checkpoint_sha256="checkpoint-hash",
        stage9_manifest_sha256="manifest-hash",
        stage9_table_sha256="table-hash",
        stage9_archive_sha256="archive-hash",
        implementation_git_commit="commit-hash",
        device="cpu",
        batch_size=256,
    )

    assert configuration[
        "primary_fidelity_threshold_selected"
    ] is False
    assert configuration["stage11_calibration_performed"] is False
    assert configuration["checkpoint_step"] == 9050


def test_stage10_validate_only_cli() -> None:
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_fourier_sanity_check.py",
            "--run-id",
            "modular-addition-training-s1-5f1bc9dee7ab",
            "--checkpoint-manifest",
            "manifests/checkpoints_seed_1.json",
            "--stage9-manifest",
            str(STAGE9_MANIFEST.relative_to(REPOSITORY)),
            "--stage9-table",
            str(STAGE9_TABLE.relative_to(REPOSITORY)),
            "--stage9-archive",
            str(STAGE9_ARCHIVE.relative_to(REPOSITORY)),
            "--device",
            "cpu",
            "--batch-size",
            "256",
            "--repository-root",
            ".",
            "--validate-inputs-only",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "validation_status: passed" in result.stdout
    assert "scientific_outputs_generated: false" in result.stdout
    assert "stable_post_circuit_count: 6" in result.stdout


def test_selected_removal_count_is_bounded_per_circuit() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        ComponentAssociationRecord,
        component_association_ranking,
        select_retained_components_for_removal,
    )

    circuits = load_stable_post_stage9_circuits(
        stage9_manifest_path=STAGE9_MANIFEST,
        stage9_table_path=STAGE9_TABLE,
        stage9_archive_path=STAGE9_ARCHIVE,
    )

    records = tuple(
        ComponentAssociationRecord(
            component_identifier=identifier,
            component_type=(
                "attention_head"
                if identifier.startswith("H")
                else "mlp_neuron"
            ),
            component_index=index,
            primary_fidelity=1.0,
            prediction_agreement_count=12769,
            prediction_disagreement_count=0,
            ground_truth_accuracy_change=0.0,
            cross_entropy_change=0.0,
            mean_kl_divergence=0.0,
            mean_jensen_shannon_divergence=0.0,
            maximum_absolute_logit_change=0.0,
            total_delta_fourier_power=float(516 - index),
            addition_manifold_delta_power=float(516 - index),
            addition_manifold_delta_fraction=1.0,
            correct_shift_rank=1,
            correct_shift_selectivity=2.0,
            dominant_canonical_frequency_pair=1,
            activation_diagonal_power_fraction=(
                None if identifier.startswith("H") else 1.0
            ),
            activation_near_constant=(
                None if identifier.startswith("H") else False
            ),
            retained_flags=(True,) * 6,
        )
        for index, identifier in enumerate(
            SEARCHABLE_COMPONENT_IDS
        )
    )
    ranking = component_association_ranking(records)

    for circuit in circuits:
        selections = select_retained_components_for_removal(
            circuit.mask,
            ranking,
        )

        assert 2 <= len(selections) <= 4
        assert len(
            {
                identifier
                for _, identifier in selections
            }
        ) == len(selections)
        assert all(
            identifier in circuit.mask.retained_component_ids
            for _, identifier in selections
        )


def test_every_stable_post_circuit_retains_a_neuron() -> None:
    circuits = load_stable_post_stage9_circuits(
        stage9_manifest_path=STAGE9_MANIFEST,
        stage9_table_path=STAGE9_TABLE,
        stage9_archive_path=STAGE9_ARCHIVE,
    )

    assert all(
        circuit.retained_neurons > 0
        for circuit in circuits
    )


def test_validate_only_cli_writes_no_stage10_outputs() -> None:
    import json
    import subprocess

    from circuit_families.analysis.fourier_sanity_check import (
        deterministic_stage10_run_id,
        stage10_configuration_record,
        stage10_output_paths,
    )
    from circuit_families.training import file_sha256

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    checkpoint_manifest = json.loads(
        (
            REPOSITORY / "manifests/checkpoints_seed_1.json"
        ).read_text(encoding="utf-8")
    )
    stable = checkpoint_manifest[
        "selected_stable_post_checkpoint"
    ]

    configuration = stage10_configuration_record(
        source_training_run_id=(
            "modular-addition-training-s1-5f1bc9dee7ab"
        ),
        checkpoint_sha256=stable["checkpoint_sha256"],
        stage9_manifest_sha256=file_sha256(STAGE9_MANIFEST),
        stage9_table_sha256=file_sha256(STAGE9_TABLE),
        stage9_archive_sha256=file_sha256(STAGE9_ARCHIVE),
        implementation_git_commit=head,
        device="cpu",
        batch_size=256,
    )
    run_id = deterministic_stage10_run_id(configuration)
    paths = stage10_output_paths(
        REPOSITORY,
        stage10_run_id=run_id,
    )

    watched_paths = (
        paths.output_directory,
        paths.manifest,
        paths.component_table,
        paths.circuit_table,
        paths.removal_table,
        paths.embedding_table,
        paths.activation_table,
    )
    state_before = {
        path: (
            path.exists(),
            path.stat().st_size if path.is_file() else None,
            path.stat().st_mtime_ns if path.exists() else None,
        )
        for path in watched_paths
    }

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_fourier_sanity_check.py",
            "--run-id",
            "modular-addition-training-s1-5f1bc9dee7ab",
            "--checkpoint-manifest",
            "manifests/checkpoints_seed_1.json",
            "--stage9-manifest",
            str(STAGE9_MANIFEST.relative_to(REPOSITORY)),
            "--stage9-table",
            str(STAGE9_TABLE.relative_to(REPOSITORY)),
            "--stage9-archive",
            str(STAGE9_ARCHIVE.relative_to(REPOSITORY)),
            "--device",
            "cpu",
            "--batch-size",
            "256",
            "--repository-root",
            ".",
            "--validate-inputs-only",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )

    state_after = {
        path: (
            path.exists(),
            path.stat().st_size if path.is_file() else None,
            path.stat().st_mtime_ns if path.exists() else None,
        )
        for path in watched_paths
    }

    assert state_after == state_before
