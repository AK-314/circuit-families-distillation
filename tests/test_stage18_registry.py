from __future__ import annotations

from fractions import Fraction

from circuit_families.analysis.stage18_scaling import (
    CHECKPOINT_STEPS,
    COMPUTE_ONLY_CEILING,
    FRESH_CELL_COUNT,
    FRESH_TRAINING_SEEDS,
    PRIMARY_MAIN_SEEDS,
    PRODUCTION_WORKERS,
    REFERENCE_CELL_COUNT,
    RESERVE_SEEDS,
    TOTAL_CELL_COUNT,
    build_main_seed_registry,
    build_stage18_registry,
    build_worker_shards,
)


def test_seed_and_checkpoint_registries_are_exact() -> None:
    assert PRIMARY_MAIN_SEEDS == (0, 1, 2, 3, 4)
    assert FRESH_TRAINING_SEEDS == (0, 2, 3, 4)
    assert RESERVE_SEEDS == (5, 6, 7, 8, 9)
    assert CHECKPOINT_STEPS == (200, 3400, 7450, 8150, 8500, 8650, 9050)
    assert len(build_main_seed_registry()) == 5


def test_complete_registry_identity_and_order() -> None:
    cells = build_stage18_registry()
    assert len(cells) == TOTAL_CELL_COUNT == 5 * 7 * 18
    assert [cell.global_cell_index for cell in cells] == list(range(1, 631))
    assert len({cell.cell_id for cell in cells}) == 630
    assert cells[0].model_seed == 0
    assert cells[0].checkpoint_step == 200
    assert cells[0].fidelity == Fraction(4, 5)
    assert cells[0].distinctness == Fraction(1, 4)


def test_exact_fresh_and_reference_cells() -> None:
    cells = build_stage18_registry()
    references = [
        cell for cell in cells if cell.family_search_execution_mode == "reference_existing_result"
    ]
    fresh = [cell for cell in cells if cell.family_search_execution_mode == "fresh_execution"]
    assert len(references) == REFERENCE_CELL_COUNT == 18
    assert len(fresh) == FRESH_CELL_COUNT == 612
    assert all(cell.model_seed == 1 and cell.checkpoint_step == 9050 for cell in references)
    assert all(cell.transfer_execution_mode == "reference_existing_result" for cell in references)


def test_worker_shards_are_disjoint_complete_and_balanced() -> None:
    shards = build_worker_shards()
    assert PRODUCTION_WORKERS == 12
    assert COMPUTE_ONLY_CEILING == 14
    assert len(shards) == 12
    assert all(len(shard.cells) == 51 for shard in shards)
    assigned = [cell.cell_id for shard in shards for cell in shard.cells]
    assert len(assigned) == 612
    assert len(set(assigned)) == 612
    assert all(
        cell.worker_id == f"worker_{cell.fresh_cell_index % 12:02d}"
        for shard in shards
        for cell in shard.cells
    )


def test_shard_hashes_are_deterministic() -> None:
    first = build_worker_shards()
    second = build_worker_shards()
    assert [shard.shard_sha256 for shard in first] == [shard.shard_sha256 for shard in second]
