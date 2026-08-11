from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from circuit_families.analysis.fourier_sanity_check import (
    MODULUS,
    POWER_ABSOLUTE_TOLERANCE,
    analyse_activation_matrix,
    analyse_logit_tensor,
    canonical_pair_power,
    centre_logits,
    modular_addition_indices,
    reshape_lexicographic_logits,
    synthetic_relation_tensor,
)


def test_cosine_addition_tensor_places_power_on_m0() -> None:
    tensor = synthetic_relation_tensor(frequency=7)
    result = analyse_logit_tensor(tensor)

    assert result.shifted_relations.correct_shift_rank == 1
    assert result.addition_manifold_fraction > 1.0 - 1.0e-10


def test_sine_addition_tensor_places_power_on_m0() -> None:
    tensor = synthetic_relation_tensor(
        frequency=11,
        use_sine=True,
    )
    result = analyse_logit_tensor(tensor)

    assert result.shifted_relations.correct_shift_rank == 1
    assert result.addition_manifold_fraction > 1.0 - 1.0e-10


def test_conjugate_frequency_pair_is_not_double_counted() -> None:
    values = np.zeros(MODULUS)
    values[9] = 2.0
    values[MODULUS - 9] = 3.0

    paired = canonical_pair_power(values)

    assert paired[8] == 5.0
    assert sum(paired) == 5.0


def test_shifted_frequency_index_relation_is_assigned_correctly() -> None:
    tensor = synthetic_relation_tensor(
        frequency=13,
        shift=5,
    )
    result = analyse_logit_tensor(tensor)

    ranking = sorted(
        range(MODULUS),
        key=lambda shift: (
            -result.shifted_relations.power_by_shift[shift],
            shift,
        ),
    )

    assert ranking[0] == 5


def test_output_centering_removes_only_class_independent_offsets() -> None:
    base = synthetic_relation_tensor(frequency=3)
    offsets = np.arange(MODULUS * MODULUS).reshape(
        MODULUS,
        MODULUS,
        1,
    )
    shifted = base + offsets

    np.testing.assert_allclose(
        centre_logits(shifted),
        centre_logits(base),
        atol=5.0e-12,
        rtol=0.0,
    )


def test_orthonormal_fft_satisfies_parseval() -> None:
    tensor = synthetic_relation_tensor(frequency=17)
    centred = centre_logits(tensor)
    spectrum = np.fft.fftn(centred, norm="ortho")

    assert math.isclose(
        float(np.square(centred).sum()),
        float(np.square(np.abs(spectrum)).sum()),
        abs_tol=POWER_ABSOLUTE_TOLERANCE,
        rel_tol=1.0e-12,
    )


def test_function_of_a_plus_b_has_diagonal_activation_power() -> None:
    a, b = np.meshgrid(
        np.arange(MODULUS),
        np.arange(MODULUS),
        indexing="ij",
    )
    activation = np.cos(
        2.0 * np.pi * 19 * (a + b) / MODULUS
    )
    result = analyse_activation_matrix(activation)

    assert result.diagonal_power_fraction > 1.0 - 1.0e-10
    assert result.dominant_frequency_pair == 19


def test_function_of_a_alone_is_not_diagonal_addition_power() -> None:
    a, _ = np.meshgrid(
        np.arange(MODULUS),
        np.arange(MODULUS),
        indexing="ij",
    )
    activation = np.cos(2.0 * np.pi * 23 * a / MODULUS)
    result = analyse_activation_matrix(activation)

    assert result.diagonal_power_fraction < 1.0e-10


def test_constant_activation_is_near_constant() -> None:
    result = analyse_activation_matrix(
        np.full((MODULUS, MODULUS), 4.5)
    )

    assert result.near_constant is True
    assert result.dominant_frequency_pair is None
    assert result.total_non_dc_power == pytest.approx(0.0, abs=1.0e-20)


def test_incorrect_input_ordering_is_rejected() -> None:
    left = torch.arange(MODULUS).repeat_interleave(MODULUS)
    right = torch.arange(MODULUS).repeat(MODULUS)
    inputs = torch.stack(
        [left, right, torch.full_like(left, MODULUS)],
        dim=1,
    )
    logits = torch.zeros(MODULUS * MODULUS, MODULUS)

    reshape_lexicographic_logits(inputs, logits)

    swapped = inputs.clone()
    swapped[[0, 1]] = swapped[[1, 0]]

    with pytest.raises(ValueError, match="lexicographic"):
        reshape_lexicographic_logits(swapped, logits)


def test_declared_m0_indices_have_expected_sign_convention() -> None:
    k = 29
    indices = modular_addition_indices()

    assert indices[0][k - 1] == k
    assert indices[1][k - 1] == k
    assert indices[2][k - 1] == (-k) % MODULUS


def test_behaviour_metrics_identity_is_exact() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        behaviour_metrics_from_logits,
    )

    logits = torch.tensor(
        [
            [4.0, 1.0, -2.0],
            [-1.0, 3.0, 0.0],
        ],
        dtype=torch.float64,
    )
    padded = torch.full((2, MODULUS), -10.0, dtype=torch.float64)
    padded[:, :3] = logits
    targets = torch.tensor([0, 1], dtype=torch.long)

    metrics = behaviour_metrics_from_logits(
        padded,
        padded.clone(),
        targets,
    )

    assert metrics.primary_fidelity == 1.0
    assert metrics.prediction_disagreement_count == 0
    assert metrics.accuracy_change == 0.0
    assert metrics.cross_entropy_change == 0.0
    assert metrics.maximum_absolute_logit_difference == 0.0
    assert metrics.mean_kl_divergence == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert metrics.mean_jensen_shannon_divergence == pytest.approx(
        0.0,
        abs=1.0e-15,
    )


def test_component_ranking_uses_frozen_index_for_ties() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        ComponentAssociationRecord,
        component_association_ranking,
    )
    from circuit_families.interpretability.masks import (
        SEARCHABLE_COMPONENT_IDS,
    )

    records = []

    for index, identifier in enumerate(SEARCHABLE_COMPONENT_IDS):
        records.append(
            ComponentAssociationRecord(
                component_identifier=identifier,
                component_type=(
                    "attention_head"
                    if identifier.startswith("H")
                    else "mlp_neuron"
                ),
                component_index=index,
                primary_fidelity=1.0,
                prediction_agreement_count=1,
                prediction_disagreement_count=0,
                ground_truth_accuracy_change=0.0,
                cross_entropy_change=0.0,
                mean_kl_divergence=0.0,
                mean_jensen_shannon_divergence=0.0,
                maximum_absolute_logit_change=0.0,
                total_delta_fourier_power=1.0,
                addition_manifold_delta_power=1.0,
                addition_manifold_delta_fraction=1.0,
                correct_shift_rank=1,
                correct_shift_selectivity=2.0,
                dominant_canonical_frequency_pair=1,
                activation_diagonal_power_fraction=None,
                activation_near_constant=None,
                retained_flags=(True,) * 6,
            )
        )

    ranking = component_association_ranking(records)

    assert tuple(
        record.component_identifier for record in ranking
    ) == SEARCHABLE_COMPONENT_IDS


def test_retained_removal_selection_is_deterministic() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        ComponentAssociationRecord,
        select_retained_components_for_removal,
    )
    from circuit_families.interpretability.masks import ComponentMask

    ranking = (
        ComponentAssociationRecord(
            component_identifier="N4",
            component_type="mlp_neuron",
            component_index=8,
            primary_fidelity=0.9,
            prediction_agreement_count=9,
            prediction_disagreement_count=1,
            ground_truth_accuracy_change=-0.1,
            cross_entropy_change=0.1,
            mean_kl_divergence=0.1,
            mean_jensen_shannon_divergence=0.1,
            maximum_absolute_logit_change=1.0,
            total_delta_fourier_power=10.0,
            addition_manifold_delta_power=8.0,
            addition_manifold_delta_fraction=0.8,
            correct_shift_rank=1,
            correct_shift_selectivity=3.0,
            dominant_canonical_frequency_pair=4,
            activation_diagonal_power_fraction=0.8,
            activation_near_constant=False,
            retained_flags=(True,) * 6,
        ),
        ComponentAssociationRecord(
            component_identifier="H2",
            component_type="attention_head",
            component_index=2,
            primary_fidelity=0.95,
            prediction_agreement_count=9,
            prediction_disagreement_count=1,
            ground_truth_accuracy_change=-0.05,
            cross_entropy_change=0.05,
            mean_kl_divergence=0.05,
            mean_jensen_shannon_divergence=0.05,
            maximum_absolute_logit_change=0.5,
            total_delta_fourier_power=5.0,
            addition_manifold_delta_power=4.0,
            addition_manifold_delta_fraction=0.8,
            correct_shift_rank=1,
            correct_shift_selectivity=2.0,
            dominant_canonical_frequency_pair=2,
            activation_diagonal_power_fraction=None,
            activation_near_constant=None,
            retained_flags=(True,) * 6,
        ),
        ComponentAssociationRecord(
            component_identifier="N9",
            component_type="mlp_neuron",
            component_index=13,
            primary_fidelity=1.0,
            prediction_agreement_count=10,
            prediction_disagreement_count=0,
            ground_truth_accuracy_change=0.0,
            cross_entropy_change=0.0,
            mean_kl_divergence=0.0,
            mean_jensen_shannon_divergence=0.0,
            maximum_absolute_logit_change=0.0,
            total_delta_fourier_power=1.0,
            addition_manifold_delta_power=0.1,
            addition_manifold_delta_fraction=0.1,
            correct_shift_rank=3,
            correct_shift_selectivity=0.5,
            dominant_canonical_frequency_pair=9,
            activation_diagonal_power_fraction=0.1,
            activation_near_constant=False,
            retained_flags=(True,) * 6,
        ),
    )

    mask = ComponentMask.from_retained_identifiers(
        ("H2", "N4", "N9")
    )

    selections = select_retained_components_for_removal(
        mask,
        ranking,
    )

    assert selections == (
        ("highest_associated_retained_overall", "N4"),
        ("highest_associated_retained_attention_head", "H2"),
        ("lowest_associated_retained_component", "N9"),
    )


def test_clear_match_classification() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        classify_fourier_diagnostic,
    )

    result = analyse_logit_tensor(
        synthetic_relation_tensor(frequency=7)
    )

    assert classify_fourier_diagnostic(result) == "clear_match"


def test_mismatch_classification_for_shifted_relation() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        classify_fourier_diagnostic,
    )

    result = analyse_logit_tensor(
        synthetic_relation_tensor(
            frequency=7,
            shift=9,
        )
    )

    assert classify_fourier_diagnostic(result) == "mismatch"


def test_degenerate_classification_for_zero_tensor() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        classify_fourier_diagnostic,
    )

    result = analyse_logit_tensor(
        np.zeros((MODULUS, MODULUS, MODULUS))
    )

    assert classify_fourier_diagnostic(result) == "degenerate"


def test_stage10_run_id_is_deterministic() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        deterministic_stage10_run_id,
    )

    first = deterministic_stage10_run_id(
        {"b": 2, "a": 1}
    )
    second = deterministic_stage10_run_id(
        {"a": 1, "b": 2}
    )

    assert first == second
    assert first.startswith("stage10-fourier-s1-")


def test_deterministic_json_has_sorted_keys_and_no_nan(
    tmp_path,
) -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        write_deterministic_json,
    )

    output = tmp_path / "record.json"
    write_deterministic_json(
        output,
        {
            "z": 1,
            "a": float("nan"),
            "infinity": float("inf"),
        },
    )

    assert output.read_text(encoding="utf-8") == (
        '{\n'
        '  "a": null,\n'
        '  "infinity": "positive_infinity",\n'
        '  "z": 1\n'
        '}\n'
    )


def test_association_power_summary_uses_retained_components() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        ComponentAssociationRecord,
        association_power_summary,
    )
    from circuit_families.interpretability.masks import ComponentMask

    records = (
        ComponentAssociationRecord(
            component_identifier="H0",
            component_type="attention_head",
            component_index=0,
            primary_fidelity=1.0,
            prediction_agreement_count=1,
            prediction_disagreement_count=0,
            ground_truth_accuracy_change=0.0,
            cross_entropy_change=0.0,
            mean_kl_divergence=0.0,
            mean_jensen_shannon_divergence=0.0,
            maximum_absolute_logit_change=0.0,
            total_delta_fourier_power=10.0,
            addition_manifold_delta_power=8.0,
            addition_manifold_delta_fraction=0.8,
            correct_shift_rank=1,
            correct_shift_selectivity=4.0,
            dominant_canonical_frequency_pair=1,
            activation_diagonal_power_fraction=None,
            activation_near_constant=None,
            retained_flags=(True,) * 6,
        ),
        ComponentAssociationRecord(
            component_identifier="N0",
            component_type="mlp_neuron",
            component_index=4,
            primary_fidelity=1.0,
            prediction_agreement_count=1,
            prediction_disagreement_count=0,
            ground_truth_accuracy_change=0.0,
            cross_entropy_change=0.0,
            mean_kl_divergence=0.0,
            mean_jensen_shannon_divergence=0.0,
            maximum_absolute_logit_change=0.0,
            total_delta_fourier_power=5.0,
            addition_manifold_delta_power=2.0,
            addition_manifold_delta_fraction=0.4,
            correct_shift_rank=2,
            correct_shift_selectivity=0.5,
            dominant_canonical_frequency_pair=2,
            activation_diagonal_power_fraction=0.4,
            activation_near_constant=False,
            retained_flags=(False,) * 6,
        ),
    )

    summary = association_power_summary(
        ComponentMask.from_retained_identifiers(("H0",)),
        records,
    )

    assert summary == (8.0, 10.0, 0.8, "H0", "H0")


def test_weight_fourier_records_cover_all_canonical_pairs() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        weight_fourier_records,
    )

    index = np.arange(MODULUS)
    matrix = np.stack(
        [
            np.cos(2.0 * np.pi * 6 * index / MODULUS),
            np.sin(2.0 * np.pi * 6 * index / MODULUS),
        ],
        axis=1,
    )
    records = weight_fourier_records(
        weight_name="synthetic",
        matrix=matrix,
    )

    assert len(records) == 56
    assert records[5].canonical_frequency_pair == 6
    assert records[5].descending_rank == 1
    assert records[5].normalized_pair_power == pytest.approx(
        1.0,
        abs=1.0e-12,
    )


def test_activation_records_preserve_neuron_order() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        ActivationSpectrumDiagnostics,
        activation_fourier_records,
    )

    diagnostics = tuple(
        ActivationSpectrumDiagnostics(
            total_non_dc_power=float(index),
            diagonal_power=float(index),
            diagonal_power_fraction=0.5,
            canonical_pair_power=(0.0,) * 56,
            normalized_canonical_pair_power=(0.0,) * 56,
            dominant_frequency_pair=None,
            activation_variance=float(index),
            activation_mean=float(-index),
            near_constant=index == 0,
        )
        for index in range(512)
    )

    records = activation_fourier_records(diagnostics)

    assert len(records) == 512
    assert records[0].component_identifier == "N0"
    assert records[511].component_identifier == "N511"
    assert records[317].neuron_index == 317


def test_model_weight_records_exclude_equals_token() -> None:
    from types import SimpleNamespace

    from circuit_families.analysis.fourier_sanity_check import (
        model_weight_fourier_records,
    )

    model = SimpleNamespace(
        W_E=torch.randn(114, 8),
        W_U=torch.randn(8, 114),
    )

    embedding, unembedding = model_weight_fourier_records(model)

    assert len(embedding) == 56
    assert len(unembedding) == 56
    assert {
        record.weight_name for record in embedding
    } == {"token_embedding_W_E"}
    assert {
        record.weight_name for record in unembedding
    } == {"valid_class_unembedding_W_U"}


def test_weight_record_rejects_short_vocabulary() -> None:
    from types import SimpleNamespace

    from circuit_families.analysis.fourier_sanity_check import (
        model_weight_fourier_records,
    )

    model = SimpleNamespace(
        W_E=torch.randn(112, 8),
        W_U=torch.randn(8, 113),
    )

    with pytest.raises(ValueError, match="W_E"):
        model_weight_fourier_records(model)


def test_component_table_records_flatten_retention_flags() -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        ComponentAssociationExecution,
        ComponentAssociationRecord,
        component_table_records,
    )

    record = ComponentAssociationRecord(
        component_identifier="H0",
        component_type="attention_head",
        component_index=0,
        primary_fidelity=1.0,
        prediction_agreement_count=10,
        prediction_disagreement_count=0,
        ground_truth_accuracy_change=0.0,
        cross_entropy_change=0.0,
        mean_kl_divergence=0.0,
        mean_jensen_shannon_divergence=0.0,
        maximum_absolute_logit_change=0.0,
        total_delta_fourier_power=1.0,
        addition_manifold_delta_power=1.0,
        addition_manifold_delta_fraction=1.0,
        correct_shift_rank=1,
        correct_shift_selectivity=2.0,
        dominant_canonical_frequency_pair=1,
        activation_diagonal_power_fraction=None,
        activation_near_constant=None,
        retained_flags=(True, False, True, False, True, False),
    )
    execution = ComponentAssociationExecution(
        records=(record,),
        model_state_sha256_before="before",
        model_state_sha256_after="before",
        hook_counts_before=(),
        hook_counts_after=(),
        gradients_absent_after=True,
    )

    rows = component_table_records(execution)

    assert len(rows) == 1
    assert "retained_flags" not in rows[0]
    assert rows[0]["retained_at_threshold_0_99"] is True
    assert rows[0]["retained_at_threshold_0_975"] is False
    assert rows[0]["retained_at_threshold_0_8"] is False


def test_mapping_csv_is_byte_stable(tmp_path) -> None:
    from circuit_families.analysis.fourier_sanity_check import (
        write_mapping_csv,
    )

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    records = (
        {"a": 1, "b": 2.5},
        {"a": 3, "b": 4.5},
    )

    write_mapping_csv(first, records)
    write_mapping_csv(second, records)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8") == (
        "a,b\n1,2.5\n3,4.5\n"
    )
