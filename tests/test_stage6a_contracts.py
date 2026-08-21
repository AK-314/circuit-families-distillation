from circuit_families.stage6a import (
    COMPONENT_COUNT,
    ExactBudgetUsage,
    ExactLedgerBuilder,
    TechnicalBudgetPolicy,
    TerminationStatus,
    reduce_endpoint1,
    validate_within_allowance,
)


def test_stage6a_integrated_contract():
    builder = ExactLedgerBuilder(
        evaluator=lambda mask: 1.0,
        fidelity_threshold=0.9,
    )

    full = (1,) * COMPONENT_COUNT
    half = (1,) * (COMPONENT_COUNT // 2) + (
        0,
    ) * (COMPONENT_COUNT // 2)

    builder.add_mask(full, proposal_index=0)
    builder.add_mask(half, proposal_index=1)

    entries = builder.seal()

    validate_within_allowance(
        ExactBudgetUsage(
            evaluation_count=len(entries),
            charged_count=len(entries),
        ),
        TechnicalBudgetPolicy(10),
    )

    result = reduce_endpoint1(
        entries,
        termination=TerminationStatus(
            status="completed",
            procedure_censored=False,
        ),
    )

    assert result.retained_proportion == 0.5
    assert result.global_minimum_claim is False


def test_stage6a_budget_boundary():
    try:
        validate_within_allowance(
            ExactBudgetUsage(3, 3),
            TechnicalBudgetPolicy(2),
        )
    except ValueError:
        return

    raise AssertionError("budget overflow accepted")


def test_stage6a_duplicate_proposals_do_not_duplicate_evaluations():
    builder = ExactLedgerBuilder(
        evaluator=lambda mask: 1.0,
        fidelity_threshold=0.9,
    )

    full = (1,) * COMPONENT_COUNT

    builder.add_mask(full, proposal_index=0)
    builder.add_mask(full, proposal_index=1)

    assert len(builder.proposals) == 2
    assert len(builder.evaluations) == 1


def test_stage6a_record_roundtrip():
    import json

    from circuit_families.stage6a.models import (
        ExactEvaluationEntry,
        exact_evaluation_entry_from_record,
        exact_evaluation_entry_to_record,
    )

    entry = ExactEvaluationEntry(
        mask_identity="abc",
        retained_count=10,
        retained_proportion=10 / 516,
        fidelity=0.99,
        qualifies=True,
        evaluation_order=1,
        exact_budget_charge=1,
    )

    record = exact_evaluation_entry_to_record(entry)

    assert json.loads(
        json.dumps(record, allow_nan=False)
    ) == record

    assert exact_evaluation_entry_from_record(record) == entry


def test_stage6a_part_c_model_validation():
    from circuit_families.stage6a.models import validate_mask_identity

    assert validate_mask_identity([3, 1, 3]) == (1, 3)

    for bad in ([True], [-1], [516], [1.5]):
        try:
            validate_mask_identity(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mask identity accepted")

def test_stage6a_part_c_record_invariant_rejection():
    from circuit_families.stage6a import (
        Endpoint1Result,
        ExactEvaluationEntry,
        ProposalEvent,
        TerminationStatus,
    )

    for bad in ("", None):
        try:
            ProposalEvent(
                proposal_index=bad,
                mask_identity="x",
            )
        except Exception:
            pass
        else:
            raise AssertionError("invalid proposal index accepted")

    try:
        ExactEvaluationEntry(
            mask_identity="x",
            retained_count=10,
            retained_proportion=float("nan"),
            fidelity=0.9,
            qualifies=True,
            evaluation_order=0,
            exact_budget_charge=1,
        )
    except Exception:
        pass

    result = Endpoint1Result(
        retained_proportion=0.5,
        mask_identity="x",
        global_minimum_claim=False,
        termination_status="completed",
        procedure_censored=False,
    )

    assert result.procedure_censored is False

    termination = TerminationStatus(
        status="completed",
        procedure_censored=False,
    )

    assert termination.status == "completed"


def test_stage6a_part_c_endpoint_record_preserves_censoring():
    from circuit_families.stage6a import (
        Endpoint1Result,
    )
    from circuit_families.stage6a.models import (
        endpoint1_result_from_record,
        endpoint1_result_to_record,
    )

    result = Endpoint1Result(
        retained_proportion=1.0,
        mask_identity="intact",
        global_minimum_claim=False,
        termination_status="completed",
        procedure_censored=True,
    )

    restored = endpoint1_result_from_record(
        endpoint1_result_to_record(result)
    )

    assert restored == result

def test_stage6a_part_c_termination_and_endpoint_validation():
    from circuit_families.stage6a import (
        Endpoint1Result,
        TerminationStatus,
    )

    try:
        TerminationStatus(
            status="unknown",
            procedure_censored=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid termination status accepted")

    try:
        Endpoint1Result(
            retained_proportion=2.0,
            mask_identity="x",
            global_minimum_claim=False,
            termination_status="completed",
            procedure_censored=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid endpoint proportion accepted")
