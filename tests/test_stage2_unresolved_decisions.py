from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from circuit_families.stage2_unresolved_decisions import (
    UnresolvedDecisionError,
    load_unresolved_decisions,
    validate_unresolved_decisions,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "followup/configs/stage2_unresolved_decisions_v1.json"


def canonical() -> dict:
    return json.loads(REGISTER.read_text(encoding="utf-8"))


def decision(record: dict, decision_id: str) -> dict:
    return next(d for d in record["decisions"] if d["decision_id"] == decision_id)


def test_canonical_register_passes() -> None:
    record = load_unresolved_decisions(REGISTER)
    assert len(record["decisions"]) == 14
    assert record["coverage"]["protocol_section15_expected_count"] == 19


def test_missing_decision_is_rejected() -> None:
    record = canonical()
    record["decisions"] = [
        d for d in record["decisions"] if d["decision_id"] != "UD-007"
    ]
    with pytest.raises(
        UnresolvedDecisionError,
        match="missing unresolved decision IDs",
    ):
        validate_unresolved_decisions(record)


def test_duplicate_decision_is_rejected() -> None:
    record = canonical()
    record["decisions"].append(copy.deepcopy(record["decisions"][0]))
    with pytest.raises(
        UnresolvedDecisionError,
        match="duplicate unresolved decision ID",
    ):
        validate_unresolved_decisions(record)


def test_unknown_decision_is_rejected() -> None:
    record = canonical()
    record["decisions"][-1]["decision_id"] = "UD-999"
    with pytest.raises(
        UnresolvedDecisionError,
        match="unknown unresolved decision IDs",
    ):
        validate_unresolved_decisions(record)


def test_premature_resolution_is_rejected() -> None:
    record = canonical()
    decision(record, "UD-007")["status"] = "resolved"
    with pytest.raises(
        UnresolvedDecisionError,
        match="prematurely resolved",
    ):
        validate_unresolved_decisions(record)


def test_ownerless_decision_is_rejected() -> None:
    record = canonical()
    decision(record, "UD-007")["owner"] = ""
    with pytest.raises(
        UnresolvedDecisionError,
        match="missing or unknown owner",
    ):
        validate_unresolved_decisions(record)


def test_stageless_decision_is_rejected() -> None:
    record = canonical()
    decision(record, "UD-007")["resolution_stage"] = ""
    with pytest.raises(
        UnresolvedDecisionError,
        match="missing or unknown resolution stage",
    ):
        validate_unresolved_decisions(record)


def test_wrong_lane_is_rejected() -> None:
    record = canonical()
    decision(record, "UD-007")["lane"] = "Lane B"
    with pytest.raises(
        UnresolvedDecisionError,
        match="lane mismatch",
    ):
        validate_unresolved_decisions(record)


def test_missing_section15_row_is_rejected() -> None:
    record = canonical()
    decision(record, "UD-011")["section15_rows"] = [
        "Phase contrasts",
        "Student cell summary",
    ]
    with pytest.raises(
        UnresolvedDecisionError,
        match="Section 15 coverage mismatch",
    ):
        validate_unresolved_decisions(record)


def test_unknown_section15_row_is_rejected() -> None:
    record = canonical()
    decision(record, "UD-011")["section15_rows"] = [
        "Phase contrasts",
        "Student cell summary",
        "Invented rule",
    ]
    with pytest.raises(
        UnresolvedDecisionError,
        match="Section 15 coverage mismatch",
    ):
        validate_unresolved_decisions(record)


def test_missing_cell_rule_cannot_move_back_to_stage11() -> None:
    record = canonical()
    decision(record, "UD-004")["section15_rows"] = [
        "Student replication",
        "Missing-cell rule",
    ]
    with pytest.raises(
        UnresolvedDecisionError,
        match="UD-004 Section 15 coverage mismatch",
    ):
        validate_unresolved_decisions(record)


def test_settled_population_unit_cannot_be_demoted_to_open_decision() -> None:
    record = canonical()
    decision(record, "UD-007")["decision_family"] = "Population-level unit"
    with pytest.raises(
        UnresolvedDecisionError,
        match="attempts to demote settled Stage 2 concept",
    ):
        validate_unresolved_decisions(record)


def test_settled_hard_soft_estimand_cannot_be_demoted() -> None:
    record = canonical()
    decision(record, "UD-006")["decision_family"] = "Hard and soft estimands"
    with pytest.raises(
        UnresolvedDecisionError,
        match="attempts to demote settled Stage 2 concept",
    ):
        validate_unresolved_decisions(record)


def test_recommended_values_must_remain_nonbinding() -> None:
    record = canonical()
    record["coverage"]["recommended_values_are_nonbinding"] = False
    with pytest.raises(
        UnresolvedDecisionError,
        match="recommended_values_are_nonbinding must remain true",
    ):
        validate_unresolved_decisions(record)
