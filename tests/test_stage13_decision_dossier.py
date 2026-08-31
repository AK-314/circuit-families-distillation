from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.validate_stage13_decision_dossier import (
    DOSSIER,
    EXPECTED_DECISIONS,
    validate,
    validate_dossier_mapping,
)


def test_stage13_preapproval_dossier_is_complete_and_deterministic() -> None:
    first = validate()
    second = validate()
    assert first == second
    assert tuple(first["decision_ids"]) == EXPECTED_DECISIONS
    assert first["approval_status"] == "pending"
    assert first["validation"] == "PASS"


def test_stage13_dossier_schema_rejects_unknown_fields() -> None:
    dossier = json.loads(DOSSIER.read_text(encoding="utf-8"))
    corrupted = copy.deepcopy(dossier)
    corrupted["silent_default"] = 0.99
    with pytest.raises(ValueError, match="closed schema"):
        validate_dossier_mapping(corrupted)


def test_stage13_dossier_has_no_private_or_registered_path() -> None:
    text = DOSSIER.read_text(encoding="utf-8").lower()
    forbidden = ("/users/", "\\users\\", "/home/", ".ssh/", "s3://")
    assert not any(token in text for token in forbidden)
    assert Path(DOSSIER).is_file()
