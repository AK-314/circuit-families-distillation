from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ScientificSkeletonError(ValueError):
    pass


FROZEN_ITEM_KEYS = {
    "item_id",
    "name",
    "status",
    "authority_refs",
    "normative_statement",
    "claim_limit",
}

EXPECTED_FROZEN_ITEM_METADATA = {'FS-001': {'name': 'Exact research question',
            'status': 'frozen_stage2',
            'authority_refs': ['protocol']},
 'FS-002': {'name': 'Experimental hierarchy',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-003': {'name': 'Population-level unit',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-004': {'name': 'Student-initialization interpretation',
            'status': 'frozen_stage2',
            'authority_refs': ['protocol']},
 'FS-005': {'name': 'Repeated-measurement layers',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-006': {'name': 'Hard and soft estimands',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-007': {'name': 'Direct teacher evaluation',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-008': {'name': 'Hard-target eligibility',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-009': {'name': 'Student circuit-fidelity reference',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-010': {'name': 'Proposed primary fidelity family',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-011': {'name': 'Endpoint 1 definition',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol', 'workflow']},
 'FS-012': {'name': 'Endpoint 1 interpretation limit',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol', 'workflow']},
 'FS-013': {'name': 'Endpoint 2 definition and interpretation',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol', 'workflow']},
 'FS-014': {'name': 'Method-budget interpretation',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol', 'workflow']},
 'FS-015': {'name': 'Outcome-category status',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master']},
 'FS-016': {'name': 'Fourier interchange scope',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-017': {'name': 'Gated extensions',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']},
 'FS-018': {'name': 'Method-development firewall',
            'status': 'frozen_stage2',
            'authority_refs': ['implementation_master', 'protocol']}}

FROZEN_TEXT_SHA256 = {
    "FS-001": {
        "normative_statement": "513d0e44b2fe22c04f9e1520a6f6319c264e16a5b0a32129816125c94ec535f3",
        "claim_limit": "ecb482a909ac29a4747d615f227efd38586a55b21646bb529b82e78a65ef4549",
    },
    "FS-002": {
        "normative_statement": "48d40f2aaf6f2da39ef4d7de6ed918efb7e3738a46ffb1f5b228f7af8ead01b3",
        "claim_limit": "cad195745c6a290a4876c01f432ec5cdca8e21149214050a3cfa23258cede4b4",
    },
    "FS-003": {
        "normative_statement": "97c4a6dea51c4e9de498de86ed39a96321a8a37f13ac4fe3c5b93bdd36689172",
        "claim_limit": "569aed955cacc068bfea55a1bb1afd0cdf446c8cd7f2bf2a184673e96f19e097",
    },
    "FS-004": {
        "normative_statement": "0f2f0ce8b0bc1eacbe8082317a921c0032c096f996da9261bddb1549b9043775",
        "claim_limit": "ee0517777efe328d95798896bd78dbe4cf3a3fa6a374bd99b3ed1877495131cf",
    },
    "FS-005": {
        "normative_statement": "9efbcfa4e4e752ea6d786fb034a59218fc928b7cdf1b9e92ee742653fbe3636b",
        "claim_limit": "0867e82b204a3125e673d10ae074a8a1c574ad7ec5e2af01e50d5e342f7ec74b",
    },
    "FS-006": {
        "normative_statement": "1f7abe42dea3f7063c80737fc0aa51263b77ce9dde1100528e684bb22b263529",
        "claim_limit": "e760ce306c7473cfa66f0792ccba447678dcade9730a81149011c960c87cec3c",
    },
    "FS-007": {
        "normative_statement": "ede1bd232dd772ed07db8d45f83b6a720eea3b50612f69465dde88df720fd0c3",
        "claim_limit": "6f67ebafbe2e8d6eff72c731ef0ffa096626f1e77a09e182158d41f65348c85d",
    },
    "FS-008": {
        "normative_statement": "f45bc74aaa17c747e1ba70c0951d90bee73d981d81ced50c64a4adf4ceb4fb6d",
        "claim_limit": "bd2e5ad5a3c79fb4f1812865ae3ee9666e8b3230afcebad8554d514564304e45",
    },
    "FS-009": {
        "normative_statement": "63f49c8e518f8d1c973bffa45c08d7f65ee305038ee37ce35df1a780e29e6ed5",
        "claim_limit": "0afdf62a9b1deb1b853fde77a1ad1d2f657700ee2ce51c00f1cfa4db211b8348",
    },
    "FS-010": {
        "normative_statement": "5391c433e3a4a6e7682e88036237fb431c174f9882a7ba6ededc306d2cce032e",
        "claim_limit": "0f3b93235c6cbe25a1687bf9873ee778eb2efd83afafc6979e6fc2d50ac272c5",
    },
    "FS-011": {
        "normative_statement": "a93d68ca42fa0611ba18f7230b563811b48551c942430c260597f34ce4fa50ef",
        "claim_limit": "3a7e64fbdf851b6d2d1908427008bbcaba7ad682735a4345a94319389809a925",
    },
    "FS-012": {
        "normative_statement": "c6e0774a960e549bd314dfcef27350aebc97eb7b24a590c78a910608dc3594ba",
        "claim_limit": "5fde4bd357bcf1aa431a68745830e8c2e640b924959f961946c1855fbaba920f",
    },
    "FS-013": {
        "normative_statement": "2af484d3d8ea72f47a3989814fba5adc351df2be01ad61a853d64d4bd1e2916d",
        "claim_limit": "53e6c92cf53a5fb55ddcdc652aaca88e1dc48953f24ea8d9d71c1565dc2a84ab",
    },
    "FS-014": {
        "normative_statement": "c18d7cf411a338652260e5561cd5ff1224a6002fe94fef710e11e08a973e2ab5",
        "claim_limit": "b0f20a5b36c6a019b147737eaa790266838c5fd73d938d6e2d14570ec84a06fe",
    },
    "FS-015": {
        "normative_statement": "01e1377061a6848ad99a7feccdfcae80a6c6210762a15692e46f3ab527ad74b5",
        "claim_limit": "5ed882c812662c9eaed7603b3901814e1d4cd6d01ff90ab793e82232ab347a48",
    },
    "FS-016": {
        "normative_statement": "9dfa1056802d35e0a4c6fc5f0d391e654d4e856889df2d2617d44f5e4a545de4",
        "claim_limit": "b75cbcea233a6fbc705ad23fa56fd34bb9416ff0ffd72f715015ef9f34889d9e",
    },
    "FS-017": {
        "normative_statement": "8be70be45ec9d67c90d3ac69dd09c9204ede8dd6a146194e573ff659c1471c2d",
        "claim_limit": "d2ea947dfedb1716ffd7a0bff8ee714c60456f8f6db67226c2b91aba7820d0f4",
    },
    "FS-018": {
        "normative_statement": "c8e6048ebecb19a767f800774e2ae04e8cb3474abd13fe3beebca2e11f9dc1bf",
        "claim_limit": "c1f7c59ec286e8825a73c6227bc0e5075ef0be6797b2cdc7c1f0578ebf5ca627",
    },
}

EXPECTED_CLAIMS_BOUNDARY = {'endpoint_outputs_present': False,
 'full_numeric_protocol_frozen': False,
 'production_ready': False,
 'prohibited_claims': ['Endpoint 1 is the true or globally minimal sufficient circuit '
                       'size.',
                       'Endpoint 2 is the true number of distinct circuits.',
                       'Raw packing counts across discovery methods are perfectly '
                       'resource-matched.',
                       'Student initializations are independent population-level '
                       'replicates.',
                       'Hard-target and soft-target students may be pooled.',
                       'Fourier interchange establishes a unique algorithm or '
                       'mechanism.',
                       'Stage 2 makes the full protocol numerically final.'],
 'scientific_results_present': False,
 'stage3_authorized': False}


def validate_scientific_skeleton(record: dict[str, Any]) -> None:
    expected_root = {
        "schema_version",
        "namespace_version",
        "metadata",
        "authority",
        "freeze_scope",
        "frozen_items",
        "unresolved_register",
        "firewall",
        "claims_boundary",
    }

    if set(record) != expected_root:
        raise ScientificSkeletonError(
            f"root fields mismatch: expected={sorted(expected_root)} actual={sorted(record)}"
        )

    if record["schema_version"] != 1:
        raise ScientificSkeletonError("schema_version must equal 1")

    if record["namespace_version"] != 1:
        raise ScientificSkeletonError("namespace_version must equal 1")

    metadata = record["metadata"]
    if metadata != {
        "record_type": "stage2_scientific_skeleton_freeze",
        "stage": 2,
        "status": "partial_scientific_skeleton_frozen",
        "scientific_execution": False,
        "created_from_commit": "9118ecd239753c54fa5c66766e5d80b54d2a6259",
    }:
        raise ScientificSkeletonError("metadata violates Stage 2 freeze identity")

    if record["freeze_scope"] != {
        "is_partial_freeze": True,
        "numeric_protocol_fully_frozen": False,
        "teacher_registry_frozen": False,
        "stage3_started": False,
    }:
        raise ScientificSkeletonError("freeze_scope violates partial-freeze boundary")

    items = record["frozen_items"]
    ids = [x["item_id"] for x in items]

    expected_ids = [f"FS-{i:03d}" for i in range(1, 19)]

    if ids != expected_ids:
        raise ScientificSkeletonError(
            f"frozen item IDs must be exactly FS-001..FS-018; actual={ids}"
        )

    by_id = {x["item_id"]: x for x in items}

    for item_id in expected_ids:
        item = by_id[item_id]

        if not isinstance(item, dict) or set(item) != FROZEN_ITEM_KEYS:
            raise ScientificSkeletonError(
                f"{item_id} frozen-item fields changed"
            )

        metadata = EXPECTED_FROZEN_ITEM_METADATA[item_id]
        for field in ("name", "status", "authority_refs"):
            if item[field] != metadata[field]:
                raise ScientificSkeletonError(
                    f"{item_id} frozen-item metadata changed: {field}"
                )

        expected_digests = FROZEN_TEXT_SHA256[item_id]

        for field in ("normative_statement", "claim_limit"):
            value = item[field]

            if not isinstance(value, str):
                raise ScientificSkeletonError(
                    f"{item_id} {field} must remain a string"
                )

            actual_digest = hashlib.sha256(
                value.encode("utf-8")
            ).hexdigest()

            if actual_digest != expected_digests[field]:
                if field == "normative_statement":
                    raise ScientificSkeletonError(
                        f"{item_id} lost required frozen meaning"
                    )

                raise ScientificSkeletonError(
                    f"{item_id} lost required claim boundary"
                )

    claims_boundary = record["claims_boundary"]

    if claims_boundary != EXPECTED_CLAIMS_BOUNDARY:
        if not isinstance(claims_boundary, dict):
            raise ScientificSkeletonError(
                "claims_boundary violates frozen Stage 2 claim contract"
            )

        expected_keys = set(EXPECTED_CLAIMS_BOUNDARY)
        actual_keys = set(claims_boundary)
        mismatched_fields = sorted(
            (expected_keys ^ actual_keys)
            | {
                field
                for field in expected_keys & actual_keys
                if claims_boundary[field] != EXPECTED_CLAIMS_BOUNDARY[field]
            }
        )

        raise ScientificSkeletonError(
            "claims_boundary violates frozen Stage 2 claim contract; "
            f"mismatched_fields={mismatched_fields}"
        )


def load_scientific_skeleton(path: str | Path) -> dict[str, Any]:
    source = Path(path)

    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScientificSkeletonError(f"invalid JSON: {exc}") from exc

    if not isinstance(record, dict):
        raise ScientificSkeletonError("record root must be an object")

    validate_scientific_skeleton(record)
    return record
