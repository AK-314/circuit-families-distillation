from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class MethodDevelopmentFirewallError(ValueError):
    pass


ROOT_KEYS = {
    "schema_version",
    "namespace_version",
    "metadata",
    "authority",
    "firewall",
    "required_entry_fields",
    "allowed_dispositions",
    "entries",
}

METADATA_EXPECTED = {
    "record_type": "stage2_excluded_development_register",
    "stage": 2,
    "status": "firewall_active",
    "scientific_execution": False,
    "created_from_commit": "9118ecd239753c54fa5c66766e5d80b54d2a6259",
}

FIREWALL_EXPECTED = {
    "excluded_development_root": "followup/excluded_development/",
    "primary_analysis_eligible": False,
    "scientific_selection_eligible": False,
    "accidental_endpoint_output_action": "register_do_not_promote",
    "post_freeze_regeneration_required": True,
    "predecessor_output_forbidden": True,
    "absolute_private_canonical_paths_forbidden": True,
    "pilot_effects_may_select_protocol_values": False,
}

ENTRY_FIELDS = [
    "exclusion_id",
    "artifact_identity",
    "development_context",
    "exclusion_reason",
    "endpoint_values_emitted",
    "primary_analysis_eligible",
    "scientific_selection_eligible",
    "regeneration_required",
    "regenerate_after",
    "disposition",
    "promotion_in_place_permitted",
]

ALLOWED_DISPOSITIONS = [
    "registered_excluded",
    "regenerated_post_freeze",
    "superseded_by_post_freeze_regeneration",
]

FORBIDDEN_PREDECESSOR_STYLE_ROOTS = {
    "outputs",
    "results",
    "models",
    "checkpoints",
    "artifacts",
    "manifests",
}


def _strict_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)

    if missing:
        raise MethodDevelopmentFirewallError(
            f"{field} is missing required fields: {missing}"
        )

    if unknown:
        raise MethodDevelopmentFirewallError(
            f"{field} contains unknown fields: {unknown}"
        )


def _validate_artifact_identity(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MethodDevelopmentFirewallError(
            f"{field} must be a non-empty canonical artifact identity"
        )

    text = value.strip()

    if Path(text).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", text):
        raise MethodDevelopmentFirewallError(
            f"{field} must not be an absolute private canonical path"
        )

    parts = Path(text).parts

    if ".." in parts:
        raise MethodDevelopmentFirewallError(
            f"{field} must not contain parent traversal"
        )

    if parts and parts[0] in FORBIDDEN_PREDECESSOR_STYLE_ROOTS:
        raise MethodDevelopmentFirewallError(
            f"{field} points at a forbidden predecessor/output-style root: {parts[0]}"
        )

    if text.startswith("followup/") and not text.startswith(
        "followup/excluded_development/"
    ):
        raise MethodDevelopmentFirewallError(
            f"{field} is a follow-up development artifact outside "
            "followup/excluded_development/"
        )


def validate_exclusion_entry(entry: dict[str, Any]) -> None:
    if not isinstance(entry, dict):
        raise MethodDevelopmentFirewallError("exclusion entry must be an object")

    _strict_keys(entry, set(ENTRY_FIELDS), "exclusion entry")

    exclusion_id = entry["exclusion_id"]
    if not isinstance(exclusion_id, str) or not re.fullmatch(
        r"EXCL-\d{3,}", exclusion_id
    ):
        raise MethodDevelopmentFirewallError(
            "exclusion_id must match EXCL-NNN or a longer numeric suffix"
        )

    _validate_artifact_identity(
        entry["artifact_identity"],
        f"{exclusion_id}.artifact_identity",
    )

    if not isinstance(entry["development_context"], str) or not entry[
        "development_context"
    ].strip():
        raise MethodDevelopmentFirewallError(
            f"{exclusion_id}.development_context must be non-empty"
        )

    if not isinstance(entry["exclusion_reason"], str) or not entry[
        "exclusion_reason"
    ].strip():
        raise MethodDevelopmentFirewallError(
            f"{exclusion_id}.exclusion_reason must be non-empty"
        )

    if not isinstance(entry["endpoint_values_emitted"], bool):
        raise MethodDevelopmentFirewallError(
            f"{exclusion_id}.endpoint_values_emitted must be boolean"
        )

    if entry["primary_analysis_eligible"] is not False:
        raise MethodDevelopmentFirewallError(
            f"{exclusion_id} attempted primary-output classification; "
            "primary_analysis_eligible must remain false"
        )

    if entry["scientific_selection_eligible"] is not False:
        raise MethodDevelopmentFirewallError(
            f"{exclusion_id} attempted scientific-selection classification; "
            "scientific_selection_eligible must remain false"
        )

    if entry["regeneration_required"] is not True:
        raise MethodDevelopmentFirewallError(
            f"{exclusion_id} is missing mandatory post-freeze regeneration"
        )

    if not isinstance(entry["regenerate_after"], str) or not entry[
        "regenerate_after"
    ].strip():
        raise MethodDevelopmentFirewallError(
            f"{exclusion_id}.regenerate_after must identify a later freeze/stage"
        )

    if entry["disposition"] not in ALLOWED_DISPOSITIONS:
        raise MethodDevelopmentFirewallError(
            f"{exclusion_id} has unknown disposition: {entry['disposition']!r}"
        )

    if entry["promotion_in_place_permitted"] is not False:
        raise MethodDevelopmentFirewallError(
            f"{exclusion_id} cannot be promoted in place"
        )


def validate_firewall_register(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise MethodDevelopmentFirewallError("firewall register root must be an object")

    _strict_keys(record, ROOT_KEYS, "firewall register")

    if record["schema_version"] != 1:
        raise MethodDevelopmentFirewallError("schema_version must equal 1")

    if record["namespace_version"] != 1:
        raise MethodDevelopmentFirewallError("namespace_version must equal 1")

    if record["metadata"] != METADATA_EXPECTED:
        raise MethodDevelopmentFirewallError(
            "metadata violates Stage 2 firewall identity"
        )

    authority = record["authority"]
    if not isinstance(authority, list) or not authority:
        raise MethodDevelopmentFirewallError(
            "authority must be a non-empty array"
        )

    for index, entry in enumerate(authority):
        if not isinstance(entry, dict):
            raise MethodDevelopmentFirewallError(
                f"authority[{index}] must be an object"
            )

        expected = {
            "authority_id",
            "precedence",
            "repository_path",
            "git_blob",
            "sha256",
        }
        _strict_keys(entry, expected, f"authority[{index}]")

        path = entry["repository_path"]
        if not isinstance(path, str) or not path:
            raise MethodDevelopmentFirewallError(
                f"authority[{index}].repository_path must be non-empty"
            )
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise MethodDevelopmentFirewallError(
                f"authority[{index}].repository_path must be portable"
            )

    if record["firewall"] != FIREWALL_EXPECTED:
        raise MethodDevelopmentFirewallError(
            "firewall semantics changed: excluded output must remain "
            "non-primary, non-selective, and regeneration-required"
        )

    if record["required_entry_fields"] != ENTRY_FIELDS:
        raise MethodDevelopmentFirewallError(
            "required_entry_fields do not match the frozen Stage 2 exclusion contract"
        )

    if record["allowed_dispositions"] != ALLOWED_DISPOSITIONS:
        raise MethodDevelopmentFirewallError(
            "allowed_dispositions do not match the frozen Stage 2 exclusion contract"
        )

    entries = record["entries"]
    if not isinstance(entries, list):
        raise MethodDevelopmentFirewallError("entries must be an array")

    ids: list[str] = []
    for entry in entries:
        validate_exclusion_entry(entry)
        ids.append(entry["exclusion_id"])

    if len(ids) != len(set(ids)):
        raise MethodDevelopmentFirewallError(
            "duplicate exclusion ID in excluded-development register"
        )


def require_registered_development_output(
    register: dict[str, Any],
    artifact_identity: str,
) -> dict[str, Any]:
    validate_firewall_register(register)

    matches = [
        entry
        for entry in register["entries"]
        if entry["artifact_identity"] == artifact_identity
    ]

    if not matches:
        raise MethodDevelopmentFirewallError(
            f"unregistered development output: {artifact_identity}"
        )

    if len(matches) != 1:
        raise MethodDevelopmentFirewallError(
            f"development output has non-unique exclusion registration: "
            f"{artifact_identity}"
        )

    return matches[0]


def assert_pilot_evidence_may_not_select_protocol_values(
    *,
    evidence_kind: str,
    intended_use: str,
) -> None:
    forbidden_evidence = {
        "pilot_phase_effect",
        "pilot_condition_effect",
        "pilot_student_effect",
        "pilot_endpoint_effect",
    }

    protocol_selection_uses = {
        "choose_protocol_value",
        "choose_threshold",
        "choose_phase_landmark",
        "choose_student_count",
        "choose_method_roster",
        "choose_search_cutoff",
        "choose_budget",
    }

    if evidence_kind in forbidden_evidence and intended_use in protocol_selection_uses:
        raise MethodDevelopmentFirewallError(
            "pilot phase/condition/student/endpoint effects are forbidden "
            "evidence for choosing prospective protocol values"
        )


def load_firewall_register(path: str | Path) -> dict[str, Any]:
    source = Path(path)

    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MethodDevelopmentFirewallError(
            f"invalid firewall-register JSON: {exc}"
        ) from exc

    if not isinstance(record, dict):
        raise MethodDevelopmentFirewallError(
            "firewall register root must be an object"
        )

    validate_firewall_register(record)
    return record
