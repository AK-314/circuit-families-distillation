from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


class Stage11DesignResolutionError(ValueError):
    """Raised when the Stage 11 prospective design-resolution record is invalid."""


EXPECTED_RD_IDS = tuple(f"RD-{number:03d}" for number in range(1, 15))
EXPECTED_RED_TEAM_COUNT = 20

MATRIX_PATH = Path("docs/distillation_followup/red_team/red_team_resolution_matrix.md")
OPEN_DECISIONS_PATH = Path("followup/configs/post_red_team_open_decisions_v1.json")
HISTORICAL_STAGE2_PATH = Path("followup/configs/stage2_unresolved_decisions_v1.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11DesignResolutionError(message)


def _matrix_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]

        if len(cells) != 4:
            continue

        if cells[0] == "Issue":
            continue

        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue

        rows.append(
            {
                "issue": cells[0],
                "disposition": cells[1],
                "protocol_consequence": cells[2],
                "remaining_decision": cells[3],
            }
        )

    return rows


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Stage11DesignResolutionError(f"missing Stage 11 authority: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Stage11DesignResolutionError(
            f"invalid JSON in {path}: line={exc.lineno} column={exc.colno}"
        ) from exc


def _validate_string_list(
    value: Any,
    *,
    field: str,
    nonempty: bool = True,
) -> None:
    _require(isinstance(value, list), f"{field} must be a list")
    _require(
        all(isinstance(item, str) and item.strip() for item in value),
        f"{field} must contain only non-empty strings",
    )
    if nonempty:
        _require(bool(value), f"{field} must not be empty")


def validate_stage11_resolution_record(
    record: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    _require(
        record.get("record_type") == "stage11_red_team_resolution",
        "wrong record_type",
    )
    _require(record.get("schema_version") == 1, "wrong schema_version")
    _require(record.get("stage") == 11, "wrong stage")
    _require(
        record.get("scientific_execution") is False,
        "scientific_execution must remain false",
    )
    _require(
        record.get("definitive_outcomes_consulted") is False,
        "definitive_outcomes_consulted must remain false",
    )
    _require(
        record.get("historical_stage2_register_mutated") is False,
        "historical Stage 2 register must remain immutable",
    )

    historical = record.get("historical_stage2_guard")
    _require(
        isinstance(historical, dict),
        "historical_stage2_guard must be an object",
    )
    _require(
        historical.get("repository_path") == HISTORICAL_STAGE2_PATH.as_posix(),
        "historical Stage 2 path mismatch",
    )

    historical_path = repo_root / HISTORICAL_STAGE2_PATH
    _require(
        historical_path.is_file(),
        "historical Stage 2 register missing",
    )
    _require(
        historical.get("sha256") == sha256(historical_path),
        "historical Stage 2 register mutation detected",
    )

    authority = record.get("authority")
    _require(isinstance(authority, dict), "authority must be an object")

    matrix_path = repo_root / MATRIX_PATH
    decisions_path = repo_root / OPEN_DECISIONS_PATH

    _require(matrix_path.is_file(), "red-team matrix missing")
    _require(decisions_path.is_file(), "post-red-team decision register missing")

    _require(
        authority.get("red_team_matrix", {}).get("repository_path") == MATRIX_PATH.as_posix(),
        "red-team matrix authority path mismatch",
    )
    _require(
        authority.get("red_team_matrix", {}).get("sha256") == sha256(matrix_path),
        "red-team matrix authority hash mismatch",
    )
    _require(
        authority.get("open_decisions", {}).get("repository_path")
        == OPEN_DECISIONS_PATH.as_posix(),
        "open-decision authority path mismatch",
    )
    _require(
        authority.get("open_decisions", {}).get("sha256") == sha256(decisions_path),
        "open-decision authority hash mismatch",
    )

    canonical_matrix = _matrix_rows(matrix_path)
    _require(
        len(canonical_matrix) == EXPECTED_RED_TEAM_COUNT,
        "canonical red-team matrix row count drifted",
    )

    red_team_items = record.get("red_team_items")
    _require(
        isinstance(red_team_items, list),
        "red_team_items must be a list",
    )
    _require(
        len(red_team_items) == EXPECTED_RED_TEAM_COUNT,
        "missing red-team rows",
    )

    ids = [item.get("item_id") for item in red_team_items]
    expected_rt_ids = [f"RT-{number:03d}" for number in range(1, EXPECTED_RED_TEAM_COUNT + 1)]
    _require(
        ids == expected_rt_ids,
        "red-team row IDs must be complete, unique, and ordered",
    )

    for index, (item, canonical) in enumerate(
        zip(red_team_items, canonical_matrix, strict=True),
        1,
    ):
        for field in (
            "issue",
            "disposition",
            "protocol_consequence",
            "remaining_decision",
        ):
            _require(
                item.get(field) == canonical[field],
                f"red-team row RT-{index:03d} drifted field={field}",
            )

        for field in (
            "scientific_reason",
            "required_implementation_packages",
            "required_freeze_stages",
            "analysis_consequence",
            "permitted_claim",
            "prohibited_claim",
        ):
            value = item.get(field)
            if field in {
                "required_implementation_packages",
                "required_freeze_stages",
            }:
                _validate_string_list(
                    value,
                    field=f"RT-{index:03d}.{field}",
                )
            else:
                _require(
                    isinstance(value, str) and value.strip(),
                    f"RT-{index:03d}.{field} must be non-empty",
                )

        _require(
            isinstance(item.get("failure_or_unavailability_reportable"), bool),
            f"RT-{index:03d}.failure_or_unavailability_reportable must be boolean",
        )
        _require(
            isinstance(item.get("blocks_stage15"), bool),
            f"RT-{index:03d}.blocks_stage15 must be boolean",
        )

        disposition = str(item["disposition"]).lower()
        if disposition.startswith("accepted"):
            _require(
                bool(item["required_implementation_packages"]),
                f"accepted item RT-{index:03d} has no implementation consequence",
            )

    prohibited_claims = {
        item["prohibited_claim"].strip()
        for item in red_team_items
        if item["prohibited_claim"].strip()
    }
    for item in red_team_items:
        permitted = item["permitted_claim"].strip()
        _require(
            permitted not in prohibited_claims,
            "rejected claim silently reintroduced elsewhere",
        )

    open_decisions = _load_json(decisions_path)
    canonical_decisions = open_decisions.get("decisions")
    _require(
        isinstance(canonical_decisions, list),
        "canonical RD decisions must be a list",
    )

    canonical_by_id = {item["decision_id"]: item for item in canonical_decisions}
    _require(
        tuple(sorted(canonical_by_id)) == EXPECTED_RD_IDS,
        "canonical RD register does not contain exact RD-001 through RD-014",
    )

    rd_items = record.get("rd_items")
    _require(isinstance(rd_items, list), "rd_items must be a list")

    rd_ids = [item.get("decision_id") for item in rd_items]
    counts = Counter(rd_ids)
    duplicates = sorted(str(decision_id) for decision_id, count in counts.items() if count > 1)
    missing = sorted(set(EXPECTED_RD_IDS) - set(rd_ids))
    unknown = sorted(set(rd_ids) - set(EXPECTED_RD_IDS))

    _require(not missing, f"missing RD identifiers: {missing}")
    _require(not duplicates, f"duplicate RD identifiers: {duplicates}")
    _require(not unknown, f"unknown RD identifiers: {unknown}")
    _require(
        tuple(rd_ids) == EXPECTED_RD_IDS,
        "RD identifiers must be ordered RD-001 through RD-014",
    )

    for item in rd_items:
        rid = item["decision_id"]
        canonical = canonical_by_id[rid]

        # A production-blocking decision may only transition to "resolved"
        # when an explicit versioned resolution record accompanies it. Check
        # this before canonical-status anchoring so adversarial mutations receive
        # the specific diagnostic required by the Stage 11 handoff.
        if item.get("status") == "resolved":
            resolution_record = item.get("resolution_record")
            _require(
                isinstance(resolution_record, dict),
                f"{rid} marked resolved without a resolution record",
            )
            _require(
                isinstance(resolution_record.get("repository_path"), str)
                and resolution_record["repository_path"],
                f"{rid} resolution record lacks repository_path",
            )
            _require(
                isinstance(resolution_record.get("sha256"), str)
                and re.fullmatch(
                    r"[0-9a-f]{64}",
                    resolution_record["sha256"],
                ),
                f"{rid} resolution record lacks valid sha256",
            )

        for field in (
            "family",
            "historical_links",
            "resolution_stage",
            "required_resolution",
            "working_direction",
            "status",
        ):
            _require(
                item.get(field) == canonical[field],
                f"{rid} canonical field drifted: {field}",
            )

        for field in (
            "disposition",
            "scientific_reason",
            "analysis_consequence",
            "permitted_claim",
            "prohibited_claim",
        ):
            _require(
                isinstance(item.get(field), str) and item[field].strip(),
                f"{rid}.{field} must be non-empty",
            )

        _validate_string_list(
            item.get("required_implementation_packages"),
            field=f"{rid}.required_implementation_packages",
        )
        _validate_string_list(
            item.get("required_freeze_stages"),
            field=f"{rid}.required_freeze_stages",
        )

        _require(
            isinstance(item.get("failure_or_unavailability_reportable"), bool),
            f"{rid}.failure_or_unavailability_reportable must be boolean",
        )
        _require(
            isinstance(item.get("blocks_stage15"), bool),
            f"{rid}.blocks_stage15 must be boolean",
        )
        _require(
            item.get("blocks_stage15") is True,
            f"{rid} is production-blocking and must block Stage 15",
        )

        resolution_state = item.get("stage11_resolution_state")
        _require(
            resolution_state
            in {
                "accepted_direction_only",
                "candidate_interface_only",
                "production_value_still_open",
            },
            f"{rid} has invalid stage11_resolution_state",
        )

    _require(
        record.get("full_factorial_required") is False,
        "literal full factorial must remain rejected",
    )


def load_stage11_resolution_record(
    path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    record = _load_json(path)
    _require(isinstance(record, dict), "Stage 11 record root must be an object")
    validate_stage11_resolution_record(record, repo_root=repo_root)
    return record


def validate_stage11_design_candidates(record: dict[str, Any]) -> None:
    _require(
        record.get("record_type") == "stage11_post_red_team_design_candidates",
        "wrong Stage 11 candidate record_type",
    )
    _require(record.get("scientific_execution") is False, "scientific execution forbidden")
    _require(
        record.get("production_eligible") is False,
        "Stage 11 candidates cannot be production eligible",
    )
    _require(
        record.get("historical_stage4_identity_mutated") is False,
        "Stage 4 identity must not be mutated",
    )

    population = record["population"]
    _require(
        population["population_unit"] == "teacher_seed", "teacher seed must remain population unit"
    )
    _require(
        population["protected_primary_task"] == "modular_addition_mod_113",
        "protected primary task drifted",
    )
    _require(
        population["phase_selection_evidence"] == "training_metrics_only",
        "phase selection must use training metrics only",
    )
    _require(
        population["exact_teacher_roster"] is None,
        "exact teacher roster must remain open in Stage 11",
    )
    _require(
        population["exact_teacher_count"] is None,
        "exact teacher count must remain open in Stage 11",
    )
    _require(
        population["unavailable_teacher_phase_cells_reportable"] is True,
        "unavailable teacher cells must remain reportable",
    )

    tasks = record["tasks"]
    _require(
        [t["task_slot"] for t in tasks] == ["Task 1", "Task 2", "Task 3"],
        "Task 1-3 roster mismatch",
    )
    _require(tasks[0]["task_family"] == "modular_addition", "Task 1 must be modular addition")
    _require(
        tasks[0]["modulus"] == 113 and tasks[0]["scope"] == "full_domain", "Task 1 contract drifted"
    )
    _require(
        tasks[1]["task_family"] == "modular_multiplication", "Task 2 must be modular multiplication"
    )
    _require(tasks[1]["scope"] == "reduced", "Task 2 must remain reduced")
    _require(tasks[1]["modulus"] is None, "Task 2 production modulus must remain open")
    _require(tasks[2]["task_family"] == "modular_polynomial", "Task 3 must be modular polynomial")
    _require(tasks[2]["formula_terms"] is None, "Task 3 formula must remain open")
    _require(
        tasks[2]["selection_must_be_outcome_independent"] is True,
        "Task 3 formula selection must be outcome independent",
    )
    _require(
        record["task_scope_rules"]["modular_subtraction_counts_as_task_breadth"] is False,
        "subtraction cannot substitute for task breadth",
    )

    arch = record["architectures"]
    _require(
        arch["canonical_family"] == "predecessor_matched",
        "canonical architecture must remain predecessor matched",
    )
    _require(
        arch["candidate_panel_maximum_families"] <= 5, "architecture panel exceeds accepted maximum"
    )
    _require(arch["exact_family_roster"] is None, "exact architecture roster must remain open")
    _require(
        arch["assignment_strategy"] == "balanced_sparse_assignment",
        "architecture assignment must be balanced sparse",
    )
    _require(arch["literal_full_factorial"] is False, "literal full factorial is prohibited")
    _require(
        arch["isolated_depth_or_width_causal_claim_permitted"] is False,
        "heterogeneous panel cannot imply isolated depth/width causality",
    )

    bases = {item["basis_family"]: item for item in record["component_bases"]}
    _require(
        set(bases)
        == {
            "canonical",
            "attention_refinement",
            "coarse_mlp_blocks",
            "orientation_sensitivity",
            "accounting_sensitivity",
        },
        "required basis families incomplete",
    )
    _require(
        bases["canonical"]["definition"] == "attention_heads_plus_individual_mlp_neurons",
        "canonical basis drifted",
    )
    _require(
        bases["attention_refinement"]["definition"]
        == "pre_output_projection_attention_coordinates",
        "attention refinement must be pre-output-projection",
    )
    _require(bases["coarse_mlp_blocks"]["block_count"] is None, "MLP block count must remain open")
    _require(
        bases["orientation_sensitivity"]["rotation_count"] is None,
        "rotation count must remain open",
    )
    _require(
        record["basis_claim_boundary"]["claims_limited_to_tested_bases"] is True,
        "basis claims must be limited to tested bases",
    )
    _require(
        record["basis_claim_boundary"]["universal_basis_invariance_claim_permitted"] is False,
        "universal basis invariance claim forbidden",
    )

    compatibility = record["compatibility"]
    _require(
        compatibility["stage4_identity_schema_rewritten"] is False,
        "Stage 4 identity rewrite forbidden",
    )
    _require(
        compatibility["new_factors_are_versioned_references_not_stage4_identity_mutations"] is True,
        "new factors must be versioned references",
    )


def load_stage11_design_candidates(path: Path) -> dict[str, Any]:
    record = _load_json(path)
    _require(isinstance(record, dict), "Stage 11 candidate root must be an object")
    validate_stage11_design_candidates(record)
    return record


def validate_stage11_part_d(record: dict[str, Any]) -> None:
    endpoint1 = record["endpoints"]["endpoint1"]
    endpoint2 = record["endpoints"]["endpoint2"]

    _require(endpoint1["status"] == "primary", "Endpoint 1 must remain primary")
    _require(
        endpoint1["intact_mask_always_included"] is True, "Endpoint 1 must include intact mask"
    )
    _require(endpoint1["global_minimum_claim_permitted"] is False, "global minimum claim forbidden")
    _require(
        endpoint1["primary_fidelity_threshold"] is None,
        "primary fidelity threshold must remain open",
    )
    _require(endpoint1["fidelity_frontier"] is None, "fidelity frontier grid must remain open")
    _require(
        endpoint1["ledger_reuse_required_where_supported"] is True, "ledger reuse must be required"
    )

    _require(
        endpoint2["status"] == "key_secondary", "Endpoint 2 must remain key secondary in Stage 11"
    )
    _require(
        endpoint2["co_primary_requires_preproduction_validation"] is True,
        "Endpoint 2 co-primary status requires validation",
    )
    _require(
        endpoint2["mechanism_count_claim_permitted"] is False, "mechanism-count claim forbidden"
    )
    for field in ("component_cap", "overlap_metric", "overlap_cutoff", "packing_solver"):
        _require(endpoint2[field] is None, f"Endpoint 2 {field} must remain open")

    discovery = record["discovery"]
    families = discovery["required_families"]
    _require(
        [x["family"] for x in families]
        == [
            "inherited_discrete",
            "independent_continuous_or_stochastic",
        ],
        "discovery family roster drifted",
    )
    _require(
        discovery["algorithmic_independence_required"] is True,
        "independent discovery family required",
    )
    _require(
        discovery["cosmetic_restart_of_inherited_method_acceptable"] is False,
        "cosmetic restart cannot satisfy independence",
    )
    _require(
        discovery["common_exact_evaluation_allowance"] is None, "exact allowance must remain open"
    )
    _require(
        discovery["native_budgets_declared_equivalent"] is False,
        "unlike native budgets cannot be declared equivalent",
    )
    for family in families:
        _require(family["native_budget"] is None, "production native budget must remain open")

    calibration = record["packing_calibration"]
    _require(
        set(calibration["required_nulls"])
        == {
            "combinatorial_size_and_type_matched_floor",
            "ordinary_restart_baseline",
            "local_fidelity_retaining_perturbation",
            "tractable_feasible_region_calibration",
        },
        "all four packing calibration layers are required",
    )
    _require(calibration["null_draw_counts"] is None, "null draw counts must remain open")
    _require(
        calibration["nulls_may_be_collapsed_into_one"] is False, "packing nulls cannot be collapsed"
    )
    _require(
        calibration["tractable_model"]["task_and_model"] is None,
        "tractable calibration model must remain open",
    )
    _require(
        calibration["tractable_model"]["population_replicate"] is False,
        "tractable model is not a population replicate",
    )

    fourier = record["fourier_interchange"]
    _require(
        fourier["status"] == "registered_key_secondary", "Fourier analysis must be key secondary"
    )
    _require(
        fourier["execute_regardless_of_primary_direction"] is True,
        "Fourier analysis must be outcome-independent",
    )
    _require(
        fourier["pair_selection_may_use_candidate_outcomes"] is False,
        "Fourier pair selection cannot use candidate outcomes",
    )
    _require(
        set(fourier["required_controls"])
        == {
            "wrong_fourier_mode",
            "shuffled_coefficients",
            "mismatched_input",
            "equal_norm_random_state",
            "unaligned_ordinary_activation_patching",
        },
        "Fourier control roster incomplete",
    )
    _require(
        fourier["shared_abstraction_claim_requires_all_controls_outperformed"] is True,
        "shared-abstraction claim requires all controls",
    )
    _require(
        fourier["uniquely_identified_algorithm_claim_permitted"] is False,
        "Fourier uniqueness claim forbidden",
    )
    for field in (
        "intervention_location",
        "alignment_rule",
        "capacity_matching_rule",
        "outcome",
        "trial_roster",
    ):
        _require(fourier[field] is None, f"Fourier {field} must remain open")

    tiering = record["tiering"]
    _require(tiering["literal_full_factorial"] is False, "literal full factorial is prohibited")
    _require(tiering["tier1"]["status"] == "protected_primary", "Tier 1 must be protected")
    _require(tiering["tier2"]["status"] == "protected_minimum", "Tier 2 minimum must be protected")
    _require(tiering["tier2"]["exact_minimum"] is None, "Tier 2 exact minimum must remain open")
    _require(tiering["tier3"]["status"] == "optional", "Tier 3 must remain optional")
    _require(tiering["tier3"]["priority_order"] is None, "Tier 3 exact priority must remain open")
    _require(
        tiering["tier3"]["may_run_before_tier1_complete_and_tier2_minimum_secure"] is False,
        "Tier 3 cannot pre-empt protected work",
    )

    resources = record["resources"]
    _require(
        resources["symbolica_unavailable_branch_required"] is True,
        "Symbolica-unavailable branch required",
    )
    _require(resources["school_mac_branch_required"] is True, "school-Mac branch required")
    _require(
        resources["incomplete_campaign_recovery_branch_required"] is True,
        "incomplete recovery branch required",
    )
    _require(
        resources["school_mac_priority"]
        == [
            "protected_primary_recovery",
            "reproduction_and_integrity",
            "task2_after_protected_requirements",
            "task3_only_after_completeness_gate",
        ],
        "school-Mac priority drifted",
    )
    _require(
        resources["reverse_priority_shedding"]
        == [
            "tier3_optional_breadth",
            "unprotected_tier2_work",
            "protected_tier2_minimum",
            "tier1_never_shed_for_optional_work",
        ],
        "reverse-priority shedding order drifted",
    )
    for field in (
        "cluster_concurrency",
        "retry_caps",
        "storage_quotas",
        "final_audit_export_reserve",
        "rebuttal_reserve_amount",
    ):
        _require(resources[field] is None, f"resource {field} must remain open")


def load_stage11_complete_candidates(path: Path) -> dict[str, Any]:
    record = load_stage11_design_candidates(path)
    validate_stage11_part_d(record)
    return record


def validate_stage11_report_skeleton(record: dict[str, Any]) -> None:
    report = record["planned_report_skeleton"]
    _require(
        report["synthetic_only"] is True,
        "planned report must remain synthetic",
    )
    _require(
        report["scientific_values_present"] is False,
        "Stage 11 report cannot contain scientific values",
    )
    _require(
        report["numeric_placeholders"] == "forbidden_in_stage11",
        "Stage 11 must not invent numeric report values",
    )
    expected = [
        "attempt_eligibility_and_unavailability_accounting",
        "direct_teacher_endpoint1",
        "hard_student_endpoint1",
        "soft_student_endpoint1",
        "realization_and_teacher_student_contrasts",
        "discovery_method_dependence",
        "packing_and_null_calibration",
        "architecture_external_validity",
        "basis_sensitivity",
        "fidelity_frontier",
        "fourier_interchange_and_controls",
        "claim_resolution",
    ]
    _require(
        report["sections"] == expected,
        "planned report section order drifted",
    )
