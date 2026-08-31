#!/usr/bin/env python3
"""Build the revised, still-unapproved Stage 13 decision dossier."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "followup/decisions/stage13_decision_dossier_v1.json"
OUTPUT = ROOT / "followup/decisions/stage13_decision_dossier_v2.json"
BENCHMARK = ROOT / "followup/benchmarks/stage13_search_profile_benchmark_v1.json"
PROJECTION = ROOT / "followup/manifests/stage13_package_resource_projection_v2.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(record: dict[str, object], decision_id: str) -> dict[str, object]:
    return next(item for item in record["decisions"] if item["decision_id"] == decision_id)


def main() -> int:
    dossier = json.loads(V1.read_text(encoding="utf-8"))
    dossier["schema_version"] = "stage13-decision-dossier/v2"
    dossier["recommended_package_id"] = "stage13-package-a-conservative-protected/v2"
    dossier["supersedes"] = {
        "path": "followup/decisions/stage13_decision_dossier_v1.json",
        "file_sha256": _sha256(V1),
        "approval_status": "never_approved",
    }
    dossier["benchmark_evidence"] = {
        "path": "followup/benchmarks/stage13_search_profile_benchmark_v1.json",
        "file_sha256": _sha256(BENCHMARK),
        "classification": "constructed non-scientific evidence",
        "hard_concrete_profile": "516 components, 5000 steps, checkpoint every 50, retention 2; median 2623.751 native steps/s",
        "ordinary_restart_profile": "16 restarts, 256 exact allowance including intact; median 0.667390 s for bridge mechanics",
        "scope_limit": "does not benchmark model forward/backward work, CUDA, or campaign concurrency",
    }
    dossier["resource_projection"] = {
        "path": "followup/manifests/stage13_package_resource_projection_v2.json",
        "file_sha256": _sha256(PROJECTION),
        "envelope": "96 hours total; 84 hours science plus an inviolate final 12-hour audit; 256 CPU cores; 16 CUDA GPUs; 4 TiB scratch; 1 TiB persistent; grant still unverified",
        "conclusion": "A and C fit lower and central scenarios but not conservative; B fits lower and central but not conservative; no package may launch until a complete Stage 14 execution branch passes frozen throughput, memory, storage, and backend qualification",
    }

    rd2 = _decision(dossier, "RD-002")
    rd2["stage12_evidence"] = (
        "P2 validates injected multi-layer architectures, architecture-bound identities, parameter/component accounting, "
        "checkpoint resume, eligibility, and sealing. The v1 disjoint-seed allocation was not contrast-identifiable and is superseded."
    )
    rd2["viable_options"][0]["description"] = (
        "Five-family panel: canonical plus depth, head-granularity, narrow-MLP, and wide-MLP variants, all crossed with the same fixed anchor seeds 0-4."
    )
    rd2["protected_tier_consequences"]["tier2"] = (
        "All four alternates share anchor seeds 0-4, pre/stable phases, both conditions, and two eligible students per cell; 80 cells and 160 eligible-student targets."
    )
    rd2["recommendation"]["tier2_assignment"] = {
        "anchor_teacher_seeds": [0, 1, 2, 3, 4],
        "architectures_on_every_anchor": [
            "depth-2-matched-width/v1",
            "heads-8-matched-width/v1",
            "mlp-256-matched-residual/v1",
            "mlp-1024-matched-residual/v1",
        ],
        "phases": ["pre-grokking", "stable post-grokking"],
        "conditions": ["hard", "soft"],
        "assignment_type": "complete shared fixed-anchor block",
        "unavailable_rule": "retain unavailable; no seed or checkpoint replacement",
    }
    rd2["recommendation"]["permitted_contrasts"] = (
        "Within the identical teacher seed, phase, and condition, alternate-architecture student-cell median minus canonical-architecture student-cell median; aggregate only complete pairs over teacher seeds. "
        "Architecture results are conditional external-validity contrasts, not isolated causal effects of depth, heads, or width."
    )
    rd2["rationale"] = (
        "Every alternate is connected to canonical through the same teacher checkpoints, so architecture contrasts no longer confound architecture with teacher seed."
    )

    rd4 = _decision(dossier, "RD-004")
    rd4["stage12_evidence"] = (
        "R2 implements canonical round trips, pre-output attention coordinates, seeded balanced blocks, seeded orthogonal rotations, cross-basis lifecycle checks, and parameter/type accounting. The v1 family-balanced but disjoint assignment is superseded."
    )
    rd4["recommendation"]["assignment"] = {
        "anchor_teacher_seeds": [0, 1, 2, 3, 4],
        "phases": ["pre-grokking", "stable post-grokking"],
        "model_roles_per_checkpoint": [
            "direct teacher",
            "lowest fixed eligible canonical hard student",
            "lowest fixed eligible canonical soft student",
        ],
        "bases_on_every_available_model": [
            "canonical heads-plus-neurons",
            "attention-coordinate refinement",
            "balanced 32-neuron blocks partition-0",
            "MLP rotation-0",
            "MLP rotation-1",
        ],
        "maximum_model_checkpoint_identities": 30,
        "alternative_basis_ledgers_per_method": 120,
        "assignment_type": "complete shared anchor/checkpoint panel",
        "unavailable_rule": "retain unavailable and compare only exact checkpoint pairs; no replacement",
    }
    rd4["recommendation"]["permitted_contrasts"] = (
        "For the same model/checkpoint, discovery method, phase, condition, and teacher seed, compare each alternative-basis Endpoint 1 and packing reduction with its canonical-basis reduction. Aggregate paired differences by teacher seed; raw component counts are never compared without raw, parameter-weighted, and component-type denominators."
    )
    rd4["rationale"] = (
        "Every basis is evaluated on the same fixed model/checkpoint identities, making all permitted comparisons paired and connected across teacher seeds."
    )

    rd6 = _decision(dossier, "RD-006")
    rd6["protected_tier_consequences"]["tier2"] = (
        "The complete 3 x 4 component-cap/overlap grid is recomputed from every applicable sealed exact ledger; all 12 cells are descriptive sensitivities."
    )
    rd6["recommendation"]["sensitivity_grid"] = {
        "component_cap_proportions": [0.125, 0.25, 0.375],
        "maximum_pairwise_jaccard_overlaps": [0.0, 0.125, 0.25, 0.5],
        "cartesian_cell_count": 12,
        "cells": [
            {"component_cap": cap, "maximum_pairwise_overlap": overlap}
            for cap in (0.125, 0.25, 0.375)
            for overlap in (0.0, 0.125, 0.25, 0.5)
        ],
        "hierarchy": {
            "primary": "Endpoint 1 only",
            "key_secondary": "packing at component cap 0.25 and maximum Jaccard overlap 0.25",
            "descriptive_sensitivity": "the other 11 grid cells",
        },
        "execution": "reducer-only from masks and exact fidelities already present in each sealed ledger; no rediscovery, new model inference, new exact evaluation, or ledger mutation",
        "ledger_limit": "if a mask is absent from the sealed ledger, no sensitivity cell may generate or evaluate it",
    }
    rd6["rationale"] = (
        "Endpoint 1 remains the sole primary. The key-secondary packing setting is fixed, and the full Cartesian sensitivity grid measures procedure dependence without creating new search opportunities."
    )

    rd7 = _decision(dossier, "RD-007")
    rd7["stage12_evidence"] = (
        "Stage 6D provides inherited adapters and budget ledgers; R1 provides continuous stochastic gates, optimizer resume, proposal extraction, and exact bridge. Stage 10 measured exact evaluation at 0.088445375 seconds on the technical CPU fixture. The constructed Stage 13 profile measured 5000 native gate steps at median 1.905668 seconds and the 16-restart bridge at median 0.667390 seconds; model-in-the-loop CUDA cost remains unbenchmarked."
    )
    rd7["recommendation"]["common_exact_evaluation_allowance"] = 256
    rd7["recommendation"]["exact_allowance_semantics"] = (
        "256 total exact evaluations per model-method ledger, including the mandatory intact mask; duplicate requests reuse evidence and do not recharge the ledger"
    )
    rd7["uncertainty"] = (
        "Native mechanics are benchmarked, but the production objective forward/backward pass and CUDA concurrency are not; Stage 14 must pass a frozen complete-branch throughput equation or block launch."
    )

    rd8 = _decision(dossier, "RD-008")
    rd8["protected_tier_consequences"]["tier1"] = (
        "A fixed 60-slot canonical anchor calibration panel: seeds 0-4 x pre/stable x direct/lowest-hard/lowest-soft x two methods; unavailable slots remain unavailable."
    )
    rd8["recommendation"]["panel_assignment"] = {
        "teacher_seeds": [0, 1, 2, 3, 4],
        "phases": ["pre-grokking", "stable post-grokking"],
        "model_roles": ["direct teacher", "lowest fixed eligible canonical hard student", "lowest fixed eligible canonical soft student"],
        "discovery_methods": ["greedy-deletion-centred-logit/v1", "hard-concrete-gates-centred-logit/v1"],
        "maximum_profile_slots": 60,
        "replacement": "none",
    }
    rd8["recommendation"]["ordinary_restart"]["exact_allowance_semantics"] = (
        "256 total including intact in Package A/C; 128 total including intact in Package B"
    )

    rd11 = _decision(dossier, "RD-011")
    rd11["viable_options"][0]["description"] = (
        "Teacher-to-canonical-student pairs on a fixed internal MLP-post location; diagonal modular-sum Fourier characters; disjoint-fit alignment; counterfactual centered-logit displacement fidelity."
    )
    rd11["protected_tier_consequences"]["tier2"] = (
        "Up to 60 comparison sets (seed 0-14 x pre/stable x hard/soft), each with aligned intervention plus all five capacity-matched controls and 256 fixed recipient/source trials per condition."
    )
    rd11["recommendation"] = {
        "option": "A",
        "pair_rule": "for each seed 0-14, each primary phase, and each hard/soft condition, source is the direct teacher and recipient is the lowest fixed initialization slot that is eligible; if none is eligible the pair is unavailable and is not replaced",
        "maximum_pair_count": 60,
        "internal_location": "final-token blocks.0.mlp.hook_post, after the nonlinear MLP activation and before W_out writes into the residual stream",
        "location_justification": "modular-addition task variables are expected to be synthesized in nonlinear MLP features; intervention before W_out tests an internal causal state and leaves the recipient's W_out, residual addition, normalization, and unembedding to translate that state into behavior",
        "representation": "unitary 2-D DFT over the ordered modular input grid; complex coefficients stored as ordered real/imaginary pairs with conjugate partners implied by real activations",
        "primary_task_modes": [[1, 1], [2, 2], [3, 3], [4, 4]],
        "mode_justification": "for any function g(x+y mod p), the 2-D Fourier transform is supported on the diagonal frequency pairs (k,k) under the frozen negative-exponent DFT convention because character_k(x+y)=character_k(x) character_k(y); the first four nonzero diagonal harmonics are fixed without observing activations or outcomes",
        "wrong_mode_control_support": [[1, 2], [2, 1], [1, -1], [2, -2]],
        "wrong_mode_separation": "off-diagonal support is disjoint from the primary diagonal support, has the same number of complex modes and real degrees of freedom, and is never used to fit, rank, or amend the primary mode subspace",
        "alignment": "fit source and recipient linear task-code subspaces and an orthogonal Procrustes map in float64 on the fixed training-domain split only; deterministic SVD/sign convention; no test trials or outcome logits enter fitting",
        "interchanged_variable": "the projected internal code for the modular sum s=(x+y) mod p on the four primary diagonal harmonics",
        "intervention_formula": "h_rec_intervened = h_rec_base - P_rec(h_rec_base) + A(P_src(h_src_donor)), at the frozen MLP-post location; P_rec and P_src are fixed fitted task-code projections and A is the fixed capacity-bounded source-to-recipient alignment",
        "trial_construction": "recipient base input is (x,y); donor input is (x_d,y_d) with donor sum s_d different from base sum s; the canonical recipient counterfactual is (x, (s_d-x) mod p), preserving the recipient first operand while changing only the abstract sum to s_d",
        "expected_behavior": "the intervened recipient should move from its intact base behavior toward its own intact behavior on the canonical counterfactual input, changing the predicted result from s to s_d while preserving the recipient network's centered-logit pattern for that counterfactual",
        "primary_outcome": {
            "name": "counterfactual_centered_logit_displacement_fidelity",
            "formula": "1 - ||C(logits_intervened)-C(logits_recipient_counterfactual)||^2 / ||C(logits_recipient_base)-C(logits_recipient_counterfactual)||^2",
            "centering": "C subtracts the per-input class mean",
            "interpretation": "1 is exact recipient-counterfactual behavior; 0 is no displacement from intact base; negative values are retained",
            "zero_denominator": "trial unavailable and reported, never epsilon-regularized",
        },
        "why_causal_abstraction": "the donor contributes only an internal sum-code variable; the recipient's downstream weights must causally realize the donor sum as the recipient's own counterfactual behavior. Matching donor output logits or patching the final residual is neither the intervention nor the target.",
        "secondary_outcomes": ["counterfactual argmax success", "base-behavior preservation on matched no-change trials", "nonfinite/failure/censoring diagnostics"],
        "capacity": "same ordered support of four complex modes/eight real degrees of freedom with identical conjugate bookkeeping, fitted rank, float32 write precision, recipient write budget, identifier count, and side-information bytes for aligned plus all five controls",
        "trial_inputs": "256 fixed canonically ordered recipient/donor trials per comparison set and condition; no replacement",
        "conditions": [
            "aligned_fourier_interchange",
            "wrong_fourier_mode",
            "shuffled_coefficients",
            "mismatched_input",
            "equal_norm_random_state",
            "unaligned_ordinary_activation_patching",
        ],
        "aggregation": "within comparison set, aligned minus each control on the primary outcome; then teacher-seed paired summaries separately by phase and hard/soft condition",
        "success_rule": "shared-causal-abstraction claim eligible only when aligned exceeds every one of the five controls under the predeclared teacher-seed interval rule, counterfactual argmax is favorable, no-change preservation passes, and the comparison inventory is complete; otherwise not_met, indeterminate, or unavailable",
        "uniqueness_claim": False,
    }
    rd11["rationale"] = (
        "Diagonal Fourier characters are the algebraic support of modular sum, and replacing that code before the MLP output projection asks the recipient's downstream computation to realize a donor abstract value as its own counterfactual behavior."
    )
    rd11["uncertainty"] = (
        "The fixed internal location or first four harmonics may be insufficient; that is a registered negative result, not permission to move the hook, add modes, or select trials."
    )

    rd12 = _decision(dossier, "RD-012")
    rd12["recommendation"]["tier2_protected_minimum"][0] = (
        "RD-002 all four alternates on the shared anchor seeds 0-4 at pre/stable phases and both conditions"
    )
    rd12["recommendation"]["tier2_protected_minimum"][1] = (
        "RD-004 every alternative basis on the same seed-0-4 direct/hard/soft model-checkpoint panel"
    )
    rd12["uncertainty"] = (
        "The complete package counts and scenarios are now projected. Package A/C central cases are near the envelope and every package's conservative case exceeds it; Stage 14 complete-branch qualification is therefore a launch blocker."
    )

    rd14 = _decision(dossier, "RD-014")
    rd14["viable_options"][0]["description"] = (
        "Two infrastructure retries, rolling two-checkpoint recovery, 12-hour final audit reserve, and frozen complete-workload E16/E8/CPU256 branches selected only by measured Stage 14 thresholds."
    )
    rd14["recommendation"]["intended_envelope_branch_not_verified_grant"] = {
        "continuous_hours": 96,
        "science_execution_hours": 84,
        "cpu_cores": 256,
        "cuda_gpus": 16,
        "scratch_bytes": 4398046511104,
        "persistent_bytes": 1099511627776,
    }
    rd14["recommendation"]["stage14_throughput_branch_freeze"] = {
        "qualification_repeats": 3,
        "rate_statistic": "minimum of three after one discarded warmup",
        "efficiency_haircut": 0.70,
        "memory_and_storage_haircut": 0.80,
        "branches": [
            {
                "id": "E16",
                "resources": "16 qualified CUDA workers plus up to 256 CPU workers",
                "complete_scope": "every logical job and sensitivity in the Alex-approved package; no tier, seed, method, control, or exact allowance changes",
                "selection": "first preference when available and the frozen wall-time equation passes",
            },
            {
                "id": "E8",
                "resources": "8 qualified CUDA workers plus up to 256 CPU workers",
                "complete_scope": "identical complete approved package; only concurrency and worker mapping differ",
                "selection": "second preference if E16 is unavailable or fails but E8 passes",
            },
            {
                "id": "CPU256",
                "resources": "up to 256 deterministic CPU workers",
                "complete_scope": "identical complete approved package; no scientific shrinkage",
                "selection": "recovery branch only if its independently measured equation passes",
            },
        ],
        "wall_time_equation": "training_equivalent_updates/(workers*measured_training_updates_per_second*0.70) + hard_concrete_steps/(workers*measured_model_in_loop_steps_per_second*0.70) + measured_CPU_critical_path + fixed_orchestration_allowance <= 84 hours",
        "package_workload_bindings": {
            "A": {"central_training_equivalent_updates": 15713440, "hard_concrete_steps": 15900000, "fixed_orchestration_hours": 4},
            "B": {"central_training_equivalent_updates": 8842000, "hard_concrete_steps": 2675000, "fixed_orchestration_hours": 4},
            "C": {"central_training_equivalent_updates": 15813440, "hard_concrete_steps": 15900000, "fixed_orchestration_hours": 4},
        },
        "reference_rate_floors_for_E16_central_projection": {
            "training_canonical_equivalent_updates_per_second_per_gpu": 28.5714285714,
            "hard_concrete_model_in_loop_steps_per_second_per_gpu": 7.1428571429,
            "exact_full_domain_evaluations_per_second_per_cpu_worker": 11.306431,
        },
        "E8_rate_floors_when_CPU_path_is_unchanged": {
            "training_canonical_equivalent_updates_per_second_per_gpu": 57.1428571429,
            "hard_concrete_model_in_loop_steps_per_second_per_gpu": 14.2857142857,
        },
        "additional_pass_conditions": [
            "all exact and semantic backend qualification checks pass",
            "peak worker memory and aggregate host memory are <=80% of verified capacity",
            "projected persistent and scratch bytes are <=80% of verified quotas",
            "the final 12 hours remain unallocated to scientific jobs",
            "provider queue latency and measured merge/export critical path are included",
        ],
        "selection_rule": "select the highest-preference complete branch that passes all conditions; if none passes, launch is blocked and a new Alex-approved Stage 13 amendment is required",
        "prohibited_fallback": "never switch package, reduce seeds, omit a method/control/sensitivity, lower exact 2^18 calibration, or consume audit reserve to make a branch pass",
    }
    rd14["recommendation"]["stage14_only_unknowns"].extend([
        "model-in-the-loop hard-concrete throughput",
        "CUDA training throughput and VRAM",
        "measured complete-branch wall-time equation",
    ])
    rd14["uncertainty"] = (
        "The grant and production hardware remain unverified. Package A projects to 72.991 science hours centrally but 383.213 conservatively; B to 21.072 and 152.751; C to 73.078 and 384.195. No package is feasible by assertion alone."
    )

    dossier["package_alternatives"] = [
        {
            "package_id": "stage13-package-a-conservative-protected/v2",
            "summary": "Fixed Task 1 seeds 0-14 without replacement; 3-of-6 Tier 1 students; all four architectures on shared anchors 0-4; all four alternative bases on the same checkpoint panel; 256 exact evaluations per method ledger; four x 5000 hard-concrete restarts; full nulls/Fourier/exact 2^18 calibration. Central: 690 student attempts, 771.103 GPU-h, 25.628 CPU-core-h, 72.991 science wall h, 24.711 GiB persistent.",
        },
        {
            "package_id": "stage13-package-b-reduced-protected/v2",
            "summary": "Same fixed Task 1 seeds and primary scientific definitions; 2-of-4 Tier 1 students; depth and eight-head variants on all shared anchors; three alternative basis families; 128 exact allowance; two x 2500 hard-concrete restarts; reduced null counts; exact 2^18 calibration retained. Central: 390 attempts, 189.992 GPU-h, 19.453 CPU-core-h, 21.072 science wall h, 12.231 GiB persistent.",
        },
        {
            "package_id": "stage13-package-c-expanded-roster/v2",
            "summary": "Train Task 1 seeds 0-19 and prospectively select the first 15 complete by seed order, then run Package A unchanged on those 15; this changes the population roster rule and is not an automatic fallback. Central: 690 attempts, 772.075 GPU-h, 25.628 CPU-core-h, 73.078 science wall h, 24.763 GiB persistent.",
        },
    ]

    OUTPUT.write_text(
        json.dumps(dossier, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
