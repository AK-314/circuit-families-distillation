"""Stage 5A centred-logit predictive-fidelity technical contracts.

The numerical profiles in this module exist only to make synthetic and
technical Stage 5A validation executable. They are not production protocol
choices and do not resolve UD-007.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformer_lens import HookedTransformer

FIDELITY_FORMULA_REF = "centred-logit-predictive-fidelity/v1"
TECHNICAL_PROFILE_SET_VERSION = "stage5a-technical-fidelity-profiles/v1"
VERSION_REFERENCE_RE = re.compile(r"^[a-z][a-z0-9._-]*/v[1-9][0-9]*$")

_FORBIDDEN_PROFILE_KEYS = frozenset(
    {
        "fidelity_threshold",
        "primary_fidelity_threshold",
        "threshold",
        "sensitivity_grid",
        "fidelity_sensitivity_grid",
        "production_threshold",
        "production_precision",
        "final_precision",
        "final_denominator_guard",
        "final_reduction_order",
    }
)

_ALLOWED_ACCUMULATION_DTYPES = frozenset({"float64"})
_ALLOWED_CENTERING_DTYPES = frozenset({"preserve_input_float_dtype"})
_ALLOWED_ACCUMULATION_ORDERS = frozenset(
    {"canonical_example_order_then_class_sum"}
)
_ALLOWED_CANONICAL_ORDER_POLICIES = frozenset(
    {"explicit_index_strict_contiguous_ascending"}
)
_ALLOWED_BATCH_SEMANTICS = frozenset(
    {"batch_boundaries_do_not_change_logical_order"}
)
_ALLOWED_NONFINITE_POLICIES = frozenset({"reject"})
_ALLOWED_DENOMINATOR_POLICIES = frozenset(
    {"classify_exact_zero_else_positive"}
)


@dataclass(frozen=True)
class TechnicalNumericalProfile:
    """Validated, injectable Stage 5A technical numerical profile."""

    profile_ref: str
    formula_ref: str
    profile_status: str
    scientific_data: bool
    production_eligible: bool
    resolves_ud007: bool
    accumulation_dtype: str
    centering_dtype: str
    accumulation_order: str
    canonical_order_policy: str
    batch_semantics: str
    nonfinite_policy: str
    denominator_guard_candidate: str
    near_zero_guard_defined: bool
    notes: str

    def __post_init__(self) -> None:
        if not VERSION_REFERENCE_RE.fullmatch(self.profile_ref):
            raise ValueError(
                "profile_ref must match the Stage 4 version-reference grammar."
            )
        if not VERSION_REFERENCE_RE.fullmatch(self.formula_ref):
            raise ValueError(
                "formula_ref must match the Stage 4 version-reference grammar."
            )
        if self.formula_ref != FIDELITY_FORMULA_REF:
            raise ValueError(
                f"formula_ref must equal {FIDELITY_FORMULA_REF!r}."
            )
        if self.profile_status != "technical_candidate":
            raise ValueError(
                "Stage 5A numerical profiles must have "
                "profile_status='technical_candidate'."
            )
        if self.scientific_data is not False:
            raise ValueError(
                "Stage 5A technical profiles must declare scientific_data=false."
            )
        if self.production_eligible is not False:
            raise ValueError(
                "Stage 5A technical profiles must declare "
                "production_eligible=false."
            )
        if self.resolves_ud007 is not False:
            raise ValueError(
                "Stage 5A technical profiles must declare resolves_ud007=false."
            )
        if self.accumulation_dtype not in _ALLOWED_ACCUMULATION_DTYPES:
            raise ValueError(
                "Unsupported Stage 5A technical accumulation_dtype."
            )
        if self.centering_dtype not in _ALLOWED_CENTERING_DTYPES:
            raise ValueError("Unsupported Stage 5A technical centering_dtype.")
        if self.accumulation_order not in _ALLOWED_ACCUMULATION_ORDERS:
            raise ValueError(
                "Unsupported Stage 5A technical accumulation_order."
            )
        if (
            self.canonical_order_policy
            not in _ALLOWED_CANONICAL_ORDER_POLICIES
        ):
            raise ValueError(
                "Unsupported Stage 5A technical canonical_order_policy."
            )
        if self.batch_semantics not in _ALLOWED_BATCH_SEMANTICS:
            raise ValueError(
                "Unsupported Stage 5A technical batch_semantics."
            )
        if self.nonfinite_policy not in _ALLOWED_NONFINITE_POLICIES:
            raise ValueError(
                "Unsupported Stage 5A technical nonfinite_policy."
            )
        if (
            self.denominator_guard_candidate
            not in _ALLOWED_DENOMINATOR_POLICIES
        ):
            raise ValueError(
                "Unsupported Stage 5A technical denominator-guard candidate."
            )
        if self.near_zero_guard_defined is not False:
            raise ValueError(
                "Stage 5A technical profiles must leave the near-zero "
                "production guard unresolved."
            )
        if not isinstance(self.notes, str) or not self.notes.strip():
            raise ValueError("Technical profile notes must be non-empty.")

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe profile record."""
        record = {
            "profile_ref": self.profile_ref,
            "formula_ref": self.formula_ref,
            "profile_status": self.profile_status,
            "scientific_data": self.scientific_data,
            "production_eligible": self.production_eligible,
            "resolves_ud007": self.resolves_ud007,
            "accumulation_dtype": self.accumulation_dtype,
            "centering_dtype": self.centering_dtype,
            "accumulation_order": self.accumulation_order,
            "canonical_order_policy": self.canonical_order_policy,
            "batch_semantics": self.batch_semantics,
            "nonfinite_policy": self.nonfinite_policy,
            "denominator_guard_candidate": (
                self.denominator_guard_candidate
            ),
            "near_zero_guard_defined": self.near_zero_guard_defined,
            "notes": self.notes,
        }
        json.dumps(record, allow_nan=False, sort_keys=True)
        return record



def centre_logits_across_classes(logits: torch.Tensor) -> torch.Tensor:
    """Centre logits independently for each input across the class dimension.

    Input shape must be [example, class]. The operation removes additive
    per-example class-gauge shifts while preserving example ordering.
    """


    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a PyTorch tensor.")

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape [example, class]."
        )

    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError(
            "logits must contain at least one example and class."
        )

    if not torch.is_floating_point(logits):
        raise TypeError(
            "logits must have a floating dtype."
        )

    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError(
            "logits must contain only finite values."
        )

    class_mean = logits.mean(dim=1, keepdim=True)
    return logits - class_mean



class CentredLogitPredictiveAccumulator:
    """Deterministic streaming accumulator for centred-logit fidelity."""

    def __init__(
        self,
        *,
        expected_example_count: int,
        class_count: int,
    ) -> None:
        if expected_example_count <= 0:
            raise ValueError("expected_example_count must be positive.")
        if class_count <= 0:
            raise ValueError("class_count must be positive.")

        self.expected_example_count = expected_example_count
        self.class_count = class_count
        self._seen: set[int] = set()
        self._next_expected_index = 0
        self._numerator = 0.0
        self._denominator = 0.0

    @property
    def numerator(self) -> float:
        return self._numerator

    @property
    def denominator(self) -> float:
        return self._denominator

    @property
    def seen_count(self) -> int:
        return len(self._seen)

    def update(
        self,
        full_centred_logits: torch.Tensor,
        masked_centred_logits: torch.Tensor,
        *,
        start_index: int,
    ) -> None:
        if not isinstance(start_index, int):
            raise TypeError("start_index must be an integer.")

        if start_index != self._next_expected_index:
            raise ValueError(
                "Batches must arrive in canonical example order."
            )

        if full_centred_logits.shape != masked_centred_logits.shape:
            raise ValueError(
                "Full and masked centred logits must have identical shapes."
            )

        if full_centred_logits.ndim != 2:
            raise ValueError(
                "Centred logits must have shape [example, class]."
            )

        if full_centred_logits.shape[1] != self.class_count:
            raise ValueError(
                "Centred logits class count does not match accumulator."
            )

        batch_size = full_centred_logits.shape[0]
        indices = range(start_index, start_index + batch_size)

        if any(index in self._seen for index in indices):
            raise ValueError("Duplicate example index encountered.")

        if start_index + batch_size > self.expected_example_count:
            raise ValueError("Received too many examples.")

        diff = masked_centred_logits - full_centred_logits

        self._numerator += float(
            torch.sum(diff * diff).item()
        )
        self._denominator += float(
            torch.sum(full_centred_logits * full_centred_logits).item()
        )

        self._seen.update(indices)
        self._next_expected_index += batch_size

    def finalize(self) -> float:
        if self.seen_count != self.expected_example_count:
            raise ValueError(
                "Cannot finalize before all examples are accumulated."
            )

        if self._denominator == 0.0:
            raise ZeroDivisionError(
                "Predictive-fidelity denominator is exactly zero."
            )

        return 1.0 - self._numerator / self._denominator


@dataclass(frozen=True)
class CentredLogitPredictiveMetricRecord:
    """Technical candidate record for one centred-logit metric evaluation."""

    formula_ref: str
    profile_ref: str
    record_status: str
    evaluated_example_count: int
    class_count: int
    numerator: float
    denominator: float
    predictive_fidelity: float
    denominator_status: str
    canonical_order_policy: str
    accumulation_order: str
    nonfinite_rejected: bool
    notes: str

    def __post_init__(self) -> None:
        if self.formula_ref != FIDELITY_FORMULA_REF:
            raise ValueError(
                "Metric record formula_ref must match Stage 5A formula."
            )
        if self.record_status != "technical_candidate":
            raise ValueError(
                "Metric record must remain technical_candidate."
            )
        if self.evaluated_example_count <= 0:
            raise ValueError(
                "evaluated_example_count must be positive."
            )
        if self.class_count <= 0:
            raise ValueError(
                "class_count must be positive."
            )
        for value_name, value in (
            ("numerator", self.numerator),
            ("denominator", self.denominator),
            ("predictive_fidelity", self.predictive_fidelity),
        ):
            require_finite_scalar(value, label=value_name)
        if self.nonfinite_rejected is not True:
            raise ValueError(
                "Metric records must declare nonfinite_rejected=true."
            )
        if not self.notes.strip():
            raise ValueError("Metric record notes must be non-empty.")

    def to_record(self) -> dict[str, Any]:
        record = {
            "formula_ref": self.formula_ref,
            "profile_ref": self.profile_ref,
            "record_status": self.record_status,
            "evaluated_example_count": self.evaluated_example_count,
            "class_count": self.class_count,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "predictive_fidelity": self.predictive_fidelity,
            "denominator_status": self.denominator_status,
            "canonical_order_policy": self.canonical_order_policy,
            "accumulation_order": self.accumulation_order,
            "nonfinite_rejected": self.nonfinite_rejected,
            "notes": self.notes,
        }
        json.dumps(record, allow_nan=False, sort_keys=True)
        return record


def _require_exact_keys(
    record: Mapping[str, Any],
    *,
    required: frozenset[str],
    label: str,
) -> None:
    actual = frozenset(record)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing:
        raise ValueError(
            f"{label} is missing required fields: {', '.join(missing)}"
        )
    if extra:
        raise ValueError(
            f"{label} contains unsupported fields: {', '.join(extra)}"
        )



def centred_logit_predictive_metric_record_from_record(
    record: Mapping[str, Any],
) -> CentredLogitPredictiveMetricRecord:
    """Validate and reconstruct one Stage 5A metric record.

    This reconstructs technical metric records only. It does not resolve
    UD-007 or define a production scientific endpoint.
    """

    _require_exact_keys(
        record,
        required={
            "formula_ref",
            "profile_ref",
            "record_status",
            "evaluated_example_count",
            "class_count",
            "numerator",
            "denominator",
            "predictive_fidelity",
            "denominator_status",
            "canonical_order_policy",
            "accumulation_order",
            "nonfinite_rejected",
            "notes",
        },
        label="metric record",
    )

    return CentredLogitPredictiveMetricRecord(
        formula_ref=record["formula_ref"],
        profile_ref=record["profile_ref"],
        record_status=record["record_status"],
        evaluated_example_count=record["evaluated_example_count"],
        class_count=record["class_count"],
        numerator=record["numerator"],
        denominator=record["denominator"],
        predictive_fidelity=record["predictive_fidelity"],
        denominator_status=record["denominator_status"],
        canonical_order_policy=record["canonical_order_policy"],
        accumulation_order=record["accumulation_order"],
        nonfinite_rejected=record["nonfinite_rejected"],
        notes=record["notes"],
    )


def technical_profile_from_record(
    record: Mapping[str, Any],
) -> TechnicalNumericalProfile:
    """Validate and construct one Stage 5A technical profile."""
    if not isinstance(record, Mapping):
        raise TypeError("technical profile must be a mapping.")

    forbidden = sorted(_FORBIDDEN_PROFILE_KEYS.intersection(record))
    if forbidden:
        raise ValueError(
            "Stage 5A technical profile contains premature scientific "
            f"choice field(s): {', '.join(forbidden)}"
        )

    required = frozenset(
        {
            "profile_ref",
            "formula_ref",
            "profile_status",
            "scientific_data",
            "production_eligible",
            "resolves_ud007",
            "accumulation_dtype",
            "centering_dtype",
            "accumulation_order",
            "canonical_order_policy",
            "batch_semantics",
            "nonfinite_policy",
            "denominator_guard_candidate",
            "near_zero_guard_defined",
            "notes",
        }
    )
    _require_exact_keys(record, required=required, label="technical profile")

    string_fields = (
        "profile_ref",
        "formula_ref",
        "profile_status",
        "accumulation_dtype",
        "centering_dtype",
        "accumulation_order",
        "canonical_order_policy",
        "batch_semantics",
        "nonfinite_policy",
        "denominator_guard_candidate",
        "notes",
    )
    for field in string_fields:
        if not isinstance(record[field], str):
            raise TypeError(f"{field} must be a string.")

    boolean_fields = (
        "scientific_data",
        "production_eligible",
        "resolves_ud007",
        "near_zero_guard_defined",
    )
    for field in boolean_fields:
        if not isinstance(record[field], bool):
            raise TypeError(f"{field} must be boolean.")

    return TechnicalNumericalProfile(
        profile_ref=record["profile_ref"],
        formula_ref=record["formula_ref"],
        profile_status=record["profile_status"],
        scientific_data=record["scientific_data"],
        production_eligible=record["production_eligible"],
        resolves_ud007=record["resolves_ud007"],
        accumulation_dtype=record["accumulation_dtype"],
        centering_dtype=record["centering_dtype"],
        accumulation_order=record["accumulation_order"],
        canonical_order_policy=record["canonical_order_policy"],
        batch_semantics=record["batch_semantics"],
        nonfinite_policy=record["nonfinite_policy"],
        denominator_guard_candidate=record[
            "denominator_guard_candidate"
        ],
        near_zero_guard_defined=record["near_zero_guard_defined"],
        notes=record["notes"],
    )


def validate_technical_profile_set(
    record: Mapping[str, Any],
) -> tuple[TechnicalNumericalProfile, ...]:
    """Validate the complete Stage 5A technical profile set."""
    if not isinstance(record, Mapping):
        raise TypeError("technical profile set must be a mapping.")

    required = frozenset(
        {
            "profile_set_version",
            "stage",
            "purpose",
            "scientific_data",
            "production_eligible",
            "resolves_ud007",
            "profiles",
        }
    )
    _require_exact_keys(
        record,
        required=required,
        label="technical profile set",
    )

    if record["profile_set_version"] != TECHNICAL_PROFILE_SET_VERSION:
        raise ValueError("Unsupported technical profile-set version.")
    if record["stage"] != "5A":
        raise ValueError("Technical profile set must belong to Stage 5A.")
    if record["purpose"] != "synthetic_and_technical_validation_only":
        raise ValueError(
            "Technical profile set purpose must remain "
            "synthetic_and_technical_validation_only."
        )
    if record["scientific_data"] is not False:
        raise ValueError(
            "Technical profile set must declare scientific_data=false."
        )
    if record["production_eligible"] is not False:
        raise ValueError(
            "Technical profile set must declare production_eligible=false."
        )
    if record["resolves_ud007"] is not False:
        raise ValueError(
            "Technical profile set must declare resolves_ud007=false."
        )

    profiles = record["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(
            "Technical profile set must contain at least one profile."
        )

    validated = tuple(
        technical_profile_from_record(profile) for profile in profiles
    )
    refs = [profile.profile_ref for profile in validated]
    if len(refs) != len(set(refs)):
        raise ValueError("Technical profile_ref values must be unique.")
    return validated


def load_technical_profile_set(
    path: str | Path,
) -> tuple[TechnicalNumericalProfile, ...]:
    """Load an injectable Stage 5A technical profile set from JSON."""
    input_path = Path(path)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    return validate_technical_profile_set(raw)



@dataclass(frozen=True)
class CentredLogitFullModelReference:
    """Frozen centred-logit full-model outputs for reusable mask evaluation."""

    centred_final_logits: torch.Tensor
    evaluated_example_count: int
    inference_batch_size: int

    def __post_init__(self) -> None:
        if self.centred_final_logits.ndim != 2:
            raise ValueError(
                "centred_final_logits must have shape [example, class]."
            )
        if self.centred_final_logits.requires_grad:
            raise ValueError(
                "centred_final_logits must be detached."
            )
        if not bool(torch.isfinite(self.centred_final_logits).all().item()):
            raise FloatingPointError(
                "centred_final_logits must contain only finite values."
            )


def compute_centred_logit_full_model_reference(
    model: HookedTransformer,
    inputs: torch.Tensor,
    *,
    batch_size: int,
) -> CentredLogitFullModelReference:
    """Compute reusable detached centred final-position full-model logits."""

    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a PyTorch tensor.")

    if inputs.ndim != 2:
        raise ValueError(
            "inputs must have shape [example, sequence_position]."
        )

    if inputs.shape[0] == 0:
        raise ValueError("inputs must contain examples.")

    if inputs.dtype != torch.long:
        raise TypeError("inputs must have dtype torch.long.")

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    was_training = model.training
    parameter_state = tuple(
        parameter.detach().clone()
        for parameter in model.parameters()
    )

    batches: list[torch.Tensor] = []

    try:
        model.eval()

        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])

            with torch.inference_mode():
                logits = model(inputs[start:stop])

            final_logits = logits[:, -1, :]
            centred = centre_logits_across_classes(final_logits)

            batches.append(
                centred.detach().clone()
            )
    finally:
        model.train(was_training)

    stored = torch.cat(batches, dim=0)

    for before, after in zip(parameter_state, model.parameters(), strict=True):
        if not torch.equal(before, after.detach()):
            raise RuntimeError(
                "Full-model reference computation altered parameters."
            )

    return CentredLogitFullModelReference(
        centred_final_logits=stored,
        evaluated_example_count=inputs.shape[0],
        inference_batch_size=batch_size,
    )


def evaluate_centred_logit_component_mask(
    full_reference: CentredLogitFullModelReference,
    masked_logits: torch.Tensor,
    *,
    start_index: int,
) -> float:
    """Evaluate one masked-model output against a centred-logit reference.

    This consumes existing mask-evaluation outputs and does not change
    component basis, mask semantics, discovery, search, ledger, or endpoint
    behaviour.
    """

    if not isinstance(full_reference, CentredLogitFullModelReference):
        raise TypeError(
            "full_reference must be a CentredLogitFullModelReference."
        )

    if not isinstance(masked_logits, torch.Tensor):
        raise TypeError("masked_logits must be a PyTorch tensor.")

    if masked_logits.ndim != 2:
        raise ValueError(
            "masked_logits must have shape [example, class]."
        )

    if start_index < 0:
        raise ValueError("start_index must be non-negative.")

    stop_index = start_index + masked_logits.shape[0]

    if stop_index > full_reference.evaluated_example_count:
        raise ValueError(
            "Masked logits exceed full-model reference coverage."
        )

    masked_centred = centre_logits_across_classes(masked_logits)

    accumulator = CentredLogitPredictiveAccumulator(
        expected_example_count=masked_logits.shape[0],
        class_count=masked_logits.shape[1],
    )

    accumulator.update(
        full_reference.centred_final_logits[start_index:stop_index],
        masked_centred,
        start_index=0,
    )

    return accumulator.finalize()

def require_finite_scalar(value: float, *, label: str) -> float:
    """Reject non-finite scalar state under the technical profile contract."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{label} must be finite.")
    return result
