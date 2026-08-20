"""Portable synthetic-only Stage 5A centred-logit validator."""

from __future__ import annotations

import torch

from circuit_families.interpretability.centred_logit_fidelity import (
    FIDELITY_FORMULA_REF,
    TECHNICAL_PROFILE_SET_VERSION,
    CentredLogitPredictiveAccumulator,
    centre_logits_across_classes,
)


def main() -> int:
    full = torch.tensor(
        [[1.0, 2.0, 3.0],
         [2.0, 4.0, 6.0]],
        dtype=torch.float64,
    )

    masked = full + torch.tensor(
        [[1.0, -1.0, 0.0],
         [0.5, -0.5, 0.0]],
        dtype=torch.float64,
    )

    full_centred = centre_logits_across_classes(full)
    masked_centred = centre_logits_across_classes(masked)

    acc = CentredLogitPredictiveAccumulator(
        expected_example_count=2,
        class_count=3,
    )

    acc.update(
        full_centred,
        masked_centred,
        start_index=0,
    )

    fidelity = acc.finalize()

    gauge_delta = float(
        torch.abs(
            centre_logits_across_classes(full + 7.0)
            - full_centred
        ).max().item()
    )

    batch_acc = CentredLogitPredictiveAccumulator(
        expected_example_count=2,
        class_count=3,
    )
    batch_acc.update(
        full_centred[:1],
        masked_centred[:1],
        start_index=0,
    )
    batch_acc.update(
        full_centred[1:],
        masked_centred[1:],
        start_index=1,
    )

    batch_delta = abs(fidelity - batch_acc.finalize())

    passed = (
        gauge_delta == 0.0
        and batch_delta < 1e-12
    )

    print("formula_ref=", FIDELITY_FORMULA_REF)
    print("profile_set_version=", TECHNICAL_PROFILE_SET_VERSION)
    print("example_count=", 2)
    print("class_count=", 3)
    print("fidelity=", fidelity)
    print("gauge_delta=", gauge_delta)
    print("batch_delta=", batch_delta)
    print("PASS" if passed else "FAIL")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
