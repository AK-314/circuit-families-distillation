#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from circuit_families.stage11_design_resolution import (  # noqa: E402
    load_stage11_complete_candidates,
    load_stage11_resolution_record,
)

MANIFEST = ROOT / "followup/manifests/stage11_red_team_resolution_v1.json"
CANDIDATES = ROOT / "followup/configs/stage11_post_red_team_design_candidates_v1.json"
DOCUMENT = ROOT / "docs/distillation_followup/stage11_post_red_team_design_resolution.md"


def main() -> int:
    try:
        resolution = load_stage11_resolution_record(MANIFEST, repo_root=ROOT)
        candidates = load_stage11_complete_candidates(CANDIDATES)

        if not DOCUMENT.is_file():
            raise ValueError("missing Stage 11 design-resolution document")

        text = DOCUMENT.read_text(encoding="utf-8")
        required_phrases = (
            "Teacher seed remains the population-level unit",
            "Task 2:",
            "Task 3:",
            "Endpoint 1 remains primary",
            "Endpoint 2 remains a key secondary",
            "two algorithmically distinct discovery families",
            "four distinct calibration layers",
            "Fourier interchange is a registered key secondary",
            "A literal full factorial is rejected",
            "Stage 15 remains unstarted",
        )
        missing = [phrase for phrase in required_phrases if phrase not in text]
        if missing:
            raise ValueError(f"document coverage missing: {missing}")

        print(f"red_team_rows={len(resolution['red_team_items'])}")
        print(f"rd_items={len(resolution['rd_items'])}")
        print(f"tasks={len(candidates['tasks'])}")
        print(f"basis_families={len(candidates['component_bases'])}")
        print(f"discovery_families={len(candidates['discovery']['required_families'])}")
        print(f"packing_nulls={len(candidates['packing_calibration']['required_nulls'])}")
        print(f"fourier_controls={len(candidates['fourier_interchange']['required_controls'])}")
        print("scientific_execution=NO")
        print("artifact_generation=NO")
        print("STAGE11_VALIDATE=PASS")
        return 0
    except Exception as exc:
        print(f"STAGE11_VALIDATE=FAIL reason={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
