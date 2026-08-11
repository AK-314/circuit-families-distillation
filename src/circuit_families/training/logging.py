"""Stable JSON Lines logging for training metrics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def append_jsonl(
    path: str | Path,
    record: Mapping[str, Any],
) -> Path:
    """Append one finite, stable JSON record to a JSONL file."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialised = json.dumps(
        dict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )

    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(serialised)
        handle.write("\n")

    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file as a list of JSON objects."""

    input_path = Path(path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"JSONL file does not exist: {input_path}"
        )

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise ValueError(
                f"JSONL line {line_number} must not be empty."
            )

        value = json.loads(line)

        if not isinstance(value, dict):
            raise ValueError(
                f"JSONL line {line_number} must contain an object."
            )

        records.append(value)

    return records
