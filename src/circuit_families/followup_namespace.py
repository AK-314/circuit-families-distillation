"""Portable path safety for the circuit-families distillation follow-up."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Final

NAMESPACE_VERSION: Final = "circuit-families-distillation/v1"
FOLLOWUP_ROOT: Final = PurePosixPath("followup")

APPROVED_LOGICAL_ROOTS: Final[dict[str, PurePosixPath]] = {
    "configs": FOLLOWUP_ROOT / "configs",
    "local_scratch": FOLLOWUP_ROOT / "local" / "scratch",
    "teacher_cache": FOLLOWUP_ROOT / "artifacts" / "teacher_cache",
    "student_checkpoints": (
        FOLLOWUP_ROOT / "artifacts" / "student_checkpoints"
    ),
    "student_outputs": FOLLOWUP_ROOT / "artifacts" / "student_outputs",
    "discovery_raw": FOLLOWUP_ROOT / "artifacts" / "discovery_raw",
    "manifests": FOLLOWUP_ROOT / "manifests",
    "reviewed_tables": FOLLOWUP_ROOT / "reviewed" / "tables",
    "notes": FOLLOWUP_ROOT / "reviewed" / "notes",
    "figures": FOLLOWUP_ROOT / "reviewed" / "figures",
    "archives": FOLLOWUP_ROOT / "artifacts" / "archives",
    "excluded_development": FOLLOWUP_ROOT / "excluded_development",
    "reproduction_bundles": (
        FOLLOWUP_ROOT / "artifacts" / "reproduction_bundles"
    ),
}

LEGACY_OUTPUT_ROOTS: Final[frozenset[str]] = frozenset(
    {"checkpoints", "results", "manifests", "figures"}
)


class NamespacePathError(ValueError):
    """Raised when a follow-up path violates the preservation boundary."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_repository_roots(
    *,
    successor_root: str | Path,
    predecessor_root: str | Path,
) -> tuple[Path, Path]:
    """Return distinct resolved successor/predecessor repository roots."""

    successor = _resolved(successor_root)
    predecessor = _resolved(predecessor_root)

    if successor == predecessor:
        raise NamespacePathError(
            "Successor repository root resolves to the predecessor repository "
            f"root: {successor}"
        )

    if _is_within(successor, predecessor):
        raise NamespacePathError(
            "Successor repository root must not be inside the predecessor "
            f"repository: {successor}"
        )

    if _is_within(predecessor, successor):
        raise NamespacePathError(
            "Predecessor repository root must not be inside the successor "
            f"repository: {predecessor}"
        )

    return successor, predecessor


def logical_root(name: str) -> PurePosixPath:
    """Return the portable repository-relative root for a logical namespace."""

    try:
        return APPROVED_LOGICAL_ROOTS[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(APPROVED_LOGICAL_ROOTS))
        raise NamespacePathError(
            f"Unknown follow-up logical root {name!r}. "
            f"Approved logical roots: {allowed}."
        ) from exc


def build_followup_relative_path(
    logical_name: str,
    *parts: str,
) -> PurePosixPath:
    """Build a deterministic portable path beneath an approved logical root."""

    root = logical_root(logical_name)
    candidate = root.joinpath(*parts)

    if candidate.is_absolute():
        raise NamespacePathError(
            f"Follow-up path must be repository-relative: {candidate}"
        )

    if ".." in candidate.parts:
        raise NamespacePathError(
            f"Follow-up path traversal is not permitted: {candidate}"
        )

    if not candidate.parts or candidate.parts[0] != FOLLOWUP_ROOT.as_posix():
        raise NamespacePathError(
            f"Follow-up path must remain beneath {FOLLOWUP_ROOT}/: {candidate}"
        )

    return candidate


def validate_followup_output_path(
    path: str | Path,
    *,
    successor_root: str | Path,
    predecessor_root: str | Path,
) -> Path:
    """Validate and return a resolved physical follow-up output path.

    ``path`` may be repository-relative or absolute. A valid path must resolve
    beneath ``<successor_root>/followup`` and must never resolve to or inside
    the predecessor repository.
    """

    successor, predecessor = validate_repository_roots(
        successor_root=successor_root,
        predecessor_root=predecessor_root,
    )

    supplied = Path(path).expanduser()
    candidate = supplied if supplied.is_absolute() else successor / supplied
    resolved = candidate.resolve(strict=False)

    if resolved == predecessor or _is_within(resolved, predecessor):
        raise NamespacePathError(
            "Follow-up output path collides with the immutable predecessor "
            f"repository: supplied={path!s}, resolved={resolved}"
        )

    authorized_root = (successor / FOLLOWUP_ROOT.as_posix()).resolve(
        strict=False
    )

    if resolved == successor:
        raise NamespacePathError(
            "The successor repository root itself is not an output root. "
            f"Use a path beneath {FOLLOWUP_ROOT}/."
        )

    if _is_within(resolved, successor):
        relative = resolved.relative_to(successor)
        if relative.parts and relative.parts[0] in LEGACY_OUTPUT_ROOTS:
            raise NamespacePathError(
                "New follow-up commands may not write to inherited legacy "
                f"output root {relative.parts[0]!r}: {resolved}"
            )

    if resolved != authorized_root and not _is_within(
        resolved,
        authorized_root,
    ):
        raise NamespacePathError(
            "Follow-up output path escapes the authorized namespace: "
            f"supplied={path!s}, resolved={resolved}, "
            f"authorized_root={authorized_root}"
        )

    return resolved


def resolve_logical_output_path(
    logical_name: str,
    *parts: str,
    successor_root: str | Path,
    predecessor_root: str | Path,
) -> Path:
    """Build and validate one approved logical follow-up output path."""

    relative = build_followup_relative_path(logical_name, *parts)
    return validate_followup_output_path(
        relative,
        successor_root=successor_root,
        predecessor_root=predecessor_root,
    )
