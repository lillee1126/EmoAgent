from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_OUTPUT_DIRECTORY_NAMES = frozenset(
    {"process", "result", "results", "outputs", "emoagent_runs"}
)


@dataclass(frozen=True)
class SourceImage:
    path: Path
    relative_path: Path

    @property
    def key(self) -> Path:
        return self.relative_path.with_suffix("")

    @property
    def assignment_key(self) -> str:
        return self.relative_path.as_posix()


def _is_within(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def discover_source_images(
    input_dir: Path,
    *,
    excluded_roots: Iterable[Path] = (),
) -> list[SourceImage]:
    """Discover unlabeled PNG inputs while excluding generated output trees."""

    root = input_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    excluded = tuple(excluded_roots)
    sources: list[SourceImage] = []
    seen: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() != ".png":
            continue
        relative = path.relative_to(root)
        # Notebook checkpoints are duplicate working copies, not benchmark inputs.
        if any(part.startswith(".") for part in relative.parts[:-1]):
            continue
        if any(part.casefold() in _OUTPUT_DIRECTORY_NAMES for part in relative.parts[:-1]):
            continue
        if _is_within(path, excluded):
            continue
        key = relative.as_posix().casefold()
        if key in seen:
            raise ValueError(f"Duplicate source path after normalization: {relative.as_posix()}")
        seen.add(key)
        sources.append(SourceImage(path=path.resolve(), relative_path=relative))
    if not sources:
        raise FileNotFoundError(f"No source PNG images found in {root}")
    return sources
