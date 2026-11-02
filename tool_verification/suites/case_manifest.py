"""Integrity checks shared by frozen image evaluation suites."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_cases(
    cases: Iterable[Mapping[str, object]],
    data_root: Path,
    *,
    hash_key: str = "sha256",
) -> None:
    """Fail before inference if a frozen image is absent or has changed."""
    root = data_root.resolve()
    for case in cases:
        relative = case.get("image_path")
        expected = case.get(hash_key)
        if not isinstance(relative, str) or not relative:
            raise ValueError("frozen case has no image_path")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"frozen case has no valid {hash_key}: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"frozen image path escapes data root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"frozen image is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected.lower():
            raise ValueError(
                f"frozen image SHA256 mismatch for {relative}: "
                f"expected {expected.lower()}, got {actual}"
            )
