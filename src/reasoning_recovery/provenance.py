"""Small deterministic helpers for provenance and artifact identities."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    """Encode structured data deterministically for hashes and manifests."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    """Return a UTF-8 SHA-256 digest."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical structured data."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one local provenance asset without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_token_ids(token_ids: list[int] | tuple[int, ...]) -> str:
    """Hash token IDs without decoding or re-tokenizing them."""

    return sha256_json([int(token_id) for token_id in token_ids])


def utc_now_iso() -> str:
    """Return an unambiguous UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def safe_component(value: str, *, max_length: int = 72) -> str:
    """Create a readable, filesystem-safe component without relying on it as an ID."""

    compact = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not compact:
        compact = "item"
    return compact[:max_length]


def hashed_filename(logical_id: str, *, suffix: str = ".json") -> str:
    """Keep a readable filename while retaining collision-resistant identity."""

    digest = sha256_text(logical_id)[:16]
    return f"{safe_component(logical_id)}--{digest}{suffix}"


def source_git_commit(root: Path) -> str | None:
    """Read the current source commit without changing repository state."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def source_git_identity(root: Path) -> dict[str, object]:
    """Return a minimal, read-only source identity for a model-backed run."""

    commit = source_git_commit(root)
    if commit is None:
        return {"commit": None, "is_clean": None}
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": commit, "is_clean": None}
    return {"commit": commit, "is_clean": not bool(completed.stdout.strip())}


def package_version(name: str) -> str | None:
    """Return an installed package version, or None when the package is absent."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def base_runtime_provenance() -> dict[str, object]:
    """Capture non-model runtime facts without importing optional ML packages."""

    packages = (
        "torch",
        "transformers",
        "datasets",
        "huggingface_hub",
        "math-verify",
        "PyYAML",
    )
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "package_versions": {name: package_version(name) for name in packages},
    }


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")
