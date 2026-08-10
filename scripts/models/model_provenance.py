#!/usr/bin/env python3
"""Deterministic provenance sidecars for generated model artifacts.

This module intentionally uses only the Python standard library.  Exporter
tests and package composition must be able to validate cached artifacts on a
machine that does not have PyTorch, ONNX, or TensorRT installed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


PROVENANCE_SCHEMA_VERSION = 1
SIDECAR_SUFFIX = ".provenance.json"


class ProvenanceError(RuntimeError):
    """Raised when an artifact and its provenance contract are incompatible."""


@dataclass(frozen=True)
class ProvenanceValidation:
    valid: bool
    reason: str
    document: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.name + SIDECAR_SUFFIX)


def artifact_record(artifact_path: Path) -> dict[str, Any]:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {artifact_path}")
    return {
        "file": artifact_path.name,
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }


def provenance_document(
    artifact_path: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact": artifact_record(artifact_path),
        "contract": dict(contract),
    }


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_provenance(
    artifact_path: Path,
    contract: Mapping[str, Any],
    sidecar_path: Path | None = None,
) -> Path:
    sidecar = sidecar_path or provenance_path(artifact_path)
    _write_json_atomic(sidecar, provenance_document(artifact_path, contract))
    return sidecar


def read_provenance(sidecar_path: Path) -> dict[str, Any]:
    try:
        document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(
            f"Cannot read provenance sidecar {sidecar_path}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ProvenanceError(f"Provenance sidecar is not a JSON object: {sidecar_path}")
    return document


def validate_provenance(
    artifact_path: Path,
    expected_contract: Mapping[str, Any] | None = None,
    sidecar_path: Path | None = None,
) -> ProvenanceValidation:
    sidecar = sidecar_path or provenance_path(artifact_path)
    if not artifact_path.is_file():
        return ProvenanceValidation(False, f"artifact is missing: {artifact_path}")
    if not sidecar.is_file():
        return ProvenanceValidation(False, f"provenance sidecar is missing: {sidecar}")
    try:
        document = read_provenance(sidecar)
        actual_record = artifact_record(artifact_path)
    except (OSError, ProvenanceError) as error:
        return ProvenanceValidation(False, str(error))

    if document.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        return ProvenanceValidation(
            False,
            "provenance schema mismatch: "
            f"expected {PROVENANCE_SCHEMA_VERSION}, "
            f"got {document.get('schema_version')!r}",
            document,
        )
    if document.get("artifact") != actual_record:
        return ProvenanceValidation(
            False,
            "artifact name, size, or SHA-256 does not match its provenance sidecar",
            document,
        )
    if expected_contract is not None and document.get("contract") != dict(expected_contract):
        return ProvenanceValidation(
            False,
            "export contract does not exactly match the provenance sidecar",
            document,
        )
    if set(document) != {"schema_version", "artifact", "contract"}:
        return ProvenanceValidation(
            False,
            "provenance sidecar contains unsupported top-level fields",
            document,
        )
    if not isinstance(document.get("contract"), dict):
        return ProvenanceValidation(False, "provenance contract is not an object", document)
    return ProvenanceValidation(True, "", document)


def require_provenance(
    artifact_path: Path,
    expected_contract: Mapping[str, Any] | None = None,
    sidecar_path: Path | None = None,
) -> dict[str, Any]:
    validation = validate_provenance(
        artifact_path, expected_contract=expected_contract, sidecar_path=sidecar_path
    )
    if not validation.valid or validation.document is None:
        raise ProvenanceError(f"Cannot reuse {artifact_path}: {validation.reason}")
    return validation.document


def invalidate_provenance(artifact_path: Path) -> None:
    provenance_path(artifact_path).unlink(missing_ok=True)


def source_file_records(
    files: Mapping[str, Path] | Sequence[tuple[str, Path]],
) -> list[dict[str, Any]]:
    items = files.items() if isinstance(files, Mapping) else files
    records = []
    for name, path in sorted(items, key=lambda item: item[0]):
        if not path.is_file():
            raise FileNotFoundError(f"Source weight/checkpoint does not exist: {path}")
        records.append(
            {
                "name": name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def installed_tool_versions(
    distributions: Mapping[str, str] | Sequence[str],
) -> dict[str, str]:
    items = (
        distributions.items()
        if isinstance(distributions, Mapping)
        else ((name, name) for name in distributions)
    )
    versions = {"python": sys.version.split()[0]}
    for label, distribution in sorted(items):
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "unavailable"
    return versions


@lru_cache(maxsize=None)
def git_source_revision(repository: Path) -> dict[str, Any]:
    resolved = repository.resolve()
    source_digest = hashlib.sha256()
    source_file_count = 0
    for path in sorted(resolved.rglob("*.py")):
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        source_digest.update(len(relative).to_bytes(8, "big"))
        source_digest.update(relative)
        source_digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                source_digest.update(block)
        source_file_count += 1
    source_tree_sha256 = source_digest.hexdigest()

    def run_git(*arguments: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(resolved), *arguments],
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = run_git("rev-parse", "HEAD").decode("ascii")
        diff = subprocess.check_output(
            ["git", "-C", str(resolved), "diff", "--binary", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        return {
            "commit": "unknown",
            "dirty": None,
            "dirty_diff_sha256": None,
            "source_file_count": source_file_count,
            "source_tree_sha256": source_tree_sha256,
        }
    return {
        "commit": commit,
        "dirty": bool(diff),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
        "source_file_count": source_file_count,
        "source_tree_sha256": source_tree_sha256,
    }


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Public atomic JSON writer used by package composers."""

    _write_json_atomic(path, document)
