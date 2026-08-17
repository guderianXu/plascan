"""MVS manifest loading, latest-record indexing, and artifact resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FrameKey = tuple[int, str]
FrameIndex = dict[FrameKey, tuple[str, dict[str, Any]]]


def load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"MVS manifest root must be an object: {resolved}")
    return resolved, document


def image_basename(frame: dict[str, Any], manifest_path: Path) -> str:
    configured = str(frame.get("ref_image", "")).strip().replace("\\", "/")
    name = configured.rsplit("/", maxsplit=1)[-1]
    if not name:
        raise ValueError(
            f"Frame {frame.get('ref_index')} has no ref_image basename: {manifest_path}"
        )
    return name


def frame_index(
    manifest: dict[str, Any], manifest_path: Path
) -> tuple[FrameIndex, int]:
    """Index the last record for each frame key.

    Workspace manifests normally upsert records, but replay-style documents
    can retain earlier entries. Last-record-wins matches PlaScan's latest
    artifact semantics and prevents an interrupted earlier record from hiding
    a later completed artifact.
    """
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Manifest must contain a frames array: {manifest_path}")
    indexed: FrameIndex = {}
    duplicate_count = 0
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError(f"Manifest frame must be an object: {manifest_path}")
        ref_index = frame.get("ref_index")
        if isinstance(ref_index, bool) or not isinstance(ref_index, int):
            raise ValueError(f"Manifest frame has invalid ref_index: {manifest_path}")
        basename = image_basename(frame, manifest_path)
        key = (ref_index, basename.casefold())
        duplicate_count += int(key in indexed)
        indexed[key] = (basename, frame)
    return indexed, duplicate_count


def artifact_path(
    frame: dict[str, Any], key: str, manifest_path: Path, required: bool
) -> Path | None:
    """Resolve a configured artifact relative to its own manifest directory."""
    configured = str(frame.get(key, "")).strip()
    if not configured:
        if required:
            raise ValueError(
                f"Frame {frame.get('ref_index')} has no required {key}: {manifest_path}"
            )
        return None
    path = Path(configured.replace("\\", "/"))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Frame {frame.get('ref_index')} {key} does not exist: {path}"
        )
    return path


def acceptance(frame: dict[str, Any]) -> str:
    value = str(frame.get("acceptance", "")).strip()
    if not value:
        decision = frame.get("quality_decision", {})
        if isinstance(decision, dict):
            value = str(decision.get("acceptance", "")).strip()
    return value or "unknown"


def fusion_eligible(
    frame: dict[str, Any], frame_acceptance: str
) -> tuple[bool | None, str]:
    configured = frame.get("fusion_eligible")
    if isinstance(configured, bool):
        return configured, "manifest"
    if frame_acceptance != "unknown":
        return frame_acceptance == "accepted", "derived_from_acceptance"
    return None, "unknown"


def key_record(key: FrameKey, basename: str) -> dict[str, Any]:
    return {"ref_index": key[0], "image_basename": basename}
