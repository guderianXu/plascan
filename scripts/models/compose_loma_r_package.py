#!/usr/bin/env python3
"""Compose the only supported portable LoMa-R package layout.

The composer consumes already-exported ONNX artifacts and their validated
provenance sidecars.  Package metadata is derived from those artifacts rather
than from exporter command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from model_provenance import (
    ProvenanceError,
    require_provenance,
)


PACKAGE_SCHEMA_VERSION = 2
PACKAGE_KEYPOINT_BUCKETS = (1024, 2048, 3840)
FEATURE_KEYPOINT_COUNT = 3840
DESCRIPTOR_DIMENSION = 256


def _write_release_manifest(path: Path, document: Mapping[str, Any]) -> None:
    """Write the byte-stable manifest format used by models-v1.1.0.

    The release whitelist verifies manifest bytes as well as their semantic
    fields.  Keep this serialization deterministic so the composer remains
    the single source of the published K1024/K2048/K3840 manifests.
    """

    text = json.dumps(
        document,
        ensure_ascii=False,
        indent=4,
        separators=(",", ":  "),
    )
    payload = (text.replace("\n", "\r\n") + "\n").encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _contract_value(contract: dict[str, Any], *keys: str) -> Any:
    value: Any = contract
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ProvenanceError(
                "LoMa-R provenance is missing contract field " + ".".join(keys)
            )
        value = value[key]
    return value


def require_matcher_only_feature(
    feature_onnx: Path, expected_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject matcher-only export when the existing feature is not this contract."""

    try:
        return require_provenance(feature_onnx, expected_contract)
    except ProvenanceError as error:
        raise ProvenanceError(
            "--matcher-only requires a compatible existing feature ONNX: "
            f"{error}"
        ) from error


def compose_package(
    feature_onnx: Path,
    matcher_onnx: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    feature_onnx = feature_onnx.resolve()
    matcher_onnx = matcher_onnx.resolve()
    destination = (output_dir or feature_onnx.parent).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    feature_document = require_provenance(feature_onnx)
    matcher_document = require_provenance(matcher_onnx)
    feature_contract = feature_document["contract"]
    matcher_contract = matcher_document["contract"]

    if feature_contract.get("artifact_kind") != "loma_r_feature_onnx":
        raise ProvenanceError("Feature provenance does not describe a LoMa-R feature ONNX")
    if matcher_contract.get("artifact_kind") != "loma_r_matcher_onnx":
        raise ProvenanceError("Matcher provenance does not describe a LoMa-R matcher ONNX")

    feature_count = _contract_value(
        feature_contract, "profile", "feature_keypoint_count"
    )
    matcher_maximum = _contract_value(
        matcher_contract, "profile", "keypoints", "maximum"
    )
    matcher_dynamic = _contract_value(
        matcher_contract, "profile", "keypoints", "dynamic"
    )
    matcher_minimum = _contract_value(
        matcher_contract, "profile", "keypoints", "minimum"
    )
    matcher_buckets = _contract_value(
        matcher_contract, "profile", "keypoints", "package_buckets"
    )
    if feature_count != FEATURE_KEYPOINT_COUNT:
        raise ProvenanceError(
            f"LoMa-R feature provenance must declare K{FEATURE_KEYPOINT_COUNT}"
        )
    if (
        not isinstance(matcher_minimum, int)
        or not isinstance(matcher_maximum, int)
        or matcher_minimum > min(PACKAGE_KEYPOINT_BUCKETS)
        or matcher_maximum < FEATURE_KEYPOINT_COUNT
        or matcher_dynamic is not True
        or matcher_buckets != list(PACKAGE_KEYPOINT_BUCKETS)
    ):
        raise ProvenanceError("LoMa-R matcher provenance must declare dynamic K up to 3840")

    feature_precision = _contract_value(feature_contract, "precision")
    matcher_precision = _contract_value(matcher_contract, "precision")
    if feature_precision != matcher_precision or feature_precision not in {"fp16", "fp32"}:
        raise ProvenanceError("LoMa-R feature and matcher precision contracts do not match")

    feature_revision = _contract_value(feature_contract, "source", "revision")
    matcher_revision = _contract_value(matcher_contract, "source", "revision")
    if feature_revision != matcher_revision:
        raise ProvenanceError("LoMa-R feature and matcher source revisions do not match")

    input_width = _contract_value(feature_contract, "input", "width")
    input_height = _contract_value(feature_contract, "input", "height")
    descriptor_dimension = _contract_value(
        feature_contract, "model", "configuration", "descriptor_dimension"
    )
    matcher_descriptor_dimension = _contract_value(
        matcher_contract, "model", "configuration", "descriptor_dimension"
    )
    if descriptor_dimension != DESCRIPTOR_DIMENSION or (
        matcher_descriptor_dimension != descriptor_dimension
    ):
        raise ProvenanceError("LoMa-R descriptor dimensions are incompatible")
    if (
        not isinstance(input_width, int)
        or not isinstance(input_height, int)
        or input_width <= 0
        or input_height <= 0
    ):
        raise ProvenanceError("LoMa-R feature input dimensions are invalid")

    expected_feature_name = (
        f"loma_r_features_k{FEATURE_KEYPOINT_COUNT}_{feature_precision}.onnx"
    )
    expected_matcher_name = f"loma_r_matcher_dynamic_{feature_precision}.onnx"
    if feature_onnx.name != expected_feature_name:
        raise ProvenanceError(
            f"Shared LoMa-R feature ONNX must be named {expected_feature_name}"
        )
    if matcher_onnx.name != expected_matcher_name:
        raise ProvenanceError(
            f"Dynamic LoMa-R matcher ONNX must be named {expected_matcher_name}"
        )
    if feature_onnx.parent != destination or matcher_onnx.parent != destination:
        raise ProvenanceError(
            "LoMa-R ONNX artifacts must already be in the package output directory; "
            "manual rename/copy is not supported"
        )

    feature_record = feature_document["artifact"]
    matcher_record = matcher_document["artifact"]
    manifests = []
    for keypoint_count in PACKAGE_KEYPOINT_BUCKETS:
        manifest = destination / f"loma_r_k{keypoint_count}_{feature_precision}.json"
        document = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "algorithm_id": "loma_r",
            "algorithm_version": 1,
            "source": "LoMa-R (DaD + DeDoDe-G/DINOv2 + LoMa-R)",
            "precision": feature_precision,
            "input_width": input_width,
            "input_height": input_height,
            "keypoint_count": keypoint_count,
            "feature_keypoint_count": feature_count,
            "descriptor_dimension": descriptor_dimension,
            "feature_onnx": feature_record["file"],
            "matcher_onnx": matcher_record["file"],
            "feature_onnx_sha256": feature_record["sha256"],
            "matcher_onnx_sha256": matcher_record["sha256"],
        }
        _write_release_manifest(manifest, document)
        manifests.append(manifest)
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose validated LoMa-R K1024/K2048/K3840 manifests"
    )
    parser.add_argument("--feature-onnx", type=Path, required=True)
    parser.add_argument("--matcher-onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = compose_package(
        args.feature_onnx, args.matcher_onnx, args.output_dir
    )
    print("Portable LoMa-R package manifests:")
    for manifest in manifests:
        print(f"  {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
