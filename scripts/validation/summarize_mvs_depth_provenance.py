#!/usr/bin/env python3
"""Validate and summarize persisted MVS depth-provenance artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROVENANCE_KEYS = (
    "native_patchmatch_pixel_count",
    "targeted_patchmatch_pixel_count",
    "cross_view_measured_pixel_count",
    "anchored_interpolation_pixel_count",
)


def _integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return int(value)


def _latest_frames(document: dict[str, Any]) -> list[dict[str, Any]]:
    records = document.get("depth_artifacts", document.get("frames"))
    if not isinstance(records, list):
        raise ValueError("input must contain a depth_artifacts or frames array")

    latest: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every depth frame record must be an object")
        ref_index = record.get("ref_index")
        if isinstance(ref_index, bool) or not isinstance(ref_index, int):
            raise ValueError("every depth frame record must contain integer ref_index")
        if record.get("status", "completed") == "completed":
            latest[ref_index] = record
    return [latest[index] for index in sorted(latest)]


def summarize(document: dict[str, Any], input_path: Path) -> dict[str, Any]:
    frames = _latest_frames(document)
    errors: list[str] = []
    totals = {key: 0 for key in PROVENANCE_KEYS}
    totals.update(
        valid_pixel_count=0,
        unclassified_valid_pixel_count=0,
        support_pixel_count=0,
        missing_pixel_count=0,
        targeted_requested_gap_pixel_count=0,
        targeted_candidate_pixel_count=0,
        targeted_recovered_pixel_count=0,
    )
    existing_artifact_count = 0

    for frame in frames:
        ref_index = frame["ref_index"]
        provenance = frame.get("depth_provenance_summary")
        if not isinstance(provenance, dict) or not provenance.get("available", False):
            errors.append(f"frame {ref_index}: depth provenance summary is unavailable")
            continue

        valid_count = _integer(provenance, "valid_pixel_count")
        classified_count = 0
        for key in PROVENANCE_KEYS:
            count = _integer(provenance, key)
            totals[key] += count
            classified_count += count
        unclassified_count = _integer(
            provenance, "unclassified_valid_pixel_count"
        )
        totals["valid_pixel_count"] += valid_count
        totals["unclassified_valid_pixel_count"] += unclassified_count
        if classified_count != valid_count or unclassified_count != 0:
            errors.append(
                f"frame {ref_index}: classified={classified_count}, "
                f"valid={valid_count}, unclassified={unclassified_count}"
            )

        artifact_text = frame.get("depth_provenance_path", "")
        if not isinstance(artifact_text, str) or not artifact_text:
            errors.append(f"frame {ref_index}: depth_provenance_path is empty")
        else:
            artifact_path = Path(artifact_text)
            if not artifact_path.is_absolute():
                artifact_path = input_path.parent / artifact_path
            if artifact_path.is_file():
                existing_artifact_count += 1
            else:
                errors.append(
                    f"frame {ref_index}: provenance artifact does not exist: "
                    f"{artifact_path}"
                )

        missing = frame.get("missing_reason_summary", {})
        if isinstance(missing, dict):
            totals["support_pixel_count"] += _integer(
                missing, "support_pixel_count"
            )
            totals["missing_pixel_count"] += _integer(
                missing, "missing_pixel_count"
            )
        targeted = frame.get("targeted_gap_recovery_diagnostics", {})
        if isinstance(targeted, dict):
            totals["targeted_requested_gap_pixel_count"] += _integer(
                targeted, "requested_gap_pixel_count"
            )
            totals["targeted_candidate_pixel_count"] += _integer(
                targeted, "candidate_pixel_count"
            )
            totals["targeted_recovered_pixel_count"] += _integer(
                targeted, "recovered_pixel_count"
            )

    valid_count = totals["valid_pixel_count"]
    measured_count = sum(totals[key] for key in PROVENANCE_KEYS[:3])
    support_count = totals["support_pixel_count"]
    return {
        "schema": "plascan.mvs.depth_provenance_validation.v1",
        "input": str(input_path.resolve()),
        "frame_count": len(frames),
        "existing_provenance_artifact_count": existing_artifact_count,
        **totals,
        "classified_valid_pixel_count": sum(
            totals[key] for key in PROVENANCE_KEYS
        ),
        "measured_pixel_count": measured_count,
        "measured_valid_ratio": measured_count / valid_count if valid_count else 0.0,
        "interpolated_valid_ratio": (
            totals["anchored_interpolation_pixel_count"] / valid_count
            if valid_count
            else 0.0
        ),
        "missing_within_support_ratio": (
            totals["missing_pixel_count"] / support_count if support_count else 0.0
        ),
        "passed": not errors and bool(frames),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that every final valid depth pixel has exactly one persisted "
            "provenance class and summarize the batch."
        )
    )
    parser.add_argument("input", type=Path, help="mvs_replay_report.json or mvs_manifest.json")
    parser.add_argument("--output", type=Path, help="optional summary JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("input root must be a JSON object")
        result = summarize(document, args.input)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"depth provenance validation failed: {error}", file=sys.stderr)
        return 2

    output_text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    print(output_text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
