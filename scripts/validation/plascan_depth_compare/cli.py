"""Command-line interface for PlaScan depth-run comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .accumulators import DEFAULT_SAMPLE_LIMIT
from .comparison import compare_depth_runs
from .gates import evaluate_gates


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return parsed


def probability(value: str) -> float:
    parsed = nonnegative_float(value)
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and candidate PlaScan mvs_manifest.json depth artifacts."
        )
    )
    parser.add_argument(
        "--baseline-manifest",
        "--baseline",
        dest="baseline_manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--candidate-manifest",
        "--candidate",
        dest="candidate_manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-coverage-decline-pp",
        type=nonnegative_float,
        help="Maximum aggregate candidate coverage decline in percentage points.",
    )
    parser.add_argument(
        "--min-aggregate-mask-iou",
        type=probability,
        help="Minimum pooled valid-mask intersection over union.",
    )
    parser.add_argument(
        "--max-relative-depth-p95",
        type=nonnegative_float,
        help="Maximum pooled common-valid relative depth difference P95.",
    )
    parser.add_argument(
        "--require-zero-nonfinite-negative",
        "--zero-tolerance-nonfinite-negative",
        dest="require_zero_nonfinite_negative",
        action="store_true",
        help="Fail if the candidate contains any non-finite or finite negative depth.",
    )
    parser.add_argument(
        "--quantile-sample-limit",
        type=positive_int,
        default=DEFAULT_SAMPLE_LIMIT,
        help=(
            "Maximum deterministic reservoir size for each frame/run distribution "
            f"(default: {DEFAULT_SAMPLE_LIMIT})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_depth_runs(
        args.baseline_manifest,
        args.candidate_manifest,
        quantile_sample_limit=args.quantile_sample_limit,
    )
    report["quality_gate"] = evaluate_gates(
        report,
        max_coverage_decline_pp=args.max_coverage_decline_pp,
        min_aggregate_mask_iou=args.min_aggregate_mask_iou,
        max_relative_depth_p95=args.max_relative_depth_p95,
        require_zero_nonfinite_negative=args.require_zero_nonfinite_negative,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    aggregate = report["aggregate"]["comparison"]
    print(
        f"Compared {report['pairing']['paired_frame_count']} PlaScan depth frames: "
        f"coverage decline={aggregate['coverage_decline_pp']:.3f} pp, "
        f"mask IoU={aggregate['mask_iou']:.6f}, "
        f"relative P95={aggregate['relative_depth_difference']['p95']}, "
        f"gate={report['quality_gate']['mode']}/"
        f"{'passed' if report['quality_gate']['passed'] else 'failed'}"
    )
    return 0 if report["quality_gate"]["passed"] else 2
