"""Optional acceptance gates for a completed depth comparison report."""

from __future__ import annotations

from typing import Any


def evaluate_gates(
    report: dict[str, Any],
    max_coverage_decline_pp: float | None = None,
    min_aggregate_mask_iou: float | None = None,
    max_relative_depth_p95: float | None = None,
    require_zero_nonfinite_negative: bool = False,
) -> dict[str, Any]:
    configured = {
        "max_coverage_decline_pp": max_coverage_decline_pp,
        "min_aggregate_mask_iou": min_aggregate_mask_iou,
        "max_relative_depth_p95": max_relative_depth_p95,
        "require_zero_candidate_nonfinite_negative": require_zero_nonfinite_negative,
    }
    enforced = any(
        value is not None
        for value in (
            max_coverage_decline_pp,
            min_aggregate_mask_iou,
            max_relative_depth_p95,
        )
    ) or require_zero_nonfinite_negative
    failures: list[dict[str, Any]] = []
    comparison = report["aggregate"]["comparison"]
    candidate_depth = report["aggregate"]["candidate"]["depth"]

    if (
        max_coverage_decline_pp is not None
        and comparison["coverage_decline_pp"] > max_coverage_decline_pp
    ):
        failures.append(
            {
                "code": "coverage_decline_pp_exceeded",
                "actual": comparison["coverage_decline_pp"],
                "limit": max_coverage_decline_pp,
            }
        )
    if (
        min_aggregate_mask_iou is not None
        and comparison["mask_iou"] < min_aggregate_mask_iou
    ):
        failures.append(
            {
                "code": "aggregate_mask_iou_below_minimum",
                "actual": comparison["mask_iou"],
                "limit": min_aggregate_mask_iou,
            }
        )
    if max_relative_depth_p95 is not None:
        relative_p95 = comparison["relative_depth_difference"]["p95"]
        if relative_p95 is None:
            failures.append(
                {
                    "code": "relative_depth_p95_unavailable",
                    "actual": None,
                    "limit": max_relative_depth_p95,
                }
            )
        elif relative_p95 > max_relative_depth_p95:
            failures.append(
                {
                    "code": "relative_depth_p95_exceeded",
                    "actual": relative_p95,
                    "limit": max_relative_depth_p95,
                }
            )
    if require_zero_nonfinite_negative:
        invalid_count = int(candidate_depth["nonfinite_depth_count"]) + int(
            candidate_depth["negative_depth_count"]
        )
        if invalid_count:
            failures.append(
                {
                    "code": "candidate_nonfinite_or_negative_depth",
                    "actual": invalid_count,
                    "limit": 0,
                    "nonfinite_depth_count": candidate_depth[
                        "nonfinite_depth_count"
                    ],
                    "negative_depth_count": candidate_depth[
                        "negative_depth_count"
                    ],
                }
            )
    return {
        "mode": "enforced" if enforced else "report_only",
        "configured": configured,
        "passed": not failures,
        "failures": failures,
    }
