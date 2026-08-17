"""Orchestrate frame pairing and construct the depth A/B report."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .accumulators import (
    DEFAULT_SAMPLE_LIMIT,
    DepthAggregate,
    DistributionAccumulator,
    SpreadAggregate,
    SupportAggregate,
)
from .fast_matrix import CV_32FC1, read_fast_matrix
from .manifest import (
    acceptance,
    artifact_path,
    frame_index,
    fusion_eligible,
    key_record,
    load_manifest,
)
from .metrics import (
    depth_metrics,
    relative_depth_metrics,
    spread_metrics,
    support_metrics,
)


class RunAggregates:
    def __init__(self, sample_limit: int, seed: int) -> None:
        self.depth = DepthAggregate()
        self.support = SupportAggregate(sample_limit, seed + 1)
        self.spread = SpreadAggregate(sample_limit, seed + 2)
        self.acceptance: Counter[str] = Counter()
        self.eligibility: Counter[str] = Counter()

    def add_qualification(
        self, frame_acceptance: str, eligible: bool | None
    ) -> None:
        self.acceptance[frame_acceptance] += 1
        label = "unknown" if eligible is None else str(eligible).lower()
        self.eligibility[label] += 1

    def summary(self, paired_frame_count: int) -> dict[str, Any]:
        return {
            "depth": self.depth.summary(),
            "acceptance_counts": dict(sorted(self.acceptance.items())),
            "fusion_eligible_counts": dict(sorted(self.eligibility.items())),
            "geometry_support": self.support.summary(paired_frame_count),
            "inverse_depth_spread": self.spread.summary(paired_frame_count),
        }


def side_report(
    frame: dict[str, Any],
    manifest_path: Path,
    depth_path: Path,
    depth: np.ndarray,
    depth_report: dict[str, Any],
    valid_depth: np.ndarray,
    sample_limit: int,
    aggregate: RunAggregates,
    seed: int,
) -> tuple[dict[str, Any], str, bool | None]:
    frame_acceptance = acceptance(frame)
    eligible, eligibility_source = fusion_eligible(frame, frame_acceptance)
    aggregate.depth.add(depth_report)
    aggregate.add_qualification(frame_acceptance, eligible)
    support_path = artifact_path(
        frame, "raw_geometry_support_path", manifest_path, required=False
    )
    spread_path = artifact_path(
        frame, "raw_inverse_depth_spread_path", manifest_path, required=False
    )
    return {
        "raw_depth_path": str(depth_path),
        "status": str(frame.get("status", "")),
        "acceptance": frame_acceptance,
        "fusion_eligible": eligible,
        "fusion_eligible_source": eligibility_source,
        "depth": depth_report,
        "geometry_support": support_metrics(
            support_path,
            depth.shape,
            valid_depth,
            sample_limit,
            aggregate.support,
            seed=seed + 1,
        ),
        "inverse_depth_spread": spread_metrics(
            spread_path,
            depth.shape,
            valid_depth,
            sample_limit,
            aggregate.spread,
            seed=seed + 2,
        ),
    }, frame_acceptance, eligible


def eligibility_label(value: bool | None) -> str:
    return "unknown" if value is None else str(value).lower()


def compare_depth_runs(
    baseline_manifest: Path,
    candidate_manifest: Path,
    quantile_sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    if quantile_sample_limit <= 0:
        raise ValueError("quantile_sample_limit must be positive")
    baseline_path, baseline_document = load_manifest(baseline_manifest)
    candidate_path, candidate_document = load_manifest(candidate_manifest)
    baseline_frames, baseline_duplicates = frame_index(
        baseline_document, baseline_path
    )
    candidate_frames, candidate_duplicates = frame_index(
        candidate_document, candidate_path
    )
    paired_keys = sorted(baseline_frames.keys() & candidate_frames.keys())
    if not paired_keys:
        raise ValueError(
            f"No frames pair by ref_index and ref_image basename: "
            f"{baseline_path}, {candidate_path}"
        )

    baseline_run = RunAggregates(quantile_sample_limit, 2000)
    candidate_run = RunAggregates(quantile_sample_limit, 3000)
    relative_aggregate = DistributionAccumulator(quantile_sample_limit, 1001)
    acceptance_transitions: Counter[str] = Counter()
    eligibility_transitions: Counter[str] = Counter()
    common_valid_pixel_count = 0
    union_valid_pixel_count = 0
    baseline_only_pixel_count = 0
    candidate_only_pixel_count = 0
    frame_reports: list[dict[str, Any]] = []

    for ordinal, key in enumerate(paired_keys):
        baseline_name, baseline_frame = baseline_frames[key]
        candidate_name, candidate_frame = candidate_frames[key]
        baseline_depth_path = artifact_path(
            baseline_frame, "raw_depth_path", baseline_path, required=True
        )
        candidate_depth_path = artifact_path(
            candidate_frame, "raw_depth_path", candidate_path, required=True
        )
        assert baseline_depth_path is not None and candidate_depth_path is not None
        baseline_depth = read_fast_matrix(baseline_depth_path, CV_32FC1)
        candidate_depth = read_fast_matrix(candidate_depth_path, CV_32FC1)
        if baseline_depth.shape != candidate_depth.shape:
            raise ValueError(
                f"Depth shape mismatch for ref_index={key[0]}, image={baseline_name}: "
                f"baseline={baseline_depth.shape}, candidate={candidate_depth.shape}"
            )

        baseline_depth_report, baseline_valid = depth_metrics(baseline_depth)
        candidate_depth_report, candidate_valid = depth_metrics(candidate_depth)
        common_valid = baseline_valid & candidate_valid
        baseline_valid_count = int(baseline_depth_report["valid_pixel_count"])
        candidate_valid_count = int(candidate_depth_report["valid_pixel_count"])
        common_count = int(np.count_nonzero(common_valid))
        union_count = baseline_valid_count + candidate_valid_count - common_count
        baseline_only_count = baseline_valid_count - common_count
        candidate_only_count = candidate_valid_count - common_count
        pixel_count = int(baseline_depth.size)
        common_valid_pixel_count += common_count
        union_valid_pixel_count += union_count
        baseline_only_pixel_count += baseline_only_count
        candidate_only_pixel_count += candidate_only_count

        relative_report = relative_depth_metrics(
            baseline_depth,
            candidate_depth,
            common_valid,
            quantile_sample_limit,
            relative_aggregate,
            seed=4000 + ordinal,
        )
        baseline_side, baseline_acceptance, baseline_eligible = side_report(
            baseline_frame,
            baseline_path,
            baseline_depth_path,
            baseline_depth,
            baseline_depth_report,
            baseline_valid,
            quantile_sample_limit,
            baseline_run,
            seed=5000 + ordinal * 10,
        )
        candidate_side, candidate_acceptance, candidate_eligible = side_report(
            candidate_frame,
            candidate_path,
            candidate_depth_path,
            candidate_depth,
            candidate_depth_report,
            candidate_valid,
            quantile_sample_limit,
            candidate_run,
            seed=6000 + ordinal * 10,
        )
        acceptance_transitions[
            f"{baseline_acceptance}->{candidate_acceptance}"
        ] += 1
        eligibility_transitions[
            f"{eligibility_label(baseline_eligible)}->"
            f"{eligibility_label(candidate_eligible)}"
        ] += 1
        frame_reports.append(
            {
                "ref_index": key[0],
                "image_basename": baseline_name,
                "candidate_image_basename": candidate_name,
                "shape": {
                    "rows": int(baseline_depth.shape[0]),
                    "columns": int(baseline_depth.shape[1]),
                },
                "baseline": baseline_side,
                "candidate": candidate_side,
                "comparison": {
                    "common_valid_pixel_count": common_count,
                    "union_valid_pixel_count": union_count,
                    "mask_iou": common_count / union_count if union_count else 1.0,
                    "baseline_only_pixel_count": baseline_only_count,
                    "baseline_only_fraction": baseline_only_count / pixel_count,
                    "candidate_only_pixel_count": candidate_only_count,
                    "candidate_only_fraction": candidate_only_count / pixel_count,
                    "coverage_decline_pp": 100.0
                    * (
                        float(baseline_depth_report["coverage"])
                        - float(candidate_depth_report["coverage"])
                    ),
                    "relative_depth_difference": relative_report,
                },
            }
        )

    paired_frame_count = len(paired_keys)
    baseline_summary = baseline_run.summary(paired_frame_count)
    candidate_summary = candidate_run.summary(paired_frame_count)
    total_pixel_count = int(baseline_summary["depth"]["pixel_count"])
    unmatched_baseline = [
        key_record(key, baseline_frames[key][0])
        for key in sorted(baseline_frames.keys() - candidate_frames.keys())
    ]
    unmatched_candidate = [
        key_record(key, candidate_frames[key][0])
        for key in sorted(candidate_frames.keys() - baseline_frames.keys())
    ]
    baseline_coverage = float(baseline_summary["depth"]["coverage"])
    candidate_coverage = float(candidate_summary["depth"]["coverage"])
    return {
        "schema_version": 1,
        "baseline_manifest": str(baseline_path),
        "candidate_manifest": str(candidate_path),
        "relative_depth_definition": (
            "abs(candidate-baseline)/max(baseline,float32_epsilon), "
            "common valid pixels"
        ),
        "quantile_sampling": {
            "method": "deterministic_uniform_reservoir",
            "sample_limit_per_distribution": quantile_sample_limit,
            "exact_count_mean_and_max": True,
        },
        "pairing": {
            "key": "ref_index + case-insensitive ref_image basename",
            "duplicate_semantics": "last record wins",
            "baseline_input_record_count": len(baseline_document["frames"]),
            "candidate_input_record_count": len(candidate_document["frames"]),
            "baseline_duplicate_record_count": baseline_duplicates,
            "candidate_duplicate_record_count": candidate_duplicates,
            "baseline_frame_count": len(baseline_frames),
            "candidate_frame_count": len(candidate_frames),
            "paired_frame_count": paired_frame_count,
            "unmatched_baseline": unmatched_baseline,
            "unmatched_candidate": unmatched_candidate,
        },
        "aggregate": {
            "qualification_scope": "paired_latest_frames",
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "comparison": {
                "pixel_count": total_pixel_count,
                "common_valid_pixel_count": common_valid_pixel_count,
                "union_valid_pixel_count": union_valid_pixel_count,
                "mask_iou": (
                    common_valid_pixel_count / union_valid_pixel_count
                    if union_valid_pixel_count
                    else 1.0
                ),
                "baseline_only_pixel_count": baseline_only_pixel_count,
                "baseline_only_fraction": baseline_only_pixel_count
                / total_pixel_count,
                "candidate_only_pixel_count": candidate_only_pixel_count,
                "candidate_only_fraction": candidate_only_pixel_count
                / total_pixel_count,
                "coverage_decline_pp": 100.0
                * (baseline_coverage - candidate_coverage),
                "relative_depth_difference": relative_aggregate.summary(),
                "acceptance_transitions": dict(
                    sorted(acceptance_transitions.items())
                ),
                "fusion_eligible_transitions": dict(
                    sorted(eligibility_transitions.items())
                ),
            },
        },
        "frames": frame_reports,
    }
