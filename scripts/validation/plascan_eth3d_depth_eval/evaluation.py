"""Depth-domain metrics for an ETH3D remapped ground truth image."""

from __future__ import annotations

import numpy as np


def evaluate_depth_prediction(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, object]:
    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Depth shape mismatch: prediction={prediction.shape}, "
            f"ground_truth={ground_truth.shape}"
        )
    if prediction.ndim != 2:
        raise ValueError("Depth arrays must be two-dimensional")

    prediction_valid = np.isfinite(prediction) & (prediction > 0.0)
    ground_truth_valid = np.isfinite(ground_truth) & (ground_truth > 0.0)
    if not np.any(ground_truth_valid):
        raise ValueError("Remapped ground truth contains no valid depth pixels")
    common = prediction_valid & ground_truth_valid
    ground_truth_count = int(np.count_nonzero(ground_truth_valid))
    prediction_count = int(np.count_nonzero(prediction_valid))
    common_count = int(np.count_nonzero(common))

    absolute_error = np.abs(
        prediction[common].astype(np.float64)
        - ground_truth[common].astype(np.float64)
    )
    relative_error = absolute_error / np.maximum(
        ground_truth[common].astype(np.float64),
        np.finfo(np.float64).eps,
    )
    signed_error = (
        prediction[common].astype(np.float64)
        - ground_truth[common].astype(np.float64)
    )
    return {
        "pixel_count": int(prediction.size),
        "ground_truth_valid_pixel_count": ground_truth_count,
        "prediction_valid_pixel_count": prediction_count,
        "common_valid_pixel_count": common_count,
        "prediction_valid_outside_ground_truth_pixel_count": int(
            np.count_nonzero(prediction_valid & ~ground_truth_valid)
        ),
        "ground_truth_missing_prediction_pixel_count": int(
            np.count_nonzero(ground_truth_valid & ~prediction_valid)
        ),
        "prediction_coverage_of_ground_truth": float(
            common_count / ground_truth_count
        ),
        "absolute_depth_error": _distribution(absolute_error, include_rmse=True),
        "relative_absolute_depth_error": _distribution(
            relative_error, include_rmse=True
        ),
        "signed_depth_error": _distribution(signed_error, include_rmse=False),
        "accuracy": {
            "within_0_5_percent": _fraction_within(relative_error, 0.005),
            "within_1_percent": _fraction_within(relative_error, 0.01),
            "within_2_percent": _fraction_within(relative_error, 0.02),
            "within_5_percent": _fraction_within(relative_error, 0.05),
        },
    }


def _distribution(
    values: np.ndarray,
    *,
    include_rmse: bool,
) -> dict[str, float | int | None]:
    if values.size == 0:
        result: dict[str, float | int | None] = {
            "sample_count": 0,
            "minimum": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "mean": None,
            "maximum": None,
        }
        if include_rmse:
            result["rmse"] = None
        return result
    result = {
        "sample_count": int(values.size),
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }
    if include_rmse:
        result["rmse"] = float(np.sqrt(np.mean(np.square(values))))
    return result


def _fraction_within(values: np.ndarray, threshold: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values <= threshold))
