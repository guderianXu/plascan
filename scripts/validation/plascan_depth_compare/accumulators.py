"""Bounded-memory distribution and run-level metric accumulators."""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_SAMPLE_LIMIT = 1_000_000
QUANTILES = (0.50, 0.90, 0.95, 0.99)
QUANTILE_NAMES = ("p50", "p90", "p95", "p99")


class DistributionAccumulator:
    """Exact count/mean/max plus a deterministic uniform reservoir.

    Merging uses a hypergeometric draw to choose how many items in the final
    reservoir come from the new batch. A uniform subset of the old uniform
    reservoir plus a uniform subset of the new batch remains a uniform sample
    of the combined population.
    """

    def __init__(self, sample_limit: int, seed: int) -> None:
        self.sample_limit = sample_limit
        self.population_count = 0
        self.total = 0.0
        self.maximum: float | None = None
        self.sample = np.empty(0, dtype=np.float64)
        self._random = np.random.default_rng(seed)

    def add(self, values: np.ndarray) -> None:
        flattened = np.asarray(values).reshape(-1)
        if flattened.size == 0:
            return
        finite_values = flattened[np.isfinite(flattened)]
        if finite_values.size == 0:
            return
        finite_values = np.asarray(finite_values, dtype=np.float64)
        new_count = int(finite_values.size)
        old_count = self.population_count
        self.population_count += new_count
        self.total += float(np.sum(finite_values, dtype=np.float64))
        batch_maximum = float(np.max(finite_values))
        self.maximum = (
            batch_maximum
            if self.maximum is None
            else max(self.maximum, batch_maximum)
        )

        if self.population_count <= self.sample_limit:
            self.sample = np.concatenate((self.sample, finite_values))
            return
        if old_count == 0:
            selected = self._random.choice(
                new_count, size=self.sample_limit, replace=False
            )
            self.sample = finite_values[selected].copy()
            return

        take_new = int(
            self._random.hypergeometric(
                ngood=new_count,
                nbad=old_count,
                nsample=self.sample_limit,
            )
        )
        keep_old = self.sample_limit - take_new
        old_values = self._sample_old(keep_old)
        new_values = self._sample_new(finite_values, take_new)
        self.sample = np.concatenate((old_values, new_values))

    def _sample_old(self, count: int) -> np.ndarray:
        if count == 0:
            return np.empty(0, dtype=np.float64)
        indices = self._random.choice(self.sample.size, size=count, replace=False)
        return self.sample[indices]

    def _sample_new(self, values: np.ndarray, count: int) -> np.ndarray:
        if count == 0:
            return np.empty(0, dtype=np.float64)
        indices = self._random.choice(values.size, size=count, replace=False)
        return values[indices]

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "count": self.population_count,
            "sample_count": int(self.sample.size),
            "quantiles_approximate": self.population_count > self.sample.size,
        }
        if self.population_count == 0:
            result.update({name: None for name in QUANTILE_NAMES})
            result.update({"mean": None, "max": None})
            return result
        values = np.quantile(self.sample, QUANTILES)
        result.update(
            {name: float(value) for name, value in zip(QUANTILE_NAMES, values)}
        )
        result["mean"] = self.total / self.population_count
        result["max"] = self.maximum
        return result


class DepthAggregate:
    FIELDS = (
        "pixel_count",
        "valid_pixel_count",
        "zero_depth_count",
        "negative_depth_count",
        "nonfinite_depth_count",
    )

    def __init__(self) -> None:
        for name in self.FIELDS:
            setattr(self, name, 0)

    def add(self, metrics: dict[str, Any]) -> None:
        for name in self.FIELDS:
            setattr(self, name, getattr(self, name) + int(metrics[name]))

    def summary(self) -> dict[str, Any]:
        return {
            "pixel_count": self.pixel_count,
            "valid_pixel_count": self.valid_pixel_count,
            "coverage": (
                self.valid_pixel_count / self.pixel_count
                if self.pixel_count
                else None
            ),
            "zero_depth_count": self.zero_depth_count,
            "negative_depth_count": self.negative_depth_count,
            "nonfinite_depth_count": self.nonfinite_depth_count,
        }


class SupportAggregate:
    def __init__(self, sample_limit: int, seed: int) -> None:
        self.distribution = DistributionAccumulator(sample_limit, seed)
        self.available_frame_count = 0
        self.scope_valid_pixel_count = 0
        self.count_ge_2 = 0
        self.count_ge_3 = 0

    def summary(self, paired_frame_count: int) -> dict[str, Any]:
        count = self.scope_valid_pixel_count
        return {
            "scope": "valid_depth_pixels_with_artifact",
            "available_frame_count": self.available_frame_count,
            "missing_frame_count": paired_frame_count - self.available_frame_count,
            "scope_valid_pixel_count": count,
            "distribution": self.distribution.summary(),
            "ratio_ge_2": self.count_ge_2 / count if count else None,
            "ratio_ge_3": self.count_ge_3 / count if count else None,
        }


class SpreadAggregate:
    def __init__(self, sample_limit: int, seed: int) -> None:
        self.distribution = DistributionAccumulator(sample_limit, seed)
        self.available_frame_count = 0
        self.scope_valid_pixel_count = 0
        self.nonfinite_count = 0
        self.negative_count = 0

    def summary(self, paired_frame_count: int) -> dict[str, Any]:
        return {
            "scope": "valid_depth_pixels_with_artifact",
            "available_frame_count": self.available_frame_count,
            "missing_frame_count": paired_frame_count - self.available_frame_count,
            "scope_valid_pixel_count": self.scope_valid_pixel_count,
            "nonfinite_count": self.nonfinite_count,
            "negative_count": self.negative_count,
            "distribution": self.distribution.summary(),
        }
