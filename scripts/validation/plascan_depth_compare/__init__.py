"""Reusable PlaScan depth-run comparison API."""

from .accumulators import DEFAULT_SAMPLE_LIMIT
from .comparison import compare_depth_runs
from .fast_matrix import (
    CV_16UC1,
    CV_32FC1,
    FAST_MATRIX_HEADER,
    FAST_MATRIX_MAGIC,
    read_fast_matrix,
)
from .gates import evaluate_gates

__all__ = [
    "CV_16UC1",
    "CV_32FC1",
    "DEFAULT_SAMPLE_LIMIT",
    "FAST_MATRIX_HEADER",
    "FAST_MATRIX_MAGIC",
    "compare_depth_runs",
    "evaluate_gates",
    "read_fast_matrix",
]
