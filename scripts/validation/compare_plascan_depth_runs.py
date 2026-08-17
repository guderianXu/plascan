#!/usr/bin/env python3
"""CLI compatibility entry point for PlaScan depth-run comparison."""

from __future__ import annotations

from pathlib import Path
import sys


VALIDATION_DIR = Path(__file__).resolve().parent
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from plascan_depth_compare import (  # noqa: E402,F401
    CV_16UC1,
    CV_32FC1,
    DEFAULT_SAMPLE_LIMIT,
    FAST_MATRIX_HEADER,
    FAST_MATRIX_MAGIC,
    compare_depth_runs,
    evaluate_gates,
    read_fast_matrix,
)
from plascan_depth_compare.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
