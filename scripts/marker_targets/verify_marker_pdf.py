#!/usr/bin/env python3
"""Rasterize a marker PDF and verify its coded IDs with marker_detect_cli."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable


def parse_expected_ids(text: str) -> list[int]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected IDs must not be empty")
    try:
        ids = [int(value) for value in values]
    except ValueError as exc:
        raise ValueError("expected IDs must be comma-separated integers") from exc
    if any(target_id < 0 for target_id in ids):
        raise ValueError("expected IDs must be non-negative")
    if len(set(ids)) != len(ids):
        raise ValueError("expected IDs must not contain duplicates")
    return ids


def validate_detected_ids(expected: Iterable[int], detected: Iterable[int]) -> None:
    expected_counts = Counter(expected)
    detected_counts = Counter(detected)
    if detected_counts == expected_counts:
        return

    missing = sorted((expected_counts - detected_counts).elements())
    extras = sorted((detected_counts - expected_counts).elements())
    raise ValueError(f"marker ID mismatch: missing={missing}, extra_or_duplicate={extras}")


def render_pdf(pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required; run scripts/env/setup_python_runtime.py first"
        ) from exc

    document = fitz.open(pdf_path)
    try:
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pages: list[Path] = []
        for page_index in range(document.page_count):
            output = output_dir / f"page_{page_index + 1:04d}.png"
            pixmap = document.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(output)
            pages.append(output)
        return pages
    finally:
        document.close()


def detect_page(detector: Path, image: Path, family: str, output: Path) -> list[int]:
    command = [
        str(detector),
        "--image",
        str(image),
        "--family",
        family,
        "--output",
        str(output),
        "--min-decision-margin",
        "0",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        details = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"marker detector failed for {image}: {details}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("schema") != "plascan.marker-detections.v1":
        raise ValueError(f"unexpected detector schema in {output}")
    return [int(item["target_id"]) for item in payload.get("observations", [])]


def verify_pdf(
    pdf_path: Path,
    detector: Path,
    family: str,
    expected_ids: list[int],
    dpi: int,
    output_dir: Path,
) -> list[int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = render_pdf(pdf_path, output_dir, dpi)
    detected: list[int] = []
    for page_index, image in enumerate(pages, start=1):
        detected.extend(
            detect_page(
                detector,
                image,
                family,
                output_dir / f"detections_{page_index:04d}.json",
            )
        )
    validate_detected_ids(expected_ids, detected)
    return detected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rasterize a marker PDF and verify every requested coded target ID."
    )
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--expected-ids", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Keep rasterized pages and detector JSON in this directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    if not args.detector.is_file():
        raise FileNotFoundError(f"detector not found: {args.detector}")
    if args.dpi < 72:
        raise ValueError("DPI must be at least 72")
    expected = parse_expected_ids(args.expected_ids)

    if args.work_dir:
        detected = verify_pdf(
            args.pdf, args.detector, args.family, expected, args.dpi, args.work_dir
        )
    else:
        with tempfile.TemporaryDirectory(prefix="plascan-marker-pdf-") as directory:
            detected = verify_pdf(
                args.pdf,
                args.detector,
                args.family,
                expected,
                args.dpi,
                Path(directory),
            )
    print(f"marker PDF verified: family={args.family}, ids={sorted(detected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
