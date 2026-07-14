#!/usr/bin/env python3
"""Import licensed Metashape marker exports into PlaScan's golden corpus.

The input directory must contain an ``index.csv`` with these columns:

    source,family,id,page,x,y,width,height

``page`` is one-based for PDF files. The crop rectangle is expressed in PDF
points; leave x/y/width/height empty when one raster image contains one target.
The script never reads Metashape program files. It only imports user-exported
PDFs or images and records deterministic SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_FAMILIES = {"circular12", "circular14", "circular16", "circular20"}


@dataclass(frozen=True)
class CorpusEntry:
    source: str
    family: str
    target_id: int
    page: int
    x: float | None
    y: float | None
    width: float | None
    height: float | None


def optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def load_index(index_path: Path) -> list[CorpusEntry]:
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source", "family", "id", "page", "x", "y", "width", "height"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"index.csv missing columns: {', '.join(sorted(missing))}")

        entries: list[CorpusEntry] = []
        for row_number, row in enumerate(reader, start=2):
            family = row["family"].strip().lower()
            if family not in SUPPORTED_FAMILIES:
                raise ValueError(f"index.csv row {row_number}: unsupported family {family!r}")
            entry = CorpusEntry(
                source=row["source"].strip(),
                family=family,
                target_id=int(row["id"]),
                page=int(row["page"] or "1"),
                x=optional_float(row["x"]),
                y=optional_float(row["y"]),
                width=optional_float(row["width"]),
                height=optional_float(row["height"]),
            )
            crop_values = (entry.x, entry.y, entry.width, entry.height)
            if any(value is None for value in crop_values) != all(value is None for value in crop_values):
                raise ValueError(f"index.csv row {row_number}: crop rectangle must be complete or empty")
            if entry.page < 1 or entry.target_id < 0:
                raise ValueError(f"index.csv row {row_number}: page must be >= 1 and id must be >= 0")
            entries.append(entry)
    return entries


def resolved_source(input_dir: Path, relative_source: str) -> Path:
    source = (input_dir / relative_source).resolve()
    try:
        source.relative_to(input_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"source escapes input directory: {relative_source}") from exc
    if not source.is_file():
        raise FileNotFoundError(f"source does not exist: {source}")
    return source


def render_entry(source: Path, entry: CorpusEntry, output_path: Path, dpi: int) -> None:
    if source.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF import requires pymupdf; run setup_python_runtime.py") from exc

        document = fitz.open(source)
        try:
            if entry.page > document.page_count:
                raise ValueError(f"page {entry.page} is outside {source.name} ({document.page_count} pages)")
            page = document.load_page(entry.page - 1)
            clip = None
            if entry.x is not None:
                clip = fitz.Rect(entry.x, entry.y, entry.x + entry.width, entry.y + entry.height)
                if not page.rect.contains(clip):
                    raise ValueError(f"crop rectangle is outside page {entry.page} of {source.name}")
            scale = dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
            pixmap.save(output_path)
        finally:
            document.close()
        return

    if entry.x is not None:
        raise ValueError(f"raster crop is intentionally unsupported; crop {source.name} before import")
    shutil.copyfile(source, output_path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_corpus(input_dir: Path, output_dir: Path, index_path: Path, dpi: int) -> Path:
    entries = load_index(index_path)
    if not entries:
        raise ValueError("index.csv does not contain any targets")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        key = (entry.family, entry.target_id)
        if key in seen:
            raise ValueError(f"duplicate family/id in corpus: {entry.family}/{entry.target_id}")
        seen.add(key)

        source = resolved_source(input_dir, entry.source)
        output_name = f"{entry.family}_{entry.target_id:06d}.png"
        output_path = image_dir / output_name
        render_entry(source, entry, output_path, dpi)
        manifest_entries.append(
            {
                "family": entry.family,
                "id": entry.target_id,
                "page": entry.page,
                "source": entry.source,
                "image": f"images/{output_name}",
                "sha256": sha256(output_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "provenance": "User-exported licensed Agisoft Metashape marker PDF/image",
        "dpi": dpi,
        "entries": manifest_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory containing exported PDFs/images")
    parser.add_argument("--output", required=True, type=Path, help="Golden corpus output directory")
    parser.add_argument("--index", type=Path, help="CSV index; defaults to <input>/index.csv")
    parser.add_argument("--dpi", type=int, default=600, help="PDF rasterization DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    index_path = args.index.expanduser().resolve() if args.index else input_dir / "index.csv"
    if args.dpi < 72:
        raise ValueError("dpi must be at least 72")
    manifest_path = import_corpus(input_dir, output_dir, index_path, args.dpi)
    print(f"Imported corpus manifest: {manifest_path}")


if __name__ == "__main__":
    main()
