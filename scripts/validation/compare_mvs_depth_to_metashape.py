#!/usr/bin/env python3
"""Compare PlaScan MVS depth artifacts with exported Metashape float depths.

Inputs are matched by image label, so Metashape projects that omit depth for a
subset of aligned cameras can still be compared without inventing reference
data. The report separates mask coverage from depth accuracy so an apparently
dense but geometrically wrong PlaScan map cannot pass by covering more pixels
than the reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import cv2
import numpy as np


FAST_MATRIX_HEADER = struct.Struct("<16siii4xQ")
FAST_MATRIX_MAGIC = b"PLASDEPTHMAT01\x00\x00"
CV_32FC1 = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvs-manifest", required=True, type=Path)
    parser.add_argument("--metashape-export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        help="Optional per-frame depth/error panels and an error contact sheet.",
    )
    return parser.parse_args()


def read_fast_float_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(FAST_MATRIX_HEADER.size)
        magic, rows, columns, cv_type, payload_size = FAST_MATRIX_HEADER.unpack(header)
        if magic != FAST_MATRIX_MAGIC or cv_type != CV_32FC1:
            raise ValueError(f"Unsupported PlaScan depth matrix: {path}")
        payload = stream.read(payload_size)
    values = np.frombuffer(payload, dtype="<f4")
    if values.size != rows * columns:
        raise ValueError(f"Invalid PlaScan depth payload size: {path}")
    return values.reshape(rows, columns)


def resolve_candidate_depth_path(
    frame: dict[str, object], mvs_root: Path
) -> Path:
    configured = Path(str(frame["raw_depth_path"]))
    if not configured.is_absolute():
        configured = mvs_root / configured
    if configured.is_file():
        return configured

    candidates: list[tuple[int, Path]] = []
    for level in frame.get("pyramid_levels", []):
        if not level.get("success", False):
            continue
        raw_path = Path(str(level.get("raw_depth_path", "")))
        if not raw_path.is_absolute():
            raw_path = mvs_root / raw_path
        if not raw_path.is_file():
            continue
        area = int(level.get("artifact_width", 0)) * int(
            level.get("artifact_height", 0)
        )
        candidates.append((area, raw_path))
    if not candidates:
        raise FileNotFoundError(configured)
    return max(candidates, key=lambda item: item[0])[1]


def finite_quantile(values: np.ndarray, quantile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, quantile)) if finite.size else None


def colorize_depth(depth: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    scale = max(high - low, np.finfo(np.float32).eps)
    normalized = np.clip((depth - low) / scale, 0.0, 1.0)
    color = cv2.applyColorMap(np.asarray(normalized * 255.0, dtype=np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def make_diagnostic_panel(
    candidate: np.ndarray,
    reference: np.ndarray,
    candidate_valid: np.ndarray,
    reference_valid: np.ndarray,
    label: str,
) -> tuple[np.ndarray, dict[str, object]]:
    reference_values = reference[reference_valid]
    low = float(np.quantile(reference_values, 0.01))
    high = float(np.quantile(reference_values, 0.99))
    candidate_color = colorize_depth(candidate, candidate_valid, low, high)
    reference_color = colorize_depth(reference, reference_valid, low, high)

    overlap = candidate_valid & reference_valid
    residual_image = np.zeros(candidate.shape, dtype=np.float32)
    residual_image[overlap] = np.abs(candidate[overlap] - reference[overlap]) / np.maximum(
        reference[overlap], np.finfo(np.float32).eps
    )
    error_normalized = np.clip(residual_image / 0.05, 0.0, 1.0)
    error_color = cv2.applyColorMap(
        np.asarray(error_normalized * 255.0, dtype=np.uint8), cv2.COLORMAP_TURBO
    )
    error_color[~overlap] = 0
    error_color[candidate_valid & ~reference_valid] = (255, 0, 255)
    error_color[reference_valid & ~candidate_valid] = (0, 255, 255)

    catastrophic = np.asarray(overlap & (residual_image > 0.05), dtype=np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(catastrophic, 8)
    largest_area = 0
    largest_bbox: list[int] | None = None
    if component_count > 1:
        largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_area = int(stats[largest_index, cv2.CC_STAT_AREA])
        largest_bbox = [
            int(stats[largest_index, cv2.CC_STAT_LEFT]),
            int(stats[largest_index, cv2.CC_STAT_TOP]),
            int(stats[largest_index, cv2.CC_STAT_WIDTH]),
            int(stats[largest_index, cv2.CC_STAT_HEIGHT]),
        ]
        x, y, width, height = largest_bbox
        cv2.rectangle(error_color, (x, y), (x + width - 1, y + height - 1), (255, 255, 255), 2)

    panels = [candidate_color, reference_color, error_color]
    titles = ["PlaScan", "Metashape", "relative error (5%=red)"]
    for panel, title in zip(panels, titles, strict=True):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    combined = np.hstack(panels)
    cv2.putText(combined, label, (10, combined.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    diagnostics = {
        "catastrophic_over_5_percent_fraction": float(np.mean(catastrophic != 0)),
        "catastrophic_component_count": max(0, component_count - 1),
        "largest_catastrophic_component_pixels": largest_area,
        "largest_catastrophic_component_bbox": largest_bbox,
    }
    return combined, diagnostics


def main() -> int:
    args = parse_args()
    mvs_manifest_path = args.mvs_manifest.resolve()
    mvs_root = mvs_manifest_path.parent
    mvs_manifest = json.loads(mvs_manifest_path.read_text(encoding="utf-8"))
    metashape_root = args.metashape_export.resolve()
    metashape_manifest = json.loads(
        (metashape_root / "metashape_depth_manifest.json").read_text(encoding="utf-8")
    )

    frames = sorted(mvs_manifest["frames"], key=lambda item: item["ref_index"])
    exported = metashape_manifest["depth_maps"]
    exported_by_label = {
        str(item["label"]).casefold(): item for item in exported
    }
    if len(exported_by_label) != len(exported):
        raise ValueError("Metashape export contains duplicate camera labels")

    frame_reports: list[dict[str, object]] = []
    all_relative_residuals: list[np.ndarray] = []
    all_depth_ratios: list[np.ndarray] = []
    diagnostic_panels: list[np.ndarray] = []
    diagnostic_dir = args.diagnostic_dir.resolve() if args.diagnostic_dir else None
    missing_frame_indices: list[int] = []
    if diagnostic_dir:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        index = int(frame["ref_index"])
        image_label = Path(str(frame["ref_image"])).stem.casefold()
        reference_record = exported_by_label.get(image_label)
        if reference_record is None:
            missing_frame_indices.append(index)
            continue
        candidate_path = resolve_candidate_depth_path(frame, mvs_root)
        reference_path = metashape_root / reference_record["depth_path"]
        candidate = read_fast_float_matrix(candidate_path)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
        if reference is None:
            raise FileNotFoundError(reference_path)
        reference = np.asarray(reference, dtype=np.float32)
        if candidate.shape != reference.shape:
            reference = cv2.resize(
                reference,
                (candidate.shape[1], candidate.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        candidate_valid = np.isfinite(candidate) & (candidate > 0)
        reference_valid = np.isfinite(reference) & (reference > 0)
        overlap = candidate_valid & reference_valid
        union = candidate_valid | reference_valid
        relative_residual = np.abs(candidate[overlap] - reference[overlap]) / np.maximum(
            reference[overlap], np.finfo(np.float32).eps
        )
        depth_ratio = candidate[overlap] / np.maximum(
            reference[overlap], np.finfo(np.float32).eps
        )
        all_relative_residuals.append(relative_residual)
        all_depth_ratios.append(depth_ratio)
        pixel_count = candidate.size
        frame_report: dict[str, object] = {
                "frame_index": index,
                "image_label": image_label,
                "metashape_camera_index": int(reference_record["camera_index"]),
                "candidate_valid_fraction": float(np.count_nonzero(candidate_valid) / pixel_count),
                "reference_valid_fraction": float(np.count_nonzero(reference_valid) / pixel_count),
                "mask_iou": float(np.count_nonzero(overlap) / max(1, np.count_nonzero(union))),
                "candidate_only_fraction": float(
                    np.count_nonzero(candidate_valid & ~reference_valid) / pixel_count
                ),
                "reference_only_fraction": float(
                    np.count_nonzero(reference_valid & ~candidate_valid) / pixel_count
                ),
                "relative_residual_p50": finite_quantile(relative_residual, 0.50),
                "relative_residual_p90": finite_quantile(relative_residual, 0.90),
                "relative_residual_p99": finite_quantile(relative_residual, 0.99),
                "depth_ratio_p50": finite_quantile(depth_ratio, 0.50),
                "candidate_depth_p50": finite_quantile(candidate[candidate_valid], 0.50),
                "reference_depth_p50": finite_quantile(reference[reference_valid], 0.50),
                "within_0_2_percent": float(np.mean(relative_residual <= 0.002))
                if relative_residual.size
                else None,
                "within_1_percent": float(np.mean(relative_residual <= 0.01))
                if relative_residual.size
                else None,
        }
        if diagnostic_dir:
            panel, diagnostics = make_diagnostic_panel(
                candidate,
                reference,
                candidate_valid,
                reference_valid,
                image_label,
            )
            frame_report.update(diagnostics)
            diagnostic_panels.append(panel)
            cv2.imwrite(str(diagnostic_dir / f"frame_{index:02d}.png"), panel)
        frame_reports.append(frame_report)

    combined = np.concatenate(all_relative_residuals) if all_relative_residuals else np.array([])
    combined_ratios = np.concatenate(all_depth_ratios) if all_depth_ratios else np.array([])
    report = {
        "mvs_manifest": str(mvs_manifest_path),
        "metashape_export": str(metashape_root),
        "frame_count": len(frame_reports),
        "mvs_frame_count": len(frames),
        "missing_metashape_frame_indices": missing_frame_indices,
        "frames": frame_reports,
        "aggregate": {
            "relative_residual_p50": finite_quantile(combined, 0.50),
            "relative_residual_p90": finite_quantile(combined, 0.90),
            "relative_residual_p99": finite_quantile(combined, 0.99),
            "depth_ratio_p50": finite_quantile(combined_ratios, 0.50),
            "depth_ratio_p10": finite_quantile(combined_ratios, 0.10),
            "depth_ratio_p90": finite_quantile(combined_ratios, 0.90),
            "within_0_2_percent": float(np.mean(combined <= 0.002)) if combined.size else None,
            "within_1_percent": float(np.mean(combined <= 0.01)) if combined.size else None,
            "mean_candidate_only_fraction": float(
                np.mean([item["candidate_only_fraction"] for item in frame_reports])
            ),
            "mean_reference_only_fraction": float(
                np.mean([item["reference_only_fraction"] for item in frame_reports])
            ),
        },
    }
    args.output.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if diagnostic_dir and diagnostic_panels:
        thumbnail_width = 960
        thumbnails = [
            cv2.resize(panel, (thumbnail_width, max(1, panel.shape[0] * thumbnail_width // panel.shape[1])))
            for panel in diagnostic_panels
        ]
        cv2.imwrite(str(diagnostic_dir / "contact.png"), np.vstack(thumbnails))
    print(json.dumps(report["aggregate"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
