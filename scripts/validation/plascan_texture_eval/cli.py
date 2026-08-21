"""CLI for geometry-separated held-out evaluation of textured OBJ models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .image_io import decoded_rgb_sha256, load_binary_mask, load_linear_reference
from .metrics import appearance_metrics, linear_to_srgb, seam_metrics, uv_metrics
from .model import load_textured_obj
from .rasterizer import PinholeCamera, render_textured_mesh


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a textured OBJ into declared held-out cameras and report silhouette, "
            "linear-RGB appearance, real-edge seam, and UV-layout metrics."
        )
    )
    parser.add_argument("--obj", required=True, type=Path, help="Textured OBJ with MTL map_Kd textures.")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="JSON containing disjoint training_images and held-out views.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1600,
        help="Downscale held-out images and K for deterministic CPU evaluation; 0 keeps native size.",
    )
    parser.add_argument("--near-depth", type=float, default=1.0e-6)
    parser.add_argument("--occupancy-resolution", type=int, default=512)
    parser.add_argument("--cull-backfaces", action="store_true")
    parser.add_argument(
        "--allow-unmasked-diagnostic",
        action="store_true",
        help=(
            "Allow views without a foreground mask for non-scored diagnostics. "
            "Formal evaluation requires one mask per held-out view."
        ),
    )
    parser.add_argument("--save-buffers", action="store_true", help="Also save float depth and face-id NPY files.")
    parser.add_argument("--force", action="store_true", help="Allow replacing evaluator-owned files in output-dir.")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_manifest(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Texture evaluation manifest root must be an object")
    if value.get("schema") != "plascan.texture_eval.v1":
        raise ValueError("Texture evaluation manifest schema must be plascan.texture_eval.v1")
    if not isinstance(value.get("training_images"), list) or not value["training_images"]:
        raise ValueError("Manifest must declare a non-empty training_images list")
    if not isinstance(value.get("views"), list) or not value["views"]:
        raise ValueError("Manifest must declare a non-empty held-out views list")
    return value


def _validate_held_out_split(manifest: dict[str, object], base: Path) -> tuple[list[Path], list[dict[str, object]]]:
    training_paths = [_resolve_path(base, value, "training_images[]") for value in manifest["training_images"]]
    views = manifest["views"]
    if not all(isinstance(view, dict) for view in views):
        raise ValueError("Each held-out view must be an object")
    held_out_paths = [_resolve_path(base, view.get("image"), "views[].image") for view in views]
    missing = [str(path) for path in training_paths + held_out_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Training or held-out images do not exist: {missing}")
    training_set = set(training_paths)
    direct_overlap = training_set.intersection(held_out_paths)
    if direct_overlap:
        raise ValueError(f"Held-out images also occur in training_images: {sorted(map(str, direct_overlap))}")
    training_hashes = {_sha256(path): path for path in training_paths}
    duplicate_content = [
        (training_hashes[digest], path)
        for path in held_out_paths
        if (digest := _sha256(path)) in training_hashes
    ]
    if duplicate_content:
        raise ValueError(
            "Held-out image content duplicates training data: "
            + ", ".join(f"{training} == {held_out}" for training, held_out in duplicate_content)
        )
    training_decoded_hashes = {decoded_rgb_sha256(path): path for path in training_paths}
    decoded_duplicates = [
        (training_decoded_hashes[digest], path)
        for path in held_out_paths
        if (digest := decoded_rgb_sha256(path)) in training_decoded_hashes
    ]
    if decoded_duplicates:
        raise ValueError(
            "Held-out decoded pixels duplicate training data: "
            + ", ".join(f"{training} == {held_out}" for training, held_out in decoded_duplicates)
        )
    return training_paths, views


def _safe_view_id(value: object, index: int, used: set[str]) -> str:
    raw = str(value).strip() if value is not None else f"view_{index:04d}"
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or f"view_{index:04d}"
    if identifier in used:
        raise ValueError(f"Held-out view IDs collide after filename sanitization: {identifier}")
    used.add(identifier)
    return identifier


def _save_rgb(path: Path, linear_rgb: np.ndarray) -> None:
    srgb = linear_to_srgb(linear_rgb)
    Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8), mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _silhouette_metrics(rendered: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    intersection = int(np.count_nonzero(rendered & reference))
    union = int(np.count_nonzero(rendered | reference))
    rendered_count = int(np.count_nonzero(rendered))
    reference_count = int(np.count_nonzero(reference))
    return {
        "intersection_pixel_count": intersection,
        "union_pixel_count": union,
        "rendered_pixel_count": rendered_count,
        "reference_pixel_count": reference_count,
        "iou": intersection / union if union else 1.0,
        "precision": intersection / rendered_count if rendered_count else 0.0,
        "recall": intersection / reference_count if reference_count else 0.0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.max_dimension < 0:
        raise ValueError("--max-dimension must be non-negative")
    if args.near_depth <= 0.0:
        raise ValueError("--near-depth must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    training_paths, views = _validate_held_out_split(manifest, manifest_path.parent)
    mesh = load_textured_obj(args.obj)

    output_dir = args.output_dir.expanduser().resolve()
    report_path = output_dir / "texture_evaluation.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty; pass --force to reuse it: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seam_report = seam_metrics(mesh)
    uv_report = uv_metrics(mesh, args.occupancy_resolution)
    per_view: list[dict[str, object]] = []
    used_ids: set[str] = set()
    total_squared_error = 0.0
    total_samples = 0
    total_intersection = 0
    total_union = 0
    total_rendered_silhouette_pixels = 0
    total_reference_silhouette_pixels = 0
    silhouette_view_count = 0
    unmasked_view_count = 0
    weighted_ssim = 0.0
    weighted_sharpness = 0.0
    hashes: dict[str, str] = {
        str(mesh.source_path): _sha256(mesh.source_path),
        str(manifest_path): _sha256(manifest_path),
    }
    decoded_rgb_hashes: dict[str, str] = {}
    for material in mesh.materials:
        hashes[str(material.texture_path)] = _sha256(material.texture_path)
    for material_library in mesh.material_library_paths:
        hashes[str(material_library)] = _sha256(material_library)
    for path in training_paths:
        hashes[str(path)] = _sha256(path)
        decoded_rgb_hashes[str(path)] = decoded_rgb_sha256(path)

    for index, view in enumerate(views):
        view_id = _safe_view_id(view.get("id"), index, used_ids)
        image_path = _resolve_path(manifest_path.parent, view.get("image"), "views[].image")
        hashes[str(image_path)] = _sha256(image_path)
        loaded_reference = load_linear_reference(image_path, args.max_dimension)
        reference = loaded_reference.linear_rgb
        original_size = loaded_reference.original_size
        scale = loaded_reference.scale
        decoded_rgb_hashes[str(image_path)] = loaded_reference.decoded_rgb_sha256
        height, width = reference.shape[:2]
        camera_value = view.get("camera_model")
        if not isinstance(camera_value, dict):
            raise ValueError(f"View {view_id} is missing camera_model")
        camera = PinholeCamera.from_json(camera_value).scaled(*scale)
        rendered = render_textured_mesh(
            mesh,
            camera,
            width,
            height,
            near_depth=args.near_depth,
            cull_backfaces=args.cull_backfaces,
        )
        rendered_mask = np.isfinite(rendered.depth)
        mask_value = view.get("mask")
        reference_mask: np.ndarray | None = None
        mask_path: Path | None = None
        if mask_value is not None:
            mask_path = _resolve_path(manifest_path.parent, mask_value, "views[].mask")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Held-out mask does not exist: {mask_path}")
            hashes[str(mask_path)] = _sha256(mask_path)
            reference_mask = load_binary_mask(mask_path, original_size, (width, height))
        elif not args.allow_unmasked_diagnostic:
            raise ValueError(
                f"View {view_id} has no foreground mask; formal evaluation requires views[].mask"
            )
        else:
            unmasked_view_count += 1
        evaluation_mask = rendered_mask if reference_mask is None else rendered_mask & reference_mask
        if not np.any(evaluation_mask):
            raise ValueError(f"View {view_id} has no rendered pixels inside its evaluation mask")
        appearance = appearance_metrics(rendered.linear_rgb, reference, evaluation_mask)
        silhouette = _silhouette_metrics(rendered_mask, reference_mask) if reference_mask is not None else None

        render_path = output_dir / f"{view_id}_render.png"
        visibility_path = output_dir / f"{view_id}_visibility.png"
        error_path = output_dir / f"{view_id}_linear_error.png"
        _save_rgb(render_path, rendered.linear_rgb)
        _save_mask(visibility_path, rendered_mask)
        absolute_error = np.abs(rendered.linear_rgb - reference)
        absolute_error[~evaluation_mask] = 0.0
        _save_rgb(error_path, np.clip(absolute_error * 4.0, 0.0, 1.0))
        if args.save_buffers:
            np.save(output_dir / f"{view_id}_depth.npy", rendered.depth, allow_pickle=False)
            np.save(output_dir / f"{view_id}_face_ids.npy", rendered.face_ids, allow_pickle=False)

        pixel_count = int(appearance["valid_pixel_count"])
        total_squared_error += float(appearance["linear_rgb_squared_error_sum"])
        total_samples += int(appearance["linear_rgb_sample_count"])
        weighted_ssim += float(appearance["masked_ssim"]) * pixel_count
        weighted_sharpness += float(appearance["sharpness_ratio"]) * pixel_count
        if silhouette is not None:
            silhouette_view_count += 1
            total_intersection += int(silhouette["intersection_pixel_count"])
            total_union += int(silhouette["union_pixel_count"])
            total_rendered_silhouette_pixels += int(silhouette["rendered_pixel_count"])
            total_reference_silhouette_pixels += int(silhouette["reference_pixel_count"])
        per_view.append(
            {
                "id": view_id,
                "image": str(image_path),
                "mask": str(mask_path) if mask_path else None,
                "original_size": list(original_size),
                "evaluated_size": [width, height],
                "scale": list(scale),
                "render": str(render_path),
                "visibility_mask": str(visibility_path),
                "linear_error_preview": str(error_path),
                "appearance": appearance,
                "silhouette": silhouette,
                "render_diagnostics": {
                    "visible_pixel_count": int(np.count_nonzero(rendered_mask)),
                    "rendered_face_count": rendered.rendered_face_count,
                    "backface_count": rendered.backface_count,
                    "near_clipped_face_count": rendered.near_clipped_face_count,
                },
            }
        )

    aggregate_mse = total_squared_error / total_samples
    aggregate_psnr = 120.0 if aggregate_mse <= 1.0e-12 else float(10.0 * np.log10(1.0 / aggregate_mse))
    total_valid_pixels = sum(int(view["appearance"]["valid_pixel_count"]) for view in per_view)
    return {
        "schema": "plascan.texture_evaluation.v1",
        "status": "ok",
        "inputs": {
            "obj": str(mesh.source_path),
            "manifest": str(manifest_path),
            "training_image_count": len(training_paths),
            "held_out_view_count": len(per_view),
            "sha256": hashes,
            "decoded_rgb_sha256": decoded_rgb_hashes,
        },
        "settings": {
            "max_dimension": args.max_dimension,
            "near_depth": args.near_depth,
            "cull_backfaces": args.cull_backfaces,
            "occupancy_resolution": args.occupancy_resolution,
            "pixel_convention": "top-left pixel center is (0,0)",
            "appearance_color_space": "linear sRGB",
            "reference_resize_space": "linear sRGB",
            "evaluation_mode": (
                "unmasked_diagnostic" if unmasked_view_count else "formal_masked"
            ),
            "unmasked_view_count": unmasked_view_count,
        },
        "mesh": {
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "material_count": len(mesh.materials),
            "seams": seam_report,
            "uv": uv_report,
        },
        "aggregate": {
            "appearance_valid_pixel_count": total_valid_pixels,
            "linear_rgb_mse": aggregate_mse,
            "linear_rgb_psnr_db": aggregate_psnr,
            "masked_ssim_weighted_mean": weighted_ssim / total_valid_pixels,
            "sharpness_ratio_weighted_mean": weighted_sharpness / total_valid_pixels,
            "silhouette_view_count": silhouette_view_count,
            "silhouette_iou": total_intersection / total_union if total_union else None,
            "silhouette_precision": (
                total_intersection / total_rendered_silhouette_pixels
                if total_rendered_silhouette_pixels
                else None
            ),
            "silhouette_recall": (
                total_intersection / total_reference_silhouette_pixels
                if total_reference_silhouette_pixels
                else None
            ),
        },
        "views": per_view,
        "report": str(report_path),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        report = evaluate(args)
        report_path = args.output_dir.expanduser().resolve() / "texture_evaluation.json"
        _write_json(report_path, report)
        aggregate = report["aggregate"]
        print(
            "Texture evaluation passed: "
            f"views={len(report['views'])}, PSNR={aggregate['linear_rgb_psnr_db']:.3f} dB, "
            f"SSIM={aggregate['masked_ssim_weighted_mean']:.5f}, report={report_path}"
        )
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Texture evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
