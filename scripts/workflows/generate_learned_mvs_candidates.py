#!/usr/bin/env python3
"""Run an optional TorchScript MVS model and write geometry-gated candidates.

The TorchScript adapter receives ``(reference, sources, intrinsics, poses)``:
reference is [1,3,H,W], sources is [1,N,3,H,W], intrinsics is
[1,N+1,3,3], and poses is [1,N,4,4] mapping reference-camera coordinates to
each source camera. It must return ``(depth, confidence)`` in reference-camera
positive-Z depth. This script only creates candidates; PlaScan performs the
independent multi-view geometry gate after classical MVS consistency.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import torch


FAST_DEPTH_HEADER = struct.Struct("<16siii4xQ")
FAST_DEPTH_MAGIC = b"PLASDEPTHMAT01\x00\x00"
CV_32FC1 = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvs-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-source-views", type=int, default=6)
    parser.add_argument("--frame", type=int, action="append")
    return parser.parse_args()


def camera_matrix(camera: dict) -> np.ndarray:
    return np.asarray(
        [
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def world_to_camera(camera: dict) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        camera["rotation_world_to_camera"], dtype=np.float64
    ).reshape(3, 3)
    transform[:3, 3] = np.asarray(
        camera["translation_world_to_camera"], dtype=np.float64
    )
    return transform


def load_image(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read MVS image: {path}")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))


def normalize_output(value: torch.Tensor, width: int, height: int) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    array = np.squeeze(array)
    if array.shape != (height, width):
        raise ValueError(
            f"Model output is {array.shape}, expected {(height, width)}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def write_float_grid(path: Path, values: np.ndarray) -> None:
    if values.ndim != 2 or values.dtype != np.float32:
        raise ValueError("PlaScan learned candidate must be a float32 HxW grid")
    height, width = values.shape
    payload_bytes = values.nbytes
    with path.open("wb") as stream:
        stream.write(
            FAST_DEPTH_HEADER.pack(
                FAST_DEPTH_MAGIC,
                height,
                width,
                CV_32FC1,
                payload_bytes,
            )
        )
        stream.write(values.tobytes(order="C"))


def main() -> int:
    args = parse_args()
    manifest_path = args.mvs_manifest.resolve()
    model_path = args.model.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MVS manifest not found: {manifest_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"TorchScript MVS model not found: {model_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but torch.cuda.is_available() is false: "
            f"{args.device}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    frame_by_index = {int(frame["ref_index"]): frame for frame in frames}
    selected_frames = set(args.frame or frame_by_index.keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = torch.jit.load(str(model_path), map_location=device).eval()

    completed = 0
    with torch.inference_mode():
        for frame_index in sorted(selected_frames):
            frame = frame_by_index.get(frame_index)
            if frame is None:
                raise ValueError(f"Frame {frame_index} is absent from manifest")
            width = int(frame["grid_width"])
            height = int(frame["grid_height"])
            source_indices = [
                int(index)
                for index in frame.get("source_indices", [])
                if int(index) in frame_by_index
            ][: max(1, args.maximum_source_views)]
            if not source_indices:
                raise ValueError(f"Frame {frame_index} has no available sources")

            reference = load_image(
                Path(frame["ref_image"]), width, height
            )
            source_images = []
            intrinsics = [camera_matrix(frame["camera_model"])]
            relative_poses = []
            reference_camera_to_world = np.linalg.inv(
                world_to_camera(frame["camera_model"])
            )
            for source_index in source_indices:
                source = frame_by_index[source_index]
                source_images.append(
                    load_image(Path(source["ref_image"]), width, height)
                )
                intrinsics.append(camera_matrix(source["camera_model"]))
                relative_poses.append(
                    world_to_camera(source["camera_model"])
                    @ reference_camera_to_world
                )

            reference_tensor = torch.from_numpy(reference)[None].to(device)
            sources_tensor = torch.from_numpy(
                np.stack(source_images, axis=0)
            )[None].to(device)
            intrinsics_tensor = torch.from_numpy(
                np.stack(intrinsics, axis=0)
            )[None].to(device)
            poses_tensor = torch.from_numpy(
                np.stack(relative_poses, axis=0).astype(np.float32)
            )[None].to(device)
            output = model(
                reference_tensor,
                sources_tensor,
                intrinsics_tensor,
                poses_tensor,
            )
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise TypeError(
                    "TorchScript model must return (depth, confidence)"
                )
            depth = normalize_output(output[0], width, height)
            confidence = np.clip(
                normalize_output(output[1], width, height), 0.0, 1.0
            )
            valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(confidence)
            depth = np.where(valid, depth, 0.0).astype(np.float32)
            confidence = np.where(valid, confidence, 0.0).astype(np.float32)
            write_float_grid(
                args.output_dir / f"learned_depth_{frame_index}.bin", depth
            )
            write_float_grid(
                args.output_dir / f"learned_depth_{frame_index}_conf.bin",
                confidence,
            )
            completed += 1
            print(
                f"frame={frame_index} sources={len(source_indices)} "
                f"valid={int(np.count_nonzero(valid))}/{width * height}"
            )

    print(f"Wrote learned MVS candidates for {completed} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
