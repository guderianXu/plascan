#!/usr/bin/env python3
"""Export PlaScan's SIFT LightGlue core matcher to ONNX and TensorRT.

The generated TensorRT engine keeps FP32 inputs and output for a stable C++ ABI.
With ``--precision fp16``, NVIDIA ModelOpt AutoCast converts only safe internal
operations to FP16 before TensorRT builds a strongly typed engine.

The engine intentionally exports similarity and matchability logits instead of
the final assignment matrix. TensorRT's Myelin compiler can generate an invalid
dynamic LogSoftmax kernel for this model on Windows/Blackwell. PlaScan computes
the mathematically equivalent double-softmax assignment in its own postprocessor.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_provenance import (
    git_source_revision,
    installed_tool_versions,
    invalidate_provenance,
    provenance_path,
    require_provenance,
    sha256_file,
    source_file_records,
    validate_provenance,
    write_json_atomic,
    write_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "build" / "model_cache" / "lightglue_tensorrt"
OFFICIAL_WEIGHTS_ID = "sift_lightglue:v0.1_arxiv"
OFFICIAL_WEIGHTS_RELEASE = "v0.1_arxiv"
OFFICIAL_WEIGHTS_NAME = "sift_lightglue_v0-1_arxiv.pth"
OFFICIAL_WEIGHTS_URL = (
    "https://github.com/cvg/LightGlue/releases/download/"
    f"{OFFICIAL_WEIGHTS_RELEASE}/sift_lightglue.pth"
)
EXPORTER_SCHEMA_VERSION = 1


def add_lightglue_to_path() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("LIGHTGLUE_REPO", "").strip()
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            ROOT / "3rdparty" / "LightGlue-main",
            ROOT / "3rdparty" / "LightGlue",
            ROOT / "third_party" / "LightGlue-main",
            ROOT / "third_party" / "LightGlue",
        ]
    )
    for candidate in candidates:
        if (candidate / "lightglue" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return candidate.resolve()
    return None


LIGHTGLUE_SOURCE_ROOT = add_lightglue_to_path()
from lightglue import LightGlue  # noqa: E402


class SiftLightGlueScoreModel(torch.nn.Module):
    def __init__(self, weights_path: Path | None):
        super().__init__()
        common = {
            "flash": False,
            "mp": False,
            "depth_confidence": -1,
            "width_confidence": -1,
        }
        if weights_path is None:
            self.model = LightGlue(features="sift", **common).eval()
            return

        self.model = LightGlue(
            features=None,
            input_dim=128,
            add_scale_ori=True,
            weights=None,
            **common,
        ).eval()
        raw_state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        if isinstance(raw_state, dict) and "state_dict" in raw_state:
            raw_state = raw_state["state_dict"]
        if not isinstance(raw_state, dict):
            raise RuntimeError(f"Unsupported LightGlue weights file: {weights_path}")
        state = {
            name.removeprefix("model."): value.detach().cpu()
            for name, value in raw_state.items()
        }
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Weights do not match the SIFT LightGlue module: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    @staticmethod
    def normalize_keypoints(keypoints: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
        shift = image_size[:, None, :] * 0.5
        max_side = torch.maximum(image_size[:, 0:1], image_size[:, 1:2])
        scale = max_side[:, None, :] * 0.5
        return (keypoints - shift) / scale

    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        image_size0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size1: torch.Tensor,
        valid0: torch.Tensor,
        valid1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy0 = self.normalize_keypoints(keypoints0[..., :2], image_size0)
        xy1 = self.normalize_keypoints(keypoints1[..., :2], image_size1)
        geometry0 = torch.cat([xy0, keypoints0[..., 2:4]], dim=-1)
        geometry1 = torch.cat([xy1, keypoints1[..., 2:4]], dim=-1)

        descriptor0 = self.model.input_proj(descriptors0.contiguous())
        descriptor1 = self.model.input_proj(descriptors1.contiguous())
        encoding0 = self.model.posenc(geometry0)
        encoding1 = self.model.posenc(geometry1)
        mask0 = valid0.unsqueeze(-1)
        mask1 = valid1.unsqueeze(-1)
        for layer in self.model.transformers:
            descriptor0, descriptor1 = layer(
                descriptor0,
                descriptor1,
                encoding0,
                encoding1,
                mask0=mask0,
                mask1=mask1,
            )
        assignment = self.model.log_assignment[-1]
        projected0 = assignment.final_proj(descriptor0)
        projected1 = assignment.final_proj(descriptor1)
        descriptor_dimension = projected0.shape[-1]
        scale = float(descriptor_dimension) ** 0.25
        similarity = torch.einsum(
            "bmd,bnd->bmn", projected0 / scale, projected1 / scale
        )
        matchability0 = assignment.matchability(descriptor0).squeeze(-1)
        matchability1 = assignment.matchability(descriptor1).squeeze(-1)
        return similarity, matchability0, matchability1


def exporter_tool_versions() -> dict[str, str]:
    return installed_tool_versions(
        {
            "lightglue": "lightglue",
            "modelopt": "nvidia-modelopt",
            "numpy": "numpy",
            "onnx": "onnx",
            "onnxscript": "onnxscript",
            "tensorrt": "tensorrt",
            "torch": "torch",
        }
    )


def resolve_weights_source(weights_path: Path | None) -> list[dict[str, Any]]:
    if weights_path is not None:
        return source_file_records({weights_path.name: weights_path})

    torch.hub.load_state_dict_from_url(
        OFFICIAL_WEIGHTS_URL,
        map_location="cpu",
        file_name=OFFICIAL_WEIGHTS_NAME,
    )
    cached = Path(torch.hub.get_dir()) / "checkpoints" / OFFICIAL_WEIGHTS_NAME
    if not cached.is_file():
        raise FileNotFoundError(
            "Official LightGlue weights were loaded but the cached checkpoint "
            f"cannot be found: {cached}"
        )
    records = source_file_records({OFFICIAL_WEIGHTS_NAME: cached})
    records[0]["source_id"] = OFFICIAL_WEIGHTS_ID
    records[0]["source_url"] = OFFICIAL_WEIGHTS_URL
    return records


def lightglue_source_revision() -> dict[str, Any]:
    implementation_file = inspect.getsourcefile(LightGlue)
    implementation = (
        git_source_revision(LIGHTGLUE_SOURCE_ROOT)
        if LIGHTGLUE_SOURCE_ROOT is not None
        else {"commit": "installed-package", "dirty": None, "dirty_diff_sha256": None}
    )
    if implementation_file is not None and Path(implementation_file).is_file():
        implementation = {
            **implementation,
            "source_file": Path(implementation_file).name,
            "source_file_sha256": sha256_file(Path(implementation_file)),
        }
    return {
        "model_release": OFFICIAL_WEIGHTS_RELEASE,
        "implementation": implementation,
    }


def base_onnx_contract(
    weights: list[dict[str, Any]], bucket_keypoints: int
) -> dict[str, Any]:
    dynamic = bucket_keypoints <= 0
    return {
        "artifact_kind": "lightglue_sift_base_onnx",
        "exporter": {
            "name": Path(__file__).name,
            "schema_version": EXPORTER_SCHEMA_VERSION,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "source": {
            "repository": "cvg/LightGlue",
            "revision": lightglue_source_revision(),
            "weights": weights,
        },
        "model": {
            "id": "sift_lightglue",
            "configuration": {
                "descriptor_dimension": 128,
                "add_scale_orientation": True,
                "depth_confidence": -1,
                "width_confidence": -1,
                "flash_attention": False,
            },
        },
        "input": {
            "keypoints": "[1,K,4] float32",
            "descriptors": "[1,K,128] float32",
            "image_size": "[1,2] float32",
            "valid": "[1,K] bool",
        },
        "profile": {
            "dynamic_keypoints": dynamic,
            "bucket_keypoints": bucket_keypoints,
            "minimum": 1 if dynamic else bucket_keypoints,
            "maximum": 65536 if dynamic else bucket_keypoints,
        },
        "opset": 20,
        "precision": "fp32",
        "tools": exporter_tool_versions(),
    }


def fp16_onnx_contract(
    base_path: Path,
    base_contract: dict[str, Any],
    calibration_count: int,
    bucket_keypoints: int,
) -> dict[str, Any]:
    return {
        **base_contract,
        "artifact_kind": "lightglue_sift_fp16_onnx",
        "precision": "fp16",
        "conversion": {
            "source_onnx": {
                "file": base_path.name,
                "sha256": sha256_file(base_path),
            },
            "modelopt_autocast": {
                "calibration_keypoints": calibration_count,
                "calibration_pair_keypoints": (
                    bucket_keypoints if bucket_keypoints > 0 else calibration_count + 7
                ),
                "keep_io_types": True,
                "low_precision_type": "fp16",
                "providers": ["cpu"],
            },
        },
    }


def sample_inputs(count0: int, count1: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    keypoints0 = torch.rand((1, count0, 4), generator=generator)
    keypoints1 = torch.rand((1, count1, 4), generator=generator)
    image_size0 = torch.tensor([[640.0, 480.0]], dtype=torch.float32)
    image_size1 = torch.tensor([[768.0, 512.0]], dtype=torch.float32)
    keypoints0[..., :2] *= image_size0[:, None, :]
    keypoints1[..., :2] *= image_size1[:, None, :]
    keypoints0[..., 2] = 1.0 + keypoints0[..., 2] * 15.0
    keypoints1[..., 2] = 1.0 + keypoints1[..., 2] * 15.0
    keypoints0[..., 3] = (keypoints0[..., 3] * 2.0 - 1.0) * torch.pi
    keypoints1[..., 3] = (keypoints1[..., 3] * 2.0 - 1.0) * torch.pi
    descriptors0 = torch.rand((1, count0, 128), generator=generator)
    descriptors1 = torch.rand((1, count1, 128), generator=generator)
    valid0 = torch.ones((1, count0), dtype=torch.bool)
    valid1 = torch.ones((1, count1), dtype=torch.bool)
    return (
        keypoints0,
        descriptors0,
        image_size0,
        keypoints1,
        descriptors1,
        image_size1,
        valid0,
        valid1,
    )


def export_onnx(
    weights_path: Path | None,
    output_path: Path,
    bucket_keypoints: int,
    contract: dict[str, Any],
    force: bool,
) -> None:
    validation = validate_provenance(output_path, contract)
    if validation.valid and not force:
        print(f"reuse provenance-validated ONNX: {output_path}")
        return
    if output_path.exists() and not force:
        print(f"re-export ONNX because cached provenance is invalid: {validation.reason}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    invalidate_provenance(output_path)
    output_path.unlink(missing_ok=True)
    model = SiftLightGlueScoreModel(weights_path).eval()
    sample_count0 = bucket_keypoints if bucket_keypoints > 0 else 32
    sample_count1 = bucket_keypoints if bucket_keypoints > 0 else 40
    inputs = sample_inputs(sample_count0, sample_count1)
    dynamic_shapes = None
    if bucket_keypoints <= 0:
        keypoint_count0 = torch.export.Dim("num_keypoints0", min=1, max=65536)
        keypoint_count1 = torch.export.Dim("num_keypoints1", min=1, max=65536)
        dynamic_shapes = (
            {1: keypoint_count0},
            {1: keypoint_count0},
            None,
            {1: keypoint_count1},
            {1: keypoint_count1},
            None,
            {1: keypoint_count0},
            {1: keypoint_count1},
        )

    started = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(output_path),
            input_names=[
                "keypoints0",
                "descriptors0",
                "image_size0",
                "keypoints1",
                "descriptors1",
                "image_size1",
                "valid0",
                "valid1",
            ],
            output_names=["similarity", "matchability0", "matchability1"],
            dynamic_shapes=dynamic_shapes,
            opset_version=20,
            dynamo=True,
            external_data=False,
            optimize=False,
        )
    write_provenance(output_path, contract)
    print(f"exported ONNX in {time.perf_counter() - started:.2f}s: {output_path}")


def write_calibration_data(path: Path, count0: int, count1: int) -> None:
    inputs = sample_inputs(count0, count1)
    names = [
        "keypoints0",
        "descriptors0",
        "image_size0",
        "keypoints1",
        "descriptors1",
        "image_size1",
        "valid0",
        "valid1",
    ]
    np.savez(path, **{name: value.numpy() for name, value in zip(names, inputs)})


def convert_fp16(
    input_path: Path,
    output_path: Path,
    calibration_count: int,
    bucket_keypoints: int,
    contract: dict[str, Any],
    force: bool,
) -> None:
    validation = validate_provenance(output_path, contract)
    if validation.valid and not force:
        print(f"reuse provenance-validated mixed-precision ONNX: {output_path}")
        return
    if output_path.exists() and not force:
        print(
            "re-convert mixed-precision ONNX because cached provenance is invalid: "
            f"{validation.reason}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    invalidate_provenance(output_path)
    output_path.unlink(missing_ok=True)
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="plascan_lightglue_", dir=DEFAULT_CACHE_DIR
    ) as temporary_directory:
        calibration_path = Path(temporary_directory) / "calibration.npz"
        calibration_count1 = (
            bucket_keypoints if bucket_keypoints > 0 else calibration_count + 7
        )
        calibration_count0 = (
            bucket_keypoints if bucket_keypoints > 0 else calibration_count
        )
        write_calibration_data(
            calibration_path, calibration_count0, calibration_count1
        )
        command = [
            sys.executable,
            "-m",
            "modelopt.onnx.autocast",
            "--onnx_path",
            str(input_path),
            "--output_path",
            str(output_path),
            "--low_precision_type",
            "fp16",
            "--calibration_data",
            str(calibration_path),
            "--providers",
            "cpu",
            "--keep_io_types",
            "--log_level",
            "INFO",
        ]
        started = time.perf_counter()
        subprocess.run(command, check=True)
        print(
            "converted strongly typed FP16 ONNX in "
            f"{time.perf_counter() - started:.2f}s: {output_path}"
        )
    write_provenance(output_path, contract)


def tensor_rt_environment() -> dict[str, Any]:
    import tensorrt as trt

    gpu_name = "unknown"
    compute_capability = "unknown"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = f"{major}.{minor}"
    return {
        "tensorrt_version": trt.__version__,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
    }


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    minimum: int,
    optimum: int,
    maximum: int,
    workspace_gib: float,
    builder_optimization_level: int,
    maximum_auxiliary_streams: int,
) -> float:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 1024 * 1024 * 1024)
    )
    config.builder_optimization_level = builder_optimization_level
    config.max_aux_streams = maximum_auxiliary_streams
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

    has_dynamic_input = any(
        -1 in tuple(network.get_input(index).shape)
        for index in range(network.num_inputs)
    )
    if has_dynamic_input:
        profile = builder.create_optimization_profile()
        shapes = {
            "keypoints0": ((1, minimum, 4), (1, optimum, 4), (1, maximum, 4)),
            "descriptors0": (
                (1, minimum, 128),
                (1, optimum, 128),
                (1, maximum, 128),
            ),
            "image_size0": ((1, 2), (1, 2), (1, 2)),
            "keypoints1": ((1, minimum, 4), (1, optimum, 4), (1, maximum, 4)),
            "descriptors1": (
                (1, minimum, 128),
                (1, optimum, 128),
                (1, maximum, 128),
            ),
            "image_size1": ((1, 2), (1, 2), (1, 2)),
            "valid0": ((1, minimum), (1, optimum), (1, maximum)),
            "valid1": ((1, minimum), (1, optimum), (1, maximum)),
        }
        for name, (minimum_shape, optimum_shape, maximum_shape) in shapes.items():
            profile_result = profile.set_shape(
                name, minimum_shape, optimum_shape, maximum_shape
            )
            if profile_result is False:
                raise RuntimeError(
                    f"Cannot set TensorRT optimization profile for {name}"
                )
        profile_index = config.add_optimization_profile(profile)
        if profile_index is not None and profile_index < 0:
            raise RuntimeError("Cannot add TensorRT optimization profile")

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    build_seconds = time.perf_counter() - started
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return build_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the PlaScan SIFT LightGlue TensorRT engine"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Optional SIFT LightGlue .pth weights. When omitted, the official "
            "weights are downloaded through torch.hub and cached locally."
        ),
    )
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="FP32 is accuracy-preserving; FP16 is experimental and may change matches",
    )
    parser.add_argument("--min-keypoints", type=int, default=1)
    parser.add_argument("--opt-keypoints", type=int, default=1024)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--calibration-keypoints", type=int, default=64)
    parser.add_argument(
        "--bucket-keypoints",
        type=int,
        default=1024,
        help="Fixed padded keypoint capacity; use 0 only for dynamic-shape diagnostics",
    )
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument(
        "--builder-optimization-level", type=int, choices=range(0, 6), default=3
    )
    parser.add_argument("--max-aux-streams", type=int, default=0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = args.weights.resolve() if args.weights is not None else None
    if weights_path is not None and not weights_path.is_file():
        raise FileNotFoundError(f"LightGlue weights not found: {weights_path}")
    if not (1 <= args.min_keypoints <= args.opt_keypoints <= args.max_keypoints):
        raise ValueError("Expected 1 <= min-keypoints <= opt-keypoints <= max-keypoints")
    if args.bucket_keypoints < 0:
        raise ValueError("bucket-keypoints must be non-negative")

    weights_source = resolve_weights_source(weights_path)
    base_contract = base_onnx_contract(weights_source, args.bucket_keypoints)

    shape_tag = (
        f"bucket{args.bucket_keypoints}"
        if args.bucket_keypoints > 0
        else "dynamic"
    )
    onnx_path = args.onnx
    if onnx_path is None:
        onnx_path = DEFAULT_CACHE_DIR / f"lightglue_sift_{shape_tag}.onnx"
    onnx_path = onnx_path.resolve()

    engine_path = args.engine
    if engine_path is None:
        engine_path = (
            DEFAULT_CACHE_DIR /
            f"lightglue_sift_{shape_tag}_{args.precision}.engine"
        )
    engine_path = engine_path.resolve()

    if not args.skip_export:
        export_onnx(
            weights_path,
            onnx_path,
            args.bucket_keypoints,
            base_contract,
            args.force,
        )
    else:
        require_provenance(onnx_path, base_contract)

    build_onnx_path = onnx_path
    if args.precision == "fp16":
        build_onnx_path = onnx_path.with_name(f"{onnx_path.stem}_fp16.onnx")
        converted_contract = fp16_onnx_contract(
            onnx_path,
            base_contract,
            args.calibration_keypoints,
            args.bucket_keypoints,
        )
        convert_fp16(
            onnx_path,
            build_onnx_path,
            args.calibration_keypoints,
            args.bucket_keypoints,
            converted_contract,
            args.force,
        )

    if args.skip_build:
        return 0

    environment = tensor_rt_environment()
    cache_key = {
        "schema": 3,
        "weights_source": weights_source,
        "onnx_sha256": sha256_file(build_onnx_path),
        "onnx_provenance_sha256": sha256_file(provenance_path(build_onnx_path)),
        "precision": args.precision,
        "bucket_keypoints": args.bucket_keypoints,
        "profile": {
            "minimum": args.min_keypoints,
            "optimum": args.opt_keypoints,
            "maximum": args.max_keypoints,
        },
        "workspace_bytes": int(args.workspace_gib * 1024 * 1024 * 1024),
        "builder_optimization_level": args.builder_optimization_level,
        "max_aux_streams": args.max_aux_streams,
        **environment,
    }
    metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
    if engine_path.is_file() and metadata_path.is_file() and not args.force:
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        contract_matches = all(
            existing.get(key) == value for key, value in cache_key.items()
        )
        artifact_matches = (
            existing.get("engine_sha256") == sha256_file(engine_path)
            and existing.get("engine_bytes") == engine_path.stat().st_size
        )
        if contract_matches and artifact_matches:
            print(f"reuse TensorRT engine: {engine_path}")
            return 0
        print("rebuild TensorRT engine because metadata or artifact hash is stale")

    build_seconds = build_engine(
        build_onnx_path,
        engine_path,
        args.min_keypoints,
        args.opt_keypoints,
        args.max_keypoints,
        args.workspace_gib,
        args.builder_optimization_level,
        args.max_aux_streams,
    )
    metadata = {
        **cache_key,
        "engine_sha256": sha256_file(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "build_seconds": build_seconds,
    }
    write_json_atomic(metadata_path, metadata)
    print(
        f"built TensorRT {args.precision} engine in {build_seconds:.2f}s: "
        f"{engine_path} ({engine_path.stat().st_size / (1024 * 1024):.1f} MiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
