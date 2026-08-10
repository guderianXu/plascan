#!/usr/bin/env python3
"""Export the official LoMa-R pipeline to PlaScan portable model artifacts.

The source architecture is loaded from a local LoMa-R checkout.  Four official
checkpoints are required locally and this script never downloads weights.  The
portable result contains one shared K3840 DaD + DeDoDe-G feature ONNX, one
dynamic LoMa-R matcher ONNX, and three validated K-bucket manifests. Optional
local TensorRT engines are diagnostic artifacts and are never package inputs.

LoMa-R source: https://github.com/davnords/loma (MIT; matcher derives from
LightGlue under Apache-2.0).  Checkpoint redistribution is intentionally left
to the model authors; PlaScan stores only generated local artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import MethodType
from urllib.parse import urlparse

import torch
import torch.nn.functional as functional

from compose_loma_r_package import (
    FEATURE_KEYPOINT_COUNT,
    PACKAGE_KEYPOINT_BUCKETS,
    compose_package,
    require_matcher_only_feature,
)
from model_provenance import (
    git_source_revision,
    installed_tool_versions,
    invalidate_provenance,
    provenance_path,
    require_provenance,
    sha256_file,
    source_file_records,
    validate_provenance,
    write_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "build" / "model_cache" / "loma_r_tensorrt"
REQUIRED_WEIGHTS = (
    "loma_R.pth",
    "dad.pth",
    "dedode_descriptor_G.pth",
    "dinov2_vitl14_pretrain.pth",
)
FEATURE_WEIGHTS = (
    "dad.pth",
    "dedode_descriptor_G.pth",
    "dinov2_vitl14_pretrain.pth",
)
MATCHER_WEIGHTS = ("loma_R.pth",)
MATCHER_EXPORT_SAMPLE_KEYPOINTS = 1024
EXPORTER_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the validated portable DaD + DeDoDe-G + LoMa-R package"
    )
    parser.add_argument("--loma-repo", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-size", type=int, default=784)
    parser.add_argument(
        "--keypoints",
        type=int,
        default=FEATURE_KEYPOINT_COUNT,
        help=(
            "Shared feature capacity. The portable package contract requires 3840; "
            "the option is retained only for explicit compatibility checks."
        ),
    )
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument(
        "--matcher-only",
        action="store_true",
        help="validate the existing feature ONNX and export/reuse only the matcher",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def exporter_tool_versions() -> dict[str, str]:
    return installed_tool_versions(
        {
            "einops": "einops",
            "modelopt": "nvidia-modelopt",
            "onnx": "onnx",
            "onnxscript": "onnxscript",
            "tensorrt": "tensorrt",
            "torch": "torch",
        }
    )


def checkpoint_records(weights_dir: Path, names: tuple[str, ...]) -> list[dict[str, object]]:
    return source_file_records({name: weights_dir / name for name in names})


def common_contract(
    repo: Path,
    weights_dir: Path,
    checkpoint_names: tuple[str, ...],
    precision: str,
) -> dict[str, object]:
    return {
        "exporter": {
            "name": Path(__file__).name,
            "schema_version": EXPORTER_SCHEMA_VERSION,
            "script_sha256": sha256_file(Path(__file__)),
            "external_data": True,
            "dynamo": False,
        },
        "source": {
            "repository": "davnords/loma",
            "revision": git_source_revision(repo),
            "checkpoints": checkpoint_records(weights_dir, checkpoint_names),
        },
        "opset": 18,
        "precision": precision,
        "onnx_io_precision": "fp32",
        "tools": exporter_tool_versions(),
    }


def feature_onnx_contract(
    repo: Path,
    weights_dir: Path,
    input_size: int,
    precision: str,
) -> dict[str, object]:
    return {
        **common_contract(repo, weights_dir, FEATURE_WEIGHTS, precision),
        "artifact_kind": "loma_r_feature_onnx",
        "model": {
            "id": "loma_r_feature",
            "configuration": {
                "detector": "DaD",
                "descriptor": "DeDoDe-G/DINOv2-ViT-L14",
                "descriptor_dimension": 256,
            },
        },
        "input": {
            "layout": "NCHW",
            "shape": [1, 3, input_size, input_size],
            "width": input_size,
            "height": input_size,
            "dtype": "float32",
        },
        "profile": {
            "feature_keypoint_count": FEATURE_KEYPOINT_COUNT,
            "dynamic_input": False,
        },
    }


def matcher_onnx_contract(
    repo: Path,
    weights_dir: Path,
    precision: str,
) -> dict[str, object]:
    return {
        **common_contract(repo, weights_dir, MATCHER_WEIGHTS, precision),
        "artifact_kind": "loma_r_matcher_onnx",
        "model": {
            "id": "loma_r_matcher",
            "configuration": {
                "descriptor_dimension": 256,
                "rotation_invariant": True,
            },
        },
        "input": {
            "keypoints": "[1,K,2] float32",
            "descriptors": "[1,K,256] float32",
            "valid": "[1,K] bool",
        },
        "profile": {
            "keypoints": {
                "dynamic": True,
                "minimum": 1,
                "optimum": 2048,
                "maximum": FEATURE_KEYPOINT_COUNT,
                "package_buckets": list(PACKAGE_KEYPOINT_BUCKETS),
                "export_sample": MATCHER_EXPORT_SAMPLE_KEYPOINTS,
            }
        },
    }


def load_model(repo: Path, weights_dir: Path) -> torch.nn.Module:
    source_dir = repo / "src"
    if not (source_dir / "loma" / "loma.py").is_file():
        raise FileNotFoundError(f"LoMa-R source tree not found: {source_dir}")
    missing = [name for name in REQUIRED_WEIGHTS if not (weights_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing LoMa-R checkpoints in {weights_dir}: {', '.join(missing)}"
        )
    sys.path.insert(0, str(source_dir))
    from loma import LoMa, LoMaR  # pylint: disable=import-outside-toplevel

    original_loader = torch.hub.load_state_dict_from_url

    def local_loader(url: str, *_args: object, **_kwargs: object) -> object:
        path = weights_dir / Path(urlparse(url).path).name
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint is not available locally: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    torch.hub.load_state_dict_from_url = local_loader  # type: ignore[assignment]
    try:
        # Disable the matcher's Python autocast context in the exported graph.
        # TensorRT applies FP16 at engine-build time while preserving the fixed
        # float32 ABI consumed by the C++ runtime.
        model = LoMa(LoMaR(mp=False)).eval().cuda()
    finally:
        torch.hub.load_state_dict_from_url = original_loader  # type: ignore[assignment]

    # Export a deterministic graph. TensorRT chooses FP16 kernels during build;
    # Python autocast inside the source model must not bake mixed dtypes into ABI.
    for module in model.modules():
        if hasattr(module, "amp"):
            module.amp = False
        if hasattr(module, "amp_dtype"):
            module.amp_dtype = torch.float32
    return model.float().eval()


def make_dinov2_position_encoding_exportable(
    model: torch.nn.Module, input_size: int
) -> None:
    """Replace tensor-derived interpolation scales with fixed export dimensions."""
    dinov2 = model._descriptor.encoder.frozen_dinov2.dinov2_vitl14  # noqa: SLF001
    target_size = input_size // int(dinov2.patch_size)

    def interpolate_pos_encoding(
        module: torch.nn.Module,
        tokens: torch.Tensor,
        _width: int,
        _height: int,
    ) -> torch.Tensor:
        previous_dtype = tokens.dtype
        position = module.pos_embed.float()
        class_position = position[:, :1]
        patch_position = position[:, 1:]
        source_size = int(patch_position.shape[1] ** 0.5)
        dimension = patch_position.shape[-1]
        patch_position = functional.interpolate(
            patch_position.reshape(1, source_size, source_size, dimension)
            .permute(0, 3, 1, 2),
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
        )
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(
            1, target_size * target_size, dimension
        )
        return torch.cat((class_position, patch_position), dim=1).to(previous_dtype)

    dinov2.interpolate_pos_encoding = MethodType(interpolate_pos_encoding, dinov2)


class FeatureWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, keypoints: int):
        super().__init__()
        self.detector = model._detector  # noqa: SLF001 - official model composition
        self.descriptor = model._descriptor  # noqa: SLF001
        self.keypoints = keypoints

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        logits = self.detector.forward_impl(image)[:, 0]
        height, width = logits.shape[-2:]
        probabilities = logits.flatten(1).softmax(dim=1).reshape_as(logits)
        local_maximum = functional.max_pool2d(
            logits[:, None], kernel_size=3, stride=1, padding=1
        )[:, 0]
        nms_probabilities = torch.where(
            logits == local_maximum,
            probabilities,
            torch.zeros_like(probabilities),
        )
        keypoint_probs, indices = torch.topk(
            nms_probabilities.flatten(1), self.keypoints, dim=1
        )
        x_coord = torch.remainder(indices, width).float() + 0.5
        y_coord = torch.div(indices, width, rounding_mode="floor").float() + 0.5
        keypoints = torch.stack(
            (
                2.0 * x_coord / float(width) - 1.0,
                2.0 * y_coord / float(height) - 1.0,
            ),
            dim=2,
        )
        description_grid = self.descriptor(image).float()
        descriptions = functional.grid_sample(
            description_grid,
            keypoints[:, None],
            mode="bilinear",
            align_corners=False,
        )[:, :, 0].transpose(1, 2)
        return keypoints.float(), keypoint_probs.float(), descriptions.float()


class MatcherWrapper(torch.nn.Module):
    """TensorRT-friendly LoMa-R forward with dynamic keypoint count."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.heads = int(model.cfg.num_heads)
        self.dimension = int(model.cfg.embed_dim)
        self.head_dimension = self.dimension // self.heads
        self.attention_scale = self.head_dimension**-0.5

    def _rotate_half(self, value: torch.Tensor) -> torch.Tensor:
        paired = value.reshape(
            1, self.heads, -1, self.head_dimension // 2, 2
        )
        return torch.stack((-paired[..., 1], paired[..., 0]), dim=4).reshape(
            1, self.heads, -1, self.head_dimension
        )

    def _apply_rotary(
        self, encoding: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return value * encoding[0] + self._rotate_half(value) * encoding[1]

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_valid: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.matmul(query, key.permute(0, 1, 3, 2)) * self.attention_scale
        logits = torch.where(
            key_valid[:, None, None, :],
            logits,
            torch.full_like(logits, -10000.0),
        )
        context = torch.matmul(logits.softmax(dim=3), value)
        return context * query_valid[:, None, :, None].float()

    def _self_attention(
        self,
        block: torch.nn.Module,
        descriptors: torch.Tensor,
        encoding: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        qkv = block.Wqkv(descriptors).reshape(
            1, -1, self.heads, self.head_dimension, 3
        ).permute(0, 2, 1, 3, 4)
        query = self._apply_rotary(encoding, qkv[..., 0])
        key = self._apply_rotary(encoding, qkv[..., 1])
        context = self._attention(query, key, qkv[..., 2], valid, valid)
        message = block.out_proj(
            context.permute(0, 2, 1, 3).reshape(
                1, -1, self.dimension
            )
        )
        return descriptors + block.ffn(torch.cat((descriptors, message), dim=2))

    def _cross_attention(
        self,
        block: torch.nn.Module,
        descriptors0: torch.Tensor,
        descriptors1: torch.Tensor,
        valid0: torch.Tensor,
        valid1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                1, -1, self.heads, self.head_dimension
            ).permute(0, 2, 1, 3)

        query_key0 = split_heads(block.to_qk(descriptors0))
        query_key1 = split_heads(block.to_qk(descriptors1))
        value0 = split_heads(block.to_v(descriptors0))
        value1 = split_heads(block.to_v(descriptors1))
        message0 = self._attention(query_key0, query_key1, value1, valid0, valid1)
        message1 = self._attention(query_key1, query_key0, value0, valid1, valid0)

        def merge_heads(value: torch.Tensor) -> torch.Tensor:
            return value.permute(0, 2, 1, 3).reshape(
                1, -1, self.dimension
            )

        message0 = block.to_out(merge_heads(message0))
        message1 = block.to_out(merge_heads(message1))
        output0 = descriptors0 + block.ffn(
            torch.cat((descriptors0, message0), dim=2)
        )
        output1 = descriptors1 + block.ffn(
            torch.cat((descriptors1, message1), dim=2)
        )
        return output0, output1

    def forward(
        self,
        keypoints0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors0: torch.Tensor,
        descriptors1: torch.Tensor,
        valid0: torch.Tensor,
        valid1: torch.Tensor,
    ) -> torch.Tensor:
        descriptors0 = self.model.input_proj(descriptors0)
        descriptors1 = self.model.input_proj(descriptors1)
        encoding0 = self.model.posenc(keypoints0)
        encoding1 = self.model.posenc(keypoints1)
        for layer in self.model.transformers:
            descriptors0 = self._self_attention(
                layer.self_attn, descriptors0, encoding0, valid0
            )
            descriptors1 = self._self_attention(
                layer.self_attn, descriptors1, encoding1, valid1
            )
            descriptors0, descriptors1 = self._cross_attention(
                layer.cross_attn, descriptors0, descriptors1, valid0, valid1
            )
        assignment = self.model.log_assignment[-1]
        matched0 = assignment.final_proj(descriptors0) / self.dimension**0.25
        matched1 = assignment.final_proj(descriptors1) / self.dimension**0.25
        similarity = torch.matmul(matched0, matched1.permute(0, 2, 1))
        scores = similarity.softmax(dim=2) * similarity.softmax(dim=1)
        valid = valid0[:, :, None] & valid1[:, None, :]
        return torch.where(valid, scores.float(), torch.zeros_like(scores, dtype=torch.float32))


def export_onnx(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    contract: dict[str, object],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    invalidate_provenance(path)
    path.unlink(missing_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        external_data=True,
    )
    write_provenance(path, contract)


def needs_export(
    path: Path,
    contract: dict[str, object],
    force: bool,
) -> bool:
    validation = validate_provenance(path, contract)
    if validation.valid and not force:
        print(f"reuse provenance-validated ONNX: {path}")
        return False
    if path.exists() and not force:
        print(f"re-export {path.name} because cached provenance is invalid: {validation.reason}")
    return True


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace: int,
    fixed_keypoints: int = 0,
) -> None:
    import tensorrt as trt  # pylint: disable=import-outside-toplevel

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("The selected GPU does not provide fast FP16 TensorRT kernels")
        config.set_flag(trt.BuilderFlag.FP16)
    if fixed_keypoints > 0:
        profile = builder.create_optimization_profile()
        shapes = {
            "keypoints0": (1, fixed_keypoints, 2),
            "keypoints1": (1, fixed_keypoints, 2),
            "descriptors0": (1, fixed_keypoints, 256),
            "descriptors1": (1, fixed_keypoints, 256),
            "valid0": (1, fixed_keypoints),
            "valid1": (1, fixed_keypoints),
        }
        for name, shape in shapes.items():
            if profile.set_shape(name, shape, shape, shape) is False:
                raise RuntimeError(f"Cannot set TensorRT optimization profile for {name}")
        profile_index = config.add_optimization_profile(profile)
        if profile_index is not None and profile_index < 0:
            raise RuntimeError("Cannot add the LoMa-R TensorRT optimization profile")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_path.name}")
    engine_path.write_bytes(bytes(serialized))


def build_engine_with_provenance(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace: int,
    fixed_keypoints: int,
    force: bool,
) -> None:
    onnx_document = require_provenance(onnx_path)
    contract = {
        "artifact_kind": "loma_r_tensorrt_engine",
        "source_onnx": onnx_document["artifact"],
        "source_provenance_sha256": sha256_file(provenance_path(onnx_path)),
        "precision": precision,
        "profile": {"fixed_keypoints": fixed_keypoints},
        "workspace_bytes": workspace,
        "gpu": {
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability()
            ),
        },
        "tools": exporter_tool_versions(),
    }
    validation = validate_provenance(engine_path, contract)
    if validation.valid and not force:
        print(f"reuse provenance-validated TensorRT engine: {engine_path}")
        return
    if engine_path.exists() and not force:
        print(f"rebuild {engine_path.name}: {validation.reason}")
    invalidate_provenance(engine_path)
    engine_path.unlink(missing_ok=True)
    build_engine(
        onnx_path,
        engine_path,
        precision,
        workspace,
        fixed_keypoints=fixed_keypoints,
    )
    write_provenance(engine_path, contract)


def main() -> None:
    args = parse_args()
    if args.input_size <= 0 or args.input_size % 14 != 0:
        raise ValueError("--input-size must be positive and divisible by DINOv2 patch size 14")
    if args.keypoints != FEATURE_KEYPOINT_COUNT:
        raise ValueError(
            f"--keypoints must be {FEATURE_KEYPOINT_COUNT}; the portable package "
            "uses one shared K3840 feature ONNX and three matcher profiles"
        )

    repo = args.loma_repo.resolve()
    weights_dir = args.weights_dir.resolve()
    source_dir = repo / "src"
    if not (source_dir / "loma" / "loma.py").is_file():
        raise FileNotFoundError(f"LoMa-R source tree not found: {source_dir}")
    missing = [name for name in REQUIRED_WEIGHTS if not (weights_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing LoMa-R checkpoints in {weights_dir}: {', '.join(missing)}"
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_onnx = output / (
        f"loma_r_features_k{FEATURE_KEYPOINT_COUNT}_{args.precision}.onnx"
    )
    matcher_onnx = output / f"loma_r_matcher_dynamic_{args.precision}.onnx"
    feature_contract = feature_onnx_contract(
        repo, weights_dir, args.input_size, args.precision
    )
    matcher_contract = matcher_onnx_contract(repo, weights_dir, args.precision)

    if args.matcher_only:
        require_matcher_only_feature(feature_onnx, feature_contract)
        export_feature = False
    else:
        export_feature = needs_export(feature_onnx, feature_contract, args.force)
    export_matcher = needs_export(matcher_onnx, matcher_contract, args.force)

    if export_feature or export_matcher:
        if not torch.cuda.is_available():
            raise RuntimeError("LoMa-R ONNX export requires an NVIDIA CUDA device")
        # ONNX tracing does not need cuDNN kernels. Disabling cuDNN also avoids
        # a system cuDNN DLL shadowing the version bundled with the Torch wheel.
        torch.backends.cudnn.enabled = False
        model = load_model(repo, weights_dir)
    else:
        model = None

    size = args.input_size
    if export_feature:
        assert model is not None
        make_dinov2_position_encoding_exportable(model, size)
        image = torch.zeros((1, 3, size, size), device="cuda", dtype=torch.float32)
        export_onnx(
            FeatureWrapper(model, FEATURE_KEYPOINT_COUNT).eval(),
            (image,),
            feature_onnx,
            ["image"],
            ["keypoints", "keypoint_scores", "descriptors"],
            feature_contract,
        )
    if export_matcher:
        assert model is not None
        count = MATCHER_EXPORT_SAMPLE_KEYPOINTS
        keypoints = torch.zeros((1, count, 2), device="cuda", dtype=torch.float32)
        descriptors = torch.zeros((1, count, 256), device="cuda", dtype=torch.float32)
        valid = torch.ones((1, count), device="cuda", dtype=torch.bool)
        export_onnx(
            MatcherWrapper(model).eval(),
            (keypoints, keypoints, descriptors, descriptors, valid, valid),
            matcher_onnx,
            [
                "keypoints0",
                "keypoints1",
                "descriptors0",
                "descriptors1",
                "valid0",
                "valid1",
            ],
            ["scores"],
            matcher_contract,
            dynamic_axes={
                "keypoints0": {1: "keypoint_count"},
                "keypoints1": {1: "keypoint_count"},
                "descriptors0": {1: "keypoint_count"},
                "descriptors1": {1: "keypoint_count"},
                "valid0": {1: "keypoint_count"},
                "valid1": {1: "keypoint_count"},
                "scores": {1: "keypoint_count", 2: "keypoint_count"},
            },
        )

    manifests = compose_package(feature_onnx, matcher_onnx, output)
    print("Portable LoMa-R package written:")
    for manifest in manifests:
        print(f"  {manifest}")
    if args.onnx_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("LoMa-R TensorRT engine build requires an NVIDIA CUDA device")
    workspace = int(max(1.0, args.workspace_gib) * 1024**3)
    feature_engine = output / (
        f"loma_r_features_k{FEATURE_KEYPOINT_COUNT}_{args.precision}.engine"
    )
    matcher_engine = output / (
        f"loma_r_matcher_k{FEATURE_KEYPOINT_COUNT}_{args.precision}.engine"
    )
    if not args.matcher_only:
        build_engine_with_provenance(
            feature_onnx,
            feature_engine,
            args.precision,
            workspace,
            fixed_keypoints=0,
            force=args.force,
        )
    build_engine_with_provenance(
        matcher_onnx,
        matcher_engine,
        args.precision,
        workspace,
        fixed_keypoints=FEATURE_KEYPOINT_COUNT,
        force=args.force,
    )


if __name__ == "__main__":
    main()
