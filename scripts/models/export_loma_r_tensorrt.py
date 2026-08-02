#!/usr/bin/env python3
"""Export the official LoMa-R pipeline to PlaScan TensorRT engines.

The source architecture is loaded from a local LoMa-R checkout.  Four official
checkpoints are required locally and this script never downloads weights.  The
result contains a per-image DaD + DeDoDe-G feature engine, a LoMa-R matcher
engine, and a manifest that prevents incompatible engines from being combined.

LoMa-R source: https://github.com/davnords/loma (MIT; matcher derives from
LightGlue under Apache-2.0).  Checkpoint redistribution is intentionally left
to the model authors; PlaScan stores only generated local artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType
from urllib.parse import urlparse

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "build" / "model_cache" / "loma_r_tensorrt"
REQUIRED_WEIGHTS = (
    "loma_R.pth",
    "dad.pth",
    "dedode_descriptor_G.pth",
    "dinov2_vitl14_pretrain.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DaD + DeDoDe-G + LoMa-R to TensorRT"
    )
    parser.add_argument("--loma-repo", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-size", type=int, default=784)
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument(
        "--matcher-only",
        action="store_true",
        help="reuse an existing feature engine and rebuild only the matcher",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """TensorRT-friendly LoMa-R forward with explicit fixed-shape attention."""

    def __init__(self, model: torch.nn.Module, keypoints: int):
        super().__init__()
        self.model = model
        self.keypoints = keypoints
        self.heads = int(model.cfg.num_heads)
        self.dimension = int(model.cfg.embed_dim)
        self.head_dimension = self.dimension // self.heads
        self.attention_scale = self.head_dimension**-0.5

    def _rotate_half(self, value: torch.Tensor) -> torch.Tensor:
        paired = value.reshape(
            1, self.heads, self.keypoints, self.head_dimension // 2, 2
        )
        return torch.stack((-paired[..., 1], paired[..., 0]), dim=4).reshape(
            1, self.heads, self.keypoints, self.head_dimension
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
            1, self.keypoints, self.heads, self.head_dimension, 3
        ).permute(0, 2, 1, 3, 4)
        query = self._apply_rotary(encoding, qkv[..., 0])
        key = self._apply_rotary(encoding, qkv[..., 1])
        context = self._attention(query, key, qkv[..., 2], valid, valid)
        message = block.out_proj(
            context.permute(0, 2, 1, 3).reshape(
                1, self.keypoints, self.dimension
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
                1, self.keypoints, self.heads, self.head_dimension
            ).permute(0, 2, 1, 3)

        query_key0 = split_heads(block.to_qk(descriptors0))
        query_key1 = split_heads(block.to_qk(descriptors1))
        value0 = split_heads(block.to_v(descriptors0))
        value1 = split_heads(block.to_v(descriptors1))
        message0 = self._attention(query_key0, query_key1, value1, valid0, valid1)
        message1 = self._attention(query_key1, query_key0, value0, valid1, valid0)

        def merge_heads(value: torch.Tensor) -> torch.Tensor:
            return value.permute(0, 2, 1, 3).reshape(
                1, self.keypoints, self.dimension
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        external_data=True,
    )


def build_engine(onnx_path: Path, engine_path: Path, precision: str, workspace: int) -> None:
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
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_path.name}")
    engine_path.write_bytes(bytes(serialized))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LoMa-R TensorRT export requires an NVIDIA CUDA device")
    # ONNX tracing does not need cuDNN kernels. Disabling cuDNN here also keeps
    # export usable on Windows machines where a system cuDNN DLL shadows the
    # version bundled with the selected PyTorch wheel. TensorRT remains the
    # production inference backend and is unaffected by this setting.
    torch.backends.cudnn.enabled = False
    if args.input_size <= 0 or args.input_size % 14 != 0:
        raise ValueError("--input-size must be positive and divisible by DINOv2 patch size 14")
    if args.keypoints <= 0:
        raise ValueError("--keypoints must be positive")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"k{args.keypoints}_{args.precision}"
    feature_onnx = output / f"loma_r_features_{suffix}.onnx"
    matcher_onnx = output / f"loma_r_matcher_{suffix}.onnx"
    feature_engine = output / f"loma_r_features_{suffix}.engine"
    matcher_engine = output / f"loma_r_matcher_{suffix}.engine"
    manifest = output / "loma_r_tensorrt.json"
    generated = ((matcher_onnx, matcher_engine) if args.matcher_only else
                 (feature_onnx, matcher_onnx, feature_engine, matcher_engine, manifest))
    if not args.force and any(path.exists() for path in generated):
        raise FileExistsError("Output exists; use --force to replace the LoMa-R package")

    model = load_model(args.loma_repo.resolve(), args.weights_dir.resolve())
    size = args.input_size
    count = args.keypoints
    make_dinov2_position_encoding_exportable(model, size)
    image = torch.zeros((1, 3, size, size), device="cuda", dtype=torch.float32)
    keypoints = torch.zeros((1, count, 2), device="cuda", dtype=torch.float32)
    descriptors = torch.zeros((1, count, 256), device="cuda", dtype=torch.float32)
    valid = torch.ones((1, count), device="cuda", dtype=torch.bool)

    if not args.matcher_only:
        export_onnx(
            FeatureWrapper(model, count).eval(),
            (image,),
            feature_onnx,
            ["image"],
            ["keypoints", "keypoint_scores", "descriptors"],
        )
    export_onnx(
        MatcherWrapper(model, count).eval(),
        (keypoints, keypoints, descriptors, descriptors, valid, valid),
        matcher_onnx,
        ["keypoints0", "keypoints1", "descriptors0", "descriptors1", "valid0", "valid1"],
        ["scores"],
    )
    if args.onnx_only:
        print(f"ONNX models written to {output}")
        return

    workspace = int(max(1.0, args.workspace_gib) * 1024**3)
    if not args.matcher_only:
        build_engine(feature_onnx, feature_engine, args.precision, workspace)
    build_engine(matcher_onnx, matcher_engine, args.precision, workspace)
    if not feature_engine.is_file():
        raise FileNotFoundError(
            f"Feature engine required for manifest was not found: {feature_engine}"
        )
    metadata = {
        "schema_version": 1,
        "algorithm_id": "loma_r",
        "algorithm_version": 1,
        "source": "LoMa-R (DaD + DeDoDe-G/DINOv2 + LoMa-R)",
        "precision": args.precision,
        "input_width": size,
        "input_height": size,
        "keypoint_count": count,
        "descriptor_dimension": 256,
        "feature_engine": feature_engine.name,
        "matcher_engine": matcher_engine.name,
        "feature_engine_sha256": sha256(feature_engine),
        "matcher_engine_sha256": sha256(matcher_engine),
        "checkpoints": {name: sha256(args.weights_dir / name) for name in REQUIRED_WEIGHTS},
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "tensorrt": __import__("tensorrt").__version__,
    }
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"LoMa-R TensorRT package written to {manifest}")


if __name__ == "__main__":
    main()
