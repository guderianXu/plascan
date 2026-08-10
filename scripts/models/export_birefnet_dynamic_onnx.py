#!/usr/bin/env python3
"""Export the pinned BiRefNet Dynamic checkpoint to PlaScan's ONNX contract.

The upstream checkpoint was trained with dynamic image shapes, but PlaScan's
first production TensorRT profile is deliberately fixed to 1024 x 1024.  The
runtime preserves the source aspect ratio with letterboxing.  The exported
graph exposes raw foreground logits; sigmoid remains an explicit part of the
C++ model contract.

Upstream model: https://huggingface.co/ZhengPeng7/BiRefNet_dynamic (MIT)
Official export recipe:
https://github.com/ZhengPeng7/BiRefNet/blob/main/tutorials/BiRefNet_pth2onnx.ipynb
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import typing
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "ZhengPeng7/BiRefNet_dynamic"
MODEL_REVISION = "280306042f57b7a33854319da62fd86aaa89ec4c"
MODEL_FILE = "model.safetensors"
MODEL_BYTES = 444_473_596
MODEL_SHA256 = "e3d2e4884e51ff30f0cd630edc6b1e41b06b7f23a0a2a5169f7b7cb33a711c2d"
INPUT_SIZE = 1024
INPUT_NAME = "input_image"
OUTPUT_NAME = "output_image"
OPSET_VERSION = 17
DEFORM_EXPORTER_REVISION = "ed84d9114b4ab17c5e0a5dc8a412c0d57743a402"
DEFORM_EXPORTER_URL = (
    "https://raw.githubusercontent.com/masamitsu-murase/"
    "deform_conv2d_onnx_exporter/"
    f"{DEFORM_EXPORTER_REVISION}/src/deform_conv2d_onnx_exporter.py"
)
DEFORM_EXPORTER_BYTES = 22_977
DEFORM_EXPORTER_SHA256 = (
    "4856926c0d55ecd1185e21f8056f76670c25603e71f96b98c6b12d263de721ba"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "resources"
            / "models"
            / "birefnet_dynamic"
            / "BiRefNet_dynamic_1024.onnx"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache" / "huggingface",
        help="Hugging Face cache used for the pinned source snapshot.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-checker",
        action="store_true",
        help="Skip onnx.checker after export (not allowed for release assets).",
    )
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="Skip PyTorch/ONNX Runtime equivalence (not allowed for release assets).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Unexpected size for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Unexpected SHA-256 for {path}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )


def download_source_snapshot(cache_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required; initialize PlaScan's .venv first"
        ) from error

    # Download into a real local directory instead of the Hub's symlink-based
    # snapshot cache.  The latter requires Windows Developer Mode or elevated
    # symlink privileges and otherwise fails after transferring the large
    # checkpoint.
    snapshot_dir = cache_dir / "birefnet_dynamic_snapshot" / MODEL_REVISION
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(snapshot_dir),
            allow_patterns=[
                MODEL_FILE,
                "config.json",
                "BiRefNet_config.py",
                "birefnet.py",
            ],
            max_workers=1,
        )
    )
    weights = snapshot / MODEL_FILE
    if not weights.is_file():
        raise RuntimeError(f"Pinned BiRefNet weights were not downloaded: {weights}")
    verify_file(weights, MODEL_BYTES, MODEL_SHA256)
    return snapshot


def load_deform_exporter() -> ModuleType:
    try:
        with urllib.request.urlopen(DEFORM_EXPORTER_URL, timeout=60) as response:
            source = response.read()
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(
            "Cannot download the pinned deform_conv2d ONNX exporter from "
            f"{DEFORM_EXPORTER_URL}"
        ) from error
    if len(source) != DEFORM_EXPORTER_BYTES:
        raise RuntimeError(
            "Pinned deform_conv2d exporter has an unexpected length: "
            f"expected {DEFORM_EXPORTER_BYTES}, got {len(source)}"
        )
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != DEFORM_EXPORTER_SHA256:
        raise RuntimeError(
            "Pinned deform_conv2d exporter failed SHA-256 verification: "
            f"expected {DEFORM_EXPORTER_SHA256}, got {actual_sha256}"
        )

    temporary = tempfile.TemporaryDirectory(prefix="plascan_birefnet_export_")
    module_path = Path(temporary.name) / "deform_conv2d_onnx_exporter.py"
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(
        "plascan_deform_conv2d_onnx_exporter", module_path
    )
    if spec is None or spec.loader is None:
        temporary.cleanup()
        raise RuntimeError("Cannot import the pinned deform_conv2d exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep the temporary source alive as long as the imported module is used.
    module._plascan_temporary_directory = temporary  # type: ignore[attr-defined]
    return module


def register_deform_conv2d_symbolic(torch_module: typing.Any) -> None:
    exporter = load_deform_exporter()
    from torch.onnx import symbolic_helper

    dimension_getter = getattr(symbolic_helper, "_get_tensor_dim_size", None)
    tensor_type = getattr(getattr(torch_module, "_C", None), "TensorType", None)
    if not callable(dimension_getter) or tensor_type is None:
        raise RuntimeError(
            "The installed PyTorch no longer exposes the legacy ONNX APIs required "
            "by BiRefNet's pinned deform_conv2d exporter"
        )

    def fixed_dimension(value: typing.Any, dimension: int) -> int | None:
        size = dimension_getter(value, dimension)
        if size is not None:
            return int(size)
        if dimension not in (0, 2, 3):
            return None

        value_type = typing.cast(typing.Any, value.type())
        strides_getter = getattr(value_type, "strides", None)
        if not callable(strides_getter):
            raise RuntimeError(
                "Cannot inspect PyTorch strides while exporting BiRefNet's "
                f"deform_conv2d tensor for dimension {dimension}"
            )
        strides = strides_getter()
        if (
            strides is None
            or len(strides) != 4
            or any(stride is None or int(stride) <= 0 for stride in strides)
        ):
            raise RuntimeError(
                "Cannot infer a static BiRefNet deform_conv2d tensor shape from "
                f"PyTorch strides for dimension {dimension}: {strides}"
            )
        static_strides = [int(stride) for stride in strides]
        if dimension == 0:
            inferred = static_strides[3]
        elif dimension == 2:
            if static_strides[1] % static_strides[2] != 0:
                raise RuntimeError(
                    "Cannot infer the BiRefNet deform_conv2d height from non-divisible "
                    f"PyTorch strides: {static_strides}"
                )
            inferred = static_strides[1] // static_strides[2]
        else:
            inferred = static_strides[2]
        if inferred <= 0:
            raise RuntimeError(
                "BiRefNet deform_conv2d produced a non-positive inferred dimension: "
                f"dimension={dimension}, value={inferred}"
            )
        return inferred

    exporter.get_tensor_dim_size = fixed_dimension
    exporter.register_deform_conv2d_onnx_op()


def load_model(snapshot: Path) -> typing.Any:
    try:
        import torch
        from transformers import AutoModelForImageSegmentation
    except ImportError as error:
        raise RuntimeError(
            "torch, torchvision and transformers are required; initialize "
            "PlaScan's .venv first"
        ) from error

    model = AutoModelForImageSegmentation.from_pretrained(
        str(snapshot),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    return model.float().cpu().eval()


def export_onnx(model: typing.Any, output: Path) -> None:
    import torch

    class FinalLogits(torch.nn.Module):
        def __init__(self, source: typing.Any):
            super().__init__()
            self.source = source

        def forward(self, image: typing.Any) -> typing.Any:
            predictions = self.source(image)
            if not isinstance(predictions, (list, tuple)) or not predictions:
                raise RuntimeError("BiRefNet did not return its multi-scale logits")
            return predictions[-1].float()

    register_deform_conv2d_symbolic(torch)
    wrapper = FinalLogits(model).eval()
    sample = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                sample,
                str(temporary),
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                opset_version=OPSET_VERSION,
                do_constant_folding=True,
                dynamo=False,
                external_data=False,
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Exported ONNX in {time.perf_counter() - started:.1f}s: {output}")


def check_onnx_contract(output: Path) -> dict[str, typing.Any]:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("onnx is required to verify the exported graph") from error

    graph = onnx.load(str(output), load_external_data=False)
    onnx.checker.check_model(graph, full_check=False)
    standard_opsets = [
        int(item.version) for item in graph.opset_import if item.domain in ("", "ai.onnx")
    ]
    if standard_opsets != [OPSET_VERSION]:
        raise RuntimeError(f"Unexpected BiRefNet ONNX opset imports: {standard_opsets}")
    if len(graph.graph.input) != 1 or graph.graph.input[0].name != INPUT_NAME:
        raise RuntimeError("BiRefNet ONNX input contract does not match input_image")
    if len(graph.graph.output) != 1 or graph.graph.output[0].name != OUTPUT_NAME:
        raise RuntimeError("BiRefNet ONNX output contract does not match output_image")

    expected_type = onnx.TensorProto.FLOAT
    input_type = graph.graph.input[0].type.tensor_type.elem_type
    output_type = graph.graph.output[0].type.tensor_type.elem_type
    if input_type != expected_type or output_type != expected_type:
        raise RuntimeError(
            "BiRefNet ONNX input and output must both expose float32 tensors: "
            f"input={input_type}, output={output_type}"
        )

    def dimensions(value: typing.Any) -> list[int]:
        return [int(item.dim_value) for item in value.type.tensor_type.shape.dim]

    input_shape = dimensions(graph.graph.input[0])
    output_shape = dimensions(graph.graph.output[0])
    expected = [1, 3, INPUT_SIZE, INPUT_SIZE]
    if input_shape != expected:
        raise RuntimeError(f"Unexpected BiRefNet input shape: {input_shape}")
    if output_shape != [1, 1, INPUT_SIZE, INPUT_SIZE]:
        raise RuntimeError(f"Unexpected BiRefNet output shape: {output_shape}")
    custom_domains = sorted(
        {node.domain for node in graph.graph.node if node.domain not in ("", "ai.onnx")}
    )
    if custom_domains:
        raise RuntimeError(f"BiRefNet ONNX contains custom operator domains: {custom_domains}")
    external_initializers = [
        initializer.name
        for initializer in graph.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        or bool(initializer.external_data)
    ]
    if external_initializers:
        raise RuntimeError(
            "BiRefNet ONNX must be a single self-contained file, but external data "
            f"was referenced by {len(external_initializers)} initializer(s)"
        )
    return {
        "input_name": INPUT_NAME,
        "input_shape": input_shape,
        "output_name": OUTPUT_NAME,
        "output_shape": output_shape,
        "output_semantics": "foreground_logits_apply_sigmoid",
        "opset": OPSET_VERSION,
    }


def verify_runtime_equivalence(
    model: typing.Any, output: Path
) -> dict[str, typing.Any]:
    try:
        import numpy as np
        import onnxruntime
        import torch
    except ImportError as error:
        raise RuntimeError(
            "numpy, torch and onnxruntime are required for release verification"
        ) from error

    sample = torch.linspace(
        -1.0,
        1.0,
        steps=3 * INPUT_SIZE * INPUT_SIZE,
        dtype=torch.float32,
    ).reshape(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        predictions = model(sample)
    if not isinstance(predictions, (list, tuple)) or not predictions:
        raise RuntimeError("BiRefNet did not return multi-scale logits during verification")
    torch_output = predictions[-1].detach().cpu().numpy().astype(np.float32, copy=False)
    expected_shape = (1, 1, INPUT_SIZE, INPUT_SIZE)
    if torch_output.shape != expected_shape:
        raise RuntimeError(f"Unexpected PyTorch verification output: {torch_output.shape}")

    session = onnxruntime.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    onnx_outputs = session.run([OUTPUT_NAME], {INPUT_NAME: sample.numpy()})
    if len(onnx_outputs) != 1 or onnx_outputs[0].shape != expected_shape:
        shape = None if not onnx_outputs else onnx_outputs[0].shape
        raise RuntimeError(f"Unexpected ONNX Runtime verification output: {shape}")
    onnx_output = onnx_outputs[0]
    if not np.isfinite(torch_output).all() or not np.isfinite(onnx_output).all():
        raise RuntimeError("BiRefNet verification produced non-finite logits")

    absolute_error = np.abs(torch_output - onnx_output)
    max_absolute_error = float(absolute_error.max())
    mean_absolute_error = float(absolute_error.mean())
    if max_absolute_error > 5.0e-3 or mean_absolute_error > 5.0e-4:
        raise RuntimeError(
            "BiRefNet ONNX Runtime output differs from PyTorch: "
            f"max_abs={max_absolute_error:.8g}, mean_abs={mean_absolute_error:.8g}"
        )
    print(
        "PyTorch/ONNX Runtime equivalence passed: "
        f"max_abs={max_absolute_error:.8g}, mean_abs={mean_absolute_error:.8g}"
    )
    return {
        "provider": "CPUExecutionProvider",
        "input": "deterministic_linear_-1_to_1",
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "max_absolute_error_limit": 5.0e-3,
        "mean_absolute_error_limit": 5.0e-4,
    }


def write_provenance(
    output: Path,
    contract: dict[str, typing.Any],
    runtime_validation: dict[str, typing.Any] | None,
    elapsed_seconds: float,
) -> Path:
    import onnx
    import onnxruntime
    import torch
    import torchvision
    import transformers

    provenance = {
        "schema": "plascan_birefnet_dynamic_onnx_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_file": MODEL_FILE,
            "weights_bytes": MODEL_BYTES,
            "weights_sha256": MODEL_SHA256,
            "license": "MIT",
        },
        "deform_conv2d_exporter": {
            "revision": DEFORM_EXPORTER_REVISION,
            "source_sha256": DEFORM_EXPORTER_SHA256,
            "license": "MIT",
        },
        "contract": contract,
        "toolchain": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "runtime_validation": runtime_validation,
        "artifact": {
            "file": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "external_data": False,
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
    }
    path = output.with_suffix(".provenance.json")
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists; pass --force to replace it: {output}")

    started = time.perf_counter()
    snapshot = download_source_snapshot(args.cache_dir.expanduser().resolve())
    model = load_model(snapshot)
    export_onnx(model, output)
    contract = (
        {
            "input_name": INPUT_NAME,
            "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "output_name": OUTPUT_NAME,
            "output_shape": [1, 1, INPUT_SIZE, INPUT_SIZE],
            "output_semantics": "foreground_logits_apply_sigmoid",
            "opset": OPSET_VERSION,
        }
        if args.skip_checker
        else check_onnx_contract(output)
    )
    runtime_validation = (
        None if args.skip_runtime_check else verify_runtime_equivalence(model, output)
    )
    provenance = write_provenance(
        output, contract, runtime_validation, time.perf_counter() - started
    )
    print(f"ONNX bytes: {output.stat().st_size}")
    print(f"ONNX SHA-256: {sha256_file(output)}")
    print(f"Provenance: {provenance}")


if __name__ == "__main__":
    main()
