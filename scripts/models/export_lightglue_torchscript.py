#!/usr/bin/env python3
"""Export LightGlue matchers as TorchScript modules for PlaScan C++.

C++ interface:
    forward(kpts0, descs0, image_size0, kpts1, descs1, image_size1) -> scores

Inputs:
    kpts*:      [1, N, 2] pixel coordinates; SIFT may pass [1, N, 4] as
                [x, y, scale, orientation_radians]
    descs*:     [1, N, D] descriptors, D depends on feature type
    image_size: [1, 2] as [width, height]

Output:
    scores: [1, N0 + 1, N1 + 1] log assignment matrix
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "resources" / "models"
FEATURE_DIMS = {
    "disk": 128,
    "aliked": 128,
    "sift": 128,
    "superpoint": 256,
}


def add_vendored_lightglue_to_path() -> str:
    candidates: list[Path] = []
    env_repo = os.environ.get("LIGHTGLUE_REPO", "").strip()
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.extend([
        ROOT / "3rdparty" / "LightGlue-main",
        ROOT / "3rdparty" / "LightGlue",
        ROOT / "third_party" / "LightGlue-main",
        ROOT / "third_party" / "LightGlue",
    ])

    for candidate in candidates:
        if (candidate / "lightglue" / "__init__.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str
    return ""


add_vendored_lightglue_to_path()
from lightglue import LightGlue
from lightglue.lightglue import normalize_keypoints


class LightGlueScoreWrapper(torch.nn.Module):
    def __init__(self, feature: str):
        super().__init__()
        self.model = LightGlue(
            features=feature,
            flash=False,
            mp=False,
            depth_confidence=-1,
            width_confidence=-1,
        ).eval()

    def forward(
        self,
        kpts0: torch.Tensor,
        descs0: torch.Tensor,
        image_size0: torch.Tensor,
        kpts1: torch.Tensor,
        descs1: torch.Tensor,
        image_size1: torch.Tensor,
    ) -> torch.Tensor:
        xy0 = normalize_keypoints(kpts0[..., :2], image_size0).clone()
        xy1 = normalize_keypoints(kpts1[..., :2], image_size1).clone()
        if self.model.conf.add_scale_ori:
            if kpts0.size(-1) >= 4:
                scales0 = kpts0[..., 2:3]
                oris0 = kpts0[..., 3:4]
            else:
                scales0 = torch.ones_like(xy0[..., :1])
                oris0 = torch.zeros_like(xy0[..., :1])
            if kpts1.size(-1) >= 4:
                scales1 = kpts1[..., 2:3]
                oris1 = kpts1[..., 3:4]
            else:
                scales1 = torch.ones_like(xy1[..., :1])
                oris1 = torch.zeros_like(xy1[..., :1])
            kpts0 = torch.cat([xy0, scales0, oris0], dim=-1)
            kpts1 = torch.cat([xy1, scales1, oris1], dim=-1)
        else:
            kpts0 = xy0
            kpts1 = xy1

        desc0 = descs0.detach().contiguous()
        desc1 = descs1.detach().contiguous()

        desc0 = self.model.input_proj(desc0)
        desc1 = self.model.input_proj(desc1)
        encoding0 = self.model.posenc(kpts0)
        encoding1 = self.model.posenc(kpts1)

        for layer in self.model.transformers:
            desc0, desc1 = layer(desc0, desc1, encoding0, encoding1)

        scores, _ = self.model.log_assignment[-1](desc0, desc1)
        return scores


def parse_features(value: str) -> list[str]:
    if value == "all":
        return ["disk", "aliked", "sift", "superpoint"]
    features = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in features if item not in FEATURE_DIMS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown feature(s): {', '.join(unknown)}")
    return features


def parse_devices(value: str) -> list[str]:
    if value == "all":
        return ["cpu", "cuda"]
    devices = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in devices if item not in {"cpu", "cuda"}]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown device(s): {', '.join(unknown)}")
    return devices


def export_one(feature: str, device_name: str, output_dir: Path) -> Path:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but torch.cuda.is_available() is false")

    device = torch.device(device_name)
    dim = FEATURE_DIMS[feature]
    model = LightGlueScoreWrapper(feature).eval().to(device)

    torch.manual_seed(42)
    kpt_dim = 4 if feature == "sift" else 2
    kpts0 = torch.rand(1, 16, kpt_dim, device=device)
    kpts1 = torch.rand(1, 18, kpt_dim, device=device)
    kpts0[..., :2] *= torch.tensor([640.0, 480.0], device=device)
    kpts1[..., :2] *= torch.tensor([640.0, 480.0], device=device)
    if feature == "sift":
        kpts0[..., 2:3] = 1.0 + kpts0[..., 2:3] * 15.0
        kpts1[..., 2:3] = 1.0 + kpts1[..., 2:3] * 15.0
        kpts0[..., 3:4] = (kpts0[..., 3:4] * 2.0 - 1.0) * torch.pi
        kpts1[..., 3:4] = (kpts1[..., 3:4] * 2.0 - 1.0) * torch.pi
    desc0 = torch.rand(1, 16, dim, device=device)
    desc1 = torch.rand(1, 18, dim, device=device)
    image_size = torch.tensor([[640.0, 480.0]], device=device)

    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (kpts0, desc0, image_size, kpts1, desc1, image_size),
            strict=False,
        )
        scores = traced(kpts0, desc0, image_size, kpts1, desc1, image_size)

    expected_shape = (1, 17, 19)
    if tuple(scores.shape) != expected_shape:
        raise RuntimeError(f"{feature}/{device_name} produced {tuple(scores.shape)}, expected {expected_shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lightglue_{feature}_{device_name}.torchscript"
    traced.save(str(output_path))
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export LightGlue TorchScript matchers")
    parser.add_argument(
        "--features",
        type=parse_features,
        default=parse_features("disk,aliked,sift"),
        help="Comma-separated: disk,aliked,sift,superpoint or all. Default: disk,aliked,sift",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=parse_devices("all"),
        help="Comma-separated: cpu,cuda or all. Default: all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    for feature in args.features:
        for device_name in args.devices:
            path = export_one(feature, device_name, args.output_dir)
            print(f"exported {feature}/{device_name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
