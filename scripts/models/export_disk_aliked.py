#!/usr/bin/env python3
"""Export DISK/ALIKED as TorchScript for PlaScan C++ extractors.

Fixes:
  - DISK: bypass kornia heatmap_to_keypoints (nonzero→topk, NMS via max_pool2d)
  - ALIKED: monkey-patch DeformableConv2d → regular Conv2d (avoids torchvision::deform_conv2d)
  - Bitwise_xor: monkey-patch Attention.forward

C++ interface: forward(image [1,1,H,W], orig_wh [W,H]) -> (kpts [N,2], descs [N,D], scores [N])
"""
import os
import argparse
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

OUT = Path(os.environ.get(
    "PLASCAN_MODEL_OUT",
    Path(__file__).resolve().parents[2] / "resources" / "models",
))

# ── Monkey-patch 1: Attention (bitwise_xor bug) ──
def _patched_attn(self, q, k, v, mask=None):
    if self.has_sdp:
        args = [x.contiguous() for x in [q, k, v]]
        out = F.scaled_dot_product_attention(*args, attn_mask=mask)
        return out if mask is None else out.nan_to_num()
    s = q.shape[-1] ** -0.5
    sim = torch.einsum("...id,...jd->...ij", q, k) * s
    if mask is not None:
        sim = sim.masked_fill(torch.logical_not(mask), -float("inf"))
    attn = F.softmax(sim, -1)
    return torch.einsum("...ij,...jd->...id", attn, v)

# ── Monkey-patch 2: DeformableConv2d with pure-PyTorch DCN (avoids torchvision) ──
def _pure_deform_conv2d(x, offset, weight, bias, stride=1, padding=1):
    """Trace-safe deformable conv2d using grid_sample (no torchvision dependency)."""
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    kH, kW = weight.shape[2], weight.shape[3]
    pad = padding[0] if isinstance(padding, tuple) else padding
    x_pad = F.pad(x, (pad, pad, pad, pad))
    # offset [B, 2*kH*kW, H, W] → [B, kH*kW, H, W, 2]
    o = offset.reshape(B, 2, kH * kW, H, W).permute(0, 2, 3, 4, 1)
    dy, dx = o[..., 0], o[..., 1]
    # Grid positions (linspace is trace-safe)
    gy = torch.linspace(float(pad), float(H + pad - 1), H, device=x.device)
    gx = torch.linspace(float(pad), float(W + pad - 1), W, device=x.device)
    # Kernel base offsets
    ky = torch.arange(kH, device=x.device, dtype=torch.float32) - (kH - 1) / 2.0
    ky = ky.repeat_interleave(kW)
    kx = torch.arange(kW, device=x.device, dtype=torch.float32) - (kW - 1) / 2.0
    kx = kx.repeat(kH)
    ky = ky.reshape(1, kH * kW, 1, 1)
    kx = kx.reshape(1, kH * kW, 1, 1)
    # Sample positions [B, N, H, W]
    sy = gy.reshape(1, 1, H, 1) + ky + dy
    sx = gx.reshape(1, 1, 1, W) + kx + dx
    Hp, Wp = H + 2 * pad, W + 2 * pad
    sy = sy / (Hp - 1) * 2.0 - 1.0
    sx = sx / (Wp - 1) * 2.0 - 1.0
    # Stack into grid [B, N, H, W, 2] → [B, H, N*W, 2] for grid_sample
    grid = torch.stack([sx, sy], dim=-1)
    grid = grid.permute(0, 2, 1, 3, 4).reshape(B, H, kH * kW * W, 2)
    sampled = F.grid_sample(x_pad, grid, mode='bilinear', align_corners=True)
    sampled = sampled.reshape(B, C_in, H, kH * kW, W).permute(0, 1, 3, 2, 4)
    sampled = sampled.reshape(B, C_in * kH * kW, H, W)
    w = weight.reshape(C_out, C_in * kH * kW, 1, 1)
    return F.conv2d(sampled, w, bias)

def _patched_deform_forward(self, x):
    h, w = x.shape[2:]
    max_offset = max(h, w) / 4.0
    # offset_conv predicts offsets, with optional mask
    out = self.offset_conv(x)
    if self.mask:
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
    else:
        offset = out
    offset = offset.clamp(-max_offset, max_offset)
    return _pure_deform_conv2d(x, offset, self.regular_conv.weight,
                               self.regular_conv.bias, padding=self.padding)

def patch_lightglue_for_export():
    from lightglue.lightglue import Attention
    from lightglue.aliked import DeformableConv2d

    Attention.forward = _patched_attn
    DeformableConv2d.forward = _patched_deform_forward

# ── DISK: bypass kornia heatmap_to_keypoints ──
class DiskExtractorWrap(nn.Module):
    def __init__(self, max_kpts=2048):
        super().__init__()
        from kornia.feature import DISK as _DISK
        self.disk = _DISK.from_pretrained("depth").eval()
        self.max_kpts = max_kpts

    def forward(self, image, orig_wh):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        h, w = image.shape[2], image.shape[3]
        pd_h = (16 - h % 16) % 16
        pd_w = (16 - w % 16) % 16
        image = F.pad(image, (0, pd_w, 0, pd_h))
        heatmaps, descriptors = self.disk.heatmap_and_dense_descriptors(image)
        heatmaps = heatmaps[..., :h, :w]
        descriptors = descriptors[..., :h, :w]
        heatmap = heatmaps.squeeze(1)
        pooled = F.max_pool2d(heatmap, kernel_size=5, stride=1, padding=2)
        nms_mask = (heatmap == pooled) & (heatmap > 0.0)
        scores_masked = heatmap * nms_mask.float()
        flat = scores_masked.reshape(1, -1)
        shape = torch._shape_as_tensor(heatmap)
        W_index = shape[2]
        k = min(self.max_kpts, flat.shape[1])
        top_vals, top_idx = torch.topk(flat, k, dim=1)
        top_y = torch.div(top_idx, W_index, rounding_mode="floor").float()
        top_x = torch.remainder(top_idx, W_index).float()
        kpts = torch.stack([top_x, top_y], dim=-1)
        wh = shape[[2, 1]].to(device=top_x.device, dtype=top_x.dtype)
        denom = torch.clamp(wh - 1.0, min=1.0)
        kpts_norm = torch.stack([
            top_x / denom[0] * 2.0 - 1.0,
            top_y / denom[1] * 2.0 - 1.0
        ], dim=-1).unsqueeze(2)
        desc_sampled = F.grid_sample(descriptors, kpts_norm, mode='bilinear', align_corners=True)
        descs = F.normalize(desc_sampled.squeeze(-1).permute(0, 2, 1), p=2, dim=2)
        return kpts, descs, top_vals

# ── ALIKED: force top_k mode, DCN already patched ──
class AlikedExtractorWrap(nn.Module):
    def __init__(self, max_kpts=2048):
        super().__init__()
        patch_lightglue_for_export()
        from lightglue import ALIKED as _ALIKED
        self.ext = _ALIKED(max_num_keypoints=max_kpts).eval()
        self.ext.dkd.top_k = max_kpts
        self.ext.dkd.scores_th = 0.0

    def forward(self, image, orig_wh):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        feature_map, score_map = self.ext.extract_dense_map(image)
        keypoints, kptscores, _ = self.ext.dkd(score_map)
        shape = torch._shape_as_tensor(image).to(device=image.device, dtype=torch.float32)
        wh = torch.stack([shape[3] - 1.0, shape[2] - 1.0])
        kpts = (torch.stack(keypoints) + 1.0) / 2.0 * wh[None, None, :]
        scores = torch.stack(kptscores)
        descriptors, _ = self.ext.desc_head(feature_map, keypoints)
        descs = F.normalize(torch.stack(descriptors), p=2, dim=2)
        return kpts, descs, scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export PlaScan DISK/ALIKED extractor TorchScript models."
    )
    parser.add_argument(
        "--models",
        choices=("disk", "aliked", "all"),
        default="disk",
        help="models to export; default exports the DISK 8192 main model",
    )
    parser.add_argument(
        "--devices",
        choices=("auto", "cpu", "cuda", "all"),
        default="auto",
        help="devices to trace; auto exports CPU and CUDA when CUDA is available",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="output directory, defaults to PLASCAN_MODEL_OUT or resources/models",
    )
    parser.add_argument(
        "--disk-max-kpts",
        type=int,
        default=8192,
        help="maximum DISK keypoints to export into the main model",
    )
    parser.add_argument(
        "--aliked-max-kpts",
        type=int,
        default=480,
        help="maximum ALIKED keypoints when --models includes aliked",
    )
    return parser.parse_args()


def selected_models(models):
    if models == "all":
        return ("disk", "aliked")
    return (models,)


def selected_devices(devices):
    cuda_available = torch.cuda.is_available()
    if devices == "cpu":
        return ("cpu",)
    if devices == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return ("cuda",)
    if devices == "all":
        if not cuda_available:
            print("CUDA is not available; exporting CPU only")
            return ("cpu",)
        return ("cpu", "cuda")
    if cuda_available:
        return ("cpu", "cuda")
    return ("cpu",)


def export_model(name, cls, max_kpts, device, output_dir):
    filename = f"{name}_{device}_{max_kpts}.torchscript"
    path = output_dir / filename
    print(f"Exporting {filename}...")
    torch_device = torch.device(device)
    model = cls(max_kpts=max_kpts).eval().to(torch_device)
    dummy = torch.rand(1, 1, 480, 640, device=torch_device)
    dummy_wh = torch.tensor([640., 480.], device=torch_device)
    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy, dummy_wh), strict=False)
    traced.save(str(path))
    print(f"  OK: {path}")
    return path


def export_all(args=None):
    args = args or parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "disk": ("disk_extractor", DiskExtractorWrap, args.disk_max_kpts),
        "aliked": ("aliked_extractor", AlikedExtractorWrap, args.aliked_max_kpts),
    }
    exported = []
    for model_key in selected_models(args.models):
        name, cls, max_kpts = specs[model_key]
        for device in selected_devices(args.devices):
            exported.append(export_model(name, cls, max_kpts, device, output_dir))
    print("Done")
    return exported


if __name__ == "__main__":
    export_all()
