#!/usr/bin/env python3
"""Export DISK/ALIKED as TorchScript for PlaScan C++ extractors.

Fixes:
  - DISK: bypass kornia heatmap_to_keypoints (nonzero→topk, NMS via max_pool2d)
  - ALIKED: monkey-patch DeformableConv2d → regular Conv2d (avoids torchvision::deform_conv2d)
  - Bitwise_xor: monkey-patch Attention.forward

C++ interface: forward(image [1,1,H,W], orig_wh [W,H]) -> (kpts [N,2], descs [N,D], scores [N])
"""
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

OUT = Path("/home/guderian/code/plascan/resources/models")

# ── Monkey-patch 1: Attention (bitwise_xor bug) ──
from lightglue.lightglue import Attention
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
Attention.forward = _patched_attn

# ── Monkey-patch 2: DeformableConv2d with pure-PyTorch DCN (avoids torchvision) ──
from lightglue.aliked import DeformableConv2d
_orig_deform_forward = DeformableConv2d.forward

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
        W = heatmap.shape[2]
        k = min(self.max_kpts, flat.shape[1])
        top_vals, top_idx = torch.topk(flat, k, dim=1)
        top_y = (top_idx // W).float()
        top_x = (top_idx % W).float()
        kpts = torch.stack([top_x, top_y], dim=-1)
        H = heatmap.shape[1]
        kpts_norm = torch.stack([
            top_x / (float(W) - 1.0) * 2.0 - 1.0,
            top_y / (float(H) - 1.0) * 2.0 - 1.0
        ], dim=-1).unsqueeze(2)
        desc_sampled = F.grid_sample(descriptors, kpts_norm, mode='bilinear', align_corners=True)
        descs = F.normalize(desc_sampled.squeeze(-1).permute(0, 2, 1), p=2, dim=2)
        return kpts, descs, top_vals

# ── ALIKED: force top_k mode, DCN already patched ──
class AlikedExtractorWrap(nn.Module):
    def __init__(self, max_kpts=2048):
        super().__init__()
        from lightglue import ALIKED as _ALIKED
        self.ext = _ALIKED(max_num_keypoints=max_kpts).eval()
        self.ext.dkd.top_k = max_kpts
        self.ext.dkd.scores_th = 0.0

    def forward(self, image, orig_wh):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        feature_map, score_map = self.ext.extract_dense_map(image)
        keypoints, kptscores, _ = self.ext.dkd(score_map)
        _, _, h, w = image.shape
        wh = torch.tensor([float(w) - 1.0, float(h) - 1.0], device=image.device)
        kpts = (torch.stack(keypoints) + 1.0) / 2.0 * wh[None, None, :]
        scores = torch.stack(kptscores)
        descriptors, _ = self.ext.desc_head(feature_map, keypoints)
        descs = F.normalize(torch.stack(descriptors), p=2, dim=2)
        return kpts, descs, scores

# ── Export ──
OUT.mkdir(parents=True, exist_ok=True)
for name, cls, max_k in [("disk_extractor_cpu_1200", DiskExtractorWrap, 1200),
                          ("aliked_extractor_cpu_480", AlikedExtractorWrap, 480)]:
    print(f"Exporting {name}...")
    model = cls(max_kpts=max_k)
    dummy = torch.rand(1, 1, 480, 640)
    dummy_wh = torch.tensor([640., 480.])
    traced = torch.jit.trace(model, (dummy, dummy_wh), strict=False)
    path = OUT / f"{name}.pt"
    traced.save(str(path))
    # Also save cuda variant
    traced.save(str(OUT / f"{name.replace('_cpu_', '_cuda_')}.pt"))
    print(f"  OK: {path}")
print("Done")
