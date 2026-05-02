#!/usr/bin/env python3
"""为测试生成 TorchScript 占位模型，接口匹配 C++ DiskExtractor/AlikedExtractor

C++ 期望 forward(image [1,1,H,W] float32) -> Tuple(kpts [1,N,2], descs [1,N,D], scores [1,N])
kpts 是像素坐标，会经 tensorToFeatureOutput 除以 coordScale 映射回原图坐标。
"""
import torch
import os

models_dir = os.path.join(os.path.dirname(__file__), "..", "resources", "models")
os.makedirs(models_dir, exist_ok=True)

class ExtractorModel(torch.nn.Module):
    def __init__(self, desc_dim: int = 128):
        super().__init__()
        self.desc_dim = desc_dim

    def forward(self, x: torch.Tensor, orig_wh: torch.Tensor):
        N = 100
        kpts = torch.rand(1, N, 2, device=x.device, dtype=x.dtype) * 450.0
        descs = torch.nn.functional.normalize(
            torch.randn(1, N, self.desc_dim, device=x.device, dtype=x.dtype), dim=2)
        scores = torch.rand(1, N, device=x.device, dtype=x.dtype)
        return kpts, descs, scores


def make_model(desc_dim: int):
    m = ExtractorModel(desc_dim=desc_dim)
    m.eval()
    dummy = torch.rand(1, 1, 480, 640)
    dummy_wh = torch.tensor([640.0, 480.0])
    return torch.jit.trace(m, (dummy, dummy_wh), strict=False)


# DISK (128-dim)
disk = make_model(128)
disk.save(os.path.join(models_dir, "disk_extractor_cuda_1200.pt"))
disk.save(os.path.join(models_dir, "disk_extractor_cpu_1200.pt"))
print("[DISK] models saved")

# ALIKED (128-dim)
aliked = make_model(128)
aliked.save(os.path.join(models_dir, "aliked_extractor_cpu_480.pt"))
aliked.save(os.path.join(models_dir, "aliked_extractor_cuda_480.pt"))
print("[ALIKED] models saved")

# SuperPoint (256-dim descriptors + dense descriptors map)
# C++ 调用 forward(image [1,1,H,W], orig_wh [W,H]) -> (kpts [N,2], scores [N], desc_dense [1,256,H/8,W/8])
# 注意: kpts/scores 不带 batch 维度 (N,2) / (N,) 而非 (1,N,2) / (1,N)
class SuperPointModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, orig_wh: torch.Tensor):
        N = 80
        H = x.shape[2]
        W = x.shape[3]
        kpts = (torch.rand(1, N, 2, device=x.device, dtype=x.dtype) * 450.0).squeeze(0)
        scores = (torch.rand(1, N, device=x.device, dtype=x.dtype) * 0.8 + 0.1).squeeze(0)
        dense = torch.rand(1, 256, ((H + 7) // 8), ((W + 7) // 8),
                           device=x.device, dtype=x.dtype)
        return kpts, scores, dense

sp = SuperPointModel()
sp.eval()
dummy = torch.rand(1, 1, 480, 640)
dummy_wh = torch.tensor([640.0, 480.0])
traced = torch.jit.trace(sp, (dummy, dummy_wh), strict=False)
traced.save(os.path.join(models_dir, "superpoint_test.pt"))
print("[SuperPoint] test model saved")

print("Done")
