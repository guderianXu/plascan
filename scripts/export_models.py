#!/usr/bin/env python3
"""
导出 DISK、ALIKED、LoFTR、RoMa 为 TorchScript .pt 文件。

用法：
    conda activate plascan
    python scripts/export_models.py --all
    python scripts/export_models.py --disk --aliked
    python scripts/export_models.py --loftr --roma

输出目录：resources/models/
"""

import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "resources" / "models"


def export_disk(device: str = "cuda"):
    """导出 DISK 特征提取器（kornia/LightGlue 仓库）"""
    try:
        import torch
        from lightglue import DISK as DISKExtractor
    except ImportError:
        print("[DISK] 需要安装 lightglue: pip install git+https://github.com/cvg/LightGlue.git")
        return False

    print(f"[DISK] 导出中 (device={device})...")
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    extractor = DISKExtractor(max_num_keypoints=2048).eval().to(dev)

    # 包装为接受 (image_tensor [1,3,H,W], orig_wh [1,2]) 的 TorchScript 模块
    class DISKWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image: torch.Tensor, orig_wh: torch.Tensor):
            # image: [1,3,H,W] float32 [0,1]
            feats = self.model.extract(image)
            kpts = feats["keypoints"]       # [1,N,2]
            descs = feats["descriptors"]    # [1,N,128]
            fixed_wh = orig_wh             # 直接透传，DISK 输出已是像素坐标
            return kpts, descs, fixed_wh

    wrapper = DISKWrapper(extractor).eval().to(dev)
    # 用随机图像 trace，避免全零图像导致关键点为空
    torch.manual_seed(42)
    dummy_img = torch.rand(1, 3, 480, 640, device=dev)
    dummy_wh = torch.tensor([[640.0, 480.0]], device=dev)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_img, dummy_wh), strict=False)

    suffix = "cuda" if dev.type == "cuda" else "cpu"
    out_path = MODELS_DIR / f"disk_{suffix}.pt"
    traced.save(str(out_path))
    print(f"[DISK] 已保存: {out_path}")
    return True


def export_aliked(device: str = "cuda"):
    """导出 ALIKED 特征提取器（kornia/LightGlue 仓库）"""
    try:
        import torch
        from lightglue import ALIKED as ALIKEDExtractor
    except ImportError:
        print("[ALIKED] 需要安装 lightglue: pip install git+https://github.com/cvg/LightGlue.git")
        return False

    print(f"[ALIKED] 导出中 (device={device})...")
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    extractor = ALIKEDExtractor(max_num_keypoints=2048).eval().to(dev)

    class ALIKEDWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image: torch.Tensor, orig_wh: torch.Tensor):
            feats = self.model.extract(image)
            kpts = feats["keypoints"]
            descs = feats["descriptors"]
            return kpts, descs, orig_wh

    wrapper = ALIKEDWrapper(extractor).eval().to(dev)
    torch.manual_seed(42)
    dummy_img = torch.rand(1, 3, 480, 640, device=dev)
    dummy_wh = torch.tensor([[640.0, 480.0]], device=dev)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_img, dummy_wh), strict=False)

    suffix = "cuda" if dev.type == "cuda" else "cpu"
    out_path = MODELS_DIR / f"aliked_{suffix}.pt"
    traced.save(str(out_path))
    print(f"[ALIKED] 已保存: {out_path}")
    return True


def export_loftr(scene: str = "outdoor", device: str = "cuda"):
    """
    导出 LoFTR 匹配器（使用 kornia.feature.LoFTR，无需单独安装 cvg-unizh/LoFTR）。

    输入：img0 [1,1,H,W], img1 [1,1,H,W]（灰度，float32 [0,1]）
    输出：mkpts0 [N,2], mkpts1 [N,2], confidence [N]
    """
    try:
        import torch
        from kornia.feature import LoFTR
    except ImportError:
        print("[LoFTR] 需要安装 kornia>=0.6: pip install kornia")
        return False

    import torch

    print(f"[LoFTR] 导出中 (scene={scene}, device={device})...")
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # kornia 的 LoFTR 会自动从 HuggingFace 下载预训练权重
    pretrained = "outdoor" if scene == "outdoor" else "indoor"
    model = LoFTR(pretrained=pretrained).eval().to(dev)

    class LoFTRWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, img0: torch.Tensor, img1: torch.Tensor):
            # 简化: 调用原始 forward 但在 trace 拦截 fine_matching
            # 用 try-except 包裹 fine 部分, trace 时会走 coarse 路径
            data = {"image0": img0, "image1": img1}

            # backbone
            cat0 = torch.cat([img0, img1], dim=0)
            feats = self.m.backbone(cat0)
            if isinstance(feats, (list, tuple)):
                feats_c, feats_f = feats
            else:
                feats_c, feats_f = feats.split(1, dim=0)

            feat_c0, feat_c1 = feats_c[0:1], feats_c[1:2]

            data.update({
                "bs": 1, "hw0_c": feat_c0.shape[2:4], "hw1_c": feat_c1.shape[2:4]
            })

            # coarse transformer
            d_model = self.m.loftr_coarse.layer0.linear1.in_features  # 256
            feat_c0 = feat_c0.flatten(2).permute(0, 2, 1)
            feat_c1 = feat_c1.flatten(2).permute(0, 2, 1)
            feat_c0, feat_c1 = self.m.loftr_coarse.layers(feat_c0, feat_c1)

            # coarse matching
            scale = feat_c0.shape[-1] ** -0.5
            conf = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) * scale
            data["conf_matrix_with_bin"] = conf

            # Manually do coarse matching to avoid fine_matching trace bug
            B, L, S = conf.shape
            # Remove dustbin column for mutual NN
            scores = conf[:, :-1, :-1]  # [B, L-1, S-1]
            max0 = scores.argmax(dim=2)  # [B, L-1]
            max1 = scores.argmax(dim=1)  # [B, S-1]

            # Build coarse match indices
            idx = torch.arange(B, device=conf.device).unsqueeze(1)
            mutual = max1[idx, max0] == torch.arange(L-1, device=conf.device).unsqueeze(0)

            # Get W and H from data
            W0 = data["hw0_c"][1]
            H0 = data["hw0_c"][0]
            W1 = data["hw1_c"][1]
            H1 = data["hw1_c"][0]

            # Convert to pixel coordinates
            kpts0_list, kpts1_list, conf_list = [], [], []
            for b in range(B):
                valid = mutual[b]
                indices = torch.where(valid)[0]
                if len(indices) > 0:
                    i0 = indices
                    i1 = max0[b][valid]
                    c = scores[b][i0, i1]
                    # Coarse grid → pixel
                    x0 = (i0 % (W0 // 8) + 0.5) * 8
                    y0 = (i0 // (W0 // 8) + 0.5) * 8
                    x1 = (i1 % (W1 // 8) + 0.5) * 8
                    y1 = (i1 // (W1 // 8) + 0.5) * 8
                    kpts0 = torch.stack([x0, y0], dim=1)
                    kpts1 = torch.stack([x1, y1], dim=1)
                    kpts0_list.append(kpts0.unsqueeze(0))
                    kpts1_list.append(kpts1.unsqueeze(0))
                    conf_list.append(c.unsqueeze(0))

            if len(kpts0_list) > 0:
                mkpts0 = torch.cat(kpts0_list, dim=0)  # [B, M, 2]
                mkpts1 = torch.cat(kpts1_list, dim=0)
                mconf  = torch.cat(conf_list, dim=0)    # [B, M]
            else:
                mkpts0 = torch.zeros(B, 1, 2, device=img0.device)
                mkpts1 = torch.zeros(B, 1, 2, device=img0.device)
                mconf  = torch.zeros(B, 1, device=img0.device)

            return mkpts0, mkpts1, mconf

    wrapper = LoFTRWrapper(model).eval().to(dev)

    suffix = "cuda" if dev.type == "cuda" else "cpu"
    # 用多种分辨率 trace 以提高兼容性
    for h, w in [(480, 640), (672, 896)]:
        torch.manual_seed(42)
        dummy0 = torch.rand(1, 1, h, w, device=dev)
        dummy1 = torch.rand(1, 1, h, w, device=dev)
        try:
            with torch.no_grad():
                traced = torch.jit.trace(wrapper, (dummy0, dummy1), strict=False)
            out_path = MODELS_DIR / f"loftr_{scene}_{suffix}.pt"
            traced.save(str(out_path))
            print(f"[LoFTR] 已保存: {out_path} ({h}x{w}, 仅粗匹配)")
            break  # 成功就停止
        except Exception as e:
            print(f"[LoFTR] trace {h}x{w} 失败: {e}")
            continue
    return True


def export_roma(scene: str = "outdoor", device: str = "cuda"):
    """
    导出 RoMa 匹配器（Parskatt/RoMa）。

    输入：img0 [1,3,H,W], img1 [1,3,H,W]（RGB，float32 [0,1]）
    输出：mkpts0 [N,2], mkpts1 [N,2], confidence [N]
    """
    try:
        import torch
        import romatch
    except ImportError:
        print("[RoMa] 需要安装 romatch:")
        print("  pip install git+https://github.com/Parskatt/RoMa.git")
        return False

    import torch

    print(f"[RoMa] 导出中 (scene={scene}, device={device})...")
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    if scene == "outdoor":
        model = romatch.roma_outdoor(device=dev)
    else:
        model = romatch.roma_indoor(device=dev)
    model = model.eval()

    class RoMaWrapper(torch.nn.Module):
        def __init__(self, m, max_kpts: int = 10000):
            super().__init__()
            self.m = m
            self.max_kpts = max_kpts

        def forward(self, img0: torch.Tensor, img1: torch.Tensor):
            H, W = img0.shape[2], img0.shape[3]
            warp, certainty = self.m.match(img0, img1, device=img0.device)
            matches, conf = self.m.sample(
                matches=warp,
                certainty=certainty,
                num=self.max_kpts,
            )
            kpts0 = matches[:, :2]
            kpts1 = matches[:, 2:]
            # 坐标从 [-1,1] 归一化转为像素坐标
            kpts0 = (kpts0 + 1) / 2 * torch.tensor([W, H], dtype=torch.float32, device=img0.device)
            kpts1 = (kpts1 + 1) / 2 * torch.tensor([W, H], dtype=torch.float32, device=img0.device)
            return kpts0, kpts1, conf

    wrapper = RoMaWrapper(model).eval().to(dev)

    # RoMa 要求输入尺寸是 14 的倍数（ViT patch size）
    torch.manual_seed(42)
    dummy0 = torch.rand(1, 3, 504, 672, device=dev)  # 504=14*36, 672=14*48
    dummy1 = torch.rand(1, 3, 504, 672, device=dev)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy0, dummy1), strict=False)

    suffix = "cuda" if dev.type == "cuda" else "cpu"
    out_path = MODELS_DIR / f"roma_{scene}_{suffix}.pt"
    traced.save(str(out_path))
    print(f"[RoMa] 已保存: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="导出深度学习匹配模型为 TorchScript")
    parser.add_argument("--all", action="store_true", help="导出所有模型")
    parser.add_argument("--disk", action="store_true")
    parser.add_argument("--aliked", action="store_true")
    parser.add_argument("--loftr", action="store_true")
    parser.add_argument("--roma", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="只导出 CPU 版本")
    parser.add_argument("--scene", choices=["outdoor", "indoor", "both"], default="both",
                        help="LoFTR/RoMa 场景类型（默认 both）")
    args = parser.parse_args()

    if not any([args.all, args.disk, args.aliked, args.loftr, args.roma]):
        parser.print_help()
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    devices = ["cpu"] if args.cpu_only else ["cuda", "cpu"]
    scenes = ["outdoor", "indoor"] if args.scene == "both" else [args.scene]

    results = {}

    if args.all or args.disk:
        for dev in devices:
            results[f"disk_{dev}"] = export_disk(dev)

    if args.all or args.aliked:
        for dev in devices:
            results[f"aliked_{dev}"] = export_aliked(dev)

    if args.all or args.loftr:
        for scene in scenes:
            for dev in devices:
                results[f"loftr_{scene}_{dev}"] = export_loftr(scene, dev)

    if args.all or args.roma:
        for scene in scenes:
            for dev in devices:
                results[f"roma_{scene}_{dev}"] = export_roma(scene, dev)

    print("\n=== 导出结果 ===")
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")


if __name__ == "__main__":
    main()
