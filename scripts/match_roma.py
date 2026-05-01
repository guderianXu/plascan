#!/usr/bin/env python3
"""
RoMa 特征匹配脚本，输出与 SuperGlue/LightGlue 相同格式的 .match 文件。

用法：
    conda activate plascan
    python scripts/match_roma.py --scene outdoor --pairs img1__img2 img3__img4 \
        --image-dir /path/to/images --output /path/to/output

输出：每个匹配对对应一个 .match 文件（JSON 格式）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import cv2


def load_model(scene: str, device: torch.device):
    import romatch
    if scene == "outdoor":
        return romatch.roma_outdoor(device=device)
    else:
        return romatch.roma_indoor(device=device)


def preprocess_image(img_path: Path, device: torch.device, target_size=(672, 504)):
    """加载图像并 resize 到 RoMa 要求的尺寸（14 的倍数）"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    t = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device), orig_w, orig_h, target_size[0], target_size[1]


def match_pair(model, img0_path: Path, img1_path: Path, device: torch.device,
               threshold: float = 0.05, max_keypoints: int = 10000) -> dict:
    t0, ow0, oh0, rw0, rh0 = preprocess_image(img0_path, device)
    t1, ow1, oh1, rw1, rh1 = preprocess_image(img1_path, device)

    with torch.no_grad():
        warp, certainty = model.match(t0, t1, device=device)
        matches, conf = model.sample(matches=warp, certainty=certainty, num=max_keypoints)

    matches = matches.cpu().numpy()
    conf = conf.cpu().numpy()

    # 过滤低置信度匹配
    mask = conf >= threshold
    matches = matches[mask]
    conf = conf[mask]

    # 坐标从 [-1,1] 转为原图像素坐标
    kpts0 = (matches[:, :2] + 1) / 2 * np.array([ow0, oh0])
    kpts1 = (matches[:, 2:] + 1) / 2 * np.array([ow1, oh1])

    return {
        "keypoints0": kpts0.tolist(),
        "keypoints1": kpts1.tolist(),
        "confidence": conf.tolist(),
        "num_matches": int(len(conf)),
    }


def main():
    parser = argparse.ArgumentParser(description="RoMa 特征匹配")
    parser.add_argument("--scene", choices=["outdoor", "indoor"], default="outdoor")
    parser.add_argument("--pairs", nargs="+", required=True,
                        help="匹配对，格式 img1__img2（不含扩展名）")
    parser.add_argument("--image-dir", required=True, help="图像目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-keypoints", type=int, default=10000)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--image-ext", default=".jpg",
                        help="图像扩展名（默认 .jpg）")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[RoMa] 使用设备: {dev}, 场景: {args.scene}")

    model = load_model(args.scene, dev).eval()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, pair in enumerate(args.pairs):
        parts = pair.split("__")
        if len(parts) != 2:
            print(f"  [{i+1}] 跳过无效匹配对格式: {pair}", file=sys.stderr)
            continue

        name0, name1 = parts

        # 自动查找图像文件（支持多种扩展名）
        def find_image(name):
            for ext in [args.image_ext, ".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                p = image_dir / (name + ext)
                if p.exists():
                    return p
            return None

        img0 = find_image(name0)
        img1 = find_image(name1)

        if img0 is None or img1 is None:
            print(f"  [{i+1}] 找不到图像: {name0} 或 {name1}", file=sys.stderr)
            failed.append(pair)
            continue

        out_path = out_dir / f"{pair}.match"
        try:
            result = match_pair(model, img0, img1, dev, args.threshold, args.max_keypoints)
            with open(out_path, "w") as f:
                json.dump(result, f)
            print(f"  [{i+1}/{len(args.pairs)}] {pair} → {result['num_matches']} 个匹配")
        except Exception as e:
            print(f"  [{i+1}/{len(args.pairs)}] {pair} 失败: {e}", file=sys.stderr)
            failed.append(pair)

    print(f"\n完成: {len(args.pairs) - len(failed)}/{len(args.pairs)} 成功")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
