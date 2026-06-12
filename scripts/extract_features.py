#!/usr/bin/env python3
"""
DISK / ALIKED 特征提取器，输出与 PlaScan 格式兼容的特征文件。

用法：
    python scripts/extract_features.py --algo disk   --images /path/to/images/*.jpg --output /path/to/output
    python scripts/extract_features.py --algo aliked --images /path/to/images/*.jpg --output /path/to/output

输出：每张图像对应一个算法特定的特征文件（.dsk / .alk）。
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

# 算法 → 文件后缀 + Magic Bytes 映射（与 C++ FeatureFileIO / ExtractorSuffix 一致）
ALGO_CONFIG = {
    "disk":   {"suffix": ".dsk", "magic": b"DSKB"},
    "aliked": {"suffix": ".alk", "magic": b"ALKB"},
}


def is_missing_lightglue_error(exc: ModuleNotFoundError) -> bool:
    missing_name = getattr(exc, "name", None)
    return missing_name == "lightglue" or "No module named 'lightglue'" in str(exc)


def add_vendored_lightglue_to_path(project_root: Path | None = None) -> Path | None:
    root = project_root if project_root is not None else Path(__file__).resolve().parents[1]
    vendored = root / "3rdparty" / "LightGlue-main"
    if not (vendored / "lightglue" / "__init__.py").exists():
        return None

    vendored_str = str(vendored)
    if vendored_str not in sys.path:
        sys.path.insert(0, vendored_str)
    return vendored


def load_extractor(algo: str, max_keypoints: int, device: torch.device):
    add_vendored_lightglue_to_path()

    if algo == "disk":
        from lightglue import DISK
        # DISK 的 max_num_keypoints 在关键点少于该值时会触发 kthvalue() 错误
        # 使用 None 避免该问题，稍后手动截断
        return DISK(max_num_keypoints=None).eval().to(device)
    elif algo == "aliked":
        from lightglue import ALIKED
        return ALIKED(max_num_keypoints=max_keypoints).eval().to(device)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def validate_grayscale_range(grayscale_min: float, grayscale_max: float) -> None:
    if not 0.0 <= grayscale_min <= grayscale_max <= 1.0:
        raise ValueError(
            "灰度阈值范围无效: "
            f"grayscale_min={grayscale_min}, grayscale_max={grayscale_max}; "
            "需要满足 0.0 <= min <= max <= 1.0"
        )


def filter_features_by_grayscale_range(kpts: np.ndarray, descs: np.ndarray, scores: np.ndarray,
                                       img_f: np.ndarray, grayscale_min: float,
                                       grayscale_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_grayscale_range(grayscale_min, grayscale_max)
    if len(kpts) == 0 or (grayscale_min <= 0.0 and grayscale_max >= 1.0):
        return kpts, descs, scores

    if img_f.ndim == 3:
        gray_f = img_f.mean(axis=2)
    else:
        gray_f = img_f

    height, width = gray_f.shape[:2]
    pixel_xy = np.rint(kpts).astype(np.int64)
    x = pixel_xy[:, 0]
    y = pixel_xy[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    keep = np.zeros(len(kpts), dtype=bool)
    if np.any(inside):
        gray_values = gray_f[y[inside], x[inside]]
        keep[inside] = (gray_values >= grayscale_min) & (gray_values <= grayscale_max)

    return kpts[keep], descs[keep], scores[keep]


def extract(extractor, img_path: Path, device: torch.device,
            max_keypoints: int = -1, score_threshold: float = 0.0,
            grayscale_min: float = 0.0, grayscale_max: float = 1.0) -> dict:
    """返回 {'keypoints': np.ndarray [N,2], 'descriptors': np.ndarray [N,D], 'scores': np.ndarray [N]}"""
    validate_grayscale_range(grayscale_min, grayscale_max)

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # 处理单通道（灰度/遥感）图像：转为 3 通道 RGB
    if img_bgr.ndim == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    elif img_bgr.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_bgr[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif img_bgr.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 归一化到 [0, 1]（支持 16-bit 遥感影像）
    if img_rgb.dtype == np.uint16:
        img_f = img_rgb.astype(np.float32) / 65535.0
    else:
        img_f = img_rgb.astype(np.float32) / 255.0

    img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = extractor.extract(img_t)

    kpts = feats["keypoints"][0].cpu().numpy()       # [N, 2]
    descs = feats["descriptors"][0].cpu().numpy()    # [N, D]
    scores = feats.get("keypoint_scores", feats.get("scores", None))
    if scores is not None:
        scores = scores[0].cpu().numpy()
    else:
        scores = np.ones(len(kpts), dtype=np.float32)

    if score_threshold > 0 and len(kpts) > 0:
        keep = scores >= score_threshold
        kpts = kpts[keep]
        descs = descs[keep]
        scores = scores[keep]

    kpts, descs, scores = filter_features_by_grayscale_range(
        kpts,
        descs,
        scores,
        img_f,
        grayscale_min,
        grayscale_max,
    )

    # 手动截断（DISK 使用 max_num_keypoints=None 时需要）
    if max_keypoints > 0 and len(kpts) > max_keypoints:
        idx = np.argsort(scores)[::-1][:max_keypoints]
        kpts = kpts[idx]
        descs = descs[idx]
        scores = scores[idx]

    return {
        "keypoints": kpts.astype(np.float32),
        "descriptors": descs.astype(np.float32),
        "scores": scores.astype(np.float32),
        "image_width": img_bgr.shape[1],
        "image_height": img_bgr.shape[0],
    }


def save_features(data: dict, out_path: Path, magic: bytes):
    """
    保存特征文件（二进制格式，小端序，与 C++ FeatureFileIO::write 完全一致）。

    格式：
        magic:      char[4] (SPBT/DSKB/ALKB/etc.)
        version:    uint32 = 1
        name_len:   uint32
        name:       utf8[name_len]  (图像文件名)
        N:          uint32          (关键点数量)
        for i in N: x float32, y float32, score float32
        desc_dim:   uint32
        for i in N, j in D: float32
    """
    kpts = data["keypoints"]    # [N, 2] float32
    descs = data["descriptors"] # [N, D] float32
    scores = data["scores"]     # [N] float32
    N = len(kpts)
    D = descs.shape[1] if N > 0 else 0
    image_name = out_path.stem

    name_bytes = image_name.encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(magic)                                      # magic (算法特定)
        f.write(struct.pack("<I", 1))                       # version
        f.write(struct.pack("<I", len(name_bytes)))         # name_len
        f.write(name_bytes)                                 # name
        f.write(struct.pack("<I", N))                       # N
        for i in range(N):
            f.write(struct.pack("<fff", float(kpts[i, 0]), float(kpts[i, 1]), float(scores[i])))
        f.write(struct.pack("<I", D))                       # desc_dim
        if N > 0 and D > 0:
            f.write(descs.astype(np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser(description="DISK/ALIKED 特征提取")
    parser.add_argument("--algo", choices=["disk", "aliked"], required=True)
    parser.add_argument("--images", nargs="+", required=True, help="输入图像路径（支持通配符）")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--grayscale-min", type=float, default=0.0)
    parser.add_argument("--grayscale-max", type=float, default=1.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    try:
        validate_grayscale_range(args.grayscale_min, args.grayscale_max)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    cfg = ALGO_CONFIG.get(args.algo)
    if cfg is None:
        print(f"不支持的算法: {args.algo}", file=sys.stderr)
        sys.exit(1)

    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[{args.algo.upper()}] 使用设备: {dev}")

    try:
        extractor = load_extractor(args.algo, args.max_keypoints, dev)
    except ModuleNotFoundError as exc:
        if not is_missing_lightglue_error(exc):
            raise
        print(
            f"[{args.algo.upper()}] 无法加载 Python 依赖 lightglue。"
            "DISK/ALIKED 特征提取需要 lightglue；未生成替代 DISK/ALIKED 输出。"
            "请使用包含 lightglue 的 Python 环境，例如执行: "
            "python -m pip install git+https://github.com/cvg/LightGlue.git；"
            "或在 GUI 中选择 SuperPoint/SIFT/ORB/AKAZE/SURF。",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 展开通配符
    import glob
    image_paths = []
    for pattern in args.images:
        expanded = glob.glob(pattern)
        if expanded:
            image_paths.extend(expanded)
        else:
            image_paths.append(pattern)

    suffix = cfg["suffix"]
    magic = cfg["magic"]

    print(f"[{args.algo.upper()}] 处理 {len(image_paths)} 张图像...")
    print(f"[{args.algo.upper()}] 灰度阈值: [{args.grayscale_min}, {args.grayscale_max}]")
    failed = []
    for i, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        out_path = out_dir / (img_path.stem + suffix)
        try:
            data = extract(
                extractor,
                img_path,
                dev,
                args.max_keypoints,
                args.score_threshold,
                args.grayscale_min,
                args.grayscale_max,
            )
            save_features(data, out_path, magic)
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name} → {data['keypoints'].shape[0]} 个关键点 → {out_path.name}")
        except Exception as e:
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name} 失败: {e}", file=sys.stderr)
            failed.append(img_path)

    print(f"\n完成: {len(image_paths) - len(failed)}/{len(image_paths)} 成功")
    if failed:
        print(f"失败: {[str(p) for p in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
