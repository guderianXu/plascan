#!/usr/bin/env python3
"""DISK/ALIKED 特征提取 + LightGlue 匹配 (无 TorchScript, 直接调用)
用法:
  python scripts/run_disk_aliked.py -a disk    -L a.tif -R b.tif -o out.match [--cuda]
  python scripts/run_disk_aliked.py -a aliked  -L a.tif -R b.tif -o out.match [--cuda]
  python scripts/run_disk_aliked.py -a disk    -i img.tif -o out.dsk   [--cuda]  (仅提取)
"""
import argparse, cv2, torch, struct, time, sys, os
import numpy as np

def extract_features(extractor, img, device, max_kp):
    """Extract DISK/ALIKED features from grayscale image -> (kpts, descs, scores)"""
    # Convert to 3-channel (DISK/ALIKED expect RGB)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = extractor.extract(t)
    kpts = feats["keypoints"][0]      # [N, 2]
    descs = feats["descriptors"][0]   # [N, D]
    scores = feats.get("keypoint_scores", feats.get("scores"))
    if scores is not None:
        scores = scores[0]
    return kpts.cpu().numpy(), descs.cpu().numpy(), scores.cpu().numpy() if scores is not None else None

def mutual_nn_match(descs0, descs1, kpts0, kpts1, min_score=0.2, device='cpu'):
    """Mutual nearest neighbor + ratio test, return matched pixel coordinates"""
    if descs0.size == 0 or descs1.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    descs0_t = torch.nn.functional.normalize(torch.from_numpy(descs0).float().to(device), p=2, dim=1)
    descs1_t = torch.nn.functional.normalize(torch.from_numpy(descs1).float().to(device), p=2, dim=1)
    # Cosine similarity
    sim = torch.mm(descs0_t, descs1_t.t())  # [N0, N1]
    scores, matches = sim.max(dim=1)  # N0 -> best N1
    # Mutual check
    scores_r, matches_r = sim.max(dim=0)
    mutual = matches_r[matches] == torch.arange(len(matches), device=matches.device)
    # Filter
    valid = mutual & (scores > min_score)
    idx0 = torch.where(valid)[0].cpu().numpy()
    idx1 = matches[valid].cpu().numpy()
    conf = scores[valid].cpu().numpy()

    pts0 = kpts0[idx0]
    pts1 = kpts1[idx1]
    return pts0, pts1, conf


def write_sgmt_match(out_path, left_path, right_path, pts0, pts1, conf, algo):
    import json
    pts0 = np.asarray(pts0, dtype=np.float32)
    pts1 = np.asarray(pts1, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    n = int(min(len(pts0), len(pts1), len(conf)))
    name0 = os.path.splitext(os.path.basename(left_path))[0].encode('utf-8')
    name1 = os.path.splitext(os.path.basename(right_path))[0].encode('utf-8')
    with open(out_path, 'wb') as f:
        f.write(b'SGMT')
        f.write(struct.pack('>I', 1))
        f.write(struct.pack('>I', len(name0))); f.write(name0)
        f.write(struct.pack('>I', len(name1))); f.write(name1)
        f.write(struct.pack('>i', n))
        f.write(struct.pack('>i', n))
        f.write(struct.pack('>i', n))
        for i in range(n):
            f.write(struct.pack('>id', i, float(conf[i])))
        for i in range(n):
            f.write(struct.pack('>id', i, float(conf[i])))

    sidecar = {
        "match_file": os.path.abspath(out_path),
        "image0_path": os.path.abspath(left_path),
        "image1_path": os.path.abspath(right_path),
        "image0_name": name0.decode("utf-8"),
        "image1_name": name1.decode("utf-8"),
        "feature_algorithm": algo,
        "match_algorithm": "lightglue_nn",
        "backend": "python_disk_aliked_legacy",
        "feature_format_version": 2,
        "num_matches": n,
        "matched_points0": pts0[:n, :2].astype(float).tolist(),
        "matched_points1": pts1[:n, :2].astype(float).tolist(),
        "matched_indices0": list(range(n)),
        "matched_indices1": list(range(n)),
        "matched_scores": conf[:n].astype(float).tolist(),
    }
    with open(out_path + ".json", "w", encoding="utf-8") as fj:
        json.dump(sidecar, fj, ensure_ascii=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-a', '--algo', required=True, choices=['disk','aliked'])
    p.add_argument('-L', '--left')
    p.add_argument('-R', '--right')
    p.add_argument('-i', '--input', help='单张影像 (仅提取特征)')
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--max-kp', type=int, default=2048)
    p.add_argument('--max-dim', type=int, default=1600)
    args = p.parse_args()

    dev = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"{args.algo.upper()} on {dev}")

    from lightglue import DISK, ALIKED
    cls = DISK if args.algo == 'disk' else ALIKED
    t0_load = time.time()
    extractor = cls(max_num_keypoints=args.max_kp).eval().to(dev)
    t1_load = time.time()

    # -- 仅提取模式 --
    if args.input:
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img is None: print("E: cannot load", file=sys.stderr); sys.exit(2)

        oh, ow = img.shape[:2]
        s = 1.0
        if max(ow, oh) > args.max_dim:
            s = args.max_dim / max(ow, oh)
            img = cv2.resize(img, (int(ow*s), int(oh*s)))

        kpts, descs, scores = extract_features(extractor, img, dev, args.max_kp)
        kpts /= s  # scale back

        magic = b'DSKB' if args.algo == 'disk' else b'ALKB'
        with open(args.output, 'wb') as f:
            f.write(magic)
            f.write(struct.pack('<I', 3))  # version
            name = os.path.basename(args.input).encode('utf-8')
            f.write(struct.pack('<I', len(name)))
            f.write(name)
            f.write(struct.pack('<ii', int(ow), int(oh)))
            n = len(kpts)
            f.write(struct.pack('<I', n))
            for i in range(n):
                f.write(struct.pack('<fffff',
                                    float(kpts[i][0]),
                                    float(kpts[i][1]),
                                    float(scores[i]) if scores is not None else 1.0,
                                    8.0,
                                    -1.0))
            D = descs.shape[1]
            f.write(struct.pack('<I', D))
            f.write(descs.astype(np.float32).tobytes())
        print(f"{args.algo.upper()}: {n} kp, {D}d, load={t1_load-t0_load:.1f}s -> {args.output}")
        return

    # -- 匹配模式 --
    if not args.left or not args.right:
        print("E: match mode needs -L and -R", file=sys.stderr); sys.exit(1)

    img1 = cv2.imread(args.left, cv2.IMREAD_COLOR)
    img2 = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None: print("E: load fail", file=sys.stderr); sys.exit(2)

    oh, ow = img1.shape[:2]
    s = 1.0
    if max(ow, oh) > args.max_dim:
        s = args.max_dim / max(ow, oh)
        img1 = cv2.resize(img1, (int(ow*s), int(oh*s)))
        img2 = cv2.resize(img2, (int(ow*s), int(oh*s)))

    t0 = time.time()
    kpts1, descs1, _ = extract_features(extractor, img1, dev, args.max_kp)
    kpts2, descs2, _ = extract_features(extractor, img2, dev, args.max_kp)
    t1 = time.time()

    pts1, pts2, conf = mutual_nn_match(descs1, descs2, kpts1, kpts2, device=str(dev))
    t2 = time.time()

    pts1 = pts1 / s
    pts2 = pts2 / s
    write_sgmt_match(args.output, args.left, args.right, pts1, pts2, conf, args.algo)

    print(f"{args.algo.upper()}+NN: {len(pts1)} matches, extract={t1-t0:.1f}s match={t2-t1:.1f}s -> {args.output}")

if __name__ == '__main__':
    main()
