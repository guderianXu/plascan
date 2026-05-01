#!/usr/bin/env python3
"""DISK/ALIKED 特征提取 + LightGlue 匹配 (无 TorchScript, 直接调用)
用法:
  python scripts/run_disk_aliked.py -a disk    -L a.tif -R b.tif -o out.match [--cuda]
  python scripts/run_disk_aliked.py -a aliked  -L a.tif -R b.tif -o out.match [--cuda]
  python scripts/run_disk_aliked.py -a disk    -i img.tif -o out.sp    [--cuda]  (仅提取)
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

def mutual_nn_match(descs0, descs1, kpts0, kpts1, min_score=0.2):
    """Mutual nearest neighbor + ratio test, return matched pixel coordinates"""
    descs0_t = torch.from_numpy(descs0).cuda()
    descs1_t = torch.from_numpy(descs1).cuda()
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

        # Save as .sp (reuse QFileBinaryIO format)
        # Write using SP binary format: magic, version, name_len, name, n_kp, kpts, desc_dim, descs
        with open(args.output, 'wb') as f:
            f.write(b'SPBT')
            f.write(struct.pack('<I', 1))  # version
            name = os.path.basename(args.input).encode('utf-8')
            f.write(struct.pack('<I', len(name)))
            f.write(name)
            n = len(kpts)
            f.write(struct.pack('<I', n))
            for i in range(n):
                f.write(struct.pack('<fff', float(kpts[i][0]), float(kpts[i][1]), float(scores[i]) if scores is not None else 1.0))
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

    pts1, pts2, conf = mutual_nn_match(descs1, descs2, kpts1, kpts2)
    t2 = time.time()

    n = len(pts1)
    with open(args.output, 'wb') as f:
        f.write(struct.pack('>i', n))
        for i in range(n):
            f.write(struct.pack('>ffff',
                pts1[i][0]/s, pts1[i][1]/s, pts2[i][0]/s, pts2[i][1]/s))

    print(f"{args.algo.upper()}+NN: {n} matches, extract={t1-t0:.1f}s match={t2-t1:.1f}s -> {args.output}")

if __name__ == '__main__':
    main()
