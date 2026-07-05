#!/usr/bin/env python3
"""LoFTR 端到端匹配 (无 TorchScript, 直接调用 kornia)
用法: python scripts/run_loftr.py -L img1.tif -R img2.tif -o out.match [--cuda]
"""
import argparse, cv2, torch, struct, time, sys
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-L', '--left', required=True)
    p.add_argument('-R', '--right', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--max-dim', type=int, default=1200)
    p.add_argument('--min-conf', type=float, default=0.2)
    args = p.parse_args()

    dev = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"LoFTR matching on {dev}")

    from kornia.feature import LoFTR
    model = LoFTR(pretrained='outdoor').eval().to(dev)

    img1 = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print(f"ERROR: cannot load images", file=sys.stderr); sys.exit(2)

    ow, oh = img1.shape[1], img1.shape[0]
    max_side = max(ow, oh)
    scale = 1.0
    if max_side > args.max_dim:
        scale = args.max_dim / max_side
        nw, nh = int(ow * scale), int(oh * scale)
        # Make divisible by 32 for LoFTR
        nw = (nw // 32) * 32
        nh = (nh // 32) * 32
        img1 = cv2.resize(img1, (nw, nh))
        img2 = cv2.resize(img2, (nw, nh))
        print(f"Resized: {ow}x{oh} -> {nw}x{nh}")

    t0 = time.time()
    t1 = torch.from_numpy(img1.astype(np.float32)/255).unsqueeze(0).unsqueeze(0).to(dev)
    t2 = torch.from_numpy(img2.astype(np.float32)/255).unsqueeze(0).unsqueeze(0).to(dev)
    t_load = time.time()

    with torch.no_grad():
        out = model({"image0": t1, "image1": t2})
    t_infer = time.time()

    mkpts0 = out["keypoints0"].cpu().numpy()
    mkpts1 = out["keypoints1"].cpu().numpy()
    mconf  = out["confidence"].cpu().numpy()

    # Filter by confidence
    mask = mconf > args.min_conf
    mkpts0, mkpts1, mconf = mkpts0[mask], mkpts1[mask], mconf[mask]
    n = len(mkpts0)

    # Save .match (SGMT binary format, BigEndian + double scores, matching QDataStream)
    scale_back_pts0 = mkpts0 / scale if scale != 1.0 else mkpts0
    scale_back_pts1 = mkpts1 / scale if scale != 1.0 else mkpts1

    with open(args.output, 'wb') as f:
        # SGMT header
        f.write(b'SGMT')
        f.write(struct.pack('>I', 1))  # version
        # Image names
        import os
        name0 = os.path.splitext(os.path.basename(args.left))[0].encode('utf-8')
        name1 = os.path.splitext(os.path.basename(args.right))[0].encode('utf-8')
        f.write(struct.pack('>I', len(name0))); f.write(name0)
        f.write(struct.pack('>I', len(name1))); f.write(name1)
        # Match stats
        f.write(struct.pack('>i', n))     # numMatches
        f.write(struct.pack('>i', n))     # num_keypoints0
        f.write(struct.pack('>i', n))     # num_keypoints1
        # Per-kp0: match_idx + score
        for i in range(n):
            f.write(struct.pack('>id', i, float(mconf[i])))
        # Per-kp1: match_idx + score
        for i in range(n):
            f.write(struct.pack('>id', i, float(mconf[i])))

    # Write sidecar .match.json (required by PlaScan viewer for coord loading)
    import json
    sidecar = {
        "match_file": args.output,
        "image0_path": args.left, "image1_path": args.right,
        "image0_name": os.path.splitext(os.path.basename(args.left))[0],
        "image1_name": os.path.splitext(os.path.basename(args.right))[0],
        "num_matches": n,
        "match_algorithm": "loftr",
        "matched_points0": scale_back_pts0[:, :2].tolist(),
        "matched_points1": scale_back_pts1[:, :2].tolist()
    }
    with open(args.output + ".json", "w") as fj:
        json.dump(sidecar, fj)

    t_total = time.time() - t0
    print(f"LoFTR: {n} matches, load={t_load-t0:.1f}s infer={t_infer-t_load:.1f}s, saved to {args.output}")

if __name__ == '__main__':
    main()
