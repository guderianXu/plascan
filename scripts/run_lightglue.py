#!/usr/bin/env python3
"""LightGlue 通用匹配 — 支持任意特征类型的 Python 后端
用法: python scripts/run_lightglue.py -f1 A.sp -f2 B.sp -o out.match [--cuda]
      python scripts/run_lightglue.py -f1 A.dsk -f2 B.dsk -o out.match [--cuda]
"""
import argparse, struct, json, sys, os
import numpy as np
import torch, cv2


def read_feature_file(path):
    """Read PlaScan binary feature file (.sp/.dsk/.alk/.sift/.orb/.akz)"""
    with open(path, 'rb') as f:
        magic = f.read(4).decode('ascii', errors='replace')
        version = struct.unpack('<I', f.read(4))[0]
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8', errors='replace')
        N = struct.unpack('<I', f.read(4))[0]
        kpts = np.zeros((N, 2), dtype=np.float32)
        scores = np.zeros(N, dtype=np.float32)
        for i in range(N):
            x, y, s = struct.unpack('<fff', f.read(12))
            kpts[i] = [x, y]
            scores[i] = s
        D = struct.unpack('<I', f.read(4))[0]
        descs = np.frombuffer(f.read(N * D * 4), dtype=np.float32).reshape(N, D) if N > 0 and D > 0 else None
    return {'keypoints': kpts, 'descriptors': descs, 'scores': scores, 'name': name, 'N': N}


def match_pair(feat_path0, feat_path1, out_path, device='cuda', threshold=0.15, max_kpts=2048):
    """Match two feature files via LightGlue Python API (supports any descriptor dim)"""
    data0 = read_feature_file(feat_path0)
    data1 = read_feature_file(feat_path1)

    if data0['N'] == 0 or data1['N'] == 0:
        print(f"ERROR: empty feature file", file=sys.stderr)
        return 0

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    from lightglue import LightGlue
    from lightglue.utils import rbd

    D = data0['descriptors'].shape[1]
    lg = LightGlue(features=f'auto_{D}d').eval().to(dev)

    kpts0 = torch.from_numpy(data0['keypoints']).unsqueeze(0).to(dev)
    kpts1 = torch.from_numpy(data1['keypoints']).unsqueeze(0).to(dev)
    descs0 = torch.from_numpy(data0['descriptors']).unsqueeze(0).to(dev)
    descs1 = torch.from_numpy(data1['descriptors']).unsqueeze(0).to(dev)

    with torch.no_grad():
        result = lg({'image0': {'keypoints': kpts0, 'descriptors': descs0},
                     'image1': {'keypoints': kpts1, 'descriptors': descs1}})
        matches = result['matches0'][0].cpu().numpy()  # [N, 2] indices
        mscores = result.get('matching_scores0', result.get('scores'))
        if mscores is not None:
            mscores = mscores[0].cpu().numpy()

    valid = matches[:, 0] > -1
    matches = matches[valid]
    n = len(matches)

    # Write SGMT binary + sidecar JSON
    name0 = os.path.splitext(os.path.basename(feat_path0))[0]
    name0_enc = name0.encode('utf-8')
    name1 = os.path.splitext(os.path.basename(feat_path1))[0]
    name1_enc = name1.encode('utf-8')

    with open(out_path, 'wb') as f:
        f.write(b'SGMT')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(name0_enc))); f.write(name0_enc)
        f.write(struct.pack('<I', len(name1_enc))); f.write(name1_enc)
        f.write(struct.pack('<i', n)); f.write(struct.pack('<i', n)); f.write(struct.pack('<i', n))
        for i in range(n):
            score = float(mscores[i]) if mscores is not None else 1.0
            f.write(struct.pack('<if', int(matches[i, 0]), score))
        for i in range(n):
            score = float(mscores[i]) if mscores is not None else 1.0
            f.write(struct.pack('<if', int(matches[i, 1]), score))

    # Sidecar JSON
    pts0 = data0['keypoints'][matches[:, 0]].tolist()
    pts1 = data1['keypoints'][matches[:, 1]].tolist()
    sidecar = {
        "match_file": out_path,
        "image0_name": name0, "image1_name": name1,
        "sp0_path": os.path.abspath(feat_path0),
        "sp1_path": os.path.abspath(feat_path1),
        "num_matches": int(n),
        "match_algorithm": "lightglue_py",
        "matched_points0": [[float(p[0]), float(p[1])] for p in pts0],
        "matched_points1": [[float(p[0]), float(p[1])] for p in pts1]
    }
    with open(out_path + ".json", "w") as fj:
        json.dump(sidecar, fj)

    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f1', required=True, help="Feature file for image 0")
    ap.add_argument('-f2', required=True, help="Feature file for image 1")
    ap.add_argument('-o', required=True, help="Output .match file")
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--threshold', type=float, default=0.15)
    ap.add_argument('--max-kpts', type=int, default=2048)
    args = ap.parse_args()

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"LightGlue matching on {device}")

    n = match_pair(args.f1, args.f2, args.o, device, args.threshold, args.max_kpts)
    print(f"LightGlue: {n} matches saved to {args.o}")
