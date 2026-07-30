#!/usr/bin/env python3
"""DeDoDe 特征提取 + NN 匹配 (kornia DeDoDe, 直接调用)
用法:
  python scripts/workflows/run_dedode.py -L a.tif -R b.tif -o out.match [--cuda]  (提取+匹配)
  python scripts/workflows/run_dedode.py -i img.tif -o out.dedode [--cuda]        (仅提取)
  python scripts/workflows/run_dedode.py --feature-left a.dedode --feature-right b.dedode -o out.match
"""
import argparse, cv2, json, torch, struct, time, sys, os, numpy as np

def mutual_nn(descs0, descs1, kpts0, kpts1, min_score=0.2, device='cuda'):
    if descs0.size == 0 or descs1.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    d0 = torch.nn.functional.normalize(torch.from_numpy(descs0).float().to(device), p=2, dim=1)
    d1 = torch.nn.functional.normalize(torch.from_numpy(descs1).float().to(device), p=2, dim=1)
    sim = torch.mm(d0, d1.t())
    vals, idx = sim.max(dim=1)
    _, idx_r = sim.max(dim=0)
    mutual = idx_r[idx] == torch.arange(len(idx), device=device)
    valid = mutual & (vals > min_score)
    i0 = valid.nonzero(as_tuple=True)[0].cpu().numpy()
    i1 = idx[valid].cpu().numpy()
    return kpts0[i0], kpts1[i1], vals[valid].cpu().numpy()


def write_feature_file(path, image_path, kpts, scores, descs, image_width, image_height):
    name = os.path.basename(image_path).encode("utf-8")
    kpts = np.asarray(kpts, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    descs = np.asarray(descs, dtype=np.float32)
    n = int(kpts.shape[0])
    d = int(descs.shape[1]) if n > 0 and descs.ndim == 2 else 0
    with open(path, "wb") as f:
        f.write(b"DEDE")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<I", len(name)))
        f.write(name)
        f.write(struct.pack("<ii", int(image_width), int(image_height)))
        f.write(struct.pack("<I", n))
        for i in range(n):
            score = float(scores[i]) if i < len(scores) else 1.0
            f.write(struct.pack("<fffff", float(kpts[i, 0]), float(kpts[i, 1]), score, 8.0, -1.0))
        f.write(struct.pack("<I", d))
        if n > 0 and d > 0:
            f.write(descs.astype(np.float32).tobytes())


def read_feature_file(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"DEDE":
            raise ValueError(f"not a DeDoDe feature file: {path}")
        version = struct.unpack("<I", f.read(4))[0]
        name_len = struct.unpack("<I", f.read(4))[0]
        name = f.read(name_len).decode("utf-8", errors="replace")
        image_width = 0
        image_height = 0
        if version >= 3:
            image_width, image_height = struct.unpack("<ii", f.read(8))
        n = struct.unpack("<I", f.read(4))[0]
        kpts = np.zeros((n, 2), dtype=np.float32)
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if version >= 2:
                x, y, score, _, _ = struct.unpack("<fffff", f.read(20))
            else:
                x, y, score = struct.unpack("<fff", f.read(12))
            kpts[i] = [x, y]
            scores[i] = score
        d = struct.unpack("<I", f.read(4))[0]
        if n > 0 and d > 0:
            descs = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d).copy()
        else:
            descs = np.zeros((n, 0), dtype=np.float32)
    return {
        "path": os.path.abspath(path),
        "name": name,
        "keypoints": kpts,
        "scores": scores,
        "descriptors": descs,
        "image_width": image_width,
        "image_height": image_height,
    }


def write_sgmt_match(out_path, image0_name, image1_name, pts0, pts1, conf,
                     algorithm="dedode", feature0_path=None, feature1_path=None,
                     image0_path=None, image1_path=None):
    pts0 = np.asarray(pts0, dtype=np.float32)
    pts1 = np.asarray(pts1, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    n = int(min(len(pts0), len(pts1), len(conf)))
    name0 = os.path.splitext(os.path.basename(image0_name))[0].encode("utf-8")
    name1 = os.path.splitext(os.path.basename(image1_name))[0].encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(b"SGMT")
        f.write(struct.pack(">I", 1))
        f.write(struct.pack(">I", len(name0))); f.write(name0)
        f.write(struct.pack(">I", len(name1))); f.write(name1)
        f.write(struct.pack(">i", n))
        f.write(struct.pack(">i", n))
        f.write(struct.pack(">i", n))
        for i in range(n):
            f.write(struct.pack(">id", i, float(conf[i])))
        for i in range(n):
            f.write(struct.pack(">id", i, float(conf[i])))

    sidecar = {
        "match_file": os.path.abspath(out_path),
        "image0_name": name0.decode("utf-8"),
        "image1_name": name1.decode("utf-8"),
        "feature_algorithm": "dedode",
        "match_algorithm": algorithm,
        "backend": "python_dedode",
        "feature_format_version": 2,
        "num_matches": n,
        "matched_points0": pts0[:n, :2].astype(float).tolist(),
        "matched_points1": pts1[:n, :2].astype(float).tolist(),
        "matched_indices0": list(range(n)),
        "matched_indices1": list(range(n)),
        "matched_scores": conf[:n].astype(float).tolist(),
    }
    if feature0_path:
        sidecar["feature0_path"] = os.path.abspath(feature0_path)
        sidecar["sp0_path"] = os.path.abspath(feature0_path)
    if feature1_path:
        sidecar["feature1_path"] = os.path.abspath(feature1_path)
        sidecar["sp1_path"] = os.path.abspath(feature1_path)
    if image0_path:
        sidecar["image0_path"] = os.path.abspath(image0_path)
    if image1_path:
        sidecar["image1_path"] = os.path.abspath(image1_path)
    with open(out_path + ".json", "w", encoding="utf-8") as fj:
        json.dump(sidecar, fj, ensure_ascii=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-L','--left')
    p.add_argument('-R','--right')
    p.add_argument('-i','--input',help='单张提取')
    p.add_argument('--feature-left', help='左 DeDoDe 特征文件')
    p.add_argument('--feature-right', help='右 DeDoDe 特征文件')
    p.add_argument('-o','--output',required=True)
    p.add_argument('--cuda',action='store_true')
    p.add_argument('--max-dim',type=int,default=1600)
    p.add_argument('--max-kp',type=int,default=8192)
    p.add_argument('--min-score',type=float,default=0.2)
    args = p.parse_args()

    dev = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    if args.feature_left or args.feature_right:
        if not args.feature_left or not args.feature_right:
            print("E: need --feature-left and --feature-right together", file=sys.stderr)
            sys.exit(1)
        feat0 = read_feature_file(args.feature_left)
        feat1 = read_feature_file(args.feature_right)
        pts0, pts1, conf = mutual_nn(
            feat0["descriptors"],
            feat1["descriptors"],
            feat0["keypoints"],
            feat1["keypoints"],
            min_score=args.min_score,
            device=str(dev),
        )
        write_sgmt_match(
            args.output,
            feat0["name"],
            feat1["name"],
            pts0,
            pts1,
            conf,
            feature0_path=args.feature_left,
            feature1_path=args.feature_right,
        )
        print(f"DeDoDe+NN: {len(pts0)} matches -> {args.output}")
        return

    from kornia.feature import DeDoDe

    t0 = time.time()
    m = DeDoDe(amp_dtype=None).eval().to(dev)
    t1 = time.time()

    def extract(img_bgr):
        oh, ow = img_bgr.shape[:2]
        s = 1.0
        if max(ow,oh) > args.max_dim:
            s = args.max_dim / max(ow,oh)
            nw = (int(ow*s)//14)*14; nh = (int(oh*s)//14)*14
            img_bgr = cv2.resize(img_bgr,(nw,nh))
        t = torch.from_numpy(img_bgr.astype('float32')/255.0).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad():
            kps_t, scores_t, descs_t = m.forward(t)
        kps = kps_t[0].cpu().numpy()
        sc  = scores_t[0].cpu().numpy()
        dc  = descs_t[0].cpu().numpy()
        if args.max_kp > 0 and len(kps) > args.max_kp:
            top = np.argsort(sc)[-args.max_kp:]
            kps, sc, dc = kps[top], sc[top], dc[top]
        return kps / s, sc, dc, s, ow, oh

    if args.input:
        img = cv2.imread(args.input)
        if img is None: print("E: load fail",file=sys.stderr); sys.exit(2)
        kps, sc, dc, scale, ow, oh = extract(img)
        D = dc.shape[1] if len(kps) > 0 else 0
        write_feature_file(args.output, args.input, kps, sc, dc, ow, oh)
        print(f"DeDoDe: {len(kps)} kp {D}d load={t1-t0:.1f}s -> {args.output}")
        return

    if not args.left or not args.right: print("E: need -L -R",file=sys.stderr); sys.exit(1)
    img1 = cv2.imread(args.left); img2 = cv2.imread(args.right)
    if img1 is None or img2 is None: print("E: load",file=sys.stderr); sys.exit(2)

    kps1, _, dc1, s1, _, _ = extract(img1)
    kps2, _, dc2, s2, _, _ = extract(img2)
    t2 = time.time()
    pts1, pts2, conf = mutual_nn(dc1, dc2, kps1, kps2, min_score=args.min_score, device=str(dev))
    t3 = time.time()

    write_sgmt_match(
        args.output,
        args.left,
        args.right,
        pts1,
        pts2,
        conf,
        image0_path=args.left,
        image1_path=args.right,
    )
    print(f"DeDoDe+NN: {len(pts1)} matches extract={t2-t1:.1f}s match={t3-t2:.1f}s -> {args.output}")

if __name__ == '__main__':
    main()
