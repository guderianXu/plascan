#!/usr/bin/env python3
"""DeDoDe 特征提取 + NN 匹配 (kornia DeDoDe, 直接调用)
用法:
  python scripts/run_dedode.py -L a.tif -R b.tif -o out.match [--cuda]  (提取+匹配)
  python scripts/run_dedode.py -i img.tif -o out.dedode [--cuda]        (仅提取)
"""
import argparse, cv2, torch, struct, time, sys, os, numpy as np

def mutual_nn(descs0, descs1, kpts0, kpts1, min_score=0.2, device='cuda'):
    d0 = torch.from_numpy(descs0).to(device)
    d1 = torch.from_numpy(descs1).to(device)
    sim = torch.mm(d0, d1.t())
    vals, idx = sim.max(dim=1)
    _, idx_r = sim.max(dim=0)
    mutual = idx_r[idx] == torch.arange(len(idx), device=device)
    valid = mutual & (vals > min_score)
    i0 = valid.nonzero(as_tuple=True)[0].cpu().numpy()
    i1 = idx[valid].cpu().numpy()
    return kpts0[i0], kpts1[i1], vals[valid].cpu().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-L','--left')
    p.add_argument('-R','--right')
    p.add_argument('-i','--input',help='单张提取')
    p.add_argument('-o','--output',required=True)
    p.add_argument('--cuda',action='store_true')
    p.add_argument('--max-dim',type=int,default=1600)
    p.add_argument('--max-kp',type=int,default=8192)
    args = p.parse_args()

    dev = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
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
        return kps / s, sc, dc, s

    if args.input:
        img = cv2.imread(args.input)
        if img is None: print("E: load fail",file=sys.stderr); sys.exit(2)
        kps, sc, dc, scale = extract(img)
        suffix = '.dedode'
        with open(args.output, 'wb') as f:
            f.write(b'DEDE'); f.write(struct.pack('<I',1))
            name = os.path.basename(args.input).encode(); f.write(struct.pack('<I',len(name))); f.write(name)
            f.write(struct.pack('<I',len(kps)))
            for i in range(len(kps)): f.write(struct.pack('<fff',kps[i][0],kps[i][1],sc[i]))
            D = dc.shape[1]; f.write(struct.pack('<I',D))
            f.write(dc.astype(np.float32).tobytes())
        print(f"DeDoDe: {len(kps)} kp {D}d load={t1-t0:.1f}s -> {args.output}")
        return

    if not args.left or not args.right: print("E: need -L -R",file=sys.stderr); sys.exit(1)
    img1 = cv2.imread(args.left); img2 = cv2.imread(args.right)
    if img1 is None or img2 is None: print("E: load",file=sys.stderr); sys.exit(2)

    kps1, _, dc1, s1 = extract(img1)
    kps2, _, dc2, s2 = extract(img2)
    t2 = time.time()
    pts1, pts2, conf = mutual_nn(dc1, dc2, kps1, kps2, device=str(dev))
    t3 = time.time()

    with open(args.output, 'wb') as f:
        f.write(struct.pack('>i', len(pts1)))
        for i in range(len(pts1)):
            f.write(struct.pack('>ffff', pts1[i][0], pts1[i][1], pts2[i][0], pts2[i][1]))
    print(f"DeDoDe+NN: {len(pts1)} matches extract={t2-t1:.1f}s match={t3-t2:.1f}s -> {args.output}")

if __name__ == '__main__':
    main()
