#!/usr/bin/env python3
"""LightGlue matcher backend for PlaScan feature files.

The script is used as a fallback when algorithm-specific TorchScript
LightGlue models are unavailable, especially for DISK/ALIKED descriptors.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


def add_vendored_lightglue_to_path():
    root = Path(__file__).resolve().parents[1]
    candidates = []
    env_repo = os.environ.get("LIGHTGLUE_REPO", "").strip()
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.extend([
        root / "3rdparty" / "LightGlue-main",
        root / "3rdparty" / "LightGlue",
        root / "third_party" / "LightGlue-main",
        root / "third_party" / "LightGlue",
    ])

    for candidate in candidates:
        if (candidate / "lightglue" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return str(candidate)
    return ""


def read_feature_file(path):
    """Read a PlaScan binary feature file (.sp/.dsk/.alk/.sift/.orb/.akz)."""
    with open(path, "rb") as f:
        magic = f.read(4).decode("ascii", errors="replace")
        version = struct.unpack("<I", f.read(4))[0]
        name_len = struct.unpack("<I", f.read(4))[0]
        name = f.read(name_len).decode("utf-8", errors="replace")
        n_keypoints = struct.unpack("<I", f.read(4))[0]

        keypoints = np.zeros((n_keypoints, 2), dtype=np.float32)
        scores = np.zeros(n_keypoints, dtype=np.float32)
        for idx in range(n_keypoints):
            x, y, score = struct.unpack("<fff", f.read(12))
            keypoints[idx] = [x, y]
            scores[idx] = score

        desc_dim = struct.unpack("<I", f.read(4))[0]
        if n_keypoints > 0 and desc_dim > 0:
            raw = f.read(n_keypoints * desc_dim * 4)
            descriptors = np.frombuffer(raw, dtype=np.float32).reshape(n_keypoints, desc_dim).copy()
        else:
            descriptors = np.zeros((n_keypoints, 0), dtype=np.float32)

    return {
        "path": os.path.abspath(path),
        "magic": magic,
        "version": version,
        "name": name,
        "keypoints": keypoints,
        "scores": scores,
        "descriptors": descriptors,
        "n_keypoints": n_keypoints,
        "desc_dim": int(desc_dim),
    }


def estimate_image_size(keypoints):
    if keypoints.size == 0:
        return np.array([1.0, 1.0], dtype=np.float32)
    width = max(float(np.max(keypoints[:, 0])) + 1.0, 1.0)
    height = max(float(np.max(keypoints[:, 1])) + 1.0, 1.0)
    return np.array([width, height], dtype=np.float32)


def torch_features(data, device, torch_module):
    keypoints = torch_module.from_numpy(data["keypoints"]).unsqueeze(0).to(device)
    descriptors = torch_module.from_numpy(data["descriptors"]).unsqueeze(0).to(device)
    scores = torch_module.from_numpy(data["scores"]).unsqueeze(0).to(device)
    image_size = torch_module.from_numpy(estimate_image_size(data["keypoints"])).unsqueeze(0).to(device)
    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "keypoint_scores": scores,
        "image_size": image_size,
    }


def lightglue_feature_candidates(feature_algorithm, desc_dim):
    feature_algorithm = (feature_algorithm or "").strip().lower()
    if feature_algorithm == "disk":
        return ["disk", f"auto_{desc_dim}d"]
    if feature_algorithm == "aliked":
        return ["aliked", f"auto_{desc_dim}d"]
    if feature_algorithm == "superpoint":
        return ["superpoint", f"auto_{desc_dim}d"]
    if desc_dim == 128:
        return ["disk", "aliked", "auto_128d"]
    if desc_dim == 256:
        return ["superpoint", "auto_256d"]
    return [f"auto_{desc_dim}d"]


def make_lightglue(LightGlue, feature_algorithm, desc_dim, threshold):
    errors = []
    seen = set()
    for feature_name in lightglue_feature_candidates(feature_algorithm, desc_dim):
        if feature_name in seen:
            continue
        seen.add(feature_name)
        try:
            try:
                return LightGlue(features=feature_name, filter_threshold=threshold), feature_name
            except TypeError:
                return LightGlue(features=feature_name), feature_name
        except Exception as exc:
            errors.append(f"{feature_name}: {exc}")
    raise RuntimeError("cannot create LightGlue matcher; tried " + "; ".join(errors))


def tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def squeeze_batch(array, preserve_pair_rows=False):
    if array is None:
        return None
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    elif not preserve_pair_rows and array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return array


def extract_score_array(result, n0, pairs):
    for key in ("scores", "matching_scores", "matching_scores0"):
        scores = squeeze_batch(tensor_to_numpy(result.get(key)))
        if scores is None:
            continue
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim == 0:
            continue
        if scores.shape[0] == len(pairs):
            return scores
        if scores.shape[0] == n0:
            return np.array([scores[int(q)] for q, _ in pairs], dtype=np.float32)
    return np.ones(len(pairs), dtype=np.float32)


def extract_match_pairs(result, n0, n1):
    matches = squeeze_batch(tensor_to_numpy(result.get("matches")), preserve_pair_rows=True)
    if matches is not None and matches.size > 0:
        matches = np.asarray(matches, dtype=np.int64)
        if matches.ndim == 2 and matches.shape[1] == 2:
            pairs = matches[(matches[:, 0] >= 0) & (matches[:, 1] >= 0)]
            pairs = pairs[(pairs[:, 0] < n0) & (pairs[:, 1] < n1)]
            scores = extract_score_array(result, n0, pairs)
            return pairs, scores

    matches0 = squeeze_batch(tensor_to_numpy(result.get("matches0")))
    if matches0 is None:
        raise RuntimeError("LightGlue output does not contain matches or matches0")

    matches0 = np.asarray(matches0, dtype=np.int64)
    if matches0.ndim == 2 and matches0.shape[1] == 2:
        pairs = matches0[(matches0[:, 0] >= 0) & (matches0[:, 1] >= 0)]
        pairs = pairs[(pairs[:, 0] < n0) & (pairs[:, 1] < n1)]
    else:
        matches0 = matches0.reshape(-1)
        valid = np.where((matches0 >= 0) & (matches0 < n1))[0]
        pairs = np.stack([valid, matches0[valid]], axis=1) if valid.size else np.zeros((0, 2), dtype=np.int64)

    scores = extract_score_array(result, n0, pairs)
    return pairs.astype(np.int64), scores.astype(np.float32)


def write_sgmt_match(out_path, feat_path0, feat_path1, data0, data1, pairs, scores, feature_algorithm, match_algorithm):
    name0 = os.path.splitext(os.path.basename(feat_path0))[0]
    name1 = os.path.splitext(os.path.basename(feat_path1))[0]
    name0_enc = name0.encode("utf-8")
    name1_enc = name1.encode("utf-8")
    n0 = int(data0["n_keypoints"])
    n1 = int(data1["n_keypoints"])

    matches0 = np.full(n0, -1, dtype=np.int32)
    matches1 = np.full(n1, -1, dtype=np.int32)
    scores0 = np.zeros(n0, dtype=np.float32)
    scores1 = np.zeros(n1, dtype=np.float32)
    for (query_idx, train_idx), score in zip(pairs, scores):
        query_idx = int(query_idx)
        train_idx = int(train_idx)
        score = float(score)
        matches0[query_idx] = train_idx
        matches1[train_idx] = query_idx
        scores0[query_idx] = score
        scores1[train_idx] = score

    with open(out_path, "wb") as f:
        f.write(b"SGMT")
        f.write(struct.pack(">I", 1))
        f.write(struct.pack(">I", len(name0_enc)))
        f.write(name0_enc)
        f.write(struct.pack(">I", len(name1_enc)))
        f.write(name1_enc)
        f.write(struct.pack(">i", int(len(pairs))))
        f.write(struct.pack(">i", n0))
        f.write(struct.pack(">i", n1))
        for match_idx, score in zip(matches0, scores0):
            f.write(struct.pack(">id", int(match_idx), float(score)))
        for match_idx, score in zip(matches1, scores1):
            f.write(struct.pack(">id", int(match_idx), float(score)))

    pts0 = data0["keypoints"][pairs[:, 0]].tolist() if len(pairs) else []
    pts1 = data1["keypoints"][pairs[:, 1]].tolist() if len(pairs) else []
    sidecar = {
        "match_file": os.path.abspath(out_path),
        "image0_name": name0,
        "image1_name": name1,
        "feature0_path": os.path.abspath(feat_path0),
        "feature1_path": os.path.abspath(feat_path1),
        "sp0_path": os.path.abspath(feat_path0),
        "sp1_path": os.path.abspath(feat_path1),
        "feature_algorithm": feature_algorithm,
        "match_algorithm": match_algorithm,
        "backend": "python_lightglue",
        "num_matches": int(len(pairs)),
        "matched_points0": [[float(p[0]), float(p[1])] for p in pts0],
        "matched_points1": [[float(p[0]), float(p[1])] for p in pts1],
    }
    with open(out_path + ".json", "w", encoding="utf-8") as fj:
        json.dump(sidecar, fj, ensure_ascii=False)


def match_pair(feat_path0, feat_path1, out_path, use_cuda, threshold, feature_algorithm, match_algorithm):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is not installed in this Python environment") from exc

    add_vendored_lightglue_to_path()
    try:
        from lightglue import LightGlue
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Python LightGlue backend is unavailable. Install LightGlue in PLASCAN_PYTHON "
            "or set LIGHTGLUE_REPO to a LightGlue checkout."
        ) from exc

    data0 = read_feature_file(feat_path0)
    data1 = read_feature_file(feat_path1)
    if data0["n_keypoints"] == 0 or data1["n_keypoints"] == 0:
        raise RuntimeError("empty feature file")
    if data0["desc_dim"] != data1["desc_dim"]:
        raise RuntimeError(f"descriptor dimension mismatch: {data0['desc_dim']} vs {data1['desc_dim']}")

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    matcher, feature_name = make_lightglue(LightGlue, feature_algorithm, data0["desc_dim"], threshold)
    matcher = matcher.eval().to(device)

    batch = {
        "image0": torch_features(data0, device, torch),
        "image1": torch_features(data1, device, torch),
    }
    with torch.no_grad():
        result = matcher(batch)

    pairs, scores = extract_match_pairs(result, data0["n_keypoints"], data1["n_keypoints"])
    write_sgmt_match(out_path, feat_path0, feat_path1, data0, data1,
                     pairs, scores, feature_algorithm, match_algorithm)
    return len(pairs), str(device), feature_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", required=True, help="Feature file for image 0")
    parser.add_argument("-f2", required=True, help="Feature file for image 1")
    parser.add_argument("-o", required=True, help="Output .match file")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--max-kpts", type=int, default=2048, help="Reserved for CLI compatibility")
    parser.add_argument("--feature-algorithm", default="auto")
    parser.add_argument("--match-algorithm", default="lightglue")
    args = parser.parse_args()

    try:
        count, device, feature_name = match_pair(args.f1, args.f2, args.o, args.cuda, args.threshold,
                                                 args.feature_algorithm, args.match_algorithm)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"LightGlue Python backend: features={feature_name} device={device} matches={count} output={args.o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
