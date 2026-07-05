import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = Path(
    os.environ.get(
        "PLASCAN_FEATURE_MATCH_CLI",
        REPO_ROOT / "build" / "bin" / "feature_match_cli",
    )
)


def write_sift_feature(path: Path, image_name: str, keypoints, descriptors) -> None:
    with path.open("wb") as handle:
        handle.write(b"SFTB")
        handle.write(struct.pack("<I", 1))
        name_bytes = image_name.encode("utf-8")
        handle.write(struct.pack("<I", len(name_bytes)))
        handle.write(name_bytes)
        handle.write(struct.pack("<I", len(keypoints)))
        for x, y, score in keypoints:
            handle.write(struct.pack("<fff", x, y, score))
        desc_dim = len(descriptors[0]) if descriptors else 0
        handle.write(struct.pack("<I", desc_dim))
        for row in descriptors:
            handle.write(struct.pack("<" + "f" * desc_dim, *row))


class FeatureMatchCliSidecarTest(unittest.TestCase):
    def run_cli(self, args):
        if not CLI_PATH.exists():
            self.skipTest(f"feature_match_cli not found: {CLI_PATH}")
        return subprocess.run(
            [str(CLI_PATH), *map(str, args)],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_bf_sift_writes_ba_v2_sidecar_indices_and_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left.sift"
            right = root / "right.sift"
            out = root / "left__right.match"

            write_sift_feature(
                left,
                "left.jpg",
                [(10.0, 20.0, 0.9), (30.0, 40.0, 0.8)],
                [[1.0, 0.0], [0.0, 1.0]],
            )
            write_sift_feature(
                right,
                "right.jpg",
                [(11.0, 21.0, 0.95), (31.0, 41.0, 0.85)],
                [[1.0, 0.0], [0.0, 1.0]],
            )

            result = self.run_cli([
                "--algorithm", "bf",
                "--sp1", left,
                "--sp2", right,
                "--output", out,
                "--match-threshold", "0.0",
            ])

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            sidecar_path = Path(str(out) + ".json")
            self.assertTrue(sidecar_path.exists(), result.stdout + result.stderr)

            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["feature_format_version"], 2)
            self.assertEqual(sidecar["image0_name"], "left.jpg")
            self.assertEqual(sidecar["image1_name"], "right.jpg")
            self.assertEqual(sidecar["feature0_path"], str(left))
            self.assertEqual(sidecar["feature1_path"], str(right))
            self.assertEqual(sidecar["matched_indices0"], [0, 1])
            self.assertEqual(sidecar["matched_indices1"], [0, 1])
            self.assertEqual(sidecar["matched_points0"], [[10.0, 20.0], [30.0, 40.0]])
            self.assertEqual(sidecar["matched_points1"], [[11.0, 21.0], [31.0, 41.0]])
            self.assertEqual(sidecar["num_matches"], 2)
            self.assertEqual(len(sidecar["matched_scores"]), 2)


if __name__ == "__main__":
    unittest.main()
