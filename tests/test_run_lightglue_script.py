import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_run_lightglue():
    spec = importlib.util.spec_from_file_location("run_lightglue", ROOT / "scripts/run_lightglue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunLightGlueScriptTest(unittest.TestCase):
    def test_read_feature_file_supports_version_two_keypoint_geometry(self):
        module = load_run_lightglue()
        with tempfile.TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "a.sift"
            with feature_path.open("wb") as f:
                f.write(b"SFTB")
                f.write(struct.pack("<I", 2))
                name = b"a.jpg"
                f.write(struct.pack("<I", len(name)))
                f.write(name)
                f.write(struct.pack("<I", 2))
                f.write(struct.pack("<fffff", 1.0, 2.0, 0.5, 4.0, 90.0))
                f.write(struct.pack("<fffff", 3.0, 4.0, 0.6, 8.0, 180.0))
                f.write(struct.pack("<I", 2))
                f.write(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32).tobytes())

            data = module.read_feature_file(str(feature_path))

            self.assertEqual(data["version"], 2)
            np.testing.assert_allclose(data["keypoints"], np.array([[1.0, 2.0], [3.0, 4.0]],
                                                                   dtype=np.float32))
            np.testing.assert_allclose(data["scales"], np.array([4.0, 8.0], dtype=np.float32))
            np.testing.assert_allclose(data["orientations"], np.array([90.0, 180.0],
                                                                      dtype=np.float32))

    def test_read_feature_file_keeps_version_one_compatibility(self):
        module = load_run_lightglue()
        with tempfile.TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "legacy.sift"
            with feature_path.open("wb") as f:
                f.write(b"SFTB")
                f.write(struct.pack("<I", 1))
                name = b"legacy.jpg"
                f.write(struct.pack("<I", len(name)))
                f.write(name)
                f.write(struct.pack("<I", 1))
                f.write(struct.pack("<fff", 1.0, 2.0, 0.5))
                f.write(struct.pack("<I", 2))
                f.write(np.array([[0.1, 0.2]], dtype=np.float32).tobytes())

            data = module.read_feature_file(str(feature_path))

            self.assertEqual(data["version"], 1)
            np.testing.assert_allclose(data["keypoints"], np.array([[1.0, 2.0]],
                                                                   dtype=np.float32))
            np.testing.assert_allclose(data["scales"], np.array([1.0], dtype=np.float32))
            np.testing.assert_allclose(data["orientations"], np.array([-1.0], dtype=np.float32))

    def test_read_feature_file_supports_version_three_image_size(self):
        module = load_run_lightglue()
        with tempfile.TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "v3.dsk"
            with feature_path.open("wb") as f:
                f.write(b"DSKB")
                f.write(struct.pack("<I", 3))
                name = b"v3.jpg"
                f.write(struct.pack("<I", len(name)))
                f.write(name)
                f.write(struct.pack("<ii", 640, 480))
                f.write(struct.pack("<I", 1))
                f.write(struct.pack("<fffff", 1.0, 2.0, 0.5, 4.0, 90.0))
                f.write(struct.pack("<I", 2))
                f.write(np.array([[0.1, 0.2]], dtype=np.float32).tobytes())

            data = module.read_feature_file(str(feature_path))

            self.assertEqual(data["version"], 3)
            self.assertEqual(data["image_width"], 640)
            self.assertEqual(data["image_height"], 480)
            np.testing.assert_allclose(module.estimate_image_size(data),
                                       np.array([640.0, 480.0], dtype=np.float32))

    def test_sift_uses_sift_lightglue_backend_before_generic_128d(self):
        module = load_run_lightglue()

        candidates = module.lightglue_feature_candidates("sift", 128)

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0], "sift")
        self.assertIn("auto_128d", candidates)

    def test_sgmt_scores_use_qdatastream_double_precision(self):
        module = load_run_lightglue()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "a__b.match"
            data0 = {
                "n_keypoints": 2,
                "keypoints": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            }
            data1 = {
                "n_keypoints": 3,
                "keypoints": np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
            }
            pairs = np.array([[0, 2], [1, 1]], dtype=np.int64)
            scores = np.array([0.75, 0.5], dtype=np.float32)

            module.write_sgmt_match(str(out_path), "a.dsk", "b.dsk", data0, data1,
                                    pairs, scores, "disk", "lightglue")

            raw = out_path.read_bytes()
            offset = 4 + 4
            name0_len = struct.unpack(">I", raw[offset:offset + 4])[0]
            offset += 4 + name0_len
            name1_len = struct.unpack(">I", raw[offset:offset + 4])[0]
            offset += 4 + name1_len
            self.assertEqual(struct.unpack(">iii", raw[offset:offset + 12]), (2, 2, 3))
            offset += 12

            expected_size = offset + (data0["n_keypoints"] + data1["n_keypoints"]) * 12
            self.assertEqual(len(raw), expected_size)
            self.assertEqual(struct.unpack(">id", raw[offset:offset + 12]), (2, 0.75))

            sidecar = json.loads(Path(str(out_path) + ".json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["feature_format_version"], 2)
            self.assertEqual(sidecar["lightglue_keypoint_budget"], 3)
            self.assertEqual(sidecar["matched_indices0"], [0, 1])
            self.assertEqual(sidecar["matched_indices1"], [2, 1])
            self.assertEqual(sidecar["matched_scores"], [0.75, 0.5])


if __name__ == "__main__":
    unittest.main()
