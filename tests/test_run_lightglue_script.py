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
            self.assertEqual(sidecar["matched_indices0"], [0, 1])
            self.assertEqual(sidecar["matched_indices1"], [2, 1])
            self.assertEqual(sidecar["matched_scores"], [0.75, 0.5])


if __name__ == "__main__":
    unittest.main()
