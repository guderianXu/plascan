import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validation"
    / "summarize_mvs_depth_provenance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_mvs_depth_provenance", SCRIPT_PATH
)
summarizer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = summarizer
SPEC.loader.exec_module(summarizer)


class SummarizeMvsDepthProvenanceTest(unittest.TestCase):
    def _record(self, root: Path, ref_index: int = 3) -> dict:
        artifact = root / f"depth_{ref_index}_provenance.png"
        artifact.write_bytes(b"png")
        return {
            "ref_index": ref_index,
            "status": "completed",
            "depth_provenance_path": str(artifact),
            "depth_provenance_summary": {
                "available": True,
                "valid_pixel_count": 10,
                "native_patchmatch_pixel_count": 6,
                "targeted_patchmatch_pixel_count": 1,
                "cross_view_measured_pixel_count": 2,
                "anchored_interpolation_pixel_count": 1,
                "unclassified_valid_pixel_count": 0,
            },
            "missing_reason_summary": {
                "support_pixel_count": 12,
                "missing_pixel_count": 2,
            },
            "targeted_gap_recovery_diagnostics": {
                "requested_gap_pixel_count": 4,
                "candidate_pixel_count": 3,
                "recovered_pixel_count": 1,
            },
        }

    def test_summarize_accepts_complete_partition_and_deduplicates_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            early = self._record(root)
            early["depth_provenance_summary"] = {
                **early["depth_provenance_summary"],
                "valid_pixel_count": 0,
                "native_patchmatch_pixel_count": 0,
                "targeted_patchmatch_pixel_count": 0,
                "cross_view_measured_pixel_count": 0,
                "anchored_interpolation_pixel_count": 0,
            }
            final = self._record(root)
            result = summarizer.summarize(
                {"depth_artifacts": [early, final]}, root / "report.json"
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["frame_count"], 1)
            self.assertEqual(result["valid_pixel_count"], 10)
            self.assertEqual(result["classified_valid_pixel_count"], 10)
            self.assertEqual(result["measured_pixel_count"], 9)
            self.assertAlmostEqual(result["interpolated_valid_ratio"], 0.1)

    def test_summarize_rejects_unclassified_valid_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root)
            record["depth_provenance_summary"][
                "unclassified_valid_pixel_count"
            ] = 1
            record["depth_provenance_summary"][
                "native_patchmatch_pixel_count"
            ] = 5

            result = summarizer.summarize(
                {"frames": [record]}, root / "manifest.json"
            )

            self.assertFalse(result["passed"])
            self.assertTrue(any("unclassified=1" in error for error in result["errors"]))

    def test_summarize_rejects_missing_provenance_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root)
            Path(record["depth_provenance_path"]).unlink()

            result = summarizer.summarize(
                {"frames": [record]}, root / "manifest.json"
            )

            self.assertFalse(result["passed"])
            self.assertTrue(any("does not exist" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
