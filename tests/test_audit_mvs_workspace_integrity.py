import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validation"
    / "audit_mvs_workspace_integrity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_mvs_workspace_integrity", SCRIPT_PATH
)
auditor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = auditor
SPEC.loader.exec_module(auditor)


def _metrics(width: int, height: int, coverage: float) -> object:
    pixels = width * height
    nonzero = round(pixels * coverage)
    return auditor.MaskMetrics(
        width=width,
        height=height,
        channels=1,
        dtype="uint8",
        pixel_count=pixels,
        nonzero_pixel_count=nonzero,
        nonzero_coverage=nonzero / pixels,
        minimum_value=255.0 if coverage == 1.0 else 0.0,
        maximum_value=0.0 if coverage == 0.0 else 255.0,
        all_white=coverage == 1.0,
        all_black=coverage == 0.0,
    )


def _source_metrics(
    width: int, height: int, *, nonzero: int, maximum: int, observed_bits: int
) -> object:
    pixels = width * height
    return auditor.SourceMaskMetrics(
        width=width,
        height=height,
        cv_type=2,
        data_bytes=pixels * 2,
        file_size=40 + pixels * 2,
        value_count=pixels,
        nonzero_value_count=nonzero,
        maximum_value=maximum,
        observed_bit_mask=observed_bits,
    )


def _write_fast_matrix(
    path: Path,
    *,
    magic: bytes = b"PLASDEPTHMAT01\x00\x00",
    rows: int = 2,
    columns: int = 3,
    cv_type: int = 2,
    data_bytes: int | None = None,
    payload: bytes | None = None,
    suffix: bytes = b"",
) -> None:
    if payload is None:
        payload = bytes(max(rows, 0) * max(columns, 0) * 2)
    if data_bytes is None:
        data_bytes = len(payload)
    header = struct.pack(
        "<16siii4xQ", magic, rows, columns, cv_type, data_bytes
    )
    path.write_bytes(header + payload + suffix)


class AuditMvsWorkspaceIntegrityTest(unittest.TestCase):
    def _frame(
        self,
        root: Path,
        ref_index: int,
        acceptance: str,
        coverage: float,
        *,
        available: bool,
    ) -> dict:
        path = root / f"support_{ref_index}.png"
        path.write_bytes(b"mask")
        return {
            "ref_index": ref_index,
            "status": "completed",
            "acceptance": acceptance,
            "fusion_eligible": acceptance == "accepted",
            "algorithm_revision": 36,
            "support_mask_path": path.name,
            "grid_width": 2,
            "grid_height": 2,
            "mask_coverage": coverage,
            "quality_decision": {
                "sparse_absolute_depth_residual": {
                    "available": available,
                    "projected_sample_count": 40,
                    "valid_sample_count": 30,
                    "valid_sample_ratio": 0.75,
                    "median_absolute_log_error": 0.002,
                }
            },
        }

    def test_audit_summarizes_frame_masks_revisions_and_sparse_residuals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = [
                self._frame(root, 1, "accepted", 0.5, available=True),
                self._frame(
                    root, 2, "validation_only", 0.75, available=False
                ),
            ]
            metrics = {
                "support_1.png": _metrics(2, 2, 0.5),
                "support_2.png": _metrics(2, 2, 0.75),
            }

            result = auditor.audit_manifest(
                {"algorithm_revision": 36, "frames": frames},
                root / "mvs_manifest.json",
                mask_reader=lambda path: metrics[path.name],
            )

            self.assertTrue(result["integrity_passed"])
            self.assertEqual(result["counts"]["frame"], 2)
            self.assertEqual(result["counts"]["accepted"], 1)
            self.assertEqual(result["counts"]["validation"], 1)
            self.assertEqual(result["counts"]["fusion_eligible"], 1)
            self.assertEqual(
                result["algorithm_revision"]["frame_distribution"], {"36": 2}
            )
            sparse = result["sparse_absolute_depth_residual"]
            self.assertEqual(sparse["available_count"], 1)
            self.assertEqual(
                sparse["valid_sample_count_distribution"]["p50"], 30.0
            )

    def test_audit_reports_mask_size_coverage_and_empty_mask_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            white = self._frame(root, 1, "accepted", 0.5, available=True)
            black = self._frame(root, 2, "rejected", 0.0, available=True)
            missing_path = self._frame(
                root, 3, "validation_only", 0.5, available=True
            )
            missing_path["support_mask_path"] = ""
            metrics = {
                "support_1.png": _metrics(3, 2, 1.0),
                "support_2.png": _metrics(2, 2, 0.0),
            }

            result = auditor.audit_manifest(
                {"frames": [white, black, missing_path]},
                root / "mvs_manifest.json",
                mask_reader=lambda path: metrics[path.name],
            )

            self.assertFalse(result["integrity_passed"])
            masks = result["support_masks"]
            self.assertEqual(masks["all_white_count"], 1)
            self.assertEqual(masks["all_black_count"], 1)
            self.assertEqual(masks["dimension_mismatch_count"], 1)
            self.assertEqual(masks["coverage_mismatch_count"], 1)
            self.assertEqual(masks["missing_path_count"], 1)

    def test_opencv_reader_measures_binary_mask_when_available(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV runtime is not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            image = np.array([[0, 255], [255, 255]], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))

            metrics = auditor._read_mask_metrics(path)

            self.assertEqual((metrics.width, metrics.height), (2, 2))
            self.assertEqual(metrics.nonzero_pixel_count, 3)
            self.assertEqual(metrics.nonzero_coverage, 0.75)
            self.assertFalse(metrics.all_white)
            self.assertFalse(metrics.all_black)

    def test_revision_37_geometry_source_mask_contract_passes_and_is_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = self._frame(
                root, 1, "accepted", 0.5, available=True
            )
            accepted.update(
                {
                    "algorithm_revision": 37,
                    "raw_geometry_source_mask_path": "source_1.bin",
                    "geometry_source_indices": [2, 3, 4],
                }
            )
            (root / "source_1.bin").write_bytes(b"source mask")
            rejected = self._frame(
                root, 5, "rejected", 0.5, available=True
            )
            rejected["algorithm_revision"] = 37
            support_metrics = {
                "support_1.png": _metrics(2, 2, 0.5),
                "support_5.png": _metrics(2, 2, 0.5),
            }

            result = auditor.audit_manifest(
                {"algorithm_revision": 37, "frames": [accepted, rejected]},
                root / "mvs_manifest.json",
                mask_reader=lambda path: support_metrics[path.name],
                source_mask_reader=lambda path: _source_metrics(
                    2, 2, nonzero=3, maximum=7, observed_bits=0b111
                ),
            )

            self.assertTrue(result["integrity_passed"])
            self.assertEqual(
                result["schema"], "plascan.mvs.workspace_integrity_audit.v2"
            )
            sources = result["geometry_source_masks"]
            self.assertEqual(sources["revision_completed_count"], 2)
            self.assertEqual(sources["required_count"], 1)
            self.assertEqual(sources["existing_count"], 1)
            self.assertEqual(sources["valid_table_count"], 1)
            self.assertEqual(sources["invalid_table_count"], 0)
            self.assertEqual(sources["out_of_range_bit_mask_count"], 0)
            self.assertEqual(sources["aggregate_observed_bit_mask"], 0b111)
            self.assertEqual(sources["details"][0]["maximum_value"], 7)

    def test_revision_37_geometry_source_contract_rejects_missing_invalid_and_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overflow = self._frame(
                root, 1, "accepted", 0.5, available=True
            )
            overflow.update(
                {
                    "algorithm_revision": 37,
                    "raw_geometry_source_mask_path": "overflow.bin",
                    "geometry_source_indices": [2, 3],
                }
            )
            (root / "overflow.bin").write_bytes(b"source mask")

            missing = self._frame(
                root, 4, "validation_only", 0.5, available=True
            )
            missing["algorithm_revision"] = 37

            invalid = self._frame(
                root, 6, "rejected", 0.5, available=True
            )
            invalid.update(
                {
                    "algorithm_revision": 37,
                    "raw_geometry_source_mask_path": "invalid.bin",
                    "geometry_source_indices": [6, 8, 8],
                }
            )
            (root / "invalid.bin").write_bytes(b"source mask")
            support_metrics = {
                f"support_{index}.png": _metrics(2, 2, 0.5)
                for index in (1, 4, 6)
            }
            source_metrics = {
                "overflow.bin": _source_metrics(
                    2, 2, nonzero=1, maximum=4, observed_bits=0b100
                ),
                "invalid.bin": _source_metrics(
                    2, 2, nonzero=1, maximum=1, observed_bits=0b001
                ),
            }

            result = auditor.audit_manifest(
                {
                    "algorithm_revision": 37,
                    "frames": [overflow, missing, invalid],
                },
                root / "mvs_manifest.json",
                mask_reader=lambda path: support_metrics[path.name],
                source_mask_reader=lambda path: source_metrics[path.name],
            )

            self.assertFalse(result["integrity_passed"])
            sources = result["geometry_source_masks"]
            self.assertEqual(sources["missing_required_path_count"], 1)
            self.assertEqual(sources["invalid_table_count"], 2)
            self.assertEqual(sources["out_of_range_bit_mask_count"], 1)
            self.assertEqual(sources["aggregate_out_of_range_bit_mask"], 0b100)
            issues = sources["table_issue_distribution"]
            self.assertEqual(issues["missing_or_non_array_table"], 1)
            self.assertEqual(issues["duplicate_source_index"], 1)
            self.assertEqual(issues["table_contains_reference_index"], 1)

    def test_geometry_source_reader_validates_header_and_measures_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.bin"
            values = (0, 1, 3, 8, 16, 4)
            _write_fast_matrix(valid, payload=struct.pack("<6H", *values))

            metrics = auditor._read_geometry_source_mask_metrics(valid)

            self.assertEqual((metrics.width, metrics.height), (3, 2))
            self.assertEqual(metrics.cv_type, 2)
            self.assertEqual(metrics.data_bytes, 12)
            self.assertEqual(metrics.file_size, 52)
            self.assertEqual(metrics.value_count, 6)
            self.assertEqual(metrics.nonzero_value_count, 5)
            self.assertEqual(metrics.maximum_value, 16)
            self.assertEqual(metrics.observed_bit_mask, 0b11111)

            invalid_cases = {
                "magic": {"magic": b"not-a-plascan-matrix"},
                "rows": {"rows": 0, "payload": b"", "data_bytes": 0},
                "columns": {
                    "columns": 0,
                    "payload": b"",
                    "data_bytes": 0,
                },
                "type": {"cv_type": 5},
                "data_bytes": {"data_bytes": 10},
                "file_size": {"suffix": b"extra"},
            }
            for name, arguments in invalid_cases.items():
                with self.subTest(name=name):
                    path = root / f"invalid_{name}.bin"
                    _write_fast_matrix(path, **arguments)
                    with self.assertRaises(ValueError):
                        auditor._read_geometry_source_mask_metrics(path)


if __name__ == "__main__":
    unittest.main()
