from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "scripts" / "models"
sys.path.insert(0, str(MODELS_DIR))

from compose_loma_r_package import (  # noqa: E402
    PACKAGE_KEYPOINT_BUCKETS,
    compose_package,
    require_matcher_only_feature,
)
from model_provenance import (  # noqa: E402
    ProvenanceError,
    provenance_path,
    sha256_file,
    validate_provenance,
    write_provenance,
)


def feature_contract(revision: str = "feature-revision") -> dict:
    return {
        "artifact_kind": "loma_r_feature_onnx",
        "exporter": {"name": "export_loma_r_tensorrt.py", "schema_version": 1},
        "source": {
            "repository": "davnords/loma",
            "revision": {"commit": revision, "dirty": False},
            "checkpoints": [{"name": "dad.pth", "sha256": "1" * 64}],
        },
        "model": {
            "id": "loma_r_feature",
            "configuration": {"descriptor_dimension": 256},
        },
        "input": {"width": 784, "height": 784},
        "profile": {"feature_keypoint_count": 3840, "dynamic_input": False},
        "opset": 18,
        "precision": "fp16",
        "tools": {"python": "test"},
    }


def matcher_contract(revision: str = "feature-revision") -> dict:
    return {
        "artifact_kind": "loma_r_matcher_onnx",
        "exporter": {"name": "export_loma_r_tensorrt.py", "schema_version": 1},
        "source": {
            "repository": "davnords/loma",
            "revision": {"commit": revision, "dirty": False},
            "checkpoints": [{"name": "loma_R.pth", "sha256": "2" * 64}],
        },
        "model": {
            "id": "loma_r_matcher",
            "configuration": {"descriptor_dimension": 256},
        },
        "input": {"keypoints": "[1,K,2] float32"},
        "profile": {
            "keypoints": {
                "dynamic": True,
                "minimum": 1,
                "optimum": 2048,
                "maximum": 3840,
                "package_buckets": list(PACKAGE_KEYPOINT_BUCKETS),
            }
        },
        "opset": 18,
        "precision": "fp16",
        "tools": {"python": "test"},
    }


class ModelProvenanceTest(unittest.TestCase):
    def test_helpers_import_without_torch_or_tensorrt(self):
        code = f"""
import builtins
import sys
sys.path.insert(0, {str(MODELS_DIR)!r})
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {{'torch', 'tensorrt'}}:
        raise AssertionError('heavy runtime import attempted: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import model_provenance
import compose_loma_r_package
"""
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reuse_requires_exact_contract_and_artifact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.onnx"
            artifact.write_bytes(b"onnx-v1")
            contract = {"model": {"id": "test"}, "opset": 18}
            write_provenance(artifact, contract)

            self.assertTrue(validate_provenance(artifact, contract).valid)
            self.assertFalse(
                validate_provenance(
                    artifact, {"model": {"id": "test"}, "opset": 20}
                ).valid
            )
            artifact.write_bytes(b"onnx-v2")
            validation = validate_provenance(artifact, contract)
            self.assertFalse(validation.valid)
            self.assertIn("SHA-256", validation.reason)

    def test_matcher_only_rejects_incompatible_feature_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            feature = Path(directory) / "loma_r_features_k3840_fp16.onnx"
            feature.write_bytes(b"feature")
            write_provenance(feature, feature_contract("old-revision"))

            with self.assertRaisesRegex(ProvenanceError, "--matcher-only"):
                require_matcher_only_feature(
                    feature, feature_contract("current-revision")
                )

    def test_composer_generates_only_shared_onnx_bucket_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            feature = output / "loma_r_features_k3840_fp16.onnx"
            matcher = output / "loma_r_matcher_dynamic_fp16.onnx"
            feature.write_bytes(b"feature-onnx")
            matcher.write_bytes(b"matcher-onnx")
            write_provenance(feature, feature_contract())
            write_provenance(matcher, matcher_contract())

            manifests = compose_package(feature, matcher, output)

            self.assertEqual(
                [path.name for path in manifests],
                [f"loma_r_k{count}_fp16.json" for count in PACKAGE_KEYPOINT_BUCKETS],
            )
            for path, count in zip(manifests, PACKAGE_KEYPOINT_BUCKETS):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(document["schema_version"], 2)
                self.assertEqual(document["keypoint_count"], count)
                self.assertEqual(document["feature_keypoint_count"], 3840)
                self.assertEqual(document["feature_onnx"], feature.name)
                self.assertEqual(document["matcher_onnx"], matcher.name)
                self.assertEqual(document["feature_onnx_sha256"], sha256_file(feature))
                self.assertEqual(document["matcher_onnx_sha256"], sha256_file(matcher))
                self.assertNotIn("checkpoints", document)
                self.assertNotIn("feature_provenance", document)
            self.assertTrue(provenance_path(feature).is_file())
            self.assertTrue(provenance_path(matcher).is_file())

    def test_composer_matches_published_manifest_whitelist(self):
        expected_hashes = {
            1024: "db3b242ed7cda10e16fd7c304844c1f809a3b37bc40af10a81c7248ca9e51aea",
            2048: "68ae6a68bb184375285d486384344b7a6500195d5f372e96c8b429bd8787c91e",
            3840: "5d55026fe3e0bc59bb93bc997d928ec46940905e22c7836b831a859e1dae2715",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            feature = output / "loma_r_features_k3840_fp16.onnx"
            matcher = output / "loma_r_matcher_dynamic_fp16.onnx"
            feature.touch()
            matcher.touch()
            provenance_documents = [
                {
                    "artifact": {
                        "file": feature.name,
                        "sha256": "2b2671850f6a79f071a171eb9b523a8807474bcde19b5ded0191b9593ed97e19",
                    },
                    "contract": feature_contract(),
                },
                {
                    "artifact": {
                        "file": matcher.name,
                        "sha256": "5c91444393c2245e66553e8f493e5b35dc39e8a099b9988a684391fdcdf90195",
                    },
                    "contract": matcher_contract(),
                },
            ]
            with mock.patch(
                "compose_loma_r_package.require_provenance",
                side_effect=provenance_documents,
            ):
                manifests = compose_package(feature, matcher, output)

            for manifest, keypoint_count in zip(
                manifests, PACKAGE_KEYPOINT_BUCKETS
            ):
                self.assertEqual(manifest.stat().st_size, 644)
                self.assertEqual(
                    sha256_file(manifest), expected_hashes[keypoint_count]
                )


if __name__ == "__main__":
    unittest.main()
