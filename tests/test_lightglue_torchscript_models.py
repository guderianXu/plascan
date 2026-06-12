import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "resources" / "models"


class LightGlueTorchscriptModelTest(unittest.TestCase):
    def test_disk_and_aliked_models_are_packaged_for_cpp_matching(self):
        expected = [
            "lightglue_disk_cpu.torchscript",
            "lightglue_disk_cuda.torchscript",
            "lightglue_aliked_cpu.torchscript",
            "lightglue_aliked_cuda.torchscript",
        ]

        missing = [name for name in expected if not (MODELS / name).exists()]

        self.assertEqual(missing, [])

    def test_disk_and_aliked_cpu_models_accept_128d_descriptors_only(self):
        torch = self._import_or_skip("torch")

        for algorithm in ("disk", "aliked"):
            with self.subTest(algorithm=algorithm):
                model_path = MODELS / f"lightglue_{algorithm}_cpu.torchscript"
                self.assertTrue(model_path.exists(), model_path)

                model = torch.jit.load(str(model_path), map_location="cpu").eval()
                kpts0 = torch.rand(1, 5, 2)
                kpts1 = torch.rand(1, 7, 2)
                desc0 = torch.rand(1, 5, 128)
                desc1 = torch.rand(1, 7, 128)
                image_size = torch.tensor([[640.0, 480.0]])

                with torch.no_grad():
                    scores = model(kpts0, desc0, image_size, kpts1, desc1, image_size)

                self.assertEqual(tuple(scores.shape), (1, 6, 8))

                wrong_desc0 = torch.rand(1, 5, 256)
                wrong_desc1 = torch.rand(1, 7, 256)
                with self.assertRaises(Exception):
                    model(kpts0, wrong_desc0, image_size, kpts1, wrong_desc1, image_size)

    def test_legacy_superpoint_cpu_model_accepts_256d_descriptors_only(self):
        torch = self._import_or_skip("torch")

        model_path = MODELS / "lightglue_matcher_cpu.torchscript"
        self.assertTrue(model_path.exists(), model_path)

        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        kpts0 = torch.rand(1, 5, 2)
        kpts1 = torch.rand(1, 7, 2)
        image_size = torch.tensor([[640.0, 480.0]])

        desc0 = torch.rand(1, 5, 256)
        desc1 = torch.rand(1, 7, 256)
        with torch.no_grad():
            scores = model(kpts0, desc0, image_size, kpts1, desc1, image_size)

        self.assertEqual(tuple(scores.shape), (1, 6, 8))

        wrong_desc0 = torch.rand(1, 5, 128)
        wrong_desc1 = torch.rand(1, 7, 128)
        with self.assertRaises(Exception):
            model(kpts0, wrong_desc0, image_size, kpts1, wrong_desc1, image_size)

    @staticmethod
    def _import_or_skip(module_name):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise unittest.SkipTest(f"{module_name} is not installed") from exc


if __name__ == "__main__":
    unittest.main()
