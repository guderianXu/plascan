import unittest
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DiskTorchscriptModelTest(unittest.TestCase):
    def test_descriptors_are_sampled_with_runtime_image_shape(self):
        model_path = ROOT / "resources/models/disk_extractor_cpu_8192.torchscript"
        image_path = ROOT / "testData/img/1.png"
        self.assertEqual(model_path.suffix, ".torchscript")
        self.assertTrue(model_path.exists(), model_path)
        self.assertTrue(image_path.exists(), image_path)

        torch = self._import_or_skip("torch")
        cv2 = self._import_or_skip("cv2")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        scale = 1200.0 / max(image.shape)
        resized = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.assertNotEqual((resized.shape[1], resized.shape[0]), (640, 480))

        torch.set_num_threads(1)
        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        tensor = torch.from_numpy(resized.astype("float32") / 255.0).view(1, 1, resized.shape[0], resized.shape[1])
        orig_wh = torch.tensor([image.shape[1], image.shape[0]], dtype=torch.float32)

        with torch.no_grad():
            keypoints, descriptors, scores = model(tensor, orig_wh)

        self.assertEqual(keypoints.shape[1], 8192)
        self.assertEqual(descriptors.shape[1], 8192)
        self.assertEqual(scores.shape[1], 8192)

        norms = torch.linalg.norm(descriptors[0], dim=1)
        zero_count = int((norms < 1e-6).sum().item())
        self.assertLessEqual(zero_count, 1)
        self.assertGreater(float(norms.mean().item()), 0.99)

    @staticmethod
    def _import_or_skip(module_name):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise unittest.SkipTest(f"{module_name} is not installed") from exc


if __name__ == "__main__":
    unittest.main()
