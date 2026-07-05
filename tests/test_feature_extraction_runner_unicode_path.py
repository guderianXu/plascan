from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "gui" / "tasks" / "FeatureExtractionRunner.cpp"


class FeatureExtractionRunnerUnicodePathTest(unittest.TestCase):
    def test_runner_decodes_images_from_qfile_bytes(self):
        source = SOURCE.read_text(encoding="utf-8")

        self.assertIn("readImageWithQtPath", source)
        self.assertIn("QFile imageFile(imagePath)", source)
        self.assertIn("cv::imdecode", source)
        self.assertNotIn("cv::imread(imagePath.toStdString()", source)


if __name__ == "__main__":
    unittest.main()
