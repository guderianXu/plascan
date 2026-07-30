import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "models" / "install_sam21_model.py"
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "models" / "export_sam21_torchscript.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("install_sam21_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class InstallSam21ModelScriptTest(unittest.TestCase):
    def test_variants_use_official_sam21_checkpoint_urls(self):
        module = load_script_module()

        self.assertEqual(
            module.VARIANTS["tiny"].url,
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        )
        self.assertEqual(module.VARIANTS["small"].checkpoint_name, "sam2.1_hiera_small.pt")
        self.assertEqual(module.VARIANTS["base_plus"].token, "base_plus")
        self.assertEqual(module.VARIANTS["large"].checkpoint_name, "sam2.1_hiera_large.pt")

    def test_build_export_command_uses_bundled_export_script_and_model_dir(self):
        module = load_script_module()

        command = module.build_export_command(
            python_exe=Path("C:/PlaScan/runtime/python.exe"),
            source_dir=Path("E:/code/plascan"),
            model_dir=Path("E:/code/plascan/resources/models"),
            variant=module.VARIANTS["small"],
            devices="cpu,cuda",
            input_size=1024,
        )

        self.assertEqual(command[0], "C:/PlaScan/runtime/python.exe")
        self.assertEqual(command[1], "E:/code/plascan/scripts/models/export_sam21_torchscript.py")
        self.assertIn("--variant", command)
        self.assertIn("small", command)
        self.assertIn("--checkpoint", command)
        self.assertIn("E:/code/plascan/resources/models/sam2.1_hiera_small.pt", command)
        self.assertIn("--devices", command)
        self.assertIn("cpu,cuda", command)

    def test_export_script_uses_reshape_after_noncontiguous_permute(self):
        source = EXPORT_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn(".reshape(batch_size, -1, feat_size[0], feat_size[1])", source)
        self.assertNotIn("feat.permute(1, 2, 0).view(", source)
        self.assertNotIn("src.transpose(1, 2).view(", source)


if __name__ == "__main__":
    unittest.main()
