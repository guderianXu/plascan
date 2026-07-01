from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "env" / "setup_python_runtime.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("setup_python_runtime", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        try:
            sys.path.remove(str(SCRIPT_PATH.parent))
        except ValueError:
            pass
    return module


class PythonRuntimeSetupTest(unittest.TestCase):
    def test_default_runtime_dir_is_fixed_under_build_env(self):
        runtime = load_runtime_module()

        self.assertEqual(
            runtime.default_runtime_dir(Path("E:/code/plascan")),
            Path("E:/code/plascan") / "build" / "env" / "python-runtime",
        )

    def test_runtime_python_path_is_platform_specific(self):
        runtime = load_runtime_module()
        runtime_dir = Path("E:/code/plascan") / "build" / "env" / "python-runtime"

        self.assertEqual(
            runtime.runtime_python_path(runtime_dir, "windows"),
            runtime_dir / "Scripts" / "python.exe",
        )
        self.assertEqual(
            runtime.runtime_python_path(runtime_dir, "linux"),
            runtime_dir / "bin" / "python",
        )

    def test_environment_values_include_runtime_model_and_script_paths(self):
        runtime = load_runtime_module()
        repo_root = Path("E:/code/plascan")
        runtime_dir = repo_root / "build" / "env" / "python-runtime"
        python_exe = runtime_dir / "Scripts" / "python.exe"

        values = runtime.environment_values(
            repo_root=repo_root,
            runtime_dir=runtime_dir,
            python_exe=python_exe,
            device="cuda",
            cuda_wheel="cu130",
            torch_dir="",
            cuda_root="",
        )

        self.assertEqual(values["PLASCAN_PYTHON_EXECUTABLE"], str(python_exe))
        self.assertEqual(values["PLASCAN_PYTHON"], str(python_exe))
        self.assertEqual(values["PLASCAN_MODEL_DIR"], str(repo_root / "resources" / "models"))
        self.assertEqual(values["PLASCAN_SCRIPT_DIR"], str(repo_root / "scripts"))
        self.assertEqual(values["PYTHONUTF8"], "1")
        self.assertEqual(values["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(values["PLASCAN_PYTHON_RUNTIME_DIR"], str(runtime_dir))
        self.assertEqual(values["PLASCAN_PYTHON_DEVICE"], "cuda")
        self.assertEqual(values["PLASCAN_PYTHON_CUDA_WHEEL"], "cu130")

    def test_dependency_plan_uses_cuda_index_and_lightglue(self):
        runtime = load_runtime_module()

        plan = runtime.dependency_plan(device="cuda", cuda_wheel="cu130", extra_packages=["pytest"])

        self.assertEqual(plan.torch_index_url, "https://download.pytorch.org/whl/cu130")
        self.assertIn("torch", plan.torch_packages)
        self.assertIn("torchvision", plan.torch_packages)
        self.assertIn("numpy", plan.base_packages)
        self.assertIn("opencv-python", plan.base_packages)
        self.assertIn("kornia", plan.base_packages)
        self.assertTrue(any("lightglue" in package.lower() for package in plan.base_packages))
        self.assertIn("pytest", plan.extra_packages)

    def test_write_runtime_env_files(self):
        runtime = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "repo"
            runtime_dir = repo_root / "build" / "env" / "python-runtime"
            python_exe = runtime.runtime_python_path(runtime_dir, "linux")
            json_path = runtime.write_runtime_env_files(
                output_dir=tmp_path / "env",
                repo_root=repo_root,
                runtime_dir=runtime_dir,
                python_exe=python_exe,
                device="cpu",
                cuda_wheel="cpu",
                torch_dir="",
                cuda_root="",
            )

            self.assertTrue(json_path.exists())
            ps1_text = (tmp_path / "env" / "plascan-env.ps1").read_text(encoding="utf-8")
            sh_text = (tmp_path / "env" / "plascan-env.sh").read_text(encoding="utf-8")
            self.assertIn("PLASCAN_PYTHON_EXECUTABLE", ps1_text)
            self.assertIn("PLASCAN_MODEL_DIR", ps1_text)
            self.assertIn("PLASCAN_SCRIPT_DIR", sh_text)

    def test_windows_dev_shell_auto_detects_fixed_runtime(self):
        script = (REPO_ROOT / "scripts" / "build_win" / "enter_plascan_dev_shell.ps1").read_text(encoding="utf-8")

        self.assertIn("python-runtime", script)
        self.assertIn("PLASCAN_PYTHON_EXECUTABLE", script)
        self.assertIn("PLASCAN_MODEL_DIR", script)
        self.assertIn("PLASCAN_SCRIPT_DIR", script)


if __name__ == "__main__":
    unittest.main()
