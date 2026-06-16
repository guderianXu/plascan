import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = Path(
    os.environ.get(
        "PLASCAN_RECONSTRUCT_PIPELINE_CLI",
        REPO_ROOT / "build" / "bin" / "reconstruct_pipeline_cli",
    )
)


def write_camera(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "VERSION_3",
                "PINHOLE",
                "TSAI",
                "fu = 100",
                "fv = 100",
                "cu = 50",
                "cv = 50",
                "u_direction = 1 0 0",
                "v_direction = 0 1 0",
                "w_direction = 0 0 1",
                "C = 0 0 0",
                "R = 1 0 0 0 1 0 0 0 1",
                "pitch = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


class ReconstructPipelineCliTest(unittest.TestCase):
    def run_cli(self, args):
        if not CLI_PATH.exists():
            self.skipTest(f"reconstruct_pipeline_cli not found: {CLI_PATH}")
        return subprocess.run(
            [str(CLI_PATH), *map(str, args)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_plapoint_progress_uses_utf8_console_output(self):
        source = (REPO_ROOT / "src" / "cli" / "cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")

        self.assertNotIn("message.toLocal8Bit()", source)
        self.assertIn("qUtf8Printable(message)", source)

    def test_non_empty_output_dir_requires_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            output_dir.mkdir()
            (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
            missing_list = root / "missing.lis"

            result = self.run_cli([missing_list, "--output-dir", output_dir])

            self.assertNotEqual(result.returncode, 0)
            combined = result.stdout + result.stderr
            self.assertIn("输出目录", combined)
            self.assertIn("非空", combined)

            forced = self.run_cli([missing_list, "--output-dir", output_dir, "--force"])

            self.assertNotEqual(forced.returncode, 0)
            forced_output = forced.stdout + forced.stderr
            self.assertIn("列表读取失败", forced_output)
            self.assertNotIn("非空", forced_output)

    def test_shell_quoted_lis_paths_support_spaces_and_unicode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "影像 一.png"
            camera = root / "相机 一.tsai"
            image.write_bytes(b"placeholder")
            write_camera(camera)
            list_path = root / "input.lis"
            list_path.write_text("'影像 一.png' '相机 一.tsai'\n", encoding="utf-8")

            result = self.run_cli([list_path, "--output-dir", root / "out"])

            self.assertNotEqual(result.returncode, 0)
            combined = result.stdout + result.stderr
            self.assertIn("至少需要 2 组", combined)
            self.assertNotIn("需要 '<image> <camera.tsai>'", combined)
            self.assertNotIn("影像不存在", combined)

    def test_quoted_csv_lis_paths_allow_commas_spaces_and_unicode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "影像, 一.png"
            camera = root / "相机 一.tsai"
            image.write_bytes(b"placeholder")
            write_camera(camera)
            list_path = root / "input.lis"
            list_path.write_text('"影像, 一.png","相机 一.tsai"\n', encoding="utf-8")

            result = self.run_cli([list_path, "--output-dir", root / "out"])

            self.assertNotEqual(result.returncode, 0)
            combined = result.stdout + result.stderr
            self.assertIn("至少需要 2 组", combined)
            self.assertNotIn("影像不存在", combined)


if __name__ == "__main__":
    unittest.main()
