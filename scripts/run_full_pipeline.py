#!/usr/bin/env python3
"""One-command PlaScan reconstruction pipeline from an image/camera .lis file."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PlaScan GUI-equivalent reconstruction pipeline"
    )
    parser.add_argument("list_file", type=Path, help="image/camera list file")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--quality", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cuda-parallel-pairs", type=int, default=1)
    parser.add_argument("--dem-resolution", type=float, default=0.0)
    parser.add_argument("--mesh-resolution", type=int, default=160)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-terrain", action="store_true")
    parser.add_argument("--export-obj", action="store_true")
    parser.add_argument(
        "--legacy-stereo-test",
        action="store_true",
        help="run the old dense_match_cli/triangulate_cli validation pipeline",
    )
    return parser.parse_args(argv)


def command_path(build_dir: Path, name: str) -> Path:
    direct = build_dir / name
    if direct.exists():
        return direct
    return build_dir / "bin" / name


def build_command(args: argparse.Namespace) -> list[str]:
    root = repo_root()
    build_dir = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
    build_dir = build_dir.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = build_dir / "测试用临时文件" / f"full_pipeline_{stamp}"
    elif not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    tool = command_path(build_dir, "reconstruct_pipeline_cli")
    cmd = [
        str(tool),
        str(args.list_file.resolve()),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--quality",
        str(args.quality),
        "--threads",
        str(args.threads),
        "--cuda-parallel-pairs",
        str(args.cuda_parallel_pairs),
        "--dem-resolution",
        str(args.dem_resolution),
        "--mesh-resolution",
        str(args.mesh_resolution),
    ]
    if args.skip_model:
        cmd.append("--skip-model")
    if args.skip_terrain:
        cmd.append("--skip-terrain")
    if args.export_obj:
        cmd.append("--export-obj")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.legacy_stereo_test:
        from run_full_pipeline_test import main as legacy_main

        return legacy_main()

    cmd = build_command(args)
    tool = Path(cmd[0])
    if not tool.exists():
        print(f"reconstruct_pipeline_cli not found: {tool}", file=sys.stderr)
        print("请先编译: cmake --build build --target reconstruct_pipeline_cli", file=sys.stderr)
        return 127

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
