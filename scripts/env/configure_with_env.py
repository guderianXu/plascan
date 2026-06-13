#!/usr/bin/env python3
"""Configure PlaScan with a generated environment file and CMake preset."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from env_common import default_output_dir, host_platform, load_env_json, merged_environment, quote_command


def default_preset(build_type: str) -> str:
    prefix = "windows" if host_platform() == "windows" else "linux"
    return f"{prefix}-vcpkg-{build_type}"


def cmake_defines(values: dict[str, str]) -> list[str]:
    defines = []
    mapping = {
        "PLASCAN_TORCH_DIR": "PLASCAN_TORCH_DIR",
        "Torch_DIR": "Torch_DIR",
        "CUDAToolkit_ROOT": "CUDAToolkit_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR": "CUDA_TOOLKIT_ROOT_DIR",
        "CMAKE_CUDA_COMPILER": "CMAKE_CUDA_COMPILER",
        "CMAKE_TOOLCHAIN_FILE": "CMAKE_TOOLCHAIN_FILE",
        "VCPKG_TARGET_TRIPLET": "VCPKG_TARGET_TRIPLET",
    }
    for env_key, cmake_key in mapping.items():
        value = values.get(env_key, "").strip()
        if value:
            defines.append(f"-D{cmake_key}={value}")
    return defines


def run(cmd: list[str], dry_run: bool, env: dict[str, str]) -> None:
    print(f"+ {quote_command(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(default_output_dir() / "plascan-env.json"))
    parser.add_argument("--vcpkg-file", default=str(default_output_dir() / "plascan-vcpkg.json"))
    parser.add_argument("--require-env-file", action="store_true")
    parser.add_argument("--require-vcpkg-file", action="store_true")
    parser.add_argument("--build-type", choices=["debug", "release"], default="release")
    parser.add_argument("--preset", help="Override CMake configure preset")
    parser.add_argument("--build", action="store_true", help="Run cmake --build after configure")
    parser.add_argument("--test", action="store_true", help="Run ctest after configure/build")
    parser.add_argument("--package", action="store_true", help="Run cpack after configure/build")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("cmake_args", nargs="*", help="Additional -D arguments passed to configure")
    args = parser.parse_args()

    values = {}
    env_path = Path(args.env_file).expanduser().resolve()
    vcpkg_path = Path(args.vcpkg_file).expanduser().resolve()

    if env_path.exists():
        values.update(load_env_json(env_path))
    elif args.require_env_file:
        values.update(load_env_json(env_path))

    if vcpkg_path.exists():
        values.update(load_env_json(vcpkg_path))
    elif args.require_vcpkg_file:
        values.update(load_env_json(vcpkg_path))

    env = merged_environment(values)
    preset = args.preset or default_preset(args.build_type)

    configure_cmd = ["cmake", "--preset", preset, *cmake_defines(values), *args.cmake_args]
    run(configure_cmd, args.dry_run, env)

    if args.build:
        run(["cmake", "--build", "--preset", preset], args.dry_run, env)
    if args.test:
        run(["ctest", "--preset", preset], args.dry_run, env)
    if args.package:
        run(["cpack", "--preset", preset], args.dry_run, env)


if __name__ == "__main__":
    main()
