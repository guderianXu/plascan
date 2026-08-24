#!/usr/bin/env python3
"""Configure PlaScan with a generated environment file and CMake preset."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

from env_common import default_output_dir, fail, host_platform, load_env_json, merged_environment, quote_command
from run_tests import build_ctest_command
from setup_tensorrt_sdk import prepare_tensorrt_sdk


def default_preset(build_type: str) -> str:
    prefix = {"windows": "windows", "linux": "linux", "macos": "macos"}[host_platform()]
    return f"{prefix}-vcpkg-{build_type}"


def source_preset(build_type: str, dependencies: bool = False) -> str:
    prefix = {"windows": "windows", "linux": "linux", "macos": "macos"}[host_platform()]
    middle = "source-deps" if dependencies else "source"
    return f"{prefix}-{middle}-{build_type}"


def default_vcpkg_triplet() -> str:
    return {
        "windows": "x64-windows",
        "linux": "x64-linux-dynamic",
        "macos": "arm64-osx",
    }[host_platform()]


def cmake_defines(values: dict[str, str], *, include_gpu_runtime: bool = True) -> list[str]:
    defines = []
    mapping = {
        "CUDAToolkit_ROOT": "CUDAToolkit_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR": "CUDA_TOOLKIT_ROOT_DIR",
        "CMAKE_CUDA_COMPILER": "CMAKE_CUDA_COMPILER",
        "CMAKE_TOOLCHAIN_FILE": "CMAKE_TOOLCHAIN_FILE",
        "PLASCAN_PYTHON_EXECUTABLE": "Python3_EXECUTABLE",
        "VCPKG_TARGET_TRIPLET": "VCPKG_TARGET_TRIPLET",
    }
    if include_gpu_runtime:
        mapping.update(
            {
                "PLASCAN_CUDA_ARCHITECTURES": "PLASCAN_CUDA_ARCHITECTURES",
                "TensorRT_ROOT": "TensorRT_ROOT",
            }
        )
    for env_key, cmake_key in mapping.items():
        value = values.get(env_key, "").strip()
        if value:
            defines.append(f"-D{cmake_key}={value}")
    return defines


def is_foreign_platform_path(value: str) -> bool:
    if host_platform() == "windows":
        return False
    return bool(re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("\\\\"))


def validate_environment_values(values: dict[str, str]) -> None:
    path_keys = [
        "CUDAToolkit_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_TOOLCHAIN_FILE",
        "PLASCAN_PYTHON_EXECUTABLE",
    ]
    for key in path_keys:
        value = values.get(key, "").strip()
        if value and is_foreign_platform_path(value):
            fail(
                f"{key} contains a foreign platform path for {host_platform()}: {value}. "
                "Regenerate build/env on this host or pass --env-file for the current platform."
            )


def run(cmd: list[str], dry_run: bool, env: dict[str, str]) -> None:
    print(f"+ {quote_command(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def cmake_option(cmake_args: list[str], name: str, default: bool) -> bool:
    prefix = f"-D{name}="
    for argument in reversed(cmake_args):
        if argument.startswith(prefix):
            return argument[len(prefix) :].strip().lower() not in {"0", "off", "false", "no"}
    return default


def cuda_compiler_available(values: dict[str, str], env: dict[str, str]) -> bool:
    configured = values.get("CMAKE_CUDA_COMPILER", "").strip()
    if configured and Path(configured).expanduser().is_file():
        return True
    return shutil.which("nvcc", path=env.get("PATH")) is not None


def compute_capability_architectures(output: str) -> str:
    architectures = set()
    for line in output.splitlines():
        match = re.search(r"(?<!\d)(\d+)\.(\d+)(?!\d)", line.strip())
        if match:
            architectures.add(f"{int(match.group(1))}{int(match.group(2))}")
    return ";".join(sorted(architectures, key=int))


def detect_cuda_architectures(env: dict[str, str]) -> str:
    nvidia_smi = shutil.which("nvidia-smi", path=env.get("PATH"))
    if not nvidia_smi:
        return ""
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return compute_capability_architectures(result.stdout)


def default_build_jobs(environment: dict[str, str]) -> int:
    configured = environment.get("CMAKE_BUILD_PARALLEL_LEVEL", "").strip()
    if configured.isdigit() and int(configured) > 0:
        return int(configured)
    return max(1, os.cpu_count() or 1)


def prepend_runtime_path(environment: dict[str, str], directories: list[Path]) -> None:
    existing = environment.get("PATH", "")
    resolved = []
    for directory in directories:
        path = directory.expanduser().resolve()
        if path.is_dir() and str(path).lower() not in {entry.lower() for entry in resolved}:
            resolved.append(str(path))
    if resolved:
        environment["PATH"] = os.pathsep.join([*resolved, existing])


def custom_build_layout(build_dir: str) -> tuple[Path, Path, Path, Path]:
    main_dir = Path(build_dir).expanduser().resolve()
    dependencies_dir = main_dir / "source-deps"
    dependency_prefix = dependencies_dir / "install"
    vcpkg_installed_dir = dependencies_dir / "vcpkg_installed"
    return main_dir, dependencies_dir, dependency_prefix, vcpkg_installed_dir


def cmake_path(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(default_output_dir() / "plascan-env.json"))
    parser.add_argument("--vcpkg-file", default=str(default_output_dir() / "plascan-vcpkg.json"))
    parser.add_argument("--require-env-file", action="store_true")
    parser.add_argument("--require-vcpkg-file", action="store_true")
    parser.add_argument("--build-type", choices=["debug", "release"], default="release")
    parser.add_argument("--preset", help="Override CMake configure preset")
    parser.add_argument(
        "--build-dir",
        help=(
            "Override the PlaScan build directory. With --source-deps, dependency builds are "
            "placed below <build-dir>/source-deps"
        ),
    )
    parser.add_argument(
        "--source-deps",
        action="store_true",
        help="Build pinned Qt/OpenCV/GDAL sources before configuring PlaScan",
    )
    parser.add_argument(
        "--source-deps-preset",
        help="Override the source dependency configure/build preset",
    )
    parser.add_argument("--build", action="store_true", help="Run cmake --build after configure")
    parser.add_argument("--build-jobs", type=int, help="Override the default all-logical-CPU build parallelism")
    parser.add_argument("--test", action="store_true", help="Run ctest after configure/build")
    parser.add_argument("--test-jobs", type=int, help="Override the default all-logical-CPU CTest worker count")
    parser.add_argument("--package", action="store_true", help="Run cpack after configure/build")
    parser.add_argument(
        "--accept-tensorrt-license",
        action="store_true",
        help="Accept the NVIDIA TensorRT license and allow the pinned SDK download",
    )
    parser.add_argument("--tensorrt-root", default="", help="Use an existing TensorRT C++ SDK")
    parser.add_argument("--tensorrt-archive", default="", help="Install from an official TensorRT ZIP")
    parser.add_argument(
        "--no-tensorrt-auto-install",
        action="store_true",
        help="Do not find or install the pinned TensorRT SDK",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("cmake_args", nargs="*", help="Additional -D arguments passed to configure")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    custom_main_dir = None
    custom_dependencies_dir = None
    custom_dependency_prefix = None
    custom_vcpkg_installed_dir = None
    if args.build_dir:
        (
            custom_main_dir,
            custom_dependencies_dir,
            custom_dependency_prefix,
            custom_vcpkg_installed_dir,
        ) = custom_build_layout(args.build_dir)
        if custom_main_dir == repository_root:
            parser.error("--build-dir must not be the PlaScan source directory")

    if args.source_deps and args.build_type != "release":
        parser.error("--source-deps currently supports --build-type release only")
    if args.source_deps and args.package:
        parser.error("--package is not yet available for source dependency presets")

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

    validate_environment_values(values)

    env = merged_environment(values)
    cuda_requested = cmake_option(args.cmake_args, "PLASCAN_ENABLE_CUDA", True)
    tensorrt_requested = cmake_option(args.cmake_args, "PLASCAN_ENABLE_TENSORRT", True)
    architecture_overridden = any(
        argument.startswith("-DPLASCAN_CUDA_ARCHITECTURES=") for argument in args.cmake_args
    )
    if cuda_requested and not architecture_overridden and not values.get(
        "PLASCAN_CUDA_ARCHITECTURES", ""
    ).strip():
        detected_architectures = detect_cuda_architectures(env)
        if detected_architectures:
            values["PLASCAN_CUDA_ARCHITECTURES"] = detected_architectures
            print(f"Detected CUDA architectures: {detected_architectures}")
    if cuda_requested and tensorrt_requested and cuda_compiler_available(values, env):
        accept_license = args.accept_tensorrt_license or env.get(
            "PLASCAN_ACCEPT_TENSORRT_LICENSE", ""
        ).strip().lower() in {"1", "on", "true", "yes"}
        tensorrt_root = prepare_tensorrt_sdk(
            Path(__file__).resolve().parents[2],
            env,
            requested_root=args.tensorrt_root,
            requested_archive=args.tensorrt_archive,
            auto_install=not args.no_tensorrt_auto_install,
            accept_license=accept_license,
            dry_run=args.dry_run,
        )
        if tensorrt_root:
            values["TensorRT_ROOT"] = str(tensorrt_root)
            env["TENSORRT_ROOT"] = str(tensorrt_root)
            cuda_root_value = (
                values.get("CUDAToolkit_ROOT", "")
                or values.get("CUDA_TOOLKIT_ROOT_DIR", "")
                or env.get("CUDA_PATH", "")
            )
            runtime_directories = [tensorrt_root / "bin", tensorrt_root / "lib"]
            if cuda_root_value:
                runtime_directories.append(Path(cuda_root_value) / "bin")
            prepend_runtime_path(env, runtime_directories)
    build_jobs = args.build_jobs or default_build_jobs(env)
    if build_jobs < 1:
        parser.error("--build-jobs must be greater than zero")
    if args.source_deps:
        dependencies_preset = args.source_deps_preset or source_preset(
            args.build_type, dependencies=True
        )
        dependencies_configure_cmd = [
            "cmake",
            "--preset",
            dependencies_preset,
        ]
        if custom_dependencies_dir is not None:
            dependencies_configure_cmd.extend(
                [
                    "-B",
                    cmake_path(custom_dependencies_dir),
                    f"-DPLASCAN_SOURCE_DEPENDENCY_PREFIX={cmake_path(custom_dependency_prefix)}",
                    f"-DVCPKG_INSTALLED_DIR={cmake_path(custom_vcpkg_installed_dir)}",
                ]
            )
        dependencies_configure_cmd.extend(cmake_defines(values, include_gpu_runtime=False))
        run(dependencies_configure_cmd, args.dry_run, env)
        dependencies_build_selector = (
            [cmake_path(custom_dependencies_dir)]
            if custom_dependencies_dir is not None
            else ["--preset", dependencies_preset]
        )
        run(
            [
                "cmake",
                "--build",
                *dependencies_build_selector,
                "--parallel",
                str(build_jobs),
            ],
            args.dry_run,
            env,
        )

        if custom_dependency_prefix is not None:
            prepend_runtime_path(
                env,
                [
                    custom_dependency_prefix / "bin",
                    custom_dependency_prefix / "x64" / "vc17" / "bin",
                    custom_vcpkg_installed_dir / "x64-windows" / "bin",
                ],
            )
            env["GDAL_DATA"] = str(custom_dependency_prefix / "share" / "gdal")
            triplet = values.get("VCPKG_TARGET_TRIPLET", "").strip()
            if not triplet:
                triplet = default_vcpkg_triplet()
            env["PROJ_DATA"] = str(custom_vcpkg_installed_dir / triplet / "share" / "proj")

    preset = args.preset or (
        source_preset(args.build_type) if args.source_deps else default_preset(args.build_type)
    )

    configure_cmd = ["cmake", "--preset", preset]
    if custom_main_dir is not None:
        configure_cmd.extend(["-B", cmake_path(custom_main_dir)])
        if args.source_deps:
            configure_cmd.extend(
                [
                    f"-DCMAKE_PREFIX_PATH={cmake_path(custom_dependency_prefix)}",
                    f"-DPLASCAN_SOURCE_DEPENDENCY_PREFIX={cmake_path(custom_dependency_prefix)}",
                    f"-DVCPKG_INSTALLED_DIR={cmake_path(custom_vcpkg_installed_dir)}",
                ]
            )
        else:
            configure_cmd.append(
                f"-DVCPKG_INSTALLED_DIR={cmake_path(custom_main_dir / 'vcpkg_installed')}"
            )
    configure_cmd.extend([*cmake_defines(values), *args.cmake_args])
    run(configure_cmd, args.dry_run, env)

    if args.build:
        build_selector = (
            [cmake_path(custom_main_dir)]
            if custom_main_dir is not None
            else ["--preset", preset]
        )
        run(
            ["cmake", "--build", *build_selector, "--parallel", str(build_jobs)],
            args.dry_run,
            env,
        )
    if args.test:
        test_selector = (
            ["--test-dir", cmake_path(custom_main_dir), "--output-on-failure"]
            if custom_main_dir is not None
            else ["--preset", preset]
        )
        test_cmd = build_ctest_command(test_selector, jobs=args.test_jobs, environment=env)
        run(test_cmd, args.dry_run, env)
    if args.package:
        package_selector = (
            ["--config", cmake_path(custom_main_dir / "CPackConfig.cmake")]
            if custom_main_dir is not None
            else ["--preset", preset]
        )
        run(["cpack", *package_selector], args.dry_run, env)


if __name__ == "__main__":
    main()
