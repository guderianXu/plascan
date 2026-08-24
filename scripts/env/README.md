# PlaScan Environment Scripts

This directory contains machine-local environment setup helpers. The scripts write generated environment files to
`build/env/` by default so machine-local paths do not enter the source tree.

## vcpkg

Register an existing vcpkg checkout and write PlaScan vcpkg config:

```bash
python scripts/env/setup_vcpkg.py --root /path/to/vcpkg
```

Clone, bootstrap, and install PlaScan manifest dependencies:

```bash
python scripts/env/setup_vcpkg.py --root build/env/vcpkg --clone --install
```

Windows PowerShell:

```powershell
python scripts\env\setup_vcpkg.py --root C:\src\vcpkg --clone --install --triplet x64-windows
```

The script writes:

- `build/env/plascan-vcpkg.json`
- `build/env/plascan-vcpkg.cmake`
- `build/env/plascan-vcpkg.sh`
- `build/env/plascan-vcpkg.ps1`

Use `--dry-run` to print the clone/bootstrap/install commands without running them.

## Python Runtime

The installed GUI checks for Python on startup. If no runtime is available, it can download a signed Python installer
from python.org and create a per-user managed runtime without administrator privileges. The same workflow remains available
from `Help > Update Python Environment...`; users may dismiss the startup prompt and suppress future reminders. CPack installs
`bootstrap_python_runtime.ps1`, `setup_python_runtime.py`, and `env_common.py` under `share/plascan/scripts/env`.

The recommended development runtime is the repository-local standard-library `venv` at `.venv`. Use this runtime for
model export helpers such as LightGlue and LoMa-R TensorRT export and for Python tests, so PlaScan development does not depend
on a user-managed conda environment or a per-build virtual environment.
The base runtime also installs NumPy and SciPy; the LiDAR benchmark adapters use SciPy's KD-tree for scalable surface-normal
estimation instead of an all-pairs point search.

Windows CUDA development runtime:

```powershell
python scripts\env\setup_python_runtime.py --device cuda --cuda-wheel cu130
```

Linux CPU runtime:

```bash
python3 scripts/env/setup_python_runtime.py --device cpu
```

Linux CUDA runtime:

```bash
python3 scripts/env/setup_python_runtime.py --device cuda --cuda-wheel cu130
```

Use `--skip-install` when the runtime already contains the required packages and only the generated environment
files need to be refreshed.
LightGlue is installed from its GitHub source archive, so the full install requires network access to GitHub but does not
require a separate `git` executable.

The `.venv/` directory is intentionally ignored by git. Reuse it for development instead of creating new virtual
environments under individual build directories. Use `--runtime-dir` only when packaging or CI needs a different
machine-local runtime location.

The runtime setup script writes:

- `build/env/plascan-env.json`
- `build/env/plascan-env.cmake`
- `build/env/plascan-env.sh`
- `build/env/plascan-env.ps1`

`scripts/build_win/enter_plascan_dev_shell.ps1` automatically detects `.venv/Scripts/python.exe`, prepends it to
`PATH`, and exports:

- `PLASCAN_PYTHON_EXECUTABLE`
- `PLASCAN_PYTHON`
- `PLASCAN_MODEL_DIR`
- `PLASCAN_SCRIPT_DIR`

The older `setup_python_env.py` script remains available for custom conda or non-standard venv locations:

```bash
python scripts/env/setup_python_env.py --manager conda --name plascan --device cpu
python scripts/env/setup_python_env.py --manager conda --name plascan --device cuda --cuda-wheel cu128
```

## Configure

Configure with the generated environment file:

```bash
python scripts/env/configure_with_env.py --source-deps --build-type release
```

Configure, build, and test with the pinned OpenCV 5 source package:

```bash
python scripts/env/configure_with_env.py --source-deps --build-type release --build --test
```

## Pinned Qt/OpenCV/GDAL/AprilTag source build

Initialize only the source modules required by PlaScan (do not recursively initialize the complete Qt supermodule):

```bash
git submodule update --init 3rdparty/qt 3rdparty/opencv 3rdparty/gdal 3rdparty/apriltag 3rdparty/PoissonRecon
git -C 3rdparty/qt submodule update --init qtbase qtshadertools
```

Build and install Qt 6.11.2, OpenCV 5.0.0, GDAL 3.12.4, and AprilTag 3.4.5 before configuring PlaScan. OpenCV contrib is not required. PoissonRecon is consumed directly from its pinned submodule:

```bash
python scripts/env/configure_with_env.py --source-deps --build --test
```

To place the PlaScan build tree and all source-dependency build content below a custom directory, pass
`--build-dir`; no `CMakeUserPresets.json` override is required:

```powershell
python scripts\env\configure_with_env.py --source-deps --build --test `
  --build-dir E:\code\plascan\build\8_23build
```

PlaScan is built directly in the selected directory. Source dependency builds, their shared install prefix, and
their vcpkg installed tree are kept below `<build-dir>/source-deps/`.

The host-specific source presets use `build/<platform>-source-deps-release/install` as their shared prefix and a
dedicated vcpkg manifest that excludes Qt, OpenCV, GDAL, and AprilTag. PlaScan configuration verifies exact versions and that
the loaded package configs came from this prefix. ONNX Runtime 1.29.0 release archives are SHA-256 verified and reused
from `build/env/downloads/onnxruntime/1.29.0/`.

CUDA and TensorRT are preferred by default on supported Windows and Linux hosts. Missing CUDA falls back to CPU; a
missing TensorRT SDK keeps native CUDA enabled and falls back to ONNX Runtime CPU inference. On Windows, accept the
NVIDIA license once to automatically download, verify, cache, and install the pinned TensorRT 10.15.1.29 CUDA 13.1 SDK:

```powershell
python scripts\env\configure_with_env.py --source-deps --build --test --accept-tensorrt-license
```

The archive cache is `build/env/downloads/tensorrt/10.15.1.29/` and the extracted SDK is
`build/env/sdk/tensorrt/10.15.1.29/`. Subsequent builds discover it automatically. Existing SDKs and archives can be
selected with `--tensorrt-root` and `--tensorrt-archive`; `--no-tensorrt-auto-install` disables acquisition.
The configure script detects local GPU compute capabilities through `nvidia-smi`, so a workstation build targets only
the installed GPUs by default. Pass `-DPLASCAN_CUDA_ARCHITECTURES=75;86;89;120` for a portable release build. Builds
use all logical CPUs by default and can be overridden with `--build-jobs <n>` or `CMAKE_BUILD_PARALLEL_LEVEL`.

`--test` 默认使用全部逻辑线程并行运行 CTest，可通过 `--test-jobs <n>` 或
`CTEST_PARALLEL_LEVEL` 覆盖。只运行已有构建树中的测试时，使用统一测试入口：

```bash
python scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure
```

Windows、Linux 都默认使用全部逻辑线程；其余参数会原样转发给 CTest，
也可以直接传入原生 `--parallel <n>` 或 `-j<n>` 覆盖。

The configure script passes `CUDAToolkit_ROOT` and `CUDA_TOOLKIT_ROOT_DIR` to CMake when
those values exist in `plascan-env.json`. It also
reads `plascan-vcpkg.json` when present and exports `VCPKG_ROOT` for presets that use the
vcpkg toolchain.
