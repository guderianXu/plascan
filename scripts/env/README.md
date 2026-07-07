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

The recommended development runtime is the repository-local standard-library `venv` at `.venv`. Use this runtime for
model export helpers such as LightGlue TorchScript export and for Python tests, so PlaScan development does not depend
on a user-managed conda environment or a per-build virtual environment.

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
LightGlue is installed from `https://github.com/cvg/LightGlue.git`, so the full install requires `git` and network
access to GitHub.

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

## LibTorch

Register an existing LibTorch directory:

```bash
python scripts/env/setup_libtorch.py --libtorch-root /opt/libtorch --cuda-root /usr/local/cuda-12.8
```

Download and extract LibTorch into `build/env/libtorch`:

```bash
python scripts/env/setup_libtorch.py --device cuda --version 2.7.1 --cuda-wheel cu128
```

Use `--url` when the default PyTorch archive URL does not match the release you want.

## Configure

Configure with the generated environment file:

```bash
python scripts/env/configure_with_env.py --build-type release
```

Configure, build, test, and package:

```bash
python scripts/env/configure_with_env.py --build-type release --build --test --package
```

The configure script passes `Torch_DIR`, `PLASCAN_TORCH_DIR`, `CUDAToolkit_ROOT`, and
`CUDA_TOOLKIT_ROOT_DIR` to CMake when those values exist in `plascan-env.json`. It also
reads `plascan-vcpkg.json` when present and exports `VCPKG_ROOT` for presets that use the
vcpkg toolchain.
