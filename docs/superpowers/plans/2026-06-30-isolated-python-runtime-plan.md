# Isolated Python Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed, cross-platform PlaScan Python runtime at `build/env/python-runtime` and wire it into developer shells and model export scripts.

**Architecture:** Keep the existing conda/venv helper for compatibility, and add a focused runtime setup script that always targets `build/env/python-runtime` unless explicitly overridden. Generated env files remain under `build/env/`, while Windows developer shells auto-detect the runtime and export `PLASCAN_PYTHON_EXECUTABLE`, `PLASCAN_MODEL_DIR`, and `PLASCAN_SCRIPT_DIR`.

**Tech Stack:** Python 3 venv, PowerShell, CMake/CTest Python unittest.

---

### Task 1: Add Runtime Behavior Tests

**Files:**
- Create: `tests/test_setup_python_runtime.py`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing tests**

Create tests that import `scripts/env/setup_python_runtime.py` and assert:

```python
runtime_dir = spr.default_runtime_dir(repo_root)
assert runtime_dir == repo_root / "build" / "env" / "python-runtime"
assert spr.runtime_python_path(runtime_dir, "windows") == runtime_dir / "Scripts" / "python.exe"
assert spr.runtime_python_path(runtime_dir, "linux") == runtime_dir / "bin" / "python"
```

Also test that `environment_values()` writes `PLASCAN_PYTHON_EXECUTABLE`, `PLASCAN_PYTHON`, `PLASCAN_MODEL_DIR`, `PLASCAN_SCRIPT_DIR`, `PYTHONUTF8`, and `PYTHONIOENCODING`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m unittest tests.test_setup_python_runtime
```

Expected: failure because `scripts/env/setup_python_runtime.py` does not exist.

### Task 2: Implement Runtime Setup Script

**Files:**
- Create: `scripts/env/setup_python_runtime.py`

- [ ] **Step 1: Implement path helpers**

Add `default_runtime_dir(repo_root)`, `runtime_python_path(runtime_dir, platform_name)`, `dependency_plan(device, cuda_wheel)`, and `environment_values(...)`.

- [ ] **Step 2: Implement CLI**

Support:

```text
--source-dir
--runtime-dir
--device cpu|cuda
--cuda-wheel
--python
--skip-create
--skip-install
--dry-run
--extra-package
```

Use `python -m venv` for creation, install PyTorch from the selected wheel index, install `numpy`, `opencv-python`, `kornia`, and LightGlue, then write `plascan-env.json/.cmake/.sh/.ps1`.

- [ ] **Step 3: Run tests to verify pass**

Run:

```powershell
python -m unittest tests.test_setup_python_runtime
python -m py_compile scripts/env/setup_python_runtime.py
```

Expected: both pass.

### Task 3: Wire Runtime Into Windows Dev Shell

**Files:**
- Modify: `scripts/build_win/enter_plascan_dev_shell.ps1`
- Extend: `tests/test_setup_python_runtime.py`

- [ ] **Step 1: Add failing static test**

Assert the PowerShell script contains `PLASCAN_PYTHON_EXECUTABLE`, `PLASCAN_MODEL_DIR`, `PLASCAN_SCRIPT_DIR`, and `python-runtime`.

- [ ] **Step 2: Implement shell wiring**

If `build/env/python-runtime/Scripts/python.exe` exists, prepend `Scripts` and runtime root to `PATH`, then set `PLASCAN_PYTHON_EXECUTABLE` and `PLASCAN_PYTHON`. Always set model/script directories.

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m unittest tests.test_setup_python_runtime
```

Expected: pass.

### Task 4: Update Documentation

**Files:**
- Modify: `scripts/env/README.md`

- [ ] **Step 1: Document fixed runtime path**

Show Windows and Linux commands for:

```powershell
python scripts\env\setup_python_runtime.py --device cuda --cuda-wheel cu130
```

```bash
python3 scripts/env/setup_python_runtime.py --device cpu
```

- [ ] **Step 2: Mention legacy script**

Clarify `setup_python_env.py` remains for custom conda/venv environments, but the recommended development and packaging path is `build/env/python-runtime`.

### Task 5: Final Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run focused tests**

```powershell
python -m unittest tests.test_setup_python_runtime
python -m py_compile scripts/env/setup_python_runtime.py scripts/env/setup_python_env.py
```

- [ ] **Step 2: Run CTest entry if configured**

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release -R PythonRuntimeSetupTest --output-on-failure
```

Expected: pass if the build tree has been configured after the CMake test entry was added.
