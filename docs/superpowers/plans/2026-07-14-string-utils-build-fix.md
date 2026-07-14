# String Utils Build Integration Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复字符串工具迁移后遗漏的测试目标依赖，并避免两个 QC 测试在 Windows 普通构建阶段因运行时 DLL 路径中断。

**Architecture:** 所有直接编译 `FeatureData.cpp` 的测试目标显式链接 `plascan_common_string_utils`。两个 QC 测试只把 GTest 发现时机改为 `PRE_TEST`，不做全局 PATH 注入或无关测试重构。

**Tech Stack:** CMake 3.31、Ninja、MSVC、GoogleTest、CTest、vcpkg。

---

### Task 1: 固化失败基线

**Files:**
- Inspect: `tests/CMakeLists.txt`
- Inspect: `src/core/qc/CMakeLists.txt`

- [x] **Step 1: 在不添加 vcpkg DLL 路径的环境构建失败目标**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --parallel 3 --target test_feature_file_io test_layer_renderer_batched_overlay test_survey_control_import test_point_cloud_alignment
```

Expected: `test_feature_file_io` 或 `test_layer_renderer_batched_overlay` 因 `StringTransform` 依赖缺失失败，QC 测试在构建期发现阶段可能以 `0xc0000135` 失败。

### Task 2: 补齐 FeatureData 测试目标依赖

**Files:**
- Modify: `tests/CMakeLists.txt`

- [x] **Step 1: 为三个直接编译 FeatureData.cpp 的测试目标添加显式链接依赖**

在下列目标各自的 `target_link_libraries(... PRIVATE ...)` 中加入：

```cmake
plascan_common_string_utils
```

目标为：

```text
test_gui_project_utils
test_layer_renderer_batched_overlay
test_feature_file_io
```

- [x] **Step 2: 构建两个已知失败的 FeatureData 测试目标**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --parallel 3 --target test_feature_file_io test_layer_renderer_batched_overlay
```

Expected: 两个目标均编译、链接成功。

### Task 3: 延迟 QC 测试发现

**Files:**
- Modify: `src/core/qc/CMakeLists.txt`

- [x] **Step 1: 修改两个 QC 测试的发现模式**

使用：

```cmake
gtest_discover_tests(test_survey_control_import DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_point_cloud_alignment DISCOVERY_MODE PRE_TEST)
```

- [x] **Step 2: 在不添加 vcpkg DLL 路径的环境构建 QC 测试**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --parallel 3 --target test_survey_control_import test_point_cloud_alignment
```

Expected: 构建阶段不启动测试可执行文件，命令成功结束。

### Task 4: 集成验证

**Files:**
- Verify: `tests/CMakeLists.txt`
- Verify: `src/core/qc/CMakeLists.txt`

- [x] **Step 1: 使用用户原始形式执行普通构建**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --parallel 32 --config Release
```

Expected: 不再出现本计划覆盖的头文件缺失、未解析符号或 QC 构建期 `0xc0000135`。

- [x] **Step 2: 添加运行时 DLL 路径并运行相关测试**

Run:

```powershell
$vcpkg = (Resolve-Path 'build/windows-vcpkg-cuda-release/vcpkg_installed/x64-windows/bin').Path
$env:PATH = "$vcpkg;$env:PATH"
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R 'CommonString|SurveyControlImport|PointCloudAlignment|test_layer_renderer_batched_overlay|FeatureFileIO'
```

Expected: 本次相关测试全部通过。

- [x] **Step 3: 检查补丁质量和暂存状态**

Run:

```powershell
git diff --check -- tests/CMakeLists.txt src/core/qc/CMakeLists.txt docs/superpowers/specs/2026-07-13-common-string-utils-design.md docs/superpowers/plans/2026-07-14-string-utils-build-fix.md
git diff --cached --quiet
```

Expected: 无空白错误，且没有暂存改动。根据用户约束不提交。
