# Shared Model CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 GUI/CLI 共用的模型生成核心入口，并用 Dino 的真实 GUI 参数修复 Poisson 主体清理失败。

**Architecture:** `ModelWorkflowService::buildModel()` 是唯一的参数映射和模型流程分发入口。Qt GUI 与
`mesh_reconstruct_cli` 只解析各自上下文并提交相同的 `ModelBuildRequest`，从而使 CLI 回归等价于 GUI。

**Tech Stack:** C++17、Qt6 Core JSON、CLI11、PlaPoint、GoogleTest、CMake/Ninja。

---

### Task 1: 共享模型入口

**Files:**
- Modify: `src/core/mesh/ModelWorkflowService.h`
- Modify: `src/core/mesh/ModelWorkflowService.cpp`
- Test: `tests/test_mesh_reconstructor.cpp`

- [ ] 先写使用 `ModelBuildRequest` 和 `buildModel()` 的失败测试。
- [ ] 编译 `test_mesh_reconstructor`，确认因接口不存在而失败。
- [ ] 实现设置映射、深度图分发、点云分发和 payload 源信息。
- [ ] 运行 `test_mesh_reconstructor`，确认测试通过。

### Task 2: GUI 迁移到共享入口

**Files:**
- Modify: `src/gui/project/manager/ProjectModelManager.cpp`
- Test: `tests/test_source_contracts.cpp`

- [ ] 先写源码契约测试，要求 manager 只调用 `workflow::buildModel()`。
- [ ] 运行契约测试，确认旧的两处分支调用导致失败。
- [ ] 保留 GUI 的源路径解析和异步回调，使用一个 `ModelBuildRequest` 调用共享入口。
- [ ] 运行 GUI 契约测试。

### Task 3: 等价模型 CLI

**Files:**
- Create: `src/cli/cli_mesh_reconstruct.cpp`
- Modify: `src/cli/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Modify: `tests/test_cli_contracts.cpp`

- [ ] 先写 CLI 源码/构建契约测试及最小进程测试。
- [ ] 运行测试，确认缺少目标和可执行文件。
- [ ] 实现 `--source-data`、`--depth-map-dir`、`--dense-cloud`、`--point-cloud`、
  `--output-dir`、`--settings-json`、`--settings-key` 参数。
- [ ] CLI 读取与 GUI 相同的设置对象并调用 `workflow::buildModel()`。
- [ ] 构建并运行 CLI 测试。

### Task 4: Dino 等价复现与 Poisson 修复

**Files:**
- Modify: `src/core/mesh/SurfaceReconstructor.cpp`
- Modify: `src/core/mesh/SurfaceReconstructorPostprocess.cpp`
- Test: `tests/test_mesh_reconstructor.cpp`

- [ ] 用 CLI 和 Dino `generate_model` JSON 运行，记录清理前后顶点、面和连通分量统计。
- [ ] 根据复现结果写最小失败测试，要求清理保留最大的有效主体，且不回退高度格网。
- [ ] 修改清理策略，使阈值不会把所有分量清空，并在错误中保留清理统计。
- [ ] 运行单元测试和 Dino CLI，确认生成有效 PLY 网格。

### Task 5: 验证

**Files:**
- Verify only.

- [ ] 构建 `mesh_reconstruct_cli`、`plascan_gui`、`test_mesh_reconstructor`、`test_cli_contracts`、
  `test_source_contracts`。
- [ ] 运行相关测试并确认零失败。
- [ ] 使用 Dino GUI JSON 再跑 CLI，检查 payload、PLY 顶点/面及算法字段。
- [ ] 运行 `git diff --check`，确认无临时本机测试残留。
