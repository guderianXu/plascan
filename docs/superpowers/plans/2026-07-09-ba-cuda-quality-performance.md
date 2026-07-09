# BA CUDA Quality and Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前“Ceres CUDA 可用但不一定更快/更准”的 BA 后端优化成质量受控、性能可预测、能自动选择最合适 CPU/GPU 路径的生产级光束法平差。

**Architecture:** 先建立真实数据和 synthetic 的可重复基准，避免只看单次耗时。然后把 BA 分成输入尺度归一化、残差质量门控、后端调度、Ceres solver 策略和报告输出五层；每层都用测试保护。CUDA 只在总耗时和质量都达标时进入自动路径，否则显式 fallback 到 legacy 或 Ceres CPU。

**Tech Stack:** C++17, CMake, GTest, Ceres Solver, CUDA, Qt6, Python benchmark runner, hyb2 project smoke data.

---

## File Structure

- Modify: `src/core/bundle_adjust/BundleAdjust.h`
  - 增加 BA 输入统计、质量门控、后端选择原因和尺度归一化配置。
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
  - 完善 auto backend 选择策略，加入质量失败 fallback 和 legacy/Ceres 对照入口。
- Modify: `src/core/bundle_adjust/BundleAdjustCeres.cpp`
  - 优化 Ceres solver options、参数尺度、残差有效性、CUDA/CPU solver 切换。
- Modify: `src/core/bundle_adjust/tools/ba_backend_benchmark.cpp`
  - 增加真实 BA 问题结构统计、pose/point-only 分组、JSON/CSV 输出字段。
- Modify: `scripts/bench/run_ba_backend_benchmark.py`
  - 扩展为批量运行 small/medium/hyb2，自动汇总速度、RMS、有效点比例。
- Create: `src/core/bundle_adjust/tests/test_bundle_adjust_quality_gate.cpp`
  - 覆盖“Ceres/CUDA 结果变差时必须拒绝或 fallback”的规则。
- Modify: `src/core/sfm/pipeline/IncrementalSfm.cpp`
  - 把 known-pose BA 拒绝回写逻辑抽成清晰 helper，并记录拒绝原因。
- Create: `docs/benchmarks/ba-cuda-hyb2-2026-07-09.md`
  - 记录 hyb2 三后端真实对照结果和后续阈值依据。

---

## Task 1: 固化 BA 质量和性能基准

**Files:**
- Modify: `src/core/bundle_adjust/tools/ba_backend_benchmark.cpp`
- Modify: `scripts/bench/run_ba_backend_benchmark.py`
- Create: `docs/benchmarks/ba-cuda-hyb2-2026-07-09.md`

- [x] **Step 1: 扩展 benchmark 输出字段**

在 `ba_backend_benchmark.cpp` 的输出中补充：

```text
requested,used,gpu,fallback,solver,observations,valid_ratio,rms_before,rms_after,setup_seconds,solve_seconds,total_seconds
```

Expected: synthetic 和 hyb2 对照都能直接比较“质量 + 总耗时”，不只看 solver 时间。

- [x] **Step 2: 增加批量运行脚本参数**

在 `scripts/bench/run_ba_backend_benchmark.py` 增加：

```powershell
--cases small,medium,large
--backends legacy_cpu,ceres_cpu,ceres_cuda,auto
--repeat 3
--summary-json <path>
```

Expected: 每个 case/backend 重复 3 次，输出 median total seconds。

- [x] **Step 3: 跑 synthetic 基线**

Run:

```powershell
python scripts\bench\run_ba_backend_benchmark.py --exe E:\code\plascan\build\windows-vcpkg-cuda-release\bin\ba_backend_benchmark.exe --out E:\code\plascan\build\ba-bench\synthetic-baseline.csv --summary-json E:\code\plascan\build\ba-bench\synthetic-baseline.json --cases small,medium,large --backends legacy_cpu,ceres_cpu,ceres_cuda,auto --repeat 3
```

Expected: CSV/JSON 中每个 backend 都有 median 耗时、RMS 和有效点比例。

- [x] **Step 4: 跑 hyb2 三后端真实基线**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\bundle_adjust_cli.exe E:\code\test\hyb2\hyb2.plascan --output-dir E:\code\plascan\build\ba-bench\hyb2-legacy --ba-backend legacy_cpu --threads 32 --max-iterations 3 --max-point-iterations 3 --max-camera-iterations 2 --force
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\bundle_adjust_cli.exe E:\code\test\hyb2\hyb2.plascan --output-dir E:\code\plascan\build\ba-bench\hyb2-ceres-cpu --ba-backend ceres_cpu --threads 32 --max-iterations 3 --max-point-iterations 3 --max-camera-iterations 2 --force
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\bundle_adjust_cli.exe E:\code\test\hyb2\hyb2.plascan --output-dir E:\code\plascan\build\ba-bench\hyb2-ceres-cuda --ba-backend ceres_cuda --threads 32 --max-iterations 3 --max-point-iterations 3 --max-camera-iterations 2 --force
```

Expected: 文档记录三者 RMS、有效 tracks、总耗时、GPU/fallback 状态。

---

## Task 2: 增加 BA 结果质量门控和 fallback

**Files:**
- Create: `src/core/bundle_adjust/tests/test_bundle_adjust_quality_gate.cpp`
- Modify: `src/core/bundle_adjust/BundleAdjust.h`
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
- Modify: `tests/CMakeLists.txt`

- [x] **Step 1: 写失败测试**

测试目标：当 Ceres/CUDA 输出 `meanRmsAfter` 非有限、超过输入 RMS 明显过多，或有效点比例过低时，`Auto` 必须 fallback 到 legacy。

Expected red: 当前 `Auto` 只按规模选后端，不会做质量对照 fallback。

- [x] **Step 2: 增加配置字段**

在 `BAOptions` 增加：

```cpp
bool enableBackendQualityGate = true;
double maxAcceptedRmsGrowth = 1.25;
double minAcceptedValidTrackRatio = 0.60;
bool compareAutoBackendWithLegacy = true;
```

在 `BAResult` 增加：

```cpp
bool qualityGateRejected = false;
std::string qualityGateMessage;
double validTrackRatio = 0.0;
```

- [x] **Step 3: 实现 Auto 质量对照**

`BundleAdjust::optimizePoints` 在 `backend=Auto` 时：

1. 先按规模选择 candidate backend。
2. candidate 为 Ceres/CUDA 时运行 candidate。
3. 如果质量门控失败，运行 legacy。
4. 返回 legacy 结果，并记录 `backendFallback=true`、`qualityGateRejected=true`。

Expected green: 新测试通过，hyb2 不会在 CUDA RMS/耗时明显差于 legacy 时自动采用 CUDA。

---

## Task 3: 优化 Ceres 参数尺度和 residual 稳定性

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjustCeres.cpp`
- Create/Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_ceres_backend.cpp`

- [ ] **Step 1: 增加尺度敏感回归测试**

构造大坐标/小像素残差场景，验证 Ceres CPU/CUDA 不会把点推到负深度或 `inf RMS`。

- [ ] **Step 2: 在 Ceres BA 中做问题尺度归一化**

实现策略：

```text
center = tracks initialPoint median center
scale = median distance to center
point parameter = (world - center) / scale
projection before residual = center + scale * normalizedPoint
```

Expected: Ceres 对真实工程尺度更稳定，减少大坐标病态。

- [ ] **Step 3: 收紧 Ceres solver options**

调整：

```cpp
solverOptions.max_num_iterations = options.maxIterations;
solverOptions.function_tolerance = 1e-8;
solverOptions.gradient_tolerance = 1e-10;
solverOptions.parameter_tolerance = options.stepTolerance;
solverOptions.use_nonmonotonic_steps = false;
```

Expected: Ceres 不接受让重投影误差变差的解。

---

## Task 4: 分离 point-only BA 和 pose+point BA 后端策略

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Modify: `src/core/sfm/pipeline/IncrementalSfm.cpp`
- Modify: `tests/test_ba_cuda_contracts.py`

- [x] **Step 1: 明确策略**

策略表：

```text
point-only BA: 默认 legacy_cpu，除非用户显式 ceres_cuda
pose+point BA small: legacy_cpu 或 ceres_cpu，禁止 auto CUDA
pose+point BA large: Ceres CUDA candidate + quality gate
known-pose BA: 必须通过 prior/RMS gate 才回写
```

- [x] **Step 2: 修改 auto backend 选择**

`selectBackendForProblem` 增加 `refineCameraPose` 分支：

```text
refineCameraPose=false -> prefer legacy_cpu
refineCameraPose=true && observations >= cudaThreshold -> ceres_cuda candidate
refineCameraPose=true && observations >= cpuThreshold -> ceres_cpu candidate
```

Expected: hyb2 如果是固定相机 point-only BA，auto 不再盲目选择 CUDA。

- [x] **Step 3: 更新空三默认**

空三内部 BA 使用 `Auto + quality gate`，并把实际选择原因写进日志：

```text
BA backend auto: selected legacy_cpu because point-only BA
BA backend auto: selected ceres_cuda because pose+point observations >= threshold
BA backend auto: rejected ceres_cuda because RMS grew
```

---

## Task 5: 报告和 UI 可解释化

**Files:**
- Modify: `src/gui/dialogs/BundleAdjustDialog.cpp`
- Modify: `src/gui/project/services/BundleAdjustService.cpp`
- Modify: `src/cli/cli_bundle_adjust.cpp`
- Modify: `README.md`

- [x] **Step 1: CLI 输出选择原因**

`bundle_adjust_cli` 输出增加：

```text
backend_selection_reason
quality_gate_rejected
quality_gate_message
valid_track_ratio
```

- [x] **Step 2: GUI 摘要显示真实后端和拒绝原因**

GUI 完成后摘要显示：

```text
请求后端: auto
候选后端: ceres_cuda
实际后端: legacy_cpu
原因: CUDA 候选 RMS 增长超过阈值，已回退
```

- [x] **Step 3: README 写清能力边界**

写明：

```text
Ceres CUDA 当前加速 dense Schur 线性求解，不加速 residual/Jacobian 构建。
默认 auto 会优先保证质量和总耗时，不保证只要有 GPU 就使用 GPU。
```

---

## Task 6: 最终验证矩阵

**Files:**
- Modify: `docs/benchmarks/ba-cuda-hyb2-2026-07-09.md`

- [ ] **Step 1: 编译**

Run:

```powershell
cmd.exe /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target bundle_adjust test_bundle_adjust_projection test_bundle_adjust_backend_selection test_bundle_adjust_ceres_backend test_bundle_adjust_quality_gate test_sfm_pipeline ba_backend_benchmark bundle_adjust_cli reconstruct_pipeline_cli plascan_gui --parallel 32"
```

Expected: exit `0`.

- [ ] **Step 2: 聚焦测试**

Run:

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "BundleAdjust|bundle_adjust|Sfm|Aerial"
python -m unittest tests.test_ba_cuda_contracts
```

Expected: CTest 全部通过，Python 合约测试通过。

- [ ] **Step 3: hyb2 验证**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\bundle_adjust_cli.exe E:\code\test\hyb2\hyb2.plascan --output-dir E:\code\plascan\build\ba-bench\hyb2-auto-final --ba-backend auto --threads 32 --max-iterations 3 --max-point-iterations 3 --max-camera-iterations 2 --force
```

Expected:

```text
RMS 后有限
valid_track_ratio >= 0.60
若 CUDA 比 legacy 慢或 RMS 更差，auto 应回退并写明原因
```

## Success Criteria

- CUDA backend 在显式 `ceres_cuda` 时能实际使用 `dense_schur_cuda`。
- `auto` 不再只按问题规模选择 CUDA，而是同时考虑质量门控和 backend 类型。
- hyb2 的 `auto` 结果不能比 legacy 明显更差；如果 CUDA 更差，应自动 fallback。
- 所有 BA/SfM/Aerial 聚焦测试通过。
- 文档中有真实 benchmark 表，不再只写“支持 CUDA”。
