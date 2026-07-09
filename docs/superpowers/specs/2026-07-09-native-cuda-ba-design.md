# PlaScan 自研 CUDA 全局 BA 后端设计

## 背景

当前 PlaScan 的 BA 已有三类路径：

- `legacy_cpu`：项目原有 CPU/OpenMP BA，当前 hyb2 数据上一轮 BA 的质量最好且速度很快。
- `ceres_cpu`：基于 Ceres 的 CPU BA，便于表达统一残差和约束。
- `ceres_cuda`：基于 Ceres 的 CUDA dense Schur 线性求解路径，只把线性求解的一部分放到 GPU。

现有 `ceres_cuda` 已能确认使用 `dense_schur_cuda`，但在 hyb2 数据上提升有限。主要原因是 residual/Jacobian 构建、问题装配、质量门控对照和部分迭代控制仍在 CPU，且 Ceres CUDA 路径对当前相机模型和数据规模不一定优于 legacy BA。

本设计目标是在现有 BA 接口下新增 PlaScan 自研 CUDA 后端，用于普通视觉空三/SfM 的全局 BA，直接优化相机外参和三维点，减少对 Ceres CUDA 后端能力边界的依赖。

## 首期范围

新增一个自研 CUDA BA 后端，暂定名称为 `native_cuda`，并接入现有入口：

```cpp
BAResult BundleAdjust::optimizePoints(const std::vector<Camera> &cameras,
                                      const std::vector<BATrack> &tracks,
                                      const BAOptions &options);
```

首期支持：

- 固定相机内参和畸变参数。
- 优化相机外参，参数化为每台相机 6 自由度李代数增量。
- 优化三维点坐标，每条有效 track 一个 3 维参数块。
- 支持普通重投影残差、观测权重、Huber 鲁棒核。
- 支持固定相机列表，用于消除 BA gauge 自由度。
- 支持 Auto 后端选择、质量门控、CPU fallback、benchmark 和结果报告。

首期不支持：

- 不优化焦距、主点、畸变参数。
- 不在 CUDA 后端内处理 LiDAR 点到面约束、GCP 软约束、scale bar 约束和相机位姿软先验。
- 不替代所有 legacy/Ceres 场景；遇到不支持的约束或数值失败时回退到现有后端。
- 不在第一版实现稀疏 Cholesky；首期使用 GPU Schur + GPU PCG。

## 架构

新增模块文件建议放在 `src/core/bundle_adjust`：

- `BundleAdjustNativeCuda.h/.cpp`：CPU 侧后端入口、可用性检测、输入过滤、结果组装。
- `BundleAdjustNativeCuda.cu`：CUDA kernel 和 GPU 求解流程。
- `BundleAdjustNativeCudaTypes.h`：GPU 侧扁平数据结构，避免 `.cu` 文件直接依赖 Qt 或复杂 C++ 对象。
- `BundleAdjustNativeCudaKernels.cuh`：必要时拆出 kernel 声明和小型设备函数。

`BundleAdjust.cpp` 继续作为统一调度层：

- 增加 `BABackend::NativeCuda`。
- `isBackendAvailable()` 判断 CUDA 编译开关和运行时设备。
- `selectBackendForProblem()` 在 Auto 模式下只对足够大的全局 BA 选择 `native_cuda`。
- 质量门控沿用现有 `meanRmsAfter`、`validTrackRatio` 和 legacy 对照机制。

## 数据流

CPU 侧输入仍然是 `Camera` 和 `BATrack`。进入 CUDA 后端前，先构建扁平工作集：

1. 过滤无效 track：初值非有限、少于 2 个有效观测、少于 2 个唯一相机的 track 不进入 CUDA 求解。
2. 过滤无效观测：相机索引越界、像点非有限、权重非有限或权重为负的观测被剔除。
3. 生成连续索引：
   - `camera_params[Nc]`：每台相机的固定内参、畸变、当前外参。
   - `point_params[Np]`：每个有效 track 的三维点。
   - `observations[No]`：每条观测记录 camera id、point id、u/v、weight。
4. 生成 block offset：
   - 每个 point 的观测范围。
   - 每个 camera-camera Schur block 的索引。

CUDA 后端输出：

- 优化后的相机外参列表。
- 优化后的三维点，按原始 track 下标写回。
- 每条 track 的 RMS before/after 和 valid 标志。
- 后端统计：观测数、迭代数、PCG 迭代数、setup/solve/total 耗时、fallback 原因。

## 数值算法

首期算法采用 damped Gauss-Newton / Levenberg-Marquardt 风格迭代：

1. GPU kernel 按观测计算重投影残差和解析 Jacobian：
   - `Jc`: 2x6，相机外参增量 Jacobian。
   - `Jp`: 2x3，三维点 Jacobian。
2. GPU 聚合 normal equation block：
   - 相机块 `U_i = sum(Jc^T Jc)`。
   - 点块 `V_j = sum(Jp^T Jp)`。
   - 交叉块 `W_ij = Jc^T Jp`。
   - 右端 `bc_i = -sum(Jc^T r)`，`bp_j = -sum(Jp^T r)`。
3. 对每个点块加阻尼并求 `3x3` 逆：
   - `V_j_lambda = V_j + lambda * diag(V_j)`。
   - 无法稳定求逆的点标记为本轮无效。
4. 构造 Schur reduced camera system：
   - `S = U - W V^-1 W^T`。
   - `rhs = bc - W V^-1 bp`。
5. GPU PCG 求解相机增量：
   - 首期使用 block-Jacobi 预条件。
   - 固定相机对应参数块置零或从系统中消除。
6. GPU 回代求点增量：
   - `dp = V^-1 (bp - W^T dc)`。
7. GPU/CPU 协同计算 trial cost，CPU 侧执行 LM 接受/拒绝和阻尼更新。
8. 收敛后写回相机和点，复用现有质量门控。

## 后端选择策略

`native_cuda` 不作为小问题默认后端。Auto 模式建议满足以下条件才候选：

- `refineCameraPose == true`。
- 没有 LiDAR/GCP/scale bar/pose prior 等 CUDA 首期不支持的软约束。
- 有效相机数不低于 `minNativeCudaCameras`。
- 有效观测数不低于 `minNativeCudaObservations`。
- 固定相机集合非空，或调度层自动固定首个已注册相机。

显式请求 `native_cuda` 时：

- 如果编译或运行时 CUDA 不可用，按 `allowBackendFallback` 回退。
- 如果输入包含不支持约束，按 `allowBackendFallback` 回退。
- 如果 CUDA 求解中出现非有限 cost、PCG 失败或质量门控失败，返回明确 `backendMessage`，Auto 下回退 legacy。

## 与现有功能的关系

CLI/GUI 新增后端字符串 `native_cuda`。现有字段继续保留：

- `ba_requested_backend`
- `ba_used_backend`
- `ba_used_gpu`
- `ba_backend_fallback`
- `ba_backend_selection_reason`
- `ba_quality_gate_rejected`
- `ba_valid_track_ratio`

新增诊断字段建议：

- `ba_native_cuda_pcg_iterations`
- `ba_native_cuda_linear_residual`
- `ba_native_cuda_accepted_steps`
- `ba_native_cuda_rejected_steps`
- `ba_native_cuda_active_cameras`
- `ba_native_cuda_active_tracks`
- `ba_native_cuda_active_observations`

benchmark 工具增加 `native_cuda` 后端，并默认输出 legacy、ceres_cuda、native_cuda、auto 对照。

## 测试策略

采用测试驱动开发，先写测试再实现：

1. 编译契约测试：
   - 无 CUDA 构建时 `native_cuda` 不可用且能回退。
   - 有 CUDA 构建时 `isBackendAvailable(NativeCuda)` 返回真实设备状态。
2. 小型合成数值测试：
   - 两到五台相机、几十个点，CUDA BA 的 RMS 应下降。
   - 固定首相机后不发生整体 gauge 漂移。
   - CUDA 与 Ceres/legacy 在简单场景达到相近 RMS。
3. 后端回退测试：
   - 含 LiDAR/GCP/scale bar 时回退。
   - PCG 失败或非有限输入时返回明确错误并回退。
4. SfM 集成测试：
   - 现有 `SfmPipelineTest.ThreeImageIncremental` 不回归。
   - 新增一个中等规模 synthetic global BA 测试，验证 Auto 可选择 `native_cuda`。
5. 性能基准：
   - synthetic small/medium/large 三档。
   - hyb2 数据至少记录 1 轮、3 轮、8 轮 BA 的 RMS、有效 track 比例、总耗时和 GPU 使用情况。

## 验收标准

首期完成时应满足：

- CUDA 构建可通过，CPU-only 构建不受影响。
- `native_cuda` 后端能完成固定内参的 camera + point BA。
- 在 synthetic 中等规模问题上，RMS 明显下降，且与 Ceres/legacy 结果同量级。
- Auto 模式不会在不适合 CUDA 的小问题或带未支持约束的问题上误选 `native_cuda`。
- hyb2 数据上，如果 `native_cuda` 质量低于 legacy，Auto 必须回退；如果质量满足门控，需要报告实际速度和 GPU 求解指标。

## 风险与控制

- Schur block 图构建复杂，容易出现索引错误。控制方式是先做 CPU reference builder，并用小图逐项比对 GPU 输出。
- PCG 收敛依赖尺度和预条件。首期保留最大迭代、残差阈值和 fallback，避免卡死 GUI。
- 自研投影/Jacobian 需要与 `Camera::projectWorldPoint` 对齐。首期必须做有限差分校验测试。
- GPU 原子累加可能引入非确定性。验收应使用误差容差，不要求逐 bit 一致。
- 首期不支持约束残差，必须在后端选择阶段显式检测并回退，不能静默忽略约束。

## 实施边界

该设计文档只定义自研 CUDA BA 的首期设计，不包含代码实现。实现前需要再拆分为详细 TDD 计划，按可编译、可验证、可回退的阶段逐步推进。
