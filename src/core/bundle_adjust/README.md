# core/bundle_adjust 模块

本模块只负责光束法平差问题的数值求解、后端选择和结果质量门控，不负责连接点生成、
增量影像注册或 GUI 项目写回。

## 空三调用链

```text
AerialTriangulationWorkflow
  -> AerialTriangulationPipeline
  -> SfmAttemptRunner
  -> IncrementalSfm
  -> SfmBundleAdjustCoordinator
  -> BundleAdjust::optimizePoints
```

因此 `aerial_triangulation` 不直接包含 `BundleAdjust.h`，但每次局部或全局 BA 都会通过
`sfm` 的协调层调用本模块。协调层负责活动相机、固定边界相机、相似变换规范、进度转发和结果回写；
本模块负责形成并求解非线性最小二乘问题。

## 后端能力

| 后端 | 三维点 | 相机位姿 | 共享焦距 | 控制约束 | Auto 用途 |
|------|--------|----------|----------|----------|-----------|
| Legacy CPU/OpenMP | 支持 | 支持 | 支持 | 支持 | 小型固定焦距问题和明确回退 |
| Ceres CPU | 支持 | 支持 | 联合优化 | 支持 | 共享焦距或中大型 CPU 问题 |
| Ceres CUDA | 支持 | 支持 | 联合优化 | 支持 | 达到相机/观测阈值且 Ceres CUDA 可用 |
| Native CUDA | 支持 | 不支持 | 不支持 | 不支持 | 当前不用于需要相机更新的正式空三 |

Native CUDA 当前是固定相机的三维点块优化器。它尚未实现相机 Schur 补、PCG 和回代，
因此不能称为联合 BA；显式请求不支持的配置会返回 `UnsupportedConfiguration`，
Auto 也不会在 `refineCameraPose=true` 时选择它。

## 性能策略

- Ceres 重投影、物方约束和相机位姿先验使用 AutoDiff/解析导数，不再在热点路径执行中央数值微分。
- Ceres CPU 的 Auto 策略按可变相机规模选择 Dense Schur、Sparse Schur 或
  Iterative Schur；point-only 问题使用 Dense QR。
- Ceres CUDA 在创建求解器前保守估算 dense 工作集，并与当前空闲显存预算比较。
  超限时自动改用 CPU Sparse/Iterative Schur，避免进入求解后才发生显存不足。
- 小型固定焦距问题继续使用 Legacy CPU/OpenMP；小规模联合问题使用 Ceres CPU。只有同时满足
  相机数、观测数和 CUDA 求解器可用性门槛时，Auto 才会选择 Ceres CUDA。
- CUDA 选择依据是完整 setup + solve 墙钟时间和质量门控，不只比较线性求解器内部耗时。
  16 张影像的 dino 回归包含 7856 个 BA 观测，Ceres CPU 完成正式 BA 只需约 70 ms，
  因而不会强制迁移到 GPU。
- `ba_backend_benchmark` 用于比较 Legacy、Ceres CPU、Ceres CUDA 和 Native CUDA；
  `aerial_geometry_benchmark` 负责测量空三外围几何阶段，避免把低收益 kernel 接入默认流程。
- Native CUDA 的公开参数只描述当前真实的固定相机点块能力：
  `nativeCudaMaxPointStepNorm` 限制单次三维点更新；结果报告优化前后代价、接受步和
  拒绝试探次数，不再输出并未执行的 PCG 指标。

## 结果状态与回写

`BAResult` 统一提供：

- `BASolveStatus`：成功、未收敛、取消、输入无效、配置不支持、后端不可用或数值失败。
- `solutionUsable`：求解器结果是否允许调用方使用。
- 请求/实际后端、回退标记与原因。
- 相机、轨迹、观测规模，RMS，setup/solve/total 耗时。

取消、数值失败或不可用结果不会修改调用方传入的相机和三维点。Auto 先运行一个满足能力和规模条件的
候选后端，只有状态不可用或 RMS、有效轨迹等质量门控失败时才运行回退后端。SfM 协调层另行记录
`ba_result_applied`，表示通过规范恢复和质量门控后的结果是否已经写回重建。

## 联合共享焦距

Ceres 使用一个共享绝对焦距参数块，并以 `log(focalPx)` 参数化和设置上下界。相机可以有不同初始焦距，
但同一相机组在求解后得到同一个焦距。焦距、相机位姿和三维点处于同一个 Ceres Problem，
不使用外层交替更新模拟联合自标定。`cameraCalibrationGroupIds` 可为不同镜头/焦段建立独立参数，
默认先固定焦距预热几何，再释放各组焦距。默认只释放焦距，不自动释放主点或畸变。

## 规范、鲁棒目标与正深度

- 单目 SfM 的全局 7 自由度规范由 `SfmBundleAdjustCoordinator` 和
  `SimilarityGaugeNormalizer` 管理，不在各后端内用不同方式重复实现。
- Legacy 的法方程线性化与 LM trial step 使用同一个 Huber robust cost。
- Legacy、Ceres 和 Native CUDA 统一执行正深度检查、基于全局中位数的自适应点过滤与结果统计。
- 后端选择和求解规划只统计有限像点、正权重且形成双相机轨迹的有效观测；零权重、负权重和
  非有限权重不会再被不同后端解释成不同残差贡献。
- Auto 质量门控除重投影 RMS 和有效 track 比例外，还检查 LiDAR、控制点和比例尺 RMS 不得恶化。
- 点的前后方由 `Camera::positiveDepth()` / `isPointInFront()` 定义，后端不得直接把原始相机 Z
  当作跨相机格式的统一正深度。

## 验证

模块测试位于 `tests/`，重点覆盖后端能力和选择、取消/失败保护、共享焦距、质量门控、
投影模型以及 Native CUDA 的固定相机能力。SfM 规范和局部/全局 BA 协调测试位于
`src/core/sfm/test/`。

## PlaMatrix 与 PlaPoint 边界

- BA 的 3x3/6x6 小型法方程和有限样本中位数直接复用 PlaMatrix；相关能力放在
  `plamatrix/ops/small_matrix.h` 与 `plamatrix/ops/statistics.h`，避免 PlaScan
  维护第二套数值实现。
- BA 输入是带观测索引的 track，而不是需要邻域查询的点云容器，因此不强行依赖 PlaPoint。
  LiDAR 最近邻、法线和空间关联仍由 `core/lidar`/PlaPoint 在构建 BA 约束前完成。
