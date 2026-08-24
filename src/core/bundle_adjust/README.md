# core/bundle_adjust 模块

本模块只负责光束法平差问题的数值求解、后端选择和结果质量门控，不负责连接点生成、
增量影像注册或 GUI 项目写回。

## 公共接口分层

- `BundleAdjustTypes.h`：后端、求解状态、能力和问题规模等基础类型。
- `BundleAdjustProblem.h`：像点观测、三维轨迹以及控制点、比例尺和 LiDAR 约束。
- `BundleAdjustOptions.h`：数值策略、标定参数、后端选择、任务取消和进度回调。
- `BundleAdjustResult.h`：优化后的相机、点、独立激光 shot 和后端诊断统计。
- `BundleAdjustSolver.h`：`BundleAdjust` 求解器门面，也是执行优化的正式公共入口。

各调用方必须直接包含所需的细分头，不保留旧聚合头或类型转发层。细分头只组织公共契约，
不改变 `bundle_adjust` 的独立 target。SfM 和 LiDAR 继续共同依赖该 target，
SfM 自己的局部/全局/分层 BA 调度仍位于 `sfm/pipeline`。

## 空三调用链

```text
AerialTriangulationWorkflow
  -> AerialTriangulationPipeline
  -> SfmAttemptRunner
  -> IncrementalSfm
  -> SfmBundleAdjustCoordinator
  -> BundleAdjust::optimizePoints
```

因此 `aerial_triangulation` 不直接包含 `BundleAdjustSolver.h`，但每次局部或全局 BA 都会通过
`sfm` 的协调层调用本模块。协调层负责活动相机、固定边界相机、相似变换规范、进度转发和结果回写；
本模块负责形成并求解非线性最小二乘问题。

## 后端能力

| 后端 | 三维点 | 相机位姿 | 共享焦距 | 控制约束 | Auto 用途 |
|------|--------|----------|----------|----------|-----------|
| Legacy CPU/OpenMP | 支持 | 有限 | 有限 | 有限 | point-only 固定相机问题 |
| PlaMatrix CPU/CUDA/OpenCL | 支持 | 支持 | 完整 Brown 分组共享内参 | GCP/LiDAR/比例尺/姿态/激光测距 | 联合 BA 正式后端，按规模自动选择 |

三个 PlaMatrix 后端共用解析 Brown-Conrady 相机、点和
九参数共享内参雅可比，以及 GCP、LiDAR 平面/测距、比例尺、位姿先验和相机平面约束；只切换
CPU/CUDA/OpenCL Schur-PCG。CUDA/OpenCL 在设备端装配 Schur CSR 数值并显式报告该路径，设备不可用时
回退 PlaMatrix CPU；显式禁止回退时返回 `BackendUnavailable`。

## 性能策略

- PlaMatrix 使用解析 Brown-Conrady 重投影雅可比和约束雅可比；line-scan 投影使用共享有限差分装配。
- PlaMatrix LM 在一次求解内复用法方程块拓扑、线程局部装配器、轨迹负载分区、Schur pattern 和 trial 状态
  缓冲；重复迭代只清零数值，不再重建相同容器。CPU 矩阵自由 Schur-PCG 默认使用开销更低的
  block-Jacobi，保留可配置的 cluster-Jacobi 供特定强耦合网络基准选用；早期 LM 步采用动态非精确线性容差，
  接近收敛时自动收紧。CPU 容差上限为 `2e-3`，CUDA/OpenCL 上限为 `1e-3`，以保持跨后端结果等价性。
- 小规模稠密 Schur Cholesky 数值失败时明确记录原因并切换 CPU Schur-PCG；结果分别报告法方程装配、
  trial 目标函数、状态复制、预条件器、实际线性容差范围和稠密回退次数。
- point-only 固定相机问题使用 Legacy CPU/OpenMP；小规模联合问题使用 PlaMatrix CPU。达到相机数和
  观测数门槛时，Auto 优先选择 PlaMatrix CUDA，其次选择 PlaMatrix OpenCL。
- 层级 BA 的独立块固定使用 PlaMatrix CPU 并发，避免多个并行块争抢同一 GPU 上下文。
- CUDA 选择依据是完整 setup + solve 墙钟时间和质量门控，不只比较线性求解器内部耗时；小问题不会强制迁移到 GPU。
- `ba_backend_benchmark` 用于比较 Legacy 和 PlaMatrix CPU/CUDA/OpenCL；
  `aerial_geometry_benchmark` 负责测量空三外围几何阶段，避免把低收益 kernel 接入默认流程。
- PlaMatrix CPU/CUDA/OpenCL 的同数据数值与耗时对比见
  `docs/benchmarks/2026-08-19-plamatrix-ba-backend-parity.md`。

## 结果状态与回写

`BAResult` 统一提供：

- `BASolveStatus`：成功、未收敛、取消、输入无效、配置不支持、后端不可用或数值失败。
- `solutionUsable`：求解器结果是否允许调用方使用。
- 请求/实际后端、回退标记与原因。
- 相机、轨迹、观测规模，RMS，setup/solve/postprocess/total 耗时。
- PlaMatrix 初始/最终鲁棒代价、LM 接受/拒绝步、初始 gross gate、线性后端/设备、PCG 迭代、
  Schur pattern 构建/复用次数、数值装配位置及耗时。

取消、数值失败或不可用结果不会修改调用方传入的相机和三维点。Auto 先运行一个满足能力和规模条件的
候选后端，只有状态不可用或 RMS、有效轨迹等质量门控失败时才运行回退后端。SfM 协调层另行记录
`ba_result_applied`，表示通过规范恢复和质量门控后的结果是否已经写回重建。

## 联合共享内参与自适应模型

PlaMatrix 使用每个标定组一个九参数内参块：`log(fx)`、`log(fy/fx)`、主点偏移、
`k1/k2/k3/p1/p2`，并使用同一参数掩码、参考标定、边界、弱先验和分阶段释放策略。
相机可以有不同初始焦距，但同一相机组在求解后得到同一个焦距。焦距、相机位姿和三维点处于同一个
PlaMatrix Schur-LM 问题，不使用外层交替更新模拟联合自标定。`cameraCalibrationGroupIds` 可为不同镜头/焦段建立独立参数，
默认先固定焦距预热几何，再释放各组焦距。默认只释放焦距，不自动释放主点或畸变。

空三的 `adaptiveCameraModelFitting` 会把完整 Brown 模型作为允许上限，再由
`BundleAdjustAdaptiveCameraModel` 从当前粗解逐项选择 `f/aspect/cx/cy/k1/k2/k3/p1/p2`。策略使用与
统一的有效轨迹筛选、像面中心/外围和方向覆盖、光轴多样性、交会角及多视轨迹比例；启发式可靠性
评分会先逐轨 Schur 消元三维点，降低可由场景结构吸收的响应被当作独立标定证据的风险。近似平行的对地块通常只释放
`f+k1`，多高度汇聚或环拍在证据充分时再释放更多参数。未释放参数在残差和写回阶段均保留每台相机原值。
像面方向覆盖以当前观测半径的相对外围计算；对于长焦、窄视场或行星影像，`k1/p1/p2` 的灵敏度、硬边界和弱先验
会换算到统一参考像场，避免固定的普通镜头半径门槛把本可观测的低阶畸变永久冻结。`k2/k3` 仍要求足够大的绝对
归一化半径和汇聚几何，防止高阶参数在平行航带中吸收地形穹顶。该尺度作为单独 BA 选项应用，不会在多轮 BA 中复合。
证据不足时保持 `fixed`；调用方已有的逐参数掩码始终作为不可越过的上限。
多起点预览、完整精化及多轮重三角化可以继承上一轮数值初值，但焦距/宽高比/主点边界和全部内参弱先验
始终相对稳定的 `sharedIntrinsicReferenceCameras`，不会把相对范围逐轮复合放大。`IncrementalSfm::run()`
在整个生命周期内按影像 ID 保存首次有效参考，周期性、最终及重试全局 BA 共用同一锚点；单次迭代的大型
自标定多起点只在首轮执行。逐参数掩码在 PlaMatrix 与 Legacy 焦距路径中使用同一生效判定，Legacy 会先把越界 warm start
投影回稳定焦距范围，并拒绝把多标定组或扩展模型静默合并成不等价的单组焦距解。
MetaShape 用户手册只公开了按数据条件自适应选择相机参数的工作流概念，未披露内部评分、矩阵构造或阈值。
PlaScan 仅参考这一公开概念；本文所述判据、评分与阈值均为独立设计，不主张与 MetaShape 内部实现等价。

正式空三把状态分为 `requested/scheduled/effective/applied`：用户开启不等于模型已经生效，只有最终选中
的非 `fixed` 模型通过 BA 质量门并写回才标记为 applied。完整外部/工程已知位姿继续复用输入标定，
不会运行自由网络自标定；多轮重三角化时保留之前已经应用的模型状态，末轮数值 no-op 不会抹掉它。

## 规范、鲁棒目标与正深度

- 单目 SfM 的全局 7 自由度规范由 `SfmBundleAdjustCoordinator` 和
  `SimilarityGaugeNormalizer` 管理，不在各后端内用不同方式重复实现。
- `fixedTrackIndices` 使轨迹的三维点参数块保持常量，但不删除其对相机的重投影残差；
  PlaMatrix 和 Legacy 后端语义一致。
- Legacy 的法方程线性化与 LM trial step 使用同一个 Huber robust cost。
- PlaMatrix 的 Huber 代价、观测权重和活动轨迹 gross gate 使用统一定义。
- Legacy 和 PlaMatrix 统一执行正深度检查、基于全局中位数的自适应点过滤与结果统计。
- 后端选择和求解规划只统计有限像点、正权重且形成双相机轨迹的有效观测；零权重、负权重和
  非有限权重不会再被不同后端解释成不同残差贡献。
- Auto 质量门控除重投影 RMS 和有效 track 比例外，还检查 LiDAR、控制点和比例尺 RMS 不得恶化。
- LiDAR 点到面权重采用统计权重 `1/sigma^2`，并同步缩放残差和 Huber 阈值，使其物理米制
  阈值及目标函数保持一致。仅“存在 LiDAR 平面”不再被视为完整 7 自由度 gauge，
  联合 BA 默认仍使用与纯影像分支相同的相机锚点。
- 点的前后方由 `FramePinholeCamera::positiveDepth()` / `isPointInFront()` 定义，后端不得直接把原始相机 Z
  当作跨相机格式的统一正深度。

## 验证

模块测试位于 `tests/`，重点覆盖后端能力和选择、取消/失败保护、完整 Brown 内参、全部物方约束、质量门控、
PlaMatrix CPU/CUDA/OpenCL 同数据一致性和解析 Jacobian。SfM 规范和局部/全局 BA 协调测试位于
`src/core/sfm/test/`。

## PlaMatrix 与 PlaPoint 边界

- BA 的 3x3/6x6 小型法方程、有限样本中位数，以及通用 Huber、块法方程、Schur-PCG 和 LM 状态
  直接复用 PlaMatrix；摄影测量投影、活动轨迹、gauge、约束和结果质量仍留在 PlaScan，避免数值库
  反向依赖业务模型。
- BA 输入是带观测索引的 track，而不是需要邻域查询的点云容器，因此不强行依赖 PlaPoint。
  LiDAR 最近邻、法线和空间关联仍由 `core/lidar`/PlaPoint 在构建 BA 约束前完成。
