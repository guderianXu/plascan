# 空中三角测量与光束法平差完整优化计划

## 1. 目标

本计划用于修正 PlaScan 空中三角测量中光束法平差（Bundle Adjustment，BA）的
正确性、可观测性、后端一致性和性能问题。最终目标是：

1. GUI、CLI 和测试统一通过 `aerial_triangulation` 工作流进入同一套 SfM/BA。
2. 每次局部或全局 BA 都可观察、可取消、可诊断，不再静默跳过或静默回写失败结果。
3. 无控制点的单目 SfM 正确消除 7 自由度相似变换规范，局部 BA 不改变全局坐标系。
4. Legacy CPU、Ceres CPU、Ceres CUDA 和 Native CUDA 的能力边界明确，结果可比较。
5. 自适应焦距在同一非线性问题中与相机位姿、三维点联合优化。
6. 所有求解器使用一致的相机正深度、鲁棒损失、质量阈值和结果状态语义。
7. 大问题使用加速后端时不再无条件重复运行完整 Legacy BA。

## 2. 当前实际调用链

```text
GUI / CLI
  -> AerialTriangulationWorkflow
  -> AerialTriangulationPipeline
  -> SfmAttemptRunner
  -> IncrementalSfm::run
     -> InitialPairInitializer
        -> SfmBundleAdjustCoordinator::run(global)
     -> ImageRegistrationEngine
        -> SfmBundleAdjustCoordinator::run(local)
        -> SfmBundleAdjustCoordinator::iterative(global)
     -> KnownPoseReconstructor
        -> SfmBundleAdjustCoordinator::run(global)
  -> BundleAdjust::optimizePoints
```

`aerial_triangulation` 不直接包含 `BundleAdjust.h`。它链接 `sfm`，`sfm_core`
再链接 `bundle_adjust`。这是职责隔离，不是未调用 BA。

当前用户难以确认 BA 是否运行，主要原因是：

- BA 进度没有接入空三 `progressFn`。
- 参与 BA 的相机少于 2 台时直接返回。
- 没有包含至少两个有效观测的三维轨迹时直接返回。
- 部分候选粗筛关闭 BA 迭代日志。
- 空三结果没有正式记录每次 BA 的后端、状态、规模和耗时。

## 3. 已确认问题

### 3.1 规范自由度约束不完整

- 全局 BA 只固定第一台相机完整位姿，仍保留尺度自由度。
- 局部 BA 没有固定相机，也没有引入固定边界相机。
- 局部相机集合来自无序容器，顺序和潜在锚点不确定。

### 3.2 Native CUDA 不是联合 BA

- Auto 可以在 `refineCameraPose=true` 时选择 Native CUDA。
- 当前 CUDA kernel 只更新三维点，不更新相机位姿。
- PCG 和相机位姿步长参数尚未真正生效。
- 测试只验证重投影 RMS 下降，没有扰动并验证相机位姿恢复。

### 3.3 Ceres 失败或取消结果可能被回写

- Ceres 回调可以返回 `SOLVER_ABORT`。
- 求解后没有检查 `summary.IsSolutionUsable()`。
- `BAResult` 没有统一的成功、未收敛、取消和数值失败状态。
- SfM 协调器不能可靠区分“结果不可用”和“正常但没有更新”。

### 3.4 自适应焦距不是联合自标定

- 当前共享焦距只由 Legacy CPU 支持。
- 焦距、相机位姿和三维点通过分块交替方式优化。
- 勾选自适应相机模型拟合会改变后端并关闭 Ceres/CUDA 路径。

### 3.5 Legacy BA 的鲁棒目标不一致

- 法方程使用 Huber 权重。
- LM trial step 的接受准则使用普通 RMS 或不同单位残差的直接相加。
- 线性化目标和接受目标不一致，粗差条件下可能拒绝有效步长。

### 3.6 正深度约定不统一

- `Camera` 支持 `depthAxisFlipped=true`。
- SfM 的部分负深度过滤仍直接判断 `cameraPoint[2] < 0`。
- 相机投影、三角化、PnP 和过滤可能对同一点给出不同的前后方判断。

### 3.7 阈值和后端质量门控不一致

- 高质量 SfM 使用 1.5 像素过滤阈值，BA 仍可能使用默认 2.5 像素。
- Auto 加速后端可以无条件再运行一次完整 Legacy BA 进行比较。
- GPU 与 CPU 串行双求解会抵消加速收益。

## 4. 实施原则

1. 测试先行：每项行为修改先增加失败测试，再实现修复。
2. 保持职责边界：
   - `aerial_triangulation` 负责任务编排、进度和结果报告。
   - `sfm` 负责选择 BA 活动相机、边界相机、规范约束和结果回写。
   - `bundle_adjust` 负责求解器、后端能力、状态和数值目标。
   - `camera` 负责统一相机坐标与正深度定义。
3. 不在 GUI 中实现摄影测量算法。
4. 不以“降低过滤阈值”代替几何和求解器修复。
5. 后端不可用或能力不满足时必须明确回退并记录原因。
6. 所有失败和取消路径保持输入相机、三维点不变。

## 5. 分阶段实施

### 阶段 A：BA 可观测性和统一结果状态

#### 实现

- 在 `BAResult` 中增加：
  - `BASolveStatus`
  - `solutionUsable`
  - `cancelled`
  - `terminationMessage`
- Ceres 求解后检查 `summary.IsSolutionUsable()`。
- Legacy 和 Native CUDA 返回相同状态语义。
- `SfmBundleAdjustCoordinator` 只回写可用结果。
- 增加 BA 调用记录：
  - local/global
  - 相机、轨迹、观测数量
  - 请求后端和实际后端
  - 求解状态
  - RMS、有效轨迹比例和耗时
- 将 BA 进度转发到空三 `progressFn`。
- BA 被跳过时记录原因，不再静默返回。

#### 测试

- Ceres 取消后不改变相机和点。
- Ceres 数值失败后不回写。
- 空轨迹和少于两台相机时返回明确跳过状态。
- 空三进度能够观察到局部和全局 BA 阶段。

### 阶段 B：全局尺度规范和局部 BA 固定边界

#### 实现

- 对 BA 图像 ID 排序，保证确定性。
- 增加相机基线规范约束：
  - 第一台相机固定完整位姿。
  - 第二台相机保持初始基线尺度。
- 有控制点、比例尺或足够位姿先验时，不重复施加无尺度规范。
- 局部 BA 分为：
  - 活动相机：新注册相机及其局部邻域。
  - 边界相机：观测局部轨迹但不属于活动窗口的已注册相机。
- 边界相机参与残差但保持固定。
- 局部 BA 没有有效边界时使用确定性的局部规范约束。

#### 测试

- 全局 BA 前后第一基线长度保持不变。
- 局部 BA 多次运行后全局坐标系不漂移。
- 不同插入顺序得到一致结果。
- 有控制点或比例尺时绝对约束优先。

### 阶段 C：统一正深度

#### 实现

- 在 `Camera` 或共享投影几何中提供：
  - `positiveDepth(worldPoint)`
  - `isPointInFront(worldPoint)`
- 替换 SfM、BA、三角化和质量过滤中的原始 Z 判断。
- `CameraBaseline` 使用同一正深度接口。

#### 测试

- 普通正 Z 相机和 flipped-depth 相机投影结果一致。
- flipped-depth 相机的合法点不会被 BA 后过滤。
- 三角化质量、PnP 和 BA 对点的前后方判断一致。

### 阶段 D：Legacy 鲁棒目标一致化

#### 实现

- 提取统一的 Huber loss：
  - `rho(s)`
  - `rho'(s)`
- 点、相机、控制点、比例尺和位姿先验统一使用同一代价定义。
- LM 线性化和 trial step 接受使用同一个总目标。
- 不再直接相加不同单位的 RMS。
- 日志记录总 robust cost、重投影 RMS 和约束 RMS。

#### 测试

- 无粗差时结果与现有实现保持可比。
- 10%、20%、30%粗差下 robust cost 单调下降。
- 粗差不会将相机中心拉离真值。

### 阶段 E：Ceres 联合共享焦距自标定

#### 实现

- 在 Ceres 重投影残差中加入共享焦距参数块。
- 使用 `log(focalScale)` 参数化并设置上下界。
- 支持按相机组共享焦距。
- 在同一个 Ceres Problem 中联合优化：
  - 相机位姿
  - 三维点
  - 共享焦距
- 默认只释放焦距，不释放主点和畸变。
- Ceres 不可用时才回退 Legacy，并明确记录。

#### 测试

- 错误初始焦距能够收敛到真值附近。
- CPU/CUDA Ceres 的焦距和位姿结果可比较。
- 焦距不会越过配置上下界。
- 转台弱基线数据不会因焦距自由度导致相机塌缩。

### 阶段 F：Native CUDA 能力修正和联合 BA

#### 安全修复

- 增加 `BABackendCapabilities`。
- 当前 Native CUDA 标记为只支持点优化。
- `refineCameraPose=true` 时 Auto 不选择点优化后端。
- 未实现的 PCG 和位姿参数不再对外宣称生效。
- 运行时检查真实 CUDA 设备数量、设备选择和显存可用性。

#### 联合 BA 实现

- 构造相机 6-DOF 和点 3-DOF 雅可比。
- GPU 上形成 Schur 补。
- PCG 求解相机增量。
- 回代三维点增量。
- 实现 LM 阻尼、步长接受、固定相机、基线规范和 Huber loss。
- 支持取消和明确的失败状态。

#### 测试

- 同时扰动相机位姿与三维点。
- Native CUDA 必须恢复相机并返回非零 `refinedCameraCount`。
- 与 Ceres CPU 做 Sim(3) 对齐后比较：
  - 相机中心误差
  - 旋转误差
  - 重投影 RMS
  - 有效轨迹比例
- 不支持的控制点或焦距参数必须明确回退。

### 阶段 G：阈值统一和质量门控性能

#### 实现

- 建立统一质量预设，集中生成：
  - PnP 阈值
  - 三角化阈值
  - BA Huber 阈值
  - BA 点过滤阈值
  - 观测过滤阈值
- 空三配置同时设置 SfM 和 BA 阈值。
- Auto 后端默认只运行一次候选求解。
- 仅在以下情况回退：
  - 后端不可用
  - 结果状态不可用
  - RMS、有效轨迹比例、正深度比例或位姿增量越界
- Legacy 对照改成显式诊断模式或低频抽样模式。

#### 测试

- 质量预设在 workflow、SfM 和 BA 中保持一致。
- 正常候选后端不触发第二次完整求解。
- 质量异常时能够回退并记录原因。
- 性能基准分别记录 setup、solve、传输和总耗时。

## 6. 测试矩阵

### 单元测试

- `bundle_adjust/tests`
  - 状态、取消、失败保护
  - gauge 和基线
  - 鲁棒目标
  - Ceres 共享焦距
  - Native CUDA 相机恢复
- `sfm/test`
  - 局部 BA 边界相机
  - 正深度
  - 阈值传递
  - BA 跳过原因
- `aerial_triangulation/tests`
  - BA 进度转发
  - GUI/CLI 配置一致性
  - 空三结果中的 BA 诊断

### 真实数据回归

#### `E:/code/test/temple`

- 目标：16/16 注册。
- 相机中心形成连续闭环。
- 无单台相机吸附到模型中心。
- 全局 BA 后相邻基线没有异常跳变。

#### `E:/code/test/dino`

- 目标：16/16 注册。
- 相机环绕轨迹连续。
- 自适应焦距开启和关闭均不发生模型塌缩。

#### `E:/code/test/hyb2`

- 验证蒙版、弱纹理、焦距先验和 flipped-depth 相机兼容性。
- 记录注册数、点数、RMS、正深度比例和相机基线统计。

### 后端一致性

对同一个保存的 BA 问题分别运行：

- Legacy CPU
- Ceres CPU
- Ceres CUDA
- Native CUDA

使用 Sim(3) 对齐后比较结果，避免把单目坐标系自由度误判为求解差异。

## 7. 验证命令

```powershell
cmake --build build/aerial-validation --config Release --parallel

ctest --test-dir build/aerial-validation -C Release --output-on-failure `
  -R "BundleAdjust|NativeCuda|SfmInit|SfmPipeline|AerialTriangulation"
```

真实数据通过更新后的 CLI 运行，并保存每次运行的：

- resolved settings
- BA invocation records
- SfM diagnostics
- sparse quality report
- 相机中心和基线统计

## 8. 完成标准

只有同时满足以下条件才认为本计划完成：

1. 七项问题均有针对性自动化测试。
2. BA 取消或失败不再修改重建。
3. 全局和局部 BA 不存在未约束的相似变换规范。
4. Native CUDA 不再在未优化相机时宣称完成联合 BA。
5. 自适应焦距进入 Ceres 联合优化。
6. 正深度和质量阈值在各模块中一致。
7. 默认 Auto 路径不再无条件运行两套完整 BA。
8. temple、dino 和 hyb2 回归结果写入验证记录。
9. `docs/PROJECT_ARCHITECTURE.md` 与实际模块职责同步。

## 9. 回退策略

- 每个阶段保持独立、可测试，避免一次同时改变所有求解行为。
- 新后端能力和状态字段先兼容旧调用，完成调用方迁移后再删除旧语义。
- Native CUDA 联合 BA 未达到 Ceres 对照精度前，Auto 不启用其相机优化能力。
- 真实数据质量下降时保留失败工件和诊断，不通过放宽阈值隐藏问题。

## 10. 实施结果（2026-07-30）

### 10.1 七项问题的处理状态

| 阶段 | 状态 | 实际结果 |
|------|------|----------|
| A：状态、取消与诊断 | 已完成 | `BAResult` 统一返回状态、可用性、回退原因和分阶段耗时；取消、失败结果不回写；SfM 将 BA 进度和诊断传到空三报告。 |
| B：相似变换规范 | 已完成 | 全局 BA 固定确定性锚点，并在求解后恢复初始基线尺度；局部 BA 保留固定边界相机；新增 `SimilarityGaugeNormalizer` 及回归测试。 |
| C：统一正深度 | 已完成 | `Camera::positiveDepth()` 和 `isPointInFront()` 成为统一入口，SfM 协调、三角化和投影过滤不再直接解释相机 Z 符号。 |
| D：Legacy 鲁棒目标 | 已完成 | 线性化和 LM trial step 使用一致的 Huber 目标，避免“法方程接受、trial RMS 拒绝”的目标不一致。 |
| E：Ceres 共享焦距 | 已完成 | 共享绝对焦距使用对数参数化，与相机位姿和三维点放入同一个 Ceres Problem；异构初始焦距回归测试验证最终只保留一个共同焦距。 |
| F：Native CUDA 能力边界 | 安全修复完成 | 当前实现明确标记为“仅优化三维点”，Auto 在需要相机或共享焦距优化时不会选中它；显式请求不支持配置时返回 `unsupported_configuration`。完整相机 Schur/PCG CUDA 求解仍是后续性能工作，未伪装成已完成能力。 |
| G：阈值与质量门控 | 已完成 | 空三统一下发 SfM/BA 阈值；Auto 只运行一次候选后端，只有不可用或未通过质量门控时回退；正常路径不再无条件重复完整 Legacy BA。 |

### 10.2 额外完成的搜索与性能修复

- 无相机焦距搜索扩展为 `0.55` 到 `10.0` 的 15 个尺度，覆盖普通相机、长焦转台和 Hayabusa2 ONC-T。
- 粗搜索最多并发 4 个候选，并按总线程预算拆分每个候选的线程数；整体进度按所有候选的完成量聚合。
- 候选排序不再让“单个最大相邻基线比”覆盖所有网络质量指标。闭环序列质量使用有界分数，并与交会角、多视轨迹、覆盖率和重投影误差共同排序。
- 焦距正式重放与自适应精化均保留独立诊断；自适应结果只有通过同一候选质量比较才会替换粗搜索结果。

### 10.3 自动化验证

- `BundleAdjust`、`Camera`、`SimilarityGaugeNormalizer`、`SfmPipeline` 等 27 项相关测试通过。
- `AerialTriangulationPipelineTest` 与 `SfmSearchPolicyTest` 共 23 项测试通过。
- 最终以 32 路并行执行扩展正则测试集，共 106 项通过、0 项失败；另有 1 项依赖外部真实
  `.tsai` 文件的测试按设计跳过。
- 新增测试覆盖：
  - Ceres 共享绝对焦距联合优化；
  - 取消/失败结果不回写；
  - Native CUDA 能力拒绝与 Auto 选择；
  - 相似变换基线恢复；
  - 正深度统一；
  - 焦距候选并行度不超过总线程预算；
  - 长焦候选不会仅凭序列最大基线比压过整体更优解。

### 10.4 真实数据回归

| 数据 | 注册 | 稀疏点 | 平均重投影误差 | 焦距/轨迹核验 |
|------|------|--------|----------------|---------------|
| Middlebury temple | 16/16 | 4730 | 0.303607 px | 共同焦距 1520.72 px；相对官方位姿做 Sim(3) 对齐后，相机中心 RMSE 为环半径的 0.263%，平均旋转误差 0.776°。 |
| Middlebury dino | 16/16 | 2876 | 0.479962 px | 共同焦距 3328 px；相对官方位姿做 Sim(3) 对齐后，相机中心 RMSE 为环半径的 1.277%，平均旋转误差 0.884°。 |
| Hayabusa2 hyb2 12 时刻 | 12/12 | 2330 | 0.398461 px | 共同焦距 9063.90 px；相机中心拟合圆径向 CV 0.12%，离面 RMS/半径 0.07%，相邻角步长约 30°，光轴指向目标余弦最小值 0.999981。 |

Temple 的焦距粗搜索由串行 42.334 秒下降到并行 22.838 秒，耗时减少约 46%。
真实数据验证工件分别保存在各测试工程的 `validation/ba-optimization-20260730*`
目录，原项目、原连接点和原匹配缓存未被覆盖。
