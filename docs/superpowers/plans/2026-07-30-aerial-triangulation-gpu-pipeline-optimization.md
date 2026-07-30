# 空中三角测量 GPU 流水线优化计划

## 1. 目标

在不改变摄影测量结果语义、不降低可重复性和不增加默认显存风险的前提下，提高空中三角测量中
CUDA SIFT、LightGlue 和 BA 的有效 GPU 占用率，并为大规模工程补齐可扩展的并行路径。

本次优化必须保持以下约束：

- GUI、CLI 和核心工作流继续统一调用 `aerial_triangulation` 与 `matchphototask`。
- 已有特征、匹配和连接点缓存仍可复用；缓存命中时不得为了提高 GPU 占用率强制重算。
- 蒙版、关键点限制、连接点限制、取消、真实进度和失败回退语义保持不变。
- GPU 内存不足时按受控策略降低并发或回退串行，不允许整个 GUI 进程崩溃。
- 先写测试，再实现功能；每个阶段通过后再进入下一阶段。

## 2. 当前问题

### 2.1 LightGlue CUDA 实际串行

`AerialTriangulationOptions::cudaParallelPairs` 已由 GUI/CLI 提供，但未传入
`MatchPhotosOptions`，工作流报告中的 `cuda_parallel_pairs_effective` 固定为 `0`。
`MatchingStage` 只构造一个 `LightGlueMatcher`，随后逐匹配对串行推理。

### 2.2 CUDA SIFT 前后处理未流水化

每张影像按“读取 -> 缩放 -> CUDA SIFT -> 蒙版过滤 -> 写缓存”串行执行。CUDA kernel
运行期间没有准备下一张影像，GPU 等待影像解码；写特征文件时 GPU 同样空闲。

### 2.3 Ceres CUDA 只加速线性代数

重投影、控制点、比例尺和位姿先验大量使用 `NumericDiffCostFunction`。数值微分在 CPU
重复调用投影函数，即使线性求解选择 `dense_schur_cuda`，前端残差和 Jacobian 仍是瓶颈。

### 2.4 Native CUDA BA 未联合求解相机块

当前原生 CUDA 后端只更新三维点，不能优化相机位姿、共享焦距和软约束。正式 SfM 请求联合
BA 时会回退 Ceres/Legacy CPU。

### 2.5 SfM 几何阶段全部在 CPU

F/E/H RANSAC、PnP、三角化、重投影筛选和轨迹图构建均在 CPU。前四者可做批处理并行，
但小型工程的 GPU 搬运开销可能高于收益；轨迹图属于不规则图操作，不适合直接迁移 CUDA。

## 3. 实施阶段

### 阶段 A：配置、诊断和测试基线

1. 在 `MatchPhotosOptions` 增加 `cudaParallelPairs` 和特征预取深度。
2. 工作流把用户请求值传入连接点任务，并报告真实有效并发数。
3. 为并发数解析增加纯函数：
   - CPU 模式固定为 1。
   - 显式值限制在候选对数量与安全上限内。
   - 自动模式根据 CUDA 可用性、可用显存和关键点预算计算。
4. 匹配 sidecar 记录请求并发、有效并发、worker 编号和是否发生显存降级。
5. 新增阶段耗时：特征读取、预算裁剪、GPU 推理、蒙版过滤、结果写盘。

验收：

- GUI/CLI 参数能到达 `MatchingStage`。
- 进度计数严格单调，完成值等于候选对总数。
- CPU 模式和单对任务保持串行。

### 阶段 B：LightGlue 显存感知多匹配对调度

1. 将单对处理提取为无共享可变状态的 worker 函数。
2. 每个 CUDA worker 独立持有 `LightGlueMatcher`，禁止多个线程同时调用同一 TorchScript
   module。
3. 使用有界 worker 池处理候选对；结果按原候选顺序合并，保证输出稳定。
4. 蒙版缓存改为预加载只读数据或加锁缓存，禁止并发修改 `QMap`。
5. 写匹配文件与 sidecar 使用每对唯一文件，主线程汇总 `matchRecords`。
6. 捕获 CUDA OOM：
   - 停止提交新任务；
   - 释放 worker 模型与 CUDA cache；
   - 未完成任务以一半并发重试；
   - 并发降至 1 后仍失败才报告该匹配对失败。

验收：

- 合成匹配器测试证明最大同时执行数达到配置值。
- 串行和并行输出记录顺序、匹配数及文件内容一致。
- 取消后不再启动新任务，已运行任务可安全收尾。
- CUDA OOM 降级测试不会丢失尚未执行的匹配对。

### 阶段 C：CUDA SIFT 前后处理流水线

1. 将影像处理拆为：
   - CPU 准备：读取、灰度化、缩放、解析蒙版；
   - GPU 提取：单 CUDA SIFT 执行通道；
   - CPU 完成：恢复坐标、蒙版过滤、写特征文件。
2. 使用有界队列，默认预取 2 张影像，防止大 TIFF 导致内存膨胀。
3. 复用缓存的影像不进入队列。
4. CUDA SIFT 保持单执行上下文；先重叠 CPU IO 与 GPU，不盲目并发多个 SIFT context。
5. 输出按输入影像顺序合并，进度在特征文件成功落盘后递增。

验收：

- 串行与流水线关键点坐标、描述子、蒙版过滤结果一致。
- 峰值队列长度不超过配置值。
- 读取失败、取消和写入失败不会留下被当成有效缓存的半文件。

### 阶段 D：Ceres BA Jacobian 与求解策略

1. 将重投影残差从中央数值微分迁移到 AutoDiff 或解析 Jacobian。
2. 保持现有相机模型、畸变模型、共享焦距和位姿增量参数化语义不变。
3. 控制点、比例尺、激光平面、位姿先验逐项迁移；不能模板化的残差实现
   `SizedCostFunction::Evaluate` 解析 Jacobian。
4. 联合 BA 默认使用 `SPARSE_SCHUR` CPU；达到规模阈值且 Ceres CUDA 可用时使用
   `DENSE_SCHUR + CUDA`。
5. 基准比较 NumericDiff 与新 Jacobian 的 RMS、梯度和耗时。

验收：

- Jacobian 与有限差分相对误差满足测试阈值。
- temple、dino、hyb2 注册数、RMS 和相机轨迹不退化。
- 小问题不强制 GPU，大问题能真实报告 `ceres_cuda`。

### 阶段 E：Native CUDA 能力审计与联合 BA 前置条件

实现审计确认当前 Native CUDA 只是逐 track 点块核：没有相机块、点-相机交叉块、
Schur 消元或 PCG，报告中的 `pcgIterations` 固定为 0。直接启用相机能力会造成错误的
自动后端选择，因此本轮执行范围调整为：

1. 保持 Native CUDA 的 point-only 能力声明，不虚假标记联合 BA。
2. 修正 `depthAxisFlipped` 在 Host/Device 数据和投影/Jacobian 中的完整传递。
3. 正式联合相机/点 BA 继续走阶段 D 已优化的 Ceres CPU/CUDA。
4. 将完整块稀疏 Schur/PCG 作为独立里程碑：GPU 计算点块、相机块和交叉块，
   点块消元，PCG 求相机增量，再回代三维点。
5. 该里程碑必须同时支持 gauge 固定、共享焦距、软约束、LM 回滚和 Ceres 对照，
   全部通过后才允许 Auto 选择 Native CUDA。

本轮验收：

- flipped-depth 相机在 CPU、Host CUDA 数学层和设备核采用相同物理前向语义。
- 联合 BA 不会误选 point-only Native CUDA，也不会返回半优化状态。
- Ceres CPU/CUDA 与新 AutoDiff 投影保持相机中心、三维点和 RMS 一致。

### 阶段 F：SfM 批处理 GPU 收益评估

先增加可重复基准，再决定是否启用：

1. 几何验证：批量匹配对 F/E/H 假设评分。
2. 三角化：批量 DLT、正深度、重投影误差和三角化角筛选。
3. PnP：只评估大批量待注册影像，不替换小规模 `solvePnPRansac`。
4. 轨迹图构建继续使用 CPU，并优化并行分区和内存布局。

启用门槛：

- 端到端阶段耗时至少降低 20%。
- 结果与 CPU 路径在固定随机种子下满足同等质量门控。
- 小于门槛的数据自动保留 CPU 路径。

## 4. 测试策略

### 单元测试

- 并发数解析、显存预算、OOM 降级状态机。
- 有界队列、取消、顺序合并和原子进度。
- BA Jacobian 有限差分对照。
- Native CUDA 相机块、Schur、PCG 和 gauge 约束。
- 批量三角化 CPU/GPU 对照。

### 集成测试

- `MatchPhotosTaskTest`
- `AerialTriangulationWorkflowTest`
- `BundleAdjust*Test`
- `Sfm*Test`

### 真实数据回归

- `E:/code/test/temple`
- `E:/code/test/dino`
- `E:/code/test/hyb2`

记录注册影像数、连接点数、平均重投影 RMS、相机轨迹闭环质量、各阶段耗时、峰值显存和
实际后端。

## 5. 实施顺序与停止条件

实施顺序固定为 A -> B -> C -> D -> E -> F。每个阶段只有在相关测试和构建通过后才进入
下一阶段。

若某个 GPU 方案未达到阶段 F 的收益门槛，保留基准与诊断，不把低收益实验路径接入默认
工作流。正确性、稳定性和可回退性优先于任务管理器中显示的 GPU 利用率。

## 6. 执行结果（2026-07-30）

| 阶段 | 结果 |
|------|------|
| A/B | 已完成显存感知 LightGlue CUDA 多 worker 调度、独立 matcher/stream、真实有效并发报告和 OOM 串行补跑。 |
| C | 已完成 CUDA SIFT 前的有界 CPU 影像准备队列；GPU 提取保持单上下文，避免多 context 抢占显存。 |
| D | Ceres 热点残差已迁移到 AutoDiff/解析导数；Auto 继续按问题规模选择 CPU/CUDA。 |
| E | 已确认 Native CUDA 仍是 point-only；补齐 flipped-depth 语义，不把它误报为联合 BA。 |
| F | 已增加 `aerial_geometry_benchmark`。关键点专用读取达到约 20.5 倍加速；USAC、PnP 和三角化未达到 GPU 启用门槛，保留 CPU。 |

真实 dino 回归在 RTX 5080 上完成 16 张影像、120 个 LightGlue CUDA 匹配对，自动并发为 3，
没有 OOM；几何验证使用 8 路 CPU 并发。随后复用 2599 条连接点轨迹完成正式空三，
注册 16/16 张影像，输出 2794 个稀疏点，平均重投影误差 0.521 px，MVS 质量门控为 `ok`。
详细命令与数据见 `docs/benchmarks/2026-07-30-aerial-triangulation-gpu-pipeline.md`。

收尾审查补充了匹配 sidecar 的原子一致性：核心耗时在 sidecar 写入前固定，元数据写入失败时
删除对应二进制匹配文件；串行恢复日志区分 CUDA OOM、并发 worker 异常退出和未完成任务，
不再把所有回退误报为显存不足。
