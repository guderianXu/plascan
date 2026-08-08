# MVS 异构计算前期计划

更新日期：2026-08-08

## 目标与边界

目标是在不改变 PatchMatch 深度语义、置信度门限和现有 CUDA 性能的前提下，使 MVS 深度估计
可以在原生 CPU、多块 NVIDIA CUDA GPU，以及 AMD/Intel/NVIDIA OpenCL GPU 上运行。计算后端明确
采用 CPU/CUDA/OpenCL；自动模式按 CUDA、OpenCL、CPU 逐级选择一个后端族，不再混合不同后端。
MVS 深度计算不保留 Vulkan Compute 接口，GUI 的 Vulkan 图形渲染不受影响。

## 前四阶段

### 阶段 1：基线和风险审计（已完成）

- 记录当前 Windows/MSVC/CUDA 环境、输入用例、质量指标和耗时。
- 固定并行 sweep 与传统 sweep 的回归门限。
- 修复灰度图 GPU 缓存的主机地址重用问题，确保连续 benchmark 结果可重复。
- 基线见 `docs/benchmarks/2026-08-05-patchmatch-backend-baseline.md`。

### 阶段 2：后端边界拆分（已完成）

- `PatchMatchEstimator.cpp` 只负责输入校验、后端选择和回退。
- `PatchMatchCPU.cpp` 保留真实 CPU 实现，`PatchMatchCUDA.cu` 仅保留 CUDA 资源与 kernel 路径。
- `PLASCAN_ENABLE_CUDA=OFF` 可在安装了 CUDA 的开发机上强制生成真实 CPU-only 构建。
- 无 CUDA 构建继续编译和运行 CPU PatchMatch，不再只提供公共入口空存根。

### 阶段 3：按 CUDA 设备隔离资源（已完成）

- `PatchMatchConfig::cudaDeviceIndex` 可显式选择 CUDA 设备，`-1` 使用当前设备。
- 每块 CUDA GPU 拥有独立的 workspace、执行互斥量和缓存生命期锁。
- 灰度图缓存 key 包含设备编号；缓存驱逐、显存释放和线程上传流都在正确 CUDA context 内执行。
- 同一块 GPU 仍串行修改 constant memory，不同 GPU 可以并行运行。

### 阶段 4：统一异构调度骨架（已完成）

- `DepthComputeScheduler` 定义 CPU/CUDA/OpenCL worker 的统一身份和优先级帧队列。
- 选定后端族的 CPU 或全部同类 GPU 从同一个线程安全队列领取帧；计算快的 worker 自然获得更多任务，
  不需要预估同后端多设备的固定比例。
- 调度器统计每个 worker 的完成数、失败数和累计耗时，用于后续设备权重和 GUI 诊断。
- Vulkan Compute 占位类型已经删除，避免形成一套不会实现的兼容接口。

## 阶段 5：OpenCL GPU 深度估计（已完成首期实现）

- 使用 OpenCL C 1.2 kernel 实现并行逆深度搜索、局部细化、多源 NCC、有效蒙版、稀疏深度提示和
  光度唯一性置信度，运行时兼容 OpenCL 1.2 及以上驱动。
- 每个 OpenCL GPU 缓存独立 context、command queue、program 和 kernel；设备内部串行提交，设备之间
  可并行领取帧。
- Auto 模式严格按 CUDA → OpenCL → CPU 选择整批后端：检测到可用 CUDA 时不再建立 OpenCL/CPU
  worker；CUDA 不可用时先创建 OpenCL context 并在线编译全部 PatchMatch kernel，预检成功才固定为
  OpenCL，预检失败则在首帧开始前使用原生 CPU。
- 主重建 CLI 的 `--mvs-backend opencl` 以及深度重放 CLI 的 `--device opencl` 可显式选择 OpenCL；
  `PLASCAN_ENABLE_OPENCL=OFF` 可构建无 OpenCL 版本。
- Intel Iris Xe/NEO 已完成真实 kernel smoke test。无 OpenCL GPU 的测试机跳过设备测试，但仍编译
  OpenCL 或无 OpenCL 存根路径。

## CPU 策略

CPU 不通过 OpenCL 执行。现有 C++/OpenMP 路径没有 JIT、驱动与主机内存搬运开销，适合作为稳定回退、
数值参考和调试路径。只有将来出现经过基准证明更快的特定 CPU OpenCL 驱动时，才重新评估。

## 阶段 6：多 GPU 流水线连续性优化（已完成）

- 多 GPU 采用同后端族的帧级并行：一张参考帧的深度图不跨设备拆分，不同参考帧由多块 CUDA GPU
  或多块 OpenCL GPU 从共享队列领取；较快设备会自然处理更多帧。
- worker 池先为选定后端的每块物理 GPU 保留一个执行来源，再增加同设备主机准备槽，使下一帧的
  CPU 准备和上传可与当前帧 kernel 重叠。自动任务不会同时建立 CUDA、OpenCL 和 CPU worker。
- 每块 CUDA/OpenCL GPU 都只有一个设备执行槽，避免 kernel、显存带宽和核显共享内存过度竞争；第二个
  主机帧槽只负责提前准备下一帧。OpenCL 共享 context/program，但只创建一个 command queue/kernel。
- OpenCL 按输入影像存储身份与工作分辨率缓存缩放后的 float 影像，并复用全部输入/输出 buffer；参考
  patch 由 work-group 协作载入 local memory，深度假设采用粗到细搜索。相机/蒙版打包和 OpenCV 后处理
  仍在执行槽外完成。
- 深度估计开始前按 PCI 物理设备标识逐卡获取跨进程租约。部分 CUDA 卡被其它 PlaScan GUI/CLI 占用时，
  Auto 继续使用其余可租用 CUDA 卡；CUDA 全部不可租用时再尝试 OpenCL，OpenCL 也不可用时使用 CPU。
  显式指定具体设备仍严格失败。最终后端和可用设备在 workspace hash 生成前固定，批次中途不切换后端。
- 进度中的“物理 GPU”只统计真实 CUDA/OpenCL 设备；重复的主机准备/执行 lane 单独显示为“活跃 GPU
  帧槽”，不再把 `CUDA×2 + OpenCL×1/2` 误报为三块或四块显卡。
- CUDA 日志记录 `prepare/wait/gpu_slot/post/total`，OpenCL 日志记录
  `prepare/wait/queue/kernel/read/post/total`；其中 `queue` 是提交到完成的墙钟时间，`kernel` 是设备事件的
  实际执行时间。`wait` 持续偏高表示同设备主机槽已成功前置准备；设备监控中的短暂空档若对应 `prepare`
  或 `post` 偏高，则应继续优化 CPU 数据准备或后处理；`queue-kernel` 偏高则指向驱动/JIT/排队开销。
- OpenCL 批次结束时汇总首个队列开始至末个队列结束的 `queue_occupancy`、`inter_call_idle`、
  `queue_non_kernel` 与 `end_to_end_kernel_duty`。一致性后残余深度重估也复用双主机槽，但内部像素线程
  预算按槽数均分，避免 CPU 过度订阅；设备执行槽和 command queue 仍各只有一个。

## 可行性结论

项目已具备可运行的跨厂商 OpenCL GPU 深度估计和 CPU/CUDA/OpenCL 统一帧调度。自动选择严格采用
CUDA → OpenCL → CPU，显式后端不可用时明确失败。首期 OpenCL 路径
采用独立的并行深度假设搜索，并未逐行翻译 CUDA 的传播、法向与几何一致性 kernel；当前已加入有界
输入缓存、持久 buffer、local-memory patch 分块和粗到细搜索。后续仍需用真实航测/环拍数据持续做质量
与吞吐基准，再按收益补充更复杂的传播。
