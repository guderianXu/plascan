# MVS 异构计算前期计划

更新日期：2026-08-06

## 目标与边界

目标是在不改变 PatchMatch 深度语义、置信度门限和现有 CUDA 性能的前提下，使 MVS 深度估计
可以同时调度原生 CPU、多块 NVIDIA CUDA GPU，以及 AMD/Intel/NVIDIA OpenCL GPU。计算后端明确
采用 CPU/CUDA/OpenCL，不再为 MVS 深度计算保留 Vulkan Compute 接口。GUI 的 Vulkan 图形渲染不受影响。

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
- CPU 和所有 CUDA 设备从同一个线程安全队列领取帧；计算快的 worker 自然获得更多任务，
  不需要预估不同厂商设备的固定比例。
- 调度器统计每个 worker 的完成数、失败数和累计耗时，用于后续设备权重和 GUI 诊断。
- Vulkan Compute 占位类型已经删除，避免形成一套不会实现的兼容接口。

## 阶段 5：OpenCL GPU 深度估计（已完成首期实现）

- 使用 OpenCL C 1.2 kernel 实现并行逆深度搜索、局部细化、多源 NCC、有效蒙版、稀疏深度提示和
  光度唯一性置信度，运行时兼容 OpenCL 1.2 及以上驱动。
- 每个 OpenCL GPU 缓存独立 context、command queue、program 和 kernel；设备内部串行提交，设备之间
  可并行领取帧。
- Auto 模式同时调度 CUDA 与非重复的 OpenCL 设备。CUDA 已使用 NVIDIA 卡时，不再为同一张 NVIDIA 卡
  创建 OpenCL worker；Intel/AMD GPU 仍参与计算。
- 主重建 CLI 的 `--mvs-backend opencl` 以及深度重放 CLI 的 `--device opencl` 可显式选择 OpenCL；
  `PLASCAN_ENABLE_OPENCL=OFF` 可构建无 OpenCL 版本。
- Intel Iris Xe/NEO 已完成真实 kernel smoke test。无 OpenCL GPU 的测试机跳过设备测试，但仍编译
  OpenCL 或无 OpenCL 存根路径。

## CPU 策略

CPU 不通过 OpenCL 执行。现有 C++/OpenMP 路径没有 JIT、驱动与主机内存搬运开销，适合作为稳定回退、
数值参考和调试路径。只有将来出现经过基准证明更快的特定 CPU OpenCL 驱动时，才重新评估。

## 阶段 6：多 GPU 流水线连续性优化（已完成）

- 多 GPU 采用帧级并行：一张参考帧的深度图不跨设备拆分，不同参考帧由 CUDA、Intel/AMD OpenCL
  设备从共享队列并行领取；较快设备会自然处理更多帧。
- worker 池先为每块物理 GPU 保留一个执行来源，再增加同设备主机准备槽。CUDA + OpenCL 默认场景下，
  两块设备都会参与，同时为 CUDA 增加第二个主机槽，使下一帧的 CPU 准备和灰度图上传与当前帧 kernel
  重叠，不因加入核显而退回单槽 CUDA 流水线。
- 每块 CUDA/OpenCL GPU 仍只有一个设备执行槽，避免共享 constant memory、workspace、command queue 和
  kernel 参数发生竞争；重复的是主机准备槽，不是让同一设备并发修改共享资源。
- OpenCL 的设备锁缩小到 kernel 参数设置、提交、等待和结果回读。灰度缩放、相机/蒙版打包、buffer 创建
  以及 OpenCV 后处理不再占用设备执行槽；仅 OpenCL 的机器也可利用第二个主机槽预备下一帧。
- CUDA 日志记录 `prepare/wait/gpu_slot/post/total`，OpenCL 日志记录
  `prepare/wait/queue/kernel/read/post/total`；其中 `queue` 是提交到完成的墙钟时间，`kernel` 是设备事件的
  实际执行时间。`wait` 持续偏高表示同设备主机槽已成功前置准备；设备监控中的短暂空档若对应 `prepare`
  或 `post` 偏高，则应继续优化 CPU 数据准备或后处理；`queue-kernel` 偏高则指向驱动/JIT/排队开销。

## 可行性结论

项目已具备可运行的跨厂商 OpenCL GPU 深度估计和 CPU/CUDA/OpenCL 帧级异构调度。首期 OpenCL 路径
采用独立的并行深度假设搜索，并未逐行翻译 CUDA 的传播、法向与几何一致性 kernel；因此后续重点是
用真实航测/环拍数据做质量和吞吐基准，再按收益补充更复杂的传播与设备缓冲复用。
