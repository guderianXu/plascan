# MVS 异构计算前期计划

更新日期：2026-08-05

## 目标与边界

目标是在不改变 PatchMatch 深度语义、置信度门限和现有 CUDA 性能的前提下，使 MVS 深度估计
可以同时调度 CPU、多块 NVIDIA GPU，并为 AMD/Intel GPU 的 OpenCL 或 Vulkan Compute 后端提供
稳定接入面。本轮是前四个阶段，不包含 OpenCL/Vulkan kernel 本身的移植。

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

- `DepthComputeScheduler` 定义 CPU/CUDA/OpenCL/Vulkan worker 的统一身份和优先级帧队列。
- CPU 和所有 CUDA 设备从同一个线程安全队列领取帧；计算快的 worker 自然获得更多任务，
  不需要预估不同厂商设备的固定比例。
- 调度器统计每个 worker 的完成数、失败数和累计耗时，用于后续设备权重和 GUI 诊断。
- 已预留 OpenCL/Vulkan 后端类型，但当前不会枚举或调用这两类设备。

## 后续决策点

1. 先用一个小型光度代价 kernel 完成 OpenCL 和 Vulkan Compute 的 A/B 原型，对比 AMD/Intel 驱动覆盖、
   shader/kernel 编译链、调试性和内存复用。
2. 若优先要“同一份计算代码跨厂商”，优先评估 OpenCL/SYCL；若优先要依赖统一、长期驱动支持和
   与 Qt/Vulkan 设备共享，优先评估 Vulkan Compute。
3. 真正移植时先落地纯标量深度假设和 NCC，再移植传播、法向和几何一致性；每一步都与阶段 1 的
   质量基线比对。

## 可行性结论

项目已具备跨后端调度的代码边界，接入 AMD/Intel GPU 在架构上可行。主要成本不在任务调度，
而在约 10 类 PatchMatch CUDA kernel、constant memory 相机参数、随机数和缓存/传输管理的跨 API 移植与
数值一致性验证。建议先做可运行的小型原型，再决定 OpenCL 还是 Vulkan Compute，不同时维护两套完整 kernel。
