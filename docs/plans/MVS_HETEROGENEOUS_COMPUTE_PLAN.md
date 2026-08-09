# MVS 异构计算前期计划

更新日期：2026-08-09

## 目标与边界

目标是在不改变 PatchMatch 深度语义、置信度门限和现有 CUDA 性能的前提下，使 MVS 深度估计
可以在原生 CPU、多块 NVIDIA CUDA GPU，以及 AMD/Intel/NVIDIA OpenCL GPU 上运行。计算后端明确
采用 CPU/CUDA/OpenCL；自动模式会同时使用可租用的 CUDA 与 OpenCL 物理 GPU，并在没有可用加速器时
回退到原生 CPU。显式 CUDA/OpenCL/CPU 仍是严格的单后端请求。
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
- 原生 CPU 或所有可用 CUDA/OpenCL GPU 从同一个线程安全队列领取帧；计算快的 worker 自然获得更多任务，
  不需要预估跨后端多设备的固定比例。Auto 有 GPU 参与时不混入 CPU worker。
- 调度器统计每个 worker 的完成数、失败数、累计耗时和帧耗时移动平均，用于收益调度和 GUI 诊断。
- Vulkan Compute 占位类型已经删除，避免形成一套不会实现的兼容接口。

## 阶段 5：OpenCL GPU 深度估计（已完成首期实现）

- 使用 OpenCL C 1.2 kernel 实现并行逆深度搜索、局部细化、多源 NCC、有效蒙版、稀疏深度提示和
  光度唯一性置信度，运行时兼容 OpenCL 1.2 及以上驱动。
- 每个 OpenCL GPU 缓存独立 context、command queue、program 和 kernel；设备内部串行提交，设备之间
  可并行领取帧。
- Auto 同时探测 CUDA 和 OpenCL；OpenCL 必须先通过 context 创建与 PatchMatch kernel 在线编译
  预检才能加入 worker 池。同一物理 GPU 如果同时通过 CUDA/OpenCL 暴露，则按 PCI 物理标识去重并
  优先保留 CUDA 路径；因此 NVIDIA 独显不会被重复计数，AMD/Intel 核显仍可与它并行。
- 每个自动任务逐卡获取跨进程租约；一块忙卡不会阻塞其他 CUDA/OpenCL GPU。只有没有任何通过
  预检且租用成功的加速器时，Auto 才在首帧开始前使用原生 CPU。
- 主重建 CLI 的 `--mvs-backend opencl` 以及深度重放 CLI 的 `--device opencl` 可显式选择 OpenCL；
  `PLASCAN_ENABLE_OPENCL=OFF` 可构建无 OpenCL 版本。
- Intel Iris Xe/NEO 已完成真实 kernel smoke test。无 OpenCL GPU 的测试机跳过设备测试，但仍编译
  OpenCL 或无 OpenCL 存根路径。

## CPU 策略

CPU 不通过 OpenCL 执行。现有 C++/OpenMP 路径没有 JIT、驱动与主机内存搬运开销，适合作为稳定回退、
数值参考和调试路径。只有将来出现经过基准证明更快的特定 CPU OpenCL 驱动时，才重新评估。

## 阶段 6：多 GPU 流水线连续性优化（已完成）

- 多 GPU 采用跨后端族的帧级并行：一张参考帧的深度图不跨设备拆分，不同参考帧可由 CUDA 独显和
  OpenCL 核显从同一优先级队列领取。调度器为每个参与设备保留一张初始标定帧，并按帧耗时的
  指数移动平均让较快设备自然领取更多任务。
- 队列进入尾部时会进行收益判定：如果计入在途帧后，慢设备领取候选帧的预计完成时间已不优于
  最快设备完成其在途帧并清空所有可领取剩余帧，慢设备不再领取新帧，避免核显尾帧延长整批墙钟。
  被最快后端排除的跨后端重试不计入其清队能力，能够处理该重试的慢设备不会因此被错误暂停。
- worker 池先为每块选中的物理 GPU 保留一个执行来源，再增加同设备主机准备槽，使下一帧的
  CPU 准备和上传可与当前帧 kernel 重叠。Auto 可同时建立 CUDA 和 OpenCL worker，但不会在已有 GPU
  worker 时再混入 CPU worker。
- 每块 CUDA/OpenCL GPU 都只有一个设备执行槽，避免 kernel、显存带宽和核显共享内存过度竞争；第二个
  主机帧槽只负责提前准备下一帧。OpenCL 共享 context/program，但只创建一个 command queue/kernel。
- OpenCL 按输入影像存储身份与工作分辨率缓存缩放后的 float 影像，并复用全部输入/输出 buffer；参考
  patch 由 work-group 协作载入 local memory，深度假设采用粗到细搜索。相机/蒙版打包和 OpenCV 后处理
  仍在执行槽外完成。
- 深度估计开始前按 PCI 物理设备标识逐卡获取跨进程租约。部分 CUDA/OpenCL 卡被其它 PlaScan GUI/CLI
  占用时，Auto 继续使用其余可租用的两类 GPU。显式指定具体后端/设备仍严格失败，不会切换或静默回退。
- OpenCL 驱动不提供标准 PCI 身份时，先用规范化设备名匹配 CUDA 枚举并复用 CUDA PCI 身份；仍无法
  确认身份的 NVIDIA OpenCL 接口在 CUDA 已启用时保守跳过，避免同一物理独显取得两把不同租约并建立
  两条 kernel 通道。
- 异构帧第一次失败时仅允许交给不同后端重试一次，并让故障设备停止领取普通帧；尾部因收益暂停的
  设备可以被重新唤醒接管重试。第二次失败直接终结该帧，显式/单后端任务保持原有严格失败语义。
- 流式且至少两块物理 GPU 的任务使用两个产物保存 worker；保存队列同时限制驻留任务数和按实际
  `cv::Mat` allocation 去重统计的字节数，等待入队、queued 与 active 均计入预算。缓存模式和单 GPU
  保持单保存 worker，文件格式、逐帧 manifest 锁与初始/过滤后阶段 barrier 不变。生产者预约只有在
  保存任务成功入队后才原子转交；计算与保存线程都捕获异常并回收在途计数，最终过滤后保存等待也可
  响应取消，且同一取消标志会传入 BFS 融合内部行循环，避免 `std::terminate`、容量泄漏或取消后继续融合。
- GUI 不再暴露工作线程数后，MVS 默认总 CPU 预算统一为逻辑线程数减 2；CLI 显式预算不再被历史
  7 线程上限截断。帧级计算按运行时实际 CUDA/OpenCL 主机槽数分摊像素线程并分完整数余量，例如
  32 线程机器默认 30 线程、4 个槽分为 8/8/7/7；每槽 OpenMP 后处理使用对应份额，一致性、预载、
  可见性与融合等独占 CPU 阶段使用完整总预算，既避免四个准备槽各自占满整机，也不浪费预算线程。
- 异构批次在 workspace hash 中保留稳定 `auto` token，每帧记录 `CUDA:N` 或 `OpenCL:N`，项目批次元数据
  标记为 `hybrid`。Auto 可复用兼容的单后端或异构批次；显式 CUDA/OpenCL/CPU 只复用同名的单后端批次。
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

项目已具备可运行的跨厂商 OpenCL GPU 深度估计和 CPU/CUDA/OpenCL 统一帧调度。Auto 可在同一批次中同时
使用去重后的 CUDA 独显与 OpenCL 核显，并通过标定和尾部收益门避免慢设备拖长总耗时；显式后端不可用时
仍明确失败。首期 OpenCL 路径
采用独立的并行深度假设搜索，并未逐行翻译 CUDA 的传播、法向与几何一致性 kernel；当前已加入有界
输入缓存、持久 buffer、local-memory patch 分块和粗到细搜索。后续仍需用真实航测/环拍数据持续做质量
与吞吐基准，再按收益补充更复杂的传播。
