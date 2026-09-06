# 参考 BA CPU/CUDA/OpenCL 效率对比（2026-09-04）

> 本报告记录原生稀疏求解器接入前的 PlaMatrix CPU 稠密 Schur 基线。当前 CPU 默认已改为 PlaMatrix
> 自包含块稀疏直接解；现行接线结果见 `2026-09-04-plamatrix-native-sparse-direct.md`。因此下述 GPU 交叉点不能直接
> 作为当前版本的新结论。当前 CUDA 专项优化与稀疏 CPU 同输入复测见
> `2026-09-04-plamatrix-cuda-ba-optimization.md`。

## 测试环境与口径

- 构建：Linux/GCC Release，`PLASCAN_ENABLE_CUDA=ON`，`PLASCAN_ENABLE_OPENCL=ON`。
- CPU：Intel Core i7-13700H，20 逻辑线程；基准使用 20 线程。
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，驱动 595.84。OpenCL 组也显式使用该 NVIDIA GPU。
- 配置：固定内参、联合优化相机位姿与三维点、最多 10 次非线性迭代、严格 `1e-12`
  线性容差、关闭后端回退和质量对照求解。
- 计时：表中是 6 次 API wall 中位数，不包含真实 JSON/TSAI 加载。合成组按 CPU→CUDA→OpenCL
  交错运行；South Building 每个后端在同一进程重复 6 次。
- 峰值内存是单独进程的 `ru_maxrss`，包含运行时/驱动库，不是纯求解器工作区大小。

计时期间未停止用户的远程桌面和图形会话；因此使用中位数抑制系统调度器干扰。OpenCL 在
South Building 的一次运行出现 13.70 s 尾延迟，其余 5 次为 2.31–2.78 s；主表不使用该异常值代表
典型性能。

## 主结果

“相对 CPU”大于 1 表示比 CPU 快，小于 1 表示比 CPU 慢。合成组的三后端 RMS 和 cost 均一致到当前输出精度。

| 问题 | 后端 | API wall 中位数 | 相对 CPU | 峰值 RSS |
|---|---|---:|---:|---:|
| 12 相机 / 240 tracks / 960 observations | CPU | 0.0074 s | 1.00x | 122 MiB |
| 同上 | CUDA | 0.0865 s | 0.085x | 248 MiB |
| 同上 | OpenCL | 0.0926 s | 0.080x | 296 MiB |
| 80 相机 / 3,000 tracks / 24,000 observations | CPU | 0.3177 s | 1.00x | 151 MiB |
| 同上 | CUDA | 0.3541 s | 0.90x | 283 MiB |
| 同上 | OpenCL | 0.4015 s | 0.79x | 347 MiB |
| 256 相机 / 10,000 tracks / 80,000 observations | CPU | 2.3272 s | 1.00x | 317 MiB |
| 同上 | CUDA | 1.1189 s | **2.08x** | 366 MiB |
| 同上 | OpenCL | 1.2207 s | **1.91x** | 465 MiB |
| South Building: 123 相机 / 33,007 tracks / 223,593 observations | CPU | 2.9023 s | 1.00x | 636 MiB |
| 同上 | CUDA | 1.8771 s | **1.55x** | 709 MiB |
| 同上 | OpenCL | 2.5277 s | **1.15x** | 852 MiB |

小问题中 GPU 被运行时初始化、CSR 构建和 PCG 调度开销完全淹没。相机数增大后，CPU 稠密 Schur
Cholesky 成本增长更快，设备 PCG 开始占优。

## 规模交叉扫描

下表固定每台相机 40 条 track、每 track 8 个观测，每点是 3 次 API wall 的中位数。

| 相机 | 观测 | CPU | CUDA | OpenCL | CUDA 加速 | OpenCL 加速 |
|---:|---:|---:|---:|---:|---:|---:|
| 96 | 30,720 | 0.296 s | 0.374 s | 0.437 s | 0.79x | 0.68x |
| 128 | 40,960 | 0.461 s | 0.424 s | 0.504 s | 1.09x | 0.92x |
| 160 | 51,200 | 0.671 s | 0.545 s | 0.634 s | 1.23x | 1.06x |
| 192 | 61,440 | 0.963 s | 0.644 s | 0.731 s | 1.50x | 1.32x |
| 224 | 71,680 | 1.155 s | 0.657 s | 0.761 s | 1.76x | 1.52x |

在本机与该拓扑上，CUDA 交叉点约为 128 台相机，OpenCL 约为 160 台相机。这不是只由相机数决定的硬阈值；
South Building 虽然只有 123 台相机，但有 22.36 万观测，CUDA 仍有明显收益。

## South Building 质量与收敛差异

| 后端 | RMS after | final cost | 线性化次数 | 10 次预算状态 |
|---|---:|---:|---:|---|
| CPU | 0.2722883255 px | 20728.03853 | 5 | `no_convergence`，结果可用 |
| CUDA | 0.2722912038 px | 20728.36042 | 3 | `success` |
| OpenCL | 0.2722892724 px | 20728.14831 | 4 | `success` |

CUDA 的 RMS 相对 CPU 高约 0.0011%，OpenCL 高约 0.00035%，属于不同线性求解器的微小浮点/收敛轨迹差异。
它们共用同一 Armijo 接受规则，但不保证迭代次数相同。因此 South Building 的 1.55x CUDA 端到端收益同时包含
“线性求解加速”和“更早达到终止条件”。将预算提高到 30 后，CPU 在 6 个接受步 + 6 个拒绝步后也收敛，
单次 wall 为 2.985 s；CUDA/OpenCL 仍分别约 1.577/2.292 s。

## 结论与调度建议

- 小于约 100 台相机且观测不密集时，CPU 直接 Cholesky 明显更快。
- NVIDIA 设备上优先 CUDA；它在大合成组和 South Building 上分别快 2.08x 和 1.55x，且比 NVIDIA OpenCL 稳定。
- OpenCL 在本机约 160 台相机后才稳定超过 CPU，并有更大峰值内存和尾延迟。本次未测 Intel Iris Xe
  OpenCL，不能把 NVIDIA OpenCL 数据外推到所有设备。
- 旧 Auto 阈值 24 相机且 30,000 观测会在 96 相机/30,720 观测的合成组过早选择 GPU。
- 基于本报告的后续实现采用两层策略：CUDA 常规门槛 128/30000，OpenCL 常规门槛 160/50000；
  120 相机以上的高密度网络分别在 150000/200000 观测进入 CUDA/OpenCL。South Building
  123/223593 因高密度层仍选择 GPU，显式指定后端不受 Auto 门槛限制。
- 这些值是本机默认策略，不是跨硬件绝对交叉点；Intel/AMD OpenCL 等未覆盖设备仍需单独基准校准。

报告采集阶段未修改选择策略；上述阈值由紧随其后的实现任务落地。
