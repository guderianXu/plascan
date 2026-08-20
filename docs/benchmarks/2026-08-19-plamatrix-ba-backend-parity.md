# PlaMatrix BA CPU/CUDA/OpenCL 对比验证（2026-08-19）

## 目的

在相同联合 BA 数据、解析完整 Brown-Conrady 雅可比、物方约束、Huber 目标和 LM 外层策略下，
只切换 PlaMatrix Schur 线性求解后端，验证 CPU、CUDA、OpenCL 与 Ceres CPU 的数值一致性。
Ceres 只作为显式开启的行为基准；生产环境 Auto 与失败回退均不再依赖 Ceres。

## 环境

- Linux / GCC 13
- NVIDIA GeForce RTX 4060 Laptop GPU
- CUDA 13.1.115，计算架构 `sm_89`
- NVIDIA OpenCL ICD
- vcpkg 动态依赖树：`build/linux-vcpkg-cuda-opencl-release`

CUDA/OpenCL 显式后端均在结果中返回真实设备名，测试同时断言 `usedGpu=true`、
`plamatrix_schur_assembly_on_device=1` 且无隐式 CPU 回退。

## 可复现命令

下列含 `ceres_cpu` 的命令需要先启用 `PLASCAN_ENABLE_CERES_REFERENCE=ON` 及对应可选 vcpkg feature；
生产构建可去掉列表中的 `ceres_cpu`，直接复测三个 PlaMatrix 后端。

```bash
build/linux-vcpkg-cuda-opencl-release/bin/ba_backend_benchmark \
  12 240 6 30 8 1 \
  plamatrix_cpu,plamatrix_cuda,plamatrix_opencl,ceres_cpu

build/linux-vcpkg-cuda-opencl-release/bin/ba_backend_benchmark \
  80 3000 8 8 16 1 \
  plamatrix_cpu,plamatrix_cuda,plamatrix_opencl,ceres_cpu
```

## 结果

### 12 相机 / 240 轨迹 / 1440 观测

| 后端 | 成功 | 最终 RMS | 最终鲁棒代价 | 墙钟时间（秒） | PCG 迭代 | pattern 构建/复用 |
|------|------|----------|--------------|----------------|----------|-------------------|
| PlaMatrix CPU | 是 | 0.04124380628 | 2.584247139 | 0.08373 | 2703 | — |
| PlaMatrix CUDA | 是 | 0.04124357756 | 2.584227907 | 0.18339 | 954 | 1/7 |
| PlaMatrix OpenCL | 是 | 0.04124357756 | 2.584227907 | 0.17030 | 942 | 1/7 |
| Ceres CPU | 是 | 0.04124360793 | 约 2.584229 | 0.00636 | — | — |

设备块 Jacobi 将原标量 Jacobi 的 CUDA/OpenCL 累计迭代数从 `6868/6065` 降至 `960/949`；
原墙钟约 `0.47625/0.47250 s`。设备数值装配版本中 CUDA 报告 Schur 装配约 `0.01559 s`、
线性求解约 `0.06636 s`；OpenCL 分别约为 `0.02581 s` 和 `0.05879 s`。

### 80 相机 / 3000 轨迹 / 24000 观测

| 后端 | 成功 | 最终 RMS | 最终鲁棒代价 | 墙钟时间（秒） | PCG 迭代 | pattern 构建/复用 |
|------|------|----------|--------------|----------------|----------|-------------------|
| PlaMatrix CPU | 是 | 0.04404584261 | 48.32036955 | 0.55377 | 815 | — |
| PlaMatrix CUDA | 是 | 0.04404584261 | 48.32036955 | 0.34719 | 815 | 1/5 |
| PlaMatrix OpenCL | 是 | 0.04404584261 | 48.32036955 | 0.40189 | 816 | 1/5 |
| Ceres CPU | 是 | 0.04404585629 | — | 0.06249 | — | — |

设备 PCG 迭代数已与 CPU 块 Jacobi 的 `815` 次一致，原标量 Jacobi 为 `4634/4617` 次。
CUDA 设备端 Schur 数值装配约 `0.06898 s`、线性求解约 `0.06390 s`；OpenCL 分别约为
`0.12475 s` 和 `0.06880 s`。装配索引按块槽缓存，避免把每个 Schur 项按 9x9 标量重复上传。
墙钟受冷启动和设备 runtime 状态影响，迭代、pattern 与设备装配标志更适合作为确定性回归指标。

## 结论

- CPU、CUDA、OpenCL 在两组相同输入上均收敛，RMS、鲁棒代价、相机和三维点结果满足自动化对比容差。
- CUDA/OpenCL 路径会显式检查设备并报告设备名；显式禁止回退时，设备不可用或索引无效会返回
  `BackendUnavailable`。允许回退时只转入语义等价的 PlaMatrix CPU，不会转入 Ceres。
- CUDA/OpenCL 已使用设备端块 Jacobi；80/3000 数据的迭代数与 CPU 一致，并在本次测量中快于
  PlaMatrix CPU，但仍明显慢于 Ceres CPU。
- CSR 拓扑经过完整邻接签名验证后复用，数值、阻尼和 eliminated 逆块每轮重新计算，不会复用陈旧值。
- CUDA/OpenCL 的 Schur 乘加已迁移到设备 kernel；主机保留拓扑验证、块逆和传输。80/3000 数据中
  CUDA/OpenCL 的设备装配分别比此前主机数值装配约快 46%/持平，并保持完全相同的最终结果。
- 完成约束、退化输入、取消和无 Ceres 构建门禁后，PlaMatrix CPU/CUDA/OpenCL 已纳入 Auto 默认选择。

## 2026-08-20 性能优化复测

在保持上述 Ceres parity 用例全部通过的前提下，新增 LM 拒绝步线性化复用、确定性并行装配、
CPU 稠密 Schur Cholesky、渐进式 PCG 容差、CUDA/OpenCL 常驻装配缓冲及 CUDA Schur device-to-device handoff。

| 数据 | PlaMatrix CPU | PlaMatrix CUDA | PlaMatrix OpenCL | Ceres CPU |
|------|---------------|----------------|------------------|-----------|
| 12/240/1440 | 0.00780 s | 0.13972 s | 0.16651 s | 0.00635 s |
| 80/3000/24000 | 0.24380 s | 0.24730 s | 0.30419 s | 0.07338 s |

两组 PlaMatrix 最终 RMS 分别为 `0.04124357756`、`0.04404584261`；对应 Ceres 为
`0.04124360793`、`0.04404585629`。12/240 CPU 相对优化前的 `0.08373 s` 约提升 10.7 倍，
80/3000 CPU 相对 `0.55377 s` 约提升 2.3 倍。80/3000 GPU 累计 PCG 从约 815 降到 641；
CUDA Schur 装配累计时间由约 `0.06898 s` 降到 `0.03004 s`，OpenCL 由约 `0.12475 s`
降到 `0.08663 s`。混合精度在 RTX 4060 实测变慢，因此实现保留为显式实验开关，默认关闭。

规模扫描显示 80 相机时 CPU 稠密直接解仍优于 GPU，而 150 相机/40000 观测时 CUDA 已快于 CPU；
Auto 默认交叉阈值据此调整为 128 相机且 30000 观测。OpenCL 跨队列零拷贝在 NVIDIA 595.84
驱动触发异步事件线程崩溃，生产路径保留稳定主机 handoff；CUDA 已完全移除 Schur 数值 D2H/H2D 往返。

## 2026-08-20 PlaMatrix 原生 CPU 线性代数复测

生产构建关闭 PlaMatrix 的可选系统 BLAS/LAPACK 后端，并从 vcpkg manifest 与 Debian 运行时依赖中
移除了 LAPACK/OpenBLAS。原生 GEMM 使用分块、SIMD 和 OpenMP；SVD/eigh 使用按标量精度与矩阵尺度
收敛的 Jacobi；128 阶以上的稠密 Schur 使用 32 列分块的持久 OpenMP Cholesky。

| 数据 | 原生 PlaMatrix CPU | 之前的 LAPACK CPU | 最终 RMS | 最终鲁棒代价 |
|------|--------------------|-------------------|----------|--------------|
| 12/240/1440 | 0.00879 s | 0.00780 s | 0.04124357756 | 2.584227907 |
| 80/3000/24000 | 0.29709 s | 0.24380 s | 0.04404584261 | 48.32036955 |

80/3000 的首个标量原生版本为 0.41593 s；分块 Cholesky 将累计稠密线性求解从 0.16063 s
降到 0.04468 s。相对系统 LAPACK 的应用级差距约 22%，但数值结果不变，且 GUI 与 BA benchmark
的动态依赖不再包含 CPU `libblas`、`liblapack`、`libopenblas` 或 `libgfortran`。CUDA 13.1 的
`libcublas`/`libcublasLt` 属于 GPU 后端，继续保留。

## 生产替代门禁

- 默认 `vcpkg.json` 不包含 Ceres，`PLASCAN_ENABLE_CERES_REFERENCE` 默认关闭。
- 全新 CUDA 13.1/OpenCL 构建可在 vcpkg 实际移除 Ceres 后完成；GUI、frame BA CLI 和 line-scan BA CLI
  的动态依赖及编译命令均不含 Ceres/glog/gflags/SuiteSparse。
- 层级 BA、GCP、尺度条、姿态、相机平面、行星激光与 line-scan 流程不再强制选择或失败回退 Ceres。
- Ceres 对照构建使用 `-DPLASCAN_ENABLE_CERES_REFERENCE=ON` 与可选 manifest feature，仍可重复运行同数据 parity。

## 自动化验证

- PlaMatrix 原生 CPU：278/278 通过；无 CUDA 构建中的 9 个 GPU 用例按设计跳过。
- PlaMatrix 可选系统 BLAS/LAPACK：268/268 通过（该构建未启用 benchmark 注册测试）。
- PlaMatrix CUDA 13.1/OpenCL + 原生 CPU：508/508 通过。
- 最终无 Ceres 生产构建：2673 项发现，2672 个已执行测试零失败；其中外部模型/数据相关用例按既有条件跳过，
  另有一个既有 PatchMatch benchmark 被显式禁用。
- BA 同数据测试会在设备存在时同时运行 PlaMatrix CPU/CUDA/OpenCL 和 Ceres，并检查后端身份、
  GPU 执行标志、设备名、设备 Schur 装配、迭代/耗时指标及相机、点、RMS、代价的一致性。
- 完整 Brown 回归覆盖共享 `f/aspect/cx/cy/k1/k2/k3/p1/p2`；联合约束回归覆盖 GCP、LiDAR 平面、
  比例尺、姿态先验、相机平面、激光测距、激光点先验和激光像点，并与 Ceres 比较。
