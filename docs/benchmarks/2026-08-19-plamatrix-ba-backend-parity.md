# PlaMatrix BA CPU/CUDA/OpenCL 对比验证（2026-08-19）

## 目的

在相同联合 BA 数据、解析完整 Brown-Conrady 雅可比、物方约束、Huber 目标和 LM 外层策略下，
只切换 PlaMatrix Schur 线性求解后端，验证 CPU、CUDA、OpenCL 与 Ceres CPU 的数值一致性。
本阶段不改变生产环境的 Auto 后端选择，Ceres 继续作为行为基准与回退后端。

## 环境

- Linux / GCC 13
- NVIDIA GeForce RTX 4060 Laptop GPU
- CUDA 13.1.115，计算架构 `sm_89`
- NVIDIA OpenCL ICD
- vcpkg 动态依赖树：`build/linux-vcpkg-cuda-opencl-release`

CUDA/OpenCL 显式后端均在结果中返回真实设备名，测试同时断言 `usedGpu=true`、
`plamatrix_schur_assembly_on_device=1` 且无隐式 CPU 回退。

## 可复现命令

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
- CUDA/OpenCL 路径会显式检查设备并报告设备名；设备不可用或索引无效时返回
  `BackendUnavailable`，不会静默转入 PlaMatrix CPU。
- CUDA/OpenCL 已使用设备端块 Jacobi；80/3000 数据的迭代数与 CPU 一致，并在本次测量中快于
  PlaMatrix CPU，但仍明显慢于 Ceres CPU。
- CSR 拓扑经过完整邻接签名验证后复用，数值、阻尼和 eliminated 逆块每轮重新计算，不会复用陈旧值。
- CUDA/OpenCL 的 Schur 乘加已迁移到设备 kernel；主机保留拓扑验证、块逆和传输。80/3000 数据中
  CUDA/OpenCL 的设备装配分别比此前主机数值装配约快 46%/持平，并保持完全相同的最终结果。
- 在更多真实数据验证前仍不把 PlaMatrix CUDA/OpenCL 纳入 Auto 默认选择。

## 自动化验证

- PlaMatrix CPU：273/273 通过；无 CUDA 构建中的 9 个 GPU 用例按设计跳过。
- PlaMatrix CUDA 13.1/OpenCL：502/502 通过。
- PlaScan 全量：2698 项发现，2697 个非禁用测试零失败；其中外部模型/数据相关用例按既有条件跳过，
  另有一个既有 PatchMatch benchmark 被显式禁用。
- BA 同数据测试会在设备存在时同时运行 PlaMatrix CPU/CUDA/OpenCL 和 Ceres，并检查后端身份、
  GPU 执行标志、设备名、设备 Schur 装配、迭代/耗时指标及相机、点、RMS、代价的一致性。
- 完整 Brown 回归覆盖共享 `f/aspect/cx/cy/k1/k2/k3/p1/p2`；联合约束回归覆盖 GCP、LiDAR 平面、
  比例尺、姿态先验、相机平面、激光测距、激光点先验和激光像点，并与 Ceres 比较。
